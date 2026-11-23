"""Per-GPU and per-process VRAM, read without new dependencies.

Copied, not moved, from Kilix ``src/kitty/kilix_gpu.py`` (kitty fork commit
629035ef, blob 23ad1b04); MIGRATION.md lists what was kept and what changed.
Kilix core still carries its own copy until phase 2 points it here.

* ``/sys/class/drm/card*`` lists the GPUs, their PCI address, driver and
  runtime power state. A device whose ``power/runtime_status`` is
  ``suspended`` or ``suspending`` is reported asleep, and nothing else is read
  from it unless the caller asks to wake it.
* NVIDIA devices are read through NVML (``libnvidia-ml.so.1``) with ctypes:
  total, used and free memory, and the per-process memory lists. ``nvmlInit``
  attaches every NVIDIA GPU, so without ``wake=True`` NVML is not opened at
  all while any NVIDIA GPU is asleep. Each snapshot closes NVML again, since an
  open session holds every NVIDIA GPU awake.
* When NVML cannot be loaded, ``nvidia-smi`` is asked instead.
* AMD devices are read from ``mem_info_vram_*``, and their per-process memory
  from DRM fdinfo, only for fds that point at ``/dev/dri`` nodes of a device
  with a VRAM counter. Integrated GPUs without a VRAM counter are listed but
  not measured.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB
NVIDIA_VENDOR = 0x10de
_PCI_ADDRESS = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]\Z")
_CARD_NAME = re.compile(r"card\d+\Z")
_NODE_NAME = re.compile(r"(card|renderD)\d+\Z")
_VENDORS = {0x10de: "NVIDIA", 0x1002: "AMD", 0x8086: "Intel"}
# The fallback command when NVML cannot be loaded. Tests point it at a fake.
NVIDIA_SMI = "nvidia-smi"
_SMI_TIMEOUT = 10.0


def sys_root() -> str:
    return os.path.abspath(os.environ.get("KILIX_GPU_SYSFS_ROOT") or "/sys")


def proc_root() -> str:
    return os.path.abspath(os.environ.get("KILIX_GPU_PROC_ROOT") or "/proc")


def normalise_pci(text: str) -> str | None:
    """``0000:01:00.0`` from any case, or from NVML's 8-digit domain form."""
    value = text.strip().lower()
    if len(value) == 16 and value[8] == ":":
        value = value[4:]
    return value if _PCI_ADDRESS.match(value) else None


def _read(path: str, limit: int = 4096) -> str | None:
    try:
        with open(path, encoding="ascii", errors="replace") as stream:
            return stream.read(limit).strip()
    except OSError:
        return None


def _read_int(path: str, base: int = 10) -> int | None:
    text = _read(path, 64)
    if not text:
        return None
    try:
        return int(text, base)
    except ValueError:
        return None


# Discovery {{{

@dataclass(frozen=True)
class DrmDevice:
    """The static facts about one GPU, from sysfs."""
    pci: str
    sysfs: str
    vendor_id: int
    device_id: int
    driver: str
    nodes: tuple[str, ...]
    boot_vga: bool = False

    @property
    def is_nvidia(self) -> bool:
        return self.vendor_id == NVIDIA_VENDOR and self.driver == "nvidia"

    @property
    def discrete(self) -> bool:
        # An integrated GPU sits on the root bus (Intel's is always
        # 0000:00:02.0), or is the boot VGA device behind an internal bridge.
        if self.vendor_id == NVIDIA_VENDOR:
            return True
        if self.boot_vga:
            return False
        try:
            return int(self.pci.split(":")[1], 16) != 0
        except (IndexError, ValueError):
            return False


def discover(root: str | None = None) -> tuple[DrmDevice, ...]:
    drm = os.path.join(root or sys_root(), "class", "drm")
    try:
        names = sorted(os.listdir(drm))
    except OSError:
        return ()
    by_device: dict[str, list[str]] = {}
    cards: dict[str, str] = {}
    for name in names:
        if not _NODE_NAME.match(name):
            continue
        device = os.path.realpath(os.path.join(drm, name, "device"))
        if not _PCI_ADDRESS.match(os.path.basename(device)):
            continue  # a platform or virtual GPU without a PCI function
        by_device.setdefault(device, []).append(name)
        if _CARD_NAME.match(name):
            cards.setdefault(device, name)
    devices = []
    for device, nodes in by_device.items():
        if device not in cards:
            continue
        try:
            driver = os.path.basename(os.readlink(os.path.join(device, "driver")))
        except OSError:
            driver = ""
        devices.append(DrmDevice(
            os.path.basename(device), device, _read_int(os.path.join(device, "vendor"), 16) or 0,
            _read_int(os.path.join(device, "device"), 16) or 0, driver, tuple(nodes),
            _read(os.path.join(device, "boot_vga"), 8) == "1"))
    devices.sort(key=lambda d: d.pci)
    return tuple(devices)


def is_asleep(device: DrmDevice) -> bool:
    # 'suspending' is on its way down; touching it now would resume it.
    status = (_read(os.path.join(device.sysfs, "power", "runtime_status"), 64) or "").lower()
    return status in ("suspended", "suspending")

# }}}


# NVML {{{

NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_ERROR_FUNCTION_NOT_FOUND = 13
NVML_VALUE_NOT_AVAILABLE = (1 << 64) - 1


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _NvmlProcessInfoV1(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong)]


class _NvmlProcessInfoV2(ctypes.Structure):
    # nvmlProcessInfo_v2_t; the _v3 entry points use the same layout.
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


# Newest first. Each entry is (suffix, struct).
PROCESS_ABIS: tuple[tuple[str, type[ctypes.Structure]], ...] = (
    ("_v3", _NvmlProcessInfoV2), ("_v2", _NvmlProcessInfoV2), ("", _NvmlProcessInfoV1))
PROCESS_LISTS = ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses")


def _default_nvml_loader() -> Any:
    return ctypes.CDLL("libnvidia-ml.so.1")


# Replaced by tests with a loader that returns a fake library.
nvml_loader: Callable[[], Any] = _default_nvml_loader


class Nvml:
    """A minimal NVML client, opened for one snapshot and closed after it."""

    def __init__(self, lib: Any) -> None:
        self.lib = lib

    @classmethod
    def open(cls) -> Nvml | None:
        try:
            lib = nvml_loader()
        except (OSError, AttributeError):
            return None
        init = getattr(lib, "nvmlInit_v2", None) or getattr(lib, "nvmlInit", None)
        if init is None or init() != NVML_SUCCESS:
            return None
        return cls(lib)

    def close(self) -> None:
        try:
            self.lib.nvmlShutdown()
        except Exception:
            pass

    def handle(self, pci: str) -> ctypes.c_void_p | None:
        handle = ctypes.c_void_p()
        bus_id = pci.encode("ascii")
        for name in ("nvmlDeviceGetHandleByPciBusId_v2", "nvmlDeviceGetHandleByPciBusId"):
            fn = getattr(self.lib, name, None)
            if fn is not None and fn(bus_id, ctypes.byref(handle)) == NVML_SUCCESS:
                return handle
        return None

    def name(self, handle: ctypes.c_void_p) -> str:
        buffer = ctypes.create_string_buffer(96)
        get_name = getattr(self.lib, "nvmlDeviceGetName", None)
        if get_name is None or get_name(handle, buffer, ctypes.c_uint(96)) != NVML_SUCCESS:
            return ""
        return buffer.value.decode("utf-8", "replace")

    def memory(self, handle: ctypes.c_void_p) -> tuple[int, int, int] | None:
        """(total, used, free) bytes."""
        info = _NvmlMemory()
        if self.lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)) != NVML_SUCCESS:
            return None
        return int(info.total), int(info.used), int(info.free)

    def _process_list(self, base: str, handle: ctypes.c_void_p) -> dict[int, int] | None:
        for suffix, struct in PROCESS_ABIS:
            fn = getattr(self.lib, base + suffix, None)
            if fn is None:
                continue
            capacity = 32
            for _attempt in range(3):
                count = ctypes.c_uint(capacity)
                array = (struct * capacity)()
                result = fn(handle, ctypes.byref(count), array)
                if result == NVML_ERROR_INSUFFICIENT_SIZE:
                    capacity = max(capacity * 2, int(count.value) + 8)
                    continue
                break
            if result == NVML_ERROR_FUNCTION_NOT_FOUND:
                continue
            if result != NVML_SUCCESS:
                return None
            found: dict[int, int] = {}
            for entry in array[:min(int(count.value), capacity)]:
                used = int(entry.usedGpuMemory)
                if entry.pid > 0 and used != NVML_VALUE_NOT_AVAILABLE:
                    found[int(entry.pid)] = max(found.get(int(entry.pid), 0), used)
            return found
        return None

    def processes(self, handle: ctypes.c_void_p) -> dict[int, int]:
        """PID -> bytes on this device. A process in both the compute and the
        graphics list reports the same allocation in each, so take the larger."""
        found: dict[int, int] = {}
        for base in PROCESS_LISTS:
            for pid, used in (self._process_list(base, handle) or {}).items():
                found[pid] = max(found.get(pid, 0), used)
        return found

# }}}


# nvidia-smi {{{

def _smi(args: list[str]) -> list[list[str]] | None:
    command = shutil.which(NVIDIA_SMI) if os.sep not in NVIDIA_SMI else NVIDIA_SMI
    if not command:
        return None
    try:
        done = subprocess.run([command, *args, "--format=csv,noheader,nounits"], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=_SMI_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return [[cell.strip() for cell in line.split(",")] for line in done.stdout.splitlines() if line.strip()]


def smi_devices() -> tuple[dict[str, tuple[int, int, int]], dict[str, dict[int, int]]] | None:
    """({pci: (total, used, free)}, {pci: {pid: bytes}}) from nvidia-smi, in bytes."""
    rows = _smi(["--query-gpu=pci.bus_id,memory.total,memory.used,memory.free"])
    if rows is None:
        return None
    memory: dict[str, tuple[int, int, int]] = {}
    for row in rows:
        try:
            pci = normalise_pci(row[0])
            values = tuple(int(float(cell)) * MIB for cell in row[1:4])
        except (IndexError, ValueError):
            continue
        if pci and len(values) == 3:
            memory[pci] = values  # type: ignore[assignment]
    processes: dict[str, dict[int, int]] = {pci: {} for pci in memory}
    for row in _smi(["--query-compute-apps=gpu_bus_id,pid,used_memory"]) or ():
        try:
            pci, pid, used = normalise_pci(row[0]), int(row[1]), int(float(row[2])) * MIB
        except (IndexError, ValueError):
            continue
        if pci in processes and pid > 0:
            processes[pci][pid] = max(processes[pci].get(pid, 0), used)
    return memory, processes

# }}}


# DRM fdinfo {{{

_SIZE_UNITS = {"": 1, "B": 1, "KiB": KIB, "MiB": MIB, "GiB": GIB, "kB": KIB}
_MEMORY_PREFIXES = ("drm-resident-", "drm-total-", "drm-memory-")


def _size(text: str) -> int | None:
    parts = text.split()
    if not parts:
        return None
    try:
        value = int(parts[0])
    except ValueError:
        return None
    multiplier = _SIZE_UNITS.get(parts[1] if len(parts) > 1 else "")
    return None if multiplier is None else max(0, value) * multiplier


@dataclass(frozen=True)
class DrmClient:
    driver: str
    pdev: str
    client_id: str
    memory_bytes: int


def parse_drm_fdinfo(text: str) -> DrmClient | None:
    """Parse one /proc/<pid>/fdinfo/<fd> of a DRM file.

    Per memory region the resident size is preferred, then the total, then
    the legacy amdgpu ``drm-memory-``. When a client reports device-local
    memory (``vram*`` or ``local*``) only that counts."""
    driver = pdev = client_id = ""
    regions: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key == "drm-driver":
            driver = value
        elif key == "drm-pdev":
            pdev = value
        elif key == "drm-client-id":
            client_id = value
        else:
            for prefix in _MEMORY_PREFIXES:
                if key.startswith(prefix):
                    region = key[len(prefix):]
                    # drm-total-cycles-<engine> is time, not memory.
                    if region and not region.startswith("cycles-"):
                        size = _size(value)
                        if size is not None:
                            regions.setdefault(region, {})[prefix] = size
                    break
    if not driver:
        return None
    local = {name: values for name, values in regions.items() if name.startswith(("vram", "local"))}
    total = 0
    for values in (local or regions).values():
        for prefix in _MEMORY_PREFIXES:
            if prefix in values:
                total += values[prefix]
                break
    return DrmClient(driver, pdev, client_id, total)


def fdinfo_processes(nodes: Mapping[str, str]) -> dict[str, dict[int, int]]:
    """{pci: {pid: bytes}} for every readable process holding a DRM fd on ``nodes``
    (DRM node name -> PCI address). Duplicated and inherited fds share one client,
    which is counted once, for the first process seen holding it."""
    found: dict[str, dict[int, int]] = {pci: {} for pci in nodes.values()}
    if not nodes:
        return found
    proc = proc_root()
    seen: set[tuple[str, str]] = set()
    try:
        pids = sorted(int(name) for name in os.listdir(proc) if name.isdigit())
    except OSError:
        return found
    for pid in pids:
        fd_dir = os.path.join(proc, str(pid), "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if not target.startswith("/dev/dri/"):
                continue
            pci = nodes.get(target[len("/dev/dri/"):])
            if pci is None:
                continue
            client = parse_drm_fdinfo(_read(os.path.join(proc, str(pid), "fdinfo", fd), 16384) or "")
            if client is None:
                continue
            key = (client.pdev or pci, client.client_id or f"{pid}/{fd}")
            if key in seen:
                continue
            seen.add(key)
            found[pci][pid] = found[pci].get(pid, 0) + client.memory_bytes
    return found

# }}}


# Snapshot {{{

@dataclass(frozen=True)
class Device:
    """One GPU as measured now. Byte counts are None when not measured."""
    pci: str
    name: str
    vendor: str
    driver: str
    discrete: bool
    asleep: bool
    total_bytes: int | None
    used_bytes: int | None
    free_bytes: int | None
    # 'nvml', 'nvidia-smi', 'sysfs', 'asleep', 'held' (an awake NVIDIA GPU left
    # unread because another is asleep and waking was not asked for) or 'none'
    source: str
    processes: Mapping[int, int] = field(default_factory=dict)
    # The device has a VRAM counter, even if it was not read now (asleep or held).
    counter: bool = False

    @property
    def measured(self) -> bool:
        return self.free_bytes is not None


def _basic_name(device: DrmDevice) -> str:
    vendor = _VENDORS.get(device.vendor_id, f"{device.vendor_id:04x}")
    return f"{vendor} GPU {device.vendor_id:04x}:{device.device_id:04x}"


def has_counter(drm: DrmDevice) -> bool:
    """NVIDIA GPUs always have one; others when sysfs lists ``mem_info_vram_total``.
    Checking that the attribute exists does not reach the driver."""
    return drm.is_nvidia or os.path.exists(os.path.join(drm.sysfs, "mem_info_vram_total"))


def _device(drm: DrmDevice, source: str, *, name: str = "", memory: tuple[int, int, int] | None = None,
            processes: Mapping[int, int] | None = None, asleep: bool = False) -> Device:
    total, used, free = memory if memory is not None else (None, None, None)
    return Device(drm.pci, name or _basic_name(drm), _VENDORS.get(drm.vendor_id, f"{drm.vendor_id:04x}"),
                  drm.driver, drm.discrete, asleep, total, used, free, source, dict(processes or {}),
                  has_counter(drm))


def snapshot(*, wake: bool = False) -> tuple[Device, ...]:
    """Every GPU, with its memory and per-process memory where measurable.

    Without ``wake`` nothing is read from a sleeping GPU, and NVML is not
    opened while any NVIDIA GPU sleeps. With it, NVML is opened regardless,
    which resumes sleeping NVIDIA GPUs: a caller about to load a model asks
    for that."""
    drms = discover()
    slots: dict[str, Device] = {}
    nvidia: list[DrmDevice] = []
    nodes: dict[str, str] = {}
    nvidia_asleep = False
    for drm in drms:
        if drm.is_nvidia:
            nvidia.append(drm)
            nvidia_asleep = nvidia_asleep or is_asleep(drm)
            continue
        if is_asleep(drm):
            slots[drm.pci] = _device(drm, "asleep", asleep=True)
            continue
        total = _read_int(os.path.join(drm.sysfs, "mem_info_vram_total"))
        used = _read_int(os.path.join(drm.sysfs, "mem_info_vram_used"))
        if total is not None and used is not None:
            slots[drm.pci] = _device(drm, "sysfs", memory=(total, used, max(0, total - used)))
            for node in drm.nodes:
                nodes[node] = drm.pci
        else:
            slots[drm.pci] = _device(drm, "none")
    if nvidia:
        if nvidia_asleep and not wake:
            for drm in nvidia:
                slots[drm.pci] = _device(drm, "asleep", asleep=True) if is_asleep(drm) else _device(drm, "held")
        else:
            library = Nvml.open()
            if library is not None:
                try:
                    for drm in nvidia:
                        handle = library.handle(drm.pci)
                        if handle is None:
                            slots[drm.pci] = _device(drm, "none")
                            continue
                        memory = library.memory(handle)
                        slots[drm.pci] = _device(drm, "nvml" if memory else "none", name=library.name(handle),
                                                 memory=memory, processes=library.processes(handle))
                finally:
                    library.close()
            else:
                answer = smi_devices()
                for drm in nvidia:
                    if answer is not None and drm.pci in answer[0]:
                        slots[drm.pci] = _device(drm, "nvidia-smi", memory=answer[0][drm.pci],
                                                 processes=answer[1].get(drm.pci, {}))
                    else:
                        slots[drm.pci] = _device(drm, "none")
    for pci, processes in fdinfo_processes(nodes).items():
        current = slots[pci]
        slots[pci] = Device(current.pci, current.name, current.vendor, current.driver, current.discrete,
                            current.asleep, current.total_bytes, current.used_bytes, current.free_bytes,
                            current.source, processes, current.counter)
    return tuple(slots[drm.pci] for drm in drms)

# }}}
