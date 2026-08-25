#!/usr/bin/env python3
"""Emit a coarse, redacted hardware observation without network or privilege."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import select
import signal
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NEVER_COLLECTED = [
    "asset_tag",
    "hostname",
    "ip_address",
    "mac_address",
    "machine_id",
    "serial_number",
    "system_uuid",
    "username",
]
VENDORS = {
    "0x1002": "amd",
    "0x10de": "nvidia",
    "0x8086": "intel",
}
PROBE_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"


def _read_bytes(path: Path, limit: int = 4096) -> bytes | None:
    if limit < 0:
        return None
    try:
        with path.open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            value = handle.read(limit + 1)
    except OSError:
        return None
    return value if len(value) <= limit else None


def _read_text(path: Path, limit: int = 4096) -> str | None:
    value = _read_bytes(path, limit)
    if value is None:
        return None
    try:
        return value.decode("ascii", errors="strict").strip()
    except UnicodeError:
        return None


def _read_int(path: Path, *, minimum: int | None = None, maximum: int | None = None) -> int | None:
    raw = _read_text(path)
    if raw is None or re.fullmatch(r"-?[0-9]+", raw) is None:
        return None
    value = int(raw)
    if minimum is not None and value < minimum:
        return None
    if maximum is not None and value > maximum:
        return None
    return value


def _find_executable(command: str) -> str | None:
    if re.fullmatch(r"[a-zA-Z0-9_.+-]{1,80}", command) is None:
        return None
    for root in PROBE_PATH.split(":"):
        candidate = Path(root) / command
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return str(resolved)
    return None


def _safe_token(value: str | None, limit: int = 80) -> str | None:
    if value is None or len(value) > limit:
        return None
    return value if re.fullmatch(r"[a-zA-Z0-9_.+-]+", value) else None


def _pci_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower().removeprefix("0x")
    return normalized if re.fullmatch(r"[0-9a-f]{4}", normalized) else None


def _cpu_list_count(raw: str | None) -> int | None:
    if not raw:
        return None
    selected: set[int] = set()
    for part in raw.split(","):
        fields = part.strip().split("-")
        if len(fields) == 1 and fields[0].isdigit():
            start = end = int(fields[0])
        elif len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            start, end = int(fields[0]), int(fields[1])
        else:
            return None
        if start > end or end > 65535 or end - start > 16384:
            return None
        selected.update(range(start, end + 1))
        if len(selected) > 16384:
            return None
    return len(selected) or None


def _memory() -> tuple[dict[str, int | None], str]:
    values: dict[str, int] = {}
    text = _read_text(Path("/proc/meminfo"), 128 * 1024)
    if text:
        for line in text.splitlines():
            name, separator, raw = line.partition(":")
            if not separator:
                continue
            fields = raw.strip().split()
            if len(fields) != 2 or fields[1] != "kB" or not fields[0].isdigit():
                continue
            if name in {
                "MemAvailable",
                "MemTotal",
                "Hugepagesize",
                "SwapFree",
                "SwapTotal",
            }:
                values[name] = int(fields[0]) * 1024
    huge_total_count = _read_int(Path("/proc/sys/vm/nr_hugepages"), minimum=0)
    huge_free_count = values.get("HugePages_Free")
    # HugePages_Total/Free lines have a count rather than a kB suffix.
    if text:
        for line in text.splitlines():
            name, separator, raw = line.partition(":")
            if not separator or name not in {"HugePages_Total", "HugePages_Free"}:
                continue
            raw_count = raw.strip()
            if raw_count.isdigit():
                if name == "HugePages_Total":
                    huge_total_count = int(raw_count)
                else:
                    huge_free_count = int(raw_count)
    raw_limit = _read_text(Path("/sys/fs/cgroup/memory.max"))
    total = values.get("MemTotal")
    if raw_limit == "max":
        limit = None
        limit_status = "unlimited"
    elif raw_limit and raw_limit.isdigit() and int(raw_limit) > 0:
        limit = int(raw_limit)
        limit_status = "observed"
    else:
        limit = None
        limit_status = "unknown"
    if limit is not None and total is not None and limit >= total:
        limit = None
        limit_status = "unlimited"
    huge_size = values.get("Hugepagesize")
    memory = {
        "total_bytes": total,
        "available_bytes": values.get("MemAvailable"),
        "effective_limit_bytes": limit,
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
        "hugepage_size_bytes": huge_size,
        "hugepage_total_bytes": None
        if huge_size is None or huge_total_count is None
        else huge_size * huge_total_count,
        "hugepage_free_bytes": None
        if huge_size is None or huge_free_count is None
        else huge_size * huge_free_count,
        "numa_nodes": _numa_node_count(),
    }
    return memory, limit_status


def _cpu_quota() -> tuple[float | None, str]:
    raw = _read_text(Path("/sys/fs/cgroup/cpu.max"))
    if not raw:
        return None, "unknown"
    fields = raw.split()
    if len(fields) != 2 or not fields[1].isdigit() or int(fields[1]) <= 0:
        return None, "unknown"
    if fields[0] == "max":
        return None, "unlimited"
    if not fields[0].isdigit() or int(fields[0]) <= 0:
        return None, "unknown"
    return int(fields[0]) / int(fields[1]), "observed"


def _effective_cpus(*values: int | float | None) -> int | float | None:
    candidates = [float(value) for value in values if value is not None and value > 0]
    if not candidates:
        return None
    effective = min(candidates)
    return int(effective) if effective.is_integer() else effective


def _numa_node_count() -> int | None:
    nodes = [
        path
        for path in Path("/sys/devices/system/node").glob("node*")
        if re.fullmatch(r"node[0-9]+", path.name)
    ]
    return len(nodes) or None


def _cpu_topology() -> tuple[int | None, int | None, str]:
    packages: set[int] = set()
    cores: set[tuple[int, int]] = set()
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu*")):
        if re.fullmatch(r"cpu[0-9]+", cpu.name) is None:
            continue
        package = _read_int(cpu / "topology" / "physical_package_id", minimum=0)
        core = _read_int(cpu / "topology" / "core_id", minimum=0)
        if package is not None:
            packages.add(package)
        if package is not None and core is not None:
            cores.add((package, core))
    return len(packages) or None, len(cores) or None, "observed" if cores else "unknown"


def _cpu_frequency() -> dict[str, int | None]:
    current: list[int] = []
    hardware_minimum: list[int] = []
    hardware_maximum: list[int] = []
    for policy in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        if re.fullmatch(r"policy[0-9]+", policy.name) is None:
            continue
        for path, target in (
            (policy / "scaling_cur_freq", current),
            (policy / "cpuinfo_min_freq", hardware_minimum),
            (policy / "cpuinfo_max_freq", hardware_maximum),
        ):
            value_khz = _read_int(path, minimum=1, maximum=100_000_000)
            if value_khz is not None:
                target.append(value_khz * 1000)
    return {
        "current_min_hz": min(current) if current else None,
        "current_max_hz": max(current) if current else None,
        "hardware_min_hz": min(hardware_minimum) if hardware_minimum else None,
        "hardware_max_hz": max(hardware_maximum) if hardware_maximum else None,
    }


def _cache_size(raw: str | None) -> int | None:
    if raw is None:
        return None
    match = re.fullmatch(r"([0-9]+)([KMG])", raw)
    if not match:
        return None
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    size = int(match.group(1)) * scale
    return size if 0 < size <= 16 * 1024**3 else None


def _cpu_cache_totals() -> dict[str, int | None]:
    totals = {
        "l1_data_bytes": 0,
        "l1_instruction_bytes": 0,
        "l2_bytes": 0,
        "l3_bytes": 0,
    }
    observed = {name: False for name in totals}
    seen: set[tuple[str, str, str]] = set()
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu*")):
        if re.fullmatch(r"cpu[0-9]+", cpu.name) is None:
            continue
        for cache in sorted((cpu / "cache").glob("index*")):
            if re.fullmatch(r"index[0-9]+", cache.name) is None:
                continue
            level = _read_text(cache / "level")
            kind = (_read_text(cache / "type") or "").lower()
            size = _cache_size(_read_text(cache / "size"))
            cache_id = _read_text(cache / "shared_cpu_list") or _read_text(cache / "id")
            if level is None or size is None or cache_id is None:
                continue
            identity = (level, kind, cache_id)
            if identity in seen:
                continue
            seen.add(identity)
            if level == "1" and kind == "data":
                field = "l1_data_bytes"
            elif level == "1" and kind == "instruction":
                field = "l1_instruction_bytes"
            elif level == "2" and kind == "unified":
                field = "l2_bytes"
            elif level == "3" and kind == "unified":
                field = "l3_bytes"
            else:
                continue
            totals[field] += size
            observed[field] = True
    return {name: value if observed[name] else None for name, value in totals.items()}


def _isa_features() -> list[str]:
    text = _read_text(Path("/proc/cpuinfo"), 4 * 1024 * 1024)
    raw_features: set[str] = set()
    if text:
        for line in text.splitlines():
            name, separator, raw = line.partition(":")
            if separator and name.strip() in {"Features", "flags"}:
                raw_features.update(raw.strip().lower().split())
                break
    mapping = {
        "amx_tile": "amx",
        "asimd": "asimd",
        "avx": "avx",
        "avx2": "avx2",
        "avx512f": "avx512",
        "avx_vnni": "vnni",
        "fma": "fma",
        "sve": "sve",
        "vnni": "vnni",
    }
    return sorted({normalized for raw, normalized in mapping.items() if raw in raw_features})


def _run_bounded(executable: str, arguments: list[str]) -> tuple[int | None, bytes | None]:
    """Run fixed argv with a five-second and 64-KiB stdout boundary."""
    try:
        process = subprocess.Popen(
            [executable, *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": PROBE_PATH},
            start_new_session=True,
        )
    except OSError:
        return None, None
    output = bytearray()
    deadline = time.monotonic() + 5
    try:
        if process.stdout is None:
            raise OSError("stdout pipe unavailable")
        descriptor = process.stdout.fileno()
        os.set_blocking(descriptor, False)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired([executable, *arguments], 5)
            readable, _, _ = select.select([descriptor], [], [], remaining)
            if not readable:
                raise subprocess.TimeoutExpired([executable, *arguments], 5)
            chunk = os.read(descriptor, min(8192, 64 * 1024 + 1 - len(output)))
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > 64 * 1024:
                raise OSError("stdout exceeds boundary")
        return process.wait(timeout=max(0.001, deadline - time.monotonic())), bytes(output)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
        process.wait()
        return None, None
    finally:
        if process.stdout is not None:
            process.stdout.close()


def _command_probe(command: str, arguments: list[str]) -> tuple[str, str, str | None]:
    executable = _find_executable(command)
    if executable is None:
        return "unknown", "command-unavailable", None
    returncode, stdout = _run_bounded(executable, arguments)
    if returncode is None or stdout is None:
        return "unknown", "unknown", None
    if returncode != 0:
        return "unknown", "executable-probe", None
    first_line = stdout.decode("utf-8", errors="replace").splitlines()[:1]
    version = _safe_token(first_line[0].strip()) if first_line else None
    return "available", "executable-probe", version


def _nvidia_probe(device: Path) -> tuple[str, str, str | None]:
    """Bind an nvidia-smi result to one sysfs PCI device without emitting its BDF."""
    executable = _find_executable("nvidia-smi")
    if executable is None:
        return "unknown", "command-unavailable", None
    returncode, stdout = _run_bounded(
        executable,
        [
            "--query-gpu=pci.bus_id,driver_version",
            "--format=csv,noheader,nounits",
        ],
    )
    if returncode is None or stdout is None:
        return "unknown", "unknown", None
    if returncode != 0:
        return "unknown", "executable-probe", None
    try:
        local_bdf = device.resolve(strict=True).name.lower()
    except OSError:
        return "unknown", "unknown", None
    if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", local_bdf) is None:
        return "unknown", "unknown", None
    local_suffix = local_bdf[-12:]
    matches: list[str | None] = []
    for raw_line in stdout.decode("utf-8", errors="replace").splitlines():
        raw_bdf, separator, raw_version = raw_line.partition(",")
        normalized_bdf = raw_bdf.strip().lower()
        if (
            separator
            and re.fullmatch(
                r"(?:[0-9a-f]{8}|[0-9a-f]{4}):[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",
                normalized_bdf,
            )
            and normalized_bdf[-12:] == local_suffix
        ):
            matches.append(_safe_token(raw_version.strip()))
    if len(matches) != 1:
        return "unknown", "contradictory", None
    return "available", "executable-probe", matches[0]


def _render_access(device: Path, device_root: Path = Path("/dev/dri")) -> bool | None:
    drm = device / "drm"
    try:
        names = sorted(path.name for path in drm.glob("renderD*"))
    except OSError:
        return None
    if not names:
        return None
    return any(os.access(device_root / name, os.R_OK | os.W_OK) for name in names)


def _pcie_speed(path: Path) -> float | None:
    raw = _read_text(path)
    if not raw:
        return None
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s+GT/s", raw)
    if not match:
        return None
    value = float(match.group(1))
    return value if 0 < value <= 128 else None


def _pcie_width(path: Path) -> int | None:
    raw = _read_text(path)
    if not raw:
        return None
    match = re.match(r"^([0-9]+)$", raw)
    if not match:
        return None
    value = int(match.group(1))
    return value if 1 <= value <= 32 else None


def _gpu_inventory(
    drm_root: Path = Path("/sys/class/drm"),
    iommu_root: Path = Path("/sys/kernel/iommu_groups"),
    device_root: Path = Path("/dev/dri"),
) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for card in sorted(drm_root.glob("card*")):
        if re.fullmatch(r"card[0-9]+", card.name) is None:
            continue
        device = card / "device"
        try:
            resolved = device.resolve(strict=True)
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        raw_vendor = (_read_text(device / "vendor") or "").lower()
        raw_device = (_read_text(device / "device") or "").lower()
        vendor = VENDORS.get(raw_vendor, "other" if raw_vendor else "unknown")
        try:
            driver = _safe_token((device / "driver").resolve(strict=True).name)
        except OSError:
            driver = None
        backends: list[dict[str, Any]] = []
        if vendor == "nvidia":
            status, evidence, version = _nvidia_probe(device)
            backends.append({"name": "cuda", "status": status, "evidence": evidence, "version": version})
        if vendor == "amd":
            status, evidence, _ = _command_probe("rocminfo", [])
            if status == "available":
                # Global enumeration is not proof that this device owns the backend.
                status, evidence = "unknown", "contradictory"
            backends.append({"name": "rocm", "status": status, "evidence": evidence, "version": None})
        for name, command, arguments in (
            ("opencl", "clinfo", ["--list"]),
            ("vulkan", "vulkaninfo", ["--summary"]),
        ):
            status, evidence, _ = _command_probe(command, arguments)
            if status == "available":
                # A global tool success does not prove which enumerated device owns the backend.
                status, evidence = "unknown", "contradictory"
            backends.append({"name": name, "status": status, "evidence": evidence, "version": None})
        backends.sort(key=lambda item: item["name"])
        devices.append(
            {
                "index": len(devices),
                "vendor": vendor,
                "vendor_id": _pci_id(raw_vendor),
                "device_id": _pci_id(raw_device),
                "device_class": "unknown",
                "kernel_driver": driver,
                "render_access": _render_access(device, device_root),
                "vram_bytes": None,
                "memory_kind": "unknown",
                "shared_memory_bytes": None,
                "numa_node": _read_int(device / "numa_node", minimum=-1),
                "iommu_group_present": (device / "iommu_group").exists()
                if iommu_root.exists()
                else None,
                "pcie": {
                    "current_width": _pcie_width(device / "current_link_width"),
                    "maximum_width": _pcie_width(device / "max_link_width"),
                    "current_gtps": _pcie_speed(device / "current_link_speed"),
                    "maximum_gtps": _pcie_speed(device / "max_link_speed"),
                },
                "backends": backends,
            }
        )
    return devices


def _micro_value(path: Path) -> float | None:
    value = _read_int(path, minimum=0)
    return None if value is None else value / 1_000_000


def _power(power_root: Path = Path("/sys/class/power_supply")) -> dict[str, Any]:
    ac_values: list[bool] = []
    batteries: list[dict[str, Any]] = []
    root_accessible = power_root.is_dir()
    for supply in sorted(power_root.glob("*")):
        kind = _read_text(supply / "type")
        if kind in {"Mains", "USB", "USB_C", "USB_PD"}:
            raw = _read_text(supply / "online")
            if raw in {"0", "1"}:
                ac_values.append(raw == "1")
        elif kind == "Battery":
            percent = _read_int(supply / "capacity", minimum=0, maximum=100)
            raw_status = (_read_text(supply / "status") or "").strip().lower()
            status = {
                "charging": "charging",
                "discharging": "discharging",
                "full": "full",
                "not charging": "not-charging",
            }.get(raw_status, "unknown")
            energy_full = _micro_value(supply / "energy_full")
            energy_design = _micro_value(supply / "energy_full_design")
            wear = None
            if energy_full is not None and energy_design and energy_design > 0:
                wear = round(max(0.0, min(100.0, 100.0 * (1.0 - energy_full / energy_design))), 2)
            batteries.append(
                {
                    "index": len(batteries),
                    "status": status,
                    "percent": percent,
                    "energy_full_wh": energy_full,
                    "energy_design_wh": energy_design,
                    "power_watts": _micro_value(supply / "power_now"),
                    "wear_percent": wear,
                }
            )
    percentages = [battery["percent"] for battery in batteries]
    battery_percent = (
        min(percentages)
        if percentages and all(value is not None for value in percentages)
        else None
    )
    return {
        "ac_online": any(ac_values) if ac_values else None,
        "battery_present": bool(batteries) if root_accessible else None,
        "battery_percent": battery_percent,
        "batteries": batteries,
    }


def _platform_state() -> dict[str, str]:
    efi_root = Path("/sys/firmware/efi")
    firmware_root = Path("/sys/firmware")
    firmware_mode = "uefi" if efi_root.is_dir() else "bios" if firmware_root.is_dir() else "unknown"
    secure_boot = "unknown"
    try:
        variables = sorted((efi_root / "efivars").glob("SecureBoot-*"))
    except OSError:
        variables = []
    for variable in variables[:1]:
        payload = _read_bytes(variable, 64)
        if payload is None:
            continue
        if len(payload) >= 5 and payload[4] in {0, 1}:
            secure_boot = "enabled" if payload[4] == 1 else "disabled"
    iommu_root = Path("/sys/kernel/iommu_groups")
    if not iommu_root.is_dir():
        iommu = "unknown"
    else:
        try:
            iommu = "enabled" if any(path.is_dir() for path in iommu_root.iterdir()) else "unknown"
        except OSError:
            iommu = "unknown"
    return {
        "firmware_mode": firmware_mode,
        "secure_boot": secure_boot,
        "iommu": iommu,
        "dmi_access": "not-probed",
    }


def _bus_counts() -> dict[str, int | None]:
    try:
        pci_devices = [
            path
            for path in Path("/sys/bus/pci/devices").iterdir()
            if (path / "vendor").is_file()
        ]
        pci = len(pci_devices)
        storage = sum(
            1
            for path in pci_devices
            if (_read_text(path / "class") or "").lower().startswith("0x01")
        )
    except OSError:
        pci = None
        storage = None
    try:
        usb = sum(
            1
            for path in Path("/sys/bus/usb/devices").iterdir()
            if (path / "idVendor").is_file() and (path / "idProduct").is_file()
        )
    except OSError:
        usb = None
    return {
        "pci_devices_count": pci,
        "usb_devices_count": usb,
        "storage_controllers_count": storage,
    }


def _network(network_root: Path = Path("/sys/class/net")) -> dict[str, Any]:
    interfaces: list[dict[str, Any]] = []
    root_accessible = network_root.is_dir()
    for interface in sorted(network_root.glob("*")):
        # The name is used only to omit the loopback device and is never emitted.
        if interface.name == "lo":
            continue
        raw_type = _read_int(interface / "type", minimum=0)
        if (interface / "wireless").exists():
            interface_type = "wireless"
        elif raw_type == 1:
            interface_type = "ethernet"
        elif raw_type is None:
            interface_type = "unknown"
        else:
            interface_type = "other"
        raw_state = (_read_text(interface / "operstate") or "").lower()
        online = True if raw_state == "up" else False if raw_state in {"down", "dormant"} else None
        speed = _read_int(interface / "speed", minimum=0, maximum=1_000_000)
        device = interface / "device"
        try:
            driver = _safe_token((device / "driver").resolve(strict=True).name)
        except OSError:
            driver = None
        try:
            subsystem = (device / "subsystem").resolve(strict=True).name.lower()
        except OSError:
            subsystem = ""
        if subsystem == "pci":
            bus = "pci"
        elif subsystem == "usb":
            bus = "usb"
        elif not device.exists():
            bus = "virtual"
        elif subsystem:
            bus = "other"
        else:
            bus = "unknown"
        interfaces.append(
            {
                "index": len(interfaces),
                "type": interface_type,
                "online": online,
                "link_mbps": speed,
                "driver": driver,
                "bus": bus,
            }
        )
    states = [interface["online"] for interface in interfaces]
    if any(state is True for state in states):
        offline = False
    elif interfaces and all(state is False for state in states):
        offline = True
    elif not interfaces and root_accessible:
        offline = True
    else:
        offline = None
    return {"interfaces": interfaces, "offline": offline}


def _thermal() -> dict[str, int | float | str | None]:
    thermal_root = Path("/sys/class/thermal")
    temperatures: list[float] = []
    temperature_zones = 0
    for zone in sorted(thermal_root.glob("thermal_zone*")):
        if re.fullmatch(r"thermal_zone[0-9]+", zone.name) is None:
            continue
        temperature_zones += 1
        raw = _read_int(zone / "temp", minimum=-100_000, maximum=250_000)
        if raw is not None:
            temperatures.append(raw / 1000.0)
    fan_count = 0
    fan_observed = False
    for fan in sorted(Path("/sys/class/hwmon").glob("hwmon*/fan*_input")):
        if re.fullmatch(r"fan[0-9]+_input", fan.name) is None:
            continue
        if _read_int(fan, minimum=0, maximum=200_000) is not None:
            fan_count += 1
            fan_observed = True
    return {
        "sensor_count": len(temperatures)
        if temperatures or (thermal_root.is_dir() and temperature_zones == 0)
        else None,
        "maximum_celsius": max(temperatures) if temperatures else None,
        "fan_count": fan_count if fan_observed else None,
        "throttle": "unknown",
    }


def _systemd_virtualization(executable: str) -> str:
    container_status, _ = _run_bounded(executable, ["--container", "--quiet"])
    if container_status == 0:
        return "container"
    if container_status != 1:
        return "unknown"
    virtual_status, _ = _run_bounded(executable, ["--vm", "--quiet"])
    if virtual_status == 0:
        return "virtual-machine"
    return "none" if virtual_status == 1 else "unknown"


def _virtualization() -> str:
    if Path("/run/systemd/container").exists() or Path("/.dockerenv").exists():
        return "container"
    executable = _find_executable("systemd-detect-virt")
    return _systemd_virtualization(executable) if executable else "unknown"


def _capability_id(document: dict[str, Any]) -> str:
    total = document["memory"]["total_bytes"]
    memory_bucket = None if total is None else total // (4 * 1024**3)
    material = {
        "architecture": document["cpu"]["architecture"],
        "cpu_bucket": None if document["cpu"]["effective_cpus"] is None else int(document["cpu"]["effective_cpus"]),
        "gpu": [
            {
                "backend": [(item["name"], item["status"]) for item in gpu["backends"]],
                "class": gpu["device_class"],
                "vendor": gpu["vendor"],
                "vram_bucket": None if gpu["vram_bytes"] is None else gpu["vram_bytes"] // (2 * 1024**3),
            }
            for gpu in document["gpus"]
        ],
        "memory_bucket": memory_bucket,
        "virtualization": document["virtualization"],
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"local:capability-{digest[:16]}"


def collect(scope: str) -> dict[str, Any]:
    if scope not in {"inventory", "gpu"}:
        raise ValueError("unsupported hardware observation scope")
    machine = platform.machine().lower()
    architecture = machine if machine in {"x86_64", "aarch64"} else "other" if machine else "unknown"
    logical = os.cpu_count()
    if not isinstance(logical, int) or logical < 1:
        logical = None
    online = _cpu_list_count(_read_text(Path("/sys/devices/system/cpu/online")))
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    cpuset = _cpu_list_count(_read_text(Path("/sys/fs/cgroup/cpuset.cpus.effective")))
    quota, quota_status = _cpu_quota()
    packages, physical_cores, topology_status = _cpu_topology()
    numa_nodes = _numa_node_count()
    isa_features = _isa_features()
    frequency = _cpu_frequency()
    cache_totals = _cpu_cache_totals()
    if online is not None and physical_cores is not None:
        smt = "enabled" if online > physical_cores else "disabled" if online == physical_cores else "unknown"
    else:
        smt = "unknown"
    memory, memory_limit_status = _memory()
    gpus = _gpu_inventory()
    platform_state = _platform_state()
    buses = _bus_counts()
    network = _network()
    thermal = _thermal()
    unknowns: set[str] = set()
    cpu_fields = {
        "logical": logical,
        "online": online,
        "affinity": affinity,
        "cpuset": cpuset,
        "physical-cores": physical_cores,
        "packages": packages,
        "numa": numa_nodes,
    }
    for name, value in cpu_fields.items():
        if value is None:
            unknowns.add(f"cpu.{name}")
    if quota_status == "unknown":
        unknowns.add("cpu.quota")
    if not isa_features:
        unknowns.add("cpu.isa")
    for name, value in frequency.items():
        if value is None:
            unknowns.add("cpu.frequency." + name.removesuffix("_hz").replace("_", "-"))
    for name, value in cache_totals.items():
        if value is None:
            unknowns.add("cpu.cache." + name.removesuffix("_bytes").replace("_", "-"))
    if smt == "unknown":
        unknowns.add("cpu.smt")
    if memory["available_bytes"] is None:
        unknowns.add("memory.available")
    if memory_limit_status == "unknown":
        unknowns.add("memory.effective-limit")
    for name in (
        "swap_total_bytes",
        "swap_free_bytes",
        "hugepage_size_bytes",
        "hugepage_total_bytes",
        "hugepage_free_bytes",
        "numa_nodes",
    ):
        if memory[name] is None:
            unknowns.add("memory." + name.removesuffix("_bytes").replace("_", "-"))
    for gpu in gpus:
        for field in ("vendor_id", "device_id", "kernel_driver"):
            if gpu[field] is None:
                unknowns.add(f"gpu.{gpu['index']}.{field.replace('_', '-')}")
        if gpu["device_class"] == "unknown":
            unknowns.add(f"gpu.{gpu['index']}.class")
        if gpu["render_access"] is None:
            unknowns.add(f"gpu.{gpu['index']}.render-access")
        if gpu["vram_bytes"] is None:
            unknowns.add(f"gpu.{gpu['index']}.vram")
        if gpu["memory_kind"] == "unknown":
            unknowns.add(f"gpu.{gpu['index']}.memory-kind")
        if gpu["shared_memory_bytes"] is None:
            unknowns.add(f"gpu.{gpu['index']}.shared-memory")
        if gpu["numa_node"] is None:
            unknowns.add(f"gpu.{gpu['index']}.numa")
        if gpu["iommu_group_present"] is None:
            unknowns.add(f"gpu.{gpu['index']}.iommu")
        if any(value is None for value in gpu["pcie"].values()):
            unknowns.add(f"gpu.{gpu['index']}.pcie")
        for backend in gpu["backends"]:
            if backend["status"] == "unknown":
                unknowns.add(f"gpu.{gpu['index']}.{backend['name']}")
    if not gpus:
        unknowns.add("gpu.inventory")
    power = _power()
    if power["ac_online"] is None:
        unknowns.add("power.ac")
    if power["battery_present"] is None:
        unknowns.add("power.battery-present")
    for battery in power["batteries"]:
        for field in (
            "percent",
            "energy_full_wh",
            "energy_design_wh",
            "power_watts",
            "wear_percent",
        ):
            if battery[field] is None:
                unknowns.add(f"power.battery.{battery['index']}.{field.replace('_', '-')}")
        if battery["status"] == "unknown":
            unknowns.add(f"power.battery.{battery['index']}.status")
    for field in ("firmware_mode", "secure_boot", "iommu"):
        if platform_state[field] == "unknown":
            unknowns.add("platform." + field.replace("_", "-"))
    unknowns.add("platform.dmi")
    for field, value in buses.items():
        if value is None:
            unknowns.add("buses." + field.removesuffix("_count").replace("_", "-"))
    for interface in network["interfaces"]:
        for field in ("online", "link_mbps", "driver"):
            if interface[field] is None:
                unknowns.add(f"network.{interface['index']}.{field.replace('_', '-')}")
        if interface["type"] == "unknown":
            unknowns.add(f"network.{interface['index']}.type")
        if interface["bus"] == "unknown":
            unknowns.add(f"network.{interface['index']}.bus")
    if network["offline"] is None:
        unknowns.add("network.offline")
    for field in ("sensor_count", "maximum_celsius", "fan_count"):
        if thermal[field] is None:
            unknowns.add("thermal." + field.replace("_", "-"))
    if thermal["throttle"] == "unknown":
        unknowns.add("thermal.throttle")
    document: dict[str, Any] = {
        "schema": "plebian.hardware/v1",
        "snapshot_id": "local:pending",
        "capture": {
            "source": "live-probe",
            "scope": scope,
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "redaction": "default",
            "network_used": False,
            "privileged": False,
            "qualification_eligible": False,
        },
        "never_collected": NEVER_COLLECTED,
        "privacy": {
            "classification": "fingerprinting-grade-local",
            "telemetry_eligible": False,
            "export_requires_warning": True,
            "cache_mode": "0600",
        },
        "virtualization": _virtualization(),
        "cpu": {
            "architecture": architecture,
            "model_bucket": "unknown",
            "logical_cpus": logical,
            "online_cpus": online,
            "affinity_cpus": affinity,
            "cpuset_cpus": cpuset,
            "quota_cpus": quota,
            "effective_cpus": _effective_cpus(logical, online, affinity, cpuset, quota),
            "packages": packages,
            "physical_cores": physical_cores,
            "numa_nodes": numa_nodes,
            "smt": smt,
            "isa_features": isa_features,
            "frequency_hz": frequency,
            "cache_bytes": cache_totals,
        },
        "memory": memory,
        "gpus": gpus,
        "platform": platform_state,
        "buses": buses,
        "storage": {
            "scope": "model-store",
            "free_bytes": None,
            "total_bytes": None,
            "filesystem_type": None,
            "read_only": None,
        },
        "network": network,
        "power": power,
        "thermal": thermal,
        "pressure": {
            name: Path(f"/proc/pressure/{name}").is_file() for name in ("cpu", "memory", "io")
        },
        "evidence": [
            {"field_prefix": "buses", "source": "sysfs", "status": "partial", "confidence": "observed"},
            {"field_prefix": "cpu.affinity", "source": "python-runtime", "status": "observed", "confidence": "observed"},
            {"field_prefix": "cpu.cache", "source": "sysfs", "status": "observed" if all(value is not None for value in cache_totals.values()) else "partial", "confidence": "observed"},
            {"field_prefix": "cpu.cgroup", "source": "cgroup", "status": "partial", "confidence": "observed"},
            {"field_prefix": "cpu.frequency", "source": "sysfs", "status": "observed" if all(value is not None for value in frequency.values()) else "partial", "confidence": "observed"},
            {"field_prefix": "cpu.isa", "source": "procfs", "status": "observed" if isa_features else "unknown", "confidence": "observed" if isa_features else "unknown"},
            {"field_prefix": "cpu.topology", "source": "sysfs", "status": topology_status, "confidence": "observed" if topology_status == "observed" else "unknown"},
            {"field_prefix": "gpus", "source": "sysfs", "status": "partial", "confidence": "observed"},
            {"field_prefix": "memory", "source": "procfs", "status": "partial", "confidence": "observed"},
            {"field_prefix": "network", "source": "sysfs", "status": "partial", "confidence": "observed"},
            {"field_prefix": "platform", "source": "sysfs", "status": "partial", "confidence": "observed"},
            {"field_prefix": "power", "source": "sysfs", "status": "partial", "confidence": "observed"},
            {"field_prefix": "pressure", "source": "procfs", "status": "observed", "confidence": "observed"},
            {"field_prefix": "storage", "source": "sysfs", "status": "unknown", "confidence": "unknown"},
            {"field_prefix": "thermal", "source": "sysfs", "status": "partial", "confidence": "observed"},
            {"field_prefix": "virtualization", "source": "command", "status": "observed", "confidence": "observed"},
        ],
        "unknowns": sorted(
            unknowns
            | {
                "cpu.model-bucket",
                "storage.filesystem-type",
                "storage.free",
                "storage.read-only",
                "storage.total",
            }
        ),
    }
    document["snapshot_id"] = _capability_id(document)
    return document
