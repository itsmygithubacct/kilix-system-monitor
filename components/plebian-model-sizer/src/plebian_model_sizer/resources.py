"""Read-only, bounded resource observations using the hardware provider's probes.

This additive development snapshot does not change plebian.hardware/v1. No
identifiers, cgroup paths, filesystem paths or command output are returned.
"""
from __future__ import annotations

import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from plebian_hardware import probe

MIB = 1024**2
GIB = 1024**3
SCHEMA = "plebian.models.resources/v1-development"


def cgroup_headroom(proc: Path = Path("/proc")) -> tuple[int | None, str]:
    """Minimum remaining hard limit over visible cgroup-v2 ancestors.

    Hidden ancestors, v1/hybrid hierarchies and unreadable counters remain
    unknown. A host-root cgroup has no memory.max/current by kernel design.
    """
    membership = probe._read_text(proc / "self/cgroup", 65536)
    mounts = probe._read_text(proc / "self/mountinfo", 1024 * 1024)
    if membership is None or mounts is None:
        return None, "unknown"
    rows = membership.splitlines()
    if len(rows) != 1 or not rows[0].startswith("0::/"):
        return None, "unknown"
    relative = PurePosixPath(rows[0][3:])
    if ".." in relative.parts:
        return None, "unknown"
    candidates = []
    for row in mounts.splitlines():
        left, sep, right = row.partition(" - ")
        parts = left.split()
        if sep and right.split()[0] == "cgroup2" and len(parts) >= 6:
            # Escaped mount paths need decoding; refusing is safer than guessing.
            if parts[3] == "/" and "\\" not in parts[4]:
                candidates.append(Path(parts[4]))
    if len(candidates) != 1:
        return None, "unknown"
    mount = candidates[0]
    current = mount.joinpath(*relative.parts[1:])
    if not current.is_dir():
        return None, "unknown"
    limits = []
    while current != mount:
        maximum = probe._read_text(current / "memory.max")
        used = probe._read_int(current / "memory.current", minimum=0)
        if maximum is None or used is None:
            return None, "unknown"
        if maximum != "max":
            if not maximum.isdigit() or len(maximum) > 20:
                return None, "unknown"
            limits.append(max(0, int(maximum) - used))
        current = current.parent
    # A non-root cgroup namespace can conceal host limits. Its mount root is
    # also '/'; inode 1 is the kernel cgroup2 hierarchy root on Linux.
    try:
        if mount.stat().st_ino != 1:
            return None, "unknown"
    except OSError:
        return None, "unknown"
    return (min(limits), "limited") if limits else (None, "unlimited")


def parse_nvidia(text: str) -> list[dict]:
    devices = []
    for row in text.splitlines():
        fields = [part.strip() for part in row.split(",")]
        if len(fields) != 3 or any(re.fullmatch(r"[0-9]{1,12}", p) is None for p in fields):
            return []
        index, total, free = map(int, fields)
        if index > 255 or not 0 <= free <= total or total == 0:
            return []
        if any(device["index"] == index for device in devices):
            return []
        devices.append({"index": index, "backend": "cuda", "total_bytes": total * MIB,
                        "available_bytes": free * MIB, "evidence": "nvidia-smi"})
    return devices


def gpu_headroom() -> list[dict]:
    executable = probe._find_executable("nvidia-smi")
    if executable is None:
        return []
    code, output = probe._run_bounded(executable, [
        "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"])
    if code != 0 or output is None:
        return []
    try:
        return parse_nvidia(output.decode("ascii"))
    except UnicodeError:
        return []


def storage_headroom(path: Path) -> int | None:
    """Inspect the nearest existing ancestor without creating user state."""
    try:
        while not path.exists() and path != path.parent:
            path = path.parent
        if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
            return None
        info = os.statvfs(path)
        if info.f_flag & os.ST_RDONLY:
            return None
        return info.f_bavail * info.f_frsize
    except OSError:
        return None


def data_root() -> Path:
    return Path(os.environ.get("GPU_TERMINAL_HOME", "~/.local/gpu_terminal")).expanduser() / "kilix-help-llm"


def voice_data_root() -> Path:
    shared = Path(os.environ.get("GPU_TERMINAL_HOME") or "~/.local/gpu_terminal").expanduser()
    storage = Path(os.environ.get("KILIX_STORAGE_HOME") or str(shared / "kilix")).expanduser()
    data = Path(os.environ.get("KILIX_DATA_HOME") or str(storage / "data")).expanduser()
    return data / "voice"


def collect(storage: Path | None = None) -> dict:
    memory, _ = probe._memory()
    available, total = memory["available_bytes"], memory["total_bytes"]
    limit, status = cgroup_headroom()
    if status == "unknown" or total is None or available is None or available > total:
        available = None
    elif limit is not None:
        available = min(available, limit)
    devices = gpu_headroom()
    mapping = ('single-unmasked-gpu-zero' if len(devices)==1 and devices[0]['index']==0
               and not any(key in os.environ for key in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER'))
               else 'unverified')
    return {"schema": SCHEMA, "observed_at": datetime.now(timezone.utc).isoformat(),
            "architecture": platform.machine(),
            "ram_total_bytes": total, "ram_available_bytes": available,
            "cgroup_status": status, "gpus": devices, 'cuda_device_mapping':mapping,
            "disk_available_bytes": storage_headroom(storage or data_root())}


def validate(snapshot: dict) -> None:
    if not isinstance(snapshot, dict) or snapshot.get("schema") != SCHEMA:
        raise ValueError("expected a development resource snapshot")
    mapping = snapshot.get('cuda_device_mapping','unverified')
    if not isinstance(mapping,str) or mapping not in {'unverified','single-unmasked-gpu-zero'}:
        raise ValueError('invalid CUDA device mapping')
    for key in ("ram_total_bytes", "ram_available_bytes", "disk_available_bytes"):
        value = snapshot.get(key)
        if value is not None and (type(value) is not int or not 0 <= value <= 2**63):
            raise ValueError(f"invalid {key}")
    total, available = snapshot.get("ram_total_bytes"), snapshot.get("ram_available_bytes")
    if available is not None and (total is None or available > total):
        raise ValueError("RAM headroom exceeds total RAM")
    if not isinstance(snapshot.get("cgroup_status"), str) or snapshot["cgroup_status"] not in {"unknown", "limited", "unlimited"}:
        raise ValueError("invalid cgroup status")
    devices = snapshot.get("gpus")
    if not isinstance(devices, list) or len(devices) > 256:
        raise ValueError("invalid GPU observations")
    seen = set()
    for device in devices:
        if not isinstance(device, dict):
            raise ValueError("invalid GPU observation")
        index, total, free = (device.get(k) for k in ("index", "total_bytes", "available_bytes"))
        if (any(type(v) is not int for v in (index, total, free)) or not 0 <= index <= 255
                or index in seen or not 0 <= free <= total <= 2**63 or total == 0
                or device.get("backend") != "cuda"):
            raise ValueError("invalid GPU memory observation")
        seen.add(index)
    try:
        observed = datetime.fromisoformat(snapshot["observed_at"])
        if observed.tzinfo is None:
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise ValueError("snapshot requires a timestamp with timezone") from None


def budgets(snapshot: dict, backend: str, gpu: int | None = None, *, now: datetime | None = None,
            ram_reserve: int = 2 * GIB, vram_reserve: int = GIB // 2, disk_reserve: int = GIB) -> dict:
    validate(snapshot)
    if any(type(value) is not int or value < 0 for value in (ram_reserve, vram_reserve, disk_reserve)):
        raise ValueError("resource reserves must be nonnegative integers")
    now = now or datetime.now(timezone.utc)
    age = (now - datetime.fromisoformat(snapshot["observed_at"])).total_seconds()
    fresh = 0 <= age <= 300
    devices = [d for d in snapshot["gpus"] if gpu is None or d["index"] == gpu]
    device = max(devices, key=lambda d: d["available_bytes"], default=None)
    if backend == "auto":
        backend = "cuda" if device or gpu is not None else "cpu"
    if backend not in {"cpu", "cuda"}:
        raise ValueError("only CPU and observed CUDA budgets are supported")
    ram = snapshot.get("ram_available_bytes")
    if snapshot["cgroup_status"] == "unknown":
        ram = None
    vram = device["available_bytes"] if device and backend == "cuda" else None
    disk = snapshot.get("disk_available_bytes")
    return {"backend": backend, "gpu_index": device["index"] if device and backend == "cuda" else None,
            "fresh": fresh,
            "ram_bytes": max(0, ram - ram_reserve) if fresh and ram is not None else None,
            "vram_bytes": max(0, vram - vram_reserve) if fresh and vram is not None else None,
            "disk_bytes": max(0, disk - disk_reserve) if fresh and disk is not None else None}
