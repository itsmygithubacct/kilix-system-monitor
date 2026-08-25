"""Private local persistence for redacted hardware snapshots.

The state boundary deliberately accepts an explicit root.  CLI policy may map
that root from XDG state, while tests and a future trusted launcher can inject
the exact directory without teaching this module about a user or machine ID.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CACHE_FILENAME = "snapshot-v1.json"
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
SNAPSHOT_ID = re.compile(r"^(fixture|local|redacted):[a-z0-9][a-z0-9._:-]{0,127}$")
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
FORBIDDEN_SNAPSHOT_KEYS = {
    *NEVER_COLLECTED,
    "serial",
    "uuid",
}
EXPECTED_PRIVACY = {
    "cache_mode": "0600",
    "classification": "fingerprinting-grade-local",
    "export_requires_warning": True,
    "telemetry_eligible": False,
}


class StateError(RuntimeError):
    """Base class for a redacted state-boundary refusal."""


class StateUnavailable(StateError):
    """The private state root or requested snapshot is unavailable."""


class StateMissing(StateUnavailable):
    """The private state root has not been created yet."""


class SnapshotInvalid(StateError):
    """A snapshot is malformed or violates the privacy boundary."""


class _DuplicateKey(ValueError):
    pass


def default_state_root(environment: Mapping[str, str] | None = None) -> Path:
    """Return the private component state root without resolving any symlink."""
    values = os.environ if environment is None else environment
    raw_base = values.get("XDG_STATE_HOME")
    if raw_base:
        base = Path(raw_base)
    else:
        raw_home = values.get("HOME")
        if not raw_home:
            raise StateUnavailable("no private state base")
        base = Path(raw_home) / ".local" / "state"
    if not base.is_absolute() or ".." in base.parts:
        raise StateUnavailable("invalid private state base")
    return base / "plebian-hardware"


def _directory_flags() -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _file_flags() -> int:
    return os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)


def _open_private_directory(root: Path, *, create: bool) -> int:
    if not root.is_absolute() or root == Path("/") or ".." in root.parts:
        raise StateUnavailable("invalid private state root")
    descriptor = os.open("/", _directory_flags())
    try:
        for component in root.parts[1:]:
            if component in {"", ".", ".."}:
                raise StateUnavailable("invalid private state component")
            try:
                child = os.open(component, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise StateMissing("private state root absent") from None
                try:
                    os.mkdir(component, PRIVATE_DIRECTORY_MODE, dir_fd=descriptor)
                    child = os.open(component, _directory_flags(), dir_fd=descriptor)
                    os.fchmod(child, PRIVATE_DIRECTORY_MODE)
                except OSError as error:
                    raise StateUnavailable("cannot create private state root") from error
            except OSError as error:
                raise StateUnavailable("private state path refused") from error
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != PRIVATE_DIRECTORY_MODE
        ):
            raise StateUnavailable("private state root ownership or mode refused")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKey(key)
        value[key] = item
    return value


def _forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key.lower() in FORBIDDEN_SNAPSHOT_KEYS or _forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_forbidden_key(item) for item in value)
    return False


def validate_snapshot(document: Any) -> dict[str, Any]:
    """Validate the cache-level privacy invariants without claiming P1 freeze."""
    if not isinstance(document, dict):
        raise SnapshotInvalid("snapshot is not an object")
    if document.get("schema") != "plebian.hardware/v1":
        raise SnapshotInvalid("snapshot schema is unsupported")
    snapshot_id = document.get("snapshot_id")
    if not isinstance(snapshot_id, str) or SNAPSHOT_ID.fullmatch(snapshot_id) is None:
        raise SnapshotInvalid("snapshot identity is invalid")
    capture = document.get("capture")
    if not isinstance(capture, dict) or any(
        (
            capture.get("network_used") is not False,
            capture.get("privileged") is not False,
            capture.get("qualification_eligible") is not False,
        )
    ):
        raise SnapshotInvalid("snapshot crossed the local observation boundary")
    if document.get("privacy") != EXPECTED_PRIVACY:
        raise SnapshotInvalid("snapshot privacy projection is not exact")
    if document.get("never_collected") != NEVER_COLLECTED:
        raise SnapshotInvalid("snapshot denylist is not exact")
    if _forbidden_key(document):
        raise SnapshotInvalid("snapshot contains a forbidden identifier field")
    try:
        _canonical_bytes(document)
    except (RecursionError, TypeError, ValueError) as error:
        raise SnapshotInvalid("snapshot is not finite JSON") from error
    return document


def _canonical_bytes(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _parse_snapshot(payload: bytes, *, require_canonical: bool) -> dict[str, Any]:
    if not payload or len(payload) > MAX_SNAPSHOT_BYTES:
        raise SnapshotInvalid("snapshot byte boundary violated")
    try:
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (RecursionError, UnicodeError, ValueError) as error:
        raise SnapshotInvalid("snapshot JSON refused") from error
    try:
        validated = validate_snapshot(document)
    except RecursionError as error:
        raise SnapshotInvalid("snapshot nesting boundary violated") from error
    if require_canonical and payload != _canonical_bytes(validated):
        raise SnapshotInvalid("cached snapshot is not canonical")
    return validated


def _read_open_file(descriptor: int) -> bytes:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != PRIVATE_FILE_MODE
        or metadata.st_size < 1
        or metadata.st_size > MAX_SNAPSHOT_BYTES
    ):
        raise SnapshotInvalid("snapshot file ownership, mode or size refused")
    chunks: list[bytes] = []
    remaining = MAX_SNAPSHOT_BYTES + 1
    while remaining:
        chunk = os.read(descriptor, min(65536, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) > MAX_SNAPSHOT_BYTES:
        raise SnapshotInvalid("snapshot byte boundary violated")
    return payload


def read_snapshot(root: Path) -> dict[str, Any]:
    """Read the canonical private cache without following a path component."""
    directory = _open_private_directory(root, create=False)
    try:
        try:
            descriptor = os.open(CACHE_FILENAME, _file_flags(), dir_fd=directory)
        except FileNotFoundError as error:
            raise StateUnavailable("cached snapshot absent") from error
        except OSError as error:
            raise StateUnavailable("cached snapshot path refused") from error
        try:
            return _parse_snapshot(_read_open_file(descriptor), require_canonical=True)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def read_snapshot_file(path: Path) -> dict[str, Any]:
    """Read a private exported snapshot for a local diff."""
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise StateUnavailable("snapshot path is not absolute")
    parent = _open_private_directory(path.parent, create=False)
    try:
        try:
            descriptor = os.open(path.name, _file_flags(), dir_fd=parent)
        except OSError as error:
            raise StateUnavailable("snapshot path refused") from error
        try:
            return _parse_snapshot(_read_open_file(descriptor), require_canonical=False)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def write_snapshot(root: Path, document: dict[str, Any]) -> None:
    """Atomically replace the canonical cache through a retained directory fd."""
    payload = _canonical_bytes(validate_snapshot(document))
    if len(payload) > MAX_SNAPSHOT_BYTES:
        raise SnapshotInvalid("snapshot byte boundary violated")
    directory = _open_private_directory(root, create=True)
    temporary = f".snapshot-{secrets.token_hex(16)}.tmp"
    temporary_present = False
    try:
        try:
            existing = os.stat(CACHE_FILENAME, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        except OSError as error:
            raise StateUnavailable("cached snapshot path refused") from error
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode)
            or existing.st_uid != os.geteuid()
            or existing.st_nlink != 1
            or stat.S_IMODE(existing.st_mode) != PRIVATE_FILE_MODE
        ):
            raise StateUnavailable("cached snapshot target refused")
        if existing is not None:
            try:
                current = os.open(CACHE_FILENAME, _file_flags(), dir_fd=directory)
            except OSError as error:
                raise StateUnavailable("cached snapshot path refused") from error
            try:
                _parse_snapshot(_read_open_file(current), require_canonical=True)
            finally:
                os.close(current)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(
                temporary,
                flags,
                PRIVATE_FILE_MODE,
                dir_fd=directory,
            )
            temporary_present = True
            try:
                os.fchmod(descriptor, PRIVATE_FILE_MODE)
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written < 1:
                        raise OSError("snapshot write made no progress")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(
                temporary,
                CACHE_FILENAME,
                src_dir_fd=directory,
                dst_dir_fd=directory,
            )
            temporary_present = False
            os.fsync(directory)
        except OSError as error:
            raise StateUnavailable("cannot persist private snapshot") from error
    finally:
        if temporary_present:
            try:
                os.unlink(temporary, dir_fd=directory)
            except OSError:
                pass
        os.close(directory)


def cache_status(root: Path) -> str:
    """Return absent, empty or valid; malformed/private-state failures refuse."""
    try:
        directory = _open_private_directory(root, create=False)
    except StateMissing:
        return "absent"
    try:
        try:
            descriptor = os.open(CACHE_FILENAME, _file_flags(), dir_fd=directory)
        except FileNotFoundError:
            return "empty"
        except OSError as error:
            raise StateUnavailable("cached snapshot path refused") from error
        try:
            _parse_snapshot(_read_open_file(descriptor), require_canonical=True)
            return "valid"
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _path(document: dict[str, Any], *parts: str) -> Any:
    value: Any = document
    for part in parts:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _gpu_projection(document: dict[str, Any]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for gpu in document.get("gpus", []):
        if not isinstance(gpu, dict):
            continue
        projected.append(
            {
                "backends": [
                    {
                        "name": backend.get("name"),
                        "status": backend.get("status"),
                    }
                    for backend in gpu.get("backends", [])
                    if isinstance(backend, dict)
                ],
                "device_class": gpu.get("device_class"),
                "memory_kind": gpu.get("memory_kind"),
                "render_access": gpu.get("render_access"),
                "shared_memory_bytes": gpu.get("shared_memory_bytes"),
                "vendor": gpu.get("vendor"),
                "vram_bytes": gpu.get("vram_bytes"),
            }
        )
    return projected


def _projection(document: dict[str, Any]) -> dict[str, Any]:
    """Select local capability facts while excluding identity and timestamps."""
    fields = {
        "buses.pci-devices": _path(document, "buses", "pci_devices_count"),
        "buses.storage-controllers": _path(
            document, "buses", "storage_controllers_count"
        ),
        "buses.usb-devices": _path(document, "buses", "usb_devices_count"),
        "cpu.architecture": _path(document, "cpu", "architecture"),
        "cpu.effective": _path(document, "cpu", "effective_cpus"),
        "cpu.isa": _path(document, "cpu", "isa_features"),
        "cpu.logical": _path(document, "cpu", "logical_cpus"),
        "cpu.numa": _path(document, "cpu", "numa_nodes"),
        "cpu.physical-cores": _path(document, "cpu", "physical_cores"),
        "gpu.capabilities": _gpu_projection(document),
        "memory.effective-limit": _path(
            document, "memory", "effective_limit_bytes"
        ),
        "memory.total": _path(document, "memory", "total_bytes"),
        "network.interfaces": [
            {
                "bus": interface.get("bus"),
                "driver": interface.get("driver"),
                "online": interface.get("online"),
                "type": interface.get("type"),
            }
            for interface in _path(document, "network", "interfaces") or []
            if isinstance(interface, dict)
        ],
        "platform.firmware": _path(document, "platform", "firmware_mode"),
        "platform.iommu": _path(document, "platform", "iommu"),
        "platform.secure-boot": _path(document, "platform", "secure_boot"),
        "power.ac": _path(document, "power", "ac_online"),
        "power.battery-percent": _path(document, "power", "battery_percent"),
        "power.battery-present": _path(document, "power", "battery_present"),
        "pressure.cpu": _path(document, "pressure", "cpu"),
        "pressure.io": _path(document, "pressure", "io"),
        "pressure.memory": _path(document, "pressure", "memory"),
        "storage.filesystem": _path(document, "storage", "filesystem_type"),
        "storage.free": _path(document, "storage", "free_bytes"),
        "storage.read-only": _path(document, "storage", "read_only"),
        "storage.total": _path(document, "storage", "total_bytes"),
        "thermal.fans": _path(document, "thermal", "fan_count"),
        "thermal.sensors": _path(document, "thermal", "sensor_count"),
        "virtualization": document.get("virtualization"),
    }
    return fields


def snapshot_changes(
    previous: dict[str, Any], current: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return stable redacted changes without volatile or identity fields."""
    before = _projection(validate_snapshot(previous))
    after = _projection(validate_snapshot(current))
    return [
        {"field": field, "before": before[field], "after": after[field]}
        for field in sorted(before)
        if before[field] != after[field]
    ]
