"""One per-user record of which models are resident on which GPU.

Apps register the models they load, mark them busy or idle, and call
``ensure_room`` before loading. ``ensure_room`` measures free VRAM and, when
it is short, unloads other residents that are idle and not pinned, least
recently used first, but only when doing so frees enough; otherwise it
unloads nothing and raises ``RoomError``. Models Ollama has loaded are found
through its ``/api/ps`` whoever loaded them. GPU processes with no record are
listed as unmanaged and are never touched.

There is no daemon. Every call takes the registry's ``flock`` for its
duration. The trust boundary is one user's cooperating processes, as for the
device lease; this records residency and does not change or use that lease.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import stat
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from . import ollama as _ollama
from . import vram

VERSION = "kilix.gpu-residency/v1"
BACKENDS = ("ollama", "process", "http")
STATES = ("loading", "idle", "busy", "unloading")
UNLOAD_KINDS = ("ollama", "signal")
POLICIES = ("evict-idle", "never")
ERROR_CODES = ("gone", "invalid-request", "no-room", "not-found", "refused", "unavailable", "unmeasured")
NAMESPACE_DEFAULT = "/run/user/{euid}/kilix-gpu-residency-v1"
LOCK_WAIT_SECONDS = 120.0
TERM_GRACE_SECONDS = 5.0
KILL_GRACE_SECONDS = 5.0
RELEASE_WAIT_SECONDS = 10.0
MAX_RECORDS = 64
MAX_EVENTS = 32
DEVICE_MAP_ENV = "KILIX_GPU_DEVICES"
_POLL_SECONDS = 0.05
_MAX_BYTES = 65536
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z", re.ASCII)
_MODEL = re.compile(r"[^\x00-\x1f\x7f-\x9f]{1,128}\Z")
_ID = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_LEAF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z", re.ASCII)
_RECORD_NAME = re.compile(r"([0-9a-f]{32})\.json\Z", re.ASCII)
_FIXED_NAMES = ("registry.lock", "pins.json", "events.json")
_FILE_FLAGS = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_RECORD_FIELDS = {"version", "id", "app", "model", "backend", "device", "holder", "pgid", "need_bytes",
                  "state", "pinned", "registered_at", "last_used", "unload"}
# Command names an Ollama server or runner process has (comm is cut to 15 bytes).
# Ollama 0.40 runs each model in a llama-server that is a child of "ollama serve",
# so a process whose parent is named ollama is Ollama's too.
_OLLAMA_COMMANDS = ("ollama", "ollama_llama_se", "ollama_llama_server")


class ResidencyError(RuntimeError):
    """A stable machine-readable refusal; ``code`` is one of ERROR_CODES."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RoomError(ResidencyError):
    """Not enough VRAM could be freed. Nothing was unloaded unless ``evicted`` says so."""

    def __init__(self, message: str, *, need_bytes: int, free_bytes: int | None, device: str | None,
                 plan: tuple[ResidentInfo, ...] = (), evicted: tuple[Eviction, ...] = (),
                 code: str = "no-room") -> None:
        super().__init__(code, message)
        self.need_bytes = need_bytes
        self.free_bytes = free_bytes
        self.device = device
        self.plan = plan
        self.evicted = evicted


@dataclass(frozen=True)
class ResidentInfo:
    id: str
    app: str
    model: str
    backend: str
    device_label: str | None
    device_pci: str | None
    pid: int | None
    pgid: int | None
    vram_bytes: int  # measured now, on device_pci (or on every GPU when that is unknown)
    need_bytes: int | None
    state: str
    pinned: bool
    last_used: float | None
    registered: bool  # False for an Ollama model no app registered
    unload_kind: str | None


@dataclass(frozen=True)
class Unmanaged:
    pid: int
    command: str
    device_pci: str
    vram_bytes: int


@dataclass(frozen=True)
class Eviction:
    id: str
    app: str
    model: str
    freed_bytes: int
    time: float
    by: str  # the requester, or "kilix-gpu" for a manual unload
    reason: str  # "room" or "manual"


@dataclass(frozen=True)
class Room:
    device_pci: str
    need_bytes: int
    free_bytes: int
    evicted: tuple[Eviction, ...] = ()
    plan: tuple[ResidentInfo, ...] = ()


@dataclass(frozen=True)
class Status:
    devices: tuple[vram.Device, ...]
    residents: tuple[ResidentInfo, ...]
    unmanaged: tuple[Unmanaged, ...]
    ollama: str  # "reachable", "unreachable" or "not-loopback"
    events: tuple[Eviction, ...] = field(default=())


def _refuse(message: str) -> None:
    raise ResidencyError("unavailable", message)


def _invalid(message: str) -> None:
    raise ResidencyError("invalid-request", message)


# Descriptors {{{

# Every registry lock descriptor this module holds. A child forked without exec
# shares its open file description, and so the lock: the registry would stay
# locked until the child exits. The child closes its copies at once.
_HELD: set[int] = set()


def _forget_in_child() -> None:
    for fd in list(_HELD):
        try:
            os.close(fd)
        except OSError:
            pass
    _HELD.clear()


os.register_at_fork(after_in_child=_forget_in_child)


def _identity(info: os.stat_result) -> tuple[int, int]:
    return (info.st_dev, info.st_ino)


def _file_info(fd: int) -> os.stat_result:
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
        _refuse("Residency file ownership, mode or link count is invalid")
    return info


def _directory(fd: int, *, private: bool) -> os.stat_result:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        _refuse("Residency directory is not a directory")
    if private:
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            _refuse("Residency directory must be user-owned mode 0700")
    elif (info.st_uid not in (0, os.geteuid())
          or (info.st_mode & 0o022 and not (info.st_uid == 0 and info.st_mode & stat.S_ISVTX))):
        _refuse("Residency ancestor is not trusted")
    return info


def _open_parent(path: str) -> int:
    """No-follow traversal from /; the parent itself must be user-owned 0700."""
    current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        _directory(current, private=False)
        for component in path.split("/")[1:]:
            if not component or component in (".", ".."):
                _refuse("Residency namespace must use canonical absolute components")
            child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
            os.close(current)
            current = child
            _directory(current, private=False)
        _directory(current, private=True)
        return current
    except BaseException:
        os.close(current)
        raise


def default_namespace() -> str:
    return NAMESPACE_DEFAULT.format(euid=os.geteuid())


class _Namespace:
    """The registry directory, opened without following links and locked."""

    def __init__(self, path: str | None, wait: float) -> None:
        self.parent = self.directory = self.lock = -1
        path = default_namespace() if path is None else path
        parent_path, self.leaf = os.path.split(path)
        try:
            if (not _LEAF.fullmatch(self.leaf) or not parent_path.startswith("/")
                    or os.path.normpath(path) != path):
                _invalid("Invalid residency namespace")
            self.parent = _open_parent(parent_path)
            try:
                os.mkdir(self.leaf, 0o700, dir_fd=self.parent)
            except FileExistsError:
                pass
            self.directory = os.open(self.leaf, _DIRECTORY_FLAGS, dir_fd=self.parent)
            _directory(self.directory, private=True)
            self.lock = os.open("registry.lock", _FILE_FLAGS | os.O_CREAT, 0o600, dir_fd=self.directory)
            _HELD.add(self.lock)
            _file_info(self.lock)
            deadline = time.monotonic() + wait
            while True:
                try:
                    fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        _refuse("Residency registry is busy")
                    time.sleep(_POLL_SECONDS)
            self.check()
            self.clean()
        except BaseException as error:
            self.close()
            if isinstance(error, OSError):
                raise ResidencyError("unavailable", "Residency namespace cannot be opened safely") from error
            raise

    def check(self) -> None:
        """Refuse a directory or lock that was replaced since it was opened."""
        here = os.stat(self.leaf, dir_fd=self.parent, follow_symlinks=False)
        if _identity(here) != _identity(_directory(self.directory, private=True)):
            _refuse("Residency directory was replaced")
        lock = os.stat("registry.lock", dir_fd=self.directory, follow_symlinks=False)
        if _identity(lock) != _identity(_file_info(self.lock)):
            _refuse("Residency lock was replaced")

    def names(self) -> list[str]:
        names = os.listdir(self.directory)
        for name in names:
            if not (name in _FIXED_NAMES or _RECORD_NAME.fullmatch(name)
                    or (name.startswith(".") and name.endswith(".tmp"))):
                _refuse("Residency directory has an unexpected entry")
        return names

    def clean(self) -> None:
        # A writer killed between creating and renaming its temporary file.
        for name in self.names():
            if name.startswith("."):
                os.unlink(name, dir_fd=self.directory)

    def read(self, name: str) -> object | None:
        try:
            fd = os.open(name, _FILE_FLAGS, dir_fd=self.directory)
        except FileNotFoundError:
            return None
        try:
            info = _file_info(fd)
            if not 0 < info.st_size <= _MAX_BYTES:
                _refuse("Residency file size is invalid")
            raw = os.pread(fd, _MAX_BYTES + 1, 0)
        finally:
            os.close(fd)
        try:
            return json.loads(raw, object_pairs_hook=_pairs,
                              parse_constant=lambda _: _refuse("Non-finite residency value"))
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ResidencyError("unavailable", "Residency file is malformed") from error

    def write(self, name: str, value: object) -> None:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False).encode() + b"\n"
        if len(raw) > _MAX_BYTES:
            _refuse("Residency file exceeds its bound")
        self.check()
        temporary = "." + name + ".tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                     dir_fd=self.directory)
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(raw)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, name, src_dir_fd=self.directory, dst_dir_fd=self.directory)
        os.fsync(self.directory)

    def remove(self, name: str) -> None:
        self.check()
        try:
            os.unlink(name, dir_fd=self.directory)
        except FileNotFoundError:
            pass

    def records(self) -> dict[str, dict]:
        found = {}
        for name in sorted(self.names()):
            match = _RECORD_NAME.fullmatch(name)
            if match is None:
                continue
            value = self.read(name)
            if value is None:
                continue
            if not _valid_record(value) or value["id"] != match.group(1):
                _refuse("Residency record has an unsupported schema")
            found[value["id"]] = value
        return found

    def pins(self) -> set[str]:
        value = self.read("pins.json")
        if value is None:
            return set()
        if (type(value) is not dict or set(value) != {"version", "ollama"} or value["version"] != VERSION
                or type(value["ollama"]) is not list
                or not all(type(n) is str and _ollama.MODEL_NAME.fullmatch(n) for n in value["ollama"])):
            _refuse("Residency pins have an unsupported schema")
        return set(value["ollama"])

    def save_pins(self, names: set[str]) -> None:
        self.write("pins.json", {"version": VERSION, "ollama": sorted(names)})

    def events(self) -> list[dict]:
        value = self.read("events.json")
        if value is None:
            return []
        if (type(value) is not dict or set(value) != {"version", "events"} or value["version"] != VERSION
                or type(value["events"]) is not list or not all(_valid_event(e) for e in value["events"])):
            _refuse("Residency events have an unsupported schema")
        return value["events"]

    def add_event(self, event: Eviction) -> None:
        events = self.events() + [event.__dict__.copy()]
        self.write("events.json", {"version": VERSION, "events": events[-MAX_EVENTS:]})

    def close(self) -> None:
        for name in ("lock", "directory", "parent"):
            fd = getattr(self, name)
            if fd >= 0:
                os.close(fd)
                _HELD.discard(fd)
            setattr(self, name, -1)

    def __enter__(self) -> _Namespace:
        return self

    def __exit__(self, kind: object, error: BaseException | None, _: object) -> None:
        self.close()
        if isinstance(error, OSError):
            raise ResidencyError("unavailable", "Residency namespace cannot be used safely") from error


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _refuse("Residency file has duplicate fields")
        result[key] = value
    return result

# }}}


# Validation {{{

def _int(value: object, low: int = 0) -> bool:
    return type(value) is int and low <= value < 2**63


def _time(value: object) -> bool:
    return type(value) in (int, float) and 0 <= value < 2**40


def _pci(value: object) -> bool:
    return type(value) is str and vram.normalise_pci(value) == value


def _valid_unload(value: object) -> bool:
    if value is None:
        return True
    if type(value) is not dict:
        return False
    if value.get("kind") == "signal":
        return set(value) == {"kind", "pgid"} and _int(value["pgid"], 2)
    if value.get("kind") == "ollama":
        return (set(value) == {"kind", "model"} and type(value["model"]) is str
                and _ollama.MODEL_NAME.fullmatch(value["model"]) is not None)
    return False


def _valid_record(value: object) -> bool:
    if type(value) is not dict or set(value) != _RECORD_FIELDS or value["version"] != VERSION:
        return False
    device, holder, unload = value["device"], value["holder"], value["unload"]
    if not (type(device) is dict and set(device) == {"label", "pci"}
            and (device["label"] is None or (type(device["label"]) is str and _LABEL.fullmatch(device["label"])))
            and (device["pci"] is None or _pci(device["pci"]))):
        return False
    if not (type(holder) is dict and set(holder) == {"pid", "start"}
            and _int(holder["pid"], 1) and _int(holder["start"])):
        return False
    if not _valid_unload(unload):
        return False
    # An Ollama resident, and only one, unloads through Ollama; a signal names the record's group.
    kind = None if unload is None else unload["kind"]
    if (value["backend"] == "ollama") != (kind == "ollama"):
        return False
    if kind == "signal" and (unload["pgid"] != value["pgid"] or holder["pid"] != value["pgid"]):
        return False
    return (type(value["id"]) is str and _ID.fullmatch(value["id"]) is not None
            and type(value["app"]) is str and _LABEL.fullmatch(value["app"]) is not None
            and type(value["model"]) is str and _MODEL.fullmatch(value["model"]) is not None
            and value["backend"] in BACKENDS and value["state"] in STATES
            and type(value["pinned"]) is bool
            and (value["pgid"] is None or _int(value["pgid"], 2))
            and (value["need_bytes"] is None or _int(value["need_bytes"], 1))
            and _time(value["registered_at"]) and _time(value["last_used"]))


def _valid_event(value: object) -> bool:
    return (type(value) is dict and set(value) == {"id", "app", "model", "freed_bytes", "time", "by", "reason"}
            and type(value["id"]) is str and len(value["id"]) <= 200
            and type(value["app"]) is str and len(value["app"]) <= 200
            and type(value["model"]) is str and len(value["model"]) <= 200
            and _int(value["freed_bytes"]) and _time(value["time"])
            and type(value["by"]) is str and _LABEL.fullmatch(value["by"]) is not None
            and value["reason"] in ("room", "manual"))

# }}}


# Processes {{{

def _stat(pid: int) -> tuple[str, int, int] | None:
    """(state, process group, start time in clock ticks) of a live process."""
    try:
        with open(os.path.join(vram.proc_root(), str(pid), "stat"), "rb") as stream:
            text = stream.read(4096).decode("ascii", "replace")
    except OSError:
        return None
    fields = text[text.rfind(")") + 2:].split()
    try:
        state, pgrp, start = fields[0], int(fields[2]), int(fields[19])
    except (IndexError, ValueError):
        return None
    return None if state in ("Z", "X") else (state, pgrp, start)


def _start(pid: int) -> int | None:
    found = _stat(pid)
    return None if found is None else found[2]


def _alive(holder: Mapping[str, int]) -> bool:
    return _start(holder["pid"]) == holder["start"]


def _command(pid: int) -> str:
    try:
        with open(os.path.join(vram.proc_root(), str(pid), "comm"), "rb") as stream:
            return stream.read(64).decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _parent(pid: int) -> int | None:
    try:
        with open(os.path.join(vram.proc_root(), str(pid), "stat"), "rb") as stream:
            text = stream.read(4096).decode("ascii", "replace")
        return int(text[text.rfind(")") + 2:].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _is_ollama(pid: int) -> bool:
    if _command(pid) in _OLLAMA_COMMANDS:
        return True
    parent = _parent(pid)
    return parent is not None and parent > 1 and _command(parent) in _OLLAMA_COMMANDS


def _pids() -> list[int]:
    try:
        return [int(name) for name in os.listdir(vram.proc_root()) if name.isdigit()]
    except OSError:
        return []


def _groups() -> dict[int, set[int]]:
    """Process group -> its live members."""
    groups: dict[int, set[int]] = {}
    for pid in _pids():
        found = _stat(pid)
        if found is not None:
            groups.setdefault(found[1], set()).add(pid)
    return groups

# }}}


# The view {{{

@dataclass
class _View:
    records: dict[str, dict]
    devices: tuple[vram.Device, ...]
    residents: list[ResidentInfo]
    unmanaged: list[Unmanaged]
    ollama: str
    url: str


def _record_pids(record: dict, groups: Mapping[int, set[int]]) -> set[int]:
    pids = {record["holder"]["pid"]}
    if record["pgid"] is not None:
        pids |= groups.get(record["pgid"], set())
    return pids


def _bytes_on(devices: Iterable[vram.Device], pids: set[int], pci: str | None) -> int:
    return sum(used for device in devices if pci is None or device.pci == pci
               for pid, used in device.processes.items() if pid in pids)


def _info(record: dict, vram_bytes: int, pci: str | None) -> ResidentInfo:
    unload = record["unload"]
    return ResidentInfo(record["id"], record["app"], record["model"], record["backend"],
                        record["device"]["label"], pci, record["holder"]["pid"], record["pgid"], vram_bytes,
                        record["need_bytes"], record["state"], record["pinned"], record["last_used"], True,
                        None if unload is None else unload["kind"])


def _view(ns: _Namespace, *, wake: bool, url: str | None) -> _View:
    """Prune dead records, then measure every GPU and merge Ollama's own list."""
    url = _ollama.default_url() if url is None else url
    records = ns.records()
    for key, record in list(records.items()):
        if not _alive(record["holder"]):
            ns.remove(key + ".json")
            del records[key]
    devices = vram.snapshot(wake=wake)
    loaded: tuple[_ollama.Loaded, ...] = ()
    if _ollama.loopback(url) is None:
        state = "not-loopback"
    else:
        try:
            loaded = _ollama.loaded(url)
            state = "reachable"
        except _ollama.OllamaError:
            state = "unreachable"
    names = {model.name for model in loaded}
    if state == "reachable":
        for key, record in list(records.items()):
            if record["backend"] == "ollama" and record["state"] != "loading" and record["model"] not in names:
                ns.remove(key + ".json")
                del records[key]
    groups = _groups()
    ollama_pids = {pid for device in devices for pid in device.processes if _is_ollama(pid)}
    ollama_device = None
    if ollama_pids:
        held = {device.pci: _bytes_on([device], ollama_pids, device.pci) for device in devices}
        best = max(held, key=held.get)
        ollama_device = best if held[best] > 0 else None
    by_name = {model.name: model for model in loaded}
    residents = []
    managed: set[int] = set()
    for record in records.values():
        if record["backend"] == "ollama":
            model = by_name.pop(record["model"], None)
            residents.append(_info(record, model.vram_bytes if model else 0,
                                   record["device"]["pci"] or ollama_device))
            continue
        pids = _record_pids(record, groups)
        managed |= pids
        pci = record["device"]["pci"]
        residents.append(_info(record, _bytes_on(devices, pids, pci), pci))
    pins = ns.pins()
    for model in by_name.values():
        last = None if model.expires_at is None else model.expires_at - 300.0
        residents.append(ResidentInfo("ollama:" + model.name, "ollama", model.name, "ollama", None, ollama_device,
                                      None, None, model.vram_bytes, None, "idle", model.name in pins, last,
                                      False, "ollama"))
    if loaded:
        managed |= ollama_pids
    unmanaged = [Unmanaged(pid, _command(pid), device.pci, used)
                 for device in devices for pid, used in sorted(device.processes.items())
                 if pid not in managed and used > 0]
    return _View(records, devices, residents, unmanaged, state, url)

# }}}


# Devices {{{

def _device_map() -> dict[str, str]:
    mapping = {}
    for item in os.environ.get(DEVICE_MAP_ENV, "").replace(";", ",").split(","):
        label, sep, pci = item.partition("=")
        if sep and _LABEL.fullmatch(label.strip()) and vram.normalise_pci(pci):
            mapping[label.strip()] = vram.normalise_pci(pci)  # type: ignore[assignment]
    return mapping


def resolve_device(label: str | None, devices: Iterable[vram.Device]) -> str | None:
    """The PCI address a device label names, or None when it cannot be told.

    A PCI address names itself; then KILIX_GPU_DEVICES (``Vulkan1=0000:01:00.0,...``)
    is consulted; otherwise the label is taken to mean the one GPU whose VRAM can
    be measured, when exactly one can."""
    devices = tuple(devices)
    if label is not None:
        pci = vram.normalise_pci(label)
        if pci is not None:
            return pci
        mapped = _device_map().get(label)
        if mapped is not None:
            return mapped
    measured = [device.pci for device in devices if device.counter]
    return measured[0] if len(measured) == 1 else None

# }}}


# Unloading {{{

def _wait_gone(pgid: int, leader: Mapping[str, int], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        if not _alive(leader) and not _groups().get(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)


def _signal_unload(record: dict) -> None:
    """TERM the record's process group, then KILL it after a grace period.

    Only while the group's leader is still the process that was registered:
    a PID reused since then is never signalled."""
    pgid = record["unload"]["pgid"]
    leader = record["holder"]
    if pgid != leader["pid"] or pgid <= 1 or pgid == os.getpgrp():
        raise ResidencyError("refused", "Refusing to signal a process group the record does not lead")
    # The first signal goes only to a group whose registered leader is still alive.
    # Linux does not reuse a process group's number while any member is alive, so
    # escalating to KILL is safe while the group has members, even if its leader died.
    if not _alive(leader):
        raise ResidencyError("refused", "The registered leader is gone; its group is not signalled")
    for sig, grace in ((signal.SIGTERM, TERM_GRACE_SECONDS), (signal.SIGKILL, KILL_GRACE_SECONDS)):
        if not _alive(leader) and not _groups().get(pgid):
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        except PermissionError as error:
            raise ResidencyError("refused", "The process group belongs to another user") from error
        if _wait_gone(pgid, leader, grace):
            return
    raise ResidencyError("refused", "The process group did not exit")


def _perform_unload(ns: _Namespace, view: _View, info: ResidentInfo) -> None:
    if not info.registered:
        try:
            _ollama.unload(view.url, info.model)
        except _ollama.OllamaError as error:
            raise ResidencyError("unavailable", str(error)) from error
        return
    record = view.records[info.id]
    record["state"] = "unloading"
    ns.write(info.id + ".json", record)
    try:
        if record["unload"]["kind"] == "signal":
            _signal_unload(record)
        else:
            try:
                _ollama.unload(view.url, record["unload"]["model"])
            except _ollama.OllamaError as error:
                raise ResidencyError("unavailable", str(error)) from error
    except BaseException:
        # Unloading failed: the model is still resident, and must not look evictable.
        if _alive(record["holder"]):
            record["state"] = "busy"
            ns.write(info.id + ".json", record)
        raise
    ns.remove(info.id + ".json")


def _evict(ns: _Namespace, view: _View, info: ResidentInfo, by: str, reason: str) -> Eviction:
    _perform_unload(ns, view, info)
    event = Eviction(info.id, info.app, info.model, info.vram_bytes, time.time(), by, reason)
    ns.add_event(event)
    return event

# }}}


# Room {{{

def _reserved(view: _View, pci: str) -> int:
    """VRAM claimed by records still loading on ``pci`` beyond what they hold yet."""
    return sum(max(0, info.need_bytes - info.vram_bytes) for info in view.residents
               if info.registered and info.state == "loading" and info.need_bytes
               and info.device_pci == pci)


def _candidates(view: _View, pci: str) -> list[ResidentInfo]:
    found = [info for info in view.residents
             if info.device_pci == pci and info.state == "idle" and not info.pinned
             and info.unload_kind is not None and info.vram_bytes > 0]
    found.sort(key=lambda info: (info.last_used is None, info.last_used or 0.0, info.id))
    return found


def _room(ns: _Namespace, need_bytes: int, device: str | None, requester: str, policy: str,
          url: str | None) -> tuple[Room, _View]:
    if not _int(need_bytes, 1):
        _invalid("need_bytes must be a positive integer")
    if policy not in POLICIES:
        _invalid("Unknown residency policy")
    if not _LABEL.fullmatch(requester or ""):
        _invalid("Invalid requester label")
    view = _view(ns, wake=True, url=url)
    pci = resolve_device(device, view.devices)
    target = next((d for d in view.devices if d.pci == pci), None)
    if target is None or target.free_bytes is None:
        raise RoomError("The device's free VRAM cannot be measured", need_bytes=need_bytes, free_bytes=None,
                        device=pci, code="unmeasured")
    free = target.free_bytes - _reserved(view, pci)
    if free >= need_bytes:
        return Room(pci, need_bytes, free), view
    plan, freed = [], 0
    for info in _candidates(view, pci):
        plan.append(info)
        freed += info.vram_bytes
        if free + freed >= need_bytes:
            break
    if free + freed < need_bytes:
        raise RoomError("Not enough VRAM even if every idle resident were unloaded", need_bytes=need_bytes,
                        free_bytes=free, device=pci, plan=tuple(plan))
    if policy == "never":
        raise RoomError("Room needs unloading, and the policy is never", need_bytes=need_bytes,
                        free_bytes=free, device=pci, plan=tuple(plan))
    evicted = []
    try:
        for info in plan:
            evicted.append(_evict(ns, view, info, requester, "room"))
    except ResidencyError as error:
        raise RoomError(f"Unloading failed: {error}", need_bytes=need_bytes, free_bytes=free, device=pci,
                        plan=tuple(plan), evicted=tuple(evicted)) from error
    deadline = time.monotonic() + RELEASE_WAIT_SECONDS
    while True:
        view = _view(ns, wake=True, url=url)
        target = next((d for d in view.devices if d.pci == pci), None)
        free = (target.free_bytes if target and target.free_bytes is not None else 0) - _reserved(view, pci)
        if free >= need_bytes:
            return Room(pci, need_bytes, free, tuple(evicted), tuple(plan)), view
        if time.monotonic() >= deadline:
            raise RoomError("Unloaded residents, but the VRAM did not come back", need_bytes=need_bytes,
                            free_bytes=free, device=pci, plan=tuple(plan), evicted=tuple(evicted))
        time.sleep(_POLL_SECONDS * 4)


def ensure_room(need_bytes: int, *, device: str | None, requester: str, policy: str = "evict-idle",
                namespace: str | None = None, ollama: str | None = None,
                wait: float = LOCK_WAIT_SECONDS) -> Room:
    """Make ``need_bytes`` of VRAM free on ``device``, unloading idle residents if allowed.

    Returns Room(device_pci, need_bytes, free_bytes, evicted). Raises RoomError,
    having unloaded nothing, when even every candidate would not be enough or the
    policy is "never"; its ``plan`` lists what would have been unloaded."""
    with _Namespace(namespace, wait) as ns:
        return _room(ns, need_bytes, device, requester, policy, ollama)[0]

# }}}


# Registration {{{

def _holder(pid: int) -> dict[str, int]:
    found = _stat(pid)
    if found is None:
        raise ResidencyError("invalid-request", "The process to register is not running")
    return {"pid": pid, "start": found[2]}


class Resident:
    """A registered model. Leaving a ``with`` block removes the record, unloading
    the model first if it is still loaded."""

    def __init__(self, record: dict, namespace: str | None, url: str | None, room: Room | None) -> None:
        self.id = record["id"]
        self.app = record["app"]
        self.model = record["model"]
        self.room = room
        self._namespace = namespace
        self._url = url

    def _update(self, **changes: object) -> dict:
        with _Namespace(self._namespace, LOCK_WAIT_SECONDS) as ns:
            record = ns.records().get(self.id)
            if record is None or not _alive(record["holder"]):
                if record is not None:
                    ns.remove(self.id + ".json")
                raise ResidencyError("gone", "The residency record is gone")
            record.update(changes)
            record["last_used"] = time.time()
            if not _valid_record(record):
                _invalid("Invalid residency update")
            ns.write(self.id + ".json", record)
            return record

    def busy(self) -> None:
        self._update(state="busy")

    def idle(self) -> None:
        self._update(state="idle")

    def attach(self, *, pid: int | None = None, pgid: int | None = None) -> None:
        """Follow a model process started after registering (its group, when given)."""
        if (pid is None) == (pgid is None):
            _invalid("Attach a pid or a pgid")
        if pgid is not None:
            changes = _process_fields(None, pgid, "auto", "process", self.model)
        else:
            changes = {"holder": _holder(pid), "pgid": None}
        self._update(**changes)

    @property
    def evicted(self) -> Eviction | None:
        """How this record was unloaded by another app or by hand, if it was."""
        with _Namespace(self._namespace, LOCK_WAIT_SECONDS) as ns:
            for event in reversed(ns.events()):
                if event["id"] == self.id:
                    return Eviction(**event)
        return None

    def close(self, *, unload: bool = True) -> None:
        with _Namespace(self._namespace, LOCK_WAIT_SECONDS) as ns:
            record = ns.records().get(self.id)
            if record is None:
                return
            if unload and record["unload"] is not None and _alive(record["holder"]):
                view = _view(ns, wake=False, url=self._url)
                info = next((i for i in view.residents if i.id == self.id), None)
                if info is not None and (info.backend != "ollama" or info.vram_bytes or info.state != "loading"):
                    _perform_unload(ns, view, info)
            ns.remove(self.id + ".json")

    def __enter__(self) -> Resident:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _process_fields(pid: int | None, pgid: int | None, unload: object, backend: str, model: str) -> dict:
    if pgid is not None:
        if not _int(pgid, 2):
            _invalid("Invalid process group")
        found = _stat(pgid)
        if found is None or found[1] != pgid:
            _invalid("The process group's leader is not running")
        holder = {"pid": pgid, "start": found[2]}
    elif pid is not None:
        if not _int(pid, 1):
            _invalid("Invalid pid")
        holder = _holder(pid)
    else:
        holder = _holder(os.getpid())
    if unload == "auto":
        if backend == "ollama":
            unload = {"kind": "ollama", "model": model}
        elif pgid is not None:
            unload = {"kind": "signal", "pgid": pgid}
        else:
            unload = None
    if type(unload) is dict and unload.get("kind") == "signal":
        if unload.get("pgid") != pgid:
            _invalid("A signal unload must name the registered process group")
        if pgid == os.getpgrp():
            _invalid("Refusing to register this process's own group for signalling")
    if not _valid_unload(unload):
        _invalid("Invalid unload descriptor")
    return {"holder": holder, "pgid": pgid, "unload": unload}


def register(*, app: str, model: str, backend: str, device: str | None = None, pid: int | None = None,
             pgid: int | None = None, need_bytes: int | None = None, policy: str = "evict-idle",
             pinned: bool = False, unload: object = "auto", namespace: str | None = None,
             ollama: str | None = None, wait: float = LOCK_WAIT_SECONDS) -> Resident:
    """Record a model as resident (state ``loading``) and return its handle.

    With ``need_bytes``, room is made first under the same lock, so no other app
    can take the space between the check and the record; the record then reserves
    the bytes it has not yet allocated until it leaves ``loading``. Without a pid
    or pgid the record follows the calling process; ``attach`` points it at a
    model process started afterwards. For backend ``ollama`` the model is the
    Ollama model name and the unload is ``keep_alive: 0``; for a process group the
    unload is TERM, then KILL."""
    if not _LABEL.fullmatch(app or ""):
        _invalid("Invalid app label")
    if backend not in BACKENDS:
        _invalid("Unknown backend")
    if type(model) is not str or not _MODEL.fullmatch(model):
        _invalid("Invalid model display name")
    if backend == "ollama" and not _ollama.MODEL_NAME.fullmatch(model):
        _invalid("Invalid Ollama model name")
    if device is not None and not _LABEL.fullmatch(device):
        _invalid("Invalid device label")
    if need_bytes is not None and not _int(need_bytes, 1):
        _invalid("need_bytes must be a positive integer")
    if type(pinned) is not bool:
        _invalid("pinned must be a bool")
    fields = _process_fields(pid, pgid, unload, backend, model)
    if backend == "ollama" and (fields["unload"] is None or fields["unload"]["kind"] != "ollama"):
        _invalid("An Ollama resident unloads through Ollama")
    if backend != "ollama" and fields["unload"] is not None and fields["unload"]["kind"] == "ollama":
        _invalid("Only an Ollama resident unloads through Ollama")
    with _Namespace(namespace, wait) as ns:
        room = None
        if need_bytes is not None:
            room, view = _room(ns, need_bytes, device, app, policy, ollama)
            pci = room.device_pci
        else:
            view = _view(ns, wake=False, url=ollama)
            pci = resolve_device(device, view.devices)
        if len(view.records) >= MAX_RECORDS:
            _refuse("The residency registry is full")
        now = time.time()
        record = {"version": VERSION, "id": uuid.uuid4().hex, "app": app, "model": model, "backend": backend,
                  "device": {"label": device if device is None or vram.normalise_pci(device) is None else None,
                             "pci": pci},
                  "need_bytes": need_bytes, "state": "loading", "pinned": pinned, "registered_at": now,
                  "last_used": now, **fields}
        ns.write(record["id"] + ".json", record)
    return Resident(record, namespace, ollama, room)

# }}}


# Reading and administration {{{

def status(*, namespace: str | None = None, ollama: str | None = None, wake: bool = False,
           wait: float = LOCK_WAIT_SECONDS) -> Status:
    """Devices, residents (registered and Ollama-discovered), unmanaged GPU processes
    and recent evictions. Without ``wake`` a sleeping GPU is not woken to be read."""
    with _Namespace(namespace, wait) as ns:
        view = _view(ns, wake=wake, url=ollama)
        events = tuple(Eviction(**event) for event in ns.events())
    return Status(view.devices, tuple(view.residents), tuple(view.unmanaged), view.ollama, events)


def residents(*, namespace: str | None = None, ollama: str | None = None,
              wait: float = LOCK_WAIT_SECONDS) -> tuple[ResidentInfo, ...]:
    return status(namespace=namespace, ollama=ollama, wait=wait).residents


def _find(view: _View, target: str) -> ResidentInfo:
    if type(target) is not str or not target:
        _invalid("Name a resident by id or model")
    by_id = [info for info in view.residents if info.id == target]
    if by_id:
        return by_id[0]
    by_model = [info for info in view.residents if info.model == target]
    if len(by_model) == 1:
        return by_model[0]
    if by_model:
        _invalid("Several residents have that model; name one by id")
    if any(u.command == target or str(u.pid) == target for u in view.unmanaged):
        raise ResidencyError("refused", "Unmanaged GPU processes are never unloaded")
    raise ResidencyError("not-found", "No such resident")


def unload(target: str, *, force: bool = False, by: str = "kilix-gpu", namespace: str | None = None,
           ollama: str | None = None, wait: float = LOCK_WAIT_SECONDS) -> Eviction:
    """Unload one resident by id or model name. A pinned, busy or loading one is
    refused unless ``force``; one without an unload action always is."""
    if not _LABEL.fullmatch(by or ""):
        _invalid("Invalid requester label")
    with _Namespace(namespace, wait) as ns:
        view = _view(ns, wake=False, url=ollama)
        info = _find(view, target)
        if info.unload_kind is None:
            raise ResidencyError("refused", "This resident has no unload action")
        if not force and info.pinned:
            raise ResidencyError("refused", "This resident is pinned")
        if not force and info.state != "idle":
            raise ResidencyError("refused", f"This resident is {info.state}")
        return _evict(ns, view, info, by, "manual")


def pin(target: str, pinned: bool = True, *, namespace: str | None = None, ollama: str | None = None,
        wait: float = LOCK_WAIT_SECONDS) -> ResidentInfo:
    """Pin (or unpin) a resident, so ``ensure_room`` never unloads it."""
    with _Namespace(namespace, wait) as ns:
        view = _view(ns, wake=False, url=ollama)
        info = _find(view, target)
        if info.registered:
            record = view.records[info.id]
            record["pinned"] = bool(pinned)
            ns.write(info.id + ".json", record)
        else:
            names = ns.pins()
            (names.add if pinned else names.discard)(info.model)
            ns.save_pins(names)
    return ResidentInfo(**{**info.__dict__, "pinned": bool(pinned)})


def unpin(target: str, **kwargs: object) -> ResidentInfo:
    return pin(target, False, **kwargs)  # type: ignore[arg-type]

# }}}
