"""One per-user record of which models are resident on which GPU.

Apps register the models they load, mark them busy or idle, and call
``ensure_room`` before loading. ``ensure_room`` measures free VRAM and, when
it is short, unloads other residents that are idle and not pinned, least
recently used first, but only when doing so frees enough; otherwise it
unloads nothing and raises ``RoomError``. Models Ollama has loaded are found
through its ``/api/ps`` whoever loaded them. GPU processes with no record are
listed as unmanaged and are never touched.

What is unloaded is a *unit*: one Ollama model, whichever records name it, or
one process group, whichever records name it. A unit is a candidate only when
every live record in it is idle and none pins it.

There is no daemon. Every call takes the registry's ``flock`` while it reads
or writes the registry; unloads and the wait for VRAM to come back happen with
the lock released. The trust boundary is one user's cooperating processes in
one PID namespace, as for the device lease; this records residency and does
not change or use that lease.
"""

from __future__ import annotations

import fcntl
import functools
import json
import os
import re
import select
import signal
import stat
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from . import ollama as _ollama
from . import vram

VERSION = "kilix.gpu-residency/v1"
VERSION_PREFIX = "kilix.gpu-residency/"
BACKENDS = ("ollama", "process", "http")
STATES = ("loading", "idle", "busy", "unloading")
UNLOAD_KINDS = ("ollama", "signal")
POLICIES = ("evict-idle", "never")
ERROR_CODES = ("gone", "invalid-request", "no-room", "not-found", "not-released", "policy-never", "refused",
               "unavailable", "unload-failed", "unmeasured")
ROOM_ERROR_CODES = ("no-room", "not-released", "policy-never", "unload-failed", "unmeasured")
NAMESPACE_DEFAULT = "/run/user/{euid}/kilix-gpu-residency-v1"
LOCK_WAIT_SECONDS = 120.0
TERM_GRACE_SECONDS = 5.0
KILL_GRACE_SECONDS = 5.0
RELEASE_WAIT_SECONDS = 10.0
LOAD_TIMEOUT_SECONDS = 600.0  # the longest a need_bytes reservation lasts without loaded() or idle()
# An Ollama model no app registered may be generating for another client: /api/ps cannot say.
# It is not unloaded while its estimated last use is this recent.
RECENT_USE_SECONDS = 120.0
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
_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_RECORD_FIELDS = {"version", "id", "app", "model", "backend", "device", "holder", "pgid", "need_bytes",
                  "state", "pinned", "registered_at", "last_used", "unload", "reserved_until", "pidns", "claim"}
_EVENT_FIELDS = {"id", "app", "model", "freed_bytes", "time", "by", "reason", "confirmed"}
# Command names an Ollama server or runner process has (comm is cut to 15 bytes). Only a
# process with one of these names that holds Ollama's listening socket is taken as Ollama's
# server; its descendants are its runners. A name alone proves nothing: anyone can take it.
_OLLAMA_COMMANDS = ("ollama", "ollama_llama_se", "ollama_llama_server")
# Liveness, process groups and start times are always read from the real /proc. Tests may
# point this attribute elsewhere; no environment variable can.
_PROC = "/proc"


class ResidencyError(RuntimeError):
    """A stable machine-readable refusal; ``code`` is one of ERROR_CODES."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RoomError(ResidencyError):
    """Room could not be made. ``code`` says why (ROOM_ERROR_CODES); ``evicted`` lists what was
    unloaded anyway, which is empty for ``no-room``, ``policy-never`` and ``unmeasured``."""

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
    # Measured now, once per unit: the unit's first row carries it, its other rows 0.
    vram_bytes: int
    need_bytes: int | None
    state: str
    pinned: bool  # the record's own pin, or a pin on its Ollama model
    last_used: float | None
    registered: bool  # False for an Ollama model no app registered
    unload_kind: str | None
    unit: str = ""  # "ollama:NAME", "pgid:N" or "record:ID": rows that unload together share it
    reserved_bytes: int = 0  # VRAM this record still reserves on its device for its load


@dataclass(frozen=True)
class Foreign:
    """A record this copy does not act on: written by another version, or from another PID namespace."""
    id: str
    version: str
    app: str
    model: str
    pid: int | None
    vram_bytes: int
    reason: str  # "newer-schema" or "other-pid-namespace"


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
    confirmed: bool = True  # False: Ollama was asked to unload, but had not when the wait ran out


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
    foreign: tuple[Foreign, ...] = field(default=())


def _refuse(message: str) -> None:
    raise ResidencyError("unavailable", message)


def _invalid(message: str) -> None:
    raise ResidencyError("invalid-request", message)


def _api(function):
    """The public API raises only ResidencyError: an OSError becomes ``unavailable``."""
    @functools.wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ResidencyError:
            raise
        except OSError as error:
            raise ResidencyError("unavailable", f"The registry could not be used: {error.strerror or error}") from error
        except _ollama.OllamaError as error:
            raise ResidencyError("invalid-request", str(error)) from error
    return call


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
    """The registry directory, opened without following links and locked.

    A writer creates the directory and lock when they are missing, takes the lock
    exclusively and removes temporary files a killed writer left. A reader
    (``write=False``) creates nothing, takes the lock shared, and sees an absent
    registry as an empty one."""

    def __init__(self, path: str | None, wait: float, *, write: bool = True) -> None:
        self.parent = self.directory = self.lock = -1
        self.writable = write
        self.absent = False
        path = default_namespace() if path is None else path
        parent_path, self.leaf = os.path.split(path)
        try:
            if (not _LEAF.fullmatch(self.leaf) or not parent_path.startswith("/")
                    or os.path.normpath(path) != path):
                _invalid("Invalid residency namespace")
            self.parent = _open_parent(parent_path)
            created = False
            if write:
                try:
                    os.mkdir(self.leaf, 0o700, dir_fd=self.parent)
                    created = True
                except FileExistsError:
                    pass
            try:
                self.directory = os.open(self.leaf, _DIRECTORY_FLAGS, dir_fd=self.parent)
            except FileNotFoundError:
                if write:
                    raise
                self.absent = True
                return
            if created:
                os.fchmod(self.directory, 0o700)  # whatever the umask
            _directory(self.directory, private=True)
            self.lock = self._open_lock()
            if self.lock < 0:
                self.absent = True
                return
            _HELD.add(self.lock)
            _file_info(self.lock)
            mode = fcntl.LOCK_EX if write else fcntl.LOCK_SH
            deadline = time.monotonic() + wait
            while True:
                try:
                    fcntl.flock(self.lock, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        _refuse("Residency registry is busy")
                    time.sleep(_POLL_SECONDS)
            self.check()
            if write:
                self.clean()
        except BaseException as error:
            self.close()
            if isinstance(error, OSError):
                raise ResidencyError("unavailable", "Residency namespace cannot be opened safely") from error
            raise

    def _open_lock(self) -> int:
        if not self.writable:
            try:
                return os.open("registry.lock", _READ_FLAGS, dir_fd=self.directory)
            except FileNotFoundError:
                return -1
        try:
            fd = os.open("registry.lock", _FILE_FLAGS | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self.directory)
        except FileExistsError:
            return os.open("registry.lock", _FILE_FLAGS, dir_fd=self.directory)
        try:
            os.fchmod(fd, 0o600)  # whatever the umask
        except BaseException:
            os.close(fd)
            raise
        return fd

    def check(self) -> None:
        """Refuse a directory or lock that was replaced since it was opened."""
        if self.absent:
            return
        here = os.stat(self.leaf, dir_fd=self.parent, follow_symlinks=False)
        if _identity(here) != _identity(_directory(self.directory, private=True)):
            _refuse("Residency directory was replaced")
        lock = os.stat("registry.lock", dir_fd=self.directory, follow_symlinks=False)
        if _identity(lock) != _identity(_file_info(self.lock)):
            _refuse("Residency lock was replaced")

    def names(self) -> list[str]:
        if self.absent:
            return []
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
        if self.absent:
            return None
        try:
            fd = os.open(name, _READ_FLAGS | os.O_NONBLOCK, dir_fd=self.directory)
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
        if not self.writable:
            raise AssertionError("a read-only registry view was written")
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
        if not self.writable:
            raise AssertionError("a read-only registry view was written")
        self.check()
        try:
            os.unlink(name, dir_fd=self.directory)
        except FileNotFoundError:
            pass

    def records(self) -> tuple[dict[str, dict], dict[str, dict]]:
        """(records this version understands, records it does not), by id."""
        known, foreign = {}, {}
        for name in sorted(self.names()):
            match = _RECORD_NAME.fullmatch(name)
            if match is None:
                continue
            value = self.read(name)
            if value is None:
                continue
            if _classify(value, match.group(1)) == "known":
                known[value["id"]] = value
            else:
                foreign[value["id"]] = value
        return known, foreign

    def pins(self) -> set[str]:
        """Pinned Ollama model names. A newer version's pins are honoured, never rewritten."""
        value = self.read("pins.json")
        if value is None:
            return set()
        names = value.get("ollama") if type(value) is dict else None
        if (type(value) is not dict or type(value.get("version")) is not str
                or not value["version"].startswith(VERSION_PREFIX) or type(names) is not list
                or not all(type(n) is str and _ollama.MODEL_NAME.fullmatch(n) for n in names)
                or (value["version"] == VERSION and set(value) != {"version", "ollama"})):
            _refuse("Residency pins have an unsupported schema")
        return {_ollama.canonical(name) for name in names}

    def pins_writable(self) -> bool:
        value = self.read("pins.json")
        return value is None or value.get("version") == VERSION

    def save_pins(self, names: set[str]) -> None:
        if not self.pins_writable():
            _refuse("The pins were written by a newer version; they are not rewritten")
        self.write("pins.json", {"version": VERSION, "ollama": sorted(names)})

    def events(self) -> list[dict]:
        """The unload log. A newer version's log is read as empty and never rewritten."""
        value = self.read("events.json")
        if value is None:
            return []
        if (type(value) is dict and type(value.get("version")) is str and value["version"] != VERSION
                and value["version"].startswith(VERSION_PREFIX)):
            return []
        if (type(value) is not dict or set(value) != {"version", "events"} or value["version"] != VERSION
                or type(value["events"]) is not list or not all(_valid_event(e) for e in value["events"])):
            _refuse("Residency events have an unsupported schema")
        return value["events"]

    def add_event(self, event: Eviction) -> None:
        value = self.read("events.json")
        if type(value) is dict and value.get("version") not in (None, VERSION):
            return  # a newer version's log: left as it is
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


def _process(value: object) -> bool:
    return type(value) is dict and set(value) == {"pid", "start"} and _int(value["pid"], 1) and _int(value["start"])


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
    if not _process(holder):
        return False
    if not _valid_unload(unload):
        return False
    # An Ollama resident, and only one, unloads through Ollama; a signal names the record's group.
    kind = None if unload is None else unload["kind"]
    if (value["backend"] == "ollama") != (kind == "ollama"):
        return False
    if kind == "signal" and (unload["pgid"] != value["pgid"] or holder["pid"] != value["pgid"]):
        return False
    if kind == "ollama" and unload["model"] != value["model"]:
        return False
    if value["claim"] is not None and (value["state"] != "unloading" or not _process(value["claim"])):
        return False
    return (type(value["id"]) is str and _ID.fullmatch(value["id"]) is not None
            and type(value["app"]) is str and _LABEL.fullmatch(value["app"]) is not None
            and type(value["model"]) is str and _MODEL.fullmatch(value["model"]) is not None
            and value["backend"] in BACKENDS and value["state"] in STATES
            and type(value["pinned"]) is bool
            and (value["pgid"] is None or _int(value["pgid"], 2))
            and (value["need_bytes"] is None or _int(value["need_bytes"], 1))
            and (value["reserved_until"] is None or (_time(value["reserved_until"]) and value["need_bytes"]))
            and _int(value["pidns"], 1)
            and _time(value["registered_at"]) and _time(value["last_used"]))


def _classify(value: object, ident: str) -> str:
    """"known" for a v1 record; "foreign" for one that is well formed but uses what this version
    does not know (another kilix.gpu-residency version, an added field, an unload kind or backend
    added later). Anything else is refused, and with it the registry."""
    if (type(value) is not dict or value.get("id") != ident or type(value.get("version")) is not str
            or not value["version"].startswith(VERSION_PREFIX) or len(value["version"]) > 64):
        _refuse("Residency record has an unsupported schema")
    if _valid_record(value):
        return "known"
    if value["version"] != VERSION:
        return "foreign"
    unload = value.get("unload")
    if (set(value) - _RECORD_FIELDS or value.get("backend") not in BACKENDS
            or (type(unload) is dict and unload.get("kind") not in UNLOAD_KINDS)):
        return "foreign"
    _refuse("Residency record has an unsupported schema")
    return ""


def _valid_event(value: object) -> bool:
    return (type(value) is dict and set(value) == _EVENT_FIELDS
            and type(value["id"]) is str and len(value["id"]) <= 200
            and type(value["app"]) is str and len(value["app"]) <= 200
            and type(value["model"]) is str and len(value["model"]) <= 200
            and _int(value["freed_bytes"]) and _time(value["time"])
            and type(value["by"]) is str and _LABEL.fullmatch(value["by"]) is not None
            and value["reason"] in ("room", "manual") and type(value["confirmed"]) is bool)

# }}}


# Processes {{{

def _stat(pid: int) -> tuple[str, int, int, int] | None:
    """(state, parent, process group, start time in clock ticks) of a live process."""
    try:
        with open(os.path.join(_PROC, str(pid), "stat"), "rb") as stream:
            text = stream.read(4096).decode("ascii", "replace")
    except OSError:
        return None
    fields = text[text.rfind(")") + 2:].split()
    try:
        state, parent, pgrp, start = fields[0], int(fields[1]), int(fields[2]), int(fields[19])
    except (IndexError, ValueError):
        return None
    return None if state in ("Z", "X") else (state, parent, pgrp, start)


def _start(pid: int) -> int | None:
    found = _stat(pid)
    return None if found is None else found[3]


def _alive(holder: Mapping[str, int]) -> bool:
    return _start(holder["pid"]) == holder["start"]


def _me() -> dict[str, int]:
    start = _start(os.getpid())
    if start is None:
        _refuse("This process cannot see itself in /proc")
    return {"pid": os.getpid(), "start": start}


def _pidns(pid: int | str = "self") -> int | None:
    """The inode of a process's PID namespace."""
    try:
        return os.stat(os.path.join(_PROC, str(pid), "ns", "pid")).st_ino
    except OSError:
        return None


def _own_pidns() -> int:
    found = _pidns()
    if found is None:
        _refuse("This process's PID namespace cannot be read")
    return found


def _command(pid: int) -> str:
    try:
        with open(os.path.join(_PROC, str(pid), "comm"), "rb") as stream:
            return stream.read(64).decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _uid(pid: int) -> int | None:
    try:
        with open(os.path.join(_PROC, str(pid), "status"), "rb") as stream:
            for line in stream.read(8192).decode("ascii", "replace").splitlines():
                if line.startswith("Uid:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _pids() -> list[int]:
    try:
        return [int(name) for name in os.listdir(_PROC) if name.isdigit()]
    except OSError:
        return []


def _table() -> dict[int, tuple[int, int]]:
    """pid -> (parent, process group) for every live process."""
    table = {}
    for pid in _pids():
        found = _stat(pid)
        if found is not None:
            table[pid] = (found[1], found[2])
    return table


def _groups(table: Mapping[int, tuple[int, int]] | None = None) -> dict[int, set[int]]:
    """Process group -> its live members."""
    groups: dict[int, set[int]] = {}
    for pid, (_, pgrp) in (_table() if table is None else table).items():
        groups.setdefault(pgrp, set()).add(pid)
    return groups


def _descends(pid: int, ancestors: set[int] | int, table: Mapping[int, tuple[int, int]] | None = None) -> bool:
    """Whether ``pid`` is one of ``ancestors`` or below one (pid 1 and 0 are never ancestors)."""
    roots = {ancestors} if isinstance(ancestors, int) else ancestors
    seen = 0
    while pid > 1 and seen < 4096:
        if pid in roots:
            return True
        if table is not None:
            parent = table.get(pid, (None, None))[0]
        else:
            found = _stat(pid)
            parent = None if found is None else found[1]
        if parent is None:
            return False
        pid, seen = parent, seen + 1
    return False


def _listeners(port: int) -> list[tuple[int, int]]:
    """(uid, socket inode) of each TCP socket listening on ``port``."""
    found = []
    for name in ("tcp", "tcp6"):
        try:
            with open(os.path.join(_PROC, "net", name), "rb") as stream:
                lines = stream.read(1 << 22).decode("ascii", "replace").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            try:
                local, state, uid, inode = fields[1], fields[3], int(fields[7]), int(fields[9])
                if state == "0A" and int(local.rsplit(":", 1)[1], 16) == port and inode > 0:
                    found.append((uid, inode))
            except (IndexError, ValueError):
                continue
    return found


def _holds_socket(pid: int, inodes: set[int]) -> bool:
    directory = os.path.join(_PROC, str(pid), "fd")
    try:
        descriptors = os.listdir(directory)
    except OSError:
        return False
    wanted = {f"socket:[{inode}]" for inode in inodes}
    for descriptor in descriptors:
        try:
            if os.readlink(os.path.join(directory, descriptor)) in wanted:
                return True
        except OSError:
            continue
    return False


def _ollama_identity(url: str, table: Mapping[int, tuple[int, int]]):
    """A predicate telling Ollama's processes, from something a stranger cannot set.

    Ollama's server is the process named like Ollama that holds the socket listening on the
    configured port; its runners are its descendants. When another user owns that socket (a
    system service), that user's processes are Ollama's: no other user can take its uid."""
    address = _ollama.loopback(url)
    if address is None:
        return lambda pid: False
    listeners = _listeners(address[1])
    own = {inode for uid, inode in listeners if uid == os.geteuid()}
    others = {uid for uid, _ in listeners if uid != os.geteuid()}
    roots = set()
    if own:
        roots = {pid for pid in table if _command(pid) in _OLLAMA_COMMANDS and _holds_socket(pid, own)}

    def identify(pid: int) -> bool:
        if roots and _descends(pid, roots, table):
            return True
        return bool(others) and _uid(pid) in others
    return identify

# }}}


# The view {{{

@dataclass
class _Unit:
    key: str
    kind: str | None  # "ollama", "signal" or None (nothing unloads it)
    records: list[dict]
    model: str | None = None  # the Ollama name
    loaded: _ollama.Loaded | None = None
    pgid: int | None = None
    device_pci: str | None = None
    vram_bytes: int = 0
    pinned: bool = False
    last_used: float | None = None
    pids: set[int] = field(default_factory=set)

    @property
    def registered(self) -> bool:
        return bool(self.records)

    @property
    def idle(self) -> bool:
        return all(_state(record) == "idle" for record in self.records)


@dataclass
class _View:
    records: dict[str, dict]  # live, understood, this PID namespace
    others: list[dict]  # understood records written in another PID namespace
    devices: tuple[vram.Device, ...]
    units: list[_Unit]
    residents: list[ResidentInfo]
    unmanaged: list[Unmanaged]
    foreign: list[Foreign]
    ollama: str
    url: str
    pins: set[str]
    ollama_bytes: dict[str, int]  # what Ollama's processes measurably hold, per device
    pidns: int
    reserved: dict[str, tuple[str | None, int]] = field(default_factory=dict)  # id -> (device, bytes)


def _state(record: dict) -> str:
    """A record left ``unloading`` by an unloader that is gone is idle again."""
    if record["state"] == "unloading" and (record["claim"] is None or not _alive(record["claim"])):
        return "idle"
    return record["state"]


def _record_pids(record: dict, groups: Mapping[int, set[int]]) -> set[int]:
    pids = {record["holder"]["pid"]}
    if record["pgid"] is not None:
        pids |= groups.get(record["pgid"], set())
    return pids


def _bytes_on(devices: Iterable[vram.Device], pids: set[int], pci: str | None) -> int:
    return sum(used for device in devices if pci is None or device.pci == pci
               for pid, used in device.processes.items() if pid in pids)


def _reservation(record: dict, measured: int, now: float) -> int:
    if not record["need_bytes"] or record["reserved_until"] is None or record["reserved_until"] <= now:
        return 0
    return max(0, record["need_bytes"] - measured)


def _keep_alive_estimate(loaded: _ollama.Loaded, now: float) -> float | None:
    """When an unregistered Ollama model was last used: an estimate from its expiry.

    Ollama keeps a model for its keep-alive (OLLAMA_KEEP_ALIVE, else 5 minutes) after each use.
    When the expiry is further away than that allows, the model was given a longer keep-alive
    that /api/ps does not show; the estimate then assumes as much of it has passed as is left.
    None (no estimate) for a model kept for ever, or when the keep-alive cannot be read."""
    if loaded.expires_at is None:
        return None
    keep = _ollama.keep_alive_seconds()
    if keep is None:
        return None
    last = loaded.expires_at - keep
    if last > now:
        last = now - (loaded.expires_at - now)
    return min(last, now)


def _view(ns: _Namespace, *, wake: bool, url: str | None, prune: bool | None = None) -> _View:
    """Measure every GPU, merge Ollama's own list, and group records into units.

    Writers (``prune``) also delete records whose holder is gone and Ollama records whose
    model Ollama no longer lists; readers leave them, but never show them."""
    prune = ns.writable if prune is None else prune
    url = _ollama.default_url() if url is None else url
    me = _own_pidns()
    # A /proc that does not show this very process is not this PID namespace's: nothing in it
    # can be trusted to say a holder is gone.
    trusted = _start(os.getpid()) is not None
    known, unknown = ns.records()
    records, others, foreign_rows = {}, [], []
    for key, record in known.items():
        if record["pidns"] != me:
            others.append(record)
            continue
        if trusted and not _alive(record["holder"]):
            if prune:
                ns.remove(key + ".json")
            continue
        records[key] = record
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
    names = {_ollama.canonical(model.name) for model in loaded}
    if state == "reachable":
        for key, record in list(records.items()):
            if (record["backend"] == "ollama" and record["state"] != "loading"
                    and _ollama.canonical(record["model"]) not in names):
                if prune:
                    ns.remove(key + ".json")
                del records[key]
    if prune:
        for key, record in records.items():
            if record["state"] == "unloading" and _state(record) == "idle":
                record["state"], record["claim"] = "idle", None
                ns.write(key + ".json", record)
    table = _table()
    groups = _groups(table)
    gpu_pids = {pid for device in devices for pid in device.processes}
    identify = _ollama_identity(url, table) if loaded else (lambda pid: False)
    ollama_pids = {pid for pid in gpu_pids if identify(pid)}
    ollama_bytes = {device.pci: _bytes_on([device], ollama_pids, device.pci) for device in devices}
    held = [pci for pci, used in ollama_bytes.items() if used > 0]
    # Ollama models are attributed to a GPU only when Ollama's processes hold memory on just one.
    ollama_device = held[0] if len(held) == 1 else None
    pins = ns.pins()
    now = time.time()
    units: dict[str, _Unit] = {}
    for record in sorted(records.values(), key=lambda r: r["id"]):
        if record["backend"] == "ollama":
            name = _ollama.canonical(record["model"])
            unit = units.setdefault("ollama:" + name, _Unit("ollama:" + name, "ollama", [], model=name))
        elif record["pgid"] is not None:
            key = f"pgid:{record['pgid']}"
            kind = None if record["unload"] is None else record["unload"]["kind"]
            unit = units.setdefault(key, _Unit(key, kind, [], pgid=record["pgid"]))
            if unit.kind != kind:
                unit.kind = None  # records that disagree on how to unload the group: nothing unloads it
        else:
            key = "record:" + record["id"]
            unit = units.setdefault(key, _Unit(key, None, []))
        unit.records.append(record)
    by_name = {_ollama.canonical(model.name): model for model in loaded}
    for name, model in by_name.items():
        units.setdefault("ollama:" + name, _Unit("ollama:" + name, "ollama", [], model=name)).loaded = model
    attributed: set[int] = set()
    for unit in sorted(units.values(), key=lambda u: u.key):
        for record in unit.records:
            if record["backend"] == "ollama":
                attributed |= _record_pids(record, groups)  # the app's own processes, not the model's
        if unit.kind == "ollama":
            unit.device_pci = ollama_device
            declared = unit.loaded.vram_bytes if unit.loaded else 0
            # Ollama's own figure, but never more than its processes measurably hold there.
            unit.vram_bytes = min(declared, ollama_bytes.get(ollama_device, 0)) if ollama_device else declared
            unit.pinned = unit.model in pins or any(r["pinned"] for r in unit.records)
            if unit.records:
                unit.last_used = max(r["last_used"] for r in unit.records)
            elif unit.loaded is not None:
                unit.last_used = _keep_alive_estimate(unit.loaded, now)
            continue
        pids = set()
        for record in unit.records:
            pids |= _record_pids(record, groups)
        unit.pids = pids - attributed
        attributed |= pids
        devices_named = {record["device"]["pci"] for record in unit.records}
        unit.device_pci = devices_named.pop() if len(devices_named) == 1 else None
        unit.vram_bytes = _bytes_on(devices, unit.pids, unit.device_pci)
        unit.pinned = any(r["pinned"] for r in unit.records)
        unit.last_used = max(r["last_used"] for r in unit.records)
    residents = []
    reserved: dict[str, tuple[str | None, int]] = {}
    for unit in sorted(units.values(), key=lambda u: (u.key.startswith("ollama:") and not u.records, u.key)):
        for index, record in enumerate(unit.records):
            measured = _bytes_on(devices, _record_pids(record, groups), record["device"]["pci"])
            pinned = record["pinned"] or (unit.kind == "ollama" and unit.model in pins)
            holds = unit.vram_bytes if unit.kind == "ollama" else measured
            reserved[record["id"]] = (record["device"]["pci"], _reservation(record, holds, now))
            residents.append(ResidentInfo(
                record["id"], record["app"], record["model"], record["backend"], record["device"]["label"],
                unit.device_pci if unit.kind == "ollama" else record["device"]["pci"], record["holder"]["pid"],
                record["pgid"], unit.vram_bytes if index == 0 else 0, record["need_bytes"], _state(record),
                pinned, record["last_used"], True, None if record["unload"] is None else record["unload"]["kind"],
                unit.key, reserved[record["id"]][1]))
        if not unit.records and unit.loaded is not None:
            model = unit.loaded
            residents.append(ResidentInfo("ollama:" + model.name, "ollama", model.name, "ollama", None,
                                          ollama_device, None, None, unit.vram_bytes, None, "idle", unit.pinned,
                                          unit.last_used, False, "ollama", unit.key))
    managed = set(attributed)
    if loaded:
        managed |= ollama_pids
    for key, value in unknown.items():
        holder = value.get("holder")
        pid = holder["pid"] if _process(holder) else None
        pidns = value.get("pidns")
        same = pidns is None or pidns == me
        if pid is not None and same and trusted and not _alive(holder):
            if prune:
                ns.remove(key + ".json")
            continue
        pids = set()
        if pid is not None and same:
            pids = {pid}
            if _int(value.get("pgid"), 2):
                pids |= groups.get(value["pgid"], set())
        managed |= pids
        foreign_rows.append(Foreign(key, value["version"], _label_or(value.get("app")), _model_or(value.get("model")),
                                    pid, _bytes_on(devices, pids, None), "newer-schema"))
    for record in others:
        foreign_rows.append(Foreign(record["id"], record["version"], record["app"], record["model"],
                                    record["holder"]["pid"], 0, "other-pid-namespace"))
    unmanaged = [Unmanaged(pid, _command(pid), device.pci, used)
                 for device in devices for pid, used in sorted(device.processes.items())
                 if pid not in managed and used > 0]
    return _View(records, others, devices, list(units.values()), residents, unmanaged, foreign_rows, state, url,
                 pins, ollama_bytes, me, reserved)


def _label_or(value: object) -> str:
    return value if type(value) is str and _LABEL.fullmatch(value) else "?"


def _model_or(value: object) -> str:
    return value if type(value) is str and _MODEL.fullmatch(value) else "?"

# }}}


# Devices {{{

def _device_map() -> dict[str, str]:
    """KILIX_GPU_DEVICES: an input on purpose (the installer writes it from vulkaninfo). It only
    names devices; it cannot widen what is unloaded or signalled."""
    mapping = {}
    for item in os.environ.get(DEVICE_MAP_ENV, "").replace(";", ",").split(","):
        label, sep, pci = item.partition("=")
        if sep and _LABEL.fullmatch(label.strip()) and vram.normalise_pci(pci):
            mapping[label.strip()] = vram.normalise_pci(pci)  # type: ignore[assignment]
    return mapping


def resolve_device(label: str | None, devices: Iterable[vram.Device]) -> str | None:
    """The PCI address a device label names.

    A PCI address names itself; any other label must be in KILIX_GPU_DEVICES
    (``Vulkan1=0000:01:00.0,...``). Either must name a GPU this machine has, or
    ResidencyError("invalid-request") is raised. Only ``None`` means "the one GPU
    whose VRAM can be measured"; it gives None when there is not exactly one."""
    devices = tuple(devices)
    present = {device.pci for device in devices}
    if label is None:
        measured = [device.pci for device in devices if device.counter]
        return measured[0] if len(measured) == 1 else None
    if type(label) is not str or not _LABEL.fullmatch(label):
        _invalid("Invalid device label")
    pci = vram.normalise_pci(label)
    if pci is not None:
        if pci not in present:
            _invalid(f"There is no GPU at {pci}")
        return pci
    mapped = _device_map().get(label)
    if mapped is None:
        _invalid(f"Unknown device label {label}: name a PCI address, or map the label in {DEVICE_MAP_ENV}")
    if mapped not in present:
        _invalid(f"{DEVICE_MAP_ENV} maps {label} to {mapped}, which is not a GPU here")
    return mapped

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


def _leader_handle(leader: Mapping[str, int]) -> int | None:
    """A pidfd for the registered leader, opened only while its start time still matches."""
    try:
        handle = os.pidfd_open(leader["pid"])
    except (AttributeError, OSError):
        return None
    if not _alive(leader):
        os.close(handle)
        return None
    return handle


def _exited(handle: int | None) -> bool:
    if handle is None:
        return False
    return bool(select.select([handle], [], [], 0)[0])


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
    handle = _leader_handle(leader)
    try:
        for sig, grace in ((signal.SIGTERM, TERM_GRACE_SECONDS), (signal.SIGKILL, KILL_GRACE_SECONDS)):
            if not _alive(leader) and not _groups().get(pgid):
                return
            if sig == signal.SIGTERM and (_exited(handle) or not _alive(leader)):
                raise ResidencyError("refused", "The registered leader is gone; its group is not signalled")
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            except PermissionError as error:
                raise ResidencyError("refused", "The process group belongs to another user") from error
            if _wait_gone(pgid, leader, grace):
                return
        raise ResidencyError("refused", "The process group did not exit")
    finally:
        if handle is not None:
            os.close(handle)


@dataclass
class _Claim:
    """One unit marked ``unloading`` by this process, to be unloaded with the lock released."""
    unit: _Unit
    ids: list[str]  # the records marked, including a placeholder for an unregistered Ollama model
    prior: dict[str, str]  # id -> the state before
    placeholder: str | None = None
    rows: tuple[ResidentInfo, ...] = ()


def _claim(ns: _Namespace, view: _View, units: Iterable[_Unit], *, me: dict[str, int]) -> list[_Claim]:
    claims = []
    for unit in units:
        ids, prior = [], {}
        for record in unit.records:
            prior[record["id"]] = _state(record)
            record["state"], record["claim"] = "unloading", dict(me)
            ns.write(record["id"] + ".json", record)
            ids.append(record["id"])
        placeholder = None
        if not unit.records:
            # An Ollama model nobody registered: a placeholder record says it is being unloaded, so
            # nobody else plans it meanwhile. It follows this process and is pruned if it dies.
            now = time.time()
            placeholder = uuid.uuid4().hex
            ns.write(placeholder + ".json", {
                "version": VERSION, "id": placeholder, "app": "ollama", "model": unit.model, "backend": "ollama",
                "device": {"label": None, "pci": unit.device_pci}, "holder": dict(me), "pgid": None,
                "need_bytes": None, "state": "unloading", "pinned": False, "registered_at": now, "last_used": now,
                "unload": {"kind": "ollama", "model": unit.model}, "reserved_until": None, "pidns": view.pidns,
                "claim": dict(me)})
            ids.append(placeholder)
        rows = tuple(r for r in view.residents if r.unit == unit.key)
        claims.append(_Claim(unit, ids, prior, placeholder, rows))
    return claims


def _unload_one(claim: _Claim, url: str) -> bool:
    """Unload a claimed unit, without the lock. True when confirmed; False when Ollama was asked
    but had not unloaded the model when the wait ran out. Raises ResidencyError on failure."""
    unit = claim.unit
    if unit.kind == "signal":
        record = next(r for r in unit.records if r["holder"]["pid"] == unit.pgid)
        _signal_unload(record)
        return True
    try:
        _ollama.unload(url, unit.loaded.name if unit.loaded else unit.model)
    except _ollama.OllamaTimeout:
        return False
    except _ollama.OllamaError as error:
        raise ResidencyError("unavailable", str(error)) from error
    return True


def _execute(namespace: str | None, url: str, claims: list[_Claim], by: str, reason: str
             ) -> tuple[list[Eviction], ResidencyError | None]:
    """Unload the claimed units in order, stopping at the first failure, then settle the registry.

    The units unloaded are removed and logged; the one that failed is left busy (it is still
    resident and must not look evictable); the ones not reached return to their state."""
    outcomes: dict[int, str] = {}
    failure: ResidencyError | None = None
    try:
        for index, claim in enumerate(claims):
            outcomes[index] = "failed"  # until it is known otherwise: an interrupt leaves it busy
            try:
                outcomes[index] = "done" if _unload_one(claim, url) else "unconfirmed"
            except ResidencyError as error:
                failure = error
                break
    finally:
        events = _settle(namespace, claims, outcomes, by, reason)
    return events, failure


def _settle(namespace: str | None, claims: list[_Claim], outcomes: Mapping[int, str], by: str,
            reason: str) -> list[Eviction]:
    events = []
    with _Namespace(namespace, LOCK_WAIT_SECONDS) as ns:
        known, _ = ns.records()
        for index, claim in enumerate(claims):
            outcome = outcomes.get(index, "skipped")
            if outcome in ("done", "unconfirmed"):
                for ident in claim.ids:
                    ns.remove(ident + ".json")
                rows = claim.rows or ()
                for position, row in enumerate(rows):
                    event = Eviction(row.id, row.app, row.model, claim.unit.vram_bytes if position == 0 else 0,
                                     time.time(), by, reason, outcome == "done")
                    ns.add_event(event)
                    events.append(event)
                continue
            if claim.placeholder is not None:
                ns.remove(claim.placeholder + ".json")
            for ident in claim.ids:
                record = known.get(ident)
                if record is None or record["state"] != "unloading":
                    continue  # gone, or its app has since changed it
                if outcome == "failed":
                    if not _alive(record["holder"]):
                        ns.remove(ident + ".json")  # its holder is gone: nothing is resident
                        continue
                    record["state"] = "busy"  # still resident: it must not look evictable
                else:
                    record["state"] = claim.prior.get(ident, "idle")
                    if record["state"] == "unloading":
                        record["state"] = "idle"
                record["claim"] = None
                ns.write(ident + ".json", record)
    return events

# }}}


# Room {{{

def _reserved(view: _View, pci: str, exclude: str | None = None) -> int:
    """VRAM records reserve on ``pci`` for their loads, beyond what they hold yet."""
    now = time.time()
    total = sum(amount for ident, (device, amount) in view.reserved.items()
                if device == pci and ident != exclude)
    for record in view.others:  # written elsewhere: what they hold cannot be measured from here
        if record["device"]["pci"] == pci and record["id"] != exclude:
            total += _reservation(record, 0, now)
    return total


def _recent(unit: _Unit, now: float) -> bool:
    """An unregistered Ollama model that may be in use: its estimated last use is within
    RECENT_USE_SECONDS, or cannot be estimated (a model kept for ever)."""
    return not unit.registered and (unit.last_used is None or now - unit.last_used < RECENT_USE_SECONDS)


def _candidate(unit: _Unit, pci: str, now: float) -> bool:
    return (unit.device_pci == pci and unit.kind is not None and not unit.pinned and unit.idle
            and unit.vram_bytes > 0 and (unit.kind != "ollama" or unit.loaded is not None)
            and not _recent(unit, now))


def _plan(view: _View, pci: str, need_bytes: int, exclude: str | None = None
          ) -> tuple[int, list[_Unit], int]:
    """(free now, the units to unload, what unloading them frees). Registered residents go before
    Ollama models nobody registered, each least recently used first; an Ollama model counts for
    no more than Ollama's processes measurably hold on the device."""
    target = next(d for d in view.devices if d.pci == pci)
    free = target.free_bytes - _reserved(view, pci, exclude)
    if free >= need_bytes:
        return free, [], 0
    budget = view.ollama_bytes.get(pci, 0)
    now = time.time()
    found = sorted((u for u in view.units if _candidate(u, pci, now)),
                   key=lambda u: (not u.registered, u.last_used is None, u.last_used or 0.0, u.key))
    plan, freed = [], 0
    for unit in found:
        gain = unit.vram_bytes
        if unit.kind == "ollama":
            gain = min(gain, budget)
            budget -= gain
        if gain <= 0:
            continue
        plan.append(unit)
        freed += gain
        if free + freed >= need_bytes:
            break
    return free, plan, freed


def _rows(view: _View, units: Iterable[_Unit]) -> tuple[ResidentInfo, ...]:
    keys = [unit.key for unit in units]
    return tuple(row for key in keys for row in view.residents if row.unit == key)


def _target(view: _View, device: str | None, need_bytes: int) -> str:
    pci = resolve_device(device, view.devices)
    target = next((d for d in view.devices if d.pci == pci), None)
    if target is None or target.free_bytes is None:
        raise RoomError("The device's free VRAM cannot be measured", need_bytes=need_bytes, free_bytes=None,
                        device=pci, code="unmeasured")
    return pci


def _check_room_request(need_bytes: object, requester: object, policy: object) -> None:
    if not _int(need_bytes, 1):
        _invalid("need_bytes must be a positive integer")
    if policy not in POLICIES:
        _invalid("Unknown residency policy")
    if type(requester) is not str or not _LABEL.fullmatch(requester):
        _invalid("Invalid requester label")


def _make_room(namespace: str | None, url: str | None, need_bytes: int, device: str | None, requester: str,
               policy: str, wait: float, record: dict | None = None) -> Room:
    """Plan under the lock; unload with it released; then wait, re-locking to measure.

    With ``record`` (register), the record is written under the planning lock, so its
    reservation keeps the room being freed from other apps, and removed again on failure."""
    write = record is not None or policy != "never"
    me = _me()
    exclude = None if record is None else record["id"]
    with _Namespace(namespace, wait, write=write) as ns:
        view = _view(ns, wake=True, url=url)
        if record is not None and len(view.records) + len(view.others) + len(view.foreign) >= MAX_RECORDS:
            _refuse("The residency registry is full")
        pci = _target(view, device, need_bytes)
        if record is not None:
            record["device"]["pci"] = pci
        free, plan, freed = _plan(view, pci, need_bytes, exclude)
        rows = _rows(view, plan)
        if not plan and free >= need_bytes:
            if record is not None:
                ns.write(record["id"] + ".json", record)
            return Room(pci, need_bytes, free)
        if free + freed < need_bytes:
            raise RoomError("Not enough VRAM even if every idle resident were unloaded", need_bytes=need_bytes,
                            free_bytes=free, device=pci, plan=rows)
        if policy == "never":
            raise RoomError("Room needs unloading, and the policy is never", need_bytes=need_bytes,
                            free_bytes=free, device=pci, plan=rows, code="policy-never")
        if record is not None:
            ns.write(record["id"] + ".json", record)
        claims = _claim(ns, view, plan, me=me)
        url = view.url
    try:
        evicted, failure = _execute(namespace, url, claims, requester, "room")
    except BaseException:
        if record is not None:
            _forget(namespace, record["id"])
        raise
    if failure is not None or any(not e.confirmed for e in evicted):
        if record is not None:
            _forget(namespace, record["id"])
        message = f"Unloading failed: {failure}" if failure else "Ollama had not unloaded the model in time"
        raise RoomError(message, need_bytes=need_bytes, free_bytes=free, device=pci, plan=rows,
                        evicted=tuple(evicted), code="unload-failed") from failure
    deadline = time.monotonic() + RELEASE_WAIT_SECONDS
    while True:
        with _Namespace(namespace, wait, write=False) as ns:
            view = _view(ns, wake=True, url=url)
            target = next((d for d in view.devices if d.pci == pci), None)
            free = (target.free_bytes if target and target.free_bytes is not None else 0) - _reserved(view, pci,
                                                                                                    exclude)
        if free >= need_bytes:
            return Room(pci, need_bytes, free, tuple(evicted), rows)
        if time.monotonic() >= deadline:
            if record is not None:
                _forget(namespace, record["id"])
            raise RoomError("Unloaded residents, but the VRAM did not come back", need_bytes=need_bytes,
                            free_bytes=free, device=pci, plan=rows, evicted=tuple(evicted), code="not-released")
        time.sleep(_POLL_SECONDS * 4)


def _forget(namespace: str | None, ident: str) -> None:
    with _Namespace(namespace, LOCK_WAIT_SECONDS) as ns:
        ns.remove(ident + ".json")


@_api
def ensure_room(need_bytes: int, *, device: str | None, requester: str, policy: str = "evict-idle",
                namespace: str | None = None, ollama: str | None = None,
                wait: float = LOCK_WAIT_SECONDS) -> Room:
    """Make ``need_bytes`` of VRAM free on ``device``, unloading idle residents if allowed.

    Returns Room(device_pci, need_bytes, free_bytes, evicted, plan). Raises RoomError: with
    ``no-room`` or ``policy-never`` having unloaded nothing (its ``plan`` lists what would have
    been unloaded), or ``unload-failed`` / ``not-released`` after unloading what ``evicted``
    lists. With policy ``never`` it writes nothing at all."""
    _check_room_request(need_bytes, requester, policy)
    return _make_room(namespace, ollama, need_bytes, device, requester, policy, wait)

# }}}


# Registration {{{

def _holder(pid: int) -> dict[str, int]:
    found = _stat(pid)
    if found is None:
        raise ResidencyError("invalid-request", "The process to register is not running")
    return {"pid": pid, "start": found[3]}


class Resident:
    """A registered model. Leaving a ``with`` block removes the record, unloading
    the model first if it is still loaded and nothing else holds or pins it."""

    def __init__(self, record: dict, namespace: str | None, url: str | None, room: Room | None) -> None:
        self.id = record["id"]
        self.app = record["app"]
        self.model = record["model"]
        self.backend = record["backend"]
        self.room = room
        self._namespace = namespace
        self._url = url

    @_api
    def _update(self, **changes: object) -> dict:
        with _Namespace(self._namespace, LOCK_WAIT_SECONDS) as ns:
            record = ns.records()[0].get(self.id)
            if record is None or not _alive(record["holder"]):
                if record is not None:
                    ns.remove(self.id + ".json")
                raise ResidencyError("gone", "The residency record is gone")
            record.update(changes)
            if "state" in changes:
                record["claim"] = None
            if changes.get("state") == "idle" or changes.get("loaded"):
                record["reserved_until"] = None
            record.pop("loaded", None)
            record["last_used"] = time.time()
            if not _valid_record(record):
                _invalid("Invalid residency update")
            ns.write(self.id + ".json", record)
            return record

    def busy(self) -> None:
        """Working now: not evictable. A load reservation still stands."""
        self._update(state="busy")

    def idle(self) -> None:
        """Loaded and not working: evictable. Ends the load reservation."""
        self._update(state="idle")

    def loaded(self) -> None:
        """The model is fully loaded: end the load reservation, keeping the state."""
        self._update(loaded=True)

    def attach(self, *, pid: int | None = None, pgid: int | None = None) -> None:
        """Follow a model process started after registering (its group, when given). It must be
        this process's descendant. A group becomes the unload (TERM, then KILL) of a process or
        http resident; an Ollama resident still unloads through Ollama."""
        if (pid is None) == (pgid is None):
            _invalid("Attach a pid or a pgid")
        fields = _process_fields(pid, pgid, "auto", self.backend, self.model)
        self._update(**fields)

    @property
    @_api
    def evicted(self) -> Eviction | None:
        """How this record was unloaded by another app or by hand, if it was."""
        with _Namespace(self._namespace, LOCK_WAIT_SECONDS, write=False) as ns:
            for event in reversed(ns.events()):
                if event["id"] == self.id:
                    return Eviction(**event)
        return None

    @_api
    def close(self, *, unload: bool = True) -> None:
        """Remove the record. With ``unload`` the model is unloaded first, unless another live
        record holds the same model or group, or it is pinned: then only this record goes."""
        me = _me()
        claims = []
        with _Namespace(self._namespace, LOCK_WAIT_SECONDS) as ns:
            record = ns.records()[0].get(self.id)
            if record is None:
                return
            if unload and record["unload"] is not None and _alive(record["holder"]) and record["claim"] is None:
                view = _view(ns, wake=False, url=self._url)
                unit = next((u for u in view.units if any(r["id"] == self.id for r in u.records)), None)
                shared = unit is None or len(unit.records) > 1 or unit.pinned
                never_loaded = unit is not None and unit.kind == "ollama" and (
                    unit.loaded is None or (unit.vram_bytes == 0 and record["state"] == "loading"))
                if not shared and not never_loaded and unit.kind is not None:
                    claims = _claim(ns, view, [unit], me=me)
                    url = view.url
            if not claims:
                ns.remove(self.id + ".json")
                return
        _execute(self._namespace, url, claims, self.app, "manual")
        _forget(self._namespace, self.id)

    def __enter__(self) -> Resident:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _check_ours(pid: int, what: str) -> None:
    """Only this process, or a process it started (its descendant), may be followed or signalled."""
    if pid != os.getpid() and not _descends(pid, os.getpid()):
        _invalid(f"The {what} is not this process's own: only a process it started may be registered")


def _process_fields(pid: int | None, pgid: int | None, unload: object, backend: str, model: str) -> dict:
    if pgid is not None:
        if not _int(pgid, 2):
            _invalid("Invalid process group")
        found = _stat(pgid)
        if found is None or found[2] != pgid:
            _invalid("The process group's leader is not running")
        holder = {"pid": pgid, "start": found[3]}
        if pgid != os.getpgrp():
            _check_ours(pgid, "process group")
    elif pid is not None:
        if not _int(pid, 1):
            _invalid("Invalid pid")
        holder = _holder(pid)
        _check_ours(pid, "process")
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


@_api
def register(*, app: str, model: str, backend: str, device: str | None = None, pid: int | None = None,
             pgid: int | None = None, need_bytes: int | None = None, policy: str = "evict-idle",
             pinned: bool = False, unload: object = "auto", namespace: str | None = None,
             ollama: str | None = None, wait: float = LOCK_WAIT_SECONDS) -> Resident:
    """Record a model as resident (state ``loading``) and return its handle.

    With ``need_bytes``, room is made first and the record is written under the same lock as
    the plan, so no other app can take the space being freed; the record then reserves
    ``need_bytes`` beyond what its processes hold until ``loaded()`` or ``idle()``, or for
    LOAD_TIMEOUT_SECONDS. ``busy()`` keeps the reservation. Without a pid or pgid the record
    follows the calling process; ``attach`` points it at a model process started afterwards,
    which must be this process's descendant. For backend ``ollama`` the model is the Ollama
    model name and the unload is ``keep_alive: 0``; for a process group the unload is TERM,
    then KILL. Every refusal is raised before anything is unloaded or written."""
    if type(app) is not str or not _LABEL.fullmatch(app):
        _invalid("Invalid app label")
    if backend not in BACKENDS:
        _invalid("Unknown backend")
    if type(model) is not str or not _MODEL.fullmatch(model):
        _invalid("Invalid model display name")
    if backend == "ollama" and not _ollama.MODEL_NAME.fullmatch(model):
        _invalid("Invalid Ollama model name")
    if device is not None and (type(device) is not str or not _LABEL.fullmatch(device)):
        _invalid("Invalid device label")
    if need_bytes is not None:
        _check_room_request(need_bytes, app, policy)
    elif policy not in POLICIES:
        _invalid("Unknown residency policy")
    if type(pinned) is not bool:
        _invalid("pinned must be a bool")
    fields = _process_fields(pid, pgid, unload, backend, model)
    if backend == "ollama" and (fields["unload"] is None or fields["unload"]["kind"] != "ollama"):
        _invalid("An Ollama resident unloads through Ollama")
    if backend != "ollama" and fields["unload"] is not None and fields["unload"]["kind"] == "ollama":
        _invalid("Only an Ollama resident unloads through Ollama")
    now = time.time()
    record = {"version": VERSION, "id": uuid.uuid4().hex, "app": app, "model": model, "backend": backend,
              "device": {"label": device if device is None or vram.normalise_pci(device) is None else None,
                         "pci": None},
              "need_bytes": need_bytes, "state": "loading", "pinned": pinned, "registered_at": now,
              "last_used": now, "reserved_until": None if need_bytes is None else now + LOAD_TIMEOUT_SECONDS,
              "pidns": _own_pidns(), "claim": None, **fields}
    if not _valid_record(record):
        _invalid("Invalid registration")
    room = None
    if need_bytes is not None:
        room = _make_room(namespace, ollama, need_bytes, device, app, policy, wait, record)
    else:
        with _Namespace(namespace, wait) as ns:
            view = _view(ns, wake=False, url=ollama)
            if len(view.records) + len(view.others) + len(view.foreign) >= MAX_RECORDS:
                _refuse("The residency registry is full")
            record["device"]["pci"] = resolve_device(device, view.devices)
            ns.write(record["id"] + ".json", record)
    return Resident(record, namespace, ollama, room)

# }}}


# Reading and administration {{{

@_api
def status(*, namespace: str | None = None, ollama: str | None = None, wake: bool = False,
           wait: float = LOCK_WAIT_SECONDS) -> Status:
    """Devices, residents (registered and Ollama-discovered), unmanaged GPU processes,
    records this copy does not act on, and recent evictions. It writes nothing, and
    creates nothing: an absent registry reads as empty. Without ``wake`` a sleeping GPU
    is not woken to be read."""
    with _Namespace(namespace, wait, write=False) as ns:
        view = _view(ns, wake=wake, url=ollama)
        events = tuple(Eviction(**event) for event in ns.events())
    return Status(view.devices, tuple(view.residents), tuple(view.unmanaged), view.ollama, events,
                  tuple(view.foreign))


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
    if by_model and len({info.unit for info in by_model}) == 1:
        return by_model[0]
    if by_model:
        _invalid("Several residents have that model; name one by id")
    if any(u.command == target or str(u.pid) == target for u in view.unmanaged):
        raise ResidencyError("refused", "Unmanaged GPU processes are never unloaded")
    if any(f.id == target for f in view.foreign):
        raise ResidencyError("refused", "This record was written by another version or PID namespace; it is "
                                        "never unloaded from here")
    raise ResidencyError("not-found", "No such resident")


@_api
def unload(target: str, *, force: bool = False, shared: bool = False, by: str = "kilix-gpu",
           namespace: str | None = None, ollama: str | None = None, wait: float = LOCK_WAIT_SECONDS) -> Eviction:
    """Unload one resident by id or model name, with everything that unloads with it: one Ollama
    model, or one process group. Refused unless ``force`` when any record holding it is pinned,
    busy, loading or unloading, or when the model is pinned. A unit other live records also hold
    is refused, naming them, unless ``shared`` (every holder is still checked) or ``force``. One
    without an unload action is always refused. Returns the first Eviction; ``confirmed`` is
    False when Ollama had not unloaded the model when the wait ran out."""
    if type(by) is not str or not _LABEL.fullmatch(by):
        _invalid("Invalid requester label")
    me = _me()
    with _Namespace(namespace, wait) as ns:
        view = _view(ns, wake=False, url=ollama)
        info = _find(view, target)
        unit = next(u for u in view.units if u.key == info.unit)
        if unit.kind is None:
            raise ResidencyError("refused", "This resident has no unload action")
        if any(_state(r) == "unloading" for r in unit.records) and any(r["claim"] for r in unit.records):
            raise ResidencyError("refused", "This resident is unloading")
        if not force:
            others = [r for r in unit.records if r["id"] != info.id]
            if others and not shared:
                names = ", ".join(f"{r['app']} ({r['id']})" for r in others)
                raise ResidencyError("refused", f"This model is also held by {names}; use force")
            if unit.pinned:
                raise ResidencyError("refused", "This resident is pinned")
            for record in unit.records:
                if _state(record) != "idle":
                    raise ResidencyError("refused", f"This resident is {_state(record)}")
        claims = _claim(ns, view, [unit], me=me)
        url = view.url
    events, failure = _execute(namespace, url, claims, by, "manual")
    if failure is not None:
        raise failure
    return next((e for e in events if e.id == info.id), events[0])


@_api
def pin(target: str, pinned: bool = True, *, namespace: str | None = None, ollama: str | None = None,
        wait: float = LOCK_WAIT_SECONDS) -> ResidentInfo:
    """Pin (or unpin) a resident, so ``ensure_room`` never unloads it.

    A record id pins that record. An Ollama model, named by its name or as ``ollama:NAME``,
    is pinned as a model: the pin holds for every record that names it, now or later, and for
    the model when nobody registered it. Any other model name pins the one record that has it.
    A record's own pin and a model pin both count; unpinning one leaves the other."""
    with _Namespace(namespace, wait) as ns:
        view = _view(ns, wake=False, url=ollama)
        name = target[len("ollama:"):] if type(target) is str and target.startswith("ollama:") else None
        info = None
        if name is not None:
            info = next((r for r in view.residents if r.backend == "ollama"
                         and _ollama.canonical(r.model) == _ollama.canonical(name)), None)
            if info is None:
                raise ResidencyError("not-found", "No such resident")
        else:
            info = _find(view, target)
        unit = next(u for u in view.units if u.key == info.unit)
        if unit.kind == "ollama" and (name is not None or info.id != target):
            names = ns.pins()
            (names.add if pinned else names.discard)(unit.model)
            ns.save_pins(names)
            view.pins = names
            effective = bool(pinned) or any(r["pinned"] for r in unit.records)
        else:
            record = view.records[info.id]
            record["pinned"] = bool(pinned)
            ns.write(info.id + ".json", record)
            effective = bool(pinned) or (unit.kind == "ollama" and unit.model in view.pins)
    return ResidentInfo(**{**info.__dict__, "pinned": effective})


def unpin(target: str, *, namespace: str | None = None, ollama: str | None = None,
          wait: float = LOCK_WAIT_SECONDS) -> ResidentInfo:
    """Undo ``pin``: the same target forms. A record's own pin and a model pin are separate;
    the returned ``pinned`` says whether one still holds."""
    return pin(target, False, namespace=namespace, ollama=ollama, wait=wait)

# }}}
