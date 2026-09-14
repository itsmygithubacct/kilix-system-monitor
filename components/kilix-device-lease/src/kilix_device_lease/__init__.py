"""Versioned, cooperative per-user accelerator ownership.

Every workload uses one conservative execution lock. A grant is not model,
driver, hardware, VRAM-fit or co-residency admission. Pass ``Lease.guard_fd``
to the dedicated job supervisor before allocating a model, and retain it
through complete descendant cleanup. Propagate it to engines where possible.

Successful cleanup requires explicit ``release(cleanup_complete=True)``.
Default/context release leaves the persistent grant quarantined. Process
death, PID reuse or the disappearance of all FD copies is never cleanup proof.
No existing microphone, half-duplex or voice daemon behavior is changed.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import stat
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

VERSION = "kilix.device-lease/v1"
WORKLOADS = ("tts-utterance", "stt-job", "llm-turn")
MAX_QUEUE = 24
MAX_WORKLOAD_QUEUE = 8
MAX_WAIT_SECONDS = 3600.0
_POLL_SECONDS = 0.02
_MAX_BYTES = 32768
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z", re.ASCII)
_TOKEN = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_LEAF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z", re.ASCII)
_FILE_FLAGS = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW


class LeaseError(RuntimeError):
    """A stable machine-readable refusal, with no user content in its text."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class QueueStatus:
    version: str
    ticket: str
    state: str
    position: int


def _refuse(message: str) -> None:
    raise LeaseError("unavailable", message)


def _identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino]


def _file_info(fd: int) -> os.stat_result:
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
        _refuse("Shared lease file ownership, mode or link count is invalid")
    return info


def _directory(fd: int, *, private: bool) -> os.stat_result:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        _refuse("Shared lease directory is not a directory")
    if private:
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            _refuse("Shared lease directory must be user-owned mode 0700")
    elif (info.st_uid not in (0, os.geteuid())
          or (info.st_mode & 0o022 and not (info.st_uid == 0 and info.st_mode & stat.S_ISVTX))):
        _refuse("Shared lease ancestor is not trusted")
    return info


def _open_parent(path: str) -> int:
    """No-follow traversal; public root-owned sticky ancestors may contain fixtures."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        _directory(fd, private=False)
        for component in path.split("/")[1:]:
            if not component or component in (".", ".."):
                _refuse("Shared lease namespace must use canonical absolute components")
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=fd)
            os.close(fd)
            fd = child
            _directory(fd, private=False)
        _directory(fd, private=True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _refuse("Shared lease record has duplicate fields")
        result[key] = value
    return result


def _read(fd: int) -> dict:
    info = _file_info(fd)
    if not 0 < info.st_size <= _MAX_BYTES:
        _refuse("Shared lease record size is invalid")
    raw = os.pread(fd, _MAX_BYTES + 1, 0)
    if len(raw) != info.st_size:
        _refuse("Shared lease record changed during reading")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs,
                           parse_constant=lambda _: _refuse("Non-finite shared lease record"))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise LeaseError("unavailable", "Shared lease record is malformed") from error
    if type(value) is not dict:
        _refuse("Shared lease record is not an object")
    return value


def _write(fd: int, value: dict) -> None:
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if len(raw) > _MAX_BYTES:
        _refuse("Shared lease record exceeds its bound")
    os.ftruncate(fd, 0)
    done = 0
    while done < len(raw):
        count = os.pwrite(fd, raw[done:], done)
        if count <= 0:
            _refuse("Cannot write shared lease record")
        done += count
    os.fsync(fd)


def _token(value: object) -> bool:
    return type(value) is str and _TOKEN.fullmatch(value) is not None


def _label(value: object) -> bool:
    return type(value) is str and _LABEL.fullmatch(value) is not None


def _number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _entry(value: object, *, queued: bool) -> bool:
    fields = {"ticket", "job_id", "workload", "device"}
    fields |= {"sequence", "deadline", "inode"} if queued else {"state"}
    if type(value) is not dict or set(value) != fields:
        return False
    if not (_token(value["ticket"]) and _label(value["job_id"]) and _label(value["device"])
            and type(value["workload"]) is str and value["workload"] in WORKLOADS):
        return False
    if not queued:
        return value["state"] in ("held", "releasing")
    return (type(value["sequence"]) is int and 0 <= value["sequence"] < 2**63
            and _number(value["deadline"]) and _inode(value["inode"]))


def _inode(value: object) -> bool:
    return (type(value) is list and len(value) == 2
            and all(type(x) is int and x >= 0 for x in value))


def _state(value: dict) -> None:
    if (set(value) != {"version", "next", "last", "active", "queue"}
            or value["version"] != VERSION
            or type(value["next"]) is not int or not 0 <= value["next"] < 2**63
            or value["last"] not in (None, *WORKLOADS)
            or type(value["queue"]) is not list or len(value["queue"]) > MAX_QUEUE
            or (value["active"] is not None and not _entry(value["active"], queued=False))
            or not all(_entry(item, queued=True) for item in value["queue"])):
        _refuse("Shared lease state has an unsupported schema")
    tickets = [item["ticket"] for item in value["queue"]]
    sequences = [item["sequence"] for item in value["queue"]]
    if (len(set(tickets)) != len(tickets) or len(set(sequences)) != len(sequences)
            or any(seq >= value["next"] for seq in sequences)
            or (value["active"] and value["active"]["ticket"] in tickets)
            or any(sum(item["workload"] == kind for item in value["queue"]) > MAX_WORKLOAD_QUEUE
                   for kind in WORKLOADS)):
        _refuse("Shared lease queue identities are invalid")


def _check_request(deadline: float, cancelled: Callable[[], bool] | None,
                   disconnected: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise LeaseError("cancelled", "Shared lease request was cancelled")
    if disconnected is not None and disconnected():
        raise LeaseError("cancelled", "Shared lease client disconnected")
    if time.monotonic() >= deadline:
        raise LeaseError("deadline", "Shared lease deadline expired")


class _Registry:
    """Persistent namespace anchor; all queue mutations hold its kernel lock."""

    def __init__(self, path: str, deadline: float,
                 cancelled: Callable[[], bool] | None = None,
                 disconnected: Callable[[], bool] | None = None, *, blocking: bool = True) -> None:
        self.parent = self.anchor = self.directory = self.resource = -1
        self.path = path
        self.parent_path, self.leaf = os.path.split(path)
        try:
            if not _LEAF.fullmatch(self.leaf) or not self.parent_path.startswith("/"):
                _refuse("Invalid shared lease namespace")
            self.parent = _open_parent(self.parent_path)
            self.parent_identity = _identity(os.fstat(self.parent))
            self.anchor_name = "." + self.leaf + ".lease-v1.anchor"
            created = False
            try:
                self.anchor = os.open(self.anchor_name, _FILE_FLAGS | os.O_CREAT | os.O_EXCL,
                                      0o600, dir_fd=self.parent)
                os.fchmod(self.anchor, 0o600)
                created = True
            except FileExistsError:
                self.anchor = os.open(self.anchor_name, _FILE_FLAGS, dir_fd=self.parent)
            self.anchor_identity = _identity(_file_info(self.anchor))
            while True:
                _check_request(deadline, cancelled, disconnected)
                try:
                    fcntl.flock(self.anchor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not blocking:
                        _refuse("Shared lease registry is busy")
                    time.sleep(_POLL_SECONDS)
            if created:
                os.mkdir(self.leaf, mode=0o700, dir_fd=self.parent)
                self.directory = os.open(self.leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                         dir_fd=self.parent)
                os.fchmod(self.directory, 0o700)
                self.resource = os.open("accelerator.lock", _FILE_FLAGS | os.O_CREAT | os.O_EXCL,
                                        0o600, dir_fd=self.directory)
                os.fchmod(self.resource, 0o600)
                self.identity = {"version": VERSION, "directory": _identity(_directory(self.directory, private=True)),
                                 "resource": _identity(_file_info(self.resource))}
                self.value = {"version": VERSION, "next": 0, "last": None, "active": None, "queue": []}
                self.save()
                _write(self.anchor, self.identity)
                os.fsync(self.parent)
            else:
                self.identity = _read(self.anchor)
                if (set(self.identity) != {"version", "directory", "resource"}
                        or self.identity["version"] != VERSION
                        or not _inode(self.identity["directory"]) or not _inode(self.identity["resource"])):
                    _refuse("Shared lease namespace identity is invalid")
                self.directory = os.open(self.leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                         dir_fd=self.parent)
                self.resource = os.open("accelerator.lock", _FILE_FLAGS, dir_fd=self.directory)
                fd = os.open("state.json", _FILE_FLAGS, dir_fd=self.directory)
                try:
                    self.value = _read(fd)
                finally:
                    os.close(fd)
                _state(self.value)
            self.validate()
        except BaseException as error:
            self.close()
            if isinstance(error, (OSError, ValueError, TypeError)):
                raise LeaseError("unavailable", "Shared lease namespace cannot be opened safely") from error
            raise

    def __enter__(self) -> _Registry:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        for name in ("resource", "directory", "anchor", "parent"):
            fd = getattr(self, name)
            setattr(self, name, -1)
            if fd >= 0:
                os.close(fd)

    def validate(self) -> None:
        parent = _open_parent(self.parent_path)
        try:
            if _identity(os.fstat(parent)) != self.parent_identity:
                _refuse("Shared lease parent was replaced")
        finally:
            os.close(parent)
        identities = ((self.parent, self.anchor_name, self.anchor_identity),
                      (self.parent, self.leaf, self.identity["directory"]),
                      (self.directory, "accelerator.lock", self.identity["resource"]))
        for base, name, expected in identities:
            if _identity(os.stat(name, dir_fd=base, follow_symlinks=False)) != expected:
                _refuse("Shared lease namespace or lock was replaced")
        if (_identity(_directory(self.directory, private=True)) != self.identity["directory"]
                or _identity(_file_info(self.resource)) != self.identity["resource"]):
            _refuse("Shared lease resource identity differs from its anchor")
        _file_info(self.anchor)

    def save(self) -> None:
        _state(self.value)
        # A previous coordinator can die during atomic replacement. The private
        # permanent anchor excludes every live writer before this stale temp is removed.
        try:
            stale = os.open("state.next", _FILE_FLAGS, dir_fd=self.directory)
        except FileNotFoundError:
            pass
        else:
            try:
                _file_info(stale)
                os.unlink("state.next", dir_fd=self.directory)
            finally:
                os.close(stale)
        fd = os.open("state.next", _FILE_FLAGS | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=self.directory)
        try:
            os.fchmod(fd, 0o600)
            _write(fd, self.value)
            os.replace("state.next", "state.json", src_dir_fd=self.directory, dst_dir_fd=self.directory)
            os.fsync(self.directory)
        finally:
            os.close(fd)

    def unlink_ticket(self, item: dict) -> None:
        name = item["ticket"] + ".ticket"
        if _identity(os.stat(name, dir_fd=self.directory, follow_symlinks=False)) != item["inode"]:
            _refuse("Shared lease ticket was replaced")
        os.unlink(name, dir_fd=self.directory)

    def prune(self) -> None:
        retained = []
        for item in self.value["queue"]:
            fd = os.open(item["ticket"] + ".ticket", _FILE_FLAGS, dir_fd=self.directory)
            try:
                if _identity(_file_info(fd)) != item["inode"]:
                    _refuse("Shared lease ticket identity changed")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    abandoned = True
                except BlockingIOError:
                    abandoned = False
                if abandoned or time.monotonic() >= item["deadline"]:
                    self.unlink_ticket(item)
                else:
                    retained.append(item)
            finally:
                os.close(fd)
        if retained != self.value["queue"]:
            self.value["queue"] = retained
            self.save()
        # A crash between creating a ticket and committing its queue entry must
        # not grow an unbounded collection of unreferenced request files.
        names = os.listdir(self.directory)
        if len(names) > MAX_QUEUE + 5:
            _refuse("Shared lease namespace contains too many entries")
        known = {item["ticket"] + ".ticket" for item in retained}
        for name in names:
            if not name.endswith(".ticket") or not _token(name[:-7]) or name in known:
                continue
            fd = os.open(name, _FILE_FLAGS, dir_fd=self.directory)
            try:
                _file_info(fd)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                os.unlink(name, dir_fd=self.directory)
            finally:
                os.close(fd)

    def order(self) -> list[dict]:
        groups = {kind: sorted((item for item in self.value["queue"] if item["workload"] == kind),
                               key=lambda item: item["sequence"]) for kind in WORKLOADS}
        last = self.value["last"]
        index = (WORKLOADS.index(last) + 1) % len(WORKLOADS) if last else 0
        result = []
        while any(groups.values()):
            kind = WORKLOADS[index]
            if groups[kind]:
                result.append(groups[kind].pop(0))
            index = (index + 1) % len(WORKLOADS)
        return result


class Lease:
    """A grant bound to its acquiring process, permanent resource and opaque ticket."""

    def __init__(self, fd: int, registry: _Registry, item: dict, deadline: float,
                 cancelled: Callable[[], bool] | None,
                 disconnected: Callable[[], bool] | None) -> None:
        self._fd = fd
        self._pid = os.getpid()
        self._path = registry.path
        self._identity = registry.identity.copy()
        self._anchor_identity = registry.anchor_identity.copy()
        self.ticket = item["ticket"]
        self.version = VERSION
        self.device = item["device"]
        self.workload = item["workload"]
        self._deadline = deadline
        self._cancelled = cancelled
        self._disconnected = disconnected

    def _owned(self, registry: _Registry) -> None:
        active = registry.value["active"]
        if (self._fd < 0 or os.getpid() != self._pid or registry.identity != self._identity
                or registry.anchor_identity != self._anchor_identity or active is None
                or active["ticket"] != self.ticket or active["state"] != "held"
                or _identity(_file_info(self._fd)) != self._identity["resource"]):
            raise LeaseError("lost-lease", "Shared lease identity is no longer current")

    def check(self) -> None:
        if self._fd < 0 or os.getpid() != self._pid:
            raise LeaseError("lost-lease", "Shared lease is closed or belongs to another process")
        _check_request(self._deadline, self._cancelled, self._disconnected)
        try:
            with _Registry(self._path, self._deadline, self._cancelled, self._disconnected) as registry:
                self._owned(registry)
        except OSError as error:
            raise LeaseError("lost-lease", "Shared lease descriptor or namespace was lost") from error

    @property
    def guard_fd(self) -> int:
        self.check()
        return self._fd

    def release(self, *, cleanup_complete: bool = False) -> None:
        if type(cleanup_complete) is not bool:
            raise LeaseError("invalid-request", "Cleanup acknowledgement must be boolean")
        if self._fd < 0:
            return
        if os.getpid() != self._pid:
            raise LeaseError("lost-lease", "Only the acquiring process may acknowledge cleanup")
        try:
            if cleanup_complete:
                with _Registry(self._path, time.monotonic() + 1.0) as registry:
                    self._owned(registry)
                    registry.value["active"]["state"] = "releasing"
                    registry.save()
        finally:
            fd, self._fd = self._fd, -1
            # LOCK_UN would unlock every inherited copy of this open description.
            # Closing only this process's copy preserves the supervisor/engine guard.
            try:
                identity = _identity(os.fstat(fd))
            except OSError as error:
                raise LeaseError("lost-lease", "Shared lease descriptor was closed externally") from error
            if identity != self._identity["resource"]:
                raise LeaseError("lost-lease", "Refusing to close a reused unrelated descriptor")
            os.close(fd)

    def __enter__(self) -> Lease:
        self.check()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def acquire(*, job_id: str, workload: str, device: str, deadline: float,
            cancelled: Callable[[], bool] | None = None,
            disconnected: Callable[[], bool] | None = None,
            progress: Callable[[QueueStatus], None] | None = None,
            namespace: str | None = None) -> Lease:
    """Wait fairly for exclusive execution; refuse expired/unavailable ownership.

    The optional namespace is for explicitly coordinated private deployments
    and isolated tests. Every cooperating provider must use the same namespace.
    An existing incomplete or replaced namespace is never silently recreated.
    """
    if (not _label(job_id) or not _label(device) or type(workload) is not str or workload not in WORKLOADS
            or not _number(deadline) or deadline > time.monotonic() + MAX_WAIT_SECONDS
            or any(callback is not None and not callable(callback) for callback in (cancelled, disconnected, progress))
            or (namespace is not None and (type(namespace) is not str or not namespace))):
        raise LeaseError("invalid-request", "Invalid shared lease request")
    path = namespace or f"/run/user/{os.geteuid()}/kilix-device-leases-v1"
    if not path.startswith("/") or path != os.path.normpath(path):
        raise LeaseError("invalid-request", "Lease namespace must be a canonical absolute path")
    _check_request(deadline, cancelled, disconnected)
    item = None
    ticket_fd = -1
    previous = None
    try:
        while True:
            _check_request(deadline, cancelled, disconnected)
            with _Registry(path, deadline, cancelled, disconnected) as registry:
                registry.prune()
                if item is None:
                    if (len(registry.value["queue"]) >= MAX_QUEUE
                            or sum(row["workload"] == workload for row in registry.value["queue"]) >= MAX_WORKLOAD_QUEUE):
                        raise LeaseError("queue-full", "Shared accelerator queue is full")
                    ticket = uuid.uuid4().hex
                    ticket_fd = os.open(ticket + ".ticket", _FILE_FLAGS | os.O_CREAT | os.O_EXCL,
                                        0o600, dir_fd=registry.directory)
                    os.fchmod(ticket_fd, 0o600)
                    fcntl.flock(ticket_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    item = {"ticket": ticket, "job_id": job_id, "workload": workload, "device": device,
                            "sequence": registry.value["next"], "deadline": deadline,
                            "inode": _identity(_file_info(ticket_fd))}
                    registry.value["next"] += 1
                    registry.value["queue"].append(item)
                    registry.save()
                ordered = registry.order()
                if not any(row["ticket"] == item["ticket"] for row in ordered):
                    _check_request(deadline, cancelled, disconnected)
                    raise LeaseError("lost-lease", "Queued shared lease ticket disappeared")
                try:
                    fcntl.flock(registry.resource, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    available = True
                except BlockingIOError:
                    available = False
                if available:
                    active = registry.value["active"]
                    if active is not None and active["state"] == "held":
                        _refuse("Previous accelerator cleanup is unproven; ownership is quarantined")
                    if ordered[0]["ticket"] == item["ticket"]:
                        _check_request(deadline, cancelled, disconnected)
                        registry.validate()
                        registry.value["active"] = {key: item[key] for key in ("ticket", "job_id", "workload", "device")}
                        registry.value["active"]["state"] = "held"
                        registry.value["last"] = workload
                        registry.value["queue"] = [row for row in registry.value["queue"] if row["ticket"] != item["ticket"]]
                        registry.save()
                        registry.unlink_ticket(item)
                        fd = os.dup(registry.resource)
                        lease = Lease(fd, registry, item, deadline, cancelled, disconnected)
                        os.close(ticket_fd)
                        ticket_fd = -1
                        item = None
                        return lease
                update = QueueStatus(VERSION, item["ticket"], "queued",
                                     1 + next(i for i, row in enumerate(ordered) if row["ticket"] == item["ticket"]))
            if progress is not None and update != previous:
                progress(update)
            previous = update
            time.sleep(_POLL_SECONDS)
    except OSError as error:
        raise LeaseError("unavailable", "Shared lease filesystem or descriptor operation failed") from error
    finally:
        if item is not None:
            try:
                # This request never acquired an engine grant. Refusal must
                # not add a fresh wait beyond its cancellation/deadline. If the
                # anchor is busy, closing the ticket below lets a later holder
                # prune the abandoned queue entry using its kernel lock.
                with _Registry(path, time.monotonic() + 1.0, blocking=False) as registry:
                    found = next((row for row in registry.value["queue"] if row["ticket"] == item["ticket"]), None)
                    if found is not None:
                        if found != item:
                            _refuse("Refusing to remove a replaced shared lease ticket")
                        registry.unlink_ticket(item)
                        registry.value["queue"] = [row for row in registry.value["queue"] if row["ticket"] != item["ticket"]]
                        registry.save()
            except (OSError, LeaseError):
                # Preserve unproven namespace state; the next request refuses it.
                pass
        if ticket_fd >= 0:
            os.close(ticket_fd)
