"""One shared catalog of local models: install, list, verify and remove them.

The catalog (catalog.json beside this module) pins each downloadable model to
an exact file, size, sha256 and Hugging Face revision, and names the folder role
it belongs in. llama.cpp GGUFs go to the folder kilix-llm and puzzllm already
search; text-diffusion GGUFs go to the folder puzzllm's Bench and kilix-chat
search. Ollama models are pulled, listed and deleted through a loopback Ollama.

Every change is aware of residency. A model is never removed while a registry
record holds it, while Ollama has it loaded, or while any process has the file
open or mapped. ``unload=True`` unloads it through the registry first, which
still refuses pinned, busy and loading residents, and it never touches a
process the registry does not manage.

Downloads keep the semantics of puzzllm-fetch and share its per-folder lock:
the file is verified before it gets its real name, a file of the same name that
does not match is refused, and so is a symlink. In addition, an interrupted
download is resumed from its ``.<file>.part``, within one call and across calls.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import http.client
import ipaddress
import json
import os
import re
import ssl
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from typing import Callable, Iterator, Mapping
from urllib.parse import unquote, urljoin, urlsplit

from . import ResidencyError, ResidentInfo, residents as _residents, unload as _unload
from . import ollama as _ollama

SCHEMA = "kilix.gpu-residency.catalog/v1"
RESULT_SCHEMA = "kilix.gpu-residency.models/v1"
KINDS = ("gguf", "diffusion-gguf", "ollama")
FOLDER_ROLES = ("llama.cpp", "diffusion")
FOLDER_FOR_KIND = {"gguf": "llama.cpp", "diffusion-gguf": "diffusion"}
# Every outcome, and the exit code kilix-gpu models gives it:
# 0 done, 1 failed, 2 refused, 3 in use (``--unload`` may help, unless pinned or busy), 4 does not match.
OUTCOMES = {
    "saved": 0, "already-saved": 0, "pulled": 0, "already-installed": 0, "removed": 0, "verified": 0,
    "download-failed": 1, "downloaded-mismatch": 1, "pull-failed": 1, "remove-failed": 1, "unavailable": 1,
    "unknown": 2, "invalid-name": 2, "conflict": 2, "symlink": 2, "busy-folder": 2, "not-installed": 2,
    "outside-folders": 2, "ambiguous": 2, "no-space": 2, "unsafe-folder": 2, "not-verifiable": 2,
    "loaded": 3, "pinned": 3, "busy": 3, "open": 3,
    "mismatch": 4,
}
EXIT_MEANINGS = {0: "done", 1: "failed", 2: "refused", 3: "in use", 4: "does not match"}
ACTIONS = ("list", "install", "remove", "verify")
LOCK_NAME = ".fetch.lock"  # the same lock file puzzllm-fetch takes, so the two exclude each other
PART_TEMPLATE = ".{file}.part"
ATTEMPTS = 5  # consecutive failed connections before a download gives up (the .part is kept)
BACKOFF_SECONDS = 2.0
TIMEOUT_SECONDS = 30.0
LOCK_WAIT_SECONDS = 1.0
RELEASE_WAIT_SECONDS = 10.0
MAX_REDIRECTS = 5
USER_AGENT = "kilix-gpu-residency/0.1"
_CHUNK = 1 << 20
_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z", re.ASCII)
_GGUF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.gguf\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_HF = re.compile(r"https://huggingface\.co/[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
                 r"/resolve/[0-9a-f]{40}/(?P<file>[^/?#]+)\Z", re.ASCII)
_OLLAMA_NAME = re.compile(r"[A-Za-z0-9_.:/-]+\Z", re.ASCII)
_TEXT = re.compile(r"[^\x00-\x1f\x7f-\x9f]{1,200}\Z")
_ENV = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z", re.ASCII)
_FILE_FIELDS = {"id", "name", "kind", "folder", "file", "bytes", "sha256", "url", "architecture",
                "context_length", "source", "licence"}
_OLLAMA_FIELDS = {"id", "name", "kind", "ollama", "architecture", "context_length", "source", "licence"}
_FOLDER_FIELDS = {"env", "base_env", "base_default", "subpath"}

Progress = Callable[[str, int | None, int | None], None]


class CatalogError(RuntimeError):
    """A refusal or failure. ``outcome`` is one of OUTCOMES; ``str()`` is a one-line notice."""

    def __init__(self, outcome: str, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.detail = detail

    @property
    def exit_code(self) -> int:
        return OUTCOMES[self.outcome]


@dataclass(frozen=True)
class Entry:
    id: str
    name: str
    kind: str
    architecture: str
    context_length: int
    source: str
    licence: str
    folder: str | None = None  # a folder role, for the GGUF kinds
    file: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    url: str | None = None
    ollama: str | None = None  # the Ollama name, for the ollama kind


@dataclass(frozen=True)
class Folder:
    env: str
    base_env: str
    base_default: str
    subpath: str


@dataclass(frozen=True)
class Catalog:
    entries: tuple[Entry, ...]
    folders: Mapping[str, Folder]

    def get(self, ident: str) -> Entry | None:
        return next((entry for entry in self.entries if entry.id == ident), None)


@dataclass(frozen=True)
class Result:
    action: str
    outcome: str
    message: str
    name: str
    path: str | None = None

    @property
    def exit_code(self) -> int:
        return OUTCOMES[self.outcome]


@dataclass(frozen=True)
class Listed:
    id: str  # a catalog id, a local file name, or ollama:NAME
    name: str
    kind: str
    source: str  # catalog, local or ollama
    state: str  # installed, available, partial, mismatch, link or local
    folder: str | None
    path: str | None
    bytes: int | None  # the catalog's size, else the size on disk
    partial_bytes: int | None
    resident: tuple[str, ...]  # who holds it now: "kilix-chat", "ollama", "pid 4242 (llama-server)"


@dataclass(frozen=True)
class Inventory:
    models: tuple[Listed, ...]
    folders: Mapping[str, str]
    ollama: str  # reachable, unreachable or not-loopback
    residency: str  # known, or unavailable when the registry could not be read


# Catalog data {{{

def _pairs(pairs: list[tuple[str, object]]) -> dict:
    value: dict = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key {key!r}")
        value[key] = item
    return value


def _loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _valid_url(url: object, file: str, local: bool) -> bool:
    if type(url) is not str:
        return False
    match = _HF.fullmatch(url)
    if match is not None:
        return unquote(match.group("file")) == file
    if not local:
        return False
    parts = urlsplit(url)
    if parts.scheme == "file":
        return parts.netloc in ("", "localhost") and parts.path.startswith("/")
    return parts.scheme == "http" and _loopback_host(parts.hostname)


def _bad(message: str) -> None:
    raise CatalogError("unavailable", f"The model catalog is malformed: {message}")


def _parse(raw: bytes, *, local: bool) -> Catalog:
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except ValueError as error:
        _bad(str(error))
    if type(value) is not dict or set(value) != {"schema", "folders", "models"} or value["schema"] != SCHEMA:
        _bad("it is not a kilix.gpu-residency.catalog/v1 document")
    folders = value["folders"]
    if type(folders) is not dict or set(folders) != set(FOLDER_ROLES):
        _bad(f"folders must be exactly {', '.join(FOLDER_ROLES)}")
    rules = {}
    for role, rule in folders.items():
        if type(rule) is not dict or set(rule) != _FOLDER_FIELDS or not all(type(v) is str for v in rule.values()):
            _bad(f"folder {role} is malformed")
        subpath = rule["subpath"]
        if (not _ENV.fullmatch(rule["env"]) or not _ENV.fullmatch(rule["base_env"])
                or not rule["base_default"].startswith("~/") or ".." in rule["base_default"].split("/")
                or not subpath or subpath.startswith("/") or ".." in subpath.split("/")):
            _bad(f"folder {role} is malformed")
        rules[role] = Folder(**rule)
    models = value["models"]
    if type(models) is not list:
        _bad("models must be a list")
    entries, ids, files = [], set(), set()
    for item in models:
        if type(item) is not dict:
            _bad("a model is not an object")
        kind = item.get("kind")
        if kind not in KINDS or set(item) != (_OLLAMA_FIELDS if kind == "ollama" else _FILE_FIELDS):
            _bad(f"model {item.get('id')!r} has the wrong fields")
        ident = item["id"]
        if type(ident) is not str or not _ID.fullmatch(ident) or ident.endswith(".gguf") or ident in ids:
            _bad(f"model id {ident!r} is invalid or repeated")
        ids.add(ident)
        for key in ("name", "architecture", "source", "licence"):
            if type(item[key]) is not str or not _TEXT.fullmatch(item[key]):
                _bad(f"model {ident} has an invalid {key}")
        if type(item["context_length"]) is not int or item["context_length"] <= 0:
            _bad(f"model {ident} has an invalid context_length")
        if kind == "ollama":
            if type(item["ollama"]) is not str or not ollama_name_ok(item["ollama"]):
                _bad(f"model {ident} has an invalid Ollama name")
        else:
            file = item["file"]
            if item["folder"] != FOLDER_FOR_KIND[kind]:
                _bad(f"model {ident} is a {kind} but names folder {item['folder']!r}")
            if type(file) is not str or not plain_gguf(file) or (item["folder"], file) in files:
                _bad(f"model {ident} has an invalid or repeated file name")
            files.add((item["folder"], file))
            if type(item["bytes"]) is not int or item["bytes"] <= 0:
                _bad(f"model {ident} has an invalid size")
            if type(item["sha256"]) is not str or not _SHA.fullmatch(item["sha256"]):
                _bad(f"model {ident} has an invalid sha256")
            if not _valid_url(item["url"], file, local):
                _bad(f"model {ident} is not pinned to a Hugging Face revision and file")
        entries.append(Entry(**item))
    return Catalog(tuple(entries), rules)


def load(path: str | None = None) -> Catalog:
    """The packaged catalog, or one at ``path``. Only a catalog given by path may use
    ``file://`` or loopback ``http://`` URLs (for tests and local mirrors)."""
    try:
        if path is None:
            raw = resources.files(__package__).joinpath("catalog.json").read_bytes()
        else:
            with open(path, "rb") as stream:
                raw = stream.read(1 << 20)
    except OSError as error:
        raise CatalogError("unavailable", f"The model catalog could not be read: {error.strerror}") from error
    return _parse(raw, local=path is not None)


def plain_gguf(name: str) -> bool:
    """puzzllm-fetch's rule: a plain ``*.gguf`` name with no path and no ``..``."""
    return bool(_GGUF.fullmatch(name)) and ".." not in name and "/" not in name


def ollama_name_ok(name: str) -> bool:
    """puzzllm-fetch's rule for Ollama names."""
    return (bool(name) and len(name) <= 128 and ".." not in name and "//" not in name
            and name[0] not in "/:.-" and name[-1] not in "/:" and bool(_OLLAMA_NAME.fullmatch(name)))


def _canonical(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"

# }}}


# Folders {{{

def folder(role: str, catalog: Catalog | None = None) -> str:
    """The folder for ``role``: its variable, else its base variable (or default) plus subpath."""
    rule = (catalog or load()).folders.get(role)
    if rule is None:
        raise CatalogError("unknown", f"unknown folder role {role}")
    value = os.environ.get(rule.env) or os.path.join(
        os.environ.get(rule.base_env) or os.path.expanduser(rule.base_default), rule.subpath)
    if not os.path.isabs(value):
        raise CatalogError("unsafe-folder", f"the models folder must be an absolute path ({rule.env})")
    return os.path.normpath(value)


def _make_private(path: str) -> None:
    missing = []
    while not os.path.lexists(path):
        missing.append(path)
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            continue
        os.chmod(directory, 0o700)  # whatever the umask


def _open_folder(path: str, *, create: bool) -> int | None:
    if create:
        try:
            _make_private(path)
        except OSError as error:
            raise CatalogError("unsafe-folder", f"the models folder {path} could not be made") from error
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise CatalogError("unsafe-folder", f"the models folder {path} is not a folder") from error
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        os.close(fd)
        raise CatalogError("unsafe-folder", f"the models folder {path} must be yours and not writable by others")
    return fd


@contextmanager
def _folder_lock(fd: int) -> Iterator[None]:
    try:
        lock = os.open(LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=fd)
    except OSError as error:
        raise CatalogError("unsafe-folder", "the models folder's lock could not be opened") from error
    try:
        if not stat.S_ISREG(os.fstat(lock).st_mode):
            raise CatalogError("unsafe-folder", "the models folder's lock is not a file")
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise CatalogError("busy-folder", "another download is using the models folder") from None
                time.sleep(0.05)
        yield
    finally:
        os.close(lock)


def _lstat(name: str, fd: int) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _part(file: str) -> str:
    return PART_TEMPLATE.format(file=file)


def _sha256(name: str, fd: int, progress: Progress | None = None) -> tuple[str, int]:
    handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
    try:
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode):
            raise CatalogError("conflict", f"{name} is not a regular file")
        digest, done = hashlib.sha256(), 0
        while chunk := os.read(handle, _CHUNK):
            digest.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress("verifying", done, info.st_size)
        return digest.hexdigest(), done
    finally:
        os.close(handle)


def free_bytes(fd: int) -> int:
    """Space an unprivileged user may still write in the folder open at ``fd``."""
    info = os.fstatvfs(fd)
    return info.f_bavail * info.f_frsize

# }}}


# Who holds a model {{{

def _command(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm", "rb") as stream:
            return stream.read(64).decode("utf-8", "replace").strip()
    except OSError:
        return "?"


def open_by(targets: Mapping[tuple[int, int], str]) -> dict[tuple[int, int], list[tuple[int, str]]]:
    """For each (st_dev, st_ino) in ``targets`` (mapped to its file name), the processes this
    user can inspect that hold it open or mapped. Other users' processes are not visible."""
    found: dict[tuple[int, int], list[tuple[int, str]]] = {}
    if not targets:
        return found
    inodes = {str(ino) for _, ino in targets}
    me = os.getpid()
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == me:
            continue
        hits = set()
        try:
            descriptors = os.listdir(f"/proc/{name}/fd")
        except OSError:
            descriptors = []
        for descriptor in descriptors:
            try:
                info = os.stat(f"/proc/{name}/fd/{descriptor}")
            except OSError:
                continue
            if _identity(info) in targets:
                hits.add(_identity(info))
        try:
            with open(f"/proc/{name}/maps", "rb") as stream:
                maps = stream.read().decode("utf-8", "replace")
        except OSError:
            maps = ""
        for line in maps.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 5 or fields[4] not in inodes:
                continue
            try:
                major, minor = (int(part, 16) for part in fields[3].split(":"))
            except ValueError:
                continue
            for key, file in targets.items():
                # btrfs maps show the filesystem's device, not the subvolume's: then the name decides.
                same_device = key[0] == os.makedev(major, minor)
                same_name = len(fields) == 6 and os.path.basename(fields[5].strip()) == file
                if str(key[1]) == fields[4] and (same_device or same_name):
                    hits.add(key)
        for key in hits:
            found.setdefault(key, []).append((int(name), _command(int(name))))
    return found


def _holders(namespace: str | None, url: str, what: str) -> tuple[ResidentInfo, ...]:
    try:
        return _residents(namespace=namespace, ollama=url)
    except ResidencyError as error:
        raise CatalogError("unavailable", f"cannot check whether {what} is loaded: {error}") from error


def _file_names(entry: Entry | None, directory: str, file: str) -> set[str]:
    """What an app may have registered a GGUF as: its catalog id, file name, stem or path."""
    path = os.path.join(directory, file)
    names = {file, file[:-len(".gguf")], path, os.path.realpath(path)}
    if entry is not None:
        names.add(entry.id)
    return names


def _clear(holders: list[ResidentInfo], label: str, unload: bool, namespace: str | None, url: str) -> bool:
    """Refuse while ``holders`` hold the model, or unload them with ``unload``. True when one was unloaded."""
    if not holders:
        return False
    for info in holders:
        if info.pinned:
            raise CatalogError("pinned", f"{label} is pinned by {info.app}; unpin it first")
    apps = ", ".join(sorted({info.app for info in holders}))
    if not unload:
        raise CatalogError("loaded", f"{label} is loaded by {apps}; use --unload")
    for info in holders:
        if info.state != "idle":
            raise CatalogError("busy", f"{label} is {info.state} in {info.app}; try again when it is idle")
    for info in holders:
        try:
            _unload(info.id, by="kilix-gpu-models", namespace=namespace, ollama=url)
        except ResidencyError as error:
            if error.code != "not-found":
                raise CatalogError("busy", f"could not unload {label} from {info.app}: {error}") from error
    return True


def _describe(holders: list[tuple[int, str]]) -> str:
    return ", ".join(f"pid {pid} ({command})" for pid, command in sorted(holders))

# }}}


# Downloads {{{

class _Retry(Exception):
    """The connection failed in a way worth retrying; the bytes written so far are kept."""


class _Fail(Exception):
    """The download cannot succeed by retrying."""


class _Mismatch(Exception):
    """The bytes received are not the catalog's file."""


class _State:
    def __init__(self, handle: int, have: int, total: int, progress: Progress | None) -> None:
        self.handle, self.have, self.total, self.progress = handle, have, total, progress
        self.digest = hashlib.sha256()

    def restart(self) -> None:
        os.ftruncate(self.handle, 0)
        os.lseek(self.handle, 0, os.SEEK_SET)
        self.have, self.digest = 0, hashlib.sha256()

    def copy(self, read: Callable[[int], bytes]) -> None:
        while True:
            chunk = read(min(_CHUNK, self.total - self.have + 1))  # one byte more shows an overrun
            if not chunk:
                return
            if self.have + len(chunk) > self.total:
                raise _Mismatch("the server sent more than the catalog's size")
            view = memoryview(chunk)
            try:
                while view:
                    view = view[os.write(self.handle, view):]
            except OSError as error:
                raise _Fail(f"writing failed: {error.strerror}") from error
            self.digest.update(chunk)
            self.have += len(chunk)
            if self.progress is not None:
                self.progress("downloading", self.have, self.total)


def _content_range_start(value: str | None, total: int) -> int:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", (value or "").strip())
    if match is None:
        raise _Fail("the server's Content-Range is malformed")
    if match.group(3) != "*" and int(match.group(3)) != total:
        raise _Mismatch("the server's file is not the catalog's size")
    return int(match.group(1))


def _connection(parts) -> http.client.HTTPConnection:
    if parts.scheme == "https":
        return http.client.HTTPSConnection(parts.hostname, parts.port or 443, timeout=TIMEOUT_SECONDS,
                                           context=ssl.create_default_context())
    if parts.scheme == "http" and _loopback_host(parts.hostname):
        return http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=TIMEOUT_SECONDS)
    raise _Fail("only https, or http to a loopback address, is fetched")


def _redirect_allowed(old, new) -> bool:
    if new.scheme == "https":
        return True
    return old.scheme == "http" and new.scheme == "http" and _loopback_host(new.hostname)


def _fetch_file(parts, state: _State) -> None:
    try:
        source = os.open(unquote(parts.path), os.O_RDONLY | os.O_CLOEXEC)
    except OSError as error:
        raise _Fail(f"the source could not be opened: {error.strerror}") from error
    try:
        os.lseek(source, state.have, os.SEEK_SET)
        state.copy(lambda size: os.read(source, size))
    except OSError as error:
        raise _Retry(f"reading the source failed: {error.strerror}") from error
    finally:
        os.close(source)


def _fetch(url: str, state: _State) -> None:
    """One attempt: fetch what is still missing into ``state``, following redirects."""
    for _ in range(MAX_REDIRECTS + 1):
        parts = urlsplit(url)
        if parts.scheme == "file":
            return _fetch_file(parts, state)
        connection = _connection(parts)
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if state.have:
            headers["Range"] = f"bytes={state.have}-"
        response = None
        try:
            connection.request("GET", parts.path + (f"?{parts.query}" if parts.query else ""), headers=headers)
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                target = urljoin(url, location or "")
                if not location or not _redirect_allowed(parts, urlsplit(target)):
                    raise _Fail("the server redirected somewhere that is not fetched")
                url = target
                continue
            if response.status == 416:
                if state.have == state.total:
                    return
                state.restart()
                raise _Retry("the server could not resume; starting again")
            if response.status == 200:
                length = response.getheader("Content-Length")
                if length is not None and length.strip().isdigit() and int(length) != state.total:
                    raise _Mismatch("the server's file is not the catalog's size")
                if state.have:
                    state.restart()  # the server ignored Range
            elif response.status == 206:
                if _content_range_start(response.getheader("Content-Range"), state.total) != state.have:
                    raise _Fail("the server resumed at the wrong offset")
            elif response.status == 429 or response.status >= 500:
                raise _Retry(f"HTTP {response.status}")
            else:
                raise _Fail(f"HTTP {response.status}")
            state.copy(response.read)
            return
        except (OSError, http.client.HTTPException) as error:
            raise _Retry(type(error).__name__) from error
        finally:
            if response is not None:
                response.close()
            connection.close()
    raise _Fail("too many redirects")


def _download(entry: Entry, fd: int, progress: Progress | None) -> None:
    """Fill ``.<file>.part`` to the catalog's size and sha256, resuming what is there."""
    part = _part(entry.file)
    handle = os.open(part, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=fd)
    try:
        info = os.fstat(handle)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CatalogError("conflict", f"{part} is not a plain file")
        state = _State(handle, 0, entry.bytes, progress)
        if info.st_size > entry.bytes:
            state.restart()
        else:
            # Resume: hash what is already there, then append.
            while state.have < info.st_size:
                chunk = os.pread(handle, min(_CHUNK, info.st_size - state.have), state.have)
                if not chunk:
                    break
                state.digest.update(chunk)
                state.have += len(chunk)
                if progress is not None:
                    progress("checking", state.have, entry.bytes)
            os.lseek(handle, state.have, os.SEEK_SET)
        room = free_bytes(fd)
        if room < entry.bytes - state.have:
            raise CatalogError("no-space", f"not enough disk space for {entry.file}: "
                                           f"{(entry.bytes - state.have) / 1e9:.1f} GB needed, {room / 1e9:.1f} GB free")
        failures = 0
        while state.have < entry.bytes:
            before = state.have
            try:
                _fetch(entry.url, state)
                if state.have < entry.bytes:
                    raise _Retry("the connection closed early")
            except _Retry as error:
                failures = 1 if state.have > before else failures + 1
                if failures >= ATTEMPTS:
                    raise CatalogError("download-failed", f"could not download {entry.file}",
                                       detail=str(error)) from error
                time.sleep(min(BACKOFF_SECONDS * failures, 10.0))
            except _Fail as error:
                raise CatalogError("download-failed", f"could not download {entry.file}", detail=str(error)) from error
        if state.digest.hexdigest() != entry.sha256:
            raise _Mismatch("sha256 differs")
        os.fsync(handle)
    except _Mismatch as error:
        os.close(handle)
        handle = -1
        os.unlink(part, dir_fd=fd)
        raise CatalogError("downloaded-mismatch", f"downloaded {entry.file} did not match", detail=str(error)) from error
    except CatalogError:
        if os.fstat(handle).st_size == 0:
            os.unlink(part, dir_fd=fd)  # nothing to resume
        raise
    finally:
        if handle >= 0:
            os.close(handle)


def _publish(fd: int, part: str, file: str) -> None:
    """Give the verified part its real name, never replacing a file that appeared meanwhile."""
    try:
        os.link(part, file, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
    except FileExistsError:
        raise CatalogError("conflict", f"{file} is already in the models folder and does not match") from None
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK, errno.ENOSYS):
            raise
        if _lstat(file, fd) is not None:  # a filesystem without hard links
            raise CatalogError("conflict", f"{file} is already in the models folder and does not match") from None
        os.rename(part, file, src_dir_fd=fd, dst_dir_fd=fd)
    else:
        os.unlink(part, dir_fd=fd)
    os.fsync(fd)


def _install_file(entry: Entry, catalog: Catalog, progress: Progress | None) -> Result:
    directory = folder(entry.folder, catalog)
    path = os.path.join(directory, entry.file)
    fd = _open_folder(directory, create=True)
    try:
        with _folder_lock(fd):
            present = _lstat(entry.file, fd)
            if present is not None:
                if stat.S_ISLNK(present.st_mode):
                    raise CatalogError("symlink", f"refusing to replace a symlink: {entry.file}")
                if (not stat.S_ISREG(present.st_mode) or present.st_size != entry.bytes
                        or _sha256(entry.file, fd, progress)[0] != entry.sha256):
                    raise CatalogError("conflict", f"{entry.file} is already in the models folder and does not match")
                return Result("install", "already-saved", f"already saved {entry.file}", entry.file, path)
            part = _lstat(_part(entry.file), fd)
            if part is not None:
                if stat.S_ISLNK(part.st_mode):
                    raise CatalogError("symlink", f"refusing to use a symlink: {_part(entry.file)}")
                writers = open_by({_identity(part): _part(entry.file)}).get(_identity(part))
                if writers:
                    raise CatalogError("open", f"{entry.file} is being downloaded by {_describe(writers)}")
            _download(entry, fd, progress)
            _publish(fd, _part(entry.file), entry.file)
    finally:
        os.close(fd)
    return Result("install", "saved", f"saved {entry.file}", entry.file, path)


def _ollama_url(url: str | None) -> str:
    return url or _ollama.default_url()


def _install_ollama(name: str, url: str, progress: Progress | None) -> Result:
    if not ollama_name_ok(name):
        raise CatalogError("invalid-name", "Ollama names use letters, digits, and . _ : / -")
    try:
        if _ollama.has(url, name):
            return Result("install", "already-installed", f"already installed {name}", name)
    except _ollama.OllamaError as error:
        raise CatalogError("unavailable", "Ollama is not reachable", detail=str(error)) from error
    try:
        _ollama.pull(url, name, progress)
    except _ollama.OllamaError as error:
        raise CatalogError("pull-failed", f"could not pull {name}", detail=str(error)) from error
    return Result("install", "pulled", f"pulled {name}", name)

# }}}


# Public API {{{

def install(target: str, *, catalog: Catalog | None = None, ollama: str | None = None,
            progress: Progress | None = None) -> Result:
    """Install a catalog model by id, or an Ollama model as ``ollama:NAME``.

    ``progress(stage, done, total)`` is called while downloading or pulling.
    Raises CatalogError with a one-line notice when refused or failed."""
    catalog = catalog or load()
    if target.startswith("ollama:"):
        return _install_ollama(target[len("ollama:"):], _ollama_url(ollama), progress)
    entry = catalog.get(target)
    if entry is None:
        raise CatalogError("unknown", f"unknown model {target}")
    if entry.kind == "ollama":
        return _install_ollama(entry.ollama, _ollama_url(ollama), progress)
    return _install_file(entry, catalog, progress)


def _locate(target: str, catalog: Catalog) -> tuple[str, str, Entry | None] | str:
    """(folder, file, entry) for a file target, or the Ollama name for an Ollama one."""
    if target.startswith("ollama:"):
        return target[len("ollama:"):]
    entry = catalog.get(target)
    if entry is not None:
        return entry.ollama if entry.kind == "ollama" else (folder(entry.folder, catalog), entry.file, entry)
    folders = {role: folder(role, catalog) for role in FOLDER_ROLES}
    if "/" in target:
        directory, file = os.path.split(os.path.abspath(target))
        real = os.path.realpath(directory)
        roles = [role for role, path in folders.items() if os.path.realpath(path) == real]
        if not roles or not plain_gguf(file):
            raise CatalogError("outside-folders", f"{target} is not a model file in a models folder")
        role = roles[0]
    elif plain_gguf(target):
        file = target
        roles = [role for role, path in folders.items() if os.path.lexists(os.path.join(path, file))
                 or os.path.lexists(os.path.join(path, _part(file)))]
        if len(roles) > 1:
            raise CatalogError("ambiguous", f"{file} is in more than one models folder; give its path")
        if not roles:
            raise CatalogError("not-installed", f"{file} is not installed")
        role = roles[0]
    else:
        raise CatalogError("unknown", f"unknown model {target}")
    match = next((e for e in catalog.entries if e.folder == role and e.file == file), None)
    return folders[role], file, match


def _remove_file(directory: str, file: str, entry: Entry | None, unload: bool, namespace: str | None,
                 url: str) -> Result:
    fd = _open_folder(directory, create=False)
    if fd is None:
        raise CatalogError("not-installed", f"{file} is not installed")
    try:
        with _folder_lock(fd):
            present, part = _lstat(file, fd), _lstat(_part(file), fd)
            if present is not None and stat.S_ISLNK(present.st_mode):
                raise CatalogError("symlink", f"refusing to remove a symlink: {file}")
            if present is not None and not stat.S_ISREG(present.st_mode):
                raise CatalogError("conflict", f"{file} is not a regular file")
            if part is not None and not stat.S_ISREG(part.st_mode):
                part = None  # never follow or remove anything but a plain part file
            if present is None and part is None:
                raise CatalogError("not-installed", f"{file} is not installed")
            unloaded = False
            if present is not None:
                names = _file_names(entry, directory, file)
                holders = [info for info in _holders(namespace, url, file)
                           if info.registered and info.backend != "ollama" and info.model in names]
                unloaded = _clear(holders, file, unload, namespace, url)
            targets = {_identity(info): name for name, info in ((file, present), (_part(file), part)) if info}
            deadline = time.monotonic() + (RELEASE_WAIT_SECONDS if unloaded else 0)
            while True:
                holding = open_by(targets)
                if present is None or _identity(present) not in holding or time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            if present is not None and _identity(present) in holding:
                raise CatalogError("open", f"{file} is open by {_describe(holding[_identity(present)])}; close it first")
            removed = []
            for name, info in ((file, present), (_part(file), part)):
                if info is None or _identity(info) in holding:
                    continue  # a part someone is writing is never removed
                again = _lstat(name, fd)
                if again is None or _identity(again) != _identity(info):
                    continue
                os.unlink(name, dir_fd=fd)
                removed.append(name)
            os.fsync(fd)
    finally:
        os.close(fd)
    if present is None and not removed:
        raise CatalogError("open", f"the partial download of {file} is being written; leaving it")
    message = f"removed {file}" if present is not None else f"removed the partial download of {file}"
    return Result("remove", "removed", message, file, os.path.join(directory, file))


def _remove_ollama(name: str, unload: bool, namespace: str | None, url: str) -> Result:
    if not ollama_name_ok(name):
        raise CatalogError("invalid-name", "Ollama names use letters, digits, and . _ : / -")
    try:
        names = {_canonical(model.name): model.name for model in _ollama.installed(url)}
    except _ollama.OllamaError as error:
        raise CatalogError("unavailable", "Ollama is not reachable", detail=str(error)) from error
    if _canonical(name) not in names:
        raise CatalogError("not-installed", f"{name} is not installed")
    tag = names[_canonical(name)]  # delete exactly what Ollama listed
    holders = [info for info in _holders(namespace, url, name)
               if info.backend == "ollama" and _canonical(info.model) == _canonical(name)]
    _clear(holders, name, unload, namespace, url)
    try:
        _ollama.delete(url, tag)
    except _ollama.OllamaMissing as error:
        raise CatalogError("not-installed", f"{name} is not installed") from error
    except _ollama.OllamaError as error:
        raise CatalogError("remove-failed", f"could not remove {name}", detail=str(error)) from error
    return Result("remove", "removed", f"removed {name}", name)


def remove(target: str, *, unload: bool = False, catalog: Catalog | None = None, namespace: str | None = None,
           ollama: str | None = None) -> Result:
    """Remove a model: a catalog id, a GGUF's file name or path in a models folder, or
    ``ollama:NAME``. Refused while it is resident or open; ``unload`` unloads it through
    the registry first, which still refuses pinned, busy and loading residents."""
    catalog = catalog or load()
    url = _ollama_url(ollama)
    located = _locate(target, catalog)
    if isinstance(located, str):
        return _remove_ollama(located, unload, namespace, url)
    return _remove_file(*located, unload, namespace, url)


def verify(target: str, *, catalog: Catalog | None = None, progress: Progress | None = None) -> Result:
    """Recompute an installed catalog file's size and sha256 and compare them with the catalog."""
    catalog = catalog or load()
    located = _locate(target, catalog)
    if isinstance(located, str):
        raise CatalogError("not-verifiable", "Ollama checks its own models; there is nothing to verify")
    directory, file, entry = located
    if entry is None:
        raise CatalogError("not-verifiable", f"{file} is not in the catalog; there is nothing to compare it with")
    fd = _open_folder(directory, create=False)
    if fd is None:
        raise CatalogError("not-installed", f"{file} is not installed")
    try:
        present = _lstat(file, fd)
        if present is None:
            raise CatalogError("not-installed", f"{file} is not installed")
        if stat.S_ISLNK(present.st_mode):
            raise CatalogError("symlink", f"{file} is a symlink; it is not verified")
        if present.st_size != entry.bytes:
            raise CatalogError("mismatch", f"{file} does not match the catalog (size)")
        if _sha256(file, fd, progress)[0] != entry.sha256:
            raise CatalogError("mismatch", f"{file} does not match the catalog (sha256)")
    finally:
        os.close(fd)
    return Result("verify", "verified", f"verified {file}", file, os.path.join(directory, file))


def list_models(*, catalog: Catalog | None = None, namespace: str | None = None,
                ollama: str | None = None) -> Inventory:
    """Catalog entries (installed, partial or available), unlisted local GGUFs, Ollama's
    installed models, and who holds each now. Sizes are read, not hashed: use verify."""
    catalog = catalog or load()
    url = _ollama_url(ollama)
    folders = {role: folder(role, catalog) for role in FOLDER_ROLES}
    try:
        holders = _residents(namespace=namespace, ollama=url)
        residency = "known"
    except ResidencyError:
        holders, residency = (), "unavailable"
    if _ollama.loopback(url) is None:
        tags, ollama_state = (), "not-loopback"
    else:
        try:
            tags, ollama_state = _ollama.installed(url), "reachable"
        except _ollama.OllamaError:
            tags, ollama_state = (), "unreachable"
    rows: list[dict] = []
    for role, directory in folders.items():
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            names = []
        present = {}
        for name in names:
            try:
                present[name] = os.stat(os.path.join(directory, name), follow_symlinks=False)
            except OSError:
                pass
        listed = set()
        for entry in (e for e in catalog.entries if e.folder == role):
            info, part = present.get(entry.file), present.get(_part(entry.file))
            listed.add(entry.file)
            if info is not None and stat.S_ISLNK(info.st_mode):
                state = "link"
            elif info is not None:
                state = "installed" if stat.S_ISREG(info.st_mode) and info.st_size == entry.bytes else "mismatch"
            elif part is not None and stat.S_ISREG(part.st_mode):
                state = "partial"
            else:
                state = "available"
            rows.append({"id": entry.id, "name": entry.name, "kind": entry.kind, "source": "catalog",
                         "state": state, "folder": role, "path": os.path.join(directory, entry.file),
                         "bytes": entry.bytes, "info": info, "file": entry.file, "entry": entry,
                         "partial_bytes": part.st_size if state == "partial" else None})
        kind = next(k for k, r in FOLDER_FOR_KIND.items() if r == role)
        for name, info in present.items():
            if name in listed or not plain_gguf(name) or not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                continue
            rows.append({"id": name, "name": name, "kind": kind, "source": "local",
                         "state": "link" if stat.S_ISLNK(info.st_mode) else "local", "folder": role,
                         "path": os.path.join(directory, name), "bytes": info.st_size, "info": info,
                         "file": name, "entry": None, "partial_bytes": None})
    targets = {_identity(row["info"]): row["file"] for row in rows
               if row["info"] is not None and stat.S_ISREG(row["info"].st_mode)}
    opened = open_by(targets)
    models = []
    for row in rows:
        info = row.pop("info")
        names = _file_names(row.pop("entry"), os.path.dirname(row["path"]), row.pop("file"))
        who = sorted({h.app for h in holders if h.registered and h.backend != "ollama" and h.model in names})
        if info is not None and stat.S_ISREG(info.st_mode):
            who += [f"pid {pid} ({command})" for pid, command in sorted(opened.get(_identity(info), []))]
        models.append(Listed(**row, resident=tuple(who)))
    on_disk = {_canonical(tag.name): tag for tag in tags}
    for entry in (e for e in catalog.entries if e.kind == "ollama"):
        tag = on_disk.pop(_canonical(entry.ollama), None)
        models.append(_ollama_row(entry.id, entry.name, entry.ollama, tag, holders, "catalog"))
    for key, tag in sorted(on_disk.items()):
        models.append(_ollama_row(f"ollama:{tag.name}", tag.name, tag.name, tag, holders, "ollama"))
    return Inventory(tuple(models), folders, ollama_state, residency)


def _ollama_row(ident: str, name: str, model: str, tag: _ollama.Installed | None,
                holders: tuple[ResidentInfo, ...], source: str) -> Listed:
    who = sorted({h.app for h in holders if h.backend == "ollama" and _canonical(h.model) == _canonical(model)})
    return Listed(ident, name, "ollama", source, "installed" if tag else "available", None, None,
                  tag.size_bytes if tag else None, None, tuple(who))

# }}}
