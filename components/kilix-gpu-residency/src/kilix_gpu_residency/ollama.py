"""The Ollama calls the registry and the model catalog make.

The registry lists loaded models and unloads one; the catalog lists, pulls and
deletes installed ones. Only a loopback Ollama is spoken to: a remote server's
models are not on this machine's GPUs or disks, and an unload or delete sent
there would change nothing here.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from urllib.parse import urlsplit

DEFAULT_URL = "http://127.0.0.1:11434"
_TIMEOUT = 2.0  # the whole of one exchange: connect, send and every read
_UNLOAD_TIMEOUT = 30.0  # the whole of one unload: the request and the wait until /api/ps drops it
KEEP_ALIVE_DEFAULT = 300.0  # Ollama's own default keep-alive
_PULL_TIMEOUT = 600.0  # longest wait for the next progress line of a pull
_MAX_BYTES = 1 << 20
_MAX_LINE = 1 << 16
# Ollama model names: name[:tag], with an optional registry/namespace path.
_DIGEST = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,159}\Z", re.ASCII)


class OllamaError(RuntimeError):
    """Ollama could not be reached or answered something unexpected."""


class OllamaMissing(OllamaError):
    """Ollama answered 404: it does not have that model."""


class OllamaTimeout(OllamaError):
    """The exchange's total deadline passed. For an unload: Ollama may still unload the model."""


@dataclass(frozen=True)
class Installed:
    name: str
    size_bytes: int
    digest: str | None = None  # the manifest's sha256, as GET /api/tags reports it


@dataclass(frozen=True)
class Loaded:
    name: str
    size_bytes: int
    vram_bytes: int
    expires_at: float | None  # unix time; None when unknown or far in the future


def default_url() -> str:
    """The URL OLLAMA_HOST names, as Ollama's own client reads it.

    A bare host or host:port gets http:// and port 11434; an IPv6 host keeps its brackets;
    the unspecified addresses a server binds to (0.0.0.0, ::) mean this machine's loopback.
    A host that is not loopback stays as given and is never contacted. Raises OllamaError
    for a value that is not a URL (a port out of range, for example)."""
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if not host:
        return DEFAULT_URL
    if "://" not in host:
        host = "http://" + host
    try:
        parts = urlsplit(host)
        port = parts.port or 11434
        name = parts.hostname or "127.0.0.1"
    except ValueError as error:
        raise OllamaError("OLLAMA_HOST is not a valid address") from error
    if parts.username is not None or parts.password is not None:
        return f"{parts.scheme}://{parts.netloc}"  # kept as written, so loopback() refuses it
    if name in ("0.0.0.0", "::"):
        name = "127.0.0.1" if name == "0.0.0.0" else "::1"
    if ":" in name:
        name = f"[{name}]"
    return f"{parts.scheme}://{name}:{port}"


def canonical(name: str) -> str:
    """The name Ollama lists a model under: ``llama3`` is ``llama3:latest``."""
    base = name.rsplit("/", 1)[-1]
    return name if ":" in base else name + ":latest"


def keep_alive_seconds() -> float | None:
    """Ollama's keep-alive from OLLAMA_KEEP_ALIVE (seconds, or a Go duration such as ``1h30m``),
    else 5 minutes; None when it is negative (models are kept for ever) or unreadable."""
    text = os.environ.get("OLLAMA_KEEP_ALIVE", "").strip()
    if not text:
        return KEEP_ALIVE_DEFAULT
    try:
        value = float(text)
    except ValueError:
        value = _duration(text)
    if value is None or value < 0 or value != value:
        return None
    return value


_UNITS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}
_PART = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(ns|us|µs|ms|s|m|h)")


def _duration(text: str) -> float | None:
    sign = -1.0 if text.startswith("-") else 1.0
    body = text.lstrip("+-")
    total, position = 0.0, 0
    for match in _PART.finditer(body):
        if match.start() != position:
            return None
        total += float(match.group(1)) * _UNITS[match.group(2)]
        position = match.end()
    if position != len(body) or not body:
        return None
    return sign * total


def loopback(url: str) -> tuple[str, int] | None:
    """(host, port) when ``url`` is plain http to a loopback address, else None."""
    try:
        parts = urlsplit(url)
        if parts.scheme != "http" or parts.hostname is None or parts.path not in ("", "/"):
            return None
        host = parts.hostname
        if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
            return None
        return host, parts.port or 80
    except ValueError:
        return None


class _Deadline:
    """Shuts a connection's socket down when an exchange's total time runs out, so a peer that
    trickles its answer a byte at a time cannot hold the caller past it."""

    def __init__(self, connection: http.client.HTTPConnection, seconds: float) -> None:
        self.connection = connection
        self.fired = False
        self.timer = threading.Timer(seconds, self.fire)
        self.timer.daemon = True
        self.timer.start()

    def fire(self) -> None:
        self.fired = True
        sock = self.connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def cancel(self) -> None:
        self.timer.cancel()


def _connect(url: str, method: str, path: str, body: dict | None, timeout: float, *, total: float | None = None
             ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse, _Deadline | None]:
    address = loopback(url)
    if address is None:
        raise OllamaError("Ollama is only contacted on a loopback address")
    host = address[0].strip("[]")
    connection = http.client.HTTPConnection(host, address[1], timeout=timeout)
    deadline = None
    try:
        connection.connect()
        if total is not None:
            deadline = _Deadline(connection, total)
        payload = None if body is None else json.dumps(body).encode()
        connection.request(method, path, body=payload,
                           headers={"Content-Type": "application/json"} if payload else {})
        return connection, connection.getresponse(), deadline
    except (OSError, http.client.HTTPException) as error:
        if deadline is not None:
            deadline.cancel()
        connection.close()
        if deadline is not None and deadline.fired:
            raise OllamaTimeout(f"Ollama did not answer {path} in time") from error
        raise OllamaError(f"Ollama is not reachable: {type(error).__name__}") from error


def _request(url: str, method: str, path: str, body: dict | None, timeout: float) -> dict:
    """One exchange, which ends within ``timeout`` seconds however the peer answers."""
    connection, response, deadline = _connect(url, method, path, body, timeout, total=timeout)
    try:
        raw = response.read(_MAX_BYTES + 1)
        if deadline is not None and deadline.fired:
            raise OllamaTimeout(f"Ollama did not answer {path} in time")
        if response.status == 404:
            raise OllamaMissing(f"Ollama does not have that model ({path})")
        if response.status != 200:
            raise OllamaError(f"Ollama answered HTTP {response.status} to {path}")
    except (OSError, http.client.HTTPException) as error:
        if deadline is not None and deadline.fired:
            raise OllamaTimeout(f"Ollama did not answer {path} in time") from error
        raise OllamaError(f"Ollama is not reachable: {type(error).__name__}") from error
    finally:
        if deadline is not None:
            deadline.cancel()
        connection.close()
    if len(raw) > _MAX_BYTES:
        raise OllamaError("Ollama's answer is too large")
    try:
        value = json.loads(raw.splitlines()[-1] if raw.strip() else b"{}")
    except (ValueError, IndexError, RecursionError) as error:
        raise OllamaError("Ollama's answer is not JSON") from error
    if type(value) is not dict:
        raise OllamaError("Ollama's answer is not an object")
    return value


def _expiry(text: object) -> float | None:
    if type(text) is not str or not text:
        return None
    try:
        when = datetime.fromisoformat(text)  # Python 3.11+ reads Go's nanoseconds and "Z"
    except ValueError:
        return None
    if when.tzinfo is None:
        return None
    # keep_alive < 0 keeps a model forever; Ollama reports a far-future time.
    stamp = when.timestamp()
    return None if stamp > time.time() + 86400 * 365 else stamp


def loaded(url: str) -> tuple[Loaded, ...]:
    """The models Ollama has loaded now, from ``GET /api/ps``."""
    return _loaded(url, _TIMEOUT)


def _loaded(url: str, timeout: float) -> tuple[Loaded, ...]:
    value = _request(url, "GET", "/api/ps", None, timeout)
    models = value.get("models")
    if type(models) is not list:
        raise OllamaError("Ollama's model list is malformed")
    found = []
    for item in models:
        if type(item) is not dict:
            continue
        name = item.get("name") or item.get("model")
        size, vram = item.get("size", 0), item.get("size_vram", 0)
        if type(name) is not str or not MODEL_NAME.fullmatch(name):
            continue
        if type(size) is not int or type(vram) is not int or size < 0 or vram < 0:
            continue
        found.append(Loaded(name, size, vram, _expiry(item.get("expires_at"))))
    return tuple(found)


def installed(url: str) -> tuple[Installed, ...]:
    """The models Ollama has on disk, from ``GET /api/tags``."""
    value = _request(url, "GET", "/api/tags", None, _TIMEOUT)
    models = value.get("models")
    if type(models) is not list:
        raise OllamaError("Ollama's model list is malformed")
    found = []
    for item in models:
        if type(item) is not dict:
            continue
        name, size = item.get("name") or item.get("model"), item.get("size", 0)
        digest = item.get("digest")
        if type(digest) is not str or not _DIGEST.fullmatch(digest):
            digest = None
        if type(name) is str and MODEL_NAME.fullmatch(name) and type(size) is int and size >= 0:
            found.append(Installed(name, size, digest))
    return tuple(found)


def has(url: str, name: str) -> bool:
    """Whether Ollama has ``name`` on disk (``POST /api/show``)."""
    if not MODEL_NAME.fullmatch(name):
        raise OllamaError("Not an Ollama model name")
    try:
        _request(url, "POST", "/api/show", {"model": name}, _TIMEOUT * 5)
    except OllamaMissing:
        return False
    return True


def _printable(text: object) -> str:
    return "".join(c for c in str(text) if c.isprintable())[:200]


def pull(url: str, name: str, progress: Callable[[str, int | None, int | None], None] | None = None,
         *, timeout: float = _PULL_TIMEOUT) -> None:
    """Pull ``name`` (``POST /api/pull``, streamed) and return once Ollama reports success."""
    if not MODEL_NAME.fullmatch(name):
        raise OllamaError("Not an Ollama model name")
    connection, response, _ = _connect(url, "POST", "/api/pull", {"model": name, "stream": True}, timeout)
    try:
        if response.status != 200:
            raw = response.read(_MAX_LINE)
            try:
                detail = json.loads(raw).get("error")
            except (ValueError, AttributeError, RecursionError):
                detail = None
            raise OllamaError(f"Ollama could not pull it: {_printable(detail or f'HTTP {response.status}')}")
        done = False
        while True:
            line = response.readline(_MAX_LINE + 1)
            if not line:
                break
            if len(line) > _MAX_LINE:
                raise OllamaError("Ollama's pull progress line is too long")
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (ValueError, RecursionError) as error:
                raise OllamaError("Ollama's pull progress is not JSON") from error
            if type(value) is not dict:
                raise OllamaError("Ollama's pull progress is not an object")
            if "error" in value:
                raise OllamaError(f"Ollama could not pull it: {_printable(value['error'])}")
            status = value.get("status")
            if status == "success":
                done = True
            elif progress is not None and type(status) is str:
                completed, total = value.get("completed"), value.get("total")
                progress(_printable(status), completed if type(completed) is int else None,
                         total if type(total) is int else None)
    except (OSError, http.client.HTTPException) as error:
        raise OllamaError(f"Ollama stopped answering during the pull: {type(error).__name__}") from error
    finally:
        connection.close()
    if not done:
        raise OllamaError("Ollama ended the pull without reporting success")


def delete(url: str, name: str) -> None:
    """Delete ``name`` from Ollama's disk (``DELETE /api/delete``); OllamaMissing when absent."""
    if not MODEL_NAME.fullmatch(name):
        raise OllamaError("Not an Ollama model name")
    _request(url, "DELETE", "/api/delete", {"model": name}, _TIMEOUT * 15)


def unload(url: str, name: str, *, timeout: float = _UNLOAD_TIMEOUT) -> None:
    """Ask Ollama to unload ``name`` (keep_alive 0) and wait until /api/ps no longer lists it.

    The whole takes at most ``timeout`` seconds. Ollama finishes a request already running on
    the model before it unloads; OllamaTimeout means it had not unloaded it by then (it may
    still), OllamaError that it could not be asked."""
    if not MODEL_NAME.fullmatch(name):
        raise OllamaError("Not an Ollama model name")
    deadline = time.monotonic() + timeout
    _request(url, "POST", "/api/generate", {"model": name, "keep_alive": 0}, timeout)
    wanted = canonical(name)
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise OllamaTimeout("Ollama did not unload the model in time")
        try:
            present = any(canonical(model.name) == wanted for model in _loaded(url, min(_TIMEOUT, left)))
        except OllamaTimeout:
            continue
        if not present:
            return
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
