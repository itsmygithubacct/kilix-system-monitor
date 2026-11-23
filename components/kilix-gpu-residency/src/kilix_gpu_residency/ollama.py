"""The two Ollama calls the registry makes: list loaded models, and unload one.

Only a loopback Ollama is spoken to: a remote server's models are not on this
machine's GPUs, and an unload sent there would free nothing here.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlsplit

DEFAULT_URL = "http://127.0.0.1:11434"
_TIMEOUT = 2.0
_UNLOAD_TIMEOUT = 30.0
_MAX_BYTES = 1 << 20
# Ollama model names: name[:tag], with an optional registry/namespace path.
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,159}\Z", re.ASCII)


class OllamaError(RuntimeError):
    """Ollama could not be reached or answered something unexpected."""


@dataclass(frozen=True)
class Loaded:
    name: str
    size_bytes: int
    vram_bytes: int
    expires_at: float | None  # unix time; None when unknown or far in the future


def default_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if not host:
        return DEFAULT_URL
    if "://" not in host:
        host = "http://" + host
    parts = urlsplit(host)
    return f"{parts.scheme}://{parts.hostname or '127.0.0.1'}:{parts.port or 11434}"


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


def _request(url: str, method: str, path: str, body: dict | None, timeout: float) -> dict:
    address = loopback(url)
    if address is None:
        raise OllamaError("Ollama is only contacted on a loopback address")
    connection = http.client.HTTPConnection(*address, timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body).encode()
        connection.request(method, path, body=payload,
                           headers={"Content-Type": "application/json"} if payload else {})
        response = connection.getresponse()
        raw = response.read(_MAX_BYTES + 1)
        if response.status != 200:
            raise OllamaError(f"Ollama answered HTTP {response.status} to {path}")
    except (OSError, http.client.HTTPException) as error:
        raise OllamaError(f"Ollama is not reachable: {type(error).__name__}") from error
    finally:
        connection.close()
    if len(raw) > _MAX_BYTES:
        raise OllamaError("Ollama's answer is too large")
    try:
        value = json.loads(raw.splitlines()[-1] if raw.strip() else b"{}")
    except (ValueError, IndexError) as error:
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
    value = _request(url, "GET", "/api/ps", None, _TIMEOUT)
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


def unload(url: str, name: str, *, timeout: float = _UNLOAD_TIMEOUT) -> None:
    """Ask Ollama to unload ``name`` (keep_alive 0) and wait until it is gone.

    Ollama finishes a request already running on the model before it unloads."""
    if not MODEL_NAME.fullmatch(name):
        raise OllamaError("Not an Ollama model name")
    _request(url, "POST", "/api/generate", {"model": name, "keep_alive": 0}, timeout)
    deadline = time.monotonic() + timeout
    while any(model.name == name for model in loaded(url)):
        if time.monotonic() >= deadline:
            raise OllamaError("Ollama did not unload the model in time")
        time.sleep(0.1)
