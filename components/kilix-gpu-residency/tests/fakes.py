"""Fake GPUs, a fake NVML, a fake Ollama and real process groups for the residency tests.

The fake NVML's memory follows real processes: a process listed as holding
VRAM holds it only while it is alive (not a zombie), and the device's used
memory is a base plus what live processes hold. Unloading a model, by killing
its process group or through the fake Ollama, therefore frees fake VRAM the
same way it frees real VRAM.
"""
from __future__ import annotations

import ctypes
import http.server
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from typing import Any, Callable

import residency_containment
import kilix_gpu_residency as residency
from kilix_gpu_residency import ollama as ollama_module
from kilix_gpu_residency import vram

GIB = 1 << 30
MIB = 1 << 20
NVIDIA = ("0000:01:00.0", 0x10de, 0x1f36, "nvidia", ("card1", "renderD129"))
SECOND_NVIDIA = ("0000:02:00.0", 0x10de, 0x2484, "nvidia", ("card2", "renderD130"))
INTEL = ("0000:00:02.0", 0x8086, 0x3e9b, "i915", ("card0", "renderD128"))
AMD = ("0000:03:00.0", 0x1002, 0x73bf, "amdgpu", ("card3", "renderD131"))


def make_gpu(root: Path, pci: str, vendor: int, device: int, driver: str, nodes: tuple[str, ...],
             status: str = "active", files: dict[str, str] | None = None) -> Path:
    """A sysfs PCI function and its DRM nodes, linked the way the kernel links them."""
    dev = root / "devices" / "pci0000:00" / pci
    (dev / "power").mkdir(parents=True)
    (dev / "vendor").write_text(f"0x{vendor:04x}\n")
    (dev / "device").write_text(f"0x{device:04x}\n")
    (dev / "power" / "runtime_status").write_text(status + "\n")
    drivers = root / "bus" / "pci" / "drivers" / driver
    drivers.mkdir(parents=True, exist_ok=True)
    os.symlink(drivers, dev / "driver")
    for name, value in (files or {}).items():
        (dev / name).write_text(value + "\n")
    drm = root / "class" / "drm"
    drm.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        (drm / node).mkdir()
        os.symlink(dev, drm / node / "device")
    # A connector shares the device but is not a GPU of its own.
    (drm / f"{nodes[0]}-DP-1").mkdir()
    os.symlink(dev, drm / f"{nodes[0]}-DP-1" / "device")
    return dev


def alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as stream:
            text = stream.read(4096).decode("ascii", "replace")
    except OSError:
        return False
    return text[text.rfind(")") + 2:].split()[0] not in ("Z", "X")


class FakeNvml:
    """Enough of libnvidia-ml, keyed by PCI bus id."""

    STRUCT_SIZES = {"_v3": 24, "_v2": 24, "": 16}

    def __init__(self, devices: dict[str, dict[str, Any]], versions: tuple[str, ...] = ("_v3", "_v2", "")) -> None:
        self.devices = devices
        self.order = list(devices)
        self.versions = versions
        self.calls: list[str] = []
        self.open_sessions = 0

    def __getattr__(self, name: str) -> Any:
        for base in vram.PROCESS_LISTS:
            if name.startswith(base) and name[len(base):] in self.versions:
                suffix = name[len(base):]
                return lambda handle, count, array: self._processes(name, base, suffix, handle, count, array)
        raise AttributeError(name)

    def _device(self, handle: ctypes.c_void_p) -> dict[str, Any]:
        return self.devices[self.order[int(handle.value or 0) - 1]]

    def live(self, device: dict[str, Any], key: str = "compute") -> list[tuple[int, int]]:
        entries = device.get(key, {})
        if callable(entries):
            entries = entries()
        return [(pid, used) for pid, used in dict(entries).items() if alive(pid)]

    def nvmlInit_v2(self) -> int:
        self.calls.append("init")
        self.open_sessions += 1
        return 0

    def nvmlShutdown(self) -> int:
        self.calls.append("shutdown")
        self.open_sessions -= 1
        return 0

    def nvmlDeviceGetHandleByPciBusId_v2(self, bus_id: bytes, handle: Any) -> int:
        if bus_id.decode() not in self.devices:
            return 13
        handle._obj.value = self.order.index(bus_id.decode()) + 1
        return 0

    def nvmlDeviceGetName(self, handle: ctypes.c_void_p, buffer: Any, length: Any) -> int:
        buffer.value = self._device(handle).get("name", "Fake GPU").encode()
        return 0

    def nvmlDeviceGetMemoryInfo(self, handle: ctypes.c_void_p, info: Any) -> int:
        device = self._device(handle)
        extra = device.get("extra", 0)
        used = device.get("base", 0) + sum(used for _, used in self.live(device)) + (extra() if callable(extra) else extra)
        total = device.get("total", 6 * GIB)
        info._obj.total, info._obj.used, info._obj.free = total, used, max(0, total - used)
        return 0

    def _processes(self, name: str, base: str, suffix: str, handle: ctypes.c_void_p, count: Any, array: Any) -> int:
        self.calls.append(name)
        if ctypes.sizeof(array._type_) != self.STRUCT_SIZES[suffix]:
            raise ValueError(f"{name} given a {ctypes.sizeof(array._type_)}-byte struct")
        entries = self.live(self._device(handle), "compute" if "Compute" in base else "graphics")
        if count._obj.value < len(entries):
            count._obj.value = len(entries)
            return vram.NVML_ERROR_INSUFFICIENT_SIZE
        for i, (pid, used) in enumerate(entries):
            array[i].pid, array[i].usedGpuMemory = pid, used
        count._obj.value = len(entries)
        return 0


def digest_of(name: str) -> str:
    """The manifest digest FakeOllama lists for a model, unless told otherwise."""
    import hashlib
    return hashlib.sha256(name.encode()).hexdigest()


class FakeOllama:
    """GET /api/ps and POST /api/generate with keep_alive 0, on a loopback port.

    Like Ollama 0.40, a real process named ``ollama`` (the server) has a child
    named ``llama-server`` (the runner); the fake NVML lists the runner as
    holding the VRAM of every loaded model. With ``legacy`` the runner is the
    process named ``ollama`` itself, as in older releases."""

    def __init__(self, scratch: Path, *, legacy: bool = False) -> None:
        self.models: dict[str, dict[str, Any]] = {}
        self.disk: dict[str, int] = {}  # installed models (GET /api/tags) and their sizes
        self.digests: dict[str, Any] = {}  # a manifest digest to list instead of digest_of(name)
        self.pull_digest: str | None = None  # the digest the next pulls install, as if the tag had moved
        self.pull_error: str | None = None  # an error line the next pulls report
        self.pull_truncated = False  # end the next pulls' progress before "success"
        self.requests: list[tuple[str, str, Any]] = []
        self.ignore_unload = False
        self.lock = threading.Lock()
        binaries = scratch / "bin"
        binaries.mkdir(parents=True, exist_ok=True)
        if legacy:
            os.symlink(shutil.which("sleep"), binaries / "ollama")
            self.server = None
            self.runner = subprocess.Popen([str(binaries / "ollama"), "600"], stdin=subprocess.DEVNULL,
                                           start_new_session=True)
            self.runner_pid = self.runner.pid
        else:
            os.symlink(shutil.which("sh"), binaries / "ollama")
            os.symlink(shutil.which("sleep"), binaries / "llama-server")
            self.runner = subprocess.Popen([str(binaries / "ollama"), "-c", f'"{binaries}/llama-server" 600 & wait'],
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            self.runner_pid = self._child_of(self.runner.pid)
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def reply(self, value: Any, code: int = 200) -> None:
                raw = json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def body(self) -> Any:
                return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

            def do_GET(self) -> None:
                owner.requests.append(("GET", self.path, None))
                with owner.lock:
                    if self.path == "/api/tags":
                        models = [{"name": name, "model": name, "size": size,
                                   "digest": owner.digests.get(name, digest_of(name))}
                                  for name, size in owner.disk.items()]
                    else:
                        models = [{"name": name, "model": name, **value} for name, value in owner.models.items()]
                self.reply({"models": models})

            def do_DELETE(self) -> None:
                body = self.body()
                owner.requests.append(("DELETE", self.path, body))
                with owner.lock:
                    found = owner.disk.pop(body.get("model"), None) is not None
                self.reply({} if found else {"error": "model not found"}, 200 if found else 404)

            def do_POST(self) -> None:
                body = self.body()
                owner.requests.append(("POST", self.path, body))
                if self.path == "/api/show":
                    with owner.lock:
                        found = body.get("model") in owner.disk
                    self.reply({"modelfile": "FROM x"} if found else {"error": "not found"}, 200 if found else 404)
                    return
                if self.path == "/api/pull":
                    lines = [{"status": "pulling manifest"},
                             {"status": "pulling abc", "completed": 5, "total": 10},
                             {"error": owner.pull_error} if owner.pull_error else {"status": "success"}]
                    if owner.pull_truncated:
                        lines = lines[:-1]
                    raw = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
                    if not owner.pull_error:
                        with owner.lock:
                            owner.disk[body.get("model")] = 2 * GIB
                            if owner.pull_digest:
                                owner.digests[body.get("model")] = owner.pull_digest
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if self.path == "/api/generate" and body.get("keep_alive") == 0 and not owner.ignore_unload:
                    with owner.lock:
                        owner.models.pop(body.get("model"), None)
                self.reply({"model": body.get("model"), "done": True, "done_reason": "unload"})

        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    @staticmethod
    def _child_of(parent: int) -> int:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for name in os.listdir("/proc"):
                if name.isdigit():
                    try:
                        text = Path(f"/proc/{name}/stat").read_text()
                    except OSError:
                        continue
                    if int(text[text.rfind(")") + 2:].split()[1]) == parent:
                        return int(name)
            time.sleep(0.01)
        raise RuntimeError("the fake Ollama runner did not start")

    def load(self, name: str, vram_bytes: int, expires_in: float = 300.0) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + expires_in)) + ".123456789Z"
        with self.lock:
            self.models[name] = {"size": vram_bytes + 512 * MIB, "size_vram": vram_bytes, "expires_at": stamp}

    def held(self) -> dict[int, int]:
        with self.lock:
            total = sum(model["size_vram"] for model in self.models.values())
        return {self.runner_pid: total} if total else {}

    def unloads(self) -> list[str]:
        return [body["model"] for method, path, body in self.requests
                if method == "POST" and path == "/api/generate" and body.get("keep_alive") == 0]

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()
        stop(self.runner)


class FakeDownloads:
    """A loopback file server shaped like Hugging Face: ``/resolve/<name>`` redirects to
    ``/files/<name>``, which honours ``Range``. ``drops[name]`` lists the byte offsets at
    which the next responses break the connection; ``ignore_range`` answers 200 to a Range."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.drops: dict[str, list[int]] = {}
        self.ignore_range = False
        self.shift = 0  # resume this many bytes before the offset asked for
        self.redirect: str | None = None  # where /resolve/ sends instead, when set
        self.status: int | None = None  # answer every /files/ request with this status
        self.requests: list[tuple[str, str | None]] = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def do_GET(self) -> None:
                owner.requests.append((self.path, self.headers.get("Range")))
                if self.path.startswith("/resolve/"):
                    self.send_response(302)
                    self.send_header("Location", owner.redirect or "/files/" + self.path[len("/resolve/"):])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                name = self.path[len("/files/"):]
                data = owner.files.get(name)
                if owner.status is not None or data is None:
                    self.send_response(owner.status or 404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = 0
                wanted = self.headers.get("Range")
                if wanted and not owner.ignore_range:
                    start = max(0, int(wanted[len("bytes="):].rstrip("-")) - owner.shift)
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                drops = owner.drops.get(name)
                stop_at = drops.pop(0) if drops else len(data)
                try:
                    self.wfile.write(data[start:max(start, stop_at)])
                    if stop_at < len(data):
                        self.wfile.flush()
                        self.connection.shutdown(socket.SHUT_RDWR)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client stopped reading (a mismatch it noticed early)
                self.close_connection = True

        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    def ranges(self, name: str) -> list[str | None]:
        return [wanted for path, wanted in self.requests if path == "/files/" + name]

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)  # a leader takes its group with it
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.kill()  # a member of another leader's group
    process.wait(timeout=10)
    if process.stdout is not None:
        process.stdout.close()


IGNORES_TERM = ("import signal, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "print('ready', flush=True)\n"
                "time.sleep(600)\n")


class ResidencyCase(unittest.TestCase):
    """A private namespace, a fake machine with one 6 GiB NVIDIA GPU (plus an Intel
    iGPU without a VRAM counter) and a fake Ollama."""

    gpus: tuple[tuple[Any, ...], ...] = (INTEL, NVIDIA)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="gpu-residency-")
        self.addCleanup(self.temp.cleanup)
        self.scratch = Path(self.temp.name)
        os.chmod(self.scratch, 0o700)
        self.namespace = str(self.scratch / "registry")
        self.sys = self.scratch / "sys"
        for spec in self.gpus:
            make_gpu(self.sys, *spec)
        self.environ = {key: os.environ.get(key) for key in (
            "KILIX_GPU_SYSFS_ROOT", "KILIX_GPU_PROC_ROOT", "KILIX_GPU_DEVICES", "OLLAMA_HOST", "HOME",
            "KILIX_LLM_MODELS_DIR", "PUZZLLM_DIFFUSION_MODELS", "GPU_TERMINAL_HOME", "XDG_DATA_HOME")}
        self.addCleanup(self.restore_environ)
        os.environ["KILIX_GPU_SYSFS_ROOT"] = str(self.sys)
        self.ollama = FakeOllama(self.scratch)
        self.addCleanup(self.ollama.close)
        self.processes: dict[int, int] = {}
        self.nvml_devices = {spec[0]: {"name": "Fake RTX", "total": 6 * GIB, "base": 100 * MIB,
                                       "compute": self.held}
                             for spec in self.gpus if spec[3] == "nvidia"}
        self.nvml = FakeNvml(self.nvml_devices)
        self.loader_calls = 0
        previous = vram.nvml_loader
        self.addCleanup(setattr, vram, "nvml_loader", previous)
        vram.nvml_loader = self.load_nvml
        for name in ("TERM_GRACE_SECONDS", "KILL_GRACE_SECONDS", "RELEASE_WAIT_SECONDS"):
            self.addCleanup(setattr, residency, name, getattr(residency, name))
        residency.TERM_GRACE_SECONDS = 2.0
        residency.KILL_GRACE_SECONDS = 2.0
        residency.RELEASE_WAIT_SECONDS = 1.0

    def restore_environ(self) -> None:
        for key, value in self.environ.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def load_nvml(self) -> FakeNvml:
        self.loader_calls += 1
        return self.nvml

    def held(self) -> dict[int, int]:
        return {**self.processes, **self.ollama.held()}

    def where(self) -> dict[str, str]:
        return {"namespace": self.namespace, "ollama": self.ollama.url}

    def spawn(self, vram_bytes: int = 0, *, ignore_term: bool = False) -> subprocess.Popen:
        """A real process group, holding ``vram_bytes`` of fake VRAM while it lives."""
        if ignore_term:
            process = subprocess.Popen([sys.executable, "-c", IGNORES_TERM], stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, text=True, process_group=0,
                                       env=residency_containment.child_env())
            self.assertEqual(process.stdout.readline().strip(), "ready")
        else:
            process = subprocess.Popen(["sleep", "600"], stdin=subprocess.DEVNULL, process_group=0)
        self.addCleanup(stop, process)
        if vram_bytes:
            self.processes[process.pid] = vram_bytes
        return process

    def join(self, leader: subprocess.Popen, vram_bytes: int = 0) -> subprocess.Popen:
        """Another process in ``leader``'s process group."""
        process = subprocess.Popen(["sleep", "600"], stdin=subprocess.DEVNULL,
                                   process_group=leader.pid)
        self.addCleanup(stop, process)
        if vram_bytes:
            self.processes[process.pid] = vram_bytes
        return process

    def model(self, app: str, model: str, vram_bytes: int, *, state: str = "idle", pinned: bool = False,
              ignore_term: bool = False, **kwargs: Any) -> tuple[residency.Resident, subprocess.Popen]:
        """A registered process-group resident in ``state``, holding ``vram_bytes``."""
        process = self.spawn(vram_bytes, ignore_term=ignore_term)
        resident = residency.register(app=app, model=model, backend="process", device="Vulkan1",
                                      pgid=process.pid, pinned=pinned, **kwargs, **self.where())
        if state == "idle":
            resident.idle()
        elif state == "busy":
            resident.busy()
        return resident, process

    def status(self, **kwargs: Any) -> residency.Status:
        return residency.status(**self.where(), **kwargs)

    def reap(self, process: subprocess.Popen, timeout: float = 10.0) -> int:
        return process.wait(timeout=timeout)

    def records(self) -> list[dict]:
        if not os.path.isdir(self.namespace):
            return []
        names = sorted(name for name in os.listdir(self.namespace) if name.endswith(".json")
                       and name not in ("pins.json", "events.json"))
        return [json.loads((Path(self.namespace) / name).read_text()) for name in names]


def wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


__all__ = ["ResidencyCase", "FakeNvml", "FakeOllama", "FakeDownloads", "make_gpu", "wait_for", "alive", "stop", "GIB", "MIB",
           "NVIDIA", "SECOND_NVIDIA", "INTEL", "AMD", "ollama_module", "digest_of"]


def blob(size: int, seed: int) -> bytes:
    import random
    return random.Random(seed).randbytes(size)


class CatalogCase(ResidencyCase):
    """ResidencyCase plus a private home with the catalog's default folders, a test
    catalog served from a fake loopback download server and a file:// mirror, and
    fast retry, lock and release timings."""

    SMALL = "Small-Q4.gguf"
    TINY = "Tiny-Diffusion.Q4.gguf"

    def setUp(self) -> None:
        super().setUp()
        from kilix_gpu_residency import catalog
        self.catalog_module = catalog
        self.home = self.scratch / "home"
        self.home.mkdir(mode=0o700)
        os.environ["HOME"] = str(self.home)
        for name in ("KILIX_LLM_MODELS_DIR", "PUZZLLM_DIFFUSION_MODELS", "GPU_TERMINAL_HOME", "XDG_DATA_HOME"):
            os.environ.pop(name, None)
        self.llm = self.home / ".local" / "gpu_terminal" / "kilix-llm" / "models"
        self.diffusion = self.home / ".local" / "share" / "puzzllm" / "diffusion"
        for name in ("ATTEMPTS", "BACKOFF_SECONDS", "LOCK_WAIT_SECONDS", "RELEASE_WAIT_SECONDS", "free_bytes"):
            self.addCleanup(setattr, catalog, name, getattr(catalog, name))
        catalog.BACKOFF_SECONDS = 0.0
        catalog.LOCK_WAIT_SECONDS = 0.2
        catalog.RELEASE_WAIT_SECONDS = 3.0
        self.downloads = FakeDownloads()
        self.addCleanup(self.downloads.close)
        self.small = blob(3 * MIB + 123, 1)
        self.tiny = blob(2 * MIB + 7, 2)
        self.downloads.files[self.SMALL] = self.small
        mirror = self.scratch / "mirror"
        mirror.mkdir(mode=0o700)
        (mirror / self.TINY).write_bytes(self.tiny)
        self.catalog_path = self.scratch / "catalog.json"
        self.write_catalog()

    def document(self) -> dict:
        import hashlib
        packaged = json.loads((Path(residency.__file__).parent / "catalog.json").read_text())
        common = {"architecture": "test", "context_length": 4096, "source": "a test fixture", "licence": "mit"}
        return {"schema": "kilix.gpu-residency.catalog/v1", "folders": packaged["folders"], "models": [
            {"id": "small", "name": "Small (Q4)", "kind": "gguf", "folder": "llama.cpp", "file": self.SMALL,
             "bytes": len(self.small), "sha256": hashlib.sha256(self.small).hexdigest(),
             "url": f"{self.downloads.url}/resolve/{self.SMALL}", **common},
            {"id": "tiny-diffusion", "name": "Tiny diffusion (Q4)", "kind": "diffusion-gguf", "folder": "diffusion",
             "file": self.TINY, "bytes": len(self.tiny), "sha256": hashlib.sha256(self.tiny).hexdigest(),
             "url": (self.scratch / "mirror" / self.TINY).as_uri(), **common},
            {"id": "tiny-ollama", "name": "Tiny (Ollama)", "kind": "ollama", "ollama": "tiny:1b", "bytes": 10**9,
             "digest": digest_of("tiny:1b"), **common},
        ]}

    def write_catalog(self, document: dict | None = None) -> None:
        self.catalog_path.write_text(json.dumps(document or self.document()))
        self.catalog = self.catalog_module.load(str(self.catalog_path))

    def install(self, target: str, **kwargs: Any):
        return self.catalog_module.install(target, catalog=self.catalog, ollama=self.ollama.url, **kwargs)

    def remove(self, target: str, **kwargs: Any):
        return self.catalog_module.remove(target, catalog=self.catalog, **self.where(), **kwargs)

    def verify(self, target: str, **kwargs: Any):
        return self.catalog_module.verify(target, catalog=self.catalog, **{"ollama": self.ollama.url, **kwargs})

    def inventory(self, **kwargs: Any):
        return self.catalog_module.list_models(catalog=self.catalog, **{**self.where(), **kwargs})

    def refused(self, outcome: str, call: Callable[..., Any], *args: Any, **kwargs: Any):
        with self.assertRaises(self.catalog_module.CatalogError) as caught:
            call(*args, **kwargs)
        self.assertEqual(caught.exception.outcome, outcome, str(caught.exception))
        return caught.exception

    def place(self, folder: Path, name: str, data: bytes) -> Path:
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        for parent in (folder, *folder.parents):
            if parent in (self.home, self.scratch):
                break
            os.chmod(parent, 0o700)
        path = folder / name
        path.write_bytes(data)
        return path

    def holding(self, path: Path) -> subprocess.Popen:
        """A process this suite started that holds ``path`` open (on its stdin) and nothing else."""
        with open(path, "rb") as stream:
            process = subprocess.Popen(["sleep", "600"], stdin=stream, process_group=0)
        self.addCleanup(stop, process)
        return process

    def cli(self, *argv: str) -> tuple[int, str, str]:
        import io
        from kilix_gpu_residency import cli
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(["--namespace", self.namespace, "--ollama", self.ollama.url, "models",
                         "--catalog", str(self.catalog_path), *argv], out, err)
        return code, out.getvalue(), err.getvalue()


__all__ += ["CatalogCase", "blob"]
