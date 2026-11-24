"""Shared helpers for the security seat's controls, adopted from review-p1/security/controls/seccommon.py.

Everything reuses the component's own harness: tests/fakes.py (ResidencyCase, the
fake sysfs, the fake NVML that follows real processes, the fake loopback Ollama)
and tests/residency_containment.py (the path guard, installed on import). A
control that diverges from the component therefore diverges in the code, not in
a private fixture.

Run with PYTHONPATH=<component>/tests (see run-controls.sh).
"""
from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time

import residency_containment  # noqa: F401  (installs the guard first)
from fakes import stop

# A child that renames itself (prctl PR_SET_NAME, i.e. /proc/<pid>/comm) and sleeps.
# Any user can do this to any of their own processes; the kernel does not escape it.
NAMED = ("import ctypes, sys, time\n"
         "ctypes.CDLL(None).prctl(15, sys.argv[1].encode('latin-1'), 0, 0, 0)\n"
         "print('ready', flush=True)\n"
         "time.sleep(600)\n")


def spawn_named(case, comm: str, vram_bytes: int, *, holds_on: dict | None = None) -> subprocess.Popen:
    """A real process group whose comm is ``comm``, holding fake VRAM while it lives.

    ``holds_on`` is the dict the fake NVML reads for the device (default: case.processes)."""
    process = subprocess.Popen([sys.executable, "-c", NAMED, comm], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, text=True, process_group=0,
                               env=residency_containment.child_env())
    case.addCleanup(stop, process)
    case.assertEqual(process.stdout.readline().strip(), "ready")
    with open(f"/proc/{process.pid}/comm", "rb") as stream:
        case.assertEqual(stream.read().rstrip(b"\n"), comm.encode("latin-1")[:15])
    (case.processes if holds_on is None else holds_on)[process.pid] = vram_bytes
    return process


class Peer(threading.Thread):
    """A hostile loopback HTTP peer standing where Ollama is expected.

    mode "trickle": a 200 header promising ``length`` bytes, then one byte every
    ``interval`` seconds for ``seconds`` seconds, then close.
    mode "nested": a complete 200 answer of ``depth`` '[' characters.
    mode "redirect": a 302 to a non-loopback address.
    """

    def __init__(self, mode: str, *, seconds: float = 10.0, interval: float = 0.5, depth: int = 200_000):
        super().__init__(daemon=True)
        self.mode, self.seconds, self.interval, self.depth = mode, seconds, interval, depth
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(8)
        self.url = f"http://127.0.0.1:{self.server.getsockname()[1]}"
        self.requests: list[bytes] = []
        self.done = threading.Event()
        self.start()

    def run(self) -> None:
        while not self.done.is_set():
            try:
                self.server.settimeout(0.2)
                connection, _ = self.server.accept()
            except (TimeoutError, OSError):
                continue
            threading.Thread(target=self.serve, args=(connection,), daemon=True).start()

    def serve(self, connection: socket.socket) -> None:
        with connection:
            connection.settimeout(5)
            try:
                self.requests.append(connection.recv(65536))
                if self.mode == "trickle":
                    connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                                       b"Content-Length: 100000\r\n\r\n")
                    deadline = time.monotonic() + self.seconds
                    while time.monotonic() < deadline and not self.done.is_set():
                        connection.sendall(b" ")
                        time.sleep(self.interval)
                elif self.mode == "nested":
                    body = b"[" * self.depth
                    connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                                       + str(len(body)).encode() + b"\r\n\r\n" + body)
                elif self.mode == "redirect":
                    connection.sendall(b"HTTP/1.1 302 Found\r\nLocation: http://192.0.2.1:11434/api/ps\r\n"
                                       b"Content-Length: 0\r\n\r\n")
            except OSError:
                pass

    def close(self) -> None:
        self.done.set()
        self.server.close()
