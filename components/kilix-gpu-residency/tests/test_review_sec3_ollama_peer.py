"""Adopted from the security seat's P1 review controls (review-p1/security/controls/test_sec3_ollama_peer.py).
Only the Required controls are kept; the Observed ones recorded the defects this round fixed.

S3, S4, S5: what the registry believes from whatever answers on the Ollama port.

The Ollama URL is loopback, but a loopback port is not an identity: on the laptop
Ollama runs as the `ollama` user, so the registry's user is talking to another
user's server, and while that server is down any local user can bind
127.0.0.1:11434. Every call holds the registry's exclusive flock while it talks to
that peer.

S3: http.client's timeout is per socket operation. A peer that sends one byte
    every 0.5 s keeps `status`, `register` and `ensure_room` inside the lock for as
    long as it likes, and every cooperating app's call fails "busy" after its wait.
S4: a deeply nested JSON answer raises RecursionError, which ollama._request does
    not catch (it catches ValueError and IndexError), so it escapes as a non-
    ResidencyError from status() and a traceback from `kilix-gpu status`.
S5: an unregistered Ollama model's vram_bytes is Ollama's self-reported
    size_vram, not a measurement. ensure_room plans with it, so when the claim is
    larger than what Ollama's processes hold on the GPU, the all-or-nothing check
    passes, a model is unloaded, and the call then fails anyway. The README says
    per-resident VRAM is "measured on every read and never stored or declared".
"""
from __future__ import annotations

import io
import threading
import time

import residency_containment  # noqa: F401
import kilix_gpu_residency as residency
from kilix_gpu_residency import cli, ollama
from fakes import GIB, MIB, ResidencyCase
from review_common import Peer

TRICKLE_SECONDS = 10.0


class _Peer(ResidencyCase):
    def peer(self, mode: str, **kwargs) -> Peer:
        peer = Peer(mode, **kwargs)
        self.addCleanup(peer.close)
        return peer


# S3 -------------------------------------------------------------------------

class _Trickle(_Peer):
    def held_lock(self):
        """(seconds status() took, the error a second caller got meanwhile)."""
        # Adapted for fix round 1: status() is a reader now (a shared lock), so the call that talks to
        # the peer is a writer's, ensure_room, and the other app registers (a writer too).
        peer = self.peer("trickle", seconds=TRICKLE_SECONDS)
        self.prune()  # create the registry first
        second: dict = {}

        def other_app():
            time.sleep(1.0)
            try:
                residency.register(app="other", model="m", backend="process", namespace=self.namespace,
                                   ollama=self.ollama.url, wait=2 * ollama._TIMEOUT + 2.0).close(unload=False)
                second["error"] = None
            except residency.ResidencyError as error:
                second["error"] = error
        thread = threading.Thread(target=other_app)
        thread.start()
        started = time.monotonic()
        residency.ensure_room(1, device=None, requester="test", namespace=self.namespace, ollama=peer.url)
        value = residency.status(namespace=self.namespace, ollama=peer.url)
        elapsed = time.monotonic() - started
        thread.join()
        # The branch under test: the request reached the peer, which answered nothing usable.
        self.assertTrue(peer.requests and peer.requests[0].startswith(b"GET /api/ps "))
        self.assertEqual(value.ollama, "unreachable")
        self.assertIsNone(second["error"])
        return elapsed, second["error"]


class RequiredTrickleHoldsLock(_Trickle):
    def test_an_ollama_exchange_is_bounded_by_its_timeout(self):
        elapsed, error = self.held_lock()
        self.assertLess(elapsed, 2 * (2 * ollama._TIMEOUT) + 1.0,  # two calls, each bounded
                        f"S3: a loopback peer trickling one byte per 0.5 s kept status() inside the registry lock "
                        f"for {elapsed:.1f} s (ollama._TIMEOUT is {ollama._TIMEOUT} s); another app meanwhile got "
                        f"{error!r}")


# S4 -------------------------------------------------------------------------


class RequiredNestedAnswer(_Peer):
    def test_a_malformed_answer_means_unreachable(self):
        peer = self.peer("nested")
        try:
            value = residency.status(namespace=self.namespace, ollama=peer.url)
        except BaseException as error:  # noqa: BLE001 - the finding is what escapes
            self.fail(f"S4: a 200 kB nested-array answer from the Ollama port made status() raise "
                      f"{type(error).__name__}: {error}; expected ollama='unreachable' (or a ResidencyError)")
        self.assertEqual(value.ollama, "unreachable")


# S5 -------------------------------------------------------------------------

class _SelfReport(_Peer):
    def overclaim(self):
        """Ollama lists phantom:7b at 5 GiB of VRAM; its runner measurably holds 512 MiB."""
        self.ollama.load("phantom:7b", 5 * GIB)
        runner = self.ollama.runner_pid
        self.ollama.held = lambda: {runner: 512 * MIB} if self.ollama.models else {}
        [info] = self.status().residents
        device = next(d for d in self.status(wake=True).devices if d.pci == "0000:01:00.0")
        measured = device.processes.get(runner)
        # The branch under test: declared and measured differ, and Ollama is attributed the device.
        # Adapted for fix round 1 (FX-19): status shows the measured figure, so the declared one is read
        # from Ollama's own list.
        [declared] = ollama.loaded(self.ollama.url)
        self.assertEqual((declared.vram_bytes, info.vram_bytes, measured, info.device_pci),
                         (5 * GIB, 512 * MIB, 512 * MIB, "0000:01:00.0"))
        need = device.total_bytes - 50 * MIB  # more than unloading every Ollama byte could ever free
        try:
            residency.ensure_room(need, device="Vulkan1", requester="bonsai", **self.where())
        except residency.RoomError as error:
            return error, measured, device
        self.fail("ensure_room unexpectedly made room")


class RequiredSelfReportedVram(_SelfReport):
    def test_all_or_nothing_uses_measured_vram(self):
        error, measured, device = self.overclaim()
        self.assertEqual(error.evicted, (),
                         f"S5: ensure_room unloaded {[e.model for e in error.evicted]} counting Ollama's declared "
                         f"size_vram 5 GiB although its processes measurably hold {measured // MIB} MiB; afterwards "
                         f"free is {error.free_bytes} < need {error.need_bytes}; "
                         f"README: 'If even every candidate would not be enough, it raises RoomError and unloads "
                         f"nothing' and 'Per-resident VRAM is measured on every read and never ... declared'")
