"""Ollama models are found through /api/ps whoever loaded them, and unloaded with keep_alive 0."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import time

import residency_containment  # noqa: F401
import kilix_gpu_residency as residency
from kilix_gpu_residency import ollama
from fakes import GIB, MIB, FakeOllama, ResidencyCase


class DiscoveryTests(ResidencyCase):
    def test_models_nobody_registered_are_listed_with_ollamas_own_vram_figure(self):
        self.ollama.load("bonsai-2:27b", 5 * GIB)
        status = self.status()
        self.assertEqual(status.ollama, "reachable")
        [found] = status.residents
        self.assertEqual((found.id, found.app, found.model, found.backend, found.registered, found.unload_kind),
                         ("ollama:bonsai-2:27b", "ollama", "bonsai-2:27b", "ollama", False, "ollama"))
        self.assertEqual((found.vram_bytes, found.state, found.pinned), (5 * GIB, "idle", False))
        # Attributed to the GPU where the Ollama runner holds memory; the runner is not unmanaged.
        self.assertEqual(found.device_pci, "0000:01:00.0")
        self.assertEqual(status.unmanaged, ())
        self.assertEqual(self.ollama.requests[0][:2], ("GET", "/api/ps"))

    def test_last_use_is_estimated_from_the_keep_alive_expiry(self):
        self.ollama.load("a", GIB, expires_in=60)
        self.ollama.load("b", GIB, expires_in=240)
        by_name = {r.model: r for r in self.status().residents}
        self.assertLess(by_name["a"].last_used, by_name["b"].last_used)
        self.assertAlmostEqual(by_name["b"].last_used, time.time() - 60, delta=5)

    def test_an_unreachable_ollama_leaves_its_runner_unmanaged(self):
        self.ollama.load("bonsai-2:27b", 5 * GIB)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead = f"http://127.0.0.1:{probe.getsockname()[1]}"
        status = residency.status(namespace=self.namespace, ollama=dead)
        self.assertEqual((status.ollama, status.residents), ("unreachable", ()))
        self.assertEqual([(u.pid, u.command, u.vram_bytes) for u in status.unmanaged],
                         [(self.ollama.runner_pid, "llama-server", 5 * GIB)])

    def test_a_remote_ollama_is_never_contacted(self):
        for url in ("http://192.0.2.1:11434", "https://127.0.0.1:11434", "http://127.0.0.1:1/api",
                    "http://example.invalid:11434"):
            with self.subTest(url=url):
                self.assertIsNone(ollama.loopback(url))
                status = residency.status(namespace=self.namespace, ollama=url)
                self.assertEqual(status.ollama, "not-loopback")
        for url in ("http://127.0.0.1:11434", "http://localhost:8080", "http://[::1]:11434", "http://127.0.0.2"):
            with self.subTest(url=url):
                self.assertIsNotNone(ollama.loopback(url))

    def test_the_default_url_follows_ollama_host(self):
        self.assertEqual(ollama.default_url(), "http://127.0.0.1:11434")
        import os
        # As Ollama's own client reads it: the unspecified addresses a server binds to mean loopback.
        for value, expected in (("127.0.0.1:9999", "http://127.0.0.1:9999"), ("http://localhost", "http://localhost:11434"),
                                ("0.0.0.0", "http://127.0.0.1:11434"), ("0.0.0.0:8080", "http://127.0.0.1:8080"),
                                ("[::]:11434", "http://[::1]:11434"), ("[::1]:11434", "http://[::1]:11434"),
                                ("http://[::1]", "http://[::1]:11434"), ("ip6-localhost:11434", "http://ip6-localhost:11434")):
            with self.subTest(value=value):
                os.environ["OLLAMA_HOST"] = value
                self.assertEqual(ollama.default_url(), expected)
        # Only "localhost" is trusted by name: another name that resolves to loopback is not.
        self.assertIsNone(ollama.loopback("http://ip6-localhost:11434"))
        self.assertIsNotNone(ollama.loopback("http://[::1]:11434"))
        for bad in ("127.0.0.1:99999", "127.0.0.1:port"):
            with self.subTest(bad=bad):
                os.environ["OLLAMA_HOST"] = bad
                with self.assertRaises(ollama.OllamaError):
                    ollama.default_url()
                with self.assertRaises(residency.ResidencyError) as caught:
                    residency.status(namespace=self.namespace)
                self.assertEqual(caught.exception.code, "invalid-request")

    def test_a_bind_all_ollama_host_is_discovered(self):
        import os
        self.ollama.load("bonsai-2:27b", 5 * GIB)
        os.environ["OLLAMA_HOST"] = "0.0.0.0:" + self.ollama.url.rsplit(":", 1)[1]
        status = residency.status(namespace=self.namespace)
        self.assertEqual((status.ollama, [r.model for r in status.residents]), ("reachable", ["bonsai-2:27b"]))

    def test_keep_alive_is_read_as_ollama_reads_it(self):
        import os
        self.addCleanup(os.environ.pop, "OLLAMA_KEEP_ALIVE", None)
        for value, seconds in (("", 300.0), ("600", 600.0), ("1h", 3600.0), ("1h30m", 5400.0), ("90s", 90.0),
                               ("1.5m", 90.0), ("-1", None), ("-1m", None), ("forever", None), ("5x", None)):
            with self.subTest(value=value):
                os.environ["OLLAMA_KEEP_ALIVE"] = value
                self.assertEqual(ollama.keep_alive_seconds(), seconds)

    def test_a_long_keep_alive_orders_models_by_their_real_last_use(self):
        import os
        os.environ["OLLAMA_KEEP_ALIVE"] = "1h"
        self.addCleanup(os.environ.pop, "OLLAMA_KEEP_ALIVE", None)
        self.ollama.load("stale", GIB, expires_in=600)  # used 50 minutes ago
        self.ollama.load("fresh", GIB, expires_in=3500)  # used 100 seconds ago
        by_name = {r.model: r for r in self.status().residents}
        self.assertAlmostEqual(by_name["stale"].last_used, time.time() - 3000, delta=5)
        self.assertLess(by_name["stale"].last_used, by_name["fresh"].last_used)
        # A 5 minute default would put "fresh" in the future; the estimate never passes now.
        os.environ.pop("OLLAMA_KEEP_ALIVE")
        by_name = {r.model: r for r in self.status().residents}
        self.assertLessEqual(by_name["fresh"].last_used, time.time())

    def test_malformed_entries_are_skipped(self):
        self.ollama.models.update({"bad name": {"size": 1, "size_vram": 1}, "neg": {"size": -1, "size_vram": 1},
                                   "str": {"size": "1", "size_vram": 1}})
        self.ollama.load("good:1b", GIB)
        self.assertEqual([r.model for r in self.status().residents], ["good:1b"])

    def test_expiry_times_parse_and_far_future_means_unknown(self):
        now = time.time()
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now + 100))
        self.assertAlmostEqual(ollama._expiry(stamp + ".123456789Z"), now + 100, delta=2)
        self.assertAlmostEqual(ollama._expiry(stamp + "+00:00"), now + 100, delta=2)
        self.assertIsNone(ollama._expiry("2318-01-01T00:00:00.0Z"))
        for bad in ("", None, 5, "yesterday", "2026-13-01T00:00:00Z", stamp):  # no zone: refused
            self.assertIsNone(ollama._expiry(bad))


class LegacyRunnerTests(ResidencyCase):
    """Before 0.40 the runner itself was named ollama (or ollama_llama_server)."""

    def setUp(self):
        super().setUp()
        self.ollama.close()
        self.ollama = FakeOllama(self.scratch / "legacy", legacy=True)
        self.addCleanup(self.ollama.close)

    def test_a_runner_named_ollama_is_ollamas(self):
        self.ollama.load("bonsai-2:27b", 5 * GIB)
        status = self.status()
        self.assertEqual([(r.model, r.device_pci) for r in status.residents], [("bonsai-2:27b", "0000:01:00.0")])
        self.assertEqual(status.unmanaged, ())


class OllamaUnloadTests(ResidencyCase):
    def test_unload_sends_keep_alive_zero_and_waits_until_it_is_gone(self):
        self.ollama.load("bonsai-2:27b", 5 * GIB)
        event = residency.unload("ollama:bonsai-2:27b", **self.where())
        self.assertEqual((event.model, event.app, event.freed_bytes), ("bonsai-2:27b", "ollama", 5 * GIB))
        posts = [body for method, path, body in self.ollama.requests if method == "POST"]
        self.assertEqual(posts, [{"model": "bonsai-2:27b", "keep_alive": 0}])
        self.assertEqual(self.ollama.models, {})

    def test_a_model_ollama_will_not_unload_times_out(self):
        self.ollama.load("sticky", GIB)
        self.ollama.ignore_unload = True
        with self.assertRaises(ollama.OllamaError):
            ollama.unload(self.ollama.url, "sticky", timeout=0.3)
        with self.assertRaises(ollama.OllamaError):
            ollama.unload(self.ollama.url, "not a name")

    def test_pinning_a_discovered_model_is_remembered(self):
        self.ollama.load("bonsai-2:27b", 5 * GIB)
        residency.pin("bonsai-2:27b", **self.where())
        pins = json.loads((Path(self.namespace) / "pins.json").read_text())
        self.assertEqual(pins, {"version": residency.VERSION, "ollama": ["bonsai-2:27b"]})
        self.assertTrue(self.status().residents[0].pinned)
        with self.assertRaises(residency.ResidencyError) as caught:
            residency.unload("bonsai-2:27b", **self.where())
        self.assertEqual(caught.exception.code, "refused")
        residency.unpin("bonsai-2:27b", **self.where())
        self.assertFalse(self.status().residents[0].pinned)


class RegisteredOllamaTests(ResidencyCase):
    def test_a_registered_model_is_merged_with_ollamas_entry(self):
        resident = residency.register(app="kilix-chat", model="qwen3.5:4b", backend="ollama", **self.where())
        [record] = self.records()
        self.assertEqual(record["unload"], {"kind": "ollama", "model": "qwen3.5:4b"})
        self.ollama.load("qwen3.5:4b", 3 * GIB)
        resident.idle()
        [info] = self.status().residents
        self.assertEqual((info.id, info.app, info.registered, info.vram_bytes, info.device_pci),
                         (resident.id, "kilix-chat", True, 3 * GIB, "0000:01:00.0"))

    def test_a_registered_model_ollama_dropped_is_pruned_unless_still_loading(self):
        resident = residency.register(app="kilix-chat", model="qwen3.5:4b", backend="ollama", **self.where())
        self.assertEqual(len(self.status().residents), 1)  # loading: not in /api/ps yet
        resident.idle()
        self.assertEqual(self.status().residents, ())  # keep_alive expired
        self.prune()
        self.assertEqual(self.records(), [])

    def test_an_unreachable_ollama_does_not_prune_registered_models(self):
        resident = residency.register(app="kilix-chat", model="qwen3.5:4b", backend="ollama", **self.where())
        resident.idle()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead = f"http://127.0.0.1:{probe.getsockname()[1]}"
        status = residency.status(namespace=self.namespace, ollama=dead)
        self.assertEqual([r.id for r in status.residents], [resident.id])
        self.assertEqual(len(self.records()), 1)

    def test_leaving_the_block_unloads_through_ollama(self):
        with residency.register(app="kilix-chat", model="qwen3.5:4b", backend="ollama", **self.where()) as r:
            self.ollama.load("qwen3.5:4b", 3 * GIB)
            r.idle()
        self.assertEqual(self.ollama.unloads(), ["qwen3.5:4b"])
        self.assertEqual(self.records(), [])

    def test_a_registered_model_is_evicted_by_room_like_any_other(self):
        resident = residency.register(app="kilix-chat", model="qwen3.5:4b", backend="ollama", **self.where())
        self.ollama.load("qwen3.5:4b", 4 * GIB)
        resident.idle()
        room = residency.ensure_room(3 * GIB, device="Vulkan1", requester="puzzllm", **self.where())
        self.assertEqual([e.id for e in room.evicted], [resident.id])
        self.assertEqual(resident.evicted.by, "puzzllm")
        self.assertGreaterEqual(room.free_bytes, 3 * GIB - 0 * MIB)


if __name__ == "__main__":
    import unittest
    unittest.main()
