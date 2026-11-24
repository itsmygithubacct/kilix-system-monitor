"""Adopted from the security seat's P1 review controls (review-p1/security/controls/test_sec_probes.py).
Only the Required controls are kept; the Observed ones recorded the defects this round fixed.

Probes that found nothing. Each is an Observed control that passes; together with
the component's own suite they cover the obvious attack paths that hold.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import residency_containment  # noqa: F401
import kilix_gpu_residency as residency
from kilix_gpu_residency import ollama
from fakes import GIB, ResidencyCase
from review_common import Peer


class ProbeNamespace(ResidencyCase):
    def refused(self, namespace: str) -> residency.ResidencyError:
        with self.assertRaises(residency.ResidencyError) as caught:
            residency.status(namespace=namespace, ollama=self.ollama.url)
        return caught.exception

    def test_a_symlinked_intermediate_ancestor_is_refused(self):
        real = self.scratch / "real"
        real.mkdir(mode=0o700)
        (self.scratch / "link").symlink_to(real)
        self.assertEqual(self.refused(str(self.scratch / "link" / "registry")).code, "unavailable")
        self.assertEqual(os.listdir(real), [])

    def test_a_user_owned_world_writable_sticky_ancestor_is_refused(self):
        outer = self.scratch / "outer"
        outer.mkdir()
        os.chmod(outer, 0o1777)
        inner = outer / "inner"
        inner.mkdir(mode=0o700)
        self.assertEqual(self.refused(str(inner / "registry")).code, "unavailable")

    def test_a_fifo_planted_as_a_record_is_refused_without_blocking(self):
        self.prune()  # adapted for fix round 1: status() no longer creates the registry
        os.mkfifo(Path(self.namespace) / ("0" * 32 + ".json"), 0o600)
        started = time.monotonic()
        self.assertEqual(self.refused(self.namespace).code, "unavailable")
        self.assertLess(time.monotonic() - started, 5)

    def test_a_forged_signal_record_that_does_not_lead_its_group_is_refused(self):
        resident, process = self.model("puzzllm", "m", GIB)
        path = Path(self.namespace) / (resident.id + ".json")
        record = json.loads(path.read_text())
        record["unload"]["pgid"] = record["pgid"] = os.getpgrp()  # holder.pid stays the model's
        path.write_text(json.dumps(record))
        os.chmod(path, 0o600)
        self.assertEqual(self.refused(self.namespace).code, "unavailable")
        self.assertIsNone(process.poll())


class ProbeModelNames(ResidencyCase):
    def test_escape_sequences_in_registered_names_are_refused(self):
        for model in ("\x1b[2J", "a\x9bb", "x\ny"):
            with self.subTest(model=model), self.assertRaises(residency.ResidencyError) as caught:
                residency.register(app="a", model=model, backend="process", **self.where())
            self.assertEqual(caught.exception.code, "invalid-request")

    def test_escape_sequences_in_ollama_names_are_dropped(self):
        self.ollama.models["\x1b]52;c;aGk=\x07"] = {"size": GIB, "size_vram": GIB}
        self.assertEqual(self.status().residents, ())


class ProbeOllamaReach(ResidencyCase):
    def test_tricky_hosts_are_not_loopback(self):
        for url in ("http://127.0.0.1:11434@192.0.2.1", "http://127.1:11434", "http://0.0.0.0:11434",
                    "http://[::ffff:192.0.2.1]:11434", "http://localhost.:11434", "http://127.0.0.1:11434/x",
                    "http://user@192.0.2.1:11434"):
            with self.subTest(url=url):
                self.assertIsNone(ollama.loopback(url))
        os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434@192.0.2.1"
        self.assertIsNone(ollama.loopback(ollama.default_url()))

    def test_a_redirect_is_not_followed(self):
        peer = Peer("redirect")
        self.addCleanup(peer.close)
        self.assertEqual(residency.status(namespace=self.namespace, ollama=peer.url).ollama, "unreachable")
        self.assertEqual(len(peer.requests), 1)

    def test_proxy_variables_are_not_used(self):
        for key in ("http_proxy", "HTTP_PROXY", "all_proxy"):
            os.environ[key] = "http://192.0.2.1:3128"
            self.addCleanup(os.environ.pop, key, None)
        self.ollama.load("m:1b", GIB)
        self.assertEqual(self.status().ollama, "reachable")


class ProbeInterruption(ResidencyCase):
    def test_an_interrupt_during_an_unload_releases_the_lock_and_does_not_leave_it_evictable(self):
        resident, process = self.model("puzzllm", "m", GIB, ignore_term=True)
        original = residency._wait_gone

        def interrupted(*_):
            raise KeyboardInterrupt
        residency._wait_gone = interrupted
        self.addCleanup(setattr, residency, "_wait_gone", original)
        with self.assertRaises(KeyboardInterrupt):
            residency.unload(resident.id, **self.where())
        residency._wait_gone = original
        [info] = residency.status(**self.where(), wait=1.0).residents  # the lock was released
        self.assertEqual(info.state, "busy")  # not idle, so not a candidate
