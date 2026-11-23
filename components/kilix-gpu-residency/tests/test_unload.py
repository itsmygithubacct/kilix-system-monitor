"""Unloading: process groups get TERM then KILL, only while their registered leader lives."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal

import subprocess
import sys

import residency_containment
import kilix_gpu_residency as residency
from fakes import GIB, IGNORES_TERM, ResidencyCase, stop, wait_for


class SignalUnloadTests(ResidencyCase):
    def test_term_ends_the_group_and_the_unload_is_recorded(self):
        resident, process = self.model("puzzllm", "m", GIB)
        member = self.join(process, GIB)
        event = residency.unload(resident.id, **self.where())
        self.assertEqual((event.id, event.model, event.app, event.by, event.reason, event.freed_bytes),
                         (resident.id, "m", "puzzllm", "kilix-gpu", "manual", 2 * GIB))
        self.assertEqual(self.reap(process), -signal.SIGTERM)
        self.assertEqual(self.reap(member), -signal.SIGTERM)
        self.assertEqual(self.records(), [])
        self.assertEqual([e.id for e in self.status().events], [resident.id])

    def test_a_group_that_ignores_term_is_killed_after_the_grace_period(self):
        residency.TERM_GRACE_SECONDS = 0.3
        resident, process = self.model("puzzllm", "stubborn", GIB, ignore_term=True)
        residency.unload("stubborn", **self.where())
        self.assertEqual(self.reap(process), -signal.SIGKILL)

    def test_members_that_outlive_their_leader_are_killed_too(self):
        residency.TERM_GRACE_SECONDS = 0.5
        resident, process = self.model("puzzllm", "m", GIB)
        member = subprocess.Popen([sys.executable, "-c", IGNORES_TERM], stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, text=True, process_group=process.pid,
                                  env=residency_containment.child_env())
        self.addCleanup(stop, member)
        self.assertEqual(member.stdout.readline().strip(), "ready")
        residency.unload(resident.id, **self.where())
        self.assertEqual(self.reap(process), -signal.SIGTERM)
        self.assertEqual(self.reap(member), -signal.SIGKILL)

    def test_a_leader_whose_start_time_changed_is_never_signalled(self):
        resident, process = self.model("puzzllm", "m", GIB)
        record = self.records()[0]
        record["holder"]["start"] += 1
        with self.assertRaises(residency.ResidencyError) as caught:
            residency._signal_unload(record)
        self.assertEqual(caught.exception.code, "refused")
        self.assertIsNone(process.poll())

    def test_a_signal_is_never_sent_to_a_group_the_record_does_not_lead(self):
        _, process = self.model("puzzllm", "m", GIB)
        record = self.records()[0]
        for pgid in (os.getpgrp(), 1):
            forged = json.loads(json.dumps(record))
            forged["unload"]["pgid"] = forged["pgid"] = forged["holder"]["pid"] = pgid
            forged["holder"]["start"] = residency._start(pgid) or 0  # a live leader: only the rule refuses
            with self.subTest(pgid=pgid), self.assertRaises(residency.ResidencyError) as caught:
                residency._signal_unload(forged)
            self.assertEqual(caught.exception.code, "refused")
        self.assertIsNone(process.poll())

    def test_busy_loading_and_pinned_residents_need_force(self):
        for state, pinned in (("busy", False), ("loading", False), ("idle", True)):
            with self.subTest(state=state, pinned=pinned):
                resident, process = self.model("puzzllm", f"{state}-{pinned}", GIB, state=state, pinned=pinned)
                with self.assertRaises(residency.ResidencyError) as caught:
                    residency.unload(resident.id, **self.where())
                self.assertEqual(caught.exception.code, "refused")
                self.assertIsNone(process.poll())
                residency.unload(resident.id, force=True, **self.where())
                self.assertEqual(self.reap(process), -signal.SIGTERM)

    def test_a_resident_without_an_unload_action_is_never_unloaded(self):
        resident = residency.register(app="kilix-chat", model="in-process", backend="process", **self.where())
        resident.idle()
        with self.assertRaises(residency.ResidencyError) as caught:
            residency.unload("in-process", force=True, **self.where())
        self.assertEqual(caught.exception.code, "refused")
        self.assertEqual(len(self.records()), 1)

    def test_the_unload_log_keeps_only_the_newest_events(self):
        self.addCleanup(setattr, residency, "MAX_EVENTS", residency.MAX_EVENTS)
        residency.MAX_EVENTS = 2
        for name in ("one", "two", "three"):
            resident, _ = self.model("puzzllm", name, GIB)
            residency.unload(resident.id, **self.where())
        self.assertEqual([e.model for e in self.status().events], ["two", "three"])

    def test_unknown_and_ambiguous_targets(self):
        self.model("puzzllm", "twin", GIB)
        self.model("bonsai", "twin", GIB)
        with self.assertRaises(residency.ResidencyError) as caught:
            residency.unload("twin", **self.where())
        self.assertEqual(caught.exception.code, "invalid-request")
        with self.assertRaises(residency.ResidencyError) as caught:
            residency.unload("nothing-like-it", **self.where())
        self.assertEqual(caught.exception.code, "not-found")
        self.assertEqual(len(self.records()), 2)

    def test_a_record_being_unloaded_says_so(self):
        resident, process = self.model("puzzllm", "m", GIB, ignore_term=True)
        residency.TERM_GRACE_SECONDS = 1.0
        seen = []
        original = residency._wait_gone

        def watching(pgid, leader, seconds):
            seen.append(json.loads((Path(self.namespace) / (resident.id + ".json")).read_text())["state"])
            return original(pgid, leader, seconds)
        residency._wait_gone = watching
        self.addCleanup(setattr, residency, "_wait_gone", original)
        residency.unload(resident.id, **self.where())
        self.assertEqual(seen[0], "unloading")
        self.assertTrue(wait_for(lambda: process.poll() is not None))


if __name__ == "__main__":
    import unittest
    unittest.main()
