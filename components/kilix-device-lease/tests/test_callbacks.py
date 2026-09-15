"""A caller's own callback failure reaches the caller unchanged, never as a lease refusal."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import lease_containment  # noqa: F401
import kilix_device_lease as leases


class CallbackBug(OSError):
    """An OSError subclass, because the module translates OSError into lease codes."""


class CallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="device-lease-callback-")
        self.addCleanup(self.temp.cleanup)
        self.namespace = self.temp.name + "/leases"

    def acquire(self, **kwargs):
        request = dict(job_id="callback-job", workload="tts-utterance", device="callback-device",
                       deadline=time.monotonic() + 2, namespace=self.namespace)
        request.update(kwargs)
        return leases.acquire(**request)

    def queue(self):
        return json.loads((Path(self.namespace) / "state.json").read_text(encoding="utf-8"))["queue"]

    @staticmethod
    def failing_on(call, error):
        calls = {"count": 0}

        def callback(*_args):
            calls["count"] += 1
            if calls["count"] == call:
                raise error
            return False

        return callback, calls

    def outcome(self, call, **kwargs):
        try:
            call(**kwargs)
        except leases.LeaseError as refused:
            return refused
        except BaseException as raised:  # noqa: BLE001 - the test inspects exactly what escaped
            return raised
        return None

    def test_raising_request_callbacks_escape_unchanged_at_every_call(self):
        holder = self.acquire(job_id="holder", workload="llm-turn", deadline=time.monotonic() + 60)
        try:
            for name in ("cancelled", "disconnected"):
                for kind in (ValueError, CallbackBug):
                    for call in range(1, 7):
                        with self.subTest(callback=name, error=kind.__name__, call=call):
                            error = kind("caller bug")
                            callback, calls = self.failing_on(call, error)
                            before = len(os.listdir("/proc/self/fd"))
                            escaped = self.outcome(self.acquire, **{name: callback})
                            self.assertGreaterEqual(calls["count"], call)
                            self.assertIs(escaped, error)
                            self.assertEqual(len(os.listdir("/proc/self/fd")), before)
                            self.assertEqual(self.queue(), [])
        finally:
            holder.release(cleanup_complete=True)

    def test_raising_progress_callback_escapes_unchanged(self):
        holder = self.acquire(job_id="holder", workload="llm-turn", deadline=time.monotonic() + 60)
        try:
            for kind in (ValueError, CallbackBug):
                with self.subTest(error=kind.__name__):
                    error = kind("caller bug")
                    progress, calls = self.failing_on(1, error)
                    escaped = self.outcome(self.acquire, progress=progress)
                    self.assertEqual(calls["count"], 1)
                    self.assertIs(escaped, error)
                    self.assertEqual(self.queue(), [])
        finally:
            holder.release(cleanup_complete=True)

    def test_raising_callback_during_check_escapes_unchanged(self):
        # check() consults the callback before opening the registry and again
        # while it waits for the registry lock.
        for kind in (ValueError, CallbackBug):
            for call in (1, 2):
                with self.subTest(error=kind.__name__, call=call):
                    error = kind("caller bug")
                    armed = {"on": False}
                    callback, calls = self.failing_on(call, error)
                    lease = self.acquire(cancelled=lambda: armed["on"] and callback())
                    try:
                        armed["on"] = True
                        escaped = self.outcome(lease.check)
                        self.assertEqual(calls["count"], call)
                        self.assertIs(escaped, error)
                    finally:
                        armed["on"] = False
                        lease.release(cleanup_complete=True)
