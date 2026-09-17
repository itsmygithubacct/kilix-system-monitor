"""A caller's own callback failure reaches the caller unchanged, never as a lease refusal."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import lease_containment
import kilix_device_lease as leases

# After the resource flock, a cancelled callback sleeps. That used to be inside
# the registry lock, so a second requester waited out the sleep.
BLOCKING_AFTER_RESOURCE = r"""
import fcntl, json, os, sys, time
import kilix_device_lease as leases
namespace, block, marker = sys.argv[1], float(sys.argv[2]), sys.argv[3]
real_flock = fcntl.flock
held = {"resource": False, "blocked_s": 0.0}

def flock(fd, operation):
    real_flock(fd, operation)
    try:
        held["resource"] = os.readlink("/proc/self/fd/%s" % fd).endswith("/accelerator.lock")
    except OSError:
        held["resource"] = False

fcntl.flock = flock

def cancelled():
    if held["resource"]:
        began = time.monotonic()
        time.sleep(block)
        held["blocked_s"] += time.monotonic() - began
        held["resource"] = False
    return False

began = time.monotonic()
try:
    lease = leases.acquire(job_id="blocker", workload="tts-utterance", device="dev",
                           deadline=time.monotonic() + 30, namespace=namespace, cancelled=cancelled,
                           progress=lambda status: print(json.dumps({"queued": status.position}), flush=True))
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print(json.dumps({"result": "overlap", "elapsed_s": round(time.monotonic() - began, 2),
                          "blocked_s": round(held["blocked_s"], 2)}), flush=True)
        raise SystemExit(3)
    os.close(fd)
    os.unlink(marker)
    lease.release(cleanup_complete=True)
    print(json.dumps({"result": "granted", "elapsed_s": round(time.monotonic() - began, 2),
                      "blocked_s": round(held["blocked_s"], 2)}), flush=True)
except leases.LeaseError as error:
    print(json.dumps({"result": error.code, "elapsed_s": round(time.monotonic() - began, 2),
                      "blocked_s": round(held["blocked_s"], 2)}), flush=True)
"""

WAITER = r"""
import json, os, sys, time
from kilix_device_lease import LeaseError, acquire
namespace, marker, seconds = sys.argv[1:4]
began = time.monotonic()
try:
    lease = acquire(job_id="waiter", workload="stt-job", device="dev2",
                    deadline=time.monotonic() + float(seconds), namespace=namespace,
                    progress=lambda status: print(json.dumps({"queued": status.position}), flush=True))
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print(json.dumps({"result": "overlap", "elapsed_s": round(time.monotonic() - began, 2)}), flush=True)
        raise SystemExit(3)
    os.close(fd)
    os.unlink(marker)
    lease.release(cleanup_complete=True)
    print(json.dumps({"result": "granted", "elapsed_s": round(time.monotonic() - began, 2)}), flush=True)
except LeaseError as error:
    print(json.dumps({"result": error.code, "elapsed_s": round(time.monotonic() - began, 2)}), flush=True)
"""


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

    def test_blocking_callback_does_not_stall_other_requesters(self):
        env = lease_containment.child_env()
        marker = self.temp.name + "/critical"
        holder = self.acquire(job_id="holder", workload="llm-turn", deadline=time.monotonic() + 60)
        children = []

        def stop():
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
                for stream in (child.stdout, child.stderr):
                    if stream and not stream.closed:
                        stream.close()

        self.addCleanup(stop)
        try:
            blocker = subprocess.Popen(
                [sys.executable, "-c", BLOCKING_AFTER_RESOURCE, self.namespace, "3.0", marker],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                bufsize=1, env=env)
            children.append(blocker)
            self.assertEqual(json.loads(blocker.stdout.readline()), {"queued": 1})
            waiter = subprocess.Popen(
                [sys.executable, "-c", WAITER, self.namespace, marker, "6"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                bufsize=1, env=env)
            children.append(waiter)
            self.assertEqual(json.loads(waiter.stdout.readline()), {"queued": 2})
            holder.release(cleanup_complete=True)
            holder = None
            rows = []
            for child in (blocker, waiter):
                out, err = child.communicate(timeout=20)
                self.assertEqual(child.returncode, 0, err)
                rows.append(json.loads(out.splitlines()[-1]))
        finally:
            if holder is not None:
                holder.release(cleanup_complete=True)
        blocker_row, waiter_row = rows
        self.assertEqual(blocker_row["result"], "granted", blocker_row)
        self.assertEqual(waiter_row["result"], "granted", waiter_row)
        self.assertEqual(blocker_row["blocked_s"], 0.0, blocker_row)
        self.assertLess(waiter_row["elapsed_s"], 1.5, waiter_row)
        self.assertFalse(Path(marker).exists())
