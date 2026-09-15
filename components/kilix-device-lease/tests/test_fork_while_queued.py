"""A process forked from a queued requester never keeps that request alive."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import lease_containment
import kilix_device_lease as leases

# A queued requester forks a worker without exec from its progress callback.
# The worker only sleeps; it never touches the lease API. The requester then
# waits to be killed, as a crashed provider would be.
FORKING_REQUESTER = r"""
import json, os, sys, time
import kilix_device_lease as leases
forked = {"pid": None}
def progress(status):
    if forked["pid"] is None:
        pid = os.fork()
        if pid == 0:
            time.sleep(60)
            os._exit(0)
        forked["pid"] = pid
        print(json.dumps({"worker": pid}), flush=True)
leases.acquire(job_id="forker", workload="tts-utterance", device="d", deadline=time.monotonic() + 60,
               namespace=sys.argv[1], progress=progress)
"""

# The child of a fork carries on inside the parent's acquire call.
CONTINUING_CHILD = r"""
import json, os, sys, time
import kilix_device_lease as leases
forked = {"done": False}
def progress(status):
    if not forked["done"]:
        forked["done"] = True
        if os.fork() == 0:
            forked["child"] = True
def report(outcome):
    print(json.dumps({"role": "child" if forked.get("child") else "parent", "outcome": outcome}), flush=True)
try:
    lease = leases.acquire(job_id="forker", workload="tts-utterance", device="d",
                           deadline=time.monotonic() + 20, namespace=sys.argv[1], progress=progress)
except leases.LeaseError as error:
    report(error.code)
else:
    report("granted")
    lease.release(cleanup_complete=True)
if forked.get("child"):
    os._exit(0)
"""


class ForkWhileQueuedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="device-lease-fork-")
        self.addCleanup(self.temp.cleanup)
        self.namespace = self.temp.name + "/leases"
        self.env = lease_containment.child_env()

    def acquire(self, **kwargs):
        request = dict(job_id="holder", workload="llm-turn", device="d",
                       deadline=time.monotonic() + 30, namespace=self.namespace)
        request.update(kwargs)
        return leases.acquire(**request)

    def spawn(self, code):
        child = subprocess.Popen([sys.executable, "-c", code, self.namespace], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env)

        def stop():
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdout.close()
            child.stderr.close()

        self.addCleanup(stop)
        return child

    def line(self, child, timeout=10):
        self.assertTrue(select.select([child.stdout], [], [], timeout)[0], "no line from child")
        return json.loads(child.stdout.readline())

    def test_dead_requesters_forked_worker_does_not_hold_its_queue_place(self):
        holder = self.acquire()
        requester = self.spawn(FORKING_REQUESTER)
        worker = self.line(requester)["worker"]
        self.addCleanup(lambda: os.kill(worker, signal.SIGKILL) if os.path.exists(f"/proc/{worker}") else None)
        requester.kill()
        requester.wait(timeout=5)
        holder.release(cleanup_complete=True)
        self.assertTrue(os.path.exists(f"/proc/{worker}"), "the forked worker must still be alive")
        self.acquire(job_id="successor", workload="stt-job", deadline=time.monotonic() + 1).release(
            cleanup_complete=True)
        state = json.loads((Path(self.namespace) / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["queue"], [])

    def test_forked_child_cannot_carry_on_with_its_parents_request(self):
        holder = self.acquire()
        requester = self.spawn(CONTINUING_CHILD)
        child = self.line(requester)
        self.assertEqual(child, {"role": "child", "outcome": "lost-lease"})
        holder.release(cleanup_complete=True)
        self.assertEqual(self.line(requester), {"role": "parent", "outcome": "granted"})
        self.assertEqual(requester.wait(timeout=10), 0)


if __name__ == "__main__":
    unittest.main()
