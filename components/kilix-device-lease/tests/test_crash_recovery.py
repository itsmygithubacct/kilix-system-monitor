"""A coordinator killed at any step of queue bookkeeping never wedges the namespace."""
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

# The victim kills itself immediately before its Nth mutating filesystem or lock
# operation. Python audit events fire before the operation runs, so point N means
# operations 1..N-1 completed and operation N never started. Read-only directory
# traversal is not counted. A victim that reaches the end reports its count.
VICTIM = r"""
import json, os, signal, sys, time
import kilix_device_lease as leases
namespace, scenario, kill_at = sys.argv[1], sys.argv[2], int(sys.argv[3])
MUTATING = os.O_CREAT | os.O_WRONLY | os.O_RDWR
state = {"armed": False, "count": 0}
def hook(event, args):
    if not state["armed"]:
        return
    if event == "open" and not (args[2] & MUTATING):
        return
    if event not in ("open", "os.remove", "os.rename", "os.mkdir", "os.rmdir", "fcntl.flock", "os.truncate",
                     "os.chmod"):
        return
    state["count"] += 1
    if state["count"] == kill_at:
        os.kill(os.getpid(), signal.SIGKILL)
sys.addaudithook(hook)
queued = {"seen": False}
state["armed"] = True
try:
    lease = leases.acquire(job_id="victim", workload="stt-job", device="victim",
                           deadline=time.monotonic() + 10, namespace=namespace,
                           cancelled=lambda: scenario == "cancel" and queued["seen"],
                           progress=lambda _status: queued.update(seen=True))
    lease.release(cleanup_complete=True)
    outcome = "granted"
except leases.LeaseError as error:
    outcome = error.code
state["armed"] = False
print(json.dumps({"count": state["count"], "outcome": outcome}), flush=True)
"""

QUEUED = r"""
import json, sys, time
import kilix_device_lease as leases
leases.acquire(job_id="abandoned", workload="tts-utterance", device="peer", deadline=time.monotonic() + 60,
               namespace=sys.argv[1], progress=lambda status: print(json.dumps(status.ticket), flush=True))
"""


class CrashRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="device-lease-crash-")
        self.addCleanup(self.temp.cleanup)
        self.env = lease_containment.child_env()

    def acquire(self, namespace, **kwargs):
        request = dict(job_id="observer", workload="llm-turn", device="observer",
                       deadline=time.monotonic() + 3, namespace=namespace)
        request.update(kwargs)
        return leases.acquire(**request)

    def state(self, namespace):
        return json.loads((Path(namespace) / "state.json").read_text(encoding="utf-8"))

    def victim(self, namespace, scenario, kill_at):
        return subprocess.run([sys.executable, "-c", VICTIM, namespace, scenario, str(kill_at)],
                              stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              env=self.env, timeout=30)

    def abandon_a_queued_request(self, namespace):
        holder = self.acquire(namespace, job_id="holder")
        peer = subprocess.Popen([sys.executable, "-c", QUEUED, namespace], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.env)
        try:
            self.assertTrue(json.loads(peer.stdout.readline()))
        finally:
            peer.kill()
            peer.wait(timeout=5)
            peer.stdout.close()
            peer.stderr.close()
        holder.release(cleanup_complete=True)

    def sweep(self, scenario):
        points = 0
        for kill_at in range(1, 400):
            namespace = f"{self.temp.name}/{scenario}-{kill_at}/leases"
            os.mkdir(os.path.dirname(namespace), 0o700)
            holder = None
            if scenario == "cancel":
                holder = self.acquire(namespace, job_id="holder")
            elif scenario == "prune":
                self.abandon_a_queued_request(namespace)
            result = self.victim(namespace, scenario, kill_at)
            if holder is not None:
                holder.release(cleanup_complete=True)
            completed = result.returncode == 0
            if completed:
                self.assertEqual(json.loads(result.stdout)["outcome"],
                                 "cancelled" if scenario == "cancel" else "granted", result.stderr)
            else:
                self.assertEqual(result.returncode, -9, result.stderr)
            try:
                self.acquire(namespace).release(cleanup_complete=True)
                successor = "granted"
            except leases.LeaseError as error:
                successor = error.code
            if successor != "granted":
                # Only a grant the victim had already persisted may stay quarantined.
                self.assertIn(scenario, ("prune", "fresh"), f"kill point {kill_at}")
                active = self.state(namespace)["active"]
                self.assertEqual((successor, active and active["state"], active and active["job_id"]),
                                 ("unavailable", "held", "victim"), f"kill point {kill_at}")
            self.assertEqual(self.state(namespace)["queue"], [], f"kill point {kill_at}")
            # Nothing a killed creator built is left beside the namespace.
            self.assertEqual(sorted(os.listdir(os.path.dirname(namespace))), [".leases.lease-v1.anchor", "leases"],
                             f"kill point {kill_at}")
            points += 1
            if completed:
                return points
        self.fail(f"{scenario} victim never completed")

    def test_refusal_cleanup_killed_at_any_step_never_wedges_the_namespace(self):
        self.assertGreater(self.sweep("cancel"), 3)

    def test_prune_killed_at_any_step_never_wedges_the_namespace(self):
        self.assertGreater(self.sweep("prune"), 3)

    def test_first_creation_killed_at_any_step_never_wedges_the_namespace(self):
        # The victim is the first requester of a namespace that does not exist yet.
        self.assertGreater(self.sweep("fresh"), 10)

    def test_queue_entry_whose_ticket_vanished_is_dropped(self):
        for expired in (True, False):
            with self.subTest(expired=expired):
                namespace = f"{self.temp.name}/vanished-{expired}/leases"
                os.mkdir(os.path.dirname(namespace), 0o700)
                self.acquire(namespace, job_id="seed").release(cleanup_complete=True)
                state = self.state(namespace)
                # The state an earlier coordinator left when it died after unlinking
                # a ticket but before saving the queue that referenced it.
                state["queue"] = [{"ticket": "0" * 32, "job_id": "vanished", "workload": "stt-job",
                                   "device": "d", "sequence": state["next"],
                                   "deadline": time.monotonic() + (-1 if expired else 600), "inode": [1, 1]}]
                state["next"] += 1
                path = Path(namespace) / "state.next"
                path.write_text(json.dumps(state, separators=(",", ":")) + "\n", encoding="utf-8")
                path.chmod(0o600)
                os.replace(path, Path(namespace) / "state.json")
                self.acquire(namespace).release(cleanup_complete=True)
                self.assertEqual(self.state(namespace)["queue"], [])


if __name__ == "__main__":
    unittest.main()
