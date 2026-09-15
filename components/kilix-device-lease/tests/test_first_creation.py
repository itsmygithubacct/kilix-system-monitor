"""Simultaneous first creation of a namespace never refuses a requester."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import kilix_device_lease as leases

SOURCE = Path(__file__).resolve().parents[1] / "src"
WORKERS = 8
TRIALS = 15

# Every worker blocks on one pipe and starts together when the test writes to it,
# so all of them race to create the same fresh namespace.
WORKER = r"""
import json, os, sys, time
from kilix_device_lease import LeaseError, acquire
namespace, gate = sys.argv[1], int(sys.argv[2])
os.read(gate, 1)
try:
    acquire(job_id="w" + str(os.getpid()), workload="stt-job", device="d",
            deadline=time.monotonic() + 60, namespace=namespace).release(cleanup_complete=True)
    print(json.dumps("granted"), flush=True)
except LeaseError as error:
    print(json.dumps(error.code + ": " + str(error)), flush=True)
"""


class FirstCreationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="device-lease-create-")
        self.addCleanup(self.temp.cleanup)

    def acquire(self, namespace, **kwargs):
        request = dict(job_id="creator", workload="tts-utterance", device="d",
                       deadline=time.monotonic() + 3, namespace=namespace)
        request.update(kwargs)
        return leases.acquire(**request)

    def test_simultaneous_first_creation_grants_every_requester(self):
        outcomes = {}
        env = dict(os.environ, PYTHONPATH=str(SOURCE))
        for trial in range(TRIALS):
            parent = f"{self.temp.name}/trial-{trial}"
            os.mkdir(parent, 0o700)
            read_end, write_end = os.pipe()
            workers = []
            try:
                for _ in range(WORKERS):
                    workers.append(subprocess.Popen(
                        [sys.executable, "-c", WORKER, parent + "/leases", str(read_end)],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, env=env, pass_fds=(read_end,)))
            finally:
                os.close(read_end)
                os.write(write_end, b"x" * WORKERS)
                os.close(write_end)
            for worker in workers:
                out, err = worker.communicate(timeout=120)
                outcome = json.loads(out.splitlines()[-1]) if out.strip() else f"exit {worker.returncode}: {err[-200:]}"
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
        self.assertEqual(outcomes, {"granted": WORKERS * TRIALS})

    def test_anchor_left_empty_by_a_creator_is_initialised_by_the_next_requester(self):
        # A creator makes the anchor exclusively and only then locks it. A creator
        # that has not locked it yet, or died before initialising anything, leaves
        # exactly this: an empty private anchor and no namespace directory.
        namespace = self.temp.name + "/leases"
        anchor = Path(self.temp.name) / ".leases.lease-v1.anchor"
        os.close(os.open(anchor, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        anchor.chmod(0o600)
        self.acquire(namespace).release(cleanup_complete=True)
        self.acquire(namespace, workload="stt-job").release(cleanup_complete=True)
        state = json.loads((Path(namespace) / "state.json").read_text(encoding="utf-8"))
        self.assertEqual((state["next"], state["queue"], state["active"]["state"]), (2, [], "releasing"))

    def test_empty_anchor_beside_an_existing_directory_is_still_refused(self):
        namespace = self.temp.name + "/leases"
        anchor = Path(self.temp.name) / ".leases.lease-v1.anchor"
        os.close(os.open(anchor, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        anchor.chmod(0o600)
        os.mkdir(namespace, 0o700)
        with self.assertRaises(leases.LeaseError) as caught:
            self.acquire(namespace)
        self.assertEqual(caught.exception.code, "unavailable")
        self.assertEqual(os.listdir(namespace), [])
        self.assertEqual(anchor.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
