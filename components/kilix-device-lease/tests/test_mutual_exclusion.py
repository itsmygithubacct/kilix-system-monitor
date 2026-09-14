"""Concurrent real processes never hold overlapping grants, whatever their labels."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src"
WORKLOADS = ("tts-utterance", "stt-job", "llm-turn")
WORKERS_PER_WORKLOAD = 3
ROUNDS = 3

# Each grant creates one exclusive marker before its critical section and
# removes it before acknowledging cleanup. A second concurrent holder cannot
# create the marker, so any overlap is a named exit, not a timing guess.
WORKER = r"""
import os, sys, time
from kilix_device_lease import LeaseError, acquire
namespace, marker, workload, job, rounds = sys.argv[1:6]
for index in range(int(rounds)):
    try:
        lease = acquire(job_id=f"{job}-{index}", workload=workload, device=f"label-{job}",
                        deadline=time.monotonic() + 60, namespace=namespace)
    except LeaseError as error:
        print(f"refused {error.code}", flush=True)
        raise SystemExit(2)
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print("overlap", flush=True)
        raise SystemExit(3)
    os.close(fd)
    time.sleep(0.02)
    os.unlink(marker)
    lease.release(cleanup_complete=True)
print("done", flush=True)
"""


class MutualExclusionTests(unittest.TestCase):
    def test_concurrent_workers_across_labels_never_overlap(self):
        with tempfile.TemporaryDirectory(prefix="device-lease-exclusion-") as folder:
            namespace = folder + "/leases"
            marker = folder + "/critical-section"
            env = dict(os.environ, PYTHONPATH=str(SOURCE))
            workers = []

            def stop():
                for worker in workers:
                    if worker.poll() is None:
                        worker.kill()
                        worker.wait(timeout=5)

            self.addCleanup(stop)
            # The namespace does not exist yet, so first creation is contended too.
            for workload in WORKLOADS:
                for number in range(WORKERS_PER_WORKLOAD):
                    workers.append(subprocess.Popen(
                        [sys.executable, "-c", WORKER, namespace, marker, workload,
                         f"{workload}-{number}", str(ROUNDS)],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, env=env))
            results = []
            for worker in workers:
                out, err = worker.communicate(timeout=120)
                results.append((worker.returncode, out.strip(), err))
            for code, out, err in results:
                self.assertEqual((code, out), (0, "done"), err)
            self.assertFalse(Path(marker).exists())
            state = json.loads((Path(namespace) / "state.json").read_text(encoding="utf-8"))
            grants = len(WORKLOADS) * WORKERS_PER_WORKLOAD * ROUNDS
            self.assertEqual(state["next"], grants, "every grant must pass through the queue")
            self.assertEqual(state["queue"], [])
            self.assertEqual(state["active"]["state"], "releasing")


if __name__ == "__main__":
    unittest.main()
