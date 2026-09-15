"""A cleanup acknowledgement is never lost to a busy registry."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import kilix_device_lease as leases

# A cooperating process that holds the registry anchor lock until told to stop,
# or for a fixed number of seconds.
ANCHOR_HOLDER = r"""
import fcntl, os, select, sys, time
fd = os.open(sys.argv[1], os.O_RDWR)
fcntl.flock(fd, fcntl.LOCK_EX)
print("held", flush=True)
select.select([sys.stdin], [], [], float(sys.argv[2]))
"""


class ReleaseAcknowledgementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="device-lease-ack-")
        self.addCleanup(self.temp.cleanup)
        self.namespace = self.temp.name + "/leases"
        self.anchor = self.temp.name + "/.leases.lease-v1.anchor"

    def acquire(self, **kwargs):
        request = dict(job_id="ack-job", workload="llm-turn", device="ack-device",
                       deadline=time.monotonic() + 30, namespace=self.namespace)
        request.update(kwargs)
        return leases.acquire(**request)

    def active(self):
        return json.loads((Path(self.namespace) / "state.json").read_text(encoding="utf-8"))["active"]

    def hold_anchor(self, seconds):
        holder = subprocess.Popen([sys.executable, "-c", ANCHOR_HOLDER, self.anchor, str(seconds)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def stop():
            if holder.poll() is None:
                holder.kill()
            holder.wait(timeout=5)
            for stream in (holder.stdin, holder.stdout, holder.stderr):
                stream.close()

        self.addCleanup(stop)
        self.assertTrue(select.select([holder.stdout], [], [], 10)[0], "anchor holder did not start")
        self.assertEqual(holder.stdout.readline(), b"held\n")
        return holder

    def successor(self, seconds=3):
        try:
            self.acquire(job_id="successor", workload="stt-job",
                         deadline=time.monotonic() + seconds).release(cleanup_complete=True)
            return "granted"
        except leases.LeaseError as error:
            return error.code

    def test_acknowledgement_outlasts_a_registry_held_longer_than_a_second(self):
        for seconds in (1.6, 3.0):
            with self.subTest(seconds=seconds):
                folder = tempfile.mkdtemp(prefix=f"hold-{seconds}-", dir=self.temp.name)
                self.namespace = folder + "/leases"
                self.anchor = folder + "/.leases.lease-v1.anchor"
                lease = self.acquire()
                self.hold_anchor(seconds)
                lease.release(cleanup_complete=True)
                self.assertEqual(self.active()["state"], "releasing")
                self.assertEqual(self.successor(), "granted")

    def test_unrecorded_acknowledgement_keeps_the_guard_until_a_retry_records_it(self):
        with mock.patch.object(leases, "_ACK_WAIT_SECONDS", 0.3):
            lease = self.acquire()
            holder = self.hold_anchor(60)
            with self.assertRaises(leases.LeaseError) as caught:
                lease.release(cleanup_complete=True)
            self.assertEqual(caught.exception.code, "deadline")
            holder.stdin.close()
            holder.wait(timeout=5)
            self.assertEqual(self.active()["state"], "held")
            # The acquiring process still holds the guard, so nothing else is granted.
            self.assertEqual(self.successor(seconds=0.3), "deadline")
            lease.release(cleanup_complete=True)
        self.assertEqual(self.active()["state"], "releasing")
        self.assertEqual(self.successor(), "granted")

    def test_default_release_still_quarantines_without_waiting(self):
        lease = self.acquire()
        self.hold_anchor(60)
        began = time.monotonic()
        lease.release()
        self.assertLess(time.monotonic() - began, 1.0)
        self.assertEqual(self.active()["state"], "held")


if __name__ == "__main__":
    unittest.main()
