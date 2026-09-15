"""Simultaneous first creation of a namespace never refuses a requester."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import lease_containment
import kilix_device_lease as leases

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
        env = lease_containment.child_env()
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

    @staticmethod
    def private_file(path, data=b""):
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, data)
        finally:
            os.close(fd)

    def fresh_directory(self, path, **state):
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700)
        self.private_file(Path(path) / "accelerator.lock")
        value = {"version": leases.VERSION, "next": 0, "last": None, "active": None, "queue": []}
        value.update(state)
        self.private_file(Path(path) / "state.json", json.dumps(value, separators=(",", ":")).encode() + b"\n")

    def empty_anchor(self):
        anchor = Path(self.temp.name) / ".leases.lease-v1.anchor"
        self.private_file(anchor)
        return anchor

    def snapshot(self):
        return sorted((str(path.relative_to(self.temp.name)), path.stat().st_mode, path.stat().st_ino,
                       path.read_bytes() if path.is_file() else None)
                      for path in Path(self.temp.name).rglob("*"))

    def test_complete_unused_namespace_beside_an_empty_anchor_is_adopted(self):
        # A creator killed after renaming its finished directory into place, and
        # before recording it in the anchor, leaves exactly this. No requester can
        # have used it, because every requester reads the anchor first.
        namespace = self.temp.name + "/leases"
        self.fresh_directory(namespace)
        anchor = self.empty_anchor()
        inode = os.stat(namespace).st_ino
        self.acquire(namespace).release(cleanup_complete=True)
        identity = json.loads(anchor.read_text(encoding="utf-8"))
        self.assertEqual(identity["directory"][1], inode)
        self.assertEqual(identity["resource"][1], os.stat(namespace + "/accelerator.lock").st_ino)
        self.assertEqual(json.loads((Path(namespace) / "state.json").read_text(encoding="utf-8"))["next"], 1)

    def test_partial_build_directory_is_removed_before_first_creation(self):
        # A creator killed while building its private directory leaves it partial.
        build = Path(self.temp.name) / ".leases.lease-v1.build"
        os.mkdir(build, 0o700)
        self.private_file(build / "accelerator.lock")
        self.private_file(build / "state.next", b'{"partial"')
        self.empty_anchor()
        self.acquire(self.temp.name + "/leases").release(cleanup_complete=True)
        self.assertEqual(sorted(os.listdir(self.temp.name)), [".leases.lease-v1.anchor", "leases"])

    def test_anything_else_beside_an_empty_anchor_is_refused_untouched(self):
        namespace = self.temp.name + "/leases"
        build = Path(self.temp.name) / ".leases.lease-v1.build"
        cases = {
            "extra entry in the namespace": lambda: (self.fresh_directory(namespace),
                                                     self.private_file(Path(namespace) / "other")),
            "namespace that recorded a request": lambda: self.fresh_directory(namespace, next=1),
            "namespace without its lock": lambda: (self.fresh_directory(namespace),
                                                   os.unlink(Path(namespace) / "accelerator.lock")),
            "unexpected entry in the build directory": lambda: (os.mkdir(build, 0o700),
                                                                self.private_file(build / "other")),
        }
        for case, arrange in cases.items():
            with self.subTest(case=case):
                for entry in Path(self.temp.name).iterdir():
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                arrange()
                self.empty_anchor()
                before = self.snapshot()
                with self.assertRaises(leases.LeaseError) as caught:
                    self.acquire(namespace)
                self.assertEqual(caught.exception.code, "unavailable")
                self.assertEqual(self.snapshot(), before)

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
