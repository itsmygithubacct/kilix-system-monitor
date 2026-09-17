"""K-ACC-R1 windows: grant-before-guard, close-side fd reuse, refused __enter__."""
from __future__ import annotations

import inspect
import json
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import lease_containment
import kilix_device_lease as leases


def _line_of(function, needle):
    lines, first = inspect.getsourcelines(function)
    hits = [first + i for i, text in enumerate(lines) if needle in text]
    if len(hits) != 1:
        raise AssertionError((function, needle, hits))
    return hits[0]


class GrantCommitTests(unittest.TestCase):
    """F-01: do not persist active:held until this process owns the guard."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="device-lease-grant-")
        self.addCleanup(self.temp.cleanup)
        self.namespace = self.temp.name + "/leases"
        self.env = lease_containment.child_env()

    def successor(self):
        done = subprocess.run(
            [sys.executable, "-c",
             "import sys,time,kilix_device_lease as leases\n"
             "try:\n"
             "    leases.acquire(job_id='successor', workload='tts-utterance', device='d',"
             " deadline=time.monotonic()+3, namespace=sys.argv[1]).release(cleanup_complete=True)\n"
             "    print('granted')\n"
             "except leases.LeaseError as error:\n"
             "    print(error.code)\n",
             self.namespace],
            capture_output=True, text=True, timeout=30, env=self.env)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.strip()

    def fire_at(self, needle, interrupt):
        target = _line_of(leases._acquire, needle)
        fillers = []
        box = {"fired": False}

        def local(frame, event, arg):
            if event == "exception" and fillers:
                while fillers:
                    os.close(fillers.pop())
            if event == "line" and frame.f_lineno == target and not box["fired"]:
                box["fired"] = True
                if interrupt:
                    raise KeyboardInterrupt
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(512, hard), hard))
                while True:
                    try:
                        fillers.append(os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC))
                    except OSError:
                        break
            return local

        def tracer(frame, event, arg):
            return local if frame.f_code is leases._acquire.__code__ else None

        sys.settrace(tracer)
        try:
            leases.acquire(job_id="p2", workload="stt-job", device="d",
                           deadline=time.monotonic() + 10, namespace=self.namespace)
            outcome = "granted"
        except leases.LeaseError as error:
            outcome = error.code
        except KeyboardInterrupt:
            outcome = "KeyboardInterrupt"
        finally:
            sys.settrace(None)
            while fillers:
                os.close(fillers.pop())
        return box["fired"], outcome

    def test_control_interrupt_before_the_grant_is_saved_leaves_the_namespace_usable(self):
        fired, outcome = self.fire_at("registry.validate()", interrupt=True)
        self.assertTrue(fired)
        self.assertEqual(outcome, "KeyboardInterrupt")
        self.assertEqual(self.successor(), "granted")

    def test_emfile_at_guard_dup_does_not_quarantine_the_namespace(self):
        fired, outcome = self.fire_at("fd = os.dup(registry.resource)", interrupt=False)
        self.assertTrue(fired)
        self.assertEqual(outcome, "unavailable")
        self.assertEqual(self.successor(), "granted")

    def test_interrupt_at_guard_dup_does_not_quarantine_the_namespace(self):
        fired, outcome = self.fire_at("fd = os.dup(registry.resource)", interrupt=True)
        self.assertTrue(fired)
        self.assertEqual(outcome, "KeyboardInterrupt")
        self.assertEqual(self.successor(), "granted")


class ForkDuringCloseTests(unittest.TestCase):
    """F-02: a child forked after a close syscall keeps an unrelated reused number."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="device-lease-close-")
        self.addCleanup(self.temp.cleanup)
        self.namespace = self.temp.name + "/leases"
        self.report_r, self.report_w = os.pipe()
        self.addCleanup(os.close, self.report_r)
        self.addCleanup(os.close, self.report_w)

    def fork_with_fresh_pipe(self, recorded):
        out = {}

        def run():
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                state = {}
                for label, fd in (("r", r), ("w", w)):
                    try:
                        os.fstat(fd)
                        state[label] = "open"
                    except OSError as error:
                        state[label] = "EBADF" if error.errno == 9 else str(error)
                os.write(self.report_w, (json.dumps(state) + "\n").encode())
                os._exit(0)
            os.waitpid(pid, 0)
            out["pipe"] = [r, w]
            os.close(r)
            os.close(w)

        worker = threading.Thread(target=run)
        worker.start()
        worker.join()
        child = json.loads(os.read(self.report_r, 4096).decode())
        return {"recorded_number": recorded, "pipe": out["pipe"], "child": child,
                "reused": recorded in out["pipe"]}

    def close_window(self, arm):
        registry_line = _line_of(leases._Registry.close, "setattr(self, name, -1)")
        descriptor_line = _line_of(leases._Descriptor.close, "self.fd = -1")
        box = {"done": False, "result": None}

        def local(frame, event, arg):
            if box["done"] or event != "line":
                return local
            code = frame.f_code
            if arm == "registry" and code is leases._Registry.close.__code__ and frame.f_lineno == registry_line \
                    and frame.f_locals.get("name") == "anchor":
                box["done"] = True
                box["result"] = self.fork_with_fresh_pipe(frame.f_locals["fd"])
            elif arm == "descriptor" and code is leases._Descriptor.close.__code__ and frame.f_lineno == descriptor_line:
                box["done"] = True
                box["result"] = self.fork_with_fresh_pipe(frame.f_locals["fd"])
            return local

        def tracer(frame, event, arg):
            if frame.f_code in (leases._Registry.close.__code__, leases._Descriptor.close.__code__):
                return local
            return None

        if arm == "control":
            return self.fork_with_fresh_pipe(None)
        sys.settrace(tracer)
        try:
            lease = leases.acquire(job_id="p1", workload="stt-job", device="d",
                                   deadline=time.monotonic() + 10, namespace=self.namespace)
        finally:
            sys.settrace(None)
        lease.release(cleanup_complete=True)
        return box["result"] or {"armed": False}

    def test_control_fork_outside_close_keeps_the_pipe(self):
        row = self.close_window("control")
        self.assertEqual(row["child"], {"r": "open", "w": "open"}, row)

    def test_fork_after_registry_close_keeps_an_unrelated_pipe(self):
        row = self.close_window("registry")
        self.assertTrue(row.get("reused"), row)
        self.assertEqual(row["child"], {"r": "open", "w": "open"}, row)

    def test_fork_after_descriptor_close_keeps_an_unrelated_pipe(self):
        row = self.close_window("descriptor")
        self.assertTrue(row.get("reused"), row)
        self.assertEqual(row["child"], {"r": "open", "w": "open"}, row)


class EnterFailureTests(unittest.TestCase):
    """F-03: with lease: must release() if __enter__'s check refuses."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="device-lease-enter-")
        self.addCleanup(self.temp.cleanup)
        self.namespace = self.temp.name + "/leases"
        self.env = lease_containment.child_env()

    def other_provider(self):
        done = subprocess.run(
            [sys.executable, "-c",
             "import sys,time,kilix_device_lease as leases\n"
             "began=time.monotonic()\n"
             "try:\n"
             "    leases.acquire(job_id='other-provider', workload='stt-job', device='d',"
             " deadline=time.monotonic()+3, namespace=sys.argv[1]).release(cleanup_complete=True)\n"
             "    print('granted %.2f' % (time.monotonic()-began))\n"
             "except leases.LeaseError as error:\n"
             "    print('%s %.2f' % (error.code, time.monotonic()-began))\n",
             self.namespace],
            capture_output=True, text=True, timeout=30, env=self.env)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.strip()

    def test_control_refusal_inside_the_block_closes_the_guard(self):
        flags = {"gone": False}
        lease = leases.acquire(job_id="svc-job-1", workload="tts-utterance", device="d",
                               deadline=time.monotonic() + 30, namespace=self.namespace,
                               disconnected=lambda: flags["gone"])
        guard = lease._fd
        try:
            with lease:
                flags["gone"] = True
                lease.guard_fd
            raised = None
        except leases.LeaseError as error:
            raised = error.code
        self.assertEqual(raised, "cancelled")
        with self.assertRaises(OSError):
            os.fstat(guard)
        self.assertTrue(self.other_provider().startswith("unavailable"))

    def test_refusal_in_enter_closes_the_guard(self):
        flags = {"gone": False}
        lease = leases.acquire(job_id="svc-job-1", workload="tts-utterance", device="d",
                               deadline=time.monotonic() + 30, namespace=self.namespace,
                               disconnected=lambda: flags["gone"])
        guard = lease._fd
        flags["gone"] = True
        try:
            with lease:
                pass
            raised = None
        except leases.LeaseError as error:
            raised = error.code
        self.assertEqual(raised, "cancelled")
        with self.assertRaises(OSError):
            os.fstat(guard)
        self.assertTrue(self.other_provider().startswith("unavailable"))


if __name__ == "__main__":
    unittest.main()
