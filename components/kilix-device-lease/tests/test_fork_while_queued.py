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
import threading
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


# A requester whose cancelled or disconnected callback forks once, at the first
# call made from a chosen place, which the callback identifies from outside the
# module by the namespace anchor:
#   outside-registry  no descriptor of the anchor is open in this process
#   registry-opening  one is open and the anchor is not locked
#   registry-locked   one is open and the anchor is locked, which for a lone
#                     requester only happens inside its own registry pass
# The child either sleeps without returning (sleep), or returns False (continue)
# or True (cancel) into the lease call and reports what that call did to it.
FORKING_CALLBACK = r"""
import fcntl, json, os, sys, time
import kilix_device_lease as leases
namespace, site, kind, mode = sys.argv[1:5]
anchor = os.path.join(os.path.dirname(namespace), "." + os.path.basename(namespace) + ".lease-v1.anchor")
parent = os.getpid()
state = {"forked": False, "spare": []}

def report(role, outcome, **extra):
    print(json.dumps({"role": role, "outcome": outcome, **extra}), flush=True)

def observed_site():
    target = os.stat(anchor)
    for name in os.listdir("/proc/self/fd"):
        try:
            info = os.stat("/proc/self/fd/" + name)
        except OSError:
            continue
        if (info.st_dev, info.st_ino) == (target.st_dev, target.st_ino):
            break
    else:
        return "outside-registry"
    probe = os.open(anchor, os.O_RDONLY | os.O_CLOEXEC)
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return "registry-locked"
    finally:
        os.close(probe)
    return "registry-opening"

def queued(_status):
    if not state.get("queued"):
        state["queued"] = True
        report("parent", "queued")

def callback():
    # Only once the request is queued, so the child inherits a request to act on.
    if os.getpid() != parent or state["forked"] or not state.get("queued") or observed_site() != site:
        return False
    state["forked"] = True
    pid = os.fork()
    if pid:
        report("parent", "forked", child=pid)
        if mode != "sleep":
            # Stay out of the registry until the child is done, so anything the
            # child tries on the parent's request meets a free anchor.
            os.waitpid(pid, 0)
        return False
    if mode == "sleep":
        time.sleep(60)
        os._exit(0)
    # Take the lowest free descriptor numbers, so a later close of a number the
    # lease call no longer owns would close one of these.
    state["spare"] = [os.open(os.devnull, os.O_RDONLY) for _ in range(16)]
    return mode == "cancel"

try:
    lease = leases.acquire(job_id="forker", workload="tts-utterance", device="d",
                           deadline=time.monotonic() + 30, namespace=namespace,
                           progress=queued, **{kind: callback})
except BaseException as error:
    outcome = error.code if isinstance(error, leases.LeaseError) else type(error).__name__
else:
    outcome = "granted"
    try:
        lease.release(cleanup_complete=True)
    except BaseException as error:
        outcome = "release " + (error.code if isinstance(error, leases.LeaseError) else type(error).__name__)
if os.getpid() != parent:
    lost = 0
    for fd in state["spare"]:
        try:
            os.fstat(fd)
        except OSError:
            lost += 1
    report("child", outcome, lost_fds=lost)
    os._exit(0)
report("parent", outcome)
"""
# A holder whose cancelled callback forks once while it checks its own lease. The
# child returns False into the parent's check and reports what that check did.
FORKING_CHECK = r"""
import json, os, sys, time
import kilix_device_lease as leases
parent = os.getpid()
state = {"checking": False, "forked": False}
def cancelled():
    if os.getpid() != parent or not state["checking"] or state["forked"]:
        return False
    state["forked"] = True
    pid = os.fork()
    if pid:
        os.waitpid(pid, 0)
    return False
lease = leases.acquire(job_id="checker", workload="stt-job", device="d", deadline=time.monotonic() + 30,
                       namespace=sys.argv[1], cancelled=cancelled)
state["checking"] = True
try:
    lease.check()
    outcome = "current"
except BaseException as error:
    outcome = error.code if isinstance(error, leases.LeaseError) else type(error).__name__
role = "parent" if os.getpid() == parent else "child"
print(json.dumps({"role": role, "outcome": outcome}), flush=True)
if role == "child":
    os._exit(0)
lease.release(cleanup_complete=True)
"""

# Bounds on waiting for an event that should follow at once.
PATIENCE_SECONDS = 10


class ForkWhileQueuedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="device-lease-fork-")
        self.addCleanup(self.temp.cleanup)
        self.namespace = self.temp.name + "/leases"
        self.env = lease_containment.child_env()
        self.pending = {}

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

    def forking(self, site, kind, mode):
        namespace = tempfile.mkdtemp(prefix="fork-", dir=self.temp.name) + "/leases"
        holder = self.acquire(namespace=namespace)
        child = subprocess.Popen([sys.executable, "-c", FORKING_CALLBACK, namespace, site, kind, mode],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 bufsize=0, env=self.env)
        self.pending[child.pid] = b""

        def stop():
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdout.close()
            child.stderr.close()

        self.addCleanup(stop)
        return namespace, holder, child

    def next_line(self, child):
        # The parent and its forked child share one pipe, so lines can arrive
        # together; read raw bytes rather than trust select on a buffered reader.
        buffer = self.pending[child.pid]
        end = time.monotonic() + PATIENCE_SECONDS
        while b"\n" not in buffer:
            remaining = end - time.monotonic()
            self.assertGreater(remaining, 0, "no line from the requester")
            if select.select([child.stdout], [], [], remaining)[0]:
                chunk = os.read(child.stdout.fileno(), 4096)
                self.assertTrue(chunk, "the requester closed its output")
                buffer += chunk
        line, _, self.pending[child.pid] = buffer.partition(b"\n")
        return json.loads(line)

    def forked_child(self, lines):
        forked = [line for line in lines if line.get("outcome") == "forked"]
        self.assertEqual(len(forked), 1, lines)
        pid = forked[0]["child"]
        self.addCleanup(lambda: os.kill(pid, signal.SIGKILL) if os.path.exists(f"/proc/{pid}") else None)
        return pid

    def released_within_patience(self, holder):
        done = threading.Event()

        def release():
            holder.release(cleanup_complete=True)
            done.set()

        # A daemon thread: if the acknowledgement stalls, the test fails and the
        # thread finishes once the forked child is killed during cleanup.
        threading.Thread(target=release, daemon=True).start()
        return done.wait(PATIENCE_SECONDS)

    def test_child_forked_inside_a_registry_pass_never_stalls_the_namespace(self):
        # The child keeps running without returning into the lease call. It must
        # not keep the registry locked, whether it was forked while its parent's
        # anchor descriptor was open or while that descriptor held the lock.
        for site, kind in (("registry-opening", "cancelled"), ("registry-opening", "disconnected"),
                           ("registry-locked", "cancelled"), ("registry-locked", "disconnected")):
            with self.subTest(site=site, kind=kind):
                namespace, holder, requester = self.forking(site, kind, "sleep")
                self.assertEqual(self.next_line(requester), {"role": "parent", "outcome": "queued"})
                if site == "registry-opening":
                    sleeper = self.forked_child([self.next_line(requester)])
                    self.assertTrue(self.released_within_patience(holder),
                                    "the holder's acknowledgement waited on the forked child")
                else:
                    holder.release(cleanup_complete=True)
                    sleeper = self.forked_child([self.next_line(requester)])
                # The requester's own grant and acknowledgement follow.
                self.assertEqual(self.next_line(requester), {"role": "parent", "outcome": "granted"})
                self.acquire(namespace=namespace, job_id="successor", workload="stt-job",
                             deadline=time.monotonic() + PATIENCE_SECONDS).release(cleanup_complete=True)
                self.assertTrue(os.path.exists(f"/proc/{sleeper}"), "the forked child must still be alive")

    def test_child_forked_inside_a_callback_never_acts_on_the_parents_request(self):
        runs = [(site, mode) for site in ("outside-registry", "registry-opening", "registry-locked")
                for mode in ("continue", "cancel")]
        for index, (site, mode) in enumerate(runs):
            kind = ("cancelled", "disconnected")[index % 2]
            with self.subTest(site=site, mode=mode, kind=kind):
                namespace, holder, requester = self.forking(site, kind, mode)
                self.assertEqual(self.next_line(requester), {"role": "parent", "outcome": "queued"})
                if site == "registry-locked":
                    holder.release(cleanup_complete=True)
                lines = [self.next_line(requester), self.next_line(requester)]
                if site != "registry-locked":
                    holder.release(cleanup_complete=True)
                lines.append(self.next_line(requester))
                self.forked_child(lines)
                self.assertEqual(sorted((line["role"], line["outcome"], line.get("lost_fds")) for line in lines),
                                 [("child", "lost-lease", 0), ("parent", "forked", None), ("parent", "granted", None)])
                self.assertEqual(requester.wait(timeout=PATIENCE_SECONDS), 0)
                state = json.loads((Path(namespace) / "state.json").read_text(encoding="utf-8"))
                self.assertEqual((state["next"], state["queue"], state["active"]["job_id"], state["active"]["state"]),
                                 (2, [], "forker", "releasing"))
                self.acquire(namespace=namespace, job_id="successor", workload="stt-job",
                             deadline=time.monotonic() + PATIENCE_SECONDS).release(cleanup_complete=True)

    def test_child_forked_inside_a_check_callback_is_refused_lost_lease(self):
        done = subprocess.run([sys.executable, "-c", FORKING_CHECK, self.namespace], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, env=self.env, timeout=PATIENCE_SECONDS * 6)
        self.assertEqual(done.returncode, 0, done.stderr[-4000:])
        lines = [json.loads(line) for line in done.stdout.splitlines()]
        self.assertEqual(sorted((line["role"], line["outcome"]) for line in lines),
                         [("child", "lost-lease"), ("parent", "current")])
        self.acquire(job_id="successor", workload="stt-job",
                     deadline=time.monotonic() + PATIENCE_SECONDS).release(cleanup_complete=True)

if __name__ == "__main__":
    unittest.main()
