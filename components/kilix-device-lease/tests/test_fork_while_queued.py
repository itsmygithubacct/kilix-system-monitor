"""A process forked from a queued requester never keeps that request alive."""
from __future__ import annotations

from collections import Counter
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
import uuid

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
# A queued requester that forks from a second thread at each distinct call site
# the lease module reaches: every audited call, and every os.fstat, which Python
# does not audit and which the module makes while it holds a descriptor it opened
# for one step only. Each forked child lists every descriptor it holds that the
# requester did not hold before it started, and exits at once. The requester
# cancels itself after a number of registry passes, so it is never granted.
THREAD_FORKING_REQUESTER = r"""
import json, os, sys, threading, time, warnings
import kilix_device_lease as leases
warnings.simplefilter("ignore", DeprecationWarning)
namespace = sys.argv[1]
module = leases.__file__
parent = os.getpid()
main = threading.main_thread()

def descriptors():
    found = {}
    for name in os.listdir("/proc/self/fd"):
        try:
            found[int(name)] = os.readlink("/proc/self/fd/" + name)
        except OSError:
            pass
    return found

before = set(descriptors())
seen = set()
rows = []
state = {"busy": False, "checks": 0}

def fork_from_another_thread(stack, event):
    read_end, write_end = os.pipe()
    ignored = before | {read_end, write_end}
    forked = {}
    def fork():
        pid = os.fork()
        if pid == 0:
            try:
                held = sorted(target for fd, target in descriptors().items() if fd not in ignored)
                os.write(write_end, json.dumps(held).encode())
            finally:
                os._exit(0)
        forked["pid"] = pid
    thread = threading.Thread(target=fork)
    thread.start()
    thread.join()
    os.close(write_end)
    report = b""
    while True:
        chunk = os.read(read_end, 65536)
        if not chunk:
            break
        report += chunk
    os.close(read_end)
    os.waitpid(forked["pid"], 0)
    held = sorted(target for fd, target in descriptors().items() if fd not in ignored)
    rows.append({"stack": stack, "event": event, "child": json.loads(report), "parent": held})

def observe(event, args):
    if state["busy"] or os.getpid() != parent or threading.current_thread() is not main:
        return
    state["busy"] = True
    try:
        names, lines = [], []
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_filename == module:
                names.append(frame.f_code.co_name)
                lines.append(frame.f_lineno)
            frame = frame.f_back
        key = (tuple(lines), event)
        if names and key not in seen:
            seen.add(key)
            fork_from_another_thread(">".join(reversed(names)), event)
    finally:
        state["busy"] = False

real_fstat = os.fstat
def fstat(fd):
    observe("os.fstat", (fd,))
    return real_fstat(fd)

def cancelled():
    state["checks"] += 1
    return state["checks"] > 40

sys.addaudithook(observe)
os.fstat = fstat
try:
    leases.acquire(job_id="thread-forker", workload="tts-utterance", device="d",
                   deadline=time.monotonic() + 60, namespace=namespace, cancelled=cancelled)
    outcome = "granted"
except leases.LeaseError as error:
    outcome = error.code
os.fstat = real_fstat
print(json.dumps({"outcome": outcome, "rows": rows}), flush=True)
"""

# A queued requester whose second thread forks a sleeping child at one of two
# moments, copied from the close-side stall probe:
#   tracked     prune's os.listdir: the registry is open, locked, and still tracked
#   close-side  inside _Registry.close, after it dropped itself from tracking and
#               before it closes the still-locked anchor
# The child lists namespace-path descriptors. The holder then releases and
# reports how long acknowledgement took. A child that still holds the anchor
# stalls that acknowledgement for the child's remaining lifetime.
CLOSE_SIDE_STALL = r"""
import json, os, signal, subprocess, sys, tempfile, threading, time, warnings
import kilix_device_lease as leases
warnings.simplefilter("ignore", DeprecationWarning)
REAL_CLOSE = os.close
HOLDER = r'''
import sys, time
import kilix_device_lease as leases
lease = leases.acquire(job_id="holder", workload="llm-turn", device="h",
                       deadline=time.monotonic() + 60, namespace=sys.argv[1])
print("held", flush=True)
sys.stdin.readline()
started = time.monotonic()
lease.release(cleanup_complete=True)
print("ack %.2f" % (time.monotonic() - started), flush=True)
'''
BOX = {"arm": sys.argv[1], "armed": False, "child": None, "folder": "", "w": -1}


def fork_sleeper():
    def fork():
        pid = os.fork()
        if pid == 0:
            held = []
            for name in os.listdir("/proc/self/fd"):
                try:
                    target = os.readlink("/proc/self/fd/" + name)
                except OSError:
                    continue
                if BOX["folder"] in target:
                    held.append(target.replace(BOX["folder"], "<ns-folder>"))
            os.write(BOX["w"], (json.dumps(sorted(held)) + "\n").encode())
            time.sleep(6)
            os._exit(0)
        BOX["child"] = pid
    BOX["armed"] = False
    thread = threading.Thread(target=fork)
    thread.start()
    thread.join()


def audit(event, args):
    if (BOX["armed"] and BOX["arm"] == "tracked" and event == "os.listdir"
            and threading.current_thread() is threading.main_thread()):
        if sys._getframe(1).f_code.co_name == "prune":
            fork_sleeper()


sys.addaudithook(audit)


def close(fd):
    frame = sys._getframe(1)
    if (BOX["armed"] and BOX["arm"] == "close-side" and frame.f_code.co_name == "close"
            and type(frame.f_locals.get("self")).__name__ == "_Registry"):
        fork_sleeper()
    return REAL_CLOSE(fd)


BOX["folder"] = tempfile.mkdtemp(prefix="closestall-" + BOX["arm"] + "-")
ns = BOX["folder"] + "/leases"
read_end, BOX["w"] = os.pipe()
holder = subprocess.Popen([sys.executable, "-c", HOLDER, ns],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
assert holder.stdout.readline().strip() == "held"
passes = {"n": 0}


def progress(_status):
    if BOX["child"] is None:
        BOX["armed"] = True


def cancelled():
    if BOX["child"] is not None:
        passes["n"] += 1
    return passes["n"] > 2


os.close = close
try:
    leases.acquire(job_id="requester", workload="stt-job", device="r",
                   deadline=time.monotonic() + 20, namespace=ns,
                   progress=progress, cancelled=cancelled)
    outcome = "granted?!"
except leases.LeaseError as error:
    outcome = error.code
finally:
    BOX["armed"] = False
    os.close = REAL_CLOSE
REAL_CLOSE(BOX["w"])
raw = os.read(read_end, 65536).decode().splitlines()
REAL_CLOSE(read_end)
held = json.loads(raw[0]) if raw else None
child_alive_at_release = BOX["child"] is not None and os.path.exists("/proc/%s" % BOX["child"])
holder.stdin.write("\n")
holder.stdin.flush()
ack = holder.stdout.readline().strip()
holder.wait(timeout=30)
holder.stdin.close()
holder.stdout.close()
if BOX["child"] is not None:
    os.kill(BOX["child"], signal.SIGKILL)
    os.waitpid(BOX["child"], 0)
print(json.dumps({"arm": BOX["arm"], "requester": outcome, "child_held": held,
                  "child_alive_at_release": child_alive_at_release, "holder_ack": ack}),
      flush=True)
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

    def test_child_forked_by_another_thread_during_a_registry_pass_holds_no_descriptor(self):
        # The requester queues behind a holder, and its passes prune an unreferenced
        # ticket as well as its own, so prune and validate each hold a descriptor
        # of their own while another thread forks.
        holder = self.acquire()
        orphan = uuid.uuid4().hex + ".ticket"
        fd = os.open(Path(self.namespace) / orphan, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC, 0o600)
        try:
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        try:
            done = subprocess.run([sys.executable, "-c", THREAD_FORKING_REQUESTER, self.namespace],
                                  stdin=subprocess.DEVNULL, capture_output=True, text=True, env=self.env,
                                  timeout=PATIENCE_SECONDS * 30)
        finally:
            holder.release(cleanup_complete=True)
        self.assertEqual(done.returncode, 0, done.stderr[-4000:])
        report = json.loads(done.stdout.splitlines()[-1])
        self.assertEqual(report["outcome"], "cancelled")
        rows = report["rows"]
        # The module only opens the namespace's parent directory, what lies under
        # it, and the directories above it; the interpreter's own descriptors,
        # such as the one os.urandom keeps open, are not the module's.
        directory = os.path.realpath(os.path.dirname(self.namespace))
        path = {directory, *(str(ancestor) for ancestor in Path(directory).parents)}

        def lease_descriptors(targets):
            return [target for target in targets if target in path or target.startswith(directory + "/")]

        held = [(row["stack"], row["event"], lease_descriptors(row["child"]))
                for row in rows if lease_descriptors(row["child"])]
        self.assertEqual(held, [], json.dumps(held, indent=1))
        # Those forks did happen while a pass held a descriptor opened for one step:
        # prune's second copy of the requester's own ticket, and of the orphan...
        prune = [Counter(target for target in row["parent"] if target.endswith(".ticket"))
                 for row in rows if row["stack"].endswith(">prune") and row["event"] == "fcntl.flock"]
        self.assertTrue(any(counts[name] >= 2 for counts in prune for name in counts), prune)
        self.assertTrue(any(name.endswith("/" + orphan) for counts in prune for name in counts), prune)
        # ...validate's second copy of the namespace's parent directory, and the
        # directories its no-follow traversal passes through.
        directory = os.path.realpath(os.path.dirname(self.namespace))
        validate = [row["parent"] for row in rows if row["stack"].endswith(">validate") and row["event"] == "os.fstat"]
        self.assertTrue(any(held.count(directory) >= 2 for held in validate), validate)
        traversal = [row["parent"] for row in rows
                     if row["stack"].endswith(">validate>_open_parent") and row["event"] == "open"]
        self.assertTrue(any("/" in held for held in traversal), traversal)
        self.assertFalse((Path(self.namespace) / orphan).exists())

    def test_child_forked_inside_registry_close_does_not_stall_the_holder(self):
        # A fork from another thread during _Registry.close must not copy the
        # locked anchor. On the previous module the close-side child held it
        # and the holder's release(cleanup_complete=True) waited out the sleep.
        for arm in ("tracked", "close-side"):
            with self.subTest(arm=arm):
                done = subprocess.run([sys.executable, "-c", CLOSE_SIDE_STALL, arm],
                                      stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                      env=self.env, timeout=PATIENCE_SECONDS * 6)
                self.assertEqual(done.returncode, 0, done.stderr[-4000:])
                report = json.loads(done.stdout.splitlines()[-1])
                self.assertEqual(report["requester"], "cancelled")
                self.assertTrue(report["child_alive_at_release"], report)
                self.assertEqual(report["child_held"], [])
                self.assertTrue(report["holder_ack"].startswith("ack "), report["holder_ack"])
                self.assertLess(float(report["holder_ack"].split()[1]), 1.0, report["holder_ack"])

if __name__ == "__main__":
    unittest.main()
