"""Real private processes and kernel grants; no model or accelerator required."""
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

import kilix_device_lease as leases

PEER = Path(__file__).with_name("device_lease_peer.py")
PROJECT = PEER.parent.parent / "src"


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="voice-lease-test-")
        self.addCleanup(self.temp.cleanup)
        self.namespace = self.temp.name + "/leases"
        self.children = []
        self.addCleanup(self.stop_children)

    def stop_children(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream and not stream.closed:
                    stream.close()

    def acquire(self, **kwargs):
        request = dict(job_id="test-job", workload="tts-utterance", device="test-accelerator",
                       deadline=time.monotonic() + 2, namespace=self.namespace)
        request.update(kwargs)
        return leases.acquire(**request)

    def peer(self, kind="tts-utterance", job="peer", seconds=10):
        env = dict(os.environ, PYTHONPATH=str(PROJECT))
        child = subprocess.Popen([sys.executable, str(PEER), self.namespace, kind, job, str(seconds)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 env=env, text=True, bufsize=1)
        self.children.append(child)
        return child

    def event(self, child, state=None, timeout=5):
        end = time.monotonic() + timeout
        buffered = getattr(child, "_lease_events", b"")
        while time.monotonic() < end:
            if b"\n" not in buffered:
                if not select.select([child.stdout], [], [], max(0, end-time.monotonic()))[0]:
                    break
                chunk = os.read(child.stdout.fileno(), 4096)
                if not chunk:
                    self.fail(f"peer exited {child.poll()}: {child.stderr.read()}")
                buffered += chunk
                continue
            line, buffered = buffered.split(b"\n", 1)
            child._lease_events = buffered
            value = json.loads(line)
            if state is None or value.get("state") == state:
                return value
        self.fail(f"no peer event {state}")

    def send(self, child, text):
        child.stdin.write(text + "\n")
        child.stdin.flush()

    def assert_error(self, code, call):
        with self.assertRaises(leases.LeaseError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def test_clean_release_and_reused_object_cannot_release_successor(self):
        first = self.acquire()
        self.assertFalse(os.get_inheritable(first.guard_fd))
        first.release(cleanup_complete=True)
        second = self.acquire(job_id="successor", workload="stt-job", device="different-device-alias")
        try:
            first.release(cleanup_complete=True)
            self.assert_error("lost-lease", first.check)
            second.check()
        finally:
            second.release(cleanup_complete=True)

    def test_context_exit_quarantines_without_cleanup_acknowledgement(self):
        with self.acquire() as lease:
            lease.check()
        self.assert_error("unavailable", self.acquire)
        value = json.loads((Path(self.namespace)/"state.json").read_text())
        self.assertEqual(value["active"]["state"], "held")

    def test_explicit_cleanup_in_context_releases_normally(self):
        with self.acquire() as lease:
            lease.release(cleanup_complete=True)
        next_lease = self.acquire(workload="llm-turn")
        next_lease.release(cleanup_complete=True)

    def test_deadline_cancel_and_disconnect_while_queued(self):
        holder = self.acquire()
        try:
            for command in ("cancel", "disconnect", "deadline"):
                peer = self.peer(job=command, seconds=.25 if command == "deadline" else 5)
                self.event(peer, "queued")
                if command == "cancel":
                    self.send(peer, "cancel")
                elif command == "disconnect":
                    peer.stdin.close()
                result = self.event(peer, "refused")
                self.assertEqual(result["code"], "deadline" if command == "deadline" else "cancelled")
                self.assertEqual(peer.wait(timeout=3), 0)
            self.assertEqual(json.loads((Path(self.namespace)/"state.json").read_text())["queue"], [])
        finally:
            holder.release(cleanup_complete=True)

    def test_workloads_rotate_and_fifo_is_preserved(self):
        holder = self.acquire()
        peers = {}
        for job, kind in (("tts-a", "tts-utterance"), ("tts-b", "tts-utterance"),
                          ("stt-a", "stt-job"), ("stt-b", "stt-job"), ("llm-a", "llm-turn")):
            peers[job] = self.peer(kind, job)
            self.event(peers[job], "queued")
        holder.release(cleanup_complete=True)
        for job in ("stt-a", "llm-a", "tts-a", "stt-b", "tts-b"):
            self.assertEqual(self.event(peers[job], "held")["job"], job)
            self.send(peers[job], "release")
            self.event(peers[job], "released")
            self.assertEqual(peers[job].wait(timeout=3), 0)

    def test_workload_queue_capacity_cannot_consume_other_workload_slots(self):
        holder = self.acquire()
        queued = []
        try:
            for i in range(leases.MAX_WORKLOAD_QUEUE):
                child = self.peer(job=f"queued-{i}")
                self.event(child, "queued")
                queued.append(child)
            self.assert_error("queue-full", self.acquire)
            other = self.peer("stt-job", "other-workload")
            self.event(other, "queued")
            self.send(other, "cancel")
            self.assertEqual(self.event(other, "refused")["code"], "cancelled")
            for child in queued:
                self.send(child, "cancel")
                self.event(child, "refused")
        finally:
            holder.release(cleanup_complete=True)

    def test_queued_process_death_is_pruned_by_kernel_ticket_not_pid(self):
        holder = self.acquire()
        child = self.peer()
        event = self.event(child, "queued")
        child.kill()
        child.wait(timeout=3)
        holder.release(cleanup_complete=True)
        successor = self.acquire()
        self.assertFalse((Path(self.namespace)/(event["ticket"] + ".ticket")).exists())
        successor.release(cleanup_complete=True)

    def contended_refusal(self, *, cancelled):
        holder = self.acquire(deadline=time.monotonic()+5)
        child = self.peer(job="contended", seconds=5 if cancelled else .5)
        queued = self.event(child, "queued")
        program = ("import fcntl,os,sys;f=os.open(sys.argv[1],os.O_RDWR);"
                   "fcntl.flock(f,fcntl.LOCK_EX);print('held',flush=True);sys.stdin.read(1)")
        blocker = subprocess.Popen([sys.executable, "-c", program,
                                    str(Path(self.temp.name)/".leases.lease-v1.anchor")],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.children.append(blocker)
        self.assertTrue(select.select([blocker.stdout], [], [], 2)[0])
        self.assertEqual(blocker.stdout.readline(), b"held\n")
        began = time.monotonic()
        try:
            if cancelled:
                self.send(child, "cancel")
            result = self.event(child, "refused", timeout=.8)
            self.assertEqual(result["code"], "cancelled" if cancelled else "deadline")
            self.assertIsNone(blocker.poll())
            if cancelled:
                self.assertLess(time.monotonic()-began, .3)
        finally:
            blocker.communicate(b"x", timeout=2)
            holder.release(cleanup_complete=True)
        successor = self.acquire()
        self.assertFalse((Path(self.namespace)/(queued["ticket"]+".ticket")).exists())
        successor.release(cleanup_complete=True)

    def test_cancelled_ticket_does_not_wait_again_for_busy_anchor(self):
        self.contended_refusal(cancelled=True)

    def test_expired_ticket_does_not_wait_again_for_busy_anchor(self):
        self.contended_refusal(cancelled=False)

    def test_inherited_guard_survives_acknowledged_parent_release(self):
        lease = self.acquire()
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, pass_fds=(lease.guard_fd,))
        self.children.append(child)
        lease.release(cleanup_complete=True)
        self.assert_error("deadline", lambda: self.acquire(deadline=time.monotonic()+.15))
        child.terminate()
        child.wait(timeout=3)
        self.acquire().release(cleanup_complete=True)

    def test_provider_crash_does_not_authorize_restart(self):
        child = self.peer()
        self.event(child, "held")
        child.kill()
        child.wait(timeout=3)
        self.assert_error("unavailable", self.acquire)

    def test_supervisor_loss_with_orphan_engine_is_quarantined(self):
        # Only a new private observer becomes a subreaper, never the library or
        # this embedding test process. Every child named below belongs to it.
        code = r'''
import ctypes,json,os,select,signal,subprocess,sys,time
from kilix_device_lease import acquire,LeaseError
assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
peer,namespace=sys.argv[1:]
provider=subprocess.Popen([sys.executable,peer,namespace,'tts-utterance','owner','10'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
unrelated=subprocess.Popen(['/bin/sleep','30']); owned=[]
try:
 assert select.select([provider.stdout],[],[],5)[0]
 grant=json.loads(provider.stdout.readline()); assert grant['state']=='held'
 provider.stdin.write('supervisor\n');provider.stdin.flush()
 assert select.select([provider.stdout],[],[],5)[0]
 descendants=json.loads(provider.stdout.readline());owned=[descendants['supervisor'],descendants['engine']]
 provider.kill();provider.wait(timeout=3)
 os.kill(owned[0],signal.SIGKILL);os.waitpid(owned[0],0)
 assert os.path.exists('/proc/'+str(owned[1]))
 try:acquire(job_id='successor',workload='stt-job',device='test-accelerator',deadline=time.monotonic()+1,namespace=namespace)
 except LeaseError as error:assert error.code=='unavailable'
 else:raise AssertionError('orphan engine overlapped successor')
 assert unrelated.poll() is None
 print(json.dumps({'supervisor_killed':True,'engine_alive':True,'successor_refused':True,'unrelated_preserved':True}))
finally:
 if provider.poll() is None:provider.kill();provider.wait(timeout=3)
 for pid in owned:
  try:os.kill(pid,signal.SIGKILL)
  except ProcessLookupError:pass
  try:os.waitpid(pid,0)
  except ChildProcessError:pass
 unrelated.terminate();unrelated.wait(timeout=3)
 for stream in (provider.stdin,provider.stdout,provider.stderr):stream.close()
'''
        result = subprocess.run([sys.executable, "-c", code, str(PEER), self.namespace],
                                env=dict(os.environ, PYTHONPATH=str(PROJECT)), capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["successor_refused"])

    def test_replaced_resource_preserves_both_entries(self):
        lease = self.acquire()
        path = Path(self.namespace)/"accelerator.lock"
        saved = path.with_name("saved-lock")
        path.rename(saved)
        path.write_bytes(b"unrelated")
        path.chmod(0o600)
        self.assert_error("unavailable", self.acquire)
        self.assertEqual(path.read_bytes(), b"unrelated")
        self.assertTrue(saved.exists())
        lease.release()

    def test_missing_replaced_or_symlink_namespace_is_not_recreated(self):
        lease = self.acquire()
        saved = Path(self.temp.name)/"saved-namespace"
        Path(self.namespace).rename(saved)
        self.assert_error("unavailable", self.acquire)
        self.assertFalse(Path(self.namespace).exists())
        Path(self.namespace).symlink_to(saved, target_is_directory=True)
        self.assert_error("unavailable", self.acquire)
        self.assertTrue(Path(self.namespace).is_symlink())
        lease.release()

    def test_unsafe_namespace_and_hardlinked_resource_are_refused(self):
        lease = self.acquire()
        directory = Path(self.namespace)
        directory.chmod(0o755)
        self.assert_error("unavailable", self.acquire)
        directory.chmod(0o700)
        os.link(directory/"accelerator.lock", directory/"hardlink")
        self.assert_error("unavailable", self.acquire)
        lease.release()

    def test_reused_unrelated_descriptor_is_preserved(self):
        lease = self.acquire()
        fd = lease.guard_fd
        os.close(fd)
        replacement = os.open("/dev/null", os.O_RDONLY)
        if replacement != fd:
            os.dup2(replacement, fd)
            os.close(replacement)
        try:
            self.assert_error("lost-lease", lease.release)
            os.fstat(fd)
        finally:
            os.close(fd)

    def test_invalid_input_is_typed_without_initializing_namespace(self):
        for key, value in (("job_id", "../private"), ("job_id", "x"*97), ("device", []),
                           ("workload", "microphone"), ("workload", []), ("deadline", True),
                           ("deadline", float("nan")), ("deadline", 10**1000),
                           ("namespace", self.namespace+"/../other"), ("namespace", ""), ("cancelled", 42)):
            with self.subTest(key=key, value=str(value)[:50]):
                self.assert_error("invalid-request", lambda: self.acquire(**{key: value}))
        self.assertFalse(Path(self.namespace).exists())

    def test_active_deadline_and_cancel_need_cleanup_acknowledgement(self):
        flags = {"cancel": False}
        lease = self.acquire(cancelled=lambda: flags["cancel"])
        flags["cancel"] = True
        self.assert_error("cancelled", lease.check)
        lease.release(cleanup_complete=True)
        deadline = self.acquire(deadline=time.monotonic()+.08)
        time.sleep(.1)
        self.assert_error("deadline", deadline.check)
        deadline.release(cleanup_complete=True)
        self.acquire().release(cleanup_complete=True)

    def test_duplicate_or_unsupported_record_schema_is_refused(self):
        self.acquire().release(cleanup_complete=True)
        state = Path(self.namespace)/"state.json"
        original = state.read_bytes()
        for value in (b'{"version":1,"version":2}', b'[]', b'{"version":NaN}', b'{"version":"future"}'):
            state.write_bytes(value)
            self.assert_error("unavailable", self.acquire)
            self.assertEqual(state.read_bytes(), value)
        state.write_bytes(original)
        self.acquire().release(cleanup_complete=True)


if __name__ == "__main__":
    unittest.main()
