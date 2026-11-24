"""Adopted from the security seat's P1 review controls (review-p1/security/controls/test_sec10_pid_namespace.py).
Only the Required controls are kept; the Observed ones recorded the defects this round fixed.

S10: a participant in another PID namespace deletes everyone else's records.

Liveness is "pid and start time in this process's /proc". A process in another
PID namespace (a bwrap or flatpak sandbox that shares the runtime directory, as
the design's puzzllm sandbox case contemplates) cannot see the host's holders,
so the first read it makes (status, register, ensure_room) prunes every host
record as dead. Their owners then get "gone", their loading reservations vanish,
and their still-running model processes drop to "unmanaged". Nothing in the
module notices or refuses a namespace mismatch, and the README and contract do
not say that all participants must share one PID namespace.

The other namespace is real: bubblewrap --unshare-pid with its own /proc,
started from this (already sandboxed) test process.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import unittest

import residency_containment
import kilix_gpu_residency as residency
from fakes import GIB, ResidencyCase

READER = ("import os, sys\n"
          "import kilix_gpu_residency as residency\n"
          "print('pid', os.getpid(), flush=True)\n"
          "value = residency.status(namespace=sys.argv[1], ollama='http://192.0.2.1:1')\n"
          "print('residents', len(value.residents), flush=True)\n")
BWRAP = ["bwrap", "--bind", "/", "/", "--dev-bind", "/dev", "/dev", "--unshare-pid", "--proc", "/proc",
         "--die-with-parent"]


def nested_pid_namespace_works() -> bool:
    if shutil.which("bwrap") is None:
        return False
    done = subprocess.run([*BWRAP, "true"], capture_output=True, timeout=30)
    return done.returncode == 0


@unittest.skipUnless(nested_pid_namespace_works(), "bubblewrap cannot make a PID namespace here")
class _OtherNamespace(ResidencyCase):
    def read_from_another_namespace(self):
        resident, process = self.model("kilix-chat", "LLaDA-MoE-7B", 2 * GIB)
        self.assertEqual(len(self.records()), 1)
        done = subprocess.run([*BWRAP, sys.executable, "-c", READER, self.namespace], capture_output=True, text=True,
                              env=residency_containment.child_env(KILIX_GPU_SYSFS_ROOT=str(self.scratch / "no-sys")),
                              timeout=60)
        # The branch under test: the reader really ran in another PID namespace (its own pid is tiny).
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertLess(int(done.stdout.split()[1]), 10, done.stdout)
        return resident, process, done


class RequiredOtherNamespace(_OtherNamespace):
    def test_a_participant_that_cannot_see_a_holder_does_not_delete_its_record(self):
        resident, _, done = self.read_from_another_namespace()
        self.assertEqual(len(self.records()), 1,
                         f"S10: one status() call from another PID namespace deleted record {resident.id} whose "
                         f"holder is alive in this namespace (reader said {done.stdout.split()[-2:]})")
