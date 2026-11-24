"""Adopted from the security seat's P1 review controls (review-p1/security/controls/test_sec7_foreign_group.py).
Only the Required controls are kept; the Observed ones recorded the defects this round fixed.

S7: any caller can enrol any of the user's process groups for TERM/KILL.

register(pgid=N) checks only that N leads a live group and is not the caller's
own group or group 1. It does not check that the group is the caller's, or
descends from it. A process listed as unmanaged ("never touched") stops being
unmanaged the moment some other process registers its pgid, after which any
ensure_room may TERM and KILL the whole group. The record follows the group's
leader, so it outlives the registrant.

Here a short-lived "app" (a separate Python process, so the victim is not its
descendant) registers a sibling group that it did not start, marks it idle and
exits; a later ensure_room by a different app signals the group.
"""
from __future__ import annotations

import signal
import subprocess
import sys
import textwrap

import residency_containment
import kilix_gpu_residency as residency
from fakes import GIB, ResidencyCase

REGISTRANT = textwrap.dedent("""
    import sys
    import kilix_gpu_residency as residency
    # The child sees no GPUs (an empty sysfs), so it never loads NVML; it names the GPU by PCI address.
    namespace, url, pgid = sys.argv[1], sys.argv[2], int(sys.argv[3])
    try:
        # Adapted for fix round 1: no device (a PCI address must now be a GPU the child can see, and it
        # sees none), so the only reason left to refuse is the group's ancestry.
        resident = residency.register(app="careless-app", model="not-mine", backend="process",
                                      pgid=pgid, namespace=namespace, ollama=url)
        resident.idle()
    except residency.ResidencyError as error:
        print("refused", error.code, flush=True)
        raise SystemExit(3)
    print("registered", resident.id, flush=True)
""")


class _Foreign(ResidencyCase):
    def enrol_stranger(self):
        stranger = self.spawn(2 * GIB)  # e.g. a game or another tool using the GPU
        self.assertIn(stranger.pid, [u.pid for u in self.status().unmanaged])  # unmanaged before
        done = subprocess.run([sys.executable, "-c", REGISTRANT, self.namespace, self.ollama.url, str(stranger.pid)],
                              env=residency_containment.child_env(KILIX_GPU_SYSFS_ROOT=str(self.scratch / "no-sys")),
                              capture_output=True, text=True, timeout=60)
        return stranger, done


class RequiredForeignGroup(_Foreign):
    def test_register_refuses_a_group_the_caller_did_not_start(self):
        stranger, done = self.enrol_stranger()
        self.assertIn("refused invalid-request", done.stdout)
        self.assertEqual(done.returncode, 3,
                         f"S7: a process registered pgid {stranger.pid}, a group it neither leads nor started "
                         f"(not its descendant), as its own signal-unloadable resident: {done.stdout.strip()!r}")
