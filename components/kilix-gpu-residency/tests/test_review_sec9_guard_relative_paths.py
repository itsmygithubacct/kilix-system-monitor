"""Adopted from the security seat's P1 review control S9 (review-p1/security/controls/test_sec9_guard_relative_paths.py).

S9: the test path guard did not refuse "any path under /run/user". It is an
audit hook that inspected path arguments, but _open_parent walks to a
namespace one descriptor-relative component at a time ("/", then "run",
"user", "<uid>"), so only an absolute name or the default leaf was refused.

Adapted for fix round 1: the seat ran this only inside its own sandbox, since a
broken guard writes to /run/user. Here the probe runs in a nested bubblewrap
with its own private /run, so it is safe on any machine, and a pure in-process
check covers the descriptor-relative resolution with a stand-in directory.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import residency_containment
import residency_path_guard as guard

UID = os.geteuid()
PROBE = f"/run/user/{UID}/review-security-probe"
READER = ("import sys\n"
          "import residency_path_guard as guard\n"
          "import kilix_gpu_residency as residency\n"
          "try:\n"
          "    residency.status(namespace=sys.argv[1], ollama='http://192.0.2.1:1')\n"
          "    residency.register(app='a', model='m', backend='process', namespace=sys.argv[1],\n"
          "                       ollama='http://192.0.2.1:1').close(unload=False)\n"
          "except guard.RealMachineRefused as error:\n"
          "    print('refused', error, flush=True)\n"
          "    raise SystemExit(3)\n"
          "print('used', flush=True)\n")
# A private /run with an empty /run/user/<uid> (0700), so nothing real is ever reachable.
BWRAP = ["bwrap", "--bind", "/", "/", "--dev-bind", "/dev", "/dev", "--tmpfs", "/run", "--dir", "/run/user",
         "--perms", "0700", "--dir", f"/run/user/{UID}", "--die-with-parent"]


def nested_sandbox_works() -> bool:
    if shutil.which("bwrap") is None:
        return False
    return subprocess.run([*BWRAP, "true"], capture_output=True, timeout=30).returncode == 0


class RequiredGuardRelative(unittest.TestCase):
    @unittest.skipUnless(nested_sandbox_works(), "bubblewrap cannot make a private /run here")
    def test_the_guard_refuses_any_path_under_run_user(self):
        for namespace in (PROBE, f"/run/user/{UID}/kilix-gpu-residency-v1"):
            with self.subTest(namespace=namespace):
                done = subprocess.run([*BWRAP, sys.executable, "-c", READER, namespace], capture_output=True,
                                      text=True, timeout=60,
                                      env=residency_containment.child_env(KILIX_GPU_SYSFS_ROOT="/nonexistent-sysfs"))
                self.assertEqual(done.returncode, 3,
                                 f"S9: under the path guard, status/register(namespace={namespace!r}) was not "
                                 f"refused: {done.stdout.strip()!r} {done.stderr[-500:]!r}")

    def test_descriptor_relative_names_are_resolved_before_they_are_judged(self):
        with tempfile.TemporaryDirectory(prefix="gpu-residency-guard-fd-") as temp:
            fake = Path(temp) / "run-user"
            fake.mkdir(mode=0o700)
            parent = os.open(temp, os.O_RDONLY | os.O_DIRECTORY)
            inside = os.open(fake, os.O_RDONLY | os.O_DIRECTORY)
            before = guard.RUNTIME
            guard.RUNTIME = str(fake)  # stand in for /run/user
            try:
                for call in (lambda: os.open("x", os.O_CREAT | os.O_WRONLY, 0o600, dir_fd=inside),
                             lambda: os.mkdir("y", 0o700, dir_fd=inside),
                             lambda: os.open("run-user", os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent),
                             lambda: os.listdir(inside),
                             lambda: os.stat("x", dir_fd=inside),
                             lambda: os.unlink("x", dir_fd=inside),
                             lambda: os.rename("a", "b", src_dir_fd=parent, dst_dir_fd=inside)):
                    with self.assertRaises(guard.RealMachineRefused):
                        call()
            finally:
                guard.RUNTIME = before
                os.close(inside)
                os.close(parent)
            self.assertEqual(os.listdir(fake), [])


if __name__ == "__main__":
    unittest.main()
