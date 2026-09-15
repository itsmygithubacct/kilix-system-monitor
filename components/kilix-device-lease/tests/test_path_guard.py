"""No lease test process, or child it starts, can reach the real runtime directory.

Every planted call aims at a private directory whose name carries the default
namespace leaf, never at the real runtime directory: with the guard, each call
must be refused and the directory left byte-identical; without it, the same
calls must change the directory, so the guarded arm had something to refuse.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import lease_containment
from lease_containment import guard

TESTS = Path(__file__).resolve().parent

PLANTED = r"""
import json, os, pathlib, sys
target = sys.argv[1]
results = {"guarded": bool(getattr(sys, "_kilix_lease_path_guard", False))}
def attempt(name, call):
    try:
        call()
        results[name] = "done"
    except BaseException as error:
        results[name] = type(error).__name__
attempt("makedirs", lambda: os.makedirs(target + "/made/deeper"))
attempt("mkdir", lambda: os.mkdir(target + "/single"))
attempt("rename", lambda: os.rename(target + "/rename-me", target + "/renamed"))
attempt("unlink", lambda: os.unlink(target + "/unlink-me"))
attempt("open", lambda: open(target + "/opened", "w").close())
attempt("pathlib", lambda: pathlib.Path(target, "touched").touch())
attempt("mkfifo", lambda: os.mkfifo(target + "/fifo"))
attempt("posix.mknod", lambda: __import__("posix").mknod(target + "/node", 0o600 | 0o10000))
print(json.dumps(results))
"""

FIXTURE = r"""
import os, sys
target = sys.argv[1]
os.mkdir(target, 0o700)
for name in ("rename-me", "unlink-me"):
    with open(os.path.join(target, name), "wb") as handle:
        handle.write(b"canary")
"""
CALLS = ("makedirs", "mkdir", "rename", "unlink", "open", "pathlib", "mkfifo", "posix.mknod")


class PathGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="device-lease-guard-")
        self.addCleanup(self.temp.cleanup)
        # A private stand-in for the real namespace: same leaf, private parent.
        self.target = self.temp.name + "/" + guard.LEAF
        # This process refuses to remove the leaf, so a child without the guard
        # removes the stand-in before the temporary directory is cleaned up.
        self.addCleanup(self.child, "import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)",
                        lease_containment.env_without_path_guard())

    def child(self, code, env):
        done = subprocess.run([sys.executable, "-c", code, self.target], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=30, env=env)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout

    def snapshot(self):
        # Listed through a child without the guard, which is how the fixture was made.
        code = ("import json, os, sys; t = sys.argv[1]; print(json.dumps(sorted("
                "(os.path.relpath(os.path.join(d, n), t), open(os.path.join(d, n), 'rb').read().decode() "
                "if os.path.isfile(os.path.join(d, n)) else None) for d, ds, fs in os.walk(t) for n in ds + fs)))")
        return json.loads(self.child(code, lease_containment.env_without_path_guard()))

    def test_children_started_by_the_suite_refuse_every_planted_call(self):
        self.child(FIXTURE, lease_containment.env_without_path_guard())
        before = self.snapshot()
        results = json.loads(self.child(PLANTED, lease_containment.child_env()))
        self.assertEqual(results, {"guarded": True, **{call: "RealRuntimePathRefused" for call in CALLS}})
        self.assertEqual(self.snapshot(), before)

    def test_children_without_the_guard_would_change_the_directory(self):
        self.child(FIXTURE, lease_containment.env_without_path_guard())
        before = self.snapshot()
        results = json.loads(self.child(PLANTED, lease_containment.env_without_path_guard()))
        self.assertEqual(results, {"guarded": False, **{call: "done" for call in CALLS}})
        self.assertNotEqual(self.snapshot(), before)

    def test_this_process_refuses_the_leaf_and_the_runtime_directory(self):
        with self.assertRaises(guard.RealRuntimePathRefused):
            Path(self.target).mkdir()
        self.assertEqual(guard.refused[-1], ("os.mkdir", self.target))
        for name in ("/run/user", "/run/user/1000", "/run/../run/user/1000/x", "/run//user/0",
                     "kilix-device-leases-v1", ".kilix-device-leases-v1.lease-v1.anchor",
                     "/private/kilix-device-leases-v1-shadow"):
            with self.subTest(refused=name):
                self.assertTrue(guard.refuses(name))
        for name in ("/run/users", "/run", "/private/leases", ".leases.lease-v1.anchor", "run/user/1000",
                     "accelerator.lock"):
            with self.subTest(allowed=name):
                self.assertFalse(guard.refuses(name))

    def test_every_test_module_installs_the_guard_and_no_child_bypasses_it(self):
        bypass = {"env_without_path_guard"}
        variable = "PYTHON" + "PATH"  # spelt apart so this file does not match itself
        modules = sorted(TESTS.glob("test_*.py"))
        self.assertIn(Path(__file__).resolve(), modules)
        for path in modules:
            with self.subTest(module=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                imported = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                            for alias in node.names} | {node.module for node in ast.walk(tree)
                                                        if isinstance(node, ast.ImportFrom)}
                self.assertIn("lease_containment", imported)
                self.assertNotIn(variable, path.read_text(encoding="utf-8"))
                users = {node.attr for node in ast.walk(tree)
                         if isinstance(node, ast.Attribute) and node.attr in bypass}
                if path.name not in ("test_path_guard.py", "test_interface_document.py"):
                    self.assertEqual(users, set())


if __name__ == "__main__":
    unittest.main()
