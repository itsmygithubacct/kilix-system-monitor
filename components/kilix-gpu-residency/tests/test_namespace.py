"""The registry directory: created private, and refused when foreign, linked, replaced or malformed."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

import residency_containment
import kilix_gpu_residency as residency


class NamespaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gpu-residency-ns-")
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        os.chmod(self.parent, 0o700)
        self.path = self.parent / "registry"
        os.environ["KILIX_GPU_SYSFS_ROOT"] = str(self.parent / "no-sys")
        self.addCleanup(os.environ.pop, "KILIX_GPU_SYSFS_ROOT", None)

    def use(self, **kwargs):
        return residency.status(namespace=str(self.path), ollama="http://192.0.2.1:1", **kwargs)

    def refused(self, code="unavailable"):
        with self.assertRaises(residency.ResidencyError) as caught:
            self.use()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def register(self):
        return residency.register(app="a", model="m", backend="process", namespace=str(self.path),
                                  ollama="http://192.0.2.1:1")

    def test_first_use_creates_a_private_directory_and_lock(self):
        self.use()
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(self.path / "registry.lock").st_mode & 0o777, 0o600)
        self.assertEqual(sorted(os.listdir(self.path)), ["registry.lock"])

    def test_the_default_namespace_is_the_users_runtime_directory(self):
        self.assertEqual(residency.default_namespace(), f"/run/user/{os.geteuid()}/kilix-gpu-residency-v1")
        seen = []
        original = residency._open_parent

        def recording(path):
            seen.append(path)
            raise residency.ResidencyError("unavailable", "stopped before the real directory")
        residency._open_parent = recording
        self.addCleanup(setattr, residency, "_open_parent", original)
        with self.assertRaises(residency.ResidencyError):
            residency.status()
        self.assertEqual(seen, [f"/run/user/{os.geteuid()}"])

    def test_a_parent_that_is_not_private_is_refused(self):
        for mode in (0o750, 0o755, 0o777):
            with self.subTest(mode=oct(mode)):
                os.chmod(self.parent, mode)
                self.refused()
        os.chmod(self.parent, 0o700)

    def test_a_writable_ancestor_is_refused(self):
        outer = self.parent / "outer"
        outer.mkdir(mode=0o700)
        inner = outer / "inner"
        inner.mkdir(mode=0o700)
        os.chmod(outer, 0o777)
        self.path = inner / "registry"
        self.refused()

    def test_relative_and_non_canonical_paths_are_refused(self):
        for path in ("registry", "relative/registry", str(self.parent) + "/./registry",
                     str(self.parent) + "//registry", str(self.parent) + "/../x/registry", str(self.parent) + "/a b"):
            with self.subTest(path=path):
                self.path = path
                with self.assertRaises(residency.ResidencyError):
                    self.use()

    def test_a_symlinked_namespace_is_refused(self):
        real = self.parent / "real"
        real.mkdir(mode=0o700)
        os.symlink(real, self.path)
        self.refused()
        self.assertEqual(os.listdir(real), [])

    def test_a_directory_with_the_wrong_mode_or_kind_is_refused(self):
        self.path.mkdir(mode=0o700)
        os.chmod(self.path, 0o750)
        self.refused()
        os.rmdir(self.path)
        self.path.write_text("")
        self.refused()

    def test_unexpected_entries_and_unsafe_files_are_refused(self):
        self.register()
        [name] = [n for n in os.listdir(self.path) if n.endswith(".json")]
        record = self.path / name
        for damage in ("mode", "link", "symlink", "foreign-name", "malformed", "duplicate", "schema", "id"):
            with self.subTest(damage=damage):
                saved = record.read_bytes()
                if damage == "mode":
                    os.chmod(record, 0o644)
                elif damage == "link":
                    os.link(record, self.parent / "second-link")
                elif damage == "symlink":
                    record.unlink()
                    os.symlink(self.parent / "elsewhere", record)
                elif damage == "foreign-name":
                    (self.path / "notes.txt").write_text("x")
                elif damage == "malformed":
                    record.write_bytes(b"{not json")
                elif damage == "duplicate":
                    record.write_bytes(b'{"state": "idle", ' + saved.lstrip()[1:])
                elif damage == "schema":
                    value = json.loads(saved)
                    value["state"] = "sleeping"
                    record.write_text(json.dumps(value))
                elif damage == "id":
                    value = json.loads(saved)
                    value["id"] = "f" * 32
                    record.write_text(json.dumps(value))
                self.refused()
                # Repair for the next damage.
                for extra in (self.parent / "second-link", self.path / "notes.txt"):
                    if extra.exists():
                        extra.unlink()
                if record.is_symlink():
                    record.unlink()
                record.write_bytes(saved)
                os.chmod(record, 0o600)
                self.use()

    def test_damaged_pins_and_events_are_refused(self):
        self.use()
        for name, value in (("pins.json", {"version": residency.VERSION, "ollama": ["bad name"]}),
                            ("pins.json", {"version": "other", "ollama": []}),
                            ("events.json", {"version": residency.VERSION, "events": [{"id": "x"}]})):
            with self.subTest(name=name, value=value):
                path = self.path / name
                path.write_text(json.dumps(value))
                os.chmod(path, 0o600)
                self.refused()
                path.unlink()

    def test_a_writer_killed_before_its_rename_leaves_nothing_behind(self):
        self.use()
        leftover = self.path / ".0123456789abcdef0123456789abcdef.json.tmp"
        leftover.write_text("{half")
        self.use()
        self.assertFalse(leftover.exists())

    def test_a_replaced_directory_is_refused_before_anything_is_written(self):
        resident = self.register()
        moved = self.parent / "moved"
        original = residency._Namespace.check
        calls = {"n": 0}

        def replace_then_check(ns):
            calls["n"] += 1
            if calls["n"] == 2:  # after the lock was taken, before the first write
                os.rename(self.path, moved)
                self.path.mkdir(mode=0o700)
            return original(ns)
        residency._Namespace.check = replace_then_check
        self.addCleanup(setattr, residency._Namespace, "check", original)
        with self.assertRaises(residency.ResidencyError) as caught:
            resident.busy()
        self.assertEqual(caught.exception.code, "unavailable")
        self.assertEqual(os.listdir(self.path), [])

    def test_a_replaced_lock_file_is_refused(self):
        self.use()
        (self.path / "registry.lock").unlink()
        original = residency._Namespace.check
        calls = {"n": 0}

        def swap(ns):
            calls["n"] += 1
            if calls["n"] == 1:
                (self.path / "registry.lock").unlink()
                fd = os.open(self.path / "registry.lock", os.O_CREAT | os.O_WRONLY, 0o600)
                os.close(fd)
            return original(ns)
        residency._Namespace.check = swap
        self.addCleanup(setattr, residency._Namespace, "check", original)
        self.refused()

    def test_a_busy_registry_waits_then_refuses(self):
        self.use()
        fd = os.open(self.path / "registry.lock", os.O_RDWR)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        started = time.monotonic()
        with self.assertRaises(residency.ResidencyError) as caught:
            self.use(wait=0.3)
        self.assertEqual(caught.exception.code, "unavailable")
        self.assertGreaterEqual(time.monotonic() - started, 0.3)
        self.assertLess(time.monotonic() - started, 3.0)
        fcntl.flock(fd, fcntl.LOCK_UN)
        self.use(wait=0.3)

    def test_a_child_forked_while_the_registry_is_locked_does_not_keep_it_locked(self):
        self.use()
        code = textwrap.dedent("""
            import os, sys, time
            import kilix_gpu_residency as residency
            ns = residency._Namespace(sys.argv[1], 5)
            child = os.fork()
            if child == 0:
                time.sleep(3)
                os._exit(0)
            ns.close()
            print(child, flush=True)
            time.sleep(4)
        """)
        process = subprocess.Popen([sys.executable, "-c", code, str(self.path)], stdout=subprocess.PIPE,
                                   text=True, env=residency_containment.child_env())
        self.addCleanup(process.wait)
        self.addCleanup(process.stdout.close)
        process.stdout.readline()
        started = time.monotonic()
        self.use(wait=2.0)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_lock_and_registry_descriptors_are_closed_on_exec(self):
        ns = residency._Namespace(str(self.path), 1)
        self.addCleanup(ns.close)
        for fd in (ns.lock, ns.directory, ns.parent):
            self.assertTrue(fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC)


if __name__ == "__main__":
    unittest.main()
