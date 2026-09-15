"""package-check builds every distribution without leaving anything in the checkout."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("distributions_under_test", ROOT / "tools" / "check_distributions.py")
distributions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(distributions)
UMBRELLA = "kilix-system-monitor-contracts"


class PackageResidueTests(unittest.TestCase):
    def test_root_umbrella_builds_without_touching_the_checkout(self):
        uv = os.environ.get("UV") or shutil.which("uv")
        self.assertTrue(uv, "the pinned uv must be reachable")
        details = distributions.PACKAGES[UMBRELLA]
        before = sorted(os.listdir(ROOT))
        with tempfile.TemporaryDirectory(prefix="umbrella-build-") as temporary:
            scratch = Path(temporary)
            destination = scratch / "out"
            destination.mkdir()
            distributions.build(uv, UMBRELLA, details, destination, scratch)
            built = sorted(path.name for path in destination.iterdir() if path.name.endswith((".whl", ".tar.gz")))
            private = scratch / "source" / UMBRELLA
            self.assertTrue(private.is_dir(), "the root umbrella must be built from a private copy")
            copied = sorted(os.listdir(private))
        self.assertEqual([name.split("-")[0] for name in built], ["kilix_system_monitor_contracts"] * 2)
        # The egg-info setuptools writes appeared in the private copy, not here.
        self.assertIn("kilix_system_monitor_contracts.egg-info", copied)
        self.assertEqual(sorted(os.listdir(ROOT)), before)

    def test_root_build_backend_is_accepted_only_by_its_pinned_hash(self):
        uv = os.environ.get("UV") or shutil.which("uv")
        pinned = distributions.BUILD_CONSTRAINTS.read_text(encoding="utf-8")
        digests = re.findall(r"--hash=sha256:([0-9a-f]{64})", pinned)
        self.assertEqual(len(digests), 2)
        with tempfile.TemporaryDirectory(prefix="umbrella-hash-") as temporary:
            scratch = Path(temporary)
            wrong = scratch / "build-constraints.txt"
            wrong.write_text(re.sub(r"sha256:([0-9a-f])", lambda m: "sha256:" + ("0" if m.group(1) != "0" else "1"),
                                    pinned), encoding="utf-8")
            destination = scratch / "out"
            destination.mkdir()
            with mock.patch.object(distributions, "BUILD_CONSTRAINTS", wrong):
                with self.assertRaises(RuntimeError) as caught:
                    distributions.build(uv, UMBRELLA, distributions.PACKAGES[UMBRELLA], destination, scratch)
            self.assertIn("offline build failed", str(caught.exception))
            self.assertIn("Hash mismatch for `setuptools==84.0.0`", str(caught.exception))
            self.assertEqual(sorted(path.name for path in destination.glob("*.whl")), [])

    def test_private_source_carries_no_local_state(self):
        with tempfile.TemporaryDirectory(prefix="umbrella-source-") as temporary:
            source = distributions.build_source(UMBRELLA, distributions.PACKAGES[UMBRELLA], Path(temporary))
            self.assertNotEqual(source, ROOT)
            for entry in (".git", ".venv", "kilix_system_monitor_contracts.egg-info"):
                self.assertFalse((source / entry).exists(), entry)
            for entry in ("pyproject.toml", "README.md", "LICENSE"):
                self.assertTrue((source / entry).is_file(), entry)
        for name, details in distributions.PACKAGES.items():
            if not details["isolated_build"]:
                self.assertEqual(distributions.build_source(name, details, Path("/nonexistent")), details["path"])

    def test_residue_names_every_new_entry(self):
        with tempfile.TemporaryDirectory(prefix="residue-") as temporary:
            base = Path(temporary)
            component = base / "components" / "one"
            component.mkdir(parents=True)
            watched = [base, component]
            before = distributions.snapshot(watched)
            self.assertEqual(distributions.residue(before, distributions.snapshot(watched), base), [])
            (base / "planted.egg-info").mkdir()
            (component / "build").mkdir()
            self.assertEqual(distributions.residue(before, distributions.snapshot(watched), base),
                             ["components/one/build", "planted.egg-info"])

    def test_build_residue_older_than_the_run_is_still_reported(self):
        with tempfile.TemporaryDirectory(prefix="stale-residue-") as temporary:
            base = Path(temporary)
            component = base / "components" / "one"
            component.mkdir(parents=True)
            watched = [base, component]
            self.assertEqual(distributions.build_residue(watched, base), [])
            (base / "stale.egg-info").mkdir()
            (component / "dist").mkdir()
            (component / "distribution-notes.txt").write_text("not build output", encoding="utf-8")
            before = distributions.snapshot(watched)
            # Residue already present when a run starts is invisible to the snapshot...
            self.assertEqual(distributions.residue(before, distributions.snapshot(watched), base), [])
            # ...so it is refused by name instead.
            self.assertEqual(distributions.build_residue(watched, base), ["components/one/dist", "stale.egg-info"])

    def test_main_refuses_an_entry_its_run_added_beside_a_distribution(self):
        # An entry that is not build output by name is caught only by main's
        # before-and-after snapshot of each distribution directory.
        for planted in (False, True):
            with self.subTest(planted=planted), tempfile.TemporaryDirectory(prefix="main-residue-") as temporary:
                base = Path(temporary)
                packages = {}
                for name in ("one", "two"):
                    (base / name).mkdir()
                    packages[name] = {"path": base / name, "modules": (), "version": "0.0.0",
                                      "isolated_build": False}

                def build(uv, name, details, destination, scratch, *, offline=True):
                    (destination / f"{name}-0.0.0-py3-none-any.whl").write_bytes(b"")
                    (destination / f"{name}-0.0.0.tar.gz").write_bytes(b"")
                    if planted and name == "two":
                        (details["path"] / "left-by-the-build.txt").write_text("stray", encoding="utf-8")

                with mock.patch.multiple(distributions, PACKAGES=packages, build=build,
                                         _inspect_wheel=mock.DEFAULT, _inspect_sdist=mock.DEFAULT,
                                         distribution_version=lambda _name: distributions.BUILD_BACKEND_VERSION), \
                        mock.patch.object(distributions.residue, "__defaults__", (base,)), \
                        mock.patch.object(distributions.build_residue, "__defaults__", (base,)), \
                        redirect_stdout(io.StringIO()):
                    if planted:
                        with self.assertRaises(RuntimeError) as caught:
                            distributions.main([])
                        self.assertIn("two/left-by-the-build.txt", str(caught.exception))
                    else:
                        self.assertEqual(distributions.main([]), 0)


if __name__ == "__main__":
    unittest.main()
