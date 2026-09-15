"""package-check builds every distribution without leaving anything in the checkout."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
