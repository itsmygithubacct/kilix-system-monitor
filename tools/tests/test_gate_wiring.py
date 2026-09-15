"""make check must fail when a gate is unwired, and this check must see every such removal."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gate_wiring_under_test", ROOT / "tools" / "check_gate_wiring.py")
wiring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wiring)


class GateWiringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gate-wiring-")
        self.addCleanup(self.temp.cleanup)
        self.makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    def errors_for(self, text: str) -> list[str]:
        folder = Path(tempfile.mkdtemp(dir=self.temp.name))
        (folder / "Makefile").write_text(text, encoding="utf-8")
        return wiring.wiring_errors(wiring.dry_run(folder / "Makefile"))

    def edited(self, old: str, new: str) -> str:
        self.assertEqual(self.makefile.count(old), 1, old)
        return self.makefile.replace(old, new)

    def test_this_repository_runs_every_gate(self):
        self.assertEqual(wiring.wiring_errors(wiring.dry_run(ROOT / "Makefile")), [])

    def test_a_gate_dropped_from_check_is_reported(self):
        prerequisites = next(line for line in self.makefile.splitlines() if line.startswith("check:"))
        for target in prerequisites.split(":", 1)[1].split():
            with self.subTest(dropped=target):
                text = self.edited(prerequisites, " ".join(part for part in prerequisites.split() if part != target))
                self.assertTrue(self.errors_for(text), target)

    def check_recipe_lines(self):
        """Recipe lines of check and of every target it depends on; other targets are not gates."""
        prerequisites = next(line for line in self.makefile.splitlines() if line.startswith("check:"))
        reachable = {"check", *prerequisites.split(":", 1)[1].split()}
        lines, target = [], None
        for line in self.makefile.splitlines():
            if line.startswith("\t"):
                if target in reachable:
                    lines.append(line)
            elif line and not line.startswith("#"):
                head = line.split(":", 1)[0]
                target = head if ":" in line and "=" not in head else None
        return lines

    def test_a_deleted_recipe_line_is_reported(self):
        recipe_lines = self.check_recipe_lines()
        self.assertGreaterEqual(len(recipe_lines), 13)
        for line in recipe_lines:
            with self.subTest(deleted=line.strip()[:70]):
                self.assertTrue(self.errors_for(self.edited(line + "\n", "")), line)

    def test_a_commented_out_recipe_line_is_reported(self):
        line = next(line for line in self.makefile.splitlines() if "validate_frozen_contracts.py" in line)
        self.assertTrue(self.errors_for(self.edited(line, "\t# " + line.strip())))

    def test_the_named_gates_are_the_ones_the_findings_named(self):
        self.assertIn("lease suite", wiring.GATES)
        self.assertIn("frozen bundle validator", wiring.GATES)
        prerequisites = next(line for line in self.makefile.splitlines() if line.startswith("check:"))
        lease_dropped = self.errors_for(self.edited(
            prerequisites, " ".join(part for part in prerequisites.split() if part != "lease-check")))
        self.assertEqual(lease_dropped, ["make check no longer runs the lease suite"])
        frozen_line = next(line for line in self.makefile.splitlines() if "validate_frozen_contracts.py" in line)
        frozen_deleted = self.errors_for(self.edited(frozen_line + "\n", ""))
        self.assertEqual(frozen_deleted, ["make check no longer runs the frozen bundle validator"])

    def test_make_is_available_to_ask(self):
        self.assertIsNotNone(shutil.which("make"))


if __name__ == "__main__":
    unittest.main()
