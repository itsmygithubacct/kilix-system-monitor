"""Each guard in the frozen-bundle validator must be able to fire.

The validator's own planted controls run inside main(); a guard removed together
with its control would pass unseen. These tests reach each guard from outside:
through a copied bundle that main() must refuse, or by calling the rule with a
document it must refuse, beside an unaltered positive control.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frozen = load("frozen_contracts_under_test", "tools/validate_frozen_contracts.py")
candidate = load("candidate_contracts_under_test", "tools/validate_candidate.py")
BUNDLE = ROOT / "contracts" / "v1"


def document(relative: str):
    return json.loads((BUNDLE / relative).read_text(encoding="utf-8"))


def run_main(bundle: Path) -> str:
    """Run the validator over a bundle directory; return its PASS line or its failure."""
    output = io.StringIO()
    with mock.patch.object(frozen, "BUNDLE", bundle), redirect_stdout(output):
        try:
            frozen.main()
        except frozen.Failure as error:
            return f"FAIL: {error}"
    return output.getvalue().strip()


class FrozenValidatorGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="frozen-validator-")
        self.addCleanup(self.temp.cleanup)

    def copy_bundle(self) -> Path:
        target = Path(self.temp.name) / "v1"
        shutil.copytree(BUNDLE, target)
        return target

    def test_unaltered_bundle_copy_passes(self):
        self.assertTrue(run_main(self.copy_bundle()).startswith("PASS"))

    def test_manifest_digest_is_pinned_outside_the_bundle(self):
        # A member changed and the manifest rewritten to match it is a
        # self-consistent bundle; only a pin kept outside it can refuse that.
        bundle = self.copy_bundle()
        readme = bundle / "README.md"
        readme.write_bytes(readme.read_bytes() + b"\nAn unreviewed line.\n")
        files = {path.relative_to(bundle).as_posix(): path.read_bytes()
                 for path in sorted(bundle.rglob("*")) if path.is_file() and path.name != frozen.MANIFEST}
        (bundle / frozen.MANIFEST).write_bytes(frozen.manifest_bytes(files))
        result = run_main(bundle)
        self.assertTrue(result.startswith("FAIL"), result)
        self.assertIn("differs from its pinned digest", result)
        manifest = (bundle / frozen.MANIFEST).read_bytes()
        self.assertNotEqual(hashlib.sha256(manifest).hexdigest(), frozen.EXPECTED_MANIFEST_SHA256)

    def test_measured_accelerator_row_needs_measured_vram(self):
        validator = frozen.Draft202012Validator(document(frozen.PROFILES_SCHEMA),
                                                format_checker=frozen.FormatChecker())
        row = document("profiles/res02-measured/whisper-tiny-cuda.json")
        self.assertEqual(row["profiles"][0]["backend"], "cuda")
        self.assertEqual(frozen.measured_policy_errors(row), [])
        missing = copy.deepcopy(row)
        missing["profiles"][0]["requirements"]["vram_peak_bytes"] = None
        # The schema allows null VRAM; the measured policy is the only guard.
        self.assertEqual(frozen.profile_errors(validator, missing), [])
        profile_id = row["profiles"][0]["profile_id"]
        self.assertEqual(frozen.measured_policy_errors(missing),
                         [f"{profile_id}: accelerator row has no vram_peak_bytes"])

    def test_private_path_scan_finds_each_marker(self):
        for member in sorted(BUNDLE.rglob("*")):
            if member.is_file():
                self.assertEqual(frozen.text_errors(member.read_bytes()), [], member.name)
        for segment in ("home", "tmp", "root", "mnt"):
            with self.subTest(segment=segment):
                text = "measured under " + "/" + segment + "/" + "someone" + " by a tool"
                self.assertTrue(frozen.text_errors(text.encode("utf-8")))
        self.assertEqual(frozen.text_errors(b"measured under a private scratch root"), [])
        # A marker late in a file must be found too, not only one near the start:
        # after the largest bundle member, and after far more clean text than a
        # typical read buffer holds.
        largest = max((member for member in BUNDLE.rglob("*") if member.is_file()),
                      key=lambda member: member.stat().st_size)
        padding = b"measured under a private scratch root\n" * 1000
        self.assertGreater(largest.stat().st_size, 4096)
        for prefix, where in ((largest.read_bytes(), "after " + largest.name), (padding, "after clean text")):
            for segment in ("home", "tmp", "root", "mnt"):
                with self.subTest(segment=segment, where=where):
                    marker = ("/" + segment + "/" + "someone").encode("utf-8")
                    self.assertTrue(frozen.text_errors(prefix + marker))

    def test_private_path_scan_finds_a_marker_anywhere_in_a_large_file(self):
        # A scan that keeps only a window of a file, from either end, misses a
        # marker outside it. Each file here holds 8 MiB of clean text on each side
        # of its marker, far more than any window a truncating scan could keep, and
        # the marker sits near the start, in the middle or at the end.
        clean = b"measured under a private scratch root\n"
        padding = clean * (8 * 1024 * 1024 // len(clean) + 1)
        self.assertEqual(frozen.text_errors(padding + padding), [])
        for segment in ("home", "tmp", "root", "mnt"):
            marker = ("/" + segment + "/" + "someone").encode("utf-8")
            placements = {
                "at byte 15": b"measured under " + marker + b" by a tool\n" + padding + padding,
                "in the middle": padding + marker + padding,
                "at the end": padding + padding + marker,
            }
            for where, data in placements.items():
                with self.subTest(segment=segment, where=where):
                    self.assertTrue(frozen.text_errors(data))

    def test_evidence_refusals_come_from_the_companion_semantic_rules(self):
        validator = frozen.Draft202012Validator(document(frozen.PROFILES_SCHEMA),
                                                format_checker=frozen.FormatChecker())
        refusals = document("fixtures/profiles/REFUSALS.json")
        expected = {
            "profile-estimate-qualified.json": "qualified profile lacks measured evidence",
            "profile-performance-without-measured-evidence.json": "performance number lacks measured evidence",
        }
        for name, reason in expected.items():
            with self.subTest(fixture=name):
                self.assertEqual(refusals[name]["expected"], reason)
                refused = document("fixtures/profiles/invalid/" + name)
                self.assertIn(reason, frozen.semantic_errors(refused))
                self.assertIn(reason, frozen.profile_errors(validator, refused))
                self.assertTrue(candidate.semantic_errors("plebian.models.profiles/v1", refused))

    def test_frozen_and_candidate_rules_agree_on_every_evidence_condition(self):
        base = document("profiles/res02-measured/small-en-us.json")
        base["qualification_eligible"] = True
        base["profiles"][0]["qualification"] = "qualified"
        base["profiles"][0]["artifact"]["content_sha256"] = "0" * 64
        base["profiles"][0]["artifact"]["license_decision_id"] = "planted-licence-decision"

        def decisions(doc):
            return (bool(frozen.semantic_errors(doc)),
                    bool(candidate.semantic_errors("plebian.models.profiles/v1", doc)))

        self.assertEqual(decisions(base), (False, False))
        cases = {"catalog not eligible": ("qualification_eligible", False)}
        for field in ("command", "fixture", "measured_at", "raw_evidence_sha256", "reference_hardware_class"):
            cases[f"evidence.{field} missing"] = ("evidence", field)
        for field in ("content_sha256", "license_decision_id"):
            cases[f"artifact.{field} missing"] = ("artifact", field)
        cases["confidence estimated"] = ("confidence", "estimated")
        for label, (where, what) in cases.items():
            with self.subTest(case=label):
                doc = copy.deepcopy(base)
                profile = doc["profiles"][0]
                if where == "qualification_eligible":
                    doc["qualification_eligible"] = what
                elif where == "confidence":
                    profile["evidence"]["confidence"] = what
                else:
                    profile[where][what] = None
                self.assertEqual(decisions(doc), (True, True))
        unmeasured = copy.deepcopy(document("fixtures/profiles/valid/unqualified-estimate.json"))
        performance = unmeasured["profiles"][0]["performance"]
        performance[sorted(performance)[0]] = 1.0
        self.assertEqual(decisions(unmeasured), (True, True))

    def test_main_refuses_when_frozen_and_candidate_rules_disagree(self):
        real = frozen.semantic_errors

        def drifted(doc):
            # Adds a reason only for a schema-refused fixture, so no fixture's
            # declared refusal changes and only the differential check can see it.
            profiles = doc.get("profiles", []) if isinstance(doc, dict) else []
            extra = ["planted drift"] if any(isinstance(p, dict) and "backend" not in p for p in profiles) else []
            return real(doc) + extra

        with mock.patch.object(frozen, "semantic_errors", drifted):
            result = run_main(BUNDLE)
        self.assertTrue(result.startswith("FAIL"), result)
        self.assertIn("frozen and candidate profile rules disagree", result)


if __name__ == "__main__":
    unittest.main()
