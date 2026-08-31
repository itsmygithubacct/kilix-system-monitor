from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tools.measure import profile


HARDWARE_FIXTURE = (
    profile.ROOT / "contracts/p1-candidate/fixtures/hardware/h0-cpu-only.json"
)


def write_private_json(path: Path, document: dict[str, Any]) -> None:
    path.write_bytes(profile._canonical_bytes(document))
    path.chmod(0o600)


def provider_record(root: Path, hardware: dict[str, Any]) -> dict[str, Any]:
    artifact = root / "artifact.bin"
    fixture = root / "fixture.bin"
    hardware_path = root / "hardware.json"
    return {
        "schema": "plebian.models.provider-measurement/v1",
        "measurement_id": "demo-measurement-20260831",
        "catalog_id": "demo-catalog-20260831",
        "profile_id": "demo-tts-cpu",
        "profile_version": "0.1.0",
        "provider": "demo-provider",
        "task": "tts",
        "backend": "cpu",
        "measured_at": "2026-08-31T12:00:00Z",
        "safety_margin_basis_points": 2000,
        "artifact": {
            "artifact_id": "demo-small",
            "version": "v1",
            "content_sha256": profile._sha256(artifact.read_bytes()),
            "license_decision_id": "demo-small-license-20260831",
        },
        "fixture": {
            "fixture_id": "tts-short-v1",
            "content_sha256": profile._sha256(fixture.read_bytes()),
        },
        "hardware": {
            "snapshot_id": hardware.get("snapshot_id", "fixture:h0-cpu-only"),
            "snapshot_sha256": profile._sha256(hardware_path.read_bytes()),
        },
        "command": {
            "command_id": "demo.measure.tts",
            "argv_sha256": "a" * 64,
            "executable_sha256": "b" * 64,
        },
        "reported_requirements": {
            "architecture": "x86_64",
            "download_bytes": 11,
            "disk_installed_bytes": 12,
            "temporary_bytes": 13,
            "ram_peak_bytes": 14,
            "vram_peak_bytes": 15,
        },
        "reported_performance": {
            "first_result_ms": 16,
            "realtime_factor": 0.5,
            "tokens_per_second": 17,
        },
        "measurement_boundary": {
            "status": "unaccepted",
            "method": "provider-native",
            "receipt_sha256": None,
            "reported_only": True,
        },
    }


def prepare(
    root: Path, hardware: dict[str, Any] | None = None
) -> tuple[list[str], dict[str, Path], dict[str, Any]]:
    paths = {
        "artifact": root / "artifact.bin",
        "fixture": root / "fixture.bin",
        "hardware": root / "hardware.json",
        "provider": root / "provider.json",
        "receipt": root / "receipt.json",
        "profile": root / "profile.json",
    }
    paths["artifact"].write_bytes(b"artifact-bytes")
    paths["fixture"].write_bytes(b"fixture-bytes")
    if hardware is None:
        hardware = json.loads(HARDWARE_FIXTURE.read_bytes())
    write_private_json(paths["hardware"], hardware)
    record = provider_record(root, hardware)
    write_private_json(paths["provider"], record)
    arguments = [
        "--provider-evidence",
        str(paths["provider"]),
        "--artifact",
        str(paths["artifact"]),
        "--fixture",
        str(paths["fixture"]),
        "--hardware-snapshot",
        str(paths["hardware"]),
        "--intake-receipt",
        str(paths["receipt"]),
        "--profile-catalog",
        str(paths["profile"]),
    ]
    return arguments, paths, record


class ProfileIntakeTests(unittest.TestCase):
    def test_success_records_but_does_not_promote_provider_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            previous_umask = os.umask(0o777)
            try:
                status = profile.main(arguments)
            finally:
                os.umask(previous_umask)
            self.assertEqual(status, 0)
            self.assertEqual(stat.S_IMODE(paths["receipt"].stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(paths["profile"].stat().st_mode), 0o600)

            receipt = json.loads(paths["receipt"].read_bytes())
            catalog = json.loads(paths["profile"].read_bytes())
            generated = catalog["profiles"][0]
            self.assertEqual(receipt["provider_records"], {
                "accepted_as_measurement": 0,
                "recorded": 1,
                "total": 1,
            })
            self.assertEqual(receipt["byte_identities"]["matched"], 3)
            self.assertEqual(receipt["byte_identities"]["total"], 3)
            self.assertEqual(
                receipt["promotion"]["provider_measurements"],
                {"promoted": 0, "total": 9},
            )
            self.assertEqual(receipt["outputs"], {"committed": 2, "total": 2})
            self.assertFalse(catalog["qualification_eligible"])
            self.assertEqual(generated["qualification"], "unqualified")
            self.assertEqual(generated["evidence"]["confidence"], "unknown")
            self.assertIsNone(generated["evidence"]["command"])
            self.assertIsNone(generated["evidence"]["fixture"])
            self.assertIsNone(generated["evidence"]["measured_at"])
            self.assertIsNone(generated["evidence"]["reference_hardware_class"])
            self.assertIsNone(generated["artifact"]["license_decision_id"])
            self.assertEqual(
                [generated["requirements"][key] for key in (
                    "download_bytes",
                    "disk_installed_bytes",
                    "temporary_bytes",
                    "ram_peak_bytes",
                    "vram_peak_bytes",
                )],
                [None] * 5,
            )
            self.assertEqual(set(generated["performance"].values()), {None})
            self.assertEqual(
                generated["evidence"]["raw_evidence_sha256"],
                profile._sha256(paths["provider"].read_bytes()),
            )

    def test_artifact_digest_mismatch_commits_zero_of_two_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            paths["artifact"].write_bytes(b"different-artifact")
            self.assertEqual(profile.main(arguments), 2)
            self.assertFalse(paths["receipt"].exists())
            self.assertFalse(paths["profile"].exists())

    def test_full_hardware_schema_and_privacy_are_enforced(self) -> None:
        invalid = {
            "schema": "plebian.hardware/v1",
            "snapshot_id": "fixture:h0-cpu-only",
            "hostname": "private-machine",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root, invalid)
            self.assertEqual(profile.main(arguments), 2)
            self.assertFalse(paths["receipt"].exists())
            self.assertFalse(paths["profile"].exists())

    def test_duplicate_and_noncanonical_provider_records_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            paths["provider"].write_bytes(
                b'{"schema":"plebian.models.provider-measurement/v1",'
                b'"schema":"plebian.models.provider-measurement/v1"}\n'
            )
            paths["provider"].chmod(0o600)
            self.assertEqual(profile.main(arguments), 2)
            self.assertFalse(paths["receipt"].exists())
            self.assertFalse(paths["profile"].exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, record = prepare(root)
            paths["provider"].write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            paths["provider"].chmod(0o600)
            self.assertEqual(profile.main(arguments), 2)
            self.assertFalse(paths["receipt"].exists())
            self.assertFalse(paths["profile"].exists())

    def test_output_parent_retarget_is_refused_without_redirection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            moved = root / "moved"
            redirect = root / "redirect"
            output.mkdir(mode=0o700)
            redirect.mkdir(mode=0o700)
            receipt = profile._open_output_target(output / "receipt.json")
            catalog = profile._open_output_target(output / "profile.json")
            try:
                output.rename(moved)
                output.symlink_to(redirect, target_is_directory=True)
                with self.assertRaises(profile.IntakeRefused):
                    profile._write_outputs(
                        [(receipt, b"receipt\n"), (catalog, b"profile\n")]
                    )
            finally:
                receipt.close()
                catalog.close()
            self.assertEqual(list(moved.iterdir()), [])
            self.assertEqual(list(redirect.iterdir()), [])

    def test_second_output_collision_rolls_back_first_and_preserves_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "receipt.json"
            profile_path = root / "profile.json"
            receipt = profile._open_output_target(receipt_path)
            catalog = profile._open_output_target(profile_path)
            try:
                profile_path.write_bytes(b"keep\n")
                with self.assertRaises(profile.IntakeRefused):
                    profile._write_outputs(
                        [(receipt, b"receipt\n"), (catalog, b"profile\n")]
                    )
            finally:
                receipt.close()
                catalog.close()
            self.assertFalse(receipt_path.exists())
            self.assertEqual(profile_path.read_bytes(), b"keep\n")
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["profile.json"],
            )

    def test_private_input_mode_and_symlink_input_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            paths["provider"].chmod(0o640)
            self.assertEqual(profile.main(arguments), 2)
            self.assertFalse(paths["receipt"].exists())
            self.assertFalse(paths["profile"].exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            actual = root / "provider-actual.json"
            paths["provider"].rename(actual)
            paths["provider"].symlink_to(actual)
            self.assertEqual(profile.main(arguments), 2)
            self.assertFalse(paths["receipt"].exists())
            self.assertFalse(paths["profile"].exists())

    def test_existing_output_is_preserved_and_other_output_is_not_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            paths["receipt"].write_bytes(b"keep\n")
            self.assertEqual(profile.main(arguments), 2)
            self.assertEqual(paths["receipt"].read_bytes(), b"keep\n")
            self.assertFalse(paths["profile"].exists())


if __name__ == "__main__":
    unittest.main()
