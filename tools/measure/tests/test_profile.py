from __future__ import annotations

import json
import io
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

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
    bundle = root / "intake-bundle"
    paths = {
        "artifact": root / "artifact.bin",
        "fixture": root / "fixture.bin",
        "hardware": root / "hardware.json",
        "provider": root / "provider.json",
        "bundle": bundle,
        "receipt": bundle / profile.BUNDLE_RECEIPT_NAME,
        "profile": bundle / profile.BUNDLE_PROFILE_NAME,
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
        "--output-bundle",
        str(paths["bundle"]),
    ]
    return arguments, paths, record


class ProfileIntakeTests(unittest.TestCase):
    def test_success_atomically_records_but_does_not_promote_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            previous_umask = os.umask(0o777)
            try:
                status = profile.main(arguments)
            finally:
                os.umask(previous_umask)
            self.assertEqual(status, 0)
            self.assertEqual(stat.S_IMODE(paths["bundle"].stat().st_mode), 0o700)
            self.assertEqual(
                sorted(path.name for path in paths["bundle"].iterdir()),
                [profile.BUNDLE_RECEIPT_NAME, profile.BUNDLE_PROFILE_NAME],
            )
            self.assertEqual(stat.S_IMODE(paths["receipt"].stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(paths["profile"].stat().st_mode), 0o600)

            receipt = json.loads(paths["receipt"].read_bytes())
            catalog = json.loads(paths["profile"].read_bytes())
            generated = catalog["profiles"][0]
            self.assertEqual(
                receipt["provider_records"],
                {"accepted_as_measurement": 0, "recorded": 1, "total": 1},
            )
            self.assertEqual(receipt["byte_identities"]["matched"], 3)
            self.assertEqual(receipt["byte_identities"]["total"], 3)
            self.assertEqual(
                receipt["promotion"]["provider_measurements"],
                {"promoted": 0, "total": 9},
            )
            self.assertEqual(
                receipt["bundle_files"], {"prepared": 2, "total": 2}
            )
            self.assertNotIn("outputs", receipt)
            self.assertFalse(catalog["qualification_eligible"])
            self.assertEqual(generated["qualification"], "unqualified")
            self.assertEqual(generated["evidence"]["confidence"], "unknown")
            for key in ("command", "fixture", "measured_at", "reference_hardware_class"):
                self.assertIsNone(generated["evidence"][key])
            self.assertIsNone(generated["artifact"]["license_decision_id"])
            self.assertEqual(
                [
                    generated["requirements"][key]
                    for key in (
                        "download_bytes",
                        "disk_installed_bytes",
                        "temporary_bytes",
                        "ram_peak_bytes",
                        "vram_peak_bytes",
                    )
                ],
                [None] * 5,
            )
            self.assertEqual(set(generated["performance"].values()), {None})
            self.assertEqual(
                generated["evidence"]["raw_evidence_sha256"],
                profile._sha256(paths["provider"].read_bytes()),
            )

    def test_provider_executable_artifact_is_never_executed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, record = prepare(root)
            marker = root / "executed"
            paths["artifact"].write_text(
                f"#!/bin/sh\ntouch {marker}\n",
                encoding="utf-8",
            )
            paths["artifact"].chmod(0o700)
            record["artifact"]["content_sha256"] = profile._sha256(
                paths["artifact"].read_bytes()
            )
            write_private_json(paths["provider"], record)
            self.assertEqual(profile.main(arguments), 0)
            self.assertFalse(marker.exists())

    def test_each_digest_mismatch_commits_zero_of_one_final_bundle(self) -> None:
        for role in ("artifact", "fixture", "hardware"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                arguments, paths, _record = prepare(root)
                if role == "hardware":
                    changed = json.loads(paths[role].read_bytes())
                    changed["snapshot_id"] = "fixture:h0-cpu-only-changed"
                    write_private_json(paths[role], changed)
                else:
                    paths[role].write_bytes(paths[role].read_bytes() + b"changed")
                self.assertEqual(profile.main(arguments), 2)
                self.assertFalse(paths["bundle"].exists())

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
            self.assertFalse(paths["bundle"].exists())

    def test_prefixed_and_adjacent_hex_ipv6_identifiers_are_refused(self) -> None:
        identifiers = (
            "local:2001:db8::1",
            "local:a2001:db8::1",
            "local:dead2001:db8::1",
        )
        for identifier in identifiers:
            with self.subTest(identifier=identifier):
                hardware = json.loads(HARDWARE_FIXTURE.read_bytes())
                hardware["snapshot_id"] = identifier
                self.assertTrue(profile._contains_ipv6(identifier))
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    arguments, paths, _record = prepare(root, hardware)
                    self.assertEqual(profile.main(arguments), 2)
                    self.assertFalse(paths["bundle"].exists())

    def test_duplicate_noncanonical_and_deep_json_are_handled_refusals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            paths["provider"].write_bytes(
                b'{"schema":"plebian.models.provider-measurement/v1",'
                b'"schema":"plebian.models.provider-measurement/v1"}\n'
            )
            paths["provider"].chmod(0o600)
            self.assertEqual(profile.main(arguments), 2)
            self.assertFalse(paths["bundle"].exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, record = prepare(root)
            paths["provider"].write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            paths["provider"].chmod(0o600)
            self.assertEqual(profile.main(arguments), 2)
            self.assertFalse(paths["bundle"].exists())

        for role in ("provider", "hardware"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                arguments, paths, _record = prepare(root)
                paths[role].write_bytes(
                    b'{"deep":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}\n"
                )
                paths[role].chmod(0o600)
                self.assertEqual(profile.main(arguments), 2)
                self.assertFalse(paths["bundle"].exists())

        nested: Any = 0
        for _index in range(profile.MAX_JSON_DEPTH + 1):
            nested = [nested]
        with self.assertRaises(profile.IntakeRefused):
            profile._strict_json(profile._canonical_bytes({"deep": nested}), "control")

    def test_fifo_is_nonblocking_refusal_for_four_of_four_input_roles(self) -> None:
        for role in ("provider", "artifact", "fixture", "hardware"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                arguments, paths, _record = prepare(root)
                paths[role].unlink()
                os.mkfifo(paths[role], 0o600)
                completed = subprocess.run(
                    [sys.executable, "-m", "tools.measure.profile", *arguments],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=2,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertNotIn(b"Traceback", completed.stderr)
                self.assertEqual(completed.stderr.count(b"\n"), 1)
                self.assertFalse(paths["bundle"].exists())

    def test_output_parent_retarget_cannot_redirect_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            moved = root / "moved"
            redirect = root / "redirect"
            output.mkdir(mode=0o700)
            redirect.mkdir(mode=0o700)
            target = profile._open_bundle_target(output / "bundle")
            try:
                output.rename(moved)
                output.symlink_to(redirect, target_is_directory=True)
                with self.assertRaises(profile.IntakeRefused):
                    profile._write_bundle(target, b"receipt\n", b"profile\n")
            finally:
                target.close()
            self.assertEqual(list(moved.iterdir()), [])
            self.assertEqual(list(redirect.iterdir()), [])

    def test_parent_mode_revocation_cannot_expose_partial_final_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            target = profile._open_bundle_target(bundle)
            original_rename = profile._rename_noreplace

            def revoke_then_rename(parent_fd: int, source: str, destination: str) -> None:
                root.chmod(0o500)
                original_rename(parent_fd, source, destination)

            try:
                with mock.patch.object(
                    profile,
                    "_rename_noreplace",
                    side_effect=revoke_then_rename,
                ):
                    with self.assertRaises(profile.IntakeRefused):
                        profile._write_bundle(target, b"receipt\n", b"profile\n")
            finally:
                root.chmod(0o700)
                target.close()
            self.assertFalse(bundle.exists())
            for residue in root.iterdir():
                self.assertTrue(residue.is_dir())
                self.assertEqual(list(residue.iterdir()), [])

    def test_same_uid_removal_before_rename_return_is_a_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            original_rename = profile._rename_noreplace

            def remove_then_rename(
                parent_fd: int,
                source: str,
                destination: str,
            ) -> None:
                flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
                staged_fd = os.open(source, flags, dir_fd=parent_fd)
                try:
                    os.unlink(profile.BUNDLE_PROFILE_NAME, dir_fd=staged_fd)
                finally:
                    os.close(staged_fd)
                original_rename(parent_fd, source, destination)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    profile,
                    "_rename_noreplace",
                    side_effect=remove_then_rename,
                ),
                mock.patch.object(profile.sys, "stdout", stdout),
                mock.patch.object(profile.sys, "stderr", stderr),
            ):
                self.assertEqual(profile.main(arguments), 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue().count("\n"), 1)
            self.assertFalse(paths["bundle"].exists())

    def test_bundle_race_is_preserved_and_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            target = profile._open_bundle_target(bundle)
            bundle.mkdir(mode=0o700)
            marker = bundle / "keep"
            marker.write_bytes(b"keep\n")
            try:
                with self.assertRaises(profile.IntakeRefused):
                    profile._write_bundle(target, b"receipt\n", b"profile\n")
            finally:
                target.close()
            self.assertEqual(marker.read_bytes(), b"keep\n")
            self.assertEqual(sorted(path.name for path in bundle.iterdir()), ["keep"])
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["bundle"])

    def test_private_input_mode_and_symlink_input_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            paths["provider"].chmod(0o640)
            self.assertEqual(profile.main(arguments), 2)
            self.assertFalse(paths["bundle"].exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            actual = root / "provider-actual.json"
            paths["provider"].rename(actual)
            paths["provider"].symlink_to(actual)
            self.assertEqual(profile.main(arguments), 2)
            self.assertFalse(paths["bundle"].exists())

    def test_existing_bundle_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments, paths, _record = prepare(root)
            paths["bundle"].mkdir(mode=0o700)
            marker = paths["bundle"] / "keep"
            marker.write_bytes(b"keep\n")
            self.assertEqual(profile.main(arguments), 2)
            self.assertEqual(marker.read_bytes(), b"keep\n")


if __name__ == "__main__":
    unittest.main()
