from __future__ import annotations

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from tools.measure import profile


def arguments(root: Path, command: list[str]) -> list[str]:
    artifact = root / "artifact.bin"
    fixture = root / "fixture.bin"
    hardware = root / "hardware.json"
    artifact.write_bytes(b"artifact-bytes")
    fixture.write_bytes(b"fixture-bytes")
    hardware.write_text(
        json.dumps(
            {
                "schema": "plebian.hardware/v1",
                "snapshot_id": "fixture:h0-cpu-only",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return [
        "--profile-id",
        "demo-tts-cpu",
        "--profile-version",
        "0.1.0",
        "--provider",
        "demo-provider",
        "--task",
        "tts",
        "--backend",
        "cpu",
        "--catalog-id",
        "demo-20260831",
        "--artifact-id",
        "demo-small",
        "--artifact-version",
        "v1",
        "--license-decision-id",
        "demo-small-licence-20260831",
        "--artifact",
        str(artifact),
        "--fixture-id",
        "tts-short-v1",
        "--fixture",
        str(fixture),
        "--hardware-snapshot",
        str(hardware),
        "--command-id",
        "demo.measure.tts",
        "--raw-evidence",
        str(root / "raw.json"),
        "--profile-catalog",
        str(root / "profile.json"),
        "--",
        *command,
    ]


class ProfileMeasurementTests(unittest.TestCase):
    def test_success_writes_private_measured_but_unqualified_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = profile.main(
                arguments(root, [sys.executable, "-c", "x = bytearray(1024 * 1024)"])
            )
            self.assertEqual(status, 0)
            raw_path = root / "raw.json"
            profile_path = root / "profile.json"
            self.assertEqual(stat.S_IMODE(raw_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(profile_path.stat().st_mode), 0o600)
            raw = json.loads(raw_path.read_bytes())
            catalog = json.loads(profile_path.read_bytes())
            measured = catalog["profiles"][0]
            self.assertEqual(raw["status"], "succeeded")
            self.assertGreater(raw["measurement"]["ram_peak_bytes"], 0)
            self.assertEqual(raw["command"]["command_id"], "demo.measure.tts")
            self.assertNotIn(str(root), raw_path.read_text(encoding="utf-8"))
            self.assertFalse(catalog["qualification_eligible"])
            self.assertEqual(measured["qualification"], "unqualified")
            self.assertEqual(measured["evidence"]["confidence"], "measured")
            self.assertIsNone(measured["requirements"]["vram_peak_bytes"])
            self.assertEqual(
                measured["evidence"]["raw_evidence_sha256"],
                profile._sha256(raw_path.read_bytes()),
            )

    def test_failed_command_writes_raw_evidence_but_no_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = profile.main(arguments(root, [sys.executable, "-c", "raise SystemExit(7)"]))
            self.assertEqual(status, 1)
            self.assertEqual(json.loads((root / "raw.json").read_bytes())["status"], "failed")
            self.assertFalse((root / "profile.json").exists())

    def test_timeout_kills_the_process_group_and_stays_unqualified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = arguments(root, [sys.executable, "-c", "import time; time.sleep(10)"])
            insert = argv.index("--raw-evidence")
            argv[insert:insert] = ["--timeout-seconds", "0.05", "--sample-interval-ms", "5"]
            status = profile.main(argv)
            self.assertEqual(status, 124)
            self.assertEqual(json.loads((root / "raw.json").read_bytes())["status"], "timed-out")
            self.assertFalse((root / "profile.json").exists())

    def test_symlinked_artifact_and_existing_output_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = arguments(root, [sys.executable, "-c", "pass"])
            artifact_index = argv.index("--artifact") + 1
            target = Path(argv[artifact_index])
            alias = root / "artifact-link"
            alias.symlink_to(target)
            argv[artifact_index] = str(alias)
            self.assertEqual(profile.main(argv), 2)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = arguments(root, [sys.executable, "-c", "pass"])
            (root / "raw.json").write_text("keep\n", encoding="utf-8")
            self.assertEqual(profile.main(argv), 2)
            self.assertEqual((root / "raw.json").read_text(encoding="utf-8"), "keep\n")
            self.assertFalse((root / "profile.json").exists())

    def test_duplicate_hardware_key_and_ambient_executable_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = arguments(root, [sys.executable, "-c", "pass"])
            hardware = root / "hardware.json"
            hardware.write_text(
                '{"schema":"plebian.hardware/v1","snapshot_id":"fixture:a","snapshot_id":"fixture:b"}\n',
                encoding="utf-8",
            )
            self.assertEqual(profile.main(argv), 2)
            self.assertFalse((root / "raw.json").exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = arguments(root, ["python", "-c", "pass"])
            self.assertEqual(profile.main(argv), 2)
            self.assertFalse((root / "raw.json").exists())


if __name__ == "__main__":
    unittest.main()
