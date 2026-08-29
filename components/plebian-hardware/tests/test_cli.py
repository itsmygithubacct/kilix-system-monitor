from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from plebian_hardware import cli, probe, state


def observation(scope: str = "inventory") -> dict[str, object]:
    return {
        "schema": "plebian.hardware/v1",
        "capture": {"scope": scope},
        "cpu": {"architecture": "x86_64", "effective_cpus": 4},
        "memory": {"total_bytes": 8 * 1024**3},
        "gpus": [],
        "unknowns": ["gpu.inventory"],
    }


def cache_observation(scope: str = "inventory") -> dict[str, object]:
    document = probe.collect(scope)
    document["capture"]["captured_at"] = "2026-08-25T00:00:00Z"
    return document


class CliTests(unittest.TestCase):
    def test_inventory_json_uses_exact_envelope(self) -> None:
        status, stdout, stderr = cli.dispatch(
            ["inventory", "--json"], lambda scope: observation(scope)
        )
        self.assertEqual(status, 0)
        self.assertEqual(stderr, b"")
        self.assertTrue(stdout.endswith(b"\n"))
        response = json.loads(stdout)
        self.assertEqual(response["schema"], "plebian.cli.response/v1")
        self.assertEqual(response["command"], "hardware.inventory")
        self.assertEqual(response["status"], "unknown")
        self.assertEqual(response["data"]["capture"]["scope"], "inventory")
        complete = observation()
        complete["unknowns"] = []
        status, stdout, stderr = cli.dispatch(
            ["inventory", "--json"], lambda _: complete
        )
        self.assertEqual((status, stderr), (0, b""))
        self.assertEqual(json.loads(stdout)["status"], "ok")

    def test_gpu_scope_is_not_inferred_from_option_order(self) -> None:
        status, stdout, stderr = cli.dispatch(
            ["gpu", "--json"], lambda scope: observation(scope)
        )
        self.assertEqual(status, 0)
        self.assertEqual(stderr, b"")
        self.assertEqual(json.loads(stdout)["data"]["capture"]["scope"], "gpu")
        for invalid in (["gpu"], ["--json", "gpu"], ["gpu", "--json", "extra"]):
            status, stdout, stderr = cli.dispatch(invalid)
            self.assertEqual(status, 2)
            self.assertEqual(stdout, b"")
            self.assertEqual(stderr.count(b"\n"), 1)
            self.assertLessEqual(len(stderr), cli.MAX_DIAGNOSTIC_BYTES)

    def test_show_is_redacted_and_unqualified(self) -> None:
        status, stdout, stderr = cli.dispatch(
            ["show"], lambda scope: observation(scope)
        )
        self.assertEqual((status, stderr), (0, b""))
        self.assertIn(b"redacted local observation", stdout)
        self.assertIn(b"unqualified local observation", stdout)
        self.assertNotIn(b"vendor_id", stdout)

    def test_internal_failure_does_not_expose_exception(self) -> None:
        def fail(_: str) -> dict[str, object]:
            raise RuntimeError("private probe detail")

        status, stdout, stderr = cli.dispatch(["inventory", "--json"], fail)
        self.assertEqual(status, 70)
        self.assertEqual(stdout, b"")
        self.assertNotIn(b"private probe detail", stderr)
        self.assertEqual(stderr.count(b"\n"), 1)

    def test_output_boundary_fails_closed(self) -> None:
        huge = observation()
        huge["unknowns"] = ["x" * cli.MAX_STDOUT_BYTES]
        status, stdout, stderr = cli.dispatch(
            ["inventory", "--json"], lambda _: huge
        )
        self.assertEqual(status, 70)
        self.assertEqual(stdout, b"")
        self.assertTrue(stderr.endswith(b"\n"))

    def test_refresh_writes_only_a_private_canonical_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private-state"
            document = cache_observation()
            status, stdout, stderr = cli.dispatch(
                ["refresh"], lambda _: document, state_root=root
            )
            self.assertEqual((status, stderr), (0, b""))
            self.assertIn(b"private 0600", stdout)
            self.assertNotIn(str(root).encode(), stdout)
            cached = root / state.CACHE_FILENAME
            self.assertEqual(cached.stat().st_mode & 0o777, 0o600)
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            self.assertEqual(state.read_snapshot(root), document)
            payload = cached.read_bytes()
            self.assertTrue(payload.endswith(b"\n"))
            self.assertEqual(payload, state._canonical_bytes(document))

    def test_refresh_refuses_a_symlinked_state_root_or_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            target = base / "target"
            target.mkdir(mode=0o700)
            alias = base / "alias"
            alias.symlink_to(target, target_is_directory=True)
            status, stdout, stderr = cli.dispatch(
                ["refresh"], cache_observation, state_root=alias
            )
            self.assertEqual((status, stdout), (69, b""))
            self.assertEqual(stderr, b"plebian-hardware: private state unavailable\n")

            root = base / "private"
            root.mkdir(mode=0o700)
            outside = base / "outside"
            outside.write_text("untouched\n", encoding="utf-8")
            (root / state.CACHE_FILENAME).symlink_to(outside)
            status, stdout, stderr = cli.dispatch(
                ["refresh"], cache_observation, state_root=root
            )
            self.assertEqual((status, stdout), (69, b""))
            self.assertEqual(outside.read_text(encoding="utf-8"), "untouched\n")

    def test_refresh_refuses_and_preserves_an_invalid_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private"
            root.mkdir(mode=0o700)
            cached = root / state.CACHE_FILENAME
            cached.write_text("not JSON\n", encoding="utf-8")
            cached.chmod(0o600)
            status, stdout, stderr = cli.dispatch(
                ["refresh"], cache_observation, state_root=root
            )
            self.assertEqual((status, stdout), (65, b""))
            self.assertEqual(stderr, b"plebian-hardware: snapshot input refused\n")
            self.assertEqual(cached.read_text(encoding="utf-8"), "not JSON\n")

    def test_diff_is_redacted_and_ignores_volatile_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = cache_observation()
            current = json.loads(json.dumps(previous))
            current["snapshot_id"] = "local:capability-ffffffffffffffff"
            current["capture"]["captured_at"] = "2026-08-25T01:00:00Z"
            current["memory"]["total_bytes"] += 1024
            path = root / "snapshot.json"
            path.write_bytes(state._canonical_bytes(previous))
            path.chmod(0o600)
            status, stdout, stderr = cli.dispatch(
                ["diff", str(path)], lambda _: current
            )
            self.assertEqual((status, stderr), (0, b""))
            self.assertIn(b"Changed fields: 1", stdout)
            self.assertIn(b"memory.total", stdout)
            self.assertNotIn(b"snapshot_id", stdout)
            self.assertNotIn(b"captured_at", stdout)
            for forbidden in state.NEVER_COLLECTED:
                self.assertNotIn(forbidden.encode(), stdout)

    def test_diff_refuses_public_or_duplicate_key_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "snapshot.json"
            path.write_bytes(state._canonical_bytes(cache_observation()))
            path.chmod(0o644)
            status, stdout, stderr = cli.dispatch(
                ["diff", str(path)], cache_observation
            )
            self.assertEqual((status, stdout), (65, b""))
            self.assertEqual(stderr, b"plebian-hardware: snapshot input refused\n")

            path.write_text(
                '{"schema":"plebian.hardware/v1","schema":"plebian.hardware/v1"}\n',
                encoding="utf-8",
            )
            path.chmod(0o600)
            status, stdout, stderr = cli.dispatch(
                ["diff", str(path)], cache_observation
            )
            self.assertEqual((status, stdout), (65, b""))
            self.assertEqual(stderr.count(b"\n"), 1)

    def test_doctor_reports_absent_cache_without_exposing_a_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not-created"
            status, stdout, stderr = cli.dispatch(
                ["doctor"], cache_observation, state_root=root
            )
            self.assertEqual((status, stderr), (0, b""))
            self.assertIn(b"Private cache: absent", stdout)
            self.assertIn(b"blocked pending trusted launcher", stdout)
            self.assertNotIn(str(root).encode(), stdout)

    def test_state_root_requires_absolute_private_xdg_location(self) -> None:
        with self.assertRaises(state.StateUnavailable):
            state.default_state_root({"XDG_STATE_HOME": "relative"})
        with self.assertRaises(state.StateUnavailable):
            state.default_state_root({})
        self.assertEqual(
            state.default_state_root({"XDG_STATE_HOME": "/private/state"}),
            Path("/private/state/plebian-hardware"),
        )

    def test_snapshot_cache_rejects_identifiers_and_qualification(self) -> None:
        document = cache_observation()
        document["serial"] = "private"
        with self.assertRaises(state.SnapshotInvalid):
            state.validate_snapshot(document)
        document = cache_observation()
        document["capture"]["qualification_eligible"] = True
        with self.assertRaises(state.SnapshotInvalid):
            state.validate_snapshot(document)


class ProbeBoundaryTests(unittest.TestCase):
    def test_cpu_list_parser_counts_without_emitting_ranges(self) -> None:
        self.assertEqual(probe._cpu_list_count("0-3,8,10-11"), 7)
        for invalid in (None, "", "3-1", "0-a", "0-65536", "0-20000"):
            self.assertIsNone(probe._cpu_list_count(invalid))

    def test_executable_name_rejects_shell_vocabulary(self) -> None:
        self.assertIsNone(probe._find_executable("sh -c id"))
        self.assertIsNone(probe._find_executable("../tool"))

    def test_executable_must_remain_inside_fixed_nonwritable_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            allowed = temporary_root / "allowed"
            allowed.mkdir()
            trusted = allowed / "trusted-tool"
            trusted.write_bytes(b"#!/bin/false\n")
            trusted.chmod(0o700)
            outside = temporary_root / "outside-tool"
            outside.write_bytes(b"#!/bin/false\n")
            outside.chmod(0o700)
            escape = allowed / "escape-tool"
            escape.symlink_to(outside)
            writable = allowed / "writable-tool"
            writable.write_bytes(b"#!/bin/false\n")
            writable.chmod(0o722)
            with mock.patch.object(probe, "PROBE_PATH", str(allowed)):
                self.assertEqual(probe._find_executable("trusted-tool"), str(trusted))
                self.assertIsNone(probe._find_executable("escape-tool"))
                self.assertIsNone(probe._find_executable("writable-tool"))

    def test_reads_are_bounded_regular_ascii_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = root / "value"
            value.write_bytes(b"123\n")
            self.assertEqual(probe._read_text(value, 4), "123")
            self.assertIsNone(probe._read_text(value, 3))
            value.write_bytes(b"\xff")
            self.assertIsNone(probe._read_text(value))
            directory = root / "directory"
            directory.mkdir()
            self.assertIsNone(probe._read_bytes(directory))

    def test_identifier_paths_and_final_symlinks_are_never_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "safe-target"
            target.write_text("private-value\n", encoding="ascii")
            serial = root / "serial"
            serial.write_text("private-value\n", encoding="ascii")
            alias = root / "safe-alias"
            alias.symlink_to(target)
            self.assertIsNone(probe._read_text(serial))
            self.assertIsNone(probe._read_text(alias))

    def test_probe_process_boundary_matches_privacy_contract(self) -> None:
        self.assertEqual(
            probe.EXECUTABLE_INTEGRITY,
            "resolved-within-path-and-not-group-or-world-writable",
        )
        self.assertEqual(probe.SUBPROCESS_TIMEOUT_SECONDS, 5)
        self.assertEqual(probe.SUBPROCESS_STDOUT_BYTES, 65536)
        self.assertEqual(
            probe.SUBPROCESS_ENVIRONMENT,
            {
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            },
        )

    def test_schema_tokens_fail_closed(self) -> None:
        self.assertEqual(probe._safe_token("nvidia-uvm"), "nvidia-uvm")
        self.assertIsNone(probe._safe_token("value with spaces"))
        self.assertIsNone(probe._safe_token("x" * 81))
        self.assertEqual(probe._pci_id("0x10DE"), "10de")
        self.assertIsNone(probe._pci_id("0x123"))
        self.assertIsNone(probe._pci_id("0xzzzz"))

    def test_unresolved_observation_families_have_explicit_unknown_markers(self) -> None:
        document = {
            "virtualization": "unknown",
            "cpu": {
                "architecture": "unknown",
                "model_bucket": "desktop-x86",
                "logical_cpus": 1,
                "online_cpus": 1,
                "affinity_cpus": 1,
                "cpuset_cpus": 1,
                "effective_cpus": 1,
                "physical_cores": 1,
                "packages": 1,
                "numa_nodes": 1,
                "isa_features": ["avx"],
                "frequency_hz": {
                    "current_min_hz": 1,
                    "current_max_hz": 1,
                    "hardware_min_hz": 1,
                    "hardware_max_hz": 1,
                },
                "cache_bytes": {
                    "l1_data_bytes": 1,
                    "l1_instruction_bytes": 1,
                    "l2_bytes": 1,
                    "l3_bytes": 1,
                },
                "smt": "enabled",
            },
            "memory": {
                "total_bytes": None,
                "available_bytes": 1,
                "swap_total_bytes": 0,
                "swap_free_bytes": 0,
                "hugepage_size_bytes": 1,
                "hugepage_total_bytes": 0,
                "hugepage_free_bytes": 0,
                "numa_nodes": 1,
            },
            "gpus": [
                {
                    "index": 0,
                    "vendor": "intel",
                    "vendor_id": "8086",
                    "device_id": "1234",
                    "kernel_driver": "i915",
                    "device_class": "unknown",
                    "render_access": True,
                    "vram_bytes": 0,
                    "memory_kind": "shared",
                    "shared_memory_bytes": 0,
                    "numa_node": -1,
                    "iommu_group_present": True,
                    "pcie": {
                        "current_width": 1,
                        "maximum_width": 1,
                        "current_gtps": 1.0,
                        "maximum_gtps": 1.0,
                    },
                    "backends": [
                        {"name": "opencl", "status": "unknown"},
                    ],
                }
            ],
            "power": {
                "ac_online": None,
                "battery_present": False,
                "battery_percent": None,
                "batteries": [],
            },
            "platform": {
                "firmware_mode": "uefi",
                "secure_boot": "enabled",
                "iommu": "enabled",
                "dmi_access": "not-probed",
            },
            "buses": {
                "pci_devices_count": 0,
                "usb_devices_count": 0,
                "storage_controllers_count": 0,
            },
            "network": {
                "interfaces": [
                    {
                        "index": 0,
                        "type": "ethernet",
                        "online": True,
                        "link_mbps": None,
                        "driver": "e1000e",
                        "bus": "pci",
                    }
                ],
                "offline": False,
            },
            "thermal": {
                "sensor_count": 0,
                "maximum_celsius": 0.0,
                "fan_count": 0,
                "throttle": "unknown",
            },
            "storage": {
                "filesystem_type": "ext4",
                "free_bytes": None,
                "read_only": False,
                "total_bytes": 1,
            },
        }
        self.assertEqual(
            probe._required_unknown_markers(document),
            {
                "cpu.architecture",
                "gpu.0.class",
                "gpu.0.opencl",
                "memory.total",
                "network.0.link-mbps",
                "platform.dmi",
                "power.ac",
                "storage.free",
                "thermal.throttle",
                "virtualization",
            },
        )

    def test_effective_compute_uses_the_smallest_positive_limit(self) -> None:
        self.assertEqual(probe._effective_cpus(16, 8, 3.5), 3.5)
        self.assertEqual(probe._effective_cpus(4.0, None), 4)
        self.assertIsNone(probe._effective_cpus(None, 0, -1))

    def test_subprocess_output_is_bounded(self) -> None:
        status, stdout = probe._run_bounded(
            sys.executable, ["-c", "print('ok')"]
        )
        self.assertEqual(status, 0)
        self.assertEqual(stdout, b"ok\n")
        status, stdout = probe._run_bounded(
            sys.executable,
            ["-c", "import sys; sys.stdout.write('x' * 65537)"],
        )
        self.assertIsNone(status)
        self.assertIsNone(stdout)

    def test_subprocess_receives_only_the_fixed_clean_environment(self) -> None:
        status, stdout = probe._run_bounded(
            sys.executable,
            [
                "-c",
                "import json,os; print(json.dumps(dict(os.environ),sort_keys=True,separators=(',',':')))",
            ],
        )
        self.assertEqual(status, 0)
        self.assertIsNotNone(stdout)
        self.assertEqual(json.loads(stdout), probe.SUBPROCESS_ENVIRONMENT)

    def test_battery_aggregate_stays_unknown_if_any_reading_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, percent in (("BAT0", "100"), ("BAT1", None)):
                battery = root / name
                battery.mkdir()
                (battery / "type").write_text("Battery\n", encoding="ascii")
                (battery / "status").write_text("Full\n", encoding="ascii")
                if percent is not None:
                    (battery / "capacity").write_text(percent + "\n", encoding="ascii")
            result = probe._power(root)
        self.assertTrue(result["battery_present"])
        self.assertIsNone(result["battery_percent"])
        self.assertEqual([item["index"] for item in result["batteries"]], [0, 1])

    def test_network_absence_is_unknown_not_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing"
            self.assertIsNone(probe._network(missing)["offline"])
            loopback_only = root / "net"
            (loopback_only / "lo").mkdir(parents=True)
            self.assertTrue(probe._network(loopback_only)["offline"])

    def test_gpu_identifiers_are_normalized_or_null(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            drm = root / "drm"
            card = drm / "card0"
            pci = root / "0000:01:00.0"
            card.mkdir(parents=True)
            pci.mkdir()
            (pci / "vendor").write_text("0xZZZZ\n", encoding="ascii")
            (pci / "device").write_text("0x1234\n", encoding="ascii")
            (card / "device").symlink_to(pci, target_is_directory=True)
            with mock.patch.object(
                probe,
                "_command_probe",
                return_value=("unknown", "command-unavailable", None),
            ):
                result = probe._gpu_inventory(
                    drm, root / "missing-iommu", root / "missing-devices"
                )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["vendor"], "other")
        self.assertIsNone(result[0]["vendor_id"])
        self.assertEqual(result[0]["device_id"], "1234")

    def test_global_backend_success_cannot_qualify_one_amd_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            drm = root / "drm"
            card = drm / "card0"
            pci = root / "0000:01:00.0"
            card.mkdir(parents=True)
            pci.mkdir()
            (pci / "vendor").write_text("0x1002\n", encoding="ascii")
            (pci / "device").write_text("0x1234\n", encoding="ascii")
            (card / "device").symlink_to(pci, target_is_directory=True)
            with mock.patch.object(
                probe,
                "_command_probe",
                return_value=("available", "executable-probe", "1.0"),
            ):
                result = probe._gpu_inventory(
                    drm, root / "missing-iommu", root / "missing-devices"
                )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["vendor"], "amd")
        self.assertEqual(
            result[0]["backends"],
            [
                {
                    "evidence": "contradictory",
                    "name": "opencl",
                    "status": "unknown",
                    "version": None,
                },
                {
                    "evidence": "contradictory",
                    "name": "rocm",
                    "status": "unknown",
                    "version": None,
                },
                {
                    "evidence": "contradictory",
                    "name": "vulkan",
                    "status": "unknown",
                    "version": None,
                },
            ],
        )

    def test_nvidia_result_must_bind_one_well_formed_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            device = Path(temporary) / "0000:01:00.0"
            device.mkdir()
            with (
                mock.patch.object(probe, "_find_executable", return_value="/usr/bin/nvidia-smi"),
                mock.patch.object(
                    probe,
                    "_run_bounded",
                    return_value=(0, b"00000000:01:00.0, 555.42.02, 24564\n"),
                ) as run,
            ):
                self.assertEqual(
                    probe._nvidia_probe(device),
                    (
                        "available",
                        "executable-probe",
                        "555.42.02",
                        24564 * 1024**2,
                    ),
                )
                arguments = run.call_args.args[1]
                self.assertEqual(
                    arguments,
                    [
                        "--query-gpu=pci.bus_id,driver_version,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                )
                self.assertFalse(
                    any(
                        forbidden in " ".join(arguments).lower()
                        for forbidden in ("uuid", "serial", "name", "mac", "ip")
                    )
                )
            with (
                mock.patch.object(probe, "_find_executable", return_value="/usr/bin/nvidia-smi"),
                mock.patch.object(
                    probe,
                    "_run_bounded",
                    return_value=(0, b"not-a-bdf, 555.42.02, 24564\n"),
                ),
            ):
                self.assertEqual(
                    probe._nvidia_probe(device),
                    ("unknown", "contradictory", None, None),
                )

    def test_library_scope_is_strict(self) -> None:
        with self.assertRaises(ValueError):
            probe.collect("everything")

    def test_virtualization_maps_only_expected_statuses(self) -> None:
        cases = (
            ([(0, b"")], "container"),
            ([(1, b""), (0, b"")], "virtual-machine"),
            ([(1, b""), (1, b"")], "none"),
            ([(1, b""), (2, b"")], "unknown"),
            ([(None, None)], "unknown"),
        )
        for results, expected in cases:
            with self.subTest(expected=expected), mock.patch.object(
                probe, "_run_bounded", side_effect=results
            ):
                self.assertEqual(
                    probe._systemd_virtualization("/usr/bin/systemd-detect-virt"),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
