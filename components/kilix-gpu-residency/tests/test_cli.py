"""kilix-gpu status, unload, pin, unpin and room."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import signal
import subprocess
import sys

import residency_containment
import kilix_gpu_residency as residency
from kilix_gpu_residency import cli
from fakes import GIB, MIB, ResidencyCase

LAUNCHER = Path(__file__).resolve().parents[1] / "kilix-gpu"


class CliTests(ResidencyCase):
    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(["--namespace", self.namespace, "--ollama", self.ollama.url, *argv], out, err)
        return code, out.getvalue(), err.getvalue()

    def test_status_lists_devices_residents_unmanaged_and_recent_unloads(self):
        self.model("kilix-chat", "LLaDA-MoE-7B", 2 * GIB)
        self.ollama.load("bonsai-2:27b", GIB)
        stranger = self.spawn(300 * MIB)
        residency.unload("bonsai-2:27b", **self.where())
        code, out, err = self.run_cli("status")
        self.assertEqual((code, err), (0, ""))
        self.assertIn("0000:01:00.0  Fake RTX:", out)
        self.assertIn("0000:00:02.0  Intel GPU 8086:3e9b: no VRAM counter", out)
        self.assertRegex(out, r"kilix-chat  LLaDA-MoE-7B  \[process\] idle  2\.0 GiB  idle 0m0\ds")
        self.assertIn(f"pid {stranger.pid}  sleep  300 MiB on 0000:01:00.0", out)
        self.assertIn("bonsai-2:27b (ollama), 1.0 GiB, by kilix-gpu", out)

    def test_status_json_has_a_versioned_schema(self):
        self.ollama.load("bonsai-2:27b", GIB)
        code, out, _ = self.run_cli("status", "--json")
        value = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual((value["schema"], value["version"], value["ollama"]),
                         (cli.STATUS_SCHEMA, residency.VERSION, "reachable"))
        self.assertEqual(set(value), {"schema", "version", "ollama", "devices", "residents", "unmanaged", "foreign",
                                      "events"})
        [resident] = value["residents"]
        self.assertEqual((resident["id"], resident["registered"], resident["vram_bytes"], resident["unload"]),
                         ("ollama:bonsai-2:27b", False, GIB, "ollama"))
        self.assertEqual(value["devices"][1]["free_bytes"], 6 * GIB - 100 * MIB - GIB)

    def test_unload_pin_and_unpin(self):
        resident, process = self.model("puzzllm", "m", GIB)
        self.assertEqual(self.run_cli("pin", resident.id)[:2], (0, f"Pinned m ({resident.id})\n"))
        code, _, err = self.run_cli("unload", "m")
        self.assertEqual((code, err), (1, "kilix-gpu: This resident is pinned [refused]\n"))
        self.assertEqual(self.run_cli("unpin", "m")[0], 0)
        code, out, _ = self.run_cli("unload", "m")
        self.assertEqual((code, out), (0, "Unloaded m (puzzllm), freeing about 1.0 GiB\n"))
        self.assertEqual(self.reap(process), -signal.SIGTERM)
        self.assertEqual(self.run_cli("unload", "m")[0], 1)

    def test_unload_force_takes_a_busy_resident(self):
        _, process = self.model("puzzllm", "m", GIB, state="busy")
        self.assertEqual(self.run_cli("unload", "m")[0], 1)
        self.assertEqual(self.run_cli("unload", "--force", "m")[0], 0)
        self.assertEqual(self.reap(process), -signal.SIGTERM)

    def test_room_is_a_dry_run(self):
        _, process = self.model("puzzllm", "m", 4 * GIB)
        code, out, _ = self.run_cli("room", "1G", "--device", "Vulkan1")
        self.assertEqual(code, 0)
        self.assertIn("1.0 GB fits on 0000:01:00.0: 2.0 GB free", out)  # in the units it was given
        code, out, _ = self.run_cli("room", "3GiB", "--device", "Vulkan1")
        self.assertEqual((code, out), (0, "3.0 GiB needs unloading on 0000:01:00.0: m (puzzllm, 4.0 GiB)\n"))
        code, out, _ = self.run_cli("room", "7GiB", "--json")
        value = json.loads(out)
        self.assertEqual((code, value["schema"], value["fits"], value["enough"], value["need_bytes"]),
                         (2, cli.ROOM_SCHEMA, False, False, 7 * GIB))
        self.assertIsNone(process.poll())
        self.assertEqual(code, 2)
        code, _, err = self.run_cli("room", "1G", "--device", "0000:00:02.0")
        self.assertEqual(code, 1)
        self.assertIn("[unmeasured]", err)

    def test_byte_counts(self):
        cases = {"5200000000": 5_200_000_000, "5.2G": 5_200_000_000, "4.8GiB": int(4.8 * 2**30), "512M": 512 * 10**6,
                 "512MiB": 512 * 2**20, "1 kib": 1024, "7B": 7}
        for text, expected in cases.items():
            self.assertEqual(cli.parse_bytes(text), expected)
        for bad in ("", "-1G", "0", "1.2.3G", "5 parsecs", "G"):
            with self.subTest(bad=bad), self.assertRaises(argparse.ArgumentTypeError):
                cli.parse_bytes(bad)

    def test_the_source_launcher_runs_the_command(self):
        done = subprocess.run([sys.executable, str(LAUNCHER), "--namespace", self.namespace, "--ollama",
                               self.ollama.url, "pin", "nothing"], capture_output=True, text=True, timeout=60,
                              env=residency_containment.child_env(KILIX_GPU_SYSFS_ROOT=str(self.scratch / "none")))
        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stderr, "kilix-gpu: No such resident [not-found]\n")


if __name__ == "__main__":
    import unittest
    unittest.main()
