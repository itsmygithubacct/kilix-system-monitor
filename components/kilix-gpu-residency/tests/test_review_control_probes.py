"""Adopted from the control seat's P1 review controls (review-p1/control/controls/test_control_probes.py).
Only the Required controls are kept; the Observed ones recorded the defects this round fixed.

Control-seat probes that found nothing: each asserts a README/contract claim and passes.

Same harness as test_control_findings.py (run from components/kilix-gpu-residency).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import residency_containment  # noqa: E402,F401
import kilix_gpu_residency as residency  # noqa: E402
from fakes import GIB, MIB, NVIDIA, ResidencyCase  # noqa: E402


class Probes(ResidencyCase):
    def test_ProbeAnUnloadingResidentIsRefusedWithoutForce(self):
        resident, process = self.model("puzzllm", "m", GIB)
        path = Path(self.namespace) / (resident.id + ".json")
        record = json.loads(path.read_text())
        record["state"] = "unloading"
        # Adapted for fix round 1: an unloading mark names its unloader, here a live one (this process).
        record["claim"] = {"pid": os.getpid(), "start": residency._start(os.getpid())}
        path.write_text(json.dumps(record))
        with self.assertRaises(residency.ResidencyError) as caught:
            residency.unload(resident.id, **self.where())
        self.assertEqual((caught.exception.code, str(caught.exception)), ("refused", "This resident is unloading"))
        with self.assertRaises(residency.RoomError):
            residency.ensure_room(5 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertIsNone(process.poll())

    def test_ProbeRegisterWithPolicyNeverWritesNothingAndUnloadsNothing(self):
        _, old = self.model("bonsai", "old", 4 * GIB)
        self.ollama.load("qwen3.5:2b", GIB)
        with self.assertRaises(residency.RoomError) as caught:
            residency.register(app="kilix-chat", model="new", backend="process", device="Vulkan1",
                               need_bytes=3 * GIB, policy="never", **self.where())
        # Relaxed in fix round 1 (FIXLIST "Rejected"): which candidates the plan names depends on the order;
        # registered residents now go first, so "old" alone meets the need.
        plan = sorted(r.model for r in caught.exception.plan)
        self.assertTrue(plan and set(plan) <= {"old", "qwen3.5:2b"}, plan)
        self.assertEqual(caught.exception.code, "policy-never")
        self.assertEqual(caught.exception.evicted, ())
        self.assertEqual([r["model"] for r in self.records()], ["old"])
        self.assertEqual(self.ollama.unloads(), [])
        self.assertIsNone(old.poll())

    def test_ProbeEnsureRoomWakesASleepingGpuAndStatusDoesNot(self):
        status_file = self.sys / "devices" / "pci0000:00" / NVIDIA[0] / "power" / "runtime_status"
        status_file.write_text("suspended\n")
        [device] = [d for d in self.status().devices if d.pci == NVIDIA[0]]
        self.assertEqual((device.asleep, device.free_bytes, self.loader_calls), (True, None, 0))
        room = residency.ensure_room(GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual((room.device_pci, room.free_bytes), (NVIDIA[0], 6 * GIB - 100 * MIB))
        self.assertEqual(self.loader_calls, 1)
        self.assertEqual(self.nvml.open_sessions, 0)   # closed again after the snapshot

    def test_ProbePinSurvivesAttachAndStateChanges(self):
        resident = residency.register(app="kilix-chat", model="m", backend="process", device="Vulkan1",
                                      pinned=True, **self.where())
        process = self.spawn(3 * GIB)
        resident.attach(pgid=process.pid)
        resident.busy()
        resident.idle()
        [info] = self.status().residents
        self.assertEqual((info.pinned, info.state, info.vram_bytes), (True, "idle", 3 * GIB))
        with self.assertRaises(residency.RoomError) as caught:
            residency.ensure_room(4 * GIB, device="Vulkan1", requester="puzzllm", **self.where())
        self.assertEqual(caught.exception.plan, ())
        self.assertIsNone(process.poll())

    def test_ProbeNotEnoughWithOllamaCandidatesUnloadsNothing(self):
        self.ollama.load("a:1b", GIB)
        _, p = self.model("puzzllm", "b", GIB)
        self.model("puzzllm", "busy", 2 * GIB, state="busy")
        with self.assertRaises(residency.RoomError) as caught:
            residency.ensure_room(5 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual(sorted(r.model for r in caught.exception.plan), ["a:1b", "b"])
        self.assertEqual((caught.exception.evicted, self.ollama.unloads()), ((), []))
        self.assertIsNone(p.poll())

    def test_ProbeRecordKeysOnDiskAreTheContractsRecordFields(self):
        contract = json.loads((TESTS.parents[2] / "contracts" / "kilix.gpu-residency-v1.interface.json").read_text())
        self.model("kilix-chat", "m", GIB, need_bytes=None)
        residency.register(app="kilix-chat", model="q:1b", backend="ollama", **self.where())
        for record in self.records():
            self.assertEqual(sorted(record), contract["record_fields"])


if __name__ == "__main__":
    import unittest
    unittest.main()
