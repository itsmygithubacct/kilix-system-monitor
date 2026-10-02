import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from plebian_model_sizer import resources
from plebian_model_sizer.cli import read_json


class ResourceTests(unittest.TestCase):
    def test_nested_cgroup_uses_smallest_ancestor_remaining_budget(self):
        reads = {"/proc/self/cgroup": "0::/slice/job", "/proc/self/mountinfo": "1 0 0:1 / /cgroup rw - cgroup2 cgroup2 rw",
                 "/cgroup/slice/job/memory.max": "1000", "/cgroup/slice/memory.max": "2000"}
        used = {"/cgroup/slice/job/memory.current": 100, "/cgroup/slice/memory.current": 1800}
        with patch.object(resources.probe, "_read_text", side_effect=lambda p, *a: reads.get(str(p))), \
             patch.object(resources.probe, "_read_int", side_effect=lambda p, **kw: used.get(str(p))), \
             patch.object(Path, "is_dir", return_value=True), patch.object(Path, "stat") as stat:
            stat.return_value.st_ino = 1
            self.assertEqual(resources.cgroup_headroom(), (200, "limited"))
            reads["/cgroup/slice/memory.max"] = "max"
            self.assertEqual(resources.cgroup_headroom(), (900, "limited"))
            reads["/cgroup/slice/job/memory.max"] = "0"
            self.assertEqual(resources.cgroup_headroom(), (0, "limited"))
            del reads["/cgroup/slice/memory.max"]
            self.assertEqual(resources.cgroup_headroom(), (None, "unknown"))

    def test_hidden_namespace_and_legacy_controller_unknown(self):
        with patch.object(resources.probe, "_read_text", side_effect=[
                "0::/", "1 0 0:1 / /cgroup rw - cgroup2 cgroup2 rw"]), \
             patch.object(Path, "is_dir", return_value=True), patch.object(Path, "stat") as stat:
            stat.return_value.st_ino = 100
            self.assertEqual(resources.cgroup_headroom(), (None, "unknown"))
        with patch.object(resources.probe, "_read_text", side_effect=["2:memory:/job", ""]):
            self.assertEqual(resources.cgroup_headroom(), (None, "unknown"))

    def test_gpu_rows_validate_free_memory_and_index(self):
        self.assertEqual(resources.parse_nvidia("0, 8192, 4096\n1, 16384, 1024")[0]["available_bytes"], 4 * resources.GIB)
        for text in ("0, N/A, N/A", "0, 100, 101", "0, 100, -1", "0, 100, 50\n0, 100, 50", "1000, 100, 50"):
            self.assertEqual(resources.parse_nvidia(text), [])

    def test_failed_gpu_probe_returns_no_evidence(self):
        with patch.object(resources.probe, "_find_executable", return_value="/usr/bin/nvidia-smi"), \
             patch.object(resources.probe, "_run_bounded", return_value=(1, b"0, 8000, 7000")):
            self.assertEqual(resources.gpu_headroom(), [])

    def test_collect_applies_cgroup_without_leaking_paths(self):
        with patch.object(resources.probe, "_memory", return_value=({"total_bytes": 1000, "available_bytes": 900}, "unknown")), \
             patch.object(resources, "cgroup_headroom", return_value=(400, "limited")), \
             patch.object(resources, "gpu_headroom", return_value=[]), \
             patch.object(resources, "storage_headroom", return_value=2000):
            result = resources.collect(Path("/private/documents"))
            self.assertEqual(result["ram_available_bytes"], 400)
            self.assertNotIn("/private", json.dumps(result))

    def test_disk_probe_does_not_create_state_and_uses_available_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / "data" / "models"
            expected = os.statvfs(directory)
            self.assertEqual(resources.storage_headroom(absent), expected.f_bavail * expected.f_frsize)
            self.assertFalse(absent.parent.exists())

    def test_cuda_mapping_requires_single_unmasked_physical_zero(self):
        device={'index':0,'backend':'cuda','total_bytes':6*resources.GIB,
                'available_bytes':6*resources.GIB}
        cases=[([device],{},'single-unmasked-gpu-zero'),
               ([device],{'CUDA_VISIBLE_DEVICES':'0'},'unverified'),
               ([device],{'CUDA_DEVICE_ORDER':'PCI_BUS_ID'},'unverified'),
               ([device,{**device,'index':1}],{},'unverified'),
               ([{**device,'index':1}],{},'unverified')]
        for devices,environment,expected in cases:
            with self.subTest(environment=environment,devices=devices), \
                 patch.dict(os.environ,environment,clear=True), \
                 patch.object(resources.probe,'_memory',return_value=({'total_bytes':1000,'available_bytes':900},'unknown')), \
                 patch.object(resources,'cgroup_headroom',return_value=(None,'unlimited')), \
                 patch.object(resources,'gpu_headroom',return_value=devices), \
                 patch.object(resources,'storage_headroom',return_value=1000):
                self.assertEqual(resources.collect()['cuda_device_mapping'],expected)

    def test_json_duplicate_nan_and_size_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for text in ('{"a": 1, "a": 2}', '{"a": NaN}', "[]", " " * (4 * 1024 * 1024 + 1)):
                path.write_text(text)
                with self.assertRaises(ValueError): read_json(path)


if __name__ == "__main__":
    unittest.main()
