"""The VRAM reader copied from Kilix: discovery, NVML, nvidia-smi, sysfs and DRM fdinfo."""
from __future__ import annotations

import os
from pathlib import Path
import stat

import residency_containment  # noqa: F401
from kilix_gpu_residency import vram
from fakes import AMD, GIB, INTEL, MIB, NVIDIA, SECOND_NVIDIA, FakeNvml, ResidencyCase, make_gpu

I915_FDINFO = """pos:\t0
flags:\t02100002
drm-driver:\ti915
drm-client-id:\t7
drm-pdev:\t0000:00:02.0
drm-total-system0:\t2048 KiB
drm-resident-system0:\t1024 KiB
"""
AMD_FDINFO = """drm-driver:\tamdgpu
drm-client-id:\t{client}
drm-pdev:\t0000:03:00.0
drm-memory-vram:\t{vram} KiB
drm-memory-gtt:\t999 KiB
drm-total-cycles-gfx:\t123
"""


class ReaderTests(ResidencyCase):
    gpus = (INTEL, NVIDIA)

    def test_discovery_lists_pci_gpus_once_and_skips_connectors(self):
        drm = self.sys / "class" / "drm"
        (drm / "card9").mkdir()
        os.symlink(self.sys / "devices" / "platform-gpu", drm / "card9" / "device")
        found = vram.discover()
        self.assertEqual([(d.pci, d.vendor_id, d.driver, d.nodes) for d in found],
                         [(INTEL[0], 0x8086, "i915", ("card0", "renderD128")),
                          (NVIDIA[0], 0x10de, "nvidia", ("card1", "renderD129"))])
        self.assertEqual([d.discrete for d in found], [False, True])

    def test_nvml_reads_memory_and_both_process_lists(self):
        a, b = self.spawn(), self.spawn()
        self.nvml_devices[NVIDIA[0]].update({"compute": {a.pid: GIB, b.pid: 100}, "graphics": {b.pid: 300 * MIB}})
        [intel, nvidia] = vram.snapshot()
        self.assertEqual((intel.source, intel.counter, intel.total_bytes), ("none", False, None))
        self.assertEqual((nvidia.source, nvidia.name, nvidia.counter), ("nvml", "Fake RTX", True))
        self.assertEqual((nvidia.total_bytes, nvidia.used_bytes, nvidia.free_bytes),
                         (6 * GIB, 100 * MIB + GIB + 100, 6 * GIB - 100 * MIB - GIB - 100))
        self.assertEqual(dict(nvidia.processes), {a.pid: GIB, b.pid: 300 * MIB})
        self.assertEqual(self.nvml.open_sessions, 0)  # closed again after the snapshot

    def test_process_lists_fall_back_from_v3_to_v2_to_v1(self):
        a = self.spawn()
        self.nvml_devices[NVIDIA[0]]["compute"] = {a.pid: GIB}
        for versions in (("_v2", ""), ("",)):
            with self.subTest(versions=versions):
                self.nvml.versions = versions
                self.assertEqual(dict(vram.snapshot()[1].processes), {a.pid: GIB})

    def test_a_long_process_list_is_read_whole(self):
        many = [self.spawn() for _ in range(40)]
        self.nvml_devices[NVIDIA[0]]["compute"] = {p.pid: MIB for p in many}
        self.assertEqual(len(vram.snapshot()[1].processes), 40)

    def test_unavailable_memory_values_are_skipped(self):
        a, b = self.spawn(), self.spawn()
        self.nvml_devices[NVIDIA[0]]["compute"] = {a.pid: vram.NVML_VALUE_NOT_AVAILABLE, b.pid: MIB}
        self.assertEqual(dict(vram.snapshot()[1].processes), {b.pid: MIB})

    def test_a_sleeping_nvidia_gpu_is_not_woken_unless_asked(self):
        power = self.sys / "devices" / "pci0000:00" / NVIDIA[0] / "power" / "runtime_status"
        power.write_text("suspended\n")
        nvidia = vram.snapshot()[1]
        self.assertEqual((nvidia.asleep, nvidia.source, nvidia.total_bytes, nvidia.counter),
                         (True, "asleep", None, True))
        self.assertEqual(self.loader_calls, 0)
        self.assertEqual(vram.snapshot(wake=True)[1].source, "nvml")
        self.assertEqual(self.loader_calls, 1)

    def test_a_sleeping_sibling_holds_back_nvml_for_every_nvidia_gpu(self):
        make_gpu(self.sys, *SECOND_NVIDIA, status="suspending")
        self.nvml_devices[SECOND_NVIDIA[0]] = {"total": 8 * GIB}
        self.nvml.order = list(self.nvml_devices)
        sources = [d.source for d in vram.snapshot()]
        self.assertEqual(sources, ["none", "held", "asleep"])
        self.assertEqual(self.loader_calls, 0)

    def test_without_nvml_nvidia_smi_is_asked(self):
        a = self.spawn()
        script = self.scratch / "fake-smi"
        script.write_text("#!/bin/sh\n"
                          "case \"$1\" in\n"
                          "  --query-gpu=*) echo '00000000:01:00.0, 6144, 1024, 5120' ;;\n"
                          f"  --query-compute-apps=*) echo '00000000:01:00.0, {a.pid}, 512'; echo 'junk' ;;\n"
                          "esac\n")
        script.chmod(stat.S_IRWXU)
        self.addCleanup(setattr, vram, "NVIDIA_SMI", vram.NVIDIA_SMI)
        vram.NVIDIA_SMI = str(script)

        def missing():
            raise OSError("libnvidia-ml.so.1: cannot open shared object file")
        vram.nvml_loader = missing
        nvidia = vram.snapshot()[1]
        self.assertEqual((nvidia.source, nvidia.total_bytes, nvidia.used_bytes, nvidia.free_bytes),
                         ("nvidia-smi", 6144 * MIB, 1024 * MIB, 5120 * MIB))
        self.assertEqual(dict(nvidia.processes), {a.pid: 512 * MIB})
        vram.NVIDIA_SMI = str(self.scratch / "absent")
        self.assertEqual(vram.snapshot()[1].source, "none")

    def test_pci_addresses_normalise(self):
        self.assertEqual(vram.normalise_pci("00000000:01:00.0"), "0000:01:00.0")
        self.assertEqual(vram.normalise_pci(" 0000:0A:00.1 "), "0000:0a:00.1")
        for bad in ("01:00.0", "0000:01:00", "Vulkan1", "0000:01:00.0x", ""):
            self.assertIsNone(vram.normalise_pci(bad))


class AmdTests(ResidencyCase):
    gpus = (INTEL, AMD)

    def test_amd_memory_comes_from_sysfs_and_clients_from_fdinfo(self):
        amd = self.sys / "devices" / "pci0000:00" / AMD[0]
        (amd / "mem_info_vram_total").write_text(str(8 * GIB))
        (amd / "mem_info_vram_used").write_text(str(2 * GIB))
        proc = self.scratch / "proc"
        for pid, fds in ((101, {"3": ("renderD131", AMD_FDINFO.format(client=1, vram=1024)),
                                "4": ("renderD131", AMD_FDINFO.format(client=1, vram=1024)),   # dup: same client
                                "5": ("renderD128", I915_FDINFO)}),                        # iGPU: not counted
                         (102, {"3": ("renderD131", AMD_FDINFO.format(client=2, vram=2048))}),
                         (103, {"0": (None, "pos: 0")})):
            (proc / str(pid) / "fd").mkdir(parents=True)
            (proc / str(pid) / "fdinfo").mkdir()
            for fd, (node, info) in fds.items():
                os.symlink(f"/dev/dri/{node}" if node else "/dev/null", proc / str(pid) / "fd" / fd)
                (proc / str(pid) / "fdinfo" / fd).write_text(info)
        os.environ["KILIX_GPU_PROC_ROOT"] = str(proc)
        [intel, device] = vram.snapshot()
        self.assertEqual((device.source, device.total_bytes, device.used_bytes, device.free_bytes, device.counter),
                         ("sysfs", 8 * GIB, 2 * GIB, 6 * GIB, True))
        self.assertEqual(dict(device.processes), {101: 1024 * 1024, 102: 2048 * 1024})
        self.assertEqual(dict(intel.processes), {})
        self.assertEqual(self.loader_calls, 0)

    def test_an_asleep_amd_gpu_is_not_read(self):
        amd = self.sys / "devices" / "pci0000:00" / AMD[0]
        (amd / "mem_info_vram_total").write_text(str(8 * GIB))
        (amd / "power" / "runtime_status").write_text("suspended\n")
        (amd / "mem_info_vram_used").mkdir()  # reading it would fail loudly; it must not be read
        device = vram.snapshot()[1]
        self.assertEqual((device.asleep, device.source, device.counter), (True, "asleep", True))


class FdinfoParseTests(ResidencyCase):
    def test_i915_counts_resident_system_memory(self):
        client = vram.parse_drm_fdinfo(I915_FDINFO)
        self.assertEqual((client.driver, client.pdev, client.client_id, client.memory_bytes),
                         ("i915", "0000:00:02.0", "7", 1024 * 1024))

    def test_device_local_memory_wins_and_cycles_are_not_memory(self):
        client = vram.parse_drm_fdinfo(AMD_FDINFO.format(client=3, vram=10))
        self.assertEqual(client.memory_bytes, 10 * 1024)
        self.assertEqual(vram.parse_drm_fdinfo("drm-driver: xe\ndrm-total-local0: 3 MiB\n"
                                               "drm-resident-local0: 2 MiB\ndrm-total-cycles-rcs: 9\n").memory_bytes,
                         2 * MIB)
        # Without device-local regions, every region counts, and still no cycles.
        self.assertEqual(vram.parse_drm_fdinfo("drm-driver: i915\ndrm-total-system0: 1 KiB\n"
                                               "drm-total-cycles-rcs: 5000\n").memory_bytes, 1024)
        self.assertIsNone(vram.parse_drm_fdinfo("pos: 0\nflags: 02\n"))
        self.assertEqual(vram.parse_drm_fdinfo("drm-driver: x\ndrm-total-vram0: 4 furlongs\n").memory_bytes, 0)


if __name__ == "__main__":
    import unittest
    unittest.main()
