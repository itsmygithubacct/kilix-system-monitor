"""ensure_room: measure, then unload idle residents least recently used first, or nothing."""
from __future__ import annotations

import os
import time

import residency_containment  # noqa: F401
import kilix_gpu_residency as residency
from fakes import AMD, GIB, INTEL, MIB, NVIDIA, SECOND_NVIDIA, ResidencyCase

# The fake GPU has 6 GiB with 100 MiB in base use.
FREE = 6 * GIB - 100 * MIB


class RoomTests(ResidencyCase):
    def room(self, need: int, **kwargs):
        kwargs.setdefault("device", "Vulkan1")
        kwargs.setdefault("requester", "kilix-chat")
        return residency.ensure_room(need, **kwargs, **self.where())

    def test_enough_free_vram_unloads_nothing(self):
        _, process = self.model("puzzllm", "a", GIB)
        room = self.room(4 * GIB)
        self.assertEqual((room.device_pci, room.need_bytes, room.evicted), ("0000:01:00.0", 4 * GIB, ()))
        self.assertEqual(room.free_bytes, FREE - GIB)
        self.assertIsNone(process.poll())

    def test_idle_residents_are_unloaded_least_recently_used_first_until_the_need_is_met(self):
        _, p1 = self.model("puzzllm", "oldest", 2 * GIB)
        time.sleep(0.01)
        _, p2 = self.model("bonsai", "middle", 2 * GIB)
        time.sleep(0.01)
        _, p3 = self.model("puzzllm", "newest", GIB)
        # Free is FREE - 5 GiB; 4 GiB needs the two oldest, not the newest.
        room = self.room(4 * GIB)
        self.assertEqual([e.model for e in room.evicted], ["oldest", "middle"])
        self.assertTrue(all(e.by == "kilix-chat" and e.reason == "room" for e in room.evicted))
        self.assertEqual([e.freed_bytes for e in room.evicted], [2 * GIB, 2 * GIB])
        self.assertIsNotNone(self.reap(p1))
        self.assertIsNotNone(self.reap(p2))
        self.assertIsNone(p3.poll())
        self.assertGreaterEqual(room.free_bytes, 4 * GIB)
        self.assertEqual([r.model for r in self.status().residents], ["newest"])

    def test_each_evicted_app_sees_its_own_eviction(self):
        oldest, _ = self.model("puzzllm", "oldest", 2 * GIB)
        time.sleep(0.01)
        middle, _ = self.model("bonsai", "middle", 3 * GIB)
        self.room(4 * GIB)
        self.assertEqual((oldest.evicted.model, middle.evicted.model), ("oldest", "middle"))

    def test_a_resident_without_an_unload_action_or_holding_nothing_is_left_alone(self):
        in_process = residency.register(app="voicebox", model="in-process", backend="process", **self.where())
        in_process.idle()
        self.processes[os.getpid()] = 3 * GIB  # this process holds it; nothing can unload it
        empty, empty_process = self.model("puzzllm", "empty", 0)
        time.sleep(0.01)
        _, full = self.model("puzzllm", "full", 2 * GIB)
        room = self.room(1 * GIB)
        self.assertEqual([e.model for e in room.evicted], ["full"])
        self.assertIsNone(empty_process.poll())
        self.assertEqual(sorted(r["model"] for r in self.records()), ["empty", "in-process"])

    def test_touching_a_resident_makes_it_recently_used(self):
        first, p1 = self.model("puzzllm", "first", 3 * GIB)
        time.sleep(0.01)
        self.model("puzzllm", "second", 2 * GIB)
        time.sleep(0.01)
        first.busy()
        first.idle()
        room = self.room(2 * GIB)
        self.assertEqual([e.model for e in room.evicted], ["second"])
        self.assertIsNone(p1.poll())

    def test_busy_pinned_loading_and_unmanaged_are_never_unloaded(self):
        self.model("puzzllm", "busy", GIB, state="busy")
        self.model("puzzllm", "pinned", GIB, pinned=True)
        self.model("puzzllm", "loading", GIB, state="loading")
        stranger = self.spawn(GIB)
        self.model("kilix-chat", "own-busy", GIB, state="busy")
        with self.assertRaises(residency.RoomError) as caught:
            self.room(2 * GIB)
        self.assertEqual(caught.exception.code, "no-room")
        self.assertEqual(caught.exception.plan, ())
        self.assertEqual(caught.exception.evicted, ())
        self.assertEqual(len(self.status().residents), 4)
        self.assertIsNone(stranger.poll())

    def test_the_requesters_own_idle_resident_can_be_unloaded(self):
        _, process = self.model("kilix-chat", "previous model", 3 * GIB)
        room = self.room(4 * GIB)
        self.assertEqual([e.model for e in room.evicted], ["previous model"])
        self.assertIsNotNone(self.reap(process))

    def test_when_unloading_everything_eligible_is_not_enough_nothing_is_unloaded(self):
        _, p1 = self.model("puzzllm", "a", GIB)
        _, p2 = self.model("puzzllm", "b", GIB)
        self.model("puzzllm", "busy", 3 * GIB, state="busy")
        with self.assertRaises(residency.RoomError) as caught:
            self.room(4 * GIB)
        error = caught.exception
        self.assertEqual(error.code, "no-room")
        self.assertEqual([r.model for r in error.plan], ["a", "b"])
        self.assertEqual(error.evicted, ())
        self.assertEqual((error.need_bytes, error.free_bytes, error.device), (4 * GIB, FREE - 5 * GIB, "0000:01:00.0"))
        self.assertIsNone(p1.poll())
        self.assertIsNone(p2.poll())
        self.assertEqual(self.ollama.unloads(), [])

    def test_policy_never_reports_the_plan_and_unloads_nothing(self):
        _, process = self.model("puzzllm", "a", 3 * GIB)
        with self.assertRaises(residency.RoomError) as caught:
            self.room(4 * GIB, policy="never")
        self.assertEqual([r.model for r in caught.exception.plan], ["a"])
        self.assertEqual(caught.exception.evicted, ())
        self.assertIsNone(process.poll())
        self.assertEqual(len(self.records()), 1)

    def test_unknown_policies_and_bad_sizes_are_refused(self):
        for kwargs in ({"policy": "evict-all"}, {"requester": "has space"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(residency.ResidencyError) as caught:
                self.room(GIB, **kwargs)
            self.assertEqual(caught.exception.code, "invalid-request")
        for need in (0, -1, 1.5, "1G", True):
            with self.subTest(need=need), self.assertRaises(residency.ResidencyError) as caught:
                self.room(need)
            self.assertEqual(caught.exception.code, "invalid-request")

    def test_idle_ollama_models_nobody_registered_are_evictable_unless_pinned(self):
        self.ollama.load("bonsai-2:27b", 5 * GIB)
        self.ollama.load("qwen3.5:2b", 512 * MIB, expires_in=600)
        residency.pin("qwen3.5:2b", **self.where())
        room = self.room(4 * GIB)
        self.assertEqual([e.model for e in room.evicted], ["bonsai-2:27b"])
        self.assertEqual(room.evicted[0].app, "ollama")
        self.assertEqual(self.ollama.unloads(), ["bonsai-2:27b"])
        self.assertEqual(list(self.ollama.models), ["qwen3.5:2b"])

    def test_a_loading_reservation_counts_against_free_vram(self):
        # Registering with need_bytes makes room and reserves it under one lock.
        first = residency.register(app="puzzllm", model="loading", backend="process", device="Vulkan1",
                                   need_bytes=4 * GIB, **self.where())
        self.assertEqual(first.room.evicted, ())
        with self.assertRaises(residency.RoomError):
            self.room(2 * GIB)
        # Once the loader holds what it reserved, the reservation is what it holds.
        process = self.spawn(3 * GIB)
        first.attach(pgid=process.pid)
        self.assertLess(self.room(GIB // 2).free_bytes, FREE - 3 * GIB)
        first.busy()  # loaded: only measured use counts now
        self.assertEqual(self.room(2 * GIB).free_bytes, FREE - 3 * GIB)

    def test_register_with_need_bytes_unloads_first_and_records_the_room(self):
        _, old = self.model("bonsai", "old", 4 * GIB)
        resident = residency.register(app="kilix-chat", model="new", backend="process", device="Vulkan1",
                                      need_bytes=3 * GIB, **self.where())
        self.assertEqual([e.model for e in resident.room.evicted], ["old"])
        self.assertIsNotNone(self.reap(old))
        self.assertEqual([r["model"] for r in self.records()], ["new"])

    def test_an_evicted_app_sees_its_record_gone_and_why(self):
        victim, process = self.model("puzzllm", "victim", 4 * GIB)
        self.room(3 * GIB)
        event = victim.evicted
        self.assertEqual((event.model, event.app, event.by, event.reason), ("victim", "puzzllm", "kilix-chat", "room"))
        with self.assertRaises(residency.ResidencyError) as caught:
            victim.busy()
        self.assertEqual(caught.exception.code, "gone")
        self.assertEqual([e.model for e in self.status().events], ["victim"])

    def test_a_failed_unload_stops_and_leaves_the_resident_unevictable(self):
        _, stubborn = self.model("puzzllm", "stubborn", 4 * GIB, ignore_term=True)
        self.addCleanup(setattr, residency, "KILL_GRACE_SECONDS", residency.KILL_GRACE_SECONDS)
        residency.KILL_GRACE_SECONDS = 0.2
        original = os.killpg
        self.addCleanup(setattr, os, "killpg", original)
        os.killpg = lambda pgid, sig: original(pgid, sig) if sig != 9 else None  # KILL "lost"
        with self.assertRaises(residency.RoomError) as caught:
            self.room(3 * GIB)
        self.assertEqual(caught.exception.evicted, ())
        self.assertIn("Unloading failed", str(caught.exception))
        os.killpg = original
        self.assertEqual([(r.model, r.state) for r in self.status().residents], [("stubborn", "busy")])
        with self.assertRaises(residency.RoomError):
            self.room(3 * GIB)
        self.assertIsNone(stubborn.poll())

    def test_vram_that_does_not_come_back_is_reported(self):
        _, process = self.model("puzzllm", "leaky", 4 * GIB)
        # The driver keeps the memory after the process exits.
        self.nvml_devices[NVIDIA[0]]["extra"] = lambda: 4 * GIB if process.poll() is not None else 0
        with self.assertRaises(residency.RoomError) as caught:
            self.room(3 * GIB)
        self.assertEqual([e.model for e in caught.exception.evicted], ["leaky"])
        self.assertIn("did not come back", str(caught.exception))

    def test_an_unmeasurable_device_is_reported_not_guessed(self):
        with self.assertRaises(residency.RoomError) as caught:
            self.room(GIB, device="0000:00:02.0")  # the Intel iGPU has no VRAM counter
        self.assertEqual(caught.exception.code, "unmeasured")


class DeviceTests(ResidencyCase):
    gpus = (INTEL, NVIDIA, SECOND_NVIDIA)

    def test_residents_on_another_device_are_never_unloaded(self):
        other = self.spawn(4 * GIB)
        self.nvml_devices[SECOND_NVIDIA[0]]["compute"] = lambda: {other.pid: 4 * GIB}
        self.nvml_devices[NVIDIA[0]]["compute"] = lambda: {}
        residency.register(app="puzzllm", model="elsewhere", backend="process", device=SECOND_NVIDIA[0],
                           pgid=other.pid, **self.where()).idle()
        room = residency.ensure_room(5 * GIB, device=NVIDIA[0], requester="kilix-chat", **self.where())
        self.assertEqual(room.evicted, ())
        with self.assertRaises(residency.RoomError) as caught:
            residency.ensure_room(6 * GIB, device=NVIDIA[0], requester="kilix-chat", **self.where())
        self.assertEqual((caught.exception.plan, caught.exception.evicted), ((), ()))
        self.assertIsNone(other.poll())

    def test_labels_resolve_by_pci_address_then_the_device_map(self):
        devices = residency.status(**self.where(), wake=True).devices
        self.assertEqual(residency.resolve_device("0000:02:00.0", devices), "0000:02:00.0")
        self.assertEqual(residency.resolve_device("0000:02:00.0".upper(), devices), "0000:02:00.0")
        # Two GPUs have VRAM counters, so a bare label is ambiguous without the map.
        self.assertIsNone(residency.resolve_device("Vulkan1", devices))
        with self.assertRaises(residency.RoomError) as caught:
            residency.ensure_room(GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual(caught.exception.code, "unmeasured")
        os.environ["KILIX_GPU_DEVICES"] = "Vulkan1=0000:02:00.0, CUDA0=0000:01:00.0;bad=nope"
        self.assertEqual(residency.resolve_device("Vulkan1", devices), "0000:02:00.0")
        self.assertEqual(residency.resolve_device("CUDA0", devices), "0000:01:00.0")
        self.assertIsNone(residency.resolve_device("bad", devices))
        room = residency.ensure_room(GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual(room.device_pci, "0000:02:00.0")


class OnlyMeasuredDeviceTests(ResidencyCase):
    gpus = (INTEL, AMD)

    def test_a_label_means_the_one_gpu_with_a_vram_counter(self):
        amd = self.sys / "devices" / "pci0000:00" / AMD[0]
        (amd / "mem_info_vram_total").write_text(str(8 * GIB))
        (amd / "mem_info_vram_used").write_text(str(GIB))
        room = residency.ensure_room(GIB, device="Vulkan0", requester="kilix-chat", **self.where())
        self.assertEqual((room.device_pci, room.free_bytes), (AMD[0], 7 * GIB))
        self.assertEqual(self.loader_calls, 0)  # no NVIDIA GPU: NVML is never loaded


if __name__ == "__main__":
    import unittest
    unittest.main()
