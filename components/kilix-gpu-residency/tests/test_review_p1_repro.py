"""Adopted from the third reviewer's reproductions (REVIEW-P1 / REVIEW-P1B).

Reviewer reproductions. Each assertion states the SAFE/expected behaviour;
a FAIL or ERROR here means the finding reproduces."""
from __future__ import annotations

import json
import os
from pathlib import Path
import types

import residency_containment  # noqa: F401
import kilix_gpu_residency as residency
from fakes import GIB, MIB, ResidencyCase


class Ids:
    """Deterministic record ids, so record order (sorted by id) is known."""
    def __init__(self):
        self.n = 0

    def uuid4(self):
        self.n += 1
        return types.SimpleNamespace(hex=f"{self.n:032x}")


class Repro(ResidencyCase):
    def setUp(self):
        super().setUp()
        original = residency.uuid
        residency.uuid = Ids()
        self.addCleanup(setattr, residency, "uuid", original)

    def test_R1_busy_drops_the_reservation_while_the_model_is_still_loading(self):
        a = residency.register(app="kilix-chat", model="big", backend="process", device="Vulkan1",
                               need_bytes=4 * GIB, **self.where())
        p = self.spawn(512 * MIB)          # the loader has allocated 0.5 of its 4 GiB so far
        a.attach(pgid=p.pid)
        a.busy()                           # exactly as KILIX-CHAT-INTEGRATION.md says, right after Popen
        # 5.9 GiB - 0.5 held - 3.5 still to come = 1.9 GiB really available
        with self.assertRaises(residency.RoomError):
            residency.register(app="puzzllm", model="other", backend="process", device="Vulkan1",
                               need_bytes=4 * GIB, **self.where())

    def test_R2a_evicting_an_idle_ollama_record_unloads_a_model_another_app_is_busy_with(self):
        self.ollama.load("qwen3.5:4b", 4 * GIB)
        a = residency.register(app="appa", model="qwen3.5:4b", backend="ollama", **self.where())
        a.idle()
        b = residency.register(app="appb", model="qwen3.5:4b", backend="ollama", **self.where())
        b.busy()
        try:
            residency.ensure_room(3 * GIB, device="Vulkan1", requester="puzzllm", **self.where())
        except residency.RoomError:
            pass
        self.assertEqual(self.ollama.unloads(), [], "a model a busy record uses was unloaded")

    def test_R2b_closing_one_ollama_registration_unloads_a_model_another_app_is_busy_with(self):
        self.ollama.load("qwen3.5:4b", 4 * GIB)
        b = residency.register(app="appb", model="qwen3.5:4b", backend="ollama", **self.where())
        b.busy()
        with residency.register(app="appa", model="qwen3.5:4b", backend="ollama", **self.where()) as a:
            a.idle()
        self.assertEqual(self.ollama.unloads(), [], "a model a busy record uses was unloaded")

    def test_R3_an_owner_pin_is_lost_once_an_app_registers_the_same_ollama_model(self):
        self.ollama.load("bonsai-2:27b", 5 * GIB)
        residency.pin("bonsai-2:27b", **self.where())
        r = residency.register(app="kilix-chat", model="bonsai-2:27b", backend="ollama", **self.where())
        r.idle()
        try:
            residency.ensure_room(4 * GIB, device="Vulkan1", requester="puzzllm", **self.where())
        except residency.RoomError:
            pass
        self.assertEqual(self.ollama.unloads(), [], "the owner's pinned model was unloaded")

    def test_R4_the_igpu_label_resolves_to_the_nvidia_gpu_and_evicts_there(self):
        _, p = self.model("puzzllm", "on-nvidia", 4 * GIB)
        try:
            room = residency.ensure_room(3 * GIB, device="Vulkan0", requester="kilix-chat", **self.where())
            resolved = room.device_pci
        except residency.RoomError as error:
            resolved = error.device
        self.assertIsNone(p.poll(), "a model on the NVIDIA GPU was evicted for a Vulkan0 (iGPU) request")
        self.assertNotEqual(resolved, "0000:01:00.0")

    def test_R5_proc_root_override_bypasses_the_reused_pid_check(self):
        resident, p = self.model("puzzllm", "m", GIB)
        path = Path(self.namespace) / (resident.id + ".json")
        record = json.loads(path.read_text())
        record["holder"]["start"] += 1          # the recorded owner is gone; the PID is now someone else
        path.write_text(json.dumps(record))
        fake = self.scratch / "proc" / str(p.pid)
        fake.mkdir(parents=True)
        real = Path(f"/proc/{p.pid}/stat").read_text()
        head, tail = real[:real.rfind(")") + 2], real[real.rfind(")") + 2:].split()
        tail[19] = str(record["holder"]["start"])
        (fake / "stat").write_text(head + " ".join(tail) + "\n")
        (fake / "comm").write_text("sleep\n")
        os.environ["KILIX_GPU_PROC_ROOT"] = str(self.scratch / "proc")
        try:
            residency.unload(resident.id, force=True, **self.where())
        except residency.ResidencyError:
            pass
        self.assertIsNone(p.poll(), "a process whose start time does not match the record was signalled")

    def test_R6_one_record_with_a_future_unload_kind_takes_the_whole_registry_down(self):
        resident, _ = self.model("voicebox", "tts", GIB)
        path = Path(self.namespace) / (resident.id + ".json")
        record = json.loads(path.read_text())
        record["backend"], record["unload"] = "http", {"kind": "http", "url": "http://127.0.0.1:1/models/unload"}
        record["pgid"] = None
        path.write_text(json.dumps(record))
        os.chmod(path, 0o600)
        residency.status(**self.where())   # a P1 reader beside a P2 writer

    def test_R7_status_creates_the_registry(self):
        fresh = str(self.scratch / "never-used")
        residency.status(namespace=fresh, ollama=self.ollama.url)
        self.assertFalse(os.path.exists(fresh), "status created the registry directory and lock")

    def test_R8_close_unloads_a_pinned_ollama_record(self):
        self.ollama.load("qwen3.5:4b", 3 * GIB)
        with residency.register(app="kilix-chat", model="qwen3.5:4b", backend="ollama", pinned=True,
                                **self.where()) as r:
            r.idle()
        self.assertEqual(self.ollama.unloads(), [], "a pinned model was unloaded on block exit")
