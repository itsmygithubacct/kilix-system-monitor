"""Adopted from the security seat's P1 review controls (review-p1/security/controls/test_sec6_ollama_identity.py).
Only the Required controls are kept; the Observed ones recorded the defects this round fixed.

S6: "is this process Ollama's?" is answered by a name any process can take.

_is_ollama trusts /proc/<pid>/comm (or the parent's) being "ollama",
"ollama_llama_se" or "ollama_llama_server". Any process of any user can name
itself that. Such a process
  (a) disappears from the unmanaged list whenever Ollama has a model loaded, and
  (b) decides which GPU every unregistered Ollama model is attributed to: the
      device where "Ollama's" processes hold the most memory.
With two GPUs, a decoy named "ollama" holding memory on GPU 2 moves the real
Ollama models (on GPU 1) to GPU 2, and ensure_room on GPU 2 then unloads them,
freeing nothing there. Contract: eviction.never_evicted includes "other-device"
and "unmanaged".
"""
from __future__ import annotations

import os

import residency_containment  # noqa: F401
import kilix_gpu_residency as residency
from fakes import GIB, INTEL, NVIDIA, SECOND_NVIDIA, ResidencyCase
from review_common import spawn_named

GPU1, GPU2 = NVIDIA[0], SECOND_NVIDIA[0]


class _Decoy(ResidencyCase):
    gpus = (INTEL, NVIDIA, SECOND_NVIDIA)

    def setUp(self):
        super().setUp()
        os.environ["KILIX_GPU_DEVICES"] = f"Vulkan1={GPU1},Vulkan2={GPU2}"
        # Per-device process lists (the shared harness lists every process on every GPU).
        self.on_gpu2: dict[int, int] = {}
        self.nvml_devices[GPU1]["compute"] = lambda: {**self.processes, **self.ollama.held()}
        self.nvml_devices[GPU2]["compute"] = lambda: dict(self.on_gpu2)

    def decoy_on_gpu2(self):
        self.ollama.load("real:7b", 2 * GIB)  # the real Ollama runner holds it on GPU 1
        decoy = spawn_named(self, "ollama", 3 * GIB, holds_on=self.on_gpu2)
        status = self.status(wake=True)
        gpu1 = next(d for d in status.devices if d.pci == GPU1)
        # The branch under test: the model's memory is measured on GPU 1, the decoy's on GPU 2.
        self.assertEqual(gpu1.processes.get(self.ollama.runner_pid), 2 * GIB)
        return decoy, status


class _Hidden(ResidencyCase):
    def hidden(self):
        self.ollama.load("real:7b", GIB)
        decoy = spawn_named(self, "ollama", GIB)
        return decoy, self.status()


class RequiredDecoyOllama(_Decoy):
    def test_models_are_never_unloaded_for_another_device(self):
        self.decoy_on_gpu2()
        try:
            residency.ensure_room(4 * GIB, device="Vulkan2", requester="bonsai", **self.where())
        except residency.RoomError:
            pass
        self.assertEqual(self.ollama.unloads(), [],
                         f"S6: a process that named itself 'ollama' and holds VRAM on {GPU2} made ensure_room on "
                         f"{GPU2} unload Ollama's real:7b, which is resident on {GPU1} (contract never_evicted: "
                         "other-device)")


class RequiredHiddenFromUnmanaged(_Hidden):
    def test_a_gpu_process_that_is_not_ollamas_is_listed_unmanaged(self):
        decoy, status = self.hidden()
        self.assertIn(decoy.pid, [u.pid for u in status.unmanaged],
                      f"S6: GPU process {decoy.pid} (not Ollama's; it only set its comm to 'ollama') is missing "
                      "from the unmanaged list")
