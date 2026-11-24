"""Adopted from the security seat's P1 review controls (review-p1/security/controls/test_sec1_shared_ollama_unload.py).
Only the Required controls are kept; the Observed ones recorded the defects this round fixed.

S1: an Ollama model is one unload unit, but the registry evicts it per record.

Two records may name the same Ollama model (two apps using one model, or one app
twice). _view hands the model's VRAM to whichever record sorts first and gives
the others 0. If that first record is idle and unpinned, ensure_room evicts it,
and the unload (POST keep_alive 0 for the model name) takes the model away from
the busy or pinned sibling too. Likewise a `kilix-gpu pin` of an Ollama model
(pins.json) stops applying the moment any app registers that model.

Contract: eviction.never_evicted includes "busy" and "pinned".
README: pin(...) marks "an Ollama model nobody registered, so ensure_room never unloads it".
"""
from __future__ import annotations

import residency_containment  # noqa: F401
import kilix_gpu_residency as residency
from fakes import GIB, ResidencyCase

MODEL = "shared:7b"


class _Shared(ResidencyCase):
    def ollama_record(self, app: str, *, state: str, pinned: bool = False) -> residency.Resident:
        resident = residency.register(app=app, model=MODEL, backend="ollama", device="Vulkan1", pinned=pinned,
                                      **self.where())
        getattr(resident, state)()
        return resident

    def protected_after(self, evictable: residency.Resident, protect: str) -> residency.Resident:
        """Register the protected sibling until its id sorts after the evictable one's
        (_view walks records in sorted file-name order; uuid4 ids make that a coin flip)."""
        for _ in range(64):
            if protect == "busy":
                sibling = self.ollama_record("kilix-chat", state="busy")
            else:
                sibling = self.ollama_record("kilix-chat", state="idle", pinned=True)
            if sibling.id > evictable.id:
                return sibling
            sibling.close(unload=False)
        self.fail("could not order the records")

    def scenario(self, protect: str):
        self.ollama.load(MODEL, 4 * GIB)
        evictable = self.ollama_record("puzzllm", state="idle")
        sibling = self.protected_after(evictable, protect)
        before = {r.id: r for r in self.status().residents}
        # The branch under test: both records are listed, the sibling busy or pinned.
        self.assertEqual(before[sibling.id].state, "busy" if protect == "busy" else "idle")
        self.assertEqual(before[sibling.id].pinned, protect == "pinned")
        self.assertEqual((before[evictable.id].vram_bytes, before[sibling.id].vram_bytes), (4 * GIB, 0))
        return evictable, sibling, self.room()

    def room(self):
        """Adapted for fix round 1: with the shared model no longer a candidate, the room is refused."""
        try:
            return residency.ensure_room(3 * GIB, device="Vulkan1", requester="bonsai", **self.where())
        except residency.RoomError as error:
            self.assertEqual((error.code, error.evicted), ("no-room", ()))
            return error

    def pinned_by_cli_then_registered(self):
        self.ollama.load(MODEL, 4 * GIB)
        pinned = residency.pin(MODEL, **self.where())  # kilix-gpu pin shared:7b, while nobody registered it
        self.assertEqual((pinned.id, pinned.pinned, pinned.registered), ("ollama:" + MODEL, True, False))
        self.ollama_record("puzzllm", state="idle")  # an app later registers the same model, unpinned
        return self.room()


class RequiredSharedOllamaUnload(_Shared):
    def test_a_busy_residents_ollama_model_is_never_unloaded(self):
        _, sibling, _ = self.scenario("busy")
        self.assertNotIn(MODEL, self.ollama.unloads(),
                         f"S1: ensure_room unloaded Ollama model {MODEL!r} while record {sibling.id} holding it was "
                         "busy (contract eviction.never_evicted includes 'busy')")

    def test_a_pinned_residents_ollama_model_is_never_unloaded(self):
        _, sibling, _ = self.scenario("pinned")
        self.assertNotIn(MODEL, self.ollama.unloads(),
                         f"S1: ensure_room unloaded Ollama model {MODEL!r} while record {sibling.id} holding it was "
                         "pinned (contract eviction.never_evicted includes 'pinned')")

    def test_a_cli_pin_survives_an_app_registering_the_model(self):
        self.pinned_by_cli_then_registered()
        self.assertNotIn(MODEL, self.ollama.unloads(),
                         f"S1: `kilix-gpu pin {MODEL}` was ignored after an app registered the model; "
                         "ensure_room unloaded it")
