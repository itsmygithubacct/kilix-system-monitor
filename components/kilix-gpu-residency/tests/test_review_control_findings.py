"""Adopted from the control seat's P1 review controls (review-p1/control/controls/test_control_findings.py).
Only the Required controls are kept; the Observed ones recorded the defects this round fixed.

Control-seat controls for kilix.gpu-residency/v1 P1 at ae5a6af.

Observed* controls assert what the code does now and pass: they are the evidence.
Required* controls assert the README / interface contract and fail: the failure
message is the finding. Fn numbers are those of REVIEW-P1-control.md.

Run from components/kilix-gpu-residency, so the component's own harness
(tests/fakes.py ResidencyCase, tests/residency_containment.py guard) is used:

    uv run --locked --offline python -m unittest discover -s <controls> -v

Every case uses the component's fake sysfs, fake NVML (memory follows real
processes), fake loopback Ollama and private namespace, exactly as the
component suite does; a divergence therefore means the code, not this fixture.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

TESTS = Path(__file__).resolve().parent
if not (TESTS / "fakes.py").exists():
    raise SystemExit("run from components/kilix-gpu-residency")
sys.path.insert(0, str(TESTS))

import residency_containment  # noqa: E402,F401  installs the component's guard
import kilix_gpu_residency as residency  # noqa: E402
from kilix_gpu_residency import cli, ollama as ollama_module  # noqa: E402
from fakes import GIB, MIB, NVIDIA, ResidencyCase  # noqa: E402

COMPONENT = TESTS.parent
README = (COMPONENT / "README.md").read_text(encoding="utf-8")
FREE = 6 * GIB - 100 * MIB  # the fake GPU: 6 GiB, 100 MiB base use


def nvidia_free(status: residency.Status) -> int:
    [device] = [d for d in status.devices if d.pci == NVIDIA[0]]
    return device.free_bytes


# F1  The loading reservation ends at busy(), which the documented flow calls before the child allocates.

class ReservationControls(ResidencyCase):
    """README 'Using it from an app' and KILIX-CHAT-INTEGRATION.md: register(need_bytes=)
    -> Popen -> attach(pgid=child) -> busy() -> the child loads its weights."""

    def first_app_as_documented(self):
        first = residency.register(app="kilix-chat", model="LLaDA-MoE-7B", backend="process", device="Vulkan1",
                                   need_bytes=4 * GIB, **self.where())
        child = self.spawn(0)            # started; has not allocated yet (llama-diffusion-cli is still reading the GGUF)
        first.attach(pgid=child.pid)
        first.busy()                     # "around each turn" / "not evictable while it denoises"
        return first, child


    def test_RequiredTwoAppsCannotBothTakeTheSameFreeMemory(self):
        self.first_app_as_documented()
        try:
            second = residency.register(app="puzzllm", model="Dream-7B", backend="process", device="Vulkan1",
                                        need_bytes=4 * GIB, **self.where())
        except residency.RoomError:
            return
        self.fail("REQUIRED (README: 'While loading, it reserves need_bytes ... so two apps cannot both take the "
                  "same free memory'): after the documented register(need_bytes=4GiB) -> attach -> busy() and "
                  "before the child allocated anything, a second register(need_bytes=4GiB) on the same 5.9 GiB "
                  f"device was granted room (free_bytes={second.room.free_bytes}); busy() ended the reservation")


# F3  The README's own usage example raises ResidencyError("gone").

class ReadmeExampleControls(ResidencyCase):
    """Adapted for fix round 1 (FX-23): the README's block is run from the README text itself.

    Only what an app supplies is given here: ``command`` (its model process) and
    ``wait_until_loaded``; the registry and Ollama defaults point at this test's fakes."""

    def readme_example(self):
        blocks = re.findall(r"```python\n(.*?)```", README, re.S)
        [block] = [b for b in blocks if "resident.close(unload=False)" in b and "register(" in b]
        self.addCleanup(setattr, residency, "default_namespace", residency.default_namespace)
        residency.default_namespace = lambda: self.namespace
        os.environ["OLLAMA_HOST"] = self.ollama.url
        children = []

        def wait_until_loaded(child):
            children.append(child)
            self.addCleanup(lambda: child.poll() is None and child.kill())
        exec(compile(block, "README.md", "exec"), {"command": ["sleep", "1"], "wait_until_loaded": wait_until_loaded,
                                                   "notice": lambda text: None, "__name__": "readme"})
        return children

    def test_RequiredReadmeExampleRuns(self):
        try:
            children = self.readme_example()
        except residency.ResidencyError as error:
            self.fail("REQUIRED (README 'Using it from an app'): the documented example must run; "
                      f"it raised ResidencyError({error.code!r}: {error})")
        self.assertEqual(len(children), 1)
        self.assertEqual(self.records(), [])  # the block's finally removed the record


# F4  register(need_bytes=) evicts other apps' models and then refuses because the registry is full.

class FullRegistryControls(ResidencyCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, residency, "MAX_RECORDS", residency.MAX_RECORDS)
        residency.MAX_RECORDS = 2   # the suite's own test_the_registry_holds_at_most_max_records patches it the same way
        for _ in range(2):
            residency.register(app="voicebox", model="in-process", backend="process", **self.where())
        self.ollama.load("bonsai-2:27b", 5 * GIB)   # idle, unregistered, evictable by owner decision

    def attempt(self):
        with self.assertRaises(residency.ResidencyError) as caught:
            residency.register(app="kilix-chat", model="LLaDA", backend="process", device="Vulkan1",
                               need_bytes=3 * GIB, **self.where())
        return caught.exception


    def test_RequiredARefusedRegistrationUnloadsNothing(self):
        error = self.attempt()
        self.assertEqual(self.ollama.unloads(), [],
                         "REQUIRED (README: 'With need_bytes, room is made and the record written under one lock'; "
                         "contract eviction.all_or_nothing): register() refused with "
                         f"[{error.code}] {error} after it had already unloaded {self.ollama.unloads()} for a "
                         "record it never wrote")


# F6  Ollama models nobody registered are ordered by expires_at - 300 s, not by last use.

class OllamaOrderControls(ResidencyCase):
    def setUp(self):
        super().setUp()
        # "stale": loaded with keep_alive 1h by some client and last used 30 minutes ago -> expires in 30 min.
        # "fresh": used just now with Ollama's default keep_alive 5m                    -> expires in 5 min.
        self.ollama.load("stale:7b", 2 * GIB, expires_in=1800)
        self.ollama.load("fresh:7b", 2 * GIB, expires_in=300)


    def test_RequiredLeastRecentlyUsedFirst(self):
        room = residency.ensure_room(3 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual([e.model for e in room.evicted], ["stale:7b"],
                         "REQUIRED (README 'least recently used first'; contract eviction.order "
                         "'least-recently-used'): the model idle for 30 min (keep_alive 1h) must go before the "
                         "one used just now; last_used is estimated as expires_at - 300 s, which puts the "
                         "first 25 min in the future, and the README and contract do not say the order is an estimate")


# F2  A pin on an Ollama model stops protecting it once any app registers that model.

class PinControls(ResidencyCase):
    def setUp(self):
        super().setUp()
        self.ollama.load("qwen3.5:4b", 4 * GIB)
        residency.pin("qwen3.5:4b", **self.where())            # the owner: keep this model resident
        self.resident = residency.register(app="kilix-chat", model="qwen3.5:4b", backend="ollama", **self.where())
        self.resident.idle()


    def test_RequiredAPinnedModelIsNeverUnloadedToMakeRoom(self):
        try:
            room = residency.ensure_room(3 * GIB, device="Vulkan1", requester="puzzllm", **self.where())
        except residency.RoomError:
            return
        self.fail("REQUIRED (README: 'pin(id_or_model) ... mark a resident, including an Ollama model nobody "
                  "registered, so ensure_room never unloads it'): `kilix-gpu pin qwen3.5:4b` was honoured only "
                  "until kilix-chat registered the same model; ensure_room then unloaded "
                  f"{[e.model for e in room.evicted]} although pins.json still lists it")


# F5  A process group shared by two records is counted once per record.

class SharedGroupControls(ResidencyCase):
    def setUp(self):
        super().setUp()
        # One server process group serving two models (e.g. a `serve` holding two GGUFs), 1 GiB in all.
        self.server = self.spawn(1 * GIB)
        self.a = residency.register(app="kilix-chat", model="model-a", backend="process", device="Vulkan1",
                                    pgid=self.server.pid, **self.where())
        self.a.idle()
        time.sleep(0.01)
        self.b = residency.register(app="kilix-chat", model="model-b", backend="process", device="Vulkan1",
                                    pgid=self.server.pid, **self.where())
        self.b.idle()
        time.sleep(0.01)
        _, self.other = self.model("puzzllm", "model-c", 1 * GIB)   # another 1 GiB, newest
        # Real use 2 GiB, free 3.9 GiB. 5.5 GiB is reachable by unloading the server and model-c.


    def test_RequiredRoomIsMadeWhenUnloadingCanMakeIt(self):
        try:
            residency.ensure_room(int(5.5 * GIB), device="Vulkan1", requester="bonsai", **self.where())
        except residency.RoomError as error:
            self.fail("REQUIRED (README: candidates are planned 'until the need is met'; 'If even every candidate "
                      "would not be enough, it raises RoomError and unloads nothing'): unloading the shared group "
                      "and model-c frees 2 GiB and meets 5.5 GiB, but the plan counted the one group twice "
                      f"({[r.model for r in error.plan]}), unloaded {[e.model for e in error.evicted]} and then "
                      f"failed: [{error.code}] {error}")


# F7  OLLAMA_HOST forms that name this machine disable Ollama discovery, or escape as ValueError.

class OllamaHostControls(ResidencyCase):

    def test_RequiredBracketedIpv6LoopbackIsLoopback(self):
        os.environ["OLLAMA_HOST"] = "[::1]:11434"
        self.assertIsNotNone(ollama_module.loopback(ollama_module.default_url()),
                             "REQUIRED (README: 'The default URL comes from OLLAMA_HOST'; Ollama is contacted on "
                             "loopback): OLLAMA_HOST=[::1]:11434 is loopback, but default_url() gives "
                             f"{ollama_module.default_url()!r}, so status reports 'not-loopback' and Ollama's "
                             "models are never discovered or unloaded")


    def test_RequiredEveryRefusalIsAResidencyError(self):
        os.environ["OLLAMA_HOST"] = "127.0.0.1:99999"
        try:
            residency.status(namespace=self.namespace)
        except residency.ResidencyError:
            return
        except Exception as error:  # noqa: BLE001
            self.fail("REQUIRED (README: 'ResidencyError.code is one of invalid-request, unavailable, ...'): "
                      f"status() with OLLAMA_HOST=127.0.0.1:99999 raised {type(error).__name__}: {error}")


# F8  `kilix-gpu room` exits 1 on an unmeasurable device; the README documents only 0 and 2.

class RoomExitControls(ResidencyCase):

    def test_RequiredReadmeDocumentsEveryRoomExitCode(self):
        [row] = [line for line in README.splitlines() if line.startswith("| `kilix-gpu room")]
        self.assertRegex(row, r"\b1\b",
                         "REQUIRED: the README's `kilix-gpu room` row says 'exits 0 if it fits or would fit after "
                         "unloading, 2 if it cannot' but an unmeasurable device or a refused request exits 1")


# F9  test_interface_document.py does not hold the module to the document's eviction rules.

class InterfaceDocumentControls(ResidencyCase):
    """Apply the component's own planted defect 'busy-residents-evicted' (tests/defects.py) to a private copy
    (tests/run_defects.copy_tree/apply) and run only tests/test_interface_document.py there."""

    def interface_suite_on(self, defect: str | None) -> subprocess.CompletedProcess:
        import run_defects
        from defects import DEFECTS
        component = run_defects.copy_tree(self.scratch / ("copy-" + (defect or "none")))
        if defect is not None:
            [(_, filename, edits, *_rest)] = [d for d in DEFECTS if d[0] == defect]
            run_defects.apply(component, filename, edits)
            source = (component / "src" / "kilix_gpu_residency" / "__init__.py").read_text()
            self.assertIn('_state(record) in ("idle", "busy")', source)   # the drift is really in the copy
        done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-v", "-s", "tests",
                               "-p", "test_interface_document.py"], cwd=component,
                              env=run_defects.environment(component), capture_output=True, text=True, timeout=120)
        # Branch reached: the interface tests ran (not an import error). Adapted for fix round 1: no
        # count literal, and the eviction check is the behaviour probe that replaced the literal one.
        self.assertRegex(done.stderr, r"Ran [1-9][0-9]* tests", done.stderr[-2000:])
        self.assertIn("test_every_never_evicted_reason_keeps_a_unit_out_of_the_plan", done.stderr)
        return done


    def test_RequiredTheInterfaceTestHoldsTheModuleToTheEvictionRules(self):
        done = self.interface_suite_on("busy-residents-evicted")
        self.assertNotEqual(done.returncode, 0,
                            "REQUIRED (README 'Interface document': the document records 'the eviction rules', and "
                            "test_interface_document.py 'holds the module to that document'; docs/contracts.md the "
                            "same): with the module evicting busy residents, contrary to eviction.never_evicted, "
                            "test_interface_document.py still passes (6/6); its eviction check compares the "
                            "document with a literal copy of itself")


if __name__ == "__main__":
    import unittest
    unittest.main()
