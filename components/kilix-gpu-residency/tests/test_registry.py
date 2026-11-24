"""Registration, measured VRAM, state, pruning, validation and pins."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import residency_containment
import kilix_gpu_residency as residency
from fakes import GIB, MIB, ResidencyCase, wait_for


class RegistrationTests(ResidencyCase):
    def test_a_record_holds_exactly_the_documented_fields(self):
        resident, process = self.model("kilix-chat", "LLaDA-MoE-7B Q4_K_M", 2 * GIB, state="loading",
                                       need_bytes=3 * GIB)
        [record] = self.records()
        self.assertEqual(set(record), {"version", "id", "app", "model", "backend", "device", "holder", "pgid",
                                       "need_bytes", "state", "pinned", "registered_at", "last_used", "unload",
                                       "reserved_until", "pidns", "claim"})
        self.assertEqual(record["id"], resident.id)
        self.assertEqual(record["version"], residency.VERSION)
        self.assertEqual((record["app"], record["model"], record["backend"], record["state"]),
                         ("kilix-chat", "LLaDA-MoE-7B Q4_K_M", "process", "loading"))
        self.assertEqual(record["device"], {"label": "Vulkan1", "pci": "0000:01:00.0"})
        self.assertEqual(record["pgid"], process.pid)
        self.assertEqual(record["holder"]["pid"], process.pid)
        self.assertEqual(record["unload"], {"kind": "signal", "pgid": process.pid})
        self.assertEqual(record["need_bytes"], 3 * GIB)
        self.assertNotIn("vram_bytes", record)  # measured on read, never stored or declared
        mode = os.stat(Path(self.namespace) / (resident.id + ".json")).st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_vram_is_measured_from_the_whole_process_group(self):
        resident, process = self.model("puzzllm", "Dream-Coder-7B", 1 * GIB)
        child = self.join(process, 512 * MIB)
        [info] = self.status().residents
        self.assertEqual(info.vram_bytes, 1 * GIB + 512 * MIB)
        self.assertEqual(info.device_pci, "0000:01:00.0")
        self.processes[child.pid] = 256 * MIB
        [info] = self.status().residents
        self.assertEqual(info.vram_bytes, 1 * GIB + 256 * MIB)

    def test_busy_and_idle_change_state_and_last_used(self):
        resident, _ = self.model("kilix-chat", "m", GIB, state="loading")
        first = self.records()[0]["last_used"]
        time.sleep(0.01)
        resident.busy()
        self.assertEqual(self.records()[0]["state"], "busy")
        self.assertGreater(self.records()[0]["last_used"], first)
        resident.idle()
        self.assertEqual(self.records()[0]["state"], "idle")
        [info] = self.status().residents
        self.assertEqual(info.state, "idle")

    def test_a_record_is_pruned_once_its_process_is_gone(self):
        resident, process = self.model("kilix-chat", "m", GIB)
        process.kill()
        self.reap(process)
        self.assertEqual(self.status().residents, ())
        self.assertEqual(len(self.records()), 1)  # reading prunes nothing
        self.prune()
        self.assertEqual(self.records(), [])
        with self.assertRaises(residency.ResidencyError) as caught:
            resident.busy()
        self.assertEqual(caught.exception.code, "gone")

    def test_updating_a_record_whose_process_died_says_gone(self):
        resident, process = self.model("kilix-chat", "m", GIB)
        process.kill()
        self.reap(process)
        with self.assertRaises(residency.ResidencyError) as caught:
            resident.idle()
        self.assertEqual(caught.exception.code, "gone")
        self.assertEqual(self.records(), [])

    def test_a_reused_pid_is_not_mistaken_for_the_owner(self):
        resident, process = self.model("kilix-chat", "m", GIB)
        path = Path(self.namespace) / (resident.id + ".json")
        record = json.loads(path.read_text())
        record["holder"]["start"] += 1  # the same PID, started at another time
        path.write_text(json.dumps(record))
        self.assertEqual(self.status().residents, ())
        self.assertIsNone(process.poll())  # pruned, never signalled

    def test_a_record_without_a_process_follows_the_registering_process(self):
        resident = residency.register(app="kilix-chat", model="m", backend="process", **self.where())
        [record] = self.records()
        self.assertEqual(record["holder"]["pid"], os.getpid())
        self.assertIsNone(record["unload"])
        self.assertEqual(len(self.status().residents), 1)
        resident.close()
        self.assertEqual(self.records(), [])

    def test_attach_points_the_record_at_a_process_started_later(self):
        resident = residency.register(app="kilix-chat", model="m", backend="process", device="Vulkan1",
                                      **self.where())
        process = self.spawn(GIB)
        resident.attach(pgid=process.pid)
        [record] = self.records()
        self.assertEqual(record["holder"]["pid"], process.pid)
        self.assertEqual(record["unload"], {"kind": "signal", "pgid": process.pid})
        self.assertEqual(self.status().residents[0].vram_bytes, GIB)
        process.kill()
        self.reap(process)
        self.assertEqual(self.status().residents, ())

    def test_leaving_the_block_removes_the_record_and_unloads_a_live_group(self):
        process = self.spawn(GIB)
        with residency.register(app="kilix-chat", model="m", backend="process", pgid=process.pid,
                                **self.where()) as resident:
            resident.idle()
            self.assertEqual(len(self.records()), 1)
        self.assertEqual(self.records(), [])
        self.assertTrue(wait_for(lambda: process.poll() is not None))

    def test_close_without_unload_leaves_the_process_running(self):
        resident, process = self.model("kilix-chat", "m", GIB)
        resident.close(unload=False)
        self.assertEqual(self.records(), [])
        self.assertIsNone(process.poll())

    def test_invalid_requests_are_refused_before_anything_is_written(self):
        process = self.spawn()
        bad = [
            dict(app="", model="m", backend="process"),
            dict(app="-x", model="m", backend="process"),
            dict(app="a b", model="m", backend="process"),
            dict(app="a", model="", backend="process"),
            dict(app="a", model="line\nbreak", backend="process"),
            dict(app="a", model="x" * 129, backend="process"),
            dict(app="a", model="m", backend="cuda"),
            dict(app="a", model="m", backend="process", device="Vulkan 1"),
            dict(app="a", model="m", backend="process", need_bytes=0),
            dict(app="a", model="m", backend="process", need_bytes=1.5),
            dict(app="a", model="m", backend="process", pinned="yes"),
            dict(app="a", model="has space", backend="ollama"),
            dict(app="a", model="m", backend="process", pgid=1),
            dict(app="a", model="m", backend="process", pid=process.pid, unload={"kind": "signal",
                                                                               "pgid": process.pid}),
            dict(app="a", model="m", backend="process", pgid=process.pid, unload={"kind": "ollama",
                                                                                "model": "m"}),
            dict(app="a", model="m", backend="process", pgid=process.pid, unload={"kind": "shell",
                                                                                "command": "rm"}),
            dict(app="a", model="m", backend="ollama", unload=None),
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(residency.ResidencyError) as caught:
                    residency.register(**kwargs, **self.where())
                self.assertEqual(caught.exception.code, "invalid-request")
        self.assertEqual(self.records(), [])

    def test_a_process_cannot_register_or_signal_its_own_group(self):
        # A child that leads its own group, so the leader is visible in any PID namespace.
        code = textwrap.dedent("""
            import json, os, sys
            import kilix_gpu_residency as residency
            os.setpgid(0, 0)
            results = {}
            try:
                residency.register(app="a", model="m", backend="process", pgid=os.getpgrp(),
                                   namespace=sys.argv[1], ollama=sys.argv[2])
                results["register"] = "registered"
            except residency.ResidencyError as error:
                results["register"] = error.code
            record = {"holder": {"pid": os.getpid(), "start": residency._start(os.getpid())},
                      "pgid": os.getpid(), "unload": {"kind": "signal", "pgid": os.getpid()}}
            try:
                residency._signal_unload(record)
                results["signal"] = "signalled"
            except residency.ResidencyError as error:
                results["signal"] = error.code
            print(json.dumps(results))
        """)
        done = subprocess.run([sys.executable, "-c", code, self.namespace, self.ollama.url], capture_output=True,
                              text=True, timeout=60, env=residency_containment.child_env())
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout), {"register": "invalid-request", "signal": "refused"})
        self.assertEqual(self.records(), [])

    def test_a_pgid_that_does_not_lead_a_group_is_refused(self):
        process = self.spawn()
        child = self.join(process)
        with self.assertRaises(residency.ResidencyError) as caught:
            residency.register(app="a", model="m", backend="process", pgid=child.pid, **self.where())
        self.assertEqual(caught.exception.code, "invalid-request")

    def test_pin_and_unpin_a_registered_resident(self):
        resident, _ = self.model("kilix-chat", "m", GIB)
        self.assertTrue(residency.pin(resident.id, **self.where()).pinned)
        self.assertTrue(self.records()[0]["pinned"])
        self.assertTrue(self.status().residents[0].pinned)
        self.assertFalse(residency.unpin("m", **self.where()).pinned)
        self.assertFalse(self.records()[0]["pinned"])

    def test_the_registry_holds_at_most_max_records(self):
        bound = residency.MAX_RECORDS
        self.addCleanup(setattr, residency, "MAX_RECORDS", bound)
        residency.MAX_RECORDS = 3
        for _ in range(3):
            residency.register(app="a", model="m", backend="process", **self.where())
        with self.assertRaises(residency.ResidencyError) as caught:
            residency.register(app="a", model="m", backend="process", **self.where())
        self.assertEqual(caught.exception.code, "unavailable")
        self.assertEqual(len(self.records()), 3)

    def test_unmanaged_gpu_processes_are_listed_and_never_unloaded(self):
        stranger = self.spawn(700 * MIB)
        [listed] = self.status().unmanaged
        self.assertEqual((listed.pid, listed.command, listed.device_pci, listed.vram_bytes),
                         (stranger.pid, "sleep", "0000:01:00.0", 700 * MIB))
        for target in ("sleep", str(stranger.pid)):
            with self.assertRaises(residency.ResidencyError) as caught:
                residency.unload(target, force=True, **self.where())
            self.assertEqual(caught.exception.code, "refused")
        self.assertIsNone(stranger.poll())

    def test_a_registered_groups_processes_are_not_unmanaged(self):
        self.model("kilix-chat", "m", GIB)
        self.assertEqual(self.status().unmanaged, ())


if __name__ == "__main__":
    import unittest
    unittest.main()
