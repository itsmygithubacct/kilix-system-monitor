"""Regression tests for fix round 1 (FIXLIST-20261123), one class per property.

The reviewers' own controls and reproductions are in test_review_*.py; these
cover what the fix list asked for beyond them, and each is what catches one or
more planted defects in defects.py (named in the class docstring).
"""
from __future__ import annotations

import errno
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import residency_containment
import kilix_gpu_residency as residency
from kilix_gpu_residency import catalog, cli, ollama as ollama_module, vram
from fakes import (AMD, GIB, INTEL, MIB, NVIDIA, SECOND_NVIDIA, CatalogCase, ResidencyCase, alive, make_gpu,
                   stop, wait_for)

CONTROL = re.compile("[\x00-\x09\x0b-\x1f\x7f-\x9f؜‎‏‪-‮⁦-⁩]")
HOSTILE = "a\x1b[31mb\x07c\x1b]52;c;aGk=\x07d‮e\x9bf"
NO_SYS = {"KILIX_GPU_SYSFS_ROOT": "/nonexistent-sysfs"}  # a child that sees no GPUs never loads NVML


def write_record(namespace: str, record: dict) -> Path:
    path = Path(namespace) / (record["id"] + ".json")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(record, stream)
    return path


def read_record(namespace: str, ident: str) -> dict:
    return json.loads((Path(namespace) / (ident + ".json")).read_text())


class ForeignRecordTests(ResidencyCase):
    """FX-09: foreign-record-refuses-registry, foreign-record-planned."""

    def foreign(self, process: subprocess.Popen, **extra) -> dict:
        self.prune()
        record = {"version": "kilix.gpu-residency/v2", "id": "f" * 32, "app": "voicebox", "model": "tts",
                  "backend": "http", "holder": {"pid": process.pid, "start": residency._start(process.pid)},
                  "pgid": process.pid, "state": "idle", "unload": {"kind": "http", "url": "http://127.0.0.1:1/x"},
                  "future": True, **extra}
        write_record(self.namespace, record)
        return record

    def test_a_newer_record_is_listed_counted_and_never_planned(self):
        process = self.spawn(2 * GIB)
        self.foreign(process)
        value = self.status()
        [row] = value.foreign
        self.assertEqual((row.id, row.version, row.pid, row.vram_bytes, row.reason),
                         ("f" * 32, "kilix.gpu-residency/v2", process.pid, 2 * GIB, "newer-schema"))
        self.assertNotIn(process.pid, [u.pid for u in value.unmanaged])  # its holder's VRAM is its own
        with self.assertRaises(residency.RoomError) as caught:
            residency.ensure_room(5 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual((caught.exception.code, caught.exception.plan), ("no-room", ()))
        self.assertIsNone(process.poll())
        with self.assertRaises(residency.ResidencyError) as refused:
            residency.unload("f" * 32, force=True, **self.where())
        self.assertEqual(refused.exception.code, "refused")
        stop(process)
        self.prune()  # its v1-shaped holder is dead here: pruned
        self.assertEqual(self.status().foreign, ())

    def test_newer_pins_are_honoured_and_never_rewritten(self):
        self.ollama.load("kept:1b", 4 * GIB)
        self.prune()
        pins = Path(self.namespace) / "pins.json"
        descriptor = os.open(pins, os.O_WRONLY | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"version": "kilix.gpu-residency/v2", "ollama": ["kept:1b"], "groups": []}, stream)
        before = pins.read_bytes()
        with self.assertRaises(residency.RoomError):
            residency.ensure_room(3 * GIB, device="Vulkan1", requester="puzzllm", **self.where())
        self.assertEqual(self.ollama.unloads(), [])
        with self.assertRaises(residency.ResidencyError):
            residency.pin("ollama:kept:1b", False, **self.where())
        self.assertEqual(pins.read_bytes(), before)

    def test_a_newer_event_log_reads_as_empty_and_is_never_rewritten(self):
        _, process = self.model("puzzllm", "old", 4 * GIB)
        events = Path(self.namespace) / "events.json"
        descriptor = os.open(events, os.O_WRONLY | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"version": "kilix.gpu-residency/v2", "events": [{"anything": 1}]}, stream)
        before = events.read_bytes()
        self.assertEqual(self.status().events, ())
        room = residency.ensure_room(3 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual([e.model for e in room.evicted], ["old"])
        self.assertEqual(events.read_bytes(), before)


class LockReleasedDuringUnloadsTests(ResidencyCase):
    """FX-07: unload-while-locked, unloading-state-not-written. Four groups that ignore TERM take
    TERM_GRACE_SECONDS each; another process reads and writes the registry meanwhile."""

    CHILD = ("import json, sys, time\n"
             "import kilix_gpu_residency as residency\n"
             "started = time.monotonic()\n"
             "states = [r.state for r in residency.status(namespace=sys.argv[1], ollama=sys.argv[2]).residents]\n"
             "residency.register(app='other', model='m', backend='process', namespace=sys.argv[1],\n"
             "                   ollama=sys.argv[2]).close(unload=False)\n"
             "print(json.dumps({'seconds': time.monotonic() - started, 'states': states}))\n")

    def test_status_and_register_complete_while_units_unload(self):
        processes = [self.model("puzzllm", f"m{n}", GIB, ignore_term=True)[1] for n in range(4)]
        outcome: dict = {}

        def make_room():
            try:
                outcome["room"] = residency.ensure_room(5 * GIB + 512 * MIB, device="Vulkan1", requester="bonsai",
                                                        **self.where())
            except BaseException as error:  # noqa: BLE001 - reported below
                outcome["error"] = error
        thread = threading.Thread(target=make_room)
        started = time.monotonic()
        thread.start()
        self.addCleanup(thread.join)
        self.assertTrue(wait_for(lambda: any(r["state"] == "unloading" for r in self.records()), 5.0))
        done = subprocess.run([sys.executable, "-c", self.CHILD, self.namespace, self.ollama.url], capture_output=True,
                              text=True, timeout=60, env=residency_containment.child_env(**NO_SYS))
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        seen = json.loads(done.stdout)
        thread.join(timeout=60)
        total = time.monotonic() - started
        self.assertNotIn("error", outcome, outcome.get("error"))
        self.assertEqual(len(outcome["room"].evicted), 4)
        # The bound: the other process never waits for an unload, which here take 2 s each.
        self.assertLess(seen["seconds"], residency.TERM_GRACE_SECONDS, seen)
        self.assertGreater(total, 2 * residency.TERM_GRACE_SECONDS)
        self.assertIn("unloading", seen["states"])
        self.assertTrue(all(p.poll() is not None for p in processes))


class ReservationRaceTests(ResidencyCase):
    """FX-17: a real two-process register(need_bytes) race. Each child sees one AMD GPU with
    7 GiB free and asks for 5 GiB; exactly one may have it."""

    CHILD = ("import os, sys, time\n"
             "import kilix_gpu_residency as residency\n"
             "while not os.path.exists(sys.argv[3]):\n"
             "    time.sleep(0.001)\n"
             "try:\n"
             "    residency.register(app='app' + sys.argv[4], model='m', backend='process', device=sys.argv[5],\n"
             "                       need_bytes=5 * 2**30, namespace=sys.argv[1], ollama=sys.argv[2])\n"
             "    print('ok', flush=True)\n"
             "except residency.RoomError as error:\n"
             "    print(error.code, flush=True)\n"
             "sys.stdin.read()  # keep the record alive until the parent has both answers\n")

    def test_two_apps_cannot_both_take_the_same_memory(self):
        machine = self.scratch / "amd-sys"
        make_gpu(machine, *AMD)
        card = machine / "devices" / "pci0000:00" / AMD[0]
        (card / "mem_info_vram_total").write_text(str(8 * GIB))
        (card / "mem_info_vram_used").write_text(str(GIB))
        go = self.scratch / "go"
        children = []
        for n in range(2):
            child = subprocess.Popen([sys.executable, "-c", self.CHILD, self.namespace, self.ollama.url, str(go),
                                      str(n), AMD[0]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     env=residency_containment.child_env(KILIX_GPU_SYSFS_ROOT=str(machine)))
            self.addCleanup(stop, child)
            children.append(child)
        time.sleep(0.5)
        go.write_text("")
        answers = sorted(child.stdout.readline().strip() for child in children)
        for child in children:
            child.stdin.close()
            child.wait(timeout=30)
        self.assertEqual(answers, ["no-room", "ok"])


class ReservationLifeTests(ResidencyCase):
    """FX-03, FX-29: loading-never-expires, release-wait-ignores-reservations."""

    def test_a_reservation_ends_after_the_load_timeout(self):
        first = residency.register(app="kilix-chat", model="big", backend="process", device="Vulkan1",
                                   need_bytes=4 * GIB, **self.where())
        with self.assertRaises(residency.RoomError):
            residency.register(app="puzzllm", model="other", backend="process", device="Vulkan1",
                               need_bytes=4 * GIB, **self.where())
        record = read_record(self.namespace, first.id)
        self.assertAlmostEqual(record["reserved_until"], record["registered_at"] + residency.LOAD_TIMEOUT_SECONDS)
        record["reserved_until"] = time.time() - 1  # as if LOAD_TIMEOUT_SECONDS had passed
        write_record(self.namespace, record)
        [info] = self.status().residents
        self.assertEqual((info.state, info.reserved_bytes), ("loading", 0))
        residency.register(app="puzzllm", model="other", backend="process", device="Vulkan1",
                           need_bytes=4 * GIB, **self.where())

    def test_the_release_wait_counts_reservations_made_meanwhile(self):
        self.model("puzzllm", "old", 4 * GIB)
        original = residency._unload_one

        def unload_then_another_app_registers(claim, url):
            answer = original(claim, url)
            residency.register(app="bonsai", model="x", backend="process", device="Vulkan1", need_bytes=5 * GIB,
                               **self.where())
            return answer
        with mock.patch.object(residency, "_unload_one", unload_then_another_app_registers):
            with self.assertRaises(residency.RoomError) as caught:
                residency.ensure_room(3 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual(caught.exception.code, "not-released")
        self.assertEqual([e.model for e in caught.exception.evicted], ["old"])


class RoomErrorTests(ResidencyCase):
    """FX-08: room-error-evicted-dropped, partial-unload-reported-as-no-room,
    not-released-reported-as-no-room; FX-11: unconfirmed-unload-logged-confirmed."""

    def two_units(self):
        _, process = self.model("puzzllm", "group", 2 * GIB)
        self.ollama.load("theirs:1b", 2 * GIB)
        return process

    def test_a_failure_partway_says_what_was_unloaded(self):
        process = self.two_units()

        def fails(url, name, **_):
            raise ollama_module.OllamaError("Ollama is not reachable: ConnectionRefusedError")
        with mock.patch.object(residency._ollama, "unload", fails):
            with self.assertRaises(residency.RoomError) as caught:
                residency.ensure_room(5 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual(caught.exception.code, "unload-failed")
        self.assertEqual([e.model for e in caught.exception.evicted], ["group"])
        self.assertIsNotNone(process.poll())

    def test_an_unconfirmed_unload_is_logged_unconfirmed(self):
        self.ollama.load("theirs:1b", 4 * GIB)

        def times_out(url, name, **_):
            raise ollama_module.OllamaTimeout("Ollama did not unload the model in time")
        with mock.patch.object(residency._ollama, "unload", times_out):
            with self.assertRaises(residency.RoomError) as caught:
                residency.ensure_room(3 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual(caught.exception.code, "unload-failed")
        [event] = caught.exception.evicted
        self.assertFalse(event.confirmed)
        self.assertFalse(self.status().events[-1].confirmed)
        out, err = io.StringIO(), io.StringIO()
        cli.main(["--namespace", self.namespace, "--ollama", self.ollama.url, "status"], out, err)
        self.assertIn("(asked; Ollama had not unloaded it yet)", out.getvalue())

    def test_vram_that_never_comes_back_is_not_released(self):
        self.ollama.load("sticky:1b", 4 * GIB)
        runner = self.ollama.runner_pid
        self.ollama.held = lambda: {runner: 4 * GIB}  # the runner keeps the memory after the unload
        with self.assertRaises(residency.RoomError) as caught:
            residency.ensure_room(3 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual(caught.exception.code, "not-released")
        self.assertEqual([e.model for e in caught.exception.evicted], ["sticky:1b"])


class SettleTests(ResidencyCase):
    """FX-07 adjacent: dead-claim-stays-unloading, failed-unload-of-dead-holder-rewritten."""

    def test_an_unloading_mark_whose_unloader_died_reads_idle(self):
        resident, process = self.model("puzzllm", "m", 4 * GIB)
        gone = subprocess.Popen(["true"])
        gone.wait()
        record = read_record(self.namespace, resident.id)
        record["state"], record["claim"] = "unloading", {"pid": gone.pid, "start": 1}
        write_record(self.namespace, record)
        [info] = self.status().residents
        self.assertEqual(info.state, "idle")
        room = residency.ensure_room(3 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual([e.id for e in room.evicted], [resident.id])
        self.assertIsNotNone(process.poll())

    def test_a_failed_unload_whose_holder_died_leaves_no_record(self):
        resident, process = self.model("puzzllm", "m", 4 * GIB)
        original = residency._signal_unload

        def holder_dies_first(record):
            stop(process)  # this suite's own child
            return original(record)
        with mock.patch.object(residency, "_signal_unload", holder_dies_first):
            with self.assertRaises(residency.RoomError) as caught:
                residency.ensure_room(3 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual(caught.exception.code, "unload-failed")
        self.assertEqual(self.records(), [])


class AttachTests(ResidencyCase):
    """FX-12 adjacent: attach(pid=) records the start time and keeps the descendant rule
    (holder-start-not-recorded-on-attach-pid, foreign-pid-accepted)."""

    def test_attach_pid_follows_a_child_by_its_start_time(self):
        resident = residency.register(app="kilix-chat", model="m", backend="process", device="Vulkan1",
                                      **self.where())
        child = self.spawn(GIB)
        resident.attach(pid=child.pid)
        record = read_record(self.namespace, resident.id)
        self.assertEqual(record["holder"], {"pid": child.pid, "start": residency._start(child.pid)})
        [info] = self.status().residents
        self.assertEqual((info.pid, info.vram_bytes), (child.pid, GIB))
        stop(child)
        self.prune()
        self.assertEqual(self.records(), [])

    def test_attach_pid_refuses_a_process_this_one_did_not_start(self):
        resident = residency.register(app="kilix-chat", model="m", backend="process", **self.where())
        with self.assertRaises(residency.ResidencyError) as caught:
            resident.attach(pid=os.getppid())
        self.assertEqual(caught.exception.code, "invalid-request")

    def test_pid_1_is_never_an_ancestor(self):
        self.assertFalse(residency._descends(os.getpid(), 1))
        self.assertFalse(residency._descends(os.getpid(), {1}))
        self.assertTrue(residency._descends(os.getpid(), os.getppid()))


class SharedUnitTests(ResidencyCase):
    """FX-01 adjacent: unload-shared-without-force, close-skips-loading-ollama-guard."""

    def test_unloading_a_shared_model_by_hand_names_the_other_holders(self):
        self.ollama.load("shared:7b", 3 * GIB)
        first = residency.register(app="kilix-chat", model="shared:7b", backend="ollama", **self.where())
        first.idle()
        second = residency.register(app="puzzllm", model="shared:7b", backend="ollama", **self.where())
        second.idle()
        with self.assertRaises(residency.ResidencyError) as caught:
            residency.unload(first.id, **self.where())
        self.assertEqual(caught.exception.code, "refused")
        self.assertIn(f"puzzllm ({second.id})", str(caught.exception))
        self.assertEqual(self.ollama.unloads(), [])
        event = residency.unload(first.id, shared=True, **self.where())
        self.assertEqual((event.id, self.ollama.unloads()), (first.id, ["shared:7b"]))

    def test_closing_an_ollama_record_whose_model_never_loaded_sends_no_unload(self):
        resident = residency.register(app="kilix-chat", model="never:1b", backend="ollama", **self.where())
        resident.close()
        self.assertEqual((self.ollama.unloads(), self.records()), ([], []))


class ReadOnlyTests(ResidencyCase):
    """FX-22: reads-prune."""

    def test_a_read_never_prunes(self):
        resident, process = self.model("puzzllm", "m", GIB)
        stop(process)
        self.assertEqual(self.status().residents, ())  # a dead holder is never shown
        self.assertTrue((Path(self.namespace) / (resident.id + ".json")).exists())
        self.prune()
        self.assertFalse((Path(self.namespace) / (resident.id + ".json")).exists())


class DeviceMapTests(ResidencyCase):
    """FX-04: map-to-absent-gpu-accepted."""

    def test_a_map_entry_naming_a_gpu_this_machine_lacks_is_refused(self):
        os.environ["KILIX_GPU_DEVICES"] = "Vulkan1=0000:05:00.0"
        with self.assertRaises(residency.ResidencyError) as caught:
            residency.ensure_room(GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual(caught.exception.code, "invalid-request")


class SeamTests(ResidencyCase):
    """FX-05: proc-root-from-environment, vram-seam-without-guard."""

    CHILD = ("import sys\n"
             "import kilix_gpu_residency as residency\n"
             "resident = residency.register(app='a', model='m', backend='process', namespace=sys.argv[1],\n"
             "                              ollama=sys.argv[2])\n"
             "resident.close(unload=False)\n"
             "print('ok')\n")

    def test_an_inherited_proc_root_is_ignored_by_a_new_process(self):
        resident, process = self.model("kilix-chat", "m", GIB)
        empty = self.scratch / "noproc"
        empty.mkdir()
        done = subprocess.run([sys.executable, "-c", self.CHILD, self.namespace, self.ollama.url], capture_output=True,
                              text=True, timeout=60,
                              env=residency_containment.child_env(KILIX_GPU_PROC_ROOT=str(empty), **NO_SYS))
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "ok"), done.stderr[-2000:])
        self.assertEqual([r["id"] for r in self.records()], [resident.id])
        self.assertIsNone(process.poll())

    def test_the_vram_seams_are_ignored_without_the_test_guard(self):
        os.environ["KILIX_GPU_PROC_ROOT"] = str(self.scratch / "noproc")
        self.assertEqual(vram.sys_root(), str(self.sys))  # the guard is installed here
        with mock.patch.object(sys, "_kilix_residency_guard", False):
            self.assertEqual((vram.sys_root(), vram.proc_root()), ("/sys", "/proc"))


class SanitizerSweepTests(CatalogCase):
    """FX-06: every printed field shows control characters as escapes (unmanaged-command-unescaped,
    bidi-controls-shown). JSON is json.dump's, which escapes every control character itself."""

    def assertClean(self, text: str) -> None:
        found = sorted({f"U+{ord(c):04X}" for c in CONTROL.findall(text)})
        self.assertEqual(found, [], text)
        self.assertIn("\\x1b", text)
        self.assertIn("\\u202e", text)
        self.assertIn("\\x9b", text)

    def test_status_text(self):
        device = vram.Device("0000:01:00.0", HOSTILE, "NVIDIA", "nvidia", True, False, 1, 1, 0, "nvml")
        info = residency.ResidentInfo("a" * 32, HOSTILE, HOSTILE, "process", None, None, 1, 2, 0, None, "idle",
                                      False, time.time(), True, "signal", "pgid:2", 0)
        value = residency.Status((device,), (info,), (residency.Unmanaged(3, HOSTILE, "0000:01:00.0", 1),),
                                 HOSTILE, (residency.Eviction("a" * 32, HOSTILE, HOSTILE, 1, 1.0, HOSTILE, "room"),),
                                 (residency.Foreign("b" * 32, HOSTILE, HOSTILE, HOSTILE, 4, 0, HOSTILE),))
        out = io.StringIO()
        cli.print_status(value, out)
        self.assertClean(out.getvalue())

    def test_models_list_text(self):
        listed = catalog.Listed("x", HOSTILE, "gguf", "local", "local", "llama.cpp", "/x/" + HOSTILE, 1, None,
                                ({"app": HOSTILE}, {"pid": 1, "command": HOSTILE}))
        out = io.StringIO()
        cli.print_models(catalog.Inventory((listed,), {"llama.cpp": "/x"}, "reachable", "known"), out)
        self.assertClean(out.getvalue())

    def test_a_result_message_and_detail(self):
        def refuses(*_, **__):
            raise catalog.CatalogError("download-failed", "could not " + HOSTILE, detail=HOSTILE)
        with mock.patch.object(catalog, "install", refuses):
            code, out, err = self.cli("install", "small")
        self.assertEqual(code, 1)
        self.assertClean(out)
        self.assertClean(err)


class ResultIsTheLastLineTests(CatalogCase):
    """FX-15: detail-after-result. The command's stderr merged into its stdout, as in puzzllm's pty."""

    def run_merged(self, *argv: str, ollama: str | None = None) -> tuple[int, list[str]]:
        done = subprocess.run([sys.executable, "-m", "kilix_gpu_residency.cli", "--namespace", self.namespace,
                               "--ollama", ollama or self.ollama.url, "models", "--catalog", str(self.catalog_path),
                               *argv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
                              env=residency_containment.child_env(**NO_SYS))
        return done.returncode, [line for line in done.stdout.splitlines() if line.strip()]

    def test_every_outcome_with_a_detail_ends_with_its_result(self):
        self.downloads.status = 500
        for mode in ("lines", "none"):
            with self.subTest(progress=mode):
                code, lines = self.run_merged("--progress", mode, "install", "small")
                self.assertEqual((code, lines[-1]), (1, f"could not download {self.SMALL}"), lines)
                self.assertIn("kilix-gpu: HTTP 500", lines)
        code, lines = self.run_merged("install", "ollama:tiny:1b", ollama="http://127.0.0.1:9")
        self.assertEqual((code, lines[-1]), (1, "Ollama is not reachable"), lines)
        self.assertGreater(len(lines), 1, lines)  # the detail was printed, before the result

    def test_a_failed_list_prints_one_line(self):
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(["--namespace", self.namespace, "--ollama", self.ollama.url, "models",
                         "--catalog", str(self.scratch / "no-such-catalog.json"), "list"], out, err)
        self.assertEqual(code, 1)
        self.assertEqual(len(out.getvalue().splitlines()), 1, out.getvalue())


class LastResortTests(ResidencyCase):
    """FX-14: cli-last-resort-missing."""

    def test_anything_unexpected_is_one_line_and_exit_1(self):
        def breaks(**_):
            raise RuntimeError("boom")
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cli, "status", breaks):
            code = cli.main(["--namespace", self.namespace, "status"], out, err)
        self.assertEqual((code, out.getvalue(), err.getvalue()), (1, "kilix-gpu: internal error: RuntimeError: boom\n", ""))

    def test_an_interrupt_is_one_line_and_exit_130_with_the_lock_released(self):
        def interrupted(**_):
            raise KeyboardInterrupt
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cli, "status", interrupted):
            code = cli.main(["--namespace", self.namespace, "status"], out, err)
        self.assertEqual((code, out.getvalue()), (130, "kilix-gpu: interrupted\n"))
        residency.register(app="a", model="m", backend="process", wait=1.0, **self.where()).close(unload=False)


class _Os:
    """catalog's view of os, with some calls failing; everything else is the real os."""

    def __init__(self, **failing: int) -> None:
        self.failing = failing

    def __getattr__(self, name: str):
        if name in self.failing:
            def fails(*_, **__):
                raise OSError(self.failing[name], os.strerror(self.failing[name]))
            return fails
        return getattr(os, name)


class DiskErrorTests(CatalogCase):
    """FX-14: oserror-escapes-catalog. One errno class at each of download, publish and remove;
    each is an outcome with a detail, never a traceback."""

    def failing(self, outcome: str, call, **failing: int) -> catalog.CatalogError:
        with mock.patch.object(catalog, "os", _Os(**failing)):
            with self.assertRaises(catalog.CatalogError) as caught:
                call()
        self.assertEqual(caught.exception.outcome, outcome, caught.exception)
        return caught.exception

    def test_no_space_while_writing(self):
        error = self.failing("download-failed", lambda: self.install("small"), write=errno.ENOSPC)
        self.assertIn(os.strerror(errno.ENOSPC), error.detail)

    def test_a_read_only_folder_at_publish(self):
        error = self.failing("download-failed", lambda: self.install("small"), link=errno.EROFS)
        self.assertIn(os.strerror(errno.EROFS), error.detail)
        self.assertFalse((self.llm / self.SMALL).exists())

    def test_permission_denied_at_remove(self):
        self.place(self.llm, self.SMALL, self.small)
        error = self.failing("remove-failed", lambda: self.remove("small"), unlink=errno.EACCES)
        self.assertIn(os.strerror(errno.EACCES), error.detail)
        self.assertTrue((self.llm / self.SMALL).exists())

    def test_an_io_error_while_verifying(self):
        self.place(self.llm, self.SMALL, self.small)
        error = self.failing("unavailable", lambda: self.verify("small"), read=errno.EIO, pread=errno.EIO)
        self.assertIn(os.strerror(errno.EIO), error.detail)


class FolderOwnerTests(CatalogCase):
    """FX-17: rv-folder-owner-unchecked, with a fake geteuid."""

    def test_a_folder_another_user_owns_is_refused(self):
        with mock.patch.object(catalog, "os", SimpleNamespace(**{**vars(os), "geteuid": lambda: os.geteuid() + 1})):
            with self.assertRaises(catalog.CatalogError) as caught:
                self.install("small")
        self.assertEqual(caught.exception.outcome, "unsafe-folder")


class CatalogUnloadFailureTests(CatalogCase):
    """FX-27: unload-failure-reported-busy, rv-unload-errors-ignored."""

    def test_an_unload_that_fails_stops_the_removal(self):
        path = self.place(self.llm, self.SMALL, self.small)
        _, process = self.model("kilix-chat", str(path), GIB)

        def fails(*_, **__):
            raise residency.ResidencyError("unavailable", "Residency registry is busy")
        with mock.patch.object(catalog, "_unload", fails):
            with self.assertRaises(catalog.CatalogError) as caught:
                self.remove("small", unload=True)
        self.assertEqual((caught.exception.outcome, caught.exception.exit_code), ("remove-failed", 1))
        self.assertIn("unloaded already: nothing", str(caught.exception))
        self.assertTrue(path.exists())
        self.assertIsNone(process.poll())


@unittest.skipUnless(shutil.which("openssl"), "openssl makes the test certificate")
class TlsTests(CatalogCase):
    """FX-17: rv-tls-unverified. A loopback https server with a self-signed certificate is refused;
    the test-only CAFILE hook trusts it for the positive case."""

    def setUp(self) -> None:
        super().setUp()
        self.cert, key = str(self.scratch / "cert.pem"), str(self.scratch / "key.pem")
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", self.cert,
                        "-days", "2", "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1"],
                       check=True, capture_output=True, timeout=60)
        # A path catalog's https URL must be Hugging Face's, so the loopback http entry redirects to
        # a second, https, server (an allowed redirect), as Hugging Face's own redirects to its CDN.
        secure = type(self.downloads)()
        self.addCleanup(secure.close)
        secure.files = self.downloads.files
        secure.tls(self.cert, key)
        self.downloads.redirect = f"{secure.url}/files/{self.SMALL}"
        self.addCleanup(setattr, catalog, "CAFILE", catalog.CAFILE)

    def test_an_untrusted_certificate_is_refused(self):
        with self.assertRaises(catalog.CatalogError) as caught:
            self.install("small")
        self.assertEqual(caught.exception.outcome, "download-failed")
        self.assertIn("certificate", caught.exception.detail)
        self.assertFalse((self.llm / self.SMALL).exists())

    def test_a_trusted_certificate_is_used(self):
        catalog.CAFILE = self.cert
        self.assertEqual(self.install("small").outcome, "saved")


class PinRuleTests(ResidencyCase):
    """FX-02 adjacent: what `pin NAME` pins, and which pin wins."""

    def test_pinning_a_process_records_model_name_pins_that_record_only(self):
        resident, process = self.model("kilix-chat", "llada", 4 * GIB)
        info = residency.pin("llada", **self.where())
        self.assertTrue(info.pinned)
        self.assertTrue(read_record(self.namespace, resident.id)["pinned"])
        self.assertFalse((Path(self.namespace) / "pins.json").exists())  # never a model pin
        with self.assertRaises(residency.RoomError):
            residency.ensure_room(3 * GIB, device="Vulkan1", requester="puzzllm", **self.where())
        self.assertIsNone(process.poll())

    def test_a_records_own_pin_and_a_model_pin_both_hold(self):
        self.ollama.load("kept:1b", 4 * GIB)
        resident = residency.register(app="kilix-chat", model="kept:1b", backend="ollama", pinned=True,
                                      **self.where())
        resident.idle()
        residency.pin("ollama:kept:1b", **self.where())
        self.assertTrue(residency.unpin("ollama:kept:1b", **self.where()).pinned)  # the record's own pin holds
        self.assertTrue(residency.pin(resident.id, False, **self.where()).pinned is False)
        residency.pin("ollama:kept:1b", **self.where())
        self.assertTrue(residency.unpin(resident.id, **self.where()).pinned)  # the model pin holds
        with self.assertRaises(residency.RoomError):
            residency.ensure_room(3 * GIB, device="Vulkan1", requester="puzzllm", **self.where())
        self.assertEqual(self.ollama.unloads(), [])


class RecentUseTests(ResidencyCase):
    """FX-11: recent-unregistered-planned, ollama-last-use-fixed-300."""

    def test_an_unregistered_model_used_moments_ago_is_not_planned(self):
        self.ollama.load("theirs:1b", 4 * GIB, expires_in=299.0)  # last used about a second ago
        with self.assertRaises(residency.RoomError) as caught:
            residency.ensure_room(3 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual((caught.exception.code, caught.exception.plan), ("no-room", ()))
        self.assertEqual(self.ollama.unloads(), [])
        [info] = self.status().residents
        self.assertLess(time.time() - info.last_used, residency.RECENT_USE_SECONDS)

    def test_the_estimate_follows_ollama_keep_alive(self):
        os.environ["OLLAMA_KEEP_ALIVE"] = "1h"
        self.ollama.load("theirs:1b", 4 * GIB, expires_in=3600.0 - 600.0)  # used ten minutes ago
        [info] = self.status().residents
        self.assertAlmostEqual(time.time() - info.last_used, 600.0, delta=5.0)
        room = residency.ensure_room(3 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual([e.model for e in room.evicted], ["theirs:1b"])


class LoopbackNameTests(unittest.TestCase):
    """FX-21 adjacent: only localhost and literal loopback addresses are loopback."""

    def test_a_name_that_resolves_to_loopback_is_still_refused(self):
        for url in ("http://ip6-localhost:11434", "http://localhost.localdomain:11434"):
            with self.subTest(url=url):
                self.assertIsNone(ollama_module.loopback(url))
        self.assertIsNotNone(ollama_module.loopback("http://localhost:11434"))
        self.assertIsNotNone(ollama_module.loopback("http://[::1]:11434"))


class SharedGroupTests(ResidencyCase):
    """pgid-double-counted: records sharing a process group are one unit, its memory counted once."""

    def test_a_shared_group_is_planned_whole_and_counted_once(self):
        process = self.spawn(3 * GIB)
        first = residency.register(app="kilix-chat", model="a", backend="process", device="Vulkan1",
                                   pgid=process.pid, **self.where())
        second = residency.register(app="puzzllm", model="b", backend="process", device="Vulkan1",
                                    pgid=process.pid, **self.where())
        first.idle()
        second.idle()
        with self.assertRaises(residency.RoomError) as caught:
            residency.register(app="bonsai", model="new", backend="process", device="Vulkan1",
                               need_bytes=5 * GIB, policy="never", **self.where())
        self.assertEqual(sorted(r.id for r in caught.exception.plan), sorted([first.id, second.id]))
        self.assertIsNone(process.poll())


class TwoGpuTests(ResidencyCase):
    """reservation-any-device, ollama-single-device."""

    gpus = (INTEL, NVIDIA, SECOND_NVIDIA)

    def setUp(self):
        super().setUp()
        os.environ["KILIX_GPU_DEVICES"] = f"Vulkan1={NVIDIA[0]},Vulkan2={SECOND_NVIDIA[0]}"
        self.on_second: dict[int, int] = {}
        self.nvml_devices[NVIDIA[0]]["compute"] = self.held
        self.nvml_devices[SECOND_NVIDIA[0]]["compute"] = lambda: dict(self.on_second)

    def test_a_reservation_holds_room_on_its_own_device_only(self):
        residency.register(app="puzzllm", model="big", backend="process", device="Vulkan2",
                           need_bytes=5 * GIB, **self.where())
        room = residency.ensure_room(5 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual(room.device_pci, NVIDIA[0])
        with self.assertRaises(residency.RoomError) as caught:
            residency.ensure_room(2 * GIB, device="Vulkan2", requester="kilix-chat", **self.where())
        self.assertEqual(caught.exception.code, "no-room")

    def test_ollama_on_two_gpus_is_attributed_to_neither(self):
        self.ollama.load("theirs:1b", 2 * GIB)
        self.on_second[self.ollama.runner_pid] = GIB  # Ollama's runner holds memory on both GPUs
        [info] = self.status(wake=True).residents
        self.assertIsNone(info.device_pci)
        with self.assertRaises(residency.RoomError) as caught:
            residency.ensure_room(5 * GIB, device="Vulkan1", requester="kilix-chat", **self.where())
        self.assertEqual((caught.exception.code, caught.exception.plan), ("no-room", ()))
        self.assertEqual(self.ollama.unloads(), [])


class UntrustedProcTests(ResidencyCase):
    """prune-without-self-check, prune-across-pid-namespaces, signal-leader-mismatch-allowed."""

    def test_a_proc_that_does_not_show_this_process_prunes_nothing(self):
        resident, process = self.model("kilix-chat", "m", GIB)
        fake = self.scratch / "proc"
        fake.mkdir()
        os.symlink("/proc/self", fake / "self")  # this process's namespace, but no process listed
        with mock.patch.object(residency, "_PROC", str(fake)):
            # A reader there still lists the record: nothing in that /proc says its holder is gone.
            self.assertEqual([r.id for r in self.status().residents], [resident.id])
            # A writer refuses outright rather than prune by it.
            with self.assertRaises(residency.ResidencyError) as caught:
                self.prune()
            self.assertEqual(caught.exception.code, "unavailable")
        self.assertTrue((Path(self.namespace) / (resident.id + ".json")).exists())
        self.assertIsNone(process.poll())

    def test_a_record_from_another_pid_namespace_is_never_pruned(self):
        resident = residency.register(app="kilix-chat", model="x:1b", backend="ollama", **self.where())
        record = read_record(self.namespace, resident.id)
        record["holder"] = {"pid": 4194290, "start": 1}  # gone, as far as this namespace can tell
        record["pidns"] = residency._own_pidns() + 1
        write_record(self.namespace, record)
        self.prune()
        self.assertTrue((Path(self.namespace) / (resident.id + ".json")).exists())
        status = self.status()
        self.assertEqual(status.residents, ())
        self.assertEqual([(f.id, f.reason) for f in status.foreign], [(resident.id, "other-pid-namespace")])

    def test_a_group_the_record_does_not_lead_is_never_signalled(self):
        leader, other = self.spawn(), self.spawn()
        record = {"unload": {"kind": "signal", "pgid": other.pid},
                  "holder": {"pid": leader.pid, "start": residency._start(leader.pid)}}
        with self.assertRaises(residency.ResidencyError) as caught:
            residency._signal_unload(record)
        self.assertEqual(caught.exception.code, "refused")
        time.sleep(0.2)
        self.assertIsNone(other.poll())
        self.assertIsNone(leader.poll())


class RemoveRaceTests(CatalogCase):
    """open-file-removed, rv-release-wait-dropped, rv-remove-identity-recheck-dropped,
    rv-pinned-checked-on-first-holder-only, rv-busy-checked-on-first-holder-only."""

    def setUp(self):
        super().setUp()
        self.path = self.place(self.llm, self.SMALL, self.small)

    def held_by_a_registered_holder(self):
        process = self.holding(self.path)  # the holder's own process has the file open
        resident = residency.register(app="kilix-chat", model=str(self.path), backend="process", device="Vulkan1",
                                      pgid=process.pid, **self.where())
        resident.idle()
        return resident, process

    def test_a_file_still_open_after_the_release_wait_is_not_removed(self):
        _, process = self.held_by_a_registered_holder()
        catalog.RELEASE_WAIT_SECONDS = 0.5
        with mock.patch.object(catalog, "_unload", lambda *a, **k: None):  # "unloaded", but still open
            error = self.refused("open", self.remove, "small", unload=True)
        self.assertIn("close it first", str(error))
        self.assertTrue(self.path.exists())
        self.assertIsNone(process.poll())

    def test_removal_waits_for_an_unloaded_holder_to_close_the_file(self):
        _, process = self.held_by_a_registered_holder()

        def unload_later(*_args, **_kwargs):  # the holder closes the file a moment after unload returns
            threading.Timer(0.5, stop, (process,)).start()

        with mock.patch.object(catalog, "_unload", unload_later):
            result = self.remove("small", unload=True)
        self.assertEqual(result.outcome, "removed")
        self.assertFalse(self.path.exists())

    def test_a_file_replaced_meanwhile_is_left_alone(self):
        real, calls = catalog.open_by, []

        def replacing(targets):
            calls.append(1)
            if len(calls) == 2:  # after the checks, before the unlink
                fresh = self.llm / "fresh"
                fresh.write_bytes(b"someone else's file")
                os.replace(fresh, self.path)
            return real(targets)

        with mock.patch.object(catalog, "open_by", replacing):
            try:
                self.remove("small")
            except catalog.CatalogError:
                pass
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(self.path.read_bytes(), b"someone else's file")

    def two_holders(self):
        holders = [self.model(app, str(self.path), GIB) for app in ("kilix-chat", "puzzllm")]
        return [resident for resident, _ in holders], [process for _, process in holders]

    def test_every_holder_is_checked_for_a_pin(self):
        residents, processes = self.two_holders()
        for resident in residents:  # one of the two orders puts the pinned holder second
            with self.subTest(pinned=resident.app):
                residency.pin(resident.id, **self.where())
                self.refused("pinned", self.remove, "small", unload=True)
                residency.unpin(resident.id, **self.where())
                self.assertTrue(self.path.exists())
                self.assertTrue(all(alive(p.pid) for p in processes))

    def test_every_holder_is_checked_for_idleness(self):
        residents, processes = self.two_holders()
        for resident in residents:
            with self.subTest(busy=resident.app):
                resident.busy()
                self.refused("busy", self.remove, "small", unload=True)
                resident.idle()
                self.assertTrue(self.path.exists())
                self.assertTrue(all(alive(p.pid) for p in processes))


class DownloadCheckTests(CatalogCase):
    """rv-content-range-total-unchecked, rv-overrun-unchecked, rv-hardlinked-part-accepted,
    appeared-file-not-hashed."""

    def test_a_206_claiming_another_total_is_refused(self):
        part = self.place(self.llm, self.part_name(self.SMALL), self.small[:MIB])
        self.downloads.range_total = len(self.small) + 1  # the right bytes, the wrong file
        error = self.refused("download-failed", self.install, "small")
        self.assertIn("not the catalog's size", error.detail)
        self.assertFalse((self.llm / self.SMALL).exists())
        self.assertTrue(part.exists())

    def test_a_source_longer_than_the_catalog_says_is_cut_off(self):
        (self.scratch / "mirror" / self.TINY).write_bytes(self.tiny + b"more")
        error = self.refused("download-failed", self.install, "tiny-diffusion")
        self.assertIn("more than the catalog's size", error.detail)
        self.assertFalse((self.diffusion / self.TINY).exists())

    def test_a_hard_linked_part_is_never_written(self):
        part = self.place(self.llm, self.part_name(self.SMALL), self.small[:MIB])
        other = self.scratch / "elsewhere"
        os.link(part, other)
        self.refused("conflict", self.install, "small")
        self.assertEqual(other.read_bytes(), self.small[:MIB])
        self.assertFalse((self.llm / self.SMALL).exists())

    def test_a_file_that_appears_meanwhile_is_hashed_before_it_counts(self):
        real = catalog._download

        def then_appear(entry, fd, progress):
            real(entry, fd, progress)
            (self.llm / self.SMALL).write_bytes(bytes(len(self.small)))  # the right size, other bytes

        with mock.patch.object(catalog, "_download", then_appear):
            self.refused("conflict", self.install, "small")
        self.assertEqual((self.llm / self.SMALL).read_bytes(), bytes(len(self.small)))


class OllamaRemoveTests(CatalogCase):
    """models-force-not-passed, rv-ollama-delete-unvalidated."""

    def setUp(self):
        super().setUp()
        self.ollama.disk["qwen3.5:2b"] = 2 * 10**9
        self.ollama.load("qwen3.5:2b", 2 * GIB)

    def test_the_command_passes_force(self):
        code, out, _ = self.cli("remove", "--unload", "ollama:qwen3.5:2b")
        self.assertEqual(code, 3)
        self.assertIn("--force", out)
        code, out, _ = self.cli("remove", "--unload", "--force", "ollama:qwen3.5:2b")
        self.assertEqual((code, out.strip()), (0, "removed qwen3.5:2b"))
        self.assertNotIn("qwen3.5:2b", self.ollama.disk)

    def test_delete_refuses_a_name_that_is_not_ollamas(self):
        for name in ("../x", "a b", ""):
            with self.subTest(name=name), self.assertRaises(ollama_module.OllamaError):
                ollama_module.delete(self.ollama.url, name)
        self.assertEqual([r for r in self.ollama.requests if r[0] == "DELETE"], [])


class ProxyTests(CatalogCase):
    """FX-30: downloads connect directly, and a failure says a proxy is set (proxy-hint-dropped)."""

    def test_downloads_connect_directly_and_a_failure_names_the_proxy(self):
        proxies = {"https_proxy": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "http_proxy":
                   "http://127.0.0.1:9", "no_proxy": "", "NO_PROXY": ""}
        with mock.patch.dict(os.environ, proxies):
            document = self.document()
            document["models"][0]["url"] = "http://127.0.0.1:9/resolve/" + self.SMALL  # nothing listens there
            self.write_catalog(document)
            error = self.refused("download-failed", self.install, "small")
            self.assertIn("https_proxy is set, but kilix-gpu does not use proxies", error.detail)
            self.write_catalog()
            self.assertEqual(self.install("small").outcome, "saved")
            self.assertEqual(self.downloads.ranges(self.SMALL), [None])  # the server itself was asked


class CliDeviceTests(ResidencyCase):
    """FX-04 adjacent: the command's --device takes the same path as the API."""

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(["--namespace", self.namespace, "--ollama", self.ollama.url, *argv], out, err)
        return code, out.getvalue(), err.getvalue()

    def test_room_with_an_unknown_or_unmeasured_device_exits_1(self):
        for label, code_name in (("NoSuchGPU7", "invalid-request"), ("Vulkan0", "unmeasured"),
                                 ("0000:09:00.0", "invalid-request")):
            with self.subTest(label=label):
                code, out, err = self.run_cli("room", "1G", "--device", label)
                self.assertEqual(code, 1)
                self.assertIn(f"[{code_name}]", err)
        self.assertFalse(os.path.exists(self.namespace))  # a dry run writes nothing


if __name__ == "__main__":
    unittest.main()
