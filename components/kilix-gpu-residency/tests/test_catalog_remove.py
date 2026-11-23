"""Removing, verifying and listing models, aware of residency, and the CLI.

A model is held by a registry record (a real process group this suite started),
by the fake Ollama's /api/ps, or by a process this suite started that has the
file open or mapped. Nothing outside the suite is ever unloaded or signalled.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import residency_containment
import kilix_gpu_residency as residency
from kilix_gpu_residency import catalog
from fakes import GIB, MIB, CatalogCase, alive, stop

# Maps the file through libc and closes the descriptor, so only /proc/<pid>/maps shows it
# (Python's mmap module keeps a dup of the descriptor, which the fd scan would find instead).
MAPS = ("import ctypes, os, sys, time\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "libc.mmap.restype = ctypes.c_void_p\n"
        "libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)\n"
        "fd = os.open(sys.argv[1], os.O_RDONLY)\n"
        "address = libc.mmap(None, os.fstat(fd).st_size, 1, 1, fd, 0)\n"  # PROT_READ, MAP_SHARED
        "os.close(fd)\n"
        "assert address not in (None, ctypes.c_void_p(-1).value)\n"
        "assert not [n for n in os.listdir('/proc/self/fd') if os.path.realpath('/proc/self/fd/' + n) == "
        "os.path.realpath(sys.argv[1])]\n"
        "print('ready', flush=True)\n"
        "time.sleep(600)\n")


class RemoveTests(CatalogCase):
    def setUp(self):
        super().setUp()
        self.path = self.place(self.llm, self.SMALL, self.small)

    def register(self, model: str, **kwargs):
        resident, process = self.model("kilix-chat", model, GIB, **kwargs)
        return resident, process

    def test_a_model_no_one_holds_is_removed_and_nothing_else_is(self):
        others = {".partial.Ab12Cd": b"puzzllm-fetch's", f"{self.SMALL}.download.json": b"{}",
                  "Other.gguf": b"other"}
        for name, data in others.items():
            self.place(self.llm, name, data)
        result = self.remove("small")
        self.assertEqual((result.outcome, result.message, result.exit_code), ("removed", f"removed {self.SMALL}", 0))
        self.assertFalse(self.path.exists())
        self.assertEqual(sorted(os.listdir(self.llm)), sorted([".fetch.lock", *others]))

    def test_a_model_that_is_not_there_is_not_installed(self):
        self.path.unlink()
        error = self.refused("not-installed", self.remove, "small")
        self.assertEqual((str(error), error.exit_code), (f"{self.SMALL} is not installed", 2))

    def test_a_gguf_is_named_by_id_file_name_or_path(self):
        for target in ("small", self.SMALL, str(self.path), str(self.llm / ".." / "models" / self.SMALL)):
            with self.subTest(target=target):
                self.place(self.llm, self.SMALL, self.small)
                self.assertEqual(self.remove(target).message, f"removed {self.SMALL}")
        local = self.place(self.diffusion, "Local.Q8.gguf", b"local")
        self.assertEqual(self.remove("Local.Q8.gguf").message, "removed Local.Q8.gguf")
        self.assertFalse(local.exists())

    def test_only_plain_gguf_files_directly_in_a_models_folder_are_removed(self):
        outside = self.place(self.scratch / "elsewhere", self.SMALL, b"keep")
        nested = self.place(self.llm / "artifacts", "nested.gguf", b"keep")
        lock = self.llm / ".fetch.lock"
        lock.write_bytes(b"")
        for target in (str(outside), str(nested), str(self.llm / ".partial.Ab12Cd"), str(lock),
                       str(self.llm / f"{self.SMALL}.download.json"), str(self.llm)):
            with self.subTest(target=target):
                error = self.refused("outside-folders", self.remove, target)
                self.assertEqual(error.exit_code, 2)
        for target in ("../x.gguf", "rm -rf", "Qwen3.5"):
            with self.subTest(target=target):
                self.assertIn(self.refused_any(target), ("unknown", "outside-folders", "not-installed"))
        self.assertEqual((outside.read_bytes(), nested.read_bytes()), (b"keep", b"keep"))
        self.assertTrue(self.path.exists())

    def refused_any(self, target):
        with self.assertRaises(catalog.CatalogError) as caught:
            self.remove(target)
        return caught.exception.outcome

    def test_a_symlink_is_never_removed_or_followed(self):
        outside = self.place(self.scratch / "elsewhere", "real.gguf", b"keep")
        os.symlink(outside, self.llm / "Linked.gguf")
        error = self.refused("symlink", self.remove, "Linked.gguf")
        self.assertEqual(str(error), "refusing to remove a symlink: Linked.gguf")
        self.assertTrue(os.path.islink(self.llm / "Linked.gguf"))
        self.assertEqual(outside.read_bytes(), b"keep")

    def test_a_name_in_both_folders_must_be_given_by_path(self):
        self.place(self.diffusion, self.SMALL, b"same name")
        self.refused("ambiguous", self.remove, self.SMALL)
        self.assertEqual(self.remove(str(self.diffusion / self.SMALL)).outcome, "removed")
        self.assertTrue(self.path.exists())

    def test_a_registered_model_is_refused_until_unloaded(self):
        for model in ("small", self.SMALL, "Small-Q4", str(self.path)):
            with self.subTest(registered_as=model):
                self.place(self.llm, self.SMALL, self.small)
                resident, process = self.register(model)
                error = self.refused("loaded", self.remove, "small")
                self.assertEqual((str(error), error.exit_code),
                                 (f"{self.SMALL} is loaded by kilix-chat; use --unload", 3))
                self.assertTrue(self.path.exists())
                self.assertTrue(alive(process.pid))
                result = self.remove("small", unload=True)
                self.assertEqual(result.message, f"removed {self.SMALL}")
                self.assertEqual(self.reap(process), -15)
                self.assertEqual(self.records(), [])
                self.assertFalse(self.path.exists())
                events = self.status().events
                self.assertEqual((events[-1].model, events[-1].by, events[-1].reason),
                                 (model, "kilix-gpu-models", "manual"))

    def test_a_pinned_model_is_refused_even_with_unload(self):
        resident, process = self.register("small", pinned=True)
        for unload in (False, True):
            with self.subTest(unload=unload):
                error = self.refused("pinned", self.remove, "small", unload=unload)
                self.assertEqual((str(error), error.exit_code), (f"{self.SMALL} is pinned by kilix-chat; unpin it first", 3))
        self.assertTrue(alive(process.pid))
        self.assertTrue(self.path.exists())

    def test_a_busy_or_loading_model_is_not_unloaded(self):
        for state in ("busy", "loading"):
            with self.subTest(state=state):
                resident, process = self.register("small", state=state)
                error = self.refused("busy", self.remove, "small", unload=True)
                self.assertEqual(str(error), f"{self.SMALL} is {state} in kilix-chat; try again when it is idle")
                self.assertTrue(alive(process.pid))
                self.assertTrue(self.path.exists())
                resident.close(unload=False)

    def test_a_file_an_unmanaged_process_holds_open_or_mapped_is_never_removed(self):
        mapper = subprocess.Popen([sys.executable, "-c", MAPS, str(self.path)], stdout=subprocess.PIPE, text=True,
                                  env=residency_containment.child_env(), process_group=0)
        self.addCleanup(stop, mapper)
        self.assertEqual(mapper.stdout.readline().strip(), "ready")
        for holder in (mapper, self.holding(self.path)):
            for unload in (False, True):
                with self.subTest(holder=holder.args[0], unload=unload):
                    error = self.refused("open", self.remove, "small", unload=unload)
                    self.assertIn(f"pid {holder.pid} (", str(error))
                    self.assertTrue(str(error).endswith("; close it first"))
                    self.assertTrue(alive(holder.pid))
            stop(holder)
        self.assertEqual(self.remove("small").outcome, "removed")

    def test_a_partial_download_is_removed_only_when_no_one_is_writing_it(self):
        self.path.unlink()
        part = self.place(self.llm, f".{self.SMALL}.part", self.small[:MIB])
        writer = self.holding(part)
        self.refused("open", self.remove, "small")
        self.assertTrue(part.exists())
        stop(writer)
        self.assertEqual(self.remove("small").message, f"removed the partial download of {self.SMALL}")
        self.assertFalse(part.exists())

    def test_a_model_and_its_idle_part_go_together_but_a_written_part_stays(self):
        part = self.place(self.llm, f".{self.SMALL}.part", b"stale")
        self.assertEqual(self.remove("small").outcome, "removed")
        self.assertFalse(part.exists())
        self.place(self.llm, self.SMALL, self.small)
        part = self.place(self.llm, f".{self.SMALL}.part", b"being written")
        self.holding(part)
        self.assertEqual(self.remove("small").outcome, "removed")
        self.assertEqual(part.read_bytes(), b"being written")

    def test_a_folder_a_download_holds_is_busy(self):
        import fcntl
        lock = os.open(self.llm / ".fetch.lock", os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, lock)
        fcntl.flock(lock, fcntl.LOCK_EX)
        self.refused("busy-folder", self.remove, "small")
        self.assertTrue(self.path.exists())

    def test_removal_fails_closed_when_the_registry_cannot_be_read(self):
        with self.assertRaises(catalog.CatalogError) as caught:
            catalog.remove("small", catalog=self.catalog, namespace="relative/registry", ollama=self.ollama.url)
        self.assertEqual(caught.exception.outcome, "unavailable")
        self.assertTrue(str(caught.exception).startswith(f"cannot check whether {self.SMALL} is loaded"))
        self.assertTrue(self.path.exists())


class OllamaRemoveTests(CatalogCase):
    def setUp(self):
        super().setUp()
        self.ollama.disk.update({"qwen3.5:2b": 2 * 10**9, "llama3:latest": 4 * 10**9})

    def deletes(self):
        return [body["model"] for method, path, body in self.ollama.requests if method == "DELETE"]

    def test_an_installed_model_is_deleted_through_the_api(self):
        result = self.remove("ollama:qwen3.5:2b")
        self.assertEqual((result.outcome, result.message), ("removed", "removed qwen3.5:2b"))
        self.assertEqual(self.deletes(), ["qwen3.5:2b"])
        self.assertNotIn("qwen3.5:2b", self.ollama.disk)

    def test_an_implicit_latest_tag_matches(self):
        self.assertEqual(self.remove("ollama:llama3").outcome, "removed")
        self.assertEqual(self.deletes(), ["llama3:latest"])

    def test_a_model_ollama_does_not_have_is_not_installed(self):
        error = self.refused("not-installed", self.remove, "ollama:nothing:1b")
        self.assertEqual(str(error), "nothing:1b is not installed")
        self.assertEqual(self.deletes(), [])

    def test_a_loaded_model_is_refused_until_unloaded(self):
        self.ollama.load("qwen3.5:2b", 2 * GIB)
        error = self.refused("loaded", self.remove, "ollama:qwen3.5:2b")
        self.assertEqual((str(error), error.exit_code), ("qwen3.5:2b is loaded by ollama; use --unload", 3))
        self.assertEqual((self.ollama.unloads(), self.deletes()), ([], []))
        self.assertEqual(self.remove("ollama:qwen3.5:2b", unload=True).outcome, "removed")
        self.assertEqual((self.ollama.unloads(), self.deletes()), (["qwen3.5:2b"], ["qwen3.5:2b"]))
        unload_at = next(i for i, r in enumerate(self.ollama.requests) if r[1] == "/api/generate")
        delete_at = next(i for i, r in enumerate(self.ollama.requests) if r[0] == "DELETE")
        self.assertLess(unload_at, delete_at)

    def test_a_pinned_ollama_model_is_refused_even_with_unload(self):
        self.ollama.load("qwen3.5:2b", 2 * GIB)
        residency.pin("ollama:qwen3.5:2b", **self.where())
        self.refused("pinned", self.remove, "ollama:qwen3.5:2b", unload=True)
        self.assertEqual((self.ollama.unloads(), self.deletes()), ([], []))

    def test_invalid_names_and_an_unreachable_ollama_are_refused(self):
        self.refused("invalid-name", self.remove, "ollama:../x")
        with self.assertRaises(catalog.CatalogError) as caught:
            catalog.remove("ollama:qwen3.5:2b", catalog=self.catalog, namespace=self.namespace,
                           ollama="http://127.0.0.1:9")
        self.assertEqual(caught.exception.outcome, "unavailable")
        self.assertEqual(self.deletes(), [])


class VerifyTests(CatalogCase):
    def test_an_installed_model_is_verified(self):
        self.place(self.llm, self.SMALL, self.small)
        result = self.verify("small")
        self.assertEqual((result.outcome, result.message, result.exit_code), ("verified", f"verified {self.SMALL}", 0))

    def test_a_changed_file_does_not_match(self):
        for data, why in ((self.small[:-1], "size"), (self.small[:-1] + bytes([self.small[-1] ^ 1]), "sha256")):
            with self.subTest(why=why):
                self.place(self.llm, self.SMALL, data)
                error = self.refused("mismatch", self.verify, "small")
                self.assertEqual((str(error), error.exit_code), (f"{self.SMALL} does not match the catalog ({why})", 4))

    def test_what_cannot_be_verified_says_why(self):
        self.refused("not-installed", self.verify, "small")
        self.place(self.llm, "Local.gguf", b"x")
        self.refused("not-verifiable", self.verify, "Local.gguf")
        self.refused("not-verifiable", self.verify, "ollama:qwen3.5:2b")
        self.refused("not-verifiable", self.verify, "tiny-ollama")
        self.place(self.scratch / "elsewhere", "real.gguf", self.small)
        os.symlink(self.scratch / "elsewhere" / "real.gguf", self.llm / self.SMALL)
        self.refused("symlink", self.verify, "small")


class ListTests(CatalogCase):
    def by_id(self, inventory):
        return {model.id: model for model in inventory.models}

    def test_the_inventory_shows_catalog_local_and_ollama_models_and_who_holds_them(self):
        self.place(self.llm, self.SMALL, self.small)
        self.place(self.diffusion, f".{self.TINY}.part", self.tiny[:MIB])
        self.place(self.llm, "Local-Q6.gguf", b"local model")
        for ignored in (".hidden.gguf", ".partial.Ab12Cd", "notes.txt", f"{self.SMALL}.download.json"):
            self.place(self.llm, ignored, b"x")
        (self.llm / "artifacts").mkdir()
        os.symlink(self.llm / "Local-Q6.gguf", self.llm / "Linked.gguf")
        self.model("kilix-chat", "small", GIB)
        holder = self.holding(self.llm / "Local-Q6.gguf")
        self.ollama.disk.update({"qwen3.5:2b": 2 * 10**9, "tiny:1b": 10**9})
        self.ollama.load("qwen3.5:2b", 2 * GIB)
        value = self.inventory()
        self.assertEqual((value.ollama, value.residency), ("reachable", "known"))
        self.assertEqual(dict(value.folders), {"llama.cpp": str(self.llm), "diffusion": str(self.diffusion)})
        rows = self.by_id(value)
        self.assertEqual(set(rows), {"small", "tiny-diffusion", "tiny-ollama", "Local-Q6.gguf", "Linked.gguf",
                                     "ollama:qwen3.5:2b"})
        summary = {key: (m.kind, m.source, m.state, m.folder, m.bytes, m.partial_bytes, m.resident)
                   for key, m in rows.items()}
        self.assertEqual(summary, {
            "small": ("gguf", "catalog", "installed", "llama.cpp", len(self.small), None, ("kilix-chat",)),
            "tiny-diffusion": ("diffusion-gguf", "catalog", "partial", "diffusion", len(self.tiny), MIB, ()),
            "Local-Q6.gguf": ("gguf", "local", "local", "llama.cpp", 11, None, (f"pid {holder.pid} (sleep)",)),
            "Linked.gguf": ("gguf", "local", "link", "llama.cpp",
                            len(str(self.llm / "Local-Q6.gguf")), None, ()),
            "tiny-ollama": ("ollama", "catalog", "installed", None, 10**9, None, ()),
            "ollama:qwen3.5:2b": ("ollama", "ollama", "installed", None, 2 * 10**9, None, ("ollama",)),
        })
        self.assertEqual(rows["small"].path, str(self.llm / self.SMALL))

    def test_available_and_mismatching_files_are_told_apart_by_size(self):
        rows = self.by_id(self.inventory())
        self.assertEqual((rows["small"].state, rows["tiny-diffusion"].state, rows["tiny-ollama"].state),
                         ("available", "available", "available"))
        self.place(self.llm, self.SMALL, b"short")
        self.assertEqual(self.by_id(self.inventory())["small"].state, "mismatch")

    def test_the_inventory_survives_an_unreachable_ollama_and_registry(self):
        self.place(self.llm, self.SMALL, self.small)
        value = catalog.list_models(catalog=self.catalog, namespace="relative/registry", ollama="http://127.0.0.1:9")
        self.assertEqual((value.ollama, value.residency), ("unreachable", "unavailable"))
        self.assertEqual(self.by_id(value)["small"].state, "installed")
        remote = catalog.list_models(catalog=self.catalog, namespace=self.namespace, ollama="http://192.0.2.1:11434")
        self.assertEqual(remote.ollama, "not-loopback")


class CliTests(CatalogCase):
    def test_every_action_prints_one_line_and_its_exit_code(self):
        self.assertEqual(self.cli("install", "small"), (0, f"saved {self.SMALL}\n", ""))
        self.assertEqual(self.cli("install", "small"), (0, f"already saved {self.SMALL}\n", ""))
        self.assertEqual(self.cli("install", "--", "small"), (0, f"already saved {self.SMALL}\n", ""))
        self.assertEqual(self.cli("verify", "small"), (0, f"verified {self.SMALL}\n", ""))
        self.model("kilix-chat", "small", GIB)
        self.assertEqual(self.cli("remove", "small"), (3, f"{self.SMALL} is loaded by kilix-chat; use --unload\n", ""))
        self.assertEqual(self.cli("remove", "--unload", "--", "small"), (0, f"removed {self.SMALL}\n", ""))
        self.assertEqual(self.cli("install", "nope"), (2, "unknown model nope\n", ""))
        self.place(self.llm, self.SMALL, b"short")
        self.assertEqual(self.cli("verify", "small"), (4, f"{self.SMALL} does not match the catalog (size)\n", ""))
        os.unlink(self.llm / self.SMALL)
        self.downloads.status = 404
        code, out, err = self.cli("install", "small")
        self.assertEqual((code, out), (1, f"could not download {self.SMALL}\n"))
        self.assertEqual(err, "kilix-gpu: HTTP 404\n")

    def test_progress_lines_end_with_the_result_line_for_a_menu(self):
        code, out, err = self.cli("--progress", "lines", "install", "small")
        self.assertEqual((code, out), (0, f"saved {self.SMALL}\n"))
        lines = err.splitlines()
        self.assertTrue(lines)
        self.assertTrue(all(line.startswith("downloading small") for line in lines), lines)
        self.assertEqual(lines[-1], "downloading small: 100% of 3 MiB")
        self.assertNotIn("\r", err)
        self.assertEqual(self.cli("--progress", "none", "verify", "small"), (0, f"verified {self.SMALL}\n", ""))
        self.assertEqual(self.cli("verify", "small"), (0, f"verified {self.SMALL}\n", ""))  # auto: not a terminal

    def test_json_results_carry_the_outcome_and_exit_code(self):
        code, out, err = self.cli("install", "small", "--json")
        value = json.loads(out)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(value, {"schema": catalog.RESULT_SCHEMA, "version": residency.VERSION, "action": "install",
                                 "target": "small", "outcome": "saved", "exit": 0, "message": f"saved {self.SMALL}",
                                 "detail": None, "name": self.SMALL, "path": str(self.llm / self.SMALL)})
        code, out, _ = self.cli("remove", "ollama:../x", "--json")
        self.assertEqual((code, json.loads(out)["outcome"], json.loads(out)["exit"]), (2, "invalid-name", 2))

    def test_the_list_in_text_and_json(self):
        self.place(self.llm, self.SMALL, self.small)
        self.ollama.disk["qwen3.5:2b"] = 2 * 10**9
        code, out, err = self.cli("list")
        self.assertEqual((code, err), (0, ""))
        lines = out.splitlines()
        self.assertEqual(lines[0], f"llama.cpp models: {self.llm}")
        self.assertIn(f"Diffusion models: {self.diffusion}", lines)
        self.assertIn("Ollama models", lines)
        self.assertTrue(any(line.split()[:1] == ["small"] and line.endswith("installed") for line in lines))
        self.assertTrue(any(line.split()[:1] == ["ollama:qwen3.5:2b"] for line in lines))
        code, out, _ = self.cli("list", "--json")
        value = json.loads(out)
        self.assertEqual((value["schema"], value["action"], value["ollama"], value["residency"]),
                         (catalog.RESULT_SCHEMA, "list", "reachable", "known"))
        self.assertEqual({m["id"]: m["state"] for m in value["models"]},
                         {"small": "installed", "tiny-diffusion": "available", "tiny-ollama": "available",
                          "ollama:qwen3.5:2b": "installed"})
        self.assertEqual(set(value["models"][0]), {"id", "name", "kind", "source", "state", "folder", "path",
                                                   "bytes", "partial_bytes", "resident"})

    def test_a_malformed_catalog_is_reported(self):
        self.catalog_path.write_text("{}")
        self.assertEqual(self.cli("install", "small")[:2], (1, "The model catalog is malformed: it is not a "
                                                            "kilix.gpu-residency.catalog/v1 document\n"))

    def test_the_launcher_runs_the_models_commands(self):
        launcher = Path(__file__).resolve().parents[1] / "kilix-gpu"
        done = subprocess.run([str(launcher), "--namespace", self.namespace, "--ollama", self.ollama.url, "models",
                               "--catalog", str(self.catalog_path), "install", "tiny-diffusion"],
                              capture_output=True, text=True, timeout=60,
                              env=residency_containment.child_env(KILIX_GPU_SYSFS_ROOT=str(self.sys)))
        self.assertEqual((done.returncode, done.stdout, done.stderr), (0, f"saved {self.TINY}\n", ""))
        self.assertEqual((self.diffusion / self.TINY).read_bytes(), self.tiny)


if __name__ == "__main__":
    unittest.main()
