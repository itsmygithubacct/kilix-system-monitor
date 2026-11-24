"""The model catalog's data, folders and installs.

Downloads come from a fake loopback server shaped like Hugging Face (a
redirect, then a file that honours Range) or from file:// mirrors. Ollama is
the suite's fake. No test reaches the network or the real model folders.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import unittest

import residency_containment
from kilix_gpu_residency import catalog
from fakes import digest_of, MIB, CatalogCase, stop

# The seed entries: puzzllm-fetch's pinned six (puzzllm/libexec/puzzllm-fetch) and
# d3llm-llada (owner request 2026-11-23, which puzzllm-fetch gains too), copied
# here so a drift in either is caught: id -> (kind, file, bytes, sha256, url).
# Each size and sha256 was also checked against Hugging Face's LFS metadata at the
# pinned revision.
SEED = {
    "qwen3.5-0.8b": ("gguf", "Qwen3.5-0.8B-Q8_0.gguf", 833592096,
                     "37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f",
                     "https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/"
                     "807dfd9a6fadbf703207aac3bfb0b2b3040d077c/Qwen3.5-0.8B-Q8_0.gguf"),
    "qwen3.5-4b-q4": ("gguf", "Qwen3.5-4B-Q4_K_M.gguf", 2740937888,
                      "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
                      "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/"
                      "e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf"),
    "gemma4-e2b-q4-k-m": ("gguf", "gemma-4-E2B-it-Q4_K_M.gguf", 3106738272,
                          "740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8",
                          "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/"
                          "0314792d7f1f7e229411f620751375812bb9faf2/gemma-4-E2B-it-Q4_K_M.gguf"),
    "llada-moe-7b-a1b": ("diffusion-gguf", "LLaDA-MoE-7B-A1B-Instruct.Q4_K_M.gguf", 4520661184,
                         "a8fc1d9d43718a742b55b122cdca739a9cc2e790e38b2316b1a5b10e84489b27",
                         "https://huggingface.co/mradermacher/LLaDA-MoE-7B-A1B-Instruct-GGUF/resolve/"
                         "1080e16761f6f82a92e8bfb54a4c8998dfee0219/LLaDA-MoE-7B-A1B-Instruct.Q4_K_M.gguf"),
    "llada-1.5": ("diffusion-gguf", "LLaDA-1.5.Q4_K_M.gguf", 4925226240,
                  "967765045ef1f9db6f4dff72b14a338e0c897e81618f7779105426e9e2e9fce8",
                  "https://huggingface.co/mradermacher/LLaDA-1.5-GGUF/resolve/"
                  "5acdac19a98ed7ff799324391d82c602474edf3b/LLaDA-1.5.Q4_K_M.gguf"),
    "dream-coder-7b": ("diffusion-gguf", "Dream-Coder-v0-Instruct-7B.Q4_K_M.gguf", 4683074016,
                       "c1b2c58e9f2a6e9cd548347e5845c3e5306862be1151329b22c5b3956d858e3c",
                       "https://huggingface.co/mradermacher/Dream-Coder-v0-Instruct-7B-GGUF/resolve/"
                       "e22431dd659aba45146840ac9b6b9417aff3cac5/Dream-Coder-v0-Instruct-7B.Q4_K_M.gguf"),
    "d3llm-llada": ("diffusion-gguf", "d3LLM_LLaDA.Q4_K_M.gguf", 4925226656,
                    "eb450d47abe05c1c1503d78751a8453c8834ef1a8d34802173914379b7cafa88",
                    "https://huggingface.co/mradermacher/d3LLM_LLaDA-GGUF/resolve/"
                    "6cbf038bb867cd42bb4ff9b9ce931af23b32e3df/d3LLM_LLaDA.Q4_K_M.gguf"),
}
# The Ollama pins, chosen for H2 (an 8 GiB NVIDIA GPU, the sizer's reference class) from the
# local yes/no and BoolQ rankings: id -> (Ollama name, bytes, manifest digest). Each digest is
# the sha256 of the manifest registry.ollama.ai serves for that name, and the size is its layers
# and config, as GET /api/tags reports them.
OLLAMA_SEED = {
    "d3-4b": ("d3:4b", 5833416247, "2c571f91caf64c47b6ca6bd3adc1daa07dbb84ead8e9489758498cc7e11e533d"),
    "d3-2b": ("d3:2b", 3223742935, "e1b3991aa4806936d30a6d624b234aca3f99ba89d61d6d3a17ee81739f94dafa"),
    "qwen3.5-9b": ("qwen3.5:9b", 6550825550, "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"),
    "gemma4-e4b": ("gemma4:e4b", 6583656505, "dc35e8d9c6061baa6f0fa870975ab6932e2542b579b13ea0f199fa4bb7300c9c"),
}

# Read from each file's GGUF header (general.architecture, <arch>.context_length, general.license);
# d3llm-llada's header declares no licence, so its licence is the model card's.
HEADERS = {"qwen3.5-0.8b": ("qwen35", 262144, "apache-2.0"), "qwen3.5-4b-q4": ("qwen35", 262144, "apache-2.0"),
           "gemma4-e2b-q4-k-m": ("gemma4", 131072, "apache-2.0"),
           "llada-moe-7b-a1b": ("llada-moe", 8192, "apache-2.0"), "llada-1.5": ("llada", 4096, "mit"),
           "dream-coder-7b": ("dream", 32768, "apache-2.0"), "d3llm-llada": ("llada", 4096, "apache-2.0"),
           # Ollama's /api/show model_info for the pinned manifests.
           "d3-4b": ("qwen35", 262144, "apache-2.0"), "d3-2b": ("qwen35", 262144, "apache-2.0"),
           "qwen3.5-9b": ("qwen35", 262144, "apache-2.0"), "gemma4-e4b": ("gemma4", 131072, "apache-2.0")}


class CatalogDataTests(unittest.TestCase):
    def test_the_packaged_catalog_pins_the_seed_models(self):
        loaded = catalog.load()
        self.assertEqual({e.id: (e.kind, e.file, e.bytes, e.sha256, e.url) for e in loaded.entries
                          if e.kind != "ollama"}, SEED)
        self.assertEqual({e.id: (e.ollama, e.bytes, e.digest) for e in loaded.entries if e.kind == "ollama"},
                         OLLAMA_SEED)
        for entry in loaded.entries:
            with self.subTest(entry=entry.id):
                self.assertEqual(entry.folder, catalog.FOLDER_FOR_KIND.get(entry.kind))
                self.assertEqual((entry.architecture, entry.context_length), HEADERS[entry.id][:2])
                self.assertTrue(entry.licence.startswith(HEADERS[entry.id][2] + ","))
                self.assertIn("/", entry.source)

    def test_the_packaged_folders_are_puzzllm_fetchs(self):
        folders = catalog.load().folders
        self.assertEqual(folders["llama.cpp"], catalog.Folder("KILIX_LLM_MODELS_DIR", "GPU_TERMINAL_HOME",
                                                              "~/.local/gpu_terminal", "kilix-llm/models"))
        self.assertEqual(folders["diffusion"], catalog.Folder("PUZZLLM_DIFFUSION_MODELS", "XDG_DATA_HOME",
                                                              "~/.local/share", "puzzllm/diffusion"))

    def valid(self) -> dict:
        return json.loads((Path(catalog.__file__).parent / "catalog.json").read_text())

    def test_malformed_catalogs_are_refused(self):
        def change(path, value):
            def apply(document):
                target = document
                for key in path[:-1]:
                    target = target[key]
                if value is KeyError:
                    del target[path[-1]]
                else:
                    target[path[-1]] = value
            return apply
        first = ("models", 0)
        mutations = {
            "extra field": change((*first, "mirror"), "x"),
            "missing field": change((*first, "licence"), KeyError),
            "short sha": change((*first, "sha256"), "ab" * 31),
            "uppercase sha": change((*first, "sha256"), "AB" * 32),
            "unpinned revision": change((*first, "url"), "https://huggingface.co/a/b/resolve/main/Qwen3.5-0.8B-Q8_0.gguf"),
            "another file": change((*first, "url"), "https://huggingface.co/a/b/resolve/" + "0" * 40 + "/x.gguf"),
            "another host": change((*first, "url"), "https://example.org/a/b/resolve/" + "0" * 40
                                   + "/Qwen3.5-0.8B-Q8_0.gguf"),
            "file url in the packaged catalog": change((*first, "url"), "file:///srv/Qwen3.5-0.8B-Q8_0.gguf"),
            "kind in the wrong folder": change((*first, "folder"), "diffusion"),
            "unknown kind": change((*first, "kind"), "safetensors"),
            "path in the file name": change((*first, "file"), "../Qwen3.5-0.8B-Q8_0.gguf"),
            "hidden file name": change((*first, "file"), ".Qwen.gguf"),
            "zero bytes": change((*first, "bytes"), 0),
            "boolean bytes": change((*first, "bytes"), True),
            "string context": change((*first, "context_length"), "4096"),
            "control character": change((*first, "name"), "Qwen\x1b[2J"),
            "id like a file": change((*first, "id"), "qwen.gguf"),
            "id with a colon": change((*first, "id"), "ollama:qwen"),
            "repeated id": change(("models", 1, "id"), "qwen3.5-0.8b"),
            "repeated file": change(("models", 1, "file"), "Qwen3.5-0.8B-Q8_0.gguf"),
            "absolute default": change(("folders", "llama.cpp", "base_default"), "/srv/models"),
            "escaping subpath": change(("folders", "diffusion", "subpath"), "../elsewhere"),
            "missing folder role": change(("folders", "diffusion"), KeyError),
            "lowercase variable": change(("folders", "diffusion", "env"), "puzzllm_models"),
            "wrong schema": change(("schema",), "kilix.gpu-residency.catalog/v2"),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                document = copy.deepcopy(self.valid())
                mutate(document)
                with self.assertRaises(catalog.CatalogError) as caught:
                    catalog._parse(json.dumps(document).encode(), local=False)
                self.assertEqual(caught.exception.outcome, "unavailable")
        catalog._parse(json.dumps(self.valid()).encode(), local=False)  # the unmutated control
        duplicate = json.dumps(self.valid()).replace('"schema":', '"schema": "x", "schema":', 1)
        with self.assertRaises(catalog.CatalogError):
            catalog._parse(duplicate.encode(), local=False)

    def test_only_a_catalog_given_by_path_may_use_local_urls(self):
        document = self.valid()
        document["models"] = document["models"][:1]
        for url, local_ok in (("file:///srv/m/Qwen3.5-0.8B-Q8_0.gguf", True),
                              ("http://127.0.0.1:8080/x/Qwen3.5-0.8B-Q8_0.gguf", True),
                              ("http://192.0.2.1/x/Qwen3.5-0.8B-Q8_0.gguf", False),
                              ("file://elsewhere/srv/Qwen3.5-0.8B-Q8_0.gguf", False),
                              ("ftp://127.0.0.1/Qwen3.5-0.8B-Q8_0.gguf", False)):
            document["models"][0]["url"] = url
            raw = json.dumps(document).encode()
            with self.subTest(url=url):
                with self.assertRaises(catalog.CatalogError):
                    catalog._parse(raw, local=False)
                if local_ok:
                    catalog._parse(raw, local=True)
                else:
                    with self.assertRaises(catalog.CatalogError):
                        catalog._parse(raw, local=True)

    def test_an_ollama_entry_must_pin_a_digest_and_size(self):
        packaged = json.loads((Path(catalog.__file__).parent / "catalog.json").read_text())
        good = next(m for m in packaged["models"] if m["kind"] == "ollama")
        catalog._parse(json.dumps({**packaged, "models": [good]}).encode(), local=False)
        for change in ({"digest": "sha256:" + good["digest"]}, {"digest": good["digest"].upper()},
                       {"digest": good["digest"][:-1]}, {"digest": None}, {"bytes": 0}, {"bytes": "5"},
                       {"bytes": True}, {"digest": None, "bytes": None}):
            with self.subTest(change=change):
                with self.assertRaises(catalog.CatalogError):
                    catalog._parse(json.dumps({**packaged, "models": [{**good, **change}]}).encode(), local=False)
        for missing in ("digest", "bytes"):
            with self.subTest(missing=missing):
                with self.assertRaises(catalog.CatalogError):
                    catalog._parse(json.dumps({**packaged, "models": [
                        {k: v for k, v in good.items() if k != missing}]}).encode(), local=False)

    def test_ollama_names_follow_puzzllm_fetch(self):
        for name in ("qwen3.5:2b", "library/llama3", "hf.co/user/repo:Q4_K_M", "a", "a" * 128, "bonsai-2:27b",
                     "a_b.c"):
            with self.subTest(accepted=name):
                self.assertTrue(catalog.ollama_name_ok(name))
        for name in ("", "..", "a..b", "a//b", "/a", ":a", ".a", "-a", "a/", "a:", "a b", "a;b", "ä",
                     "a" * 129, "a+b", "a\nb", "$(x)"):
            with self.subTest(refused=name):
                self.assertFalse(catalog.ollama_name_ok(name))

    def test_gguf_names_follow_puzzllm_fetch(self):
        for name in ("Qwen3.5-0.8B-Q8_0.gguf", "a.gguf", "LLaDA-1.5.Q4_K_M.gguf"):
            with self.subTest(accepted=name):
                self.assertTrue(catalog.plain_gguf(name))
        for name in (".a.gguf", "a/b.gguf", "a..gguf", "a.GGUF", "a.gguf.part", "a.gguf ", "-a.gguf", ".gguf",
                     ".partial.Ab12Cd", "a.gguf.download.json"):
            with self.subTest(refused=name):
                self.assertFalse(catalog.plain_gguf(name))


class FolderTests(CatalogCase):
    def test_folders_follow_puzzllm_fetchs_variables(self):
        home = str(self.home)
        cases = [
            ({}, home + "/.local/gpu_terminal/kilix-llm/models", home + "/.local/share/puzzllm/diffusion"),
            ({"GPU_TERMINAL_HOME": "/srv/gt", "XDG_DATA_HOME": "/srv/data"},
             "/srv/gt/kilix-llm/models", "/srv/data/puzzllm/diffusion"),
            ({"GPU_TERMINAL_HOME": "/srv/gt", "KILIX_LLM_MODELS_DIR": "/srv/llm",
              "XDG_DATA_HOME": "/srv/data", "PUZZLLM_DIFFUSION_MODELS": "/srv/dlm"}, "/srv/llm", "/srv/dlm"),
            # ${VAR:-default}: an empty variable counts as unset.
            ({"KILIX_LLM_MODELS_DIR": "", "GPU_TERMINAL_HOME": "", "PUZZLLM_DIFFUSION_MODELS": ""},
             home + "/.local/gpu_terminal/kilix-llm/models", home + "/.local/share/puzzllm/diffusion"),
        ]
        names = ("KILIX_LLM_MODELS_DIR", "PUZZLLM_DIFFUSION_MODELS", "GPU_TERMINAL_HOME", "XDG_DATA_HOME")
        for env, llm, diffusion in cases:
            with self.subTest(env=env):
                for name in names:
                    os.environ.pop(name, None)
                os.environ.update(env)
                self.assertEqual(catalog.folder("llama.cpp", self.catalog), llm)
                self.assertEqual(catalog.folder("diffusion", self.catalog), diffusion)
        for name in names:
            os.environ.pop(name, None)
        os.environ["KILIX_LLM_MODELS_DIR"] = "relative/models"
        self.refused("unsafe-folder", catalog.folder, "llama.cpp", self.catalog)

    def test_a_new_folder_and_its_parents_are_private_and_the_file_is_too(self):
        previous = os.umask(0o022)  # a permissive umask must not loosen them
        try:
            self.install("small")
        finally:
            os.umask(previous)
        for path in (self.home / ".local", self.home / ".local" / "gpu_terminal", self.llm.parent, self.llm):
            with self.subTest(folder=str(path)):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((self.llm / self.SMALL).stat().st_mode), 0o600)

    def test_a_folder_others_could_write_is_refused(self):
        self.place(self.llm, "other.gguf", b"x")
        os.chmod(self.llm, 0o770)
        self.refused("unsafe-folder", self.install, "small")
        self.refused("unsafe-folder", self.remove, "other.gguf")
        self.assertFalse((self.llm / self.SMALL).exists())
        self.assertTrue((self.llm / "other.gguf").exists())


class InstallTests(CatalogCase):
    def test_a_download_is_verified_then_named_after_one_redirect(self):
        stages = []
        result = self.install("small", progress=lambda stage, done, total: stages.append((stage, done, total)))
        self.assertEqual((result.outcome, result.message, result.exit_code), ("saved", f"saved {self.SMALL}", 0))
        self.assertEqual(result.path, str(self.llm / self.SMALL))
        self.assertEqual((self.llm / self.SMALL).read_bytes(), self.small)
        self.assertEqual(sorted(os.listdir(self.llm)), [".fetch.lock", self.SMALL])
        self.assertEqual([path for path, _ in self.downloads.requests], [f"/resolve/{self.SMALL}", f"/files/{self.SMALL}"])
        self.assertEqual(self.downloads.ranges(self.SMALL), [None])
        self.assertEqual(stages[-1], ("downloading", len(self.small), len(self.small)))

    def test_a_diffusion_model_from_a_file_mirror_goes_to_the_diffusion_folder(self):
        result = self.install("tiny-diffusion")
        self.assertEqual(result.message, f"saved {self.TINY}")
        self.assertEqual((self.diffusion / self.TINY).read_bytes(), self.tiny)
        self.assertFalse(self.llm.exists())

    def test_a_saved_model_is_already_saved(self):
        self.install("small")
        requests = len(self.downloads.requests)
        result = self.install("small")
        self.assertEqual((result.outcome, result.message), ("already-saved", f"already saved {self.SMALL}"))
        self.assertEqual(len(self.downloads.requests), requests)

    def test_a_different_file_with_the_same_name_is_refused_and_kept(self):
        for data in (b"not the model", self.small[:-1] + bytes([self.small[-1] ^ 1])):
            with self.subTest(size=len(data)):
                path = self.place(self.llm, self.SMALL, data)
                error = self.refused("conflict", self.install, "small")
                self.assertEqual(str(error), f"{self.SMALL} is already in the models folder and does not match")
                self.assertEqual(path.read_bytes(), data)
        self.assertEqual(self.downloads.requests, [])

    def test_symlinks_are_refused_as_the_file_and_as_the_part(self):
        outside = self.place(self.scratch / "outside", "real.gguf", b"untouched")
        self.llm.mkdir(parents=True, mode=0o700)
        os.symlink(outside, self.llm / self.SMALL)
        error = self.refused("symlink", self.install, "small")
        self.assertEqual(str(error), f"refusing to replace a symlink: {self.SMALL}")
        os.unlink(self.llm / self.SMALL)
        os.symlink(outside, self.llm / f".{self.SMALL}.part")
        self.refused("symlink", self.install, "small")
        self.assertEqual(outside.read_bytes(), b"untouched")
        self.assertEqual(self.downloads.requests, [])

    def test_a_dropped_connection_is_resumed_in_the_same_call(self):
        self.downloads.drops[self.SMALL] = [MIB + 5000, 2 * MIB + 17]
        self.assertEqual(self.install("small").outcome, "saved")
        self.assertEqual(self.downloads.ranges(self.SMALL), [None, f"bytes={MIB + 5000}-", f"bytes={2 * MIB + 17}-"])
        self.assertEqual((self.llm / self.SMALL).read_bytes(), self.small)

    def test_drops_that_make_progress_never_use_up_the_attempts(self):
        catalog.ATTEMPTS = 2
        self.downloads.drops[self.SMALL] = [MIB, 2 * MIB, 3 * MIB]
        self.assertEqual(self.install("small").outcome, "saved")
        self.assertEqual(len(self.downloads.ranges(self.SMALL)), 4)

    def test_a_resume_at_the_wrong_offset_is_refused(self):
        self.place(self.llm, f".{self.SMALL}.part", self.small[:MIB])
        self.downloads.shift = 1
        error = self.refused("download-failed", self.install, "small")
        self.assertIn("wrong offset", error.detail)
        self.assertEqual((self.llm / f".{self.SMALL}.part").read_bytes(), self.small[:MIB])

    def test_a_verified_part_never_replaces_a_file_that_appeared(self):
        part = self.place(self.llm, ".x.gguf.part", b"new")
        existing = self.place(self.llm, "x.gguf", b"existing")
        fd = os.open(self.llm, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, fd)
        self.refused("conflict", catalog._publish, fd, ".x.gguf.part", "x.gguf")
        self.assertEqual((existing.read_bytes(), part.read_bytes()), (b"existing", b"new"))
        os.unlink(existing)
        catalog._publish(fd, ".x.gguf.part", "x.gguf")
        self.assertEqual(sorted(os.listdir(self.llm)), ["x.gguf"])

    def test_a_symlinked_lock_is_refused_and_not_followed(self):
        target = self.place(self.scratch / "elsewhere", "lock", b"keep")
        self.llm.mkdir(parents=True, mode=0o700)
        os.symlink(target, self.llm / ".fetch.lock")
        self.refused("unsafe-folder", self.install, "small")
        self.assertEqual(target.read_bytes(), b"keep")
        self.assertEqual(self.downloads.requests, [])

    def test_a_download_that_keeps_failing_keeps_its_part_for_the_next_call(self):
        catalog.ATTEMPTS = 2
        self.downloads.drops[self.SMALL] = [2 * MIB, 2 * MIB, 2 * MIB]
        error = self.refused("download-failed", self.install, "small")
        self.assertEqual((str(error), error.exit_code), (f"could not download {self.SMALL}", 1))
        part = self.llm / f".{self.SMALL}.part"
        self.assertEqual(part.read_bytes(), self.small[:2 * MIB])
        self.assertFalse((self.llm / self.SMALL).exists())
        self.downloads.drops[self.SMALL] = []
        self.assertEqual(self.install("small").outcome, "saved")
        self.assertEqual(self.downloads.ranges(self.SMALL)[-1], f"bytes={2 * MIB}-")
        self.assertFalse(part.exists())
        self.assertEqual((self.llm / self.SMALL).read_bytes(), self.small)

    def test_a_server_that_ignores_range_restarts_from_the_beginning(self):
        self.place(self.llm, f".{self.SMALL}.part", self.small[:MIB])
        self.downloads.ignore_range = True
        self.assertEqual(self.install("small").outcome, "saved")
        self.assertEqual(self.downloads.ranges(self.SMALL), [f"bytes={MIB}-"])
        self.assertEqual((self.llm / self.SMALL).read_bytes(), self.small)

    def test_a_complete_part_is_checked_and_named_without_a_request(self):
        self.place(self.llm, f".{self.SMALL}.part", self.small)
        self.assertEqual(self.install("small").outcome, "saved")
        self.assertEqual(self.downloads.requests, [])
        self.assertEqual((self.llm / self.SMALL).read_bytes(), self.small)

    def test_a_server_that_cannot_resume_a_part_starts_again(self):
        # 416 for a range the server cannot satisfy: here, because its file is shorter than the part.
        self.place(self.llm, f".{self.SMALL}.part", self.small[:MIB])
        self.downloads.files[self.SMALL] = self.small[:MIB // 2]
        self.downloads.drops[self.SMALL] = []
        catalog.ATTEMPTS = 2
        self.refused("downloaded-mismatch", self.install, "small")
        self.assertEqual(self.downloads.ranges(self.SMALL), [f"bytes={MIB}-", None])
        self.assertEqual(sorted(os.listdir(self.llm)), [".fetch.lock"])

    def test_a_part_longer_than_the_model_starts_again(self):
        self.place(self.llm, f".{self.SMALL}.part", self.small + b"extra")
        self.assertEqual(self.install("small").outcome, "saved")
        self.assertEqual(self.downloads.ranges(self.SMALL), [None])

    def test_a_corrupt_part_fails_once_and_is_removed(self):
        part = self.place(self.llm, f".{self.SMALL}.part", b"\0" * MIB)
        error = self.refused("downloaded-mismatch", self.install, "small")
        self.assertEqual((str(error), error.exit_code), (f"downloaded {self.SMALL} did not match", 1))
        self.assertFalse(part.exists())
        self.assertFalse((self.llm / self.SMALL).exists())
        self.assertEqual(self.install("small").outcome, "saved")

    def test_a_server_with_another_file_is_refused(self):
        for data in (self.small + b"!", self.small[:-1] + b"!"):
            with self.subTest(size=len(data)):
                self.downloads.files[self.SMALL] = data
                self.refused("downloaded-mismatch", self.install, "small")
                self.assertEqual(sorted(os.listdir(self.llm)), [".fetch.lock"])

    def test_http_errors_fail_without_leaving_an_empty_part(self):
        catalog.ATTEMPTS = 3
        for status, attempts in ((404, 1), (403, 1), (503, 3), (429, 3)):
            with self.subTest(status=status):
                self.downloads.requests.clear()
                self.downloads.status = status
                error = self.refused("download-failed", self.install, "small")
                self.assertIn(str(status), error.detail)
                self.assertEqual(len(self.downloads.ranges(self.SMALL)), attempts)
                self.assertEqual(sorted(os.listdir(self.llm)), [".fetch.lock"])

    def test_redirects_may_not_leave_https_or_loopback_http(self):
        for target in ("http://192.0.2.1/files/x", "ftp://127.0.0.1/x", "file:///etc/hostname"):
            with self.subTest(redirect=target):
                self.downloads.redirect = target
                error = self.refused("download-failed", self.install, "small")
                self.assertIn("redirected", error.detail)
        self.assertEqual(self.downloads.ranges(self.SMALL), [])

    def test_a_folder_another_download_holds_is_busy(self):
        self.llm.mkdir(parents=True, mode=0o700)
        code = ("import fcntl, os, sys, time\n"
                "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
                "fcntl.flock(fd, fcntl.LOCK_EX)\nprint('ready', flush=True)\ntime.sleep(600)\n")
        holder = subprocess.Popen([sys.executable, "-c", code, str(self.llm / ".fetch.lock")], stdout=subprocess.PIPE,
                                  text=True, env=residency_containment.child_env(), process_group=0)
        self.addCleanup(stop, holder)
        self.assertEqual(holder.stdout.readline().strip(), "ready")
        error = self.refused("busy-folder", self.install, "small")
        self.assertEqual((str(error), error.exit_code), ("another download is using the models folder", 2))
        self.assertEqual(self.downloads.requests, [])

    def test_two_installs_at_once_download_once(self):
        self.downloads.drops[self.SMALL] = [MIB]  # slow the first one down with a resume
        results, errors = [], []

        def run():
            try:
                results.append(self.install("small").outcome)
            except catalog.CatalogError as error:
                errors.append(error.outcome)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        outcomes = sorted(results + errors)
        self.assertEqual(len(outcomes), 2)
        self.assertIn("saved", outcomes)
        self.assertIn([o for o in outcomes if o != "saved"][0] if outcomes.count("saved") == 1 else "x",
                      ("busy-folder", "already-saved"))
        self.assertEqual(self.downloads.ranges(self.SMALL).count(None), 1)

    def test_a_part_another_process_is_writing_is_left_alone(self):
        part = self.place(self.llm, f".{self.SMALL}.part", self.small[:MIB])
        writer = self.holding(part)
        error = self.refused("open", self.install, "small")
        self.assertIn(f"pid {writer.pid}", str(error))
        self.assertEqual(part.read_bytes(), self.small[:MIB])
        self.assertEqual(self.downloads.requests, [])

    def test_not_enough_disk_space_is_refused_before_downloading(self):
        catalog.free_bytes = lambda fd: MIB
        error = self.refused("no-space", self.install, "small")
        self.assertTrue(str(error).startswith(f"not enough disk space for {self.SMALL}"))
        self.assertEqual(self.downloads.requests, [])
        catalog.free_bytes = lambda fd: len(self.small) - MIB  # enough for what a part still lacks
        self.place(self.llm, f".{self.SMALL}.part", self.small[:2 * MIB])
        self.assertEqual(self.install("small").outcome, "saved")

    def test_an_unknown_id_is_refused(self):
        error = self.refused("unknown", self.install, "qwen-9000")
        self.assertEqual((str(error), error.exit_code), ("unknown model qwen-9000", 2))


class OllamaInstallTests(CatalogCase):
    def pulls(self):
        return [body for method, path, body in self.ollama.requests if path == "/api/pull"]

    def test_a_model_ollama_has_is_already_installed(self):
        self.ollama.disk["qwen3.5:2b"] = 2 * 10**9
        result = self.install("ollama:qwen3.5:2b")
        self.assertEqual((result.outcome, result.message), ("already-installed", "already installed qwen3.5:2b"))
        self.assertEqual(self.pulls(), [])

    def test_a_missing_model_is_pulled(self):
        stages = []
        result = self.install("ollama:qwen3.5:2b", progress=lambda *args: stages.append(args))
        self.assertEqual((result.outcome, result.message), ("pulled", "pulled qwen3.5:2b"))
        self.assertEqual(self.pulls(), [{"model": "qwen3.5:2b", "stream": True}])
        self.assertIn(("pulling abc", 5, 10), stages)
        self.assertIn("qwen3.5:2b", self.ollama.disk)

    def test_a_catalog_ollama_entry_pulls_its_name_and_then_has_its_pin(self):
        self.assertEqual(self.install("tiny-ollama").message, "pulled tiny:1b")
        result = self.install("tiny-ollama")
        self.assertEqual((result.outcome, result.message), ("already-installed", "already installed tiny:1b"))
        self.assertEqual(len(self.pulls()), 1)

    def test_an_installed_catalog_model_with_another_digest_is_refused(self):
        self.ollama.disk["tiny:latest"] = 10**9
        self.ollama.disk["tiny:1b"] = 10**9
        for listed, shown in (("0" * 64, "0" * 64), ("bad\ndigest", "no digest"), (None, "no digest")):
            with self.subTest(listed=listed):
                self.ollama.digests["tiny:1b"] = listed
                error = self.refused("conflict", self.install, "tiny-ollama")
                self.assertEqual((str(error), error.exit_code),
                                 ("tiny:1b is already installed and does not match the catalog", 2))
                self.assertEqual(error.detail, f"Ollama has {shown}; the catalog pins {digest_of('tiny:1b')}")
        self.assertEqual(self.install("ollama:tiny:1b").outcome, "already-installed")  # not a catalog pin
        self.assertEqual(self.pulls(), [])

    def test_a_pull_whose_tag_has_moved_did_not_match_and_is_kept(self):
        self.ollama.pull_digest = "1" * 64
        error = self.refused("downloaded-mismatch", self.install, "tiny-ollama")
        self.assertEqual((str(error), error.exit_code), ("pulled tiny:1b did not match the catalog", 1))
        self.assertEqual(error.detail, f"Ollama has {'1' * 64}; the catalog pins {digest_of('tiny:1b')}; "
                                       "remove it with: kilix-gpu models remove ollama:tiny:1b")
        self.assertIn("tiny:1b", self.ollama.disk)
        self.assertEqual([r for r in self.ollama.requests if r[0] == "DELETE"], [])
        self.assertEqual(self.install("ollama:qwen3.5:2b").outcome, "pulled")  # unpinned: any digest

    def test_a_failed_pull_says_so(self):
        self.ollama.pull_error = "pull model manifest: file does not exist"
        error = self.refused("pull-failed", self.install, "ollama:nothing:1b")
        self.assertEqual((str(error), error.exit_code), ("could not pull nothing:1b", 1))
        self.assertIn("file does not exist", error.detail)

    def test_a_pull_that_ends_without_success_failed(self):
        self.ollama.pull_truncated = True
        error = self.refused("pull-failed", self.install, "ollama:qwen3.5:2b")
        self.assertIn("without reporting success", error.detail)

    def test_invalid_names_are_refused_before_asking_ollama(self):
        for name in ("", "../x", "a b", "-rm", "x:"):
            with self.subTest(name=name):
                error = self.refused("invalid-name", self.install, "ollama:" + name)
                self.assertEqual(str(error), "Ollama names use letters, digits, and . _ : / -")
        self.assertEqual(self.ollama.requests, [])

    def test_an_unreachable_or_remote_ollama_is_unavailable(self):
        for url in ("http://127.0.0.1:9", "http://192.0.2.1:11434"):
            with self.subTest(url=url):
                with self.assertRaises(catalog.CatalogError) as caught:
                    catalog.install("ollama:qwen3.5:2b", catalog=self.catalog, ollama=url)
                self.assertEqual((caught.exception.outcome, str(caught.exception)),
                                 ("unavailable", "Ollama is not reachable"))


if __name__ == "__main__":
    unittest.main()
