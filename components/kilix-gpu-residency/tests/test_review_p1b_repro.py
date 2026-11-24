"""Adopted from the third reviewer's reproductions (REVIEW-P1 / REVIEW-P1B).

Reviewer reproductions for P1b. Each assertion states the SAFE/expected behaviour;
a FAIL or ERROR in a test named Rn means the finding reproduces. Tests named Sn
check something the review found sound, and should pass.

Adapted for fix round 1 (FX-26): a part is named for its pin now (CatalogCase.part_name);
R3 keeps the old name on purpose, as the part an older revision left."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import residency_containment  # noqa: F401
import kilix_gpu_residency as residency
from kilix_gpu_residency import catalog
from fakes import GIB, MIB, CatalogCase, alive, blob


class ReproP1b(CatalogCase):
    def test_R1_unload_kills_a_registered_holder_then_refuses_because_another_process_has_the_file_open(self):
        path = self.place(self.llm, self.SMALL, self.small)
        _, process = self.model("kilix-chat", self.SMALL, GIB)
        self.holding(path)                       # e.g. a llama-cli the user started by hand
        with self.assertRaises(catalog.CatalogError) as caught:
            self.remove("small", unload=True)
        self.assertEqual(caught.exception.outcome, "open")
        self.assertTrue(path.exists())
        self.assertTrue(alive(process.pid), "kilix-chat's model was unloaded for a removal that was refused")

    def test_R2_a_200_with_an_empty_body_deletes_a_resumable_part(self):
        part = self.place(self.llm, self.part_name(self.SMALL), self.small[:2 * MIB])
        self.downloads.status = 200              # a proxy/CDN hiccup: 200, Content-Length 0
        with self.assertRaises(catalog.CatalogError):
            self.install("small")
        self.assertTrue(part.exists(), "a transient bad response deleted the partial download")

    def test_R3_a_part_left_by_an_older_catalog_revision_costs_a_whole_failed_download(self):
        old = blob(len(self.small), 99)
        self.place(self.llm, f".{self.SMALL}.part", old[:2 * MIB])   # the previous revision's bytes
        result = None
        try:
            result = self.install("small")
        except catalog.CatalogError as error:
            self.fail(f"first install after a catalog update failed: {error.outcome}: {error} ({error.detail})")
        self.assertEqual(result.outcome, "saved")

    def test_R4_a_file_name_registration_holds_a_same_named_file_in_the_other_folder(self):
        self.place(self.llm, self.SMALL, self.small)
        other = self.place(self.diffusion, self.SMALL, b"an unrelated local file")
        _, process = self.model("kilix-chat", self.SMALL, GIB)   # model=path.name, as the note says
        result = self.remove(str(other), unload=True)
        self.assertEqual(result.outcome, "removed")
        self.assertTrue(alive(process.pid), "removing the diffusion copy unloaded the llama.cpp model")

    def test_R5_an_os_error_escapes_as_a_traceback_not_a_one_line_notice(self):
        self.llm.mkdir(parents=True, mode=0o700)
        (self.llm / ".fetch.lock").write_bytes(b"")
        os.chmod(self.llm, 0o500)                # yours, not writable by others, but read-only
        self.addCleanup(os.chmod, self.llm, 0o700)
        try:
            code, out, err = self.cli("install", "small")
        except OSError as error:
            self.fail(f"kilix-gpu models install raised {type(error).__name__}: {error}")
        self.assertIn(code, (1, 2))
        self.assertEqual(len(out.splitlines()), 1)

    def test_R6_ollama_host_0000_makes_a_local_ollama_unreachable_for_the_catalog(self):
        port = self.ollama.url.rsplit(":", 1)[1]
        os.environ["OLLAMA_HOST"] = f"0.0.0.0:{port}"
        try:
            result = catalog.install("ollama:tiny:1b", catalog=self.catalog)
        except catalog.CatalogError as error:
            self.fail(f"{error.outcome}: {error}")
        self.assertEqual(result.outcome, "pulled")

    def test_R7_the_owners_pin_is_lost_once_an_app_registers_the_ollama_model(self):
        self.ollama.disk["qwen3.5:2b"] = 2 * 10**9
        self.ollama.load("qwen3.5:2b", 2 * GIB)
        residency.pin("ollama:qwen3.5:2b", **self.where())
        app = residency.register(app="kilix-chat", model="qwen3.5:2b", backend="ollama", **self.where())
        app.idle()
        with self.assertRaises(catalog.CatalogError) as caught:
            self.remove("ollama:qwen3.5:2b", unload=True)
        self.assertEqual(caught.exception.outcome, "pinned")
        self.assertIn("qwen3.5:2b", self.ollama.disk, "a model the owner pinned was unloaded and deleted")

    def test_R8_a_registration_through_a_symlinked_path_does_not_hold_the_file(self):
        path = self.place(self.llm, self.SMALL, self.small)
        link = self.scratch / "my-model.gguf"
        os.symlink(path, link)
        _, process = self.model("kilix-chat", str(link), GIB)   # the full path the app opened
        with self.assertRaises(catalog.CatalogError) as caught:
            self.remove("small")
        self.assertEqual(caught.exception.outcome, "loaded")
        self.assertTrue(path.exists())

    def test_R9_an_inherited_proc_root_variable_lets_remove_delete_a_registered_model(self):
        """P1 finding 6 through the catalog: with KILIX_GPU_PROC_ROOT inherited, every
        record looks dead, is pruned, and the file it holds is removed without --unload."""
        path = self.place(self.llm, self.SMALL, self.small)
        _, process = self.model("kilix-chat", self.SMALL, GIB)
        os.environ["KILIX_GPU_PROC_ROOT"] = str(self.scratch / "noproc")
        try:
            self.remove("small")
        except catalog.CatalogError as error:
            self.assertEqual(error.outcome, "loaded")
        self.assertTrue(path.exists(), "a model kilix-chat registered was removed without --unload")
        self.assertTrue(alive(process.pid))

    def test_S1_double_dash_targets_work_as_the_wrapper_uses_them(self):
        code, out, _ = self.cli("--progress", "lines", "install", "--", "small")
        self.assertEqual((code, out.strip()), (0, f"saved {self.SMALL}"))
        code, out, _ = self.cli("remove", "--unload", "--", "small")
        self.assertEqual((code, out.strip()), (0, f"removed {self.SMALL}"))

    def test_S2_a_complete_part_hashing_wrong_is_never_published(self):
        bad = bytearray(self.small)
        bad[-1] ^= 1
        self.place(self.llm, self.part_name(self.SMALL), bytes(bad))
        with self.assertRaises(catalog.CatalogError) as caught:
            self.install("small")
        self.assertEqual(caught.exception.outcome, "downloaded-mismatch")
        self.assertFalse((self.llm / self.SMALL).exists())

    def test_S3_a_206_for_a_file_of_another_size_is_refused_and_never_published(self):
        self.place(self.llm, self.part_name(self.SMALL), self.small[:MIB])
        self.downloads.files[self.SMALL] = self.small + b"x"
        with self.assertRaises(catalog.CatalogError):
            self.install("small")
        self.assertFalse((self.llm / self.SMALL).exists())
