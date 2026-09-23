from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from plebian_model_sizer.resources import GIB, MIB, voice_data_root
from plebian_model_sizer.voice import load_profiles, recommend_voice, REQUEST_SCHEMA, validate_request
from test_sizing import snapshot


def request():
    return {"schema": REQUEST_SCHEMA, "models": [
        {"id": name, "task": task, "backend": "cpu", "installed": False, "runtime_supported": supported}
        for name, task, supported in (("espeak", "tts", True), ("mbrola", "tts", True),
                                      ("piper-en-us-kristin-medium", "tts", True),
                                      ("small-en-us", "stt", True), ("lgraph-en-us", "stt", True),
                                      ("vibevoice-asr-bitnet", "stt", False))]}


class SpeechSizingTests(unittest.TestCase):
    def setUp(self):
        self.resources = {**snapshot(), "architecture": "x86_64"}

    def test_packaged_profiles_equal_frozen_source_documents(self):
        root = Path(__file__).resolve().parents[3] / "contracts/v1/profiles/res02-measured"
        profiles = load_profiles()
        self.assertEqual(len(profiles), 13)
        for name, entry in profiles.items():
            directory = root.parent / "tts-auditions-20260923" if name.startswith("audition-") else root
            raw = (directory / (name + ".json")).read_bytes()
            self.assertEqual(entry["source_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(entry["document"], json.loads(raw))

    def test_audition_cpu_and_cuda_reference_profiles(self):
        names = [name for name in load_profiles() if name.startswith("audition-")]
        doc = {"schema": REQUEST_SCHEMA, "models": [
            {"id": name, "task": "tts", "backend": "cuda" if name.endswith("-cuda") else "cpu",
             "installed": True, "runtime_supported": True} for name in names]}
        self.resources.update(ram_total_bytes=32 * GIB, ram_available_bytes=20 * GIB)
        result = recommend_voice(doc, self.resources, task="tts")
        self.assertEqual(len(result["shortlists"]["tts"]), 6)
        pocket = next(row for row in result["candidates"]
                      if row["id"] == "audition-pocket-tts-english-python-alba-cpu")
        self.assertEqual(pocket["inference"]["resources"]["ram"]["required_bytes"],
                         (1026977792 * 12000 + 9999) // 10000)
        self.resources["ram_available_bytes"] = 800 * MIB
        result = recommend_voice(doc, self.resources, task="tts")
        self.assertEqual(set(result["shortlists"]["tts"]),
                         {"audition-espeak", "audition-mbrola", "audition-piper-en-us-kristin-medium"})
        self.resources["ram_available_bytes"] = None
        self.assertEqual(recommend_voice(doc, self.resources, task="tts")["shortlists"]["tts"], [])

    def test_known_memory_fits_do_not_promote_unknown_install_costs(self):
        result = recommend_voice(request(), self.resources)
        self.assertEqual(result["provisional_candidates"], {"tts": "espeak", "stt": "small-en-us"})
        self.assertIsNone(result["selected_model"])
        self.assertFalse(result["qualification_eligible"])
        for row in result["candidates"]:
            self.assertEqual(row["installation"]["verdict"], "unknown")
            if row["runtime_supported"]:
                self.assertEqual(row["verdict"], "estimated-fit")
                self.assertEqual(row["runtime_identity"], "unverified")

    def test_tight_memory_retains_only_small_speech_models(self):
        self.resources["ram_available_bytes"] = 300 * MIB
        result = recommend_voice(request(), self.resources)
        self.assertEqual(result["shortlists"], {"tts": ["espeak", "mbrola"], "stt": []})

    def test_cpu_speech_does_not_need_gpu_and_small_machines_can_fit(self):
        self.resources.update(gpus=[], ram_total_bytes=GIB, ram_available_bytes=800 * MIB)
        result = recommend_voice(request(), self.resources, task="stt")
        self.assertEqual(result["shortlists"], {"stt": ["small-en-us"]})

    def test_margin_is_applied_and_unsupported_model_is_never_recommended(self):
        result = recommend_voice(request(), self.resources)
        row = next(row for row in result["candidates"] if row["id"] == "small-en-us")
        self.assertEqual(row["inference"]["resources"]["ram"]["required_bytes"], (266670080 * 12000 + 9999) // 10000)
        row = next(row for row in result["candidates"] if row["id"] == "vibevoice-asr-bitnet")
        self.assertEqual(row["verdict"], "unsupported")
        self.assertIn("no-resource-profile", row["reasons"])

    def test_architecture_backend_and_task_mismatch_do_not_fit(self):
        for architecture in (None, "aarch64"):
            self.resources["architecture"] = architecture
            self.assertEqual(recommend_voice(request(), self.resources)["shortlists"], {"tts": [], "stt": []})
        self.resources["architecture"] = "x86_64"
        for key, value in (("backend", "cuda"), ("task", "stt")):
            doc = request(); doc["models"][0][key] = value
            row = recommend_voice(doc, self.resources)["candidates"][0]
            self.assertEqual(row["verdict"], "unknown")

    def test_stale_unknown_or_zero_ram_never_produces_fit(self):
        for change in ({"ram_available_bytes": 0}, {"cgroup_status": "unknown"},
                       {"observed_at": (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()}):
            resource = {**self.resources, **change}
            self.assertEqual(recommend_voice(request(), resource)["shortlists"], {"tts": [], "stt": []})

    def test_cuda_profile_requires_free_vram(self):
        doc = {"schema": REQUEST_SCHEMA, "models": [{"id": "qwen3-tts-0.6b-base-cuda", "task": "tts",
                                                     "backend": "cuda", "installed": None, "runtime_supported": True}]}
        self.resources["gpus"][0]["available_bytes"] = 2 * GIB
        self.assertEqual(recommend_voice(doc, self.resources)["candidates"][0]["verdict"], "does-not-fit")
        self.resources["gpus"] = []
        self.assertEqual(recommend_voice(doc, self.resources)["candidates"][0]["verdict"], "unknown")

    def test_bad_catalog_types_and_duplicate_models_rejected(self):
        for key, value in (("id", "../outside"), ("installed", "yes"), ("runtime_supported", 1), ("task", [])):
            doc = request(); doc["models"][0][key] = value
            with self.assertRaises(ValueError): validate_request(doc)
        doc = request(); doc["models"].append(deepcopy(doc["models"][0]))
        with self.assertRaises(ValueError): validate_request(doc)

    def test_voice_storage_environment_precedence(self):
        with patch.dict("os.environ", {"GPU_TERMINAL_HOME": "/data", "KILIX_STORAGE_HOME": "/storage", "KILIX_DATA_HOME": "/datasets"}):
            self.assertEqual(voice_data_root(), Path("/datasets/voice"))


if __name__ == "__main__":
    unittest.main()
