from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from plebian_model_sizer.resources import GIB, MIB
from plebian_model_sizer.runtime import load_profiles, recommend_runtime, REQUEST_SCHEMA
from test_sizing import snapshot


def request(*names):
    profiles = load_profiles()
    return {"schema": REQUEST_SCHEMA, "models": [
        {"id": name, "task": profiles[name]["document"]["task"],
         "manifest_digest": profiles[name]["document"]["manifest_digest"]} for name in names]}


class RuntimeSizingTests(unittest.TestCase):
    def setUp(self):
        self.resources = {**snapshot(), "architecture": "x86_64"}

    def test_profiles_match_source_documents_and_remain_unqualified(self):
        root = Path(__file__).resolve().parents[1] / "profiles/rc5-runtime-reference"
        for name, entry in load_profiles().items():
            raw = (root / (name + ".json")).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["source_sha256"])
            self.assertEqual(json.loads(raw), entry["document"])
            self.assertFalse(entry["document"]["qualification_eligible"])

    def test_default_falls_back_from_s_to_tiny_to_nano_to_none(self):
        doc = request("yolox_nano", "yolox_tiny", "yolox_s")
        requirements = {name: (entry["document"]["ram_peak_bytes"] * 12500 + 9999) // 10000
                        for name, entry in load_profiles().items() if name.startswith("yolox")}
        for name in ("yolox_s", "yolox_tiny", "yolox_nano"):
            self.resources["ram_available_bytes"] = requirements[name] + 256 * MIB
            result = recommend_runtime(doc, self.resources)
            self.assertEqual(result["defaults"]["vision"], name)
            self.assertIsNone(result["selected_model"])
        self.resources["ram_available_bytes"] = 256 * MIB
        self.assertIsNone(recommend_runtime(doc, self.resources)["defaults"]["vision"])

    def test_cpu_audio_does_not_need_gpu(self):
        self.resources["gpus"] = []
        result = recommend_runtime(request("encodec-24khz-stateful", "encodec-48khz-frame"), self.resources)
        self.assertEqual(result["defaults"]["audio"], "encodec-24khz-stateful")
        self.assertTrue(all(row["verdict"] == "estimated-fit" for row in result["candidates"]))

    def test_unknown_cgroup_headroom_never_confirms_fit(self):
        self.resources["cgroup_status"] = "unknown"
        result = recommend_runtime(request("yolox_s"), self.resources)
        self.assertEqual(result["candidates"][0]["verdict"], "unknown")
        self.assertIsNone(result["defaults"]["vision"])

    def test_catalog_change_and_architecture_mismatch_refuse_profile(self):
        for change in ("catalog", "architecture"):
            doc, resources = request("yolox_s"), deepcopy(self.resources)
            if change == "catalog": doc["models"][0]["manifest_digest"] = "0" * 64
            else: resources["architecture"] = "aarch64"
            row = recommend_runtime(doc, resources)["candidates"][0]
            self.assertEqual(row["verdict"], "unknown")
            self.assertIsNone(row["required_ram_bytes"])

    def test_image_missing_ram_cannot_become_fit_even_with_large_gpu(self):
        self.resources.update(ram_total_bytes=64 * GIB, ram_available_bytes=48 * GIB)
        self.resources["gpus"][0].update(total_bytes=24 * GIB, available_bytes=20 * GIB)
        result = recommend_runtime(request("bonsai-image-4b-ternary-gemlite"), self.resources)
        row = result["candidates"][0]
        self.assertEqual(row["verdict"], "unknown")
        self.assertIn("missing-memory-measurement", row["reasons"])
        self.assertIsNone(result["defaults"]["image"])
        self.resources["gpus"][0].update(total_bytes=6 * GIB, available_bytes=6 * GIB)
        self.assertEqual(recommend_runtime(request("bonsai-image-4b-ternary-gemlite"), self.resources)["candidates"][0]["verdict"], "does-not-fit")

    def test_unknown_alternate_never_inherits_other_model_measurement(self):
        doc = request("bonsai-image-4b-ternary-gemlite")
        doc["models"][0]["id"] = "bonsai-image-4b-binary-gemlite"
        row = recommend_runtime(doc, self.resources)["candidates"][0]
        self.assertEqual(row["verdict"], "unknown")
        self.assertEqual(row["reasons"], ["no-resource-profile"])
        self.assertIsNone(row["required_vram_bytes"])

    def test_malformed_and_duplicate_requests_rejected(self):
        for doc in ({"schema": REQUEST_SCHEMA, "models": []},
                    {"schema": REQUEST_SCHEMA, "models": request("yolox_s")["models"] * 2}):
            with self.assertRaises(ValueError): recommend_runtime(doc, self.resources)
        for task in ([], {}, None, True):
            doc = request("yolox_s")
            doc["models"][0]["task"] = task
            with self.assertRaises(ValueError): recommend_runtime(doc, self.resources)


if __name__ == "__main__":
    unittest.main()
