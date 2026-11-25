from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from plebian_model_sizer import binary_judge
from plebian_model_sizer.cli import main

GIB, MIB = 1024**3, 1024**2
QWEN_SHA = "37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f"
ENGINE = "llama.cpp@8ea290247c87ced2ab245b056ffe96dbcf90d36c"


def profile(**changes) -> dict:
    body = {
        "id": "test-qwen.cpu.x86_64",
        "model": {"id": "qwen3.5-0.8b-q8_0", "sha256": QWEN_SHA, "weights_bytes": 833_592_096,
                  "n_ctx_train": 262_144},
        "engine": {"id": ENGINE, "backend": "cpu"},
        "architecture": "x86_64",
        "standin_adapter": {"rank": 8, "bytes": 21_671_008},
        "ram": {"fixed_bytes": 1_000_000_000, "per_token_bytes": 13_312},
        "vram": None,
        "engine_cap_tokens": None,
        "template_tokens": 176,
        "bytes_per_token": 4.0,
        "margin_basis_points": 500,
        "evidence": {"note": "test fixture"},
    }
    body.update(changes)
    body["profile_sha256"] = binary_judge.profile_digest(body)
    return body


def gpu_profile() -> dict:
    return profile(id="test-qwen.cuda.x86_64", engine={"id": ENGINE, "backend": "cuda"},
                   ram={"fixed_bytes": 1_200_000_000, "per_token_bytes": 0},
                   vram={"fixed_bytes": 1_120_000_000, "per_token_bytes": 13_312})


def snapshot(ram_available: int, gpus: list | None = None, cgroup: str = "unlimited") -> dict:
    return {"schema": "plebian.models.resources/v1-development",
            "observed_at": datetime.now(timezone.utc).isoformat(), "architecture": "x86_64",
            "ram_total_bytes": 64 * GIB, "ram_available_bytes": ram_available,
            "cgroup_status": cgroup, "gpus": gpus or [], "cuda_device_mapping": "unverified",
            "disk_available_bytes": None}


def ask(snap: dict, profiles: list | None = None, **options) -> dict:
    options.setdefault("model", "qwen3.5-0.8b-q8_0")
    options.setdefault("model_sha256", QWEN_SHA)
    options.setdefault("engine", ENGINE)
    return binary_judge.recommend_binary_judge(snap, profiles=profiles or [profile()], **options)


class ArithmeticTests(unittest.TestCase):
    def test_memory_limit_with_margin_reserve_and_256_step(self):
        # budget 4 GiB - 1 GiB = 3_221_225_472; fixed 1e9 * 1.05 = 1_050_000_000;
        # per token 13_312 * 1.05 = 13_977.6 -> 13_978; 2_171_225_472 // 13_978 = 155_331 -> 155_136.
        result = ask(snapshot(4 * GIB))
        self.assertEqual(result["max_context_tokens"], 155_136)
        self.assertEqual(result["max_input_tokens"], 155_136 - 176 - 1)
        self.assertEqual(result["max_input_bytes_estimate"], 619_836)
        self.assertEqual((result["limit_reason"], result["limit_resource"]), ("memory", "ram"))
        ram = result["breakdown"]["ram"]
        self.assertEqual(ram, {"budget_bytes": 3_221_225_472, "fixed_bytes": 1_050_000_000,
                               "per_token_bytes": 13_978, "max_context_tokens": 155_136})
        self.assertEqual(result["reserve_bytes"], GIB)
        self.assertEqual(result["resource_source"], "live")

    def test_a_larger_adapter_than_the_stand_in_adds_its_excess(self):
        result = ask(snapshot(4 * GIB), adapter_bytes=31_671_008)
        self.assertEqual(result["breakdown"]["adapter_bytes_over_standin"], 10_000_000)
        self.assertEqual(result["breakdown"]["ram"]["fixed_bytes"], 1_060_000_000)
        self.assertEqual(result["max_context_tokens"], 154_368)
        # A smaller one is not credited below the measured stand-in.
        self.assertEqual(ask(snapshot(4 * GIB), adapter_bytes=0)["max_context_tokens"], 155_136)

    def test_trained_context_and_engine_cap_bound_a_large_machine(self):
        result = ask(snapshot(64 * GIB))
        self.assertEqual((result["max_context_tokens"], result["limit_reason"]), (262_144, "trained_context"))
        self.assertIsNone(result["limit_resource"])
        capped = ask(snapshot(64 * GIB), profiles=[profile(engine_cap_tokens=2_048)])
        self.assertEqual((capped["max_context_tokens"], capped["max_input_tokens"], capped["limit_reason"]),
                         (2_048, 1_871, "engine_cap"))

    def test_no_room_gives_zero_and_template_override(self):
        small = ask(snapshot(2 * GIB))
        self.assertEqual(small["max_context_tokens"], 1_536)
        self.assertEqual(small["max_input_tokens"], 1_359)
        self.assertEqual(ask(snapshot(2 * GIB), template_tokens=200)["max_input_tokens"], 1_335)
        none = ask(snapshot(GIB + 500_000_000))
        self.assertEqual((none["max_context_tokens"], none["max_input_tokens"]), (0, 0))

    def test_gpu_backend_is_bounded_by_free_vram_and_carries_the_adapter_there(self):
        gpus = [{"index": 0, "backend": "cuda", "total_bytes": 8192 * MIB,
                 "available_bytes": 3000 * MIB, "evidence": "nvidia-smi"}]
        result = ask(snapshot(16 * GIB, gpus), profiles=[gpu_profile()], backend="cuda")
        # (3000 MiB - 1 GiB - 1_176_000_000) // 13_978 = 64_099 -> 64_000.
        self.assertEqual(result["breakdown"]["vram"]["max_context_tokens"], 64_000)
        self.assertEqual((result["max_context_tokens"], result["limit_resource"]), (64_000, "vram"))
        bigger = ask(snapshot(16 * GIB, gpus), profiles=[gpu_profile()], backend="cuda",
                     adapter_bytes=31_671_008)
        self.assertEqual(bigger["breakdown"]["vram"]["fixed_bytes"], 1_186_000_000)
        self.assertEqual(bigger["breakdown"]["ram"]["fixed_bytes"], 1_260_000_000)


class RefusalTests(unittest.TestCase):
    def test_model_digest_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "refused: model sha256 000000000000 does not match "
                                                "the profiled 37ae482d3361"):
            ask(snapshot(4 * GIB), model_sha256="0" * 64)

    def test_engine_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "refused: engine llama.cpp@7be2c65 is not profiled"):
            ask(snapshot(4 * GIB), engine="llama.cpp@7be2c65")

    def test_unknown_model_backend_or_architecture_has_no_profile(self):
        with self.assertRaisesRegex(ValueError, "no binary-judge profile for model lfm2"):
            ask(snapshot(4 * GIB), model="lfm2")
        with self.assertRaisesRegex(ValueError, "on backend vulkan and architecture x86_64"):
            ask(snapshot(4 * GIB), backend="vulkan")
        arm = snapshot(4 * GIB); arm["architecture"] = "aarch64"
        with self.assertRaisesRegex(ValueError, "architecture aarch64"):
            ask(arm)

    def test_a_tampered_profile_is_refused(self):
        changed = profile()
        changed["ram"]["per_token_bytes"] = 1_000
        with self.assertRaisesRegex(ValueError, "digest does not match its content"):
            binary_judge.load_profiles({"schema": binary_judge.PROFILES_SCHEMA, "profiles": [changed]})

    def test_unknown_available_memory_gives_no_limit(self):
        with self.assertRaisesRegex(ValueError, "available RAM is unknown"):
            ask(snapshot(4 * GIB, cgroup="unknown"))


class BundledProfileTests(unittest.TestCase):
    def test_bundled_profiles_verify_and_cover_the_judge_base(self):
        profiles = binary_judge.load_profiles()
        keys = {(p["model"]["id"], p["engine"]["backend"], p["architecture"]) for p in profiles}
        self.assertIn(("qwen3.5-0.8b-q8_0", "cpu", "x86_64"), keys)
        self.assertIn(("qwen3.5-0.8b-q8_0", "cpu", "aarch64"), keys)
        for p in profiles:
            if p["model"]["id"] == "qwen3.5-0.8b-q8_0":
                self.assertEqual(p["model"]["sha256"], QWEN_SHA)
                self.assertEqual(p["model"]["n_ctx_train"], 262_144)


class CliTests(unittest.TestCase):
    def run_cli(self, *extra: str) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resources.json"
            path.write_text(json.dumps(snapshot(4 * GIB)))
            output = StringIO()
            with mock.patch.object(binary_judge, "load_profiles", return_value=[profile()]), \
                    redirect_stdout(output):
                code = main(["recommend", "binary-judge", "--model", "qwen3.5-0.8b-q8_0",
                             "--engine", ENGINE, "--resources", str(path), "--json", *extra])
        return code, json.loads(output.getvalue())

    def test_json_report(self):
        code, result = self.run_cli("--model-sha256", QWEN_SHA, "--adapter", "qwen-0.8b-binary-01")
        self.assertEqual(code, 0)
        self.assertEqual(result["schema"], "plebian.models.binary-judge-sizing/v1-development")
        self.assertEqual((result["max_input_tokens"], result["limit_reason"]), (154_959, "memory"))
        self.assertEqual((result["resource_source"], result["adapter"]), ("provided", "qwen-0.8b-binary-01"))

    def test_refusal_is_a_structured_error(self):
        code, result = self.run_cli("--model-sha256", "f" * 64)
        self.assertEqual(code, 2)
        self.assertIn("refused: model sha256", result["error"])


if __name__ == "__main__":
    unittest.main()
