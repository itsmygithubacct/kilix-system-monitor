from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from plebian_model_sizer.estimate import Workload, assess, estimate, recommend, validate_catalog
from plebian_model_sizer.resources import GIB, SCHEMA, budgets, validate


def checkpoint():
    return {"model_id": "Example/Small", "revision": "a" * 40,
            "config": {"sha256": "b" * 64}, "checkpoint_bytes": 2 * GIB,
            "checkpoint_files": [{"path": "model.safetensors", "bytes": 2 * GIB, "sha256": "c" * 64}],
            "sizing": {"config_sha256": "b" * 64, "model_type": "qwen3", "parameters": 10**9,
                       "hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 24,
                       "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 128,
                       "vocab_size": 50000, "max_position_embeddings": 32768,
                       "layer_types": ["full_attention"] * 24}}


def catalog():
    return {"schema": "kilix.help-llm.candidates/v1", "candidates": [
        {"id": "small", "generation_checkpoint": checkpoint(), "decision_checkpoint": checkpoint()}]}


def snapshot():
    return {"schema": SCHEMA, "observed_at": datetime.now(timezone.utc).isoformat(),
            "ram_total_bytes": 128 * GIB, "ram_available_bytes": 120 * GIB,
            "cgroup_status": "limited", "disk_available_bytes": 100 * GIB,
            "gpus": [{"index": 0, "backend": "cuda", "total_bytes": 24 * GIB,
                      "available_bytes": 22 * GIB}]}


class SizingTests(unittest.TestCase):
    def test_inference_fit_cannot_admit_training(self):
        resource = snapshot()
        resource["gpus"][0]["available_bytes"] = 3 * GIB
        report = recommend(catalog(), resource, Workload(task="answer"))
        row = report["candidates"][0]
        self.assertEqual(row["checks"]["infer"]["verdict"], "estimated-fit")
        self.assertEqual(row["checks"]["train"]["verdict"], "does-not-fit")
        self.assertIsNone(report["provisional_candidate"])
        inference = recommend(catalog(), resource, Workload(task="answer", phase="infer"))
        self.assertEqual(inference["provisional_candidate"], "small")
        self.assertIsNone(inference["selected_model"])
        self.assertFalse(inference["qualification_eligible"])

    def test_full_budget_fits_but_never_qualifies(self):
        report = recommend(catalog(), snapshot(), Workload())
        self.assertEqual(report["shortlist"], ["small"])
        self.assertEqual(len(report["candidates"][0]["profiles"]), 4)
        self.assertFalse(report["candidates"][0]["qualification_eligible"])
        self.assertEqual(report["candidates"][0]["quality"], "unmeasured")

    def test_smaller_candidate_chosen_regardless_of_catalog_order(self):
        doc = catalog()
        bigger = deepcopy(doc["candidates"][0]); bigger["id"] = "bigger"
        for key in ("generation_checkpoint", "decision_checkpoint"):
            bigger[key]["sizing"]["parameters"] *= 2
        doc["candidates"].insert(0, bigger)
        report = recommend(doc, snapshot(), Workload())
        self.assertEqual(report["provisional_candidate"], "small")

    def test_context_and_batch_increase_memory(self):
        for phase in ("train", "infer"):
            one = estimate(checkpoint(), "answer", phase, "cuda", Workload(context=512))
            two = estimate(checkpoint(), "answer", phase, "cuda", Workload(context=1024))
            batch = estimate(checkpoint(), "answer", phase, "cuda", Workload(context=1024, batch=2))
            self.assertLess(one["vram_peak_bytes"], two["vram_peak_bytes"])
            self.assertLess(two["vram_peak_bytes"], batch["vram_peak_bytes"])

    def test_cache_accounts_for_gqa_not_hidden_width(self):
        profile = estimate(checkpoint(), "answer", "infer", "cuda", Workload())
        self.assertEqual(profile["breakdown_bytes"]["kv_cache"], 2 * 24 * 2048 * 4 * 128 * 2)

    def test_quantization_affects_only_answer_inference(self):
        for phase, task in (("train", "answer"), ("train", "rank"), ("infer", "rank")):
            self.assertEqual(estimate(checkpoint(), task, phase, "cuda", Workload(quant="q4")),
                             estimate(checkpoint(), task, phase, "cuda", Workload(quant="f16")))
        q4 = estimate(checkpoint(), "answer", "infer", "cuda", Workload(quant="q4"))
        f16 = estimate(checkpoint(), "answer", "infer", "cuda", Workload(quant="f16"))
        self.assertLess(q4["vram_peak_bytes"], f16["vram_peak_bytes"])

    def test_adapter_rank_and_checkpointing_change_training_budget(self):
        baseline = estimate(checkpoint(), "answer", "train", "cuda", Workload())
        larger = estimate(checkpoint(), "answer", "train", "cuda", Workload(lora_rank=64))
        unchecked = estimate(checkpoint(), "answer", "train", "cuda", Workload(checkpointing=False))
        self.assertEqual(larger["trainable_parameters"], 4 * baseline["trainable_parameters"])
        self.assertGreater(unchecked["vram_peak_bytes"], baseline["vram_peak_bytes"])
        self.assertEqual(baseline["breakdown_bytes"]["adapter_and_optimizer"], baseline["trainable_parameters"] * 16)

    def test_hybrid_counts_only_full_attention_kv_and_adds_recurrence(self):
        ck = checkpoint()
        ck["sizing"].update(model_type="qwen3_5_text", layer_types=["linear_attention"] * 18 + ["full_attention"] * 6,
                            linear_num_key_heads=16, linear_num_value_heads=16, linear_key_head_dim=128,
                            linear_value_head_dim=128, linear_conv_kernel_dim=4, attn_output_gate=True)
        profile = estimate(ck, "answer", "infer", "cuda", Workload())
        self.assertEqual(profile["breakdown_bytes"]["kv_cache"], 2 * 6 * 2048 * 4 * 128 * 2)
        self.assertGreater(profile["breakdown_bytes"]["recurrent_cache"], 0)
        profile = estimate(ck, "answer", "train", "cuda", Workload())
        self.assertGreater(profile["breakdown_bytes"]["recurrent_workspace"], 0)

    def test_co_residency_adds_inference_but_training_stays_sequential(self):
        a = recommend(catalog(), snapshot(), Workload())["candidates"][0]
        b = recommend(catalog(), snapshot(), Workload(co_resident=True))["candidates"][0]
        self.assertEqual(a["checks"]["train"], b["checks"]["train"])
        self.assertGreater(b["checks"]["infer"]["resources"]["vram"]["required_bytes"],
                           a["checks"]["infer"]["resources"]["vram"]["required_bytes"])

    def test_disk_counted_per_task_not_phase_and_includes_documents(self):
        one = recommend(catalog(), snapshot(), Workload(phase="train"))["candidates"][0]
        both = recommend(catalog(), snapshot(), Workload(document_bytes=GIB))["candidates"][0]
        self.assertEqual(both["checks"]["train"]["resources"]["disk"]["required_bytes"],
                         one["checks"]["train"]["resources"]["disk"]["required_bytes"] + 4 * GIB)

    def test_disk_or_host_ram_failure_refuses_gpu_fit(self):
        for key in ("disk_available_bytes", "ram_available_bytes"):
            data = snapshot(); data[key] = GIB
            self.assertEqual(recommend(catalog(), data, Workload())["shortlist"], [])

    def test_unknown_geometry_and_context_never_fit(self):
        for mutation in (lambda g: g.pop("head_dim"), lambda g: g.update(model_type="unknown"),
                         lambda g: g.update(config_sha256="d" * 64), lambda g: g.update(layer_types=[{}] * 24)):
            doc = catalog(); mutation(doc["candidates"][0]["generation_checkpoint"]["sizing"])
            result = recommend(doc, snapshot(), Workload())
            self.assertEqual(result["candidates"][0]["verdict"], "unknown")
        result = recommend(catalog(), snapshot(), Workload(context=65536))
        self.assertEqual(result["shortlist"], [])

    def test_catalog_tampering_rejected(self):
        for mutation in (lambda c: c.update(checkpoint_bytes=1), lambda c: c.update(revision="main"),
                         lambda c: c.update(config=[]), lambda c: c["checkpoint_files"][0].update(bytes=True)):
            doc = catalog(); mutation(doc["candidates"][0]["generation_checkpoint"])
            with self.assertRaises(ValueError): validate_catalog(doc)

    def test_invalid_workloads_rejected(self):
        for values in ({"context": 0}, {"batch": True}, {"lora_rank": -1}, {"topics": 100000},
                       {"quant": "nf4"}, {"gpu": -1}, {"document_bytes": -1}, {"co_resident": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError): Workload(**values)


class BudgetTests(unittest.TestCase):
    def test_boundary_equality_and_zero(self):
        self.assertEqual(assess({"ram": 1}, {"ram_bytes": 1})["verdict"], "estimated-fit")
        self.assertEqual(assess({"ram": 1}, {"ram_bytes": 0})["verdict"], "does-not-fit")
        self.assertEqual(assess({"ram": 1}, {"ram_bytes": None})["verdict"], "unknown")

    def test_free_not_total_and_no_gpu_aggregation(self):
        data = snapshot(); data["gpus"][0]["available_bytes"] = GIB
        data["gpus"].append({**data["gpus"][0], "index": 1})
        self.assertEqual(budgets(data, "cuda")["vram_bytes"], GIB // 2)
        self.assertIsNone(budgets(data, "cuda", gpu=2)["vram_bytes"])
        self.assertEqual(budgets(data, "auto", gpu=2)["backend"], "cuda")
        self.assertIsNone(budgets(data, "auto", gpu=2)["vram_bytes"])

    def test_missing_gpu_cpu_fallback_and_explicit_cuda_unknown(self):
        data = snapshot(); data["gpus"] = []
        self.assertEqual(budgets(data, "auto")["backend"], "cpu")
        self.assertIsNone(budgets(data, "cuda")["vram_bytes"])

    def test_stale_future_and_unknown_cgroup_fail_closed(self):
        for offset in (-301, 60):
            data = snapshot(); data["observed_at"] = (datetime.now(timezone.utc) + timedelta(seconds=offset)).isoformat()
            self.assertIsNone(budgets(data, "cpu")["ram_bytes"])
            self.assertEqual(recommend(catalog(), data, Workload())["shortlist"], [])
        data = snapshot(); data["cgroup_status"] = "unknown"
        self.assertIsNone(budgets(data, "cpu")["ram_bytes"])

    def test_invalid_snapshot_rejected(self):
        for key, value in (("ram_available_bytes", True), ("observed_at", "today"), ("cgroup_status", []),
                           ("ram_available_bytes", 256 * GIB), ("gpus", None)):
            data = snapshot(); data[key] = value
            with self.assertRaises(ValueError): validate(data)


if __name__ == "__main__":
    unittest.main()
