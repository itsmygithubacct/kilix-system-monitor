"""Transparent planning arithmetic; all values are estimates, not measurements."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re

from .resources import GIB, budgets, validate
from . import __version__

RECIPE = "lora-all-linear-checkpointed-v1"


def integer(value, name: str, maximum: int, minimum: int = 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


@dataclass(frozen=True)
class Workload:
    task: str = "both"
    phase: str = "both"
    context: int = 2048
    batch: int = 1
    lora_rank: int = 16
    topics: int = 128
    quant: str = "q4"
    train_backend: str = "auto"
    infer_backend: str = "auto"
    gpu: int | None = None
    co_resident: bool = False
    checkpointing: bool = True
    document_bytes: int = 0

    def __post_init__(self):
        for name, choices in {"task": {"answer", "rank", "both"},
                              "phase": {"train", "infer", "both"},
                              "quant": {"q4", "q8", "f16"},
                              "train_backend": {"auto", "cpu", "cuda"},
                              "infer_backend": {"auto", "cpu", "cuda"}}.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"invalid {name}")
        for name, limit in {"context": 1048576, "batch": 256, "lora_rank": 1024,
                            "topics": 65536}.items():
            integer(getattr(self, name), name, limit)
        integer(self.document_bytes, "document_bytes", 2**60, 0)
        if self.gpu is not None:
            integer(self.gpu, "gpu", 255, 0)
        if type(self.co_resident) is not bool or type(self.checkpointing) is not bool:
            raise ValueError("co_resident and checkpointing must be booleans")


def geometry(checkpoint: dict) -> dict:
    """Validate the architecture inputs, including their exact config identity."""
    g = checkpoint.get("sizing")
    if not isinstance(g, dict):
        raise ValueError("missing sizing metadata")
    if g.get("config_sha256") != checkpoint.get("config", {}).get("sha256"):
        raise ValueError("sizing metadata is not bound to the checkpoint config")
    if re.fullmatch(r"[0-9a-f]{64}", str(g.get("config_sha256"))) is None:
        raise ValueError("invalid config digest")
    if not isinstance(g.get("model_type"), str) or g["model_type"] not in {"qwen3", "qwen3_5_text", "llama", "smollm3"}:
        raise ValueError("unsupported architecture")
    for name, limit in {"parameters": 10**12, "hidden_size": 65536,
                        "intermediate_size": 262144, "num_hidden_layers": 256,
                        "num_attention_heads": 1024, "num_key_value_heads": 1024,
                        "head_dim": 4096, "vocab_size": 1048576,
                        "max_position_embeddings": 1048576}.items():
        integer(g.get(name), name, limit)
    if g["num_key_value_heads"] > g["num_attention_heads"]:
        raise ValueError("key/value heads exceed attention heads")
    layers = g.get("layer_types")
    if not isinstance(layers, list) or len(layers) != g["num_hidden_layers"]:
        raise ValueError("layer types do not match layer count")
    if any(not isinstance(layer, str) or layer not in {"full_attention", "linear_attention"} for layer in layers):
        raise ValueError("unsupported layer type")
    if "linear_attention" in layers:
        if g["model_type"] != "qwen3_5_text":
            raise ValueError("unsupported recurrent architecture")
        for name in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
                     "linear_value_head_dim", "linear_conv_kernel_dim"):
            integer(g.get(name), name, 4096)
    if type(g.get("attn_output_gate", False)) is not bool:
        raise ValueError("invalid attention gate flag")
    integer(checkpoint.get("checkpoint_bytes"), "checkpoint_bytes", 2**50)
    return g


def validate_catalog(catalog: dict) -> None:
    if not isinstance(catalog, dict) or catalog.get("schema") != "kilix.help-llm.candidates/v1":
        raise ValueError("expected a kilix-help-llm candidate catalog")
    candidates = catalog.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 256:
        raise ValueError("catalog must contain 1..256 candidates")
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid candidate")
        name = candidate.get("id")
        if not isinstance(name, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", name) is None or name in seen:
            raise ValueError("invalid or duplicate candidate ID")
        seen.add(name)
        for key in ("generation_checkpoint", "decision_checkpoint"):
            checkpoint = candidate.get(key)
            if not isinstance(checkpoint, dict):
                raise ValueError("candidate requires both checkpoint identities")
            if not isinstance(checkpoint.get("model_id"), str) or re.fullmatch(
                    r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", checkpoint["model_id"]) is None:
                raise ValueError("invalid model ID")
            if re.fullmatch(r"[0-9a-f]{40}", str(checkpoint.get("revision"))) is None:
                raise ValueError("checkpoint revision must be pinned")
            if not isinstance(checkpoint.get("config"), dict):
                raise ValueError("checkpoint config identity must be an object")
            files = checkpoint.get("checkpoint_files")
            if not isinstance(files, list) or not 1 <= len(files) <= 256:
                raise ValueError("missing checkpoint file identities")
            paths = set()
            total = 0
            for file in files:
                if not isinstance(file, dict):
                    raise ValueError("invalid checkpoint file")
                path = file.get("path")
                if not isinstance(path, str) or re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", path) is None or path in paths:
                    raise ValueError("invalid or duplicate checkpoint file")
                paths.add(path)
                total += integer(file.get("bytes"), "checkpoint file bytes", 2**50)
                if re.fullmatch(r"[0-9a-f]{64}", str(file.get("sha256"))) is None:
                    raise ValueError("invalid checkpoint digest")
            if type(checkpoint.get("checkpoint_bytes")) is not int or total != checkpoint["checkpoint_bytes"]:
                raise ValueError("checkpoint byte total does not match its files")


def parameter_shapes(g: dict, rank: int) -> tuple[int, int, int]:
    """LoRA matrix sizes plus recurrent state and convolution cache elements."""
    h, inter = g["hidden_size"], g["intermediate_size"]
    full = g["layer_types"].count("full_attention")
    recurrent = g["num_hidden_layers"] - full
    q, kv = g["num_attention_heads"] * g["head_dim"], g["num_key_value_heads"] * g["head_dim"]
    targets = full * (4 * h + (3 if g.get("attn_output_gate") else 2) * q + 2 * kv)
    targets += g["num_hidden_layers"] * 3 * (h + inter)
    state = conv = 0
    if recurrent:
        keys = g["linear_num_key_heads"] * g["linear_key_head_dim"]
        values = g["linear_num_value_heads"] * g["linear_value_head_dim"]
        targets += recurrent * (5 * h + 2 * keys + 3 * values + 2 * g["linear_num_value_heads"])
        state = g["linear_num_value_heads"] * g["linear_key_head_dim"] * g["linear_value_head_dim"]
        conv = (2 * keys + values) * g["linear_conv_kernel_dim"]
    return targets * rank, state, conv


def estimate(checkpoint: dict, task: str, phase: str, backend: str, workload: Workload) -> dict:
    g = geometry(checkpoint)
    if workload.context > g["max_position_embeddings"]:
        raise ValueError("context exceeds the checkpoint's configured maximum")
    h, inter, layers = g["hidden_size"], g["intermediate_size"], g["num_hidden_layers"]
    seq, batch = workload.context, workload.batch
    full = g["layer_types"].count("full_attention")
    recurrent = layers - full
    lora, state, conv = parameter_shapes(g, workload.lora_rank)
    # The planned ranker uses a dense topic-scoring head. Vocabulary output
    # remains frozen; count all original weights until a text-only export exists.
    head = (h + 1) * workload.topics if task == "rank" else 0
    trainable = lora + head
    width = 4 if backend == "cpu" else 2
    parts = {}
    if phase == "train":
        parts["frozen_weights"] = max(g["parameters"] * width, checkpoint["checkpoint_bytes"])
        # FP32 adapter weights/master, gradients, and two FP32 Adam moments.
        parts["adapter_and_optimizer"] = trainable * 16
        live = 1 if workload.checkpointing else layers
        parts["saved_activations"] = batch * seq * h * layers * width
        parts["layer_workspace"] = batch * seq * (12 * h + 6 * inter) * width * live
        parts["attention_workspace"] = batch * g["num_attention_heads"] * seq**2 * 8 * (1 if workload.checkpointing else full)
        parts["recurrent_workspace"] = batch * seq * state * 4 * (1 if workload.checkpointing else recurrent)
        parts["output_and_loss"] = batch * seq * (g["vocab_size"] if task == "answer" else workload.topics) * 8
    else:
        if task == "answer":
            # Planning budgets, not GGUF file sizes. Keep both embedding/output
            # matrices at FP16 even if a future artifact quantizes/ties them.
            embeddings = min(g["parameters"], 2 * h * g["vocab_size"])
            numerator, denominator = {"q4": (3, 4), "q8": (9, 8), "f16": (2, 1)}[workload.quant]
            parts["weights"] = embeddings * 2 + ((g["parameters"] - embeddings) * numerator + denominator - 1) // denominator
            kv_width = 2
        else:
            parts["weights"] = max(g["parameters"] * width, checkpoint["checkpoint_bytes"])
            kv_width = width
        parts["adapter_and_head"] = trainable * 4
        parts["kv_cache"] = 2 * full * batch * seq * g["num_key_value_heads"] * g["head_dim"] * kv_width
        parts["recurrent_cache"] = recurrent * batch * (state * 4 + conv * width)
        # Full-sequence scoring cannot assume the generation runtime's chunked
        # prefill. Generation estimates assume prefill chunks <= 512 tokens.
        chunk = min(seq, 512) if task == "answer" else seq
        parts["prefill_workspace"] = batch * chunk * (12 * h + 6 * inter) * width
        parts["attention_workspace"] = batch * g["num_attention_heads"] * chunk**2 * 8
        parts["output"] = batch * chunk * (g["vocab_size"] if task == "answer" else workload.topics) * 4
    parts["runtime_allowance"] = GIB // 2
    parts["uncertainty_margin"] = (sum(parts.values()) + 3) // 4
    device_peak = sum(parts.values())
    # CUDA staging: one complete unquantized FP32 checkpoint plus host workspace.
    # CPU load/export: an additional serialized checkpoint alongside live tensors.
    host_staging = max(checkpoint["checkpoint_bytes"], g["parameters"] * 4) + GIB
    ram = host_staging if backend == "cuda" else device_peak + checkpoint["checkpoint_bytes"]
    # Cold storage: source + full merge/export scratch + 3 adapter/optimizer
    # checkpoints. Document preprocessing gets a separately declared 4x budget.
    disk = 3 * checkpoint["checkpoint_bytes"] + trainable * 16 * 3
    return {"recipe": RECIPE, "task": task, "phase": phase, "backend": backend,
            "model_id": checkpoint["model_id"], "revision": checkpoint["revision"],
            "config_sha256": checkpoint["config"]["sha256"],
            "parameters": g["parameters"], "trainable_parameters": trainable,
            "breakdown_bytes": parts, "ram_peak_bytes": ram,
            "vram_peak_bytes": device_peak if backend == "cuda" else 0,
            "disk_bytes": disk, "evidence": "estimated", "runtime_compatibility": "unverified"}


def assess(requirements: dict, budget: dict) -> dict:
    checks = {}
    for resource, required in requirements.items():
        available = budget[resource + "_bytes"]
        checks[resource] = {"required_bytes": required, "budget_bytes": available,
                            "status": "unknown" if available is None else
                            ("estimated-fit" if required <= available else "does-not-fit")}
    statuses = {check["status"] for check in checks.values()}
    verdict = "does-not-fit" if "does-not-fit" in statuses else "unknown" if "unknown" in statuses else "estimated-fit"
    return {"verdict": verdict, "resources": checks}


def recommend(catalog: dict, snapshot: dict, workload: Workload, *, source: str = "live") -> dict:
    validate_catalog(catalog)
    validate(snapshot)
    tasks = ["answer", "rank"] if workload.task == "both" else [workload.task]
    phases = ["train", "infer"] if workload.phase == "both" else [workload.phase]
    phase_budgets = {phase: budgets(snapshot, getattr(workload, phase + "_backend"), workload.gpu) for phase in phases}
    rows = []
    for candidate in catalog["candidates"]:
        profiles, errors = [], []
        for task in tasks:
            checkpoint = candidate["generation_checkpoint" if task == "answer" else "decision_checkpoint"]
            for phase in phases:
                try:
                    profiles.append(estimate(checkpoint, task, phase, phase_budgets[phase]["backend"], workload))
                except ValueError as error:
                    errors.append(f"{task}/{phase}: {error}")
        checks = {}
        if not errors:
            # Both tasks retain distinct checkpoint/adapters even when a family
            # shares an upstream revision. Count disk once per task, not phase.
            disk = sum(max(p["disk_bytes"] for p in profiles if p["task"] == task) for task in tasks)
            disk += 4 * workload.document_bytes
            for phase in phases:
                selected = [p for p in profiles if p["phase"] == phase]
                combine = sum if phase == "infer" and workload.co_resident else max
                requirements = {"ram": combine(p["ram_peak_bytes"] for p in selected), "disk": disk}
                if phase_budgets[phase]["backend"] == "cuda":
                    requirements["vram"] = combine(p["vram_peak_bytes"] for p in selected)
                checks[phase] = assess(requirements, phase_budgets[phase])
        verdicts = {check["verdict"] for check in checks.values()}
        verdict = "unknown" if errors else "does-not-fit" if "does-not-fit" in verdicts else "unknown" if "unknown" in verdicts else "estimated-fit"
        rows.append({"id": candidate["id"], "verdict": verdict, "errors": errors,
                     "profiles": profiles, "checks": checks,
                     "quality": "unmeasured", "qualification_eligible": False})
    fitting = [row for row in rows if row["verdict"] == "estimated-fit"]
    fitting.sort(key=lambda row: (max(p["parameters"] for p in row["profiles"]), row["id"]))
    digest = hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"schema": "plebian.models.llm-sizing/v1-development", "provider_version": __version__,
            "catalog_sha256": digest, "resource_source": source,
            "observed_at": snapshot["observed_at"], "workload": asdict(workload),
            "budgets": phase_budgets, "candidates": rows,
            "shortlist": [row["id"] for row in fitting],
            "provisional_candidate": fitting[0]["id"] if fitting else None,
            "selected_model": None, "qualification_eligible": False,
            "notes": ["Resource estimates only; quality and runtime compatibility require measurement.",
                      "No download, reservation, training, installation or execution is authorized by this result.",
                      "Train tasks run sequentially; inference residency follows co_resident.",
                      "CUDA indices are physical nvidia-smi indices; runtime visibility/remapping is unverified.",
                      "Data preprocessing allowance is four times document_bytes; caches must use the assessed filesystem."]}
