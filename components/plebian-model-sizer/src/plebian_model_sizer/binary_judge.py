"""Longest input a one-token judge can take on this machine, from measured profiles.

A profile is the measured memory of one engine build loading one exact base
GGUF plus a rank-8 stand-in LoRA, fitted as `fixed + per_token * context`,
on one backend and CPU architecture. It is keyed by the model sha256 and the
engine identity; a request whose digests differ is refused, never estimated.

    budget      = live MemAvailable (or free VRAM for a GPU backend) - reserve
    fixed       = profile fixed * (1 + margin) + (adapter bytes over the stand-in's)
    per_token   = profile per_token * (1 + margin)
    max_ctx     = floor((budget - fixed) / per_token), down to a multiple of 256
    max_context = min(max_ctx, n_ctx_train, engine cap)
    max_input   = max_context - template tokens - 1 output token
"""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import re

from . import __version__
from .resources import MIB, budgets, validate

RESPONSE_SCHEMA = "plebian.models.binary-judge-sizing/v1-development"
PROFILES_SCHEMA = "plebian.models.binary-judge-profiles/v1-development"
BACKENDS = ("cpu", "cuda", "vulkan")
# llama.cpp pads the context it allocates to a multiple of 256.
CONTEXT_STEP = 256
DEFAULT_RESERVE_MB = 1024


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def profile_digest(profile: dict) -> str:
    """sha256 of the profile without its own `profile_sha256` field."""
    body = {key: value for key, value in profile.items() if key != "profile_sha256"}
    return hashlib.sha256(canonical(body)).hexdigest()


def _count(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**62:
        raise ValueError(f"invalid binary-judge profile: {name}")
    return value


def check_profile(profile: dict) -> dict:
    """Shape and integrity of one profile; any doubt refuses it."""
    if not isinstance(profile, dict) or profile.get("profile_sha256") != profile_digest(profile):
        raise ValueError("invalid binary-judge profile: digest does not match its content")
    model, engine = profile.get("model"), profile.get("engine")
    if not isinstance(model, dict) or not isinstance(engine, dict):
        raise ValueError("invalid binary-judge profile: model and engine are required")
    if re.fullmatch(r"[0-9a-f]{64}", str(model.get("sha256"))) is None:
        raise ValueError("invalid binary-judge profile: model sha256")
    _count(model.get("weights_bytes"), "weights_bytes", 1)
    _count(model.get("n_ctx_train"), "n_ctx_train", 1)
    if engine.get("backend") not in BACKENDS or not isinstance(engine.get("id"), str):
        raise ValueError("invalid binary-judge profile: engine")
    if profile.get("engine_cap_tokens") is not None:
        _count(profile["engine_cap_tokens"], "engine_cap_tokens", 1)
    _count(profile.get("template_tokens"), "template_tokens")
    _count(profile.get("standin_adapter", {}).get("bytes"), "standin adapter bytes")
    _count(profile.get("margin_basis_points"), "margin_basis_points")
    for resource in ("ram", "vram"):
        part = profile.get(resource)
        if part is None and resource == "vram" and engine["backend"] == "cpu":
            continue
        if not isinstance(part, dict):
            raise ValueError(f"invalid binary-judge profile: {resource}")
        _count(part.get("fixed_bytes"), f"{resource} fixed_bytes")
        _count(part.get("per_token_bytes"), f"{resource} per_token_bytes")
    ratio = profile.get("bytes_per_token")
    if ratio is not None and (not isinstance(ratio, (int, float)) or not 0 < ratio < 64):
        raise ValueError("invalid binary-judge profile: bytes_per_token")
    return profile


def load_profiles(document: dict | None = None) -> list[dict]:
    if document is None:
        document = json.loads(files(__package__).joinpath("binary_judge_profiles.json").read_bytes())
    if not isinstance(document, dict) or document.get("schema") != PROFILES_SCHEMA:
        raise ValueError("invalid binary-judge profile set")
    profiles = [check_profile(profile) for profile in document.get("profiles", [])]
    ids = [profile["id"] for profile in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate binary-judge profile")
    return profiles


def select(profiles: list[dict], model: str, model_sha256: str, engine: str,
           backend: str, architecture: str | None) -> dict:
    """The one profile for exactly this model, engine, backend and architecture."""
    named = [p for p in profiles if p["model"]["id"] == model]
    if not named:
        raise ValueError(f"no binary-judge profile for model {model}")
    same_model = [p for p in named if p["model"]["sha256"] == model_sha256]
    if not same_model:
        raise ValueError(f"binary-judge profile refused: model sha256 {model_sha256[:12]} does not "
                         f"match the profiled {named[0]['model']['sha256'][:12]} for {model}")
    same_engine = [p for p in same_model if p["engine"]["id"] == engine]
    if not same_engine:
        known = ", ".join(sorted({p["engine"]["id"] for p in same_model}))
        raise ValueError(f"binary-judge profile refused: engine {engine} is not profiled "
                         f"for {model} (profiled: {known})")
    exact = [p for p in same_engine if p["engine"]["backend"] == backend
             and p["architecture"] == architecture]
    if not exact:
        raise ValueError(f"no binary-judge profile for {model} on backend {backend} and "
                         f"architecture {architecture or 'unknown'}")
    return exact[0]


def _with_margin(value: int, basis_points: int) -> int:
    return (value * (10000 + basis_points) + 9999) // 10000


def size(profile: dict, budget: dict, adapter_bytes: int, template_tokens: int) -> dict:
    """The arithmetic, separate from I/O so it can be checked with literal numbers."""
    margin = profile["margin_basis_points"]
    extra_adapter = max(0, adapter_bytes - profile["standin_adapter"]["bytes"])
    resources = ["ram"] + (["vram"] if profile["engine"]["backend"] != "cpu" else [])
    adapter_home = resources[-1]  # the adapter sits with the weights: VRAM on a GPU backend
    limits, breakdown = {}, {}
    for resource in resources:
        available = budget[f"{resource}_bytes"]
        if available is None:
            raise ValueError(f"available {resource.upper()} is unknown, so no input limit can be given")
        fixed = _with_margin(profile[resource]["fixed_bytes"], margin)
        if resource == adapter_home:
            fixed += extra_adapter
        per_token = _with_margin(profile[resource]["per_token_bytes"], margin)
        room = available - fixed
        tokens = room // per_token if per_token else 2**62
        limits[resource] = max(0, tokens // CONTEXT_STEP * CONTEXT_STEP) if room > 0 else 0
        breakdown[resource] = {"budget_bytes": available, "fixed_bytes": fixed,
                               "per_token_bytes": per_token, "max_context_tokens": limits[resource]}
    memory_resource = min(limits, key=lambda key: limits[key])
    memory_ctx = limits[memory_resource]
    n_ctx_train = profile["model"]["n_ctx_train"]
    cap = profile.get("engine_cap_tokens")
    candidates = [(memory_ctx, "memory"), (n_ctx_train, "trained_context")]
    if cap is not None:
        candidates.append((cap, "engine_cap"))
    max_context, reason = min(candidates, key=lambda pair: pair[0])
    max_input = max(0, max_context - template_tokens - 1)
    ratio = profile.get("bytes_per_token")
    return {
        "max_context_tokens": max_context,
        "max_input_tokens": max_input,
        "max_input_bytes_estimate": int(max_input * ratio) if ratio else None,
        "limit_reason": reason,
        "limit_resource": memory_resource if reason == "memory" else None,
        "breakdown": {**breakdown, "adapter_bytes": adapter_bytes,
                      "adapter_bytes_over_standin": extra_adapter, "margin_basis_points": margin,
                      "n_ctx_train": n_ctx_train, "engine_cap_tokens": cap,
                      "template_tokens": template_tokens, "output_tokens": 1},
    }


def recommend_binary_judge(snapshot: dict, *, model: str, model_sha256: str, engine: str,
                           backend: str = "cpu", adapter_bytes: int = 0, adapter: str | None = None,
                           reserve_mb: int = DEFAULT_RESERVE_MB, template_tokens: int | None = None,
                           source: str = "live", profiles: list[dict] | None = None) -> dict:
    validate(snapshot)
    if backend not in BACKENDS:
        raise ValueError("unsupported backend")
    if re.fullmatch(r"[0-9a-f]{64}", model_sha256 or "") is None:
        raise ValueError("--model-sha256 must be 64 lowercase hex digits")
    for name, value in (("adapter bytes", adapter_bytes), ("reserve", reserve_mb)):
        if type(value) is not int or not 0 <= value <= 2**40:
            raise ValueError(f"invalid {name}")
    profile = select(profiles if profiles is not None else load_profiles(), model, model_sha256,
                     engine, backend, snapshot.get("architecture"))
    if template_tokens is None:
        template_tokens = profile["template_tokens"]
    elif type(template_tokens) is not int or not 0 <= template_tokens <= 2**20:
        raise ValueError("invalid template tokens")
    reserve = reserve_mb * MIB
    # Vulkan's free VRAM is read like CUDA's (nvidia-smi); other devices are unknown.
    budget = budgets(snapshot, "cpu" if backend == "cpu" else "cuda",
                     ram_reserve=reserve, vram_reserve=reserve)
    if not budget["fresh"]:
        raise ValueError("resource snapshot is not fresh")
    result = size(profile, budget, adapter_bytes, template_tokens)
    return {"schema": RESPONSE_SCHEMA, "provider_version": __version__,
            "profile_id": profile["id"], "profile_sha256": profile["profile_sha256"],
            "model": model, "model_sha256": model_sha256, "adapter": adapter,
            "engine": engine, "backend": backend, "architecture": snapshot.get("architecture"),
            "resource_source": source, "observed_at": snapshot["observed_at"],
            "reserve_bytes": reserve, **result,
            "notes": ["Peak memory measured on the profiled engine with a rank-8 stand-in adapter, "
                      "plus the margin; max_input_bytes_estimate is an estimate from a measured "
                      "bytes-per-token ratio on bench items, not a limit."]}


def format_report(result: dict) -> str:
    lines = [f"binary-judge {result['model']} on {result['backend']} ({result['engine']}):",
             f"  max input {result['max_input_tokens']} tokens (limit: {result['limit_reason']}"
             + (f", {result['limit_resource']}" if result['limit_resource'] else "") + ")",
             f"  context {result['max_context_tokens']} tokens; template {result['breakdown']['template_tokens']}"]
    for resource in ("ram", "vram"):
        part = result["breakdown"].get(resource)
        if part:
            lines.append(f"  {resource.upper()}: budget {part['budget_bytes'] / MIB:.0f} MiB, "
                         f"fixed {part['fixed_bytes'] / MIB:.0f} MiB, {part['per_token_bytes']} bytes/token")
    if result["max_input_bytes_estimate"] is not None:
        lines.append(f"  about {result['max_input_bytes_estimate']} bytes of text (estimate)")
    return "\n".join(lines)
