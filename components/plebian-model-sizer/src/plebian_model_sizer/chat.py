"""Conservative CPU sizing for an installed, single-session avatar chat model."""
from __future__ import annotations

import hashlib
import json
import re

from .resources import GIB, budgets, validate

REQUEST_SCHEMA = "kilix.avatar-chat.sizing-request/v1"
RESPONSE_SCHEMA = "plebian.models.avatar-chat-sizing/v1-development"
CONTEXT = 8192


def recommend_chat(request: dict, snapshot: dict, *, source: str = "live") -> dict:
    validate(snapshot)
    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("expected an avatar chat sizing request")
    candidates = request.get("models")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 64:
        raise ValueError("avatar chat request needs 1..64 models")
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("invalid avatar chat model")
        name = candidate.get("id")
        if (not isinstance(name, str) or
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", name) is None or
                name in seen):
            raise ValueError("invalid or duplicate avatar chat model ID")
        seen.add(name)
        if (type(candidate.get("model_bytes")) is not int or
                not 1 <= candidate["model_bytes"] <= 2**50 or
                type(candidate.get("parameters")) is not int or
                not 1 <= candidate["parameters"] <= 10**12 or
                type(candidate.get("installed")) is not bool or
                type(candidate.get("runtime_supported")) is not bool):
            raise ValueError("invalid avatar chat model metadata")

    budget = budgets(snapshot, "cpu")
    available = budget["ram_bytes"]
    # Keep at least half the current *usable* memory for the desktop, voice,
    # other apps and unmodelled allocation spikes. This is not a reservation.
    limit = available // 2 if available is not None else None
    rows = []
    for candidate in candidates:
        required = 2 * candidate["model_bytes"] + GIB // 2
        verdict = ("unknown" if limit is None else
                   "estimated-fit" if required <= limit else "does-not-fit")
        rows.append({**candidate, "backend": "cpu", "context": CONTEXT,
                     "required_ram_bytes": required, "usable_ram_bytes": available,
                     "avatar_budget_bytes": limit, "verdict": verdict,
                     "qualification_eligible": False})
    fitting = [row for row in rows if row["installed"] and row["runtime_supported"]
               and row["verdict"] == "estimated-fit"]
    fitting.sort(key=lambda row: (-row["parameters"], row["id"]))
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    return {"schema": RESPONSE_SCHEMA, "request_sha256": digest,
            "resource_source": source, "observed_at": snapshot["observed_at"],
            "context": CONTEXT, "budget": budget, "candidates": rows,
            "shortlist": [row["id"] for row in fitting],
            "provisional_candidate": fitting[0]["id"] if fitting else None,
            "selected_model": None, "qualification_eligible": False,
            "notes": ["CPU-only single-session estimate: twice model bytes plus 512 MiB.",
                      "At most half of live usable RAM is assigned to avatar chat.",
                      "Installed status and model-byte identity are caller assertions.",
                      "Quality, latency, runtime compatibility and admission are not established.",
                      "No model is downloaded, installed or selected by this report."]}
