"""Vision, audio and image planning from pinned reference workloads."""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import re

from .estimate import assess
from .resources import MIB, budgets, validate
from .voice import digest

REQUEST_SCHEMA = "kilix.runtime.sizing-request/v1"
RESPONSE_SCHEMA = "plebian.models.runtime-sizing/v1-development"
DEFAULTS = {
    "vision": ("yolox_s", "yolox_tiny", "yolox_nano"),
    "audio": ("encodec-24khz-stateful", "encodec-48khz-frame"),
    "image": ("bonsai-image-4b-ternary-gemlite", "bonsai-image-4b-binary-gemlite"),
}


def load_profiles():
    result = {}
    for entry in json.loads(files(__package__).joinpath("runtime_profiles.json").read_bytes()):
        document = entry["document"]
        raw = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
        if (hashlib.sha256(raw).hexdigest() != entry["source_sha256"]
                or document["schema"] != "plebian.models.runtime-profile/v1-development"
                or document["qualification_eligible"] is not False
                or document["id"] in result or document["task"] not in DEFAULTS
                or document["backend"] not in ("cpu", "cuda")):
            raise ValueError("invalid bundled runtime profile")
        for key in ("ram_peak_bytes", "vram_peak_bytes"):
            value = document[key]
            if value is not None and (type(value) is not int or not 0 <= value <= 2**63):
                raise ValueError("invalid runtime memory measurement")
        margin = document["safety_margin_basis_points"]
        if type(margin) is not int or not 0 <= margin <= 10000:
            raise ValueError("invalid runtime profile margin")
        device = document.get('device_scope')
        if device is not None:
            if (document['backend'] != 'cuda' or not isinstance(device,dict)
                    or set(device) != {'gpu_index','minimum_total_bytes','maximum_total_bytes'}
                    or any(type(value) is not int for value in device.values())
                    or not 0 <= device['gpu_index'] <= 255
                    or not 0 < device['minimum_total_bytes'] <= device['maximum_total_bytes'] <= 2**63):
                raise ValueError('invalid runtime profile device scope')
        result[document["id"]] = entry
    return result


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {"schema", "models"} or request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("expected a runtime sizing request")
    models = request["models"]
    if not isinstance(models, list) or not 1 <= len(models) <= 64:
        raise ValueError("runtime request must contain 1..64 models")
    seen = set()
    for row in models:
        if (not isinstance(row, dict) or set(row) != {"id", "task", "manifest_digest"}
                or not isinstance(row["id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", row["id"])
                or row["id"] in seen or not isinstance(row["task"], str) or row["task"] not in DEFAULTS
                or not isinstance(row["manifest_digest"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["manifest_digest"])):
            raise ValueError("invalid runtime model metadata")
        seen.add(row["id"])


def recommend_runtime(request, snapshot, *, source="live"):
    validate_request(request)
    validate(snapshot)
    profiles = load_profiles()
    rows = []
    for model in request["models"]:
        row = {**model, "verdict": "unknown", "inference": None,
               "required_ram_bytes": None, "required_vram_bytes": None,
               "reasons": [], "qualification_eligible": False, "runtime_identity": "unverified"}
        entry = profiles.get(model["id"])
        if entry is None:
            row["reasons"].append("no-resource-profile")
        else:
            profile = entry["document"]
            row.update(profile_sha256=entry["source_sha256"], evidence=profile["evidence"],
                       workload=profile["workload"], backend=profile["backend"])
            if profile["manifest_digest"] != model["manifest_digest"] or profile["task"] != model["task"]:
                row["reasons"].append("profile-catalog-mismatch")
            elif snapshot.get("architecture") != profile["architecture"]:
                row["reasons"].append("profile-architecture-mismatch-or-unknown")
            elif profile.get('device_scope') and (
                    snapshot.get('cuda_device_mapping')!='single-unmasked-gpu-zero'
                    or len(snapshot['gpus'])!=1):
                row['reasons'].append('profile-device-mapping-unverified')
            elif profile.get('device_scope') and not any(
                    gpu['index']==profile['device_scope']['gpu_index']
                    and profile['device_scope']['minimum_total_bytes'] <= gpu['total_bytes']
                    <= profile['device_scope']['maximum_total_bytes'] for gpu in snapshot['gpus']):
                row['reasons'].append('profile-device-mode-unmeasured')
            else:
                budget = budgets(snapshot, profile["backend"],
                                 gpu=profile.get('device_scope',{}).get('gpu_index'),ram_reserve=256 * MIB,
                                 vram_reserve=256 * MIB, disk_reserve=128 * MIB)
                margin = 10000 + profile["safety_margin_basis_points"]
                needs = {}
                missing = False
                for resource in ("ram", "vram") if profile["backend"] == "cuda" else ("ram",):
                    peak = profile[resource + "_peak_bytes"]
                    if peak is None:
                        missing = True
                    else:
                        required = (peak * margin + 9999) // 10000
                        needs[resource] = required
                        row["required_" + resource + "_bytes"] = required
                row["inference"] = assess(needs, budget) if needs else None
                row["verdict"] = row["inference"]["verdict"] if needs else "unknown"
                if missing:
                    row["reasons"].append("missing-memory-measurement")
                    if row["verdict"] == "estimated-fit":
                        row["verdict"] = "unknown"
                        row["inference"]["verdict"] = "unknown"
                if profile["backend"] == "cuda" and not snapshot["gpus"]:
                    row["verdict"] = "unknown"
                    row["reasons"].append("cuda-runtime-not-observed")
                row["reasons"].append("reference-workload-only")
        rows.append(row)
    fitting = {row["id"] for row in rows if row["verdict"] == "estimated-fit"}
    tasks = sorted({row["task"] for row in rows})
    defaults = {task: next((name for name in DEFAULTS[task] if name in fitting), None) for task in tasks}
    return {"schema": RESPONSE_SCHEMA, "request_sha256": digest(request), "resource_source": source,
            "observed_at": snapshot["observed_at"], "candidates": rows, "defaults": defaults,
            "selected_model": None, "qualification_eligible": False,
            "notes": ["Reference workload memory with margin; local runtime identity and speed remain unverified.",
                      "Candidates are assessed separately; this is not a co-resident memory budget.",
                      "Image profiles cover the measured GPU-0 CPU-VAE 512x512 preview only; other modes and resolutions remain unmeasured."]}
