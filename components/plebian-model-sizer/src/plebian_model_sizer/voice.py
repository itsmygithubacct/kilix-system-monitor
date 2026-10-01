"""Speech resource planning against exact, unqualified provider profiles."""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import os
import re

from . import __version__
from .estimate import assess
from .resources import MIB, budgets, validate

REQUEST_SCHEMA = "kilix.voice.sizing-request/v1"
RESPONSE_SCHEMA = "plebian.models.voice-sizing/v1-development"


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_profiles() -> dict:
    entries = json.loads(files(__package__).joinpath("voice_profiles.json").read_bytes())
    entries += json.loads(files(__package__).joinpath("tts_audition_profiles.json").read_bytes())
    entries += json.loads(files(__package__).joinpath("stt_bench_profiles.json").read_bytes())
    result = {}
    for entry in entries:
        document = entry["document"]
        raw = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
        if (hashlib.sha256(raw).hexdigest() != entry["source_sha256"]
                or document["schema"] != "plebian.models.profiles/v1"
                or document["fixture_kind"] != "provider-catalog"
                or document["qualification_eligible"] is not False
                or len(document["profiles"]) != 1):
            raise ValueError("invalid bundled speech profile")
        if entry["id"] in result:
            raise ValueError("duplicate bundled speech profile")
        result[entry["id"]] = entry
    return result


# Default dictation model, most preferred first. A model is the default only
# when it fits and the host meets its hardware class; the sizer still never
# selects or installs anything (selected_model stays null).
STT_DEFAULTS = (
    # Whisper small.en decodes a whole utterance at once; below this class a
    # sentence takes several seconds, which is too slow for dictation.
    ("whisper-small-en", {"architecture": "x86_64", "cpu_flags": ("avx2",),
                          "logical_cpus": 8, "ram_total_bytes": 16 * 1024 ** 3}),
    ("small-en-us", {}),
)


def observe_cpu(cpuinfo: str = "/proc/cpuinfo") -> dict:
    """Logical CPU count and instruction-set flags; unknown values are None."""
    flags = None
    try:
        with open(cpuinfo, encoding="ascii", errors="replace") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key.strip() == "flags":
                    flags = sorted(set(value.split()))
                    break
    except OSError:
        pass
    return {"logical_cpus": os.cpu_count(), "flags": flags}


def hardware_class_met(rule: dict, snapshot: dict, cpu: dict | None) -> bool:
    cpu = cpu or {}
    if "architecture" in rule and snapshot.get("architecture") != rule["architecture"]:
        return False
    if "ram_total_bytes" in rule and (snapshot.get("ram_total_bytes") or 0) < rule["ram_total_bytes"]:
        return False
    if "logical_cpus" in rule and (cpu.get("logical_cpus") or 0) < rule["logical_cpus"]:
        return False
    if "cpu_flags" in rule and not set(rule["cpu_flags"]) <= set(cpu.get("flags") or ()):
        return False
    return True


def validate_request(request: dict) -> None:
    if (not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA
            or set(request) != {"schema", "models"}):
        raise ValueError("expected a kilix-voice sizing request")
    entries = request["models"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 64:
        raise ValueError("voice request must contain 1..64 models")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"id", "task", "backend", "installed", "runtime_supported"}:
            raise ValueError("invalid voice model metadata")
        name = entry["id"]
        if not isinstance(name, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", name) is None or name in seen:
            raise ValueError("invalid or duplicate voice model ID")
        seen.add(name)
        if entry["task"] not in ("tts", "stt") or entry["backend"] not in ("cpu", "cuda"):
            raise ValueError("unsupported speech task or backend")
        if type(entry["runtime_supported"]) is not bool or (entry["installed"] is not None and type(entry["installed"]) is not bool):
            raise ValueError("voice availability must be boolean or unknown")


def recommend_voice(request: dict, snapshot: dict, *, task: str = "both", source: str = "live",
                    cpu: dict | None = None) -> dict:
    validate_request(request)
    validate(snapshot)
    if task not in ("tts", "stt", "both"):
        raise ValueError("invalid speech task")
    profiles = load_profiles()
    rows = []
    for model in request["models"]:
        if task != "both" and model["task"] != task:
            continue
        row = {**model, "verdict": "unknown", "profile_id": None, "profile_sha256": None,
               "inference": None, "installation": {"verdict": "unknown"}, "reasons": [],
               "runtime_identity": "unverified", "qualification_eligible": False}
        entry = profiles.get(model["id"])
        if entry is None:
            row["reasons"].append("no-resource-profile")
        else:
            profile = entry["document"]["profiles"][0]
            requirement = profile["requirements"]
            row.update(profile_id=profile["profile_id"], profile_sha256=entry["source_sha256"],
                       artifact=profile["artifact"], evidence=profile["evidence"],
                       provider=profile["provider"], profile_version=profile["version"])
            if profile["backend"] != model["backend"] or profile["task"] != model["task"]:
                row["reasons"].append("profile-task-or-backend-mismatch")
            elif snapshot.get("architecture") != requirement["architecture"]:
                row["reasons"].append("profile-architecture-mismatch-or-unknown")
            else:
                budget = budgets(snapshot, model["backend"], ram_reserve=256 * MIB,
                                 vram_reserve=256 * MIB, disk_reserve=128 * MIB)
                row["budget"] = budget
                margin = 10000 + profile["evidence"]["safety_margin_basis_points"]
                needs = {"ram": requirement["ram_peak_bytes"]}
                if model["backend"] == "cuda":
                    needs["vram"] = requirement["vram_peak_bytes"]
                if any(value is None for value in needs.values()):
                    row["reasons"].append("missing-memory-measurement")
                else:
                    needs = {key: (value * margin + 9999) // 10000 for key, value in needs.items()}
                    row["inference"] = assess(needs, budget)
                    row["verdict"] = row["inference"]["verdict"]
                # No inference fit is allowed to pretend unknown installation
                # scratch/download costs are zero, even for an installed model.
                disk = {key: requirement[key] for key in ("download_bytes", "disk_installed_bytes", "temporary_bytes")}
                row["installation"]["profile_bytes"] = disk
                if any(value is None for value in disk.values()):
                    row["installation"]["reason"] = "incomplete-acquisition-profile"
                else:
                    row["installation"].update(assess({"disk": sum(disk.values())}, budget))
        if model["runtime_supported"] is False:
            row["verdict"] = "unsupported"
            row["reasons"].append("voice-runtime-unsupported")
        rows.append(row)
    if not rows:
        raise ValueError("request has no models for the requested speech task")
    tasks = ("tts", "stt") if task == "both" else (task,)
    shortlists = {}
    for name in tasks:
        fitting = [row for row in rows if row["task"] == name and row["verdict"] == "estimated-fit"]
        fitting.sort(key=lambda row: (row["inference"]["resources"]["ram"]["required_bytes"], row["id"]))
        shortlists[name] = [row["id"] for row in fitting]
    defaults = {}
    if "stt" in tasks:
        defaults["stt"] = next((name for name, rule in STT_DEFAULTS
                                if name in shortlists["stt"] and hardware_class_met(rule, snapshot, cpu)), None)
    return {"schema": RESPONSE_SCHEMA, "provider_version": __version__, "request_sha256": digest(request),
            "task": task, "resource_source": source, "observed_at": snapshot["observed_at"],
            "candidates": rows, "shortlists": shortlists,
            "provisional_candidates": {key: ids[0] if ids else None for key, ids in shortlists.items()},
            "defaults": defaults, "cpu": cpu,
            "selected_model": None, "qualification_eligible": False,
            "notes": ["Reference-workload memory plus the profile margin; exact local runtime/artifact identity is unverified.",
                      "Each candidate is assessed independently. This is not a co-resident TTS/STT budget.",
                      "Installation needs a complete acquisition profile; runtime fit cannot authorize downloads.",
                      "Shortlists rank memory cost, not speech quality. Installed status does not establish runtime readiness."]}


def format_report(report: dict) -> str:
    lines = ["Speech resource estimates (reference profiles; local runtime identity unverified).",
             f"{'Model':30} {'Inference':18} {'RAM MiB':>8}  {'Installed':9} Installation"]
    for row in report["candidates"]:
        memory = (row["inference"] or {}).get("resources", {}).get("ram", {}).get("required_bytes")
        ram = f"{memory / MIB:.1f}" if memory is not None else "-"
        installed = {True: "yes", False: "no", None: "unknown"}[row["installed"]]
        lines.append(f"{row['id']:30} {row['verdict']:18} {ram:>8}  {installed:9} {row['installation']['verdict']}")
    for task, candidate in report["provisional_candidates"].items():
        lines.append(f"Provisional {task} candidate: {candidate or 'none'}")
    for task, model in report.get("defaults", {}).items():
        lines.append(f"Default {task} model for this hardware: {model or 'none'}")
    lines.append("Resource planning only; model choice and installation remain explicit.")
    return "\n".join(lines)
