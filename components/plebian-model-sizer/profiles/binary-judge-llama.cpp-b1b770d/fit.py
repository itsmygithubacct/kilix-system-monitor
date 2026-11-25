#!/usr/bin/env python3
"""Fit the v1.0.0 CPU profile from measure.py's points and write it into the bundled profiles.

    fit.py MEASUREMENT.json TEMPLATE_TOKENS [PROOF.json]

The base was measured without an adapter, so the stand-in adapter's bytes are
added to every peak, as the earlier profiles' real-judge runs were counted.
The fit is the sizer's: least-squares slope per context token, and the
smallest fixed part that keeps every counted peak under the line.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from plebian_model_sizer import binary_judge  # noqa: E402

BUNDLE = ROOT / "src/plebian_model_sizer/binary_judge_profiles.json"
OLD = "qwen3.5-0.8b-q8_0.llama.cpp-8ea2902.cpu.x86_64"
NEW = "qwen3.5-0.8b-q8_0.llama.cpp-b1b770d.cpu.x86_64"
ENGINE = "llama.cpp@b1b770d90d94cad9e9ea5b993bdbc383a522fb28"
HOST = "x86_64-i7-9850h-12t-62g"


def main():
    measured = json.loads(Path(sys.argv[1]).read_text())
    template = int(sys.argv[2])
    proof = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else None
    document = json.loads(BUNDLE.read_text())
    old = next(p for p in document["profiles"] if p["id"] == OLD)
    standin = old["standin_adapter"]["bytes"]
    points = [[row["ctx"], row["peak_bytes"] + standin] for row in measured["points"]]
    n = len(points)
    mean_x, mean_y = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in points)
             / sum((x - mean_x) ** 2 for x, _ in points))
    per_token = round(slope)
    counted = points + ([[proof["ctx"], proof["peak_bytes"] + standin]] if proof else [])
    fixed = max(y - per_token * x for x, y in counted)
    evidence = {
        "bytes_per_token": old["evidence"]["bytes_per_token"],
        "hosts": {"ram": {HOST: {"intercept": round(mean_y - slope * mean_x), "slope": round(slope, 2),
                                 "points": points}}},
        "measured_at": "2026-11-24",
        "method": old["evidence"]["method"],
        "peak": ("VmHWM of llama-server after one completion whose prompt fills the context (ctx - 8 tokens), "
                 "base model alone, plus the stand-in adapter's bytes"),
        "template": f"kilix-llm judge contract prompt, Qwen chat template, thinking off: {template} tokens",
        "tool": "profiles/binary-judge-llama.cpp-b1b770d/measure.py and fit.py",
    }
    if proof:
        evidence["judge_proofs"] = (
            f"one real `kilix-llm judge` run at {proof['input_tokens']:,} input tokens (base only; the "
            "stand-in's bytes added), counted in the fixed part")
        evidence["judge_proofs_used"] = ["ram"]
        evidence["judge_proof_checks"] = {"ram": [{
            "ctx": proof["ctx"], "host": HOST, "peak": proof["peak_bytes"] + standin,
            "headroom_bytes": fixed + per_token * proof["ctx"] - (proof["peak_bytes"] + standin)}]}
    profile = {
        "id": NEW, "architecture": "x86_64", "bytes_per_token": old["bytes_per_token"],
        "engine": {"backend": "cpu", "id": ENGINE,
                   "build": "kilix-llm build-runtime.sh, GGML_NATIVE=ON, 0 patches",
                   "server_flags": old["engine"]["server_flags"]},
        "engine_cap_tokens": None, "evidence": evidence, "margin_basis_points": old["margin_basis_points"],
        "model": old["model"], "ram": {"fixed_bytes": fixed, "per_token_bytes": per_token},
        "standin_adapter": old["standin_adapter"], "template_tokens": template, "vram": None,
    }
    profile["profile_sha256"] = binary_judge.profile_digest(profile)
    binary_judge.check_profile(profile)
    document["profiles"] = [p for p in document["profiles"] if p["id"] != NEW]
    document["profiles"].insert(1, profile)
    BUNDLE.write_text(json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"id": NEW, "ram": profile["ram"], "host_slope": round(slope, 2),
                      "old_ram": old["ram"]}))


if __name__ == "__main__":
    main()
