#!/usr/bin/env python3
"""Validate the published, explicitly non-qualifying F100-C0 H2 observation."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from validate_candidate import privacy_errors


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "h2-capacity-observations-20260831.json"
PROBE = (
    ROOT
    / "components"
    / "plebian-hardware"
    / "src"
    / "plebian_hardware"
    / "probe.py"
)
LOADAVG = re.compile(
    r"^[0-9]+(?:\.[0-9]+)? [0-9]+(?:\.[0-9]+)? [0-9]+(?:\.[0-9]+)? "
    r"[0-9]+/[0-9]+ [0-9]+$"
)
EXPECTED_WINDOWS = {
    "capacity-revalidation",
    "pre-fix-inventory",
    "post-fix-worktree-control",
    "committed-candidate-inventory",
}
FIXED_COMMIT = "327f959d0d84154f8ca7d73ab97c4a197f383c28"
FIXED_PROBE_SHA256 = "f9806b8ef9c24d20a0b3845f8c994743257e75e33da2ba34bc89f48e0ee6254b"


class EvidenceFailure(ValueError):
    pass


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceFailure("duplicate JSON key")
        result[key] = value
    return result


def _load() -> dict[str, Any]:
    payload = EVIDENCE.read_bytes()
    try:
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvidenceFailure(f"non-finite value: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceFailure("evidence is not strict JSON") from error
    if not isinstance(document, dict):
        raise EvidenceFailure("evidence is not an object")
    canonical = (
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if payload != canonical:
        raise EvidenceFailure("evidence is not canonical JSON")
    return document


def _window_time(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EvidenceFailure("measurement time is not UTC")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise EvidenceFailure("measurement time is invalid") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceFailure(message)


def main() -> int:
    document = _load()
    _require(
        document.get("schema") == "f106.h2-capacity-observation/v1",
        "evidence schema differs",
    )
    _require(document.get("fixture_id") == "redacted:f100-c0-h2", "fixture differs")
    _require(document.get("tier") == "H2", "tier differs")
    failures = privacy_errors(document)
    _require(not failures, "evidence failed the central privacy scanner")

    windows = document.get("measurement_windows")
    _require(isinstance(windows, list) and len(windows) == 4, "window count differs")
    by_id = {
        item.get("window_id"): item
        for item in windows
        if isinstance(item, dict) and isinstance(item.get("window_id"), str)
    }
    _require(set(by_id) == EXPECTED_WINDOWS, "window identities differ")
    for window in windows:
        _require(isinstance(window, dict), "window is not an object")
        start = _window_time(window.get("started_at"))
        end = _window_time(window.get("ended_at"))
        _require(end >= start, "measurement window ends before it starts")
        _require(
            isinstance(window.get("loadavg_start"), str)
            and LOADAVG.fullmatch(window["loadavg_start"]) is not None,
            "measurement window lacks exact start loadavg",
        )
        _require(
            isinstance(window.get("loadavg_end"), str)
            and LOADAVG.fullmatch(window["loadavg_end"]) is not None,
            "measurement window lacks exact end loadavg",
        )

    capacity = by_id["capacity-revalidation"]["observations"]
    _require(capacity.get("architecture") == "x86_64", "H2 architecture differs")
    _require(capacity.get("memory_bytes", 0) >= 16 * 1024**3, "H2 RAM is below floor")
    _require(
        capacity.get("storage_available_bytes", 0) >= 120 * 1024**3,
        "H2 storage is below floor",
    )
    gpu = capacity.get("gpu", {})
    _require(gpu.get("vram_bytes") == 8 * 1024**3, "H2 VRAM is not at its exact floor")
    _require(gpu.get("driver_version") == "550.163.01", "H2 driver differs")
    frequency = capacity.get("frequency_policy", {})
    _require(frequency.get("governor") == "performance", "H2 governor differs")
    _require(
        frequency.get("governor_cpus") == {"observed": 16, "total": 16},
        "H2 governor coverage differs",
    )
    _require(frequency.get("no_turbo") is True, "H2 turbo control differs")

    before = by_id["pre-fix-inventory"]["observations"]
    _require(
        before
        == {
            "cuda_evidence": "command-unavailable",
            "cuda_status": "unknown",
            "vram_bytes": None,
        },
        "pre-fix causal observation differs",
    )
    for identity in ("post-fix-worktree-control", "committed-candidate-inventory"):
        observation = by_id[identity]["observations"]
        _require(observation.get("cuda_status") == "available", "CUDA remains unknown")
        _require(
            observation.get("cuda_evidence") == "executable-probe",
            "CUDA evidence is not executable-bound",
        )
        _require(observation.get("vram_bytes") == 8 * 1024**3, "VRAM remains unknown")

    committed = by_id["committed-candidate-inventory"]
    _require(committed["source"].get("commit") == FIXED_COMMIT, "source commit differs")
    _require(
        committed["source"].get("probe_sha256") == FIXED_PROBE_SHA256,
        "recorded probe digest differs",
    )
    _require(
        hashlib.sha256(PROBE.read_bytes()).hexdigest() == FIXED_PROBE_SHA256,
        "current probe differs from measured bytes",
    )
    current_hz = committed["observations"].get("cpu_current_max_hz")
    hardware_hz = committed["observations"].get("cpu_hardware_max_hz")
    _require(hardware_hz == 2_900_000_000, "base-clock ceiling differs")
    _require(
        isinstance(current_hz, int) and current_hz <= hardware_hz * 1.01,
        "observed clock exceeds the frozen ceiling tolerance",
    )
    _require(
        committed["observations"].get("network_used") is False
        and committed["observations"].get("privileged") is False
        and committed["observations"].get("qualification_eligible") is False,
        "committed observation crossed its local non-qualifying boundary",
    )

    disposition = document.get("disposition", {})
    _require(
        disposition.get("windows_with_load_pair") == {"measured": 4, "total": 4},
        "load-pair denominator differs",
    )
    _require(disposition.get("qualification_eligible") is False, "evidence self-qualifies")
    _require(
        disposition.get("installer_media_evidence") == "not-measured"
        and disposition.get("model_performance_evidence") == "not-measured",
        "evidence overclaims its measurement boundary",
    )
    print(
        "PASS (measured, non-qualifying): H2 windows with load pairs 4/4; "
        "capacity facts 8/8; causal before/after states 2/2; exact committed "
        "collector binding 2/2; installer-media evidence 0/1; model-performance "
        "evidence 0/1; qualification 0/1"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceFailure, OSError) as error:
        print(f"FAIL: H2 capacity evidence: {error}", file=sys.stderr)
        raise SystemExit(1)
