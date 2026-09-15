#!/usr/bin/env python3
"""Fail unless make check still runs every gate this repository relies on.

A gate removed from the check prerequisites, or a recipe line deleted from a
gate, leaves every remaining gate green. This asks make itself what check would
run (make -n expands every prerequisite and variable and runs nothing) and
requires one command line for each gate. tools/tests/test_gate_wiring.py runs
the same check against planted Makefile removals, and against this repository's
Makefile, so deleting this check's own line from check is seen there.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Each gate, as the fragments one expanded command line of make -n check carries.
GATES = {
    "candidate contract validator": ("tools/validate_candidate",),
    "frozen bundle validator": ("python tools/validate_frozen_contracts.py",),
    "frozen validator guard tests": ("python -m unittest discover -s tools/tests",),
    "telemetry suite": ("cd components/kilix-telemetry", "unittest discover -s tests"),
    "lease suite": ("cd components/kilix-device-lease", "unittest discover -s tests"),
    "hardware suite": ("cd components/plebian-hardware", "unittest discover -s tests"),
    "live hardware validator": ("tools/validate_candidate --live-hardware",),
    "package gate": ("python tools/check_distributions.py",),
    "profile measurement suite": ("python -m unittest discover -s tools/measure/tests",),
    "capacity evidence validator": ("python tools/validate_h2_capacity_evidence.py",),
    "launcher consumer readiness": ("python tools/check_trusted_launcher_consumer_readiness.py --self-test",),
    "model sizer block": ("python tools/check_model_sizer_block.py",),
    "gate wiring check": ("python tools/check_gate_wiring.py",),
}


def dry_run(makefile: Path, uv: str = "/absolute/uv") -> str:
    environment = {key: value for key, value in os.environ.items() if key not in ("MAKEFLAGS", "MAKELEVEL", "MFLAGS")}
    completed = subprocess.run(
        ["make", "-n", "-rR", "--no-print-directory", "-f", str(makefile), "-C", str(makefile.parent),
         "check", f"UV={uv}"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=environment, timeout=60, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"make -n check failed: {completed.stderr.strip()[-500:]}")
    return completed.stdout


def wiring_errors(commands: str) -> list[str]:
    lines = [line.strip() for line in commands.splitlines()]
    live = [line for line in lines if line and not line.startswith(("#", ":", "true", "echo"))]
    errors = []
    for gate, fragments in GATES.items():
        matches = [line for line in live if all(fragment in line for fragment in fragments)]
        if gate == "candidate contract validator":
            matches = [line for line in matches if "--live-hardware" not in line]
        if not matches:
            errors.append(f"make check no longer runs the {gate}")
    return errors


def main() -> int:
    errors = wiring_errors(dry_run(ROOT / "Makefile"))
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"PASS: make check runs {len(GATES)}/{len(GATES)} required gates")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: gate wiring could not be read: {error}", file=sys.stderr)
        raise SystemExit(1)
