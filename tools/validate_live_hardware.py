#!/usr/bin/env python3
"""Exercise the exact hardware argv and validate live redacted responses."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "contracts" / "p1-candidate"
SOURCE = ROOT / "components" / "plebian-hardware" / "src"
TIMEOUT_SECONDS = 15
MAX_STDOUT_BYTES = 4 * 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 4096


def _candidate_module() -> ModuleType:
    path = CANDIDATE / "tools" / "validate.py"
    spec = importlib.util.spec_from_file_location("f106_candidate_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment() -> dict[str, str]:
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(SOURCE),
    }


def _invoke(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", "plebian_hardware", *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=TIMEOUT_SECONDS,
        env=_environment(),
    )


def _json_document(stdout: bytes) -> dict[str, Any]:
    if not stdout.endswith(b"\n") or len(stdout) > MAX_STDOUT_BYTES:
        raise RuntimeError("successful JSON output violates its byte boundary")
    try:
        text = stdout.decode("utf-8")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("successful JSON output is not one UTF-8 document") from error
    if not isinstance(value, dict):
        raise RuntimeError("successful JSON output is not an object")
    return value


def main() -> int:
    validator = _candidate_module()
    available = validator.validators()
    contract = validator.load_json(CANDIDATE / "invocation-contract.json")
    by_command = {
        item["command_id"]: item
        for item in contract["commands"]
        if item.get("stdout") == "plebian.cli.response/v1"
    }

    observed: list[dict[str, Any]] = []
    for arguments, command in (
        (["inventory", "--json"], "hardware.inventory"),
        (["gpu", "--json"], "hardware.gpu"),
    ):
        completed = _invoke(arguments)
        if completed.returncode != 0 or completed.stderr:
            raise RuntimeError(f"{command} did not obey the success stream contract")
        document = _json_document(completed.stdout)
        errors = validator.validate_response(document, by_command, available)
        if errors:
            raise RuntimeError(f"{command} contract failure: {'; '.join(errors)}")
        capture = document["data"]["capture"]
        if (
            capture.get("network_used") is not False
            or capture.get("privileged") is not False
            or capture.get("qualification_eligible") is not False
        ):
            raise RuntimeError(f"{command} crossed the unprivileged local boundary")
        observed.append(document)

    show = _invoke(["show"])
    if show.returncode != 0 or show.stderr or not show.stdout.endswith(b"\n"):
        raise RuntimeError("hardware.show did not obey the success stream contract")
    if b"unqualified local observation" not in show.stdout:
        raise RuntimeError("hardware.show omitted the qualification exclusion")

    invalid = _invoke(["--json", "inventory"])
    if invalid.returncode != 2 or invalid.stdout:
        raise RuntimeError("invalid argv did not fail with empty stdout and status 2")
    if (
        not invalid.stderr.endswith(b"\n")
        or b"\n" in invalid.stderr[:-1]
        or len(invalid.stderr) > MAX_DIAGNOSTIC_BYTES
        or validator.privacy_errors(invalid.stderr.decode("utf-8"))
    ):
        raise RuntimeError("invalid argv violated the bounded redacted stderr contract")

    unknown_counts = [len(document["data"]["unknowns"]) for document in observed]
    gpu_counts = [len(document["data"]["gpus"]) for document in observed]
    print(
        "PASS: exact show/inventory/gpu argv; two live candidate-valid redacted "
        f"observations; unknown counts {unknown_counts}; GPU counts {gpu_counts}; "
        "no network, privilege, or qualification claim"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
