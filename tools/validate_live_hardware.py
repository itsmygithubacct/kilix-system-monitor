#!/usr/bin/env python3
"""Exercise the exact hardware argv and validate live redacted responses."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "components" / "plebian-hardware" / "src"
TIMEOUT_SECONDS = 15
MAX_STDOUT_BYTES = 4 * 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 4096


def _candidate_module(
    validator_path: Path,
    candidate_root: Path,
    site_packages: Path,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "f106_external_candidate_validator", validator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_boundary(candidate_root, site_packages)
    return module


def _environment() -> dict[str, str]:
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _invoke(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    bootstrap = (
        "import runpy,sys; "
        "sys.path.insert(0,sys.argv.pop(1)); "
        "runpy.run_module('plebian_hardware',run_name='__main__')"
    )
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            bootstrap,
            str(SOURCE),
            *arguments,
        ],
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    arguments = parser.parse_args()
    candidate = arguments.candidate_root.resolve(strict=True)
    validator = _candidate_module(
        arguments.validator.resolve(strict=True),
        candidate,
        arguments.site_packages.resolve(strict=True),
    )
    available = validator.validators()
    contract = validator.load_json(candidate / "invocation-contract.json")
    privacy_contract = validator.load_json(
        candidate / "fixtures" / "privacy" / "default-local.json"
    )
    privacy_errors = validator.validate_document(
        "plebian.hardware.privacy/v1", privacy_contract, available
    )
    if privacy_errors:
        raise RuntimeError(
            "default privacy contract is invalid: " + "; ".join(privacy_errors)
        )
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
        if document["data"].get("never_collected") != privacy_contract["collection"]["never_collected"]:
            raise RuntimeError(f"{command} does not carry the contracted denylist")
        if document["data"].get("privacy") != privacy_contract["observation_projection"]:
            raise RuntimeError(f"{command} differs from the contracted privacy projection")
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
        "privacy contract enforced; no network, privilege, or qualification claim"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
