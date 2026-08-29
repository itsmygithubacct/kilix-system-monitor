#!/usr/bin/env python3
"""Fixture-backed implementation of the candidate F106 subprocess contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESPONSES = ROOT / "fixtures" / "responses"
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024


class InputFailure(ValueError):
    pass


def _load_plan(path: Path) -> tuple[Any, bytes, str]:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise InputFailure("input plan unavailable or aliased") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= MAX_DOCUMENT_BYTES:
            raise InputFailure("input plan unavailable or oversized")
        chunks: list[bytes] = []
        remaining = MAX_DOCUMENT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_DOCUMENT_BYTES:
            raise InputFailure("input plan unavailable or oversized")
    except OSError as error:
        raise InputFailure("input plan unavailable or unreadable") from error
    finally:
        os.close(descriptor)
    try:
        plan = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise InputFailure("input plan is not valid UTF-8 JSON") from error
    canonical = (json.dumps(plan, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if payload != canonical:
        raise InputFailure("input plan is not canonical JSON")
    return plan, payload, hashlib.sha256(payload).hexdigest()


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputFailure("input plan contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> Any:
    raise InputFailure("input plan contains a non-finite number")


def _emit(path: Path) -> int:
    payload = path.read_bytes()
    if len(payload) > MAX_DOCUMENT_BYTES or not payload.endswith(b"\n"):
        print("fixture contract invariant failed", file=sys.stderr)
        return 70
    sys.stdout.buffer.write(payload)
    return 0


def _emit_bound_install(path: Path, plan_sha256: str) -> int:
    try:
        response = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, InputFailure):
        print("fixture contract invariant failed", file=sys.stderr)
        return 70
    if response.get("data", {}).get("plan_sha256") != plan_sha256:
        print("fixture plan binding invariant failed", file=sys.stderr)
        return 70
    return _emit(path)


def _usage() -> int:
    print("unsupported fixture invocation", file=sys.stderr)
    return 2


def main(program: str, argv: list[str]) -> int:
    if program == "plebian-hardware":
        routes = {
            ("show",): "hardware-show.txt",
            ("inventory", "--json"): "hardware-inventory.json",
            ("gpu", "--json"): "hardware-gpu.json",
        }
        fixture = routes.get(tuple(argv))
        return _usage() if fixture is None else _emit(RESPONSES / fixture)

    if program == "plebian-model-sizer":
        routes = {
            ("recommend", "tts", "--json"): "sizer-recommend-tts.json",
            ("plan", "local-ai-balanced", "--json"): "sizer-plan.json",
            ("install-status", "TRANSACTION_ID", "--json"): "sizer-install-status-blocked.json",
            ("cancel", "TRANSACTION_ID", "--json"): "sizer-install-cancel-blocked.json",
            ("snapshot", "--json"): "sizer-snapshot.json",
        }
        fixture = routes.get(tuple(argv))
        if fixture is not None:
            return _emit(RESPONSES / fixture)
        if (
            len(argv) == 5
            and argv[0] == "install"
            and argv[2] == "--expected-plan-sha256"
            and argv[4] == "--json"
        ):
            expected_sha256 = argv[3]
            if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
                print("expected plan SHA-256 is invalid", file=sys.stderr)
                return 65
            try:
                plan, _, actual_sha256 = _load_plan(Path(argv[1]))
            except InputFailure as error:
                print(str(error), file=sys.stderr)
                return 65
            if not isinstance(plan, dict) or plan.get("schema") != "plebian.models.install-plan/v1":
                print("input plan has an unsupported schema", file=sys.stderr)
                return 65
            if actual_sha256 != expected_sha256:
                print("input plan differs from its reviewed SHA-256", file=sys.stderr)
                return 65
            return _emit_bound_install(
                RESPONSES / "sizer-install-blocked.json", actual_sha256
            )
        return _usage()

    print("unknown fixture program", file=sys.stderr)
    return 70


if __name__ == "__main__":
    print("invoke through tools/replay-bin", file=sys.stderr)
    raise SystemExit(70)
