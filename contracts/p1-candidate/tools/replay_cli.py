#!/usr/bin/env python3
"""Fixture-backed implementation of the candidate F106 subprocess contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESPONSES = ROOT / "fixtures" / "responses"
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024


class InputFailure(ValueError):
    pass


def _load_plan(path: Path) -> Any:
    try:
        if not path.is_file() or path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise InputFailure("input plan unavailable or oversized")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InputFailure("input plan is not valid UTF-8 JSON") from error


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputFailure("input plan contains a duplicate JSON key")
        result[key] = value
    return result


def _emit(path: Path) -> int:
    payload = path.read_bytes()
    if len(payload) > MAX_DOCUMENT_BYTES or not payload.endswith(b"\n"):
        print("fixture contract invariant failed", file=sys.stderr)
        return 70
    sys.stdout.buffer.write(payload)
    return 0


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
            ("snapshot", "--json"): "sizer-snapshot.json",
        }
        fixture = routes.get(tuple(argv))
        if fixture is not None:
            return _emit(RESPONSES / fixture)
        if len(argv) == 3 and argv[0] == "install" and argv[2] == "--json":
            try:
                plan = _load_plan(Path(argv[1]))
            except InputFailure as error:
                print(str(error), file=sys.stderr)
                return 65
            if not isinstance(plan, dict) or plan.get("schema") != "plebian.models.install-plan/v1":
                print("input plan has an unsupported schema", file=sys.stderr)
                return 65
            return _emit(RESPONSES / "sizer-install-blocked.json")
        return _usage()

    print("unknown fixture program", file=sys.stderr)
    return 70


if __name__ == "__main__":
    print("invoke through tools/replay-bin", file=sys.stderr)
    raise SystemExit(70)
