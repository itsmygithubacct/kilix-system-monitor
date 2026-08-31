#!/usr/bin/env python3
"""Prove that the F100-gated component contains no provisional fit engine."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "components" / "plebian-model-sizer"
ALLOWED = {"LICENSE", "README.md"}


def main() -> int:
    files = {
        str(path.relative_to(COMPONENT))
        for path in COMPONENT.rglob("*")
        if path.is_file()
    }
    if files != ALLOWED:
        print(
            "FAIL: model-sizer gate expected only LICENSE and README.md; "
            f"found {sorted(files)}",
            file=sys.stderr,
        )
        return 1
    notice = (COMPONENT / "README.md").read_text(encoding="utf-8")
    for required in (
        "capacity fixtures are frozen **3/3**",
        "F100 has not passed U5",
        "no executable",
    ):
        if required not in notice:
            print(f"FAIL: model-sizer blocker notice lacks {required!r}", file=sys.stderr)
            return 1
    print(
        "PASS: F100-C0 capacity fixtures 3/3 frozen; D4 remains blocked on "
        "F100 U5 0/1; model-sizer executables 0/1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
