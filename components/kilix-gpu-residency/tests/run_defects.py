#!/usr/bin/env python3
"""Apply each planted defect in tests/defects.py to a private copy and run the suite there.

A defect the suite does not turn red is a hole. So is one that turns it red only
by breaking an import (a ``unittest.loader._FailedTest``) or by an error that is
not a test's own: the defect must be caught by a test that ran. Each defect's
line names the test that caught it, and ``--report FILE`` writes them all as
JSON. A defect with a fourth element is run against that one test module only.

The unmutated copy runs first and must pass, and each copy is checked to import
its own source, not the installed one, so a pass cannot come from running the
wrong code. Exits 0 only when every defect is caught and the control passes.

    python tests/run_defects.py [--jobs N] [--report FILE] [NAME ...]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

TESTS = Path(__file__).resolve().parent
COMPONENT = TESTS.parent
ROOT = COMPONENT.parents[1]
sys.path.insert(0, str(TESTS))
from defects import DEFECTS  # noqa: E402

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv")
_TIMEOUT = 900
_HEADER = re.compile(r"^(FAIL|ERROR): (\S+) \(([\w.]+)\)", re.M)
# What a defect breaking the suite's machinery, not a behaviour, looks like.
_MACHINERY = ("unittest.loader._FailedTest", "ImportError", "ModuleNotFoundError", "SyntaxError",
              "IndentationError")


def copy_tree(into: Path) -> Path:
    component = into / "components" / COMPONENT.name
    shutil.copytree(COMPONENT, component, ignore=_IGNORE)
    (into / "contracts").mkdir()
    shutil.copy2(ROOT / "contracts" / "kilix.gpu-residency-v1.interface.json", into / "contracts")
    shutil.copy2(ROOT / "manifest.toml", into)
    return component


def apply(component: Path, filename: str, edits: list[tuple[str, str]]) -> None:
    path = component / "src" / "kilix_gpu_residency" / filename
    text = path.read_text(encoding="utf-8")
    for old, new in edits:
        if text.count(old) != 1:
            raise SystemExit(f"defect text occurs {text.count(old)} times in {filename}: {old[:70]!r}")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def environment(component: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "KILIX_RESIDENCY_TEST_ROOT", "OLLAMA_HOST", "KILIX_GPU_DEVICES")}
    env["PYTHONPATH"] = str(component / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def caught_by(stderr: str) -> tuple[str, str]:
    """(outcome, test id) from a red run's output: "red" when a test that ran caught it,
    "machinery" when the first failure is an import or a loader error."""
    match = _HEADER.search(stderr)
    if match is None:
        return "machinery", ""
    kind, method, test_id = match.groups()
    block = stderr[match.end():].split("\n======", 1)[0]
    last = [line for line in block.strip().splitlines() if line and not line.startswith(" ")]
    exception = last[-1] if last else ""
    if test_id.startswith("unittest.loader") or (kind == "ERROR" and exception.startswith(_MACHINERY)):
        return "machinery", f"{test_id} {exception[:120]}"
    return "red", test_id


def run(name: str, filename: str | None, edits: list[tuple[str, str]], scratch: Path,
        only: str | None = None) -> tuple[str, str, float, str]:
    place = Path(tempfile.mkdtemp(prefix="defect-", dir=scratch))
    os.chmod(place, 0o700)
    component = copy_tree(place)
    if filename is not None:
        apply(component, filename, edits)
    env = environment(component)
    probe = subprocess.run([sys.executable, "-c", "import kilix_gpu_residency as m; print(m.__file__)"],
                           cwd=component, env=env, capture_output=True, text=True, timeout=60)
    if not probe.stdout.strip().startswith(str(component / "src")):
        return name, f"WRONG-SOURCE {probe.stdout.strip()} {probe.stderr.strip()[-200:]}", 0.0, ""
    started = time.monotonic()
    pattern = ["-p", only] if only else []
    test = ""
    try:
        done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-f", "-s", "tests", *pattern],
                              cwd=component, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=_TIMEOUT)
        if done.returncode == 0:
            outcome = "green"
        else:
            outcome, test = caught_by(done.stderr)
    except subprocess.TimeoutExpired:
        outcome = "timeout"
    finally:
        shutil.rmtree(place, ignore_errors=True)
    return name, outcome, time.monotonic() - started, test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--report", help="write each defect's outcome and catching test as JSON")
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    chosen = [(d[0], d[1], d[2], d[3] if len(d) > 3 else None) for d in DEFECTS
              if not args.names or d[0] in args.names]
    names = [d[0] for d in DEFECTS]
    if len(set(names)) != len(names):
        raise SystemExit("defect names must be unique")
    scratch = Path(tempfile.mkdtemp(prefix="gpu-residency-defects-"))
    os.chmod(scratch, 0o700)
    report = []
    try:
        # Check every defect applies before spending time on runs.
        for name, filename, edits, _ in chosen:
            check = Path(tempfile.mkdtemp(dir=scratch))
            apply(copy_tree(check), filename, edits)
            shutil.rmtree(check)
        control = run("control (no defect)", None, [], scratch)
        print(f"{control[1]:>9}  {control[2]:6.1f}s  {control[0]}", flush=True)
        if control[1] != "green":
            print("FAIL: the unmutated suite is not green", file=sys.stderr)
            return 1
        holes, odd = [], []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for name, outcome, seconds, test in pool.map(lambda d: run(d[0], d[1], d[2], scratch, d[3]), chosen):
                print(f"{outcome:>9}  {seconds:6.1f}s  {name}" + (f"  <- {test}" if test else ""), flush=True)
                report.append({"defect": name, "outcome": outcome, "test": test, "seconds": round(seconds, 1)})
                if outcome == "green":
                    holes.append(name)
                elif outcome != "red":
                    odd.append(f"{name}: {outcome} {test}".rstrip())
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if args.report:
        Path(args.report).write_text(json.dumps({"control": control[1], "defects": report}, indent=2) + "\n")
    caught = len(chosen) - len(holes) - len(odd)
    print(f"{caught}/{len(chosen)} planted defects caught; {len(holes)} holes; {len(odd)} other")
    for name in holes:
        print(f"HOLE: {name}")
    for line in odd:
        print(f"OTHER: {line}")
    return 0 if not holes and not odd else 1


if __name__ == "__main__":
    raise SystemExit(main())
