#!/usr/bin/env python3
"""Apply each planted defect in tests/defects.py to a private copy and run the suite there.

A defect the suite does not turn red is a hole. The unmutated copy runs first
and must pass, and each copy is checked to import its own source, not the
installed one, so a pass cannot come from running the wrong code. Exits 0 only
when every defect is caught and the control passes.

    python tests/run_defects.py [--jobs N] [NAME ...]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
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


def run(name: str, filename: str | None, edits: list[tuple[str, str]], scratch: Path) -> tuple[str, str, float]:
    place = Path(tempfile.mkdtemp(prefix="defect-", dir=scratch))
    os.chmod(place, 0o700)
    component = copy_tree(place)
    if filename is not None:
        apply(component, filename, edits)
    env = environment(component)
    probe = subprocess.run([sys.executable, "-c", "import kilix_gpu_residency as m; print(m.__file__)"],
                           cwd=component, env=env, capture_output=True, text=True, timeout=60)
    if not probe.stdout.strip().startswith(str(component / "src")):
        return name, f"WRONG-SOURCE {probe.stdout.strip()} {probe.stderr.strip()[-200:]}", 0.0
    started = time.monotonic()
    try:
        done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-f", "-s", "tests"], cwd=component,
                              env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=_TIMEOUT)
        outcome = "green" if done.returncode == 0 else "red"
    except subprocess.TimeoutExpired:
        outcome = "timeout"
    finally:
        shutil.rmtree(place, ignore_errors=True)
    return name, outcome, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    chosen = [d for d in DEFECTS if not args.names or d[0] in args.names]
    names = [d[0] for d in DEFECTS]
    if len(set(names)) != len(names):
        raise SystemExit("defect names must be unique")
    scratch = Path(tempfile.mkdtemp(prefix="gpu-residency-defects-"))
    os.chmod(scratch, 0o700)
    try:
        # Check every defect applies before spending time on runs.
        for name, filename, edits in chosen:
            check = Path(tempfile.mkdtemp(dir=scratch))
            apply(copy_tree(check), filename, edits)
            shutil.rmtree(check)
        control = run("control (no defect)", None, [], scratch)
        print(f"{control[1]:>8}  {control[2]:6.1f}s  {control[0]}", flush=True)
        if control[1] != "green":
            print("FAIL: the unmutated suite is not green", file=sys.stderr)
            return 1
        holes, odd = [], []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for name, outcome, seconds in pool.map(lambda d: run(d[0], d[1], d[2], scratch), chosen):
                print(f"{outcome:>8}  {seconds:6.1f}s  {name}", flush=True)
                if outcome == "green":
                    holes.append(name)
                elif outcome != "red":
                    odd.append(f"{name}: {outcome}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    caught = len(chosen) - len(holes) - len(odd)
    print(f"{caught}/{len(chosen)} planted defects caught; {len(holes)} holes; {len(odd)} other")
    for name in holes:
        print(f"HOLE: {name}")
    for line in odd:
        print(f"OTHER: {line}")
    return 0 if not holes and not odd else 1


if __name__ == "__main__":
    raise SystemExit(main())
