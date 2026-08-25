#!/usr/bin/env python3
"""Build and inspect both implemented Python component distributions offline."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "kilix-telemetry": {
        "path": ROOT / "components" / "kilix-telemetry",
        "module": "kilix_telemetry/__init__.py",
        "version": "0.1.2",
    },
    "plebian-hardware": {
        "path": ROOT / "components" / "plebian-hardware",
        "module": "plebian_hardware/__init__.py",
        "version": "0.1.0",
    },
}


def _safe(names: list[str]) -> None:
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise RuntimeError(f"distribution has unsafe member: {name}")
        if any(part in {".git", ".venv", "__pycache__", "research"} for part in path.parts):
            raise RuntimeError(f"distribution has private/build member: {name}")


def _inspect_wheel(path: Path, module: str, name: str, version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _safe(names)
        if module not in names:
            raise RuntimeError(f"{name} wheel lacks {module}")
        metadata_paths = [entry for entry in names if entry.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise RuntimeError(f"{name} wheel has an invalid METADATA set")
        metadata = archive.read(metadata_paths[0]).decode("utf-8")
        if f"Name: {name}\n" not in metadata or f"Version: {version}\n" not in metadata:
            raise RuntimeError(f"{name} wheel metadata identity mismatch")
        if "License-Expression: MIT\n" not in metadata:
            raise RuntimeError(f"{name} wheel lacks its MIT licence expression")


def _inspect_sdist(path: Path, module: str, name: str) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        _safe(names)
        suffixes = ("/LICENSE", "/README.md", "/pyproject.toml", f"/src/{module}")
        for suffix in suffixes:
            if not any(entry.endswith(suffix) for entry in names):
                raise RuntimeError(f"{name} sdist lacks {suffix.removeprefix('/')}")


def main() -> int:
    uv = os.environ.get("UV", "uv")
    with tempfile.TemporaryDirectory(prefix="kilix-system-monitor-build-") as temporary:
        output = Path(temporary)
        for name, details in PACKAGES.items():
            destination = output / name
            destination.mkdir()
            completed = subprocess.run(
                [
                    uv,
                    "build",
                    "--offline",
                    "--no-progress",
                    "--out-dir",
                    str(destination),
                    str(details["path"]),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=120,
            )
            if completed.returncode != 0:
                diagnostic = completed.stderr.decode("utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"{name} offline build failed: {diagnostic}")
            wheels = sorted(destination.glob("*.whl"))
            sdists = sorted(destination.glob("*.tar.gz"))
            if len(wheels) != 1 or len(sdists) != 1:
                raise RuntimeError(f"{name} did not produce exactly one wheel and one sdist")
            _inspect_wheel(
                wheels[0],
                str(details["module"]),
                name,
                str(details["version"]),
            )
            _inspect_sdist(sdists[0], str(details["module"]), name)
    print("PASS: offline wheel/sdist build and content inspection for 2 implemented components")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
