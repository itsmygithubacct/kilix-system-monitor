#!/usr/bin/env python3
"""Build and inspect every Python distribution this repository produces, offline.

The population is every distribution the repository can be asked to build,
including the root. The root was previously absent, and that absence is why
a commit whose own ``uv build`` returned exit 2 with 0 artifacts passed this
check at exit 0 (finding F-02). A gate that inspects only the parts that
happen to be present cannot fail on the part that is missing.

The root builds under build isolation because it uses the setuptools backend,
which is not installed in the locked environment; the two components keep
``--no-build-isolation`` because their ``uv_build`` backend is. Both remain
fully offline.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    # The root umbrella. It carries no importable module by design - the
    # distributable units are the two components below - but it must still be
    # buildable, and nothing else in this repository checks that.
    "kilix-system-monitor-contracts": {
        "path": ROOT,
        "modules": (),
        "version": "0.0.0",
        "isolated_build": True,
    },
    "kilix-telemetry": {
        "path": ROOT / "components" / "kilix-telemetry",
        "modules": ("kilix_telemetry/__init__.py",),
        "version": "0.1.2",
        "isolated_build": False,
    },
    "plebian-hardware": {
        "path": ROOT / "components" / "plebian-hardware",
        "modules": (
            "plebian_hardware/__init__.py",
            "plebian_hardware/state.py",
        ),
        "version": "0.1.0",
        "isolated_build": False,
    },
}
BUILD_BACKEND_VERSION = "0.12.5"


def _safe(names: list[str]) -> None:
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise RuntimeError(f"distribution has unsafe member: {name}")
        if any(part in {".git", ".venv", "__pycache__", "research"} for part in path.parts):
            raise RuntimeError(f"distribution has private/build member: {name}")


def _inspect_wheel(
    path: Path, modules: tuple[str, ...], name: str, version: str
) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _safe(names)
        for module in modules:
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


def _inspect_sdist(path: Path, modules: tuple[str, ...], name: str) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        _safe(names)
        suffixes = (
            "/LICENSE",
            "/README.md",
            "/pyproject.toml",
            *(f"/src/{module}" for module in modules),
        )
        for suffix in suffixes:
            if not any(entry.endswith(suffix) for entry in names):
                raise RuntimeError(f"{name} sdist lacks {suffix.removeprefix('/')}")


def main() -> int:
    uv = os.environ.get("UV", "uv")
    try:
        observed_backend = distribution_version("uv-build")
    except PackageNotFoundError as error:
        raise RuntimeError("locked uv-build backend is absent") from error
    if observed_backend != BUILD_BACKEND_VERSION:
        raise RuntimeError(
            "uv-build backend mismatch: "
            f"expected {BUILD_BACKEND_VERSION}, observed {observed_backend}"
        )
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
                    *((), ("--no-build-isolation",))[not details["isolated_build"]],
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
                tuple(str(module) for module in details["modules"]),
                name,
                str(details["version"]),
            )
            _inspect_sdist(
                sdists[0],
                tuple(str(module) for module in details["modules"]),
                name,
            )
    components = sum(1 for details in PACKAGES.values() if details["modules"])
    print(
        f"PASS: offline wheel/sdist build and content inspection for {len(PACKAGES)} "
        f"distributions ({components} implemented components plus the root umbrella)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
