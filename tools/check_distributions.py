#!/usr/bin/env python3
"""Build and inspect every Python distribution this repository produces, offline.

The population is every distribution the repository can be asked to build,
including the root. The root was previously absent, and that absence is why
a commit whose own ``uv build`` returned exit 2 with 0 artifacts passed this
check at exit 0 (finding F-02). A gate that inspects only the parts that
happen to be present cannot fail on the part that is missing.

The root builds under build isolation because it uses the setuptools backend,
which is not installed in the locked environment; the components keep
``--no-build-isolation`` because their ``uv_build`` backend is. All remain
fully offline.

setuptools writes ``<name>.egg-info`` into the directory it builds from, so
the root is built from a private copy of the checkout. The check fails if any
distribution directory gains an entry while it runs, or carries egg-info,
build or dist output however old, since residue from an earlier in-place
build would otherwise already be in the starting snapshot.

The root's build backend is pinned by hash in tools/build-constraints.txt.
``--prefetch`` runs the same builds online once, so a fresh clone's uv cache
holds every hash-verified build input; the gate itself always builds offline.
"""

from __future__ import annotations

import os
import shutil
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
    # distributable units are the components below - but it must still be
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
    "kilix-device-lease": {
        "path": ROOT / "components" / "kilix-device-lease",
        "modules": ("kilix_device_lease/__init__.py",),
        "version": "1.0.0",
        "isolated_build": False,
    },
}
BUILD_BACKEND_VERSION = "0.12.5"
BUILD_CONSTRAINTS = ROOT / "tools" / "build-constraints.txt"
# Local state a checkout may carry that is never distribution source.
_NOT_SOURCE = shutil.ignore_patterns(".git", ".venv", "__pycache__", "*.egg-info", "build", "dist")
# Build output a distribution directory must never carry.
_BUILD_RESIDUE = ("*.egg-info", "build", "dist")


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


def build_source(name: str, details: dict, scratch: Path) -> Path:
    """The directory a distribution is built from: a private copy for the root."""
    if not details["isolated_build"]:
        return details["path"]
    source = scratch / "source" / name
    shutil.copytree(details["path"], source, ignore=_NOT_SOURCE, symlinks=True)
    return source


def build(uv: str, name: str, details: dict, destination: Path, scratch: Path, *,
          offline: bool = True) -> None:
    # The isolated root build resolves its backend; the constraints pin it by hash.
    isolation = (("--build-constraints", str(BUILD_CONSTRAINTS), "--require-hashes")
                 if details["isolated_build"] else ("--no-build-isolation",))
    completed = subprocess.run(
        [
            uv,
            "build",
            *(("--offline",) if offline else ()),
            *isolation,
            "--no-progress",
            "--out-dir",
            str(destination),
            str(build_source(name, details, scratch)),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"{name} {'offline' if offline else 'online prefetch'} build failed: {diagnostic}")


def snapshot(directories: list[Path]) -> dict[Path, set[str]]:
    return {directory: set(os.listdir(directory)) for directory in directories}


def residue(before: dict[Path, set[str]], after: dict[Path, set[str]], base: Path = ROOT) -> list[str]:
    left = []
    for directory, entries in after.items():
        prefix = directory.relative_to(base).as_posix()
        for entry in entries - before.get(directory, set()):
            left.append(entry if prefix == "." else f"{prefix}/{entry}")
    return sorted(left)


def build_residue(directories: list[Path], base: Path = ROOT) -> list[str]:
    found = []
    for directory in directories:
        prefix = directory.relative_to(base).as_posix()
        for pattern in _BUILD_RESIDUE:
            for path in directory.glob(pattern):
                found.append(path.name if prefix == "." else f"{prefix}/{path.name}")
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments not in ([], ["--prefetch"]):
        raise RuntimeError("usage: check_distributions.py [--prefetch]")
    offline = arguments != ["--prefetch"]
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
    watched = [Path(details["path"]) for details in PACKAGES.values()]
    before = snapshot(watched)
    with tempfile.TemporaryDirectory(prefix="kilix-system-monitor-build-") as temporary:
        output = Path(temporary)
        for name, details in PACKAGES.items():
            destination = output / name
            destination.mkdir()
            build(uv, name, details, destination, output, offline=offline)
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
    left = sorted(set(residue(before, snapshot(watched))) | set(build_residue(watched)))
    if left:
        raise RuntimeError("package-check left artefacts in the checkout: " + ", ".join(left))
    components = sum(1 for details in PACKAGES.values() if details["modules"])
    print(
        f"PASS: {'offline' if offline else 'online prefetch'} wheel/sdist build and content inspection for {len(PACKAGES)} "
        f"distributions ({components} implemented components plus the root umbrella); "
        "0 artefacts left in the checkout"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
