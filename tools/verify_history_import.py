#!/usr/bin/env python3
"""Verify a prefix-only telemetry history import against a local source clone."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "components/kilix-telemetry/"
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "git inspection failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.stdout


def _mapping(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0].split() != ["old", "new"]:
        raise RuntimeError("commit map lacks the exact old/new header")
    result: dict[str, str] = {}
    mapped: set[str] = set()
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 2 or not all(COMMIT.fullmatch(item) for item in fields):
            raise RuntimeError("commit map has a malformed row")
        old, new = fields
        if old in result or new in mapped:
            raise RuntimeError("commit map is not one-to-one")
        result[old] = new
        mapped.add(new)
    if not result:
        raise RuntimeError("commit map is empty")
    return result


def _commit(repository: Path, commit: str) -> tuple[list[str], bytes, bytes, bytes]:
    raw = _git(repository, "cat-file", "commit", commit)
    header, separator, message = raw.partition(b"\n\n")
    if not separator:
        raise RuntimeError(f"commit object has no message boundary: {commit}")
    parents: list[str] = []
    author = b""
    committer = b""
    for line in header.splitlines():
        if line.startswith(b"parent "):
            parents.append(line.removeprefix(b"parent ").decode("ascii"))
        elif line.startswith(b"author "):
            author = line
        elif line.startswith(b"committer "):
            committer = line
    if not author or not committer:
        raise RuntimeError(f"commit object lacks identity headers: {commit}")
    return parents, author, committer, message


def _tree(repository: Path, commit: str) -> dict[str, tuple[str, str, str]]:
    raw = _git(repository, "ls-tree", "-rz", "--full-tree", commit)
    result: dict[str, tuple[str, str, str]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.decode("ascii").split()
        if not separator or len(fields) != 3:
            raise RuntimeError(f"malformed tree record in {commit}")
        path = raw_path.decode("utf-8", errors="strict")
        result[path] = (fields[0], fields[1], fields[2])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--source-ref",
        default="3affc0cc4b9a80517c452470a01e2103d29e9dbf",
    )
    parser.add_argument(
        "--map",
        dest="map_path",
        default=ROOT / "integration" / "kilix-telemetry-commit-map",
        type=Path,
    )
    parser.add_argument("--parent", default=ROOT, type=Path)
    arguments = parser.parse_args()

    mapping = _mapping(arguments.map_path)
    selected = set(
        _git(arguments.source, "rev-list", arguments.source_ref)
        .decode("ascii")
        .splitlines()
    )
    if selected != set(mapping):
        raise RuntimeError("source-ref commit set differs from the recorded map")
    tags = _git(arguments.source, "tag", "--contains", arguments.source_ref)
    if tags:
        raise RuntimeError("selected source unexpectedly has a containing tag")

    for old, new in mapping.items():
        old_parents, old_author, old_committer, old_message = _commit(
            arguments.source, old
        )
        new_parents, new_author, new_committer, new_message = _commit(
            arguments.parent, new
        )
        if new_parents != [mapping[parent] for parent in old_parents]:
            raise RuntimeError(f"mapped topology differs at {old}")
        if (old_author, old_committer, old_message) != (
            new_author,
            new_committer,
            new_message,
        ):
            raise RuntimeError(f"mapped identity, timestamp, or message differs at {old}")
        source_tree = _tree(arguments.source, old)
        parent_tree = _tree(arguments.parent, new)
        expected_tree = {PREFIX + path: value for path, value in source_tree.items()}
        if parent_tree != expected_tree:
            raise RuntimeError(f"mapped modes, blobs, or prefixed paths differ at {old}")

    print(
        f"PASS: {len(mapping)}/{len(mapping)} mapped commits preserve topology, "
        "identity, timestamps, messages, modes, blobs, and one exact component prefix"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired, UnicodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
