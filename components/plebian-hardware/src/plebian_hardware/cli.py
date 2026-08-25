"""Strict F106 command-line transport for local hardware observations."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import state
from .probe import collect


MAX_STDOUT_BYTES = 4 * 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 4096
USAGE = (
    "usage: plebian-hardware "
    "{show|inventory --json|gpu --json|refresh|diff SNAPSHOT|doctor}"
)
WARNINGS = [
    {
        "code": "UNQUALIFIED_LOCAL_OBSERVATION",
        "message": "Local redacted observation; not hardware or backend qualification evidence.",
    },
    {
        "code": "FINGERPRINTING_GRADE_LOCAL",
        "message": "Detailed hardware remains fingerprinting-grade local data even after identifier redaction.",
    },
]


Collector = Callable[[str], dict[str, Any]]


def _json_response(scope: str, document: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "plebian.cli.response/v1",
        "command": f"hardware.{scope}",
        "status": "unknown" if document.get("unknowns") else "ok",
        "warnings": WARNINGS,
        "data": document,
    }


def _format_count(value: object, unit: str) -> str:
    return f"{value} {unit}" if isinstance(value, (int, float)) else "unknown"


def _format_memory(value: object) -> str:
    if not isinstance(value, int) or value < 0:
        return "unknown"
    return f"{value / 1024**3:.1f} GiB"


def _human_summary(document: dict[str, Any]) -> str:
    cpu = document.get("cpu", {})
    memory = document.get("memory", {})
    gpus = document.get("gpus", [])
    gpu_parts: list[str] = []
    for gpu in gpus:
        vendor = gpu.get("vendor", "unknown")
        device_class = gpu.get("device_class", "unknown")
        backends = ", ".join(
            f"{backend.get('name', 'unknown')} {backend.get('status', 'unknown')}"
            for backend in gpu.get("backends", [])
        )
        gpu_parts.append(f"{vendor} {device_class}; {backends or 'backends unknown'}")
    gpu_text = " | ".join(gpu_parts) if gpu_parts else "no GPU enumerated"
    return "\n".join(
        (
            "Hardware inventory: redacted local observation",
            f"CPU: {cpu.get('architecture', 'unknown')}, "
            f"{_format_count(cpu.get('effective_cpus'), 'effective CPUs')}",
            f"Memory: {_format_memory(memory.get('total_bytes'))} total",
            f"GPU: {gpu_text}",
            "Qualification: unqualified local observation; no hardware class or backend is qualified by this output",
        )
    )


def _state_root(explicit: Path | None) -> Path:
    return state.default_state_root() if explicit is None else explicit


def _format_change(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _diff_summary(changes: list[dict[str, Any]]) -> str:
    lines = [
        "Hardware snapshot diff: redacted local change view",
        f"Changed fields: {len(changes)}",
    ]
    for change in changes:
        lines.append(
            f"{change['field']}: {_format_change(change['before'])} -> "
            f"{_format_change(change['after'])}"
        )
    lines.append("Qualification: unqualified local observation")
    return "\n".join(lines)


def dispatch(
    arguments: Sequence[str],
    collector: Collector = collect,
    *,
    state_root: Path | None = None,
) -> tuple[int, bytes, bytes]:
    """Return exit status, stdout and stderr for an exact argv tail."""
    if list(arguments) == ["show"]:
        mode = "show"
        scope = "inventory"
    elif list(arguments) == ["inventory", "--json"]:
        mode = "json"
        scope = "inventory"
    elif list(arguments) == ["gpu", "--json"]:
        mode = "json"
        scope = "gpu"
    elif list(arguments) == ["refresh"]:
        mode = "refresh"
        scope = "inventory"
    elif len(arguments) == 2 and arguments[0] == "diff":
        mode = "diff"
        scope = "inventory"
    elif list(arguments) == ["doctor"]:
        mode = "doctor"
        scope = "inventory"
    else:
        return 2, b"", (USAGE + "\n").encode("utf-8")

    try:
        document = collector(scope)
        if mode == "show":
            payload = (_human_summary(document) + "\n").encode("utf-8")
        elif mode == "json":
            payload = (
                json.dumps(
                    _json_response(scope, document),
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        elif mode == "refresh":
            state.write_snapshot(_state_root(state_root), document)
            payload = (
                "Hardware cache: refreshed private 0600 redacted snapshot\n"
                "Qualification: unqualified local observation\n"
            ).encode("utf-8")
        elif mode == "diff":
            previous = state.read_snapshot_file(Path(arguments[1]))
            payload = (
                _diff_summary(state.snapshot_changes(previous, document)) + "\n"
            ).encode("utf-8")
        else:
            state.validate_snapshot(document)
            cache = state.cache_status(_state_root(state_root))
            payload = (
                "Hardware doctor: redacted local checks\n"
                "Probe document: valid\n"
                f"Private cache: {cache}\n"
                "Startup authority: blocked pending trusted launcher\n"
                "Qualification: unqualified\n"
            ).encode("utf-8")
        if len(payload) > MAX_STDOUT_BYTES:
            raise ValueError("output boundary exceeded")
    except state.SnapshotInvalid:
        return 65, b"", b"plebian-hardware: snapshot input refused\n"
    except state.StateUnavailable:
        return 69, b"", b"plebian-hardware: private state unavailable\n"
    except Exception:
        diagnostic = b"plebian-hardware: local hardware observation failed\n"
        return 70, b"", diagnostic[:MAX_DIAGNOSTIC_BYTES]
    return 0, payload, b""


def main(arguments: Sequence[str] | None = None) -> int:
    status, stdout, stderr = dispatch(
        sys.argv[1:] if arguments is None else arguments
    )
    if stdout:
        sys.stdout.buffer.write(stdout)
    if stderr:
        sys.stderr.buffer.write(stderr)
    return status
