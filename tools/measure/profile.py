#!/usr/bin/env python3
"""Measure 1/1 provider command without turning the result into qualification.

The harness deliberately emits an unqualified ``plebian.models.profiles/v1``
catalog.  It measures process-tree RAM, elapsed time, and exact artifact and
fixture bytes.  Qualification remains a later owner/reviewer action over the
raw evidence; this command never upgrades its own output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_PROCESS_TREE = 4096
DEFAULT_TIMEOUT_SECONDS = 3600.0
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.01
FIXED_ENVIRONMENT = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
}
IDENTITY = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){0,2}$")
ARTIFACT_VERSION = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._+-]{0,127}$")
SNAPSHOT_ID = re.compile(r"^(?:fixture|local|redacted):[a-z0-9][a-z0-9._:-]{0,127}$")
TASKS = {
    "audio-codec",
    "chat",
    "embedding",
    "object-detection",
    "sound-detection",
    "stt",
    "tts",
    "vision",
}
BACKENDS = {"cpu", "cuda", "oneapi", "opencl", "rocm", "vulkan"}


class MeasurementRefused(ValueError):
    """The requested measurement crossed a fail-closed boundary."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int | None
    elapsed_ns: int
    ram_peak_bytes: int
    process_tree_complete: bool
    timed_out: bool
    executable_sha256: str


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_regular_file(path: Path, *, maximum: int | None = None) -> tuple[str, int]:
    if not path.is_absolute():
        raise MeasurementRefused("input paths must be absolute")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MeasurementRefused("input file cannot be opened without following links") from error
    digest = hashlib.sha256()
    size = 0
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise MeasurementRefused("input is not a regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if maximum is not None and size > maximum:
                raise MeasurementRefused("metadata input exceeds its byte boundary")
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _load_hardware_snapshot(path: Path) -> tuple[str, str]:
    if not path.is_absolute():
        raise MeasurementRefused("input paths must be absolute")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MeasurementRefused(
            "hardware snapshot cannot be opened without following links"
        ) from error
    chunks: list[bytes] = []
    size = 0
    digest = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise MeasurementRefused("hardware snapshot is not a regular file")
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_METADATA_BYTES:
                raise MeasurementRefused("metadata input exceeds its byte boundary")
            chunks.append(chunk)
            digest.update(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MeasurementRefused("hardware snapshot contains a duplicate key")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeError, ValueError) as error:
        raise MeasurementRefused("hardware snapshot is not strict JSON") from error
    if not isinstance(document, dict) or document.get("schema") != "plebian.hardware/v1":
        raise MeasurementRefused("hardware snapshot schema is unsupported")
    snapshot_id = document.get("snapshot_id")
    if not isinstance(snapshot_id, str) or SNAPSHOT_ID.fullmatch(snapshot_id) is None:
        raise MeasurementRefused("hardware snapshot identity is invalid")
    return snapshot_id, digest.hexdigest()


def _validated_executable(raw: str) -> str:
    path = Path(raw)
    if not path.is_absolute():
        raise MeasurementRefused("provider executable must be absolute")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise MeasurementRefused("provider executable is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise MeasurementRefused("provider executable is not an executable regular file")
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise MeasurementRefused("provider executable is group- or world-writable")
    return str(resolved)


def _read_proc_text(path: Path, maximum: int = 1024 * 1024) -> str | None:
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    if len(payload) > maximum:
        return None
    try:
        return payload.decode("ascii", errors="strict")
    except UnicodeError:
        return None


def _process_tree_rss(root_pid: int) -> tuple[int, bool]:
    pending = [root_pid]
    seen: set[int] = set()
    total = 0
    complete = True
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        if len(seen) >= MAX_PROCESS_TREE:
            return total, False
        seen.add(pid)
        status_text = _read_proc_text(Path("/proc") / str(pid) / "status", 65536)
        children_text = _read_proc_text(
            Path("/proc") / str(pid) / "task" / str(pid) / "children"
        )
        if status_text is None:
            complete = False
        else:
            for line in status_text.splitlines():
                if not line.startswith("VmRSS:"):
                    continue
                fields = line.split()
                if len(fields) == 3 and fields[1].isdigit() and fields[2] == "kB":
                    total += int(fields[1]) * 1024
                else:
                    complete = False
                break
        if children_text is None:
            complete = False
            continue
        for raw in children_text.split():
            if raw.isdigit():
                pending.append(int(raw))
            else:
                complete = False
    return total, complete


def _rusage_peak_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # Linux reports KiB; macOS reports bytes. F106 qualifies Linux only.
    return max(0, int(peak)) * 1024 if sys.platform.startswith("linux") else max(0, int(peak))


def run_provider_command(
    argv: Sequence[str],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
) -> CommandResult:
    if not argv:
        raise MeasurementRefused("provider command is empty")
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 86400:
        raise MeasurementRefused("timeout is outside the supported boundary")
    if not math.isfinite(sample_interval_seconds) or not 0.001 <= sample_interval_seconds <= 1:
        raise MeasurementRefused("sampling interval is outside the supported boundary")
    executable = _validated_executable(argv[0])
    executable_sha256, _ = _hash_regular_file(Path(executable))
    command = [executable, *argv[1:]]
    if any("\0" in item for item in command):
        raise MeasurementRefused("provider command contains a NUL byte")
    started = time.monotonic_ns()
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=FIXED_ENVIRONMENT,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as error:
        raise MeasurementRefused("provider command could not start") from error
    peak = 0
    complete = True
    samples = 0
    timed_out = False
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None:
        rss, sample_complete = _process_tree_rss(process.pid)
        samples += 1
        peak = max(peak, rss)
        complete = complete and sample_complete
        if time.monotonic() >= deadline:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
            break
        time.sleep(sample_interval_seconds)
    returncode = process.wait()
    elapsed = max(0, time.monotonic_ns() - started)
    peak = max(peak, _rusage_peak_bytes())
    return CommandResult(
        returncode=None if timed_out else returncode,
        elapsed_ns=elapsed,
        ram_peak_bytes=peak,
        process_tree_complete=complete and samples > 0,
        timed_out=timed_out,
        executable_sha256=executable_sha256,
    )


def _write_new(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or not path.name or path.name in {".", ".."}:
        raise MeasurementRefused("output paths must be absolute files")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise MeasurementRefused("output already exists or cannot be created safely") from error
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)


def _preflight_output(path: Path) -> None:
    if not path.is_absolute() or not path.name or path.name in {".", ".."}:
        raise MeasurementRefused("output paths must be absolute files")
    try:
        parent_metadata = path.parent.lstat()
    except OSError as error:
        raise MeasurementRefused("output parent is unavailable") from error
    if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_ISLNK(parent_metadata.st_mode):
        raise MeasurementRefused("output parent must be a real directory")
    if parent_metadata.st_uid != os.geteuid() or parent_metadata.st_mode & (
        stat.S_IWGRP | stat.S_IWOTH
    ):
        raise MeasurementRefused("output parent must be private and owned by this user")
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise MeasurementRefused("output cannot be inspected safely") from error
    raise MeasurementRefused("output already exists")


def _output_identity(path: Path) -> Path:
    try:
        return path.parent.resolve(strict=True) / path.name
    except OSError as error:
        raise MeasurementRefused("output parent cannot be resolved") from error


def _require_identity(label: str, value: str) -> str:
    if IDENTITY.fullmatch(value) is None:
        raise MeasurementRefused(f"{label} is not a stable identity")
    return value


def build_outputs(arguments: argparse.Namespace) -> tuple[bytes, bytes | None, CommandResult]:
    profile_id = _require_identity("profile id", arguments.profile_id)
    provider = _require_identity("provider", arguments.provider)
    artifact_id = _require_identity("artifact id", arguments.artifact_id)
    catalog_id = _require_identity("catalog id", arguments.catalog_id)
    fixture_id = _require_identity("fixture id", arguments.fixture_id)
    command_id = _require_identity("command id", arguments.command_id)
    if VERSION.fullmatch(arguments.profile_version) is None:
        raise MeasurementRefused("profile version is invalid")
    if ARTIFACT_VERSION.fullmatch(arguments.artifact_version) is None:
        raise MeasurementRefused("artifact version is invalid")
    license_decision_id = _require_identity(
        "license decision id", arguments.license_decision_id
    )
    if arguments.task not in TASKS or arguments.backend not in BACKENDS:
        raise MeasurementRefused("task or backend is unsupported")
    if not 0 <= arguments.safety_margin_basis_points <= 10000:
        raise MeasurementRefused("safety margin is outside 0..10000 basis points")
    architecture = platform.machine().lower()
    if architecture not in {"x86_64", "aarch64"}:
        raise MeasurementRefused("host architecture is outside the 0.2.1 profile boundary")

    artifact_sha256, artifact_bytes = _hash_regular_file(arguments.artifact)
    fixture_sha256, fixture_bytes = _hash_regular_file(
        arguments.fixture, maximum=MAX_METADATA_BYTES
    )
    snapshot_id, snapshot_sha256 = _load_hardware_snapshot(arguments.hardware_snapshot)
    command_argv = tuple(arguments.command)
    command_digest = _sha256(_canonical_bytes(list(command_argv)))
    result = run_provider_command(
        command_argv,
        timeout_seconds=arguments.timeout_seconds,
        sample_interval_seconds=arguments.sample_interval_ms / 1000.0,
    )
    measured_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    raw = {
        "schema": "plebian.models.raw-measurement/v1",
        "status": "timed-out"
        if result.timed_out
        else "succeeded"
        if result.returncode == 0
        else "failed",
        "measured_at": measured_at,
        "command": {
            "command_id": command_id,
            "argv_sha256": command_digest,
            "executable_sha256": result.executable_sha256,
            "environment": "fixed-clean",
            "returncode": result.returncode,
            "timed_out": result.timed_out,
        },
        "artifact": {
            "artifact_id": artifact_id,
            "version": arguments.artifact_version,
            "bytes": artifact_bytes,
            "content_sha256": artifact_sha256,
        },
        "fixture": {
            "fixture_id": fixture_id,
            "bytes": fixture_bytes,
            "content_sha256": fixture_sha256,
        },
        "hardware": {
            "snapshot_id": snapshot_id,
            "snapshot_sha256": snapshot_sha256,
        },
        "measurement": {
            "elapsed_ns": result.elapsed_ns,
            "ram_peak_bytes": result.ram_peak_bytes,
            "ram_method": "process-tree-rss-with-rusage-high-water-floor",
            "sample_interval_ms": arguments.sample_interval_ms,
            "process_tree_complete": result.process_tree_complete,
        },
    }
    raw_payload = _canonical_bytes(raw)
    if result.returncode != 0 or result.timed_out:
        return raw_payload, None, result

    profile = {
        "schema": "plebian.models.profiles/v1",
        "catalog_id": catalog_id,
        "fixture_kind": "provider-catalog",
        "qualification_eligible": False,
        "profiles": [
            {
                "profile_id": profile_id,
                "version": arguments.profile_version,
                "provider": provider,
                "task": arguments.task,
                "backend": arguments.backend,
                "artifact": {
                    "artifact_id": artifact_id,
                    "version": arguments.artifact_version,
                    "content_sha256": artifact_sha256,
                    "license_decision_id": license_decision_id,
                },
                "requirements": {
                    "architecture": architecture,
                    "download_bytes": None,
                    "disk_installed_bytes": None,
                    "temporary_bytes": None,
                    "ram_peak_bytes": result.ram_peak_bytes
                    if result.process_tree_complete
                    else None,
                    "vram_peak_bytes": None,
                },
                "performance": {
                    "first_result_ms": None,
                    "realtime_factor": None,
                    "tokens_per_second": None,
                },
                "evidence": {
                    "confidence": "measured",
                    "command": command_id,
                    "fixture": fixture_id,
                    "measured_at": measured_at,
                    "raw_evidence_sha256": _sha256(raw_payload),
                    "reference_hardware_class": snapshot_id,
                    "safety_margin_basis_points": arguments.safety_margin_basis_points,
                },
                "qualification": "unqualified",
            }
        ],
    }
    from jsonschema import Draft202012Validator, FormatChecker

    schema_path = Path(__file__).resolve().parents[2] / "contracts/p1-candidate/schemas/plebian.models.profiles-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    failures = list(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(profile)
    )
    if failures:
        raise MeasurementRefused("generated profile does not satisfy its candidate schema")
    return raw_payload, _canonical_bytes(profile), result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure one provider command and emit unqualified profile evidence."
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--profile-version", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--license-decision-id", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--hardware-snapshot", type=Path, required=True)
    parser.add_argument("--command-id", required=True)
    parser.add_argument("--safety-margin-basis-points", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--sample-interval-ms", type=float, default=10.0)
    parser.add_argument("--raw-evidence", type=Path, required=True)
    parser.add_argument("--profile-catalog", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command[:1] == ["--"]:
        arguments.command = arguments.command[1:]
    try:
        if _output_identity(arguments.raw_evidence) == _output_identity(
            arguments.profile_catalog
        ):
            raise MeasurementRefused("raw evidence and profile catalog outputs must differ")
        _preflight_output(arguments.raw_evidence)
        _preflight_output(arguments.profile_catalog)
        raw, profile, result = build_outputs(arguments)
        _write_new(arguments.raw_evidence, raw)
        if profile is None:
            print(
                "profile-measure: command 0/1, raw evidence 1/1, profile catalogs 0/1; qualification 0/1",
                file=sys.stderr,
            )
            return 124 if result.timed_out else 1
        _write_new(arguments.profile_catalog, profile)
    except (MeasurementRefused, OSError, ValueError) as error:
        print(f"profile-measure: REFUSED: {error}", file=sys.stderr)
        return 2
    print(
        "profile-measure: command 1/1, raw evidence 1/1, profile catalogs 1/1; qualification 0/1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
