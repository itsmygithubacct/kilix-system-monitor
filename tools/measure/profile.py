#!/usr/bin/env python3
"""Validate provider-owned measurement evidence without accepting its claims.

This module does not execute a provider.  It binds one canonical provider
record to the exact bytes supplied for an artifact, fixture, and redacted
hardware snapshot, then emits an explicitly unqualified profile.  Reported
measurements remain in the provider record; none are promoted into the profile.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROVIDER_SCHEMA = Path(__file__).with_name("provider-measurement-v1.schema.json")
RECEIPT_SCHEMA = Path(__file__).with_name("profile-intake-receipt-v1.schema.json")
HARDWARE_SCHEMA = ROOT / "contracts/p1-candidate/schemas/plebian.hardware-v1.schema.json"
PROFILE_SCHEMA = ROOT / "contracts/p1-candidate/schemas/plebian.models.profiles-v1.schema.json"
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
PROHIBITED_KEYS = {
    "asset_tag",
    "hostname",
    "ip_address",
    "mac_address",
    "machine_id",
    "serial",
    "serial_number",
    "system_uuid",
    "username",
    "uuid",
}
IPV4 = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
MAC = re.compile(r"(?i)(?<![0-9a-f])(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}(?![0-9a-f])")


class IntakeRefused(ValueError):
    """The supplied record failed a fail-closed intake control."""


@dataclass(frozen=True, slots=True)
class FileEvidence:
    sha256: str
    size: int
    payload: bytes | None


@dataclass(slots=True)
class OutputTarget:
    parent_fd: int
    parent_path: Path
    parent_device: int
    parent_inode: int
    name: str

    @property
    def identity(self) -> tuple[int, int, str]:
        return (self.parent_device, self.parent_inode, self.name)

    def close(self) -> None:
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1


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


def _open_directory_nofollow(path: Path) -> int:
    """Open an absolute directory through no-symlink component traversal."""
    if not path.is_absolute():
        raise IntakeRefused("all paths must be absolute")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open("/", flags)
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                os.close(descriptor)
                raise IntakeRefused("path contains an unsupported component")
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError:
                os.close(descriptor)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except IntakeRefused:
        raise
    except OSError as error:
        raise IntakeRefused("path ancestry is unavailable or contains a link") from error


def _read_regular_file(
    path: Path,
    *,
    capture: bool,
    maximum: int | None = None,
    private: bool = False,
) -> FileEvidence:
    """Read and hash one stable regular-file snapshot through a retained fd."""
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise IntakeRefused("all input paths must be absolute file paths")
    parent_fd = _open_directory_nofollow(path.parent)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        try:
            descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        except OSError as error:
            raise IntakeRefused("input is unavailable or is a link") from error
    finally:
        os.close(parent_fd)

    digest = hashlib.sha256()
    chunks: list[bytes] = []
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise IntakeRefused("input is not a regular file")
        if private and (
            before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise IntakeRefused(
                "private JSON input must be caller-owned, singly linked, and mode 0600"
            )
        if maximum is not None and before.st_size > maximum:
            raise IntakeRefused("JSON input exceeds its byte boundary")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            size += len(block)
            if maximum is not None and size > maximum:
                raise IntakeRefused("JSON input exceeds its byte boundary")
            digest.update(block)
            if capture:
                chunks.append(block)
        after = os.fstat(descriptor)
    except IntakeRefused:
        raise
    except OSError as error:
        raise IntakeRefused("input could not be read as one stable snapshot") from error
    finally:
        os.close(descriptor)

    stable_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    stable_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if stable_before != stable_after or size != after.st_size:
        raise IntakeRefused("input changed while it was being read")
    return FileEvidence(
        sha256=digest.hexdigest(),
        size=size,
        payload=b"".join(chunks) if capture else None,
    )


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        canonical = _canonical_bytes(document)
    except (UnicodeError, ValueError, TypeError) as error:
        raise IntakeRefused(f"{label} is not strict finite JSON") from error
    if not isinstance(document, dict):
        raise IntakeRefused(f"{label} must be a JSON object")
    if payload != canonical:
        raise IntakeRefused(f"{label} must use the canonical JSON encoding")
    return document


def _schema_errors(document: dict[str, Any], schema_path: Path) -> list[str]:
    from jsonschema import Draft202012Validator, FormatChecker

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    except (OSError, UnicodeError, ValueError) as error:
        raise IntakeRefused("a pinned intake schema is unavailable") from error
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        error.message
        for error in sorted(validator.iter_errors(document), key=lambda item: list(item.path))
    ]


def _privacy_errors(value: Any, trail: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized in PROHIBITED_KEYS:
                errors.append(f"forbidden identifier key at {trail}.{key}")
            errors.extend(_privacy_errors(item, f"{trail}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_privacy_errors(item, f"{trail}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if "/home/" in value or "/users/" in lowered or "c:\\users\\" in lowered:
            errors.append(f"home-directory value at {trail}")
        for match in IPV4.finditer(value):
            try:
                ipaddress.ip_address(match.group(0))
            except ValueError:
                continue
            errors.append(f"IP-address value at {trail}")
            break
        if MAC.search(value):
            errors.append(f"MAC-address value at {trail}")
        if ":" in value:
            candidate = value.removeprefix("[").removesuffix("]")
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                pass
            else:
                if address.version == 6:
                    errors.append(f"IP-address value at {trail}")
    return errors


def _validate_provider(document: dict[str, Any]) -> None:
    failures = _schema_errors(document, PROVIDER_SCHEMA)
    failures.extend(_privacy_errors(document))
    if failures:
        raise IntakeRefused(
            f"provider record failed schema/privacy validation ({len(failures)} error(s))"
        )


def _validate_hardware(document: dict[str, Any]) -> None:
    from tools.validate_candidate import semantic_errors

    schema_failures = _schema_errors(document, HARDWARE_SCHEMA)
    failures = list(schema_failures)
    failures.extend(_privacy_errors(document))
    if not schema_failures:
        failures.extend(semantic_errors("plebian.hardware/v1", document))
    if failures:
        raise IntakeRefused(
            f"hardware snapshot failed schema/privacy validation ({len(failures)} error(s))"
        )


def _open_output_target(path: Path) -> OutputTarget:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise IntakeRefused("all output paths must be absolute file paths")
    parent_fd = _open_directory_nofollow(path.parent)
    try:
        metadata = os.fstat(parent_fd)
        if metadata.st_uid != os.geteuid() or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise IntakeRefused("output parent must be caller-owned and not group/world-writable")
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as error:
            raise IntakeRefused("output target cannot be inspected") from error
        else:
            raise IntakeRefused("output target already exists")
        return OutputTarget(
            parent_fd=parent_fd,
            parent_path=path.parent,
            parent_device=metadata.st_dev,
            parent_inode=metadata.st_ino,
            name=path.name,
        )
    except Exception:
        os.close(parent_fd)
        raise


def _target_still_bound(target: OutputTarget) -> bool:
    try:
        descriptor = _open_directory_nofollow(target.parent_path)
    except IntakeRefused:
        return False
    try:
        metadata = os.fstat(descriptor)
        return (metadata.st_dev, metadata.st_ino) == (
            target.parent_device,
            target.parent_inode,
        )
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    position = 0
    while position < len(payload):
        written = os.write(descriptor, payload[position:])
        if written <= 0:
            raise OSError("short output write")
        position += written


def _write_outputs(outputs: list[tuple[OutputTarget, bytes]]) -> None:
    if len(outputs) != 2:
        raise IntakeRefused("the intake transaction requires exactly 2/2 outputs")
    targets = [target for target, _payload in outputs]
    if len({target.identity for target in targets}) != 2:
        raise IntakeRefused("the 2/2 output targets must be distinct")
    parent_identities = {
        (target.parent_device, target.parent_inode) for target in targets
    }
    if len(parent_identities) != 1:
        raise IntakeRefused("the 2/2 outputs must share one retained parent directory")
    if not all(_target_still_bound(target) for target in targets):
        raise IntakeRefused("an output parent changed after preflight")

    parent_fd = targets[0].parent_fd
    temporary_names: list[str] = []
    final_names: list[str] = []
    flags = os.O_WRONLY | os.O_CLOEXEC | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        for target, payload in outputs:
            temporary = f".{target.name}.intake-{secrets.token_hex(16)}"
            descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
            temporary_names.append(temporary)
            try:
                os.fchmod(descriptor, 0o600)
                _write_all(descriptor, payload)
                os.fsync(descriptor)
                if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
                    raise OSError("output mode verification failed")
            finally:
                os.close(descriptor)
        if not all(_target_still_bound(target) for target in targets):
            raise IntakeRefused("an output parent changed before commit")
        for temporary, target in zip(temporary_names, targets, strict=True):
            os.link(
                temporary,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            final_names.append(target.name)
        for temporary in temporary_names:
            os.unlink(temporary, dir_fd=parent_fd)
        temporary_names.clear()
        os.fsync(parent_fd)
        if not all(_target_still_bound(target) for target in targets):
            raise IntakeRefused("an output parent changed during commit")
        final_names.clear()
    except IntakeRefused:
        raise
    except OSError as error:
        raise IntakeRefused("the 2/2 output transaction failed") from error
    finally:
        for name in final_names:
            try:
                os.unlink(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        for name in temporary_names:
            try:
                os.unlink(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _build_outputs(
    provider: dict[str, Any],
    provider_file: FileEvidence,
    artifact: FileEvidence,
    fixture: FileEvidence,
    hardware: dict[str, Any],
    hardware_file: FileEvidence,
) -> tuple[bytes, bytes]:
    expected = {
        "artifact": provider["artifact"]["content_sha256"],
        "fixture": provider["fixture"]["content_sha256"],
        "hardware": provider["hardware"]["snapshot_sha256"],
    }
    observed = {
        "artifact": artifact.sha256,
        "fixture": fixture.sha256,
        "hardware": hardware_file.sha256,
    }
    if expected != observed:
        raise IntakeRefused("provider byte identities matched fewer than 3/3 inputs")
    if provider["hardware"]["snapshot_id"] != hardware["snapshot_id"]:
        raise IntakeRefused("provider hardware identity does not match the supplied snapshot")
    architecture = hardware["cpu"]["architecture"]
    if architecture not in {"aarch64", "x86_64"}:
        raise IntakeRefused("hardware architecture cannot populate the profile schema")
    if provider["reported_requirements"]["architecture"] != architecture:
        raise IntakeRefused("provider architecture does not match the supplied hardware snapshot")

    profile = {
        "schema": "plebian.models.profiles/v1",
        "catalog_id": provider["catalog_id"],
        "fixture_kind": "provider-catalog",
        "qualification_eligible": False,
        "profiles": [
            {
                "profile_id": provider["profile_id"],
                "version": provider["profile_version"],
                "provider": provider["provider"],
                "task": provider["task"],
                "backend": provider["backend"],
                "artifact": {
                    "artifact_id": provider["artifact"]["artifact_id"],
                    "version": provider["artifact"]["version"],
                    "content_sha256": artifact.sha256,
                    "license_decision_id": None,
                },
                "requirements": {
                    "architecture": architecture,
                    "download_bytes": None,
                    "disk_installed_bytes": None,
                    "temporary_bytes": None,
                    "ram_peak_bytes": None,
                    "vram_peak_bytes": None,
                },
                "performance": {
                    "first_result_ms": None,
                    "realtime_factor": None,
                    "tokens_per_second": None,
                },
                "evidence": {
                    "confidence": "unknown",
                    "command": None,
                    "fixture": None,
                    "measured_at": None,
                    "raw_evidence_sha256": provider_file.sha256,
                    "reference_hardware_class": None,
                    "safety_margin_basis_points": 0,
                },
                "qualification": "unqualified",
            }
        ],
    }
    profile_failures = _schema_errors(profile, PROFILE_SCHEMA)
    if profile_failures:
        raise IntakeRefused(
            f"generated profile failed its candidate schema ({len(profile_failures)} error(s))"
        )
    profile_payload = _canonical_bytes(profile)

    requirements = provider["reported_requirements"]
    performance = provider["reported_performance"]
    receipt = {
        "schema": "plebian.models.profile-intake-receipt/v1",
        "intake_id": provider["measurement_id"],
        "provider_records": {
            "recorded": 1,
            "total": 1,
            "accepted_as_measurement": 0,
        },
        "byte_identities": {
            "matched": 3,
            "total": 3,
            "artifact_sha256": artifact.sha256,
            "fixture_sha256": fixture.sha256,
            "hardware_sha256": hardware_file.sha256,
        },
        "validation": {
            "provider_schema": {"passed": 1, "total": 1},
            "hardware_schema_and_semantics": {"passed": 1, "total": 1},
            "privacy_documents": {"passed": 2, "total": 2},
        },
        "reported_fields": {
            "requirements": {
                "present": sum(key in requirements for key in (
                    "architecture",
                    "download_bytes",
                    "disk_installed_bytes",
                    "temporary_bytes",
                    "ram_peak_bytes",
                    "vram_peak_bytes",
                )),
                "total": 6,
            },
            "performance": {
                "present": sum(key in performance for key in (
                    "first_result_ms",
                    "realtime_factor",
                    "tokens_per_second",
                )),
                "total": 3,
            },
            "license_decision": {
                "present": int(provider["artifact"]["license_decision_id"] is not None),
                "total": 1,
            },
        },
        "promotion": {
            "provider_measurements": {"promoted": 0, "total": 9},
            "resource_metrics": {"promoted": 0, "total": 5},
            "performance_metrics": {"promoted": 0, "total": 3},
            "license_decisions": {"promoted": 0, "total": 1},
            "qualification": {"accepted": 0, "total": 1},
        },
        "measurement_boundary": {
            "accepted": 0,
            "total": 1,
            "status": "unaccepted",
        },
        "outputs": {"committed": 2, "total": 2},
        "provider_evidence_sha256": provider_file.sha256,
        "profile_catalog_sha256": _sha256(profile_payload),
        "disposition": "recorded-unqualified",
    }
    receipt_failures = _schema_errors(receipt, RECEIPT_SCHEMA)
    if receipt_failures:
        raise IntakeRefused(
            f"generated receipt failed its schema ({len(receipt_failures)} error(s))"
        )
    return _canonical_bytes(receipt), profile_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one provider-owned measurement record and emit an unqualified "
            "profile without executing provider code."
        )
    )
    parser.add_argument("--provider-evidence", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--hardware-snapshot", type=Path, required=True)
    parser.add_argument("--intake-receipt", type=Path, required=True)
    parser.add_argument("--profile-catalog", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    targets: list[OutputTarget] = []
    try:
        receipt_target = _open_output_target(arguments.intake_receipt)
        targets.append(receipt_target)
        profile_target = _open_output_target(arguments.profile_catalog)
        targets.append(profile_target)

        provider_file = _read_regular_file(
            arguments.provider_evidence,
            capture=True,
            maximum=MAX_DOCUMENT_BYTES,
            private=True,
        )
        artifact = _read_regular_file(arguments.artifact, capture=False)
        fixture = _read_regular_file(arguments.fixture, capture=False)
        hardware_file = _read_regular_file(
            arguments.hardware_snapshot,
            capture=True,
            maximum=MAX_DOCUMENT_BYTES,
            private=True,
        )
        assert provider_file.payload is not None
        assert hardware_file.payload is not None
        provider = _strict_json(provider_file.payload, "provider record")
        hardware = _strict_json(hardware_file.payload, "hardware snapshot")
        _validate_provider(provider)
        _validate_hardware(hardware)
        receipt_payload, profile_payload = _build_outputs(
            provider,
            provider_file,
            artifact,
            fixture,
            hardware,
            hardware_file,
        )
        _write_outputs(
            [
                (receipt_target, receipt_payload),
                (profile_target, profile_payload),
            ]
        )
    except IntakeRefused as error:
        print(f"profile-intake: refused: {error}", file=sys.stderr)
        return 2
    finally:
        for target in targets:
            target.close()
    print(
        "profile-intake: provider records 1/1 recorded, byte identities 3/3, "
        "outputs 2/2, measurement evidence 0/1 accepted, qualified profiles 0/1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
