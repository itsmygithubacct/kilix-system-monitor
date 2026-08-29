#!/usr/bin/env python3
"""Validate the installed pre-freeze Track D P1 contract candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT: Path
SCHEMA_PATHS: dict[str, Path]
FIXTURE_GROUPS: dict[str, Path]
Draft202012Validator: Any
FormatChecker: Any
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
EXPECTED_PYTHON_SHA256 = "0dc3a692fa85fcdb7f1a5877d2adf179809ac417a07ffde2373c832863800a15"
SCHEMA_RELATIVE_PATHS = {
    "f106.invocation-contract/v0-draft": "schemas/f106.invocation-contract-v0-draft.schema.json",
    "plebian.hardware/v1": "schemas/plebian.hardware-v1.schema.json",
    "plebian.hardware.privacy/v1": "schemas/plebian.hardware.privacy-v1.schema.json",
    "kilix.telemetry/schema-1-vnext": "schemas/kilix.telemetry-vnext.schema.json",
    "plebian.models.checkpoint-license/v1": "schemas/plebian.models.checkpoint-license-v1.schema.json",
    "plebian.models.license-evidence-set/v1": "schemas/plebian.models.license-evidence-set-v1.schema.json",
    "plebian.models.profiles/v1": "schemas/plebian.models.profiles-v1.schema.json",
    "plebian.models.fit-result/v1": "schemas/plebian.models.fit-result-v1.schema.json",
    "plebian.models.install-plan/v1": "schemas/plebian.models.install-plan-v1.schema.json",
    "plebian.models.snapshot/v1": "schemas/plebian.models.snapshot-v1.schema.json",
    "plebian.cli.response/v1": "schemas/plebian.cli.response-v1.schema.json",
}
FIXTURE_GROUP_RELATIVE_PATHS = {
    "plebian.hardware/v1": "fixtures/hardware",
    "plebian.hardware.privacy/v1": "fixtures/privacy",
    "kilix.telemetry/schema-1-vnext": "fixtures/telemetry-vnext-additive.json",
    "plebian.models.checkpoint-license/v1": "fixtures/checkpoint-licenses",
    "plebian.models.license-evidence-set/v1": "fixtures/license-evidence",
    "plebian.models.profiles/v1": "fixtures/profiles",
    "plebian.models.fit-result/v1": "fixtures/fit",
    "plebian.models.install-plan/v1": "fixtures/plans",
    "plebian.cli.response/v1": "fixtures/responses",
}
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
MAC = re.compile(r"(?i)(?<![0-9a-f])(?:[0-9a-f]{2}:){5}[0-9a-f]{2}(?![0-9a-f])")
EXPECTED_OBSERVATION_PRIVACY = {
    "cache_mode": "0600",
    "classification": "fingerprinting-grade-local",
    "export_requires_warning": True,
    "telemetry_eligible": False,
}
EXPECTED_SUBPROCESS_PRIVACY = {
    "environment": "fixed-clean",
    "executable_integrity": "resolved-within-path-and-not-group-or-world-writable",
    "path_resolution": "fixed-system-path",
    "shell": False,
    "stderr": "discard",
    "stdin": "dev-null",
    "stdout_bytes": 65536,
    "timeout_seconds": 5,
}
CAMPP_AUTHORITY_SHA256 = "54b36539688fe450074a0105a2e43719837c321e4cf0eff8d7884a2d92ad21ad"
APACHE_2_LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
F120_LICENSE_NOTICE_GAP_SHA256 = "f9731fffeb8a240ca05115375f36c4af1df51a8d08b0a9252d728a4b7b5e3c53"
LICENSE_EVIDENCE_PINS = {
    "44f5b2243328c5ba476cae3119fec348572df4190a7e67c5a2a015719edb0c47": {
        "artifact_count": 38,
        "artifact_set_sha256": "0f87ecd6388901210e4a19635df06c168c2131e4ad34c3a528c51da276a4a8b0",
        "authority_id": "f104-qwen-asr-weights-2026-08-25",
        "stream": "f104",
    },
    "e312ea69a4e55ca627d80dfddea6f2a03bdf41ea2eccf8872fda4b33c825b16f": {
        "artifact_count": 3,
        "artifact_set_sha256": "df50345f794a90d3baccaae0adc3804abf87da0baa5aea7d5773e21695f196cc",
        "authority_id": "f105-ollama-selected-weights-2026-08-25",
        "stream": "f105",
    },
}
CAMPP_PINS = {
    "iic/speech_campplus_sv_en_voxceleb_16k": {
        "artifact": {
            "bytes": 29_357_703,
            "checkpoint_filename": "campplus_voxceleb.bin",
            "content_sha256": "5b1a88b6f8d85826fabef804779c3372b42f3af21457fa48bd5c097c0686b2de",
            "model_card_sha256": "5219533788a85bd3dc7a2c2175d1f727906676927c17919a7c6061bf0f3131f7",
            "provider": "iic-modelscope",
            "source_commit": "032b8131a7ad812f87061955ca974c99060c5a03",
            "source_revision_label": None,
            "source_tree": "e7a3d915a1b6e7d8ea1eaba68d8472874e1a6145",
        },
        "decision_id": "f104-campp-english-voxceleb-2026-08-25",
        "disposition": "cleared-for-comparison",
    },
    "iic/speech_campplus_sv_zh-cn_16k-common": {
        "artifact": {
            "bytes": 28_036_335,
            "checkpoint_filename": "campplus_cn_common.bin",
            "content_sha256": "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8",
            "model_card_sha256": "9bd58b989a534fa0121c8d10e62c0cce9080a9a9aeb2499bb429676f9c17cc3a",
            "provider": "iic-modelscope",
            "source_commit": "930086088e5c5fa8a3f911c20795ca72f7f16397",
            "source_revision_label": "v2.0.2",
            "source_tree": "d58e716798173af958ac944d933f0bb3d27ead58",
        },
        "decision_id": "f104-campp-chinese-common-2026-08-25",
        "disposition": "eligible-unselected",
    },
}


class ValidationFailure(ValueError):
    pass


def configure_boundary(candidate_root: Path, site_packages: Path) -> None:
    global ROOT, SCHEMA_PATHS, FIXTURE_GROUPS, Draft202012Validator, FormatChecker

    if not sys.flags.isolated:
        raise ValidationFailure("Python isolated mode is not active")
    if not sys.flags.no_site:
        raise ValidationFailure("Python no-site mode is not active")
    if not sys.dont_write_bytecode:
        raise ValidationFailure("Python bytecode suppression is not active")
    if sys.version_info[:3] != (3, 12, 8):
        raise ValidationFailure(
            "Python version mismatch: expected 3.12.8, observed "
            + ".".join(str(part) for part in sys.version_info[:3])
        )
    executable = Path(sys.executable).resolve(strict=True)
    if hashlib.sha256(executable.read_bytes()).hexdigest() != EXPECTED_PYTHON_SHA256:
        raise ValidationFailure("Python executable digest mismatch")
    forbidden_environment = {
        "BASH_ENV",
        "ENV",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONPLATLIBDIR",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    }
    present = sorted(forbidden_environment.intersection(os.environ))
    if present:
        raise ValidationFailure(
            "forbidden startup environment variables reached the validator: "
            + ", ".join(present)
        )

    ROOT = candidate_root.resolve(strict=True)
    resolved_site_packages = site_packages.resolve(strict=True)
    if not ROOT.is_dir():
        raise ValidationFailure("candidate root is not a directory")
    if not resolved_site_packages.is_dir():
        raise ValidationFailure("locked site-packages path is not a directory")
    if ROOT == resolved_site_packages or ROOT in resolved_site_packages.parents:
        raise ValidationFailure("locked dependencies are inside the candidate root")

    cwd = Path.cwd().resolve(strict=True)
    if cwd == ROOT or ROOT in cwd.parents or cwd in ROOT.parents:
        raise ValidationFailure("authority cwd overlaps the candidate root")
    if any(cwd.iterdir()):
        raise ValidationFailure("authority cwd is not empty")
    for entry in sys.path:
        if not entry:
            raise ValidationFailure("an empty import-path entry reached the validator")
        try:
            resolved_entry = Path(entry).resolve(strict=False)
        except OSError as error:
            raise ValidationFailure(f"cannot resolve import-path entry: {error}") from error
        if resolved_entry == ROOT or ROOT in resolved_entry.parents or resolved_entry in ROOT.parents:
            raise ValidationFailure("candidate root is import-visible before validation")

    sys.path.append(str(resolved_site_packages))
    try:
        from jsonschema import Draft202012Validator as Validator
        from jsonschema import FormatChecker as Checker
    except ImportError as error:
        raise ValidationFailure(f"locked jsonschema environment is unavailable: {error}") from error
    Draft202012Validator = Validator
    FormatChecker = Checker
    SCHEMA_PATHS = {
        identity: ROOT / relative
        for identity, relative in SCHEMA_RELATIVE_PATHS.items()
    }
    FIXTURE_GROUPS = {
        identity: ROOT / relative
        for identity, relative in FIXTURE_GROUP_RELATIVE_PATHS.items()
    }


def load_json(path: Path) -> Any:
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise ValidationFailure(f"{path}: document exceeds {MAX_DOCUMENT_BYTES} bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationFailure(f"{path}: duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> Any:
        raise ValidationFailure(f"{path}: non-finite JSON number")

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationFailure(f"{path}: invalid UTF-8 JSON: {error}") from error


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def compact_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def strict_json_parser_controls() -> tuple[list[str], int]:
    controls = {
        "duplicate-top-level": b'{"schema":"first","schema":"second"}\n',
        "duplicate-nested": b'{"outer":{"value":1,"value":2}}\n',
        "not-a-number": b'{"value":NaN}\n',
        "positive-infinity": b'{"value":Infinity}\n',
        "trailing-document": b'{}\n{}\n',
        "invalid-utf8": b'{"value":"\xff"}\n',
    }
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="f106-json-controls-") as directory:
        control_root = Path(directory)
        for name, payload in controls.items():
            path = control_root / f"{name}.json"
            path.write_bytes(payload)
            try:
                load_json(path)
            except ValidationFailure:
                continue
            errors.append(f"strict JSON parser accepted {name} control")
    return errors, len(controls)


def verify_candidate_integrity(expected_manifest_sha256: str) -> tuple[list[str], int]:
    errors: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_symlink():
            errors.append(f"candidate tree contains a symlink: {path.relative_to(ROOT)}")
        elif not path.is_file() and not path.is_dir():
            errors.append(f"candidate tree contains a non-file entry: {path.relative_to(ROOT)}")
    for path in sorted(ROOT.rglob("*.json")):
        if path.read_bytes() != canonical_bytes(load_json(path)):
            errors.append(f"non-canonical JSON: {path.relative_to(ROOT)}")
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.name != "CANDIDATE-SHA256SUMS"
    )
    expected = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.relative_to(ROOT)}\n"
        for path in files
    )
    try:
        manifest_path = ROOT / "CANDIDATE-SHA256SUMS"
        manifest_bytes = manifest_path.read_bytes()
        actual = manifest_bytes.decode("ascii")
    except (OSError, UnicodeError) as error:
        errors.append(f"cannot read CANDIDATE-SHA256SUMS: {error}")
    else:
        if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
            errors.append("CANDIDATE-SHA256SUMS differs from the external authority pin")
        if actual != expected:
            errors.append("CANDIDATE-SHA256SUMS does not match the complete candidate tree")
    return errors, len(files)


def validators() -> dict[str, Draft202012Validator]:
    result: dict[str, Draft202012Validator] = {}
    for identity, path in SCHEMA_PATHS.items():
        schema = load_json(path)
        Draft202012Validator.check_schema(schema)
        result[identity] = Draft202012Validator(schema, format_checker=FormatChecker())
    return result


def privacy_errors(value: Any, trail: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized in PROHIBITED_KEYS:
                errors.append(f"forbidden identifier key at {trail}.{key}: {normalized}")
            errors.extend(privacy_errors(item, f"{trail}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(privacy_errors(item, f"{trail}[{index}]"))
    elif isinstance(value, str):
        if "/home/" in value or IPV4.search(value) or MAC.search(value):
            errors.append(f"identifier-shaped value at {trail}")
        else:
            try:
                ipaddress.ip_address(value.removeprefix("[").removesuffix("]"))
            except ValueError:
                pass
            else:
                errors.append(f"IP-address value at {trail}")
    return errors


def semantic_errors(identity: str, document: dict[str, Any]) -> list[str]:
    errors = privacy_errors(document)
    if identity == "f106.invocation-contract/v0-draft":
        commands = document.get("commands", [])
        command_ids = [command.get("command_id") for command in commands]
        if len(command_ids) != len(set(command_ids)):
            errors.append("duplicate invocation command_id")
        argv_vectors = [tuple(command.get("argv", [])) for command in commands]
        if len(argv_vectors) != len(set(argv_vectors)):
            errors.append("duplicate invocation argv")
        for index, command in enumerate(commands):
            argv = command.get("argv", [])
            if not argv or "/" in argv[0]:
                errors.append(f"commands[{index}] must invoke a PATH-resolved program without a shell")
            fixture_value = command.get("fixture")
            if not isinstance(fixture_value, str):
                continue
            fixture = Path(fixture_value)
            if fixture.is_absolute() or ".." in fixture.parts:
                errors.append(f"commands[{index}] fixture path escapes the candidate root")
                continue
            resolved = (ROOT / fixture).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.is_file():
                errors.append(f"commands[{index}] fixture is absent or outside the candidate root")
            if command.get("stdout") == "text" and command.get("data_schema") is not None:
                errors.append(f"commands[{index}] text output cannot declare a data schema")
            if command.get("stdout") == "plebian.cli.response/v1" and command.get("data_schema") is None:
                errors.append(f"commands[{index}] JSON output has no data schema")
        return errors

    if identity == "plebian.hardware/v1":
        expected_never = sorted(PROHIBITED_KEYS - {"serial", "uuid"})
        if document.get("never_collected") != expected_never:
            errors.append("never_collected must be the frozen sorted denylist")
        if "privacy" in document and document.get("privacy") != EXPECTED_OBSERVATION_PRIVACY:
            errors.append("hardware privacy projection differs from the default privacy contract")
        capture = document.get("capture", {})
        if capture.get("source") in {"synthetic-contract", "redacted-observation"} and capture.get("qualification_eligible"):
            errors.append("synthetic or redacted fixture cannot qualify hardware")
        if capture.get("qualification_eligible") and any(
            gpu.get("vendor") == "amd"
            or any(backend.get("name") == "rocm" for backend in gpu.get("backends", []))
            for gpu in document.get("gpus", [])
        ):
            errors.append("AMD/ROCm cannot qualify in the 0.2.1 boundary")
        unknowns = document.get("unknowns", [])
        if unknowns != sorted(set(unknowns)):
            errors.append("hardware unknowns must be sorted and unique")
        evidence = document.get("evidence", [])
        evidence_prefixes = [item.get("field_prefix") for item in evidence]
        if evidence_prefixes != sorted(set(evidence_prefixes)):
            errors.append("hardware evidence prefixes must be sorted and unique")
        if capture.get("source") in {"live-probe", "redacted-observation"}:
            required_observation_sections = {
                "buses",
                "evidence",
                "network",
                "platform",
                "privacy",
                "thermal",
            }
            missing = sorted(required_observation_sections - document.keys())
            if missing:
                errors.append("observed hardware lacks expanded sections: " + ", ".join(missing))
            storage = document.get("storage", {})
            for field in ("filesystem_type", "read_only", "total_bytes"):
                if field not in storage:
                    errors.append(f"observed hardware storage lacks {field}")
            cpu = document.get("cpu", {})
            for field in ("cache_bytes", "frequency_hz"):
                if field not in cpu:
                    errors.append(f"observed hardware CPU lacks {field}")
        gpu_indices = [gpu.get("index") for gpu in document.get("gpus", [])]
        if gpu_indices != list(range(len(gpu_indices))):
            errors.append("hardware GPU indexes must be contiguous")
        for gpu_index, gpu in enumerate(document.get("gpus", [])):
            names = [item.get("name") for item in gpu.get("backends", [])]
            if names != sorted(set(names)):
                errors.append(f"gpu {gpu_index} backends must be sorted and unique")
            for backend in gpu.get("backends", []):
                if backend.get("status") == "available" and backend.get("evidence") not in {
                    "api-query",
                    "device-open",
                    "executable-probe",
                    "runtime-query",
                }:
                    errors.append("available backend lacks a successful runtime probe")
                if backend.get("evidence") in {"command-unavailable", "contradictory", "unknown"} and backend.get("status") != "unknown":
                    errors.append("unavailable probe must produce unknown backend state")
        interfaces = document.get("network", {}).get("interfaces", [])
        interface_indices = [interface.get("index") for interface in interfaces]
        if interface_indices != list(range(len(interface_indices))):
            errors.append("hardware network indexes must be contiguous")
        batteries = document.get("power", {}).get("batteries", [])
        battery_indices = [battery.get("index") for battery in batteries]
        if battery_indices != list(range(len(battery_indices))):
            errors.append("hardware battery indexes must be contiguous")
        return errors

    if identity == "plebian.hardware.privacy/v1":
        expected_never = sorted(PROHIBITED_KEYS - {"serial", "uuid"})
        if document.get("collection", {}).get("never_collected") != expected_never:
            errors.append("privacy contract never_collected differs from the hardware denylist")
        if document.get("observation_projection") != EXPECTED_OBSERVATION_PRIVACY:
            errors.append("privacy contract observation projection is not exact")
        if document.get("subprocess") != EXPECTED_SUBPROCESS_PRIVACY:
            errors.append("privacy contract subprocess boundary is not exact")
        return errors

    if identity == "plebian.models.checkpoint-license/v1":
        artifact = document.get("artifact", {})
        decision = document.get("decision", {})
        model_id = artifact.get("model_id")
        if isinstance(model_id, str) and model_id.startswith("iic/speech_campplus"):
            if decision.get("wildcard_clearance") is not False:
                errors.append("CAM++ wildcard clearance is forbidden")
            pin = CAMPP_PINS.get(model_id)
            if pin is None or any(
                artifact.get(field) != value
                for field, value in (pin.get("artifact") or {}).items()
            ):
                errors.append("CAM++ checkpoint is not bound to the digest-specific determination")
            elif (
                decision.get("decision_id") != pin.get("decision_id")
                or decision.get("disposition") != pin.get("disposition")
            ):
                errors.append("CAM++ disposition differs from the digest-specific determination")
        if decision.get("authority_document_sha256") != CAMPP_AUTHORITY_SHA256:
            errors.append("checkpoint decision is not bound to the reviewed authority document")
        if decision.get("license_expression") == "Apache-2.0" and decision.get("license_text_sha256") != APACHE_2_LICENSE_SHA256:
            errors.append("Apache-2.0 decision carries the wrong license-text digest")
        if decision.get("disposition") in {"cleared-for-comparison", "eligible-unselected"} and decision.get("redistributable") is not True:
            errors.append("cleared checkpoint is not marked redistributable")
        return errors

    if identity == "plebian.models.license-evidence-set/v1":
        authority = document.get("authority", {})
        artifacts = document.get("artifacts", [])
        delivery = document.get("delivery", {})
        if authority.get("wildcard_inheritance") is not False:
            errors.append("license evidence wildcard inheritance is forbidden")
        decision_ids = [artifact.get("decision_id") for artifact in artifacts]
        artifact_ids = [artifact.get("artifact_id") for artifact in artifacts]
        if decision_ids != sorted(set(decision_ids)):
            errors.append("license evidence decision IDs must be sorted and unique")
        if len(artifact_ids) != len(set(artifact_ids)):
            errors.append("license evidence artifact IDs must be unique")
        for index, artifact in enumerate(artifacts):
            for field in ("license_expressions", "license_text_sha256s"):
                values = artifact.get(field, [])
                if values != sorted(set(values)):
                    errors.append(f"license evidence artifacts[{index}].{field} must be sorted and unique")
        artifact_set_sha256 = compact_sha256(artifacts)
        if document.get("artifact_set_sha256") != artifact_set_sha256:
            errors.append("license evidence artifact-set self-digest is invalid")
        pin = LICENSE_EVIDENCE_PINS.get(authority.get("authority_document_sha256"))
        if pin is None or any(
            (
                authority.get("authority_id") != pin.get("authority_id"),
                authority.get("stream") != pin.get("stream"),
                len(artifacts) != pin.get("artifact_count"),
                artifact_set_sha256 != pin.get("artifact_set_sha256"),
            )
        ):
            errors.append("license evidence artifact set differs from reviewed authority")
        if delivery.get("f120_gap_authority_sha256") != F120_LICENSE_NOTICE_GAP_SHA256:
            errors.append("license evidence is not bound to the F120 conveyance finding")
        if delivery.get("notice_transport") == "pending-f120-authority-disposition":
            if (
                delivery.get("staged_license_artifact_identity") is not None
                or delivery.get("staged_notice_paths")
            ):
                errors.append("pending notice transport carries staged identities")
            if delivery.get("transfer_eligible") is not False:
                errors.append("pending notice transport cannot authorize transfer")
        return errors

    if identity == "plebian.models.profiles/v1":
        if document.get("fixture_kind") == "synthetic-contract" and document.get("qualification_eligible"):
            errors.append("synthetic profile catalog cannot be qualification eligible")
        for profile in document.get("profiles", []):
            evidence = profile.get("evidence", {})
            performance = profile.get("performance", {})
            has_performance = any(value is not None for value in performance.values())
            if has_performance and evidence.get("confidence") != "measured":
                errors.append("performance number lacks measured evidence")
            if profile.get("qualification") == "qualified":
                required = (
                    document.get("qualification_eligible"),
                    evidence.get("confidence") == "measured",
                    evidence.get("command") is not None,
                    evidence.get("fixture") is not None,
                    evidence.get("measured_at") is not None,
                    evidence.get("raw_evidence_sha256") is not None,
                    evidence.get("reference_hardware_class") is not None,
                    profile.get("artifact", {}).get("content_sha256") is not None,
                    profile.get("artifact", {}).get("license_decision_id") is not None,
                )
                if not all(required):
                    errors.append("qualified profile lacks measured evidence")
        return errors

    if identity == "plebian.models.fit-result/v1":
        capacity = document.get("capacity_contract", {})
        positive = {"does-not-fit", "fits-tightly", "fits", "recommended"}
        if capacity.get("status") == "missing":
            if document.get("overall_verdict") in positive:
                errors.append("positive fit verdict without resolved F100-C0")
            if any(capacity.get(key) is not None for key in ("identity", "source_commit", "source_sha256")):
                errors.append("missing capacity contract carries invented identity")
            for resource in document.get("resources", []):
                if resource.get("available_bytes") is not None or resource.get("reserve_bytes") is not None or resource.get("verdict") != "unknown":
                    errors.append("missing F100-C0 must leave capacity arithmetic unknown")
        performance = document.get("performance", {})
        numbers = [performance.get(name) for name in ("first_result_ms", "realtime_factor", "tokens_per_second")]
        if any(value is not None for value in numbers) and performance.get("comparable") is not True:
            errors.append("performance number is not bound to comparable hardware")
        if document.get("overall_verdict") == "recommended":
            if not (
                document.get("qualification_eligible")
                and capacity.get("status") == "resolved"
                and performance.get("verdict") == "recommended"
                and performance.get("comparable") is True
            ):
                errors.append("recommendation lacks qualified comparable evidence")
        return errors

    if identity == "plebian.models.install-plan/v1":
        status = document.get("status")
        capacity = document.get("capacity_contract_status")
        if status == "blocked" and document.get("executable"):
            errors.append("blocked plan is executable")
        if capacity == "missing":
            if status == "ready" or document.get("executable"):
                errors.append("ready plan lacks authority")
            if any(value is not None for value in document.get("totals", {}).values()):
                errors.append("missing F100-C0 must leave plan totals unknown")
        if status == "ready":
            if not document.get("items") or not all(item.get("present") and item.get("receipt_sha256") for item in document.get("authorization_receipts", [])):
                errors.append("ready plan lacks receipts or selected items")
        if document.get("confirmation", {}).get("granted") is not False:
            errors.append("plan document must not synthesize user confirmation")
        return errors

    if identity == "plebian.models.snapshot/v1":
        if document.get("capacity_contract", {}).get("status") == "missing" and document.get("qualification_eligible"):
            errors.append("snapshot without F100-C0 cannot qualify")
        unknowns = document.get("unknowns", [])
        if unknowns != sorted(set(unknowns)):
            errors.append("snapshot unknowns must be sorted and unique")
        return errors

    return errors


def schema_identity(document: dict[str, Any], fallback: str | None = None) -> str:
    identity = document.get("schema")
    if identity == 1 and fallback == "kilix.telemetry/schema-1-vnext":
        return fallback
    if not isinstance(identity, str):
        raise ValidationFailure("document has no supported schema identity")
    return identity


def validate_document(identity: str, document: dict[str, Any], available: dict[str, Draft202012Validator]) -> list[str]:
    errors = [
        f"schema: {error.message}"
        for error in sorted(available[identity].iter_errors(document), key=lambda item: list(item.path))
    ]
    errors.extend(semantic_errors(identity, document))
    return errors


def fixture_paths(group: Path) -> list[Path]:
    return [group] if group.is_file() else sorted(group.glob("*.json"))


def validate_response(document: dict[str, Any], contract_by_command: dict[str, dict[str, Any]], available: dict[str, Draft202012Validator]) -> list[str]:
    errors = validate_document("plebian.cli.response/v1", document, available)
    command = document.get("command")
    entry = contract_by_command.get(command)
    if entry is None:
        errors.append(f"response command is absent from invocation contract: {command}")
        return errors
    data = document.get("data")
    if isinstance(data, dict):
        identity = entry.get("data_schema")
        if identity not in available:
            errors.append(f"invocation data schema is unsupported: {identity}")
        else:
            errors.extend(f"data: {error}" for error in validate_document(identity, data, available))
    else:
        return errors
    if command.startswith("sizer.plan") and document.get("status") != data.get("status"):
        errors.append("plan response status does not match plan status")
    if command == "sizer.install" and data.get("operation") != "install":
        errors.append("install response does not identify install operation")
    if command == "hardware.gpu" and data.get("capture", {}).get("scope") != "gpu":
        errors.append("GPU response does not carry GPU scope")
    return errors


def set_pointer(document: Any, pointer: str, value: Any) -> None:
    if not pointer.startswith("/"):
        raise ValidationFailure(f"invalid mutation pointer: {pointer}")
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]
    target = document
    for token in tokens[:-1]:
        target = target[int(token)] if isinstance(target, list) else target[token]
    final = tokens[-1]
    if isinstance(target, list):
        target[int(final)] = value
    else:
        target[final] = value


def run_replay_checks(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    replay = ROOT / "tools" / "replay-bin"
    plan = ROOT / "fixtures" / "plans" / "blocked-no-f100-c0.json"
    environment = {
        "LANG": contract["environment"]["locale"],
        "LC_ALL": contract["environment"]["locale"],
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    timeout = contract["limits"]["read_only_timeout_seconds"]
    stdout_limit = contract["limits"]["stdout_bytes"]
    diagnostic_limit = contract["limits"]["diagnostic_bytes"]
    for command in contract["commands"]:
        argv = [str(replay / command["argv"][0]), *command["argv"][1:]]
        argv = [str(plan) if value == "PLAN_PATH" else value for value in argv]
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=environment,
        )
        expected = (ROOT / command["fixture"]).read_bytes()
        if completed.returncode != 0:
            errors.append(f"{command['command_id']} replay exit {completed.returncode}, expected 0")
        if completed.stdout != expected:
            errors.append(f"{command['command_id']} replay stdout differs from fixture")
        if len(completed.stdout) > stdout_limit:
            errors.append(f"{command['command_id']} replay stdout exceeds the contract limit")
        if completed.stderr:
            errors.append(f"{command['command_id']} replay wrote stderr on success")

    def check_failure(label: str, argv: list[str], expected_status: int) -> None:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=environment,
        )
        if completed.returncode != expected_status:
            errors.append(f"{label} replay exit {completed.returncode}, expected {expected_status}")
        if completed.stdout:
            errors.append(f"{label} replay wrote stdout on failure")
        if (
            not completed.stderr.endswith(b"\n")
            or b"\n" in completed.stderr[:-1]
            or len(completed.stderr) > diagnostic_limit
        ):
            errors.append(f"{label} replay violates the bounded one-line stderr contract")
        try:
            diagnostic = completed.stderr.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{label} replay stderr is not UTF-8")
        else:
            if privacy_errors(diagnostic):
                errors.append(f"{label} replay stderr contains identifier-shaped data")

    check_failure(
        "unsupported-command",
        [str(replay / "plebian-hardware"), "unsupported"],
        2,
    )
    check_failure(
        "wrong-argument-order",
        [
            str(replay / "plebian-model-sizer"),
            "plan",
            "--json",
            "local-ai-balanced",
        ],
        2,
    )
    with tempfile.TemporaryDirectory(prefix="f106-contract-") as temporary:
        temporary_root = Path(temporary)
        missing = temporary_root / "missing-plan.json"
        malformed = temporary_root / "malformed-plan.json"
        duplicate = temporary_root / "duplicate-plan.json"
        wrong_schema = temporary_root / "wrong-schema-plan.json"
        oversized = temporary_root / "oversized-plan.json"
        malformed.write_text("{\n", encoding="utf-8")
        duplicate.write_text(
            '{"schema":"plebian.models.install-plan/v1","schema":"plebian.models.install-plan/v1"}\n',
            encoding="utf-8",
        )
        wrong_schema.write_text('{"schema":"unsupported/v1"}\n', encoding="utf-8")
        with oversized.open("wb") as handle:
            handle.truncate(MAX_DOCUMENT_BYTES + 1)
        for label, path in (
            ("missing-plan", missing),
            ("malformed-plan", malformed),
            ("duplicate-key-plan", duplicate),
            ("wrong-schema-plan", wrong_schema),
            ("oversized-plan", oversized),
        ):
            check_failure(
                label,
                [str(replay / "plebian-model-sizer"), "install", str(path), "--json"],
                65,
            )
    return errors


def _control_hook(marker: Path) -> str:
    return (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed\\n', encoding='utf-8')\n"
        "print('FORGED-P1-PASS')\n"
    )


def _run_launcher(
    launcher: Path,
    root: Path,
    uv: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    child_environment = dict(environment)
    child_environment["UV"] = str(uv)
    child_environment["PYTHONPATH"] = f":{root}"
    return subprocess.run(
        ["/bin/sh", str(launcher), "--startup-control-child"],
        cwd=root,
        env=child_environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )


def startup_boundary_controls(launcher: Path, uv: Path) -> list[str]:
    errors: list[str] = []
    repository = ROOT.parents[1]
    launcher = launcher.resolve(strict=True)
    uv = uv.resolve(strict=True)
    if launcher.parent != repository / "tools":
        return ["startup-control launcher is not the repository's external tool"]
    if launcher == ROOT or ROOT in launcher.parents:
        return ["startup-control launcher is inside the candidate root"]

    def expect_failure(
        label: str,
        control_launcher: Path,
        control_root: Path,
        expected: bytes,
        environment: dict[str, str],
    ) -> None:
        completed = _run_launcher(
            control_launcher,
            control_root,
            uv,
            environment,
        )
        if completed.returncode == 0:
            errors.append(f"{label}: altered launch unexpectedly passed")
        if expected not in completed.stderr:
            errors.append(
                f"{label}: missing fail-closed diagnostic {expected.decode('ascii')!r}"
            )
        if b"FORGED-P1-PASS" in completed.stdout + completed.stderr:
            errors.append(f"{label}: hostile startup hook forged output")

    with tempfile.TemporaryDirectory(prefix="kilix-p1-startup-control-") as temporary:
        temporary_root = Path(temporary)
        control_root = temporary_root / "repository"
        (control_root / "tools").mkdir(parents=True)
        (control_root / "contracts").mkdir()
        shutil.copy2(repository / "pyproject.toml", control_root / "pyproject.toml")
        shutil.copy2(repository / "uv.lock", control_root / "uv.lock")
        shutil.copy2(launcher, control_root / "tools" / "validate_candidate")
        shutil.copy2(
            repository / "tools" / "validate_candidate.py",
            control_root / "tools" / "validate_candidate.py",
        )
        shutil.copytree(ROOT, control_root / "contracts" / "p1-candidate")
        shutil.copytree(repository / ".venv", control_root / ".venv", symlinks=True)

        root_site_marker = temporary_root / "root-site-marker"
        root_user_marker = temporary_root / "root-user-marker"
        (control_root / "sitecustomize.py").write_text(
            _control_hook(root_site_marker), encoding="utf-8"
        )
        (control_root / "usercustomize.py").write_text(
            _control_hook(root_user_marker), encoding="utf-8"
        )
        hostile_environment = dict(os.environ)
        baseline = _run_launcher(
            control_root / "tools" / "validate_candidate",
            control_root,
            uv,
            hostile_environment,
        )
        if baseline.returncode != 0:
            diagnostic = baseline.stderr.decode("utf-8", errors="replace")[-2000:]
            errors.append(f"isolated production-shape control failed: {diagnostic}")
        if b"FORGED-P1-PASS" in baseline.stdout + baseline.stderr:
            errors.append("repository-root startup hook forged gate output")
        if root_site_marker.exists() or root_user_marker.exists():
            errors.append("repository-root startup hook executed before validation")

        candidate_site_marker = temporary_root / "candidate-site-marker"
        candidate_user_marker = temporary_root / "candidate-user-marker"
        candidate = control_root / "contracts" / "p1-candidate"
        (candidate / "sitecustomize.py").write_text(
            _control_hook(candidate_site_marker), encoding="utf-8"
        )
        (candidate / "usercustomize.py").write_text(
            _control_hook(candidate_user_marker), encoding="utf-8"
        )
        expect_failure(
            "candidate-extra control",
            control_root / "tools" / "validate_candidate",
            control_root,
            b"candidate tree file set differs from its externally pinned manifest",
            hostile_environment,
        )
        if candidate_site_marker.exists() or candidate_user_marker.exists():
            errors.append("candidate-root startup hook executed before exact-set refusal")
        (candidate / "sitecustomize.py").unlink()
        (candidate / "usercustomize.py").unlink()

        launcher_text = (control_root / "tools" / "validate_candidate").read_text(
            encoding="utf-8"
        )
        mutations = (
            (
                "isolated-mode control",
                '"$python_path" -I -S -B "$validator"',
                '"$python_path" -S -B "$validator"',
                b"Python isolated mode is not active",
            ),
            (
                "no-site control",
                '"$python_path" -I -S -B "$validator"',
                '"$python_path" -I -B "$validator"',
                b"Python no-site mode is not active",
            ),
            (
                "working-directory control",
                'cd "$trusted_cwd"',
                'cd "$root"',
                b"authority cwd overlaps the candidate root",
            ),
            (
                "environment control",
                "/usr/bin/env -i \\\n"
                "    HOME=/nonexistent \\\n"
                "    LANG=C.UTF-8 \\\n"
                "    LC_ALL=C.UTF-8 \\\n"
                "    PATH=\"$FIXED_PATH\" \\\n"
                "    PYTHONDONTWRITEBYTECODE=1",
                "/usr/bin/env \\\n"
                "    HOME=/nonexistent \\\n"
                "    LANG=C.UTF-8 \\\n"
                "    LC_ALL=C.UTF-8 \\\n"
                "    PATH=\"$FIXED_PATH\" \\\n"
                "    PYTHONDONTWRITEBYTECODE=1",
                b"forbidden startup environment variables reached the validator",
            ),
        )
        for label, old, new, expected in mutations:
            if launcher_text.count(old) != 1:
                errors.append(f"{label}: production launch site is not uniquely registered")
                continue
            mutated = control_root / "tools" / ("validate_candidate-" + label.split()[0])
            mutated.write_text(launcher_text.replace(old, new, 1), encoding="utf-8")
            expect_failure(label, mutated, control_root, expected, hostile_environment)

    return errors


def run_live_hardware_check(
    candidate_root: Path,
    site_packages: Path,
) -> list[str]:
    repository = ROOT.parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(repository / "tools" / "validate_live_hardware.py"),
            "--candidate-root",
            str(candidate_root.resolve(strict=True)),
            "--site-packages",
            str(site_packages.resolve(strict=True)),
            "--validator",
            str(repository / "tools" / "validate_candidate.py"),
        ],
        cwd=Path.cwd(),
        env={
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        return [f"isolated live-hardware check failed: {diagnostic}"]
    if completed.stderr:
        return ["isolated live-hardware check wrote stderr on success"]
    if len(completed.stdout) > 16 * 1024:
        return ["isolated live-hardware check exceeded its output boundary"]
    sys.stdout.buffer.write(completed.stdout)
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true", help="also apply invalid mutation fixtures and execute replay binaries")
    parser.add_argument("--live-hardware", action="store_true", help="also run the isolated live D2 hardware contract check")
    parser.add_argument("--startup-control-child", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", arguments.expected_candidate_manifest_sha256):
        raise ValidationFailure("candidate manifest authority pin is not a SHA-256 digest")
    configure_boundary(arguments.candidate_root, arguments.site_packages)
    available = validators()
    integrity_errors, hashed_count = verify_candidate_integrity(
        arguments.expected_candidate_manifest_sha256
    )
    contract = load_json(ROOT / "invocation-contract.json")
    if contract.get("schema") != "f106.invocation-contract/v0-draft" or contract.get("candidate") is not True:
        raise ValidationFailure("invocation contract has an unexpected identity or freeze state")
    contract_errors = validate_document(
        "f106.invocation-contract/v0-draft", contract, available
    )
    command_entries = contract.get("commands")
    if not isinstance(command_entries, list) or not command_entries:
        raise ValidationFailure("invocation contract has no commands")
    command_by_id = {entry["command_id"]: entry for entry in command_entries}
    if len(command_by_id) != len(command_entries):
        raise ValidationFailure("invocation contract command IDs are not unique")

    failures: list[str] = list(integrity_errors)
    if contract_errors:
        failures.append("invocation-contract.json: " + "; ".join(contract_errors))
    valid_count = 0
    for identity, group in FIXTURE_GROUPS.items():
        for path in fixture_paths(group):
            document = load_json(path)
            if identity == "plebian.cli.response/v1":
                errors = validate_response(document, command_by_id, available)
            else:
                observed = schema_identity(document, identity)
                errors = [f"identity mismatch: {observed}"] if observed != identity else validate_document(identity, document, available)
            if errors:
                failures.append(f"{path.relative_to(ROOT)}: " + "; ".join(errors))
            valid_count += 1

    invalid_count = 0
    invalid_total = 0
    parser_control_count = 0
    if arguments.self_test:
        invalid_paths = sorted((ROOT / "fixtures" / "invalid").glob("*.json"))
        invalid_total = len(invalid_paths)
        for path in invalid_paths:
            mutation = load_json(path)
            base = load_json(ROOT / mutation["base"])
            mutated = copy.deepcopy(base)
            set_pointer(mutated, mutation["path"], mutation["value"])
            identity = schema_identity(mutated)
            errors = validate_document(identity, mutated, available)
            expected = mutation["expected"]
            if not any(expected in error for error in errors):
                failures.append(f"{path.relative_to(ROOT)}: expected rejection containing {expected!r}; got {errors}")
            invalid_count += 1
        parser_errors, parser_control_count = strict_json_parser_controls()
        failures.extend(parser_errors)
        failures.extend(run_replay_checks(contract))
        if not failures and not arguments.startup_control_child:
            failures.extend(startup_boundary_controls(arguments.launcher, arguments.uv))
    if arguments.live_hardware and not failures:
        failures.extend(
            run_live_hardware_check(arguments.candidate_root, arguments.site_packages)
        )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    mode = (
        f", {invalid_count}/{invalid_total} invalid mutations rejected, "
        f"{parser_control_count}/6 strict JSON parser controls rejected, "
        "replay success/failure paths verified"
        if arguments.self_test
        else ""
    )
    if arguments.self_test and not arguments.startup_control_child:
        mode += ", partial developer startup controls exercised; release authority blocked"
    print(
        f"PASS (developer-only): {len(available)}/{len(SCHEMA_PATHS)} candidate "
        f"schemas, {valid_count}/{valid_count} valid fixtures{mode}; "
        f"{hashed_count}/{hashed_count} hashes and canonical JSON verified; "
        "no qualification claim"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValidationFailure, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
