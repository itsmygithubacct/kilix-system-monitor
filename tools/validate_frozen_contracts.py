#!/usr/bin/env python3
"""Validate the frozen contracts/v1 resource-profile bundle.

The bundle is immutable: its FROZEN-SHA256SUMS digest is pinned below, both
schemas are pinned to the exact candidate bytes consumers acknowledged, and
every measured row is pinned to the checked RES-02 evidence bytes. A change to
any byte fails here and needs a successor bundle, never an edit.

Beyond integrity, this proves the bundle carries a resource shape rather than
any profile-shaped document: the profile item schema must require a backend and
peak RAM and VRAM requirements, and a launcher-profile-shaped document must be
refused. Each rule has a planted control that must fire, so a pass cannot come
from a check that did not look.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "contracts" / "v1"
MANIFEST = "FROZEN-SHA256SUMS"
EXPECTED_MANIFEST_SHA256 = "9f0f4e85209451b98aa79a6768db9703de9abe092f88f28a8b600cec36e1e3e1"
PROFILES_SCHEMA = "plebian.models.profiles-v1.schema.json"
HARDWARE_SCHEMA = "plebian.hardware-v1.schema.json"
PINNED_SCHEMAS = {
    PROFILES_SCHEMA: "31ca9cbac3eb0dd463835419fbb89def3da533cb4b6a6083d476678cc643af1c",
    HARDWARE_SCHEMA: "f0f2a46adec28e2997e06b3976759a2dfc8e42e7e9d6a42bb1728c3453d8c8be",
}
RES02_RAW_EVIDENCE_SHA256 = "4ff037142014df81287230ab7e052f8a9e4d65281bb401e0b23ef386107953df"
PINNED_MEASURED = {
    "espeak.json": "2224df32a03fef2d7815045325854b3bd30829aa5c0c808bd6d654e3d183292d",
    "lgraph-en-us.json": "7b435eaee13785a2eb342ad0e5594760c21ba63935907e61d2fc2ebd4386c98d",
    "mbrola.json": "028ae726e12dcd637d10aa78ba6a125d3aa19c8c56bd21dc9abe7da93fa7968a",
    "piper-en-us-kristin-medium.json": "1ecc1ebc1c9f61d241bbc2bdb57a7adda9dd02e1891aac7ddbbd30bdb8f7f632",
    "qwen3-tts-0.6b-base-cuda.json": "4beaac1df73e1a49e7a1952426e76cea0d6056dfcc27968517103e71b7a35d36",
    "small-en-us.json": "cad2a651a5587ed91d46ef3eb9cbf9fd41f54d8587bf71bdc188fef3d2cbe21f",
    "whisper-tiny-cuda.json": "d0193d38310921bf403f92ff8b63c64d9879adfe034366800f10c39c069cac35",
}
BACKENDS = ["cpu", "cuda", "oneapi", "opencl", "rocm", "vulkan"]
LAUNCHER_PROFILE_KEYS = ("commands", "launcher_name", "profile_id", "schema", "subject_hash_manifests")
# Built from segments so this detector carries no literal private path itself.
FORBIDDEN_TEXT = tuple(f"/{segment}/" for segment in ("home", "tmp", "root", "mnt"))


class Failure(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def strict_json(data: bytes, label: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result

    def constant(name):
        raise Failure(f"{label}: non-finite JSON number {name}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, ValueError) as error:
        raise Failure(f"{label}: invalid UTF-8 JSON: {error}") from error


def manifest_bytes(files: dict[str, bytes]) -> bytes:
    return "".join(f"{sha256(data)}  ./{name}\n" for name, data in sorted(files.items())).encode("ascii")


def read_bundle() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(BUNDLE.rglob("*")):
        relative = path.relative_to(BUNDLE).as_posix()
        require(not path.is_symlink(), f"bundle contains a symlink: {relative}")
        if path.is_dir():
            continue
        require(path.is_file(), f"bundle contains a non-file entry: {relative}")
        if relative != MANIFEST:
            files[relative] = path.read_bytes()
    return files


def structural_resource_errors(schema: dict[str, Any]) -> list[str]:
    errors = []
    try:
        item = schema["properties"]["profiles"]["items"]
        required = set(item["required"])
        requirements = item["properties"]["requirements"]
        requirement_fields = set(requirements["required"])
        backend = item["properties"]["backend"]["enum"]
    except (KeyError, TypeError) as error:
        return [f"profile item schema lacks the resource structure: {error!r}"]
    for field in ("backend", "requirements"):
        if field not in required:
            errors.append(f"profile item schema does not require {field}")
    for field in ("ram_peak_bytes", "vram_peak_bytes"):
        if field not in requirement_fields:
            errors.append(f"profile requirements do not require {field}")
    if backend != BACKENDS:
        errors.append(f"profile backend vocabulary differs: {backend}")
    if item.get("additionalProperties") is not False or requirements.get("additionalProperties") is not False:
        errors.append("profile item or requirements object is open to undeclared keys")
    return errors


def semantic_errors(document: dict[str, Any]) -> list[str]:
    """The profile rules of the pre-freeze candidate, frozen with the bundle."""
    errors = []
    if document.get("fixture_kind") == "synthetic-contract" and document.get("qualification_eligible"):
        errors.append("synthetic profile catalog cannot be qualification eligible")
    for profile in document.get("profiles", []):
        if not isinstance(profile, dict):
            continue
        evidence = profile.get("evidence", {})
        performance = profile.get("performance", {})
        artifact = profile.get("artifact", {})
        if any(value is not None for value in performance.values()) and evidence.get("confidence") != "measured":
            errors.append("performance number lacks measured evidence")
        if profile.get("qualification") == "qualified" and not all((
            document.get("qualification_eligible"),
            evidence.get("confidence") == "measured",
            evidence.get("command") is not None,
            evidence.get("fixture") is not None,
            evidence.get("measured_at") is not None,
            evidence.get("raw_evidence_sha256") is not None,
            evidence.get("reference_hardware_class") is not None,
            artifact.get("content_sha256") is not None,
            artifact.get("license_decision_id") is not None,
        )):
            errors.append("qualified profile lacks measured evidence")
    return errors


def text_errors(data: bytes) -> list[str]:
    text = data.decode("utf-8")
    return [f"private path marker {marker!r}" for marker in FORBIDDEN_TEXT if marker in text]


def profile_errors(validator: Draft202012Validator, document: Any) -> list[str]:
    errors = [f"schema: {error.message}" for error in
              sorted(validator.iter_errors(document), key=lambda error: list(error.absolute_path))]
    if isinstance(document, dict):
        errors.extend(semantic_errors(document))
    return errors


def measured_policy_errors(document: dict[str, Any]) -> list[str]:
    errors = []
    if document.get("qualification_eligible") is not False:
        errors.append("measured catalog claims qualification eligibility")
    if document.get("fixture_kind") != "provider-catalog":
        errors.append("measured catalog is not a provider catalog")
    for profile in document.get("profiles", []):
        evidence, requirements = profile["evidence"], profile["requirements"]
        if evidence["confidence"] != "measured":
            errors.append(f"{profile['profile_id']}: confidence is not measured")
        for field in ("command", "fixture", "measured_at", "raw_evidence_sha256", "reference_hardware_class"):
            if evidence[field] is None:
                errors.append(f"{profile['profile_id']}: evidence.{field} is null")
        if evidence["raw_evidence_sha256"] != RES02_RAW_EVIDENCE_SHA256:
            errors.append(f"{profile['profile_id']}: raw evidence is not the RES-02 aggregate")
        if requirements["ram_peak_bytes"] is None:
            errors.append(f"{profile['profile_id']}: ram_peak_bytes is null")
        if profile["backend"] != "cpu" and requirements["vram_peak_bytes"] is None:
            errors.append(f"{profile['profile_id']}: accelerator row has no vram_peak_bytes")
        if profile["backend"] == "cpu" and requirements["vram_peak_bytes"] not in (0, None):
            errors.append(f"{profile['profile_id']}: cpu row reports accelerator memory")
        if profile["qualification"] != "unqualified":
            errors.append(f"{profile['profile_id']}: measured row claims qualification")
    return errors


def pointer_apply(document: Any, op: str, pointer: str, value: Any) -> Any:
    result = copy.deepcopy(document)
    parts = pointer.split("/")[1:]
    target = result
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    leaf = parts[-1]
    if op == "set":
        target[int(leaf) if isinstance(target, list) else leaf] = value
    elif op == "add":
        require(leaf not in target, f"add target already present: {pointer}")
        target[leaf] = value
    elif op == "remove":
        del target[leaf]
    else:
        raise Failure(f"unknown mutation op {op}")
    return result


def candidate_semantic_rule():
    path = ROOT / "tools" / "validate_candidate.py"
    spec = importlib.util.spec_from_file_location("frozen_bundle_candidate_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda document: module.semantic_errors("plebian.models.profiles/v1", document)


def main() -> int:
    counts: dict[str, int] = {}
    files = read_bundle()
    manifest = (BUNDLE / MANIFEST).read_bytes()
    require(manifest == manifest_bytes(files), f"{MANIFEST} does not match the complete bundle tree")
    require(sha256(manifest) == EXPECTED_MANIFEST_SHA256, f"{MANIFEST} differs from its pinned digest")
    counts["files"] = len(files)

    documents = {}
    for name, data in files.items():
        if name.endswith(".json"):
            documents[name] = strict_json(data, name)
            require(data == canonical(documents[name]), f"non-canonical JSON: {name}")
        require(not text_errors(data), f"{name}: {text_errors(data)}")
    counts["canonical_json"] = len(documents)

    for name, digest in PINNED_SCHEMAS.items():
        require(sha256(files[name]) == digest, f"{name} differs from the acknowledged candidate bytes")
        Draft202012Validator.check_schema(documents[name])
    schema = documents[PROFILES_SCHEMA]
    require(schema["properties"]["schema"]["const"] == "plebian.models.profiles/v1", "profiles schema identity")
    structural = structural_resource_errors(schema)
    require(not structural, f"structural resource assertion: {structural}")
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    valid = sorted(name for name in documents if name.startswith("fixtures/profiles/valid/"))
    invalid = sorted(name for name in documents if name.startswith("fixtures/profiles/invalid/"))
    measured = sorted(name for name in documents if name.startswith("profiles/res02-measured/"))
    expected_population = {PROFILES_SCHEMA, HARDWARE_SCHEMA, "README.md", "fixtures/profiles/REFUSALS.json",
                           *valid, *invalid, *measured}
    require(set(files) == expected_population, f"unexpected bundle members: {sorted(set(files) ^ expected_population)}")
    require(len(valid) == 3 and len(invalid) == 9 and len(measured) == 7,
            f"fixture population valid={len(valid)}/3 invalid={len(invalid)}/9 measured={len(measured)}/7")

    for name in valid:
        errors = profile_errors(validator, documents[name])
        require(not errors, f"valid fixture refused: {name}: {errors}")
    counts["valid"] = len(valid)

    refusals = documents["fixtures/profiles/REFUSALS.json"]
    require(set(refusals) == {Path(name).name for name in invalid}, "REFUSALS.json does not cover exactly the invalid fixtures")
    for name in invalid:
        spec = refusals[Path(name).name]
        require(set(spec) == {"base", "op", "pointer", "value", "expected"}, f"{name}: malformed refusal record")
        require(spec["base"] in valid, f"{name}: base is not a valid fixture")
        derived = pointer_apply(documents[spec["base"]], spec["op"], spec["pointer"], spec["value"])
        require(canonical(derived) == files[name], f"{name}: bytes are not the declared single mutation of its base")
        errors = profile_errors(validator, documents[name])
        require(any(spec["expected"] in error for error in errors),
                f"{name}: expected refusal {spec['expected']!r}, observed {errors}")
    counts["invalid"] = len(invalid)

    for name in measured:
        leaf = Path(name).name
        require(sha256(files[name]) == PINNED_MEASURED.get(leaf, ""), f"{name}: differs from the checked RES-02 bytes")
        errors = profile_errors(validator, documents[name]) or measured_policy_errors(documents[name])
        require(not errors, f"measured row refused: {name}: {errors}")
    require({Path(name).name for name in measured} == set(PINNED_MEASURED), "measured row population differs")
    profile_ids = [row["profile_id"] for name in measured for row in documents[name]["profiles"]]
    require(len(profile_ids) == len(set(profile_ids)) == 7, "measured profile ids are not 7 unique rows")
    counts["measured"] = len(profile_ids)

    # Planted controls: every rule above must be able to fire.
    controls = 0
    launcher_shaped = {key: [] if key in ("commands", "subject_hash_manifests") else "fixture" for key in LAUNCHER_PROFILE_KEYS}
    launcher_shaped["schema"] = "kilix.trusted-launcher.profile/v1"
    require(profile_errors(validator, launcher_shaped), "control: a launcher-profile-shaped document was accepted")
    controls += 1
    for field in ("backend", "requirements"):
        weakened = copy.deepcopy(schema)
        weakened["properties"]["profiles"]["items"]["required"].remove(field)
        require(structural_resource_errors(weakened), f"control: schema without required {field} passed the structural assertion")
        controls += 1
    for field in ("ram_peak_bytes", "vram_peak_bytes"):
        weakened = copy.deepcopy(schema)
        weakened["properties"]["profiles"]["items"]["properties"]["requirements"]["required"].remove(field)
        require(structural_resource_errors(weakened), f"control: schema without required {field} passed the structural assertion")
        controls += 1
    altered = dict(files)
    first_measured = measured[0]
    altered[first_measured] = altered[first_measured].replace(b'"measured"', b'"estimated"', 1)
    require(manifest_bytes(altered) != manifest, "control: an altered measured row left the manifest unchanged")
    controls += 1
    injected = copy.deepcopy(documents[valid[0]])
    injected["profiles"][0]["requirements"]["vram_estimate_bytes"] = 0
    require(profile_errors(validator, injected), "control: an injected requirements key was accepted")
    controls += 1
    unmeasured = copy.deepcopy(documents[first_measured])
    unmeasured["profiles"][0]["evidence"]["measured_at"] = None
    require(measured_policy_errors(unmeasured), "control: a measured row without measured_at passed the measured policy")
    controls += 1
    candidate_rule = candidate_semantic_rule()
    for name in valid + invalid + measured:
        require(bool(candidate_rule(documents[name])) == bool(semantic_errors(documents[name])),
                f"control: frozen and candidate profile rules disagree on {name}")
    controls += 1
    counts["controls"] = controls

    print(
        "PASS (frozen bundle, builder-landed, acceptance pending): "
        f"{counts['files']}/{counts['files']} manifest members and FROZEN-SHA256SUMS {sha256(manifest)}; "
        f"{counts['canonical_json']}/{counts['canonical_json']} canonical JSON; 2/2 schemas equal acknowledged candidate bytes; "
        "1/1 structural resource assertion; "
        f"{counts['valid']}/3 valid fixtures accepted; {counts['invalid']}/9 invalid fixtures refused for their declared reason; "
        f"{counts['measured']}/7 measured RES-02 rows pinned and within measured policy; "
        f"{counts['controls']}/{counts['controls']} planted controls fired; qualification 0/1"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as error:
        print(f"FAIL: frozen bundle: {error}", file=sys.stderr)
        raise SystemExit(1)
