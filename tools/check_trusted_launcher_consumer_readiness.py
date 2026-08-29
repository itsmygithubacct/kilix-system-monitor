#!/usr/bin/env python3
"""Check Track D inputs without claiming trusted-launcher adoption."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = ROOT / "integration" / "trusted-launcher-consumer-requirements.json"
SCHEMA = "kilix.track-d.trusted-launcher-consumer-requirements/v1"
STATUS = "developer-readiness-only-not-a-launch-profile"
COMMON_CASES = ("ID-02", "ID-04", "SUB-04", "SUB-05", "SUB-06", "RES-10")
RETURN_IDENTITIES = (
    "public_commit",
    "public_tree",
    "launcher_sha256",
    "bootstrap_sha256",
    "result_schema_sha256",
    "profile_schema_sha256",
    "td_p1_profile_sha256",
    "td_hw_profile_sha256",
)
P1_INVOCATIONS = (
    ("plebian-hardware", "show"),
    ("plebian-hardware", "inventory", "--json"),
    ("plebian-hardware", "gpu", "--json"),
    ("plebian-model-sizer", "recommend", "tts", "--json"),
    ("plebian-model-sizer", "plan", "local-ai-balanced", "--json"),
    ("plebian-model-sizer", "install", "PLAN_PATH", "--json"),
    ("plebian-model-sizer", "snapshot", "--json"),
)


class ReadinessFailure(ValueError):
    """The local requirements cannot safely accept an upstream return."""


def _load_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReadinessFailure(f"{path}: duplicate key {key!r}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReadinessFailure(f"{path}: invalid UTF-8 JSON: {error}") from error
    canonical = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.read_bytes() != canonical:
        raise ReadinessFailure(f"{path}: requirements are not canonical JSON")
    return value


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ReadinessFailure(f"{label}: expected keys {sorted(expected)}, observed {observed}")
    return value


def _regular_path(relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ReadinessFailure(f"{label}: path escapes the product root")
    path = ROOT / candidate
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReadinessFailure(f"{label}: path is unavailable: {error}") from error
    if ROOT not in resolved.parents or path.is_symlink():
        raise ReadinessFailure(f"{label}: path is aliased or outside the product root")
    if not resolved.is_file() and not resolved.is_dir():
        raise ReadinessFailure(f"{label}: path is not a regular file or directory")
    return resolved


def _expected_consumers() -> list[dict[str, Any]]:
    return [
        {
            "campaigns": [
                "common-matrix",
                "CHN-01-through-CHN-06-for-each-replay-helper",
                "python-on-PATH-replacement",
                "startup-hooks-at-every-later-import-visible-root",
                "seven-invocation-byte-compare",
            ],
            "intentional_children": [
                {
                    "entry_kind": "bound-python-authority",
                    "entry_path": "tools/validate_candidate.py",
                    "order": 1,
                    "role": "candidate-semantic-validator",
                },
                {
                    "entry_kind": "bound-python-module",
                    "entry_path": "contracts/p1-candidate/tools/replay_cli.py",
                    "logical_program": "plebian-hardware",
                    "order": 2,
                    "role": "fixture-replay",
                },
                {
                    "entry_kind": "bound-python-module",
                    "entry_path": "contracts/p1-candidate/tools/replay_cli.py",
                    "logical_program": "plebian-model-sizer",
                    "order": 3,
                    "role": "fixture-replay",
                },
            ],
            "post_child_verification": "required-after-every-child",
            "requirement_id": "TD-P1",
            "subject": "independent-git-archive-export",
            "terminal_members": [
                "candidate-semantic-validator",
                "plebian-hardware-replay",
                "plebian-model-sizer-replay",
            ],
        },
        {
            "campaigns": [
                "common-matrix",
                "CHN-01-through-CHN-06-for-installed-command",
                "physical-hardware",
                "virtual-machine",
                "denied-probe",
                "non-x86",
            ],
            "intentional_children": [
                {
                    "command_tail": ["-m", "unittest", "discover", "-s", "tests", "-v"],
                    "entry_kind": "bound-python-test-suite",
                    "order": 1,
                    "source_path": "components/plebian-hardware/tests",
                },
                {
                    "argv": ["show"],
                    "entry_kind": "staged-console",
                    "entry_point": "plebian-hardware",
                    "order": 2,
                },
                {
                    "argv": ["inventory", "--json"],
                    "entry_kind": "staged-console",
                    "entry_point": "plebian-hardware",
                    "order": 3,
                },
                {
                    "argv": ["gpu", "--json"],
                    "entry_kind": "staged-console",
                    "entry_point": "plebian-hardware",
                    "order": 4,
                },
                {
                    "argv": ["--json", "inventory"],
                    "entry_kind": "staged-console-invalid-argv",
                    "entry_point": "plebian-hardware",
                    "expected_exit": 2,
                    "order": 5,
                },
            ],
            "post_child_verification": "required-after-every-child",
            "requirement_id": "TD-HW",
            "subject": "independent-git-archive-export-plus-staged-prefix",
            "terminal_members": [
                "hardware-unittest-suite",
                "plebian-hardware-show",
                "plebian-hardware-inventory",
                "plebian-hardware-gpu",
                "plebian-hardware-invalid-argv",
            ],
        },
    ]


def _validate_upstream(value: Any) -> None:
    gate = _keys(
        value,
        {
            "accepted_result_states",
            "assignments",
            "blocked_result_states",
            "common_case_ids",
            "independent_exports",
            "required_return_identities",
            "state",
        },
        "upstream_gate",
    )
    if gate["accepted_result_states"] != ["PASS", "REFUSED-AS-NAMED"]:
        raise ReadinessFailure("upstream_gate: accepted result states changed")
    if gate["blocked_result_states"] != ["NULL", "HARNESS-FAIL"]:
        raise ReadinessFailure("upstream_gate: blocked result states changed")
    if tuple(gate["common_case_ids"]) != COMMON_CASES:
        raise ReadinessFailure("upstream_gate: common case population changed")
    if tuple(gate["required_return_identities"]) != RETURN_IDENTITIES:
        raise ReadinessFailure("upstream_gate: returned identity population changed")
    if gate["independent_exports"] != {"passing": 2, "population": 2}:
        raise ReadinessFailure("upstream_gate: two-of-two independent export requirement changed")
    if gate["state"] != "blocked-pending-results":
        raise ReadinessFailure("upstream_gate: an assignment was promoted without a result")
    expected_assignments = [
        {
            "decision_id": "OD-13",
            "owner": "reviewer2",
            "state": "assigned-not-returned",
            "work": "ID-04-facility-implementation",
        },
        {
            "decision_id": "OD-14",
            "owner": "Track H",
            "state": "assigned-not-returned-cross-family-review-required",
            "work": "non-forking-profile-child-table-interface",
        },
    ]
    if gate["assignments"] != expected_assignments:
        raise ReadinessFailure("upstream_gate: OD-13/OD-14 assignment boundary changed")


def _validate_consumer_paths(consumers: list[dict[str, Any]]) -> None:
    for consumer in consumers:
        for child in consumer["intentional_children"]:
            if "entry_path" in child:
                path = _regular_path(child["entry_path"], child["role"])
                if not path.is_file():
                    raise ReadinessFailure(f"{child['role']}: entry path is not a file")
            if "source_path" in child:
                path = _regular_path(child["source_path"], consumer["requirement_id"])
                if not path.is_dir():
                    raise ReadinessFailure(f"{consumer['requirement_id']}: test source is not a directory")

    hardware_project = ROOT / "components" / "plebian-hardware" / "pyproject.toml"
    with hardware_project.open("rb") as handle:
        project = tomllib.load(handle)
    scripts = project.get("project", {}).get("scripts", {})
    if scripts.get("plebian-hardware") != "plebian_hardware.cli:main":
        raise ReadinessFailure("TD-HW: staged plebian-hardware entry point changed")

    model_sizer = ROOT / "components" / "plebian-model-sizer"
    if (model_sizer / "pyproject.toml").exists():
        raise ReadinessFailure("TD-HW: blocked model-sizer gained a project entry point")


def _validate_invocations() -> None:
    contract_path = ROOT / "contracts" / "p1-candidate" / "invocation-contract.json"
    contract = _load_json(contract_path)
    commands = contract.get("commands") if isinstance(contract, dict) else None
    if not isinstance(commands, list):
        raise ReadinessFailure("TD-P1: invocation contract has no command population")
    observed = tuple(tuple(command.get("argv", [])) for command in commands)
    if observed != P1_INVOCATIONS:
        raise ReadinessFailure("TD-P1: seven-command invocation population changed")
    candidate_root = contract_path.parent
    for command in commands:
        fixture = command.get("fixture")
        if not isinstance(fixture, str):
            raise ReadinessFailure("TD-P1: an invocation lacks its fixture")
        path = Path(fixture)
        if path.is_absolute() or ".." in path.parts:
            raise ReadinessFailure("TD-P1: an invocation fixture escapes the candidate root")
        resolved = (candidate_root / path).resolve(strict=True)
        if candidate_root not in resolved.parents or not resolved.is_file():
            raise ReadinessFailure("TD-P1: an invocation fixture is not a regular candidate file")


def validate_requirements(value: Any) -> None:
    document = _keys(
        value,
        {"consumer_requirements", "schema", "status", "upstream_gate"},
        "requirements",
    )
    if document["schema"] != SCHEMA:
        raise ReadinessFailure("requirements: schema identity changed")
    if document["status"] != STATUS:
        raise ReadinessFailure("requirements: readiness-only disclaimer changed")
    _validate_upstream(document["upstream_gate"])
    expected_consumers = _expected_consumers()
    if document["consumer_requirements"] != expected_consumers:
        raise ReadinessFailure("requirements: TD-P1/TD-HW child or campaign surface changed")
    _validate_consumer_paths(expected_consumers)
    _validate_invocations()


def _mutations() -> list[Callable[[dict[str, Any]], None]]:
    def remove_case(value: dict[str, Any]) -> None:
        value["upstream_gate"]["common_case_ids"].pop()

    def promote_assignment(value: dict[str, Any]) -> None:
        value["upstream_gate"]["state"] = "accepted"

    def add_d4_child(value: dict[str, Any]) -> None:
        value["consumer_requirements"][0]["intentional_children"].append(
            {
                "entry_kind": "staged-console",
                "entry_point": "plebian-model-sizer",
                "order": 4,
            }
        )

    def change_interface_owner(value: dict[str, Any]) -> None:
        value["upstream_gate"]["assignments"][1]["owner"] = "Track D"

    return [remove_case, promote_assignment, add_d4_child, change_interface_owner]


def _self_test(value: dict[str, Any]) -> int:
    rejected = 0
    for mutate in _mutations():
        candidate = copy.deepcopy(value)
        mutate(candidate)
        try:
            validate_requirements(candidate)
        except ReadinessFailure:
            rejected += 1
    if rejected != len(_mutations()):
        raise ReadinessFailure(
            f"readiness self-test rejected {rejected}/{len(_mutations())} mutations"
        )
    return rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    value = _load_json(arguments.requirements.resolve(strict=True))
    if not isinstance(value, dict):
        raise ReadinessFailure("requirements: top level is not an object")
    validate_requirements(value)
    rejected = _self_test(value) if arguments.self_test else None
    message = (
        "PASS (developer readiness only; adoption remains blocked): "
        f"{len(COMMON_CASES)}/{len(COMMON_CASES)} common cases retained; "
        "2/2 consumer requirements retained; 8/8 intentional child specifications retained; "
        f"{len(P1_INVOCATIONS)}/{len(P1_INVOCATIONS)} P1 invocation vectors retained; "
        f"0/{len(RETURN_IDENTITIES)} upstream return identities consumed"
    )
    if rejected is not None:
        message += f"; {rejected}/{len(_mutations())} premature-adoption mutations rejected"
    print(message)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ReadinessFailure, tomllib.TOMLDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
