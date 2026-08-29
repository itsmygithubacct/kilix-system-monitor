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
DEFAULT_CAMPAIGN = ROOT / "integration" / "trusted-launcher-consumer-campaign.json"
SCHEMA = "kilix.track-d.trusted-launcher-consumer-requirements/v2"
STATUS = "final-track-d-consumer-input-not-a-launch-profile"
CAMPAIGN_SCHEMA = "kilix.track-d.trusted-launcher-consumer-campaign/v1"
CAMPAIGN_STATUS = "final-campaign-specification-no-results"
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
    ("/usr/bin/plebian-hardware", "show"),
    ("/usr/bin/plebian-hardware", "inventory", "--json"),
    ("/usr/bin/plebian-hardware", "gpu", "--json"),
    ("/usr/bin/plebian-model-sizer", "recommend", "tts", "--json"),
    ("/usr/bin/plebian-model-sizer", "plan", "local-ai-balanced", "--json"),
    (
        "/usr/bin/plebian-model-sizer",
        "install",
        "PLAN_PATH",
        "--expected-plan-sha256",
        "PLAN_SHA256",
        "--json",
    ),
    ("/usr/bin/plebian-model-sizer", "install-status", "TRANSACTION_ID", "--json"),
    ("/usr/bin/plebian-model-sizer", "cancel", "TRANSACTION_ID", "--json"),
    ("/usr/bin/plebian-model-sizer", "snapshot", "--json"),
)
INTERFACE_REQUIREMENT_IDS = tuple(f"TLIF-{index:02d}" for index in range(1, 21))
EVIDENCE_REQUIREMENT_IDS = tuple(f"TDE-{index:02d}" for index in range(1, 12))
FAMILY_RESULT_FIELDS = (
    "case_id",
    "target_id",
    "export_id",
    "grade",
    "variants_required",
    "variants_executed",
    "variants_pass",
    "variants_refused_as_named",
    "variants_null",
    "variants_harness_fail",
    "variant_manifest_sha256",
    "evidence_manifest_sha256",
)
COMMON_MATRIX_IDS = (
    *(f"CAL-{index:02d}" for index in range(1, 7)),
    *(f"ENV-{index:02d}" for index in range(1, 8)),
    *(f"ID-{index:02d}" for index in range(1, 7)),
    "DEP-01",
    "ORD-01",
    "ORD-02",
    *(f"SUB-{index:02d}" for index in range(1, 10)),
    *(f"RES-{index:02d}" for index in range(1, 11)),
)
CHAIN_MATRIX_IDS = tuple(f"CHN-{index:02d}" for index in range(1, 7))
EXPECTED_COMMON_OUTCOMES = {
    "CAL-01": ("CONTROL-MARKER-FORGED-TEXT-RC0-OBSERVED",),
    "CAL-02": ("CONTROL-EVERY-REACHABLE-LOCATION-FIRED",),
    "CAL-03": ("BASELINE-BODY-ONCE-PRECLOSURE-ZERO",),
    "CAL-04": ("TL-SUBJECT-CLOSURE", "TL-RESERVED-STARTUP"),
    "CAL-05": ("AC-TL-FIRST-PROCESS-UNOBSERVED",),
    "CAL-06": ("AC-TL-MUTATION-NOT-LIVE",),
    "ENV-01": ("TL-LAUNCH-ARGV", "AC-TL-MUTANT-SURVIVED"),
    "ENV-02": ("TL-LAUNCH-ENVIRONMENT",),
    "ENV-03": ("TL-LAUNCH-ENVIRONMENT",),
    "ENV-04": ("TL-LAUNCH-ENVIRONMENT",),
    "ENV-05": ("TL-LAUNCH-CWD",),
    "ENV-06": ("TL-LAUNCH-PATH",),
    "ENV-07": ("TL-LAUNCH-MODULE-STATE",),
    "ID-01": ("TL-INTERPRETER-IDENTITY",),
    "ID-02": ("TL-INTERPRETER-IDENTITY",),
    "ID-03": ("TL-INTERPRETER-IDENTITY",),
    "ID-04": ("TL-INTERPRETER-IDENTITY",),
    "ID-05": ("TL-BOOTSTRAP-IDENTITY", "TL-AUTHORITY-SOURCE"),
    "ID-06": ("TL-AUTHORITY-SOURCE",),
    "DEP-01": ("TL-DEPENDENCY-CLOSURE",),
    "ORD-01": ("TL-AUTHORITY-SOURCE",),
    "ORD-02": ("AC-TL-MARKER-EXECUTED",),
    "SUB-01": ("TL-SUBJECT-CLOSURE",),
    "SUB-02": ("TL-SUBJECT-CLOSURE",),
    "SUB-03": ("TL-SUBJECT-CLOSURE",),
    "SUB-04": ("TL-SUBJECT-ROOT",),
    "SUB-05": ("TL-SUBJECT-DRIFT",),
    "SUB-06": (
        "TL-SUBJECT-DRIFT",
        "TL-AUTHORITY-SOURCE",
        "TL-BOOTSTRAP-IDENTITY",
        "TL-DEPENDENCY-CLOSURE",
    ),
    "SUB-07": ("TL-SUBJECT-CLOSURE",),
    "SUB-08": ("TL-RESERVED-STARTUP",),
    "SUB-09": ("TL-SUBJECT-DRIFT",),
    "RES-01": ("TL-RESULT-MISSING",),
    "RES-02": ("TL-RESULT-SHAPE",),
    "RES-03": ("TL-RESULT-IDENTITY",),
    "RES-04": ("TL-RESULT-SHAPE", "TL-RESULT-MISSING", "TL-RESULT-LIFECYCLE"),
    "RES-05": ("TL-RESULT-NONTERMINAL", "TL-RESULT-LIFECYCLE"),
    "RES-06": ("TL-RESULT-LIFECYCLE", "TL-RESULT-NONTERMINAL"),
    "RES-07": ("TL-RESULT-NONTERMINAL",),
    "RES-08": ("TL-RESULT-IDENTITY", "TL-RESULT-SHAPE"),
    "RES-09": ("TL-RESULT-MISSING",),
    "RES-10": ("TL-CANCELLED",),
}
EXPECTED_CHAIN_OUTCOMES = {
    "CHN-01": ("TL-EXECUTION-CHAIN",),
    "CHN-02": ("TL-EXECUTION-CHAIN",),
    "CHN-03": ("TL-LAUNCH-ENVIRONMENT",),
    "CHN-04": ("TL-EXECUTION-CHAIN", "TL-SUBJECT-DRIFT"),
    "CHN-05": ("TL-RESULT-NONTERMINAL", "AC-TL-MUTANT-SURVIVED"),
    "CHN-06": ("AC-TL-MARKER-EXECUTED",),
}


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
        raise ReadinessFailure(f"{path}: document is not canonical JSON")
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
    if (resolved != ROOT and ROOT not in resolved.parents) or path.is_symlink():
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
                "nine-invocation-byte-compare",
            ],
            "freeze_boundary": "identical-product-tree-candidate-manifest-profile-result-and-signatory-bytes",
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
            "profile_constraints": {
                "authority_and_dependencies": "separately-retained-validator-and-external-dependency-closures",
                "child_launch": "shared-bootstrap-bound-interpreter-no-shebang",
                "environment": "exact-returned-allowlist",
                "result": "one-canonical-terminal-record-after-all-descendants",
                "subject_closure": "nested-candidate-root-within-whole-export-preservation-root",
            },
            "requirement_id": "TD-P1",
            "retained_roots": [
                {"path": ".", "role": "preservation-root"},
                {"path": "contracts/p1-candidate", "role": "subject-root"},
                {"path": "tools/validate_candidate.py", "role": "validator-authority"},
            ],
            "subject": "candidate-root-within-independent-git-archive-export",
            "success_assertions": [
                "hardware.show",
                "hardware.inventory",
                "hardware.gpu",
                "sizer.recommend.tts",
                "sizer.plan.local-ai-balanced",
                "sizer.install",
                "sizer.install.status",
                "sizer.install.cancel",
                "sizer.snapshot",
            ],
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
            "coverage_prerequisites": [
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
            "profile_constraints": {
                "authority_and_dependencies": "separately-retained-test-authority-and-external-dependency-closures",
                "child_launch": "shared-bootstrap-bound-interpreter-no-generated-shebang-authority",
                "environment": "exact-returned-allowlist",
                "result": "one-canonical-terminal-record-after-all-descendants",
                "subject_closure": "separate-source-and-staged-roots-within-whole-export-preservation",
            },
            "qualification_boundary": "launcher-integrity-does-not-qualify-hardware-coverage",
            "requirement_id": "TD-HW",
            "retained_roots": [
                {"path": ".", "role": "preservation-root"},
                {"path": "components/plebian-hardware/src", "role": "source-subject"},
                {"path": "components/plebian-hardware/tests", "role": "test-authority"},
                {"path_kind": "returned-staged-prefix", "role": "staged-subject"},
            ],
            "subject": "separate-source-and-staged-roots-within-independent-export",
            "terminal_members": [
                "hardware-unittest-suite",
                "plebian-hardware-show",
                "plebian-hardware-inventory",
                "plebian-hardware-gpu",
                "plebian-hardware-invalid-argv",
            ],
        },
    ]


def _validate_interface_contract(value: Any) -> None:
    interface = _keys(
        value,
        {"conformance_return", "must_not", "requirements", "scope"},
        "interface_contract",
    )
    if interface["scope"] != "capability-contract-not-track-h-interface-syntax":
        raise ReadinessFailure("interface_contract: scope changed")
    if interface["must_not"] != [
        "consumer-specific-native-launcher",
        "consumer-wrapper-before-native-launcher",
        "env-shebang-or-PATH-selected-authority",
        "subject-authored-identity-or-terminal-result",
        "child-result-promotion-without-post-descendant-verification",
    ]:
        raise ReadinessFailure("interface_contract: forbidden shape population changed")
    conformance = _keys(
        interface["conformance_return"],
        {"required_rows", "row_fields"},
        "interface_contract.conformance_return",
    )
    if conformance["required_rows"] != {
        "accepted": 0,
        "population": 20,
        "required": 20,
    }:
        raise ReadinessFailure("interface_contract: conformance denominator changed")
    if conformance["row_fields"] != [
        "requirement_id",
        "interface_field_or_mechanism",
        "evidence_sha256",
    ]:
        raise ReadinessFailure("interface_contract: conformance row shape changed")
    requirements = interface["requirements"]
    if not isinstance(requirements, list):
        raise ReadinessFailure("interface_contract: requirements are not a list")
    observed_ids: list[str] = []
    observed_text: set[str] = set()
    for index, requirement in enumerate(requirements):
        row = _keys(requirement, {"requirement_id", "text"}, f"TLIF row {index}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ReadinessFailure(f"TLIF row {index}: requirement text is empty")
        if row["text"] in observed_text:
            raise ReadinessFailure(f"TLIF row {index}: duplicate requirement text")
        observed_ids.append(row["requirement_id"])
        observed_text.add(row["text"])
    if tuple(observed_ids) != INTERFACE_REQUIREMENT_IDS:
        raise ReadinessFailure("interface_contract: 20-row requirement population changed")


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
    if gate["independent_exports"] != {
        "accepted": 0,
        "population": 2,
        "required": 2,
    }:
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
        raise ReadinessFailure("TD-P1: nine-command invocation population changed")
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
        {"consumer_requirements", "interface_contract", "schema", "status", "upstream_gate"},
        "requirements",
    )
    if document["schema"] != SCHEMA:
        raise ReadinessFailure("requirements: schema identity changed")
    if document["status"] != STATUS:
        raise ReadinessFailure("requirements: readiness-only disclaimer changed")
    _validate_interface_contract(document["interface_contract"])
    _validate_upstream(document["upstream_gate"])
    expected_consumers = _expected_consumers()
    if document["consumer_requirements"] != expected_consumers:
        raise ReadinessFailure("requirements: TD-P1/TD-HW child or campaign surface changed")
    _validate_consumer_paths(expected_consumers)
    _validate_invocations()


def _validate_case_matrix(
    value: Any,
    expected_ids: tuple[str, ...],
    expected_outcomes: dict[str, tuple[str, ...]],
    label: str,
) -> None:
    if not isinstance(value, list):
        raise ReadinessFailure(f"{label}: matrix is not a list")
    observed_ids: list[str] = []
    for index, item in enumerate(value):
        row = _keys(item, {"case_id", "expected_outcomes"}, f"{label} row {index}")
        case_id = row["case_id"]
        if case_id not in expected_outcomes:
            raise ReadinessFailure(f"{label} row {index}: unexpected case {case_id!r}")
        if tuple(row["expected_outcomes"]) != expected_outcomes[case_id]:
            raise ReadinessFailure(f"{label} row {index}: expected outcome changed")
        observed_ids.append(case_id)
    if tuple(observed_ids) != expected_ids:
        raise ReadinessFailure(f"{label}: case population or order changed")


def _validate_evidence_requirements(value: Any) -> None:
    if not isinstance(value, list):
        raise ReadinessFailure("campaign: evidence requirements are not a list")
    observed_ids: list[str] = []
    observed_text: set[str] = set()
    for index, item in enumerate(value):
        row = _keys(item, {"requirement_id", "text"}, f"evidence row {index}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ReadinessFailure(f"evidence row {index}: text is empty")
        if row["text"] in observed_text:
            raise ReadinessFailure(f"evidence row {index}: duplicate text")
        observed_ids.append(row["requirement_id"])
        observed_text.add(row["text"])
    if tuple(observed_ids) != EVIDENCE_REQUIREMENT_IDS:
        raise ReadinessFailure("campaign: evidence requirement population changed")


def _expected_profile_targets() -> list[dict[str, Any]]:
    return [
        {
            "baseline_case_ids": ["TD-P1-BASELINE"],
            "chain_targets": [
                "td-p1-hardware-replay",
                "td-p1-model-sizer-replay",
            ],
            "common_targets": ["td-p1-candidate-validator"],
            "profile_id": "TD-P1",
            "replay_startup_targets": [
                "td-p1-hardware-replay",
                "td-p1-model-sizer-replay",
            ],
        },
        {
            "baseline_case_ids": ["TD-HW-BASELINE"],
            "chain_targets": [
                "td-hw-unittest",
                "td-hw-staged-show",
                "td-hw-staged-inventory",
                "td-hw-staged-gpu",
                "td-hw-staged-invalid-argv",
            ],
            "common_targets": [
                "td-hw-make-hardware-check",
                "td-hw-staged-show",
                "td-hw-staged-inventory",
                "td-hw-staged-gpu",
                "td-hw-staged-invalid-argv",
            ],
            "profile_id": "TD-HW",
            "replay_startup_targets": [],
        },
    ]


def validate_campaign(value: Any) -> tuple[int, int, int]:
    campaign = _keys(
        value,
        {
            "accepted_grade_states",
            "chain_cases",
            "common_cases",
            "evidence_requirements",
            "exports",
            "family_row_completion",
            "profile_targets",
            "rejecting_grade_states",
            "replay_startup_matrix",
            "row_population",
            "schema",
            "status",
            "verdict_rules",
        },
        "campaign",
    )
    if campaign["schema"] != CAMPAIGN_SCHEMA or campaign["status"] != CAMPAIGN_STATUS:
        raise ReadinessFailure("campaign: schema or no-results status changed")
    if campaign["accepted_grade_states"] != ["PASS", "REFUSED-AS-NAMED"]:
        raise ReadinessFailure("campaign: accepted grade states changed")
    if campaign["rejecting_grade_states"] != ["NULL", "HARNESS-FAIL"]:
        raise ReadinessFailure("campaign: rejecting grade states changed")
    _validate_case_matrix(
        campaign["common_cases"],
        COMMON_MATRIX_IDS,
        EXPECTED_COMMON_OUTCOMES,
        "common matrix",
    )
    _validate_case_matrix(
        campaign["chain_cases"],
        CHAIN_MATRIX_IDS,
        EXPECTED_CHAIN_OUTCOMES,
        "chain matrix",
    )
    _validate_evidence_requirements(campaign["evidence_requirements"])
    if campaign["exports"] != {"independent": True, "population": 2, "required": 2}:
        raise ReadinessFailure("campaign: two-of-two export requirement changed")
    if campaign["family_row_completion"] != {
        "aggregation": "one-row-per-case-family-target-export",
        "required_result_fields": list(FAMILY_RESULT_FIELDS),
        "variant_coverage": {
            "enumeration": "complete-one-change-and-required-combination-population-for-the-case-family",
            "population": "derived-and-frozen-per-case-target-export-before-execution",
            "reachability": "every-required-variant-proves-mutation-delta-and-production-reachability",
        },
        "verdict": "variants-required-equals-variants-executed-equals-variants-pass-plus-variants-refused-as-named-and-variants-null-plus-variants-harness-fail-equals-zero",
    }:
        raise ReadinessFailure("campaign: case-family completion contract changed")
    expected_targets = _expected_profile_targets()
    if campaign["profile_targets"] != expected_targets:
        raise ReadinessFailure("campaign: profile target population changed")

    replay = _keys(
        campaign["replay_startup_matrix"],
        {
            "additional_mutations",
            "membership_modes",
            "mutation_kinds",
            "population_per_target",
            "roots",
        },
        "replay_startup_matrix",
    )
    if replay["additional_mutations"] != [
        {
            "expected_outcomes": ["TL-EXECUTION-CHAIN"],
            "mutation_id": "replace-python3-on-PATH",
        }
    ]:
        raise ReadinessFailure("campaign: replay PATH mutation changed")
    if replay["membership_modes"] != [
        {"expected_outcomes": ["TL-SUBJECT-CLOSURE"], "mode": "unlisted"},
        {
            "expected_outcomes": ["TL-RESERVED-STARTUP"],
            "mode": "listed-in-regenerated-manifest",
        },
    ]:
        raise ReadinessFailure("campaign: replay membership modes changed")
    if replay["mutation_kinds"] != [
        "sitecustomize.py",
        "usercustomize.py",
        "executable.pth",
    ]:
        raise ReadinessFailure("campaign: replay startup mutation kinds changed")
    if replay["roots"] != [
        "contracts/p1-candidate/tools/replay-bin",
        "contracts/p1-candidate/tools",
    ]:
        raise ReadinessFailure("campaign: replay-visible root population changed")
    replay_population = (
        len(replay["additional_mutations"])
        + len(replay["membership_modes"])
        * len(replay["mutation_kinds"])
        * len(replay["roots"])
    )
    if replay["population_per_target"] != replay_population or replay_population != 13:
        raise ReadinessFailure("campaign: replay startup denominator changed")

    case_definitions = 0
    for profile in expected_targets:
        case_definitions += len(profile["baseline_case_ids"])
        case_definitions += len(profile["common_targets"]) * len(COMMON_MATRIX_IDS)
        case_definitions += len(profile["chain_targets"]) * len(CHAIN_MATRIX_IDS)
        case_definitions += len(profile["replay_startup_targets"]) * replay_population
    export_population = campaign["exports"]["population"]
    case_export_rows = case_definitions * export_population
    expected_population = {
        "aggregation_unit": "case-family-target-export",
        "case_family_target_definitions": 316,
        "independent_exports": 2,
        "required_case_family_target_export_rows": 632,
        "required_nonrejecting_rows": 632,
    }
    if campaign["row_population"] != expected_population:
        raise ReadinessFailure("campaign: recorded row population changed")
    if (case_definitions, export_population, case_export_rows) != (316, 2, 632):
        raise ReadinessFailure("campaign: computed row population changed")
    if campaign["verdict_rules"] != [
        "every-required-row-is-PASS-or-REFUSED-AS-NAMED",
        "every-case-family-row-carries-a-complete-frozen-variant-manifest",
        "every-required-variant-is-executed-and-nonrejecting",
        "zero-required-row-is-NULL-or-HARNESS-FAIL",
        "every-mutation-proves-delta-and-production-reachability",
        "every-profile-preserves-subject-evidence-runtime-and-dependency-bytes",
        "both-independent-exports-bind-identical-product-profile-and-interface-identities",
        "zero-open-finding-at-every-severity",
        "launcher-integrity-does-not-promote-hardware-coverage-or-P1-freeze",
    ]:
        raise ReadinessFailure("campaign: verdict rules changed")
    return case_definitions, export_population, case_export_rows


def _requirements_mutations() -> list[Callable[[dict[str, Any]], None]]:
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

    def claim_interface_mapping(value: dict[str, Any]) -> None:
        value["interface_contract"]["conformance_return"]["required_rows"]["accepted"] = 1

    def claim_upstream_export(value: dict[str, Any]) -> None:
        value["upstream_gate"]["independent_exports"]["accepted"] = 1

    def remove_interface_requirement(value: dict[str, Any]) -> None:
        value["interface_contract"]["requirements"].pop()

    return [
        remove_case,
        promote_assignment,
        add_d4_child,
        change_interface_owner,
        claim_interface_mapping,
        claim_upstream_export,
        remove_interface_requirement,
    ]


def _campaign_mutations() -> list[Callable[[dict[str, Any]], None]]:
    def remove_common_case(value: dict[str, Any]) -> None:
        value["common_cases"].pop()

    def remove_hardware_surface(value: dict[str, Any]) -> None:
        value["profile_targets"][1]["common_targets"].pop()

    def weaken_export_requirement(value: dict[str, Any]) -> None:
        value["exports"]["required"] = 1

    def admit_null_row(value: dict[str, Any]) -> None:
        value["row_population"]["required_nonrejecting_rows"] = 631

    def weaken_family_completion(value: dict[str, Any]) -> None:
        value["family_row_completion"]["required_result_fields"].remove(
            "variants_required"
        )

    def remove_evidence_requirement(value: dict[str, Any]) -> None:
        value["evidence_requirements"].pop()

    def change_expected_outcome(value: dict[str, Any]) -> None:
        value["common_cases"][0]["expected_outcomes"] = ["PASS"]

    def shrink_replay_population(value: dict[str, Any]) -> None:
        value["replay_startup_matrix"]["population_per_target"] = 12

    return [
        remove_common_case,
        remove_hardware_surface,
        weaken_export_requirement,
        admit_null_row,
        weaken_family_completion,
        remove_evidence_requirement,
        change_expected_outcome,
        shrink_replay_population,
    ]


def _self_test(requirements: dict[str, Any], campaign: dict[str, Any]) -> tuple[int, int]:
    rejected = 0
    mutations = _requirements_mutations()
    for mutate in mutations:
        candidate = copy.deepcopy(requirements)
        mutate(candidate)
        try:
            validate_requirements(candidate)
        except ReadinessFailure:
            rejected += 1
    campaign_mutations = _campaign_mutations()
    for mutate in campaign_mutations:
        candidate = copy.deepcopy(campaign)
        mutate(candidate)
        try:
            validate_campaign(candidate)
        except ReadinessFailure:
            rejected += 1
    population = len(mutations) + len(campaign_mutations)
    if rejected != population:
        raise ReadinessFailure(
            f"readiness self-test rejected {rejected}/{population} mutations"
        )
    return rejected, population


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    requirements = _load_json(arguments.requirements.resolve(strict=True))
    campaign = _load_json(arguments.campaign.resolve(strict=True))
    if not isinstance(requirements, dict):
        raise ReadinessFailure("requirements: top level is not an object")
    if not isinstance(campaign, dict):
        raise ReadinessFailure("campaign: top level is not an object")
    validate_requirements(requirements)
    case_definitions, export_population, case_export_rows = validate_campaign(campaign)
    self_test_result = _self_test(requirements, campaign) if arguments.self_test else None
    message = (
        "PASS (developer readiness only; adoption remains blocked): "
        f"{len(COMMON_CASES)}/{len(COMMON_CASES)} common cases retained; "
        f"{len(INTERFACE_REQUIREMENT_IDS)}/{len(INTERFACE_REQUIREMENT_IDS)} interface requirements retained; "
        f"0/{len(INTERFACE_REQUIREMENT_IDS)} interface conformance rows accepted; "
        "5/5 forbidden interface shapes retained; 2/2 consumer requirements retained; "
        "8/8 intentional child specifications retained; "
        f"{len(P1_INVOCATIONS)}/{len(P1_INVOCATIONS)} P1 invocation vectors retained; "
        f"{len(COMMON_MATRIX_IDS)}/{len(COMMON_MATRIX_IDS)} common matrix definitions retained; "
        f"{len(CHAIN_MATRIX_IDS)}/{len(CHAIN_MATRIX_IDS)} chain definitions retained; "
        "13/13 replay-startup variants retained; "
        f"{len(FAMILY_RESULT_FIELDS)}/{len(FAMILY_RESULT_FIELDS)} case-family completion fields retained; "
        f"{case_definitions}/{case_definitions} case-family target definitions retained; "
        f"{case_export_rows}/{case_export_rows} case-family target-export rows specified across "
        f"{export_population}/{export_population} exports; "
        f"{len(EVIDENCE_REQUIREMENT_IDS)}/{len(EVIDENCE_REQUIREMENT_IDS)} evidence requirements retained; "
        f"0/{case_export_rows} result rows present; "
        f"0/{export_population} upstream independent exports accepted; "
        f"0/{len(RETURN_IDENTITIES)} upstream return identities consumed"
    )
    if self_test_result is not None:
        rejected, mutation_population = self_test_result
        message += f"; {rejected}/{mutation_population} omission/premature-adoption mutations rejected"
    print(message)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ReadinessFailure, tomllib.TOMLDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
