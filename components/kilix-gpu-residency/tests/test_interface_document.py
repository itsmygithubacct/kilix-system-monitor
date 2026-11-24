"""The module and its kilix.gpu-residency/v1 interface document cannot drift apart.

Each comparison also runs against planted drifts (a perturbed copy of the
document, or mutated module source), so a pass cannot come from a check that
did not look.
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
import re
import tomllib
import unittest

import residency_containment  # noqa: F401
import kilix_gpu_residency as residency
from kilix_gpu_residency import catalog, cli

COMPONENT = Path(__file__).resolve().parents[1]
ROOT = COMPONENT.parents[1]
DOCUMENT = ROOT / "contracts" / "kilix.gpu-residency-v1.interface.json"
SOURCES = sorted((COMPONENT / "src" / "kilix_gpu_residency").glob("*.py"))
KEYS = {"schema", "version", "distribution", "python_module", "command", "namespace_default", "label_pattern",
        "backends", "states", "unload_kinds", "policies", "default_policy", "error_codes", "record_fields",
        "max_records", "max_events", "lock_wait_seconds", "term_grace_seconds", "kill_grace_seconds",
        "release_wait_seconds", "device_map_env", "status_schema", "room_schema", "eviction", "ollama",
        "device_lease", "catalog"}


def strict_load(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate interface key: {key}")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def load() -> dict:
    return strict_load(DOCUMENT.read_bytes())


def differences(module, document) -> list[str]:
    signature = inspect.signature(module.ensure_room).parameters
    found = []
    pairs = (
        ("version", module.VERSION, document["version"]),
        ("python_module", module.__name__, document["python_module"]),
        ("namespace_default", module.NAMESPACE_DEFAULT, document["namespace_default"]),
        ("label_pattern", module._LABEL.pattern, document["label_pattern"]),
        ("backends", module.BACKENDS, tuple(document["backends"])),
        ("states", module.STATES, tuple(document["states"])),
        ("unload_kinds", module.UNLOAD_KINDS, tuple(document["unload_kinds"])),
        ("policies", module.POLICIES, tuple(document["policies"])),
        ("default_policy", signature["policy"].default, document["default_policy"]),
        ("default_policy(register)", inspect.signature(module.register).parameters["policy"].default,
         document["default_policy"]),
        ("error_codes", module.ERROR_CODES, tuple(document["error_codes"])),
        ("record_fields", sorted(module._RECORD_FIELDS), document["record_fields"]),
        ("max_records", module.MAX_RECORDS, document["max_records"]),
        ("max_events", module.MAX_EVENTS, document["max_events"]),
        ("lock_wait_seconds", module.LOCK_WAIT_SECONDS, float(document["lock_wait_seconds"])),
        ("term_grace_seconds", module.TERM_GRACE_SECONDS, float(document["term_grace_seconds"])),
        ("kill_grace_seconds", module.KILL_GRACE_SECONDS, float(document["kill_grace_seconds"])),
        ("release_wait_seconds", module.RELEASE_WAIT_SECONDS, float(document["release_wait_seconds"])),
        ("device_map_env", module.DEVICE_MAP_ENV, document["device_map_env"]),
        ("status_schema", cli.STATUS_SCHEMA, document["status_schema"]),
        ("room_schema", cli.ROOM_SCHEMA, document["room_schema"]),
    )
    for key, observed, expected in pairs:
        if observed != expected or type(observed) is not type(expected):
            found.append(f"{key}: module {observed!r}, document {expected!r}")
    if not module._LABEL.flags & re.ASCII:
        found.append("label_pattern: not ASCII-only")
    return found


def error_codes(sources: dict[str, str]) -> tuple[set[str], list[str]]:
    """Every code a ResidencyError or RoomError can carry, and every route the scan cannot follow."""
    codes, problems = set(), []
    for name, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if callee == "_refuse":
                codes.add("unavailable")
            elif callee == "_invalid":
                codes.add("invalid-request")
            elif callee == "ResidencyError":
                first = node.args[0] if node.args else None
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    codes.add(first.value)
                else:
                    problems.append(f"{name}:{node.lineno}: ResidencyError with a computed code")
            elif callee == "RoomError":
                codes.add("no-room")
                for keyword in node.keywords:
                    if keyword.arg == "code":
                        if isinstance(keyword.value, ast.Constant):
                            codes.add(keyword.value.value)
                        else:
                            problems.append(f"{name}:{node.lineno}: RoomError with a computed code")
    return codes, problems


def catalog_differences(module, document) -> list[str]:
    section = document["catalog"]
    packaged = json.loads((COMPONENT / "src" / "kilix_gpu_residency" / "catalog.json").read_text())
    found = []
    pairs = (
        ("python_module", module.__name__, section["python_module"]),
        ("actions", module.ACTIONS, tuple(section["actions"])),
        ("data_schema", module.SCHEMA, section["data_schema"]),
        ("result_schema", module.RESULT_SCHEMA, section["result_schema"]),
        ("kinds", module.KINDS, tuple(section["kinds"])),
        ("folders", packaged["folders"], section["folders"]),
        ("url_pattern", module._HF.pattern, section["url_pattern"]),
        ("lock_name", module.LOCK_NAME, section["lock_name"]),
        ("part_template", module.PART_TEMPLATE, section["part_template"]),
        ("attempts", module.ATTEMPTS, section["attempts"]),
        ("lock_wait_seconds", module.LOCK_WAIT_SECONDS, float(section["lock_wait_seconds"])),
        ("release_wait_seconds", module.RELEASE_WAIT_SECONDS, float(section["release_wait_seconds"])),
        ("max_redirects", module.MAX_REDIRECTS, section["max_redirects"]),
        ("outcomes", module.OUTCOMES, section["outcomes"]),
        ("exit_codes", {str(k): v for k, v in module.EXIT_MEANINGS.items()}, section["exit_codes"]),
        ("file_fields", sorted(module._FILE_FIELDS), section["file_fields"]),
        ("ollama_fields", sorted(module._OLLAMA_FIELDS), section["ollama_fields"]),
    )
    for key, observed, expected in pairs:
        if observed != expected or type(observed) is not type(expected):
            found.append(f"{key}: module {observed!r}, document {expected!r}")
    if set(module.OUTCOMES.values()) != set(module.EXIT_MEANINGS):
        found.append("exit_codes: an outcome has an undocumented exit code")
    return found


def outcomes(source: str) -> tuple[set[str], list[str]]:
    """Every outcome catalog.py can produce: CatalogError's first and Result's second argument."""
    found, problems = set(), []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        position = {"CatalogError": 0, "Result": 1}.get(node.func.id)
        if position is None:
            continue
        value = node.args[position] if len(node.args) > position else None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found.add(value.value)
        else:
            problems.append(f"{node.lineno}: {node.func.id} with a computed outcome")
    return found, problems


class InterfaceDocumentTests(unittest.TestCase):
    def test_the_catalog_section_matches_the_catalog_module(self):
        document = load()
        self.assertEqual(catalog_differences(catalog, document), [])
        planted = {"attempts": 4, "lock_name": ".lock", "kinds": ["gguf"], "result_schema": "x",
                   "outcomes": {**catalog.OUTCOMES, "removed": 1}, "exit_codes": {"0": "done"},
                   "part_template": "{file}.part", "url_pattern": ".*",
                   "file_fields": ["id"], "ollama_fields": sorted(catalog._OLLAMA_FIELDS - {"digest"})}
        for key, value in planted.items():
            with self.subTest(planted=key):
                drifted = copy.deepcopy(document)
                drifted["catalog"][key] = value
                self.assertTrue(any(line.startswith(key) for line in catalog_differences(catalog, drifted)))
        self.assertEqual(document["catalog"]["ollama"], {
            "list": "GET /api/tags", "check": "POST /api/show", "pull": "POST /api/pull stream",
            "delete": "DELETE /api/delete", "loopback_only": True,
            "pin": "an ollama entry's digest and bytes are what GET /api/tags reports: the registry manifest's sha256, "
                   "and its layers and config"})
        source = (COMPONENT / "src" / "kilix_gpu_residency" / "ollama.py").read_text(encoding="utf-8")
        for call in ('"GET", "/api/tags"', '"POST", "/api/show"', '"POST", "/api/pull", {"model": name, "stream": True}',
                     '"DELETE", "/api/delete"'):
            self.assertIn(call, source)

    def test_the_outcomes_are_every_outcome_the_catalog_can_produce(self):
        source = (COMPONENT / "src" / "kilix_gpu_residency" / "catalog.py").read_text(encoding="utf-8")
        found, problems = outcomes(source)
        self.assertEqual(problems, [])
        self.assertEqual(found, set(load()["catalog"]["outcomes"]))
        self.assertIn("exploded", outcomes(source + '\nraise CatalogError("exploded", "x")\n')[0])
        self.assertTrue(outcomes(source + "\nraise CatalogError(code, 'x')\n")[1])

    def test_the_document_has_exactly_its_keys_and_no_duplicates(self):
        document = load()
        self.assertEqual(set(document), KEYS)
        self.assertEqual(document["schema"], "kilix.gpu-residency.interface/v1")
        with self.assertRaises(ValueError):
            strict_load(b'{"version": 1, "version": 2}')

    def test_every_documented_constant_matches_the_module(self):
        document = load()
        self.assertEqual(differences(residency, document), [])
        planted = {"version": "kilix.gpu-residency/v2", "max_records": 65, "policies": ["evict-idle"],
                   "default_policy": "never", "term_grace_seconds": 6, "label_pattern": "[a-z]+",
                   "record_fields": document["record_fields"][1:], "error_codes": document["error_codes"][:-1],
                   "status_schema": "x", "backends": ["process"]}
        for key, value in planted.items():
            with self.subTest(planted=key):
                drifted = copy.deepcopy(document)
                drifted[key] = value
                self.assertTrue(any(line.startswith(key) for line in differences(residency, drifted)))

    def test_the_error_codes_are_every_code_the_source_can_raise(self):
        sources = {path.name: path.read_text(encoding="utf-8") for path in SOURCES}
        codes, problems = error_codes(sources)
        self.assertEqual(problems, [])
        self.assertEqual(codes, set(load()["error_codes"]))
        planted = dict(sources)
        planted["__init__.py"] += '\n\ndef _planted():\n    raise ResidencyError("exploded", "x")\n'
        self.assertIn("exploded", error_codes(planted)[0])
        planted = dict(sources)
        planted["__init__.py"] += '\n\ndef _planted(c):\n    raise ResidencyError(c, "x")\n'
        self.assertTrue(error_codes(planted)[1])
        planted = dict(sources)
        planted["__init__.py"] += '\n\ndef _planted(c):\n    raise RoomError("x", need_bytes=1, free_bytes=1, device=None, code=c)\n'
        self.assertTrue(error_codes(planted)[1])

    def test_the_residency_module_neither_uses_nor_changes_the_device_lease(self):
        self.assertEqual(load()["device_lease"], {"uses": False, "changes": False})
        for path in SOURCES:
            self.assertNotIn("kilix_device_lease", path.read_text(encoding="utf-8"), path.name)

    def test_the_documented_eviction_rules_and_ollama_calls(self):
        document = load()
        self.assertEqual(document["eviction"], {
            "candidate_state": "idle", "order": "least-recently-used", "all_or_nothing": True,
            "unknown_ollama_models_evictable": True,
            "never_evicted": ["busy", "loading", "unloading", "pinned", "unmanaged", "no-unload-action",
                              "other-device"]})
        self.assertEqual(document["ollama"], {"discovery": "GET /api/ps", "unload": "POST /api/generate keep_alive 0",
                                              "loopback_only": True})
        source = (COMPONENT / "src" / "kilix_gpu_residency" / "ollama.py").read_text(encoding="utf-8")
        self.assertIn('"GET", "/api/ps"', source)
        self.assertIn('"POST", "/api/generate", {"model": name, "keep_alive": 0}', source)

    def test_the_distribution_command_and_registrations_name_this_document(self):
        document = load()
        project = tomllib.loads((COMPONENT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(project["name"], document["distribution"])
        self.assertEqual(project["scripts"], {document["command"]: "kilix_gpu_residency.cli:main"})
        self.assertEqual(project["dependencies"], [])
        manifest = tomllib.loads((ROOT / "manifest.toml").read_text(encoding="utf-8"))
        [entry] = [c for c in manifest["components"] if c["id"] == document["distribution"]]
        self.assertEqual(entry["path"], "components/kilix-gpu-residency")
        self.assertEqual(entry["interface"], "contracts/kilix.gpu-residency-v1.interface.json")
        self.assertEqual(entry["version"], project["version"])


if __name__ == "__main__":
    unittest.main()
