"""The module and its kilix.gpu-residency/v1 interface document cannot drift apart.

Each comparison also runs against planted drifts (a perturbed copy of the
document, or mutated module source), so a pass cannot come from a check that
did not look.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import inspect
import io
import json
import os
from pathlib import Path
import re
import socket
import time
import tomllib
from types import SimpleNamespace
import unittest

import residency_containment  # noqa: F401
import kilix_gpu_residency as residency
from kilix_gpu_residency import catalog, cli, vram
from fakes import GIB, MIB, ResidencyCase

COMPONENT = Path(__file__).resolve().parents[1]
ROOT = COMPONENT.parents[1]
DOCUMENT = ROOT / "contracts" / "kilix.gpu-residency-v1.interface.json"
SOURCES = sorted((COMPONENT / "src" / "kilix_gpu_residency").glob("*.py"))
KEYS = {"schema", "version", "distribution", "python_module", "command", "namespace_default", "label_pattern",
        "backends", "states", "unload_kinds", "policies", "default_policy", "error_codes", "record_fields",
        "max_records", "max_events", "lock_wait_seconds", "term_grace_seconds", "kill_grace_seconds",
        "release_wait_seconds", "device_map_env", "status_schema", "room_schema", "eviction", "ollama",
        "device_lease", "catalog", "api", "types", "room_error_codes", "raises", "load_timeout_seconds",
        "recent_use_seconds", "reservation", "devices", "registry", "forward_compatibility", "gaps", "cli"}
PUBLIC_TYPES = ("ResidentInfo", "Room", "Eviction", "Status", "Foreign", "Unmanaged")


def signature(function) -> str:
    """A callable's parameters as the document writes them: names, ``*`` and defaults."""
    parts, star = [], False
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind is parameter.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        if parameter.kind is parameter.VAR_KEYWORD:
            parts.append("**" + parameter.name)
            continue
        parts.append(parameter.name + ("" if parameter.default is parameter.empty else "=" + repr(parameter.default)))
    return "(" + ", ".join(parts) + ")"


def api_differences(module, document) -> list[str]:
    found = []
    for name, expected in document["api"].items():
        owner, _, attribute = name.rpartition(".")
        target = getattr(getattr(module, owner), attribute) if owner else getattr(module, name, None)
        if target is None or signature(target) != expected:
            found.append(f"api {name}: module {None if target is None else signature(target)}, document {expected}")
    public = {name for name, value in vars(module).items() if inspect.isfunction(value)
              and not name.startswith("_") and value.__module__ == module.__name__}
    for name in sorted(public - set(document["api"])):
        found.append(f"api {name}: public in the module, not in the document")
    for name in PUBLIC_TYPES:
        fields = [f.name for f in dataclasses.fields(getattr(module, name))]
        if fields != document["types"].get(name):
            found.append(f"types {name}: module {fields}, document {document['types'].get(name)}")
    room = inspect.signature(module.RoomError.__init__).parameters
    if "evicted" not in room or "code" not in room:
        found.append("types RoomError: no evicted or code")
    return found


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
        ("room_error_codes", module.ROOM_ERROR_CODES, tuple(document["room_error_codes"])),
        ("load_timeout_seconds", module.LOAD_TIMEOUT_SECONDS, float(document["load_timeout_seconds"])),
        ("recent_use_seconds", module.RECENT_USE_SECONDS, float(document["recent_use_seconds"])),
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
        ("legacy_part_template", module.LEGACY_PART_TEMPLATE, section["legacy_part_template"]),
        ("error_fields", sorted(set(inspect.signature(module.CatalogError.__init__).parameters)
                                - {"self", "outcome", "message"}), section["error_fields"]),
    )
    for key, observed, expected in pairs:
        if observed != expected or type(observed) is not type(expected):
            found.append(f"{key}: module {observed!r}, document {expected!r}")
    if set(module.OUTCOMES.values()) != set(module.EXIT_MEANINGS):
        found.append("exit_codes: an outcome has an undocumented exit code")
    for name, expected in section["api"].items():
        observed = signature(getattr(module, name))
        if observed != expected:
            found.append(f"api {name}: module {observed}, document {expected}")
    for name, expected in section["types"].items():
        fields = [f.name for f in dataclasses.fields(getattr(module, name))]
        if fields != expected:
            found.append(f"types {name}: module {fields}, document {expected}")
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
                   "part_template": ".{file}.part", "url_pattern": ".*", "legacy_part_template": "x",
                   "error_fields": ["detail"], "api": {**document["catalog"]["api"], "remove": "(target)"},
                   "file_fields": ["id"], "ollama_fields": sorted(catalog._OLLAMA_FIELDS - {"digest"})}
        for key, value in planted.items():
            with self.subTest(planted=key):
                drifted = copy.deepcopy(document)
                drifted["catalog"][key] = value
                self.assertTrue(any(line.startswith(key) for line in catalog_differences(catalog, drifted)), key)
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
                   "status_schema": "x", "backends": ["process"], "recent_use_seconds": 60,
                   "load_timeout_seconds": 1, "room_error_codes": ["no-room"]}
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

    def test_the_ollama_calls(self):
        document = load()
        self.assertEqual(document["ollama"], {"discovery": "GET /api/ps", "unload": "POST /api/generate keep_alive 0",
                                              "loopback_only": True})
        source = (COMPONENT / "src" / "kilix_gpu_residency" / "ollama.py").read_text(encoding="utf-8")
        self.assertIn('"GET", "/api/ps"', source)
        self.assertIn('"POST", "/api/generate", {"model": name, "keep_alive": 0}', source)

    def test_the_api_and_its_types_match_the_module(self):
        document = load()
        self.assertEqual(api_differences(residency, document), [])
        for key, value in (("register", "(*, app)"), ("Resident.close", "()"), ("default_namespace", None)):
            with self.subTest(planted=key):
                drifted = copy.deepcopy(document)
                if value is None:
                    del drifted["api"][key]
                else:
                    drifted["api"][key] = value
                self.assertTrue(any(line.startswith(f"api {key}") for line in api_differences(residency, drifted)))
        drifted = copy.deepcopy(document)
        drifted["types"]["Eviction"] = drifted["types"]["Eviction"][:-1]
        self.assertTrue(any(line.startswith("types Eviction") for line in api_differences(residency, drifted)))

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


def _unit(**changes) -> residency._Unit:
    """An idle, unpinned process-group unit on 0000:01:00.0 holding 1 GiB: a candidate."""
    record = {"id": "a" * 32, "state": "idle", "claim": None, "pinned": False, "last_used": time.time() - 3600}
    unit = residency._Unit("pgid:4242", "signal", [record], pgid=4242, device_pci="0000:01:00.0",
                           vram_bytes=GIB, last_used=record["last_used"])
    for key, value in changes.items():
        if key == "state":
            record["state"] = value
        else:
            setattr(unit, key, value)
    return unit


def _ollama_unit(last_used: float | None, registered: bool = False) -> residency._Unit:
    loaded = SimpleNamespace(name="m:1b", vram_bytes=GIB)
    records = [{"id": "b" * 32, "state": "idle", "claim": None, "pinned": False, "last_used": last_used}] if registered else []
    return residency._Unit("ollama:m:1b", "ollama", records, model="m:1b", loaded=loaded, device_pci="0000:01:00.0",
                           vram_bytes=GIB, last_used=last_used)


# Each reason the document says a resident is never evicted, as a unit that has only that reason.
NEVER = {
    "busy": lambda: _unit(state="busy"),
    "loading": lambda: _unit(state="loading"),
    "unloading": lambda: _claimed(),  # marked by an unloader that is still running
    "pinned": lambda: _unit(pinned=True),
    "no-unload-action": lambda: _unit(kind=None),
    "other-device": lambda: _unit(device_pci="0000:02:00.0"),
    "no-vram": lambda: _unit(vram_bytes=0),
    "recently-used-unregistered-ollama": lambda: _ollama_unit(time.time() - 1),
}
# Also never planned, under the same reason: an unregistered model whose last use cannot be estimated.
UNKNOWN_LAST_USE = lambda: _ollama_unit(None)  # noqa: E731
# Never units at all, so never candidates: listed for completeness, probed elsewhere.
NOT_UNITS = {"unmanaged": "test_ensure_room / test_registry: unmanaged processes are never residents",
             "foreign": "test_fix_round_1.ForeignRecordTests"}


def _claimed() -> residency._Unit:
    unit = _unit(state="unloading")
    unit.records[0]["claim"] = {"pid": os.getpid(), "start": residency._start(os.getpid())}  # a live unloader
    return unit


class EvictionRuleTests(unittest.TestCase):
    """The document's eviction rules, held to the module by what the module does."""

    def test_every_never_evicted_reason_keeps_a_unit_out_of_the_plan(self):
        document = load()["eviction"]
        self.assertEqual(set(document["never_evicted"]), set(NEVER) | set(NOT_UNITS))
        now = time.time()
        self.assertTrue(residency._candidate(_unit(), "0000:01:00.0", now))  # the probe can say yes
        self.assertTrue(residency._candidate(_ollama_unit(now - 3600), "0000:01:00.0", now))
        self.assertTrue(residency._candidate(_ollama_unit(now - 1, registered=True), "0000:01:00.0", now))
        self.assertFalse(residency._candidate(UNKNOWN_LAST_USE(), "0000:01:00.0", now))
        for reason, make in NEVER.items():
            with self.subTest(reason=reason):
                self.assertFalse(residency._candidate(make(), "0000:01:00.0", now))
        # Every record of a unit counts: one busy record among idle ones keeps it.
        shared = _unit()
        shared.records.append({**shared.records[0], "id": "c" * 32, "state": "busy"})
        self.assertFalse(residency._candidate(shared, "0000:01:00.0", now))

    def view(self, units, free, ollama_bytes=0):
        device = SimpleNamespace(pci="0000:01:00.0", free_bytes=free)
        return SimpleNamespace(devices=[device], units=units, reserved={}, others=[],
                               ollama_bytes={"0000:01:00.0": ollama_bytes})

    def test_the_order_and_all_or_nothing(self):
        document = load()["eviction"]
        self.assertTrue(document["order"].startswith("registered units first"))
        now = time.time()
        old_ollama = _ollama_unit(now - 7200)
        old_ollama.key = "ollama:old"
        newer, older = _unit(last_used=now - 600), _unit(last_used=now - 1200)
        newer.key, older.key = "pgid:2", "pgid:3"
        free, plan, freed = residency._plan(self.view([old_ollama, newer, older], 0, ollama_bytes=GIB),
                                            "0000:01:00.0", 2 * GIB)
        self.assertEqual([u.key for u in plan], ["pgid:3", "pgid:2"])  # registered, least recently used first
        free, plan, freed = residency._plan(self.view([old_ollama, newer, older], 0, ollama_bytes=GIB),
                                            "0000:01:00.0", 3 * GIB)
        self.assertEqual([u.key for u in plan], ["pgid:3", "pgid:2", "ollama:old"])
        # An Ollama model counts for no more than Ollama's processes measurably hold.
        free, plan, freed = residency._plan(self.view([old_ollama], 0, ollama_bytes=GIB // 2), "0000:01:00.0", GIB)
        self.assertEqual(freed, GIB // 2)
        self.assertIn("nothing is unloaded when the plan cannot meet the need", document["all_or_nothing"])


class ContractBehaviourTests(ResidencyCase):
    """The CLI's JSON keys, exit codes and Ollama states, from running it."""

    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(["--namespace", self.namespace, "--ollama", self.ollama.url, *argv], out, err)
        return code, out.getvalue(), err.getvalue()

    def test_status_json_keys(self):
        document = load()["cli"]["status_json"]
        device = vram.Device("0000:01:00.0", "GPU", "NVIDIA", "nvidia", True, False, 1, 1, 0, "nvml")
        info = residency.ResidentInfo("a" * 32, "app", "m", "process", None, None, 1, 2, 0, None, "idle", False,
                                      None, True, "signal", "pgid:2", 0)
        value = residency.Status((device,), (info,), (residency.Unmanaged(3, "x", "0000:01:00.0", 1),), "reachable",
                                 (residency.Eviction("a" * 32, "app", "m", 1, 1.0, "by", "room"),),
                                 (residency.Foreign("b" * 32, "kilix.gpu-residency/v2", "app", "m", 4, 0, "newer-schema"),))
        out = cli.status_json(value)
        self.assertEqual(sorted(out), document["keys"])
        for key, name in (("devices", "device"), ("residents", "resident"), ("unmanaged", "unmanaged"),
                          ("foreign", "foreign"), ("events", "event")):
            with self.subTest(key=key):
                self.assertEqual(sorted(out[key][0]), document[name])

    def test_room_json_keys_and_every_exit_code(self):
        document = load()["cli"]
        _, process = self.model("puzzllm", "m", 4 * GIB)
        for argv, code in ((("room", "1GiB"), 0), (("room", "3GiB"), 0), (("room", "7GiB"), 2),
                           (("room", "1GiB", "--device", "0000:00:02.0"), 1),
                           (("room", "1GiB", "--device", "NoSuchGPU7"), 1)):
            with self.subTest(argv=argv):
                found, out, err = self.cli(*argv, "--json")
                self.assertEqual(found, code, err)
                self.assertIn(str(code), document["exit_codes"])
                if code != 1:
                    value = json.loads(out)
                    self.assertEqual(sorted(value), document["room_json"]["keys"])
                    for row in value["plan"]:
                        self.assertEqual(sorted(row), document["room_json"]["plan"])
        self.assertIsNone(process.poll())
        readme = (COMPONENT / "README.md").read_text(encoding="utf-8")
        [row] = [line for line in readme.splitlines() if line.startswith("| `kilix-gpu room")]
        for code in ("0", "1", "2"):
            self.assertRegex(row, rf"\b{code}\b")

    def test_models_json_keys(self):
        document = load()["cli"]
        listed = catalog.Listed("x", "x", "gguf", "local", "local", "llama.cpp", "/x", 1, None,
                                ({"app": "a"}, {"pid": 1, "command": "c"}))
        value = cli.models_json(catalog.Inventory((listed,), {"llama.cpp": "/x"}, "reachable", "known"))
        self.assertEqual(sorted(value), document["models_list_json"]["keys"])
        self.assertEqual(sorted(value["models"][0]), document["models_list_json"]["model"])
        self.assertEqual([sorted(h) for h in value["models"][0]["resident"]], document["models_list_json"]["resident"])
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(["--namespace", self.namespace, "--ollama", self.ollama.url, "models", "remove",
                         "ollama:../x", "--json"], out, err)
        self.assertEqual(code, 2)
        self.assertEqual(sorted(json.loads(out.getvalue())), document["models_result_json"]["keys"])

    def test_the_ollama_states(self):
        document = load()["cli"]["ollama_states"]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead = f"http://127.0.0.1:{probe.getsockname()[1]}"
        seen = [residency.status(namespace=self.namespace, ollama=url).ollama
                for url in (self.ollama.url, dead, "http://192.0.2.1:11434")]
        self.assertEqual(seen, document)


if __name__ == "__main__":
    unittest.main()
