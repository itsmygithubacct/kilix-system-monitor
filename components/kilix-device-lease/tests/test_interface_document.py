"""The module and its kilix.device-lease/v1 interface document cannot drift apart.

Every comparison here also runs against planted drifts: a perturbed copy of the
document, perturbed module values, or mutated module source. A comparison that
cannot see its planted drift fails, so a pass cannot come from a check that did
not look. Behaviour the document describes is exercised through the real
module, not through constants alone.
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from types import SimpleNamespace
import unittest
from unittest import mock

import kilix_device_lease as leases

COMPONENT = Path(__file__).resolve().parents[1]
DOCUMENT = COMPONENT.parents[1] / "contracts" / "kilix.device-lease-v1.interface.json"
KEYS = {
    "admission", "distribution", "error_codes", "guard_fd", "label_pattern", "max_queue",
    "max_wait_seconds", "max_workload_queue", "namespace_default", "python_module",
    "release_semantics", "schema", "version", "workloads",
}
NESTED = {
    "admission": {"co_residency", "exclusive"},
    "guard_fd": {"cloexec", "inherit_via"},
    "release_semantics": {"cleanup_ack_required", "lock_un", "quarantine_on_unproven"},
}
# Strings a label check must decide exactly as the documented pattern does,
# including the edges a wider or narrower pattern would decide differently.
LABEL_PROBES = (
    "a", "Z", "0", "a" * 96, "a" * 97, "", "_a", ".a", "-a", ":a", "a_b.c:d-e", "A9_.:-",
    "a/b", "a b", "a\n", "a\t", "a\x00", "é", "a" + "é", "a\\b", "a+b", "a@b", "a,b", "a~b",
)


def strict_load(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate interface key: {key}")
            result[key] = value
        return result

    def constant(name):
        raise ValueError(f"non-finite interface number: {name}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def module_source() -> str:
    return Path(leases.__file__).read_text(encoding="utf-8")


def constant_differences(module, document) -> list[str]:
    """Every documented scalar the module exports, compared by value and type."""
    differences = []
    pairs = (
        ("version", module.VERSION, document["version"]),
        ("workloads", module.WORKLOADS, tuple(document["workloads"])),
        ("max_queue", module.MAX_QUEUE, document["max_queue"]),
        ("max_workload_queue", module.MAX_WORKLOAD_QUEUE, document["max_workload_queue"]),
        ("max_wait_seconds", module.MAX_WAIT_SECONDS, float(document["max_wait_seconds"])),
        ("label_pattern", module._LABEL.pattern, document["label_pattern"]),
        ("python_module", module.__name__, document["python_module"]),
    )
    for key, observed, expected in pairs:
        if observed != expected or type(observed) is not type(expected):
            differences.append(f"{key}: module {observed!r}, document {expected!r}")
    if not module._LABEL.flags & re.ASCII:
        differences.append("label_pattern: module pattern is not ASCII-only")
    return differences


def error_code_findings(source: str) -> tuple[set[str], list[str]]:
    """Codes of every LeaseError the source can construct, and every route the scan cannot follow."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    codes: set[str] = set()
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "__class__":
            problems.append(f"line {node.lineno}: __class__ can construct an unscanned error")
        elif isinstance(node, ast.Constant) and node.value == "LeaseError":
            problems.append(f"line {node.lineno}: LeaseError named by a string")
        elif isinstance(node, ast.Name) and node.id == "LeaseError" and isinstance(node.ctx, ast.Load):
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                first = parent.args[0] if parent.args else None
                if isinstance(first, ast.Constant) and type(first.value) is str:
                    codes.add(first.value)
                else:
                    problems.append(f"line {node.lineno}: LeaseError without a literal code")
            elif not (isinstance(parent, ast.ExceptHandler)
                      or (isinstance(parent, ast.Tuple) and isinstance(parents.get(parent), ast.ExceptHandler))):
                problems.append(f"line {node.lineno}: LeaseError used other than by a call or except clause")
    return codes, problems


class RootedOS:
    """The module's os, with the filesystem root moved to a private directory."""

    def __init__(self, root: str) -> None:
        self.root = root

    def __getattr__(self, name):
        return getattr(os, name)

    def open(self, path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None:
            if path != "/":
                raise AssertionError(f"absolute open outside the private root: {path!r}")
            return os.open(self.root, flags, mode)
        return os.open(path, flags, mode, dir_fd=dir_fd)


class InterfaceDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = DOCUMENT.read_bytes()
        cls.doc = strict_load(cls.raw)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="device-lease-interface-")
        self.addCleanup(self.temp.cleanup)
        self.namespace = self.temp.name + "/leases"

    def acquire(self, **kwargs):
        request = dict(job_id="interface-job", workload="stt-job", device="interface-device",
                       deadline=time.monotonic() + 2, namespace=self.namespace)
        request.update(kwargs)
        return leases.acquire(**request)

    def outcome(self, **kwargs):
        try:
            self.acquire(**kwargs).release(cleanup_complete=True)
            return "granted"
        except leases.LeaseError as error:
            return error.code

    def test_document_is_canonical_closed_and_typed(self):
        canonical = (json.dumps(self.doc, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
        self.assertEqual(self.raw, canonical)
        self.assertEqual(set(self.doc), KEYS)
        self.assertEqual(self.doc["schema"], "kilix.device-lease.interface/v1")
        for key, fields in NESTED.items():
            self.assertIs(type(self.doc[key]), dict, key)
            self.assertEqual(set(self.doc[key]), fields, key)
        for key, field in (("admission", "co_residency"), ("admission", "exclusive"),
                           ("guard_fd", "cloexec"), ("release_semantics", "cleanup_ack_required"),
                           ("release_semantics", "lock_un"),
                           ("release_semantics", "quarantine_on_unproven")):
            self.assertIs(type(self.doc[key][field]), bool, f"{key}.{field}")
        self.assertIs(type(self.doc["guard_fd"]["inherit_via"]), str)
        for key in ("distribution", "label_pattern", "namespace_default", "python_module",
                    "schema", "version"):
            self.assertIs(type(self.doc[key]), str, key)
        for key in ("max_queue", "max_wait_seconds", "max_workload_queue"):
            self.assertIs(type(self.doc[key]), int, key)
        for key in ("workloads", "error_codes"):
            self.assertIs(type(self.doc[key]), list, key)
            self.assertTrue(self.doc[key] and all(type(item) is str for item in self.doc[key]), key)
            self.assertEqual(len(set(self.doc[key])), len(self.doc[key]), key)
        self.assertEqual(self.doc["error_codes"], sorted(self.doc["error_codes"]))

    def test_module_constants_equal_document(self):
        self.assertEqual(constant_differences(leases, self.doc), [])

    def test_constant_comparison_sees_planted_drifts(self):
        drifted_documents = {
            "version": "kilix.device-lease/v2", "workloads": list(reversed(self.doc["workloads"])),
            "max_queue": self.doc["max_queue"] + 1, "max_workload_queue": self.doc["max_workload_queue"] - 1,
            "max_wait_seconds": self.doc["max_wait_seconds"] * 2,
            "label_pattern": self.doc["label_pattern"].replace("{0,95}", "{0,96}"),
            "python_module": "kilix_device_lease_v2",
        }
        for key, value in drifted_documents.items():
            with self.subTest(document=key):
                document = copy.deepcopy(self.doc)
                document[key] = value
                self.assertTrue(constant_differences(leases, document))
        values = {name: getattr(leases, name) for name in
                  ("VERSION", "WORKLOADS", "MAX_QUEUE", "MAX_WORKLOAD_QUEUE", "MAX_WAIT_SECONDS", "_LABEL")}
        drifted_modules = {
            "VERSION": leases.VERSION + " ", "WORKLOADS": list(leases.WORKLOADS), "MAX_QUEUE": 25,
            "MAX_WORKLOAD_QUEUE": 9, "MAX_WAIT_SECONDS": 3600, "_LABEL": re.compile(leases._LABEL.pattern),
        }
        for name, value in drifted_modules.items():
            with self.subTest(module=name):
                module = SimpleNamespace(**{**values, name: value}, __name__=leases.__name__)
                self.assertTrue(constant_differences(module, self.doc))

    def test_distribution_identity_equals_document(self):
        with (COMPONENT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        self.assertEqual(project["name"], self.doc["distribution"])
        self.assertEqual((COMPONENT / "src" / self.doc["python_module"] / "__init__.py").resolve(),
                         Path(leases.__file__).resolve())

    def test_error_codes_equal_document(self):
        codes, problems = error_code_findings(module_source())
        self.assertEqual(problems, [])
        self.assertEqual(codes, set(self.doc["error_codes"]))
        bound = sorted(name for name, value in vars(leases).items() if value is leases.LeaseError)
        self.assertEqual(bound, ["LeaseError"])
        self.assertEqual([name for name, value in vars(leases).items()
                          if isinstance(value, type) and issubclass(value, leases.LeaseError)], ["LeaseError"])

    def test_error_code_scan_sees_planted_routes(self):
        base = module_source()
        anchor = "@dataclass(frozen=True)\nclass QueueStatus:"
        self.assertEqual(base.count(anchor), 1)
        planted = {
            "alias": "_ErrorAlias = LeaseError\n\n\ndef _vanish():\n    raise _ErrorAlias('vanished', 'x')\n",
            "subclass": "class _Vanished(LeaseError):\n    pass\n",
            "variable code": "def _vanish(code='vanished'):\n    raise LeaseError(code, 'x')\n",
            "keyword code": "def _vanish():\n    raise LeaseError(code='vanished', message='x')\n",
            "string lookup": "def _vanish():\n    raise globals()['LeaseError']('vanished', 'x')\n",
            "class attribute": "def _vanish(error):\n    raise error.__class__('vanished', 'x')\n",
        }
        for label, code in planted.items():
            with self.subTest(route=label):
                _codes, problems = error_code_findings(base.replace(anchor, code + "\n\n" + anchor))
                self.assertTrue(problems, label)
        codes, problems = error_code_findings(base.replace(
            anchor, "def _vanish():\n    raise LeaseError('vanished', 'x')\n\n\n" + anchor))
        self.assertEqual(problems, [])
        self.assertNotEqual(codes, set(self.doc["error_codes"]))

    def test_label_checks_decide_exactly_as_the_documented_pattern(self):
        pattern = re.compile(self.doc["label_pattern"], re.ASCII)
        for probe in LABEL_PROBES:
            expected = "granted" if pattern.fullmatch(probe) else "invalid-request"
            for field in ("job_id", "device"):
                with self.subTest(field=field, probe=probe):
                    self.assertEqual(self.outcome(**{field: probe}), expected)

    def test_label_probes_separate_wider_and_narrower_checks(self):
        documented = re.compile(self.doc["label_pattern"], re.ASCII)
        planted = {
            "wider character class": re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/ -]{0,95}\Z").fullmatch,
            "narrower character class": re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z").fullmatch,
            "wider first character": re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,95}\Z").fullmatch,
            "one character longer": re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,96}\Z").fullmatch,
            "unicode word characters": re.compile(r"\w[\w.:-]{0,95}\Z").fullmatch,
            "match with a newline-tolerant end": re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$").match,
        }
        for label, decide in planted.items():
            with self.subTest(check=label):
                self.assertTrue(any(bool(documented.fullmatch(probe)) != bool(decide(probe))
                                    for probe in LABEL_PROBES))

    def test_maximum_wait_is_enforced_where_the_document_says(self):
        limit = self.doc["max_wait_seconds"]
        margin = 30
        self.assertEqual(self.outcome(deadline=time.monotonic() + limit - margin), "granted")
        self.assertEqual(self.outcome(deadline=time.monotonic() + limit + margin), "invalid-request")
        # The two probes straddle only the documented limit: a bound half or
        # twice as long would decide at least one of them differently.
        for other in (limit / 2, limit * 2):
            self.assertNotEqual((limit - margin <= other, limit + margin <= other), (True, False))

    def test_default_namespace_is_created_where_the_document_says(self):
        documented = self.doc["namespace_default"].format(euid=os.geteuid())
        parent, leaf = os.path.split(documented)
        expected = sorted([leaf, f".{leaf}.lease-v1.anchor"])

        def entries_after(namespace):
            root = tempfile.mkdtemp(prefix="root-", dir=self.temp.name)
            os.makedirs(root + parent, mode=0o700)
            with unittest.mock.patch.object(leases, "os", RootedOS(root)):
                lease = leases.acquire(job_id="interface-job", workload="llm-turn",
                                       device="interface-device", deadline=time.monotonic() + 2,
                                       namespace=namespace)
                lease.release(cleanup_complete=True)
            return sorted(os.listdir(root + parent)), root

        observed, root = entries_after(None)
        self.assertEqual(observed, expected)
        self.assertTrue(os.path.isfile(root + documented + "/state.json"))
        # Planted: a namespace one character away lands somewhere else, and the
        # private root refuses to reach any real absolute path.
        shadow, _root = entries_after(documented + "-shadow")
        self.assertNotEqual(shadow, expected)
        with self.assertRaises(AssertionError):
            RootedOS(root).open(parent, os.O_RDONLY | os.O_DIRECTORY)

    def test_release_semantics_equal_document(self):
        semantics = self.doc["release_semantics"]
        self.assertEqual(semantics, {"cleanup_ack_required": True, "lock_un": False,
                                     "quarantine_on_unproven": True})
        parameter = inspect.signature(leases.Lease.release).parameters["cleanup_complete"]
        self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, False)
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == "LOCK_UN"
                             for node in ast.walk(ast.parse(module_source()))))
        lease = self.acquire()
        lease.release()
        with self.assertRaises(leases.LeaseError) as caught:
            self.acquire(workload="tts-utterance")
        self.assertEqual(caught.exception.code, "unavailable")

    def test_guard_descriptor_equals_document(self):
        self.assertEqual(self.doc["guard_fd"], {"cloexec": True, "inherit_via": "pass_fds"})
        lease = self.acquire()
        self.assertFalse(os.get_inheritable(lease.guard_fd))
        child = subprocess.Popen([sys.executable, "-c", "import sys;sys.stdin.read()"],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, pass_fds=(lease.guard_fd,))
        try:
            lease.release(cleanup_complete=True)
            with self.assertRaises(leases.LeaseError) as caught:
                self.acquire(deadline=time.monotonic() + .15)
            self.assertEqual(caught.exception.code, "deadline")
        finally:
            child.communicate(b"", timeout=5)
        self.acquire().release(cleanup_complete=True)

    def test_admission_equals_document(self):
        self.assertEqual(self.doc["admission"], {"co_residency": False, "exclusive": True})
        first = self.acquire(device="accelerator-a", workload="llm-turn")
        try:
            with self.assertRaises(leases.LeaseError) as caught:
                self.acquire(device="accelerator-b", workload="tts-utterance",
                             deadline=time.monotonic() + .15)
            self.assertEqual(caught.exception.code, "deadline")
        finally:
            first.release(cleanup_complete=True)
        self.acquire(device="accelerator-b").release(cleanup_complete=True)


if __name__ == "__main__":
    unittest.main()
