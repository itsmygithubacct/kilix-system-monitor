"""The module and its kilix.device-lease/v1 interface document cannot drift apart."""
from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tomllib
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


def module_tree() -> ast.AST:
    return ast.parse(Path(leases.__file__).read_text(encoding="utf-8"))


class Captured(Exception):
    pass


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
        self.assertEqual(leases.VERSION, self.doc["version"])
        self.assertEqual(leases.WORKLOADS, tuple(self.doc["workloads"]))
        self.assertEqual(leases.MAX_QUEUE, self.doc["max_queue"])
        self.assertEqual(leases.MAX_WORKLOAD_QUEUE, self.doc["max_workload_queue"])
        self.assertEqual(leases.MAX_WAIT_SECONDS, float(self.doc["max_wait_seconds"]))
        self.assertEqual(leases._LABEL.pattern, self.doc["label_pattern"])
        self.assertEqual(leases.__name__, self.doc["python_module"])

    def test_distribution_identity_equals_document(self):
        with (COMPONENT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        self.assertEqual(project["name"], self.doc["distribution"])
        self.assertEqual((COMPONENT / "src" / self.doc["python_module"] / "__init__.py").resolve(),
                         Path(leases.__file__).resolve())

    def test_error_codes_equal_document(self):
        codes = set()
        for node in ast.walk(module_tree()):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LeaseError":
                self.assertTrue(node.args and isinstance(node.args[0], ast.Constant)
                                and type(node.args[0].value) is str,
                                f"LeaseError at line {node.lineno} has no literal code")
                codes.add(node.args[0].value)
        self.assertEqual(codes, set(self.doc["error_codes"]))

    def test_default_namespace_equals_document(self):
        seen = []

        def registry(path, *_args, **_kwargs):
            seen.append(path)
            raise Captured()

        with mock.patch.object(leases, "_Registry", registry):
            # Never reach the real per-user runtime directory from a test.
            self.assertIs(leases._Registry, registry)
            with self.assertRaises(Captured):
                leases.acquire(job_id="interface-job", workload="llm-turn", device="interface-device",
                               deadline=time.monotonic() + 1)
        self.assertEqual(seen, [self.doc["namespace_default"].format(euid=os.geteuid())])

    def test_release_semantics_equal_document(self):
        semantics = self.doc["release_semantics"]
        self.assertEqual(semantics, {"cleanup_ack_required": True, "lock_un": False,
                                     "quarantine_on_unproven": True})
        parameter = inspect.signature(leases.Lease.release).parameters["cleanup_complete"]
        self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, False)
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == "LOCK_UN"
                             for node in ast.walk(module_tree())))
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
