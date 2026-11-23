"""No residency test process, or child it starts, can reach the real machine.

Planted calls aim at private stand-ins (a directory named like the default
namespace leaf, a program named nvidia-smi, a process outside the suite),
never at the real thing: with the guard each must be refused; without it the
same calls must succeed, so the guarded arm had something to refuse.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import residency_containment
from residency_containment import guard

TESTS = Path(__file__).resolve().parent

PLANTED = r"""
import json, os, pathlib, signal, socket, subprocess, sys
target, smi, stranger, models = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
results = {"guarded": bool(getattr(sys, "_kilix_residency_guard", False))}
def attempt(name, call):
    try:
        call()
        results[name] = "done"
    except BaseException as error:
        results[name] = type(error).__name__
attempt("makedirs", lambda: os.makedirs(target + "/made/deeper"))
attempt("rename", lambda: os.rename(target + "/rename-me", target + "/renamed"))
attempt("open", lambda: open(target + "/opened", "w").close())
attempt("pathlib", lambda: pathlib.Path(target, "touched").touch())
attempt("mkfifo", lambda: os.mkfifo(target + "/fifo"))
attempt("model-folder", lambda: os.makedirs(models + "/kilix-llm/models"))
attempt("model-file", lambda: open(models + "/puzzllm/diffusion.gguf", "wb").close())
attempt("nvidia-smi", lambda: subprocess.run([smi], check=True))
def dlopen():
    import ctypes
    try:
        ctypes.CDLL("libnvidia-ml.so.1")
    except OSError:
        pass  # absent on this host: the call itself was allowed
attempt("nvml", dlopen)
def connect(address):
    def call():
        with socket.socket() as s:
            s.settimeout(0.2)
            try:
                s.connect(address)
            except OSError:
                pass  # nothing listening: the connect itself was allowed
    return call
attempt("ollama-port", connect(("127.0.0.255", 11434)))  # nothing listens there; refused by port
attempt("remote", connect(("192.0.2.1", 9)))
attempt("signal-stranger", lambda: os.kill(stranger, signal.SIGCONT))
print(json.dumps(results))
"""
FIXTURE = r"""
import os, sys
target = sys.argv[1]
os.mkdir(target, 0o700)
open(os.path.join(target, "rename-me"), "wb").write(b"canary")
os.makedirs(os.path.join(sys.argv[2], "puzzllm"))
"""
CALLS = ("makedirs", "rename", "open", "pathlib", "mkfifo", "model-folder", "model-file", "nvidia-smi", "nvml",
         "ollama-port", "remote", "signal-stranger")


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gpu-residency-guard-")
        self.addCleanup(self.temp.cleanup)
        self.target = self.temp.name + "/" + guard.LEAF
        self.addCleanup(self.child, "import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)",
                        residency_containment.env_without_path_guard())
        self.smi = Path(self.temp.name) / "nvidia-smi"
        self.smi.write_text("#!/bin/sh\nexit 0\n")
        self.smi.chmod(0o700)
        # A stand-in home whose model folders the guarded child is told are the real ones.
        self.models = os.path.join(self.temp.name, "home")
        # A process that is not a descendant of the suite: its parent exits at once.
        self.stranger = int(subprocess.run(
            ["sh", "-c", "sleep 30 </dev/null >/dev/null 2>&1 & echo $!"], capture_output=True, text=True,
            timeout=10, env=residency_containment.env_without_path_guard()).stdout)
        self.addCleanup(self.child, f"import os, signal; os.kill({self.stranger}, signal.SIGKILL)",
                        residency_containment.env_without_path_guard(**{guard.ROOT_ENV: "0"}))

    def child(self, code, env, *args):
        done = subprocess.run([sys.executable, "-c", code, self.target, *args], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=30, env=env)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout

    def snapshot(self):
        code = ("import json, os, sys; print(json.dumps(sorted("
                "os.path.relpath(os.path.join(d, n), t) for t in sys.argv[1:] for d, ds, fs in os.walk(t) "
                "for n in ds + fs)))")
        return json.loads(self.child(code, residency_containment.env_without_path_guard(), self.models))

    def planted(self, env):
        return json.loads(self.child(PLANTED, env, str(self.smi), str(self.stranger), self.models))

    def test_children_started_by_the_suite_refuse_every_planted_call(self):
        self.child(FIXTURE, residency_containment.env_without_path_guard(), self.models)
        before = self.snapshot()
        folders = os.pathsep.join((self.models + "/kilix-llm", self.models + "/puzzllm"))
        self.assertEqual(self.planted(residency_containment.child_env(**{guard.MODELS_ENV: folders})),
                         {"guarded": True, **{call: "RealMachineRefused" for call in CALLS}})
        self.assertEqual(self.snapshot(), before)

    def test_the_suite_recorded_the_real_model_folders_before_clearing_their_variables(self):
        import pwd
        home = pwd.getpwuid(os.getuid()).pw_dir
        for path in (home + "/.local/gpu_terminal/kilix-llm", home + "/.local/share/puzzllm"):
            with self.subTest(folder=path):
                self.assertIn(path, guard.MODEL_FOLDERS)
                self.assertTrue(guard.refuses(path + "/models/x.gguf"))
                with self.assertRaises(guard.RealMachineRefused):
                    open(path + "/models/never-made.gguf", "rb")
        self.assertEqual(os.environ[guard.MODELS_ENV].split(os.pathsep), list(guard.MODEL_FOLDERS))
        for name in ("KILIX_LLM_MODELS_DIR", "PUZZLLM_DIFFUSION_MODELS", "GPU_TERMINAL_HOME", "XDG_DATA_HOME"):
            self.assertNotIn(name, os.environ)

    def test_children_without_the_guard_would_have_made_every_call(self):
        self.child(FIXTURE, residency_containment.env_without_path_guard(), self.models)
        before = self.snapshot()
        self.assertEqual(self.planted(residency_containment.env_without_path_guard()),
                         {"guarded": False, **{call: "done" for call in CALLS}})
        self.assertNotEqual(self.snapshot(), before)

    def test_this_process_refuses_the_leaf_and_the_runtime_directory(self):
        with self.assertRaises(guard.RealMachineRefused):
            Path(self.target).mkdir()
        self.assertEqual(guard.refused[-1], ("os.mkdir", self.target))
        for name in ("/run/user", "/run/user/1000", "/run/../run/user/1000/x", "kilix-gpu-residency-v1",
                     "/private/kilix-gpu-residency-v1-shadow"):
            with self.subTest(refused=name):
                self.assertTrue(guard.refuses(name))
        for name in ("/run/users", "/run", "/private/registry", "run/user/1000", "registry.lock"):
            with self.subTest(allowed=name):
                self.assertFalse(guard.refuses(name))
        for address, refused in ((("127.0.0.1", 11434), True), (("10.0.0.1", 80), True), (("::1", 5), False),
                                 (("127.0.0.1", 8080), False), ("/tmp/socket", False), (("localhost", 1), False)):
            with self.subTest(address=address):
                self.assertEqual(guard.refuses_address(address), refused)

    def test_signals_reach_only_the_suites_own_descendants(self):
        mine = subprocess.Popen(["sleep", "30"])
        self.addCleanup(mine.wait)
        os.kill(mine.pid, 15)
        with self.assertRaises(guard.RealMachineRefused):
            os.kill(self.stranger, 18)
        with self.assertRaises(guard.RealMachineRefused):
            os.killpg(os.getpgid(self.stranger), 18)
        os.kill(self.stranger, 0)  # existence checks are allowed

    def test_every_test_module_installs_the_guard_and_no_child_bypasses_it(self):
        variable = "PYTHON" + "PATH"  # spelt apart so this file does not match itself
        modules = sorted(TESTS.glob("test_*.py"))
        self.assertIn(Path(__file__).resolve(), modules)
        for path in modules:
            with self.subTest(module=path.name):
                text = path.read_text(encoding="utf-8")
                tree = ast.parse(text)
                imported = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                            for alias in node.names}
                self.assertIn("residency_containment", imported)
                self.assertNotIn(variable, text)
                users = {node.attr for node in ast.walk(tree)
                         if isinstance(node, ast.Attribute) and node.attr == "env_without_path_guard"}
                if path.name != "test_path_guard.py":
                    self.assertEqual(users, set())


if __name__ == "__main__":
    unittest.main()
