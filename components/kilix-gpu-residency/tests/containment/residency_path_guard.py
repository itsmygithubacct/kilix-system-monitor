"""Refuse anything a residency test could aim at the real machine.

Every namespace the residency tests use lives beneath a private temporary
directory, every GPU is a fake and every Ollama a local fake server. This
process-wide audit hook refuses, before it runs:

* any audited filesystem operation whose path names the default namespace
  leaf or lies under /run/user (the module walks to its namespace with
  descriptor-relative names, so the leaf itself is refused as well);
* any audited filesystem operation on the real model folders: the defaults
  under the account's home directory (from the password database, not $HOME)
  and whatever the model-folder variables named when the suite started. The
  catalog opens a folder by its absolute path before it works relative to it,
  so that first open is where a stray test is stopped;
* loading the real NVML library, and starting a program named nvidia-smi;
* connecting anywhere but loopback, or to Ollama's default port 11434;
* sending a signal to a process that is not a descendant of the suite's root
  process (signal 0, which only checks existence, is allowed).

Refusals raise RealMachineRefused, a BaseException, so an ``except Exception``
in the code under test cannot swallow one, and each is also recorded in
``refused``. Python does not audit os.mkfifo or os.mknod, so both are wrapped
with the same path refusal. Calls through ctypes are not covered; the suite's
runner is responsible for running it where the real runtime directory is
unreachable.
"""
from __future__ import annotations

import ipaddress
import os
import sys

LEAF = "kilix-gpu-residency-v1"
RUNTIME = "/run/user"
ROOT_ENV = "KILIX_RESIDENCY_TEST_ROOT"
MODELS_ENV = "KILIX_RESIDENCY_REAL_MODEL_FOLDERS"
MODEL_FOLDERS: tuple[str, ...] = ()
EVENTS = {
    "open": (0,), "os.mkdir": (0,), "os.rename": (0, 1), "os.remove": (0,), "os.rmdir": (0,),
    "os.link": (0, 1), "os.symlink": (0, 1), "os.chmod": (0,), "os.chown": (0,),
    "os.truncate": (0,), "os.utime": (0,), "os.listdir": (0,), "os.scandir": (0,),
    "os.chdir": (0,), "os.setxattr": (0,), "os.removexattr": (0,),
    "shutil.copyfile": (0, 1), "shutil.copymode": (0, 1), "shutil.copystat": (0, 1),
    "shutil.copytree": (0, 1), "shutil.move": (0, 1), "shutil.rmtree": (0,), "shutil.chown": (0,),
}
refused: list[tuple[str, str]] = []


class RealMachineRefused(BaseException):
    """A residency test tried to reach the real machine."""


def _name(value: object) -> str | None:
    if isinstance(value, int):
        return None
    try:
        value = os.fspath(value)
    except TypeError:
        return None
    return os.fsdecode(value)


def real_model_folders() -> list[str]:
    """The model folders of the account running the suite, as it started."""
    import pwd
    env = os.environ
    homes = {pwd.getpwuid(os.getuid()).pw_dir, env.get("HOME") or ""}
    found = set()
    for home in filter(None, homes):
        found.add(os.path.join(home, ".local", "gpu_terminal", "kilix-llm"))
        found.add(os.path.join(home, ".local", "share", "puzzllm"))
    for name, leaf in (("KILIX_LLM_MODELS_DIR", ""), ("PUZZLLM_DIFFUSION_MODELS", ""),
                       ("GPU_TERMINAL_HOME", "kilix-llm"), ("XDG_DATA_HOME", "puzzllm")):
        value = env.get(name)
        if value and value.startswith("/"):
            found.add(os.path.join(value, leaf) if leaf else value)
    return sorted(os.path.normpath(path) for path in found)


def refuses(name: str) -> bool:
    if LEAF in name:
        return True
    if not name.startswith("/"):
        return False
    normal = os.path.normpath(name)
    if any(normal == folder or normal.startswith(folder + "/") for folder in MODEL_FOLDERS):
        return True
    return normal == RUNTIME or normal.startswith(RUNTIME + "/")


def refuses_address(address: object) -> bool:
    if not isinstance(address, tuple) or len(address) < 2:
        return False  # a Unix socket path
    host, port = address[0], address[1]
    if port == 11434:
        return True
    try:
        return not (host == "localhost" or ipaddress.ip_address(host).is_loopback)
    except ValueError:
        return True


def _parent(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as stream:
            text = stream.read(4096).decode("ascii", "replace")
        return int(text[text.rfind(")") + 2:].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def descends(pid: int, root: int) -> bool:
    seen = 0
    while pid > 1 and seen < 4096:
        if pid == root:
            return True
        parent = _parent(pid)
        if parent is None:
            return False
        pid, seen = parent, seen + 1
    return False


def _group_descends(pgid: int, root: int) -> bool:
    if descends(pgid, root):
        return True
    # The leader may be gone; every live member must then be ours.
    members = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat", "rb") as stream:
                    text = stream.read(4096).decode("ascii", "replace")
                if int(text[text.rfind(")") + 2:].split()[2]) == pgid:
                    members.append(int(name))
            except (OSError, ValueError, IndexError):
                continue
    return bool(members) and all(descends(member, root) for member in members)


def _refuse(event: str, detail: str) -> None:
    refused.append((event, detail))
    raise RealMachineRefused(f"residency test refused {event} on {detail!r}")


def _hook(event: str, args: tuple) -> None:
    positions = EVENTS.get(event)
    if positions is not None:
        for position in positions:
            if position < len(args):
                name = _name(args[position])
                if name is not None and refuses(name):
                    _refuse(event, name)
        return
    if event == "ctypes.dlopen":
        name = _name(args[0]) if args else None
        if name and "nvidia-ml" in name:
            _refuse(event, name)
    elif event == "subprocess.Popen":
        executable, argv = args[0], args[1]
        names = [_name(executable)] if executable is not None else []
        if isinstance(argv, (list, tuple)) and argv:
            names.append(_name(argv[0]))
        elif argv is not None:
            names.append(_name(argv))
        for name in names:
            if name and os.path.basename(name) == "nvidia-smi":
                _refuse(event, name)
    elif event == "socket.connect":
        if len(args) > 1 and refuses_address(args[1]):
            _refuse(event, repr(args[1]))
    elif event in ("os.kill", "os.killpg"):
        target, sig = args[0], args[1]
        if sig == 0:
            return
        root = int(os.environ.get(ROOT_ENV, "0") or 0)
        ours = descends(target, root) if event == "os.kill" else _group_descends(target, root)
        if not root or not ours:
            _refuse(event, str(target))


def _refusing(call, event: str):
    def refusing(path, *args, **kwargs):
        name = _name(path)
        if name is not None and refuses(name):
            _refuse(event, name)
        return call(path, *args, **kwargs)
    return refusing


if not getattr(sys, "_kilix_residency_guard", False):
    sys._kilix_residency_guard = True
    os.environ.setdefault(ROOT_ENV, str(os.getpid()))
    os.environ.setdefault(MODELS_ENV, os.pathsep.join(real_model_folders()))
    MODEL_FOLDERS = tuple(path for path in os.environ[MODELS_ENV].split(os.pathsep) if path.startswith("/"))
    sys.addaudithook(_hook)
    import posix
    for _call in ("mkfifo", "mknod"):
        _refusal = _refusing(getattr(posix, _call), "os." + _call)
        setattr(os, _call, _refusal)
        setattr(posix, _call, _refusal)
