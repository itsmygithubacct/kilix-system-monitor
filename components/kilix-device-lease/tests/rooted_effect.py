"""Private child for test_interface_document: run the real lease module with its filesystem root moved.

usage: python rooted_effect.py '<json configuration>'

The configuration names the module file to load, a private root directory, the
namespace to request (null for the default) and a sandbox mode:

- "strict": the module's os is a proxy that exposes only the calls the module
  is known to make. An absolute path is accepted only as the "/" of os.open,
  which is redirected to the private root; every other absolute path, every
  name climbing out with "..", every cwd-relative name and every call outside
  the list is refused. An audit hook, installed after setup and never removed,
  refuses the same paths through any other route (another import of os,
  builtins.open, pathlib, shutil), and refuses symlinks, links and process
  creation. os.mkfifo and os.mknod, which Python does not audit, are replaced
  in os and posix by the same path refusal. The working directory is the
  private root.
- "open-only": the sandbox this test used before: only an absolute os.open is
  redirected, and everything else reaches the real filesystem. It exists only so
  the test can show what a planted call does without the strict sandbox, and it
  refuses to run without an explicit namespace, which the test always points at
  a private canary directory.

It prints one JSON object: the outcome, the entries left beside and inside the
namespace in the private root, and every refused call.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
import types
import uuid

CONFIG = json.loads(sys.argv[1])
ROOT = os.path.realpath(CONFIG["root"])
REFUSED: list[str] = []


class SandboxEscape(BaseException):
    """A call aimed outside the private root; BaseException, so no ordinary handler hides it."""


def refuse(what: str) -> None:
    REFUSED.append(what)
    raise SandboxEscape(what)


def beneath_root(name: str) -> bool:
    normal = os.path.normpath(name)
    return normal == ROOT or normal.startswith(ROOT + "/")


def relative_name_is_safe(name: str) -> bool:
    return not name.startswith("/") and ".." not in name.split("/")


def text(value: object) -> str | None:
    if isinstance(value, int):
        return None
    try:
        return os.fsdecode(os.fspath(value))
    except TypeError:
        return None


class StrictRootedOS:
    """The calls the lease module makes, and nothing else."""

    PASS = ("close", "dup", "fchmod", "fstat", "fsync", "ftruncate", "geteuid", "getpid",
            "pread", "pwrite", "register_at_fork", "stat_result", "O_CLOEXEC", "O_CREAT",
            "O_DIRECTORY", "O_EXCL", "O_NOFOLLOW", "O_RDONLY", "O_RDWR", "O_WRONLY")
    path = types.SimpleNamespace(join=os.path.join, split=os.path.split, normpath=os.path.normpath)

    def __getattr__(self, name):
        if name in self.PASS:
            return getattr(os, name)
        refuse(f"os.{name}")

    @staticmethod
    def _relative(call: str, name, dir_fd) -> None:
        value = text(name)
        if dir_fd is None or value is None or not relative_name_is_safe(value):
            refuse(f"{call}({name!r}, dir_fd={dir_fd!r})")

    def open(self, path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None and path == "/":
            return os.open(ROOT, flags, mode)
        self._relative("os.open", path, dir_fd)
        return os.open(path, flags, mode, dir_fd=dir_fd)

    def mkdir(self, path, mode=0o777, *, dir_fd=None):
        self._relative("os.mkdir", path, dir_fd)
        return os.mkdir(path, mode, dir_fd=dir_fd)

    def stat(self, path, *, dir_fd=None, follow_symlinks=True):
        self._relative("os.stat", path, dir_fd)
        return os.stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    def unlink(self, path, *, dir_fd=None):
        self._relative("os.unlink", path, dir_fd)
        return os.unlink(path, dir_fd=dir_fd)

    def replace(self, src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        self._relative("os.replace", src, src_dir_fd)
        self._relative("os.replace", dst, dst_dir_fd)
        return os.replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    def listdir(self, path):
        if not isinstance(path, int):
            refuse(f"os.listdir({path!r})")
        return os.listdir(path)


class OpenOnlyRootedOS:
    """The earlier sandbox: only an absolute os.open is redirected."""

    def __getattr__(self, name):
        return getattr(os, name)

    def open(self, path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None:
            if path != "/":
                refuse(f"os.open({path!r})")
            return os.open(ROOT, flags, mode)
        return os.open(path, flags, mode, dir_fd=dir_fd)


PATH_EVENTS = {
    "open": ((0, None),), "os.mkdir": ((0, 2),), "os.rename": ((0, 2), (1, 3)),
    "os.remove": ((0, 1),), "os.rmdir": ((0, 1),), "os.chmod": ((0, 2),), "os.chown": ((0, 3),),
    "os.truncate": ((0, None),), "os.utime": ((0, 3),), "os.listdir": ((0, None),),
    "os.scandir": ((0, None),), "os.chdir": ((0, None),), "os.setxattr": ((0, None),),
    "os.removexattr": ((0, None),), "shutil.copyfile": ((0, None), (1, None)),
    "shutil.copytree": ((0, None), (1, None)), "shutil.move": ((0, None), (1, None)),
    "shutil.rmtree": ((0, None),),
}
NEVER = {"os.symlink", "os.link", "os.fork", "os.forkpty", "os.exec", "os.posix_spawn", "os.spawn",
         "os.system", "subprocess.Popen", "pty.spawn", "ctypes.dlopen", "os.chdir"}


def audit(event: str, args: tuple) -> None:
    if event in NEVER:
        refuse(f"{event}")
    for position, dir_fd_position in PATH_EVENTS.get(event, ()):
        if position >= len(args):
            continue
        name = text(args[position])
        if name is None:
            continue
        if name.startswith("/"):
            if not beneath_root(name):
                refuse(f"{event}({name!r})")
        elif not relative_name_is_safe(name):
            refuse(f"{event}({name!r})")
        elif dir_fd_position is not None and (dir_fd_position >= len(args) or args[dir_fd_position] is None):
            refuse(f"{event}({name!r}) relative to the working directory")


def refuse_unaudited() -> None:
    """os.mkfifo and os.mknod raise no audit event, so every route to them is replaced."""
    import posix

    def refusing(call: str):
        def refused(path, *args, dir_fd=None, **kwargs):
            audit(f"os.{call}", (path, 0, dir_fd))
            return getattr(posix, "_rooted_" + call)(path, *args, dir_fd=dir_fd, **kwargs)
        return refused

    PATH_EVENTS.update({"os.mkfifo": ((0, 2),), "os.mknod": ((0, 2),)})
    for call in ("mkfifo", "mknod"):
        setattr(posix, "_rooted_" + call, getattr(posix, call))
        setattr(os, call, refusing(call))
        setattr(posix, call, getattr(os, call))


def entries(path: str) -> list[str] | None:
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return None


def main() -> None:
    mode = CONFIG["mode"]
    namespace = CONFIG["namespace"]
    if mode not in ("strict", "open-only") or (mode == "open-only" and namespace is None):
        raise SystemExit("rooted_effect: the open-only sandbox needs an explicit private namespace")
    spec = importlib.util.spec_from_file_location("kilix_device_lease", CONFIG["module"])
    leases = importlib.util.module_from_spec(spec)
    sys.modules["kilix_device_lease"] = leases
    spec.loader.exec_module(leases)
    # Tickets are drawn before the sandbox starts, because uuid4 may read
    # /dev/urandom, an absolute path outside the private root.
    tickets = [CONFIG["ticket"]] * 4 if CONFIG.get("ticket") else [uuid.uuid4().hex for _ in range(4)]
    leases.uuid = types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex=tickets.pop()))
    documented = namespace or f"/run/user/{os.geteuid()}/kilix-device-leases-v1"
    parent = os.path.dirname(documented)
    os.chdir(ROOT)
    if mode == "strict":
        leases.os = StrictRootedOS()
        refuse_unaudited()
        sys.addaudithook(audit)
    else:
        leases.os = OpenOnlyRootedOS()
    try:
        lease = leases.acquire(job_id="interface-job", workload="llm-turn", device="interface-device",
                               deadline=time.monotonic() + 5, namespace=namespace)
        lease.release(cleanup_complete=True)
        outcome = "granted"
    except leases.LeaseError as error:
        outcome = error.code
    except SandboxEscape:
        outcome = "refused"
    report = {"outcome": outcome, "refused": REFUSED,
              "beside": entries(ROOT + parent), "inside": entries(ROOT + documented)}
    # The root can hold the default leaf, which the parent's path guard refuses
    # to remove, so it is removed here, still inside the sandbox.
    shutil.rmtree(ROOT)
    report["refused"] = REFUSED
    sys.stdout.write(json.dumps(report) + "\n")
    sys.stdout.flush()


main()
