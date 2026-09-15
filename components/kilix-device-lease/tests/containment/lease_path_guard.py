"""Refuse any filesystem operation a lease test could aim at the real runtime directory.

Every namespace the lease tests use lives beneath a private temporary
directory. This process-wide audit hook refuses, before it runs, any audited
filesystem operation whose path names the default namespace leaf or lies under
/run/user. The lease module walks to its namespace with descriptor-relative
names, so the leaf itself is refused as well as absolute paths: a regression
that ignored its namespace argument would be stopped at the anchor, before it
created anything. Refusals raise RealRuntimePathRefused, a BaseException, so an
``except Exception`` in the code under test cannot swallow one, and each is
also recorded in ``refused``.

Python does not audit os.mkfifo or os.mknod, so both are wrapped in os and
posix with the same refusal. Read-only calls Python does not audit (for example
os.stat) and calls through ctypes are not covered; the suite's own runner is
responsible for running it where the real runtime directory is unreachable.
"""
from __future__ import annotations

import os
import sys

LEAF = "kilix-device-leases-v1"
RUNTIME = "/run/user"
# Audit event name -> positions of its path arguments.
EVENTS = {
    "open": (0,), "os.mkdir": (0,), "os.rename": (0, 1), "os.remove": (0,), "os.rmdir": (0,),
    "os.link": (0, 1), "os.symlink": (0, 1), "os.chmod": (0,), "os.chown": (0,),
    "os.truncate": (0,), "os.utime": (0,), "os.listdir": (0,), "os.scandir": (0,),
    "os.chdir": (0,), "os.setxattr": (0,), "os.removexattr": (0,),
    "shutil.copyfile": (0, 1), "shutil.copymode": (0, 1), "shutil.copystat": (0, 1),
    "shutil.copytree": (0, 1), "shutil.move": (0, 1), "shutil.rmtree": (0,), "shutil.chown": (0,),
}
refused: list[tuple[str, str]] = []


class RealRuntimePathRefused(BaseException):
    """A lease test tried to reach the real runtime directory."""


def _name(value: object) -> str | None:
    if isinstance(value, int):
        return None
    try:
        value = os.fspath(value)
    except TypeError:
        return None
    return os.fsdecode(value)


def refuses(name: str) -> bool:
    if LEAF in name:
        return True
    if not name.startswith("/"):
        return False
    normal = os.path.normpath(name)
    return normal == RUNTIME or normal.startswith(RUNTIME + "/")


def _hook(event: str, args: tuple) -> None:
    positions = EVENTS.get(event)
    if positions is None:
        return
    for position in positions:
        if position >= len(args):
            continue
        name = _name(args[position])
        if name is not None and refuses(name):
            refused.append((event, name))
            raise RealRuntimePathRefused(f"lease test refused {event} on {name!r}")


def _refusing(call, event: str):
    def refusing(path, *args, **kwargs):
        name = _name(path)
        if name is not None and refuses(name):
            refused.append((event, name))
            raise RealRuntimePathRefused(f"lease test refused {event} on {name!r}")
        return call(path, *args, **kwargs)
    return refusing


if not getattr(sys, "_kilix_lease_path_guard", False):
    sys._kilix_lease_path_guard = True
    sys.addaudithook(_hook)
    import posix
    for _call in ("mkfifo", "mknod"):
        _refusal = _refusing(getattr(posix, _call), "os." + _call)
        setattr(os, _call, _refusal)
        setattr(posix, _call, _refusal)
