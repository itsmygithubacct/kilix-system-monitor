"""Keeps this suite, and every process it starts, away from the real runtime directory.

Importing this module installs containment/lease_path_guard.py in the test
process and puts its directory on PYTHONPATH, so every Python child started
from here installs it too (containment/sitecustomize.py). Test modules build
child environments with child_env(), never by setting PYTHONPATH themselves,
so no child can be started without the guard.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

TESTS = Path(__file__).resolve().parent
SOURCE = TESTS.parent / "src"
GUARD = TESTS / "containment"

if str(GUARD) not in sys.path:
    sys.path.insert(0, str(GUARD))
import lease_path_guard as guard  # noqa: E402

os.environ["PYTHONPATH"] = os.pathsep.join((str(SOURCE), str(GUARD)))


def child_env(**extra: str) -> dict[str, str]:
    """The environment for a child that imports the lease module: its source, and the guard."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(SOURCE), str(GUARD))))
    env.update(extra)
    return env


def env_without_path_guard(**extra: str) -> dict[str, str]:
    """Only for rooted_effect.py, whose own sandbox is stricter, and for the guard's own controls."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(extra)
    return env
