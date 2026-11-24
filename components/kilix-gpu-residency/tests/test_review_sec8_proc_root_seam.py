"""Adopted from the security seat's P1 review controls (review-p1/security/controls/test_sec8_proc_root_seam.py).
Only the Required controls are kept; the Observed ones recorded the defects this round fixed.

S8: the PID-reuse guard reads whatever /proc the environment names.

vram.proc_root() honours KILIX_GPU_PROC_ROOT, a test seam that the shipped module
keeps and the README does not mention. The registry's liveness and start-time
checks (_stat, _alive, _groups) read through it, while os.killpg acts on the
real kernel. So in any process whose environment carries the variable, a record
whose leader has died and whose PID now belongs to an unrelated process is not
pruned, and `unload` / `ensure_room` TERM the unrelated group.

The PID reuse is simulated exactly as the component's own
test_a_leader_whose_start_time_changed_is_never_signalled does: the record's
start time is moved by one tick, so it no longer matches the live process.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal

import residency_containment  # noqa: F401
import kilix_gpu_residency as residency
from fakes import GIB, ResidencyCase


class _Seam(ResidencyCase):
    def reused(self):
        """A record whose leader 'died' and whose PID now belongs to ``process``."""
        resident, process = self.model("puzzllm", "m", GIB)
        path = Path(self.namespace) / (resident.id + ".json")
        record = json.loads(path.read_text())
        record["holder"]["start"] += 1  # the registered leader is gone; the PID is someone else's
        path.write_text(json.dumps(record))
        os.chmod(path, 0o600)
        return resident, process, record

    def shadow_proc(self, record) -> str:
        """A directory that says the reused PID still has the registered start time."""
        pid = record["holder"]["pid"]
        real = Path(f"/proc/{pid}/stat").read_text()
        head, tail = real[:real.rfind(")") + 2], real[real.rfind(")") + 2:].split()
        tail[19] = str(record["holder"]["start"])
        shadow = self.scratch / "shadow-proc" / str(pid)
        shadow.mkdir(parents=True)
        (shadow / "stat").write_text(head + " ".join(tail) + "\n")
        return str(self.scratch / "shadow-proc")


class RequiredProcRootSeam(_Seam):
    def test_signal_safety_does_not_depend_on_the_environment(self):
        resident, process, record = self.reused()
        os.environ["KILIX_GPU_PROC_ROOT"] = self.shadow_proc(record)
        try:
            residency.unload(resident.id, **self.where())
        except residency.ResidencyError:
            pass
        self.assertIsNone(process.poll(),
                          f"S8: with KILIX_GPU_PROC_ROOT set, unload() signalled process group {process.pid} whose "
                          "real start time does not match the record (the PID-reuse guard read the variable's tree "
                          f"while os.killpg acted on the real kernel): exit {process.returncode}")
