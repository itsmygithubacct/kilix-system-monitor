"""Adopted from the security seat's P1 review controls (review-p1/security/controls/test_sec2_terminal_injection.py).
Only the Required controls are kept; the Observed ones recorded the defects this round fixed.

S2: `kilix-gpu status` writes another process's raw comm to the terminal.

Unmanaged GPU processes are listed with Unmanaged.command, read from
/proc/<pid>/comm. NVML lists every process on the GPU whoever owns it (on the
laptop: root's Xorg, the ollama user's llama-server), and any process can set
its own comm to 15 arbitrary non-NUL bytes with prctl(PR_SET_NAME). The kernel
does not escape /proc/<pid>/comm. print_status prints it unfiltered, so another
local user's GPU process can put escape sequences into the operator's terminal:
here OSC 52 (set the clipboard to "hi"), which kitty-based terminals honour by
default.

Model names and labels are already restricted (C0/C1 refused); this field is not.
"""
from __future__ import annotations

import io
import re

import residency_containment  # noqa: F401
from kilix_gpu_residency import cli
from fakes import MIB, ResidencyCase
from review_common import spawn_named

OSC52 = "\x1b]52;c;aGk=\x07"  # 11 bytes: clipboard <- base64("hi")
CONTROL = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")


class _Injection(ResidencyCase):
    def status_text(self) -> str:
        self.stranger = spawn_named(self, OSC52, 300 * MIB)
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(["--namespace", self.namespace, "--ollama", self.ollama.url, "status"], out, err)
        self.assertEqual((code, err.getvalue()), (0, ""))
        text = out.getvalue()
        # The branch under test: the process is listed in the unmanaged section.
        self.assertIn(f"pid {self.stranger.pid}  ", text)
        return text


class RequiredTerminalInjection(_Injection):
    def test_status_output_has_no_control_characters(self):
        text = self.status_text()
        found = sorted({f"U+{ord(c):04X}" for c in CONTROL.findall(text)})
        line = next(line for line in text.splitlines() if f"pid {self.stranger.pid}" in line)
        self.assertEqual(found, [], f"S2: kilix-gpu status wrote control characters {found} from an unmanaged "
                                    f"process's comm to the terminal: {line!r}")
