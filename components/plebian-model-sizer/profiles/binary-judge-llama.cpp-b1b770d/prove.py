#!/usr/bin/env python3
"""One real `kilix-llm judge` at this machine's limit, with the engine's peak memory recorded.

    prove.py KILIX_LLM SIZER OUT.json

The input is longer than the limit and judge keeps its head (--truncate head),
so the engine renders, tokenizes and answers a prompt that fills the context.
"""
import json
import os
import subprocess
import sys
import threading
import time

FILLER = "The build finished and every test passed without a warning. "


def engine_peak(judge, found):
    """Poll the judge's llama-server child for VmHWM until judge exits."""
    while judge.poll() is None:
        try:
            children = subprocess.run(["pgrep", "-P", str(judge.pid), "-f", "llama-server"],
                                      capture_output=True, text=True).stdout.split()
            for pid in children:
                with open(f"/proc/{pid}/status") as handle:
                    for line in handle:
                        if line.startswith("VmHWM:"):
                            found["peak_bytes"] = max(found.get("peak_bytes", 0), int(line.split()[1]) * 1024)
        except OSError:
            pass
        time.sleep(1)


def main():
    kilix_llm, sizer, out = sys.argv[1:]
    env = dict(os.environ, KILIX_LLM_SIZER=sizer, KILIX_LLM_NO_AUTO_BUILD="1")
    text = "<information>\n" + FILLER * 60000 + "\n</information>\n\nDid the tests pass?"
    started = time.monotonic()
    judge = subprocess.Popen([sys.executable, "-m", "kilix_llm", "judge", "--model-id", "qwen3.5-0.8b", "--stdin",
                              "--json", "--truncate", "head"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=dict(env, PYTHONPATH=os.path.join(kilix_llm, "src")),
                             cwd=kilix_llm)
    found = {}
    watcher = threading.Thread(target=engine_peak, args=(judge, found))
    watcher.start()
    stdout, stderr = judge.communicate(text)
    watcher.join()
    if judge.returncode not in (0, 1):
        raise SystemExit(f"judge exited {judge.returncode}: {stderr[-600:]}")
    details = json.loads(stdout)
    result = {"ctx": details["max_input_tokens"] + 178 + 1, "input_tokens": details["input_tokens"],
              "max_input_tokens": details["max_input_tokens"], "limit_reason": details["limit_reason"],
              "answer": details["answer"], "peak_bytes": found["peak_bytes"],
              "seconds": round(time.monotonic() - started, 1)}
    with open(out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
