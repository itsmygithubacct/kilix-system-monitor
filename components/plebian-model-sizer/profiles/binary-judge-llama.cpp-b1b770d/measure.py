#!/usr/bin/env python3
"""Measure llama-server's peak memory for the one-token judge at several context sizes.

    measure.py SERVER MODEL.gguf OUT.json [CTX ...]

For each context it starts SERVER with the flags `kilix-llm judge` uses (one
slot, flash attention, f16 KV, no prompt cache, CPU only), sends one
completion whose prompt fills the context (ctx - 8 tokens, one token out) and
records the process's peak resident memory (VmHWM). No adapter is loaded; the
profile adds the stand-in adapter's bytes to each peak.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

GRAMMAR = 'root ::= "0" | "1"'
FILLER = "The build finished and every test passed without a warning. "


class Unix(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=None)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.path)


def call(sock, method, path, body=None):
    connection = Unix(sock)
    try:
        connection.request(method, path, None if body is None else json.dumps(body),
                           {"Content-Type": "application/json"})
        reply = connection.getresponse()
        data = reply.read()
    finally:
        connection.close()
    if reply.status != 200:
        raise RuntimeError(f"{path}: HTTP {reply.status} {data[:200]!r}")
    return json.loads(data)


def status(pid, key):
    with open(f"/proc/{pid}/status") as handle:
        for line in handle:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(key)


def measure(server, model, ctx, threads):
    with tempfile.TemporaryDirectory(prefix="judge-profile-") as directory:
        sock = os.path.join(directory, "engine.sock")
        argv = [server, "-m", model, "-c", str(ctx), "-t", str(threads), "--parallel", "1", "-fa", "on",
                "--fit", "off", "--cache-ram", "0", "--no-warmup", "--no-webui", "--host", sock, "--jinja",
                "--reasoning", "off", "-ngl", "0", "--device", "none"]
        with open(os.path.join(directory, "log"), "wb") as log:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited {process.returncode} at ctx {ctx}")
                try:
                    if os.path.exists(sock) and call(sock, "GET", "/health").get("status") == "ok":
                        break
                except (OSError, RuntimeError, http.client.HTTPException):
                    pass
                time.sleep(0.2)
            want = ctx - 8
            text = FILLER * (want // 8 + 16)
            tokens = call(sock, "POST", "/tokenize", {"content": text, "add_special": True})["tokens"][:want]
            if len(tokens) != want:
                raise RuntimeError("the filler is too short")
            started = time.monotonic()
            reply = call(sock, "POST", "/completion", {
                "prompt": tokens, "n_predict": 1, "temperature": 0, "grammar": GRAMMAR,
                "n_probs": 20, "cache_prompt": False})
            if reply.get("tokens_evaluated") != want:
                raise RuntimeError(f"evaluated {reply.get('tokens_evaluated')} of {want} tokens")
            return {"ctx": ctx, "prompt_tokens": want, "peak_bytes": status(process.pid, "VmHWM"),
                    "seconds": round(time.monotonic() - started, 1)}
        finally:
            process.terminate()
            process.wait()


def main():
    server, model, out, *contexts = sys.argv[1:]
    contexts = [int(c) for c in contexts] or [512, 2048, 8192, 32768, 131072, 262144]
    threads = len({line.split(":")[1].strip() for line in open("/proc/cpuinfo") if line.startswith("core id")}) or os.cpu_count()
    rows = []
    for ctx in contexts:
        rows.append(measure(server, model, ctx, threads))
        print(json.dumps(rows[-1]), flush=True)
        with open(out, "w") as handle:
            json.dump({"server": server, "model": model, "threads": threads, "points": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
