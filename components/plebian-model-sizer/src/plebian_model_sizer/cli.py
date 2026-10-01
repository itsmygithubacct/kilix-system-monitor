"""Local sizing CLI. No network, downloads, training or persistent state."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from .estimate import Workload, recommend
from .chat import recommend_chat
from .resources import collect, voice_data_root
from .voice import observe_cpu, recommend_voice, format_report


def read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    with (nullcontext(sys.stdin.buffer) if str(path) == "-" else path.open("rb")) as handle:
        raw = handle.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("JSON input exceeds 4 MiB")
    result = json.loads(raw, object_pairs_hook=pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid JSON number")))
    if not isinstance(result, dict):
        raise ValueError("JSON input must be an object")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="observe current resources without saving them")
    snapshot.add_argument("--json", action="store_true")
    snapshot.add_argument("--data-root", type=Path, help="filesystem to assess (default: help module user data)")
    recommendations = commands.add_parser("recommend", help="produce a provisional resource shortlist")
    domains = recommendations.add_subparsers(dest="domain", required=True)
    command = domains.add_parser("help-llm", help="document-model training and inference")
    chat = domains.add_parser("avatar-chat", help="installed CPU chat model resource shortlist")
    voice = domains.add_parser("voice", help="speech inference from measured reference profiles")
    for subcommand in (command, chat, voice):
        subcommand.add_argument("--catalog", required=True, type=Path, help="catalog/request JSON file, or - for stdin")
        subcommand.add_argument("--resources", type=Path, help="explicit snapshot for simulation; at most five minutes old")
        subcommand.add_argument("--data-root", type=Path)
        subcommand.add_argument("--json", action="store_true")
    voice.add_argument("--task", choices=["tts", "stt", "both"], default="both")
    command.add_argument("--task", choices=["answer", "rank", "both"], default="both")
    command.add_argument("--phase", choices=["train", "infer", "both"], default="both")
    command.add_argument("--context", type=int, default=2048)
    command.add_argument("--batch", type=int, default=1)
    command.add_argument("--lora-rank", type=int, default=16)
    command.add_argument("--topics", type=int, default=128)
    command.add_argument("--quant", choices=["q4", "q8", "f16", "f32"], default="q4")
    command.add_argument("--train-backend", choices=["auto", "cpu", "cuda"], default="auto")
    command.add_argument("--infer-backend", choices=["auto", "cpu", "cuda"], default="auto")
    command.add_argument("--gpu", type=int)
    command.add_argument("--co-resident", action="store_true")
    command.add_argument("--no-checkpointing", dest="checkpointing", action="store_false")
    command.add_argument("--document-bytes", type=int, default=0)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "snapshot":
            result = collect(args.data_root)
        else:
            if args.resources and args.data_root:
                raise ValueError("--resources and --data-root are mutually exclusive")
            catalog = read_json(args.catalog)
            data_path = args.data_root
            if args.domain == "voice" and data_path is None:
                data_path = voice_data_root()
            resources = read_json(args.resources) if args.resources else collect(data_path)
            source = "provided" if args.resources else "live"
            if args.domain == "voice":
                result = recommend_voice(catalog, resources, task=args.task, source=source,
                                         cpu=observe_cpu() if source == "live" else None)
            elif args.domain == "avatar-chat":
                result = recommend_chat(catalog, resources, source=source)
            else:
                values = {key: getattr(args, key) for key in Workload.__dataclass_fields__}
                result = recommend(catalog, resources, Workload(**values), source=source)
    except (OSError, ValueError, RecursionError) as error:
        # Do not echo local paths or arbitrary input content in diagnostics.
        message = str(error) if isinstance(error, ValueError) and not isinstance(error, UnicodeError) else type(error).__name__
        if args.json:
            print(json.dumps({"schema": "plebian.models.error/v1-development", "error": message}))
        else:
            print(f"plebian-model-sizer: {message}", file=sys.stderr)
        return 2
    if args.json or args.command == "snapshot":
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.domain == "voice":
        print(format_report(result))
    elif args.domain == "avatar-chat":
        for row in result["candidates"]:
            print(f"{row['id']}: {row['verdict']}; RAM {row['required_ram_bytes'] / 1024**3:.2f} GiB; installed {row['installed']}")
        print("Provisional resource candidate: " + (result["provisional_candidate"] or "none"))
        print("Avatar selection still requires an installed, evaluated model.")
    else:
        print("Development resource estimates; task quality and runtime support are unverified.")
        print(f"{'Candidate':24} {'Verdict':18} {'Train RAM/VRAM GiB':22} Infer RAM/VRAM GiB")
        for row in result["candidates"]:
            peaks = []
            for phase in ("train", "infer"):
                check = row["checks"].get(phase, {}).get("resources", {})
                peaks.append(" / ".join(f"{check[k]['required_bytes'] / 1024**3:.2f}" if k in check else "-" for k in ("ram", "vram")))
            print(f"{row['id']:24} {row['verdict']:18} {peaks[0]:22} {peaks[1]}")
            for error in row["errors"]:
                print(f"  {error}")
        print("Provisional candidate: " + (result["provisional_candidate"] or "none"))
        print("Selection awaits task evaluation. Use --json for assumptions, budgets and breakdowns.")
    return 0
