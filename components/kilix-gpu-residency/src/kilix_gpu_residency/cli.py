"""kilix-gpu: show and manage the models resident on this machine's GPUs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

from . import (ResidencyError, RoomError, Status, VERSION, ensure_room, pin, status, unload)

STATUS_SCHEMA = "kilix.gpu-residency.status/v1"
ROOM_SCHEMA = "kilix.gpu-residency.room/v1"
_UNITS = {"": 1, "B": 1, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12,
          "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40, "KB": 10**3, "MB": 10**6, "GB": 10**9}
_BYTES = re.compile(r"(\d+(?:\.\d+)?)\s*([A-Za-z]*)\Z")


def parse_bytes(text: str) -> int:
    """``5200000000``, ``5.2G`` (powers of ten) or ``4.8GiB`` (powers of two)."""
    match = _BYTES.fullmatch(text.strip())
    if match is None or match.group(2).upper() not in _UNITS:
        raise argparse.ArgumentTypeError(f"not a byte count: {text!r}")
    value = int(float(match.group(1)) * _UNITS[match.group(2).upper()])
    if value <= 0:
        raise argparse.ArgumentTypeError("the byte count must be positive")
    return value


def size(value: int | None) -> str:
    if value is None:
        return "?"
    return f"{value / 2**30:.1f} GiB" if value >= 2**30 else f"{value / 2**20:.0f} MiB"


def _idle(last_used: float | None, now: float) -> str:
    if last_used is None:
        return "?"
    seconds = max(0, int(now - last_used))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def status_json(value: Status) -> dict:
    return {
        "schema": STATUS_SCHEMA,
        "version": VERSION,
        "ollama": value.ollama,
        "devices": [{"pci": d.pci, "name": d.name, "vendor": d.vendor, "driver": d.driver,
                     "discrete": d.discrete, "asleep": d.asleep, "source": d.source,
                     "total_bytes": d.total_bytes, "used_bytes": d.used_bytes, "free_bytes": d.free_bytes}
                    for d in value.devices],
        "residents": [{"id": r.id, "app": r.app, "model": r.model, "backend": r.backend,
                       "device_label": r.device_label, "device_pci": r.device_pci, "pid": r.pid, "pgid": r.pgid,
                       "vram_bytes": r.vram_bytes, "need_bytes": r.need_bytes, "state": r.state,
                       "pinned": r.pinned, "last_used": r.last_used, "registered": r.registered,
                       "unload": r.unload_kind}
                      for r in value.residents],
        "unmanaged": [{"pid": u.pid, "command": u.command, "device_pci": u.device_pci, "vram_bytes": u.vram_bytes}
                      for u in value.unmanaged],
        "events": [event.__dict__ for event in value.events],
    }


def print_status(value: Status, out) -> None:
    now = time.time()
    print("Devices", file=out)
    for d in value.devices:
        if d.asleep:
            memory = "asleep (not woken to read)"
        elif d.source == "held":
            memory = "not read: another NVIDIA GPU is asleep"
        elif d.total_bytes is None:
            memory = "no VRAM counter"
        else:
            memory = f"{size(d.used_bytes)} used / {size(d.total_bytes)}, {size(d.free_bytes)} free"
        print(f"  {d.pci}  {d.name}: {memory}", file=out)
    print("Residents", file=out)
    if not value.residents:
        print("  none", file=out)
    for r in value.residents:
        flags = " pinned" if r.pinned else ""
        flags += "" if r.registered else " (found in Ollama, not registered)"
        print(f"  {r.id}  {r.app}  {r.model}  [{r.backend}] {r.state}  {size(r.vram_bytes)}"
              f"  idle {_idle(r.last_used, now) if r.state == 'idle' else '-'}{flags}", file=out)
    print("Unmanaged GPU processes (never unloaded)", file=out)
    if not value.unmanaged:
        print("  none", file=out)
    for u in value.unmanaged:
        print(f"  pid {u.pid}  {u.command or '?'}  {size(u.vram_bytes)} on {u.device_pci}", file=out)
    if value.ollama != "reachable":
        print(f"Ollama: {value.ollama}", file=out)
    if value.events:
        print("Recent unloads", file=out)
        for event in value.events[-5:]:
            stamp = time.strftime("%H:%M:%S", time.localtime(event.time))
            print(f"  {stamp}  {event.model} ({event.app}), {size(event.freed_bytes)}, by {event.by}", file=out)


def main(argv: list[str] | None = None, out=None, err=None) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    parser = argparse.ArgumentParser(prog="kilix-gpu", description=__doc__)
    parser.add_argument("--namespace", help="registry directory (default /run/user/<uid>/kilix-gpu-residency-v1)")
    parser.add_argument("--ollama", help="Ollama URL (default from OLLAMA_HOST, else http://127.0.0.1:11434)")
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("status", help="devices, residents and unmanaged GPU processes")
    show.add_argument("--json", action="store_true")
    show.add_argument("--wake", action="store_true", help="read sleeping GPUs too (wakes them)")
    drop = commands.add_parser("unload", help="unload one resident by id or model")
    drop.add_argument("target")
    drop.add_argument("--force", action="store_true", help="also when pinned, busy or loading")
    for name in ("pin", "unpin"):
        commands.add_parser(name, help=f"{name} a resident by id or model").add_argument("target")
    room = commands.add_parser("room", help="dry run of ensure_room: what would be unloaded")
    room.add_argument("bytes", type=parse_bytes)
    room.add_argument("--device", help="device label or PCI address")
    room.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    where = {"namespace": args.namespace, "ollama": args.ollama}
    try:
        if args.command == "status":
            value = status(wake=args.wake, **where)
            if args.json:
                json.dump(status_json(value), out, indent=2, sort_keys=True)
                print(file=out)
            else:
                print_status(value, out)
        elif args.command == "unload":
            event = unload(args.target, force=args.force, **where)
            print(f"Unloaded {event.model} ({event.app}), freeing about {size(event.freed_bytes)}", file=out)
        elif args.command in ("pin", "unpin"):
            info = pin(args.target, args.command == "pin", **where)
            print(f"{'Pinned' if info.pinned else 'Unpinned'} {info.model} ({info.id})", file=out)
        else:
            return _room(args, where, out)
    except ResidencyError as error:
        print(f"kilix-gpu: {error} [{error.code}]", file=err)
        return 1
    return 0


def _room(args: argparse.Namespace, where: dict, out) -> int:
    try:
        room = ensure_room(args.bytes, device=args.device, requester="kilix-gpu", policy="never", **where)
        result = {"fits": True, "device_pci": room.device_pci, "free_bytes": room.free_bytes, "plan": []}
    except RoomError as error:
        if error.code == "unmeasured":
            raise
        result = {"fits": False, "device_pci": error.device, "free_bytes": error.free_bytes,
                  "plan": [{"id": r.id, "app": r.app, "model": r.model, "vram_bytes": r.vram_bytes}
                           for r in error.plan],
                  "enough": (error.free_bytes or 0) + sum(r.vram_bytes for r in error.plan) >= args.bytes}
    if args.json:
        json.dump({"schema": ROOM_SCHEMA, "need_bytes": args.bytes, **result}, out, indent=2, sort_keys=True)
        print(file=out)
    elif result["fits"]:
        print(f"{size(args.bytes)} fits on {result['device_pci']}: {size(result['free_bytes'])} free", file=out)
    elif result["enough"]:
        names = ", ".join(f"{r['model']} ({r['app']}, {size(r['vram_bytes'])})" for r in result["plan"])
        print(f"{size(args.bytes)} needs unloading on {result['device_pci']}: {names}", file=out)
    else:
        print(f"{size(args.bytes)} does not fit on {result['device_pci']} even with every idle resident "
              f"unloaded ({size(result['free_bytes'])} free)", file=out)
    return 0 if result["fits"] or result["enough"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
