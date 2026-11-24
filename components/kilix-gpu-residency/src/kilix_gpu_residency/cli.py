"""kilix-gpu: show and manage the models resident on this machine's GPUs.

Every line printed for a person goes through clean(): text from other processes
(a command name, an app label, Ollama's error text) cannot reach the terminal as
a control sequence. JSON output is json.dump's, whose escapes already keep every
control character out of the terminal.

Exit codes: 0 done; 1 failed, refused or unavailable (``kilix-gpu: MESSAGE [code]``
on stderr); 2 ``room`` does not fit even with every idle resident unloaded; 130
interrupted. ``models`` has its own codes (catalog.OUTCOMES). Anything unexpected
prints one line on stdout and exits 1.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time

from . import (ResidencyError, RoomError, Status, VERSION, ensure_room, pin, status, unload)
from . import catalog as _catalog

STATUS_SCHEMA = "kilix.gpu-residency.status/v1"
ROOM_SCHEMA = "kilix.gpu-residency.room/v1"
_UNITS = {"": 1, "B": 1, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12,
          "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40, "KB": 10**3, "MB": 10**6, "GB": 10**9,
          "TB": 10**12}
_BYTES = re.compile(r"(\d+(?:\.\d+)?)\s*([A-Za-z]*)\Z")
# C0 (tab and newline included), DEL, C1, and the bidirectional controls that reorder what is shown.
_CONTROLS = re.compile("[\x00-\x1f\x7f-\x9f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")


def clean(text: object) -> str:
    """``text`` with every control character shown as an escape (``\\x1b``, ``\\u202e``)."""
    return _CONTROLS.sub(lambda m: f"\\x{ord(m.group()):02x}" if ord(m.group()) < 0x100
                         else f"\\u{ord(m.group()):04x}", str(text))


def say(text: str, file, **kwargs) -> None:
    print(clean(text), file=file, **kwargs)


class Bytes(int):
    """A byte count that remembers whether it was given in powers of ten."""
    decimal: bool = False


def parse_bytes(text: str) -> Bytes:
    """``5200000000``, ``5.2G`` (powers of ten) or ``4.8GiB`` (powers of two)."""
    match = _BYTES.fullmatch(text.strip())
    if match is None or match.group(2).upper() not in _UNITS:
        raise argparse.ArgumentTypeError(f"not a byte count: {text!r}")
    value = Bytes(int(float(match.group(1)) * _UNITS[match.group(2).upper()]))
    if value <= 0:
        raise argparse.ArgumentTypeError("the byte count must be positive")
    value.decimal = "I" not in match.group(2).upper() and match.group(2) != ""
    return value


def size(value: int | None, decimal: bool = False) -> str:
    """Powers of two (GiB, MiB), or of ten (GB, MB) when the person gave a size that way."""
    if value is None:
        return "?"
    if decimal:
        return f"{value / 1e9:.1f} GB" if value >= 10**9 else f"{value / 1e6:.0f} MB"
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
                       "unload": r.unload_kind, "unit": r.unit, "reserved_bytes": r.reserved_bytes}
                      for r in value.residents],
        "unmanaged": [{"pid": u.pid, "command": u.command, "device_pci": u.device_pci, "vram_bytes": u.vram_bytes}
                      for u in value.unmanaged],
        "foreign": [f.__dict__ for f in value.foreign],
        "events": [event.__dict__ for event in value.events],
    }


def print_status(value: Status, out) -> None:
    now = time.time()
    say("Devices", out)
    for d in value.devices:
        if d.asleep:
            memory = "asleep (not woken to read)"
        elif d.source == "held":
            memory = "not read: another NVIDIA GPU is asleep"
        elif d.total_bytes is None:
            memory = "no VRAM counter"
        else:
            memory = f"{size(d.used_bytes)} used / {size(d.total_bytes)}, {size(d.free_bytes)} free"
        say(f"  {d.pci}  {d.name}: {memory}", out)
    say("Residents", out)
    if not value.residents:
        say("  none", out)
    seen: set[str] = set()
    for r in value.residents:
        first = r.unit not in seen
        seen.add(r.unit)
        flags = " pinned" if r.pinned else ""
        flags += "" if r.registered else " (found in Ollama, not registered)"
        flags += f"  reserving {size(r.reserved_bytes)}" if r.reserved_bytes else ""
        shown = size(r.vram_bytes) if first else "(counted above)"  # one figure per unit
        say(f"  {r.id}  {r.app}  {r.model}  [{r.backend}] {r.state}  {shown}"
            f"  idle {_idle(r.last_used, now) if r.state == 'idle' else '-'}{flags}", out)
    say("Unmanaged GPU processes (never unloaded)", out)
    if not value.unmanaged:
        say("  none", out)
    for u in value.unmanaged:
        say(f"  pid {u.pid}  {u.command or '?'}  {size(u.vram_bytes)} on {u.device_pci}", out)
    if value.foreign:
        say("Records this version does not act on (never unloaded)", out)
        for f in value.foreign:
            say(f"  {f.id}  {f.app}  {f.model}  {f.version}  {f.reason}  {size(f.vram_bytes)}", out)
    if value.ollama != "reachable":
        say(f"Ollama: {value.ollama}", out)
    if value.events:
        say("Recent unloads", out)
        for event in value.events[-5:]:
            stamp = time.strftime("%H:%M:%S", time.localtime(event.time))
            note = "" if event.confirmed else " (asked; Ollama had not unloaded it yet)"
            say(f"  {stamp}  {event.model} ({event.app}), {size(event.freed_bytes)}, by {event.by}{note}", out)


def main(argv: list[str] | None = None, out=None, err=None) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    try:
        return _main(argv, out, err)
    except KeyboardInterrupt:
        say("kilix-gpu: interrupted", out, flush=True)
        return 130
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001 - the last resort: one line, never a traceback
        say(f"kilix-gpu: internal error: {type(error).__name__}: {error}", out, flush=True)
        return 1


def _main(argv: list[str] | None, out, err) -> int:
    parser = argparse.ArgumentParser(prog="kilix-gpu", description=__doc__)
    parser.add_argument("--namespace", help="registry directory (default /run/user/<uid>/kilix-gpu-residency-v1)")
    parser.add_argument("--ollama", help="Ollama URL (default from OLLAMA_HOST, else http://127.0.0.1:11434)")
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("status", help="devices, residents and unmanaged GPU processes")
    show.add_argument("--json", action="store_true")
    show.add_argument("--wake", action="store_true", help="read sleeping GPUs too (wakes them)")
    drop = commands.add_parser("unload", help="unload one resident by id or model")
    drop.add_argument("target")
    drop.add_argument("--force", action="store_true", help="also when pinned, busy, loading or shared")
    drop.add_argument("--shared", action="store_true",
                      help="also when other apps hold the same model or group (each must still be idle)")
    for name in ("pin", "unpin"):
        commands.add_parser(name, help=f"{name} a resident by id or model").add_argument("target")
    room = commands.add_parser("room", help="dry run of ensure_room: what would be unloaded")
    room.add_argument("bytes", type=parse_bytes)
    room.add_argument("--device", help="device label or PCI address")
    room.add_argument("--json", action="store_true")
    models = commands.add_parser("models", help="the shared model catalog: list, install, remove, verify")
    models.add_argument("--catalog", help="a catalog file instead of the packaged one")
    models.add_argument("--progress", choices=("auto", "lines", "none"), default="auto",
                        help="install and verify progress on stderr: auto redraws one line on a terminal; "
                             "lines prints a whole line about once a second (for a menu that shows the latest "
                             "line); none is silent")
    actions = models.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="catalog, local and Ollama models, and who holds them").add_argument(
        "--json", action="store_true")
    get = actions.add_parser("install", help="install a catalog id, or ollama:NAME")
    get.add_argument("target")
    get.add_argument("--json", action="store_true")
    drop_model = actions.add_parser("remove", help="remove a catalog id, a GGUF file name or path, or ollama:NAME")
    drop_model.add_argument("target")
    drop_model.add_argument("--unload", action="store_true",
                            help="unload it through the registry first (never a pinned or busy one)")
    drop_model.add_argument("--force", action="store_true",
                            help="with --unload: also an Ollama model no app registered, which may be in use")
    drop_model.add_argument("--json", action="store_true")
    check = actions.add_parser("verify", help="recompute an installed catalog file's sha256, or check a catalog Ollama model's digest")
    check.add_argument("target")
    check.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    where = {"namespace": args.namespace, "ollama": args.ollama}
    if args.command == "models":
        return _models(args, where, out, err)
    try:
        if args.command == "status":
            value = status(wake=args.wake, **where)
            if args.json:
                json.dump(status_json(value), out, indent=2, sort_keys=True)
                print(file=out)
            else:
                print_status(value, out)
        elif args.command == "unload":
            event = unload(args.target, force=args.force, shared=args.shared, **where)
            note = "" if event.confirmed else " (asked; Ollama had not unloaded it yet)"
            say(f"Unloaded {event.model} ({event.app}), freeing about {size(event.freed_bytes)}{note}", out)
        elif args.command in ("pin", "unpin"):
            info = pin(args.target, args.command == "pin", **where)
            say(f"{'Pinned' if info.pinned else 'Unpinned'} {info.model} ({info.id})", out)
        else:
            return _room(args, where, out)
    except ResidencyError as error:
        say(f"kilix-gpu: {error} [{error.code}]", err)
        return 1
    return 0


def _room(args: argparse.Namespace, where: dict, out) -> int:
    """Exit 0 when it fits or would fit after unloading, 2 when it would not, 1 when the device
    is unknown (invalid-request) or its free VRAM cannot be measured (unmeasured)."""
    decimal = getattr(args.bytes, "decimal", False)
    try:
        room = ensure_room(int(args.bytes), device=args.device, requester="kilix-gpu", policy="never", **where)
        result = {"fits": True, "enough": True, "device_pci": room.device_pci, "free_bytes": room.free_bytes,
                  "plan": []}
    except RoomError as error:
        if error.code not in ("no-room", "policy-never"):
            raise
        result = {"fits": False, "enough": error.code == "policy-never", "device_pci": error.device,
                  "free_bytes": error.free_bytes,
                  "plan": [{"id": r.id, "app": r.app, "model": r.model, "vram_bytes": r.vram_bytes}
                           for r in error.plan]}
    if args.json:
        json.dump({"schema": ROOM_SCHEMA, "need_bytes": int(args.bytes), **result}, out, indent=2, sort_keys=True)
        print(file=out)
    elif result["fits"]:
        say(f"{size(args.bytes, decimal)} fits on {result['device_pci']}: {size(result['free_bytes'], decimal)} free",
            out)
    elif result["enough"]:
        names = ", ".join(f"{r['model']} ({r['app']}, {size(r['vram_bytes'], decimal)})" for r in result["plan"])
        say(f"{size(args.bytes, decimal)} needs unloading on {result['device_pci']}: {names}", out)
    else:
        say(f"{size(args.bytes, decimal)} does not fit on {result['device_pci']} even with every idle resident "
            f"unloaded ({size(result['free_bytes'], decimal)} free)", out)
    return 0 if result["fits"] or result["enough"] else 2


def _progress(mode: str, target: str, err):
    """A progress callback for ``mode``, or None. ``auto`` redraws in place on a terminal only."""
    if mode == "none" or (mode == "auto" and not err.isatty()):
        return None
    last = [0.0, ""]

    def show(stage: str, done: int | None, total: int | None) -> None:
        now = time.monotonic()
        if now - last[0] < (0.5 if mode == "auto" else 1.0) and done != total and stage == last[1]:
            return
        last[0], last[1] = now, stage
        amount = f": {100 * done // total}% of {size(total)}" if done is not None and total else ""
        if mode == "lines":
            say(f"{stage} {target}{amount}", err, flush=True)
        else:
            print(f"\r{clean(stage)} {clean(target)}{amount}\033[K", end="", file=err, flush=True)
    return show


def models_json(value: _catalog.Inventory) -> dict:
    return {"schema": _catalog.RESULT_SCHEMA, "version": VERSION, "action": "list", "folders": dict(value.folders),
            "ollama": value.ollama, "residency": value.residency,
            "models": [dataclasses.asdict(model) for model in value.models]}


def _holder(value: dict) -> str:
    return value["app"] if "app" in value else f"pid {value['pid']} ({value['command']})"


def print_models(value: _catalog.Inventory, out) -> None:
    titles = {"llama.cpp": "llama.cpp models", "diffusion": "Diffusion models"}
    for role, directory in value.folders.items():
        say(f"{titles[role]}: {directory}", out)
        rows = [m for m in value.models if m.folder == role]
        if not rows:
            say("  none", out)
        for m in rows:
            state = f"partial {size(m.partial_bytes)}" if m.state == "partial" else m.state
            label = m.name if m.source == "catalog" else ""
            held = f"  held by {', '.join(_holder(h) for h in m.resident)}" if m.resident else ""
            say(f"  {m.id:<38} {label:<38} {size(m.bytes):>9}  {state}{held}", out)
    say("Ollama models", out)
    rows = [m for m in value.models if m.kind == "ollama"]
    if value.ollama != "reachable":
        say(f"  Ollama is {value.ollama.replace('-', ' ')}", out)
    elif not rows:
        say("  none", out)
    for m in rows:
        held = f"  loaded by {', '.join(_holder(h) for h in m.resident)}" if m.resident else ""
        say(f"  {m.id:<38} {'':<38} {size(m.bytes):>9}  {m.state}{held}", out)
    if value.residency != "known":
        say("Who holds each model is unknown: the registry could not be read", out)


def _models(args: argparse.Namespace, where: dict, out, err) -> int:
    """One line on stdout for every outcome (a game menu shows it as is); --json for one object.

    The result is always the last line: progress and any detail go to stderr, flushed, before it."""
    progress = None
    holders: tuple[dict, ...] = ()
    try:
        catalog = _catalog.load(args.catalog)
        if args.action == "list":
            value = _catalog.list_models(catalog=catalog, **where)
            if args.json:
                json.dump(models_json(value), out, indent=2, sort_keys=True)
                print(file=out)
            else:
                print_models(value, out)
            return 0
        progress = None if args.json else _progress(args.progress, args.target, err)
        if args.action == "install":
            result = _catalog.install(args.target, catalog=catalog, ollama=args.ollama, progress=progress)
        elif args.action == "remove":
            result = _catalog.remove(args.target, unload=args.unload, force=args.force, catalog=catalog, **where)
        else:
            result = _catalog.verify(args.target, catalog=catalog, progress=progress, ollama=args.ollama)
        outcome, message, detail = result.outcome, result.message, None
        name, path = result.name, result.path
    except _catalog.CatalogError as error:
        outcome, message, detail, name, path = error.outcome, str(error), error.detail, error.name, error.path
        holders = error.holders
    if progress is not None and args.progress == "auto":
        print("\r\033[K", end="", file=err, flush=True)
    code = _catalog.OUTCOMES[outcome]
    if args.json:
        json.dump({"schema": _catalog.RESULT_SCHEMA, "version": VERSION, "action": args.action,
                   "target": getattr(args, "target", None), "outcome": outcome, "exit": code,
                   "message": message, "detail": detail, "name": name, "path": path,
                   "holders": list(holders)}, out, sort_keys=True)
        print(file=out)
    else:
        if detail:
            say(f"kilix-gpu: {detail}", err, flush=True)
        err.flush()
        say(message, out, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
