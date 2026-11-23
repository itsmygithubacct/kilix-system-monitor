# kilix-gpu-residency

`kilix_gpu_residency` is the one per-user record of which models are resident
on which GPU, and the `kilix-gpu` command that shows and manages it. Apps
register the models they load, mark them busy or idle, and call `ensure_room`
before loading. When free VRAM is short, `ensure_room` unloads other apps'
idle, unpinned models, least recently used first. Models Ollama has loaded are
found through Ollama's own list, whoever loaded them. GPU processes nobody
registered are listed and never touched.

Its model catalog (`kilix_gpu_residency.catalog`, `kilix-gpu models`) is the one
place apps install, list, verify and remove local models. It refuses to remove
anything resident.

Standard library only, no daemon. Every call takes the registry's `flock` for
its duration. The trust boundary is one user's cooperating processes, as for
[kilix-device-lease](../kilix-device-lease/README.md), which this component
neither uses nor changes. A lease grant is execution coordination; a resident
record is memory bookkeeping.

## Using it from an app

```python
import os, subprocess
from kilix_gpu_residency import register, RoomError, ResidencyError

try:
    # Make room and reserve it under one lock, before the model process exists.
    resident = register(app="kilix-chat", model="LLaDA-MoE-7B Q4_K_M", backend="process",
                        device="Vulkan1", need_bytes=5_200_000_000)
except RoomError as error:
    ...  # error.free_bytes, error.need_bytes, error.plan: nothing was unloaded
for eviction in resident.room.evicted:
    notice(f"Freed {eviction.freed_bytes / 1e9:.1f} GB ({eviction.model}, {eviction.app})")
with resident:
    child = subprocess.Popen(argv, start_new_session=True)
    resident.attach(pgid=child.pid)   # the record now follows the child's process group
    resident.busy()                   # around each turn
    child.wait()
    resident.idle()
# Leaving the block removes the record, unloading the model first if it still runs.
```

- `register(app=, model=, backend=, device=None, pid=None, pgid=None, need_bytes=None, policy="evict-idle", pinned=False, unload="auto")`
  - `app` and `device` are labels: 1–96 ASCII letters, digits and `_ . : -`, starting with a letter or digit.
  - `model` is a display name of up to 128 printable characters, never a prompt or user text. For backend `ollama` it is the Ollama model name.
  - `backend` is `process`, `http` or `ollama`.
  - The record follows a process: the `pgid` leader, else `pid`, else the calling process. It is pruned once that process is gone, or has the same PID but another start time.
  - A record starts in state `loading`. While loading, it reserves `need_bytes` beyond what its processes already hold, so two apps cannot both take the same free memory.
  - `unload="auto"` gives a process group `{"kind": "signal", "pgid": N}`, and an Ollama model `{"kind": "ollama", "model": NAME}`. A record without a group gets no unload action, and nothing unloads it.
- `Resident.busy()`, `.idle()`, `.attach(pid=|pgid=)` and `.close(unload=True)`.
  - Any of them raises `ResidencyError("gone")` once the record has been unloaded or pruned.
  - `.evicted` says who unloaded it, when, and how much was freed.
- `ensure_room(need_bytes, device=, requester=, policy="evict-idle")` returns `Room(device_pci, need_bytes, free_bytes, evicted, plan)`.
  - It measures the device's free VRAM, minus other records' loading reservations.
  - If that is short, it plans unloads in least-recently-used order. Candidates are residents on that device that are `idle`, not pinned, have an unload action and hold VRAM. Busy, loading and unloading residents, pinned ones, unmanaged processes and other devices are never candidates.
  - If even every candidate would not be enough, it raises `RoomError` and unloads nothing. Its `plan` lists the candidates.
  - `policy="never"` only reports: it raises `RoomError` with the plan whenever unloading would be needed.
  - After unloading, it waits up to 10 s for the VRAM to come back.
- `status(wake=False)` returns devices, residents, unmanaged processes and recent unloads; `residents()` returns the residents alone. Without `wake`, a sleeping GPU is not woken to be read.
- `unload(id_or_model, force=False)` unloads one resident. A pinned, busy or loading one is refused unless `force`.
- `pin(id_or_model)` / `unpin(...)` mark a resident, including an Ollama model nobody registered, so `ensure_room` never unloads it.

`ResidencyError.code` is one of `invalid-request`, `unavailable`, `no-room`,
`unmeasured`, `not-found`, `refused` or `gone`. `RoomError` (codes `no-room`
and `unmeasured`) carries `need_bytes`, `free_bytes`, `device`, `plan` and
`evicted`.

## Unloading

- **A process group** gets `SIGTERM`, and `SIGKILL` after 5 s if any member remains.
  - The first signal is sent only while the group's registered leader is alive with its recorded start time.
  - Escalating is safe while the group has members, because Linux does not reuse a process group's number until it is empty.
  - The library refuses to signal its caller's own group, or group 1.
- **An Ollama model** is sent `POST /api/generate {"model": NAME, "keep_alive": 0}`, and the library waits until `/api/ps` no longer lists it. Ollama finishes a request already running on the model before it unloads.
- **Ollama is only contacted on loopback.** The default URL comes from `OLLAMA_HOST`, else `http://127.0.0.1:11434`. A remote Ollama's models are not on this machine's GPUs.
- The `http` unload kind (for example voicebox's `/models/unload`) is phase 2. In phase 1 an `http` backend unloads by its process group.

Every unload by `ensure_room` or by hand is logged in the registry. The newest
32 entries are kept, and `kilix-gpu status` shows the last five.

## Devices

A device is named by a label such as llama.cpp's `Vulkan1`, or by its PCI address. A label resolves:

1. as a PCI address, if it is one;
2. through `KILIX_GPU_DEVICES`, for example `Vulkan1=0000:01:00.0,CUDA0=0000:01:00.0`;
3. otherwise to the one GPU that has a VRAM counter, if exactly one has.

On the Quadro RTX 3000 laptop, `Vulkan0` is the Intel iGPU, which has no VRAM
counter, so `Vulkan1` resolves to the NVIDIA GPU. With two measurable GPUs and
no map, `ensure_room` refuses with `unmeasured` instead of guessing. An Ollama
model is attributed to the GPU where Ollama's processes hold memory.

VRAM is read by `kilix_gpu_residency.vram`, copied from Kilix's GPU readout:

- NVIDIA through NVML with ctypes, else `nvidia-smi`;
- AMD through sysfs `mem_info_vram_*` and per-process DRM fdinfo.

NVML is not opened while any NVIDIA GPU sleeps, unless the caller is about to
load a model. `ensure_room` asks for that; `status` does not, unless given
`--wake`. Per-resident VRAM is measured on every read and never stored or
declared.

## The registry

The registry lives in `/run/user/<euid>/kilix-gpu-residency-v1/`. The parent
must be user-owned mode 0700, and no ancestor may be writable by others except
a root-owned sticky directory. The directory is created at mode 0700, holding
`registry.lock`, one `<id>.json` per resident, `pins.json` and `events.json`,
each user-owned, mode 0600, with one link.

These are refused, not repaired:

- a symlinked, foreign or replaced directory or lock;
- an unexpected entry;
- a malformed record or one with duplicate fields.

Writes go to a temporary file and are renamed into place. A writer killed
before its rename leaves a temporary file, which the next call removes. Lock
descriptors are close-on-exec. A child forked without exec closes its copies,
so it cannot keep the registry locked. The registry holds at most 64 records.

A deliberate `namespace=` override (`--namespace` for the command) gives tests
and coordinated deployments a private registry. Every cooperating app must use
the same one.

## kilix-gpu

| Command | Effect |
| --- | --- |
| `kilix-gpu status [--json] [--wake]` | devices with total, used and free VRAM; each resident with app, model, backend, state, VRAM, pinned and idle time; unmanaged GPU processes; recent unloads |
| `kilix-gpu unload ID\|MODEL [--force]` | unload one resident |
| `kilix-gpu pin ID\|MODEL` / `unpin` | keep a resident from being unloaded to make room |
| `kilix-gpu room BYTES [--device D] [--json]` | dry run of `ensure_room` (`5.2G` or `4.8GiB`): exits 0 if it fits or would fit after unloading, 2 if it cannot |

`--json` output carries `schema` `kilix.gpu-residency.status/v1` or
`kilix.gpu-residency.room/v1`. From a source tree, run
`components/kilix-gpu-residency/kilix-gpu status`.

## The model catalog

`catalog.json` beside the module pins each downloadable model:

- an id, a display name and a kind (`gguf`, `diffusion-gguf` or `ollama`);
- its file, exact size and sha256, and a URL pinned to one Hugging Face revision (`https://huggingface.co/<owner>/<repo>/resolve/<40-hex revision>/<file>`);
- its folder role, architecture and context length (as the GGUF header has them), and a source and licence note.

It is seeded with puzzllm-fetch's six pinned models, plus `d3llm-llada`
(d3LLM's LLaDA, which the owner asked for and puzzllm-fetch gains too).

| Folder role | Folder | Kinds |
| --- | --- | --- |
| `llama.cpp` | `$KILIX_LLM_MODELS_DIR`, else `${GPU_TERMINAL_HOME:-~/.local/gpu_terminal}/kilix-llm/models` | `gguf` |
| `diffusion` | `$PUZZLLM_DIFFUSION_MODELS`, else `${XDG_DATA_HOME:-~/.local/share}/puzzllm/diffusion` | `diffusion-gguf` |

An empty variable counts as unset, as in the shell's `${VAR:-default}`. A
folder the catalog creates is mode 0700, with its parents. A folder that is not
yours, or that others can write, is refused.

| Command | Effect |
| --- | --- |
| `kilix-gpu models list [--json]` | catalog models (installed, partial, available, or mismatch by size), unlisted GGUFs in the folders (`local`, or `link` for a symlink), Ollama's installed models; and who holds each now: a registry app, Ollama, or a process with the file open or mapped |
| `kilix-gpu models install ID\|ollama:NAME [--json]` | download, verify and name a catalog file, or pull an Ollama model |
| `kilix-gpu models remove ID\|FILE\|PATH\|ollama:NAME [--unload] [--json]` | remove a model that nothing holds; `--unload` unloads it through the registry first |
| `kilix-gpu models verify ID\|FILE [--json]` | recompute an installed catalog file's size and sha256 |

`models --catalog FILE` uses another catalog. Only a catalog given by path may
use `file://` or loopback `http://` URLs, for tests and local mirrors.

`models --progress auto|lines|none` sets how install and verify report progress on stderr:
- `auto`, the default, redraws one line, and only on a terminal.
- `lines` prints a whole line about once a second, such as `downloading llada-1.5: 40% of 4.6 GiB`. This suits a menu that shows the latest line from a pty, as puzzllm's Downloads page does.
- `none` is silent.

Every action prints one line on stdout, for a game menu to show as a notice,
and exits with a code:

| Exit | Meaning | Lines |
| --- | --- | --- |
| 0 | done | `saved X`, `already saved X`, `pulled N`, `already installed N`, `removed X`, `verified X` |
| 1 | failed | `could not download X`, `downloaded X did not match`, `could not pull N`, `Ollama is not reachable` |
| 2 | refused | `unknown model ID`, `X is already in the models folder and does not match`, `refusing to replace a symlink: X`, `another download is using the models folder`, `X is not installed`, `not enough disk space for X: ...` |
| 3 | in use | `X is loaded by APP; use --unload`, `X is pinned by APP; unpin it first`, `X is busy in APP; try again when it is idle`, `X is open by pid N (COMMAND); close it first` |
| 4 | does not match | `X does not match the catalog (size)` or `(sha256)` |

A failure's cause, when there is one, goes to stderr. `--json` prints one
object with `schema` `kilix.gpu-residency.models/v1`, `outcome`, `exit`,
`message`, `detail`, `name` and `path`. The interface document lists every
outcome with its exit code.

### Installs

Installs keep puzzllm-fetch's semantics and its per-folder lock file,
`.fetch.lock`, so the two never write a folder at once:

- A matching file of the same name is `already saved`.
- A different file of that name is refused, and so is a symlink.
- The download goes to `.<file>.part`, is checked against the size and sha256, and only then is hard-linked to its name, so a file that appeared meanwhile is never replaced.

In addition, a dropped connection is resumed with `Range`, up to five
consecutive attempts that make no progress. A download that still fails keeps
its `.part`, and the next install resumes it. Other cases:
- A server that ignores `Range` starts over.
- A server that resumes at another offset, or serves a file of another size, is refused.
- A part that does not hash to the catalog's sha256 is deleted.
- A part another process has open is left alone.

Redirects may only go to https, or from loopback http to loopback http. Ollama
models are pulled through `POST /api/pull` on a loopback Ollama, with
puzzllm-fetch's name rule.

### Removal

A model is not removed while:

- a registry record holds it (matched by catalog id, file name, file stem or path);
- Ollama has it loaded;
- any process this user can inspect has the file open or mapped. Other users' processes cannot be seen.

`--unload` unloads registry and Ollama residents through `kilix_gpu_residency.unload`, then waits up to 10 s for the file to be closed. It still refuses a pinned, busy or loading resident. It never touches a process the registry does not manage: a file held by such a process is refused with its pid.

Only a regular file named `*.gguf` directly in a models folder can be
removed, with its idle `.part`. That excludes symlinks, other files such as
puzzllm-fetch's `.partial.*` and `.download.json`, and anything in a
subfolder. A part someone is writing is never removed. Ollama models are
removed with `DELETE /api/delete`, using the exact tag Ollama lists. If the
registry cannot be read, removal fails closed.

### From Python

```python
from kilix_gpu_residency import catalog

result = catalog.install("llada-moe-7b-a1b", progress=lambda stage, done, total: ...)
print(result.message)                     # "saved LLaDA-MoE-7B-A1B-Instruct.Q4_K_M.gguf"
try:
    catalog.remove("llada-moe-7b-a1b")
except catalog.CatalogError as error:
    print(error, error.outcome, error.exit_code)  # "... is loaded by kilix-chat; use --unload" loaded 3
for model in catalog.list_models().models:
    print(model.id, model.state, model.resident)
```

`install`, `remove`, `verify` and `list_models` take `catalog=` (from
`catalog.load(path)`), and `namespace=` and `ollama=` as the registry calls do.

## Interface document

`contracts/kilix.gpu-residency-v1.interface.json` at the monorepo root records
this interface:

- the version, backends, states, unload kinds, policies and error codes;
- the record fields, the bounds and timeouts;
- the default namespace, the eviction rules and the Ollama calls;
- the catalog: its data schema, kinds, folders, URL pattern, lock, part name, outcomes and exit codes.

`tests/test_interface_document.py` holds the module to that document, and each
comparison also runs against planted drifts.

## Checks

From the monorepo root, with the release-pinned uv 0.12.5:

    make residency-check UV=/absolute/path/to/uv-0.12.5
    make residency-defects UV=/absolute/path/to/uv-0.12.5

`residency-check` is part of `make check`. It uses:

- fake GPUs: a fake sysfs, and a fake NVML whose memory follows real processes;
- a fake Ollama on a loopback port;
- a fake loopback download server shaped like Hugging Face, which can redirect, drop a connection partway, ignore `Range`, or resume at the wrong offset;
- `file://` mirrors and real process groups.

Every test process, and every Python child it starts, installs
`tests/containment/residency_path_guard.py`. The guard refuses:

- any path under `/run/user` or naming the default namespace leaf;
- the real model folders: those under the account's home directory (taken from the password database), and whatever the model-folder variables named when the suite started;
- loading the real NVML, or running `nvidia-smi`;
- connecting anywhere but loopback, or to port 11434;
- signalling any process that is not a descendant of the suite.

`residency-defects` applies each of the planted defects in `tests/defects.py`
to a private copy and requires the suite to turn red for every one. Run both
inside a private mount namespace with an empty `/run/user`, because the guard
cannot see calls made through ctypes.

## Status

Phase 1 of the plebian-os GPU residency design, with the model catalog (P1b), at version 0.1.0:

- Phase 2 points Kilix core's GPU readout at `kilix_gpu_residency.vram` and adds the `http` unload kind.
- Phase 3 is a lease interface revision binding `llm-turn` grants to a resident record.

MIGRATION.md records where the VRAM reader was copied from.
