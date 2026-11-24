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

Standard library only, no daemon. Calls that change the registry take its
`flock` exclusively. Reads take it shared. Neither holds it while an unload
or a release wait runs. The trust boundary is one user's cooperating
processes, as for [kilix-device-lease](../kilix-device-lease/README.md), which
this component neither uses nor changes. A lease grant is execution
coordination; a resident record is memory bookkeeping.

The library raises only `ResidencyError` (and `RoomError`, which is one), and
the catalog only `CatalogError`, for any input and any I/O failure.

## Using it from an app

This block is run verbatim by `tests/test_review_control_findings.py`
(`ReadmeExampleControls`). `command`, `wait_until_loaded` and `notice` are the
app's own.

```python
import subprocess
from kilix_gpu_residency import register, RoomError

try:
    # Make room and reserve it under one lock, before the model process exists.
    resident = register(app="kilix-chat", model="LLaDA-MoE-7B Q4_K_M", backend="process",
                        device="Vulkan1", need_bytes=5_200_000_000)
except RoomError as error:
    notice(f"Not enough VRAM: {error}")   # error.code, .free_bytes, .need_bytes, .plan, .evicted
    raise
for eviction in resident.room.evicted:
    notice(f"Freed {eviction.freed_bytes / 1e9:.1f} GB ({eviction.model}, {eviction.app})")
try:
    child = subprocess.Popen(command, start_new_session=True)
    resident.attach(pgid=child.pid)   # the record now follows the child's process group
    wait_until_loaded(child)          # for example, poll the server's /health
    resident.loaded()                 # the weights are in VRAM: the reservation ends
    resident.busy()                   # around each turn
    resident.idle()
finally:
    # The app stops its own child; close removes the record without signalling it.
    resident.close(unload=False)
```

- `register(app=, model=, backend=, device=None, pid=None, pgid=None, need_bytes=None, policy="evict-idle", pinned=False, unload="auto")`
  - `app` and `device` are labels: 1–96 ASCII letters, digits and `_ . : -`, starting with a letter or digit.
  - `model` is a display name of up to 128 printable characters, never a prompt or user text. For backend `ollama` it is the Ollama model name. For a catalog file, register its absolute path or its catalog id.
  - `backend` is `process`, `http` or `ollama`.
  - The record follows a process: the `pgid` leader, else `pid`, else the calling process. It is pruned once that process is gone, or has the same PID but another start time.
  - `pgid=` (and `attach(pgid=)`) accepts only the caller's own group, or a group whose leader is the caller or descends from it, such as a child started with `start_new_session=True`. `pid=` follows the same rule. Anything else is `invalid-request`.
  - Every refusal (labels, the 64-record cap, the group rule) comes before any unload.
  - A record starts in state `loading`. With `need_bytes`, it reserves `need_bytes` minus what its processes already hold, on its own device. The reservation lasts until `loaded()`, `idle()` or 600 s after `register` (`LOAD_TIMEOUT_SECONDS`), whichever is first. `busy()` keeps it, so two apps cannot both take the same free memory while weights are still arriving.
  - `unload="auto"` gives a process group `{"kind": "signal", "pgid": N}`, and an Ollama model `{"kind": "ollama", "model": NAME}`. A record without a group gets no unload action, and nothing unloads it.
- `Resident.busy()`, `.idle()`, `.loaded()`, `.attach(pid=|pgid=)` and `.close(unload=True)`.
  - Any of them raises `ResidencyError("gone")` once the record has been unloaded or pruned.
  - `.evicted` says who unloaded it, when, and how much was freed.
  - `.close()` removes the record. With `unload=True` it unloads the model only when nothing else holds the unit: no other live record names the same Ollama model or process group, and it is not pinned. Otherwise it removes only its own record.
  - A `SIGTERM` handler may call `.close(unload=False)`. No call holds the lock while an unload or release wait runs, so the handler gets it within the lock wait.
- `ensure_room(need_bytes, device=, requester=, policy="evict-idle")` returns `Room(device_pci, need_bytes, free_bytes, evicted, plan)`. See [Eviction](#eviction).
- `status(wake=False)` returns devices, residents, unmanaged processes, foreign records and recent unloads; `residents()` returns the residents alone. Without `wake`, a sleeping GPU is not woken to be read.
- `unload(id_or_model, force=False, shared=False)` unloads one resident's unit. A pinned, busy or loading one is refused unless `force`. A unit other live records also hold is refused unless `shared` or `force`, and the refusal names them.
- `pin(target)` / `unpin(target)`:
  - A record id pins that record.
  - An Ollama model, named `ollama:NAME` or by its name, is pinned as a model in `pins.json`. The pin holds for every record that names it, now or later, and for the model when nobody registered it.
  - Any other model name pins the one record that has it, by that record's own flag; a process record is never pinned through `pins.json`.
  - A record's own pin and a model pin both count. Unpinning one leaves the other, and the returned `pinned` says whether one still holds.

`ResidencyError.code` is one of `invalid-request`, `unavailable`, `no-room`,
`policy-never`, `unload-failed`, `not-released`, `unmeasured`, `not-found`,
`refused` or `gone`. `RoomError` carries `code`, `need_bytes`, `free_bytes`,
`device`, `plan` and `evicted`, and its code is one of:

| Code | Meaning | Unloaded |
| --- | --- | --- |
| `no-room` | not enough even if every candidate were unloaded | nothing |
| `policy-never` | unloading would make room, but the policy is `never` | nothing |
| `unmeasured` | the device's free VRAM cannot be measured | nothing |
| `unload-failed` | an unload failed, or Ollama had not confirmed it in time | what `evicted` lists |
| `not-released` | everything planned was unloaded, but the VRAM did not come back within 10 s | what `evicted` lists |

## Eviction

`ensure_room` measures the device's free VRAM, minus other records' current
reservations on that device. If that is short, it plans unloads.

- **Units.** One Ollama model is one unit: every record naming it, and Ollama's own entry for it. One process group is one unit: every record naming it. A record with neither is its own unit. `status` shows one VRAM figure per unit, on the unit's first row; its other rows show `(counted above)`. In phase 2 an `http` unit is one endpoint, shared the same way.
- **Candidates.** A unit is a candidate when it is on the requested device, has an unload action and measured VRAM, every live record in it is `idle`, and neither a record nor `pins.json` pins it. Busy, loading and unloading units, pinned ones, unmanaged processes, foreign records and other devices are never candidates.
- **Unknown clients' Ollama models.** A model Ollama has loaded that no app registered is evictable only when it looks idle, and "idle" is an estimate. Ollama reports only when the model expires, so its last use is estimated as `expires_at` minus the keep-alive (`OLLAMA_KEEP_ALIVE` when set, else Ollama's 300 s), never later than now. A model whose estimated last use is under 120 s ago (`RECENT_USE_SECONDS`) is not a candidate. A model kept for ever (a negative keep-alive), or one whose keep-alive cannot be read, has no estimate and is never a candidate. A request that has run longer than that on a model nobody registered is not visible to the estimate.
- **Order.** Registered units first, then Ollama models nobody registered; each least recently used first.
- **All or nothing.** If even every candidate would not be enough, it raises `RoomError("no-room")` and unloads nothing. Its `plan` lists the candidates. An Ollama model counts for no more than Ollama's processes measurably hold on the device, so a model Ollama says is larger cannot pass the check alone.
- **The lock.** Planned units are marked `unloading` under the lock, with the unloader's pid and start time. The unloads and the release wait run with the lock released. Then the lock is taken again to remove and log what was unloaded, mark a failed unit `busy`, and revert any unit not reached. An `unloading` mark whose unloader is gone reads as idle.
- `policy="never"` only reports: it raises `RoomError("policy-never")` with the plan whenever unloading would be needed.
- After unloading, it waits up to 10 s for the VRAM to come back, counting reservations made meanwhile.

## Unloading

- **A process group** gets `SIGTERM`, and `SIGKILL` after 5 s if any member remains.
  - The first signal is sent only while the group's registered leader is alive with its recorded start time. The leader is held by a pidfd from that check to the signal.
  - Escalating is safe while the group has members, because Linux does not reuse a process group's number until it is empty.
  - The library refuses to signal its caller's own group, or group 1.
- **An Ollama model** is sent `POST /api/generate {"model": NAME, "keep_alive": 0}`, and the library waits up to 30 s until `/api/ps` no longer lists it. Ollama finishes a request already running on the model before it unloads. An unload Ollama has not confirmed in time is logged with `confirmed: false`.
- **Ollama is only contacted on loopback**, with a 2 s deadline on each whole exchange. The default URL comes from `OLLAMA_HOST`, as Ollama's own client reads it: a bare host or `host:port`, `[::1]:11434` for IPv6, and the unspecified `0.0.0.0` and `::` meaning `127.0.0.1` and `::1`. Any other name than `localhost`, even one that resolves to loopback such as `ip6-localhost`, is refused as `not-loopback`. A remote Ollama's models are not on this machine's GPUs.
- **Ollama's processes** are the process named like Ollama that holds the socket listening on the configured port, and its descendants. A process that only names itself `ollama` is unmanaged. Ollama's models are attributed to a GPU only while Ollama holds memory on exactly one; while it spans two, its models are listed with no device and are never candidates (Ollama's list does not say which runner holds which model).
- The `http` unload kind (for example voicebox's `/models/unload`) is phase 2. In phase 1 an `http` backend unloads by its process group.

Every unload by `ensure_room` or by hand is logged in the registry. The newest
32 entries are kept, and `kilix-gpu status` shows the last five.

## Devices

A device is named by a label such as llama.cpp's `Vulkan1`, or by its PCI address:

1. A PCI address names itself, and must be a GPU this machine has.
2. Any other label must be in `KILIX_GPU_DEVICES`, for example `Vulkan0=0000:00:02.0,Vulkan1=0000:01:00.0`, and map to a GPU this machine has.
3. No device (`device=None`, or no `--device`) means the one GPU with a VRAM counter, if exactly one has.

An unknown label, or a map entry naming a GPU this machine does not have, is
`invalid-request`. A label is never guessed. On the Quadro RTX 3000 laptop,
`Vulkan0` is the Intel iGPU, which has no VRAM counter, so `ensure_room` on it
is `unmeasured`. The plebian-os installer writes `KILIX_GPU_DEVICES` from
`vulkaninfo` (MIGRATION.md). The map only names devices. It cannot widen what
is unloaded or signalled.

VRAM is read by `kilix_gpu_residency.vram`, copied from Kilix's GPU readout:

- NVIDIA through NVML with ctypes, else `nvidia-smi`;
- AMD through sysfs `mem_info_vram_*` and per-process DRM fdinfo.

NVML is not opened while any NVIDIA GPU sleeps, unless the caller is about to
load a model. `ensure_room` asks for that; `status` does not, unless given
`--wake`. Per-resident VRAM is what the resident's processes hold, measured on
every read and never stored. An Ollama model's figure is Ollama's own report
(`size_vram`), capped at what Ollama's processes measurably hold on the device;
`status` shows the capped figure.

`KILIX_GPU_SYSFS_ROOT` and `KILIX_GPU_PROC_ROOT` are test seams. They are read
only when the test guard is installed, and ignored otherwise.

## The registry

The registry lives in `/run/user/<euid>/kilix-gpu-residency-v1/`. The parent
must be user-owned mode 0700, and no ancestor may be writable by others except
a root-owned sticky directory. The directory is created at mode 0700, holding
`registry.lock`, one `<id>.json` per resident, `pins.json` and `events.json`,
each user-owned, mode 0600 whatever the umask, with one link.

- **Reads** (`status`, `residents`, `kilix-gpu room` and `models list`) create nothing, take the lock shared, read an absent registry as empty, and prune nothing.
- **Writes** (every other call) create the registry, take the lock exclusively, remove temporary files a killed writer left, and prune records whose holder is gone.

These are refused, not repaired:

- a symlinked, foreign or replaced directory or lock;
- an unexpected entry;
- a malformed record, or one with duplicate fields.

A record that is otherwise a safe registry file (regular, 0600, one link, valid
JSON, its id matching its name, version `kilix.gpu-residency/*`) but uses what
this version does not know is **foreign**: another version, an added field, an
unknown backend or unload kind. It is listed under `foreign`, never unloaded,
its holder's VRAM counted, and pruned only when its holder is dead in this PID
namespace. `pins.json` of another version is honoured and never rewritten;
`events.json` of another version reads as empty and is never rewritten.

Every participant should share one PID namespace. A record carries its
writer's PID namespace. Records from another namespace are never pruned or
signalled from here, and are listed foreign with reason `other-pid-namespace`.
A process in another user namespace that shares the registry is in another
PID namespace too, or sees the same `/proc`, so the same rule covers it.

Writes go to a temporary file and are renamed into place. Lock descriptors are
close-on-exec. A child forked without exec closes its copies, so it cannot keep
the registry locked. The registry holds at most 64 records.

A deliberate `namespace=` override (`--namespace` for the command) gives tests
and coordinated deployments a private registry. Every cooperating app must use
the same one.

## kilix-gpu

| Command | Effect |
| --- | --- |
| `kilix-gpu status [--json] [--wake]` | devices with total, used and free VRAM; each resident with app, model, backend, state, VRAM, pinned and idle time; unmanaged GPU processes; foreign records; recent unloads |
| `kilix-gpu unload ID\|MODEL [--force] [--shared]` | unload one resident's unit |
| `kilix-gpu pin ID\|MODEL` / `unpin` | keep a resident from being unloaded to make room |
| `kilix-gpu room BYTES [--device D] [--json]` | dry run of `ensure_room` (`5.2G` or `4.8GiB`): exits 0 if it fits or would fit after unloading, 2 if it cannot, 1 for an unknown device or an unmeasured one |

Exit codes: 0 done; 1 failed, refused or unavailable, with `kilix-gpu: MESSAGE
[code]` on stderr; 2 `room` does not fit; 130 interrupted. Anything unexpected
prints one line on stdout and exits 1. A size is echoed in the units given:
`room 1G` says `1.0 GB`, `room 1GiB` says `1.0 GiB`.

Every human-readable line shows control characters (C0, C1, DEL and the bidi
controls) as escapes, so another user's process name cannot write to the
terminal. `--json` output is `json.dump`'s, whose strings escape every control
character, and carries `schema` `kilix.gpu-residency.status/v1` or
`kilix.gpu-residency.room/v1`. The interface document lists every key. From a
source tree, run `components/kilix-gpu-residency/kilix-gpu status`.

## The model catalog

`catalog.json` beside the module pins each downloadable model:

- an id, a display name and a kind (`gguf`, `diffusion-gguf` or `ollama`);
- its file, exact size and sha256, and a URL pinned to one Hugging Face revision (`https://huggingface.co/<owner>/<repo>/resolve/<40-hex revision>/<file>`);
- its folder role, architecture and context length (as the GGUF header has them), and a source and licence note.

It is seeded with puzzllm-fetch's six pinned models, plus `d3llm-llada`
(d3LLM's LLaDA, which the owner asked for and puzzllm-fetch gains too).

It also pins four Ollama models. An Ollama entry names a model and pins its
manifest by the `digest` and `bytes` that `GET /api/tags` reports, which are
the sha256 of the manifest the registry serves and the size of its layers and
config. They are the best-scoring local models that fit H2, the hardware level
the model sizer's GPU profiles are measured on (an 8 GiB NVIDIA GPU and at
least 16 GiB RAM):

Measured on an H2 machine (an RTX 3070 with 8 GiB, Ollama 0.40.16), with the
strict YES/NO contract on the OneBit development items (442) and BoolQ
validation (3,270):

| id | Ollama name | Size | OneBit | BoolQ | Seconds an answer (OneBit, BoolQ) | Chosen for |
| --- | --- | --- | --- | --- | --- | --- |
| `d3-4b` | `d3:4b` | 5.8 GB | 94.8% | 92.2% | 0.19, 0.08 | the best judge on both |
| `d3-2b` | `d3:2b` | 3.2 GB | 86.0% | 89.8% | 0.07, 0.03 | the fastest judge, second on BoolQ |
| `qwen3.5-9b` | `qwen3.5:9b` | 6.6 GB | 90.5% | 87.6% | 0.19, 0.15 | general chat; second on OneBit, and the furthest of these in the game campaign |
| `gemma4-e4b` | `gemma4:e4b` | 6.6 GB | 89.1% | 84.8% | 0.12, 0.08 | general chat; third on OneBit |

The runners-up that fit, `qwen3.5:4b` (87.6%, 84.3%) and `bonsai-2:9b`
(87.1%, 86.6%), were measured alongside them. `bonsai-2:9b` is level with
`gemma4:e4b` on the two benchmarks taken together, but slower. Models that do
not fit 8 GiB are left out: `d3:9b` and `nimble` are 9B at Q8_0, and the 27B
models are larger still. `gemma4:e4b` is the manifest `gemma4:latest` named
when it was first ranked, pinned by a name that does not move. The scores
match the earlier runs on a 6 GiB laptop GPU to within two answers.

`ollama:NAME` still installs any Ollama model, unpinned.

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
| `kilix-gpu models remove ID\|FILE\|PATH\|ollama:NAME [--unload] [--force] [--json]` | remove a model that nothing holds; `--unload` unloads it through the registry first; `--force` also unloads an Ollama model no app registered |
| `kilix-gpu models verify ID\|FILE [--json]` | recompute an installed catalog file's size and sha256, or compare a catalog Ollama model's manifest digest with its pin |

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
| 1 | failed | `could not download X`, `downloaded X did not match`, `could not pull N`, `pulled N did not match the catalog`, `Ollama is not reachable`, `could not remove X` (also when an unload failed: the line names what was already unloaded) |
| 2 | refused | `unknown model ID`, `X is already in the models folder and does not match`, `N is already installed and does not match the catalog`, `refusing to replace a symlink: X`, `a download is running in that folder; try again when it finishes`, `X is not installed`, `not enough disk space for X: ...` |
| 3 | in use | `X is loaded by APP; use --unload`, `X is loaded in Ollama by a client that did not register it and may be in use; use --force to unload it anyway`, `X is pinned by APP; unpin it first`, `X is busy in APP; try again when it is idle`, `X is open by pid N (COMMAND); close it first` |
| 4 | does not match | `X does not match the catalog (size)`, `(sha256)` or, for Ollama, `(digest)` |

A failure's cause, when there is one, goes to stderr and is flushed before the
result line, so the result is always the last line, also with `--progress lines`
and with stderr merged into stdout, as in puzzllm's pty. A `list` that fails
prints one line too. `--json` prints one object with `schema`
`kilix.gpu-residency.models/v1`, `outcome`, `exit`, `message`, `detail`,
`holders`, `name` and `path`. The interface document lists every outcome with
its exit code. Any I/O error is one of these outcomes (`download-failed`,
`remove-failed`, or `unavailable` for `verify` and `list`), never a traceback.

### Installs

Installs keep puzzllm-fetch's semantics and its per-folder lock file,
`.fetch.lock`, so the two never write a folder at once:

- A matching file of the same name is `already saved`.
- A different file of that name is refused, and so is a symlink.
- The download goes to `.<file>.<pin>.part`, where `<pin>` is the first 12 hex digits of the catalog sha256, so a part of another revision is never resumed. It is checked against the size and sha256, and only then hard-linked to its name, so a file that appeared meanwhile is never replaced. If one did appear, it is hashed: a match is `already saved` and a mismatch `conflict`, and the part is removed either way.
- A part under the older name `.<file>.part` is never resumed; `remove` deletes it with its file. puzzllm-fetch's own temporary files (`.partial.*`) are ignored.
- Free disk space is checked once, before downloading, and none is reserved.
- Downloads connect directly and never use a proxy. When `https_proxy` is set, a failure's detail says so; puzzllm keeps its curl path for proxied networks. Loopback addresses never go through a proxy.
- Certificates are verified against the system's store.

In addition, a dropped connection is resumed with `Range`, up to five
consecutive attempts that make no progress. A download that still fails keeps
its part, and the next install resumes it. Other cases:
- A server that ignores `Range` starts over.
- A server that resumes at another offset, answers with the wrong length or total, or sends more than it promised, is refused, and the part is kept.
- A part is deleted only when the complete bytes do not hash to the catalog's sha256.
- A part larger than the file (a 416) starts over.
- A part another process has open is left alone.

Redirects may only go to https, or from loopback http to loopback http. Ollama
models are pulled through `POST /api/pull` on a loopback Ollama, with
puzzllm-fetch's name rule. For a catalog Ollama entry:
- An installed copy with another digest is refused, like a different file of the same name.
- A pull is checked afterwards. If the registry's tag has moved, the install fails with `pulled N did not match the catalog`. Ollama keeps what it pulled, and the notice says how to remove it.

### Removal

A model is not removed while:

- a registry record holds it: by catalog id, by absolute path (compared after resolving symlinks), or by bare file name or stem only while no other models folder has a file of that name;
- Ollama has it loaded;
- any process this user can inspect has the file open or mapped.

Other users' processes, such as a system service's, cannot be inspected, and
they do not block removal. That is deliberate. Removing a file another process
has open or mapped does not disturb that process: the kernel keeps the data
until the process closes it, and the space is freed then. And there are always
processes this user cannot inspect, so refusing on their account would refuse
every removal. A model folder is the user's own, so it is normally only the
user's processes that load from it.

`--unload` first checks that no process outside the holders' groups has the
file open; if one has, it is refused `open` and nothing is unloaded. It then
checks every holder (pinned, busy or loading refuses) before unloading any,
unloads each unit through `kilix_gpu_residency.unload`, and waits up to 10 s
for the file to be closed. An Ollama model no app registered may be in use by
another client, so `--unload` refuses it unless `--force` is given too. An
unload that fails is `remove-failed`, naming what was already unloaded. It
never touches a process the registry does not manage. The open-file check is
best effort (other users' processes are not visible); the registry is the
real guard.

Only a regular file named `*.gguf` directly in a models folder can be
removed, with its idle parts. That excludes symlinks, other files such as
puzzllm-fetch's `.partial.*` temporaries, and anything in a subfolder. No
sidecar file is read or removed. A part someone is writing is never removed. Ollama models are
removed with `DELETE /api/delete`, using the exact tag Ollama lists. If the
registry cannot be read, removal fails closed.

### From Python

Illustrative (it would download 4.6 GB); `tests/test_catalog*.py` cover each call.

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

- the version, backends, states, unload kinds, policies and error codes, with each room error's meaning;
- the API: every public function's and method's signature, and every result type's fields;
- the record fields, the bounds and timeouts, including `load_timeout_seconds` and `recent_use_seconds`;
- the default namespace, devices, reservations, the registry's read and write rules, the eviction rules and the Ollama calls;
- the command: its JSON keys, exit codes and Ollama states;
- forward compatibility, versioning and the known gaps;
- the catalog: its API, data schema, kinds, folders, URL pattern, lock, part names, outcomes and exit codes.

`tests/test_interface_document.py` holds the module to that document by what
the module does: signatures and fields by introspection, the eviction rules by
probing the planner with one unit per rule, and the JSON keys and exit codes by
running the command. Each comparison also runs against planted drifts.

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

- any path under `/run/user` or naming the default namespace leaf, including one reached relative to a directory descriptor, which the guard resolves through `/proc/self/fd` first;
- the real model folders: those under the account's home directory (taken from the password database), and whatever the model-folder variables named when the suite started;
- loading the real NVML, or running `nvidia-smi`;
- connecting anywhere but loopback, or to port 11434;
- signalling any process that is not a descendant of the suite.

`residency-defects` applies each of the planted defects in `tests/defects.py`
to a private copy and requires the suite to turn red for every one, by a
failing test: a defect that only breaks an import counts as a hole. It records
which tests caught each defect. Run both inside a private mount namespace with
an empty `/run/user`, because the guard cannot see calls made through ctypes.
The reviewers' controls and reproductions are in `tests/test_review_*.py`.

`make prefetch` runs with `--no-python-downloads`, so run
`uv python install 3.12.8` once first.

## Status

Phase 1 of the plebian-os GPU residency design, with the model catalog (P1b), at version 0.1.0:

- Phase 2 points Kilix core's GPU readout at `kilix_gpu_residency.vram` and adds the `http` unload kind.
- Phase 3 is a lease interface revision binding `llm-turn` grants to a resident record.

MIGRATION.md records where the VRAM reader was copied from.
