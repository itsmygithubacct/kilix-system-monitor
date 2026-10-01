# plebian-model-sizer

Answer inference supports `--quant f32` for an unquantized FP32 CPU runtime,
including four-byte model weights and KV cache. Quantized defaults describe a
different runtime and must not be used to admit FP32 help-model experiments.

Local resource estimates for document and speech models. Version 0.2.0
supports `kilix-help-llm` training/inference budgets and `kilix-voice` speech
inference planning against measured reference profiles. The development CLI
also offers an avatar-chat CPU shortlist for already installed Ollama models.

The output includes a provisional resource shortlist, exact checkpoint/config
identities, memory breakdowns, and the assumptions used. Quality and runtime
compatibility remain unmeasured; `selected_model` stays null and
`qualification_eligible` stays false. These are development estimates, not
measured peak memory or a guarantee that a job will complete.

## Usage

From a checkout of this component:

```sh
./plebian-model-sizer snapshot --json
./plebian-model-sizer recommend help-llm --catalog /path/to/candidates.json
./plebian-model-sizer recommend help-llm --catalog /path/to/candidates.json \
  --task both --phase both --context 2048 --batch 1 --lora-rank 16 --json
```

The source launcher uses the sibling `plebian-hardware` component. An installed
wheel provides the same `plebian-model-sizer` console command and requires
`plebian-hardware==0.1.0`. For development, `uv sync --locked` installs both from
the monorepo. Python 3.11 or newer is required.

Useful options:

| Option | Effect |
| --- | --- |
| `--task answer`, `rank`, `both` | Generation checkpoint, decision checkpoint, or both; default both |
| `--phase train`, `infer`, `both` | Independent resource checks; default both |
| `--train-backend cpu/cuda/auto` | CPU FP32 or CUDA BF16 frozen weights; default auto |
| `--infer-backend cpu/cuda/auto` | Independently choose inference resources; default auto |
| `--gpu N` | Use one physical NVIDIA index; GPUs are never pooled |
| `--context N`, `--batch N` | Total sequence tokens (including generation space), concurrent sequences/microbatch |
| `--lora-rank N` | All-linear adapter rank; default 16 |
| `--topics N` | Dense scoring-head outputs; default 128 |
| `--quant q4/q8/f16/f32` | Answer inference weight estimate; default q4; training/ranking weights are unchanged |
| `--co-resident` | Sum both inference workloads instead of sequential maximum |
| `--no-checkpointing` | Retain training activations across all layers |
| `--document-bytes N` | Reserve four times this amount for documents/preprocessing |
| `--data-root PATH` | Assess the filesystem where models and training data will live |
| `--resources PATH` | Use an explicit development snapshot for simulation |

`auto` chooses the observed NVIDIA GPU with the most free memory, or CPU if
none is observed and no explicit GPU was requested. An unobserved `--gpu N`
returns an unknown CUDA budget. It does not move an oversized GPU workload to CPU;
request `cpu` explicitly to evaluate that alternative. Device indices come from
`nvidia-smi`, not CUDA's potentially remapped runtime namespace. Observing free
memory does not prove a training/runtime installation can use the device.

Default storage is `~/.local/gpu_terminal/kilix-help-llm`, or
`GPU_TERMINAL_HOME/kilix-help-llm`. The commands inspect the nearest existing
ancestor without creating a directory. They read metadata and current capacity;
they do not fetch weights, run models, or persist a snapshot. Redirect output
into the user-data tree when retaining a report. Direct library/download caches
to the same assessed filesystem when implementing a training job.

## Accounting

The following recipe applies to `help-llm`; speech uses the reference profiles
described in the next section.

The catalog supplies exact revisions, shard sizes/digests, upstream parameter
counts, and architecture dimensions bound to each pinned config digest. Counts
cover the entire checkpoint, including unused vision tensors, until a smaller
export has been produced and measured. Unknown architectures or incomplete
metadata return `unknown` and cannot enter the shortlist.

The `lora-all-linear-checkpointed-v1` recipe budgets frozen base weights,
all-linear LoRA matrices, and a dense topic head for ranking. Training uses
16 bytes per trainable parameter for FP32 weights/master, gradients and Adam
moments, saved hidden states, layer/MLP workspaces, dense attention score buffers,
and output/loss buffers. Checkpointing retains layer-boundary activations and
one layer's workspace. Hybrid Qwen3.5 includes recurrent state for a full
sequence in that workspace. The recipe does not estimate QLoRA, full-model
fine-tuning, distributed training, throughput, or a custom pointer-head layout.

Inference counts grouped-query KV cache for full-attention layers and separate
FP32 recurrent state plus convolution caches for hybrid layers. Answer prefill
assumes chunks of at most 512 tokens; scoring uses the full sequence. Adapter
and scoring-head weights are FP32. Q4/Q8 planning factors are 0.75/1.125 bytes
per parameter, while both embedding/output matrices are conservatively kept at
FP16. These factors are estimates, not exact GGUF artifact sizes.

Every compute-memory estimate adds a 512 MiB runtime allowance and 25% margin.
CPU loading also budgets a serialized checkpoint; CUDA host staging budgets a
full FP32 checkpoint plus 1 GiB. Training is sequential across tasks. Inference
uses a sequential maximum by default, or a sum with `--co-resident`. Storage
counts each task once: source weights, two full-size merge/export scratch copies,
three adapter/optimizer checkpoints, and the declared preprocessing allowance.
This assumes retention is capped at three checkpoints and datasets fit the
specified allowance; measure a changed workflow with an appropriate profile.

Usable budgets subtract 2 GiB RAM, 512 MiB VRAM per selected GPU, and 1 GiB disk
from **current available** resources. RAM uses Linux `MemAvailable`, bounded by
`memory.max - memory.current` at every visible cgroup-v2 ancestor. Swap is not
counted. Hidden ancestors, legacy cgroup hierarchies and unreadable counters
remain unknown. NVIDIA free memory comes from a bounded trusted-system-path
probe. ROCm/Vulkan/unified-memory/offload budgets are not inferred from totals.
Snapshots older than five minutes or dated in the future cannot produce fits.
Capacity can change after observation; this command is not a reservation.

A candidate enters `shortlist` only when every requested phase, RAM/VRAM budget
and storage check returns `estimated-fit`. Ordering uses the largest actual
checkpoint parameter count among the selected tasks, then candidate ID. The
first is `provisional_candidate`; task evaluation must establish quality before
selecting a model. `unknown` and `does-not-fit` rows include their failed checks.
JSON consumers must inspect these fields: exit 0 means a report was produced,
including when the shortlist is empty. Invalid input exits 2.

## Avatar chat (development)

`recommend avatar-chat --catalog - --json` accepts a bounded
`kilix.avatar-chat.sizing-request/v1` object on stdin. Each model supplies an
Ollama model ID, its observed model byte size, parameter count, and caller-known
installed/runtime-supported flags. The report is bound to the complete request
by SHA-256. The fixed target is one CPU chat session at 8192 context tokens.

The provisional shortlist uses a deliberately conservative estimate of twice
the model bytes plus 512 MiB, while reserving two GiB and at least half of the
remaining live RAM for the desktop, speech, other applications and unmodelled
spikes. It orders fitting installed candidates by parameter count, largest
first. This is not a measured peak, latency or quality result; the caller must
verify model identity, evaluate answers and perform the actual selection.
`selected_model` remains null. No download, installation or runtime mutation
occurs. Stale or unknown RAM observations produce no shortlist.

## Speech models

`kilix-tts --recommend` and `kilix-stt --recommend` send their current catalogs
to this provider. The direct interface is:

```sh
plebian-model-sizer recommend voice --catalog /path/to/request.json --task both --json
```

`--catalog -` accepts bounded JSON on stdin. The request has schema
`kilix.voice.sizing-request/v1` and a `models` list; each entry carries `id`,
`task` (`tts` or `stt`), `backend` (`cpu` or `cuda`), `runtime_supported`
(boolean), and `installed` (boolean or null). The response schema is
`plebian.models.voice-sizing/v1-development`. Its `request_sha256` binds the
complete request using the same sorted compact JSON convention as the LLM
catalog digest. `--resources` supports explicit simulations; live collection
uses the voice data root, honoring `KILIX_DATA_HOME`, `KILIX_STORAGE_HOME`, and
`GPU_TERMINAL_HOME`. `--data-root` can select another filesystem explicitly.

The package carries exact copies of the seven frozen `res02-measured` speech
profile documents, each bound to its original SHA256. Tests compare every copy
with the frozen source. eSpeak, MBROLA, Kristin Piper and the two Vosk profiles
cover the current Voice catalog. CUDA Qwen/Whisper profiles remain available
for consumers that support their exact IDs and backends; this does not add
those runtimes to Voice. An unknown model, mismatched backend/architecture, or
missing measurement cannot produce a fit. VibeVoice has no profile and remains
unsupported in the current Voice runtime.

Speech inference uses each profile's measured peak RAM/VRAM plus its declared
safety margin (currently 20%). It reserves 256 MiB RAM and VRAM, and 128 MiB
disk, while retaining the common cgroup, stale-snapshot and free-memory checks.
These are estimates based on the **reference workload**, with exact local
runtime/artifact identity unverified. Arbitrary utterances, other voices,
concurrent models and latency/quality are not established by this comparison.

`verdict` and `inference` assess runtime memory. `installation` separately
requires known download, installed and temporary byte counts. Current profiles
have unknown temporary space, so installation remains unknown even when
inference fits. Installed status is displayed independently and does not waive
the missing measurements. A runtime fit does not admit an installation.
`shortlists` and `provisional_candidates` are keyed by speech task and ordered
by the estimated RAM requirement, then ID. They rank resource cost, not quality.
Unsupported consumer runtimes are excluded. Model selection, installation and
execution remain separate; `selected_model` is null and qualification false.

`defaults.stt` (present when the STT task is assessed) names the dictation
model to offer by default: the first of `whisper-small-en`, `small-en-us` that
is on the STT shortlist and whose hardware class the host meets. Whisper
small.en requires x86-64 with AVX2, at least 8 logical CPUs and at least 16 GiB
of total RAM; below that a sentence takes several seconds, too slow for
dictation. The CPU class comes from `/proc/cpuinfo` and is echoed as `cpu`; it
is observed only for live snapshots, so a provided snapshot never qualifies
Whisper. The default is an offer for the desktop to present, never a selection
or an installation; `selected_model` stays null. Its profile was measured on
the owner's dictation (25 prompts, i7-9850H, 4 threads) in
`contracts/v1/profiles/stt-bench-20260929/`.

## Development interface and checks

`snapshot` emits `plebian.models.resources/v1-development` and `recommend
help-llm` emits `plebian.models.llm-sizing/v1-development`. These additive
interfaces are defined by the implementation and tests, not the frozen release
schemas. Supplied snapshots are marked `resource_source: provided`; their
contents are simulation input, not authenticated hardware evidence. Input JSON
is bounded to 4 MiB and duplicate keys/non-finite numbers are rejected.
`catalog_sha256` hashes sorted compact UTF-8 JSON with Python's default ASCII
escaping, so consumers can bind an estimate to its complete catalog metadata.

`make model-sizer-check` from the monorepo runs the sizing tests. `make check`
also validates package contents and all existing contract/provider gates.
Frozen F100-C0 fixtures, F100 U5 status, P1 acceptance, exact checkpoint licence
admission, install coordination and trusted-launcher qualification are unchanged.
The former absence-of-code gate has been replaced by functional tests; this
component does not implement the candidate replay's plan/install/lifecycle calls
or claim release-qualified D4 sizing.
