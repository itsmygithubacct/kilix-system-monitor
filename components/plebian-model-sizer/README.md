# plebian-model-sizer

Local resource estimates for choosing small document models. Version 0.1.0
supports the `kilix-help-llm` candidate catalog, separate LoRA training and
inference budgets, and both question answering and topic ranking.

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
| `--quant q4/q8/f16` | Answer inference weight estimate; default q4; training/ranking weights are unchanged |
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
