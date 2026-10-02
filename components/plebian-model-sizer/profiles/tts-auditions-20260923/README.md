# Short-workload TTS audition profiles

These five unqualified profiles add distinct `audition-*` identities. They do
not replace the frozen RES02 profiles or qualify any production provider.
Each document binds its source benchmark result by SHA-256; the packaged copy
is also hash-checked and tested against this directory.

The fixture is “Good morning. This is a short speech test for the Kilix desktop.”
One fresh process runs one first and three warm passes, with local weights and
no playback or cache flush. First-result time includes model/import loading.
Real-time factor is the median of per-pass warm synthesis/audio ratios.

RAM for resident engines is the greater of sampled process-tree RSS and parent
high-water RSS. For tiny eSpeak/MBROLA subprocess engines, the conservative sum
of parent/child high-water marks is retained. Forked probe high-water marks are
not added to Qwen's parent RSS: inherited values would double-count the model.
VRAM is the sampled per-process NVIDIA reading including context overhead,
not just PyTorch allocations. The profiles use a 20% memory margin; the shared
planner separately reserves 256 MiB RAM/VRAM. Longer text and concurrent work
can exceed these short-workload observations.

Piper uses its direct engine API, not a full provider/daemon/playback path.
Qwen uses four threads, Ryan/English/seed 0, float32/SDPA on CPU and
bfloat16/FlashAttention 2 on CUDA. Artifact identity and runtime identity remain
unverified by sizing. Download and temporary-space requirements stay unknown.

CPU reference: x86_64 i7-9850H. GPU reference: i7-10700F and RTX 3070 8 GiB,
with another user model remaining resident. This is resource-planning evidence,
not a controlled hardware comparison, a quality ranking or a maximum-memory
guarantee. Existing runtime and licence checks remain authoritative at use.
