# RC5 reference memory observations

These development profiles cover the selected catalog's three original YOLOX
ONNX graphs and native EnCodec 24/48 kHz bundles on x86_64 CPU. Each measurement
used a separate process, two inference threads and successful inference calls.
The raw evidence and reproducible harness are retained privately and identified
by SHA-256 in each document. Linux `ru_maxrss` measures one process's resident
high-water mark; it is not a cgroup aggregate or a no-escape execution proof.
The sizer applies a 25% reference margin and a 256 MiB RAM reserve. Different
inputs, runtimes, threads or concurrent models can require more memory.

Both Bonsai image entries now use successful receipt-backed local generations
at 512×512, four steps, seed 17, with the frozen Python 3.12.8 CUDA runtime.
The measured Quadro RTX 3000 has 6 GiB VRAM and compute capability 7.5. Its
CPU float32 VAE mode applies below 8 GiB; the profiles are restricted to that
mode with one unmasked GPU at physical index 0. Multiple cards or CUDA visibility/
order overrides produce unknown because their CUDA device mapping is unverified.
The bound runtime explicitly selects `cuda:0` instead of the backend's optional
device environment override. Larger cards use a different VAE mode and receive an unknown
recommendation until measured. Larger resolutions are also unmeasured.

RAM uses the measured process RSS plus the complete declared model population
as a conservative allowance for sealed memory copies outside process RSS.
It is not an aggregate cgroup measurement. VRAM uses the larger of allocator
peaks and a 0.4-second sampled process peak; sampling remains a lower bound.
The same 25% margin and 256 MiB reserves apply. At nominal 6 GiB capacity the
ternary candidate does not fit with margin and binary does, provided enough
RAM and free VRAM remain. Other running applications can prevent either fit.
The profile binds the exact runtime source and lock plus private observations
and generated preview digests. Local installed runtime identity, CUDA support
on other hardware, speed, sustained operation and release quality remain
unverified by the planner.

These are `plebian.models.runtime-profile/v1-development` documents. They do
not change the frozen provider intake protocol, grant qualification, establish
installed runtime identity or authorize acquisition. A changed catalog manifest
or a different architecture suppresses the profile's memory recommendation.
Candidates are independent; their fit does not imply a co-resident budget.
