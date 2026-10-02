# RC5 reference memory observations

These development profiles cover the selected catalog's three original YOLOX
ONNX graphs and native EnCodec 24/48 kHz bundles on x86_64 CPU. Each measurement
used a separate process, two inference threads and successful inference calls.
The raw evidence and reproducible harness are retained privately and identified
by SHA-256 in each document. Linux `ru_maxrss` measures one process's resident
high-water mark; it is not a cgroup aggregate or a no-escape execution proof.
The sizer applies a 25% reference margin and a 256 MiB RAM reserve. Different
inputs, runtimes, threads or concurrent models can require more memory.

The Bonsai ternary image entry uses the selected, digest-verified upstream
model card's approximate 6.8 GiB peak HBM at 1024×1024 on RTX 3080. Its RAM
requirement remains unknown. This entry cannot establish a fit, even on a
large GPU. The binary image alternate has no profile and cannot inherit the
ternary model's figures. CUDA kernel compatibility remains unverified.

These are `plebian.models.runtime-profile/v1-development` documents. They do
not change the frozen provider intake protocol, grant qualification, establish
installed runtime identity or authorize acquisition. A changed catalog manifest
or a different architecture suppresses the profile's memory recommendation.
Candidates are independent; their fit does not imply a co-resident budget.
