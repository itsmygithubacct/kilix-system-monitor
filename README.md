# kilix-system-monitor

kilix-system-monitor is the F106 provider monorepo for Plebian OS and Kilix.
It keeps three separately versioned components beside the contract fixtures
that couple them:

- components/kilix-telemetry preserves the public kilix-telemetry package,
  import, command and history identities.
- components/plebian-hardware implements the currently open, unprivileged D2
  observation surface.
- components/plebian-model-sizer is an intentionally non-executable skeleton.
  Fit, recommendation and planning code is blocked until F100 passes U5 and
  F100-C0 freezes real reserve policy.

The parent is local and unpublished. Its telemetry history was imported from
the exact public source with a prefix-only rewrite; MIGRATION.md records the
source, rewrite tool, complete commit map, equivalence checks and split-back
procedure. No source repository, consumer pin or public remote was changed.

## Contract state

contracts/p1-candidate contains the pre-freeze schemas, redacted fixtures,
invocation contract and replay binaries. Those bytes are installed for coupled
testing, but they are not P1-frozen and they make no hardware, backend, model,
fit or performance qualification claim. They move to a frozen root schema
surface only after all named consumers sign identical bytes.

Two additive design contracts now make privacy and checkpoint-licence admission
machine-checkable without changing the original R2 invocation or response
fixtures. They remain D-side proposals until the same P1 signoff occurs.

integration/f120-registration.json is the exact pre-repository scaffold from
the published F120 handoff. Its zero commit and metadata sentinels are retained
until the parent has a reviewed public install surface. An F120 development
manifest is therefore expected to be dirty/unresolved; qualification and
staging must refuse.

## Checks

Use the release-pinned uv 0.12.5:

    make check

The locked aggregate environment includes the exact uv-build 0.12.5 backend
needed by both component sdists, so the package gate can stay offline after the
normal locked environment sync instead of depending on an unrelated cache hit.

The aggregate check verifies contract integrity and negative fixtures, the
imported telemetry suite, the hardware unit boundaries, live inventory/GPU
schema and privacy rules, wheel/sdist contents, and the intentional model-sizer
block. Hardware checks use no network and no privilege.

## Current boundaries

Implemented now: history-preserving local parent creation, aggregate contract
tooling, exact F120 development registration, and unprivileged D2 probes for
CPU/RAM/cgroup/topology/cache/frequency/ISA, DRM/PCI GPUs, bounded backend
commands, device-bound NVIDIA driver/VRAM evidence, buses, anonymous network
links, firmware/IOMMU state, power/battery, thermal/fans, PSI and
virtualization. Identifier-named files and final-component symlinks are refused
at the common read boundary.

Not implemented or claimed: P1 freeze, privileged DMI, SMART/NVMe and private
cache lifecycle, device-bound ROCm/Vulkan/OpenCL success, D3 telemetry vNext,
D4 sizing, D5 consumers, or D6 hardware qualification. H3 physical inventory,
H3 model performance, and AMD/ROCm fit/performance/support remain unqualified
for 0.2.1.
