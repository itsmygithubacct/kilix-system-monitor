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

Three additive design contracts now make privacy, checkpoint-licence admission
and exact F104/F105 weight-evidence citation machine-checkable without changing
the original R2 invocation or response fixtures. They remain D-side proposals
until the same P1 signoff occurs. The weight evidence is explicitly
digest-enumerated with no wildcard inheritance; it authorizes no selection,
fit, staging or transfer.

integration/f120-registration.json is the exact pre-repository scaffold from
the published F120 handoff. Its zero commit and metadata sentinels are retained
until the parent has a reviewed public install surface. An F120 development
manifest is therefore expected to be dirty/unresolved; qualification and
staging must refuse.

## Checks

Use the release-pinned uv 0.12.5:

    /absolute/path/to/release-pinned-uv-0.12.5 sync --locked --offline \
      --no-install-project --managed-python --no-python-downloads --python 3.12.8
    make check UV=/absolute/path/to/release-pinned-uv-0.12.5

The locked aggregate environment includes the exact uv-build 0.12.5 backend
needed by both component sdists, so the package gate can stay offline after the
normal locked environment sync instead of depending on an unrelated cache hit.

The P1 candidate is checked through `tools/validate_candidate`, an external
shell boundary that refuses an unpinned uv or Python, closes the candidate's
complete file set before Python starts, then runs the external semantic
validator with `-I -S -B` from a private empty directory and an allowlisted
environment. Candidate-root `sitecustomize.py` and `usercustomize.py` files are
therefore rejected before they can execute.

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
at the common read boundary. Optional command probes refuse resolved targets
outside the fixed system path and group- or world-writable executables.

Not implemented or claimed: P1 freeze, privileged DMI, SMART/NVMe and private
cache lifecycle, device-bound ROCm/Vulkan/OpenCL success, D3 telemetry vNext,
D4 sizing, D5 consumers, or D6 hardware qualification. H3 physical inventory,
H3 model performance, and AMD/ROCm fit/performance/support remain unqualified
for 0.2.1.

F120-C11 also remains open: frozen F120 v1 does not prove that a staged model
payload conveys its required licence/notice bytes or represent different
obligations per payload in one component. D's licence-evidence fixtures remain
transfer-ineligible until the release authority's strict-v1/new-identity choice
is implemented; the existing S120 source/code-provider path remains usable.
