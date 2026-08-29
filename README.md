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

integration/trusted-launcher-consumer-requirements.json records the exact
Track D TD-P1 and TD-HW consumer requirements without defining or copying a
launcher profile. OD-13 assigns the ID-04 implementation to reviewer2 and
OD-14 assigns the non-forking profile/child-table interface to Track H, but the
shared result packet and independently reviewed interface have not returned.
The readiness check therefore validates the local child surfaces and rejects
premature adoption while consuming none of the required returned identities.

## Checks

Use the release-pinned uv 0.12.5 for a functional developer check:

    /absolute/path/to/release-pinned-uv-0.12.5 sync --locked --offline \
      --no-install-project --managed-python --no-python-downloads --python 3.12.8
    make check UV=/absolute/path/to/release-pinned-uv-0.12.5

The locked aggregate environment includes the exact uv-build 0.12.5 backend
needed by both component sdists, so the package gate can stay offline after the
normal locked environment sync instead of depending on an unrelated cache hit.

The current `tools/validate_candidate` shell path has useful partial isolation:
it refuses an unpinned uv or Python, checks the candidate file set before
Python, then starts the semantic validator with `-I -S -B` from a private empty
directory. It is not release authority. The normative startup contract also
requires an independently accepted launcher/bootstrap, a retained no-follow
subject descriptor, exact first-process envelope checks, a canonical result
channel and complete causal mutations. The self-test's two replay helpers still
select Python through `/usr/bin/env` and inherited `PATH`.

The aggregate developer check exercises contract integrity and negative
fixtures, the imported telemetry suite, the hardware unit boundaries, live
inventory/GPU schema and privacy rules, wheel/sdist contents, trusted-launcher
consumer readiness, and the intentional model-sizer block. Hardware checks use
no network and no privilege.

## Current boundaries

Implemented now: history-preserving local parent creation, aggregate contract
tooling, exact F120 development registration, and unprivileged D2 probes for
CPU/RAM/cgroup/topology/cache/frequency/ISA, DRM/PCI GPUs, bounded backend
commands, device-bound NVIDIA driver/VRAM evidence, buses, anonymous network
links, firmware/IOMMU state, power/battery, thermal/fans, PSI and
virtualization. Identifier-named files and final-component symlinks are refused
at the common read boundary. Optional command probes refuse resolved targets
outside the fixed system path and group- or world-writable executables.

Implemented as non-qualifying D2 construction: a private 0600 canonical snapshot
cache, symlink-refusing descriptor-relative replacement, redacted snapshot diff
and cache-aware doctor primitives. Not implemented or claimed: the shared
trusted-launcher profiles, P1 freeze, privileged DMI, SMART/NVMe,
device-bound ROCm/Vulkan/OpenCL success, D3 telemetry vNext,
D4 sizing, D5 consumers, or D6 hardware qualification. H3 physical inventory,
H3 model performance, and AMD/ROCm fit/performance/support remain unqualified
for 0.2.1.

F120-C11 also remains open: frozen F120 v1 does not prove that a staged model
payload conveys its required licence/notice bytes or represent different
obligations per payload in one component. D's licence-evidence fixtures remain
transfer-ineligible until the release authority's strict-v1/new-identity choice
is implemented; the existing S120 source/code-provider path remains usable.

Track H causally proved that the old F120 launch shape can return forged zero
and PASS output before its real validator or tests run. Track D has the same
shape at its staged console/test and replay boundaries. No `make check` result
from this repository is qualification or P1-freeze evidence until Track H's
reference trusted launcher is published and the Track D candidate-validator,
replay-helper and authority-bearing staged-probe profiles pass the normative
mutation packet.
