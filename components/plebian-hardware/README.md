# plebian-hardware

plebian-hardware emits a coarse, privacy-safe local hardware observation for
Plebian OS. The three pre-freeze F106 consumer calls are:

    plebian-hardware show
    plebian-hardware inventory --json
    plebian-hardware gpu --json

The local operator surface now also includes:

    plebian-hardware refresh
    plebian-hardware diff SNAPSHOT
    plebian-hardware doctor

These three commands are not additions to the joint F107 invocation candidate.
`refresh` atomically persists canonical redacted JSON under an explicit private
XDG state root. Every path component is opened without following symlinks; the
component directory must be owned by the caller with mode 0700 and the cache is
a caller-owned regular file with mode 0600. `diff` accepts only a private,
bounded regular snapshot and compares an allowlisted capability projection, so
timestamps, snapshot IDs, transient bus/interface names and forbidden identity
fields cannot enter its output. `doctor` reports the cache state without its
path and repeats that startup authority is blocked.

The JSON commands emit one plebian.cli.response/v1 envelope containing a
plebian.hardware/v1 observation. The schema and invocation bytes remain a
pre-freeze candidate under ../../contracts/p1-candidate until every named P1
signatory accepts identical bytes.

The collector uses bounded reads from procfs, sysfs and cgroup v2. Optional
local commands are found only in a fixed system path and run with fixed argv,
a clean locale, null stdin, a five-second deadline and a 64 KiB output limit.
Resolved executables must remain inside that path and must not be group- or
world-writable. It never uses the network or elevates privilege.

The additive plebian.hardware.privacy/v1 candidate fixes those boundaries as
data rather than leaving them as review conventions. The common file reader
refuses identifier-named inputs and final-component symlinks before opening
them. The NVIDIA query is bound to one transient PCI address and asks only for
that address, the driver version and total memory; the address is not emitted.

Hostname, user name, machine ID, system UUID, serial numbers, asset tags, MAC
addresses and IP addresses are never collected. Interface names, DRM connector
names, CPU-list text and PCI bus addresses are transient lookup inputs and are
not emitted. Even after that redaction, detailed hardware is
fingerprinting-grade local data and is not telemetry or qualification evidence.

The private cache and redacted diff primitives are implemented but remain
non-qualifying. The fixed privileged DMI helper, model-store capacity,
SMART/NVMe inspection and device-bound ROCm/Vulkan/OpenCL probes are not
implemented by this component yet. Missing evidence remains null with an
explicit unknown rather than becoming zero or a guessed pass.

The current Python console script, component-root `uv run` tests and P1 replay
helper are exposed to Python startup hooks when run against a staged/provider
tree. They are operator/developer functionality only, not release authority.
Qualification requires the shared external trusted launcher, pinned `-I -S -B`
interpreter, descriptor-bound closure, canonical result channel and Track D's
replay-interpreter profile before any staged probe result is accepted.
