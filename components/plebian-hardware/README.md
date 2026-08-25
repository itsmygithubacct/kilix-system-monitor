# plebian-hardware

plebian-hardware emits a coarse, privacy-safe local hardware observation for
Plebian OS. The implemented 0.2.1 surface is deliberately limited to the three
pre-freeze F106 consumer calls:

    plebian-hardware show
    plebian-hardware inventory --json
    plebian-hardware gpu --json

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

The fixed privileged DMI helper, private cache/diff lifecycle, model-store
capacity, SMART/NVMe inspection and device-bound ROCm/Vulkan/OpenCL probes are
not implemented by this component yet. Missing evidence remains null with an
explicit unknown rather than becoming zero or a guessed pass.
