# kilix-system-monitor

A parent repository for the components that observe a Plebian-OS machine and
size work for it: live telemetry, the shared accelerator execution lease,
unprivileged hardware observation and model resource estimates.

**Status:** a parent repository pinned by Plebian-OS. Plebian-OS 0.2.3
(release candidate `v0.2.3-rc6`) holds `KILIX_SYSTEM_MONITOR_REF=9316bc26`
from 0.2.2 in `releases/0.2.3.env`, and Kilix pins the same commit: on first
use Kilix fetches it to run `plebian-model-sizer` (speech and local-model
sizing) and the `kilix-device-lease` module (local-model and transcription
providers). The telemetry daemon Kilix runs comes from the standalone
[kilix-telemetry](https://github.com/itsmygithubacct/kilix-telemetry#readme)
repository, a Kilix submodule. The default branch is `main`.

## What it contains

kilix-system-monitor is the F106 provider monorepo for Plebian OS and Kilix.
It keeps four separately versioned components beside the contract fixtures
that couple them:

- components/kilix-telemetry preserves the public kilix-telemetry package,
  import, command and history identities.
- components/kilix-device-lease is the one home of the `kilix.device-lease/v1`
  shared accelerator execution lease that the Qwen and transcription providers
  consume. It coordinates exclusive execution only; it admits no model, device
  or capacity.
- components/plebian-hardware implements the currently open, unprivileged D2
  observation surface.
- components/plebian-model-sizer provides development estimates and provisional
  document/speech model shortlists for training and inference. Its additive commands
  do not change the frozen release contracts or close F100 U5 **0/1**.

The parent is public at `refs/heads/work/0.2.1-f106`. Its telemetry history was
imported from the exact public source with a prefix-only rewrite; MIGRATION.md
records the source, rewrite tool, complete commit map, equivalence checks and
split-back procedure. The source repository and consumer pin remain unchanged;
this parent now has its own public origin.

| Component | Version | Status (`manifest.toml`) | README |
| --- | --- | --- | --- |
| `kilix-telemetry` | 0.1.2 | history-imported | [components/kilix-telemetry](components/kilix-telemetry/README.md) |
| `kilix-device-lease` | 1.0.0 | imported, awaiting interface acceptance | [components/kilix-device-lease](components/kilix-device-lease/README.md) |
| `plebian-hardware` | 0.1.0 | unprivileged D2 | [components/plebian-hardware](components/plebian-hardware/README.md) |
| `plebian-model-sizer` | 0.2.0 | development estimates, unqualified | [components/plebian-model-sizer](components/plebian-model-sizer/README.md) |

## Quick start

The checks use the release-pinned uv 0.12.5 and its managed CPython 3.12.8.
Fill the uv cache once with network access, then run the gate offline:

```sh
git clone https://github.com/itsmygithubacct/kilix-system-monitor.git
cd kilix-system-monitor
make prefetch UV=/absolute/path/to/uv-0.12.5
CANDIDATE_SCRATCH_ROOT=/absolute/scratch/root TMPDIR=/absolute/scratch/root \
  make check UV=/absolute/path/to/uv-0.12.5
```

[docs/checks.md](docs/checks.md) has the full sequence and what each gate
covers. Each component also runs from its source tree, for example
`components/plebian-model-sizer/plebian-model-sizer snapshot`.

## Usage

| Command | Effect |
| --- | --- |
| `kilix-telemetryd` / `kilix-telemetry serve` | run the single per-user sampler |
| `kilix-telemetry snapshot` / `pane` / `history` / `status` | read the newest sample, one process tree, recent samples, or daemon and ring health |
| `plebian-hardware show` | coarse, privacy-safe hardware summary |
| `plebian-hardware inventory --json` / `gpu --json` | machine inventory or GPU observation as JSON |
| `plebian-model-sizer snapshot` | current RAM, cgroup headroom, CUDA free memory and model storage |
| `plebian-model-sizer recommend help-llm\|avatar-chat\|voice\|runtime` | provisional resource shortlist for one domain |
| `make check` | every offline gate (contracts, telemetry, lease, hardware, packages, profiles, evidence, launcher readiness, model sizer) |
| `make prefetch` | fill the uv cache that `make check` reads offline |

`kilix-device-lease` has no command; it is the `kilix_device_lease` Python
module that providers import.

## How it works

```text
 /proc, /sys ──▶ kilix-telemetryd ──▶ private mmap ring ──▶ Kilix chrome, dashboards, desktops
 /proc, /sys, DRM/PCI ──▶ plebian-hardware ──▶ plebian.hardware/v1 JSON
                                      │
                                      ▼
 measured profiles ──▶ plebian-model-sizer ──▶ provisional shortlists (Kilix model and voice setup)
 model / transcription providers ──▶ kilix_device_lease ──▶ one exclusive accelerator grant
```

`kilix-telemetry` runs one sampler per user that reads `/proc` and `/sys` and
writes versioned, compressed, CRC-checked snapshots to a bounded ring; clients
read the ring instead of each walking the process table. `plebian-hardware`
reads CPU, memory, GPU, bus, firmware, power, thermal and pressure state
without privilege or network and refuses identifier-named files and unsafe
probe targets. `plebian-model-sizer` combines that observation with measured
reference profiles to estimate training and inference cost; a shortlist is an
estimate, not a reservation or a qualification. `kilix-device-lease` is a
cooperative per-user lock that serializes accelerator use across providers.

The contracts that couple these components live under `contracts/`: the
pre-freeze `p1-candidate` bundle, the frozen `v1` resource-profile bundle and
the device-lease interface. [docs/contracts.md](docs/contracts.md) explains
each one and what has and has not been accepted.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/contracts.md](docs/contracts.md) | contract state: candidate, frozen v1, device lease, F120 and trusted-launcher records |
| [docs/checks.md](docs/checks.md) | prefetch, the offline developer gate and what it exercises |
| [docs/boundaries.md](docs/boundaries.md) | what is implemented, constructed but non-qualifying, and not claimed |
| [MIGRATION.md](MIGRATION.md) | the kilix-telemetry history import and the kilix-device-lease import |
| [integration/README.md](integration/README.md) | the F120 development registration and trusted-launcher readiness |
| [integration/TRUSTED-LAUNCHER-CONSUMER.md](integration/TRUSTED-LAUNCHER-CONSUMER.md) | the trusted-launcher consumer contract |
| [LICENSES/README.md](LICENSES/README.md) | licensing layout |

## Related repositories

- [Kilix](https://github.com/itsmygithubacct/kilix#readme) fetches this
  repository at its pinned commit for the model sizer and the device lease, and
  carries [kilix-telemetry](https://github.com/itsmygithubacct/kilix-telemetry#readme)
  as a submodule.
- [Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme) pins this
  repository as a release root.
- [kilix-voice](https://github.com/itsmygithubacct/kilix-voice#readme) and
  [kilix-help-llm](https://github.com/itsmygithubacct/kilix-help-llm#readme):
  the speech and help-model workloads the sizer estimates.
- [kilix-desktop-sdk](https://github.com/itsmygithubacct/kilix-desktop-sdk#readme)
  and [kilix-media-sdk](https://github.com/itsmygithubacct/kilix-media-sdk#readme):
  the other parent repositories Plebian-OS pins.

## Part of Plebian-OS

kilix-system-monitor is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
Plebian-OS pins it in `releases/0.2.3.env`, and Kilix fetches the same commit
on first use for model sizing and the device lease. The map of every
repository and how it ships is on the
[components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## License

MIT; see [LICENSE](LICENSE). The imported kilix-telemetry component keeps its
original MIT licence; see [LICENSES/README.md](LICENSES/README.md).
