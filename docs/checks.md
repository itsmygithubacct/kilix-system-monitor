# kilix-system-monitor checks

How to prefetch the locked environments and run the offline developer gate, and what it exercises. Moved from the [README](../README.md); the text is unchanged.

## Checks

`make check` runs offline. With network access, fill the uv cache it reads
once, using the release-pinned uv 0.12.5 and its managed CPython 3.12.8:

    make prefetch UV=/absolute/path/to/release-pinned-uv-0.12.5

prefetch syncs the root and component environments from PyPI, and uv verifies
every download against the hashes in each `uv.lock`. It then runs the package
gate online once, so the root's isolated setuptools build fetches the backend
pinned by hash in `tools/build-constraints.txt`. A fresh clone with an empty
cache needs nothing more. Then, offline, for a functional developer check:

    /absolute/path/to/release-pinned-uv-0.12.5 sync --locked --offline \
      --no-install-project --managed-python --no-python-downloads --python 3.12.8
    CANDIDATE_SCRATCH_ROOT=/absolute/scratch/root TMPDIR=/absolute/scratch/root \
      make check UV=/absolute/path/to/release-pinned-uv-0.12.5

The locked aggregate environment includes the exact uv-build 0.12.5 backend
needed by the component sdists, and the root's setuptools backend is pinned by
hash, so the package gate stays offline instead of depending on an unrelated
cache hit.
The candidate gate requires `CANDIDATE_SCRATCH_ROOT` to name an existing,
canonical scratch directory other than `/`, and refuses a `TMPDIR` that does
not resolve beneath it.

The current `tools/validate_candidate` shell path has useful partial isolation:
it refuses an unpinned uv or Python, checks the candidate file set before
Python, then starts the semantic validator with `-I -S -B` from a private empty
directory. It is not release authority. The normative startup contract also
requires an independently accepted launcher/bootstrap, a retained no-follow
subject descriptor, exact first-process envelope checks, a canonical result
channel and complete causal mutations. The self-test's two replay helpers still
select Python through `/usr/bin/env` and inherited `PATH`.

The aggregate developer check exercises contract integrity and negative
fixtures, the imported telemetry suite, the shared accelerator lease's
real-process controls and interface-document binding, the hardware unit boundaries, live
inventory/GPU schema and privacy rules, wheel/sdist contents, trusted-launcher
consumer readiness, the unqualified provider-profile intake validator, and
the model-sizer resource accounting and refusal cases. Hardware checks use no network and no
privilege. `tools/measure/profile.py` records **1/1** provider-owned evidence
record against **3/3** exact input-byte identities, promotes **0/9** provider
measurement fields, and keeps qualification acceptance at **0/1**. Provider
execution is outside this validator.

The public H2 capacity observation under `evidence/` carries the exact
`/proc/loadavg` values at both ends of **4/4** measurement windows. It records
the causal Debian `nvidia-smi` repair and one exact committed-source observation
without naming the host. The validator keeps installer-media evidence at
**0/1**, model-performance evidence at **0/1**, and qualification at **0/1**.
