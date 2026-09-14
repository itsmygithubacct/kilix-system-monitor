# Frozen F106 resource-profile bundle v1

Status: **frozen bytes, builder-landed, acceptance pending.** This directory is
immutable. A change to any byte needs a successor bundle, never an edit.
`tools/validate_frozen_contracts.py` pins the digest of `FROZEN-SHA256SUMS`, and
the bundle digest is the SHA-256 of that file.

The bytes were landed by the builder as a local commit. The acceptances that
freeze a contract here are recorded outside this repository and are not
implied by it:

- non-author adversarial review of the bundle;
- cross-family acceptance of the exact landed commit;
- identical-byte signoff by the named consumers;
- the owner's final signoff.

Until those records exist, nothing in this directory is P1-freeze evidence.

## What it carries

| Member | Content |
| --- | --- |
| `plebian.models.profiles-v1.schema.json` | the `plebian.models.profiles/v1` profile catalog schema, byte-identical to the candidate that consumers acknowledged |
| `plebian.hardware-v1.schema.json` | the `plebian.hardware/v1` schema, byte-identical to the acknowledged candidate |
| `profiles/res02-measured/` | 7 measured provider profile catalogs, byte-identical to the checked RES-02 measurement output |
| `fixtures/profiles/valid/` | 3 valid catalogs: the candidate's unqualified estimate, and copies of one measured CPU row and one measured CUDA row |
| `fixtures/profiles/invalid/` | 9 refused catalogs, each one declared mutation of a valid fixture |
| `fixtures/profiles/REFUSALS.json` | for each refused catalog: its base, the mutation, and the refusal it must produce |
| `FROZEN-SHA256SUMS` | the SHA-256 of every other member |

The profile item schema requires a `backend` from `cpu`, `cuda`, `oneapi`,
`opencl`, `rocm` and `vulkan`, and requirements that include
`ram_peak_bytes` and `vram_peak_bytes`, in bytes. Every object is closed. A
document without that resource structure is refused by the schema, and the
validator asserts that structure directly, so a launcher command profile or
any other profile-shaped document cannot stand in for this bundle.

## The measured rows

Each measured catalog holds one row with `confidence: measured`, a
repository-relative measurement command, a fixture name, `measured_at`, a
reference hardware class, and `raw_evidence_sha256` equal to the aggregate of
the measurement window they came from. Peak RAM is the scope's peak memory,
including page cache charged to it, so RAM figures are conservative ceilings.
Peak VRAM is per-process accelerator memory: 0 for CPU rows, and the measured
maximum for CUDA rows.

Every row is `qualification: unqualified` and every catalog is
`qualification_eligible: false`. A measurement is not a qualification.
Qualification belongs to F106 and its hardware policy. `license_decision_id`
is null in every row. Two rows carry a null `content_sha256`, because their
artifacts are distribution packages rather than downloaded model files.

The rows name a hardware class, never a host.

## Validation

    make contracts-check UV=/absolute/path/to/release-pinned-uv-0.12.5

The frozen-bundle step checks:

- the manifest against the complete tree, and its pinned digest;
- canonical JSON for every JSON member;
- both schemas against the acknowledged digests;
- the structural resource assertion;
- every valid fixture accepted and every refused fixture rejected for its
  declared reason;
- the declared mutation reproduces each refused fixture's exact bytes;
- every measured row against its pinned bytes and the measured-row policy;
- planted controls, each of which must fire.
