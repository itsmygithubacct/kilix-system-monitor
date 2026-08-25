# Installed F106 P1 contract candidate

This directory is a pre-freeze product-side test surface, not a frozen
contract and not release or qualification evidence. It contains only the
schema, fixture, invocation and replay assets needed for atomic provider and
consumer checks. Research-only migration drafts, probe matrices, handoff notes
and release-wording drafts are deliberately excluded from this repository.

The original eight schema files, invocation contract, 26 fixture files and two
replay binaries retain their exact R2 bytes. Three additive D-side schema
designs, five valid fixtures and eight invalid mutations cover the default
privacy boundary and digest-specific model-weight licence admission. The
checkpoint decision explicitly forbids wildcard clearance, and separate
mutations reject both an unlisted `iic/speech_campplus_*` sibling and any
attempt to set a wildcard grant. They do not alter the R2 command surface and
are not frozen. The locally generated
CANDIDATE-SHA256SUMS binds the complete installed tree, including its validator
and this status notice; it is intentionally distinct from the 46-file
research-side R2 bundle manifest.

`plebian.models.license-evidence-set/v1` adds exact citation IDs for 38 unique
F104 Qwen/transcription/VAD artifact digests and the three selected F105 Ollama
manifest digests. Each set is bound to the reviewed authority-document hash, a
canonical artifact-set digest and `wildcard_inheritance=false`; neither code
licence is used as weight evidence. These are licence citations, not package
manifests or profile selections. Notice bytes and staged paths remain empty and
transfer is false. The delivery block is bound to F120-C11, which proves frozen
F120 v1 can accept a staged payload without conveying its declared licence or
notice text and cannot express heterogeneous per-payload obligations. It stays
pending until the release authority selects and enforces either strict
one-obligation-unit components or new per-artifact contract identities.

The privacy proposal also requires optional command probes to resolve inside
the fixed system path and refuses group- or world-writable executables.

Verify it with the root locked environment:

    uv run --locked python contracts/p1-candidate/tools/validate.py --self-test

Expected output names eleven candidate schemas, twenty-one valid fixtures,
seventeen rejected invalid mutations, replayed success/failure paths, 56 installed-file
hashes, canonical JSON, and no qualification claim.
