# Installed F106 P1 contract candidate

This directory is a pre-freeze product-side test surface, not a frozen
contract and not release or qualification evidence. It contains only the
schema, fixture, invocation and replay assets needed for atomic provider and
consumer checks. Research-only migration drafts, probe matrices, handoff notes
and release-wording drafts are deliberately excluded from this repository.

This successor contains **12/12 schemas**, **23/23 valid fixtures**, **18/18
invalid fixture mutations** and **2/2 replay binaries**. It dispositions the
F107-B R2 return's **4/4 findings**: all **9/9 invocations** use frozen absolute
installed paths; install binds the SHA-256 of the exact canonical reviewed plan
bytes; status and cancellation are **2/2 explicit lifecycle calls** with the
same plan identity in resume state; and every F107-rendered population has a
schema `maxItems`. The locally generated `CANDIDATE-SHA256SUMS` binds all
**59/59 payload members**, including this status notice. This is a successor
candidate, not an R2 signature or a freeze.

The additive D-side privacy and digest-specific model-weight licence designs
remain present. The checkpoint decision explicitly forbids wildcard clearance,
and separate mutations reject both an unlisted `iic/speech_campplus_*` sibling
and any attempt to set a wildcard grant. These records are not frozen.

`plebian.models.license-evidence-set/v1` adds exact citation IDs for 38 unique
F104 Qwen/transcription/VAD artifact digests and the three selected F105 Ollama
manifest digests. Each set is bound to the reviewed authority-document hash, a
canonical artifact-set digest and `wildcard_inheritance=false`; neither code
licence is used as weight evidence. These are licence citations, not package
manifests or profile selections. Notice bytes and staged paths remain empty and
transfer is false. The delivery block is bound to F120-C11, which proves frozen
F120 v1 can accept a staged payload without conveying its declared licence or
notice text and cannot express heterogeneous per-payload obligations. The v1
evidence-set delivery block is therefore permanently admission-only rather
than a future unioned conveyance record. It stays pending until the release
authority selects and enforces either strict one-obligation-unit components or
new per-artifact contract identities in a successor shape.

The privacy proposal also requires optional command probes to resolve inside
the fixed system path and refuses group- or world-writable executables.

From the monorepo root, exercise the current functional developer check with
the release-pinned uv 0.12.5 executable:

    make contracts-check UV=/absolute/path/to/release-pinned-uv-0.12.5

The output names **12/12 candidate schemas**, **23/23 valid fixtures**, **18/18
rejected invalid mutations**, all replayed success/failure paths, **59/59
candidate payload files**, canonical JSON, the partial developer startup controls and
no qualification claim. It is not P1 freeze or release authority.
The current repository shell path partially isolates the outer validator, but
it does not implement the normative retained-descriptor/bootstrap/canonical-
result contract, and both replay helpers still select site-enabled Python
through `/usr/bin/env` and inherited `PATH`. P1 remains blocked until the shared
trusted launcher and Track D replay-interpreter overlay pass their full causal
mutation packet.
