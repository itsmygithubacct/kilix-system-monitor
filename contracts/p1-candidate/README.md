# Installed F106 P1 contract candidate

This directory is a pre-freeze product-side test surface, not a frozen
contract and not release or qualification evidence. It contains only the
schema, fixture, invocation and replay assets needed for atomic provider and
consumer checks. Research-only migration drafts, probe matrices, handoff notes
and release-wording drafts are deliberately excluded from this repository.

The original eight schema files, invocation contract, 26 fixture files and two
replay binaries retain their exact R2 bytes. Two additive D-side schema designs,
three valid fixtures and four invalid mutations cover the default privacy
boundary and digest-specific checkpoint-licence admission. The checkpoint
decision explicitly forbids wildcard clearance, and separate mutations reject
both an unlisted `iic/speech_campplus_*` sibling and any attempt to set a
wildcard grant. They do not alter the R2 command surface and are not frozen. The
locally generated
CANDIDATE-SHA256SUMS binds the complete installed tree, including its validator
and this status notice; it is intentionally distinct from the 46-file
research-side R2 bundle manifest.

The privacy proposal also requires optional command probes to resolve inside
the fixed system path and refuses group- or world-writable executables.

Verify it with the root locked environment:

    uv run --locked python contracts/p1-candidate/tools/validate.py --self-test

Expected output names ten candidate schemas, nineteen valid fixtures, thirteen
rejected invalid mutations, replayed success/failure paths, 49 installed-file
hashes, canonical JSON, and no qualification claim.
