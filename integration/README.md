# F120 development registration

f120-registration.json is copied byte-for-byte from the F106 fixture in the
published F120 handoff commit
429504c2bbc7b330e40cec97eec226e59c194e38 on
work/0.2.1-f120-closure.

It intentionally retains forty-zero expected/requested commits,
local-only/unpublished disposition, unresolved component metadata and the
canonical future HTTPS URL. The workspace root is the handoff's portable
/workspace convention. Until the parent lands a reviewed public install
surface, resolving this registration without qualification must emit three
dirty/unresolved development records. Qualification and staging must refuse.

From the exact F120 tree, enter the non-packaged tool directory first; selecting
its uv project from another working directory does not put `kilix_f120` on
Python's import path:

    cd tools/closure
    make --no-print-directory check
    uv run --locked python -m kilix_f120 resolve \
      PATH_TO_THIS_FILE OUTPUT
    uv run --locked python -m kilix_f120 validate \
      OUTPUT --allow-development-state

Do not replace the sentinels merely because a local checkout exists. The F120
qualification procedure requires a landed repository/component, reviewed
public install surface, exact commit, notice and licence bytes, toolchain,
artifacts, commands and required tests.

## Model licence/notice conveyance boundary

Track H's F120-C11 finding is bound in the D-side licence-evidence candidate by
SHA-256 `f9731fffeb8a240ca05115375f36c4af1df51a8d08b0a9252d728a4b7b5e3c53`.
Frozen v1 carries component licence and source-notice identities, but a valid
stage can omit their bytes and an artifact cannot name its notice. It also
cannot distinguish heterogeneous per-payload obligations inside one component.

Consequently no staged model payload is transfer- or P9-eligible merely because
its component and `licenses_sha256` resolve. Until the release authority
dispositions PR-H-007, model integrations must retain empty staged notice paths
and refuse transfer. The two admissible outcomes are a strict v1 profile where
each component is one uniform obligation unit and every text is a matched
same-component staged `notice` artifact, or new contract identities with
explicit per-artifact references. This does not retract S120 for D's source and
code-provider work.

## Trusted-launcher consumer readiness

`TRUSTED-LAUNCHER-CONSUMER.md` is the normative human-readable index.
`trusted-launcher-consumer-requirements.json` is Track D's final capability and
child-table input to Track H, and `trusted-launcher-consumer-campaign.json`
defines the complete consumer case/evidence population. None is a Track H
launch profile, result packet, frozen interface or adoption lock. They do not
invent profile syntax or copy the F120-specific launcher.

Run the developer-only static readiness check from the repository root:

    make launcher-consumer-readiness UV=/absolute/path/to/release-pinned-uv-0.12.5

The check accounts for **20/20** interface capabilities, **2/2** consumer
requirements, **8/8** intentional child specifications, **7/7** P1 invocation
vectors, **316/316** case-family target definitions and **632/632** required
case-family target-export rows across **2/2** independent exports. Each row
must freeze and completely account for its inner mutation variants. The check
applies omission and premature-adoption controls proving that assignments
cannot be promoted, populations cannot be shrunk and D4 cannot enter a launcher
profile. A passing readiness check still contains **0/632** result rows, accepts
**0/20** interface mappings and **0/2** upstream independent exports, consumes
**0/8** returned launcher identities and makes no qualification claim.
