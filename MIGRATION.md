# kilix-telemetry history import

Status: public work ref and rollback-safe. The source repository, consumer pin,
release tag and source-publication settings remain unchanged; this parent now
has its own public origin.

## Frozen source and selected refs

The selected source was the anonymously reachable public repository
https://github.com/itsmygithubacct/kilix-telemetry.git at exact main commit
3affc0cc4b9a80517c452470a01e2103d29e9dbf. The selected history contains 15
commits, no merge commits, no tags and no gitlinks. Only refs/heads/main was
rewritten.

The ordinary telemetry checkout and its unpublished work/0.2.0-c001 branch
were not rewritten or imported. The existing Kilix consumer pin
6df4525805f875ade88dbf9d0b2da95aa847da1a maps to
7b04cf26e89262c2180629784c78ae0152be5403 in the parent. Moving that consumer
pin belongs to its owning track and was not attempted here.

## Pinned rewrite tool and transform

The import used the official git-filter-repo v2.47.0 tag from
https://github.com/newren/git-filter-repo.git, peeled to commit
6f79afc8c90c592a3052e6cc53c2ca8907515bca. The executed git-filter-repo script
had SHA-256
67447413e273fc76809289111748870b6f6072f08b17efe94863a92d810b7d94;
its COPYING file had SHA-256
6447a28bf91a61a316accc25bc8fcdb8cba667bb226cab571107e8110ea2d411.

An isolated mirror of the exact public source was transformed in place with:

    python3 PATH_TO_PINNED_GIT_FILTER_REPO --force \
      --to-subdirectory-filter components/kilix-telemetry \
      --refs refs/heads/main

The ordinary checkout was never a rewrite target. The rewritten mirror was
then cloned as this new parent, and its temporary source remote was removed.
The parent was subsequently published at `refs/heads/work/0.2.1-f106`; the
source repository was not mutated.

## Complete commit map

The machine-readable copy at integration/kilix-telemetry-commit-map has
SHA-256 c896abb6d2b326a3a776f76ad168eb5916a712ac6c109acc9ebb7f39ded7961b.

    0372699c2e3991b706ef3c81aa746c988247e275 -> 52f0fc419ab643afac6018265c0481f029a73efc
    2544e73510d04f8b571956eb4508968c81a44e56 -> 2fad503b1d1858c23480a0f5e188c44bd561e8b9
    39742cac4e419c7286da623a35d4d6a899f7caeb -> 3a2bde40bda8c91eae3052daaa98a4ee50071e10
    3affc0cc4b9a80517c452470a01e2103d29e9dbf -> 057167558ac8d7f26593e194a08f24254b47ca02
    51f36f077aa998565a20e019b109d64823de73b7 -> 4dfd2f83a86181ef1055309b35b45c7d275e0d58
    687147c9283569587a09660c69bb730138d5ee61 -> 00a5b59aacd3a1ffdcca28d4baa4ecdf2193d8fd
    6df4525805f875ade88dbf9d0b2da95aa847da1a -> 7b04cf26e89262c2180629784c78ae0152be5403
    74ce0e6537e0f943347b1c8d2218785574f8aa15 -> ed7a60fc7073596934fcf260994410f9049643c5
    8ae6e7a85ecd71e64ec78a8872ee45a987d5a9a5 -> fd7f38a0792be9f7f40967d55a0cca26c563e41d
    9f52f61588cc1147466fdf5b3c4d977e242036a3 -> 78e745032787443679a14d210dcdc8f6e2230f01
    af87c13ecbd2b147d2034d41fdd519b8bf9291d1 -> c0f9e0f4bfa2bcd9a0d4f6d8226fd53fc1a1ba20
    c7b59022c60b30fc6c7d5ea07b03cd3b6464b242 -> 2d4d8c8eb0548bf84b7e58e6b5528b7b365e8309
    cb38df598bc0405abee0aad9d749a9fb97db7508 -> ca148bd8e384c89c10c269dfe5f117f03210e7ba
    e36d9febf0f50afa616634c6759812d2bc4414f4 -> a5b565a8696e808ead79f4caf82864f6be4d8629
    f1d4f48b09265c8e78ccd5dbbc238e80b875d6d9 -> 318d5cb41d2fee1ce21598a7b5d14e16ae858567

## Equivalence evidence

All 15 mappings were checked independently after import. For every commit:

- the mapped parent list equals the source parent list after applying the map;
- author and committer names, emails and timestamps are identical;
- the complete commit message is identical;
- every source mode and blob appears exactly once beneath
  components/kilix-telemetry, with identical bytes; and
- stripping that one prefix yields the exact source tree and no extra path.

The result was 15 of 15 commits equivalent across topology, identity,
timestamps, messages, modes and prefixed tree bytes. The imported tip's locked,
offline telemetry suite passed all 33 tests. The public source tip maps to
057167558ac8d7f26593e194a08f24254b47ca02.

The committed map and verifier make that comparison repeatable against any
local clone containing the exact source commit:

    python tools/verify_history_import.py --source PATH_TO_SOURCE_CLONE

## Abort and rollback

Before a successor publication, abort by retaining the current public work ref
and discarding the successor if any combined-history hygiene, provenance,
licensing, component-identity, package, test, F120 or visibility gate fails.
No source remote needs repair because none was mutated.

After publication, rollback selects the last known-good parent and component
refs; it does not rewrite or delete either public history. Consumers
move only through their owning tracks and retain their direct read-only
fallbacks.

For split-back, freeze parent movement, select the commits affecting
components/kilix-telemetry plus explicitly reviewed shared-contract changes,
rewrite that stable prefix back to repository root in an isolated mirror, and
repeat topology/identity/tree/licence/build/test and complete-history hygiene
checks. Publish the extracted repository only with separate authorization and
a non-destructive ref plan. Shared schemas remain available at their frozen
digest so a split does not strand consumers or silently downgrade cache data.

## kilix-device-lease import

Status: local, unpublished commit. This is a byte import with recorded
provenance, not a history import.

### Source

The shared accelerator lease was written in kilix-voice on branch
`work/0.2.2-astra4-shared-device-lease`; no remote-tracking branch contains
that commit. The import
reads commit `b9c2a36f8b92fbc3100f600da44b8f4483d2f797` with `git show`; the
source checkout and branch were not modified. Two commits touch the imported
files, both authored and committed by
`itsmygithubacct <itsmygithubacct@users.noreply.github.com>`:

    095d18c1ffb57f9ab255fd698d2029cebfde8e42 2026-09-08T02:47:00Z Add conservative shared accelerator leases
    b9c2a36f8b92fbc3100f600da44b8f4483d2f797 2026-09-08T02:58:11Z Avoid waiting again when a queued lease request ends

### File map

| Source path | Source blob | Source SHA-256 | Destination | Destination SHA-256 | Change |
| --- | --- | --- | --- | --- | --- |
| `voicelib/device_leases.py` | `cbea4a9b7349ac9cd1199d5036630808301a52ae` | `382ea1de20dbf4eda21c92a529ecd95f04451a8224fb4b20b0315d17918b0ccc` | `components/kilix-device-lease/src/kilix_device_lease/__init__.py` | `382ea1de20dbf4eda21c92a529ecd95f04451a8224fb4b20b0315d17918b0ccc` | identical bytes |
| `tests/test_device_leases.py` | `cc6ac79b3ed6290793e68af5291ea6326386208e` | `67529dc68d588f3b919462fc12d678b369088787fb443095268215e7d0177a3d` | `components/kilix-device-lease/tests/test_device_leases.py` | `a8d0de2ea5675c35404280efc91fd53b1f2ccf3da6129908b58a7eaf545fa408` | 3 source line(s) replaced |
| `tests/device_lease_peer.py` | `a7f6c56b0aa0fc9489419a56861daa35f595a5fe` | `d9ba2b2f60a75f5748de306210bbe0a915dec5d05b49077f214421ca0e62a2fd` | `components/kilix-device-lease/tests/device_lease_peer.py` | `99225fd59491b030b47802718802ce19764beb204f87585ce3bafc3d1050570b` | 1 source line(s) replaced |
| `DEVICE_LEASES.md` | `554e84e91ec460f228cf61e15deb2578cc5ef126` | `f0d2eebc1bf22051eb7e76d6e8855238a2a1d886c4e7784f72b693af8f920a3d` | `components/kilix-device-lease/README.md` | `77ebe79900810606822636b7a2c3bcf1c673a13dd756b711b2737ad9dfc53446` | 6 source line(s) replaced |

The module is unchanged: its docstring names no import path, so no byte of it
needed to move. The table records the import commit only. Later commits change
the module and its tests to fix acceptance-review findings; the repository
history, not this table, records those changes. The two test files change only their import lines, and the
test module's `PROJECT` now names the component `src` directory that child
processes put on `PYTHONPATH`. The README is DEVICE_LEASES.md with the import
path replaced and two sections added (interface document, checks).

Added on import, with no source counterpart: `pyproject.toml` (uv_build 0.12.5,
distribution `kilix-device-lease` 1.0.0, no dependencies), `LICENSE`,
`tests/test_interface_document.py`, `tests/test_mutual_exclusion.py`, and the
interface document `contracts/kilix.device-lease-v1.interface.json`.

### Why not a history import

The telemetry import above was a prefix-only rewrite of a whole repository.
This import selects four files from a larger repository, renames the module
path and rewrites test imports, so no prefix-only rewrite reproduces it. The
source commits' identities are recorded above instead.

### Licence

kilix-voice is distributed under GPL-3.0. These files were written under the
same account, on a branch no remote-tracking ref contains. They are offered here
under this repository's MIT licence, which the package gate requires of every
component. The owner's confirmation of that licence choice is required before
this commit is published.

### Consumers and the one-home rule

kilix-qwen-tts and kilix-transcribe import `kilix_device_lease` and no longer
import `voicelib.device_leases`. No copy of the lease is to remain in
kilix-voice: branches carrying `voicelib/device_leases.py` must not be merged
there as they stand.

### Abort and rollback

Before publication, drop the import commit; nothing outside this repository
and the two provider work branches references it, and the kilix-voice branch is
untouched. After publication, consumers move back only through their owning
tracks.
