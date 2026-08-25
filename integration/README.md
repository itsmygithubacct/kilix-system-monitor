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
