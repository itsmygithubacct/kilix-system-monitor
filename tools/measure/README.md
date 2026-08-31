# Provider-profile intake validator

`profile.py` does not execute provider code. It validates **1/1** canonical,
provider-owned `plebian.models.provider-measurement/v1` record against **3/3**
exact byte identities: **1/1** artifact, **1/1** fixture, and **1/1** private,
redacted `plebian.hardware/v1` snapshot. The hardware document must pass its
full schema, semantic denylist, and privacy controls.

Successful intake atomically commits **1/1** new mode-0700 bundle containing
**2/2** exact-mode-0600 files: a schema-valid intake receipt and a schema-valid
`plebian.models.profiles/v1` catalog. A Linux `renameat2(RENAME_NOREPLACE)` is
the single visibility event. Before that event the final bundle is **0/1**;
after it, both files are visible **2/2**. Existing bundles are never replaced,
and symlinked path components are refused. Concurrent mutation by another
same-UID process can leave an empty private temporary directory after a refused
commit, but cannot expose a partial final bundle or a receipt claiming commit.
The protocol does not claim persistence of a caller-owned pathname against a
same-UID process that can rename that path before or after the final check.

Intake records but does not accept provider claims. The generated profile is
always `qualification_eligible: false`, `qualification: unqualified`, and
`confidence: unknown`. It promotes **0/9** provider measurement fields,
including **0/5** numeric resource metrics and **0/3** performance metrics; it
also promotes **0/1** provider licence decisions and accepts **0/1** profile
qualification. Its required architecture comes independently from the
schema-valid hardware snapshot.

Provider execution, artifact/fixture consumption proof, and evidence-grade
aggregate peak measurement remain outside this tool. Those require an accepted
no-escape launcher/cgroup boundary; sampled process trees are not accepted as
that boundary.

All paths must be absolute. Provider evidence and hardware JSON must be
canonical, caller-owned, singly linked, and mode 0600. The **1/1** output bundle
must be absent beneath a caller-owned, non-group/world-writable parent
directory.

```text
python -m tools.measure.profile \
  --provider-evidence /absolute/provider-measurement.json \
  --artifact /absolute/model.bin \
  --fixture /absolute/fixture.wav \
  --hardware-snapshot /absolute/hardware.json \
  --output-bundle /absolute/private/provider-profile-intake
```

On success the bundle contains `intake-receipt.json` and
`profile-catalog.json`, **2/2** files.

The provider record schema is
`tools/measure/provider-measurement-v1.schema.json`; the output receipt schema
is `tools/measure/profile-intake-receipt-v1.schema.json`.
