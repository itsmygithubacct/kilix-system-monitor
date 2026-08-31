# Provider-profile measurement harness

`profile.py` runs **1/1** absolute provider executable with no shell and a fixed
clean environment. It hashes **1/1** exact artifact, **1/1** exact fixture and
**1/1** redacted `plebian.hardware/v1` snapshot, samples the provider process
tree, and writes **2/2** new 0600 files:

- raw measurement evidence with command-argv, artifact, fixture and hardware
  digests; and
- **1/1** schema-valid `plebian.models.profiles/v1` catalog.

The catalog is always `qualification_eligible: false` and its profile is always
`unqualified`. The harness measures bytes and RAM; it does not invent GPU
memory or performance values, decide licences, extrapolate across hardware, or
turn its own output into an accepted profile.

Example (all file arguments must be absolute and outputs must not exist):

```text
python -m tools.measure.profile \
  --profile-id provider-small-cpu --profile-version 0.1.0 \
  --provider provider --task tts --backend cpu \
  --catalog-id provider-20260831 \
  --artifact-id provider-small --artifact-version v1 \
  --license-decision-id provider-small-licence-20260831 \
  --artifact /absolute/model.bin \
  --fixture-id tts-short-v1 --fixture /absolute/fixture.wav \
  --hardware-snapshot /absolute/hardware.json \
  --command-id provider.measure.tts \
  --raw-evidence /absolute/raw.json \
  --profile-catalog /absolute/profile.json \
  -- /absolute/provider --model /absolute/model.bin
```
