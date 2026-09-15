# kilix-device-lease

`kilix_device_lease` supplies the standard-library-only
`kilix.device-lease/v1` cooperative per-user execution contract. Qwen,
transcription and language-model providers contend on one conservative
accelerator lock. Different selected-device labels cannot create parallel
grants. The kilix-voice session/microphone arbiter and Vosk/espeak behavior are
independent of this API.

This is execution coordination. It does not authorize a model, certify a
driver/device, accept caller-supplied VRAM estimates or establish hardware fit.
The provider must separately validate its installed model/runtime and admitted
hardware policy. CPU fixtures cannot claim accelerator qualification.

```python
from kilix_device_lease import acquire

lease = acquire(
    job_id=opaque_job_id,
    workload="tts-utterance",  # also stt-job or llm-turn
    device=selected_device_id,
    deadline=absolute_monotonic_deadline,
    cancelled=is_cancelled,
    disconnected=is_disconnected,
    progress=show_queue_status,
)
with lease:
    # Before model allocation, explicitly pass lease.guard_fd to the dedicated
    # job supervisor using Popen(pass_fds=...). Retain the inherited guard until
    # every owned engine descendant is reaped; propagate to engines if possible.
    # Existing runner cancellation/deadline/disconnect control remains required.
    result, cleanup_attested = run_owned_job(lease.guard_fd)
    if cleanup_attested:
        lease.release(cleanup_complete=True)
```

Job/device labels accept 1–96 ASCII characters from letters, numbers, `_ . : -`,
starting with a letter or number. Use opaque identities; do not pass user paths,
text, transcripts, audio, consent or prompts. The deadline is an absolute
`time.monotonic()` value no more than one hour ahead. Cancellation/disconnect
predicates must be nonblocking; the queue progress callback runs outside the
registry lock. An exception raised by any of these callbacks reaches the caller
unchanged, with the request withdrawn from the queue; it is never reported as a
lease code. `QueueStatus` has version, opaque ticket, state and one-based
position. Positions reflect workload rotation with FIFO order within each
workload. Capacity is 24 queued requests, at most eight per workload.

`LeaseError.code` is `invalid-request`, `queue-full`, `cancelled`, `deadline`,
`unavailable` or `lost-lease`. Disconnect returns `cancelled`. Do not silently
continue without a required grant. `lease.check()` validates current ownership
and cancellation/deadline state. The guard descriptor is CLOEXEC unless
explicitly passed to a child.

`release(cleanup_complete=True)` acknowledges that all owned engine processes
have been reaped, or that no engine was started. A supervisor-only completion
channel inaccessible to engines is one way to carry that acknowledgement.
An engine result, `ENGINE_FAILED`, cancellation, timeout, disconnected provider
or vanished supervisor is not cleanup proof. Release marks the record releasing
and closes only the provider's descriptor; the next grant still waits for all
inherited descriptor copies to close. Recording the acknowledgement waits for a
busy registry for up to the one-hour maximum request wait. If it still cannot
be recorded, release raises `deadline` and keeps the descriptor, and calling
`release(cleanup_complete=True)` again records it. It never calls `LOCK_UN`, which could
unlock the shared open-file description while descendants still retain it.

Default `release()` and context exit close the local descriptor and preserve
an unacknowledged active grant. Even after all FD copies vanish, the persistent
grant remains unavailable. This quarantine covers killed supervisors with
surviving engines that do not retain the guard. The API has no automatic reset
of unproven ownership: resolve complete owned teardown before administering a
fresh namespace. Stale PID numbers do not authorize deletion or recovery.
No resident model cache may outlive a positively released grant; idle eviction
must finish before acknowledging cleanup.

The default namespace is `/run/user/<effective-uid>/kilix-device-leases-v1`.
Its existing parent must be user-owned mode 0700. A deliberate private
`namespace=` override supports coordinated deployments and isolated tests;
every cooperating provider must use the identical namespace. A permanent
private anchor records the directory and resource inode identities. Existing
incomplete, foreign, unsafe, hardlinked, symlinked or replaced state is refused,
not repaired or recreated. First creation is decided under the anchor lock: an
empty anchor with no namespace directory beside it has never granted anything,
so whichever requester locks it first initialises it, and simultaneous first
requesters all wait their turn instead of being refused. Dead queued requests are pruned using their kernel
ticket locks, without PID liveness guesses. Queue bookkeeping saves the queue
before it removes a ticket file, and a queue entry whose ticket file is already
gone is dropped, so a coordinator killed at any step cannot wedge the queue.
A child forked without exec closes its copies of every registry and ticket
descriptor the module has open, so it never keeps the registry locked or a dead
requester's queue place alive, even when it was forked from a callback in the
middle of a registry pass. A child that returns from a callback into its
parent's `acquire` or `check` call is refused `lost-lease` at once, without
withdrawing, taking or closing anything of its parent's request. A fork made
outside Python's fork hooks, such as a raw `fork()` in a C extension, is not
covered. The library never starts a broker
daemon, kills a process or changes the embedding process's child-reaping policy.

The threat boundary is cooperating providers under one user. A program that
deliberately bypasses this API or lies about cleanup is not prevented from
opening a device by this user-space coordination mechanism. Actual microphone
capture, persistent recording indicators and screen-lock termination require
their separately integrated device policy.

## Interface document

`contracts/kilix.device-lease-v1.interface.json` at the monorepo root is the
interface record for this distribution: its version, workloads, queue bounds,
maximum wait, label pattern, default namespace, error codes, release semantics,
guard-descriptor inheritance and admission policy. `tests/test_interface_document.py`
holds the module to that document, so a constant cannot change without the
document changing with it. Interface acceptance by F100 is a separate record
and is not implied by this repository.

## Checks

From the monorepo root, with the release-pinned uv 0.12.5:

    make lease-check UV=/absolute/path/to/release-pinned-uv-0.12.5

The suite runs private real-process controls. It uses no GPU, model payload,
microphone, network or audio output, and never touches the default namespace.
Every test process, and every Python child it starts, installs
`tests/containment/lease_path_guard.py`, which refuses any filesystem call on
a path under `/run/user` or naming the default namespace leaf. The one test
that creates the default namespace does so in a child whose filesystem root is
moved to a private directory and which refuses every other path. Planted
calls prove both refusals. Mutation runs should still use a private mount
namespace over `/run/user`, because neither refusal sees calls made through
ctypes.

The module was imported from kilix-voice; MIGRATION.md records the source
commit, the exact bytes and every change made on import.
