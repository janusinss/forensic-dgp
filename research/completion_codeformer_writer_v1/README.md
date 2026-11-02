# Evidence-writer scale repair prototype

26 October 2026 (Asia/Manila). Research-only source and local fixtures. The sealed
coverage-v2 helper, worker, packet and all returned evidence are unchanged. This
prototype is not imported by a training packet or the application.

## Failure being investigated

The actual V2 run completed64 learning cycles by923.424seconds and all896 typed
forward slots, then reached the1800-second supervisor hard kill during terminal
serialization. The320,900,800-byte `.partial` is not a valid ZIP checkpoint. It
must stay failed-save evidence, not a model file. No fragment recovery is done.
Separate quality audits also failed the V2 preservation/quality gates; fixing the
writer does not make that learned recipe acceptable or authorize repeating it.

The old capped writer calls its budget guard for every serializer write. Each
category total scans all tracked files and resolves their paths again. With the
actual624 returned regular files,620 are tracked output files. Three old
terminal-write checks made7,443 category calls and7,446 Path.resolve calls,
taking1.581seconds locally. The cached prototype made zero rescans for the same
three checks and took0.001025seconds. These measured local results demonstrate
an accounting complexity problem, not a measured estimate of VM runtime or proof
that it was the only cause of the VM timeout.

## What changes in the isolated helper

`optimized_budget.py` uses a cached extent per owned open file and O(1) totals
for terminal, teacher, metadata, PNG and array categories. Each open validates
its canonical root/path and rejects symlinks, traversal and unapproved extensions;
root and parent symlinks are checked before resolution. Writes use the owned
file handle and compare proposed extent growth with exactly the existing byte,
time, disk, metadata and pending-terminal reservations. Seek rewrites do not
count bytes twice. No public `fileno` exposes a way around the callback.

Commit renames update only ledger metadata after the caller's actual atomic file
rename, preserving the old caller contract. A committed terminal consumes its
reservation; remaining receipts retain emergency headroom and the terminal time
window. A failed cap write keeps its prefix/partial and cannot produce a committed
state. The explicit ledger consistency check is for audit boundaries, rather than
rescanning the namespace in every write. File paths are checked at open; owned
streams retain their handles. This does not claim immunity to arbitrary external
processes moving paths during a run, and the future worker must retain its source,
namespace and pre-save publication checks.

No quality gate, work counter, deadline, disk floor or evidence limit is relaxed.
This helper imports the immutable V2 contract as its policy source. Prototype
classes are not a new architecture, optimizer or training route.

## Bounded tests and measured outcomes

`namespace_scale_benchmark_v2.json` binds the final helper. At620 entries,8MiB
written in eight1MiB callbacks produced identical bytes with native, old and
new writers: native0.003817s, old6.991855s, new0.005822s. Fixture clocks/free-space
were intentionally fixed to isolate accounting cost; real quota/deadline
behavior is tested separately. No actual return file was written or altered.

`policy_tests_v2.json` records18 passing focused tests: inherited category/global
caps, path/root links/traversal, overwrite, deadline/disk rules, metadata failure
headroom, empty-to-complete terminal commit, seek extents, category rename refusal,
native NPZ/PNG byte parity and partial retention on a serialization cap failure.

`tensor_32MiB_scale_fixture.json` uses996 independent CPU tensors:332 weight-like,
332 first-moment-like and332 second-moment-like arrays. They are not a neural model
or optimizer. At620 tracked output entries,5,014 write callbacks required only two
canonical path validations. The32,927,287-byte serialized file took0.095722s with
the cached writer versus0.072416s native, with identical SHA256. Every1002 ZIP
member CRC and all996 safe CPU tensor readbacks matched. The fixture deadline was
30seconds; no long old-writer large-callback run was attempted. These measurements
are local and do not guarantee any other machine's throughput.

All limits are inherited from the original policy. Tiny fixture values/clock/free
settings only test boundaries and behavior. No model, backward, optimizer object,
optimizer update, VM call or reconstruction of the invalid `.partial` occurred.

## Required before a future integration

Independent review must confirm cached ledger invariants and all policy boundaries.
If later admitted into a distinct future worker, instrument monotonic phase starts
and ends for teacher caching, parent loading/Adam restoration, parity, each
evaluation cohort, terminal CPU extraction, state-dictionary assembly, serialization,
flush/fsync, final checks and atomic commit. Save a small pre-save receipt before
creating a `.partial`, binding counts, current logical state, planned state bytes,
free disk, output/category totals and deadline remaining. Record serialization
bytes/callback counts, phase duration and final ledger totals after save. Keep
all failure partials; never relabel an incomplete file as a checkpoint.

Use a finite serializer fixture at the intended real namespace and tensor counts
before the manual VM packet. Do not extend deadlines to hide accounting overhead.
No new training packet or automatic launch follows this infrastructure repair.
The failed V2 learning result requires a separately justified learning design.
