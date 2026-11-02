# In-place model updates

Model-set isolation ships with the Completion R2 installer 0.2.2. The exact native
0.2.1-to-0.2.2 update preserved prior models and saved records; see
[R2_VERIFICATION.md](R2_VERIFICATION.md). Published 0.2.0 and 0.2.1 installers
remain historical releases without this support. No automatic background updater
or model-quality qualification is implied.

## Storage and activation

Managed releases resolve each family to
`runtime/data/models/<implementation-or-completion>/<inventory-key>/`.
The key is the first 32 hexadecimal characters of a canonical asset-inventory
SHA-256. An exclusively published `.model-set.json` binds the full 64-character
fingerprint, family, and exact asset IDs, filenames, sizes, and hashes. A shortened
key collision or mismatched/missing marker in a nonempty directory is refused.
Resolution performs no writes; setup binds an empty namespace before copying.
Markers establish ownership, not readiness or model quality.

Changed bytes create a different family namespace. Unchanged families reuse the
same namespace across source commits and metadata-only changes. Original 0.2.0
sources continue using their legacy flat folders. Their weights are retained.
Checkout/developer installs keep their existing checkout-local model paths.

All required model bytes and the application/dependency checks must pass before
the existing distribution engine atomically changes `runtime/current.json`.
There is no separate model activation pointer. A failed candidate can retain its
own partial files, but does not redirect the previous application to new weights.
Repair stays inside an exact bound inventory and retains damaged-file backups.

The first managed migration retains the legacy models plus one new set per
family. A later changed family needs additional storage for its replacement.
No automatic deletion of model generations occurs. Model files are copied rather
than hard-linked, preserving the existing unaliased-file integrity requirement.
Offline relocation skips copying legacy flat weights for the new source layout;
it copies the pinned payload directly into the new set once. Saved records and
relocation verification remain unchanged.

## Releasing an approved replacement model

1. Update the reviewed model manifests and matching runtime/architecture/logical
   state pins together. Model-store routing never admits arbitrary new weights.
2. Prepare a new versioned offline payload containing the exact approved files.
3. Verify model loading, inference, source preservation, failed update recovery,
   and record retention before building the new installer.
4. Publish a new immutable version only after native installer acceptance.
5. Point the existing download page to that release; existing users run it over
   their current installation. Updating a Git branch alone does not update PCs.

Keep published 0.2.0 assets immutable. See [BUILD.md](BUILD.md) and
[VERIFICATION.md](VERIFICATION.md) for the delivered release and its boundaries.

## Historical source-only verification, 1 November 2026

The new tests use tiny synthetic weight files and managed temporary directories,
not actual trained-model updates or the registered installation. They cover
changed/unchanged sets, legacy preservation, marker and inventory rejection,
namespace-scoped repair, corruption, Windows temporary-file path length, and
logical/physical source directory identity. Three two-release tests exercise the actual PowerShell distribution transaction:
a failed install from a legacy release, failed final verification from a
namespaced release, and successful activation after verification with unchanged
family reuse. Prior source receipts, records, weights, and old-loader resolution
are retained in the failure cases. At that snapshot native EXE acceptance remained
a future-release check. The later real R2 update is recorded in
[R2_VERIFICATION.md](R2_VERIFICATION.md); neither check qualifies model quality
or antivirus behavior.


Verification on 2026-11-01: the full Python suite ran 376 tests in 211.091 seconds,
with 5 skips for unavailable Windows link privileges. All executed tests passed.
The scoped model/installer tests ran 87 tests with 3 link-privilege skips. The
Windows long-folder publication regression executed successfully. No model pins,
trained weights, dependencies, public release metadata, or installed app changed.
Evidence: scratch/model-update-full-suite.log and model-update-all-focused.log.

Internal delivery evaluation: accuracy 4/5 (native EXE adoption with real new
weights remains untested); completeness 4/5 (source support delivered, future
release acceptance still required); clarity 5/5 (unreleased and published
contracts separated); actionability 5/5 (release path and tests documented);
conciseness 5/5 (bounded change with no updater or unrelated UI work). Overall
4.6/5. The highest-impact next check is a new versioned installer using approved
replacement weights; model usefulness and antivirus behavior require separate
evidence.
