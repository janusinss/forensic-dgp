# Bank B r3 returned evidence and B r4 hash compatibility repair - 11 October 2026

Status: B r3 is a confirmed initialization **execution failure**, with zero
model forwards, gradients or optimizer updates. B r4 is a new independently
verified transfer; its manual VM loading, training and quality remain pending.
The complete restoration/completion goal remains ACTIVE.

## Returned evidence

Archive: `cctv-dgp-bank-comparison-v1-r3-B-results.tar.gz`.
Bytes: 811,929,274. SHA256:
`52ea8aca05f5d74403ef707915c51d4b800718ddd2f7795443255c56b4180d3a`.
The safe importer checked all 5,631 regular members and 5,623 prepared bindings.
The independent saved-artifact audit checked all 5,630 manifest bindings.
The only run-output records are environment, failure and initializer-loading
receipts. There are no new paired/native candidate or baseline image outputs,
model states, gradient results or quality-gate records to review. No new visual
sheets were generated. The previously reviewed A and B r2 sheets remain separate.

Worker stop: 8.149 seconds; supervisor:
22.026 seconds. All eight counters are zero:
original/A/B/recognizer/prior forwards, gradient queries, backwards and updates.
The failure occurred during `pinned_initializer_before_baseline`, before the
initializer hash or state loading completed. Export completion packaged the
failure; it is not training completion or image-quality evidence.

## Location, cause and correction

Location: `cctv_dgp_bank_pinned_initializer_v1.py`, `sha()` called from
`load_pinned_initializer()` before the baseline pass.
Cause: `hashlib.file_digest` is unavailable in recorded VM Python 3.10.12.
Python documents it as added in 3.11; see
[official Python hashlib documentation](https://docs.python.org/3.11/library/hashlib.html#hashlib.file_digest).
The new loader used an API unavailable in the known VM environment. This was
an implementation compatibility mistake, not evidence that the B design
cannot learn. It does not establish why B r2's constructor tensors differed.

Correction: new immutable loader `cctv_dgp_bank_pinned_initializer_v2.py`
uses streaming SHA256 over one-MiB blocks. Only its `sha()` function changes.
Hash-pinned initializer bytes, exact keys/shapes/dtypes/finite checks, permitted
new style-constructor differences, rejection of frozen/zero-fusion differences,
strict state loading and exact whole-state requirements remain unchanged.
The original loader, failed protocols, returns and receipts are preserved.

The missing API was reproduced without model import by extracting the old hash
helper and giving it a hashlib namespace without `file_digest`. The corrected
loader was tested on the real B CPU model/initializer with the API absent:
exact loading; exact restoration of a deliberately perturbed permitted style
tensor; rejection without overwrite of a frozen-prior change; rejection of a
zero-fusion change; rejection of a wrong initializer hash. Five cases passed;
three real file hashes also matched direct full-byte SHA256. State SHA256:
`98675d10be6f3405dd4bd0950d9251a61b26b9b161ae62a9ce48f775bc43e2bd`.
These are API-absence simulations on local Python 3.13, not execution in a
local Python 3.10 interpreter. They perform zero forwards/gradients/updates.
Actual VM loading and B r2's missing tensor diagnosis remain pending.

## Verified B r4 transfer

Protocol SHA256: `8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85`.
Full archive: `cctv-dgp-bank-comparison-v1-r4-B-execution.tar.gz`, 812,764,956 bytes.
Full archive SHA256: `aea4c8e14b9a0ff67744498bf776d1c74137bdae5b38030d61885181e35ed2c7`.
Small transfer: `cctv-dgp-bank-comparison-v1-r4-B-code-transfer.tar.gz`, 3,718,790 bytes
(3.55 MiB).
Small transfer SHA256: `354f41abf71ec94631a8584cf57e77a7fce3815c37e890e3eaf1de234d8c9d95`.

Independent audit checked every one of the full packet's 5,630 regular tar
members and the transfer's 15 members, both LF checksum files, all 31 Python
files against Python 3.10 syntax, and absence of runtime `file_digest` calls
in that packet. It reconstructed all 5,618 reused assets plus 11 overlays and
proved that the resulting packet exactly matches the full archive. Model
computation AST and 11 scientific worker functions remain identical. Data,
initializer, schedule, quality thresholds and finite resource caps match the
old packet; six unsafe protocol changes were rejected. The actual transfer
CLI hashes the real initializer with the repaired API-absent loader. The
data-only installer verification CLI also passed; its live VM copy is unrun.

Manual installation copies assets from the retained B r3 directory into a
new B r4 root; it never deletes or changes the source. It validates source
protocol/assets/failure/terminal supervisor, idle GPU and every copied byte,
has a 300-second stop, and writes `packet_install_receipt.json`. No model code
or training is imported. The full archive remains locally for recovery or
migration. Use the small transfer on this VM to retain the required disk margin.
Follow the [five manual upload, installation, tmux and download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_COMPARISON_B_V1_R4_VM.md>).

Scientific run remains 50 updates, 250 distinct TRAIN references, 1,250 profile
exposures and zero complete epochs / 0.3201 epoch, matching the intended A
exposures. B learns our conditioning/reconstruction with a frozen external
StyleGAN2 bank; the bank is not an own-trained generator. A's rejected trained
state is not the B initializer. A's retained raw/PNG gains remain 0.471905% /
0.473663%, below 1%, with 14 / 15 overlapping group/metric preservation failures.
A trained matched comparison is pending because neither B r2 nor B r3 updated.
No extra epochs, plateau bypass or app promotion are authorized by packaging.

Caps are unchanged: 16 GiB free after install, 1 GiB reserve, 6 GiB output,
20 GiB VRAM, worker 90 minutes, external 90.5 minutes, fitting 10 minutes,
preflight five minutes, snapshot 20 minutes, export 10 minutes and external
export 10.5 minutes. Runtime for a complete B r4 run is unmeasured.

## Authorized two-archive cleanup

Exactly two inactive VM home files were removed after full hash verification
against retained local backups: B r3 execution and B r3 result gzip archives.
Allocated reclaim: 1.512 GiB; measured recovery: 1.504 GiB.
Fresh free space: 17.145 GiB. Conservative space after the small upload
and complete B r4 copy: 16.243 GiB, meeting 16 GiB at this snapshot.
The installer and launch must check current resources again.

Independent closure checked exact paths, UID 1001, single links, inode/mtime/
size/hash bindings, idle workloads and the deletion ledger. Before/after
state matches for 416,692 research metadata
entries and 5,632 retained B r3 byte hashes.
Checkpoints, full states, data/splits, training caches, instructions, logs,
measurements, provenance and gate/failure records remain. Complete scientific
cache bytes were not rehashed; their namespace/metadata preservation is not
whole-cache byte verification. Two final post-export log tails are saved
locally without substituting any original returned archive bytes.

The separate Windows local cleanup by the other chat remains recorded in
PROJECT_HANDOFF.md and its exact previous bytes are retained. No VM start,
stop or switch, automatic pilot, gradient or training task was performed.
App source bindings and permanent checkpoint SHA256 `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b` match.

## Remaining goal and evidence

After manual B r4 execution, independently audit constructor differences,
actual counters/stops, full states and frozen tensors where present, bank
ablations, all raw/PNG records and every planned sheet. Native CCTV remains
unpaired without clean alignment; paired synthetic metrics stay separate.
Reserved final identities remain unused. No Zamboanga or ethnicity claim
follows from these sources. All five milestones, seven covering families with
separate automatic/assisted review, independent final review and bundled
inline Playwright app qualification remain pending. App unchanged this turn;
no browser test was rerun and no quality candidate was adopted.

Evidence: [B r3 failure audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_comparison_v1_r3_B_analysis/failure_execution_audit.json>),
[compatibility regression](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_comparison_v1_r3_B_analysis/hash_compat_loader_regression.json>),
[B r4 transfer audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_comparison_B4_transfer_v1_audit.json>) and
[cleanup closure](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_B3_archive_cleanup_v1/independent_audit.json>).
Future return importer: [import_cctv_dgp_bank_B4_return_v1.py](<C:/xampp/htdocs/YEAR 4/Testing/scripts/import_cctv_dgp_bank_B4_return_v1.py>).
Its contract/CLI are checked; execution awaits the three returned result files.
Publication bindings: [publication.json](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_comparison_v1_r3_B_analysis/publication.json>).
