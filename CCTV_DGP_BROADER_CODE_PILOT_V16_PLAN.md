# V16 broader code-learning preparation — 4 October 2026

**Closure,5 October2026:** manual launch stopped at its cache timing projection
with0 backward/optimizer updates. Independent return audit is complete and the
original failure is preserved. See `CCTV_DGP_BROADER_CODES_V16_FAILURE_AUDIT.md`.
Separate V16 r2 changes source/role timing and cache order while retaining the
learning recipe, schedule, cohorts and finite caps. Its verified manual runbook
is `CCTV_DGP_BROADER_CODES_V16_R2_VM.md`. No useful upgrade or app adoption.

**Subsequent R2 closure:** all 3,128 updates are independently audited; both
trained snapshots fail appearance preservation. The V17 input-feature connection
control is also audited and rejected for degraded restoration. Reports:
`CCTV_DGP_BROADER_CODES_V16_R2_RESULTS.md` and
`CCTV_DGP_FIDELITY_SPOTCHECK_V17_RESULTS.md`. Preserve all original failed pilots;
do not launch their historical instructions again.

## Historical V16 preparation

**Verified preparation; VM execution pending.** The streaming cache, finite
trainer/supervisor, success/failure importer and independent arithmetic auditor
are implemented. Fourteen boundary checks passed with zero local backward calls
or optimizer updates. The1,088,480-byte archive has23 independently checked members;
116 parent assets,6,195 data assets and2,804 historical baseline artifacts were
verified. CUDA preflight and outputs are unverified. V15 remains preserved;
the application remains unchanged.

The user selected **transfer files and pasteable VM commands only** on4 October.
This supersedes the earlier automatic gcloud launch instruction below. No cloud
query/start/SSH/training occurred. The last historical receipt recorded TERMINATED;
current cloud state is unverified. Runbook:
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_BROADER_CODES_V16_VM.md` ↔ intended
`~/forensic-dgp/CCTV_DGP_BROADER_CODES_V16_VM.md` after separate transfer.
Protocol4331c28c97a659846eab76ee0dee9150811a00da3c6139d8d24b7905173a6681.
Archivea40f68ccdeb7e7cf2e1420bef8c984a7faad6120461073da4362bcf2fb8ab65e.

Windows `C:\xampp\htdocs\YEAR 4\Testing\` ↔ VM `~/forensic-dgp/`.
This plan's intended VM counterpart is
`~/forensic-dgp/CCTV_DGP_BROADER_CODE_PILOT_V16_PLAN.md` after document sync.
Current module `dgp_broader_code_conditioner_v16.py` ↔ intended
`~/forensic-dgp/dgp_broader_code_conditioner_v16.py` after verified transfer.
The frozen local package is `outputs\cctv_dgp_broader_codes_vm_v16\`.
Its VM counterpart `~/forensic-dgp/cctv_dgp_broader_codes_vm_v16/` was not created
by this local preparation. No V16 trained checkpoint exists locally.

## Hypothesis and architecture limit

V14 could fit ten photographs; V15's fresh-image path matched that fit exactly
but failed on new faces. The next experiment tests a reset2,422,432-parameter
code-only classifier on broader data. It removes the shared mean/std objective
and the memorized fitted initialization. DGP, CodeFormer renderer/encoder,
clean-code teacher and recognizer remain frozen and explicitly declared.
This changes data, objective and optimizer recipe together: an engineering
pilot, not proof that data alone caused any improvement.

The frozen prior also changes clear appearance without our head. More code
learning cannot by itself guarantee clear identity preservation. This is a
component diagnostic; a separately evaluated output path that preserves retained
DGP/visible appearance is required before application adoption. Strict historical
preservation guards remain unchanged even if they reject the component.

## Frozen finite recipe; actual execution unverified

Use all781 existing training references (391 FFHQ/390 Asian) and their3,905
fixed camera cases from audited V9. Keep the104-reference/520-case development
validation cohort separate; retain exact IDs/source/target hashes and provenance.
No native24 or reserved32 access. No output-driven preview selection or new
validation-tuned rendering mode. No clean target or teacher is needed at inference.

Train only the reset code head with observed-token cross-entropy to frozen
clean VQGAN teacher codes. No statistics loss, feature MSE, recognizer-gradient
objective or generator/backbone fine-tuning. Proposed AdamW lr0.0003,
weight_decay0.01, betas(0.9,0.999), clip1, no AMP. Batch10 contains one case per
camera profile per source; seeded shuffles cycle the shorter source once per
profile/epoch. Eight epochs:391 updates/epoch,3,128 updates/31,280 exposures.
Each epoch covers all own training references and profiles, plus five Asian
case replays to keep source/profile balance. The seeded schedule is independently
checked by14 local boundary tests. These are frozen budgets, not executed counts.

Cache frozen DGP RGB/encoder features/base logits to disk with exact float32
metadata and checksums; store teacher labels only for training references. Use
streamed batches rather than holding the roughly9.3GB feature cache in the
VM's16GB RAM. Require at least12GiB free cache/export disk. Existing pinned
weights/runtime are reused; no dependency upgrade or full-dataset re-upload.

Expected total around20 minutes based on V15's measured frozen forwards; cache
cap900s, training/evaluation cap1200s, overall supervisor cap2400s/40 minutes,
VRAM20GiB. Measure early cache batches and25 training updates before continuing;
stop when projected time exceeds the frozen budget. Stop at epoch4 if fixed
training-preview code CE improves less than1% from epoch0. No overwrite or resume.

## Verification before and after training

Require local inference/boundary tests, exact zero-head starting-logit parity,
VM frozen-state proof and a one-batch backward preflight with zero optimizer
updates. All actual backward/training calls must run on forensic-dgp-thesis L4.
Source/cache/teacher/cohort fingerprints and exact update/exposure counts must
be auditable. The released generator/prior are not relabeled as our own training.

Predeclare epoch0/4/8 snapshots and balanced fixed training/validation preview
references. Report training code accuracy/loss and held-out delivered PNG
MSE/PSNR/SSIM/ArcFace per clear/source/profile group; compare retained DGP and
starting prior on identical inputs. Preserve failed gates, raw previews, all
PNG metrics, fixed-alignment embeddings and original256-cell grids.
No automatic `best.pth`, app replacement or checkpoint promotion in this
component diagnostic. Independent return audit and visual review precede
any broader-output claim or further experiment.

Next: follow the exact browser-SSH transfer/launch commands in the runbook.
The user's manual-transport selection supersedes automatic gcloud execution.
After returned files arrive, independently audit successful results or the failed
trace prefix and review all ten fixed grids. Local preparation/transfer receipts
are under `outputs\`; CUDA preflight and training remain unverified. The main app,
independent final review and active Goal remain unfinished.
