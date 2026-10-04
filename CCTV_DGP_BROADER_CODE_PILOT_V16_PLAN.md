# V16 broader code-learning preparation — 4 October 2026

**Preparation only; not ready to launch training.** The reset code-only module
is implemented. The cache/trainer/supervisor, independent audit and executable
frozen transfer package remain to be prepared. V15 was fully audited and rejected;
neither its fitted head nor its negative result is overwritten. The VM is stopped.

Windows `C:\xampp\htdocs\YEAR 4\Testing\` ↔ VM `~/forensic-dgp/`.
This plan's intended VM counterpart is
`~/forensic-dgp/CCTV_DGP_BROADER_CODE_PILOT_V16_PLAN.md` after document sync.
Current module `dgp_broader_code_conditioner_v16.py` ↔ intended
`~/forensic-dgp/dgp_broader_code_conditioner_v16.py` after verified transfer.
No V16 checkpoint or VM execution directory has been created.

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

## Proposed finite recipe to freeze before execution

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
case replays to keep source/profile balance. Verify the actual schedule before
freezing; these values are a proposed budget, not executed training counts.

Cache frozen DGP RGB/encoder features/base logits to disk with exact float32
metadata and checksums; store teacher labels only for training references. Use
streamed batches rather than holding the full roughly8GB feature cache in the
VM's16GB RAM. Require at least12GiB free cache/export disk. Existing pinned
weights/runtime are reused; no dependency upgrade or full-dataset re-upload.

Expected total around20 minutes based on V15's measured frozen forwards; cache
cap900s, training/evaluation cap1200s, overall supervisor cap2400s/40 minutes,
VRAM20GiB. Measure early cache batches and25 training updates before continuing;
stop when projected time exceeds the frozen budget. No overwrite or resume.

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

Next: implement and verify the streaming cache, balanced schedule, finite VM
trainer and independent audit; freeze a verified package, then launch through
the authorized gcloud connection. Return collection, handoff update and idle
shutdown stay automatic. The main app and active Goal remain unfinished.
