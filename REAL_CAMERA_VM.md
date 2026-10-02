# Real input camera/source diagnostic — 2 October 2026

This is a three-arm **detector training-fit diagnostic**, not generator training,
restoration retraining or application checkpoint selection. Its inputs and labels
are verified locally; actual CUDA execution is pending. All optimizer work must
run on the existing NVIDIA L4 VM. The runner refuses Windows/CPU before creating
outputs, models or optimizers. No local training or smoke-training was performed.

Local repository: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`.
New isolated pilot: `~/forensic-dgp/real_camera_vm_bundle/`.
Read-only prior VM assets: `~/forensic-dgp/coverage_vm_bundle/`.
The unchanged old `native_expert_vm_bundle` is not this experiment.

## What is being tested

The 36-case returned-checkpoint review finds reduced detector coverage under
blur/noise and empty obstructing-hair masks. The user-supplied Mendeley ZIP adds
reviewable hands and a hair curtain, but has no publisher removal masks or splits.
Only eight sources were separately labelled and admitted to a new supported
registry. They share a related capture cohort; this is not subject-diverse data.
The whole ZIP, non-face crops and unlabelled images are excluded from training.

The verified real registry has123 records:91 training,25 unchanged validation,
7 unchanged previously inspected test. The camera cache holds91 native/degraded
pairs, preserving targets, valid support and source geometry exactly. Blur,
downsampling, noise and JPEG affect observed source RGB only; neutral padding
does not enter the filters. Unknown observed pixels remain context.

| Arm | Actual training sources | Difference |
| --- | --- | --- |
| `native83` | Previous83 real sources, native only;280 existing reflection fixtures | Matched continuation control |
| `camera83` | Same83 real sources, mixed native/degraded; same280 fixtures | Input-only camera effect isolated from new-source additions |
| `camera91` | Same83 plus8 reviewed Mendeley sources, native/degraded; same280 fixtures | Addition effect relative to `camera83` |

All arms start independently from reflective epoch42 and **identical inherited
AdamW moments** at step672/model882 updates. Each arm has2 epochs×56 batches:
112 updates;336 updates total across three independent branches. Each final
branch has moment step784/model994 updates and global epoch44. Do not add the
three branches together as if one model received336 sequential updates.

Batch8 contains3 real samples (2 covered/1 clear) and5 reflection fixtures
(4 covered/1 clear). Every fixture appears once per epoch in all arms. `native83`
and `camera83` use identical source slots; only the real input condition differs.
In `camera91`, each new covered source appears twice per epoch (one native, one
degraded); the new clear source appears three times. Thus only34/336 real-sample
slots across two epochs belong to the related new cohort. All91 real sources
still appear each epoch. Held-out sources are never forwarded in this diagnostic.

The fixed loss is0.5 supported real +0.5 supported fixture loss. Both use
BCE+Dice+0.25 hard-visible penalty/top10%, with valid support in every supervised
reduction. Encoder LR1e-5, decoder/head LR1e-4, AdamW decay1e-4, clip1, fp32,
seed42, frozen reference head and BN running state. No LR/threshold sweep,
optimizer reset, extra epochs or generator calls occur. Threshold is0.5.

Initial and each final model score the same462 train-cohort views (182 real,
280 fixtures). New-source scores are training-cohort diagnosis, including in
arms that do not train on them; they are not a final holdout. The ten-row preview
compares input, target, source42 and all three final predictions. These are masks,
not generated facial estimates. First actual run timing is unverified; plan for
approximately5–15 minutes, with a30-minute training/measurement cap after loading.
Startup checks and archive export are outside that cap.

## Readiness and selection

The package is ready for **VM preflight and this bounded diagnostic** only.
The dataset's `training_recipe_ready=false` remains historical data-stage metadata;
the separately frozen experiment explicitly consumes it with supported reductions.
No model is qualified merely by completing this run. Five fixed fit checks are:

1. `camera83` old degraded IoU exceeds the matched `native83` control.
2. `camera83` retains old native and reflection fit versus `native83`.
3. `camera91` improves new native and degraded IoU versus `camera83`.
4. `camera91` retains old native/degraded and reflection fit versus `camera83`.
5. Every final real/fixture clear control remains empty.

Retention means nondecreasing IoU and no increase in missed fraction, visible FP,
empty-covered cases or clear errors, with identical support/denominators. All
five passing checks justify independent return audit and a further evaluation
decision, not promotion. The original425-case gates are **not evaluated here**;
reflective42's historical synthetic failure remains unchanged. No `best.pth`
is created. Each arm produces `last.pth` marked unselected. Inspect the ten-row
grid and all relevant family cases before preparing original-gate and end-to-end
evaluation. Standalone scarf/general-object training coverage is still missing.

## Transfer and exact VM commands

Upload these two prepared local files through Google Cloud SSH to `~`:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\real-camera-vm-bundle.tar.gz`
and `C:\xampp\htdocs\YEAR 4\Testing\outputs\real-camera-vm-bundle.tar.gz.sha256`.
The bundle is self-contained for new code/data. Existing weights, moments and
fixture tensors are read from the old VM asset root; no reinstall/download is
prescribed. Its checksum sidecar uses LF to avoid the previous CRLF filename error.

```bash
cd ~
sha256sum -c real-camera-vm-bundle.tar.gz.sha256 &&
test ! -e ~/forensic-dgp/real_camera_vm_bundle &&
tar -xzf real-camera-vm-bundle.tar.gz -C ~/forensic-dgp &&
tmux new-session -A -s dgp_training
```

Inside tmux:

```bash
cd ~/forensic-dgp/real_camera_vm_bundle
test -f ../feature_vm_bundle/.venv/bin/activate &&
source ../feature_vm_bundle/.venv/bin/activate &&
nvidia-smi &&
python -u scripts/train_real_camera_vm.py --dry_run --batch_size 1 &&
python -u scripts/train_real_camera_vm.py
```

`--existing /absolute/path/to/coverage_vm_bundle` is allowed only if the prior
asset directory moved; use it on both Python commands. Preflight verifies bundle
hashes, model/moment identity, cached fixture tensors, actual CUDA/VRAM, pinned
Torch2.9.1+cu129 and isolated SMP dependencies, dataset directory presence, original
repository `outputs/phase4_with_progress/split.json` hash/disjoint membership and
one finite source171 partial-support GPU loss. The dry run constructs no optimizer
and performs zero updates. A1-batch pass is not trained-quality evidence.

The runner refuses existing execution output. If it fails, retain partial logs;
do not rerun over them or silently reset an arm. No code is pushed by this bundle.
`git pull` alone cannot supply these uncommitted new files or ignored checkpoints;
it is not required for this isolated, hash-bound transfer.

## Return files

On success the runner exports:
`~/forensic-dgp/real_camera_vm_bundle/real-camera-results.tar.gz`
and `~/forensic-dgp/real_camera_vm_bundle/real-camera-results.tar.gz.sha256`.
Download both through Google Cloud SSH to
`C:\xampp\htdocs\YEAR 4\Testing\outputs\`.

The return contains three final detector checkpoints, frozen protocol/inventory,
all initial/final measured masks, per-case counts,336-step logs, fit decision and
ten-row preview. Each optimizer remains on the VM under
`~/forensic-dgp/real_camera_vm_bundle/outputs/real_camera_vm/<arm>/final_optimizer.pth`
with its hash recorded. Omitting those three large optimizer files reduces the
download; preserve them for any future audited continuation. No automatic next
training run, application swap or exact hidden-identity claim follows completion.
