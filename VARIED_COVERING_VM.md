# Varied covering detector pilot — 3 October 2026

This finite VM experiment compares existing detector data with the verified
42-source COFW extension. It addresses missed hair/hand/cloth/object coverings
and clear-face errors demonstrated in `REAL_CAMERA_RESULTS.md`. It trains only
the detector; restoration and completion weights/application remain unchanged.
No local model, optimizer or training is required to prepare this bundle.

| Location | Windows | Linux VM |
| --- | --- | --- |
| Repository | `C:\xampp\htdocs\YEAR 4\Testing\` | `~/forensic-dgp/` |
| Runbook | `C:\xampp\htdocs\YEAR 4\Testing\VARIED_COVERING_VM.md` | `~/forensic-dgp/varied_covering_vm_bundle/VARIED_COVERING_VM.md` after extraction |
| Transfer archive | `C:\xampp\htdocs\YEAR 4\Testing\outputs\varied-covering-vm-bundle-v2.tar.gz` | `~/varied-covering-vm-bundle-v2.tar.gz` after upload |
| New workspace | Source files and ignored prepared inputs under the repository | `~/forensic-dgp/varied_covering_vm_bundle/` |
| Return files | `C:\xampp\htdocs\YEAR 4\Testing\outputs\varied-covering-results.tar.gz` and `.sha256` | `~/forensic-dgp/varied_covering_vm_bundle/varied-covering-results.tar.gz` and `.sha256` |

## Matched comparison and budget

| Branch | Real training sources | Changed slots |
| --- | ---: | --- |
| `existing91` | Previous 91 sources | Two existing covered and two existing clear inputs per batch |
| `varied133` | Previous 91 plus 42 reviewed COFW sources | Replace one covered and one clear slot with the new cohort |

Both start with the same audited **camera91 `last.pth`**, SHA256
`e4b16da0ccb92b2a3a6b29f10b863320ccf3ead1fb9e1464f753a667f76c1702`.
This is an unselected development checkpoint, not `best.pth` or the Phase 3
restoration model. It already improves degraded mask/hand transfer and serves
as a shared starting point. Its historical clear/retention failures remain.

Both branches create **fresh AdamW state**, beginning at step zero. The source
has 994 cumulative model updates; its old moment-step metadata is 784. Those
optimizer files are deliberately not consumed. This is a matched new experiment,
not an identical resume. Each branch performs two epochs ×64 batches =128
updates. The two independent branches total256 updates; each final model has
1,122 cumulative updates and fresh optimizer step128, global epoch46.

Batch8 has four real inputs (two covered/two clear) and four reflection fixtures
(two covered/two clear). Replay cases and each slot's native/degraded condition
are identical between branches; the common old-covered/old-clear slots match.
New positives cycle across five reviewed groups: hands, obstructing hair,
cloth/scarf, objects and opaque eyewear/masks. Every arm source appears native
and degraded, and all280 existing training fixtures appear within the budget.
Compared with the previous diagnostic, both branches receive more clear controls;
the branch comparison isolates the new cohort/family-sampling intervention.

Loss remains0.5 supported real +0.5 supported replay loss, using BCE+Dice and
hard-visible penalty0.25/top10%. Valid support applies to every supervised
reduction; unknown observed pixels remain RGB context. Encoder LR1e-5,
decoder/head LR1e-4, decay1e-4, gradient clip1, FP32, seed42, frozen reference
head/BN running state. Detector threshold0.5 is unchanged; no margin/threshold
sweep or generator call occurs.

The initial model and two final models each score546 training-cohort views
(266 real and280 reflection). Total training/measurement budget is3,686 detector
image forwards, plus one separate preflight forward. No held-out forward.
Planning allowance: approximately5 minutes including checks/loading; the new L4
timing is not measured. The hard training/measurement limit is30 minutes after
loading, excluding transfer/startup/archive export. Preserve partial results if
the cap or any check fails; do not restart over them.

## Fit checks and selection boundary

The frozen training-fit checks are:

1. Every added covering group improves native and degraded IoU against `existing91`.
2. Old native/degraded fit retains IoU, missed coverage, visible FP and case-error counts against the matched control.
3. Reflection fit retains those same metrics against the matched control.
4. Every final clear control in both branches is empty.
5. Every new covered training example has a nonempty native/degraded prediction.

These exposed training-cohort checks diagnose learning and preservation. They
are not generalization, Philippine population performance or useful face-output
proof. All passing checks lead to independent return audit and inspection of the
fixed10-row grid, followed by a separate original425-case gate/practical-gallery
evaluation decision. The original gates are not evaluated, weakened or waived
here. No `best.pth`, selection or app swap follows this diagnostic automatically.
Each branch returns an explicitly unselected `last.pth`.

Supported V3 preserves all123 previous records/held-out bytes and adds42 author
training sources. The dataset's `training_recipe_ready=false` remains its data
stage declaration; the separate frozen protocol consumes it with supported
reductions. COFW test RGB, inspected RealOcc author validation and the full
unlabelled archives are excluded. Full source/terms/refinement history:
`C:\xampp\htdocs\YEAR 4\Testing\COFW_DATA_PREPARATION.md` ↔
`~/forensic-dgp/varied_covering_vm_bundle/COFW_DATA_PREPARATION.md` after transfer.

## Upload through the configured Windows Google Cloud SDK Shell

Run on Windows (CMD syntax), transferring one file per SCP command:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis varied-covering-vm-bundle-v2.tar.gz "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis varied-covering-vm-bundle-v2.tar.gz.sha256 "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

Use the actual VM zone if prompted; no zone is inferred from the instance name.
Existing installation/transfer details are in
`C:\xampp\htdocs\YEAR 4\Testing\WINDOWS_GCLOUD_TRANSFER.md`.
The isolated bundle supplies hash-bound new code/data; Git transfers only
committed tracked files and does not supply these ignored datasets/checkpoints.

## Google Cloud SSH: verify and extract

```bash
cd ~
sha256sum -c varied-covering-vm-bundle-v2.tar.gz.sha256 &&
test ! -e ~/forensic-dgp/varied_covering_vm_bundle &&
tar -xzf varied-covering-vm-bundle-v2.tar.gz -C ~/forensic-dgp &&
tmux new-session -A -s dgp_training
```

Inside tmux:

```bash
cd ~/forensic-dgp/varied_covering_vm_bundle
test -f ../feature_vm_bundle/.venv/bin/activate &&
source ../feature_vm_bundle/.venv/bin/activate &&
nvidia-smi &&
python -u scripts/train_varied_covering_vm.py --dry_run --batch_size 1 &&
python -u scripts/train_varied_covering_vm.py
```

Read-only prior assets are in `~/forensic-dgp/coverage_vm_bundle/` (280 reflection
fixtures/pinned isolated dependencies) and `~/forensic-dgp/real_camera_vm_bundle/`
(camera91 weights). If either moved, use `--existing /absolute/coverage/path`
or `--camera /absolute/camera/path` on both Python commands; every hash still
must match. No new installation, model download or inherited optimizer is needed.

Preflight refuses Windows/CPU before output/model/optimizer work, verifies bundle
hashes, all165 registered raw-source provenance files, CUDA/VRAM, Torch2.9.1+cu129,
pinned SMP dependencies, original Asian/FFHQ directory presence, the original
`~/forensic-dgp/outputs/phase4_with_progress/split.json` hash/disjoint membership,
reviewed registry/pairs, fixed schedule and fixture bytes. It runs one finite
partial-support hair case with **zero optimizer updates** and unchanged model
state. A preflight pass is not trained-quality evidence.

V2 resolves the preflight by the reviewed hair source ID (`cofw_train_0868`,
native case202). The unused V1 archive/protocol/audit are preserved: its positional
case194 was a valid partial-support hand input, contrary to its hair comment.
No V1 upload or CUDA run occurred in this workspace. Use only the named V2
archive above. Training schedules/budgets/data are otherwise unchanged.

## Download after completion

Run in the Windows Google Cloud SDK Shell:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/varied_covering_vm_bundle/varied-covering-results.tar.gz" .
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/varied_covering_vm_bundle/varied-covering-results.tar.gz.sha256" .
```

The return contains two checkpoints, all1,638 initial/final measured masks,
256-step logs, per-case/group metrics, protocol/inventory, fit decision and
the10-row preview. Each fresh optimizer remains on the VM under
`~/forensic-dgp/varied_covering_vm_bundle/outputs/varied_covering_vm/<arm>/final_optimizer.pth`;
its hash is recorded. Keep it for any future independently audited continuation.
New checkpoints do not replace the local application's current detector.

Next: verify the returned archive, budgets, lineage, masks/metrics and complete
removal footprints before deciding original-gate and generated-face evaluation.
Nearly hidden automatic rejection remains unresolved; this finite fit pilot
does not declare the full workflow Goal complete.
