# Native covering expert pilot — 2 October 2026

This is StageA: test whether the separately adapted face U-Net can learn the
reviewed real lens footprints while retaining its existing real/fixture fit.
It is not a deployable detector, generator run or final-quality claim. A later
learned input-only selector requires a separate protocol and the original real
and synthetic gates. Neither a target-informed oracle nor training fit selects
an application checkpoint. No actual model training is permitted locally.

Windows repository: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`.
Existing VM assets: `~/forensic-dgp/coverage_vm_bundle/` (read-only).
New pilot workspace: `~/forensic-dgp/native_expert_vm_bundle/`.

## Why this experiment

`FROZEN_COMPLEMENTARITY_RESULTS.md` records1,946 independently checked masks.
Both frozen heads miss8,205/21,853 labelled lens pixels. Union/intersection fail
the training checks; merely selecting their current binary outputs cannot create
the missing coverage. The face-specific branch already fits original73 real
examples well, but its four new native lens cases are incomplete. Preserve the
original Gated U-Net separately while testing new native coverage learning.

This pilot changes training membership, domain sampling and objective together.
It does not isolate the causal effect of native additions or removal of legacy
replay. Previous replay-removal failures remain recorded. There is no claim that
this expert retains the original synthetic gate by itself. Parameter separation
preserves the old branch; later inference routing can still regress.

## Fixed StageA recipe

The starting checkpoint is reflective epoch42, SHA256
`a51f20e8fa12cee7dec574195debc5ada9b8cfeeb289b12bad11a6d39a9acf25`.
Inherited AdamW moments, SHA256
`8075fde6c2d383d1923a6e69d5798882b0620ca435d20a18640eb49a34243935`,
start at step672/model882 updates. Do not reset moments. The original completion
parent stays untouched at SHA256
`c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93`.

| Setting | Fixed value |
| --- | --- |
| Training | 83 supported real images:51 covered/32 clear;280 existing fixtures:224 covered/56 clear |
| Budget | 6epochs×56batches=336 updates; final global epoch48 |
| Batch | 3real (2covered/1clear) +5fixture (4covered/1clear); batch8 |
| Sampling | Seed42; each fixture exactly once per epoch; real strata cycle shuffled lists; every real case appears each epoch |
| Loss | 0.5×supported real loss +0.5×supported fixture loss; each uses BCE+Dice+0.25 hard-visible penalty/top10% |
| Learning rates | Encoder1e-5; decoder/head1e-4; AdamW decay1e-4; clip1; fp32 |
| Frozen state | Reference visible-face head and BN running statistics; original parent checkpoint file |
| Inputs | Fixed256px; no new crop, resize, augmentation, threshold or checkpoint sweep |
| Final counters | Model1218 updates; optimizer step1008; this experiment336 updates |

Source171's unknown lower band remains observed RGB but has zero supervised loss.
Targets and all reductions explicitly use valid support. The ten native additions
are training-only; the fourteen pending cases remain absent. These are approximate
assistant-reviewed detector labels, not uncovered-face reconstruction targets.
All105 earlier records,25 validation/7 test membership and full held-out support
remain exact. Native labels were previously screened against4,210 references;
whole-image screening and unknown external pretraining do not prove identity separation.

Measure all83 real/280 fixture training cases before training and only at the fixed
final budget, threshold0.5, batch1 inference. No held-out model scoring occurs in
StageA. Save masks and all case counts. Six lens masks total21,853 pixels: old2,031
plus new19,822. The six-epoch budget cannot be extended after seeing results.

Five checks must all pass to justify preparing StageB (not promotion):

1. Native-four IoU improves; each of171/216/348/374 gains TP; native-four FP does not increase.
2. Original73 real training IoU, missed fraction, visible FP, empty cases and clear errors do not regress versus the same-run source42 baseline.
3. All280 fixture training metrics retain the same five safeguards versus that baseline.
4. Combined new lens recovery increases and old-two lens recovery does not decrease.
5. All32 real and56 fixture clear controls remain empty.

Both baselines/finals use identical supports and denominators. Preserve any failed
final candidate as diagnostic evidence; do not rename it `best_detector.pth`.
Inspect the fixed ten-row grid: four native coverings, two old glare cases, clear
sources207/208, first degraded reflection fixture and first degraded clear fixture.
Visual review is required even if all numerical training checks pass.

StageB is not implemented or authorized by these fit checks alone: first audit
the return, reproduce masks and inspect visuals. A proposed learned selector must
use only the image and model outputs at inference, never task/source IDs, target
labels or filenames. Original425-case development gates remain exact, including
synthetic IoU≥0.9746899906463458, missed≤0.016476187160583314,
visible FP≤0.0013327836915128428, empty≤1 and clear errors0/80. Generator/application
and Phase3 restoration remain unchanged until eligible end-to-end improvement.

## Transfer and VM execution

Local preparation does data/checkpoint-state audits and fixture tests only, with
no model/optimizer construction, backward pass on model parameters or training.
The archive contains new code, all252 supported dataset files, six lens maps and
the byte-bound original Phase4 split snapshot. Existing115MB moments,57MB weights
and125MB fixture cache are reused directly from the existing VM workspace.
No dependency download/reinstall is part of this pilot.

Upload `C:\xampp\htdocs\YEAR 4\Testing\outputs\native-expert-vm-code.tar.gz`
and its `.sha256` file to `/home/janusdominic0/` through Google Cloud SSH.
The preparation report and archive audit must exist before running this recipe.

```bash
cd ~
sha256sum -c native-expert-vm-code.tar.gz.sha256 &&
test ! -e ~/forensic-dgp/native_expert_vm_bundle &&
tar -xzf native-expert-vm-code.tar.gz -C ~/forensic-dgp &&
tmux new-session -A -s dgp_training
```

Inside tmux:

```bash
cd ~/forensic-dgp/native_expert_vm_bundle
source ../feature_vm_bundle/.venv/bin/activate
nvidia-smi
python -u scripts/train_native_expert_vm.py --dry_run --batch_size 1 &&
python -u scripts/train_native_expert_vm.py
```

The default read-only asset root is `../coverage_vm_bundle`; override with
`--existing /absolute/path/to/coverage_vm_bundle` only if it was moved. The VM guard
rejects Windows or unavailable CUDA before creating files/models/optimizers.
Preflight verifies inventory, inherited moment values, pinned dependencies and
Torch2.9.1+cu129, raw dataset-directory presence, split hash/membership, cache
hashes/support and one native input's finite GPU loss. It performs zero updates.
Full-batch allocation is exercised before the first actual update, using the
predeclared batch8. The pilot refuses existing output or archive: inspect partial
evidence instead of blindly rerunning. A1-batch dry run is not trained quality evidence.

On success the runner writes:
`~/forensic-dgp/native_expert_vm_bundle/native-expert-results.tar.gz`
and its LF `.sha256` sidecar. The return includes only the final model/moments,
input inventory, metrics,336-step log, initial/final masks and ten-row preview;
it avoids resending intermediate or initial checkpoint copies. Download those
two files through Google Cloud SSH to
`C:\xampp\htdocs\YEAR 4\Testing\outputs\`.

No git commit/push is required for this self-contained transfer. Ordinary source
changes remain available for a later explicit commit. The full goal remains active
until an eligible detector and visually reviewed completion improvement exist.
