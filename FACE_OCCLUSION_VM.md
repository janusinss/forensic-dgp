# Direct occlusion pilot: existing Linux CUDA VM

The controlled comparison tests a new direct occlusion head with fully trainable
FaceExtraction encoder/decoder versus the same randomly initialized architecture.
Both start with identical new head weights. This is an unproven pilot, not a
checkpoint selected for the application. The full recipe is in
`FACE_OCCLUSION_PILOT.md`; unchanged gates compare both arms with the original
completion parent. External FFHQ pretraining overlap remains unresolved.
Twelve local adapter/guard/runtime/package tests pass. Bash syntax and the five
pinned package imports are verified locally; actual CUDA execution is pending.

## 1. Upload the two files

Use Google Cloud SSH's Upload File control:

- `C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-vm-code.tar.gz`
- `C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-vm-code.tar.gz.sha256`

The archive includes both initial checkpoints, code, fixed recipe, MIT source
license and pinned package list. It adds files to the **existing** completed
coverage workspace at `~/forensic-dgp/coverage_vm_bundle/`. That workspace must
still contain its original inventory, protocol, parent checkpoint, datasets and
`outputs/coverage_training_vm/replay_pixels.pth`. The Windows copies are under
`C:\xampp\htdocs\YEAR 4\Testing\`; the local cached replay copy is under
`outputs\downloaded_coverage\outputs\coverage_training_vm\replay_pixels.pth`.

## 2. Verify and extract in SSH

```bash
cd ~
sha256sum -c face-occlusion-vm-code.tar.gz.sha256 &&
test -d ~/forensic-dgp/coverage_vm_bundle &&
tar --keep-old-files -xzf face-occlusion-vm-code.tar.gz -C ~/forensic-dgp/coverage_vm_bundle &&
tmux new-session -A -s dgp_training
```

The checksum has Linux LF line endings. Extraction refuses existing files. This
bundle has not been pushed to Git; `git pull` alone cannot deliver it. If any
command fails, preserve its output and inspect the error before another run.

## 3. Install the isolated dependencies and run inside tmux

```bash
cd ~/forensic-dgp/coverage_vm_bundle
if [ -f ../feature_vm_bundle/.venv/bin/activate ]; then
  source ../feature_vm_bundle/.venv/bin/activate
elif [ -f ../venv/bin/activate ]; then
  source ../venv/bin/activate
fi
bash scripts/setup_face_occlusion_vm.sh &&
python3 -u scripts/train_face_occlusion_vm.py --preflight --output outputs/face_occlusion_preflight_vm &&
python3 -u scripts/train_face_occlusion_vm.py
```

Setup first checks the active Linux CUDA runtime, required existing packages and
GPU VRAM. It installs only five pinned packages with `--no-deps` into the new
`outputs/face_occlusion_dependencies/` directory. It does not create a venv or
install Torch. The same interpreter supplies the existing CUDA Torch/torchvision.
Network access to the Python package index is needed for these five packages.

Preflight verifies both old/new inventories, fixed replay membership and the
initialization hashes, then performs one mixed batch forward per arm with zero
optimizer updates. Full training runs exactly 210 updates per arm: ten epochs
of 21 batches of eight examples, 420 total updates. Candidate validation and
mask export occur only at epochs 1, 5 and 10. Runtime and peak VRAM remain to be
measured on the L4; no production quality claim follows from a successful run.

New preflight/training output directories and the return archive must be absent.
Setup also refuses an existing dependency target. Failed/partial runs are
preserved; no deletion, resume or automatic additional epoch is included.

## 4. Download the results after completion

Use Google Cloud SSH's Download File control with:

```text
/home/janusdominic0/forensic-dgp/coverage_vm_bundle/face-occlusion-results.tar.gz
```

Save as
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-results.tar.gz`.
The VM output directory is
`~/forensic-dgp/coverage_vm_bundle/outputs/face_occlusion_pilot_vm/`.
The archive contains all candidate states, parent/candidate masks, training-fit
and validation metrics, exact executed batch indices, environment/source hashes,
runtime/peak allocated and reserved CUDA memory, and completion status.
Initial weights stay in the sent bundle and are not
duplicated in the return archive. Three states per arm total approximately
345 MB before compression; metric-eligible best copies add space if selected.

## 5. Return-artifact decision

Independently verify code/init hashes and 420 executed updates; recount the
25 real and 400 synthetic masks at each of the six candidates. Reproduce masks
from the returned checkpoints and verify frozen heads/statistics and unchanged
parent/generator. Compare both arms' training fit, human/mannequin/glare subsets
and all unchanged selection guards. `best_detector.pth` exists only if its arm
passes both real and synthetic gates; filename alone does not authorize use.

Next: only a qualifying candidate advances to a reviewed ten-row end-to-end
restoration/completion grid. Clear facial regions must remain preserved;
covered features and strong lens reflections receive plausible estimates.
If neither arm qualifies, record that result before choosing another intervention.
