# Existing VM: fixed additional-fitting pilot

Four runner tests and three replay diagnostic tests pass. Actual CUDA execution
of this new runner is unverified. This pilot starts from the **trained direct
occlusion detector**'s pretrained epoch10 state; it does not train the restoration
generator or use `dgp_zamboanga_final.pth` as its trainable initialization.
The fixed recipe is`FACE_OCCLUSION_CONTINUATION.md`.

## 1. Upload two small files to VM home

- `C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-continuation-code.tar.gz`
- `C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-continuation-code.tar.gz.sha256`

All source weights, datasets, cached replay and isolated dependencies are reused
from the completed pilot in`~/forensic-dgp/coverage_vm_bundle/`. These new files
have not been pushed; a git pull alone does not install the runner.

## 2. Paste into SSH

```bash
cd ~
sha256sum -c face-occlusion-continuation-code.tar.gz.sha256 &&
test -d ~/forensic-dgp/coverage_vm_bundle &&
tar --keep-old-files -xzf face-occlusion-continuation-code.tar.gz -C ~/forensic-dgp/coverage_vm_bundle &&
tmux new-session -A -s dgp_training
```

The checksum uses Linux LF. Extraction refuses existing files.

## 3. Paste inside tmux

```bash
cd ~/forensic-dgp/coverage_vm_bundle
if [ -f ../feature_vm_bundle/.venv/bin/activate ]; then
  source ../feature_vm_bundle/.venv/bin/activate
elif [ -f ../venv/bin/activate ]; then
  source ../venv/bin/activate
fi
python3 -u scripts/train_face_occlusion_continuation_vm.py --preflight --output outputs/face_occlusion_continuation_preflight_vm &&
python3 -u scripts/train_face_occlusion_continuation_vm.py
```

Use the existing isolated packages; there is no setup/install command. Preflight
performs one mixed CUDA forward with zero additional updates. Full execution
requires the starting source metrics to reproduce their previous values, then
performs exactly420 new optimizer steps/20 additional epochs. AdamW starts fresh
because the pilot's optimizer state was unavailable. Global epochs end at30;
this is a different detector from the older restoration epochs27–31.

Candidate checkpoints are`epoch_11.pth`,`epoch_15.pth`,`epoch_20.pth`,`epoch_30.pth`.
An eligible best model is saved only if both unchanged gates pass; no automatic
application change occurs. The final optimizer state is exported separately for
reproducibility. Source initial state and prior artifacts are preserved.

## 4. Download after completion

Use Google Cloud SSH's Download File control:

```text
/home/janusdominic0/forensic-dgp/coverage_vm_bundle/face-occlusion-continuation-results.tar.gz
```

Save locally as
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-continuation-results.tar.gz`.
The VM result directory is
`~/forensic-dgp/coverage_vm_bundle/outputs/face_occlusion_continuation_vm/`.

## 5. Return audit

Recount all candidate masks, verify both update counters/source/checkpoint states
and final optimizer binding, inspect unchanged real/synthetic gates and glare,
then perform an end-to-end completion review only for an eligible detector.
If any command fails, preserve its output/error; this runner does not retry or
resume existing directories. The overall improvement goal is not complete merely
because this VM run finishes.
