# Existing VM: matched region-weighting pilot

The package is prepared locally; its new CUDA run is unverified. This trains
the Track2 **occlusion detector**, using the verified epoch30 model and exact
saved AdamW moments. There are two matched arms, 210 updates each. The fixed
recipe is `FACE_OCCLUSION_FOCUS.md`. Actual training runs only on the VM.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`; execution root: `~/forensic-dgp/coverage_vm_bundle/`.

## 1. Upload two files to VM home

- `C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-focus-code.tar.gz`
- `C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-focus-code.tar.gz.sha256`

Upload using Google Cloud SSH's Upload File control. Source weights, cached
replay, reviewed datasets and dependencies already exist in the completed VM
bundle. The archive adds new files only; all previous inventoried files are
preserved. These new source files have not been committed or pushed. `git pull`
does not install this prepared package in the isolated execution directory.

## 2. Paste into SSH

```bash
cd ~
sha256sum -c face-occlusion-focus-code.tar.gz.sha256 &&
test -d ~/forensic-dgp/coverage_vm_bundle &&
tar --keep-old-files -xzf face-occlusion-focus-code.tar.gz -C ~/forensic-dgp/coverage_vm_bundle &&
tmux new-session -A -s dgp_training
```

The checksum file uses LF line endings. Extraction rejects any existing member.
If the session already exists, tmux attaches to it; use its shell prompt after
the previous run has finished.

## 3. Paste inside tmux

```bash
cd ~/forensic-dgp/coverage_vm_bundle
if [ -f ../feature_vm_bundle/.venv/bin/activate ]; then
  source ../feature_vm_bundle/.venv/bin/activate
elif [ -f ../venv/bin/activate ]; then
  source ../venv/bin/activate
fi
nvidia-smi &&
python3 -u scripts/train_face_occlusion_focus_vm.py --preflight --output outputs/face_occlusion_focus_preflight_vm &&
python3 -u scripts/train_face_occlusion_focus_vm.py
```

Reuse the existing CUDA environment and isolated packages; no installation is
needed. Preflight requires Linux/CUDA and verifies previous/new input hashes,
datasets/schedule, saved source/moments and every training weight map. It runs
one mixed forward with zero updates and must print:

```text
CUDA forward and source/moments/maps passed; zero optimizer updates
```

The full run rechecks source/baseline metrics before its 420 total new updates.
Each arm saves global epochs35 and40; saved optimizer steps are525 and630.
Original real/synthetic gates remain unchanged. This is separate from the older
restoration epochs27–31 and does not train `dgp_zamboanga_final.pth`.

The prior 420-update L4 continuation reported99.16 seconds excluding archive
compression. Allow roughly5–15 minutes for the new run's verification, exports
and compression; the new wall time has not been measured. Keep about2GB free
for checkpoint/optimizer files plus their archive. A missing `best_detector.pth`
means no candidate passed both guards; it does not mean the script canceled.

Press Ctrl+B, then D to detach while the VM continues. Attach later with:

```bash
tmux attach-session -t dgp_training
```

If any command fails, preserve its exact output. Do not delete results, relax
guards or restart into an existing output directory.

## 4. Download after completion

The last line prints the completed archive path. In Google Cloud SSH's Download
File control, paste:

```text
/home/janusdominic0/forensic-dgp/coverage_vm_bundle/face-occlusion-focus-results.tar.gz
```

Save locally as:

```text
C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-focus-results.tar.gz
```

Also download the same VM path with `.sha256` appended to verify transfer.
Model/log directories are `~/forensic-dgp/coverage_vm_bundle/outputs/face_occlusion_focus_vm/`
and, after safe extraction, local
`C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_face_occlusion_focus\outputs\face_occlusion_focus_vm\`.

## 5. Next after return

Audit both matched arms, all420 logged steps, 2,915 masks, source/restored/final
optimizer bindings, unchanged states and original gates. Inspect the ten-row
grid and reflection zoom. Only an eligible detector advances to reviewed
end-to-end completion. Track1 Phase3 restoration and Track2 generator/application
baselines remain retained until their outputs meet the separate quality checks.

The goal stays unmet if the run merely completes. No new training recipe is
chosen from an unverified archive or from training metrics alone.
