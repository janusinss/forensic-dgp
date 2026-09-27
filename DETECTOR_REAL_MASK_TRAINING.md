# Real-mask detector pilot

## Ready-to-run balanced training experiment

The updated workflow is ready for an experimental VM run. This is not a validated production detector. `scripts/run_real_detector_gcp.sh` runs two matched 10-epoch arms from the original epoch-2 checkpoint: balanced covered/uncovered batches with the existing BCE+Dice loss, and the same configuration plus a visible-background penalty (weight .25, hardest 10% of visible pixels per image). These are initial experimental hyperparameters, not an established optimum. Both arms use learning rate 1e-5, batch size 4 and seed 42. At this dataset size, each arm has 21 batches per epoch (210 updates); minority examples are reshuffled and cycled to maintain equal counts.

The penalty excludes covered target pixels. Validation/test loading is unbalanced and unaugmented; labels and distributions remain unchanged. Generator weights stay frozen and are checked after every epoch. The existing selection gates stay unchanged. Test data are not evaluated automatically. Compare the two arms before attributing any gain to the added loss.

### Transfer and start

Commit and push the code changes first. Upload `outputs/detector-reviewed-100.tar.gz` (about 9 MB) using Google Cloud SSH's Upload File button; it normally arrives in the VM home directory. The archive contains the manifest and all 100 reviewed image/mask pairs; data are gitignored and are not transferred by Git.

In the VM SSH terminal:

```bash
cd ~/forensic-dgp
git pull origin main
test ! -e dataset/detector_expanded_review && tar -xzf ~/detector-reviewed-100.tar.gz
tmux new -s detector_training
```

Only extract into a new dataset directory. If it already exists, verify it with `--validate_only` rather than overwriting it. Inside tmux:

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
bash scripts/run_real_detector_gcp.sh
```

The default starting checkpoint is `outputs/completion_pilot/epoch_2.pth`. Override `CHECKPOINT` if its VM path differs. `MANIFEST`, `OUTPUT`, `EPOCHS` and `PYTHON_CMD` are also configurable. Output defaults to `outputs/real_detector_balanced_vm` and must not exist already. This starts a fresh experiment, not optimizer-state resumption. Detach from tmux with Ctrl+B, then D.

When both arms finish:

```bash
tar -czf detector-balanced-results.tar.gz outputs/real_detector_balanced_vm
realpath detector-balanced-results.tar.gz
```

Use the SSH Download File button with the printed path. `cloudshell download` is specific to Cloud Shell and should not be assumed available inside the Compute Engine VM SSH session.

Each arm saves `detector_epoch_N.pth`, `metrics.jsonl` and `run.json`. `best_detector.pth` appears only if a checkpoint passes validation selection; it may be absent. These outputs are detector-only fine-tuning exports compatible with the completion loader; do not replace the deployed model until visual and held-out checks pass.

The first real-covering examples showed whole-mask misses. The current synthetic detector needs real mask supervision; one-pixel expansion does not solve this. `detector_training.py` fine-tunes only `CompletionNet.segmenter` and asserts the generator remains bitwise unchanged. CodeFormer is not loaded or trained.

## Current review package

Expanded pilot update: `dataset/detector_expanded_review` now holds 100 assistant-annotated crops, split 68/25/7. `outputs/detector-reviewed-100.tar.gz` packages the manifest and image/mask pairs for separate VM transfer. A three-epoch detector-only run from the original checkpoint reached validation IoU .2335 versus .0604 initially, but missed 70.6% of covered pixels and increased uncovered false-positive cases from 1/11 to 5/11. No checkpoint qualified; stricter diagnostic thresholds also failed the gates. The next experiment should address false positives through balanced batches and a visible-background penalty, with a longer bounded pilot budget and unchanged test reservation. See `outputs/real_detector_expanded_pilot/REPORT.md`. Do not deploy these epoch checkpoints.

September 27 update: a separate `dataset/detector_pilot_review` package now contains 28 assistant-annotated crops (16 train, 5 validation, 7 reserved test). Polygons were inspected visually and the reviewer is recorded explicitly; these are not expert-reviewed labels. A real three-epoch CPU detector pilot completed in `outputs/real_detector_micro_pilot`. Mask IoU rose from .0568 to .1413, but 85.8% of covered pixels remained missed and visible false positives increased. Every checkpoint was rejected by selection; no app weights were replaced. The next step is a larger, more diverse reviewed development/validation set, not deployment or repeating the same tiny run on the VM. Full findings: `outputs/real_detector_micro_pilot/REPORT.md`.

Run `python scripts/prepare_detector_review.py` once to export the 18 diagnostic crops and automatic mask proposals into `dataset/detector_review`. Existing outputs are refused. Every proposal starts unreviewed. The four earlier coarse polygons are deliberately not approved or included as labels.

For each usable crop, paint a matching grayscale PNG: white (255) for the complete covering to replace, black (0) for preserved regions. Include covering edges and straps when they must be removed; preserve visible eyes/skin. Reject or recrop padded/distorted images, very small faces and ambiguous multiple-person crops. These initial candidates have already been inspected and must be development data, not an untouched final test.

Add new covered crops and verified uncovered negative examples. Negative masks must be completely black. For a first pilot, aim for at least 60 reviewed crops with roughly 20 uncovered controls, distributed across train/validation/test; this is a practical starting point, not evidence that this count suffices. Keep all photos of the same known subject/source sequence and duplicates in one split. Where identity or provenance is uncertain, keep related images together rather than claiming identity-disjoint evaluation. Near-duplicate review is still manual.

Edit `manifest.json` records after visual review:

- `kind`: `covered` or `uncovered`; `reviewed`: true only after review.
- `group`: a nonempty stable subject/source group; never infer identity from numeric filenames.
- `split`: `train`, `validation` or `test`. Every split needs both kinds. Do not use previously inspected examples for the final test.
- `image_sha256` and `mask_sha256`: SHA256 of the final image and reviewed PNG. Hashes must be updated deliberately if files change. PowerShell: `(Get-FileHash -Algorithm SHA256 'path/to/file.png').Hash.ToLower()`.

Paths must be relative to the manifest directory. Remove rejected candidates from the manifest; leave original source files intact. The trainer rejects pending records, hash changes, nonbinary masks, duplicate image bytes and cross-split declared group leakage. It cannot verify that a human label or group assignment is correct.

## Validate before VM training

```bash
python detector_training.py --manifest dataset/detector_review/manifest.json --checkpoint outputs/completion_pilot/epoch_2.pth --output_dir outputs/real_detector_pilot --validate_only
```

Validation does not load the checkpoint or train. Transfer the entire reviewed directory to the VM; dataset files are gitignored and do not arrive with `git pull`.

After validation succeeds, a short VM pilot is:

```bash
python detector_training.py --manifest dataset/detector_review/manifest.json --checkpoint outputs/completion_pilot/epoch_2.pth --output_dir outputs/real_detector_pilot --epochs 3 --lr 0.00001
```

This is an initialization from the existing detector, not a continuation of generator epochs. It does not resume interrupted optimizer state; choose a new output for a new pilot. Validation runs before training and after each epoch. The test split is validated for file integrity but never used for optimization or checkpoint selection. Outputs retain the existing completion checkpoint format, with detector-only provenance and a separate fine-tune epoch field.

`best_detector.pth` is saved only when validation IoU improves without increasing visible-pixel false positives, empty-mask failures or negative-image false-positive counts relative to the initial detector. A best checkpoint may not exist. `detector_epoch_N.pth` is always saved after a completed epoch. These numerical gates are not deployment approval.

## Next step

Finish accurate mask annotation and group-based data separation first. Do not train the current pending package. After a short detector pilot, inspect output masks on the untouched test set and use the frozen completion pipeline to check real covering residue and visible-face preservation. Accurate-mask completion and crop alignment must still be investigated separately; covered photos are not clean reconstruction targets.
