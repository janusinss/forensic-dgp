# Pretrained face completion experiment

The next approach is a pretrained generator with an explicit, reviewable covering mask, optional visible-region restoration, and exact copying outside the mask. The existing detector can suggest a mask; a user can correct it. Hidden features remain estimates from one input image.

## Implemented

- `pretrained_completion.py`: official CodeFormer **inpainting** architecture/weights, explicit binary masks, RGB normalization to [-1,1], 512px internal inference with `w=1, adain=False`, exact composition at the caller's resolution, empty-mask bypass and unsupported/nonfinite-output rejection.
- `third_party/codeformer`: minimal official network source pinned to `b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`. Only import wiring/logger were adapted. No full BasicSR installation is needed. License and attribution are retained.
- `scripts/download_completion_weights.py`: official release download, SHA256 check and atomic installation. Existing mismatching weights are not overwritten. An interrupted `.part` file is retained for inspection; remove that incomplete file before retrying.
- `run_completion_benchmark.py`: frozen manifest, oracle or predicted masks, identical naming, checkpoint/manifest provenance, per-case times, errors and region scores. Inference reads input and mask only; the scoring step reads targets. Failed inference never substitutes the input as a fake success.
- `completion_web.py`: opt-in CodeFormer backend. With no detector configured, manual masks still work. The original restoration app is unchanged.

Default custom completion blending now uses radius 0. A correct mask therefore cannot fail visible preservation merely because a three-pixel external feather band was added. Detector false positives can still change truly visible pixels; selection thresholds remain strict. New training configurations carry `mask-only-v1` to prevent silently resuming an old experiment with different behavior.

The pretrained adapter uses **visible-support-normalized resizing**: it resizes `input * visible_mask` and divides by resized visible support before filling the resized hole white. Ordinary interpolation can leak the covering color into context; replacing the hole white before ordinary interpolation can bleach that context instead. Regression tests cover both effects. Run provenance records `input_policy=visible-normalized-resize-v1`. Earlier development outputs without this field are superseded and must not be used for selection.

## Start the local experimental page (PowerShell)

From the project directory:

```powershell
venv/Scripts/python.exe scripts/download_completion_weights.py
$env:COMPLETION_BACKEND = 'codeformer'
$env:CODEFORMER_CHECKPOINT = 'checkpoints/codeformer_inpainting.pth'
$env:COMPLETION_CHECKPOINT = 'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
venv/Scripts/python.exe -m uvicorn completion_web:app --host 127.0.0.1 --port 8002
```

Open http://127.0.0.1:8002. Use an upright, centered, aligned face crop with level eyes and the full chin/forehead. Review the orange region before generating. Leave visible restoration unchecked for clear images. Mask inference uses the custom detector's original resolution, then resizes its probability map to the generator resolution. The page preserves the crop resolution for composition; the generator runs at 512 internally. The display canvas is only a preview, so its resizing does not contaminate the uploaded model input. Upsampling does not make a 128px input contain genuine 512px detail.

`COMPLETION_CHECKPOINT` is optional for this backend: omit it to paint regions manually. Clear it with `Remove-Item Env:COMPLETION_CHECKPOINT` if an old shell value is present. Default backend remains `custom` when `COMPLETION_BACKEND` is unset. These are separate experimental choices, not automatic deployment selection.

## Benchmark on the existing Google Cloud VM

First push these source changes from the local repository. Then in Google Cloud SSH:

```bash
cd ~/forensic-dgp
git pull --ff-only origin main
tmux new-session -A -s completion_benchmark
```

Inside tmux:

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
bash scripts/run_completion_benchmark_gcp.sh
```

This downloads approximately 360 MB of pretrained weights once and evaluates **20 FFHQ + 20 Asian-source validation images, each with 10 cases across 4 pipelines (1,600 inferences)**, not a training run. The four pipelines are custom/oracle mask, custom/predicted mask, CodeFormer/oracle mask and CodeFormer/predicted mask. Each uses the same frozen Phase 3 restoration on degraded cases. CodeFormer is run at 512 internally and composed/scored at the common 256px benchmark size. It does not claim higher-resolution ground truth.

Inputs default to both dataset folders, the original Phase 4 split, and `outputs/completion_pilot/epoch_2.pth` as detector/custom checkpoint. Override `DATA_DIR`, `SPLIT_FILE`, `DETECTOR`, `RESTORER`, `IMAGES_PER_SOURCE` or `BENCHMARK` if paths differ. It refuses an existing benchmark/output directory to prevent mixing results. To rerun, choose a new `BENCHMARK`. No new package dependencies beyond the existing project environment are required.

Sampling is balanced by dataset root: 20 validation images per source by default. Missing or insufficient source membership fails explicitly. Reports separate source datasets and covering/degradation groups, including failures. Asian-source membership is a dataset label, not proof of Filipino representation. Add manually annotated real coverings for the next decision-quality evaluation. The local eight-image pilot is a functional/visual comparison only.

## Interpretation and next training decision

The completed local pilot used the same eight FFHQ validation images (80 synthetic cases) for all four pipelines, with no inference failures. Covered-region MAE changed from 0.091788 to 0.068397 with known masks (25.5% lower) and from 0.096025 to 0.076386 with predicted masks (20.5% lower). Clear known-mask visible error is exactly zero. Corrected pretrained outputs contain recognizable facial structure instead of smooth patches. These results support further evaluation of this candidate, not deployment: the split was reused, only eight source images were tested, and real masks/Asian-source images were not tested locally. Full local artifacts are in `outputs/completion_pretrained_benchmark/REPORT.md`.

Inspect clear and degraded cases separately, and oracle versus predicted masks. Compare reference, input, old generator and pretrained output. Region MAE rewards smooth averages; realistic-looking anatomy may have worse MAE and still be wrong for the person. Neither metric improvement nor a convincing mouth proves the true hidden appearance.

Do not select automatically or resume the old two-epoch run. First review boundary seams, leftover covering, visible-face preservation, plausible anatomy, independent identity consistency and real covering detection. If completion is useful with corrected masks but fails with automatic masks, improve the detector/data rather than retraining the generator blindly. Boundary color mismatch is a separate compositing issue and requires controlled evaluation before applying correction.

CodeFormer uses the non-commercial S-Lab License 1.0, included with the source. This integration is a research experiment. Other models have not been exhaustively compared, so this is a tested candidate rather than a claim of universally best performance.

Official references: [inpainting implementation](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/inference_inpainting.py), [repository/license](https://github.com/sczhou/CodeFormer).
