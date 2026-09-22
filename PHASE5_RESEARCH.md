# Phase 5: identity-aware restoration pilot

Prepared 22 September 2026. This is a research experiment, not a claim of improved output before training.

## What Phase 4 taught us

The 4,000-image cloud validation selected epoch 27 by PSNR. PSNR rose from 20.2795 to 20.6097 and SSIM from 0.6603 to 0.6700, while ArcFace similarity fell from 0.3568 to 0.3487. Every Phase 4 epoch scored below the starting model on that identity metric. A local 64-image check using a shared set of detected pairs reproduced the tradeoff. The deployed Phase 3 model remains unchanged.

## Research and application

| Evidence | Application here | Limit |
|---|---|---|
| [RestoreFormer, CVPR 2022, section 3.3](https://arxiv.org/html/2201.06374v2) uses pretrained ArcFace features in an identity loss. | Add frozen ArcFace cosine loss to the existing restoration objective. Gradients reach the restored image; the recognizer is never trained. | This adopts a loss concept, not RestoreFormer's architecture or reported performance. Identity embeddings do not prove recovery of true missing details. |
| [InsightFace's recognition implementation](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py) specifies RGB normalization; its [alignment code](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py) defines the five-point 112px template. | Use the existing w600k_r50.onnx weights, converted to PyTorch with [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch). Align reference and restored image with the SAME reference-landmark transform via differentiable sampling. | FAN eye-center landmarks approximate the five-point detector landmarks. This is a new fixed-alignment metric, not directly interchangeable with Phase 4 scores. |
| [PyTorch's EMA recipe](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/) averages model weights for stability. | Maintain an exponential moving average of unique model parameters and buffers. Validate/export EMA weights; preserve raw weights and optimizer state for continuation. | Its published gains are from classification. Restoration benefit and decay 0.999 must be tested here. |
| [Blau and Michaeli, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html) explains why distortion and perceived quality need not improve together. | Stop selecting checkpoints by PSNR alone. Require no baseline regression in identity similarity and SSIM, with checks for both dataset sources. | These are conservative experimental gates, not a guarantee of good visual quality. |
| [Official FFHQ documentation](https://github.com/NVlabs/ffhq-dataset) distinguishes 128px thumbnails from 1024px source images. | Record the current target-resolution limitation. Keep the current data for this controlled pilot; test genuine higher-resolution targets separately. | Enlarging a 128px image to 256px adds no new reference detail. Data amount alone cannot solve this. |

The new objective is the existing composite restoration loss plus `0.1 * mean(1 - cosine(ArcFace(restored), ArcFace(reference)))`. Missing or invalid reference landmarks contribute no identity term, and coverage is printed. The generated image is never detached before the identity loss. Target embeddings need no gradient. The existing FAN, VGG, color, Sobel and FFT losses remain active.

## The next experiment

Start from `checkpoints/dgp_zamboanga_final.pth`, which retained stronger measured identity similarity than Phase 4. This is a separate Phase 5 pilot, numbered epochs 1–2; it does not imply continuation from Phase 4 epoch 31.

- Keep BOTH datasets and the completed Phase 4 split, including all 4,000 validation images.
- Keep the existing degradation distribution, including 35% heavy primary blur.
- Use head learning rate 1e-5, backbone learning rate 2e-6, and identity weight 0.1.
- Use EMA decay 0.999. These are initial experiment settings, not an established optimum.
- Run two full epochs first. Review images and all metrics before authorizing a longer run.

No dataset-source reweighting is introduced yet. FFHQ-source and Asian-source results are reported separately. Source membership is not an ethnicity label; FFHQ itself contains varied demographics.

## Validation and checkpoint rules

Phase 5 reports `ArcFace_fixed`. It uses reference landmarks to crop both images in the same coordinates. This removes repeated output-face detection and fixes eligible pair membership for each run. Reference landmark detection is cached once, with the resized RGB pixels, image shape and detector package version in the cache key. Cache preparation uses the GPU in the launcher. Repeated epochs read cached coordinates.

`best.pth` starts as the Phase 3 baseline. A candidate replaces it only when:

1. PSNR exceeds the currently selected best.
2. Overall SSIM and fixed-alignment identity similarity do not fall below the baseline.
3. The number of valid identity pairs is unchanged.
4. Each dataset source passes its own baseline PSNR, SSIM and identity checks.

If no candidate passes, `best.pth` intentionally stays at the baseline. `best_selection.json` records the selected epoch and metrics; epoch zero means the starting baseline. Every EMA epoch is still exported for inspection. Tiny changes are not evidence of statistical significance.

The same recognition network is used in the training loss and this validation metric. Treat that metric as development feedback, not independent proof of identity preservation. Before deployment, rerun the legacy detection-based evaluation and an independent recognition/visual check on a separate, appropriately sourced Filipino test set. Earlier training may already have used the Phase 4 validation images. The inherited split is useful for regression comparison, not proof of unseen-identity performance.

## Run on the Google Cloud VM

Commit and push the new project files first. Inside the existing tmux session:

```bash
cd ~/forensic-dgp
git pull origin main
if [ -d venv ]; then source venv/bin/activate; fi
python3 -m pip install -r requirements-phase5.txt
bash scripts/run_phase5_gcp.sh
```

The launcher checks CUDA and ArcFace conversion, prepares cached landmarks with progress, evaluates the baseline, and runs two training/validation epochs. Cache preparation is additional first-run work. Runtime on the VM has not been benchmarked; do not assume the local smoke-test duration predicts full training time. The launcher fails if a dataset directory or the prior split is missing.

Outputs are saved to `outputs/phase5_identity/`:

- `best.pth` and `best_selection.json`: selected inference weights and selection record.
- `epoch_1.pth`, `epoch_2.pth`: EMA inference weights for each complete epoch.
- `last_state.pth`: raw model, EMA, optimizer, scheduler, baseline, selected metrics, configuration and random states.
- `metrics.jsonl`, `baseline.json`, `epoch_*.json`: aggregate and per-image measurements.
- `baseline.png`, `epoch_*.png`: degraded input, raw restoration and reference comparisons.

To resume an interrupted run at the last COMPLETED epoch:

```bash
bash scripts/run_phase5_gcp.sh --resume_state outputs/phase5_identity/last_state.pth
```

Work inside an interrupted partial epoch is repeated. A resume keeps the original baseline and EMA. It does not recompute the baseline or reset checkpoint-selection thresholds.

For an ablation, run a separate output directory with `--lambda_identity 0`, keeping all other settings and the data split identical. This distinguishes the contribution of identity supervision from EMA and lower learning rates; do not mix ablation results into the same output directory.

## Verification performed locally

The converted ArcFace model matched ONNX Runtime on a two-image test with maximum absolute embedding error 1.31e-6. Nonzero gradients reached the image, and recognition weights remained frozen. A real CPU smoke test completed loss computation, backpropagation, validation, checkpoint saving and resume, with the EMA restored. Its slightly lower identity score was correctly rejected despite increased PSNR. Those were single-batch tests, not trained quality results. CUDA execution is checked by the launcher on the VM because local PyTorch is CPU-only.

## Data suitability

FFHQ's current documentation explicitly excludes development or improvement of facial-recognition technologies and describes its licenses. This pilot freezes the recognizer and trains restoration, but that does not establish suitability for the proposed school identification use. Resolve permitted data use before applying the system to that use case. No new datasets or model weights were downloaded for this pilot; the existing ArcFace weights were reused.
