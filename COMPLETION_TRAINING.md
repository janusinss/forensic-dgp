# Face completion: current implementation and training

25 September 2026. Work continues in this workspace; the planned workspace handoff is canceled.

## What is implemented

This is a **new trainable baseline**, not a pretrained completion model or a demonstrated output-quality improvement. It uses the retained Phase 3 restoration model for degraded visible areas, a new gated U-Net for completion, and a separate gated U-Net for predicting covered regions. No second photo of the person is used at inference.

| File | Purpose |
|---|---|
| `completion.py`, `completion_data.py` | Model, masks, composition, deterministic synthetic training pairs |
| `train_completion.py`, `scripts/run_completion_gcp.sh` | Separate training, validation, EMA, full-state resume and inference exports |
| `completion_inference.py`, `completion_web.py` | Strict checkpoint loading and inference without target information |
| `templates/completion.html` | Single-image face crop, region estimation/painting and generated-region display |
| `completion_benchmark.py`, `tests/test_completion.py` | Fixed external benchmark export/scoring and automated contracts |

The existing `app.py`, restoration architecture, Phase 4/5 launchers and deployed Phase 3 checkpoint have not been replaced. New completion weights use a versioned format and cannot be confused with an old restoration weight file.

## Actual scope and remaining research

The implemented synthetic coverings are lower-face polygons, eye bars, rectangular objects and free-form strokes, plus uncovered controls. They vary in position/color/texture. They are **not photorealistic hands, scarves or transparent glasses**, and synthetic segmentation is not established as reliable real-world occlusion detection. The interface provides region correction for that reason.

The new completion model starts from random weights. The visible-region restorer remains frozen. Training uses area-normalized hole reconstruction, auxiliary visible-context reconstruction, segmentation BCE/Dice and optional VGG perceptual supervision. It does not yet include adversarial training, a pretrained generative face prior or an independent identity metric. Phase 5 ArcFace was not silently carried over as proof of completion fidelity. This baseline may produce smooth average features; compare it with pretrained candidates before making quality claims.

The research priorities in [FACE_COMPLETION_RESEARCH.md](FACE_COMPLETION_RESEARCH.md) remain applicable. Benchmark export/scoring is implemented; external CodeFormer/LaMa environments and pretrained comparisons have not been run in this workspace. Their outputs must be evaluated before declaring a winner.

## Data behavior

Training samples each aligned face once per epoch, selecting one of five coverage kinds and clear/degraded conditions reproducibly. Coverings are applied **before** Gaussian blur, downsampling, noise and JPEG compression. This initial degradation recipe is intentionally separate from the old stronger CCTV recipe.

Validation uses ten fixed cases per reference face: all five coverage kinds, each clear and degraded. With the old 4,000-image validation membership, that means **40,000 validation cases per pass**, not 4,000. Validation seeds are unchanged by training epoch. Content hashes detect modified images on resume; startup reads the dataset to compute these hashes.

Masks use white/1 for generation. For degraded images the synthetic geometric mask is dilated to approximate blur contamination; this is a conservative heuristic, not an exact physical support calculation. An additional bounded blend band is exported as part of the generated region. Outside that band the completion stage copies the visible-region image exactly. If visible restoration is enabled, that image is the restorer output, not the original input.

The default training resolution is 256. Original 128px targets remain a data limitation. The current split is image-disjoint, not proven identity-disjoint or unseen by historical phases. Do not report it as an independent Filipino completion benchmark.

## First VM check: one batch only

Commit/push the intended files before pulling on the VM. In the existing SSH terminal:

```bash
cd ~/forensic-dgp
git pull --ff-only origin main
if [ -d venv ]; then source venv/bin/activate; fi
python3 -m pip check
nvidia-smi
tmux new-session -A -s dgp_training
```

Inside tmux, reactivate the environment if needed, then run the CUDA mechanics check:

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
OUTPUT_DIR=outputs/completion_gpu_smoke bash scripts/run_completion_gcp.sh --dry_run --batch_size 1 --num_workers 0
```

This uses both dataset directories and the prior `outputs/phase4_with_progress/split.json`. Override `DATA_DIR`, `SPLIT_FILE` or `RESTORATION_CHECKPOINT` only when paths differ. No new dependency beyond the main project requirements is added. VGG weights can download on first use if not already cached.

Check the ten-row preview and logs. Smoke checkpoints are explicitly marked and the web app rejects them. A smoke run proves mechanics, not quality or GPU capacity at batch size 8.

## Full two-epoch baseline pilot

After the GPU check and dataset inspection, use a different output directory:

```bash
OUTPUT_DIR=outputs/completion_pilot bash scripts/run_completion_gcp.sh
```

Defaults: two epochs, batch 8, two workers, width 24, learning rate `1e-4`, perceptual weight `0.05`, EMA decay `0.99`. These are initial baseline settings, not an established optimum. Do not assume two epochs from scratch will produce convincing hidden features. GPU runtime/peak memory have not been measured locally.

Resume at the last completed epoch:

```bash
OUTPUT_DIR=outputs/completion_pilot bash scripts/run_completion_gcp.sh --resume_state outputs/completion_pilot/last_state.pth
```

Preserve original options on resume, including any batch or loss overrides. Partial-epoch work is repeated. Changing the total epoch count is rejected because it changes the learning-rate schedule. Choose a fresh experiment for changed settings; the current trainer does not implement weight-only warm-start for a new completion experiment.

Detach with Ctrl+B, then D. Do not run a second GPU training process in the same session. No cloud training was launched from the local implementation session.

## Read outputs correctly

| Output | Meaning |
|---|---|
| `epoch_1.pth`, `epoch_2.pth` | Versioned EMA inference checkpoints |
| `last_state.pth` | Raw model, EMA, optimizer, scheduler, RNG, configuration and selection state |
| `baseline.json`, `epoch_*.json`, `metrics.jsonl` | Per-case and aggregate predicted-mask/known-mask errors, source/type groups, segmentation IoU and unsupported coverage |
| `baseline.png`, `epoch_*.png` | Rows: input, known mask, known-mask completion, predicted-mask completion, clean target |
| `best.pth`, `best_selection.json` | Preliminary selected candidate; `best.pth` is absent if no trained epoch passes |

Selection requires lower predicted-mask hole MAE than the current best, no visible MAE increase over baseline, no worse aggregate segmentation IoU, no cases above 85% predicted coverage, and no visible-region regression in reported coverage/condition or source groups. Epoch zero in the selection record means restoration-only baseline and **no selected completion model**. These checks do not establish identity preservation; inspect outputs and perform independent identity/visual evaluation before deployment.

## Fixed pretrained comparison inputs

From the repository root, export a first set of 20 reference images × 10 cases (200 cases):

```bash
python3 completion_benchmark.py prepare \
  --data_dir dataset/thumbnails128x128,dataset/asian_faces \
  --split_file outputs/phase4_with_progress/split.json \
  --images 20 --size 512 --output_dir outputs/completion_benchmark
```

Folders contain `input/`, `target/`, `mask/`, and `codeformer_input/` plus a manifest with seeds and source hashes. `codeformer_input/` uses the official white-hole convention. The supplied script requires aligned 512×512 faces; these exports assume the dataset faces are already appropriately cropped/aligned. Resizing targets does not create additional detail. For a final benchmark, use a genuinely separate test split and high-resolution references.

In an isolated environment with the official CodeFormer repository and dependencies installed, its published inference entry point accepts:

```bash
python inference_inpainting.py -i /absolute/path/to/completion_benchmark/codeformer_input -o /absolute/path/to/codeformer_predictions
```

This is an external command, not verified in the current environment. Preserve the explicit masks when composing external outputs; the original script can interpret naturally pure-white visible pixels as holes. LaMa requires its own input adapter/environment; neither library is installed into the DGP environment by this change.

Score same-named PNG outputs at the benchmark's exact resolution:

```bash
python3 completion_benchmark.py score \
  --benchmark outputs/completion_benchmark \
  --predictions /absolute/path/to/predictions \
  --report outputs/completion_comparison.json
```

Missing/invalid predictions are counted and produce a nonzero exit. Region MAEs are area-normalized; empty hole regions are excluded from hole averages. These metrics are deliberately labeled as pixel errors, not perceptual/identity fidelity. LPIPS, independent identity evaluation, blinded comparisons and statistical intervals remain research work.

## Test a downloaded trained model locally

After the VM pilot finishes, archive its complete run there:

```bash
cd ~/forensic-dgp
tar -czf /tmp/completion-results.tar.gz outputs/completion_pilot
sha256sum /tmp/completion-results.tar.gz
```

Download `/tmp/completion-results.tar.gz` using SSH-in-browser. Extract locally into a fresh staging directory and compare its hash. Select an actual trained epoch for inspection; it need not have passed `best` selection, but report that distinction.

PowerShell, using the real extracted checkpoint path:

```powershell
$env:COMPLETION_CHECKPOINT = 'C:\absolute\path\to\epoch_2.pth'
venv/Scripts/python.exe -m uvicorn completion_web:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001`. Upload one image, choose a square face crop, estimate/paint/correct its covered region, and enable visible restoration only when needed. Output is a face crop; full-scene automatic alignment and paste-back are not implemented in this experimental interface. A manually supplied region replaces automatic segmentation. Near-total coverage is rejected; no ground-truth image or landmarks are read at inference.

## Verification recorded locally

The implementation session exercised unit tests for compositing, invalid masks, empty regions, deterministic coverings, hidden-pixel erasure, differentiable training, inference, smoke-checkpoint refusal, upload validation and missing benchmark outputs. Existing unit tests were also run. A 64px CPU smoke run saved and resumed across two one-batch iterations; a native 256px CPU smoke run used the VGG perceptual term. These are mechanics tests only. The browser's upload/crop/edit flow and missing-checkpoint message were checked.

A 50-step single-case overfit check reduced hole MAE from 0.2159 to 0.0608. This demonstrates learning on one training example, not generalization. A real-image benchmark export produced ten cases; scoring the references against themselves returned zero error as expected. Exporting the full cloud validation split against the locally incomplete dataset correctly failed membership validation rather than silently dropping the missing Asian-source images.

Run the automated checks again after changes:

```powershell
venv/Scripts/python.exe -m unittest discover -s tests -p 'test_*.py'
```

Next quality work is the GPU smoke, fixed pretrained comparison, real covering data/segmentation evaluation, and a bounded training pilot. Do not replace Phase 3 based on these local smoke outputs.
