# Forensic DGP: workspace handoff and training runbook

Prepared 23 September 2026. Covers both an existing Google Cloud VM and a fresh VM.

**25 September: workspace handoff canceled by the user.** Continue implementation and training here. This file remains a historical setup reference. Current completion implementation and commands are in [COMPLETION_TRAINING.md](COMPLETION_TRAINING.md).

**Scope update, 25 September 2026:** the user confirmed single-image restoration plus completion of facial regions hidden by masks or other objects. Hidden features are plausible estimates; visible degraded regions may also be restored. Read [FACE_COMPLETION_RESEARCH.md](FACE_COMPLETION_RESEARCH.md) for the researched plan and benchmark sequence, and the current runbook linked above for the implemented baseline. The Phase 5 launcher below remains restoration-only. Historical results and VM instructions below remain relevant.

## 1. Start here

The goal is better face-restoration output for a Philippine school setting, especially fidelity to the actual person. Higher sharpness or PSNR alone is insufficient. The user wants to train again after reviewing and implementing improvements.

**Current decision:** retain `checkpoints/dgp_zamboanga_final.pth` as the application baseline. Phase 4 completed, but its identity metric declined. Phase 5 is implemented and passed local tests; full GPU training and output-quality verification remain outstanding.

| Item | Verified state at handoff |
|---|---|
| Repository | https://github.com/janusinss/forensic-dgp |
| Local project | `C:\xampp\htdocs\YEAR 4\Testing` |
| Existing VM project directory | `~/forensic-dgp` |
| Local code revision before this document | `67a0ba1` (`changes`); remote push status not independently verified |
| Existing training hardware | Screenshot reported NVIDIA T4 and about 16 GB host RAM; exact machine type, project ID and zone are unverified |
| Local training runtime | Windows virtual environment, CPU-only PyTorch; not a substitute for GPU verification |
| Existing tmux session | `dgp_training` |

Read this document, [Phase 5 research](PHASE5_RESEARCH.md), and the actual training code before continuing. [The original understanding document](PROJECT_UNDERSTANDING_AND_IMPROVEMENTS.txt) provides historical context but includes outdated claims. This handoff records verified limitations explicitly.

## 2. What the project actually does

The restoration model is a feedforward residual generator with a MobileNetV2 feature pyramid, compatible with the project's DeblurGAN-v2-derived weights. It accepts RGB tensors in `[0,1]`, normalizes internally, and produces 256×256 restoration outputs. It is not a newly implemented diffusion model.

| File | Responsibility |
|---|---|
| `models/dgp_synthesizer.py` | Restoration architecture |
| `dataset.py`, `dataloader.py` | Images, synthetic degradation, landmarks and data splitting |
| `train.py`, `evaluation.py`, `training_state.py` | Phase 4 training, evaluation and checkpoint state |
| `train_phase5.py`, `phase5_utils.py`, `models/identity_loss.py` | Phase 5 identity supervision, EMA and selection |
| `app.py` | FastAPI application, preprocessing, restoration and display postprocessing |

Training creates degraded inputs from reference faces. The application additionally uses preprocessing, alignment/cropping, contrast adjustments and sharpening. Evaluate raw model output separately from these display changes.

Corrections to earlier descriptions:

1. FAN supervision is heatmap MSE, not an explicit landmark Euclidean constraint or a guarantee against hallucination.
2. A recurrent second restoration pass is not implemented merely because an old document describes one.
3. Application candidate scores are hardcoded display values, not measured loss, confidence or identity probabilities.
4. A restored face is an estimate. Current tests do not establish forensic admissibility or recovery of details absent from the input.
5. The application prioritizes `checkpoints/dgp_zamboanga_final.pth`. Saving `outputs/.../best.pth` does not automatically deploy it; its epoch fallback search also does not include epoch 31.

## 3. Completed work and measured results

### Historical training

The earlier project history describes Phase 1 as epochs 1–10, Phase 2 as 11–20, and Phase 3 as 21–26. Phase 3's named output is `dgp_zamboanga_final.pth`. Older reported Phase 2 metrics were training-batch measurements and must not be compared directly with the later held-out validation.

Phase 4 continued from Phase 3 through epochs 27–31. Changes included deterministic degradation, 35% heavy primary blur, a fixed 5% validation split, validation progress reporting, gradient clipping, complete training-state saves, and PSNR-based best-checkpoint selection. The existing component, VGG, color, FAN, Sobel and FFT losses remained active.

The actual cloud output directory was `outputs/phase4_with_progress`, although the Phase 4 launcher defaults to `outputs/phase4`.

### Full Phase 4 cloud validation

| Checkpoint | PSNR | SSIM | ArcFace similarity | Valid identity pairs |
|---|---:|---:|---:|---:|
| Phase 3 baseline | 20.2795 | 0.6603 | 0.3568 | 3886/4000 |
| Epoch 27 | 20.6097 | 0.6700 | 0.3487 | 3887/4000 |
| Epoch 28 | 20.0884 | 0.6541 | 0.3410 | 3904/4000 |
| Epoch 29 | 20.4924 | 0.6659 | 0.3461 | 3893/4000 |
| Epoch 30 | 20.1700 | 0.6532 | 0.3505 | 3881/4000 |
| Epoch 31 | 20.4048 | 0.6645 | 0.3357 | 3911/4000 |

`best.pth` contains exactly the same tensors as epoch 27. File hashes differ because serialization can differ. The full `last_state.pth` records epoch 31 and is not a dry run.

**Interpretation:** epoch 27 improved pixel/structural metrics, but every Phase 4 epoch had lower average ArcFace similarity than the starting model. “Best” meant highest validation PSNR, not proven best identity fidelity.

The run used 76,000 training images and 4,000 validation images. Training contained 66,500 FFHQ-source and 9,500 Asian-source images; validation contained 3,500 and 500 respectively. Recorded paths do not overlap. Earlier phases may already have seen these validation images, and identity-level separation was not established.

### Local checkpoint review completed 22 September

The downloaded archive is `outputs/phase4-results.tar.gz`. Its extracted run is at `outputs/downloaded_phase4/outputs/phase4_with_progress/`.

A paired check used the same degraded inputs for all models on 64 FFHQ validation images, selected with seed 20260922, degradation seed 42 and heavy-blur probability 0.35. All 64 were common valid identity pairs.

| Model | PSNR | SSIM | ArcFace, shared pairs |
|---|---:|---:|---:|
| Phase 3 | 19.4734 | 0.6376 | 0.3081 |
| Phase 4 best / epoch 27 | 19.7414 | 0.6454 | 0.3048 |
| Phase 4 epoch 31 | 19.5833 | 0.6413 | 0.2933 |

There were also 60 successful model/mode inference cases using 10 synthetic and 5 wild images, including sub-32 mode on the wild images. Visual changes were modest; severe blur remained soft. These real images have no verified pristine target here.

Review artifacts, which Git excludes:

- `outputs/phase4_review/REPORT.md` and `results.json`.
- `wild_comparison.png` and `wild_sub32_comparison.png` in that directory.
- `synthetic_comparison.png` and `paired_comparison.png` in that directory.
- Individual raw outputs and scene composites in that directory.
- `outputs/verify_phase4.py` and `outputs/write_phase4_report.py`, the local comparison helpers.

### Checkpoint meanings

| File | Meaning and intended use |
|---|---|
| `checkpoints/dgp_zamboanga_final.pth` | Phase 3 inference weights; retained deployed baseline and Phase 5 starting point |
| Phase 4 `best.pth` | Epoch 27 weights selected by PSNR; comparison candidate |
| Phase 4 `dgp_improved_epoch_31.pth` | Last Phase 4 weights; not the selected best |
| Phase 4 `last_state.pth` | Complete Phase 4 continuation state; not an inference-only weight file |
| Phase 5 `last_state.pth` | Phase 5 raw model, EMA, optimizer, scheduler, configuration, selection and random states |

Known SHA-256 values:

```text
Phase 3:      b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c
Phase 4 best: 90fd34c1f4b531cd4bb5ee02f6d6995af97f2d43122208e79412de30760a7f09
Phase 4 ep31: 64b7bf98f90cbcc8ef48697d44758dd73f0f4e8f8b929c023657528f8ca2b97a
ArcFace ONNX: 4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43
```

## 4. Phase 5: implemented, awaiting full training

Phase 5 starts a separate **two-epoch pilot numbered 1–2**, from Phase 3. It does not continue the Phase 4 optimizer or imply epochs 32–33.

| Implemented change | Purpose |
|---|---|
| Frozen differentiable ArcFace cosine loss | Penalize identity-feature changes while gradients reach restored pixels |
| Same reference-landmark alignment for output and target | Avoid changing the crop according to the generated face |
| GPU preparation and reusable landmark cache | Avoid repeating landmark detection each epoch |
| Exponential moving average (EMA) | Validate and export smoothed model weights |
| Overall and per-source selection gates | Prevent a PSNR-only win from replacing a baseline with worse measured identity |

The objective adds `0.1 * mean(1 - cosine_similarity)` to the existing restoration loss. ArcFace weights stay frozen. Invalid reference landmarks contribute no identity loss. The cache key includes resized RGB content, shape and detector version.

Defaults: both datasets, original Phase 4 split, batch size 8, two workers, seed 42, head learning rate `1e-5`, backbone learning rate `2e-6`, EMA decay `0.999`, heavy-blur probability `0.35`. These are experiment settings, not a demonstrated optimum. No dataset reweighting or mixed-precision training was introduced.

Phase 5 reports `ArcFace_fixed`, which is **not directly comparable** to Phase 4's detection-based ArcFace values. It uses fixed reference alignment and reports each dataset source separately. Source is not an ethnicity label.

`best.pth` initially contains the starting baseline. Replacement requires higher PSNR than the selected best, overall SSIM and identity at least as good as baseline, the same nonzero eligible-pair count, and no baseline regression in each source's PSNR/SSIM/identity. `best_selection.json` records the decision; epoch 0 means no trained candidate has qualified.

Local verification already completed: 11 unit tests, Python and launcher syntax checks, conversion parity against ONNX Runtime (maximum embedding difference `1.31e-6`), nonzero input gradients with frozen recognition weights, and real CPU smoke training/save/resume. Smoke tests used one batch per nominal epoch and **are not full training or quality evidence**. `outputs/phase5_smoke` must never be presented as a production training result.

Research rationale and primary references are in [PHASE5_RESEARCH.md](PHASE5_RESEARCH.md). Full GPU compatibility, runtime and restoration benefit are still unverified.

## 5. Continue on the existing VM

Commands in this section run in the VM's Linux SSH terminal. First commit/push any intended local changes; `git pull` cannot retrieve files that have not been pushed. Preserve any VM-local edits if Git reports a conflict.

### Update and inspect

```bash
cd ~/forensic-dgp
git status --short
git pull --ff-only origin main
if [ -d venv ]; then source venv/bin/activate; fi
python3 -m pip install -r requirements-phase5.txt
python3 -m pip check
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()"
ls checkpoints/dgp_zamboanga_final.pth outputs/phase4_with_progress/split.json
```

Do not reinstall working GPU drivers or replace the existing PyTorch environment merely to follow the fresh-VM section. If `venv` is absent, the historical environment may be in user site-packages; confirm the printed interpreter and CUDA check before proceeding.

### Start the pilot inside tmux

```bash
tmux new-session -A -s dgp_training
```

Inside that session, confirm no other training command is running before starting another:

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
DATA_DIR="dataset/thumbnails128x128,dataset/asian_faces" bash scripts/run_phase5_gcp.sh
```

The launcher checks ArcFace conversion/gradients, prepares landmarks, runs baseline validation, and performs two training/validation epochs. It requires the Phase 4 split and both datasets. Use a fresh output directory for a new experiment:

```bash
OUTPUT_DIR=outputs/phase5_identity_trial2 bash scripts/run_phase5_gcp.sh
```

Detach with **Ctrl+B**, release, then **D**. Reattach with `tmux attach -t dgp_training`. tmux survives an SSH disconnect; it does not keep training alive through a stopped or rebooted VM.

### Resume an interrupted pilot

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
bash scripts/run_phase5_gcp.sh --resume_state outputs/phase5_identity/last_state.pth
```

Resume starts after the last completed epoch; partial-epoch work is repeated. Keep original parameters, output directory and data membership. If a custom output directory was used, set `OUTPUT_DIR` to that directory and point `--resume_state` to its `last_state.pth`. If no complete checkpoint exists, use a new output directory to restart. A completed two-epoch run reports completion; increasing epochs is a new experiment rather than an identical resume.

## 6. Build a fresh Google Cloud VM

### Provisioning

These are proposed setup choices, not a record of the old VM's exact configuration. No new VM was provisioned while writing this handoff.

1. Select your Google Cloud project and enable Compute Engine with billing and GPU quota in the chosen zone.
2. Create an N1 VM with one NVIDIA T4; `n1-standard-8` is a reasonable starting host configuration for preprocessing, subject to budget and availability.
3. Select Ubuntu 22.04 LTS and a 100 GB persistent balanced disk as an initial allocation; allow more space for duplicate downloads, caches or higher-resolution data.
4. Use standard provisioning for the first pilot and the GPU-required terminate-on-maintenance policy. Keep training access through SSH; the training job does not require a public web port.
5. Record project ID, zone, instance name, disk size and image version before connecting with SSH.

Follow [Google's N1/T4 creation guide](https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose) for supported combinations and regional availability. The existing screenshot hostname was `forensic-dgp-thesis`; do not infer the project ID or zone from that hostname. If using Secure Boot, follow the signed-driver procedure in the driver documentation.

### OS dependencies and GPU driver

```bash
sudo apt-get update
sudo apt-get install -y git tmux python3-venv python3-dev build-essential curl unzip libgl1 libglib2.0-0
nvidia-smi
```

If the image already supplies a working NVIDIA driver, skip driver installation. On a fresh plain Ubuntu VM without one, Google's installer procedure is:

```bash
cd ~
curl -fL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz --output cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver --installation-mode=repo --installation-branch=prod
```

If an Ops Agent is collecting GPU metrics, stop it before installation as described by Google. The installer can reboot the VM; reconnect and rerun the same installation command if instructed, then verify `nvidia-smi`. Restore a previously stopped Ops Agent afterward. Secure Boot requires the additional signing steps, not just the command above. [Official driver instructions](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu).

### Repository and Python environment

Run this clone only when `~/forensic-dgp` does not already exist:

```bash
cd ~
git clone https://github.com/janusinss/forensic-dgp.git
cd ~/forensic-dgp
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install numpy cython
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install -r requirements.txt -r requirements-phase5.txt
python3 -m pip install kagglehub pyarrow pillow
python3 -m pip check
python3 -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

The CUDA 12.6 wheel channel is documented by [PyTorch](https://pytorch.org/get-started/previous-versions/). Packages in the main requirements file are unpinned, so this is a fresh installation recipe, not a byte-for-byte recreation of the historical environment. Check Python/driver compatibility using the [official selector](https://pytorch.org/get-started/locally/) if installation fails. The `CUDA Version` printed by `nvidia-smi` is not the installed PyTorch wheel version. Run the project's conversion and gradient checks before committing GPU time to training.

Record the working environment after installation:

```bash
mkdir -p outputs/environment
python3 -m pip freeze > outputs/environment/pip-freeze.txt
git rev-parse HEAD > outputs/environment/git-revision.txt
nvidia-smi > outputs/environment/nvidia-smi.txt
```

### Model assets

Confirm the Phase 3 weights were obtained with the repository:

```bash
ls -lh checkpoints/dgp_zamboanga_final.pth
sha256sum checkpoints/dgp_zamboanga_final.pth
```

The required ArcFace file is normally `~/.insightface/models/buffalo_l/w600k_r50.onnx`. Restore that cache from the old machine or initialize the existing InsightFace package to download its model pack:

```bash
python3 - <<'PY'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(256, 256))
PY
```

This CPU initialization downloads assets; Phase 5 converts the recognition network to PyTorch and uses CUDA for training. FAN and VGG pretrained assets also download on first use. Keep internet access for initial preparation or restore the corresponding `~/.cache/torch` assets. The landmark preparation and baseline stages can therefore include first-use downloads.

## 7. Datasets: install, verify and preserve the split

Both datasets were used in Phase 4 and are retained for the controlled Phase 5 comparison. Two datasets are not automatically better than one: quality, representation, split integrity and measured outcomes matter. The current mixture is about 87.5% FFHQ-source and 12.5% Asian-source, with no balancing sampler.

| Local relative directory | Downloader source | Expected previous-run count |
|---|---|---:|
| `dataset/thumbnails128x128` | Kaggle `greatgamedota/ffhq-face-data-set` | 70,000 |
| `dataset/asian_faces` | Hugging Face `hiennguyen9874/face-age-gender-asian`, first 10,000 extracted records | 10,000 |

Prefer transferring the exact existing dataset directories for reproducibility. To download into a fresh VM:

```bash
cd ~/forensic-dgp
source venv/bin/activate
python3 scripts/download_ffhq.py dataset/thumbnails128x128
python3 scripts/download_asian_faces.py dataset/asian_faces
```

The FFHQ downloader treats 5,000 existing images as sufficient to skip downloading, although the completed run used 70,000. **Do not treat its success message alone as proof of a complete dataset.** Restore the complete old directory if counts or file paths differ. Access/authentication requirements of the upstream hosts can change; never put tokens into Git.

Check counts and decode every image:

```bash
python3 - <<'PY'
from pathlib import Path
from PIL import Image
for name, expected in [('thumbnails128x128', 70000), ('asian_faces', 10000)]:
    paths = sorted(p for p in (Path('dataset') / name).rglob('*')
                   if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    print(name, len(paths), 'expected', expected, flush=True)
    assert len(paths) == expected, 'Dataset count differs from the completed run'
    for path in paths:
        with Image.open(path) as image:
            image.verify()
    print(name, 'all files decoded', flush=True)
PY
```

Copy `outputs/phase4_with_progress/split.json` from the old VM. It is excluded from Git. When restoring the downloaded archive to a fresh VM, upload your own `phase4-results.tar.gz` to the home directory and extract into a separate staging directory:

```bash
cd ~/forensic-dgp
mkdir -p outputs/restored_phase4
tar -xzf ~/phase4-results.tar.gz -C outputs/restored_phase4
ls outputs/restored_phase4/outputs/phase4_with_progress/split.json
```

Use this restored location explicitly:

```bash
SPLIT_FILE=outputs/restored_phase4/outputs/phase4_with_progress/split.json bash scripts/run_phase5_gcp.sh
```

Run from the repository root and retain original relative image paths. The trainer checks that the manifest includes all images exactly once with no overlap. Do not silently create a different split to bypass a mismatch. If the old manifest contains machine-specific absolute paths, map only the root while preserving membership and document that migration before training.

The current FFHQ targets are 128px thumbnails resized to 256px; resizing creates no additional reference detail. The Asian-source dataset is not a verified Filipino-school benchmark. [FFHQ's official documentation](https://github.com/NVlabs/ffhq-dataset) also specifies usage restrictions, including exclusion of facial-recognition development. Freezing ArcFace for restoration does not itself establish that the intended school identification use is permitted. Dataset/model permissions and a representative evaluation set remain unresolved deployment requirements.

## 8. Expected progress, files and troubleshooting

Validation measures outputs without updating the restoration model. Training updates its weights. A new run performs baseline validation, epoch training, epoch validation, checkpoint saving, and repeats. The phase number in a one-batch smoke test did not prove that real epochs had already trained.

Historical Phase 4 baseline validation took about 43 minutes for 4,000 images. Epoch 27 took about 11 hours 31 minutes plus about 43 minutes of validation. Phase 5 runtime has not been measured; cache preparation is extra first-run work.

Inspect a running VM from a second SSH session:

```bash
nvidia-smi
ps -eo pid,etime,time,pcpu,stat,args | grep '[p]ython'
```

High CPU usage with low GPU utilization can indicate CPU detection or preprocessing; a single GPU snapshot cannot establish that a job is stuck. In Phase 4, InsightFace used CPU execution and lacked visible validation progress initially. ArcFace was not removed to fix that visibility problem.

Phase 5 output directory contents:

- `best.pth` plus `best_selection.json`: selected inference weights and selection reason.
- `epoch_1.pth`, `epoch_2.pth`: EMA inference weights, including candidates that failed selection.
- `last_state.pth`: full state for resuming, not an inference-only checkpoint.
- `metrics.jsonl`, `baseline.json`, `epoch_*.json`: aggregate and per-image measurements.
- `config.json`, `split.json`, `baseline.png`, `epoch_*.png`: experiment settings, membership and comparisons.

| Location | Cause | Fix |
|---|---|---|
| CUDA preflight | CPU wheel, wrong environment or missing driver | Check interpreter, `nvidia-smi`, and PyTorch CUDA availability before launching |
| Split/data check | Missing files, partial download or different layout | Restore exact data and original manifest; verify counts and paths |
| ArcFace preflight | Missing ONNX or conversion mismatch | Restore/download the required asset; run conversion checker and inspect its error |
| Existing output/resume check | New run reuses an output directory or changes configuration | Resume its own full state with original settings, or start a separately named experiment |
| FAN/identity validation | Undetected faces or no eligible reference pairs | Review coverage and images; isolated warnings can occur, but zero baseline identity pairs stops Phase 5 |

For an out-of-memory error, inspect GPU processes first. A smaller batch is a new run setting and cannot silently replace the batch size of an existing resume. Preserve failed-run logs when creating the replacement experiment.

## 9. Download results and move to another workspace

### Export a completed cloud run

Run on the VM after training has finished so metrics and checkpoint state are consistent:

```bash
cd ~/forensic-dgp
tar -czf /tmp/phase5-results.tar.gz outputs/phase5_identity
sha256sum /tmp/phase5-results.tar.gz
```

In SSH-in-browser choose **Download file** and enter `/tmp/phase5-results.tar.gz`. For Phase 4 the equivalent path was `/tmp/phase4-results.tar.gz`, containing `outputs/phase4_with_progress`.

On local Windows, save Phase 5's archive as `outputs/phase5-results.tar.gz`. From the project root in PowerShell:

```powershell
Get-FileHash outputs/phase5-results.tar.gz -Algorithm SHA256
New-Item -ItemType Directory -Force outputs/downloaded_phase5
tar -xzf outputs/phase5-results.tar.gz -C outputs/downloaded_phase5
```

Compare the local hash with the VM's hash. Extract only the archive you exported into a new staging directory; do not overwrite the deployed checkpoint. Its nested run will be `outputs/downloaded_phase5/outputs/phase5_identity/`.

### Transfer checklist

1. Push/clone tracked project code, including this handoff and `PHASE5_RESEARCH.md`; record the commit hash.
2. Transfer `outputs/phase4-results.tar.gz`, `outputs/phase4_review/`, comparison helpers, and any completed Phase 5 run archive separately.
3. Transfer the two dataset directories and original split for exact reproduction, or use the documented fresh-download checks.
4. Preserve required pretrained weights and optionally the landmark/Torch/InsightFace caches; retain checkpoint hashes and environment records.
5. Copy local `AGENTS.md` guidance if the next workspace needs it; recreate its virtual environment instead of copying Windows `venv` to Linux.

`outputs/`, `dataset/`, `weights/`, `venv/`, `docs/`, and agent configuration directories are ignored. Most checkpoint files are ignored too, with specific named exceptions. A successful `git pull` does not bring back all research evidence. Do not include credentials, `.env` secrets or account tokens in a shared archive.

## 10. Verification in the next workspace

The prior test results are recorded above; this document does not imply tests were rerun on a fresh VM. From the project root with the correct environment:

```bash
python3 -m unittest discover -s tests -p test_phase5.py
python3 scripts/check_phase5_identity.py --model ~/.insightface/models/buffalo_l/w600k_r50.onnx --device cuda
```

For a local Windows CPU check, use `venv/Scripts/python.exe` instead of `python3` and `--device cpu` for the conversion checker. Unit tests and conversion checks validate mechanics, not trained output quality.

A separately named local smoke run is optional when validating a rebuilt environment:

```powershell
venv/Scripts/python.exe train_phase5.py --data_dir dataset/thumbnails128x128 --batch_size 1 --num_workers 0 --dry_run --output_dir outputs/phase5_smoke_new_workspace
```

That command requires existing model assets and images. Never mix its artifacts with a complete training run. To reproduce the previous Phase 4 comparison, transfer its ignored helper and inputs first; `outputs/verify_phase4.py` uses fixed archive/report paths and rewrites its review outputs when rerun.

To inspect the application locally using the retained baseline:

```powershell
venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Do not interpret the application's hardcoded candidate scores as validation results.

## 11. Next work, in order

1. Complete the two-epoch Phase 5 GPU pilot with the original data split and preserve all outputs.
2. Compare selected and per-epoch EMA weights against Phase 3 and Phase 4 epoch 27 on identical inputs, using raw images and display outputs separately.
3. Run legacy detection-based evaluation and an independent identity/visual assessment. The recognizer used in the new training loss also supplies `ArcFace_fixed`, so improvement on that metric alone is not independent evidence.
4. Run a controlled identity-loss ablation with a new output directory and `--lambda_identity 0`; retain the same data and remaining settings. This isolates identity supervision from EMA and reduced learning rates.
5. Based on measured errors, plan higher-resolution reference data, representative permitted Filipino test data, and camera-matched degradation experiments. These are proposals, not implemented or proven improvements.

Do not automatically extend training because more epochs are available. Preserve the Phase 3 baseline until the output review supports a deliberate deployment decision. Record selected epoch, checkpoint hash, configuration, data split, coverage, runtime, visual comparisons and limitations for each new result.

### Instruction for the next assistant/workspace

Read `AGENTS.md` if present, this handoff, `PHASE5_RESEARCH.md`, and the current training scripts. Inspect Git status and transferred artifacts before claiming anything is complete. Phase 4 has measured results; Phase 5 has implementation and local mechanical tests only. Verify whether new cloud results have arrived since this handoff, retain the user's focus on actual restored output, and update this document with evidence rather than assuming an improvement.
