# RGB versus grayscale detector pilot — 3 October 2026

This is the next bounded experiment on the existing L4 VM. The returned
varied-covering models still miss obstructing hair, underestimate nearly hidden
hands and leave covering edges. Reviewed masks improve most face estimates.
The completion/restoration backend stays unchanged. Actual training runs only
on the VM; local preparation and audits perform zero weight updates.

The previous V2 sampler already cycles five COFW covering families. This pilot
keeps that sampler and compares input grayscale exposure at an equal budget.
It does not claim grayscale is the established cause of the failures. It also
measures whether more RGB exposure helps before proposing additional labels.

| Artifact | Windows local | Linux VM |
| --- | --- | --- |
| Bundle / LF checksum | `C:\xampp\htdocs\YEAR 4\Testing\outputs\gray-covering-vm-bundle.tar.gz` / `.sha256` | Upload both to `/home/janusdominic0/` |
| Extracted workspace | Source remains at `C:\xampp\htdocs\YEAR 4\Testing\` | `~/forensic-dgp/gray_covering_vm_bundle/` |
| Protocol | `C:\xampp\htdocs\YEAR 4\Testing\outputs\gray_covering_protocol_v1\protocol.json` | `~/forensic-dgp/gray_covering_vm_bundle/inputs/gray_covering_protocol.json` |
| New trainer | `C:\xampp\htdocs\YEAR 4\Testing\scripts\train_gray_covering_vm.py` | `~/forensic-dgp/gray_covering_vm_bundle/scripts/train_gray_covering_vm.py` |
| Return / checksum | Download to `C:\xampp\htdocs\YEAR 4\Testing\outputs\` | `~/forensic-dgp/gray_covering_vm_bundle/gray-covering-results.tar.gz` / `.sha256` |

## Frozen comparison

Both branches start from the unchanged camera91 checkpoint (`epoch 44`, 994 model
updates), with fresh AdamW state. The branches are `rgb133` and `gray133`: eight
experiment epochs of 64 steps, 512 updates each, 1,024 updates total. The first
128 RGB steps match the executed V2 treatment schedule. At update 128 the entire
model must equal the previous `varied133/last.pth`; a mismatch stops the pilot and
preserves partial evidence. No inherited optimizer moments are consumed.

The batch is eight images: two covered/two clear real inputs and two covered/two
clear cached reflection fixtures. Both branches use the same 133 training sources,
266 native/degraded inputs, fixtures, case order, supported BCE/Dice/visible loss,
learning rates and frozen reference-head/BN state. Existing42 COFW additions and
the original 123 registry records remain byte-preserved. Previously inspected
RealOcc hair/near-hidden examples remain evaluation-only; no labels or splits change.

Only `gray133` converts real input RGB to three equal luma channels using
`0.299R + 0.587G + 0.114B`. Each source's visits cycle native-RGB, degraded-RGB,
native-gray, degraded-gray. All 133 sources see all four conditions within the
fixed budget. Targets, supervised/unknown support and geometry are unchanged.
Replay fixtures remain unchanged in both branches.

The initial, RGB128 and two final states each produce 812 training-cohort masks:
266 RGB, 266 grayscale and 280 reflection. Total 3,248 measurement forwards plus
8,192 training image forwards =11,440, with one separate zero-update CUDA
preflight. The processing cap is 30 minutes after loading; allow roughly 3–6 minutes
after loading based on the previous L4 pilot, an estimate rather than new timing.
Final branch counters are `epoch 52`, 1,506 cumulative model updates and512 fresh
updates. Those are detector history counters, not a new full 80,000-image restoration run.

## Existing VM dependencies

The package reuses `~/forensic-dgp/feature_vm_bundle/.venv/` (fallback
`~/forensic-dgp/venv/`) and PyTorch `2.9.1+cu129`. It installs no packages and
creates no venv. Required read-only assets:

| Required VM path | Local verified counterpart |
| --- | --- |
| `~/forensic-dgp/real_camera_vm_bundle/outputs/real_camera_vm/camera91/last.pth` | `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_real_camera_v1\outputs\real_camera_vm\camera91\last.pth` |
| `~/forensic-dgp/varied_covering_vm_bundle/outputs/varied_covering_vm/varied133/last.pth` | `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_varied_covering_v1\outputs\varied_covering_vm\varied133\last.pth` |
| `~/forensic-dgp/coverage_vm_bundle/outputs/reflection_coverage_data_v1/` and `outputs/face_occlusion_dependencies/` | `C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection_coverage_data_v1\` plus recorded existing VM runtime |
| `~/forensic-dgp/outputs/phase4_with_progress/split.json` | `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_phase4\outputs\phase4_with_progress\split.json` |
| `~/forensic-dgp/dataset/asian_faces/` and `dataset/thumbnails128x128/` | Same relative directories under `C:\xampp\htdocs\YEAR 4\Testing\` |

## Upload from Windows Google Cloud SDK Shell

These commands use the working SDK configuration recorded in
`C:\xampp\htdocs\YEAR 4\Testing\WINDOWS_GCLOUD_TRANSFER.md` (receiving root
`~/forensic-dgp/`). Transfer one remote file per command.

1. Select the local outputs directory:

   ```cmd
   cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
   ```

2. Upload the new archive:

   ```cmd
   gcloud compute scp --project=forensic-dgp-thesis "gray-covering-vm-bundle.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

3. Upload its LF checksum:

   ```cmd
   gcloud compute scp --project=forensic-dgp-thesis "gray-covering-vm-bundle.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
   ```

## Prepare in Google Cloud SSH

1. Select VM home:

   ```bash
   cd ~
   ```

2. Verify and extract only into an unused destination; any failure stops extraction:

   ```bash
   sha256sum -c gray-covering-vm-bundle.tar.gz.sha256 && test ! -e ~/forensic-dgp/gray_covering_vm_bundle && tar -xzf gray-covering-vm-bundle.tar.gz -C ~/forensic-dgp
   ```

3. Open the persistent session:

   ```bash
   tmux new-session -A -s dgp_training
   ```

## Run inside tmux

1. Select the new workspace:

   ```bash
   cd ~/forensic-dgp/gray_covering_vm_bundle
   ```

2. Run setup checks, one-batch CUDA preflight and the finite pilot:

   ```bash
   bash scripts/run_gray_covering_vm.sh
   ```

The wrapper activates the existing venv. Setup checks `nvidia-smi`, CUDA, at
least 4 GiB total / 2 GiB free VRAM, exact source/fixture/bundle hashes, dataset
directories and split integrity. The trainer performs `--dry_run --batch_size 1`
without constructing an optimizer. A passing preflight is required before the
full batch 8 run. A failed/partial execution cannot restart in place; preserve it
and return the failing command/output. Do not re-extract over any executed workspace.

No git commit or push has occurred for these local changes. This self-contained
bundle supplies the new code; `git pull` is not its transfer mechanism.

## Return files after completion

In the Windows SDK Shell, after selecting the local outputs directory:

1. Download the result archive:

   ```cmd
   gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/gray_covering_vm_bundle/gray-covering-results.tar.gz" .
   ```

2. Download the result checksum:

   ```cmd
   gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/gray_covering_vm_bundle/gray-covering-results.tar.gz.sha256" .
   ```

The return contains both unselected `last.pth` models, the RGB128 checkpoint and
reproduction report, 3,248 masks, logs, fit measures and ten-row RGB/grayscale
previews. Final optimizer tensors stay on the VM. Fit checks compare hair/new
grayscale coverage, RGB/reflection retention and supervised clear controls; they
cannot create `best.pth` or select an application model. Original 425 gate failures
remain unchanged and unevaluated by this training-only diagnostic.

Next after return: independently verify schedules, optimizer/model counters,
restricted-loaded states, RGB128 reproduction and saved masks; inspect both
previews and rerun the fixed practical gallery before any checkpoint selection.
Training completion alone does not meet the active Goal.
