# Forensic face restoration — IMPLEMENTATION

This branch is for application development, deployment, and real-world testing of a local Windows face-restoration app powered by **Conditioned SFT V2 (`p2_u0224`)**. It accepts one already cropped frontal or mildly turned face and produces a 256×256 result. Research and training are maintained separately.

## Requirements

- 64-bit Windows with 64-bit Python 3.12 or 3.13; enable the Python launcher or Python on PATH.
- Git for cloning and updates, plus internet access during installation.
- At least 2.5 GiB available RAM before restoration; inference uses CPU PyTorch.

Dependencies are pinned in [requirements.txt](requirements.txt). The installer creates a short, checkout-specific environment under `%LOCALAPPDATA%\DGP\` and records it in the ignored `.implementation-env.json` file. Existing project `venv` folders are preserved.

## Install and start

```bat
git clone --depth 1 --single-branch --branch IMPLEMENTATION https://github.com/janusinss/forensic-dgp.git
cd forensic-dgp
install.bat
start.bat
```

Installation downloads the three base model files from the [model release](https://github.com/janusinss/forensic-dgp/releases/tag/implementation-models-v1) and stages `v2_p2_u0224_heads.npz` from `implementation/assets/`. Each of the four model files must match its complete size and SHA-256; existing mismatched files are preserved and reported.

The browser opens after the local health check passes, normally at http://127.0.0.1:8000. Keep the terminal open and press Ctrl+C to stop. If that port is occupied, use `start.bat --port 0`; the actual address is printed and recorded in `scratch/implementation_launch.json`. After setup, inference runs locally without model-download requests.

## Update

Stop the running app with Ctrl+C and wait for it to exit. Save any local source changes before updating. From this checkout, run:

```bat
git pull --ff-only origin IMPLEMENTATION
install.bat
start.bat
```

If the pull fails, resolve the reported local changes or branch divergence before continuing. Rerunning installation updates dependencies and verifies the model files.

## Use

1. Upload one PNG/JPEG crop, up to 10 MiB and 1600 pixels per side, and inspect its visible facial structure.
2. Choose **Auto** (input-quality suggestion), **On** (restore with Conditioned SFT V2 `p2_u0224`), or **Off** (keep the resized original with no inference).
3. Bypass preparation, or paint Protect over the face and its existing soft edges. Paint Apply only where added surroundings blur is permitted.
4. Preview and confirm manual preparation. Edits invalidate confirmation; every preview rebuilds from the original. Protected pixels and pixels outside Apply stay exact during preparation.
5. Restore the selected input and compare the original, selected input, and V2 result. Download PNGs or an optional evidence ZIP containing the original, available masks, preparation records, result, and available RAW output.

Manual preparation is optional; there is no automatic blur matcher. Protection applies to preprocessing only. Generated facial details can change, and a sharper-looking result does not prove recovered identity. Request clearer or less-covered input when visible information is insufficient. Optional reviewed covering completion is described below. The models remain experimental; see [MODEL_CARD.md](MODEL_CARD.md).

## Optional covering removal and completion

The Conditioned SFT V2 restoration stage is unchanged. After reviewing its result,
choose **Review covering removal** if a mask, sunglasses, strong lens glare,
hand, obstructing hair, scarf or object still hides the face. An uncovered face
can finish with the restoration result.

The separate detector proposes an orange removal area. Correct it with Paint,
Erase, Undo, the rectangle tool or a binary mask PNG. Keep clear glasses,
non-obstructing hair and visible features outside that area. The detector can
miss coverings or select visible features; an empty proposal does not establish
an uncovered face. Manual painting remains available when proposals fail.

Confirm the exact removal area and sufficient visible facial structure, then
generate one estimate with **trained CodeFormer completion R1**. Edits invalidate
confirmation. Empty reviewed masks skip R1. Excessive coverage requests a
clearer or less-covered crop.

Completion changes only the reviewed area of the restoration result. Every
pixel outside it stays exactly equal to that result; this is different from
preserving the original upload, because restoration may already have changed
visible appearance. Completion does not rerun restoration or apply it to the
generated face. Original upload, restoration, mask and final estimate remain
separate, with PNG and evidence ZIP downloads. Automatic proposals and corrected
assisted masks are recorded separately.

Both models run locally in separate CPU workers and unload between stages.
The completion and proposal stages each have a 120-second process cap; the
existing restoration limit remains unchanged. These limits are not performance
guarantees for arbitrary Windows hardware. No inference API or training VM is
required. See [completion model limits](COMPLETION_MODEL_CARD.md) and
[verified assets/offline setup](COMPLETION_INSTALL.md).

## Privacy and troubleshooting

The service listens on loopback. Restoration records stay locally under `scratch/implementation_requests`; covering proposals, reviewed masks and completion records stay under `scratch/completion_requests`. Downloaded ZIPs also contain image data. Remove your own records and downloads when no longer needed.

- Missing dependency or model: rerun `install.bat` and inspect the failing step.
- Low memory: close unused applications before retrying. Restoration retains its 6 GiB RSS limit; completion uses a separate 6 GiB owned-process commitment limit. These are different measurements, not minimum RAM specifications.
- Worker failure: review the retained request directory before restarting `start.bat`. Requests have a 360-second limit and no automatic retry.
- Busy port: stop your own earlier server or use `start.bat --port 0`. The launcher does not kill other applications.

## Attribution

The app uses trained adaptations of S-Lab's CodeFormer architecture. Retain [the S-Lab license](third_party/codeformer/LICENSE) and [notices](third_party/codeformer/NOTICE.md); its non-commercial terms apply. Training datasets are not included.
