# Forensic DGP

A local Windows app for restoring a cropped face picture, with optional manual covering removal. Results are 256 x 256 pixels.

## Install

**`install.bat` is the updated installer on IMPLEMENTATION.** It uses pinned `uv 0.13.1`, cached dependencies, and concurrent restoration/completion asset setup. The filename stays the same so existing shortcuts and update commands keep working.

`install-integrated.bat` is the `main` branch's wrapper for its `integrated_app/` folder. This standalone branch keeps the app at the repository root and uses `install.bat` directly.

Install [Git for Windows](https://git-scm.com/download/win) for the clone command below. Open PowerShell and run:

```powershell
git clone --depth 1 --single-branch --branch IMPLEMENTATION https://github.com/janusinss/forensic-dgp.git
cd forensic-dgp
.\install.bat
.\start.bat
```

Alternatively, [download the IMPLEMENTATION ZIP](https://github.com/janusinss/forensic-dgp/archive/refs/heads/IMPLEMENTATION.zip), extract it, and double-click `install.bat`. Open `start.bat` when installation succeeds.

The installer uses 64-bit Python 3.12 or 3.13. If neither is available, it downloads a private Python runtime under `%LOCALAPPDATA%\DGP` without changing PATH. Every model file must pass its size and SHA-256 checks; existing mismatched files are preserved.

### Faster repeat and multi-PC setup

The installer checks an existing private Python runtime before probing system installations. It validates the version and 64-bit architecture before use, and falls back to a supported system Python if that private copy is invalid.

Package installation uses [uv's Windows hardlink default](https://docs.astral.sh/uv/reference/settings/#link-mode), avoiding full copies of cached dependency files when the cache and environment share a filesystem. Other filesystem layouts fall back to copying. This applies to dependency packages; model files remain separately verified ordinary files.

For a trusted preseeded setup, copy a compatible, populated uv cache intact into `implementation\uv_cache\` before installation. The installer detects that directory automatically. An explicit `UV_CACHE_DIR` setting takes precedence; without either setting, uv uses its normal per-user cache. The bundled cache is git-ignored. Keep the cache and environment on the same filesystem for [best cache performance](https://docs.astral.sh/uv/concepts/cache/#cache-directory).

Reuse existing exact model files under `models/` to avoid downloading them again; full size and SHA-256 checks still run. A copied package cache alone does not make setup fully offline: Python, the one-time uv bootstrap, missing packages, or model files can still require downloads. Installation time depends on available files, connection speed, disk speed, and security scanning.

## Requirements

- 64-bit Windows; a graphics card is not required.
- Internet access during installation. Image processing runs locally afterward.
- At least 2.5 GiB of available RAM before model processing.
- Storage for dependencies, about 720 MiB of model assets, and saved request records.

## Use

1. Upload one cropped PNG or JPEG containing a frontal or mildly turned face. The file limit is 10 MiB and 1600 pixels per side; animated images are unsupported.
2. Select **One face is clear enough to review** when its visible features are readable.
3. Choose **Auto**, **On**, or **Off** under **Choose restoration**. Auto uses an input-quality suggestion; Off returns the resized original.
4. Choose **Bypass** or **Choose areas to protect or blur** for input preparation.
5. Click **Restore selected input**. Compare the images and download a PNG or the review ZIP.

### Optional input preparation

| Selection | Effect during preparation |
| --- | --- |
| Green only | Protect green pixels; blur the remaining surroundings. |
| Blue only | Blur only the blue pixels. |
| Green and blue | Blur blue pixels; green protection takes priority. |
| Neither | Keep the input unchanged. |

Use **Paint**, **Erase both**, **Undo**, and **Blur strength** to adjust the selection. Click **Preview areas**, inspect the preview, and confirm it before continuing. Editing the selection or blur strength requires a new preview and confirmation.

Protection applies to preparation. Restoration can change the face. Preparation is skipped when restoration is Off or Auto keeps the original.

### Optional covering removal

Choose **Review covering removal** after restoration when a covering hides part of the face. Click **Start removal review**, then paint the covering orange. Keep visible features, clear glasses, and non-obstructing hair outside the selection.

Inspect the selection in the editor before clicking **Review**. Confirm both statements when they are true, then click **Generate result**. The guided app uses your manual selection; it does not choose a removal area automatically.

Completion changes only the selected area of the restoration result. An empty selection skips the completion model. PNG and evidence ZIP downloads retain the available images and records.

## Start, stop, and update

Run `.\start.bat` whenever you want to open the app. Keep its terminal open during use. Press **Ctrl+C** in that terminal to stop it.

The browser opens after the local health check succeeds. If it does not open, use the **Application URL** printed in the terminal. For an occupied port, run `.\start.bat --port 0` to select an available port.

To update a Git checkout, stop the app and save any source changes first:

```powershell
git pull --ff-only origin IMPLEMENTATION
.\install.bat
.\start.bat
```

For a ZIP installation, extract a fresh IMPLEMENTATION ZIP into a new folder and run its installer.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Python or dependency setup fails | Read the reported error; check internet access and storage before rerunning `install.bat`. |
| A model fails verification | Keep the reported file for inspection. Move the mismatched file aside before rerunning installation. |
| Available memory is too low | Close unused applications before processing. |
| A worker fails or stops unexpectedly | Inspect the saved request under `scratch/` before restarting the app. |
| Completion assets cannot be downloaded | Use the local-file instructions below if you already have both exact files. |

### Install completion assets from local files

Local staging works even when a first installation stopped before creating `.implementation-env.json`. Use the environment folder printed after **[1/3] Python environment:** and append `\Scripts\python.exe`. Replace the three example paths below:

```powershell
$python = 'C:\path-shown-by-setup\Scripts\python.exe'
& $python -B -m implementation.completion_install --local-r1 'C:\verified-assets\codeformer_completion_r1.pth' --local-detector 'C:\verified-assets\covering_detector_segmenter_v1.pth'
.\install.bat
```

The local staging command performs no downloads. Both files must match their expected size and SHA-256. A peer file can be omitted only if its installed copy already passes verification. Rerunning `install.bat` completes the remaining checks and writes the environment configuration after successful setup.

To verify installed completion files with the same interpreter:

```powershell
& $python -B -m implementation.completion_install --verify-only
```

## Local records and model information

Request images and records are saved under `scratch/implementation_requests` and `scratch/completion_requests`. Downloaded evidence ZIPs also contain image data. Remove your own saved records when they are no longer needed.

V2 is the app's restoration model; its four component files are loaded together. Optional covering removal uses the separate R1 completion model. See the [restoration model card](MODEL_CARD.md) and [completion model card](COMPLETION_MODEL_CARD.md) for asset fingerprints, provenance, and limits. Generated facial detail is an estimate, not proof of identity or hidden appearance.

The app uses adaptations of S-Lab CodeFormer under non-commercial terms. Retain the [restoration license](third_party/codeformer/LICENSE), [restoration notices](third_party/codeformer/NOTICE.md), [completion license](third_party/codeformer_completion/LICENSE), and [completion notices](third_party/codeformer_completion/NOTICE.md). Training data is not distributed with this app.
