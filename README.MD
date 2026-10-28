# Forensic face restoration — IMPLEMENTATION

A local Windows application for reviewing and restoring one cropped face to a 256 × 256 image. It uses Conditioned SFT V2 (`p2_u0224`) for restoration and offers a separate, optional R1 step for estimating areas hidden by a covering.

**Experimental research software:** generated details are estimates, not proof of identity or the hidden face. Request a clearer or less-covered crop when there is insufficient visible facial information. The CodeFormer license includes non-commercial terms; see [Attribution](#attribution) and [MODEL_CARD.md](MODEL_CARD.md).

## Before running

1. Use 64-bit Windows and install **64-bit Python 3.12 or 3.13** from [python.org](https://www.python.org/downloads/windows/). Enable the Python launcher (`py`) or add Python to PATH. A GPU is not required; inference runs on the CPU.
2. Install [Git for Windows](https://git-scm.com/download/win) to clone and update the project. Open Command Prompt or PowerShell in a folder where you can create a checkout.
3. Connect to the internet for installation: the installer obtains Python packages from PyPI and the PyTorch CPU index, and model assets from GitHub. Allow write access to the checkout and `%LOCALAPPDATA%`. After setup, inference runs locally.
4. Allow at least **2.5 GiB of available RAM** before restoration; this is a runtime admission check, not a certified minimum for the whole computer. Close unused applications if the check fails. The six model assets total about 720 MiB; dependencies, the environment, and installation staging need additional disk space.
5. Prepare one non-animated PNG or JPEG face crop: at most **10 MiB**, **1600 pixels per side**, and **2,560,000 pixels** total. Use a frontal or mildly turned face with readable visible facial structure. Crop scenes and multiple faces before uploading.

## First-time installation

Run these commands in Command Prompt (in PowerShell, use `.\install.bat` and `.\start.bat`):

```bat
git clone --depth 1 --single-branch --branch IMPLEMENTATION https://github.com/janusinss/forensic-dgp.git
cd forensic-dgp
install.bat
```

Wait for **both** the restoration setup and the separate R1 completion/proposal setup to finish successfully before starting. The installer creates a checkout-specific Python environment under `%LOCALAPPDATA%\DGP\` and records its location in `.implementation-env.json`; existing project `venv` folders are preserved. Dependencies are pinned in [requirements.txt](requirements.txt).

The installer verifies the size and SHA-256 of all six model files: three base files from [implementation-models-v1](https://github.com/janusinss/forensic-dgp/releases/tag/implementation-models-v1), the bundled V2 heads, and two completion/proposal files from [completion-r1-models-v1](https://github.com/janusinss/forensic-dgp/releases/tag/completion-r1-models-v1). It preserves mismatched existing files and reports failures. If completion downloads are unavailable, see [COMPLETION_INSTALL.md](COMPLETION_INSTALL.md) for verified local asset staging; rerun `install.bat` after staging.

## Start and stop

```bat
start.bat
```

Keep the terminal open. The launcher prints the address, checks the local service, and opens the browser after the health check succeeds. The default is [http://127.0.0.1:8000](http://127.0.0.1:8000). If the browser does not open, visit the printed address manually. To stop the application, press **Ctrl+C** in its terminal and wait for it to exit.

If port 8000 is occupied, run `start.bat --port 0` to select an available port. The actual URL is also recorded in `scratch/implementation_launch.json`. The launcher does not stop other applications.

## Restore a face

1. Upload the crop, inspect it, select **One face with readable visible facial structure**, and choose **Choose restoration**. If the face is too degraded, covered, or outside the supported crop, obtain a suitable image instead.
2. Select **Auto** for an original-input blur/noise suggestion, **On** to run V2, or **Off** to keep the resized input without restoration inference. Choose **Bypass** to use the original upload, or select optional manual Protect + Apply preparation. Select **Continue**.
3. For manual preparation, paint **Protect** over the face and its existing soft edges; paint **Apply** only over surroundings where added blur is permitted. Select **Preview areas**, inspect the preview, confirm it, and select **Continue**. Any edit requires a new preview and confirmation. There is no automatic blur matcher.
4. Select **Restore selected input**. Compare the original, selected input, and result; use **Download PNG** or **Download evidence ZIP**. The ZIP option can include the original, masks, preparation records, result, and available RAW records.

Protected pixels and pixels outside Apply stay exact during preparation. This protection does not constrain the restoration model, which can change visible appearance. A sharper image does not establish recovered identity.

## Optional covering removal

1. After reviewing the restoration result, select **Review covering removal** and **Start removal review**. If no covering needs removal, keep the restoration result.
2. Review the orange proposed removal area. Paint missed coverings and erase visible facial features, clear glasses, and non-obstructing hair. The automatic proposal can miss coverings or select visible details; an empty proposal does not prove that a face is uncovered.
3. Use **More tools** if needed to draw a rectangle or import a 256 × 256 binary PNG mask: white means removal and black means preserve. Select **Review**.
4. Confirm the exact area and that enough visible structure remains, then select **Generate one R1 estimate**. Edits invalidate confirmation; an empty reviewed mask skips R1. If too much of the face is covered, use a clearer or less-covered crop.
5. Compare the original, restoration, reviewed mask, and estimate. Download individual PNGs or the evidence ZIP.

Completion changes only the reviewed area of the restoration result; pixels outside it remain exact relative to that result, which can already differ from the original upload. R1 does not rerun restoration. Automatic proposals and assisted corrections are recorded separately. The detector and completion model are not qualified; see [COMPLETION_MODEL_CARD.md](COMPLETION_MODEL_CARD.md).

## Update an existing checkout

Stop the app with Ctrl+C and wait for it to exit. Save any local source changes before updating. From the checkout, run:

```bat
git pull --ff-only origin IMPLEMENTATION
install.bat
start.bat
```

If the pull fails, resolve the reported local changes or branch divergence first. Installation updates dependencies and verifies the model files.

## Troubleshooting and local records

- **Python not found:** install 64-bit Python 3.12 or 3.13, enable the launcher or PATH, and rerun `install.bat`.
- **Dependency or model setup fails:** read the failing installer step, check network access and disk space, and rerun `install.bat`. Mismatched assets and unsuccessful partial downloads are preserved for inspection. See [COMPLETION_INSTALL.md](COMPLETION_INSTALL.md) for completion assets.
- **Low memory or worker failure:** close unused applications and retry. Restoration has a 360-second request limit and a 6 GiB worker RSS limit; completion and proposal workers each have a 120-second cap, and completion has a separate 6 GiB owned-process commitment limit. These are limits, not minimum hardware specifications. Review retained request records when a worker fails.
- **Browser does not open or port is busy:** open the printed URL, or restart with `start.bat --port 0`.

The service listens on loopback. Restoration records are stored under `scratch/implementation_requests`; covering proposals, reviewed masks, and completion records are under `scratch/completion_requests`. Evidence ZIPs and downloaded PNGs also contain image data. Remove your own local records and downloads when they are no longer needed.

## Attribution

The app uses trained adaptations of S-Lab's CodeFormer architecture. Retain the [CodeFormer license](third_party/codeformer/LICENSE) and [notices](third_party/codeformer/NOTICE.md), and the [completion license](third_party/codeformer_completion/LICENSE) and [notices](third_party/codeformer_completion/NOTICE.md). Non-commercial terms apply. Training datasets are not included.
