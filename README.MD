# Forensic DGP — IMPLEMENTATION

Local experimental face restoration, reviewed input preparation, video face capture, and optional covering removal. Generated facial detail is an estimate; these models are not qualified for identifying people.

## Windows installer 0.2.12

Download the [offline Windows installer](https://forensic-dgp.pages.dev/). It includes Python, packages and the admitted models, including Completion R2. Choose the folder for the full app during setup. Updates preserve saved records and prior model files. Exact release details and checksums are in [v0.2.12](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.12).

## Developer batch installation

1. Download [install.bat](https://raw.githubusercontent.com/janusinss/forensic-dgp/IMPLEMENTATION/install.bat) into an empty folder. Save the actual file, not the GitHub HTML page.
2. Run `install.bat`. No Git, existing Python, administrator access, or complete repository download is required.
3. Run the `start.bat` created beside it. Keep the application terminal open; press Ctrl+C to stop.

The installer resolves the latest **IMPLEMENTATION** commit once and downloads that exact version. It prints the installed revision. The managed app lives under `%LOCALAPPDATA%\DGP\app`; an existing unrelated source checkout is not overwritten. If a different `start.bat` already exists beside the download, it is preserved; use the launcher path printed by setup.

Requirements: 64-bit Windows, internet access for setup, at least 2.5 GiB available RAM for processing, and sufficient disk space for Python dependencies, approximately 720 MiB of model files, retained app versions and your saved records. A GPU is not required.

## Update and installation behavior

Close the managed app and rerun `install.bat` to install the current published UI, flow and fixes. `start.bat` starts the last successfully installed version without an online update check. The installer cannot retrieve unpublished local edits.

- Source files are checked against the exact GitHub commit tree; model files retain their complete size/SHA-256 checks. This trusts GitHub HTTPS and the repository; it is not a separately signed release.
- New code is staged in a separate version directory. The active-version pointer changes only after dependency, asset and application import checks pass. The prior version and saved records remain available if setup fails.
- Dependencies use the pinned uv installer, its package cache and CPU-only PyTorch. An unchanged dependency fingerprint reuses its environment across UI-only updates. A changed dependency manifest creates a separate environment. The isolated capture runtime skips package installation when its exact versions and detector pass verification.
- Verified model files are shared across app versions rather than downloaded for every update. Mismatched models or edited managed source files are refused and preserved for inspection.
- A shared managed-app lock prevents concurrent setup or updating a running managed app. Setup does not kill processes or occupied ports. Previous versions and failed staging files are retained; no automatic evidence deletion occurs.

Package installation is pinned at the direct-dependency level; transitive dependencies are not a complete hash-locked supply chain. First setup needs the network. Repeated setup may still need it to check the branch or obtain missing packages; launching an already installed app is local.

## Input and review

Choose **Picture** to upload one PNG/JPEG face image, or **Video** for an uploaded clip, webcam or LAN RTSP camera. A chosen video file opens automatically. Native playback controls support Space play/pause and a separate 0.25×–4× speed selector.

Picture scans the upload with YuNet and selects **Face 1** when a crop is available. Faces are numbered from left to right. **Picture crop** lets you preview another detected face, keep the full picture, or use **Manual crop** to draw, move, and resize a rectangular crop directly on the picture (drag or Enter, Arrow keys, and Shift+arrow keys). When no face is detected, you can use the full picture, draw a manual crop, or upload a close-up. Before/After metrics refer to the chosen crop. The review ZIP retains the original upload and crop coordinates.

Freeze a frame to scan it locally. The numbered boxes mark the detected faces in the original frame. Selecting a box gives you a slightly wider crop that you can adjust before review. You can also drag on the preview or press Enter to create a manual crop. Arrow keys move it, and Shift+arrow keys resize it. **Full frame crop** returns to the entire frame while keeping the selection. **Use this face** sends the selected rectangular crop directly to Settings.

The capture scanner bundles the MIT-licensed YuNet 2023mar model from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). It scans the selected frame when frozen. Small, blurred, or obstructed faces can still require a manual crop.

Switching Picture/Video or leaving Capture pauses an opened local video while preserving its source, position, speed, and crop selection. **Resume video** continues from the frozen frame. Explicit Stop, file/source replacement, and page unload release it. Leaving capture releases live webcam and RTSP sources.

After uploading a picture, select **Continue**. Settings lists **Auto**, **Manual**, and **Off** in that order. **Auto** is selected by default. It checks the full picture or frozen frame for strong horizontal scanlines, then for pixelation. For scanlines, Auto suppresses the repeating frequency and applies 1.5 px of Gaussian smoothing at the 256-pixel scale. For pixelation, it maps the block regions, fills narrow gaps supported by repeated edges, and compares 2.5, 3.0, and 3.5 px of smoothing using remaining block edges and retained detail. It uses 3.0 px when there is too little evidence to choose another strength. Where the full-source check does not modify the selected crop, Auto can check the face for block patterns or use Protect + Auto-match when the face is measurably blurrier than its surroundings. Otherwise it restores the crop directly.

Auto runs the restoration model once per crop. The result shows the preparation used. Full-source scanline cleanup or pixelation smoothing also offers a separate full-size prepared PNG. These experimental routes improve the appearance of some inputs. Detection, region boundaries, and the chosen strength are estimates. **Manual** lets Picture users bypass preparation or mark regions. **Off** returns the resized original.

Auto can also handle tilted or perspective-warped mosaics, irregular cell mosaics, and strong grids in 32–47 px crops. When separate artifacts are supported, it can treat distinct regular and irregular mosaic regions, combine scanline correction with local mosaic smoothing, combine a supported 4 px mosaic with exterior JPEG deblocking, run DCT macroblock deblocking (8 × 8 FBCNN / 16 × 16 HEVC boundary filtering), or compose native YCrCb noise denoising, color-cast/dark-channel dehazing, spatial comb cleanup, edge deringing, gradient debanding, tangent anti-aliasing, and chroma debleeding before V2. The offline installer includes the pinned FBCNN checkpoint for JPEG and macroblock preparation.

Use the top-right triple-dot menu (**Check Update**) inside the app to check for a newer release and install verified updates in place while retaining saved records.

In reviewed preparation, green-only **Auto-match surrounding blur** measures reliable edges across the full upload, including protected areas, and adds blur to sharper areas outside green. With this matching option off or blue marks present, **Manual blur strength** sets the radius. Green takes priority. Select **Continue** on the preparation preview to use it for restoration. Generated appearance can change.

After freezing a video, webcam, or RTSP frame, select one face or click **Add all** for up to 12 faces. **Use this face** or **Use these faces** opens Settings directly. Settings shows the selected crops and one shared **Auto**, **Manual**, or **Off** choice, with **Auto** selected by default. Video **Auto** reuses preparation of the verified frozen frame across the selected crops and offers one shared full-frame PNG when full-frame preparation is applied. Video capture skips manual preparation. In Restore, the app processes the selected faces in order, one at a time.

In Restore, choose a numbered face button to see that face’s **Before** and **After** images. Each image has **Download PNG** and **Enlarge** controls. When review files are requested, **Download ZIP** appears beside the After image. Edge variation and estimated noise appear below the pair, including a **Prepared** column when preparation runs and the native-scale **Before** noise estimate when it differs from the 256-pixel scale. Add a clean unpadded crop or a 256 × 256 PNG or JPEG for that face if it matches the pose and crop. The app compares the resized original and result to the supplied reference using MAE and PSNR over the observed crop area. Without a reference, the panel displays **Ground truth unavailable**. Switching faces keeps each reference and its scores with the corresponding result.

Optional removal review uses **Trained CodeFormer Completion R2** after the picture result, including when restoration was Off. Paint only the covering orange, inspect it, confirm the two statements and generate. Pixels outside the reviewed area are copied exactly from the supplied prepared/restored canvas; an empty selection skips completion. Hidden features remain guesses. Preserve the original, masks and review ZIP when assessing a result.

R2 has its own `codeformer_completion_r2.pth` file. Updating preserves existing R1 files and saved records. Missing or changed R2 weights refuse completion instead of silently using R1. Restoration and the optional covering detector are unchanged; the guided editor continues to use manual covering selection.

## Launch and troubleshooting

`start.bat --port 0` selects an available loopback port. `--no-browser` prints the URL without opening a tab. If a port is occupied, setup/start reports it instead of opening another app's UI.

| Problem | Recovery |
| --- | --- |
| Old interface after update | Stop the older app, rerun the new installer, then run the managed launcher printed by setup. Compare its printed commit. |
| Setup or network failure | Read the error and rerun setup. The active pointer is not promoted on failure. |
| Model or managed code differs | Keep the reported file. Move only the identified mismatched version/file aside for inspection before retrying. |
| Capture session is full | Stop and restart the app; saved capture records remain on disk. |
| Browser permission or RTSP failure | Use a local video file or explicitly reconnect the camera. Do not put camera credentials in the URL. |

Managed models are in `%LOCALAPPDATA%\DGP\app\data\models`. Saved request records are in `data\scratch\implementation_requests`, `data\scratch\completion_requests` and `data\scratch\capture_requests`. Downloaded ZIPs contain image data. Existing source-checkout records are left where they were; updates do not import or delete them automatically.

## Developers and existing source checkouts

The default installer always installs/updates the managed app. To work on an extracted or Git checkout deliberately:

```powershell
git pull --ff-only origin IMPLEMENTATION
.\install.bat --source --no-pause
.\start.bat --source
```

Save local source changes before pulling. The source path keeps its own environment record and checkout-local models/records. `start.bat --source --verify-only` checks that installation without starting a server. Standalone users need a newly downloaded installer once to get this updater; an older batch file cannot update itself automatically.

Python asset verification remains available through `implementation.setup_models --verify-only` and `implementation.completion_install --verify-only` with the installation's Python. Completion assets can be staged from exact verified local files using `--local-r2` and `--local-detector`; consult the module's `--help`. Keep historical R1 files as R1; renaming another checkpoint does not pass the exact R2 size, file hash and logical-state checks.

## Model information and licenses

See [MODEL_CARD.md](MODEL_CARD.md) and [COMPLETION_MODEL_CARD.md](COMPLETION_MODEL_CARD.md) for fingerprints, provenance and limits. The app uses adaptations of S-Lab CodeFormer under **non-commercial terms**. Retain the [restoration license](third_party/codeformer/LICENSE), [restoration notices](third_party/codeformer/NOTICE.md), [completion license](third_party/codeformer_completion/LICENSE) and [completion notices](third_party/codeformer_completion/NOTICE.md). Training datasets are not distributed.
