# Forensic DGP — IMPLEMENTATION

Local experimental face restoration, reviewed input preparation, video face capture, and optional covering removal. Generated facial detail is an estimate; these models are not qualified for identifying people.

## Install on Windows

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

Freeze a frame, select a detected face or drag on the preview to crop it. Press Enter on the frozen preview to create a keyboard-adjustable crop. Arrow keys move the crop; Shift+arrow keys resize it. **Full frame crop** returns the preview to the entire frame while keeping the selected crop. **Use this face** passes the direct rectangular PNG crop into Picture review; no FAN alignment, person tracking, ranking or restoration runs during capture.

Switching Picture/Video, reviewing a crop or temporarily leaving Input pauses an opened local video while preserving its source, position, speed, crop and zoom. Returning does not autoplay. **Resume video** continues a frozen capture. Explicit Stop, file/source replacement and page unload release it. Live webcam/RTSP sources are released when leaving capture; they are never silently restarted.

Review whether one face is clear enough, then choose **Auto**, **On** or **Off** restoration and **Bypass** or optional reviewed preparation. Green protects pixels during preparation; blue selects blur, with green taking priority. Review and confirm preparation before restoration. Protection applies to preparation; generated appearance can change.

Optional removal review is available after the result, including when restoration was Off or Auto retained the original. Paint only the covering orange, inspect it, confirm the two statements and generate. Hidden features remain guesses. Preserve the original, masks and review ZIP when assessing a result.

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

Python asset verification remains available through `implementation.setup_models --verify-only` and `implementation.completion_install --verify-only` with the installation's Python. Completion assets can be staged from exact verified local files using `--local-r1` and `--local-detector`; consult the module's `--help`.

## Model information and licenses

See [MODEL_CARD.md](MODEL_CARD.md) and [COMPLETION_MODEL_CARD.md](COMPLETION_MODEL_CARD.md) for fingerprints, provenance and limits. The app uses adaptations of S-Lab CodeFormer under **non-commercial terms**. Retain the [restoration license](third_party/codeformer/LICENSE), [restoration notices](third_party/codeformer/NOTICE.md), [completion license](third_party/codeformer_completion/LICENSE) and [completion notices](third_party/codeformer_completion/NOTICE.md). Training datasets are not distributed.
