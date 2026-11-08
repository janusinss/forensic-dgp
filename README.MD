# Forensic DGP — IMPLEMENTATION

Local experimental face restoration, reviewed input preparation, video face capture, and optional covering removal. Generated facial detail is an estimate; these models are not qualified for identifying people.

## Current installer: 0.2.3

Download the [offline Windows installer](https://forensic-dgp.pages.dev/). Version
0.2.3 adds YuNet face detection for frozen video frames, the Face queue, restoration
diagnostics and a controlled benchmark. It includes the existing restoration and
Completion R2 models. See the [0.2.3 release](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.3)
for the installer and checksum.

## Historical 0.2.2 Completion R2

The [0.2.2 release](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.2) introduced Completion R2 and verified that saved records and prior models survived the update. See [R2_VERIFICATION.md](packaging/R2_VERIFICATION.md) for the exact artifact, native update and installed inference checks.

## Historical 0.2.1 path hotfix

Download [0.2.1](https://forensic-dgp.pages.dev/) for the Windows short-path
payload verification fix. Valid payloads no longer fail because Windows expands
an 8.3 temporary-folder alias. Existing-file safeguards remain enabled.
See [hotfix verification](packaging/PAYLOAD_PATH_HOTFIX.md).

## Previous offline desktop installer: 0.2.0

Version **0.2.0 is published as an unsigned research prerelease**. Download it
from the [landing page](https://forensic-dgp.pages.dev/) or
[GitHub Releases](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.0).
It bundles private Python, complete hashed Windows CPU package sets and all six
admitted models. It adds fresh/update/repair/move discovery, safe environment
replacement and verified relocation records. See [build instructions](packaging/BUILD.md).
The public download is one offline EXE containing 1,045,716,354 bytes. The live
metadata, GitHub asset digest and complete anonymous browser-download hash
match the accepted build. The previous 0.1.1 download remains available unchanged.

Verification uses the user's own PC and dedicated test folders. An earlier
0.2.0 candidate completed a fresh installation; the tested candidate passed an
in-place update while reusing its verified environment and preserving a saved
record. Source/package repair also passed with restored source bytes, a new
verified environment generation and preserved records. Capture, real restoration,
review export and scoped shutdown checks passed. Native relocation stopped safely
before switching registration because of a source-file counting defect; the helper
fix has regression coverage and the corrected c8ff04e installer is built.
Its repeat native relocation passed in 651.33 seconds with records preserved,
recreated environments and updated Windows integration. The accepted installation
also passed layout, capture, playback-rate and shutdown checks. GitHub/Pages
publication and final browser-download verification are complete. The downloaded
Internet-marked EXE was verified but not executed, so these checks do not establish
its SmartScreen/antivirus execution outcome. No clean-Windows
VM or general hardware qualification is claimed. Antivirus stays enabled. Signing remains
outside the free preview and an unsigned release can still receive Windows warnings.

## Previous public desktop installer: 0.1.1

The Windows setup EXE has a public [download page](https://forensic-dgp.pages.dev).
The following behavior describes **the historical 0.1.1 online installer**,
preserved as the previous release rather than the current main download. See [installer verification](packaging/VERIFICATION.md) for its exact artifact identity and test boundaries.

The 0.1.1 setup flow selects a dedicated local fixed-drive folder of at most 120 characters. That folder contains the desktop launcher plus `runtime` for private Python, application/capture dependencies, package caches, verified models and saved records. First setup requires Windows 10/11 x64, Microsoft Edge, internet access and at least 4 GB of free disk space. It does not require administrator access or preinstalled Python. Known-size downloads show measured progress; package setup and verification show activity when a total is unknown.

In 0.1.1, updates verify and reuse valid components in the same location. To
select another folder in that old version, uninstall the Windows launcher first; the previous runtime, saved evidence and ownership records remain. Setup creates the new installation without automatically migrating or deleting the old data. Uninstall retains `runtime`, `install-location.json` and `shell-owner.json` for safe verification on reinstall.

See [installer build and folder behavior](packaging/BUILD.md) and [verification evidence](packaging/VERIFICATION.md). These desktop changes preserve the existing developer batch workflows below, including their legacy default storage location.
## Developer batch installation on Windows

1. Download [install.bat](https://raw.githubusercontent.com/janusinss/forensic-dgp/IMPLEMENTATION/install.bat) into an empty folder. Save the actual file, not the GitHub HTML page.
2. Run `install.bat`. No Git, existing Python, administrator access, or complete repository download is required.
3. Run the `start.bat` created beside it. Keep the application terminal open; press Ctrl+C to stop.

The installer resolves the latest **IMPLEMENTATION** commit once and downloads that exact version. It prints the installed revision. The managed app lives under `%LOCALAPPDATA%\DGP\app`; an existing unrelated source checkout is not overwritten. If a different `start.bat` already exists beside the download, it is preserved; use the launcher path printed by setup.

Requirements: 64-bit Windows, internet access for setup, at least 2.5 GiB available RAM for processing, and sufficient disk space for Python dependencies, approximately 720 MiB of model files, retained app versions and your saved records. A GPU is not required.

## Developer batch update and installation behavior

Close the managed app and rerun `install.bat` to install the current published UI, flow and fixes. `start.bat` starts the last successfully installed version without an online update check. The installer cannot retrieve unpublished local edits.

- Source files are checked against the exact GitHub commit tree; model files retain their complete size/SHA-256 checks. This trusts GitHub HTTPS and the repository; it is not a separately signed release.
- New code is staged in a separate version directory. The active-version pointer changes only after dependency, asset and application import checks pass. The prior version and saved records remain available if setup fails.
- Dependencies use the pinned uv installer, its package cache and CPU-only PyTorch. An unchanged dependency fingerprint reuses its environment across UI-only updates. A changed dependency manifest creates a separate environment. The isolated capture runtime skips package installation when its exact versions and detector pass verification.
- Verified model files are shared across app versions rather than downloaded for every update. Mismatched models or edited managed source files are refused and preserved for inspection.
- A shared managed-app lock prevents concurrent setup or updating a running managed app. Setup does not kill processes or occupied ports. Previous versions and failed staging files are retained; no automatic evidence deletion occurs.

The legacy online batch path pins direct dependencies; it is not the complete
hash-locked offline package path introduced for the 0.2.0 desktop installer. First setup needs the network. Repeated setup may still need it to check the branch or obtain missing packages; launching an already installed app is local.

## Input and review

Choose **Picture** to upload one PNG/JPEG face image, or **Video** for an uploaded clip, webcam or LAN RTSP camera. A chosen video file opens automatically. Native playback controls support Space play/pause and a separate 0.25×–4× speed selector.

Freeze a frame to scan it locally. The numbered boxes mark the detected faces in the original frame. Selecting a box gives you a slightly wider crop that you can adjust before review. You can also drag on the preview or press Enter to create a manual crop. Arrow keys move it, and Shift+arrow keys resize it. **Full frame crop** returns to the entire frame while keeping the selection. **Use this face** sends the original-pixel rectangular PNG crop to Picture review.

The capture scanner bundles the MIT-licensed YuNet 2023mar model from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). It scans the selected frame when frozen. Small, blurred, or obstructed faces can still require a manual crop.

Switching Picture/Video, reviewing a crop or temporarily leaving Input pauses an opened local video while preserving its source, position, speed, crop and zoom. Returning does not autoplay. **Resume video** continues a frozen capture. Explicit Stop, file/source replacement and page unload release it. Live webcam/RTSP sources are released when leaving capture; they are never silently restarted.

Review whether one face is clear enough, then choose **Auto**, **On** or **Off** restoration and **Bypass** or optional reviewed preparation. Green protects pixels during preparation; blue selects blur, with green taking priority. Review and confirm preparation before restoration. Protection applies to preparation; generated appearance can change.

After freezing a frame from a video file, webcam, or RTSP camera, check the detection count. Use **Add all N to queue** to queue the detected faces from that frame and open **Face queue**. Use **Add selected face to queue** for one detected or manually adjusted crop, or **Queue loaded picture** for a still image. The queue holds up to 12 crops per session. If the group will not fit, select individual faces or free slots in the queue. Review or skip each crop. After every crop has a decision, **Run approved crops** processes one crop at a time in queue order. Open each finished result to inspect it or download its records. Reloading or closing the app clears the queue.

Result diagnostics show edge variation and estimated noise before and after restoration. Open **Benchmark** to try an illustrated demo or use a clean square face photo. The benchmark saves the clean reference, applies a fixed blur, restores the blurred copy, and reports MAE and PSNR for both versions against that reference. The scores apply to the controlled test case.

Optional removal review uses **Trained CodeFormer Completion R2** after the result, including when restoration was Off or Auto retained the original. Paint only the covering orange, inspect it, confirm the two statements and generate. Pixels outside the reviewed area are copied exactly from the supplied prepared/restored canvas; an empty selection skips completion. Hidden features remain guesses. Preserve the original, masks and review ZIP when assessing a result.

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
