# Build the online Windows installer

Status: version 0.1.1 is published at https://forensic-dgp.janusdominic0.workers.dev. The public download matches the verified local artifact. Clean-machine release validation remains open; see VERIFICATION.md.

## Build prerequisites

Use Windows x64 with Git, Python 3.12/3.13 x64, the Windows .NET Framework C# compiler, Node.js for JavaScript checks, and verified Inno Setup 7.1.1 x64. End users need none of these build tools. The desktop window requires Microsoft Edge.

The compiler installer is pinned to:
https://github.com/jrsoftware/issrc/releases/download/is-7_1_1/innosetup-7.1.1-x64.exe

SHA-256:
8a5a6711c3766a6e5fb74f280ba0e3551e719ce94538d901fdbd1f8f9920b3f0

Verify the checksum and a valid Authenticode signature from Pyrsys B.V. before installing the compiler. Install it for the current user without elevation. This checkout uses `scratch/tools/inno-7.1.1/ISCC.exe`; build tools are not committed.

## Build

Commit the source intended for the artifact on the isolated feature branch. A local preview build does not require merging to IMPLEMENTATION or publishing a release.

```powershell
python -B -m unittest discover -s tests
node --test tests/*.test.cjs
python packaging/build_release.py --iscc "scratch/tools/inno-7.1.1/ISCC.exe" --version 0.1.1
```

The builder refuses tracked modifications. Payloads and packaging inputs come from the exact commit; untracked records and model weights are excluded. The setup EXE bundles application source. First setup obtains the pinned private Python runtime, application packages and verified models online. The generated manifest records source identity, ancestry, input checksums and the artifact hash.

Output is `scratch/dist/ForensicDGP_Setup.exe` with `scratch/dist/release.json`. Build output is ignored by Git. The instructions identify the build inputs; byte-for-byte reproducibility is not asserted for timestamped unsigned EXEs. See `VERIFICATION.md` for the candidate actually built and the checks performed.

## End-user requirements

- Windows 10 or 11, 64-bit, with Microsoft Edge.
- Internet access for first setup, including approximately 720 MiB of models and the Python/packages downloads.
- At least 4 GB of free disk space initially; allow additional space for retained versions, caches and saved records.
- A dedicated, user-writable folder on a ready local fixed drive, with a full folder path of at most 120 characters.
- No administrator access, Git, preinstalled Python or GPU is required.

Network/device paths, drive roots, protected Windows/shared-program directories and linked folders are refused. Select a dedicated child directory rather than a personal-folder root such as Documents or Downloads.

## Full application folder

The directory page selects the full application location. The default is `%LOCALAPPDATA%\Programs\Forensic DGP`; users can choose another permitted folder. The ready page shows that location before installation.

| Location under the selected folder | Contents |
| --- | --- |
| Application root | Desktop launcher, shell ownership marker and bound `install-location.json` record. |
| `runtime\versions` | Verified application source versions. |
| `runtime\python-3.13.9-amd64` | Private Python used by this desktop installation. |
| `runtime\environments` and `runtime\cache` | Application/capture libraries and reusable package caches. |
| `runtime\data` | Verified model files, saved capture/restoration/removal records, and desktop diagnostics. |

Windows registration and shortcuts remain in normal per-user Windows locations. Temporary staging/handshake files can use Windows temporary storage. The full application folder does not require relocating existing global Python installations or changing PATH.

## Progress and cancellation

Setup presents five named stages. Known-size downloads show measured byte percentages. Package resolution, environment creation and checks use an activity bar when their total work is unknown. Details and diagnostic logs remain available; the activity bar is not a time estimate.

Cancellation waits for the current owned operation to finish before stopping at a safe boundary. Activation is the commit boundary; a late cancellation request does not turn a successful backend result into a false failure. If shell installation fails after backend activation, rerun the same setup to finish registration. Shell rollback must not be described as restoring the backend pointer.

## Updates and changing location

Run a newer tested setup EXE explicitly to update the same folder. Valid Python, environments and model files are verified and reused. Package changes use a different environment identity; mismatched or invalid existing resources are preserved and refused rather than silently overwritten. Source ancestry rejects unintended downgrades or divergent embedded payloads.

To choose a different folder while a desktop location is registered, uninstall the Windows launcher first. Uninstall retains the previous runtime and records. Setup can then create a fresh installation in the new folder. It does not copy, move, merge or delete the old models, Python, environments or saved evidence automatically. Preserve those records when comparing the installations.

The original batch-managed `%LOCALAPPDATA%\DGP\app` installation is also left untouched. A custom desktop installation has its own `runtime`; it does not adopt or relocate that older backend implicitly.

## Uninstall and reinstall

Windows Installed Apps uninstall removes the desktop launcher, registration and shortcuts. It retains `runtime`, `install-location.json` and `shell-owner.json` so saved evidence and ownership metadata remain available. Reinstalling into that same marked folder can verify and reuse retained resources. Older uninstall logs can remove the marker; setup recreates it only after the retained location configuration and schema-2 runtime ownership independently verify the same folder. An unrelated nonempty directory or mismatched binding is refused.

No cache-removal checkbox or automatic record deletion is provided. Removing the launcher does not reclaim the retained runtime's disk space.

## Developer batch workflows

The existing batch defaults are preserved: standalone `install.bat` resolves published IMPLEMENTATION and uses its legacy user-local managed root; `start.bat` launches that installed version. Deliberate checkout development continues through `install.bat --source` and `start.bat --source`. The desktop EXE uses the location bound to its own launcher directory instead of redirecting these workflows.

## Release limits

The setup and desktop launcher are unsigned. Inno packaging does not eliminate SmartScreen warnings. Direct dependency pins are not a complete transitive hash lock. A cached local smoke test is not a clean-machine test; consult `VERIFICATION.md` for actual evidence and open gates.

Offline packaging, automatic background updates, GitHub publication and merging to IMPLEMENTATION require separate authorization. The 0.1.1 website deployment and download checksum have been verified.
