# Version 0.2.0 candidate verification

Date: 2026-11-01, Asia/Manila. Status: candidate under local-PC verification;
not yet published. The previous 0.1.1 record below is historical evidence.

## New acceptance contract

The candidate bundles private Python, hashed Windows application/capture wheel
sets, admitted models and third-party notices. It supports existing-installation
discovery, source/runtime repair, and receipt-bound relocation. Active files are
preserved on preparation failure; historical saved records remain byte-exact.

The payload has 1,097,942,917 input bytes and 1,880,644,934 expanded bytes.
Manifest SHA-256: bf92c65b3ad343f0605325b48dad5c94e80300e4675fd3918c48c86a363fbb3e.
Final compressed EXE bytes and SHA-256 will be recorded after compilation.

## Checks completed before candidate compilation

- Locked application packages installed and imported from local wheels with
  LongPathsEnabled=0. No Windows registry policy change was made.
- Patched package pins: Pillow13.0.0, pip26.2.2, uv0.13.4. The isolated verification
  environment dependency audit reported no known vulnerabilities. This is an
  advisory-database check, not a complete security verdict.
- 34 JavaScript capture/playback regressions and 19 website integrity/hosting
  tests passed. The website production build and dependency audit passed.
- Actual private-Python staging proof installed pip26.2.2 from the verified
  wheel, validated ensurepip26.2.2 and a derived-runtime receipt, then published
  only the owned test base. Shared Python remained unchanged.
- An isolated Inno event fixture established that ssPostInstall follows files,
  registration and shortcuts, and precedes the optional application launch.

## Required final checks

Full integrated regression suite, final committed-source compilation, actual
fresh install/reinstall/repair/move in dedicated PC test folders, offline setup,
first-use flows, browser download and active-antivirus observations remain
pending until explicit results are appended. No VM is part of this run.

The PC reports Avast Antivirus registered. No antivirus settings were changed.
Registration alone does not prove every protection module is active. The user
provided an illustrative SmartScreen screenshot, not an exact diagnosis of a
specific release detection. Signing is not implemented and no warning-free
execution guarantee is made. Webcam/RTSP hardware and physical computers other
than this PC remain unverified. Model qualification is unchanged.

---

# Version 0.1.1 installer verification

Current download page: https://forensic-dgp.pages.dev/.
The hosting moved to Cloudflare Pages on 2026-10-31; the former Workers endpoint
was removed. The installer below is unchanged. Historical publication details
remain in this report; see ../download-site/DEPLOYMENT.md for the hosting cutover.

Date: 2026-10-31, Asia/Manila.
Status: published unsigned research preview. Clean-machine qualification remains open.

## Published artifact

| Item | Value |
| --- | --- |
| Source branch | feature/inno-desktop-installer |
| Build source commit | 90d43aa9687edf46f0cb4ca4fa9b4db5ea8a5b5d |
| Version | 0.1.1 |
| Artifact | scratch/dist/ForensicDGP_Setup.exe |
| Bytes | 6,304,374 |
| SHA-256 | 280a1f5ae936f14a42ca7ad9900edacbd82c1ef2122eb706516c394609e317e2 |
| Public page | https://forensic-dgp.janusdominic0.workers.dev/ |
| Worker version | ec2ece28-d490-437b-bc86-fe13add2e922 |

The public browser download matched this exact local artifact. The build embeds
committed source and seven hashed setup inputs. Later documentation and public
metadata commits do not change this artifact's source identity.

## Delivered installation behavior

The wizard explains prerequisites, selects a dedicated local fixed-drive folder,
offers an optional Desktop shortcut, reviews the destination, displays setup
progress and finishes with a launch option. Installation requires no elevation.
Paths longer than 120 characters, protected roots, network locations, linked
ancestors and unrecognized nonempty destinations are refused.

The selected folder contains the launcher and bound runtime child: private
Python, isolated application/capture environments, pip/uv caches, verified model
families and saved records. Existing developer batch roots remain separate.
Upgrades reuse the registered folder. Choosing another folder requires removing
the Windows launcher first; setup does not move or delete the old data.

Python and model transfers report measured per-file byte counts and percentages.
Operations without a reliable total display an activity bar. One active transfer
stays visible during parallel downloads; all parallel output remains in details.
Python progress is throttled to roughly 1 MiB intervals without reducing
cancellation or integrity checks.

Uninstall removes Windows integration and retains runtime data and bindings.
New uninstall entries retain the shell marker. Older uninstall logs can remove
it, so recovery requires both an exact bound location configuration and an exact
schema-2 runtime owner before restoring the marker. Validate remains read-only.

## Observed checks

| Check | Evidence and boundary |
| --- | --- |
| Python suite | 237 tests ran, 4 Windows privilege skips; 233 executed successfully. Final packaging/ownership/wizard changes also passed 47 focused checks. |
| Real wizard fixture | Compiled Inno EXEs recorded measured 0/50/100 progress; interleaved A/B downloads remained stable; an early-finished peer did not stall the display. This was executable logging evidence, not visual inspection of the interactive wizard. |
| Fresh chosen folder | Actual installation into a previously absent local folder downloaded private Python, application/capture dependencies and all six admitted model files. The initial successful cold candidate was ab12dee; final source changes affected wizard display and retained-marker recovery. The final artifact then passed an actual upgrade and reinstall. |
| Full-folder inventory | Private Python, both package caches, environments, models and installation records were beneath the selected runtime. The preservation baseline recorded 43,249 files totaling 2,832,472,630 bytes. |
| Cancellation | A real Python-download cancellation requested through the installer's cancellation file exited before activation; no current pointer or Windows registration was published. Download/staging evidence remained. This did not test clicking the native Cancel button. |
| Upgrade | Final 90d43aa artifact exited 0 in the existing selected folder, verified cached components and activated its exact source revision. |
| Uninstall/reinstall | Actual final uninstaller exited 0; all 43,249 baseline runtime files, the verification record, location configuration and marker remained. Final reinstall exited 0. |
| Legacy uninstall recovery | A real isolated old/new Inno upgrade demonstrated deletion by the old uninstall entry. Exact-binding recovery then passed actual PowerShell validation/preparation checks. |
| Existing saved evidence | All 149 pre-existing data files in the legacy installation retained their SHA-256 values. No old data was moved or deleted. |
| Desktop launch | Installed launcher hash matched its embedded manifest. Exact final source 90d43aa launched; its own URL record resolved the server, health returned 200 and an owned visible Forensic DGP window was observed. That window was left available for review. |
| Installed app browser | The actual installed server returned its Picture UI and Video option with no page or console errors. Screenshot: main scratch/download-v011/installed-app.png. |
| Download page | Local and live checks passed at 375, 768 and 1280 CSS pixels without overflow or browser errors. A real click downloaded the expected filename and exact 6,304,374-byte EXE with the SHA-256 above. Attachment/octet-stream headers passed; sensitive configuration paths returned 404. |
| Site checks | Seven release-preparation fixtures passed; TypeScript/Vite production build passed; npm audit reported zero vulnerabilities. |

## Remaining boundaries

Testing used a fresh chosen folder and private runtime on this development PC.
It was not a clean Windows VM/account without development tools. The native
interactive wizard's visual layout was not inspected. Clean-machine network,
low-disk, interruption and full account-path matrices remain open.

Earlier owned-window closure and process-lifetime evidence is retained below.
The final custom-root window was left running; closing it during active inference
or live capture was not newly tested. Actual webcam/RTSP hardware behavior and
model qualification are outside these installation checks.

The setup and launcher remain unsigned. Direct package pins are not a complete
transitive hash lock. Publishing this preview does not establish model quality,
identity verification or a complete security verdict.

## Evidence

Feature scratch contains build-v011-release.log, python-v011-verified-tests.log,
v011-final-focused.log, setup-custom-v011-verified.log, setup-v011-upgrade.log,
uninstall-v011.log, setup-v011-reinstall.log, custom-v011-install-report.json,
custom-v011-uninstall-report.json, custom-v011-preservation-report.json and
desktop-v011-report.json. The main workspace retains browser artifacts in
scratch/download-v011. Credentials, records and internal logs were not uploaded.

Source remains on the installer feature branch. This task did not merge or push
IMPLEMENTATION, create a GitHub release or configure automatic updates.

## Internal evaluation

Accuracy: verified artifact and public download identity. Completeness: requested
features and local maintenance checks passed; clean-account and native visual
validation remain open. Clarity: measured transfer progress and activity stages
are distinguished. Actionability: live download is available. Conciseness: user
handoff can link the download and this report.

---

# Historical 0.1.0 local preview evidence

The following report describes the earlier artifact at its original verification
time. Its publication and fixed-directory statements do not describe 0.1.1.

# Local installer preview verification

Date: 2026-10-31, Asia/Manila.
Disposition: local review preview. Full release validation remains open.

## Artifact identity

| Item | Value |
| --- | --- |
| Feature branch | feature/inno-desktop-installer |
| Application/build source commit | d0a9be8cc9f98b9a42bbc2ff86dae8b1f8ee0df2 |
| Setup version | 0.1.0 preview |
| Artifact | scratch/dist/ForensicDGP_Setup.exe |
| Size | 6,232,315 bytes |
| SHA-256 | 3c055fe550149b9c3aee7f51bda59f5af7ebed683cbf3970d3d7c9b1cb50e4a5 |
| Signing | Unsigned |
| Build tool | Inno Setup 7.1.1 x64; official installer SHA-256 and Authenticode checked |
| Publication | None; IMPLEMENTATION was not merged or pushed |

The artifact was built before this report-only documentation update. Its source
commit remains the one above. The generated scratch/dist/release.json identifies
all embedded inputs. Source is bundled; Python dependencies and models use the
online backend. No offline-install claim is made.

## Observed local results

| Check | Result and boundary |
| --- | --- |
| Python suite | 179 tests ran, 4 skipped; 175 executed successfully. Windows symlink privilege cases account for skips. Full suite is from the same Python implementation as the final candidate. Focused release/desktop/wizard checks passed after final packaging edits. |
| JavaScript | 34 existing capture tests passed. No frontend source was changed by this feature. |
| Source and integrity failures | Actual early EXE attempts rejected noncanonical source bytes and a progress sharing conflict before activation. Original b7d8437 active pointer was retained. Canonical archive generation and progress logging were repaired with regressions. |
| Installation/adoption | Actual per-user EXE installation adopted the existing marked batch backend. Verified model/runtime caches were reused. No administrator elevation was requested. This was a warm-cache test on the current machine. |
| Final reinstall | Final d0a9be8 EXE reinstalled successfully over retained backend data after shell uninstall. Installed current.json matches the artifact source commit. |
| Desktop ownership | Final installed launcher SHA-256 matches its embedded manifest. Its actual executable path, healthy server and owned visible-window diagnostics were checked. Final server URL was resolved from its launch record, not guessed. |
| Window closure | User closed the earlier local desktop window after native-tool inspection timed out. Recorded server22728 exited normally, port60813 stopped listening and zero dedicated desktop processes remained. Final launcher changes extend starter-exit waiting and preserve logs; final owned-window readiness was separately observed. |
| Process fixtures | Suspended process assignment, retained descendants after starter exit, job cleanup and survival of an unrelated control process passed against final C# source. |
| Application browser flow | Installed server passed Picture/Video layout checks at1280x720,1366x768,1920x1080; supplied MP4 auto-open;0.5x rate; mode return; freeze; keyboard crop; fresh human review; Back to capture; restoration Off result; Enlarge; removal editor. No page errors, console errors or failed HTTP responses. These flows ran on daab081 before the final C#-only startup/logging and packaging edits. |
| Evidence preservation | All97 preexisting evidence files, excluding mutable launch metadata, retained identical SHA-256. Package inventory remained unchanged. New browser-test records were retained. |
| Uninstall | Actual registered uninstaller exited0, removed shell/shortcut/registration and retained the managed backend. Existing evidence hashes were checked again. Models/environments were intentionally retained as batch-shared resources. |
| Running-app guard | Final EXE refused installation while the final desktop app was running, exited1 and left current.json unchanged. No process was killed. |
| Shortcut | Inno created the per-user Start Menu shortcut. Its target matches the exact tested installed launcher with no additional arguments. Desktop shortcut remains optional. |

## Automated safety coverage

| Area | Evidence |
| --- | --- |
| Bundles | Exact source identity, tree hashes, ZIP checksum, unsafe members, tampering and ancestry refusal. |
| Transaction | Cancellation before activation, joining owned tasks, prior pointer retention, shared-runtime verification without package mutation, invalid-runtime refusal. |
| Progress | Actual PowerShell stderr exit semantics; exact structured events; late-cancellation result precedence; actual Windows sharing conflict retries and immediate real-access denial. |
| Storage/uninstall | Exact ownership, wrong roots, junction refusal, installation mutex and full sentinel inventory retention. |
| Desktop | Random event/readiness contract, actual Windows event signaling, startup cleanup, PID-in-job readiness and owned process disposal. |

A cancellation request finishes the current operation before reaching a safe
boundary. This is not instant cancellation of package managers. Once activation
succeeds, a late request does not report a false failure. Shell registration
failure after successful backend activation can leave a prepared backend; it does
not claim to roll the backend pointer back. Rerunning the same setup completes
registration after resolving the shell error.

## Gates still open

| Gate | Missing evidence |
| --- | --- |
| Clean machine | A clean standard Windows account/VM with no Git or Python, empty caches, no admitted models, restricted network and insufficient-disk scenarios. Current warm-cache results do not satisfy this gate. |
| Native host | Actual Edge-host file selection, webcam permission, live webcam and LAN RTSP hardware. Native Edge inspection returned Computer Use app approval timed out. |
| Active-work closure | Window closure during real inference or active live capture, repeated clicks and all alternate Edge lifetime conditions. Passive closure and process primitives were tested. |
| Full maintenance matrix | Complete interactive cancellation timing, interrupted fresh runtime creation/repair, all downgrade/divergence combinations and ordinary non-ASCII Windows account paths on a clean machine. Selected boundaries have automated fixtures. |
| Release trust | Authenticode signing, complete transitive package hash locking, independent release attestations and public distribution are not delivered by this preview. |

These gates keep the release in draft status. The approved continuing goal allows
a local review artifact with unmet gates explicit; it does not authorize merging,
publishing, purchasing services or beginning an offline installer.

## Evidence locations

- Feature scratch: python-tests-final.log, javascript-tests.log, packaging-final-tests.log and final-candidate-tests.log.
- Feature scratch: setup-adoption-3.log, setup-reinstall.log, setup-running-app.log and uninstall-test.log.
- Feature scratch: preservation-report.json, desktop-close-report.json, final-desktop-report.json and running-app-guard-report.json.
- Main workspace scratch/inno-browser: report.json and installed-flow.png.
- Managed data/desktop/logs: native supervisor and server logs.

Saved evidence and screenshots stay in ignored directories. They are not embedded
in the distributable installer. The final preview was left available locally for
review; closing its dedicated window requests normal server shutdown.

## Internal evaluation

| Axis | Assessment |
| --- | --- |
| Accuracy | Results distinguish actual executable tests, process fixtures, browser checks and unverified host behavior. |
| Completeness | Local preview delivered; clean-machine and release gates remain open. |
| Clarity | Source identity and retention behavior are explicit. |
| Actionability | Setup EXE, exact manifest, build instructions and this report are local review artifacts. |
| Conciseness | Handoff should link the artifact and report without repeating the phase checklist. |

No model qualification or identity-verification claim follows from these
installation and functional checks.
