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
