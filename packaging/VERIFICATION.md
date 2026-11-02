# Version 0.2.0 offline preview verification

Date: 2026-11-01, Asia/Manila. Status: 0.2.0 unsigned offline preview published
on GitHub Releases and the Cloudflare Pages landing page after local-PC acceptance.
The full anonymous browser download and a separate anonymous HTTP transfer
completed with the exact accepted byte count and SHA-256.
Build source: c8ff04e83bd4a722bc73134f67c0c1b1719ecb36. Native relocation into
scratch/pc-moved-final passed, followed by installed-app and scoped-close checks.
The earlier 03d6fda candidate's failed-safe relocation remains below as evidence.
The main landing-page metadata now points to the accepted 0.2.0 GitHub asset.
All 0.1.x sections are historical and do not define the offline contract.

## New acceptance contract

The candidate bundles private Python, hashed Windows application/capture wheel
sets, admitted models and third-party notices. It supports existing-installation
discovery, source/runtime repair, and receipt-bound relocation. Active files are
preserved on preparation failure; historical saved records remain byte-exact.

The payload has 1,097,942,917 input bytes and 1,880,644,934 expanded bytes.
Manifest SHA-256: bf92c65b3ad343f0605325b48dad5c94e80300e4675fd3918c48c86a363fbb3e.
The successful fresh-install candidate 3dce509830e6ef9900673823ecd82e0242589ed4
was 1,045,710,387 bytes, SHA-256
437ee146d5e5af675d51fa9f21beb761c0de00be1b084a81684953aae9563505.
This identifies the earlier fresh-install candidate, not the subsequently tested artifact.
The tested 03d6fda offline EXE is 1,045,712,378 bytes, SHA-256
dc7efb36a1c08a4ceff9bc7123071563482dd138623f300d8680bb780612671f.
Evidence: scratch/dist-v020-final/release.json.

## Published artifact

| Item | Value |
| --- | --- |
| Source commit | c8ff04e83bd4a722bc73134f67c0c1b1719ecb36 |
| Version | 0.2.0 unsigned offline preview |
| Local release artifact | scratch/dist/ForensicDGP_Setup.exe; accepted copy retained in scratch/dist-v020-relocation-final |
| Public release | https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.0 |
| Landing page | https://forensic-dgp.pages.dev/ |
| Pages deployment | https://f0d4b9ce.forensic-dgp.pages.dev/ |
| Bytes | 1,045,716,354 |
| SHA-256 | 4287feb22c96c9abaf5804592938b0c4de2dbfbeecc747cb00a93543a9a5635a |

The rebuilt source differs from 03d6fda only in documentation, release metadata,
the corrected installation-state helper and its regression test; application
Python/UI source is unchanged. This establishes source equivalence for those app
flows; separate native relocation and installed-app results for c8ff04e are
recorded below. Evidence: scratch/candidate-source-equivalence.json.

GitHub release v0.2.0 is public (`isDraft=false`) and marked as a prerelease
(`isPrerelease=true`). Its asset size and GitHub-reported digest match the accepted
EXE above. The accepted artifact was promoted to scratch/dist; the previous
0.1.1 public bytes were preserved and hash-verified in previous-public-v011.
Evidence: scratch/release-promotion-report.json.

Cloudflare Pages deployment succeeded, and the canonical /release.json contains
version 0.2.0 with the exact accepted size/checksum and fixed GitHub download URL.
The anonymous Chromium download from the live landing-page button completed as
ForensicDGP_Setup-0.2.0.exe with 1,045,716,354 bytes, the exact accepted SHA-256,
and no download failure. Its Zone.Identifier contained ZoneId=3 with
HostUrl=about:internet. The browser-owned test download was removed automatically
when its context closed. The downloaded EXE was not executed.

## Checks completed

- Locked application packages installed and imported from local wheels with
  LongPathsEnabled=0. No Windows registry policy change was made.
- Patched package pins: Pillow13.0.0, pip26.2.2, uv0.13.4. The isolated verification
  environment dependency audit reported zero known vulnerabilities for covered
  packages. Torch 2.13.0+cpu was skipped because that build was not found on PyPI.
  This advisory-database result is not a complete security verdict.
- 34 JavaScript capture/playback regressions and 19 website integrity/hosting
  tests passed. The website production build and dependency audit passed.
- Actual private-Python staging proof installed pip26.2.2 from the verified
  wheel, validated ensurepip26.2.2 and a derived-runtime receipt, then published
  only the owned test base. Shared Python remained unchanged.
- An isolated Inno event fixture established that ssPostInstall follows files,
  registration and shortcuts, and precedes the optional application launch.

## Current executable and regression evidence

| Check | Observed result and boundary |
| --- | --- |
| Python regression suite | 327 tests ran, 4 Windows privilege skips; 323 executed successfully. Evidence: scratch/python-v020-cold-final.log. |
| Earlier focused regressions | 39 checks passed against the 03d6fda source after the full suite. |
| Corrected lifecycle regressions | All 52 cases have passing evidence: a combined run passed 51, and one 15-second startup timeout passed unchanged on rerun in 0.834 seconds. This was not a single clean 52-case run. |
| JavaScript and website | 34 capture/playback tests and 19 website integrity/hosting tests passed. |
| Actual fresh offline candidate | Source 3dce509 installed into the dedicated scratch/pc-fresh folder, exited 0 in 261.17 seconds, and published the launcher and active pointer. Evidence: scratch/pc-fresh-zip-outcome.json. |
| Tested candidate in-place update | The 03d6fda EXE exited 0 in 189.22 seconds. Generation de2a0a3f8f3b4b1fa2a3d088ca199ec0 was reused, and the saved record retained SHA-256 7acb5f22954ecb442e6fb8f82c63c9287100460d5afec7057d108aec50509b92. Evidence: scratch/pc-update-final-outcome.json and installation log. |
| Tested candidate source/package repair | The 03d6fda EXE exited 0 in 173.54 seconds after deliberate corruption of only the owned test README and Pillow metadata. README bytes were restored, a new generation 02a9fdbf751248a5a6b7c5ce313a928d was verified, the previous generation remained, and the saved record SHA-256 was unchanged. Evidence: scratch/pc-repair-final-outcome.json and installation log. |
| Relocation finding | The 03d6fda native Move exited 7 after 346.55 seconds while preparing the replacement backend. Windows registration stayed at scratch/pc-fresh; both saved-record copies matched, and an unrelated old-folder file remained. Evidence: scratch/pc-move-final-outcome.json. This was a safe refusal, not a successful relocation. |
| Corrected native relocation | Accepted c8ff04e EXE exited 0 after 651.33 seconds (10 minutes 51 seconds). The environment was recreated, registration switched to scratch/pc-moved-final, the relocation receipt is completed, and the saved-record SHA-256 stayed 7acb5f22954ecb442e6fb8f82c63c9287100460d5afec7057d108aec50509b92. Original records and an unrelated old-folder file remained; optional cleanup was off by default. Evidence: scratch/pc-move-fixed-outcome.json. This is one measured PC result, not a speed promise. |
| Accepted installed-app verification | c8ff04e server was healthy; widths 1280, 1366 and 1920 had no overflow or browser errors; actual video capture, return playback and retained 0.25x/4x rates passed. Evidence: main scratch/pc-final-installed-report.json. |
| Accepted launcher and close | Launcher hash and Start Menu target matched the accepted installation. Its scoped close exited 0, server PID 3548 exited, port 54862 closed and no owned PIDs remained. Evidence: scratch/pc-final-close-report.json. |
| Installed app capture flow | Six checks completed: automatic video opening, retained 0.5x speed, Space play/pause, selected-face zoom, and playable capture/Picture-mode return. The later automation stopped on an ambiguous text selector, not an app error; no complete all-stage flow is claimed from this report. Evidence: main scratch/pc-final-flow-report.json. |
| Actual restoration and review | Restoration On returned HTTP 200 with a real CPU forward; Enlarge filled the 1366 by 768 browser area, the review ZIP exported, and the removal editor opened. No page/console errors were recorded. Removal inference was not run. Evidence: main scratch/pc-app-restoration-report.json. |
| Scoped desktop closure | The close request exited 0; launcher/server exited, the resolved port stopped listening, and no owned PIDs remained. Evidence: scratch/pc-app-close-report.json. |
| Final local download page | Widths 375, 768 and 1280 had no overflow or browser errors; metadata matches the final version, byte count, checksum and intended GitHub asset URL. Evidence: main workspace scratch/offline-site-report.json. This is local page verification, not a public download check. |
| Public browser download | Anonymous Chromium clicked the actual live download button and obtained ForensicDGP_Setup-0.2.0.exe, 1,045,716,354 bytes, SHA-256 4287feb22c96c9abaf5804592938b0c4de2dbfbeecc747cb00a93543a9a5635a, failure null. Live widths 375/768/1280 had no overflow/errors; checksum and metadata matched. Evidence: main scratch/public-v020-download-report.json. |
| Previous release preserved | The 0.1.1 download returned HTTP 200 with 6,304,374 bytes and its unchanged SHA-256 280a1f5ae936f14a42ca7ad9900edacbd82c1ef2122eb706516c394609e317e2. |
| Independent anonymous HTTP transfer | HTTP 200, 1,045,716,354 bytes and the same final SHA-256 in 48.11 seconds. The file was not executed. Evidence: scratch/public-v020-http-download-report.json. |
| Network boundary | The successful installer process used unreachable child-only proxies. Host networking and antivirus settings were unchanged. This verifies no successful proxy-mediated setup fetch was required; it is not a network-disconnected clean-machine test. |

The tested 03d6fda candidate passed in-place update, source/package repair and
the application checks above, but did not pass relocation. Its unpublished draft asset was replaced by the accepted c8ff04e artifact after
the corrected native relocation passed. Only that accepted artifact was published;
the earlier candidate checks remain bound to their original source identities.

## First candidate finding

The dafb213 offline candidate compiled to 1,023,222,504 bytes with SHA-256
681db98600268f812839019645bd93d4907079e47537470ba5d67f49d1d43f0d.
The full suite passed 319 tests (4 skips) before compilation.

Its real fresh installation stopped before activation/registration: the capture
import probe exceeded 30 seconds during parallel setup. The retained environment
then passed the exact capture/OpenCV/NumPy/pip probe in 0.928 seconds alone.
This establishes a cold parallel-setup timeout, not an antivirus diagnosis.
The fix defers capture verification until bulk writes finish and allows a
bounded 120-second cold check, retaining 30 seconds for normal health checks.

An isolated R2-model Inno benchmark verified equal extracted hashes: ZIP/1
extraction 6.281 seconds versus no-compression 3.218 seconds. ZIP/1 produced
352,164,946 bytes versus 378,706,964 bytes, so the next full candidate uses
non-solid ZIP/1. Total fixture process time included substantial overhead
outside the extraction timer; no cause is asserted for that overhead.

## Relocation defect and correction

The candidate had already prepared its destination backend when health discovery
counted the array returned by `Tree-Files` as one object instead of counting the
160 source files described by the receipt. This incorrectly refused the candidate
before Windows registration switched. Existing registration and original records
were retained, and the copied record remained byte-identical.

A two-line collection-count correction and a regression using multiple source
files passed red/green checks. Read-only health discovery with the corrected
helper reported scratch/pc-moved healthy. The corrected c8ff04e installer then
passed a separate native relocation into the new empty scratch/pc-moved-final
folder, with completed receipt, switched registration and verified record hashes.
The earlier failed destination and its evidence were preserved.

## Publication and remaining boundaries

GitHub publication, canonical live release metadata, the live checksum, previous
release preservation, three-width page layout, and the complete anonymous browser
download passed. A separate full anonymous HTTP transfer independently matched
the final file. Public evidence is retained in main workspace
scratch/public-v020-download-report.json and feature scratch/
public-v020-http-download-report.json.

The browser download initially paused at 44,460 bytes before later completing.
No protection settings changed; the cause of that pause was not diagnosed.
The downloaded file had Mark of the Web but was not executed. Installation tests
used the local accepted artifact. Therefore these results do not establish the
SmartScreen outcome or antivirus acceptance when executing a freshly downloaded
Internet-marked EXE. No universal antivirus qualification is claimed.

The earlier fresh-install candidate, update/repair candidate, accepted relocation
artifact, app-source equivalence and final installed-app checks are bound
separately above. They are not one identical test or a clean-Windows validation.
No VM is part of this run. Native wizard interaction and antivirus-prompt coverage
remain limited; executable logs and automated fixtures do not establish every
interactive state. No general installation-time performance guarantee is made.

The PC reports Avast Antivirus registered. No antivirus settings were changed.
Registration alone does not prove every protection module is active. The user
provided an illustrative SmartScreen screenshot, not an exact diagnosis of a
specific release detection. Signing is not implemented and no warning-free
execution guarantee is made. Webcam/RTSP hardware and physical computers other
than this PC remain unverified. Model qualification is unchanged.

---

# Historical version 0.1.1 installer verification

Public landing page: https://forensic-dgp.pages.dev/.
The hosting moved to Cloudflare Pages on 2026-10-31; the former Workers endpoint
was removed. The 0.1.1 artifact described below is unchanged. Historical publication details
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
time. Its publication, authorization, fixed-directory, package-locking and
release-gate statements describe 0.1.0 only. They do not revoke the later user
approval for the 0.2.0 offline installer, PC testing, relocation, or release work.

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

## Historical 0.1.0 gates at the time of that report

| Gate | Missing evidence |
| --- | --- |
| Clean machine | A clean standard Windows account/VM with no Git or Python, empty caches, no admitted models, restricted network and insufficient-disk scenarios. Current warm-cache results do not satisfy this gate. |
| Native host | Actual Edge-host file selection, webcam permission, live webcam and LAN RTSP hardware. Native Edge inspection returned Computer Use app approval timed out. |
| Active-work closure | Window closure during real inference or active live capture, repeated clicks and all alternate Edge lifetime conditions. Passive closure and process primitives were tested. |
| Full maintenance matrix | Complete interactive cancellation timing, interrupted fresh runtime creation/repair, all downgrade/divergence combinations and ordinary non-ASCII Windows account paths on a clean machine. Selected boundaries have automated fixtures. |
| Release trust | Authenticode signing, complete transitive package hash locking, independent release attestations and public distribution are not delivered by this preview. |

At the time of the 0.1.0 report, these gates kept that preview in draft status.
Its then-approved goal covered only a local review artifact. Later authorization
and the current 0.2.0 delivery scope are recorded at the top of this document;
this historical paragraph is not a current prohibition on offline packaging or
publication. Purchasing signing services remains outside the free-preview scope.

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
