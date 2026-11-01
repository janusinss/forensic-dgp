# Current 0.2.0 offline installer scope

Updated 2026-11-01, Asia/Manila. Later user instructions approved a single offline
EXE, fresh/update/repair/move flows, verification on the user's own PC, and a
public downloads-only GitHub release linked from the existing Cloudflare Pages
site. This supersedes the older online-only/local-preview scope below.

The tested candidate is built from source
03d6fdaa5caa7817eff8aa721e0d784593bf0da2. An earlier fresh offline candidate
completed installation; the tested candidate passed an actual in-place update and
source/package repair with records preserved. Application capture, actual
restoration, export and scoped shutdown checks passed. Native relocation stopped
safely before switching registration because of a source-file counting defect.
The helper correction has regression coverage; a rebuilt artifact and repeat
relocation remain pending. A matching 03d6fda asset exists only in a draft release
and must not be published as the corrected accepted artifact.

Use [BUILD.md](BUILD.md) for the current build/storage/maintenance contract and
[VERIFICATION.md](VERIFICATION.md) for artifact identities, exact results and open
checks. This is PC verification, not a clean Windows VM or general hardware
qualification. Native UI/antivirus observations remain limited; the unsigned
preview provides no universal warning-free execution guarantee.

No VM setup, download mirrors, automatic background updater, or signing-service
purchase is included. No IMPLEMENTATION merge or push has been performed.

---

# Historical 0.1.0 implementation plan

The original proposal follows unchanged for provenance. Its online-only,
local-preview and authorization/deferred-work statements applied at that time;
they do not override the later approved 0.2.0 scope above.

# Forensic DGP: online Windows installer plan

Status: local preview implemented on the isolated feature branch. Phase 5 release gates remain open; see VERIFICATION.md. No merge or publication occurred.

Prepared: 2026-10-31, Asia/Manila.

## Intended result

An end user downloads `ForensicDGP_Setup.exe`, follows a graphical installer, and launches the existing Forensic DGP interface from a Start Menu or optional Desktop shortcut. A dedicated desktop window hosts the interface. Setup and launch provide visible errors without requiring a terminal.

The release reuses the existing managed installation engine. Picture/Video capture, restoration, removal, model verification, and developer `--source` workflows remain regression requirements.

## Baseline and execution boundary

- Verified local branch: `IMPLEMENTATION`, commit `b7d84372c377b773fdd268fa469dbe20d09e38d5`.
- Current checkout: `C:\xampp\htdocs\YEAR 4\Testing`. Unrelated untracked files are present and must remain untouched.
- Proposed implementation checkout: sibling `C:\xampp\htdocs\YEAR 4\Testing_INNO_INSTALLER`, branch `feature/inno-desktop-installer`. Recheck branch ownership before creation; preserve any existing directory or branch.
- The downloaded `C:\Users\janus\Downloads\inno_installer_plan.md` remains an input proposal. This plan records the revised scope.
- Wait for approval before branch creation, compiler installation, application changes, installer runs, commits, or publication. Goal registration does not authorize execution.

## Scope decisions

| Area | Planned behavior |
| --- | --- |
| First delivery | Online installer for Windows x64, installed for the current user without requested elevation. |
| User-facing files | One setup EXE for distribution; shortcuts launch a windowed application. Internal implementation files remain under managed installation directories. |
| Release identity | Each setup EXE installs one exact tested source commit. Updates require an explicit setup/update action. Launch uses the active installed version locally. |
| Existing installation | Adopt a valid marked batch installation through the same backend. Refuse unrelated nonempty managed roots. Preserve source checkouts. |
| Deferred work | Full offline installer, automatic background updates, mobile redesign, model changes, and UI flow redesign. |

## Architecture boundaries

| Component | Responsibility |
| --- | --- |
| Inno layer: `packaging/installer.iss` | Wizard, installation registration, shortcuts, bounded uninstall, and progress presentation. |
| Managed backend: `implementation/distribution.ps1` | Owned-root validation, installation lock, pinned source verification, receipts, staging, verification, and atomic activation. |
| Setup modules: `implementation/environment.py`, bootstrap and asset installers | Validated Python, reusable environments, CPU packages, isolated capture runtime, and verified models. |
| Desktop layer: `packaging/launcher/`, `implementation/launch.py` | Windowed launch, actual loopback URL, readiness, logs, process ownership, and controlled shutdown. |
| Release evidence: `packaging/BUILD.md`, `packaging/VERIFICATION.md` | Pinned build instructions, artifact identity, acceptance results, and explicit limitations. |

Inno-owned shell files should live separately, provisionally under `%LOCALAPPDATA%\Programs\Forensic DGP`. Backend-owned versions, environments, and data remain under `%LOCALAPPDATA%\DGP\app`. Setup must not populate the managed root before its ownership checks. Existing shared Python and developer environments outside that root are outside uninstallation scope.

The GUI must consume structured events from the backend rather than maintain a second installation implementation. File integrity checks provide the existing integrity boundary; they must not be presented as a separately signed release unless release signing is actually implemented and verified.

## Phase 1: isolation and test specification

1. Verify the current IMPLEMENTATION revision.
2. Create the isolated feature worktree.
3. Record the build and platform prerequisites in `packaging/BUILD.md`.
4. Specify failing safety regressions in the distribution and managed-path test suites.
5. Define the generated release-manifest schema.

The generated release manifest identifies the source commit, launcher/build-tool versions, required assets, checksums, and package resolution evidence. Generate it from an existing tested source revision; avoid a tracked manifest requiring its own future commit hash. Keep generated build output outside source tracking.

Gate: no unrelated file changes; no branch switch in the existing checkout; a bounded file list and acceptance matrix exist before implementation.

## Phase 2: desktop lifecycle proof

1. Define the desktop launch contract in `implementation/launch.py`.
2. Prototype the windowed launcher in `packaging/launcher/`.
3. Connect readiness to the server's actual loopback URL.
4. Implement shutdown through owned process handles.
5. Record the shell decision in `packaging/DESKTOP_LIFECYCLE.md`.

Edge app mode is provisional. Its acceptance test must establish reliable ownership when Edge is already running, a window is closed, a second launch occurs, or startup fails. Use a dedicated app profile where required; never terminate every Edge or Python process. Test capture and inference workers during closure. Allow bounded graceful shutdown before terminating only owned descendants.

If Edge app mode cannot meet these requirements, stop this phase for review of an owned WebView2-host design. Do not silently substitute a normal browser or install another shell dependency. Detect missing prerequisites before launch. Keep error dialogs and an accessible log location when consoles are hidden.

Gate: a shortcut-compatible launcher opens a healthy app window; closing that window releases its server, lock, and workers; unrelated browser processes remain running. Startup errors remain visible. A cold application launch after installation requires no network access.

## Phase 3: graphical access to the existing installer

1. Add structured setup events to `implementation/distribution.ps1`.
2. Add stage reporting to Python and asset setup modules.
3. Implement cancellation at defined safe boundaries.
4. Extend adoption tests for existing managed installations.
5. Extend failure tests for interrupted upgrades.

The GUI displays measured byte progress for downloads whose totals are known. Package resolution, environment creation, and verification display stage activity when duration or total work is unknown. Parallel tasks retain their own status. Completion is reported only after successful activation.

Preserve source/model checks, receipt verification, path and reparse defenses, installation locking, environment reuse, and atomic active-version changes. Hold the transaction boundary through cancellation. Retain diagnostics and partial artifacts according to existing preservation behavior; never expose a partially prepared version as active.

Treat environments shared with the active version as read-only during updates. Verify and reuse an already valid environment without changing packages. Prepare any required repair or package mutation in a separate environment generation before activation. Extend environment records if needed to distinguish generations; retaining `current.json` alone does not establish rollback safety. Include cancellation during dependency setup against a shared environment, followed by a successful launch of the previous active version.

Gate: cancellation or failure leaves the previous active version and its environment usable. Repeated setup reuses valid environments and models. Edited source, mismatched assets, unrelated roots, and concurrent app/install operations are refused without destructive repair.

## Phase 4: Inno packaging and maintenance

1. Author `packaging/installer.iss`.
2. Connect the wizard to the verified backend.
3. Add per-user launch shortcuts.
4. Add explicit upgrade and selective uninstall behavior.
5. Document the reproducible build command in `packaging/BUILD.md`.

Pin and verify the compiler and any build dependencies. Prefer Inno's built-in download capabilities where needed; add no download plugin without a concrete requirement. Use `PrivilegesRequired=lowest`. Define a stable application identifier and a user-visible version mapped to the exact source commit.

Use an immutable tested source payload for a release EXE. Preserve the existing standalone batch updater's explicit latest-IMPLEMENTATION behavior; share its verification/activation engine rather than change its public contract. Repair must verify before reuse and preserve mismatched files instead of silently overwriting them. Refuse unintended downgrades by default.

Uninstall removes registered shell files and owned application code/runtime files only within verified boundaries. Preserve captures, originals, masks, results, shared models, and reusable caches by default. Optional removal of verified model/cache paths needs an explicit user choice; saved evidence is not included in that choice. Do not recursively delete `%LOCALAPPDATA%\DGP`. If the app is running, request closure through its owned lifecycle or stop with an actionable message.

Existing installations may share environments or caches with batch workflows. Inventory ownership before removal; preserve resources whose exclusive ownership cannot be established. Keep ownership metadata needed for future verification when retained data remains.

Gate: the generated EXE performs fresh setup and explicit upgrades, registers one application entry, creates working shortcuts, and preserves retained data on uninstall. Build output stays untracked. Signing status is disclosed; no security-warning or installation-time guarantee is made.

## Phase 5: full executable verification

1. Run the focused unit and integration suites.
2. Test setup under a clean standard Windows user.
3. Exercise the installed desktop workflows.
4. Test the failure and maintenance matrix.
5. Produce `packaging/VERIFICATION.md`.

| Test group | Required evidence |
| --- | --- |
| Installation | Empty machine/profile without Git or Python; no administrator rights; ordinary paths plus spaces/non-ASCII names; truthful progress; repeated setup; valid batch-install adoption. |
| Installed app | Shortcut launch; healthy loopback server; Picture upload; supplied video auto-open; playback speed; capture/review return; restoration and removal smoke checks; window-close cleanup; offline launch. |
| Failure paths | Blocked network; cancellation during download/setup; corrupted source/model; invalid ownership; insufficient disk space; startup failure; occupied port; running app; second launch; failed upgrade retaining the prior version. |
| Maintenance | Explicit upgrade; downgrade refusal; repair behavior; uninstall with default retention; optional cache removal; reinstall over retained data; developer `install.bat --source` and `start.bat --source` parity. |
| Release identity | Source commit; tool versions; artifact SHA-256; signing status; exact commands; separate cold-install, warm-cache, and clean-machine results. |

Browser verification remains a root-agent responsibility using the workspace's bundled inline Playwright workflow and its resolved target URL. Exercise the desktop window itself for host permissions, file selection, webcam access, and shutdown; a browser-only check cannot establish those behaviors. Uploaded video is the repeatable capture fixture. Record webcam and RTSP hardware checks separately when a device is unavailable. Desktop web is the design target.

A cached same-machine smoke test is not a clean-machine test. If a clean Windows machine/VM is unavailable, document that gate as unverified; do not claim release readiness. Record performance from measurements instead of retaining the proposal's speculative file sizes or installation-time estimates.

Gate: all required checks have evidence or an explicit unresolved status. Deliver a local review artifact before any merge or publication. A failed gate keeps the delivery in draft status.

## Completion criteria

- [ ] A local `ForensicDGP_Setup.exe` installs the pinned application as a standard user.
- [ ] A shortcut opens the dedicated window without a terminal dependency.
- [ ] Closing the window releases only the application's owned processes.
- [ ] Updates preserve the last good version and saved evidence on failure.
- [ ] Build instructions and verification evidence accompany the artifact.

## Approval and progress reporting

Approval of this plan starts implementation in the proposed isolated worktree. Report completion of each phase with its evidence and next gate. Surface architectural changes before proceeding with their dependent work. Do not merge to IMPLEMENTATION, publish a release, purchase signing services, or start offline packaging without separate authorization.

Continuing-work goal:

> After the user approves this plan, implement and verify the online Inno installer and owned desktop launcher in an isolated feature worktree. Preserve the existing managed installer safeguards, developer workflows, and saved evidence. Deliver a local setup EXE, reproducible build instructions, and a verification report for review. Keep unmet gates explicit. Do not merge, publish, purchase services, or implement offline packaging without separate authorization.

## Reference material

- [Current reviewed proposal](C:/Users/janus/Downloads/inno_installer_plan.md)
- [Inno download verification and progress](https://jrsoftware.org/ishelp/topic_isxfunc_downloadtemporaryfile.htm)
- [Inno non-administrator setup](https://jrsoftware.org/ishelp/topic_setup_privilegesrequired.htm)
- [Inno uninstall deletion guidance](https://jrsoftware.org/ishelp/topic_uninstalldeletesection.htm)
- [Windows SmartScreen reputation](https://learn.microsoft.com/windows/apps/package-and-deploy/smartscreen-reputation)
