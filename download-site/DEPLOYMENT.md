# Cloudflare Pages deployment

## Current 0.2.11 offline preview

Published November 14, 2026 in Asia/Manila. The
[live download page](https://forensic-dgp.pages.dev/) runs
[Pages deployment bd2c3680](https://bd2c3680.forensic-dgp.pages.dev/).

The [GitHub prerelease](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.11)
references `ForensicDGP_Setup-0.2.11.exe` from build source
`7e78f28ae420ddb1e4c5b95668b186fb47bfdf5c` (`1,314,084,061` bytes, SHA-256
`05a4640e302a37c258d4cbd1fa388d854645269a029e7f0101de476c366b1051`). This release
adds automatic shutdown of running Forensic DGP instances during upgrade/uninstall
and `\\?\` long-path safe uninstall cleanup in `packaging/uninstall.ps1` and
`packaging/installer.iss`. All 19 site tests and `tsc --noEmit` + Vite production
build passed. Published versions remain immutable: future updates must use a new
version, build, and release rather than replacing `0.2.11`.

## Historical 0.2.10 offline preview

Published November 14, 2026 in Asia/Manila. The
[live download page](https://forensic-dgp.pages.dev/) runs
[Pages deployment 343115ac](https://343115ac.forensic-dgp.pages.dev/).

The [GitHub prerelease](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.10)
references `ForensicDGP_Setup-0.2.10.exe` from build source
`2c70789c967493ac7e9094a0332d2f2db803e142` (`1,314,080,732` bytes, SHA-256
`a4ed08386f2125c8be9937045c53a5520cb13f29761f2e6442ae9dd95ff7f66b`). This release
ships aligned GT reference evaluation across native, padded, and manual crops
plus hardened Windows PE version/manifest metadata. The published `/release.json`
and `/downloads/ForensicDGP_Setup-0.2.10.exe.sha256` checksum were verified live.
All 19 site tests and `tsc --noEmit` + Vite production build passed. Published
versions remain immutable: future updates must use a new version, build, and
release rather than replacing `0.2.10`.

## Historical 0.2.9 offline preview

Published November 14, 2026 in Asia/Manila. The
[live download page](https://forensic-dgp.pages.dev/) runs
[Pages deployment 5cbf135f](https://5cbf135f.forensic-dgp.pages.dev/).

The [GitHub prerelease](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.9)
contains `ForensicDGP_Setup-0.2.9.exe` from build source
`e874076519ca9063797cda4ea3b0307b2bb0e3cf`. GitHub confirmed
1,314,071,849 bytes (`Content-Range: bytes 0-0/1314071849`) and SHA-256
`3944d95b2c388fc7ece38b0ff214a0a58ef585030c6b68c35483eee27bcb94f6`.
The published `/release.json` and `.sha256` checksum were verified live. All 19
site tests and `tsc --noEmit` + Vite production build passed. The live page
passed Playwright checks at 375, 768 and 1280 pixels without horizontal overflow
or browser errors.

## Historical 0.2.8 offline preview

Published November 14, 2026 in Asia/Manila. The
[live download page](https://forensic-dgp.pages.dev/) runs
[Pages deployment e487a1a1](https://e487a1a1.forensic-dgp.pages.dev/).

The [GitHub prerelease](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.8)
contains `ForensicDGP_Setup-0.2.8.exe` from build source
`e837197bda6f95761fa2f16ca275e62322fa103b`. GitHub confirmed
1,314,059,206 bytes (`Content-Range: bytes 0-0/1314059206`) and SHA-256
`d804ddbca9ed6697a6c37ff4ad348a8c55541f509c6e35614c7aff2ea9a5697e`.
The published `/release.json` and `.sha256` checksum were verified live. All 19
site tests and `tsc --noEmit` + Vite production build passed. The live page
passed Playwright checks at 375, 768 and 1280 pixels without horizontal overflow
or browser errors.

## Historical 0.2.6 offline preview

Published November 9, 2026 in Asia/Manila. The
[live download page](https://forensic-dgp.pages.dev/) runs
[Pages deployment 91204018](https://91204018.forensic-dgp.pages.dev/).

The [GitHub prerelease](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.6)
contains `ForensicDGP_Setup-0.2.6.exe` from build source
`19aff929e5f18e4a928f9ca5558d93875a46cd38`. GitHub confirmed
1,045,830,949 bytes and SHA-256
`d35eeabbe27c43d86cb52e5322c5778c82fd6eb2cbcc76444708bdb91baf0ad9`.
The public download responded to a range request with the expected total size.
The published checksum, no-script download, and previous release checksum
were verified. All 19 site tests passed. The live page passed at 375, 768 and
1280 pixels without horizontal overflow or browser errors.

The previous production deployment was
[0.2.4 on Pages](https://4f620787.forensic-dgp.pages.dev/).
The [0.2.4 installer](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.4)
and its checksum remain available.

## Historical 0.2.2 Completion R2 release

Published 2026-11-02, Asia/Manila. Canonical page: https://forensic-dgp.pages.dev/.
Deployment: https://ff9edb57.forensic-dgp.pages.dev/.
Installer: https://github.com/janusinss/forensic-dgp-downloads/releases/download/v0.2.2/ForensicDGP_Setup-0.2.2.exe.
Build source: 453d563bdd6c1c36b92107b869dcc6e6f70e4a8e.
Bytes: 1,045,576,043. SHA-256:
bc9952a5e54a3955a06a7e3b636d008f8dee2f227ef470224ee927463d733a25.

The full public EXE transfer matched this size and hash in 130.93 seconds without
storing another EXE. Native update, R2 inference and preservation evidence are in
[R2_VERIFICATION.md](../packaging/R2_VERIFICATION.md). The existing personal
Cloudflare login was reused; legacy Pages downloads and older GitHub releases
remain available. Pages hosts sanitized release metadata and checksums; the large
EXE remains a versioned GitHub asset.

## Historical 0.2.1 path hotfix

Published 2026-11-02, Asia/Manila. Canonical page: https://forensic-dgp.pages.dev/.
Deployment: https://115d2e5f.forensic-dgp.pages.dev/.
Installer: https://github.com/janusinss/forensic-dgp-downloads/releases/download/v0.2.1/ForensicDGP_Setup-0.2.1.exe.
Build source: 6f7cf0b59b7391d154da9ec9be573d6a74f77b90.
Bytes: 1,045,722,009. SHA-256:
431fd74e2f643ccfc7b31407deaabeebaf12e6240e53f7ef8f2b6f35da78a884.

The release fixes Windows short-path payload verification without changing
models or existing-installation safeguards. The canonical page, download link,
checksum and 375/768/1280 layouts passed via installed Edge. The complete anonymous transfer matched the exact bytes/SHA-256 in 234.11
seconds; see ../packaging/PAYLOAD_PATH_HOTFIX.md. Legacy Pages EXEs
0.1.0/0.1.1 were preserved; published GitHub 0.2.0 assets were not changed.
The authorized personal-account login was reused from its existing isolated
configuration; no organization login or account settings changed.

## Historical 0.2.0 offline preview

Published 2026-11-01, Asia/Manila.

- Public page: https://forensic-dgp.pages.dev/
- Installer: https://github.com/janusinss/forensic-dgp-downloads/releases/download/v0.2.0/ForensicDGP_Setup-0.2.0.exe
- Deployment: https://f0d4b9ce.forensic-dgp.pages.dev/
- Build source: c8ff04e83bd4a722bc73134f67c0c1b1719ecb36.
- Installer: 1,045,716,354 bytes; SHA-256 4287feb22c96c9abaf5804592938b0c4de2dbfbeecc747cb00a93543a9a5635a.

The EXE is a public GitHub prerelease asset. Pages serves the landing page,
checksum and sanitized metadata. The exact previous 0.1.1 asset remains available
at its historical Pages URL below. Full anonymous browser and HTTP downloads of
0.2.0 matched the accepted size/hash. Production layout checks passed at
375/768/1280 pixels without overflow or page/console errors.

The browser download had Internet-zone metadata, but that downloaded copy was
not executed. Local-artifact installation and application checks are documented
separately in ../packaging/VERIFICATION.md. This remains an unsigned local-PC
preview, with no SmartScreen/antivirus or clean-machine qualification.

Future deployments run `npm run deploy` only after the accepted new EXE and
matching release.json have been promoted into ../scratch/dist. Preparation
verifies that default input before building. Published versions are immutable:
use a new version/build/release for future updates. See ../packaging/BUILD.md.
Source work remains on feature/inno-desktop-installer; IMPLEMENTATION was not
merged or pushed during this release.

## Historical 0.1.1 Pages migration

Public URL: https://forensic-dgp.pages.dev/
Direct installer: https://forensic-dgp.pages.dev/downloads/ForensicDGP_Setup-0.1.1.exe
Deployment URL: https://0b19972f.forensic-dgp.pages.dev/
Date: 2026-10-31, Asia/Manila.

Project: forensic-dgp. Production branch: main. Hosting source: 2bd942a.
Personal Cloudflare account: b9e82b6514591358168877d8118aec18.
Method: Wrangler 4.156.0, Pages Direct Upload. The existing isolated personal
login remains separate from the organization CLI configuration. Pages access
was authorized by the user for this migration.

## Existing installer

Version 0.1.1 remains the same unsigned research preview: 6,304,374 bytes.
SHA-256: 280a1f5ae936f14a42ca7ad9900edacbd82c1ef2122eb706516c394609e317e2.
Build source: 90d43aa9687edf46f0cb4ca4fa9b4db5ea8a5b5d.
The installer was neither rebuilt nor executed for this hosting migration.

## Cutover verification

The exact production Pages address returned 200. The actual download link
returned the expected filename and unchanged installer bytes/SHA-256.
Attachment and application/octet-stream headers passed. Existing CSP and other
security headers were preserved. The page passed 375, 768 and 1280 pixel checks
without horizontal overflow, page exceptions or console errors. A top-level
404.html preserves 404 responses for unknown paths, including /.env,
/.git/config, /wrangler.jsonc and /_headers.

All 16 website checks passed: nine Pages/account-isolation/CLI regressions and
seven existing release-integrity fixtures. The TypeScript/Vite build passed.
The wrapper pins the personal account in its child environment, clears primary
and legacy competing credential aliases and refuses Worker-specific commands.
The explicit Pages flag avoids Wrangler's automatic new-project delegation to
Workers. Future npm run deploy operations target this Pages project.

## Retired Workers endpoint

After successful Pages checks, the user-authorized deletion removed only the
forensic-dgp Worker from the personal account. Its version previously had only
an ASSETS binding. The old https://forensic-dgp.janusdominic0.workers.dev/ address
returned 404, and the Cloudflare deployments API returned code 10007:
This Worker does not exist on your account.

Wrangler's optional legacy KV cleanup returned an authentication error after
the Worker service deletion because the login has no KV scope. No additional
KV access was requested. Worker removal was independently verified by the HTTP
and API checks above; no KV binding was attached to this site.

No installer changes, other project deletions, IMPLEMENTATION merge, GitHub
release, custom domain or paid storage were performed.

## Evidence

Main workspace scratch/pages-migration contains live-report.json, screenshots
and the downloaded EXE. Feature scratch/pages-migration contains the previous
Worker configuration/version snapshot, retirement-report.json and the retired
Worker API response. Evidence and credentials are ignored by Git and were not
published. Installer qualification limits remain in ../packaging/VERIFICATION.md.
