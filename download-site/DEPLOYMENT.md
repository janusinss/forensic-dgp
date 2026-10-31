# Cloudflare Pages deployment

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
