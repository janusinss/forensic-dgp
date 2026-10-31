# Cloudflare deployment

Public URL: https://forensic-dgp.janusdominic0.workers.dev

Direct installer:
https://forensic-dgp.janusdominic0.workers.dev/downloads/ForensicDGP_Setup-0.1.1.exe

Deployed2026-10-31 to the user's personal Cloudflare account.
Worker: forensic-dgp.
Cloudflare version ID:ec2ece28-d490-437b-bc86-fe13add2e922.
Method: Wrangler4.156.0, Workers Static Assets,15 scanned assets.
The existing organization CLI login was preserved; deployment used the separate
ignored personal-account configuration.

## Installer identity

Version0.1.1 is an unsigned research preview. The hosted executable is exactly
6,304,374 bytes. Its SHA-256 is:

280a1f5ae936f14a42ca7ad9900edacbd82c1ef2122eb706516c394609e317e2

The public download and its checksum were fetched after deployment and matched
the original verified local artifact. No research images, private records,
source maps, credentials or internal verification logs were published.

## Verification

- React/TypeScript/Vite production build passed.
- Seven release-preparation tests passed, including tamper/size/path failures.
- Local and live Chromium checks passed at375,768,1280 CSS pixels without
  horizontal overflow, page exceptions or console errors.
- The real download link produced the correct suggested filename and exact EXE.
- Production returns attachment and application/octet-stream for the installer.
  SHA-256 and license downloads return200.
- Security headers include self-only Content-Security-Policy, nosniff,
  DENY framing and disabled camera/microphone/geolocation on this download page.
- Requests for /.env, /.git/config and /_headers return404.

Impeccable's bounded scan reported intentional speech-triangle borders as
side-tab patterns and flagged the incumbent Inter face as common. These were
retained to match the application's established branding; screenshots showed
no layout defect.

Evidence for version 0.1.1 remains in the main workspace's ignored scratch/download-v011: live-report.json, live-375.png, live-768.png, live-1280.png and the exact downloaded EXE. The report records the artifact checksum, response headers, blocked paths and browser errors.

## Scope

This deployment publishes the website and setup executable, as requested.
Installer/application source remains on feature/inno-desktop-installer.
No IMPLEMENTATION merge, GitHub release, custom domain, paid R2 storage,
analytics or user-upload service was created.

The hosted preview retains the installer verification limits recorded in
../packaging/VERIFICATION.md. Publishing this download page does not establish
clean-machine testing, camera-specific behavior or model qualification.
