# Forensic DGP download site

Live: https://forensic-dgp.pages.dev

A React/TypeScript landing page deployed with Cloudflare Pages.
The executable stays in build output; it is never committed to Git.

## Build and preview

    npm ci
    npm test
    npm run build
    npm run preview

The build verifies ../scratch/dist/ForensicDGP_Setup.exe against the generated
installer release manifest before copying it into public/downloads. A mismatch
stops the build. Public release metadata includes only version, size, SHA-256,
signing/preview status and download paths.

## Cloudflare

Deployment targets personal account b9e82b6514591358168877d8118aec18.
The default organization Wrangler login is intentionally not used.
These commands use a separate ignored config directory at
../scratch/cloudflare-personal and do not alter the other Wrangler login.

    npm run login
    npm run whoami
    npm run deploy

The wrapper targets project forensic-dgp and production branch main. The personal
account is pinned through the child environment; Pages does not support account_id
in its Wrangler configuration. The Pages-only flag prevents Wrangler from
automatically delegating a new project back to Workers. The standalone Worker
was retired after the Pages download passed verification.

Confirm the personal account before deployment. Do not use a temporary Cloudflare
account or change the configured account to bypass an authorization failure.
No custom domain or paid storage is configured. Both page and installer are
static assets in one Pages project.

## Updating the installer

Build and verify a new setup artifact first. Update any version-specific
noscript download link in index.html and content-type rule in public/_headers.
Run the build and tests again, then deploy. Download assets revalidate rather
than use immutable caching.

## Verification

The page was checked at 375, 768 and 1280 CSS pixels with no horizontal overflow
or browser exceptions. The primary link downloads the exact 6,304,374-byte EXE
whose SHA-256 is:
280a1f5ae936f14a42ca7ad9900edacbd82c1ef2122eb706516c394609e317e2

Sixteen site checks cover Pages/account isolation and release tampering, size/version validation,
ordinary-path boundaries and retention of unrelated files. Public deployment and header verification passed; see DEPLOYMENT.md.
