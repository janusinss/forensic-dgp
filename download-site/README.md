# Forensic DGP download site

A React/TypeScript landing page deployed with Cloudflare Workers Static Assets.
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

Confirm the personal account before deployment. Do not use a temporary Cloudflare
account or change the configured account to bypass an authorization failure.
No custom domain or paid storage is configured. Both page and installer are
static assets in one Worker.

## Updating the installer

Build and verify a new setup artifact first. Update any version-specific
noscript download link in index.html and content-type rule in public/_headers.
Run the build and tests again, then deploy. Download assets revalidate rather
than use immutable caching.

## Verification

The page was checked at 375, 768 and 1280 CSS pixels with no horizontal overflow
or browser exceptions. The primary link downloads the exact6,232,315-byte EXE
whose SHA-256 is:
3c055fe550149b9c3aee7f51bda59f5af7ebed683cbf3970d3d7c9b1cb50e4a5

Seven release-preparation fixtures cover tampering, size/version validation,
ordinary-path boundaries and retention of unrelated files. Public deployment
and header verification must be recorded separately after successful upload.
