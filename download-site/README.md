# Forensic DGP download site

Live: https://forensic-dgp.pages.dev

A React/TypeScript landing page deployed with Cloudflare Pages.
The executable stays in build output; it is never committed to Git.

## Build and preview

    npm ci
    npm test
    npm run build
    npm run preview

The build verifies `../scratch/dist/ForensicDGP_Setup.exe` against its
`release.json`. A mismatch stops the build. The preparation step writes public
release metadata and a checksum. For offline releases, the download URL points
to the versioned GitHub release asset.

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
Pages hosts the landing page, release metadata, checksums and historical small
installers. GitHub hosts the current offline installer.

## Updating the installer

Build and verify a new setup artifact first. Update the fallback download in
`index.html` to match the generated URL. Run the build and tests, then publish
the GitHub prerelease and verify its installer size and SHA-256 before deploying
the page. Preserve historical downloads and checksums.

## Verification

Check the page at 375, 768 and 1280 CSS pixels. The website tests cover Pages
account isolation, release integrity, ordinary paths and retention of unrelated
files. See [deployment history](DEPLOYMENT.md) for release verification.
