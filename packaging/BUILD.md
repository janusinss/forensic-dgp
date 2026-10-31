# Build the online Windows installer

Status: preview under verification. No release has been published.

## Prerequisites

Use Windows x64 with Git, Python 3.12/3.13 x64, the Windows .NET Framework C#
compiler, and verified Inno Setup 7.1.1 x64. End users need none of the build tools.
The desktop preview needs Microsoft Edge. No shell package is installed silently.

The compiler installer is pinned to:
https://github.com/jrsoftware/issrc/releases/download/is-7_1_1/innosetup-7.1.1-x64.exe

SHA-256:
8a5a6711c3766a6e5fb74f280ba0e3551e719ce94538d901fdbd1f8f9920b3f0

Verify the checksum and a valid Authenticode signature from Pyrsys B.V. before
installing the compiler. Install it for the current user without elevation.
This checkout uses scratch/tools/inno-7.1.1/ISCC.exe; tools are not committed.

## Build

Commit the source intended for the artifact on the isolated feature branch.
Do not merge or publish merely to build a preview.

    python -B -m unittest discover -s tests
    node --test tests/*.test.cjs
    python packaging/build_release.py --iscc "scratch/tools/inno-7.1.1/ISCC.exe" --version 0.1.0

The builder refuses tracked modifications. Source payloads, launcher source and
installer scripts are read from the exact commit, so untracked records and model
weights are not included. The setup EXE bundles application source, while
Python/packages/models are obtained through the existing verified online setup.
A generated manifest records the exact source tree, ancestry and ZIP checksum.

Output is scratch/dist/ForensicDGP_Setup.exe with scratch/dist/release.json.
The latter records the source commit, tool version, input hashes and artifact hash.
Build output is ignored by Git. Reproducible instructions identify all inputs;
byte-for-byte reproducibility is not asserted for timestamped unsigned EXEs.

## Install, update and uninstall

Run the EXE as the intended standard user. Installation is fixed at
%LOCALAPPDATA%\Programs\Forensic DGP. The shared managed backend remains under
%LOCALAPPDATA%\DGP\app. Setup does not use administrator privileges.
Upgrade by explicitly running a newer tested setup EXE. Source ancestry protects
against replacing a newer or divergent backend with an older embedded payload.

Progress reports measured downloads and activity stages. Cancellation waits for
the current owned operation to finish before stopping at a safe boundary.
Activation is the commit boundary; a late cancellation request does not turn a
successful backend result into a false failure. If Windows shell installation
fails after backend activation, the backend can already be prepared; rerun the
same setup EXE to finish registration. Do not assume shell rollback restores the
backend pointer.

Windows Installed Apps uninstall removes the launcher, registration and shortcuts.
All shared backend files, environments, models and saved evidence remain.
Exclusive ownership of these batch-compatible resources is not established.
No cache-removal checkbox is exposed in this preview.

## Release limits

The generated setup and desktop launcher are unsigned. Inno packaging does not
eliminate SmartScreen warnings. Signing services are outside this implementation.
The current package pins direct dependencies but does not claim a complete
transitive hash lock. Never describe a cached local installation as a cold
machine test. See VERIFICATION.md for the actual evidence and open gates.

Offline packaging, automatic background updates, GitHub publication and merging
to IMPLEMENTATION require separate authorization.
