# Build Windows installer releases

Current release: **0.2.3 unsigned offline preview**, with YuNet video face capture,
the Face queue and restoration benchmarks. Release and artifact details are in
[0.2.3 deployment](../download-site/DEPLOYMENT.md). Earlier verification remains in
[R2_VERIFICATION.md](R2_VERIFICATION.md), [PAYLOAD_PATH_HOTFIX.md](PAYLOAD_PATH_HOTFIX.md)
and [VERIFICATION.md](VERIFICATION.md).

The offline release targets Windows x64 and contains private CPython, CPU-only
application packages, an isolated capture environment, and all six admitted
models. Legacy online builds remain supported for developer workflows.

## Build preparation

Use Windows x64, Git, Python 3.13 x64, the Windows Framework C# compiler, and the
verified Inno Setup 7.1.1 compiler. End users need none of these build tools.
The desktop window requires Microsoft Edge and Microsoft .NET Framework 4.8 or newer.

The compiler is pinned to the official innosetup-7.1.1-x64.exe release with
SHA-256 8a5a6711c3766a6e5fb74f280ba0e3551e719ce94538d901fdbd1f8f9920b3f0.
Verify its Pyrsys B.V. Authenticode signature before using it.

Prepare a fresh payload outside tracked source:

```powershell
python packaging/prepare_offline.py --output scratch/offline-payload --model-source <restoration-model-directory> --model-source <completion-model-directory> --write-locks
```

Only exact size/SHA-256 matches to the admitted model manifests are copied.
The preparer obtains the fixed official Python ZIP and Windows cp313 wheels.
No research directory is modified. Review the resulting application/capture
locks and dependency audit before committing them with the source changes.
An optional --wheel-source can reuse an earlier group wheelhouse; obsolete
packages are excluded and missing pinned versions are fetched during building.

Run the Python/JavaScript suites and commit the reviewed feature source before
creating the release. The builder reads committed Git blobs and rejects tracked
changes, changed preparer bytes, or payload locks differing from that commit.

```powershell
python -B -m unittest discover -s tests
node --test tests/*.test.cjs tests/test_restoration_queue.cjs
python packaging/build_release.py --iscc scratch/tools/inno-7.1.1/ISCC.exe --version 0.2.4 --offline-payload scratch/offline-payload
```

The command uses the next release version, 0.2.4. Published versioned assets are immutable.

Output: scratch/dist/ForensicDGP_Setup.exe and release.json. The release records
source ancestry, setup-input hashes, the offline manifest fingerprint, exact
artifact bytes and SHA-256. Offline packaging uses ZIP/1 with non-solid files to
reduce extraction time and permit random access to setup helpers. The single EXE
must remain below 2 GiB. Omit
--offline-payload only when deliberately building the legacy online variant.

## Offline installation contract

The payload is immutable and inventory-bound: Python ZIP, separate hashed
package locks/wheels, six models, and package notices with original-path indexes.
End-user setup never resolves against package indexes or silently falls back to
an online download. Pip and uv are patched release pins; the installed private
Python is derived from the fixed ZIP plus the checksum-bound pip update.

The selected folder contains the launcher and runtime: private Python, owned
environment generations, source versions, models, records and diagnostics.
No global Python, PATH, antivirus exclusion or Windows long-path policy change
is required. Offline wheel installation uses the extended-length spelling of
the exact computed environment prefix instead of moving a prebuilt venv.

Preflight checks prerequisites, actual resolved destination, write access and
payload-based temporary/destination space. Protected roots, links, unsupported
paths, conflicting ownership and unintended downgrades remain refused.

## Model-update support

Release 0.2.2 adds manifest-bound model-family namespaces for in-place model
updates. Published 0.2.0 and 0.2.1 remain unchanged. See
[MODEL_UPDATES.md](MODEL_UPDATES.md) for storage, activation and legacy compatibility.

## Existing installations

Setup distinguishes fresh, same-version, update, repair and relocation using
validated files and ownership as well as Windows registration. An absent target
with stale registration does not require manual uninstall first. A running app
must close before mutation; the new launcher supports a scoped close request.

Repairs create and verify a separate environment generation. They retain prior
files. A failed candidate does not replace the active application pointer.
Cancellation stops at a safe operation boundary; details and logs remain visible.
Capture imports are verified after bulk package/model writes finish. Cold setup
allows 120 seconds; normal installed health probes retain a 30-second bound.
Reused capture environments are checked against the committed full package lock;
installation checks must not depend on a temporary payload surviving afterward.
Measured model-copy progress is throttled without delaying per-block cancellation.

Relocation stages records and verified models in the new folder, recreates
Python/environments and location bindings, and verifies the candidate before
switching Windows integration. Historical JSON bytes and hashes are preserved;
old absolute paths remain historical provenance. A separate relocation record
identifies the completed change. The app does not automatically replay old jobs.

Optional cleanup runs only after completion and removes matching owned files.
Unknown or changed files remain. Older installations may retain unproven Python
or cache files; setup reports this instead of presenting cleanup as total removal.
The old uninstaller is never used against the newly registered location.

Starting with 0.2.4, Windows uninstall removes the verified desktop runtime,
including models, private Python, caches and saved records. The confirmation
describes permanent removal. Setup-created location records and Windows
shortcuts are removed, along with the selected folder once it is empty.
Unrelated files at the top level and legacy batch-managed DGP/app roots are
preserved. If a runtime file is locked, close the program using it and retry
uninstall from Windows Settings.

## Release verification and distribution

Test on the user's own PC in dedicated test folders. Report this as local-PC
verification, never a clean-Windows or universal hardware qualification. Keep
antivirus protection enabled and record actual alerts. A SmartScreen example is
not a confirmed detection cause. The preview remains unsigned; code signing is
not a delivered feature and no warning-free antivirus promise is made.

Keep the prior public download until the candidate passes acceptance. Large
offline EXEs are GitHub Release assets in janusinss/forensic-dgp-downloads; the
Cloudflare Pages site publishes only sanitized metadata/checksums and a fixed
versioned GitHub link. Website preparation streams the EXE hash rather than
loading the large file into memory. Never upload research records or credentials.

Download mirrors and automatic background updates are not included. The source
repository and developer batch workflow remain separate from the public downloads
repository. Old release gates in VERIFICATION.md are labeled historical.

## Future release rule

Published version assets are immutable. Never replace the published v0.2.0 EXE
or checksum with different bytes under the same filename/tag. Future installer
or IMPLEMENTATION changes require a new version, committed-source build,
acceptance checks, a new GitHub release, and updated Pages metadata pointing to
that exact new asset. Launching an installed app does not fetch branch changes.

The current app artifact identifies source c8ff04e83bd4a722bc73134f67c0c1b1719ecb36
on the local installer feature branch. No IMPLEMENTATION merge or source push has
been performed by this release work. Publishing the downloads-only repository
and static landing page does not change that source-branch state.

The published browser download had Mark of the Web but was not executed. Local
installer/application checks and public download integrity do not establish a
warning-free SmartScreen result or acceptance by every antivirus product.
