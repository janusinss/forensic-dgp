# Version 0.2.1 payload path hotfix

The supplied 0.2.0 setup log stops in `Verify-Payload` with
`Unexpected offline payload file.` This happens before destination preflight,
source preparation, or relocation. That log does not establish a conflict with
an earlier installation and does not record the offending inventory name.

## Reproduced defect

Windows PowerShell expands an 8.3 directory alias when returning a child's
`FullName`. The old verifier removed the length of the original root string
from that expanded name. Equivalent short and long root spellings can have
different lengths, producing a false inventory key. A real Windows PowerShell
5.1 regression using `GetShortPathNameW` failed with the same payload error
before the fix. A space in an account name alone is not the defect; the supplied
remote log does not record whether its stage argument used an 8.3 alias.

The verifier now carries each directory's manifest-relative prefix during
enumeration and appends the child's name. Inventory names no longer depend on
absolute path spelling. An actually unlisted file is still rejected and its
relative name is included in the diagnostic.

## Preserved checks

- Manifest and release input SHA-256 verification.
- Payload file sizes and SHA-256 verification.
- Duplicate, missing, and unlisted file rejection.
- Reparse-point refusal and destination preservation on early failure.

The hotfix branches from the published 0.2.0 release documentation commit
`ba0b68a`. Separate model-isolation work at `6673f0d` remains unchanged and is
not included. Existing-installation safeguards are unchanged. Version 0.2.0
release assets remain immutable; the fixed installer uses version 0.2.1.

## Verification status

The short-path regression failed before the patch. After the patch, 68 focused
installer/offline-runtime tests ran: 67 passed and one unchanged native-copy
fixture skipped because this isolated checkout has no local compiler copy. All
11 PowerShell wrapper cases passed, including 8.3, spaced Unicode, mixed case,
trailing separators, tampering, missing/unlisted files and an actual junction
refusal. The exact release EXE passed a native update using a genuine 8.3 temporary
folder in 222.49 seconds. All 6 checked record hashes were unchanged, the
verified environment was reused, Windows registration reports 0.2.1, and the
installed launcher passed health and scoped server-shutdown checks.

The immutable GitHub prerelease and Pages deployment
`https://115d2e5f.forensic-dgp.pages.dev/` are published. The canonical page
returns 0.2.1 metadata and the exact GitHub link/checksum. Installed Microsoft
Edge, through the bundled Playwright runner, passed 375/768/1280 layout checks
without overflow or page/console errors. The previous 0.1.1 download retained
its exact bytes and SHA-256. The complete anonymous public transfer passed: 1,045,722,009 bytes and the
exact release SHA-256 matched after 234.11 seconds. The streamed public file
was verified without saving another EXE or executing that downloaded copy. The affected PC's retest is separate from local proof.
No antivirus settings, Windows path policies, or system Python are changed.


## Exact release identity

- Build source: `6f7cf0b59b7391d154da9ec9be573d6a74f77b90`.
- EXE bytes: 1,045,722,009.
- SHA-256: `431fd74e2f643ccfc7b31407deaabeebaf12e6240e53f7ef8f2b6f35da78a884`.
- Public release: https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.1.

Evidence stays in ignored scratch: hotfix-focused-tests.log,
native-short-path-update-outcome.json, native-v021-launch-report.json and
public-v021-transfer.json. The main workspace holds hotfix-v021-site-report.json
and screenshots. Temporary copied payload staging was removed after retaining
build-input records; the upload handoff is a hard link to the retained EXE.
