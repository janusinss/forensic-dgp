# Completion R2 release 0.2.2

Verified on the development PC, 2 November 2026 (Asia/Manila).

## Exact identities

| Item | Identity |
| --- | --- |
| IMPLEMENTATION application revision | `2c5d8fd7c557b68ecd1eed2c6f29091ac1dcd6d9` |
| Offline installer build source | `453d563bdd6c1c36b92107b869dcc6e6f70e4a8e` |
| EXE bytes | 1045576043 |
| EXE SHA-256 | `bc9952a5e54a3955a06a7e3b636d008f8dee2f227ef470224ee927463d733a25` |
| Offline payload manifest SHA-256 | `59a4d99c977e209273fc2ff2e4d08e4b474a224f55184be18e2660467bcde215` |
| Completion R2 SHA-256 | `ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c` |
| Completion R2 logical state | `300b13b8f01db17aa95ee95ca41dbbdfbd4e5e3257dd16657fcc7950d3351ea0` |
| Completion R2 training protocol | `463c1977e9cd74f0a4fd0240df7a784f48b613043ef667714f1ef94adb4cd60a` |

The application release changes optional completion from R1 to R2. Restoration
weights, preparation and inference math, the covering detector, and the manual
review flow remain unchanged. The installer combines this update with model-set
isolation and the 0.2.1 Windows short-path verification fix. R2 weights use their
own filename and exact admission fingerprints; missing R2 never falls back to R1.
Historical source attribution is retained alongside R2 provenance.

## Native existing-installation update

The exact EXE updated the retained task-owned 0.2.1 installation in 318.03 seconds.
It used a genuine Windows 8.3 TEMP/TMP spelling, exercising the previous short-path
failure. Child-only HTTP/HTTPS proxies pointed at a closed loopback port; this is
not an OS-enforced network-disconnection test.

All 6 prior flat model files and the 4 saved-record files checked retained their
sizes and SHA-256 values. All 6 newly namespaced model files verified. The active
pointer changed to the exact build source only after verification. The existing
verified environment generation and capture packages were reused. The first
namespaced migration intentionally retains the older flat files; no model garbage
collection was performed.

## Installed browser and independent output checks

The installed app passed real V2 restoration On, import of a fixed reviewed
cloth-mask area, explicit review/confirmation, one R2 CPU completion, and actual
PNG/ZIP downloads. R2 took 27.8 seconds on this PC. Editing the selection to empty
passed with 0 neural calls and an unchanged image; its full supervised stage took
1.5 seconds. These timings are observations, not minimum-device certification.

Independent download auditing verified ZIP CRCs, all 15 manifest entries in each
bundle, model/protocol/logical-state bindings, and saved raw-to-PNG replay. The
nonempty request preserved all 141015 RGB bytes outside its 18531-pixel mask; the
empty request preserved all 196608 RGB bytes. This proves preservation relative
to the supplied restoration-result canvas, not recovered hidden identity.

Installed Edge checks found no page/console errors and no horizontal overflow at
375, 768 and 1280 pixels. The desktop launcher became healthy and its scoped close
request ended its server. The separate test backend was stopped after the model
controllers reported no active workers.

The first browser attempt was refused before inference by the existing 2.5 GiB
available-RAM guard. A test response filter also used the wrong endpoint. The
filter was corrected and only the duplicate task-owned desktop window was closed;
the memory guard and unrelated applications were not changed. The successful
flow used the installed source and private environment without that extra window.

## Regression checks

| Check | Result |
| --- | --- |
| IMPLEMENTATION Python suite | 132 run, 4 skipped |
| Integrated installer Python suite | 391 run, 6 skipped |
| Application JavaScript | 34 passed |
| Download page tests | 19 passed |
| Download page build | Passed |

Three existing wrapper assertions required whitespace normalization because
PowerShell wrapped diagnostic text in the longer worktree path. The underlying
negative tests still require refusal, preservation and no destination creation.
No production guard changed. The accepted EXE source predates only these test
formatting corrections and final release documentation/website metadata.

Dependency versions remain unchanged. Current package auditing reported no known
vulnerabilities among covered packages; CPU-specific Torch was skipped by the
PyPI audit. The website audit reported zero vulnerabilities. This is not universal
security or antivirus certification.

## Distribution

Model release: [completion-r2-models-v1](https://github.com/janusinss/forensic-dgp/releases/tag/completion-r2-models-v1).
Full anonymous transfers of R2 and the unchanged detector matched their exact
sizes and hashes. Installer release: [v0.2.2](https://github.com/janusinss/forensic-dgp-downloads/releases/tag/v0.2.2).
The canonical download page is [forensic-dgp.pages.dev](https://forensic-dgp.pages.dev/).
The complete anonymous EXE transfer matched the exact size and SHA-256 in 130.93
seconds without saving a duplicate installer. Pages deployment:
`ff9edb57.forensic-dgp.pages.dev`. Live metadata, link and checksum checks remain
in the rollout's local audit artifacts.

This remains an unsigned, noncommercial research preview tested on this PC.
Clean-Windows first-install execution, physical-device compatibility and
Internet-Zone SmartScreen/antivirus acceptance were not established by this run.
Research promotion and functional/pixel checks do not qualify every covering
family or establish independent final hidden-face accuracy.
