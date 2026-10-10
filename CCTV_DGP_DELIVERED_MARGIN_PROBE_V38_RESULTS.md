# V38: finite preservation progress, useful restoration still unqualified

The human's V38 return is verified. The quarter and eighth trials pass the
unchanged delivered PNG preservation, source-gain and brightness checks in both
50-case TRAIN design cohorts. Neither full nor half scale passes both cohorts.
This is evidence for a small disposable displacement, not a trained replacement,
useful native CCTV restoration, an independent evaluation or app readiness.

The 388,426,632-byte archive SHA256 is
`c1403d67ba0a66a1200285bcab291cd87dc9d6d7bf0518534f3b07a12ebff876`.
The protocol SHA256 remains
`caea39469005b7066d7cdbd74e073ba674b9b9b6fb9324f3110fe827a06948ab`.
The prospective checker was frozen before the human ran the probe. It verifies
2,135 regular archive members, all 500 raw/PNG/vector measurements, 170 group
aggregates, the 210-row saved-array geometry and 100 frozen CPU replay outputs.
Runtime is 240.167 seconds inside a 241.612-second supervisor. CPU/VM maximum
raw error is 2.59653e-6, PNG difference is one byte and vector error is 1.03563e-6,
within the predeclared replay tolerances. These do not relax quality gates.

The L4 probe takes 111.690 seconds. It makes four reset trials, zero new gradient
queries, optimizer updates, committed trajectory updates or epochs, and writes
no checkpoint. The original DGP and fixed recognizer are restored. Export
`complete:true` means the archive is complete; it does not imply training success.

| Scale | Exposed TRAIN structure gain | Other TRAIN structure gain | PNG failures, exposed / other |
|---|---:|---:|---:|
| 1 | 0.299483% | 0.409106% | 1 / 4 |
| 0.5 | 0.147063% | 0.209282% | 0 / 1 |
| 0.25 | 0.068478% | 0.116937% | 0 / 0 |
| 0.125 | 0.036274% | 0.069911% | 0 / 0 |

Structure gain means reduction of the existing landmark high-frequency error
against paired photographic targets. It is not a percentage of recovered
identity or native CCTV accuracy. The historical cohort name `unexposed` denotes
TRAIN design examples already used in this investigation; it is not a holdout.
Acquisition labels asian_faces and FFHQ describe sources, not ethnicity.

The six PNG failures remain in the evidence: one source/profile ArcFace failure
in exposed/full, four source/profile SSIM failures in other/full, and one SSIM
failure in other/half. V36's 11 failures also remain. The descriptive raw MSE and
raw ArcFace recount has no failures here; it does not override delivered-image
failures and is not a claim that every raw metric has passed.

All 100 fresh original raw/PNG files are byte-identical to V36's originals.
Both smaller steps have positive structure gains for both acquisition sources
and pass the existing brightness-fraction limit of 0.2. Their maximum per-channel
PNG changes are four and two bytes respectively. All eight comparisons, including
failed larger steps, are retained.

The implementing assistant actually inspected all 20 pages at original 256-pixel
cell size: 100 cases, 500 model outputs and 200 input/target cells. A separate
checker verifies every one of the 700 cells, case order, decisions, baseline
parity and all 14 unchanged app/checkpoint bindings. Degraded eyes, nostrils, lips
and tooth boundaries remain weak. Motion generally retains more coarse structure
than severe blur/compound input, but the passing small trials do not show a
convincing additional clarity benefit. Softness alone is not rejection; a useful
visible-structure improvement has not been demonstrated.

The quarter step is the largest measured step passing both TRAIN cohorts. It is
fixed before new development inference, with state
`b7aad53d93d4fa58ca826be6162667fff2faf85578da4f421e938b8711bc49ba`.
It was checked on the existing native 24-crop gallery and separate paired
520-case development set through a frozen disposable copy. All native comparison
cells were visually reviewed; no convincing incremental clarity was demonstrated.
The paired check improves degraded high-frequency error by0.097128%, but retains
one source/profile ArcFace regression. The quarter step is rejected for adoption.
No optimizer, autograd, state fitting or checkpoint writing occurred locally.
No larger failing scale or development-selected smaller scale is substituted.
See [the complete development findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V38_QUARTER_DEVELOPMENT_RESULTS.md>).

The 1%-at50 and 10%-at800 training-capacity requirements remain. V38 performs no
training trajectory and cannot pass either. Previous capacity/development and
covering failures, splits, caches, original checkpoints and local backup remain.
Native CCTV has no aligned clean reference; no native PSNR/SSIM or hidden-identity
claim is made. Reserved final identities remain unopened. The DGP-led app and
both automatic and assisted covering quality remain unqualified. Goal active.

[Independent return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_delivered_margin_probe_v38_independent_audit.json>)
[All-eight-trial analysis](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1/analysis.json>)
[Actual full image review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1/visual_review.json>)
[Independent cell/decision verification](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1/independent_analysis_page_audit.json>)
