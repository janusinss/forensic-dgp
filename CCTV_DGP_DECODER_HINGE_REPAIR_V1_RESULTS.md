# Decoder hinge repair: verified correction, rejected model — 15 October 2026

**The manual run stopped correctly after one optimizer update. No candidate
qualifies, no extra epoch is justified, and the application baseline is unchanged.**
The inactive brightness-gradient defect is repaired, but useful restoration and
preservation did not follow. Export completion packages the failed experiment.

## Audited comparison

| Saved state | RAW structure gain | PNG structure gain | RAW failures | PNG failures |
|---|---:|---:|---:|---:|
| Retained legacy first step | 0.091556% | 0.098722% | 27 | 28 |
| Corrected first step | 0.113913% | 0.128470% | 30 | 30 |
| Diagnostic 1/16 change | 0.007835% | 0.023974% | 21 | 20 |
| Diagnostic 1/4 change | 0.030022% | 0.031829% | 22 | 22 |

Structure gain is the relative reduction in landmark-supported high-frequency
luminance MSE against the baseline, not a percentage of human-visible clarity.
ArcFace recognition similarity is a separate preservation check. The unchanged
qualification threshold is at least 1%
globally and in each source, plus all preservation and brightness requirements.
Failures count overlapping group/metric decisions, not distinct people or images.
The corrected step fails 12 MSE, 10 SSIM and 8 recognizer comparisons in each
RAW/PNG path. All sampled states also have zero brightness-gain fractions;
passing that check does not compensate for the other failures. Diagnostic probes
are evidence only and never replace the full optimizer state or select a model.

Corrected source gains remain separate:

| TRAIN source namespace | RAW gain | PNG gain |
|---|---:|---:|
| `dataset/asian_faces/degraded` | 0.175269% | 0.188322% |
| `dataset/thumbnails128x128/degraded` | 0.098526% | 0.113454% |

Clear-image MSE worsens by 7.077% RAW and 14.738% PNG. Degraded-image MSE worsens
by 7.504% RAW and 7.608% PNG. This does not establish useful improvement over the
baseline. The root agent inspected all 20 eight-column, five-row sheets. Their
800 image cells match independently recomputed output pixels exactly. Facial
detail remains soft; severe compound cases remain insufficient for confident
restoration. This is an agent visual review, not independent final human review.

[Complete numerical audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/independent_audit_r1.json>).
[Per-reference visual observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/visual_observations.json>).
[Visual file and pixel bindings](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/visual_binding_audit.json>).

## What the correction proves

All 20 initial legacy penalty-gradient arrays are finite and nonzero. All 20
corrected inactive arrays are exactly zero; scalar penalties and metadata remain
identical. The worker used the intended corrected helper. Adam remained at
1e-4, with the same 15 decoder tensors, 609,219 elements, loss weights, anchors,
reference order, original-relative radius, splits and quality requirements.
The first update changed direction, so this return is not an unchanged rerun.
It also shows that the old derivative was not the only cause of failure.

Local auditing checked 600 RAW/PNG compositions, 300 exact baseline/initial/
legacy controls, all 17 groups in each path, 40 saved gradient arrays, two full
states, the saved first Adam update, projected vector bytes, frozen tensors,
optimizer step and exact probe-state/RNG restoration. Saved embeddings and CUDA
derivatives remain VM evidence; the local auditor did not run a fresh recognizer
or neural/gradient/optimizer call. The successful R1 audit took
107.29 seconds.

## Retained local audit stop and strict repair

**Location:** original local checker `verify_projection_journal`, line 90.
**Cause:** a CPU NumPy norm and VM reserve metadata differ by one binary64 ULP:
`7.041221370338014e-6` versus `7.041221370338013e-6`. The saved VM value matches
the independently derived correctly rounded exact reference. This scalar audit
portability issue did not change model weights or explain the quality failures.
**Fix:** the separate R1 wrapper uses an exact rational squared-spacing sum and
high-precision decimal reference. It requires exact equality, rejects even a
one-ULP metadata change, and retains exact actual-vector/radius checks. It does
not widen a quality or numerical tolerance. Six saved-vector regression tests
pass. The original checker, its failed receipt, all output bytes and protocol
remain unchanged.

[Preserved original failure](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/initial_audit_stop/failure.json>).
[Independent exact reserve proof](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/projection_reserve_oracle.json>).
[R1 metadata receipt](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/projection_metadata_r1.json>).

## Run, provenance and scope

The human-manual L4 run made 1 optimizer update, 42 gradient queries and 1
backward call. It saved 20 original-restorer batches, 20 prior-feature batches,
103 decoder batches and 243 recognizer batches. The worker quality-stopped
after 169.54 seconds; its external exit was code 2 without timeout or kill.
The saved audit and export succeeded. No worker, new parameter proposal or
training was launched by the agent in this return audit.

The 587,914,575-byte archive SHA256 is
`7c06ddca900866797f31974e2c5349b1e2d1bc2f8a25ad34a4444fbb17f83b22`.
Protocol SHA256: `a4d57c933214345a6d1cab5639007d15599f85e0db0450a0aad11aea5d715518`.
The import verified 828 regular members, 410 source, 203 parent and 136 replay
bindings before the full audit. Original checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.

This is 20 exposed photographic TRAIN references with 100 synthetic views,
ten references per source. The historical `thumbnails128x128` namespace here
uses audited HQ FFHQ counterpart targets; it is not evidence of 128px clean
targets. Dataset labels do not identify ethnicity or capture country. No native
CCTV, final identity, Zamboanga sample or covering-removal claim follows.
Paired errors stay separate from native unpaired evidence. The retained baseline
is a fine-tuned DeblurGAN-v2-compatible conditional restorer; the frozen external
StyleGAN2 prior and separate pretrained completion component remain declared.

[Selected next causal review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_HINGE_REPAIR_REVIEW_V1.md>). The full five-milestone goal,
seven automatic/assisted covering families and independent final review remain
active and unqualified. This packet is USED; do not resume or rerun it.
