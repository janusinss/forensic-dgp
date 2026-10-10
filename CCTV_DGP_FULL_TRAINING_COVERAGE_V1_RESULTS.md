# Full TRAIN coverage and native input review — 10 October 2026

The frozen review verifies all 3,905 legacy TRAIN inputs, all 781 canonical TRAIN
targets and the 24 previously usable native development crops. These are input,
target, geometry and pixel-filter measurements. There are no neural calls,
gradients, optimizer updates, epochs, new thresholds or app changes in this review.
Validation and reserved-final pixels were not decoded. Native crops stay unpaired.

The independent checker verifies 5,528 source bindings, all canonical target pixel
hashes and all TRAIN geometry/pixel metrics. A separate OpenCV implementation of
the worker's SciPy high-pass filter differs by at most 3.46945e-18. This arithmetic
check does not establish the validity of the Auto quality heuristic as a face
sufficiency classifier. All native usability labels remain unchanged.

## Source-specific findings

| Retained TRAIN reference source | References | Median target landmark high-frequency energy | Clear cases selected by existing Auto |
| --- | ---: | ---: | ---: |
| Audited HQ FFHQ counterparts | 391 | 0.00281990 | 0/391 |
| Retained `dataset/asian_faces` photographic replay | 390 | 0.000628738 | 336/390 |

All 3,124 degraded TRAIN cases trigger the current Auto rule. The rule selects
11/24 native development inputs. No source label is an ethnicity label or evidence
of performance in Zamboanga City. Native capture country remains unspecified.
No new acquisition or terms change occurs here; existing provenance, terms,
source identities, exposure history and split roles remain binding.

Geometric eye spacing projected onto the TRAIN degradation sampling grid is:

| Source/profile | Minimum–maximum grid pixels |
| --- | ---: |
| HQ FFHQ blur/compound | 4.55–6.95 |
| HQ FFHQ low light / motion | 6.06–9.27 / 9.09–13.90 |
| Asian photographic replay blur/compound | 4.62–11.85 |
| Asian photographic replay low light / motion | 6.16–15.80 / 9.25–23.71 |
| Native development annotated eye spacing | 23.02–47.04 |

The full corpus has a small geometric overlap through the largest replay motion
cases. The earlier 100-case review's non-overlap statement applies to that sample,
not the whole corpus. These distances do not measure equivalent resolved detail,
blur, optics, illumination or compression. The native range still motivates
examining less severe degradation coverage after the demonstrated spatial-path
limitation is addressed; it does not prove a unique cause of reconstruction error.

## Canonical target hashes and supervision

All 781 clear inputs exactly equal their actual canonical targets. For the 391
HQ FFHQ counterparts, `mixed_protocol_v9.json` explicitly records that the legacy
`target_rgb_sha256` describes historical reduced thumbnails. The actual canonical
pixel hash is `canonical_target_rgb_sha256[reference_id]`; all 781 match it.
The 391 expected legacy mismatches are not newly discovered corrupted targets.
The 390 photographic replay targets match both fields. Historical FFHQ
`native_size:128` describes the original thumbnail, not the HQ source acquisition.
No original metadata or hashes were rewritten.

The retained identity-v2 recipe is verified as Adam with backbone learning rate
2e-6, head learning rate 1e-5, weight decay 1e-5 and gradient clipping 1. Stored
normalization statistics were frozen. The selected branch contributes two
additional epochs/226 updates; its full ancestral epoch count is unconfirmed.
These historical rates are not newly demonstrated optimal rates for a repaired
branch. Additional epochs 1/2/5 remain a later finite-study target.

## Evidence

- [Frozen plan](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_full_training_coverage_v1/plan.json>)
- [Full measured rows](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_full_training_coverage_v1/results.json>)
- [Independent audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_full_training_coverage_v1/independent_audit.json>)

This closes the broader input/target inventory. It qualifies no model, leaves
final identities outside tuning, and completes neither restoration nor the seven
completion families. The full goal remains active.
