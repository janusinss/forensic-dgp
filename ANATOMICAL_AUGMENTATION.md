# Anatomical training augmentation v1 — 29 September 2026

Experimental implementation: `anatomical_augmentation.py`. Not integrated into
production inference, the benchmark loader, or any training runner. No fitting ran.

Uses five source landmarks (eyes, nose, mouth corners) to position an oriented
eye band or lower-face polygon. Rejects missing/nonfinite/out-of-bounds anchors,
extreme roll, implausible relative geometry, polygons outside the image and
implausible area. Checks the relevant anchor pixels are covered. Pixels outside
the generated mask are unchanged. Same seed gives the same texture. These are
opaque procedural shapes, not realistic accessories or simulated lens glare.

Four tests pass: anchors covered and binary mask/determinism/outside preservation,
explicit invalid-geometry rejection, no input mutation, unknown-kind rejection.
No optimizer is involved in tests or source-landmark inference.

## Source-only trial

352 sources: the previous 363 provisional candidates minus 11 additional flags.
Existing local InsightFace `det_10g.onnx` used for input-only five-point inference.
Exactly one face above confidence 0.6 required. Source hashes checked; detector and
augmentation hashes recorded in `outputs/training_diversity_audit/landmark_trial_v1/results.json`.
This checks reliable-looking geometry, not whether the detector's points are true.

Eye generation: 340 accepted, 12 rejected. Lower generation: 305 accepted, 47
rejected. Reasons: seven missing/ambiguous face detections, three mouth-depth
failures, one nose-order failure, one eye-separation failure; lower additionally
rejects 35 polygons leaving the image. Failures remain review records, not empty
successful masks or deleted sources. Source pose/crop distribution can affect
acceptance and must be reported before selecting a training subset.

First 12 jointly accepted examples inspected in `preview.png`. Several frontal
placements improve relative to fixed geometry; profiles can still yield narrow
or incomplete anatomical coverage. Covering two landmark centers is not proof
of covering the whole facial region. Need full accepted/rejected source review
before selecting data. Clean-target status also remains provisional.

The trial retains the old square resize solely to inspect placement changes;
non-square distortion remains unresolved. Runtime emitted ONNX output-shape
metadata warnings at 256 input (model metadata sizes differ from actual dynamic
outputs); inference completed. A subsequent direct-runtime audit at 256 and 640
confirmed finite outputs with counts matching two anchors at strides 8/16/32.
At 640 the output metadata also matches; at 256 it does not. Installed InsightFace
routes the model to SCRFD, which builds decoding anchors from actual input size.
See `outputs/training_diversity_audit/detector_shape_audit.json`; warnings remain
recorded separately. This verifies dimensional consistency, not landmark accuracy.
No inference uncovered reference is introduced:
landmarks here are for synthesizing examples from training sources only.

## Next

### Crop-aware V2 update

The rejection review found all 35 border-only lower-mask failures in Asian-source
crops (none in FFHQ). Dropping these cases would selectively reduce source coverage.
An explicit `boundary_policy='clip'` now uses `anatomical-covering-v2-clip`.
It rasterizes the complete polygon on a padded canvas, then crops to the photograph;
direct OpenCV clipping had a 13-pixel edge discrepancy caught by the crop-equivalence
test. Default V1 stays strict. All six tests pass; replaying 352 sources reproduces
all cached default V1 events. New V2 counts per anatomical kind: Asian 164/170,
FFHQ 176/182. Missing/ambiguous detections and geometry failures remain rejected.

Visually inspected all 47 original rejects, 12 accepted pose extremes and the 35
new clipped lower-face outlines. New masks intersect visible mouths; profiles can
remain narrow. These are partial procedural coverings, not verified full medical
mask shapes. No landmark truth or restored-face improvement is claimed. Review:
`outputs/training_diversity_audit/landmark_trial_v1/acceptance_review/visual_review.json`.
V2 replay: `outputs/training_diversity_audit/landmark_trial_v1/clipping_v2/results.json`.

Next implementation: version the expanded loader and source-balanced sampling for
a bounded VM experiment. Preserve benchmark, generic occlusions, explicit rejected
cases and original validation safeguards. No local training occurred.

Prepare the VM package after the expanded loader is checked. Existing
benchmark targets and deployed models remain unchanged. Glare data, detector
retention and end-to-end completed-face quality are still unresolved.
