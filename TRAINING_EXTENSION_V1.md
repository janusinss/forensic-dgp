# Reviewed training extension V1 — 30 September 2026

Local: `C:\xampp\htdocs\YEAR 4\Testing\dataset\detector_training_extension_v1\`.
VM destination when a recipe is ready: `~/forensic-dgp/dataset/detector_training_extension_v1/`.
Not uploaded or used for fitting yet.

The extension copies all 100 V3 records and their image/mask files unchanged,
then adds two opaque-lens examples and one transparent-glasses control to train.
It loads through the actual `detector_training.load_manifest` validator.

| Split | Covered | Clear | Total |
| --- | ---: | ---: | ---: |
| Train | 45 | 26 | 71 |
| Validation | 15 | 10 | 25 |
| Test | 4 | 3 | 7 |

Manifest SHA256: `12adcc9ad159fd92985fbd458673551350f24d375b2f8300a8a245b12463ee57`.
Every copied image/mask hash verified; all original record metadata unchanged.
Builder: `scripts/build_reviewed_eyewear_extension.py`. It refuses an existing
output directory and rejects stale audit/reference hashes, unreviewed additions,
source membership mismatch, overlap flags and exact source-group/image reuse.
Evidence: `build_verification.json` beside the manifest.

The added annotations are approximate assistant-reviewed polygons. They are not
expert ground truth, and identity-disjointness remains unverified. They use
full-source resizing to 256x256, not geometric face alignment. Existing runners
accept this format but do **not** enforce the experiment-readiness metadata:
do not treat a loadable dataset as authorization to launch a recipe.

No application, generator, training default or original V3 dataset changed.
Three extra images alone do not establish a fix for the previous retention failure.

Next: complete a broader reviewed mixed-covering batch before a bounded VM data
coverage comparison. Native queue cases 26 (`asian_face_09432.jpg`, straw at
mouth) and 43 (`asian_face_00184.jpg`, low-resolution microphone at chin) were
inspected as candidate non-mask examples; neither has an accepted pixel label.
Original 25 validation/7 test membership stays fixed, with previous test reuse
limitations retained. Original real/synthetic selection safeguards still apply.
