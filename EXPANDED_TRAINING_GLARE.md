# Training glare diagnostic — 29 September 2026

Both expanded candidates fit glare on their two labeled training images but
fail to transfer that detection to the reviewed validation glare image. This
supports a generalization/data-coverage concern; it does not prove architecture
is sufficient for the full task or that adding data alone will fix it.

## Measurements

Inference only on all 68 real training cases. V2 cached feature image order and
hashes were checked against V3; targets were freshly loaded from V3. Old cached
V2 masks were not used as training/evaluation targets. The frozen encoder hash
matches the returned checkpoints. CPU aggregate gated real IoU differs from VM
training-fit results by -0.00000175 fixed / 0 anatomical. This close aggregate
agreement does not certify bitwise CPU/GPU equality.

| Training case | Added V3 glare pixels | Fixed recovered | Anatomical recovered |
|---|---:|---:|---:|
| uncovered_05.png | 1,345 | 1,080 (80.30%) | 1,108 (82.38%) |
| new_covered_48.png | 686 | 507 (73.91%) | 301 (43.88%) |

These are recall counts on V3-minus-V2 labeled glare pixels, not glare IoU or
precision. The second image includes a large medical mask, so whole-image IoU
would obscure poor glare recall. Presence probabilities are 0.99977 / 0.99326 for
the first example and 1.0 for the second; gating does not remove their glare masks.
The anatomical prediction has a visible hole in the second glare region.

All 25 clear training controls are empty after gating. Raw predictions contain
8 fixed / 10 anatomical false-positive pixels across two clear images per arm.
The validation glare case remains 0/491 recovered by either model even before
gating. No threshold or validation label was changed.

All 272 saved raw/gated training masks were independently recounted. A six-row
preview (both training glare cases plus first four clear controls in manifest
order) was inspected. These training examples establish fit, not generalization.

## Next justified work

Prepare a separate review queue of additional **training-only** real eyewear
and reflections, excluding existing validation/test sources and duplicate images.
Pair glare candidates with transparent-eyeglass controls. Existing source-role
review/annotation queues can supply candidates; no proposal becomes ground truth
until its visible reflection boundary and role are reviewed. Do not infer hidden
face targets or reuse the failed validation case as training data.

No additional identical VM run, threshold sweep or generator training yet.
Keep original real/synthetic safeguards, mannequin reporting and baseline models.
Synthetic retention is also still failing and must be assessed in any next pilot;
glare improvements alone cannot qualify a candidate. Independent final evaluation
and reviewed end-to-end completion improvement remain outstanding.

Script: `scripts/audit_expanded_training_glare.py`.
Local evidence: `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded_training_glare\`
(`results.json`, `preview.png`, per-arm raw/gated masks).
VM parent checkpoints remain in
`~/forensic-dgp/expanded_feature_bundle/outputs/expanded_feature_training/`.
No local training or application changes occurred.
