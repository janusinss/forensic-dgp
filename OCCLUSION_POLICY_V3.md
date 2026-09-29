# Occlusion policy V3 — strong lens glare included

User decision,29 September2026: estimate facial regions hidden by strong lens glare.
Single-image output remains a plausible estimate, not recovered identity evidence.

## Annotation scope

1. Include an opaque covering or strong lens reflection where facial structure is
   obscured. Mark the obscuring region, not the entire eyeglass lens by default.
2. Preserve transparent lenses, visible eyes, normal highlights, teeth, skin and
   clothing/background. Brightness alone is insufficient to label an occlusion.
3. Mark uncertain glare boundaries for review; do not manufacture precise hidden
   ground truth. Real images supply occlusion masks, not reference uncovered faces.
4. Preserve existing split membership and provenance. Review the complete dataset
   consistently, including covered cases, rather than relabeling only model failures.
5. Keep V2 immutable. V3 candidates cannot enter training until policy review and
   mask validation finish. Previously inspected test cases are not untouched tests.

## Evaluation transition

The V2 false-positive findings remain valid under V2's labels. They do not become
successes retroactively. New policy means new versioned labels and a fresh baseline
evaluation of the original detector and candidates on the same V3 cases. Preserve
the existing comparison rules and synthetic benchmark; never weaken thresholds
or delete difficult cases to obtain a pass. Report glare as a separate stratum.

Automatic brightness thresholds or classifier predictions cannot approve their
own ground truth. Record each review decision, rationale, mask hash and provenance.
Uncertain cases stay pending. No label changes have yet been made.

Local review queue: `c:\xampp\htdocs\YEAR 4\Testing\dataset\detector_glare_review_v3\audit_queue.json`.
VM counterpart: `~/forensic-dgp/dataset/detector_glare_review_v3/audit_queue.json`,
available only after explicit transfer. This queue is not a training manifest.

Next: inspect all100 existing crops under this policy, create proposed masks for
affected cases, review uncertain boundaries, then validate the versioned labels
before any new training or claims of improvement.

## Source-only review completed

All100 source crops inspected in four contact sheets;14 glasses cases inspected
in enlarged views.96 records have no proposed glare addition. Four provisional
polygon additions were drawn from source pixels, not detector predictions:

| Image | Existing split | Added pixels at256px |
|---|---|---:|
| covered_04.png | test | 237 |
| uncovered_05.png | train | 1345 |
| new_covered_48.png | train | 686 |
| new_uncovered_16.png | validation | 491 |

These are approximate assistant annotations, not independent ground truth.
The previously inspected test set remains unsuitable as an untouched test claim.
Verified source hashes, all100 split assignments, four binary masks and preservation
of every existing V2 masked pixel. Proposals only add regions; V2 remains immutable.

Preview: `outputs/glare_policy_review/proposals.jpg`; source sheets and enlarged
glasses review are in that same local folder. User boundary/scope clarification
requested for dark scene reflection and blue glare versus bright white glare.
All four proposals remain pending; `training_enabled=false`. Once resolved, retain
the decision in provenance, finalize consistent versioned labels and re-evaluate
baselines under V3. No model has been trained using these proposals.

## Finalized after user decision

User accepted all four marked reflection types after the preview. The pending
status above is historical. Finalized `dataset/detector_glare_review_v3/manifest.json`
SHA256 `e36ce5cf04c858d61885c2a0187d0182099ada9852eb03d21d645f5bdbb3ea18`.
Standard manifest validation passed (hashes, masks, grouping and split checks).
V3 counts: train43 covered/25 uncovered, validation15/10, test4/3. The images and
split assignments are unchanged; two formerly uncovered records now contain glare
regions. Queue finalized and labels enabled, with approximate-annotation caveats.

Original completion epoch2 detector reevaluated on validation only: IoU0.060648,
missed93.160%, visible FP1.8086%, empty6/15, negativeFP1/10. The single validation
glare case is missed completely; one case cannot establish glare generalization.
Mannequin-excluded metrics and glare/no-added-glare strata are recorded in
`outputs/glare_policy_review/v3_baseline.json`. Test images were visually reviewed
for annotation consistency but not scored in this evaluation. Synthetic baseline
and selection rules are unchanged. V2 results remain archived under V2 labels.
