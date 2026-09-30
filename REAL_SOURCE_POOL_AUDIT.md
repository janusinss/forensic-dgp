# Real-source coverage audit — 30 September 2026

No training or checkpoint promotion occurred. The previous six feature-head
candidates still fail original synthetic retention. This is dataset preparation,
not evidence that adding data will resolve that failure.

## Verified inventory

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM root: `~/forensic-dgp/`. Relative paths below resolve from either root;
the new audit artifacts currently exist locally only.

`scripts/audit_real_source_pool.py` screened all 1,510 JPG sources in
`dataset/real_occlusion_review/` against 4,200 unique reviewed source/crop,
original validation and benchmark source paths. It compared file SHA256,
EXIF-transposed RGB/dimension hashes and 63-bit DCT hashes (distance <=6).

| Finding | Count |
| --- | ---: |
| Sources with reference flags | 123 |
| Within-pool flagged pairs | 195 |
| Sources without either kind of flag | 1,100 |
| Existing queued source hashes reverified | 25 |

Flags are screening candidates, not confirmed duplicates. Counts overlap;
pairs are not counts of unique images. Whole-image similarity cannot establish
identity separation or detect all alternate crops. Nothing was deleted, assigned
to a split, or enabled for training. Original V3 manifest hash remains
`e36ce5cf04c858d61885c2a0187d0182099ada9852eb03d21d645f5bdbb3ea18`.

## Bounded visual triage

The first 24 unflagged sources in ascending SHA256 order were inspected in
two contact sheets, without model scores. Source hashes were checked again when
recording the review in `outputs/real_source_pool_audit/source_review.json`.

| Provisional category | Count |
| --- | ---: |
| Single-mask or mixed hand/mask candidates | 5 |
| Apparently clear controls | 4 |
| Digitally drawn mask appearance | 4 |
| Multi-face / authenticity / resolution / eyewear holds | 11 |

The four drawn-mask images must not be represented as real-world occlusion
examples without provenance establishing otherwise. Product imagery and possible
mannequins remain held; filenames do not establish real-person authenticity.

Native images additionally inspected:

- `masked (1655).jpg`: side-profile white respirator, usable mask contour.
- `masked (1257).jpg`: patterned fabric and dual-valve respirator.
- `masked (1439).jpg`: hand overlapping surgical mask and cheek.

These are candidate detector annotations, not uncovered reconstruction targets.
No paired photograph of the hidden facial region is available. Clothing outside
the face must not become a face-completion target simply because it touches a mask.

## Next bounded step

### Proposal follow-up

`scripts/prepare_real_expansion_proposals.py` now exports three source-derived
square crops, native-coordinate polygons, binary 256-pixel masks and a three-row
overlay. `outputs/real_expansion_proposals_v1/` preserves the initial proposals;
`outputs/real_expansion_proposals_v2/` contains one boundary correction pass.
Both remain disabled and unassigned. These output version names do not change
the reviewed dataset V3.

All three overlays were inspected. Profile lower-edge undercoverage improved,
but thin edge mismatch remains; patterned-mask strap/fabric boundaries and the
hand/cheek contour remain approximate. They are not accepted training labels.
Nearest DCT distances against the 100 V3 crops were 22, 14 and 20 (threshold 6).
The nearest images were visually inspected and show distinct scenes. This does
not establish identity separation, nor replace broader crop-to-source screening.
Source hashes, binary mask shapes and unchanged V3 manifest were verified in
`outputs/real_expansion_proposals_v2/verification.json`.

Next: finish native-resolution boundary review and expand the non-mask/eyewear
annotation batch before combining proposals into any versioned training set.
No additional model run is justified by three provisional labels alone.

Prepare source-grouped face crops and explicit pixel-mask proposals for the three
native-reviewed sources. Compare crops against existing reviewed/held-out images
before assigning training membership. Keep the unchanged validation and test
memberships, and keep ambiguous tiny glare proposals excluded. Reuse the existing
25-image non-mask/eyewear queue to broaden covering types rather than adding only
medical masks. Version accepted labels separately from V3; record source hashes,
crop coordinates, grouping, reviewer and boundary policy.

Only after reviewed data and leakage checks are complete should a bounded VM-only
training recipe be prepared. No new training command is warranted yet. Evaluate
against the unchanged real and synthetic gates, followed by completion previews;
the current generator/application baseline remains unchanged.

Evidence: `outputs/real_source_pool_audit/results.json`, `queue.json`,
`source_review.json`, `page_0.png`, `page_1.png`. Review recording can be rechecked
locally with `venv/Scripts/python.exe outputs/real_source_pool_audit/record_review.py`.
