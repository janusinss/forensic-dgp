# Bounded training-source search — 29 September 2026

Reviewed 200 additional original-training sources, 100 per dataset, excluding the
previous 400-source pool. Seed 20260929; sorted paths shuffled separately per
source, accepting the first 100 passing content checks. No model scores guided
selection. Two perceptual-match candidates were excluded and recorded.

Screening references include original validation, completion benchmark sources,
all V3 source/crop images and the prior 400-source pool: 4,600 reference paths.
Exact byte hashes, EXIF-aware decoded RGB hashes and 63-bit DCT distance <=6 were
checked, including earlier accepted candidates. This remains a heuristic whole-
image screen, not identity-disjoint certification. No held-out data was relabeled.

All five 40-image sheets were reviewed; 14 eyewear candidates were inspected at
enlarged resolution. Three localized-reflection candidates emerged:

- Search index 62: bright lens reflections near the upper eye regions.
- Search index 127: small blue/white reflections over partly visible eyes.
- Search index 141: small bright reflection in the image-left lens.

Five detailed cases remain ambiguous (1, 5, 10, 122, 196), six are proposed clear
controls (147, 165, 177, 185, 188, 195), and five other thumbnails show dark/mirrored
eyewear. The remaining 181 were not shortlisted; that is not a clean-label decision.
No pixel masks were created and none of these cases is enabled for training.

Next: prepare native-resolution boundary proposals for the three localized
reflections and inspect overlays. Preserve visible eye detail rather than filling
entire lenses. Annotation review is required before a future versioned dataset;
three new examples alone do not establish adequate coverage or generalization.
No additional random search loop or VM training is justified until these proposals
are assessed. Existing synthetic safeguards and application baseline remain fixed.

Local artifacts: `C:\xampp\htdocs\YEAR 4\Testing\outputs\glare_source_search_v1\`
(`manifest.json`, `review.json`, five sheets, two detail pages).
No VM data changed. A future packaged mirror would be
`~/forensic-dgp/outputs/glare_source_search_v1/`.

## Native-resolution boundary review

Three narrow proposals were rasterized in native source coordinates and checked:
index 62 has 52 pixels, 127 has 65, and 141 has 25. Source hashes, binary masks,
dimensions and pixel counts pass; the three-row outline preview was inspected.
Files are under `boundary_proposals/` (`proposals.json`, `verification.json`,
`review.png`, individual masks and coordinate inspection panels).

Closer inspection weakens the earlier candidate interpretation: these tiny bright
spots overlap pupils/frame regions and may be ordinary corneal catchlights or
frame reflections. The available pixels do not establish that strong lens glare
hides facial detail. All three therefore remain `unapproved_ambiguous` with
`training_enabled=false`. No reviewed dataset manifest was changed. This is a
correction of source triage confidence, not an expansion of the occlusion policy.

A user clarification is pending on whether these tiny ambiguous highlights belong
in the thesis scope. Recommendation: keep them excluded under the existing
strong-glare scope and seek clearer real reflections. Approval of scope alone
would not certify pixel-boundary accuracy or establish sufficient training data.
