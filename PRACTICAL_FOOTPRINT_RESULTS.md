# Revised removal footprints — 2 October 2026

Including complete opaque glasses changes the practical result substantially.
CodeFormer now produces useful estimated eyes in the dark-sunglasses and mirrored
eyewear examples, and reduces the strong white reflection while retaining the
clear frame. The mask with clear glasses still has a pale nose patch and a poor
transition. These six developmental cases do not establish full-family readiness.

| Location | Windows local | Linux VM after transfer |
| --- | --- | --- |
| Repository | `C:\xampp\htdocs\YEAR 4\Testing\` | `~/forensic-dgp/` |
| Report | `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_FOOTPRINT_RESULTS.md` | `~/forensic-dgp/PRACTICAL_FOOTPRINT_RESULTS.md` |
| Frozen proposals | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_footprints_v3\` | `~/forensic-dgp/outputs/practical_footprints_v3/` |
| Comparison | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_footprint_outputs_v3\` | `~/forensic-dgp/outputs/practical_footprint_outputs_v3/` |

No transfer, training, commit or push occurred. All actual training remains on
the user's L4 VM. These outputs are local CPU inference.

## Frozen scope and verification

Four covering footprints and two empty controls were source-reviewed before
generation. The difficult turned mask/clear-glasses face is a diagnostic beyond
the initial frontal/mild-turn scope. The other cases are dark sunglasses, white
lens glare, mirrored glasses, an uncovered control and an ordinary clear-glasses
control. All are previously inspected detector-training sources, not holdout data.

V3 includes full opaque rims/bridges and the facial mask strap segment. Opaque
proposals add another 3-pixel square margin to the earlier proposal, reaching up
to 6 pixels relative to old labels. White-glare lens interiors receive no extra
dilation. A source-painted wire region is excluded from the clear-glasses mask
case. These are approximate operator removal proposals, not new training labels
or expert ground truth. Original labels, splits, prior masks and results remain
unchanged. Earlier failed outputs motivated case selection; the new footprints
were painted from the original visible sources.

Twelve requests comprise eight nonempty forwards and four empty bypasses across
CodeFormer and AOT-GAN; 12 previous outputs are reused for comparison. CPU elapsed
36.2 seconds, zero failures, detector forwards or optimizer updates. Restoration
is off and the existing crops are used without another alignment operation.

The independent PIL/NumPy auditor verifies all 12 new and 12 reused outputs,
72 historical gallery assets, source/code/checkpoint hashes, proposal deltas and
recorded generator state digests. Zero pixels changed outside the respective
active masks; zero changed in the source-painted protected eyewear region. This
is byte preservation, not hidden-face accuracy. No hidden ground truth or hole
MAE is available.

| Evidence | SHA256 |
| --- | --- |
| `outputs/practical_footprints_v3/frozen_protocol.json` | `4a256b9a00c25f68414010877a8cf96c1e3a9a283842360ed64978a676a5c5d3` |
| `outputs/practical_footprint_outputs_v3/results.json` | `1c79c0627697a353e57f47adb8943e1e5fda8af1ed51ec00f53aa01dbf3ef17c` |
| `outputs/practical_footprint_outputs_v3/independent_verification.json` | `638b13e094082abdf4c8e088c6fb0c33010c843aad62fe723680e44ddc05b40f` |

## Assistant visual review

All rows were viewed in the two three-row preview crops. This is developmental
assistant triage, not independent expert assessment or deployment accuracy.

| Case | CodeFormer V3 | AOT-GAN V3 |
| --- | --- | --- |
| Mask with clear glasses | Still needs a fix: pale nose region and uneven facial join | Still needs a fix: blur and pale nose region |
| Dark sunglasses | Useful estimated eyes; opaque glasses removed, texture/appearance still approximate | Useful estimated eyes; more texture variation around nose |
| White lens glare | Useful reduction of opaque reflection and estimated visible eyes; clear frame retained | Still needs a fix: bright lens fill remains |
| Mirrored glasses | Useful eyes; full circular rim removed | Useful eyes; full circular rim removed |
| Uncovered control | Exact preservation | Exact preservation |
| Clear-glasses control | Exact preservation | Exact preservation |

CodeFormer remains the development baseline. This comparison supports complete
covering footprints and optional manual adjustment, not promotion of a failed
automatic detector or a claim that the adapter input alone caused the earlier
white-glare failure. The historical reflective detector retention failure stands.

## Next step

Freeze and compare the prepared degraded crops with completion alone, the legacy
restoration-before-completion stage, and restoration of only visible pixels after
completion. Use only known visible pixels for reference metrics; review generated
anatomy separately. Broad standalone-hand, obstructing-hair, scarf and other-object
coverage, visibility rejection and the existing main-app workflow remain required.
No unchanged VM training recipe is justified by this result.
