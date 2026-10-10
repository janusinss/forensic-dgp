# Native CCTV review contract: current-state audit, 9 October 2026

The existing source-specific selections, input-only reviews and encoded assets
remain intact. A new metadata audit verifies 137 encoded asset hashes and the
four retained raw source files without decoding an image, running a model or
using the VM. This reconfirms the evaluation boundary before any new training
design; it does not qualify restoration or complete the full goal.

## Frozen cohorts and resolution

| Cohort | Cases | Source labels | Native shorter-side range |
| --- | ---: | ---: | ---: |
| QMUL original development | 24 | 24 | 7–61 pixels |
| QMUL reserved evaluation | 32 | 32 | 7–93 pixels |
| QMUL larger-crop extension, excluded by input review | 3 | 3 | 64–96 pixels |
| ChokePoint development | 24 | 12 | 93–189 pixels |
| ChokePoint reserved metadata | 26 | 13 | 104–186 pixels |

There are 51 development/diagnostic cases and 58 reserved cases. The label counts
are source-specific, not proven globally distinct people. ChokePoint uses two
temporal positions per label; 26 cases represent 13 labels, not 26 independent
subjects. No within-source development/reserved label overlaps are found.
The three QMUL extension labels are also disjoint from the original 56 labels.

The QMUL reserved files are encoded source copies. ChokePoint's reserved entries
are frame/eye/crop metadata only: no reserved crop, prepared input or mask file
was created by its frozen selection. The new audit hashes encoded files and
reads metadata; it creates or views no reserved pixels. Past exposure statements
are retained as versioned receipts, not a claim that an exhaustive external
history of every possible exposure is available.

Native dimensions describe the observation. The 256×256 model canvas does not
make a seven-pixel face a 256-pixel observation. ChokePoint crops are derived
from original 800×600 frames, not the publisher's normalized 96×96 face archive.
The publisher confirms those formats and the frontal-camera sequence.
[Official ChokePoint page](https://arma.sourceforge.net/chokepoint/),
[publisher README](https://zenodo.org/records/815657/files/README.txt).

## Provenance and terms rechecked

The QMUL publisher continues to make the release available for research while
retaining the original owners' copyright. This is not a new commercial-use or
unrestricted redistribution permission.
[Official QMUL release and licence](https://qmul-survface.github.io/).

The primary QMUL paper lists component collections from China and Japan among
other locations. This supports its role as a mixed-source acquisition under the
Asian-source preference. It does not establish the capture country of any frozen
crop or any person's ethnicity; the local per-crop country fields remain null.
[Primary paper, Table 4](https://arxiv.org/html/1804.09691v6).

ChokePoint's notice permits noncommercial research and requires acknowledgment,
retention of the original notice and prominent identification of modifications.
The previously archived licence and derivative notices remain hash-bound with
the crops. Australian sponsorship is not proof of the sequence's capture site
or participants' ethnicity. No country or local-performance inference is made.
[Publisher licence](https://arma.sourceforge.net/chokepoint/).

The release versions and discrepancies already recorded in CCTV_BENCHMARK_STATUS.md
remain unchanged. Original recognition protocols are not relabelled as our custom
restoration protocol. No new dataset is acquired or institutional request sent.

## Input-only criteria and overlap limits

The current input reviews match all frozen case IDs and their subset hashes.
They were recorded before model outputs. The three larger QMUL crops remain
excluded for unsupported pose/framing or insufficient information; they are not
replaced with favorable outputs. Original low-resolution failures stay recorded.
The 24 ChokePoint development cases have usable rough structure according to
the pre-output review, which is not proof that any restoration improves them.

Selection manifests retain their creation-time `input_review_pending` flags.
Completed separately bound review receipts establish the later review status.
The audit joins those records without rewriting the frozen selection manifests.

The frozen criteria require one frontal or mildly turned crop with usable visible
eye/nose/mouth/outline relationships. Preserve visible appearance, clear glasses
and ordinary hair. Request a clearer or less-covered crop when structure is
insufficient. Resolution bins are diagnostic rather than automatic usability
labels. Keep every selected case and refusal in the denominator for its stated
scope; do not relabel an input because a particular model performs poorly.

Cross-source and historical/pretrained training-person overlap remains unknown.
Namespaced release labels and matching encoded bytes cannot establish complete
subject independence. V42 contains no frozen native case IDs. Its encoded inputs
also match none of the 27 recorded prepared native inputs checked here: the 24
ChokePoint development crops and three excluded extension crops. This is a direct
asset-exclusion check, not a person-overlap or re-encoding proof. The original
QMUL24 prepared hashes are outside that particular hash comparison.

## Conditions for later model and final review

Native CCTV has no aligned clean restoration target. Temporal ChokePoint views
are unpaired observations, even when one appears clearer. Do not calculate paired
PSNR/SSIM from these views or turn native crops into invented clean supervision.
Photographic synthetic-pair scores stay in their separate TRAIN/DEV evidence.

The next qualified candidate must be compared against resize, retained original
Phase3, current identity-v2 DGP and declared pretrained restoration baselines on
the same prepared inputs. Separate raw float output, PNG delivery and any display
processing. Report each source and insufficient-information case separately;
existing sharper pretrained estimates do not become our trained DGP contribution.

Freeze the candidate and processing policy before reserved review. Apply the
existing input-only criteria before opening model outputs. Use an independent
reviewer and keep source-label/temporal grouping explicit. Final results must not
be fed back into architecture, threshold or checkpoint tuning. Cross-history
overlap uncertainty remains a stated limitation of that review.

These native restoration cohorts do not establish final seven-family completion
coverage. Masks, sunglasses, strong glare, hands, obstructing hair, scarves and
objects still need their separate automatic/assisted qualification, mask review,
visible-region preservation and download-flow checks. No completion or app change
occurs here. V42 remains failed; the architecture question is pending. All five
milestones, useful development results, independent final review and bundled
inline Playwright remain required. The full goal stays active/incomplete.

Evidence: `outputs/cctv_dgp_native_contract_audit_20261009_v1/audit.json` and the
original frozen selections/reviews under `cctv_native_development_v2`,
`cctv_native_structure_extension_v1` and `cctv_chokepoint_native_development_v1`.
