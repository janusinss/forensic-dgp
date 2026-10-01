# Frozen-detector complementarity diagnostic — 2 October 2026

This is a fixed training-only feasibility diagnostic, not a trained router or
application candidate. Actual fitting remains VM-only. The independent supported
data audit and earlier failed retention recipes motivate testing whether keeping
two existing detectors frozen leaves useful complementary predictions.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`; latest execution root:
`~/forensic-dgp/coverage_vm_bundle/`.
This CPU diagnostic has no new VM counterpart or upload.

## Evidence, inference and limits

[Progressive Neural Networks](https://arxiv.org/abs/1606.04671) investigates
architecture expansion to preserve prior knowledge. The
[MICCAI 2024 Low-Rank Mixture-of-Experts paper](https://papers.miccai.org/miccai-2024/paper/1160_paper.pdf)
uses dataset-specific experts with other network parts fixed in continual medical
segmentation. These primary sources motivate parameter isolation; they do not
validate face-mask/glare performance here. Our diagnostic does not reproduce
either architecture: no lateral features, low-rank experts or learned router.
Frozen weights preserve each branch, but inference routing can still regress.

The original Gated U-Net detector remains frozen at parent SHA256
`c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93`.
The face-specific detector is fixed to the final reflective epoch42 budget,
SHA256 `a51f20e8fa12cee7dec574195debc5ada9b8cfeeb289b12bad11a6d39a9acf25`.
No validation-best checkpoint search. The latter is independently audited but
ineligible on original safeguards; using it as a diagnostic branch is not promotion.

| Input pool | Cases | Covered | Clear | Support |
| --- | ---: | ---: | ---: | --- |
| Supported real training | 83 | 51 | 32 | Explicit valid map; originals full |
| Current safe replay membership | 610 | 360 | 250 | Existing square/full definition |
| Reviewed reflection fixtures | 280 | 224 | 56 | Existing explicit valid maps |

Replay membership is the union of unique case IDs actually used by the already
executed reflection core schedule, rather than restoring all638 earlier cached
IDs. Its existing eight-source quarantine remains unchanged. Case count changes
are training-membership provenance, not relaxed held-out safeguards. Inputs and
targets stay byte-bound to the existing caches; there is no resampling or augmentation.
Source paths remain in the original Phase4 training partition. Exact shared-RGB
contradictory labels are reported on common supervised pixels, not silently merged.

There are973 unique diagnostic case IDs. Both frozen models use CPU, float32 input
in `[0,1]`, 256px images and logit-sigmoid threshold0.5. Reuse353 final-reflective
training masks (73 real/280 fixtures) only after verifying their returned hashes
and complete prior CPU reproduction. Perform973 parent and620 final-reflective
image predictions:1,593 new forwards. Check finite outputs, exact pre/post tensor
state fingerprints, eval mode and absent parameter gradients. No optimizer,
backpropagation, generator completion, held-out model forward or fitting.

## Fixed systems and diagnostic bounds

| System group | Definition | Available at inference? |
| --- | --- | --- |
| Parent / candidate | Each frozen head independently | Yes; not newly selected |
| Union / intersection | Boolean OR / AND at fixed thresholds | Yes; no target/source input |
| Pool oracle | Candidate on real/reflection, parent on replay | No; uses known pool membership |
| Dominance oracle | Candidate only when TP does not fall and FP does not rise, with strict improvement; ties parent | No; uses ground-truth mask/support |
| Pixel oracle | Union on true positives; intersection on visible pixels | No; uses ground-truth mask |

All scores use the same valid support. Ignored predictions are recorded separately;
unknown regions never become negative labels. Common false negatives remain missed
even under the pixel oracle; shared false positives remain errors. Oracles bound
what the existing binary heads might offer. They are not automatic inference,
quality guarantees, or permission to train on held-out labels.

Report counts and metrics per case, pool, covering/degradation stratum and source
pool; distinguish original73 and new10 real examples. Measure real lens-only
recovery over the two existing strong-reflection training labels plus the four new
lens annotations. Their fixed total is21,853 labelled output pixels. Keep all
remaining covering boundaries and the14 pending sources unresolved.

Before any predictions, freeze10 preview rows: the four new covering sources;
the two existing strong-reflection training cases; clear sources207/208; the first
registered degraded irregular replay case and first registered degraded clear
replay case. Show input, target, each head, union and dominance oracle. Mark oracle
columns as target-informed. This is a diagnostic selection, not representative
sampling or new holdout evidence.

## Decision boundaries and execution

The two fixed compositions receive training checks only: strict real IoU gain
against the parent; real visible FP, empty masks and clear false-mask cases no
higher; all five replay metrics no worse than the parent on this training pool;
strict labelled lens recovery gain against the final-reflective branch. Passing
would justify further independent evaluation, not application promotion. Failing
does not license thresholds, weights, checkpoints or gate changes.

First inspect whether the label-informed bounds offer useful complementary
coverage and where both heads still fail. A learned input-only router would
require a separate predeclared VM protocol and unchanged development gates.
No task ID, filename, source membership or target-derived prompt may select the
head in a deployable single-image system. Do not commission fitting from oracle
success alone; missing shared glare coverage requires more than a selector.

Runner: `C:\xampp\htdocs\YEAR 4\Testing\scripts\run_frozen_complementarity.py`.
Prepared specification:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\frozen_complementarity_protocol_v1\protocol.json`.
Planned evidence:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\frozen_complementarity_v1\`.
Both prepare and execution refuse existing output. Preserve partial output if a
process fails; do not restart solely because observation expires. Use only the
existing pinned face-extraction dependencies; do not install or download weights.

After execution: independently recount saved masks and every composition/bound,
check membership and hashes, inspect the ten-row preview, update handoff, and
choose the next justified action. Original425-case development selection gates,
test membership, generator/application and Phase3 baseline remain unchanged.
The full goal still requires an eligible candidate and reviewed end-to-end gain;
hidden facial anatomy remains an estimate.
