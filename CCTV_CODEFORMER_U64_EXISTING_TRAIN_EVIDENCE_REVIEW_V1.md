# Existing u64 TRAIN evidence: targets, corruption and loss scope

23 October 2026. Read-only review after the audited PNG-checked-step quality
stop. The required next design choice is still pending. This review changes no
dataset, loss, model, gate or training plan; original u64 remains retained and
U96 stays the experimental app incumbent.

## Data reconstruction result

All **24 retained photographic targets** exactly reproduce direct LANCZOS
resizing of their retained 1024×1024 RGB sources to 256×256. Source and target
pixel hashes and the retained official source MD5 metadata match. All **96
paired TRAIN inputs** exactly reconstruct from those targets under the declared
operators. All 24 clear controls retain exact target-file bytes. Every localized
blur leaves every pixel outside its central 128×128 pasted region unchanged.
Independent separable Gaussian arithmetic and the declared SciPy operator
produce the same final RGB8 pixels for all 24 photographs.

There is no demonstrated corrupted file, clean-target resizing mismatch,
misapplied declared blur operator or accidental second blur in these inputs.
This does not establish that the photographic targets capture every subtle
facial detail or that the synthetic inputs represent real CCTV degradations.

## What this cohort covers

The cohort contains **16 optimization photographs and eight adaptive TRAIN
monitor photographs**, each with four correlated roles. These 96 files are not
96 independent people or an independent test set. The photographic split does
not establish person-identity separation or nonoverlap with CodeFormer
pretraining. Existing reserved FINAL identities are outside this review.

| Role | Actual corruption | Important limit |
| --- | --- | --- |
| Clear | Exact clean target bytes | A control for preserving already visible appearance |
| Global blur | One Gaussian, sigma 1.2 at 256 pixels, radius 4, reflect boundary | No motion, optical or camera-calibrated blur variation |
| Local blur | The same Gaussian pasted only into `[64:192,64:192]` | Fixed square, hard boundary; no soft-edge transition |
| Low light | Linear-sRGB exposure multiplied by 0.35 | Brightness only; no added sensor noise, blur or compression |

The hard-paste local corruption is **not** the user-reviewed Protect/Apply
surroundings preparation workflow. In that workflow, the already blurred
protected region stays unchanged and only permitted surroundings receive blur.
These training inputs instead simulate corruption inside a fixed central
square. The two roles must not be conflated.

Across the 24 photographs, whole-image input-to-clean MAE averages 0.00397413
for local blur, 0.01527638 for global blur and 0.17289424 for darkening. These
values describe input corruption under known synthetic operators; they are
neither restoration-model results nor evidence of native CCTV performance.
Different changed areas and brightness shifts prevent interpreting them as
directly comparable difficulty rankings. They also do not prove why a specific
optimization trajectory failed.

## Loss and evaluation measure different things

Independent saved-array arithmetic reproduces **96 retained objectives**:
32 from the actual eight-Adam-update route and 64 from its unchanged-initializer
diagnostic. The largest difference from a saved CUDA FP32 objective is
8.60e-8, within the existing 5e-7 scalar readback bound. That bound checks audit
arithmetic; it adds no quality tolerance. No framework or autograd was replayed.

The local loss balances inside and outside means equally, giving each inside
pixel three times the outside weight, while the principal local evaluation
uses the center only. The clear loss uses twice a SmoothL1 composite; the
photographic gate uses ordinary absolute error. Low-light loss includes an
extra clean-photograph block-luminance anchor, while the inherited illumination
guard measures movement from frozen U96. A reduction in the aggregated RAW
training loss therefore does not require every separate RAW/PNG endpoint or
legacy guard to improve.

Existing original-initializer gradient measurements reinforce that distinction:
same-role objective/MAE cosine was positive in all 64 rolecases, but
objective/adjacent-error cosine was negative for local 7/16, global 8/16,
clear 12/16 and low-light 8/16. The average low-light share of the four
objective-gradient norms was 62.2285%. These are local measured tradeoffs,
not a complete replay of Adam history or proof of a uniquely correct new loss.

## Interpretation before another training decision

Correct targets and exact operator reconstruction remove specific data-pipeline
errors from the demonstrated causes. The remaining evidence supports reviewing
coverage, role weighting, objective-to-evaluation alignment and the trainable
path without assuming which one uniquely caused the failed successor.

One blur strength and one exposure level do not establish coverage of the
strong filters, varying resolution, compression or real capture conditions in
the user's stress images and native CCTV development crops. Training more
epochs on this cohort would repeat that restricted coverage. No new corruption
family, loss weighting, architecture, optimizer or epoch schedule is selected
by this review.

Recorded provenance, final input admissions and exact loss-to-gate arithmetic
are retained in separate companion receipts. Historical manifests with pending
flags must be interpreted through their later explicit admissions, not
silently rewritten. Stored source terms are provenance snapshots, not fresh
legal certification or evidence of person/pretraining independence.

Primary saved evidence is
`outputs/codeformer_u64_train_evidence_review_v1/inputs_review_v1.json`.
Exact loss arithmetic and scope are recorded in
`outputs/codeformer_u64_train_evidence_review_v1/loss_review_v1.json` and its
companion `loss_review_v1.md`; provenance is recorded separately in
`outputs/codeformer_u64_train_evidence_review_v1/provenance_review_v1.json`.
The parent protocol remains SHA256
`4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419`.
All audits here use existing files and NumPy/PIL/SciPy arithmetic only; no
neural model, gradient, optimizer, parameter application, VM or browser is run.
Native/wild evidence remains unpaired, broader DEV failures stay binding,
FINAL remains unopened/execution-unfrozen, and independent reviewers remain
pending. Preparation and deferred completion scope remain unchanged.

The full goal remains active and incomplete. Under `AGENTS.md`, further
model/training changes wait for the diagnostic choice between training
examples/loss design and reconstruction architecture. Existing authorization
for evidence-justified development remains in force; all actual future
training still requires the manual finite L4 workflow.
