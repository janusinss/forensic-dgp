# Reflection coverage: verified return and decision — 2 October 2026

The matched reflection pilot is independently audited and reproduced. Neither
arm satisfies the original synthetic safeguards; no `best_detector.pth` is
eligible. The reflective arm learns the introduced training patterns, but still
misses the real validation reflection. No generator, application or Track 1
restoration checkpoint is promoted. An identical training run is not justified.

## Transfer, execution and CPU verification

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`; execution root:
`~/forensic-dgp/coverage_vm_bundle/`.

The downloaded archive and VM-generated LF checksum now match. Size: 649,143,021
bytes. SHA256:
`a5714641943da67137a51866e4614b2b2841b2447145c816a51a39e7c52b032b`.
Windows archive: `C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection-coverage-results.tar.gz`.
VM archive: `/home/janusdominic0/forensic-dgp/coverage_vm_bundle/reflection-coverage-results.tar.gz`.
The checksum uses the same path with `.sha256` appended. Earlier preliminary
inspection/readiness artifacts retain their historical pending flags.

Safe extraction verifies 4,517 returned file hashes and all 179 sent input/
provenance members. All 504 logged updates, fixed exposures, supplemental pairs,
loss arithmetic, counters and original guard decisions reconstruct. All 4,315
binary masks independently recount. Source/final checkpoints and AdamW states
have finite, correctly shaped tensors, exact source/model bindings and unchanged
frozen BatchNorm/reference-head state. Each arm starts at model update 630 and
optimizer step 420; 252 new updates end at model update 882 and moment step 672.
The two forks are independent; neither resets its inherited optimizer.

CPU inference compares every saved mask with the audited checkpoint predictions.
Exactly one pixel differs, in reflective epoch 36 synthetic case 359; all other
original-domain masks and all 1,400 reflection-fixture masks match exactly.
The cause of that single CPU/VM difference was not independently isolated.
All CPU/VM selection decisions agree. Local model state stays unchanged; no
optimizer is constructed and no fitting occurs. The initial sandbox dependency
read failed before inference; the authorized rerun uses existing pinned modules
without reinstalling them.

The VM reports 132.19 seconds, NVIDIA L4, Torch 2.9.1+cu129 and 1,163,121,664 peak
allocated CUDA bytes. Remote original-parent invariance remains an executed-code/
log claim: its VM tensors were not returned for independent comparison.

## Candidates on unchanged validation sets

| Fork/global epoch | Real IoU | Human-only IoU | Mannequin IoU | Synthetic IoU | Selected |
| --- | ---: | ---: | ---: | ---: | --- |
| Source 30 | 0.81739 | 0.81997 | 0.79113 | 0.92039 | Ineligible source |
| Control 36 | 0.81679 | 0.81761 | 0.80840 | 0.92195 | No |
| Reflective 36 | 0.83172 | 0.83018 | 0.84772 | 0.92093 | No |
| Control 42 | 0.81207 | 0.81167 | 0.81625 | 0.93134 | No |
| Reflective 42 | 0.81462 | 0.81525 | 0.80817 | 0.93292 | No |

All four candidates pass the original real gate and fail synthetic retention.
The synthetic IoU requirement remains 0.9746899906463458. At epoch 42,
control/reflective missed fractions are 0.057166/0.053594 versus required at most
0.016476; visible false-positive fractions are 0.001815/0.002126 versus at most
0.001333. Clear synthetic errors are 1/80 and 0/80 respectively. Improved medical
mask coverage cannot override these safeguards. Source 30 already fails retention.

## Error partitions and visual review

On the 280 training fixtures, source/control 42/reflective 42 IoU is
0.18159/0.10150/0.80937. The reflective arm detects all 224 positive fixtures and
keeps all 56 clear fixtures empty. Four style IoUs are white patch 0.80221,
white streak 0.79994, scene reflection 0.79553 and blue glare 0.83962.
Clean/degraded fitting IoU is 0.89740/0.76243. These are exposed training sources.

Lens-only recovery excludes original mouth-mask pixels. The two real training
reflections contain 2,031 additional accepted pixels. Weighted recall for
source/control 36/reflective 36/control 42/reflective 42 is
59.48%/93.01%/99.41%/80.26%/67.41%. More fixture fitting does not imply improving
real reflection fit. All four candidates recover zero of 491 validation
reflection pixels. This one reused photo does not measure all reflection types.

Final reflective irregular synthetic IoU is 0.87617 versus parent 0.95988;
degraded IoU is 0.90780 versus parent 0.96005. Across all 400 cases, it loses
145,010 parent true-positive pixels and recovers 20,268 parent misses. It removes
27,389 parent false positives but adds 45,523 visible false pixels. Net coverage
loss is 124,742 pixels; net extra visible errors are 18,134. Correct clear-face
presence alone does not repair these segmentation errors. Both source pools lose
coverage; pool differences are not evidence of demographic causation.

The fixed ten real rows `[0,1,5,6,2,3,4,17,8,23]`, lens zoom, clean/degraded
fixture sheets and 20 predeclared synthetic views were inspected. Medical masks
are fuller than the original parent, clear rows stay empty, and the patterned
side-view boundary remains imperfect. The validation lens reflection is absent.
Procedural fitting improves, while degraded irregular boundaries lose coverage.
Synthetic sheets show one identity per pool; grouped numeric analysis includes
all 400 cases. No end-to-end completion improvement is established.

## Evidence and commands

Local extraction:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_reflection_coverage\outputs\reflection_coverage_vm\`.
VM source: `~/forensic-dgp/coverage_vm_bundle/outputs/reflection_coverage_vm/`.
Local audits:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection_coverage_validation\{results,reproduction,members}.json`.
Local diagnosis/review:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection_coverage_analysis_v1\{results,visual_review}.json`.
These are local inspection artifacts; no new VM upload/job is needed.

The tools refuse existing completed/partial outputs. These commands reproduce
the workflow in a fresh checkout with identical pinned inputs; do not rerun
them over completed evidence:

```powershell
Set-Location 'C:\xampp\htdocs\YEAR 4\Testing'
venv/Scripts/python.exe scripts/audit_reflection_coverage_results.py
venv/Scripts/python.exe scripts/reproduce_reflection_coverage_results.py
venv/Scripts/python.exe scripts/analyze_reflection_coverage_results.py
```

Five CPU-reproduction contracts passed during preparation. Four new analysis
counterexamples pass after observed failure before implementation: pixel-transition
conservation, clear-case errors, micro-count aggregation and binary shape checks.
These tests validate tooling, not model quality. The executed VM recipe, package,
targets, splits, thresholds and gates stay unchanged. No code was committed/pushed.

## Next bounded action before training

The subsequent predeclared input-only face-crop diagnostic is now complete and
rejected. All706 new masks independently recount. Final training lens recall
drops67.41%→50.12%, whole real/fixture IoU falls and visible FP increases; clear
controls remain empty. No held-out forwards or optimizer updates occurred.
See `FACE_CROP_RESULTS.md`. Stop this rule without a scale/confidence/threshold
or epoch sweep; do not tune against the missed validation photo.

Only two real training reflection photos exist. Additional real covering examples
need a separate training-only dataset version, native mask review and preserved
split/provenance. Viewed validation/test photos must not become training data.
Retention repair needs a distinct predeclared hypothesis; previous teacher replay
and projection runs already failed, so merely renaming them is not an intervention.

No new VM training recipe is ready from this return alone. Actual training stays
on the VM. Only an eligible detector advances to reviewed end-to-end completion.
External pretraining overlap, identity separation and generalization remain
unresolved. Hidden facial features are plausible estimates. The goal stays active
and unmet.

