# CCTV DGP identity pilot return: 4 October 2026

Windows project: `C:\xampp\htdocs\YEAR 4\Testing\`.
Training VM: `~/forensic-dgp/cctv_dgp_vm_bundle/` on the existing NVIDIA L4.
The full DGP-first Goal is active. This report records completed VM training and
independent local audit and completed native development review. The candidate
has not demonstrated useful native CCTV restoration and is not an application
default.

Later milestone, 4 October 2026: the subsequent V3 return is audited; all four
trained epochs failed selection and both best files retain this V2 start.
Results and five-grid review: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_PERCEPTUAL_RESULTS.md`
(intended VM `~/forensic-dgp/CCTV_DGP_PERCEPTUAL_RESULTS.md`). The next prepared step
is the VM-only zero-update objective diagnostic in
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_OBJECTIVE_DIAGNOSTIC_V4.md`
(VM `~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_OBJECTIVE_DIAGNOSTIC_V4.md` after
overlay extraction). It has not run and does not export a trained checkpoint.

## Verified return

The returned `cctv-dgp-normfix-v2-results.tar.gz` is 348,737,530 bytes. Its SHA256
is `82a7fee644909757d81418aae229bc3c7d6f61ea11447f5d26785ffa3d5cb527`.
The transfer checksum, safe archive inventory, immutable parent protocol,
correction source manifest and exact returned correction sources are verified.

Local extracted root:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_normfix_return_v2\`.
Canonical result beneath that root: `outputs\cctv_dgp_pilot\`.
Corresponding VM result:
`~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_pilot/`.

The original CUDA gradient preflight passed with zero optimizer updates; the
additional six-image evaluation preserved the exact starting student state.
The completed pilot used PyTorch2.9.1+cu129 on an NVIDIA L4, 452 optimizer updates,
and 311.77 seconds against a 5,400-second cap. Both arms started from the same
retained Phase3 tensors. All four epoch checkpoints changed 293 parameter tensors;
normalization buffers and frozen teachers remain unchanged. The original failed
zero-update attempt is preserved on the VM; its training recipe was not repeated
without the versioned InstanceNorm runtime correction.

The independent local audit reconstructs 2,750 prediction PNG metrics, 50 raw
float preview/composition records, 3,300 embedding cosines and all 452 update
trace entries. Fifty-two ONNX recognizer preview forwards validate stored
embeddings against the pinned recognizer. No local restoration, backward or
optimizer update occurred in this return audit. The received VM audit receipt
remains intact.

Local receipt:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_normfix_return_v2\local_independent_audit.json`.
SHA256: `8eb3ec00c31e201689a22d4682c6fa21345653990529bf9d0986133a50eae685`.
Returned results SHA256:
`61c537cf451542889d52d5b4ae820548e04970c9c0435c5ef2b760a8b8882725`.
The local receipt has no VM counterpart until explicitly transferred; do not
overwrite the separately received VM receipt.

## Paired photograph regression

These are 110 inherited validation references, with one clear and four fixed
camera-like degradation cases per reference: 550 cases total, 440 degraded and
110 clear. They are synthetic paired photographic evidence, not native CCTV or
Zamboanga footage. The Asian dataset name describes its provenance, not a claim
about an individual pictured person's ethnicity or local population coverage.
Targets are processed photographs, frequently below256 captured pixels.

The numbers follow the actual exported RGB PNG on observed support. The original
protocol's aggregate PSNR derives from aggregate mean case MSE; it is not a mean
of individual image PSNRs. ArcFace is fixed-grid embedding similarity, not verified
identity accuracy. Keep source/profile reports separate.

| Group | Phase3 PSNR | Identity epoch2 PSNR | Phase3 SSIM | Epoch2 SSIM | Phase3 ArcFace | Epoch2 ArcFace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All degraded (440) | 13.2317 | 16.2167 | 0.5275 | 0.6559 | 0.2766 | 0.3269 |
| Clear controls (110) | 25.3528 | 32.3735 | 0.8238 | 0.9453 | 0.8335 | 0.9544 |
| Asian-source degraded (204) | 13.0153 | 15.9959 | 0.5773 | 0.7043 | 0.3629 | 0.4164 |
| FFHQ-source degraded (236) | 13.4279 | 16.4171 | 0.4845 | 0.6141 | 0.2021 | 0.2494 |

Aggregate degraded MAE decreased from0.159733 to0.108000; clear MAE decreased
from0.044658 to0.018718. Numerical improvement does not establish visible detail
recovery. The fixed ten-row identity-epoch2 preview retains substantial softness
and low-light/compound cases remain difficult. Inspect native structure and
counterexamples before application selection.

## Selection and next decision

The camera-only arm's epoch2 aggregate degraded PSNR is17.2684, but its ArcFace
similarity fell from0.2766 to0.2558. It did not meet the unchanged guard and its
`best.pth` is an epoch0 baseline fallback. Do not describe that file as an improved
trained candidate.

The identity-supervised arm qualifies at epoch2. Eligible candidate:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_normfix_return_v2\outputs\cctv_dgp_pilot\camera_identity\best.pth`
↔ `~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_pilot/camera_identity/best.pth`.
Its bytes equal `camera_identity/epoch_2.pth`; SHA256:
`b30aeabecc60dff2fbd289575ce8691be9e670c653cacb6721bc154b618c3916`.
It qualifies for development review, not automatic production promotion.

The original native development plan is already frozen at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_native_pilot_review_v1\frozen_plan.json`.
Its optional VM report counterpart is
`~/forensic-dgp/outputs/cctv_dgp_native_pilot_review_v1/frozen_plan.json` after
explicit transfer. The comparison re-audits the return, runs only eligible trained
candidates on24 development cases, and compares cached Phase3 and CodeFormer on
identical inputs/composition. Six input-selected coarse frontal/mild cases form
the core group; all24 diagnostics remain. It permits at most48 forwards and480
seconds after loading with zero optimizer/backward calls.

The native run is now complete:24 candidate forwards in7.87 seconds, with exact
student state preservation and zero gradients/optimizer updates. Result SHA256:
`a2221cfdeb546abe494af0793519d6b9a4e1be3c71f47876a5721c02de9a32fc`.
Its independent composition audit rebuilt120 PNG stages and verified24 raw
outputs. It has no model forwards or optimizer updates of its own.

The assistant's development visual review inspected the ten-row preview and all
four full gallery pages. Its separate ledger is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_native_pilot_visual_review_v1\review.json`
(optional VM counterpart `~/forensic-dgp/outputs/cctv_dgp_native_pilot_visual_review_v1/review.json`
after explicit transfer). Ledger SHA256:
`c9a76e9dc7d4e180f278baeb3ccafbe0001c79da1eddfe34117f6fa2412329f7`.
The original native result and composition audit are preserved; their then-pending
visual-review fields are superseded by this separate dated ledger, not rewritten.

The six input-selected core faces remain soft with color shifts; block smoothing
does not establish useful structural restoration. Sharper CodeFormer outputs
sometimes add generic features or expressions and do not qualify as our main
restorer. Seven input-selected insufficient-information cases require a clearer
crop. Strong profiles and the covered case remain diagnostics. No aligned native
clean reference exists, so native PSNR, SSIM and identity accuracy are not reported.
The32 reserved native crops remain untouched. Native usefulness is not demonstrated;
the paired-eligible checkpoint is retained for research without default promotion.

## Historical perceptual V3 preparation — superseded by the audited results above

The frozen VGG19 loss currently samples after ReLU. A separate10-case frozen
diagnostic found signed preactivation features yield roughly7 times the unscaled
loss magnitude. It used42 VGG forwards in12.92 seconds and confirmed exact control
feature values, unchanged teacher state and zero backward/optimizer calls. This
is feature/scale evidence, not a proven cause of the native failure.

A separate training-only calibration used64 original training references,
32/source, with zero validation/native cases. Eight frozen DGP batch forwards and
32 VGG batch forwards took83.63 seconds. Starting/teacher states remained unchanged;
there were zero backward/optimizer calls. Its report at
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_perceptual_calibration_v3\scales.json`
has SHA256 `89016bd61e6599f4a8f6debb41c710f518ead97d87b671a4be67026cab0dfb5a`.
Mean loss scales do not imply equal gradient magnitudes or trained improvement.

The resulting changed finite VM comparison starts both arms at the audited V2
identity epoch2, keeps ArcFace identity supervision, and compares continued
postactivation versus training-only calibrated preactivation features. Signed
taps are cloned before subsequent in-place activation. Exact commands, source
rationale, unchanged safeguards and452-update/90-minute budget are in
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_PERCEPTUAL_V3.md`, bundled at
`~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_PERCEPTUAL_V3.md` after verified overlay
transfer. Actual V3 CUDA preflight, training, returned audit and native review
remain pending; preparation does not establish an improved model.

The preparation is now independently verified:27 relevant forward/file/guard tests
passed, nine new Python files parse for Python3.10, Bash launcher syntax passes,
the source derivation is exact, and the real V3 `--verify` CLI reports1,012 prepared
references and zero optimizer updates. The additive archive has16 files and is
724,844 bytes, SHA256
`925ef82d27ba1e3bfe010989c79d6e1e28572e8a91c47e83c5da7e09da8d469e`.
Receipt: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_perceptual_v3_checks\package_audit.json`
(optional VM copy `~/forensic-dgp/outputs/cctv_dgp_perceptual_v3_checks/package_audit.json`
after explicit transfer). It reuses VM data/weights/runtime and preserves all
historical assets. No local backward or optimizer step was performed.

The application's current combined route still uses CodeFormer for visible
restoration. DGP256 integration, insufficient-information qualification,
supporting covering-family outputs, full local browser verification and independent
final assessment remain required by the Goal. Training completion is not completion
of that Goal.
