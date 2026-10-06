# V18 returned results — 5 October 2026

**Training, transfer, independent local audit and development visual review are
complete. V18 demonstrates fitting capacity, but its unconditional output fails
the unchanged clear-image preservation guards. No checkpoint is selected or
adopted. Do not repeat the closed V18 training commands.**

The user returned the successful export after VM storage maintenance. The archive
is 481,723,919 bytes; its SHA256, sidecar, terminal receipt and extracted results
agree. All 21 frozen assets and the protocol match both original and returned
copies. All earlier failures remain preserved.

| Binding | SHA256 |
| --- | --- |
| V18 protocol | e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb |
| Returned archive | d494351f1f1f36d60f79f696633ff6e0dcc5cdf1f03662f84884f4aae71787e5 |
| Returned results JSON | f0acefaadbf2890627339038371673aeed30cf9a066b9f74ca71d28766e9c206 |

## Verified execution and evidence limits

The existing NVIDIA L4 ran all 600 updates and 6,000 exposures: 120 exposures per
case across ten training photographs/fifty paired synthetic cases. Snapshots
0/50/200/600 are all retained. The update50 fitting objective falls
0.0234957756371→0.0174959130739; ratio 0.744640796038 passes the frozen ≥1%
improvement stop. This is a fitting stop, not an output-quality pass.

Cache takes 14.9953 seconds; fitting 133.3495 seconds; trainer 148.3465 seconds;
the VM audit 24.5149 seconds; full supervisor/export 216.6474 seconds. All finite
timing limits pass. Peak allocated VRAM is 1,918,783,488 bytes, approximately
1.787 GiB. CUDA receipts contain 601 backwards and two gradient-norm traversals;
603 total autograd traversals. Frozen gradients are absent and all five frozen
state hashes agree before/after. Each trained checkpoint changes 42 tensors.

Declared forwards are verified against the receipts: 58 each retained DGP,
prior encoder/classifier, R2 code head and prior feature generator; 860 new
spatial decoder; 912 fixed-affine recognizer; zero prior RGB tail and unused V11.
Only our 684,395-parameter decoder is fitted. Retained DGP, our R2 head,
pretrained CodeFormer feature prior and pretrained ArcFace stay frozen. This
does not establish equivalence to the published DGP architecture.

The independent Windows audit takes **60.2905 seconds** and replays all 250
cached CPU decoder outputs. Maximum float difference is
**9.5367431640625e-7**, below the already frozen CPU limit 5e-5. All 250 raw/PNG
compositions and metric/cosine calculations, 50 initial exact DGP cases, eight
saved fresh VM parity cases, 600 trace rows, cache bindings, checkpoints and
550 original grid cells pass. Fresh VM/cache differences are zero, below the
unchanged VM limit 2e-6. Local audit uses zero backwards/optimizer updates and
does not rerun the DGP, prior, recognizer or CUDA optimizer/gradient execution.

There is no teacher, held-out validation, native CCTV, reserved review,
checkpoint selection or production promotion. These ten photographs are the
same identities fitted 120 times per case. Strong training resemblance and
recognizer scores do not demonstrate generalization, exact identity, native
CCTV effectiveness or performance in Zamboanga. Pretrained corpus overlap is
not excluded.

## Separate arithmetic corrections, with failures preserved

The original importer safely verifies/extracts the archive, then the frozen
local auditor fails at exact equality of float32 training-error accumulation.
Largest VM/local group mean difference is 1.4901161207725444e-9. A separate r1
auditor checks both against float64 accumulation of the same pointwise float32
squared errors, within two float32 machine epsilons, 2.384185791015625e-7 relative.
The recorded coefficients must still equal their frozen derivation exactly.
This numeric check passes; r1 then fails at exact PSNR aggregation equality.

The measured second difference comes solely from log10 evaluation: at most
7.105427357601002e-15 dB. Non-PSNR means, case counts, fitting-stop objectives/
ratio, quality booleans and failed-group lists agree exactly. Separate r2 checks
derived PSNR against independent `math.log10` within four binary64 ULPs and
propagates that bound only to PSNR gain. All MSE/SSIM/appearance means, decision
fields, fitting stop, image composition, source/data hashes and neural parity
limits remain unchanged. Ten focused numeric regressions pass across r1/r2,
including rejection of changed weights, metrics, decisions and failed lists.

The frozen V18 source/protocol/results are unchanged. The original failure and
r1 failure stay in their original directories. The successful r2 auditor and
its numeric helpers/provenance are separate; this is not a modified training
recipe, new training run or relaxed scientific quality guard.

Evidence:

1. [Original failed audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_structure_return_v18/local_audit.log>).
2. [First correction failure](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_structure_v18_audit_recovery_r1/audit.log>).
3. [Complete independent audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_structure_v18_audit_recovery_r2/local_full_audit.json>).
4. [Correction provenance and unchanged originals](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_structure_v18_audit_recovery_r2/recovery_completion.json>).
5. [Verified summary and all source/group metrics](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_structure_v18_verified_summary.json>).

## Paired synthetic training results

These are delivered-PNG metrics on identical 256×256 inputs, observed support
only. Raw float32 outputs are separately saved/audited. PNG processing only
composes padding from the input, rounds and clips; no sharpening, contrast
adjustment or preferred display transform is used.

| Fixed snapshot | Degraded PSNR dB | Degraded SSIM | Degraded fixed ArcFace | Clear SSIM | Unchanged preservation |
| --- | ---: | ---: | ---: | ---: | --- |
| 0: retained DGP | 15.5527 | 0.62517 | 0.39847 | 0.93362 | Baseline |
| 50 | 16.5515 | 0.63841 | 0.56061 | 0.92154 | Failed |
| 200 | 21.6613 | 0.66918 | 0.82093 | 0.91525 | Failed |
| 600 | 23.5201 | 0.70649 | 0.95073 | 0.92773 | Failed |

At update600, degraded MSE falls 0.0278441246566→0.00444620872303:
84.0318% reduction, +7.9674 dB. The ≥10% capacity gain passes. Four clear-image
preservation failures remain:

1. `clear:SSIM` — 0.933623893975→0.927725843050.
2. `dataset/asian_faces/clear:SSIM` — 0.971355834166→0.962517558699.
3. `dataset/thumbnails128x128/clear:MSE` — 0.000941132522979→0.000958513954638.
4. `dataset/thumbnails128x128/clear:SSIM` — 0.895891953784→0.892934127400.

The original `qualified_for_separate_generalization_protocol` remains false
at every trained snapshot. Aggregate improvement does not waive these failures.

Report the two capture-independent photographic source folders separately:

| Source/cohort | Cases | DGP→600 PSNR dB | DGP→600 SSIM | DGP→600 fixed ArcFace |
| --- | ---: | ---: | ---: | ---: |
| dataset/asian_faces, clear | 5 | 34.3930→35.4987 | 0.97136→0.96252 | 0.97783→0.99480 |
| dataset/asian_faces, degraded | 20 | 15.3258→26.1465 | 0.74500→0.82747 | 0.56141→0.95716 |
| dataset/thumbnails128x128, clear | 5 | 30.2635→30.1840 | 0.89589→0.89293 | 0.93748→0.99289 |
| dataset/thumbnails128x128, degraded | 20 | 15.7920→21.8951 | 0.50533→0.58550 | 0.23552→0.94431 |

These folder labels establish neither ethnicity nor native CCTV capture source.
No aligned clean reference exists for the project's real native CCTV evidence;
these PSNR/SSIM figures must stay separate from its unpaired review.

## Original-cell development review

All five stage sheets and five zero-prior sheets were viewed at original
resolution, covering every reference/profile: **ten sheets/550 cells**. This is
the primary agent's development review; independent final acceptance remains
pending. Per-sheet hashes and observations:
[review receipt](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_structure_review_v18.json>).

1. Clear geometry mostly survives, but glasses, skin, texture and background
   soften/change; the clear SSIM/source MSE failures remain material.
2. Blur and motion improve fitted eyes/mouths and facial coherence. Glasses
   frames, smiles/teeth, cheeks and hands still contain soft or patchy regions.
3. Low-light outputs brighten and improve fitted structure, with mottled skin,
   forehead/cheek colour patches and soft glasses remaining.
4. Compound degradation is the strongest visible limitation: coloured feature
   patches, uncertain eyes/cheeks, weak glasses and lost fine structure remain.
5. Zeroing prior features makes most degraded outputs softer and less coherent.
   It is an input-sensitivity control on the same head, not a separately trained
   no-prior model or causal proof that this prior outperforms all alternatives.

The final zero-prior degraded control gives 19.5815 dB PSNR, 0.65136 SSIM and
0.48830 fixed cosine, versus 23.5201/0.70649/0.95073 with normal features. This
shows dependence on the supplied prior features in these fitted outputs only.
Clear glasses and non-obstructing hair remain preservation targets; training
proxy resemblance does not justify object removal or exact hidden identity.

## Decision and next work

Close V18 as **capacity gain demonstrated, unconditional preservation failed**.
Keep all snapshots, frozen packages, returned outputs and failures. No V18 repeat,
best-checkpoint selection, app deployment or unconditional generalization run.

The demonstrated processing limitation is unnecessary alteration of clear input.
Diagnose a separately documented input-only preservation/automatic-selection
control on training inputs first; it must not route by known degradation labels,
targets, evaluation identities or output-quality scores. Any new control has its
own protocol, unchanged scientific thresholds and no reinterpretation of V18.
Only subsequent justified evidence can support broader finite VM training or a
separately frozen generalization/native development experiment.

Useful native CCTV restoration, insufficient-input/clearer-crop behavior,
covering-family automatic/assisted completion, DGP-led local app selection and
override, independent final review and full inline Playwright flow remain
required. The full goal stays active and incomplete.
