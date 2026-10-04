# V7 training-only DGP fit diagnostic: 4 October 2026

Both matched arms completed 200 total updates on the L4 in 35.15 seconds.
Neither met the preregistered training-fit criterion at update 100. The independent
local audit and final preview review passed; this is a completed diagnostic, not
an improved production model. The full DGP-first Goal remains active.

Windows workspace: `C:\xampp\htdocs\YEAR 4\Testing\`.
Linux bundle: `~/forensic-dgp/cctv_dgp_capacity_vm_v7/`.
All actual training uses the VM, including short pilots. The assistant now has
configured CLI start/stop/SSH/SCP access; no local CUDA setup is required.

## Frozen experiment

Ten existing approved FFHQ training pairs were selected before model execution:
the first two sorted epoch-1 case IDs per profile in the frozen V6 training cohort.
Their canonical targets, masks, camera inputs and affine geometry are copied
byte-for-byte. No new camera regeneration, role reassignment, Asian validation,
native development or reserved image is used. This is an FFHQ-only training-fit
probe, not population coverage or independent generalization evidence.

`pixel_mse` fits per-image observed RGB squared error. `retained_objective` uses
the prior Charbonnier + 0.05 color + 0.1 postactivation VGG + 0.05 Sobel + 0.1
fixed-observed identity objective. Both start from the same V2 tensors and see
the same ten inputs, shuffled traversal and fresh Adam. The diagnostic learning
rate is ten times V6 in both arms: backbone `2e-5`, other parameters `1e-4`.
Batch two, weight decay `1e-5`, clip norm one, frozen normalization/teachers,
no AMP/EMA. The matched factor is objective; cohort and rate are diagnostic
settings shared by the arms, not an exact replay of V6.

Each arm runs 100 updates; every training pair is exposed 20 times. Snapshots
25/50/100 are retained, but the declared fit decision uses the fixed final update
100 only: mean blur and motion MSE must both be at most 80% of their common starting
values. Passing would establish only fit on those examples. No `best.pth` or
production/native selection is permitted. Full-cohort safeguards remain unchanged.

The trainer has a 600-second cap and measured projection at update 16; the complete
training/audit/export supervisor has a 900-second cap. Failure preserves partial
state/logs and prohibits automatic resume/repetition. CUDA preflight records two
autograd calls and zero updates. Actual training records 200 backward calls and
200 optimizer steps. Teachers and normalization buffers remained unchanged.

## Result and audit

| Training profile | Starting MSE | Pixel-only at 100 | Retained objective at 100 |
| --- | ---: | ---: | ---: |
| Blur | 0.0095371 | 0.0201019 | 0.0197190 |
| Motion | 0.0050750 | 0.0050135 | 0.0052027 |
| Low light | 0.0592767 | 0.0118523 | 0.0221312 |
| Compound | 0.0561817 | 0.0120964 | 0.0203373 |
| Clear | 0.0009495 | 0.0008684 | 0.0004067 |

Final blur MSE ratios are 2.10775 and 2.06761; motion ratios are 0.98789 and 1.02516.
Both arms fail the fixed 0.8 criterion. Improvements in low-light/compound error
coexist with blur regression. The retained objective increases training-set blur
embedding similarity from 0.23504 to 0.67387, while blur MSE worsens; this proxy
gain does not establish accurate hidden identity, recovered detail or usefulness.

The final baseline/pixel-only/retained-objective ten-row grids were inspected.
Blurred/compound outputs remain soft, with global tone differences. Pixel-only
fitting introduces conspicuous eye/color artifacts in the blurred examples.
Stronger training embedding matching alone does not resolve these limitations.
These findings do not prove the architecture cannot fit a face: each blur example
had only 20 exposures amid other tasks. Task interference, insufficient exposure
and model/optimization limits remain competing explanations.

The VM audit/export finished in 60.93 seconds total. Local audit checked all 70
actual PNGs, 80 saved embedding arrays/70 cosines, 200 matched update/loss records, six changed
checkpoints, immutable buffers, all preview cells and the fixed final fit decision.
Each checkpoint changes 158 unique named parameter tensors. No local model forward,
backward or optimizer step was used. The audit does not replay CUDA gradients or
recognizer forwards. The terminal supervisor and its source/auditor hashes also bind
to the downloaded archive. The successful audit is reused; do not rerun it unchanged.

| Evidence | Windows local | Linux VM |
| --- | --- | --- |
| Protocol/runtime/data | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_capacity_vm_v7\` | `~/forensic-dgp/cctv_dgp_capacity_vm_v7/` |
| Returned checkpoints/metrics | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_capacity_return_v7\outputs\cctv_dgp_capacity_v7\` | `~/forensic-dgp/cctv_dgp_capacity_vm_v7/outputs/cctv_dgp_capacity_v7/` |
| Result archive/checksum | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-capacity-v7-results.tar.gz` and `.sha256` | `~/forensic-dgp/cctv_dgp_capacity_vm_v7/cctv-dgp-capacity-v7-results.tar.gz` and `.sha256` |
| Passed local audit/review | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_capacity_return_v7\local_independent_audit.json` and `assistant_preview_and_execution_review.json` | VM audit at `~/forensic-dgp/cctv_dgp_capacity_vm_v7/outputs/cctv_dgp_capacity_v7/independent_audit_vm.json`; assistant review remains local |
| Supervisor terminal receipt | `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_capacity_v7_terminal_completion.json` | `~/forensic-dgp/cctv_dgp_capacity_vm_v7/supervisor_completion_v7.json` |

Protocol SHA256: `a1d2fbb2c543c95ec1d0c7afe4670d74e7ad8009a8608daf5a2cce6209540b61`.
Input archive SHA256: `2a9f83b002a29bbe1b19408fb206ecbd127567f267ca3d472f59f31ee7ec8979`.
Return archive SHA256: `d2eecebb3bc9ffcac7e1abbd71d7455ffe5ed859b18dd9a27b497839fb0f4a9e`.
Returned results SHA256: `09af542e78fc44d1ebd3d50f84d9d5ef61104f3438f4d0814bdb04836dec142a`.
Local audit SHA256: `dab48537db9bc4f96341c0979f8e1c73c6852a561e67d636b90e00ba8fa18471`.
Assistant review/execution proof SHA256: `b68e09874b73faf3c2cdd1dbbdab3d150765106679c969023d3091914af1b4b4`.

The source/preparation/local-training guard checks passed with zero local model
operations. Its 2.05-MB upload reused three exact cached weight files only after
hash verification. Windows command length blocked the first inline launch before
VM execution; the identical hash-verified launcher was transferred as a file and
executed successfully. Failure/transport receipts are preserved. No environment
packages, historical protocols or app defaults changed.

## Next experiment decision

Follow-up completed: isolated V8 fits the same two blurred examples at the fixed
1,000-update endpoint, but collateral faces develop strong artifacts. Its local
audit and all four grids are complete; no checkpoint was promoted. Current
decision/evidence is in `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_BLUR_FIT_V8.md`
↔ intended VM `~/forensic-dgp/CCTV_DGP_BLUR_FIT_V8.md` after document sync.
The preparation paragraph below records the original V8 decision; it is no
longer a pending execution request. Next, review Asian replay sources and prepare
a broader finite mixed-source recipe before restarting the VM.

Prepare an isolated blur-fit diagnostic on the same two training pairs, preserving
the starting tensors, pixel objective, geometry and optimizer. The design budget
is 1,000 updates with checkpoints at 20, 100 and 1,000, a 600-second trainer cap
and a 900-second complete execution cap. The 20-update point matches each blur
pair's 20 exposures in V7. The longer fixed endpoint probes fitting with more
exposure. Evaluate all ten existing training pairs to report collateral effects;
do not select a production model. These comparisons help
separate observed task coupling from insufficient exposure. The design is local
at `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_blur_fit_v8_design\proposed_protocol.json`
↔ intended VM design copy only after an executable package is prepared.
Freeze the new executable protocol/runtime
before execution and run it on the VM. It remains a training-only diagnostic with
no best/default/native promotion. Do not repeat this failed mixed recipe unchanged
or infer that pixel-only loss is the next generalization recipe.

The 32 reserved native cases remain untouched. Full-cohort validation and eligible
native review are still required before DGP-led app integration or a useful-output
claim. All existing completion/covering-family requirements remain part of the Goal.
