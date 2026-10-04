# V12: bounded DGP face-code conditioning fit

**V12 r2 fitting closed, 4 October 2026:** the L4 completed 300 updates / 600
exposures on ten training faces. Trainer 125.22 seconds; complete VM
training/audit/export 177.28 seconds; peak allocated VRAM 1.43 GB. CUDA
zero-conditioner parity and gradient preflight passed. The 468,092,580-byte
return and all 116 VM source/asset bindings are verified. A 21.64-second local
no-forward audit rebuilt all 300 renders, 150 code probes and 500 grid cells.
All ten original-cell grids were reviewed. **No useful upgrade:** degraded
code accuracy stays near 2%, clear accuracy falls 21.35% to 13.78%, and facial
artifacts persist. No best.pth, selection or production promotion; no validation,
native development or reserved access. The original Python3.10 loader failure
(zero updates) and its separate r2 compatibility correction remain preserved.

Current report: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_FACE_CODE_FIT_V12_RESULTS.md`
↔ intended `~/forensic-dgp/CCTV_DGP_FACE_CODE_FIT_V12_RESULTS.md` after doc sync.
Audited local return: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_face_code_fit_return_v12_r2_verified\`
↔ executed VM results below `~/forensic-dgp/cctv_dgp_face_code_fit_vm_v12_r2/outputs/cctv_dgp_face_code_fit_v12/`.
The earlier sandbox-blocked partial extraction is retained separately. Receipts,
learning diagnosis and grid review are linked in the report. All return files
are local; idle checks passed and Cloud confirms **TERMINATED**. Source/doc/git
publication has not occurred. The active Goal remains incomplete.

**Next:** freeze an inference-only training-cohort control that separates code
prediction from original-input AdaIN/fidelity rendering. Clean teacher-label
arms are declared oracle diagnostics, not learned results. Do not repeat this
recipe or scale it up before identifying a specific repair.

Historical initial status: the first package was prepared and launched, then
stopped before updates at the Python3.10 hash API incompatibility. The current
separate r2 execution is described above. This
is not a trained upgrade or a full-data pilot. No app default changes.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`; VM root: `~/forensic-dgp/`.
Local executable: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_face_code_fit_vm_v12\`.
VM executable: `~/forensic-dgp/cctv_dgp_face_code_fit_vm_v12/` after verified transfer.
The source archive is thin; its four required weights are separately bound to
verified cached or transferred files. Missing or changed weights stop execution.

## What trains

Only our V11 conditioner's **455,072 parameters** learn. Retained V2 DGP,
official CodeFormer prior, clean VQGAN teacher and recognizer stay frozen.
The original RGB enters the prior encoder; frozen DGP RGB supplies three extra
conditioning channels. The zero-initialized residual changes face-code prediction.
Original encoder features provide AdaIN/fidelity connections when rendered.

V11 proved exact zero-conditioner parity locally, stable state/counts and coherent
clean-target prior capacity, with some feature changes. V10's RGB cascade was
rejected. This diagnostic tests whether our layer can learn the target codes;
it does not repeat either failed approach or claim the prior is our own.

The same ten V11 training references, five per source, use all five inherited
camera profiles: 50 cases. Twelve balanced epochs pair one Asian and one FFHQ
case per batch. Every case appears once per epoch: **300 updates / 600 exposures**.
No validation or native data is used, including reserved CCTV. Asian targets
remain source photographs below native 256 resolution, not high-quality truth.

Code labels come from the official clean teacher. Feature targets are the frozen
prior codebook vectors at those indices. Observed-token cross entropy (weight
0.5) plus latent feature MSE (weight 1.0) follows the declared author supervision
form. Adam lr 1e-4, betas (0.9, 0.999), gradient norm cap 1.0; no AMP. This recipe
is fixed before output, with no validation-derived tuning.
[Official training implementation](https://github.com/sczhou/CodeFormer/blob/master/basicsr/models/codeformer_model.py).

## Preflight, budgets and evidence

Both trainer entry and trainable-code graph reject machines other than the
existing `forensic-dgp-thesis` Linux NVIDIA L4. Four boundary tests pass, including
an actual rejected local trainer entry before models/output/backward. The first
test receipt failed because a dictionary Counter was interpreted as explicit
counts; the corrected key-count test passed and the earlier receipt is preserved.

CUDA preflight checks the unchanged PyTorch/torchvision runtime, a zero-conditioner
baseline comparison and one backward call with **zero optimizer updates**.
Finite gradients must reach our projection; all frozen parameters must have no
gradients or state changes. The zero projection initially leaves trunk gradients
zero; actual fitting must subsequently change both trunk and projection.

Trainer cap **600 seconds**, total training/audit/export cap **900 seconds**;
peak allocated VRAM at most 20 GiB. Update-20 measured timing projects the remaining
training and two measured-cost snapshots; excessive projection stops. Wrong
pins, nonfinite values, changed frozen state or count violations stop. Preserve
partial/failure evidence; no automatic resume, repeat or edit of frozen sources.

Snapshots 0, 100 and 300 render all 50 cases at w0 and w1: 300 output images,
raw floats and embeddings. Store all 150 code-logit/conditioned-feature probes
so independent audits can rebuild supervised losses and code accuracy. Incremental
update traces retain evidence if the run stops. Terminal neural state/count proof
is written before the ten profile/source preview grids.

No `best.pth` is created. There is no automatic selection or promotion from
training-only fit. Inspect code learning and visible facial structure before
deciding whether a broader, separately frozen held-out pilot is justified.
Training completion alone is not the Goal's useful-output criterion.

## Frozen transfer and commands

Protocol SHA256: `3452466e5618ed30cde58826f0bcb000afcafe786d07e3db2c64c120f561b939`.
Source archive: Windows `outputs\cctv-dgp-face-code-v12-execution.tar.gz` ↔
VM `/home/janusdominic0/cctv-dgp-face-code-v12-execution.tar.gz` after transfer.
108 members, 3,438,656 bytes; SHA256
`a26aecc8234676cf40d48fb4e30844ce4411b71e441f367c79c47f208278adba`.
Independent member hashes and all 50 inherited case bytes are verified locally.

The assistant operates configured gcloud start/SSH/SCP directly. These commands
document the remote launch **after** the package and external weight binding,
idle-job checks and full verification. They are not a request for duplicate runs:

```bash
cd ~/forensic-dgp/cctv_dgp_face_code_fit_vm_v12
PYTHON=~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python
"$PYTHON" -u scripts/run_cctv_dgp_face_code_fit_v12_supervised.py \
  --root "$PWD" \
  --expected-protocol-sha 3452466e5618ed30cde58826f0bcb000afcafe786d07e3db2c64c120f561b939
```

Dedicated tmux session: `dgp_face_code_v12`. Successful return:
`~/forensic-dgp/cctv_dgp_face_code_fit_vm_v12/cctv-dgp-face-code-v12-results.tar.gz`
↔ Windows `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-face-code-v12-results.tar.gz`.
Failure export uses `cctv-dgp-face-code-v12-failure.tar.gz`. Collect and verify
archive/SHA plus supervisor completion separately; audit locally with no backward.
Restore the VM's prior stopped state after verified idle checks and collection.

Next: independently audit the returned pixels, code losses, states and trace;
review all ten original-cell grids. A held-out validation pilot is conditional
on the learned conditioner improving useful structure on this fitting diagnostic.
No real Zamboanga CCTV or final independent review is yet available.

Pretraining and proxy-split limits: `CCTV_FACE_PRIOR_DATA_LIMITS.md` beneath
the Windows root, intended same filename below `~/forensic-dgp/` after document
sync. Our holdout is not proven unseen by the pretrained weights.
