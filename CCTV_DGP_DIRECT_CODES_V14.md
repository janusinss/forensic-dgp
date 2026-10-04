# V14 — direct code prediction and clean rendering statistics

Original V14 status, 4 October 2026: stopped after the CUDA gradient preflight,
with one backward and **zero optimizer updates**, at an invalid oracle decoder
identity check. The 42,190,592-byte failure return is independently audited;
the partial conditioner exactly matches its initial state. Preserve this frozen
protocol and its failure. No training snapshot or final neural-count proof exists.

A separate six-render diagnostic found identical codebook values/statistics but
float32 normalization roundoff: maximum decoder difference2.980232e-6, affecting
two uint8 channels by1 per case. Its return is independently audited; no fitting.
The numeric check is corrected in a separately frozen revision, without changing
the model, inputs, targets, objectives, LR or update budget. Starting100-output
raw/PNG parity remains unchanged. Current runbook: `CCTV_DGP_DIRECT_CODES_V14_R2.md`
under both document roots after explicit sync. Do not rerun this original package.

Protocol SHA256: `67b8df8e6cfc7857b65c4862988f34e740bbae26960b3b07af5d06852255ae4f`.
Execution archive SHA256: `84e8a5bc72c7e63f8bf7ae0a00cb78aa8ef51fc0218ca369c012a8dd29fa60ae`.
All frozen execution sources and parent assets verified; no local backward or
optimizer calls. Preserve this protocol and package after execution.

Windows root `C:\xampp\htdocs\YEAR 4\Testing\` ↔ Linux root `~/forensic-dgp/`.
Prepared execution `outputs\cctv_dgp_direct_codes_vm_v14\` ↔ actual VM
`~/forensic-dgp/cctv_dgp_direct_codes_vm_v14/`. Document counterpart after
explicit sync: `~/forensic-dgp/CCTV_DGP_DIRECT_CODES_V14.md`.

## Why this repair

V12 reduced code CE by flattening predictions without improving degraded code
accuracy. V13 showed that correct target codes can produce coherent faces,
while observed degraded AdaIN statistics carry unwanted texture/brightness.
Simply removing statistics leaves wrong-code outputs structurally unreliable.
See `CCTV_DGP_FACE_CODE_CONTROLS_V13_RESULTS.md` beneath both document roots
after explicit sync; clean-code oracles are not learned CCTV restoration.

Our **2,619,808-parameter** conditioner predicts residual logits directly and
channel mean/log-standard-deviation corrections. Its inputs are original RGB,
frozen retained-DGP RGB and frozen prior encoder features/base logits. Both
output projections start at zero. The original DGP, CodeFormer encoder,
classifier, codebook, renderer and recognizer remain frozen. The contribution
must be described as our trained conditioning plus a declared pretrained prior.

Training uses audited cached original features/logits from V12, not clean-target
features at inference. Clean teacher labels supervise observed face tokens.
Rendering-statistic targets are the channel mean/logstd of the corresponding
clean-label prior codebook vectors, using unbiased spatial variance plus1e-5.
Mean MSE and logstd MSE are supervised separately; V12 feature MSE is removed.
No identity gradient, decoder tuning, alignment change or display sharpening.
Fresh-image inference recomputes original features and DGP RGB and requires no
teacher/clean target. This interface is experimental and is not app-integrated.

## Frozen finite design

The same ten training references (five/source) and five profiles yield50 cases.
No new identities, output-based sampling, validation, native or reserved use.
Forty balanced epochs, batch2, **1,000 updates /2,000 exposures**, Adam lr0.001,
betas(0.9,0.999), CE/mean-MSE/logstd-MSE weights1/1/1, norm clip1, no AMP.
Snapshots0/300/1000 are diagnostic adapters. This changes architecture,
objectives, LR and exposures from V12; it is not a single-factor causal ablation.

Each snapshot renders all50 cases with the same code choices and observed,
omitted or predicted statistics, with fidelityw0. All450 deliverable PNGs and raw
renders,150 code/stat probes,450 recognition vectors and three checkpoints are
retained. Five ten-row grids show original256 cells. Outside observed support,
the original camera image is copied unchanged. Clear preservation and source/
profile identity are reported separately; code loss alone cannot select weights.

Expected neural calls:50 frozen-DGP cache forwards,1,151 conditioner forwards,
454 generator forwards,450 recognizer forwards. No prior encoder/classifier,
teacher or legacy-conditioner forwards during this cached fitting diagnostic.
The extra four generator forwards verify two target-informed codebook-stat
oracle pairs, separately labeled and never claimed as learned outputs.

All actual training and the one-backward/zero-update preflight run on the user's
existing NVIDIA L4 VM. Stop if either head is unreachable, frozen weights change,
starting output parity fails, predictions become nonfinite, VRAM exceeds20GiB,
or timing exceeds the finite budget. Trainer cap **600 seconds**, supervisor
including audit/export **900 seconds**; projected timing gate at update25.
Do not resume a partial failure or repeat this recipe automatically.

## Execution and independent audit

Reuse the verified V12 r2 sources/weights/cache and the existing CUDA runtime;
no installation or modification of frozen parent files. The assistant is
authorized to start/SSH/SCP/stop this VM. Verify no competing GPU/tmux/project
task before launch and before restoring its prior stopped state.

Historical launch after exact transfer/fresh extraction:

```bash
cd ~/forensic-dgp/cctv_dgp_direct_codes_vm_v14
~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python -u \
  scripts/run_cctv_dgp_direct_codes_v14_supervised.py \
  --root "$PWD" \
  --parent-bundle ~/forensic-dgp/cctv_dgp_face_code_fit_vm_v12_r2 \
  --expected-protocol-sha 67b8df8e6cfc7857b65c4862988f34e740bbae26960b3b07af5d06852255ae4f
```

VM execution root `cctv-dgp-direct-codes-v14-results.tar.gz` and `.sha256`
→ Windows `outputs\` with the same names. VM `supervisor_completion_v14.json`
→ Windows `outputs\cctv_dgp_direct_codes_v14_completion.json`.

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 -u scripts/import_cctv_dgp_direct_codes_v14.py `
  --archive outputs/cctv-dgp-direct-codes-v14-results.tar.gz `
  --completion outputs/cctv_dgp_direct_codes_v14_completion.json `
  --extract-to outputs/cctv_dgp_direct_codes_return_v14
```

The independent auditor rebuilds saved code/stat losses, pixels, embeddings,
grid cells, trace arithmetic and changed states without any neural, backward
or optimizer calls. It does not replay CUDA gradients or certify identity.
Only a coherent train-cohort improvement justifies a separately frozen broader
pilot. Ten-face fitting cannot establish held-out or real CCTV usefulness.
No real Zamboanga samples or independent final review exist yet. Keep previous
gates, failure history, dataset-overlap limitations and production weights.
