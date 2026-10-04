# V14 r2 — numeric-control correction and finite direct-code fit

Status, 4 October 2026: completed1,000 updates on the L4 in97.27 seconds;
training/audit/export163.34 seconds. The587,564,966-byte return is independently
audited and all five grids reviewed. Coherent ten-face fitting is demonstrated;
clear preservation and unseen-image usefulness remain unresolved.
See [the results report](CCTV_DGP_DIRECT_CODES_V14_R2_RESULTS.md).
**Training-cohort capacity only**; no best-checkpoint selection, held-out
evaluation or production adoption. Six local boundary/numeric tests passed;
the48,687-byte execution archive remains frozen.

Windows root `C:\xampp\htdocs\YEAR 4\Testing\` ↔ Linux root `~/forensic-dgp/`.
Windows preparation `outputs\cctv_dgp_direct_codes_vm_v14_r2\` ↔ actual VM
`~/forensic-dgp/cctv_dgp_direct_codes_vm_v14_r2/`.
Document counterpart after sync: `~/forensic-dgp/CCTV_DGP_DIRECT_CODES_V14_R2.md`.

Protocol `d7d5dcac64202c24a7c85658e6b660bdc806bf7d8c16b09542c53170f40a1da2`.
Execution archive `810e5538bbd33defce500f9e4ef954579cf4bc7522cf4da6b3d6accdeb46ff11`.
Original failed V14 protocol
`67b8df8e6cfc7857b65c4862988f34e740bbae26960b3b07af5d06852255ae4f`
and all its sources/return remain preserved. That run made zero updates.

## What is corrected

The original check incorrectly required codebook mean/std normalization to give
an almost bit-identical decoded image. Six frozen VM renders measured the
float32 roundtrip: equal codebook values/statistics, latent difference ≤4.768372e-7,
decoder difference ≤2.980232e-6, and two color channels differing by1 uint8 level
per reference. No model change or identity/quality improvement was established.

The separate revision requires **exact matching codebook statistics** and checks
the actual normalized latent against eight float32 epsilons at its value scale.
It saves all content/normalized/stat arrays and reports decoder roundoff without
asserting decoded image identity. Independent arithmetic reconstructs this proof.
The original100 starting-output raw/PNG parity checks, frozen-state/gradient gates,
timing/VRAM budgets and previous quality safeguards are unchanged. This is a
technical control correction, not a relaxation of candidate selection criteria.

Original failure archive `outputs\cctv-dgp-direct-codes-v14-failure.tar.gz`:
42,190,592 bytes, SHA256
`150d6670253c8f66b0de3152888ab1d37e93aed70beaea55a7ba15689e6c698f`.
Independent failure audit:
`outputs\cctv_dgp_direct_codes_failure_return_v14\local_failure_audit.json`.
Diagnostic archive `outputs\cctv-dgp-stats-roundtrip-v14-results.tar.gz`:
4,122,806 bytes, SHA256
`58dce947f52d14a29ee408d2486a6fd3d675acea7e928ffbfe8c75c26e78eeec`.
Diagnostic local audit:
`outputs\cctv_dgp_stats_roundtrip_return_v14\local_independent_audit.json`.
VM counterparts are under `~/forensic-dgp/cctv_dgp_direct_codes_vm_v14/`;
the independent local audits are not synced.

## Unchanged training design

Our2,619,808-parameter direct code/statistics conditioner; original DGP and
declared CodeFormer prior/recognizer frozen. Same ten training references,
five/source, five profiles/50 cases, cached original encoder features/base logits,
and frozen DGP RGB as extra conditioning. Clean teacher labels and codebook
mean/logstd supervise training; no clean target is needed at fresh-image inference.

Forty balanced epochs, batch2,1,000 updates/2,000 exposures; Adam lr0.001,
betas(0.9,0.999), CE/mean-MSE/logstd-MSE weights1/1/1, clip1, no AMP.
Snapshots0/300/1000; observed/none/predicted rendering statistics, fidelityw0.
450 deliverable renders,150 code/stat probes,450 recognition vectors,350 grid
cells plus the separately labeled oracle numeric controls are audited.
Trainer cap600 seconds, total supervisor cap900 seconds, VRAM cap20GiB;
timing gate update25. No validation/native/reserved use or automatic selection.

## Exact execution and return paths

After verified transfer and fresh extraction, the authorized assistant starts
one dedicated `dgp_direct_codes_v14_r2` tmux job. No installer is required.

```bash
cd ~/forensic-dgp/cctv_dgp_direct_codes_vm_v14_r2
~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python -u \
  scripts/run_cctv_dgp_direct_codes_v14_supervised_r2.py \
  --root "$PWD" \
  --parent-bundle ~/forensic-dgp/cctv_dgp_face_code_fit_vm_v12_r2 \
  --expected-protocol-sha d7d5dcac64202c24a7c85658e6b660bdc806bf7d8c16b09542c53170f40a1da2
```

Collect execution-root `cctv-dgp-direct-codes-v14-r2-results.tar.gz` and `.sha256`
to Windows `outputs\`; collect `supervisor_completion_v14.json` to Windows
`outputs\cctv_dgp_direct_codes_v14_r2_completion.json`.

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 -u scripts/import_cctv_dgp_direct_codes_v14_r2.py `
  --archive outputs/cctv-dgp-direct-codes-v14-r2-results.tar.gz `
  --completion outputs/cctv_dgp_direct_codes_v14_r2_completion.json `
  --extract-to outputs/cctv_dgp_direct_codes_return_v14_r2
```

Audit once independently and review all five original256-cell ten-row grids.
Report clear preservation and degraded/source/profile structure and identity
separately. No-forward audits do not replay CUDA gradients or certify identity.
Only useful training-cohort outputs justify a separately versioned broader pilot;
ten-face capacity cannot establish generalization or Zamboanga CCTV performance.
Restore the previously stopped VM after verified return collection/idle checks.
Goal completion still requires useful DGP-led local inference and final review.
