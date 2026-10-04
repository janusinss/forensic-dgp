# V13: separate saved face-code prediction from rendering

Status, 4 October 2026: the frozen L4 control completed in 52.21 seconds;
return collection, the 10.98-second independent local audit and all five grid
reviews are complete. The VM is stopped. This is an **inference-only
training-cohort diagnostic**, not another training run. Findings and evidence:
`CCTV_DGP_FACE_CODE_CONTROLS_V13_RESULTS.md` beneath both document roots after sync.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`; VM root: `~/forensic-dgp/`.
Local preparation: `outputs\cctv_dgp_face_code_render_controls_vm_v13\`.
VM execution: `~/forensic-dgp/cctv_dgp_face_code_render_controls_vm_v13/`.
Document counterpart after explicit sync: `~/forensic-dgp/CCTV_DGP_FACE_CODE_CONTROLS_V13.md`.

V12's aggregate code loss decreased, but degraded accuracy stayed near 2%,
clear code accuracy declined, and trained outputs retained displaced eyes,
invented glasses and changed mouths. Before more fitting, compare its saved
initial/learned top-1 codes against clean teacher labels through the same frozen
codebook and renderer. Separate original-input AdaIN statistics from code choice.
Fidelity is fixed at w0, so this diagnostic isolates AdaIN rather than comparing
the w1 skip connections. Clean teacher labels are deliberately target-informed
**oracle controls**; they are never deployable restoration or a trained upgrade.

Four new arms: initial codes without input statistics; learned update300 codes
without input statistics; teacher codes with original observed-input statistics;
teacher codes without input statistics. The cached learned original-stat w0
output, camera input and clean training reference accompany them in each grid.
No image/identity is added or selected based on a new outcome: retain all ten
V12 training references and all five profiles (50 cases).

There are **162 generator forwards**: 150 case-specific renders, 10 cached
teacher no-stat renders reused across profiles, and two parity forwards checking
the old cached original-stat w0 rendering (maximum float deviation ≤2e-6).
No DGP, conditioner, encoder, transformer, teacher or recognizer forwards run.
All weights remain frozen. Zero backward calls and optimizer updates; no
validation, native or reserved data. No checkpoint, selection or app promotion.
Budget: ≤240 seconds for loading/neural execution and a 360-second external
process timeout including export. Stop the L4 after verified collection/idle checks.

Protocol SHA256: `093cf67b630de4ed74d0c2fa62fc928a69c2c2464e5b9ecfd23d22a8a0a603ac`.
Four-file source archive: **19,887 bytes**, SHA256
`90f7a25bf834c7e8ffc8be2e0271d4fd945c9e83834ef3c18d24a5a51f4b7df5`.
Parent V12 protocol: `06f056ea4b1b04e6d8c631e5e78df129345c78279230c7946f6b5deb2e226846`.
Parent results: `95fb1993e93aba1972ae89d024df1ad8d50574d1fefd7eacf5dc977b6c9c4796`.
Reuse the verified cached prior and original CUDA environment. No installer or
source modification in the frozen V12 bundle is required.

The actual local runtime entry was tested: it rejects this machine before
creating outputs or constructing a neural model. Preparation and return audits
are permitted locally; this control is scheduled on the L4 to bound runtime.

Historical launch command after verified transfer/fresh extraction:

```bash
cd ~/forensic-dgp/cctv_dgp_face_code_render_controls_vm_v13
timeout --signal=TERM --kill-after=10s 360s \
  ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python -u \
  scripts/render_cctv_dgp_face_code_controls_v13.py \
  --root "$PWD" \
  --parent-bundle ~/forensic-dgp/cctv_dgp_face_code_fit_vm_v12_r2 \
  --expected-protocol-sha 093cf67b630de4ed74d0c2fa62fc928a69c2c2464e5b9ecfd23d22a8a0a603ac
```

Return archive: VM execution root `cctv-dgp-face-code-controls-v13-results.tar.gz`
and `.sha256` → Windows `outputs\` with the same basenames. Collect the separate
VM `completion.json` → Windows `outputs\cctv_dgp_face_code_controls_v13_completion.json`.
Use the frozen independent auditor once:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 -u scripts/audit_cctv_dgp_face_code_controls_v13.py `
  --archive outputs/cctv-dgp-face-code-controls-v13-results.tar.gz `
  --completion outputs/cctv_dgp_face_code_controls_v13_completion.json `
  --extract-to outputs/cctv_dgp_face_code_render_controls_return_v13
```

The audit checked 200 PNG compositions /160 distinct raw renders and all 350
original grid cells without neural calls. All five grids were reviewed. Clean
labels render coherent faces while learned labels remain distorted; observed
statistics also transfer degraded texture/brightness. Both code prediction and
rendering statistics require a distinct repair. Removing AdaIN alone worsens
several predicted-code outputs. No checkpoint, selection or app promotion.
Next: a finite VM-only direct-code / clean-statistics capacity pilot, with
starting parity and structural review before held-out evaluation. Independent
final assessment, native CCTV usefulness and DGP-led integration remain pending.
