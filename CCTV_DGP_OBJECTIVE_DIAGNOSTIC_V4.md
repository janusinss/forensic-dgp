# CCTV DGP objective diagnostic V4: existing L4 VM, zero updates

Prepared 4 October 2026. This is a diagnostic before choosing another training
recipe. The V3 return is independently audited and all four epochs failed the
unchanged guards; both V3 best files retain the starting V2 tensors. See local
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_PERCEPTUAL_RESULTS.md` (intended
`~/forensic-dgp/CCTV_DGP_PERCEPTUAL_RESULTS.md` after document transfer).

## Fixed execution scope

Reuse the existing VM bundle at `~/forensic-dgp/cctv_dgp_vm_bundle/`, its `.venv`,
902/110 reference split, data, normalization correction, V3 sources and teachers.
Windows source/data stage: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_vm_bundle_v1\`.
The retained starting checkpoint is V2 `outputs/cctv_dgp_pilot/camera_identity/best.pth`
on the VM; its local counterpart is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_normfix_return_v2\outputs\cctv_dgp_pilot\camera_identity\best.pth`.
SHA256: `b30aeabecc60dff2fbd289575ce8691be9e670c653cacb6721bc154b618c3916`.

| Limit | Fixed value |
| --- | --- |
| Cases | First four original epoch 1 cases per source/profile; 40 distinct training references |
| Groups | Two sources × clear, blur, low-light, motion and compound; 10 groups |
| Computation | Ten four-image student forwards;50 `autograd.grad` traversals on the L4 only |
| Updates and evaluation | Zero optimizer updates; no validation, native or reserved cases |
| Time | 600 seconds total cap; expected 1–3 minutes, not measured yet on this VM |

Measure weighted pixel, color, postactivation VGG, Sobel and ArcFace gradients
at one fixed model state. Coefficients remain1,0.05,0.1,0.05,0.1. Each group exports
losses and gradient Gram matrices in model-parameter and observed-image spaces.
Norms, pair cosines and summed-gradient cancellation are reconstructed from the
Gram matrices. Image gradients outside observed support are zeroed. Positive
cosines indicate locally aligned gradient directions; negative cosines identify
local conflict. These observations do not predict Adam behavior or prove that a
loss causes a quality failure. Examine both gradient spaces and all source/profile
groups before proposing a changed finite training experiment.

No optimizer is constructed, no `.backward()` is called, and no checkpoint is
exported. The student/teachers must retain exact state fingerprints throughout;
teacher parameters remain frozen and parameter `.grad` buffers stay empty.
CUDA autograd probes are deliberately guarded to the existing Linux L4 VM.
Local numeric tests/package checks make no model forwards or backward calls.

## 1. Upload from Windows Google Cloud SDK Shell

The additive package is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-objective-v4.tar.gz` plus its LF
`.sha256`. Upload each file separately; the observed Windows PuTTY backend rejects
multiple remote sources. These sources are not committed/pushed. A git pull alone
does not install this overlay. No new CUDA/environment installation is needed.

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "cctv-dgp-objective-v4.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis "cctv-dgp-objective-v4.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

Select the actual VM zone if requested; it has not been recorded in this project.

## 2. Verify and extract in Google Cloud SSH

The overlay contains only new V4 assets, its lineage receipt and new protocol.
It does not replace the original, V2 or V3 files. Extraction is permitted only
after checksum verification and while the V4 manifest does not exist.

```bash
cd ~ &&
sha256sum -c cctv-dgp-objective-v4.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_perceptual_v3/results.json &&
test ! -e ~/forensic-dgp/cctv_dgp_vm_bundle/objective_diagnostic_protocol_v4.json &&
tar -xzf cctv-dgp-objective-v4.tar.gz -C ~/forensic-dgp &&
tmux new-session -A -s dgp_training
```

If the verified overlay was already extracted, skip extraction and use
`tmux new-session -A -s dgp_training`. Preserve existing outputs and stop rather
than deleting them to rerun this diagnostic.

## 3. Run inside tmux

```bash
cd ~/forensic-dgp/cctv_dgp_vm_bundle &&
source .venv/bin/activate &&
nvidia-smi &&
bash scripts/run_cctv_dgp_objective_diagnostic_vm.sh
```

The launcher checks the existing torch/torchvision versions and CUDA, verifies
the complete original/V2/V3 chain and V4 cohort, runs the finite diagnostic, audits
reported lineage/source/state and Gram arithmetic, and exports its small return.
Progress prints once per group. Detach with Ctrl+B, then D; reattach with
`tmux attach -t dgp_training`. Do not start a second process while the first runs.

Stop on nonfinite losses/gradients, missing cohort/assets, changed model/teacher
state, unexpected parameter gradients, runtime change or deadline. Preserve
`outputs/cctv_dgp_objective_diagnostic_v4/` and `objective_diagnostic_v4.log` on
failure. Failure does not permit automatic resume/repetition. The launcher prints
completion only after its audit succeeds. CUDA backward remains unverified until
this VM execution occurs.

## 4. Download from Windows Google Cloud SDK Shell

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-objective-v4-results.tar.gz" .
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-objective-v4-results.tar.gz.sha256" .
```

Both files belong in `C:\xampp\htdocs\YEAR 4\Testing\outputs\`. This small return
contains 10 group reports and pinned execution sources, not model weights or images.

## 5. Independent local audit and subsequent decision

Run from Windows PowerShell in `C:\xampp\htdocs\YEAR 4\Testing\` after download:

```powershell
.\venv\Scripts\python.exe -X utf8 -u scripts/audit_cctv_dgp_objective_diagnostic_v4.py --root outputs/cctv_dgp_vm_bundle_v1 --bundle-dir outputs/cctv_dgp_objective_vm_v4 --perceptual-bundle-dir outputs/cctv_dgp_perceptual_vm_v3 --parent-return outputs/cctv_dgp_normfix_return_v2/outputs/cctv_dgp_pilot --v3-return outputs/cctv_dgp_perceptual_return_v3/outputs/cctv_dgp_perceptual_v3 --archive outputs/cctv-dgp-objective-v4-results.tar.gz --extract-to outputs/cctv_dgp_objective_return_v4 --receipt outputs/cctv_dgp_objective_return_v4/local_independent_audit.json
```

The expected local extracted diagnostic is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_objective_return_v4\outputs\cctv_dgp_objective_diagnostic_v4\`;
VM counterpart: `~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_objective_diagnostic_v4/`.
The auditor checks pinned sources/cohort, reported unchanged state and timing,
weight arithmetic and 20 Gram-derived summaries. It does not replay CUDA gradients
locally and cannot independently prove the reported gradients by recomputation.
Synthetic tamper fixtures are tests of the audit, not returned VM evidence.

Next: inspect per-profile gradient conflicts, identify a justified changed
training proposal with an explicit finite budget, then validate any actual
candidate through the unchanged paired and native development requirements.
Useful native output, DGP-led app integration, covering-family behavior and
independent final review remain required. The Goal is not complete.
