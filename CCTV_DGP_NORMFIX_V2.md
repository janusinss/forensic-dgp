# CCTV DGP pilot: confirmed InstanceNorm correction

Prepared 4 October 2026. Local project: `C:\xampp\htdocs\YEAR 4\Testing\`.
Existing VM package: `~/forensic-dgp/cctv_dgp_vm_bundle/`.
The autonomous Goal remains paused; these commands support the user's manual VM
pilot. No actual training or backward check runs locally.

## Cause and correction

The user's L4 / PyTorch 2.9.1+cu129 diagnostic reported no changed tensors for
batch eight and ten changed InstanceNorm running-statistic tensors for batch six.
The largest observed drift was 0.00048828125 in `smooth.1.running_var`.
PyTorch 2.9's native InstanceNorm averages/writes repeated statistics back even
in evaluation mode. The final validation and training batches each contain six
images. Merely setting `.eval()` does not preserve their exact stored bytes.

Source: [PyTorch 2.9.1 native InstanceNorm](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/Normalization.cpp#L662-L695).

The five student InstanceNorm layers pass cloned running statistics to the same
native operator. It retains the original output and gradient computation while
discarding the unwanted buffer writeback. Checkpoint keys, parameters, stored
statistics, data, order, losses, selection criteria, seed and 452-update /
90-minute budget remain. Strict matching and buffer checks are retained.

The correction is declared in `normfix_v2.json`, with hashes of its source files
and the unchanged parent protocol. The original protocol and source assets are
not edited. Runtime correction source/receipt is included in the return archive.
The original failed run had zero recorded training updates. A specifically
verified archive operation preserves it at `outputs/cctv_dgp_pilot_failed_v1/`
before a new corrected run uses `outputs/cctv_dgp_pilot/`. It refuses arbitrary
failures, any training trace, changed partial-state bytes or an existing archive.

Local verification passed 18 regression/guard checks, Python 3.10 syntax for six
files, Bash launcher syntax, and the GNU tar export filename transforms. Four
forward-only DGP calls on PyTorch 2.13.0 CPU compared corrected and original
outputs for batches eight and six: maximum output difference zero and unchanged
student state. Receipt: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_normfix_v2_checks\local_forward.json`
(copy to `~/forensic-dgp/outputs/cctv_dgp_normfix_v2_checks/` only if needed).
The legacy 2.9 writeback behavior is separately reproduced by the regression
fixture. Corrected L4 evaluation/backward compatibility remains pending.

## 1. Upload the small correction from Windows Google Cloud SDK Shell

The package is `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-normfix-v2.tar.gz`.
Its checksum is ASCII with LF. There is no data/weight download in this correction.
Use the actual VM zone if gcloud prompts; the project/VM name remains the same.

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "cctv-dgp-normfix-v2.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis "cctv-dgp-normfix-v2.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify/extract in VM SSH

These eight new source/runbook/test files and the manifest do not overwrite any
original frozen asset. Do not re-extract over an already installed correction.

```bash
cd ~ &&
sha256sum -c cctv-dgp-normfix-v2.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_vm_bundle/normfix_v2.json &&
tar -xzf cctv-dgp-normfix-v2.tar.gz -C ~/forensic-dgp &&
tmux new-session -A -s dgp_training
```

## 3. Run inside tmux

Dependencies and CUDA versions were already checked in the preceding SSH support.
This launcher checks the unchanged original package and correction manifest,
verifies the original runtime versions, preserves only the known zero-update
failure, repeats the existing CUDA gradient preflight, and adds one six-image
evaluation with a strict unchanged-state check before baseline validation.
It does not suppress the GTK dependency error or alter any dependency version.
The unused inherited PyGObject/pycairo issue was separately recorded in
`pip_check.log`; it is not a DGP dependency.

```bash
cd ~/forensic-dgp/cctv_dgp_vm_bundle &&
source .venv/bin/activate &&
python -u scripts/run_cctv_dgp_normfix_vm.py --verify &&
bash scripts/run_cctv_dgp_normfix_vm.sh --archive-zero-update-failure
```

Expected sequence: preserved zero-update failure; existing CUDA preflight passed;
`Norm correction passed: batch6, identical student state`; baseline validation;
`camera_no_identity epoch1/2 update1/226`; the two matched arms and final audit.
The corrected gradient/tail check must actually pass on the L4 VM before training
is considered started. Local verification does not establish CUDA compatibility
of the correction. Detach with Ctrl+B then D. Reattach with
`tmux attach -t dgp_training`. An error stops this attempt; preserve its evidence.

## 4. Download the corrected return in Windows Google Cloud SDK Shell

VM archive: `~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-normfix-v2-results.tar.gz`.
Local receiving directory: `C:\xampp\htdocs\YEAR 4\Testing\outputs\`.

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-normfix-v2-results.tar.gz" .
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-normfix-v2-results.tar.gz.sha256" .
```

## 5. Independently audit locally after the return arrives

Local frozen parent: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_vm_bundle_v1\`.
Local correction: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_normfix_v2\`.
VM runtime receipt: `~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_pilot/normfix_revision.json`.

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 scripts/audit_cctv_dgp_normfix_results.py --root outputs/cctv_dgp_vm_bundle_v1 --fix-dir outputs/cctv_dgp_normfix_v2 --archive outputs/cctv-dgp-normfix-v2-results.tar.gz --extract-to outputs/cctv_dgp_normfix_return_v2 --verify-recognizer --receipt outputs/cctv_dgp_normfix_return_v2/local_independent_audit.json
```

This verifies all original metrics/checkpoint/trace checks and the declared
normalization correction. The received VM receipt stays intact; the local receipt
is saved alongside the extracted archive root. It is forward-only. No app checkpoint is promoted by
training completion. After an audited return, compare qualified trained candidates
on the existing native development gallery; keep the reserved images untouched
until final evaluation. The existing parent protocol/native comparison plan stay
unchanged; save corrected-return native outputs separately if the old plan has
already produced execution artifacts. There is no corrected training result yet.
