# RGB/grayscale pilot return review — prepared 3 October 2026

The return auditor is ready locally. The new VM pilot has not been observed
running in this chat, and its result archive is not present locally. This document
does not report an executed CUDA preflight, new training or improved face output.

| Artifact | Windows local workspace | Linux VM |
| --- | --- | --- |
| Frozen training bundle | `C:\xampp\htdocs\YEAR 4\Testing\outputs\gray-covering-vm-bundle.tar.gz` | Upload to `/home/janusdominic0/`, extract into `~/forensic-dgp/gray_covering_vm_bundle/` |
| Training commands | `C:\xampp\htdocs\YEAR 4\Testing\GRAY_COVERING_VM.md` | `~/forensic-dgp/gray_covering_vm_bundle/GRAY_COVERING_VM.md` inside the unchanged bundle |
| Result archive and checksum | Download to `C:\xampp\htdocs\YEAR 4\Testing\outputs\` | `~/forensic-dgp/gray_covering_vm_bundle/gray-covering-results.tar.gz` and `.sha256` |
| Independent return auditor | `C:\xampp\htdocs\YEAR 4\Testing\scripts\audit_gray_covering_results.py` | Intended `~/forensic-dgp/scripts/audit_gray_covering_results.py` only after a separate source transfer; it is not in the frozen training bundle |
| Prepared audit evidence | `C:\xampp\htdocs\YEAR 4\Testing\outputs\gray_covering_return_audit_preparation_v1\` | Intended `~/forensic-dgp/outputs/gray_covering_return_audit_preparation_v1/` only after a separate evidence transfer |
| Returned checkpoint files after audit | `C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_gray_covering_v1\outputs\gray_covering_vm\` | `~/forensic-dgp/gray_covering_vm_bundle/outputs/gray_covering_vm/` |
| Future audit report | `C:\xampp\htdocs\YEAR 4\Testing\outputs\gray_covering_results_validation_v1\verification.json` | Intended `~/forensic-dgp/outputs/gray_covering_results_validation_v1/verification.json` only after a separate evidence transfer |

## 1. Complete the finite pilot on the VM

Use the upload/extraction/tmux commands in `GRAY_COVERING_VM.md`. Inside the new
workspace `~/forensic-dgp/gray_covering_vm_bundle/`, run:

```bash
bash scripts/run_gray_covering_vm.sh
```

The wrapper verifies the existing environment and source data, performs one
grayscale batch with zero updates, then executes both finite detector branches.
If the RGB model at update 128 differs from the previous varied133 model, the
script stops and retains the partial run. Preserve that output for diagnosis;
do not rerun into the same directory or remove the reproduction requirement.

## 2. Download after the completion message

In Windows **Google Cloud SDK Shell** (CMD), using the working project/instance
configuration recorded in `WINDOWS_GCLOUD_TRANSFER.md`:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/gray_covering_vm_bundle/gray-covering-results.tar.gz" .
gcloud compute scp --project=forensic-dgp-thesis "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/forensic-dgp/gray_covering_vm_bundle/gray-covering-results.tar.gz.sha256" .
```

Each transfer has one remote source. The checksum must have LF line endings.
Keep the camera/varied result files; the new return has a distinct name. Do not
overwrite an existing gray return or audit directory without preserving it.

## 3. Independently audit locally

In Windows PowerShell:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
.\venv\Scripts\python.exe -X utf8 scripts/audit_gray_covering_results.py
```

This command constructs no detector or optimizer and performs zero model forwards
and zero weight updates. It verifies 3,267 regular archive members, the LF checksum,
all member digests and the frozen sent protocol/inventory before extraction.
Unexpected roots, links, traversal, duplicates, missing files, oversized exports
or altered contents are rejected. Existing extraction/audit evidence is preserved.

The audit independently reconstructs both 512-step schedules, verifies all 1,024
logged real/grayscale/replay conditions and fresh-update counters, and recounts
all 3,248 binary masks from unchanged targets and supervision support. RGB and
grayscale scoring share geometry; ignored pixels remain separate from false
positives and do not count as detected hidden features.

All three checkpoints must have the expected metadata, 184 finite model tensors
and 92 unchanged reference-head/BN tensors. RGB128 must equal the previously
returned varied133 model in every state tensor. Source and RGB128 RGB/fixture
masks must also match the prior return (1,092 mask comparisons). Both ten-row
preview grids are reconstructed pixel-for-pixel. The independently computed
training-fit decision must preserve its failures and cannot select `best.pth`.

Final optimizer tensors remain on the VM. Their reported step counters and
remote timings/forward counts remain frozen-code/log claims; this audit does not
inspect the moments or independently observe VM execution. A saved-mask recount
does not reproduce final model predictions. Those limits are recorded in the report.

The separate read-only preparation check is:

```powershell
.\venv\Scripts\python.exe -X utf8 scripts/audit_gray_covering_results.py --check-contract
```

It validates the sent package, current source assets, schedules and all 812
measurement targets. Its `result_audit_completed` value remains `false`; it must
not be presented as a returned-model audit or a trained-quality result.

## 4. Review actual output before checkpoint selection

After a successful integrity audit, inspect both complete preview grids and run
local inference on the unchanged 36-case practical gallery. Report automatic and
reviewed-mask results separately. Recheck hair, covering edges, ordinary clear
glasses and the two nearly hidden inputs through the existing visibility guard.
Then generate the same ten degraded face-output comparisons through the retained
completion/restoration backend and inspect covering remnants, plausible anatomy,
visible appearance and seams.

These sources have already been inspected: results are development evidence.
The original 425-case failures, original data splits and historical protocols stay
unchanged. A training-fit pass is insufficient for promotion or Goal completion.
The application's existing checkpoint is retained until the required practical
output and regression evidence supports a change. All actual training remains
VM-only; local inference and file audits are permitted.
