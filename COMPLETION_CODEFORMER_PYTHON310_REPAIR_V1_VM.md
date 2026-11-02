# CodeFormer completion: Python 3.10 repair and manual continuation

25 October 2026. The reported failure occurred in the first protocol hash during --verify-transfer, before model loading, gradients or optimizer updates. hashlib.file_digest was introduced in Python3.11; the VM uses Python3.10. This is a compatibility defect in the prepared scripts, not a learned-model quality failure.

Windows transfer files are in the ORIGINAL C:\xampp\htdocs\YEAR 4\Testing\outputs folder as requested. The VM uses its Linux ~/forensic-dgp folder; it does not read a Windows workspace path. Completion source development remains isolated on FACE_MASK_COMPLETION.

Only the diagnostic and export SHA256 helpers change to streamed hashlib.sha256. The repair verifies all27 installed packet files, preserves the original scripts/protocol in a new sibling backup, and refuses existing diagnostic outputs/logs. Weights, data, masks, gradients, conditions and stop limits stay exact. No checkpoint re-upload or dependency/environment upgrade is needed.

Local verification:11 independent tests, Python3.10 grammar checks and a full-packet transfer dry run with hashlib.file_digest removed passed. Actual Python3.10 was not installed locally; final VM transfer verification remains required. No model/gradient/VM execution was performed during repair preparation.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "repair_completion_codeformer_gradient_python310_v1.py" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "repair_completion_codeformer_gradient_python310_v1.py.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

The repair is56,692bytes. Its SHA256 is88a77caf85ef995a185f3d7427596fc18da93e7bf23d3dacdf85385e25c33ed3. The sidecar uses Linux LF newlines.

## 2. Apply in VM SSH

```bash
cd ~ &&
sha256sum -c repair_completion_codeformer_gradient_python310_v1.py.sha256 &&
python3 ~/repair_completion_codeformer_gradient_python310_v1.py --expected-sha 88a77caf85ef995a185f3d7427596fc18da93e7bf23d3dacdf85385e25c33ed3
```

Expected: complete:true, files_changed:3, scientific_settings_unchanged:true, optimizer_updates:0. The three changed files are the two hash helpers and their exact protocol bindings. The output gives the backup directory. If already_applied:true appears, it verified the repaired files and made no changes. If any check fails, retain the error and do not bypass it or delete existing evidence.

## 3. Verify and launch inside tmux

If you are already inside tmux, stay there. Otherwise open/attach it:

```bash
tmux new-session -A -s completion_cf_gradient_v1
```

Then use the NEW protocol SHA:

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_gradient_v1_vm.py --root . --protocol-sha e79183a9f4e3ea50e9ac1d6705d9667f046364132e1f1be0628befd024835d9a --verify-transfer &&
bash scripts/run_completion_codeformer_gradient_v1.sh e79183a9f4e3ea50e9ac1d6705d9667f046364132e1f1be0628befd024835d9a
```

Transfer verification must print complete:true,27 files,8 cases,0 neural calls and0 gradient queries. Only the following wrapper starts the manual diagnostic. Maximum600seconds,24 typed forward slots and16 gradient queries; zero optimizer updates. This is a learning-path diagnostic, not a training improvement or completed goal.

Monitor from another SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm/diagnostic.log
```

Detach tmux with Ctrl+B, then D. At the end inspect the actual terminal and export receipts:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm/supervisor.json
cat ~/cctv-completion-codeformer-gradient-v1-export.json
```

Export complete:true confirms archive creation only. Diagnostic child_exit, timeout and returned gradient_path_pass remain separate. The wrapper exports retained failed/timeout results as well as successful results. Do not append epochs or rerun a failed quality recipe unchanged.

## 4. Download from Windows Google Cloud SDK Shell

One remote source per command avoids the Windows PuTTY multiple-source limitation.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-gradient-v1-results.tar.gz" "."
gcloud compute scp --project=dforensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-gradient-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-gradient-v1-export.json" "."
```

Return these three files for independent audit. Original R1 protocol e1e4b9fd8c4bf71db806deca03a66e187e9726cafff65ff799acc25aa432387e and all original packet bytes remain retained locally; the VM repair keeps its own original backup. New compatible protocol is e79183a9f4e3ea50e9ac1d6705d9667f046364132e1f1be0628befd024835d9a. Do not use the old protocol SHA after repair.
