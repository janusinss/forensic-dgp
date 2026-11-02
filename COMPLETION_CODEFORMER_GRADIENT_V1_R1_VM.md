# Superseded after VM Python3.10 compatibility failure

Use [the small repair and updated commands](COMPLETION_CODEFORMER_PYTHON310_REPAIR_V1_VM.md). Do not relaunch the old protocol SHA after applying the repair. Original instructions are retained below.

---

# CodeFormer completion: manual L4 gradient check R1

PREPARED, NOT RUN. This checks which parts can learn before fine-tuning. It creates no optimizer, performs zero weight updates and does not replace the app model. Four existing TRAIN references/eight fixed cases,24 typed forward slots and16 gradient queries; maximum10 minutes for the diagnostic. Keep the original CodeFormer checkpoint. No new training was launched locally or remotely.

Target: project forensic-dgp-thesis, instance forensic-dgp-thesis, zone us-central1-a, user janusdominic0. These commands use the existing L4 VM details; use new details you provide when moving VMs. The VM needs the existing cctv_dgp_vm_bundle/.venv and at least2GiB free for archive, extraction, bounded results and export. Execution requires one idle NVIDIA L4 on g2-standard-4.

## 1. Upload from Windows Google Cloud SDK Shell

The files are in the **FACE_MASK_COMPLETION worktree**, separate from restoration.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-gradient-v1-r1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-gradient-v1-r1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

Archive:349,969,718bytes, SHA-256 `c17dea799cc29f560b567acf138b0ee966cad4421976321152112d630c0e61ff`. Its checksum sidecar has Linux LF newlines. All28 archive members passed independent verification. Do not use the retained earlier unrun staging draft.

## 2. Install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-completion-codeformer-gradient-v1-r1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm &&
python3 -c 'import pathlib,shutil; assert shutil.disk_usage(pathlib.Path.home()).free >= 2*1024**3, "Need at least 2 GiB free; stop and inventory storage"' &&
tar -xzf cctv-completion-codeformer-gradient-v1-r1-execution.tar.gz -C ~/forensic-dgp
```

If a test fails, retain its error and stop; do not delete or overwrite an existing diagnostic folder to bypass it.

## 3. Open tmux

```bash
tmux new-session -A -s completion_cf_gradient_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_gradient_v1_vm.py --root . --protocol-sha e1e4b9fd8c4bf71db806deca03a66e187e9726cafff65ff799acc25aa432387e --verify-transfer &&
bash scripts/run_completion_codeformer_gradient_v1.sh e1e4b9fd8c4bf71db806deca03a66e187e9726cafff65ff799acc25aa432387e
```

Transfer verification prints27 files,8 cases,0 neural calls,0 gradient queries and0 optimizer updates. The actual diagnostic writes progress into diagnostic.log; its main terminal can stay quiet while it runs. Detach safely with Ctrl+B, then D.

From another SSH terminal, monitor without launching another run:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm/diagnostic.log
```

Ctrl+C here stops only tail. The wrapper attempts bounded evidence export after success, failed gradient qualification or timeout. At the end inspect:

```bash
cat ~/cctv-completion-codeformer-gradient-v1-export.json
cat ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm/supervisor.json
```

Export `complete:true` means only archive creation succeeded. Check diagnostic_child_exit, timeout and returned gradient_path_pass separately. A child exit2 or other failure is retained evidence, not a reason to add epochs or rerun unchanged. The result archive excludes model weights and is capped at160MiB; export has a180-second writer limit.

## 5. Download to Windows Google Cloud SDK Shell

Use one remote source per command because the Windows PuTTY backend rejects multiple remote sources.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-gradient-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-gradient-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-gradient-v1-export.json" "."
```

Tell this chat when the three return files are downloaded. They will be independently audited before any finite fine-tuning pilot. No automatic training follows a pass. The diagnostic uses the same-model frozen clean encoder/codebook teacher, so connectivity does not prove better anatomy or hidden identity. Reserved final data and original validation pixels are not used. Restoration development remains separate.
