> Historical V1 guide: this pilot executed and stopped before training. Use [the isolated R1 guide](COMPLETION_CODEFORMER_FINETUNE_V1_R1_VM.md); retain V1 and do not rerun it.

# CodeFormer completion: first manual fine-tuning pilot V1

25 October 2026. PREPARED, NOT RUN. The previous gradient diagnostic passed. **This new run actually updates the completion model**: maximum32 paired cycles,32 code Adam steps plus32 fusion Adam steps=64 optimizer updates. It keeps the original checkpoint and freezes the original codebook/generator. RESTORATION and IMPLEMENTATION are unchanged.

Use the existing forensic-dgp-thesis L4 VM, project forensic-dgp-thesis, zone us-central1-a, user janusdominic0. Training remains your manual tmux action. The packet is only27,675,512bytes because it reuses the exact original checkpoint already installed by the previous diagnostic; no checkpoint re-upload is needed.

The pilot uses32 FIT and eight separate MONITOR source photos, all original TRAIN-role; four fixed covering geometries,128fit/32monitor cases and32 retained unpaired real/degraded public development cases. Monitoring is adaptive TRAIN evidence, not an independent identity/final test. Actual model input follows the app's prepared256→512 visible-context path; the clean HQ target is supervision only.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-finetune-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-finetune-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

Archive SHA256: d5d463d987855dcce2d770b975980fbfb801bff2ce669a772083004ec7aa13c3. Checksum sidecar uses Linux LF newlines. The packet contains330 regular members, including its protocol; it contains no original weight file.

## 2. Install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-completion-codeformer-finetune-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth &&
test ! -e ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_vm &&
python3 -c 'import pathlib,shutil; assert shutil.disk_usage(pathlib.Path.home()).free >= 4*1024**3, "Need at least 4 GiB free; stop and inventory storage"' &&
tar -xzf cctv-completion-codeformer-finetune-v1-execution.tar.gz -C ~/forensic-dgp
```

If a test fails, stop and retain its error. Do not delete an existing experiment or change hashes to bypass it. A new VM will need the verified base checkpoint and environment prepared first; these instructions assume the same VM as the completed gradient diagnostic.

## 3. Open tmux

If already inside tmux, use that session. Otherwise:

```bash
tmux new-session -A -s completion_cf_finetune_v1
```

## 4. Verify and launch inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_finetune_v1_vm.py --root . --protocol-sha 0dfe963bd29856ea6201d6397aa25d651ca0291661fa3ec26539c9c770e31b8f --verify-transfer &&
bash scripts/run_completion_codeformer_finetune_v1.sh 0dfe963bd29856ea6201d6397aa25d651ca0291661fa3ec26539c9c770e31b8f
```

Transfer verification must print complete:true,329files,128fit cases,32monitor cases,0neural calls and0optimizer updates. That verifies files only; the following wrapper starts actual training.

The first part caches original clean-reference labels and computes unchanged baseline outputs; the terminal can be quiet until the first training cycle. The worker logs each completed cycle out of32. Detach safely with Ctrl+B then D. Do not launch a second copy.

Monitor from another SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_vm/trainer.log
```

The training child has a30-minute hard process limit and a28-minute cooperative stop to reserve time for its terminal state. It may stop earlier for non-finite values, GPU/storage bounds, failed fresh-original app parity or a prospective substantial monitoring regression. An occupied GPU stops before training. Snapshots8/16/32 retain metrics/images; they are not alternate recovery checkpoints and cannot be cherry-picked after a failed terminal result.

One terminal state retains all332 trainable tensors, both complete Adam states, ordered parameter names, RNGs, schedule, counters and any partial accumulated gradients. It is published atomically after serialization; a retained .partial is failed-save evidence, not a usable model. A hard kill can prevent recovery. No automatic resume or app adoption is allowed.

The wrapper attempts evidence export after the worker exits, including negative quality status. Export has a900-second cooperative time guard and1536MiB raw/compressed bounds. A complete export is not proof that training improved quality. Expect a substantially larger return than the upload because it contains the terminal training state (about702MB before images and compression).

At the end inspect:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_vm/supervisor.json
cat ~/cctv-completion-codeformer-finetune-v1-export.json
```

Worker exit0 means the predeclared numeric screen passed; exit2 generally means the finite pilot did not qualify numerically, rather than archive creation failed. Other nonzero exits can be execution/resource failures. Retain the exact logs and return in every case. Numeric screening requires a prospective1% relative improvement in paired delivered-PNG hole L1 plus non-worsening raw/PNG hole and seam metrics per geometry; it is not the old restoration structure-gain score, hidden-identity accuracy or full covering-family usefulness. Visual/appearance review and your acceptance remain required.

## 5. Download from Windows Google Cloud SDK Shell

Use one remote source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-finetune-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-finetune-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-finetune-v1-export.json" "."
```

Return all three files to this chat. The checkpoint/Adam/counters, raw outputs, PNGs, metrics and every planned comparison will be independently audited before creating an opt-in local trained-model test. Export or training completion alone does not change the current baseline. Keep the original CodeFormer model and all failed evidence. Goal ACTIVE.
