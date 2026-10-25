# Completed return: do not rerun this historical pilot

The 64-update run has completed and its finite TRAIN screen passed. Use the [audited return review](CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_LR_V1_RETURN_REVIEW.md). Frozen local DEV comparison is next; the upload and launch commands below are retained history, not a new action. U96 remains current pending DEV, user and independent FINAL review.

---

# U96 learning-rate comparison V1: five manual VM steps

Prepared and **UNRUN**. The smoother-anchor trial completed64 updates with global PNG structure gain0.919% and localized gain0.665%, below1%. Preserve its quality stop. This distinct comparison starts fresh from complete U96 and changes ONLY maximum Adam learning rate0.00005 to0.0001. SmoothL1 beta0.01, all data,64-update cap and preservation gates stay unchanged. It does not resume a failed checkpoint or add epochs. U96 remains current.

Upload **38,785,747 bytes**. Archive SHA256 `f1a56a6692dadb7b256e4659d5032344db5b6edb70fa2c7900fb42be44631821`. Protocol SHA256 `8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9`.

Require **8GiB free after installation**, about8.36GiB before copying weights. Previous B64 worker took about7minutes; plan10–20minutes for this new unbenchmarked learning rate, excluding transfer/verification. Hard worker cap46minutes plus20seconds termination grace; separate export cap5minutes45seconds. Maximum64 updates/256 role exposures, four passes over64 optimization cases, not64 epochs. Preservation failures stop earlier.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-postfusion-smoothanchors-lr-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-postfusion-smoothanchors-lr-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-u96-postfusion-smoothanchors-lr-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_r1_vm &&
test ! -e ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm &&
tar -xzf cctv-codeformer-u96-postfusion-smoothanchors-lr-v1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm &&
python3 -B research/codeformer_u96_postfusion_smoothanchors_lr_v1/install_weights_from_r1.py --root . --protocol-sha 8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9
```

The installer validates the successful R1 preflight and exclusively copies its exactR2/U96 initializer files. No new model or optimizer is constructed. It preserves all previous sources and checkpoints. Keep `-B`; it prevents the earlier bytecode-directory installation failure. If a source/storage guard fails, preserve the message and partial stage; do not bypass or overwrite it.

## 3. Open tmux from the normal VM SSH prompt

```bash
tmux new-session -A -s u96_smoothanchors_lr_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u96_postfusion_smoothanchors_lr_v1.py --root . --protocol-sha 8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9 --verify-transfer &&
bash scripts/run_learning.sh 8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9
```

The supervisor prints a child PID and writes training output into `run.log`. In another VM SSH window, monitor with:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/run.log
```

Detach using **Ctrl+B**, then **D**. Reattach with `tmux attach -t u96_smoothanchors_lr_v1`. Exit0 passes only the finite TRAIN screen. Exit2 retains a quality stop; other nonzero exits indicate integrity/infrastructure/resource failure. Export success is separate from model success. Each committed head/Adam/RNG/sampler state remains saved; an interruption can retain an explicitly unknown optimizer outcome. Do not append updates, rerun the started stage or promote automatically.

## 5. Download from Windows Google Cloud SDK Shell

After export completes, download each remote file separately for PuTTY compatibility:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-smoothanchors-lr-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-smoothanchors-lr-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-smoothanchors-lr-v1-export.json" "."
```

Independent archive/RAW/PNG/optimizer/gate auditing and all planned visual sheets precede any app/model decision. The same8 monitor photographs are reused adaptive TRAIN evidence after both preceding trials, not unseen validation. A higher learning rate can break preservation; the unchanged checks stop that run. No native, FINAL or new app qualification is claimed. New actual training remains your manual workflow.
