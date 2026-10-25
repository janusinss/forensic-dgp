# U96 smoother-anchor comparison V1: five manual VM steps

Prepared and **UNRUN**. The previous B64 trial completed64 real updates but failed the structure endpoint. Preserve it. This separate trial starts fresh from complete U96, changing only the clear/dark anchor loss to SmoothL1(beta0.01). It does not resume B64 or add epochs. U96 remains current.

Upload **38,239,511 bytes**. Archive SHA256 `c2ea2ecfbdb3b4cf80aea3f3c9b2a95bc75d5c4480cb40fed032a4e3a66ec7ca`. Protocol SHA256 `7bf251dd257d79dfcd8e4bd07fe7674badf4b2f30c15fac10cc048b9a36e778a`.

Require **8GiB free after installation**, about8.36GiB before copying weights. Previous B64 worker took about7minutes; plan10–20minutes for this new unbenchmarked criterion, excluding transfer/verification. Hard worker cap46minutes plus20seconds termination grace; separate export cap5minutes45seconds. Maximum64 updates/256 role exposures, four passes over64 optimization cases, not64 epochs. Preservation failures stop earlier.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-postfusion-smoothanchors-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-postfusion-smoothanchors-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-u96-postfusion-smoothanchors-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_r1_vm &&
test ! -e ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_v1_vm &&
tar -xzf cctv-codeformer-u96-postfusion-smoothanchors-v1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_v1_vm &&
python3 -B research/codeformer_u96_postfusion_smoothanchors_v1/install_weights_from_r1.py --root . --protocol-sha 7bf251dd257d79dfcd8e4bd07fe7674badf4b2f30c15fac10cc048b9a36e778a
```

The installer validates the successful R1 preflight and exclusively copies its exactR2/U96 initializer files. No new model or optimizer is constructed. It preserves all previous sources and checkpoints. Keep `-B`; it prevents the earlier bytecode-directory installation failure. If a source/storage guard fails, preserve the message and partial stage; do not bypass or overwrite it.

## 3. Open tmux from the normal VM SSH prompt

```bash
tmux new-session -A -s u96_smoothanchors_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u96_postfusion_smoothanchors_v1.py --root . --protocol-sha 7bf251dd257d79dfcd8e4bd07fe7674badf4b2f30c15fac10cc048b9a36e778a --verify-transfer &&
bash scripts/run_learning.sh 7bf251dd257d79dfcd8e4bd07fe7674badf4b2f30c15fac10cc048b9a36e778a
```

The supervisor prints a child PID and writes training output into `run.log`. In another VM SSH window, monitor with:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_v1_vm/run.log
```

Detach using **Ctrl+B**, then **D**. Reattach with `tmux attach -t u96_smoothanchors_v1`. Exit0 passes only the finite TRAIN screen. Exit2 retains a quality stop; other nonzero exits indicate integrity/infrastructure/resource failure. Export success is separate from model success. Each committed head/Adam/RNG/sampler state remains saved; an interruption can retain an explicitly unknown optimizer outcome. Do not append updates, rerun the started stage or promote automatically.

## 5. Download from Windows Google Cloud SDK Shell

After export completes, download each remote file separately for PuTTY compatibility:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-smoothanchors-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-smoothanchors-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-smoothanchors-v1-export.json" "."
```

Independent archive/RAW/PNG/optimizer/gate auditing and all planned visual sheets precede any app/model decision. The same8 monitor photographs are reused adaptive TRAIN evidence after B64 feedback, not unseen validation. No native, FINAL or new app qualification is claimed. New actual training remains your manual workflow.
