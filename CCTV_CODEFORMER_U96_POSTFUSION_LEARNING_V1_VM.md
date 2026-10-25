# U96 post-fusion learning V1: five manual VM steps

Prepared and unrun. This is a distinct image-learning trial from complete U96. It trains four new correction heads containing36 tensors, with the539 incumbent tensors frozen. Their final output projections start at zero, preserving the initial U96 output. U96 remains the selected model until returned evidence and visible outputs justify a replacement.

Upload size: **38,086,503 bytes**. Execution SHA256: `1adc830de1444e4fd1c51e9172e6df11f3738f78b59bf2a2a4b2ec051adff248`. Protocol SHA256: `11e3286c64d9de282e46d3fa95ae5c792a402c72a681fba4040d816ae5944903`.

Require **8GiB free after installation**; exclusive weight copying needs about8.36GiB before copying. Estimated training/runtime **15–35minutes**, unbenchmarked for this new worker. Worker hard cap46minutes plus up to20seconds termination grace; separate export cap5minutes45seconds. Input verification and transfers add time. Training stops at64updates or an earlier preservation/resource failure.256 role-case exposures equal four passes over64 optimization cases; this is not64epochs.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-postfusion-learning-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-postfusion-learning-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-u96-postfusion-learning-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_r1_vm &&
test ! -e ~/forensic-dgp/cctv_codeformer_u96_postfusion_learning_v1_vm &&
tar -xzf cctv-codeformer-u96-postfusion-learning-v1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_u96_postfusion_learning_v1_vm &&
python3 -B research/codeformer_u96_postfusion_learning_v1/install_weights_from_r1.py --root . --protocol-sha 11e3286c64d9de282e46d3fa95ae5c792a402c72a681fba4040d816ae5944903
```

The installer exclusively copies the two exact retained R1 initializer files after checking its successful preflight receipt. It preserves the old stage/checkpoints. It creates no optimizer or model. Bytecode is disabled in the command and source. If a required source or storage check fails, preserve the output/error; do not bypass guards or overwrite the stage.

## 3. Open tmux from the normal VM SSH prompt

```bash
tmux new-session -A -s u96_postfusion_learning_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_u96_postfusion_learning_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u96_postfusion_learning_v1.py --root . --protocol-sha 11e3286c64d9de282e46d3fa95ae5c792a402c72a681fba4040d816ae5944903 --verify-transfer &&
bash scripts/run_learning.sh 11e3286c64d9de282e46d3fa95ae5c792a402c72a681fba4040d816ae5944903
```

The supervisor prints the child PID and `run.log` path. To detach safely, press **Ctrl+B**, then **D**. Reattach with `tmux attach -t u96_postfusion_learning_v1`. In another VM SSH window, view progress with:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_u96_postfusion_learning_v1_vm/run.log
```

Exit0 means the finite TRAIN screen passed. Exit2 means a preservation/quality stop; retain it. Other nonzero codes indicate an infrastructure/integrity stop. Successful export still does not establish an improved or qualified model. Every committed update retains head/Adam/RNG/sampler state. A timeout can leave an explicitly unknown optimizer outcome; the export preserves that distinction. Never append updates or rerun this stage automatically.

## 5. Download from Windows Google Cloud SDK Shell

After export completes, use one remote source per command for Windows PuTTY compatibility:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-learning-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-learning-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-learning-v1-export.json" "."
```

The next local action is independent archive/state/RAW/PNG/gate auditing and review of all planned sheets. A side-by-side experimental app comparison follows the returned evidence; no automatic default-model replacement is enabled. Keep the existing U96/R2/original benchmarks and all failures.
