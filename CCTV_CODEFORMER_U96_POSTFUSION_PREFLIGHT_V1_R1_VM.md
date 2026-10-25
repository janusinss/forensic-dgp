# U96 post-fusion preflight R1: manual VM guide

21 October 2026. **Verified preparation; R1 has not run on the VM.**

V1 stopped with `No module named 'psutil'` before constructing a model. All
forward, gradient and optimizer counters were zero. Retain that original stage,
its failure and downloaded files. R1 uses Linux's built-in current/peak memory
readings instead of the missing dependency. It does not change model weights,
inputs, image losses, numerical checks or resource limits.

The new upload is **1,309,337 bytes (about 1.25 MiB)**. It omits
the two large model files. The installer verifies the retained V1 protocol,
zero-call failure and static files, then copies only the exact R2 and U96 files
into the fresh R1 stage. It uses no hardlinks, network download or overwrite.
The old stage must remain at `~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_vm`.

Require about **6.36 GiB free before the weight copy**, leaving at least **6 GiB
free afterward**. The installer measures the exact requirement and stops before
copying if space is insufficient. Current VM space was not queried here.
Do not run the old V1 command or delete its files.

Archive SHA256: `1fb97dbfb4975eb23eaf46c28041b8e88f102f2d95b0a968996c03bc6b5332bd`  
Protocol SHA256: `ffa457e9b7573537fb006c02910cd109af0e17d1b4b7a576502c9f7aa6d14cdf`

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-postfusion-preflight-v1-r1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-postfusion-preflight-v1-r1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-u96-postfusion-preflight-v1-r1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_vm &&
test ! -e ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_r1_vm &&
tar -xzf cctv-codeformer-u96-postfusion-preflight-v1-r1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_r1_vm &&
python3 research/codeformer_u96_postfusion_preflight_v1_r1/install_weights_from_v1.py --root . --protocol-sha ffa457e9b7573537fb006c02910cd109af0e17d1b4b7a576502c9f7aa6d14cdf
```

Expected final output includes `"weights_copied":2` and `"neural_or_optimizer_calls":0`.
If installation stops, retain its terminal output and partial files. Do not
rename, overwrite or delete a prior stage to force another attempt.

## 3. Open tmux in VM SSH

```bash
tmux new-session -A -s u96_postfusion_preflight_r1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_r1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u96_postfusion_preflight_v1_r1.py --root . --protocol-sha ffa457e9b7573537fb006c02910cd109af0e17d1b4b7a576502c9f7aa6d14cdf --verify-transfer &&
bash scripts/run_preflight.sh ffa457e9b7573537fb006c02910cd109af0e17d1b4b7a576502c9f7aa6d14cdf
```

This is still a feasibility diagnostic: **20 full forwards, 4 gradient queries,
0 optimizer constructions/updates**, using four TRAIN photographs with clear and
localized-blur witnesses. It creates no trained successor. U96 remains current.

The supervisor writes detailed progress to `run.log`. In a second VM SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_r1_vm/run.log
```

Press `Ctrl+C` to stop following the log. Detach tmux with `Ctrl+B`, then `D`.
Do not start a second copy. Wait for the final `worker_exit` and `export_exit`.
Export completion means the archive was written; it never means model quality
passed. Download a failed return as well and retain its evidence.

## 5. Download from Windows Google Cloud SDK Shell

Use each single-source command after export finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-preflight-v1-r1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-preflight-v1-r1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-preflight-v1-r1-export.json" "."
```

## Limits and next decision

Runtime is unbenchmarked. Worker limits remain 20 minutes internal, 21 minutes
external, 180 seconds per phase and up to 20 seconds termination grace. Export
limits are 330 seconds internal and 345 seconds external; file verification adds
I/O time. Continuing disk reserve is 4 GiB, current/peak host RSS cap 10 GiB,
CUDA reserved cap 18 GiB, diagnostic outputs 128 MiB and returned archive 160 MiB.
The exact NVIDIA L4 and idle-GPU guards remain. No `psutil` installation is needed.

The full installed local stage passed inert transfer verification. All 44 thin
archive members were independently read back; the two omitted model bindings
are explicit. The actual Linux installer and L4 runtime remain unverified.
Strict forward parity, finite-gradient/projection requirements and the bounded
warning-only gradient policy remain unchanged. RNG equality does not establish
bitwise repeatable CUDA gradients.

After download, the saved-only local auditor verifies the return, arrays,
PNGs, states and counters; all planned image sheets still require review.
A passing preflight does not establish useful learning, actual optimizer-step
preservation, native/FINAL quality or automatic permission to train. A distinct
finite learning pilot can be chosen only from that returned evidence.

[Failure audit and repair](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_PREFLIGHT_V1_FAILURE_REVIEW.md>) ·
[Independent R1 packet audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_preflight_v1_r1_packet/independent_packet_audit_v1.json>).
