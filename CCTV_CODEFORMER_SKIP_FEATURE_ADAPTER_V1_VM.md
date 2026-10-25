# Skip-feature adapter V1: five-step manual L4 guide

Prepared and verified, **not trained or accepted yet**. New trainable adapters
start from retained R2. All515 existing R2 tensors stay frozen. This experiment
tests whether learned features can approach part of the clean-reference diagnostic.

Allow **24GiB free after installation**, preserving4GiB. Estimated L4 runtime:
25–45minutes; the changed graph is unmeasured until manual preflight. Worker
work cap80minutes; external worker cap88minutes20seconds, with bounded shutdown.
Export cap10minutes. Stops retain failures and partial states; never rerun or
append epochs unchanged. Use the existing NVIDIA L4 VM and existing venv.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-skip-feature-adapter-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-skip-feature-adapter-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-codeformer-skip-feature-adapter-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_codeformer_skip_feature_adapter_v1_vm &&
tar -xzf cctv-codeformer-skip-feature-adapter-v1-execution.tar.gz -C ~/forensic-dgp
```

Checksum expected: `e422146394fddfab29c39996aa9605048a32dacad8bb430fe12c21f4db3c63cc`. The supplied checksum file uses ASCII/LF.
The new directory check preserves any earlier installation or partial run.

3. Open **tmux**:

```bash
tmux new-session -A -s codeformer_skip_adapter_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_codeformer_skip_feature_adapter_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -m research.codeformer_skip_feature_adapter_v1.worker --root . --protocol-sha 0d358ab29e0344c7243b6312cb1d04ca6cc0e0a1de270bca7e6b4040dd566c90 --verify-transfer &&
bash run_skip_adapter.sh 0d358ab29e0344c7243b6312cb1d04ca6cc0e0a1de270bca7e6b4040dd566c90
```

Verification performs zero neural calls. The manually invoked supervisor then
starts one finite run:110 Adam updates/440 microbatches, capped3,644 forwards
and four gradient queries. It proves identity initialization and image-loss
connectivity before Adam; failed preservation/resource checks retain the stop.
Training or export completion does not qualify the model.

Detach safely with **Ctrl+B**, release, **D**. Reattach with the step3 command.
For another SSH tab, progress is in:
`~/forensic-dgp/cctv_codeformer_skip_feature_adapter_v1_vm/outputs/supervision/worker.log`.

5. Download after export finishes, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-skip-feature-adapter-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-skip-feature-adapter-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-skip-feature-adapter-v1-export.json" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-skip-feature-adapter-v1-supervisor-final.json" "."
```

Each remote source has its own command, matching Windows/PuTTY limitations.
Keep all four returned files. Independent local model-state, saved-image and
visual review are required before changing the app or choosing more training.
If you move to another VM, update its project/name/zone and restore the runtime
before using this guide; this packet is self-contained for model/data sources.
