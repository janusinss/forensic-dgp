# Decoder activation V1: five manual steps

**VERIFIED / PREPARED / UNRUN. This measures the failed direction; it performs
zero optimizer updates, zero new amplitudes and no direction search.** It uses
the existing L4 VM and retained research directories. Do not resume the failed
hinge-repair optimizer. The permanent app checkpoint stays unchanged.

Require at least **3 GiB free**, **8 GiB available RAM**, the existing venv and
an idle L4. Estimated total runtime is **5–15 minutes**, subject to measured
projection stops. Worker cap:1200s; saved audit/export:300s each. External
caps:1230/330/330s, with30s kill grace. VRAM20GiB, RSS12GiB, output768MiB and
disk reserve1GiB are enforced; model loading is included in peak accounting.

1. Upload from **Windows Google Cloud SDK Shell (cmd.exe)**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-decoder-activation-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-decoder-activation-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-decoder-activation-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_decoder_activation_v1_vm &&
tar -xzf cctv-dgp-decoder-activation-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_decoder_activation_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_decoder_activation_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_decoder_activation_v1_vm.py --root . --protocol-sha b93c7a8ce3ab730f1d20f8ebc8786db18469e8652db26554cf0b116dc5785481 --verify-transfer &&
bash scripts/run_activation.sh b93c7a8ce3ab730f1d20f8ebc8786db18469e8652db26554cf0b116dc5785481
```

5. Download from **Windows Google Cloud SDK Shell** after export:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-decoder-activation-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-decoder-activation-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-decoder-activation-v1-export.json" "."
```

Each SCP command has one remote source, as required by Windows PuTTY. `Ctrl+B`,
then `D` detaches tmux safely. Export completion means evidence was packaged;
it cannot qualify the model. Keep failed/partial files if any stop occurs.

Expected successful measurement:400 exact retained RAW/PNG controls,80 fresh
batch-five objective records,70 gradient queries, zero optimizer/backward calls
and exact initial state/RNG restoration. Quality remains unqualified even if
all these measurements succeed. A killed partial run may have an unknown exact
query count; the export explicitly preserves that uncertainty.

Execution archive:395,607bytes. SHA256:`297cadcafeb839cea08fff89361716bb86042969cfa4c2d2e3660f9d1e2078fb`.
Protocol SHA256:`b93c7a8ce3ab730f1d20f8ebc8786db18469e8652db26554cf0b116dc5785481`.

[Independent preparation and scope](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DECODER_ACTIVATION_V1_PREPARATION.md>).
After download, the agent uses the source-pinned local importer:
[Local import and saved-data audit](<C:/xampp/htdocs/YEAR 4/Testing/scripts/import_cctv_dgp_decoder_activation_v1_return.py>).
No VM execution or VM connection was performed by the agent.
