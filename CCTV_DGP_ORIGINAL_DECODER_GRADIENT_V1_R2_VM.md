# Original-decoder gradient proof V1 R2 — manual VM commands

Run **R2 only**, once. V27 is closed; the unissued V1/R1 drafts remain preserved.
This checks the separate original decoder's differentiation path with **zero
optimizer updates/epochs**. It does not train or establish restoration quality.
The existing `cctv_dgp_feature_skips_vm_v27` directory and virtual environment are
required. Existing files stay read-only. The packet requires 2 GiB free; verified
cleanup last reported 7.48 GiB. No further cleanup or training starts automatically.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-original-decoder-gradient-v1-r2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-original-decoder-gradient-v1-r2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-original-decoder-gradient-v1-r2-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -d ~/forensic-dgp/cctv_dgp_feature_skips_vm_v27/outputs &&
test ! -e ~/forensic-dgp/cctv_dgp_original_decoder_gradient_v1_r2_vm &&
tar -xzf cctv-dgp-original-decoder-gradient-v1-r2-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_original_decoder_gradient_v1_r2
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_original_decoder_gradient_v1_r2_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_original_decoder_gradient_v1_r2_vm.py --root . --protocol-sha 81127e45a205c44ef685646a4f82911d02b2355dfe24c63772c23e62e66a41f2 --verify-transfer &&
bash scripts/run_decoder_gradient.sh 81127e45a205c44ef685646a4f82911d02b2355dfe24c63772c23e62e66a41f2
```

Ten batch messages end at 70 gradient queries. A complete proof has diagnostic
exit code 0, export exit code 0, `run_results_present: true` and
`failure_present: false`. Export `complete: true` alone means packaging completed.
Any failure stays preserved; do not rerun, relax a gate or launch another pilot.
The worker limit is 600 seconds; supervisor 660 seconds plus 30 seconds grace.
Export has a separate 90-second internal/120-second external limit plus 10 seconds
grace. To detach safely, press **Ctrl+B**, release, then **D**.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-original-decoder-gradient-v1-r2-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-original-decoder-gradient-v1-r2-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-original-decoder-gradient-v1-r2-export.json" "."
```

Use three separate downloads: Windows PuTTY rejects multiple remote sources in
one `gcloud compute scp` command. Download failure exports too. The return then
requires independent source, matrix, state, timing and CPU inference readback.
No app promotion or subsequent training is authorized by export completion.

The verified upload archive is **13,816 bytes**, SHA256
`f21634d909e880a26b8ae90e0e6393ff32fc7f2758ebb5312faf59707d82900d`.

[Decoder review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ORIGINAL_DECODER_REVIEW_V1.md>) ·
[Packet verification](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_original_decoder_gradient_v1_r2_preparation/independent_packet_audit.json>)
