# V25 fixed-state diagnostic — upload, tmux and download

6 October 2026. **Run the diagnostic below; keep the failed V25 training run.**
This15 KB transfer measures saved-state gradients with **zero weight updates**.
Keep the existing ~/forensic-dgp/cctv_dgp_spatial_features_vm_v25 folder intact.
Expected external run limit8 minutes plus30s termination grace; export1 minute
plus10s grace. Data, models, venv and250 saved features are reused.

Protocol SHA256: `884412f5e3dab571e630cdadb4f962046c8da309ebe6b62eac4d5af0e29d281a`.
Archive SHA256: `669626a5cbf480edb38b0c6787dcd59da6ae8d935a65789011e3102f0f45d5c5`.

1. Upload from **Windows Google Cloud SDK Shell**:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-v25-gradient-diagnostic-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-v25-gradient-diagnostic-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-v25-gradient-diagnostic-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -d ~/forensic-dgp/cctv_dgp_spatial_features_vm_v25/outputs/frozen_DGP_features &&
test ! -e ~/forensic-dgp/cctv_dgp_v25_gradient_diagnostic_v1_vm &&
tar -xzf cctv-dgp-v25-gradient-diagnostic-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_v25_gradient_diagnostic_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_v25_gradient_diagnostic_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py --root . --protocol-sha 884412f5e3dab571e630cdadb4f962046c8da309ebe6b62eac4d5af0e29d281a --verify-transfer &&
bash scripts/run_gradient.sh 884412f5e3dab571e630cdadb4f962046c8da309ebe6b62eac4d5af0e29d281a
```

5. Download **after the export receipt appears**, from **Windows Google Cloud SDK Shell**:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v25-gradient-diagnostic-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v25-gradient-diagnostic-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v25-gradient-diagnostic-v1-export.json" "."
```

Downloads go to **C:\xampp\htdocs\YEAR 4\Testing\outputs**. Keep three separate
download calls because Windows PuTTY rejects multiple remote sources.

`Ctrl+B`, then `D` detaches safely. Step3 reattaches. Successful measurement prints
both snapshot0 and50 gradient summaries, Diagnostic exit code:0, followed by an
export receipt. Its complete:true means packaging; it does not accept a model.
Download failure evidence too. The log is
~/forensic-dgp/cctv_dgp_v25_gradient_diagnostic_v1_vm/trainer.log.
If a guard or timing limit stops the diagnostic, preserve/paste that error; do
not remove the failed folder or repeat it unchanged. No follow-on training launches.

Returned matrices and receipts require independent audit before choosing any new
training recipe. Original V25's1% early failure remains closed. Goal active/incomplete.

[Frozen diagnostic design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_PLAN.md>) ·
[Audited V25 result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FEATURES_V25_RESULTS.md>)
