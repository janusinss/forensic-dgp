# V38: four disposable PNG-margin trials, five manual steps

V37's diagnostic is independently audited. It made zero training updates and
predicted none of V36's11 PNG failures. V38 tests a different direction with210
checks, retaining all108 earlier checks and adding102 PNG checks with empirical
margins from all four V36 scales. These are TRAIN design measurements, not a
preservation guarantee or independent evaluation. No new gradients, optimizer,
training trajectory or checkpoint. Original models and all failed gates remain.

Require **6 GiB free** on the existing L4 VM. Estimate **3–6 minutes** for the
probe plus **1–3 minutes** for export. Enforced worker900s, external930s with30s
kill grace; export300s/external330s with30s grace; VRAM20GiB. Use the existing
V32-loss, V34, V36 and V37 saved dependencies; do not delete or rerun them.
Stop on any missing/hash-mismatched dependency. The agent has not launched V38.

All17 finite PNG preservation groups, source gains and brightness fraction stay
fixed. The1%-at50 and10%-at800 training requirements remain for a separately
justified future training pilot; this four-trial probe cannot pass them.
Returned outputs need independent audit and all100 comparisons reviewed before
any training decision. V29 development failures, native unpaired criteria and
all seven covering families remain binding. No app promotion or goal completion.

Protocol SHA256: `caea39469005b7066d7cdbd74e073ba674b9b9b6fb9324f3110fe827a06948ab`
Execution archive SHA256: `6096168f41f4be1fbaae9490340c7bdab70140d744b1db3de7194bab90ea38b0`

1. Upload in **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-delivered-margin-probe-v38-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-delivered-margin-probe-v38-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-delivered-margin-probe-v38-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_delivered_guard_grad_v37_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/cctv_dgp_delivered_margin_probe_v38_vm &&
tar -xzf cctv-dgp-delivered-margin-probe-v38-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_delivered_margin_v38
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_delivered_margin_probe_v38_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_delivered_margin_probe_v38_vm.py --root . --protocol-sha caea39469005b7066d7cdbd74e073ba674b9b9b6fb9324f3110fe827a06948ab --verify-transfer &&
bash scripts/run_v38_probe.sh caea39469005b7066d7cdbd74e073ba674b9b9b6fb9324f3110fe827a06948ab
```

Detach with Ctrl+B, then D. Reattach with `tmux attach -t dgp_delivered_margin_v38`.
Inspect `tail -n 25 ~/forensic-dgp/cctv_dgp_delivered_margin_probe_v38_vm/probe.log` if needed. Download after
`Export exit code: 0`; `complete:true` in the export means packaging only.
If the probe fails, retain the failure and download it; do not rerun or resume.

5. Download in **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-delivered-margin-probe-v38-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-delivered-margin-probe-v38-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-delivered-margin-probe-v38-export.json" "."
```

The PuTTY-backed gcloud client accepts one remote source per download command.
All three files are saved in `C:\xampp\htdocs\YEAR 4\Testing\outputs`.
