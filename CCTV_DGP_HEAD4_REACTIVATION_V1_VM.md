# Connected DGP branch check: exact manual commands

This packet is prepared for the existing NVIDIA L4/g2-standard-4 VM, not launched.
It compares original and repaired gradients, with **zero optimizer updates/epochs**.
Our deepest original branch is numerically collapsed. A separate fixed initializer
preserves all100 current CPU outputs exactly. VM gradient usefulness is unverified.
This experiment decides whether the connected path can learn before more epochs.

Expected **4–10 minutes** plus **1–3 minutes** export; worker600s/external630s,
export180s/external210s,30s kill grace. Require **3GiB free after installation**;
20GiB allocated VRAM,768MiB uncompressed return,512MiB disk reserve.
Only160 autograd queries are allowed; no fitting, solver, epoch, automatic follow-on
or app promotion. Every unchanged structure/preservation requirement remains.

Execution archive: 183,353,481 bytes; SHA256 `c9199973d88e3bd0fb49f18cf74b7246c937f063f720936fcdfe6547b092a851`.
Protocol SHA256: `944f1b8d886221717433b73d6eb5a24366ba2058bc3e8c22c1edfa7f0b5d7f81`.
Use only after the independent packet audit in outputs/cctv_dgp_head4_reactivation_v1_preparation/
passes. Source, data roles and original failures are preserved. Do not rerun this
packet over a partial or completed run.

If the existing VM is stopped, start it manually in Google Cloud. Equivalent
**Windows Google Cloud SDK Shell** command (starts billed VM runtime):

```bat
gcloud compute instances start forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a
```

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-head4-reactivation-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-head4-reactivation-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-head4-reactivation-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_head4_reactivation_vm_v1 &&
tar -xzf cctv-dgp-head4-reactivation-v1-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

Stop if free space is below3GiB. No deletion is bundled. Maintenance requires
a fresh inventory and exact retained-backup hashes, preserving all research assets.

3. Open tmux:

```bash
tmux new-session -A -s dgp_head4_reactivation_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_head4_reactivation_vm_v1 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_head4_reactivation_vm_v1.py --root . --protocol-sha 944f1b8d886221717433b73d6eb5a24366ba2058bc3e8c22c1edfa7f0b5d7f81 --verify-transfer &&
python -B -u scripts/supervise_cctv_dgp_head4_reactivation_v1.py --root . --protocol-sha 944f1b8d886221717433b73d6eb5a24366ba2058bc3e8c22c1edfa7f0b5d7f81
```

Detach: Ctrl+B, release, D. Reattach:
`tmux attach-session -t dgp_head4_reactivation_v1`.
The supervisor prints worker output and exports a retained failure when possible.
It does not refuse merely because you are inside the intended tmux session.
`connected_route_pass: true` only means the repaired pieces have finite nonzero
improvement gradients. It is not a structure/preservation/quality pass.
`complete: true` in the export JSON means archive completion only.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-head4-reactivation-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-head4-reactivation-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-head4-reactivation-v1-export.json" "."
```

Return allthree for independent saved-vector/initial-output audit. There is no
autograd or optimizer replay locally. Keep original checkpoints, splits, caches
and gate failures. Additional epochs1/2/5 are a later finite study after the new
path's training capacity and native/paired development preservation justify it.
No final identities or completion families are qualified by this experiment.
The full DGP-led application goal remains active/incomplete.
