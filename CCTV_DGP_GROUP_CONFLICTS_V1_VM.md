# Group conflicts before more epochs: manual VM commands

**Prepared and verified, not run.** This measures source/profile loss conflicts
before training. There are **160 gradient queries, zero optimizer updates and
zero epochs**. If all 32 sampled losses have a certified common descent direction,
three independent reset trials test it against the unchanged preservation gates.
Otherwise the diagnostic records the conflicts and exports without trials.
Neither outcome qualifies a model or starts follow-on training.

Estimated diagnostic **8–20 minutes**, export **1–5 minutes**; the broader loss
queries have not been timed on the L4 yet. Enforced worker1500s/external1530s,
cache120s, gradients480s, solver120s, trials300s, export300s/external330s,
30s kill grace,20GiB allocated VRAM and1.5GiB uncompressed return. Require
**4 GiB free after installation**, with512MiB protected reserve and a storage
projection before derivatives. No cleanup is bundled.

Archive: 220,267,830 bytes. Protocol SHA256: `bc0dcbd92b876c53ebe4f484cc5743fad7c8e7cdce6ca8222e04de3d485ec4b2`.
Execution SHA256: `cfe829808d9280744d45633bcdfb9bcdd0dbc6a70549d6f86d0a73acf2f96ff5`.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-group-conflicts-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-group-conflicts-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-group-conflicts-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_group_conflicts_v1_vm &&
tar -xzf cctv-dgp-group-conflicts-v1-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

3. Open tmux:

```bash
tmux new-session -A -s dgp_group_conflicts_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_group_conflicts_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_group_conflicts_v1_vm.py --root . --protocol-sha bc0dcbd92b876c53ebe4f484cc5743fad7c8e7cdce6ca8222e04de3d485ec4b2 --verify-transfer &&
bash scripts/run_group_conflicts.sh bc0dcbd92b876c53ebe4f484cc5743fad7c8e7cdce6ca8222e04de3d485ec4b2
```

Detach with Ctrl+B, release, D; reattach with
`tmux attach-session -t dgp_group_conflicts_v1`. Use a fresh directory. The worker
refuses competing GPU processes, a different VM, prior outputs, resume and
automatic promotion. Every timing/storage/integrity failure is retained.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-group-conflicts-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-group-conflicts-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-group-conflicts-v1-export.json" "."
```

All three return files require independent audit and visual review. The 100
photographic TRAIN cases do not establish native CCTV, DEV or final performance.
Read CCTV_DGP_GROUP_CONFLICTS_V1_DESIGN.md for surrogate and derivative-audit
limits. The DGP-led app, all five milestones and seven covering families remain
active/incomplete. Changing VMs requires updating and reverifying the host guard
before a new packet; do not edit this immutable packet on the VM.
