# Coordinated original-DGP directions: manual VM commands

Prepared and verified before transfer; not launched. This tests a different
direction from the failed pure-structure probe. It reuses audited saved
derivatives: **zero new gradient queries, zero optimizer updates, zero epochs**.
Nine reset trials combine original decoder structure descent with original
feature identity descent. No failed pilot is resumed. No checkpoint qualifies
until structure, preservation and useful development outputs pass review.

Estimated diagnostic **5–10 minutes**, export **1–5 minutes**, based on the prior
run. Require **4 GiB free after installation**. Worker1200s/external1230s,
trials600s, cache120s, export300s/external330s,30s kill grace,20GiB allocated
VRAM,1.75GiB uncompressed return and512MiB reserve are enforced. No cleanup
is bundled. All inputs/code/weights/gradient evidence are packaged separately
from the prior immutable folders. Keep the research-cache migration backup.

Archive: **245,615,832 bytes**.
Protocol SHA256: `8e28ab1de4873bca2aff806a21ae173bc039c9463d5b2acf04848b031791d2d2`
Execution SHA256: `da08ec66d881495efbcfbfbd794ab179df8a1fbff69342b96b3f1e4dd582bede`

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-original-loss-balance-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-original-loss-balance-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-original-loss-balance-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_original_loss_balance_v1_vm &&
tar -xzf cctv-dgp-original-loss-balance-v1-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

3. Open tmux:

```bash
tmux new-session -A -s dgp_original_loss_balance_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_original_loss_balance_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_original_loss_balance_v1_vm.py --root . --protocol-sha 8e28ab1de4873bca2aff806a21ae173bc039c9463d5b2acf04848b031791d2d2 --verify-transfer &&
bash scripts/run_loss_balance.sh 8e28ab1de4873bca2aff806a21ae173bc039c9463d5b2acf04848b031791d2d2
```

Detach with Ctrl+B, release, D. Reattach with
`tmux attach-session -t dgp_original_loss_balance_v1`.
The script checks idle L4/g2-standard-4 and stops on timing/storage/VRAM or
integrity problems. A diagnostic can finish while every trial fails quality;
the archive retains every failure. It starts no follow-on training.

5. Download from **Windows Google Cloud SDK Shell**, one source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-original-loss-balance-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-original-loss-balance-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-original-loss-balance-v1-export.json" "."
```

Return all three files for independent audit and whole-face visual review.
The same1% structure and appearance requirements remain in force. Both cohorts
are photographic TRAIN diagnostics, not final identities or native CCTV quality
evidence. All five restoration milestones and seven covering families remain
required; the full goal is active/incomplete.
