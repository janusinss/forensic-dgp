# Repaired DGP branch: first manual training stage

The downloaded gradient diagnostic is audited. This new packet trains only the
three repaired original-DGP pieces for **50 updates**, with all**3,905 TRAIN**
cases checked before/after. It has not run. It completes0 full epochs and cannot
qualify restoration; later additional epochs1/2/5 require reviewed evidence.

Require **14GiB free after installation**. Estimated stage**8–25 minutes** plus
export**2–10 minutes**; enforced worker2400s/external2430s, export600s/external630s,
fit300s, each snapshot900s, VRAM20GiB, return6GiB and free-disk reserve1GiB.
The larger export has not been timed. Stop on failed gates or insufficient space.
Original checkpoints, research data/caches and failures remain protected.

The last API check found the existing VM stopped. Start it manually if needed
from **Windows Google Cloud SDK Shell**; this starts billed VM runtime:

```bat
gcloud compute instances start forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a
```

Execution archive:445,352,976 bytes; SHA256 `c7651ee72f14b3ac1b2e44f2671982b880f21b88172a87d4a03561dc8c32c8b4`.
Protocol SHA256:`3d6faf634d45867dc2edd7589aa0b5450790e307342f1897f8ea4b859a646532`. Use only after the independent packet audit passes.
No automatic start/training, failed-run resume, historical rerun or app adoption.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-head4-capacity-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-head4-capacity-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-head4-capacity-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_head4_capacity_vm_v1 &&
tar -xzf cctv-dgp-head4-capacity-v1-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp &&
python3 -c 'import shutil; from pathlib import Path; free=shutil.disk_usage(Path.home()/"forensic-dgp").free; print("Free GiB:",round(free/1024**3,2)); assert free>=14*1024**3,"Stop: need14GiB free after install"'
```

Stop if the disk check fails. No deletion is bundled. Authorized maintenance
requires fresh workload/disk inventory and exact hash-matched retained backups.

3. Open tmux:

```bash
tmux new-session -A -s dgp_head4_capacity_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_head4_capacity_vm_v1 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_head4_capacity_vm_v1.py --root . --protocol-sha 3d6faf634d45867dc2edd7589aa0b5450790e307342f1897f8ea4b859a646532 --verify-transfer &&
python -B -u scripts/supervise_cctv_dgp_head4_capacity_v1.py --root . --protocol-sha 3d6faf634d45867dc2edd7589aa0b5450790e307342f1897f8ea4b859a646532
```

Detach: Ctrl+B, release, D. Reattach:
`tmux attach-session -t dgp_head4_capacity_v1`.
The supervisor prints progress and exports retained failures when possible.
Initial parity, six real gradient checks and measured storage reservation occur
before the optimizer. The1% structure and raw/PNG preservation gates stay fixed.
`early_capacity_pass:true` is a necessary TRAIN result only; native development,
visual review, independent final review and app qualification remain pending.
`complete:true` in the export receipt means the archive finished.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-head4-capacity-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-head4-capacity-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-head4-capacity-v1-export.json" "."
```

Return allthree files, including a failed run. The independent audit checks every
saved raw/PNG record and the full training state; the visual review follows.
Keep originals on the VM until the local full-state backup is verified.
This packet is self-contained apart from the existing environment. No reserved
final images or native CCTV are paired training targets. Allfive milestones and
allseven completion families remain required; the full goal remains incomplete.
