# Finite preservation checks before another epoch run

Prepared for the existing L4 VM; use only after the independent packet audit
in outputs/cctv_dgp_finite_guard_v1_r1_preparation/ is complete. Not launched.
This separate mechanics study recomputes64 cohort/source/profile gradients.
It tests at most9 placements and accepts at most3 parameter changes. Accepted
changes ARE training, even though no torch optimizer is constructed. Both sampled TRAIN cohorts (50 cases each; 100 total)
now guide fitting; neither is independent DEV/final.

The original1% quality requirement is reported at every placement. A small
accepted mechanics change is not a qualified model or completed epoch. Three
rejected proposals or a failed direction certificate stop and export. No resume,
automatic next study, historical-pilot launch or app promotion is permitted.

Estimated run **10â€“20 minutes**, export **1â€“6 minutes**, extrapolated from
the preceding L4 diagnostic. New64-group runtime/compression is unmeasured.
Enforced: worker1800s/external1830s; cache120s; per state gradients240s,
solver120s, proposals180s; export360s/external390s;30s termination grace;
20GiB allocated VRAM;3GiB uncompressed retained return;512MiB reserve.
Require **7GiB free after installation**, including space for the return archive.
Observed prior-size projection is 3,188,317,504 bytes; live projection and
conservative per-file storage stops still apply. No deletion is bundled.

Packet: 190,367,690 bytes. Protocol: `efdd62759213136114a56f0aaa256278753cac8bac4bc6c6a52f1f8b7550598f`.
Execution SHA256: `4a620dd41d96cedfef25b0ed7bbd82ac362f0e867c3ab4ca656642bb1c74870c`.

The VM was stopped when inspected. Start the existing instance manually in
Google Cloud before uploading. To start it from **Windows Google Cloud SDK Shell**:

```bat
gcloud compute instances start forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a
```

This command starts billed VM runtime. It has not been executed by Codex.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-finite-guard-v1-r1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-finite-guard-v1-r1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-finite-guard-v1-r1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_finite_guard_v1_r1_vm &&
tar -xzf cctv-dgp-finite-guard-v1-r1-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

If there is less than7GiB free, stop here. Storage maintenance needs a new live
inventory and exact backup/hash-bound candidates. Preserve research assets,
all original checkpoints, splits, gate failures and active work.

3. Open tmux:

```bash
tmux new-session -A -s dgp_finite_guard_v1_r1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_finite_guard_v1_r1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_finite_guard_v1_r1_vm.py --root . --protocol-sha efdd62759213136114a56f0aaa256278753cac8bac4bc6c6a52f1f8b7550598f --verify-transfer &&
bash scripts/run_finite_guard.sh efdd62759213136114a56f0aaa256278753cac8bac4bc6c6a52f1f8b7550598f
```

Detach with Ctrl+B, release, D. Reattach:
`tmux attach-session -t dgp_finite_guard_v1_r1`.
An export receipt with `complete: true` only confirms the archive was created.
It includes the actual number of accepted changes and preserves worker failures.
A zero exit code can also mean all proposals were rejected cleanly.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-finite-guard-v1-r1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-finite-guard-v1-r1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-finite-guard-v1-r1-export.json" "."
```

Return all three files for independent saved-vector/decision/CPU-output audit
and full visual review before any further training decision. Individual autograd
queries are not independently replayed locally. All native CCTV/paired DEV,
independent final, covering-family and full application requirements remain.
The full goal remains active/incomplete. See CCTV_DGP_FINITE_GUARD_V1_R1_DESIGN.md.
