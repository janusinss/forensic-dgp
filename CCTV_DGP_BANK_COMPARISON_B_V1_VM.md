# Superseded B r2 commands - 11 October 2026

B r2 has run and stopped before training with `Fresh initializer differs`.
Its downloaded failure is independently audited and retained. Use the
[five manual transfer, tmux and download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_COMPARISON_B_V1_R3_VM.md>) for the new B r3 root. Do not rerun the historical commands
below or resume its failed state. Latest evidence: [B r2 return review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_B_R2_RETURN_REVIEW.md>).

---

# Bank comparison: manual route B

Verified maintenance snapshot 2026-10-10T17:56:46.712372+00:00: A independent return/visual audit
and exact backup-bound cleanup closed. VM free space 18.129 GiB;
conservative B post-install free space 16.523 GiB. B is ready for the
user's manual transfer/preflight at this snapshot; runtime resources are checked
again by the fixed launch. Full review: [CCTV_DGP_BANK_A_R3_RETURN_REVIEW.md](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_A_R3_RETURN_REVIEW.md>).


Use this separate B root after A's independent audit and the backup-verified
storage cleanup close. Read the latest PROJECT_HANDOFF.md status before launch.
The transfer packet is already frozen; no automatic run occurs.

B starts from its declared initializer, not A's rejected state. Our learned
input conditioning and 256x256 reconstruction use a frozen external StyleGAN2
face-generating bank. The bank weights are pretrained external components;
they are not our own trained generator or a pretrained restoration output.
This run does not train or qualify covered-region completion.

Match A: **50 updates**, **250 distinct TRAIN references** (125 per source),
**1,250 profile exposures**, **zero complete epochs**. Retain the original
checkpoint, all failure evidence and the unchanged raw/PNG preservation checks.
Extra epochs require a preserved, useful reviewed result; this command never
resumes A or automatically schedules additional training.

Require **16 GiB free after installation**. A's worker took 20.6 minutes and
its export took 4.0 minutes. B's runtime is unmeasured; its preflight measures
and checks projections before training. Enforced worker limit: **90 minutes**;
supervised export: **10.5 minutes**. Fitting limit 10 minutes, snapshot limit
20 minutes, preflight limit five minutes, VRAM limit 20 GiB, output limit 6 GiB
and disk reserve 1 GiB. A stopped for quality, not storage or execution.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-comparison-v1-r2-B-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-comparison-v1-r2-B-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-comparison-v1-r2-B-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_comparison_v1_r2_B_vm &&
tar -xzf cctv-dgp-bank-comparison-v1-r2-B-execution.tar.gz -C ~/forensic-dgp
```

Expected archive SHA256:
`239e461210c7ba5e80127fcdd004d0528e51085647af447fe29c027174283c7a`.
The CR filter handles Windows checksum line endings without replacing the upload.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_comparison_B
```

4. Launch **inside that tmux session**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_comparison_v1_r2_B_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_comparison_v1_vm.py --root . --protocol-sha 68197fd20bef1eae52c4c232164a027e095d567d4bfb37aea79c4f908853ea42 --verify-transfer &&
bash scripts/run_bank_comparison.sh 68197fd20bef1eae52c4c232164a027e095d567d4bfb37aea79c4f908853ea42
```

Detach with **Ctrl+B**, then **D**. Reattach to the same session to view progress.
Do not repeat step 4 in a used or partial root. A quality rejection has trainer
exit code 2 and a saved quality-stop record; export success preserves that
evidence. A resource, parity, gradient or execution stop also stays retained.
Do not change stop thresholds or continue a rejected state.

5. Download after export finishes, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r2-B-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r2-B-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-r2-B-export.json" "."
```

Each SCP download has one remote source for Windows/PuTTY. Return all three
files for independent full-state, raw/PNG, bank-ablation and matched A/B audits
plus all26 planned visual sheets. Native CCTV remains unpaired. Reserved final
identities stay outside tuning. No success, app promotion, further epoch run or
seven-family completion qualification follows from export completion alone.
