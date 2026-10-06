# V22 R1 — manual upload, tmux run and download

**Latest status — 6 October 2026: V22 R1 is independently audited and closed.**
The downloaded76,931,848-byte archive matches its checksum. All366 files,
100 raw/PNG pairs/metrics, both50-case snapshots and stopped checkpoint pass
integrity/replay checks. All50 cases are reviewed: no visible structure gain.
The original update50 early stop remains failed; no final800 gate was executed.
The retained failure and diagnostic report remain available.

The commands below are historical V22 procedure. Do not rerun steps1–4 or restart
this unchanged failed recipe. A different verified V23 detail path is now prepared:
[V23 upload/tmux/download commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_VM.md>).
[V22 R1 independent result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md>).
Original pre-update runbook bytes are preserved in
`outputs/dgp_detail_prior_v22_r1_audit_and_skip_v23_milestone/before_docs/`.

Historical status and procedure follow.


**Current status — 5 October 2026:** the user reports successful L4/CUDA preflight
and a stop at update50: feature-error reduction0.0000322991%, below the frozen1%
early requirement. Training exit1 is retained; export exit0 packages the failure.
Expected return SHA256: `4a734450e9ea8fca8fb240a5e06b55f99a28316a770869938b2bd2aa05ad9cad`,
76,931,848 bytes. Returned-file integrity and independent output audit are pending.
Proceed to step5; do not repeat steps1–4 or restart this failed recipe.

Prepared 5 October 2026. This is a **training-only capacity pilot**, not an accepted
app model. Same ten previously exposed training photographs/50 cases; no native
or final/reserved images. New own-DGP detail head,800 updates/80 epochs, batch5.
Preflight cap5 minutes; fitting cap25 minutes; worker cap30 minutes; outer stop35
minutes plus30-second kill grace. Failed and partial outputs are exported.

Verified transfer archive:217,556,126 bytes (207.48MiB),198 regular members.
SHA256: `6978e423c23909caebff65c7299267ce1a6803d15e2818e38bac6b1f685fdaba`.
Frozen protocol: `c431af07b8e0bd2310f67fdc1b9ccfac7118adc08dc6cddd22131af6a5a4ca96`.
The existing VM venv must remain installed. The package includes original DGP/
ArcFace weights and cached training data, so it does not require the old V12,
V18 or V19 directories to remain on the VM. It requires3GiB free disk.
No assistant VM action or local training has occurred. The user-reported CUDA
preflight passed; the downloaded evidence still needs independent verification.

Use **R1** for Python3.10 compatibility: streaming SHA256 replaces the newer
checksum API. Model, data, schedule, objective and training gates are unchanged.
The original unexecuted V22 packet remains archived. R1 verifies196 assets and
198 archive members. The commands below preserve the original execution procedure.

1. Upload from **Windows Google Cloud SDK Shell**:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-detail-prior-v22-r1-execution.tar.gz" "cctv-dgp-detail-prior-v22-r1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-detail-prior-v22-r1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_detail_prior_vm_v22_r1 &&
tar -xzf cctv-dgp-detail-prior-v22-r1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_detail_prior_v22_r1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_detail_prior_vm_v22_r1 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_detail_prior_v22_r1.py --root . --protocol-sha c431af07b8e0bd2310f67fdc1b9ccfac7118adc08dc6cddd22131af6a5a4ca96 --preflight &&
bash scripts/run_v22.sh c431af07b8e0bd2310f67fdc1b9ccfac7118adc08dc6cddd22131af6a5a4ca96
```

5. Download after the **export receipt** appears, from **Windows Google Cloud SDK Shell**:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-detail-prior-v22-r1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-detail-prior-v22-r1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-detail-prior-v22-r1-export.json" "."
```

`Ctrl+B`, then `D` detaches from tmux while the finite supervisor continues.
The tmux command in step3 reattaches. Training progress is displayed and saved
in `~/forensic-dgp/cctv_dgp_detail_prior_vm_v22_r1/trainer.log`.

Export receipt `"complete":true` means packaging completed; the actual trained
head still needs its capacity gates, local independent audit and visual review.
Even a capacity pass is not native CCTV or final/app acceptance. Download the
failure evidence if the early structural/timing/appearance checks stop the pilot.
Do not relaunch the same recipe or delete the partial outputs.

If a command stops before training, preserve its traceback. The guard rejects a
different host/machine/GPU, a competing GPU process, changed files, unavailable
dependencies, insufficient disk or failed parity. It does not reject idle tmux
sessions or terminate other work. No automatic installation or cleanup is used.

Run the three download commands separately. The user's Windows PuTTY backend
rejects multiple remote sources in one command. The upload uses local sources;
only the download command is split. General syntax follows the
[Google Cloud CLI scp reference](https://docs.cloud.google.com/sdk/gcloud/reference/compute/scp).
The original runbook is preserved in
`outputs/cctv_dgp_detail_prior_v22_r1_putty_download_fix_v1/before_docs/`;
the training source, protocol, package and failed gates are unchanged.

After download, the assistant verifies SHA256 before safe extraction and audits
returned sources, timing, budgets, saved metric arithmetic, CPU head/recognizer
replay and all50 training-case images. Keep CodeFormer as a declared comparison
baseline and preserve all original gates. Further training or app integration
requires its own justified frozen stage.
