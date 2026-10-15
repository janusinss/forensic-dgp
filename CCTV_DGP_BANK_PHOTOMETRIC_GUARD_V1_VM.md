# Executed and audited — 11 October 2026: do not rerun this historical recipe

This transfer has now been executed. Its 50-update candidate failed the retained
quality requirements and is not eligible for extra epochs or unchanged retry.
The prepared/unrun wording and launch commands below are historical.
[Audited return and user feedback](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_PHOTOMETRIC_GUARD_V1_RESULTS.md>).
The current next action is the **unrun frozen-inference diagnostic**:
[Exact new manual commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_PATH_ABLATION_V1_VM.md>).
Original guide bytes are preserved as the suffix below and in the publication backup.

---

# Brightness-guard comparison: manual L4 VM commands

**Prepared and independently transfer-verified; training is unrun.** The prior
B run loaded correctly, trained 50 updates and failed quality. This new run adds
one degraded TRAIN brightness-gain loss; it preserves bank features and starts
from the unchanged initializer. It cannot resume rejected B50. No extra epochs
are authorized by archive completion. [Evidence and rationale](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_AB_COMPARISON_V1_RESULTS.md>).

Upload only the **3.58 MiB code transfer**.
The data-only installer copies **5,626 hash-verified assets**
from the retained B r4 root into a distinct folder with **17 overlays**.
It imports no model code, runs no training, deletes no source file and has a
300-second limit. The **814,054,443-byte full execution archive**
stays locally for recovery; these commands use the small transfer.

Current verified free: **16.988 GiB**; conservative post-install: **16.051 GiB**.
Require **16 GiB after installation**; installer and launch recheck. Estimate
25–35 minutes based on B r4's 23.90-minute worker plus export, not a measured
new-run ETA. Finite caps: worker 90 minutes/external 90.5, fit 10, preflight 5,
snapshot 20, export 10/external 10.5, VRAM 20 GiB, output 6 GiB, reserve 1 GiB.
50 updates, 250 distinct TRAIN references, 1,250 profiles, **zero complete epochs**.
Existing 1% structure and preservation/brightness requirements remain unchanged.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-photometric-guard-v1-code-transfer.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-photometric-guard-v1-code-transfer.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-photometric-guard-v1-code-transfer.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/cctv_dgp_bank_photometric_guard_v1_transfer &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
tar -xzf cctv-dgp-bank-photometric-guard-v1-code-transfer.tar.gz -C ~ &&
python3 -B ~/cctv_dgp_bank_photometric_guard_v1_transfer/install_packet.py --protocol-sha f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12
```

Expected transfer SHA256: `2b92e62db55aee71a3eb210d57eca0f90e82a24c2039855134d244cd09b5b864`.
Expected protocol SHA256: `f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12`.
The source B r4 assets and quality stop must remain intact. The installer
does not need its removed completed RGB outputs. Preserve a partial install
for diagnosis; do not rerun a used root or historical recipe.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_photometric_guard_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_comparison_v1_vm.py --root . --protocol-sha f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12 --verify-transfer &&
bash scripts/run_bank_comparison.sh f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12
```

Transfer verification must print `"loader_hash_runtime_checked": true`.
The actual trainer separately checks L4/VM/tmux, source hashes, space, timing,
fixed assets, output parity and gradients before/after update 1. Detach with
**Ctrl+B**, then **D**; reattach using step 3. Wait for supervised export.
Exit 2 means quality stop; `"complete": true` in export means the archive
was produced, not that the model passed. Preserve results from any stop.

5. Download after export, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-photometric-guard-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-photometric-guard-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-photometric-guard-v1-export.json" "."
```

Each download has one remote source for Windows/PuTTY. Return all three for
independent import, full state/exposure/fit-time audits, separate raw/PNG checks,
matched B r4 comparison and every planned visual sheet. Final identities stay
outside tuning. No automatic extra epochs, model adoption or completion claim.
The app's restoration/covering-family scope remains required in the active goal.
