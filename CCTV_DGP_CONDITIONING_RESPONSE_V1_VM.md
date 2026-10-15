# USED — diagnostic completed and independently audited, 14 October 2026

This packet has already run. Retain its outputs and do not repeat the historical
launch commands below. Zero optimizer updates or training epochs; no model
qualified. [Current result and retained limits](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_RESPONSE_V1_RESULTS.md>).
The next input-response diagnostic is a design review, with no verified transfer
packet yet. Exact previous preparation and commands follow as history.

# Manual conditioning-response diagnostic V1 — 14 October 2026

**Verified and UNRUN.** This measures why predicted learning differs from real
image changes. It accepts no training updates and selects no checkpoint.
The previous RAW R1 pilot remains failed. New neural work stays human-manual
inside tmux on the existing NVIDIA L4 VM.

Require **9 GiB free after installation**. Estimated diagnostic **6–12 minutes**,
or **8–15 minutes including export**; those are estimates from prior measured
work. External stop limits total **29.5 minutes including both termination
graces**. Storage, VRAM, RSS, projection and output stops also apply.
Packet: **10,290,328 bytes**, SHA256 `5f272c22ab735ce9a293633d7a4ec595bdfd77835f02cb64fece9976c0e5afd8`.
Protocol: `e72d76e3f2330420ee93b43cf528d0b49da132f356a30c336a1ec578c8f82f44`.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-conditioning-response-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-conditioning-response-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-conditioning-response-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_conditioning_response_v1_vm &&
tar -xzf cctv-dgp-conditioning-response-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_conditioning_response_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_conditioning_response_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_conditioning_response_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha e72d76e3f2330420ee93b43cf528d0b49da132f356a30c336a1ec578c8f82f44 --verify-transfer &&
bash scripts/run_response.sh e72d76e3f2330420ee93b43cf528d0b49da132f356a30c336a1ec578c8f82f44
```

The first receipt is hash-only: 100 cases, maximum640 derivative queries,
six temporary displacements, zero optimizer updates and zero neural calls in
that receipt. The wrapper writes progress into `review.log`; its full log prints
when the diagnostic ends. In a second VM SSH window, monitor with
`tail -F ~/forensic-dgp/cctv_dgp_conditioning_response_v1_vm/review.log ~/forensic-dgp/cctv_dgp_conditioning_response_v1_vm/export.log`.
Stop that monitor with Ctrl+C. Detach tmux with Ctrl+B, release, then D.
Do not rerun a used directory or remove a failure to get past the guard.

5. Download from **Windows Google Cloud SDK Shell**, after export finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-response-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-response-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-conditioning-response-v1-export.json" "."
```

Each download has one remote source, as required by PuTTY-backed gcloud SCP.
Keep all three files together in `outputs`. Return them for the independently
prepared audit. `Diagnostic exit code: 0` means the measurements finished;
`complete: true` in export means the archive exists. Neither is a model-quality
or epoch-training success. All complete outputs and all planned sheets need
review before any next learning decision.

Original checkpoints, app design, splits, provenance, caches, full states and
failed gates remain. This is photographic paired TRAIN evidence only. Native
CCTV, final identities and seven automatic/assisted completion families remain
separate and unqualified by this diagnostic.
[Design, evidence and exact limits](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_CONDITIONING_RESPONSE_V1_PREPARATION.md>).
