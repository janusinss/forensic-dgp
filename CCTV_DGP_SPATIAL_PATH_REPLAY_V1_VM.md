# USED — saved replay imported and audited, 15 October 2026

The manual replay and independent local audit are complete. All sampled
quality gates fail; no new optimizer update or model qualification resulted.
Do not rerun this immutable packet. Historical commands below are retained
for provenance. [Results and next decision](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_PATH_REPLAY_V1_RESULTS.md>).

Exact previous document follows.

# Saved learning-path replay — manual L4 commands

**VERIFIED / PREPARED / UNRUN.** This measures existing A/B reconstruction states;
it performs zero new training updates, gradients or epochs. The original app
checkpoint stays retained. This is the diagnostic selected after both endpoints
failed. The agent has not uploaded or launched it.

Require **8 GiB free after install** and **8 GiB available RAM before models**.
Use the existing L4 VM, `forensic-dgp-thesis`, in `us-central1-a`. Keep the existing
`cctv_dgp_spatial_readout_v1_vm`, `cctv_dgp_bank_photometric_guard_v1_vm` and shared
venv. The thin packet reuses their hash-bound saved states and research assets.
Estimate **15–30 minutes including audit/export**; the new runtime is unmeasured.
Enforced maxima including kill grace: measurement 41 minutes, saved audit
11 minutes, export 11 minutes; hash-only verification adds time separately.
VRAM 20 GiB, RSS 12 GiB, output 3 GiB, retained free reserve 1 GiB.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-spatial-path-replay-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-spatial-path-replay-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-spatial-path-replay-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_spatial_readout_v1_vm/protocol.json &&
test -f ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm/protocol.json &&
test ! -e ~/forensic-dgp/cctv_dgp_spatial_path_replay_v1_vm &&
tar -xzf cctv-dgp-spatial-path-replay-v1-execution.tar.gz -C ~/forensic-dgp
```

Expected checksum: `OK`. The SHA256 sidecar is verified LF UTF-8 without BOM or
carriage returns. If the destination exists, stop and preserve that previous run;
do not delete it or bypass the guard.

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_spatial_path_replay_v1
```

4. Measure **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_spatial_path_replay_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_spatial_path_replay_v1_vm.py --root . --protocol-sha 6f9a8757466578ef57457994ded7fe9b6b170657f2ee7dd3b8b23f0bd2cc1bc7 --verify-transfer &&
bash scripts/run_spatial_path.sh 6f9a8757466578ef57457994ded7fe9b6b170657f2ee7dd3b8b23f0bd2cc1bc7
```

The hash-only check should report `complete: true`, 10 packet assets, 410 source
assets, 203 parent assets, 14 saved states and zero neural calls/optimizer updates.
The supervised measurement then prints progress for both routes, the saved audit
and export. `Ctrl+B`, then `D`, detaches tmux without stopping it. `Ctrl+C` in the
running worker interrupts it. Do not rerun any historical training worker.

Expected diagnostic completion: `review_exit: 0`, `audit_exit: 0`, `export_exit: 0`
and `optimizer_updates: 0`. Numeric gates can still fail. Export `complete: true`
confirms packaging only; it is not training or model-quality success. If any stage
fails, preserve the printed traceback and all export files for independent audit.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-path-replay-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-path-replay-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-spatial-path-replay-v1-export.json" "."
```

This avoids the Windows PuTTY multiple-remote-sources error. Keep all three files
in the named local `outputs` directory. Do not delete the VM research assets while
the run or transfer is active. No cleanup is included in these commands.

Execution archive SHA256: `228e7bb9c1a957021c86be471f0eea1f3f0a5c38ada99dedf4fb69756b8c7f88`; **85,745 bytes**.
Protocol SHA256: `6f9a8757466578ef57457994ded7fe9b6b170657f2ee7dd3b8b23f0bd2cc1bc7`.

After return: verify the archive/manifest, independently check all saved outputs,
all 17 RAW/PNG groups and exact 0/50 controls, then directly inspect every one of
the 40 planned sheets. This packet does not qualify native CCTV, final identities,
completion families or a replacement app checkpoint.

[Verified preparation and limits](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_PATH_REPLAY_V1_PREPARATION.md>).
[Why 0.87% and 0.13% are rejected experiments](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_EXPERIMENT_PROGRESS_20261014.md>).
