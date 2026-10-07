# V31 command-only import correction

7 October 2026. This supersedes only step4 of
`CCTV_DGP_PROFILE_BATCHES_V31_VM.md`. The original guide, execution archive,
protocol and failed terminal evidence remain unchanged.

Location: `paired_batch_and_closed_check`, line137 of the frozen VM worker.
Cause: launching a file in `scripts/` puts that directory on Python's import path;
the new schedule helper is in the packet root. The transfer check therefore raised
`ModuleNotFoundError: No module named 'cctv_dgp_profile_batches_v31_schedule'`.
The previous packet audit did not check this initial script import path.
Fix: export the packet root in `PYTHONPATH` for the transfer check and its Bash
child. This changes launch configuration, not model code, losses, training data,
optimizer, update limits or acceptance thresholds.

The read-only maintenance snapshot at 08:59:44 UTC found two copies of this traceback
in the V31 tmux pane, no V31 process, no trainer log, no outputs and an idle GPU.
The exact corrected transfer check then passed on the existing VM in 2.77 seconds:
9 packet files, 246 original assets, 3,905 training cases, 781 references and zero
neural/gradient or optimizer calls. No training was started by the agent. The
initial local sandbox credential-access failure and successful maintenance retry
are retained separately; neither launches training.

Protocol SHA256:
`ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e`.
Execution archive SHA256:
`bdf79346b10376b75328dfd2fab3ab3902935982cb9910b4478b401c0c55b9b8`.
There is no new transfer file. Retain the original finite bounds and gates:
800 maximum updates, 1% minimum structure gain at 50, final 10%, all preservation
groups and source requirements, preflight 300s/cache 900s/fit 3600s/worker 4500s,
external 4800s +30s, export 900s/external 930s +30s. Do not rerun if a new training
output or trainer log is already present; preserve its receipt instead.

1. Launch in the existing **V31 tmux pane**:

```bash
cd ~/forensic-dgp/cctv_dgp_profile_batches_vm_v31 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" &&
python -B -u scripts/cctv_dgp_profile_batches_v31_vm.py --root . --protocol-sha ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e --verify-transfer &&
bash scripts/run_v31.sh ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e
```

2. Check the first printed JSON: `complete:true`, `training_cases:3905`,
   `training_references:781`, `optimizer_updates:0`. This confirms transfer
   verification only. Subsequent gradient/cache/snapshot/update output belongs to
   the finite pilot. Expected runtime remains 15-35 minutes plus export.

3. Detach with **Ctrl+B**, release both keys, then **D**.

4. Monitor from a second **VM SSH** window:

```bash
tail -F ~/forensic-dgp/cctv_dgp_profile_batches_vm_v31/trainer.log
```

5. After export finishes, download from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-profile-batches-v31-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-profile-batches-v31-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-profile-batches-v31-export.json" "."
```

Three separate download calls avoid PuTTY's multiple-remote-source limitation.
Export `complete:true` is packaging status, not training success or useful
restoration. Preserve any stop and its failure records. Local independent return
audit and visible development review remain required; no app promotion follows
automatically. No reserved final data, native CCTV images or human identity claim
is involved in this maintenance check. Goal active/incomplete.

Evidence: `outputs/cctv_dgp_v31_launch_status_v1/` (failed local transport),
`outputs/cctv_dgp_v31_launch_status_v1_r1/` (read-only VM status), and
`outputs/cctv_dgp_v31_launch_import_v1/` (corrected read-only transfer check).
