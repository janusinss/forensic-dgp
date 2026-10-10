# V41 spatial DGP PCGrad pilot: five manual steps

The audited V40 endpoint diagnostic finds conflicting improvement/preservation
gradients. V41 tests one learning change: PCGrad combines the same seven weighted
component gradients before the same AdamW update. No loss weights or preservation
thresholds change. Fixed random order seed20261008 is declared before execution.
The same original spatial decoder seed,781 TRAIN references/3,905 cases and
paired800-batch schedule remain. Original DGP, initial reference and recognizer
stay frozen. No pretrained restoration targets or primary substitution are used.

Maximum800 updates (one epoch plus19 batches), with the unchanged1% structure
requirement at50 and10% at800. All17 preservation groups, both-source nonregression
and20% mean-only fraction remain. Failure at50 stops and exports every failed gate;
there is no resume, rerun unchanged, sweep, automatic launch or app promotion.
Each update retains seven raw gradient arrays, projected/clipped gradients,
before/after parameters, AdamW moments, scalar losses and input IDs for independent
readback. Gradients and actual training occur only on the existing L4 VM.

Require the existing **NVIDIA L4/g2-standard-4**, existing venv and **6GiB free
after installation**. Packet:444,567,499 bytes. Estimate **20–50 minutes** training
and **3–15 minutes** export; these are V41 estimates, not measured results.
Cache900s, fit3600s, worker4500s/external4800s+30s grace, export900s/external930s+30s,
allocated VRAM20GiB and uncompressed return3GiB are enforced. Update20 timing
projection stops before exceeding the fit budget. The pilot performs no cleanup.

Protocol SHA256: `46dbeab9515719fdd5571cdbfdf5e52e6673bc78e44ccff84bd2f3f15e69c56e`
Execution archive SHA256: `0f1cb2e8b6d11dade6fb0483b4ba1f285ed2c926bc0bf9f0442763b73bae0d53`

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-pcgrad-fit-v41-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-pcgrad-fit-v41-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-pcgrad-fit-v41-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_pcgrad_fit_vm_v41 &&
tar -xzf cctv-dgp-pcgrad-fit-v41-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_pcgrad_fit_v41
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_pcgrad_fit_vm_v41 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_pcgrad_fit_v41_vm.py --root . --protocol-sha 46dbeab9515719fdd5571cdbfdf5e52e6673bc78e44ccff84bd2f3f15e69c56e --verify-transfer &&
bash scripts/run_v41.sh 46dbeab9515719fdd5571cdbfdf5e52e6673bc78e44ccff84bd2f3f15e69c56e
```

Detach safely with **Ctrl+B**, release, then **D**. Keep the VM running while
training/export is active. Trainer exit1 means retain the stop; complete:true
on the export receipt means packaging completed, not restoration qualification.

5. Download in **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-pcgrad-fit-v41-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-pcgrad-fit-v41-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-pcgrad-fit-v41-export.json" "."
```

The prospective local auditor uses `venv\Scripts\python.exe`, frozen inference
and saved-array arithmetic only. Return arrays are never independently
differentiated locally. Parameter arithmetic tolerance3e-7 is for float32
CPU/CUDA reduction readback. Combined-gradient arithmetic allows two float32
ULPs plus1e-12 for reduction rounding; parameter chains and snapshot tensors
stay exact. All image/gate thresholds remain unchanged.
Useful native output, covering-family quality and independent final review
remain required. No real Zamboanga samples exist; source labels do not establish
ethnicity. Reserved final pixels remain unopened. Goal active/incomplete.
