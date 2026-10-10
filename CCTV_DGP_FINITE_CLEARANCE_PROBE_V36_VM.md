# V36: four disposable finite-clearance trials, five manual steps

V35's independently audited500 outputs improve detail but all four scales fail
preservation in both TRAIN subsets. Raw outputs also regress; PNG conversion
alone does not explain the failure. All100 case comparisons were actually viewed.

V36 changes the direction. All108 original rows remain;65 raw MSE/ArcFace rows
require twice V35 scale1's measured positive raw/PNG departure from the linear
prediction. SSIM and six restoration rows retain zero minimum clearance because
the saved skimage SSIM differs from the differentiable VM definition. This is
an empirical hypothesis, not a validated bound or a weakened quality gate.
The independent full-row KKT check must pass before using these commands.

Four reset trials at scales1,1/2,1/4,1/8;100 original plus400 trial outputs.
Zero optimizer updates, gradients, backwards, epochs, committed trajectory or
new checkpoint. No V32/V35 continuation or historical pilot. The original DGP
and recognizer stay frozen. Actual17 PNG groups, source gains, brightness and
full-corpus1%-at50/10%-at800 gates remain. TRAIN subsets cannot qualify the app,
independent final faces or native CCTV. Source labels are not ethnicity.

Require idle running NVIDIA L4/g2-standard-4 and6GiB free. V35 took94seconds
plus26seconds export. Estimate2-4minutes plus1-2minutes export for V36; unrun.
Worker900s/external930s+30s grace; export300s/external330s+30s grace;
20GiB allocated VRAM;768MiB uncompressed return;2300 files. Failures are exported.

Protocol SHA256:9af4cbf10d7c141e2cbef2248bc282c135a9cc6117f37b47feaef76d71dadd38
Execution archive SHA256:ee51b5bc9e888dec32e0dfd90c96a19f9e93e21acd1fd4bef3e8500d12f0a67e

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-finite-clearance-probe-v36-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-finite-clearance-probe-v36-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-finite-clearance-probe-v36-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_finite_clearance_probe_v36_vm &&
tar -xzf cctv-dgp-finite-clearance-probe-v36-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_finite_clearance_v36
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_finite_clearance_probe_v36_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_finite_clearance_probe_v36_vm.py --root . --protocol-sha 9af4cbf10d7c141e2cbef2248bc282c135a9cc6117f37b47feaef76d71dadd38 --verify-transfer &&
bash scripts/run_v36_probe.sh 9af4cbf10d7c141e2cbef2248bc282c135a9cc6117f37b47feaef76d71dadd38
```

Detach with **Ctrl+B**, release, **D**. Reattach with
`tmux attach -t dgp_finite_clearance_v36`. Inspect
`tail -n 25 ~/forensic-dgp/cctv_dgp_finite_clearance_probe_v36_vm/probe.log`.
An export `complete:true` records a transfer; it does not imply preservation passed.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-finite-clearance-probe-v36-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-finite-clearance-probe-v36-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-finite-clearance-probe-v36-export.json" "."
```

No VM training or connection is started by packet preparation. Returned files
require independent bounded CPU replay and complete numerical/visual review.
Do not rerun an existing or failed directory. Original checkpoints, splits,
research caches, failures, local backups and the current app remain preserved.
