# V33 finite update probe: five manual steps

The returned V32 loss diagnostic passed independent audit. At its stopped state,
the summed raw loss-gradient direction increases the facial-detail term in both
measured TRAIN cohorts. Preservation gradients remain necessary for measured
regressions. This probe changes update formation on disposable model copies.

**Eight fresh AdamW proposal steps; zero new gradient queries; no continuing
training trajectory or new checkpoint.** Each original/stopped state and exposed/
unexposed cohort resets completely. The eight steps use saved gradients, not new
backpropagation. They do change disposable parameters, so run this manually on
the existing L4 VM. No agent launch or automatic follow-on training occurs.

It compares the original summed-loss proposal, a restoration-only diagnostic
control, and the restoration proposal constrained by all seven nonzero loss
gradients at four fixed scales:1,1/2,1/4,1/8. AdamW's actual displacement is
projected after its adaptive transformation. The original seven loss values,
17 delivered preservation groups, source and brightness limits remain. The
1% at50 and10% at800 capacity requirements remain for later finite training.
The unconstrained control cannot qualify a training recipe. Linear projection,
finite probe completion and packaging cannot establish useful restoration.

Require an idle **NVIDIA L4/g2-standard-4** with **6GiB free**. Estimate **3-10
minutes for the probe plus1-4 minutes for export**; this new finite probe has not
been timed on the VM. Enforced worker900s; external930s plus30s kill grace;
export300s/external330s plus30s grace; allocated VRAM20GiB; return2GiB.
There are exactly1,200 trial outputs plus200 before outputs. Nonfinite values,
changed dependencies, missing assets, count violations or time/space violations
stop and retain evidence. Existing checkpoints, caches, splits and failures stay.

Protocol SHA256: ba8d1f87cae38cac8e1b3b893378a185c6126c25151dadbcfa0ebb373a51adee
Archive SHA256: 282931a3b265c834a5736ac7f70c0816e08b74a15948d4c94645679a309eeace

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-loss-cone-probe-v33-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-loss-cone-probe-v33-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-loss-cone-probe-v33-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_v32_loss_gradient_v1_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/cctv_dgp_loss_cone_probe_v33_vm &&
tar -xzf cctv-dgp-loss-cone-probe-v33-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_cone_v33
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_loss_cone_probe_v33_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_loss_cone_probe_v33_vm.py --root . --protocol-sha ba8d1f87cae38cac8e1b3b893378a185c6126c25151dadbcfa0ebb373a51adee --verify-transfer &&
bash scripts/run_v33_probe.sh ba8d1f87cae38cac8e1b3b893378a185c6126c25151dadbcfa0ebb373a51adee
```

Detach with **Ctrl+B**, release, then **D**. Step3 reconnects. Expected completed
probe: optimizer_updates:8, committed_trajectory_updates:0, raw_outputs:1400.
Export complete:true means packaging. Download retained failures too. Preserve
the run and its assertions; do not rerun it or launch an800-update continuation.

5. Download from **Windows Google Cloud SDK Shell**, after export completes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-loss-cone-probe-v33-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-loss-cone-probe-v33-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-loss-cone-probe-v33-export.json" "."
```

Each download uses one remote source, as required by Windows PuTTY. Returned
files need independent hash, displacement/moment/projection and output audits
before another training design is justified. This is paired photographic TRAIN
evidence; source labels are not ethnicity, native CCTV or Zamboanga performance.
The application and all seven completion-family requirements remain unchanged.
