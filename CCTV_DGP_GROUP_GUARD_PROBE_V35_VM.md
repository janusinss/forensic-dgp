# V35 finite image probe: five manual steps

V34's independent audit passed:300 saved gradient queries, zero parameter
updates,230 files verified and280 frozen CPU outputs replayed. Original raw
outputs match V33 exactly. Its raw preservation derivatives expose12 harmful
checks per TRAIN subset under the old proposal, all in fixed ArcFace.

V35 tests a changed direction: project the mean of the two audited original
AdamW displacements against all102 non-hinged source/profile preservation
rows plus all6 nonzero existing restoration-loss rows. Independent group
reassembly and full-row KKT/primal checks pass. The zero-at-original4 hinges
are retained in output measurements. This mean displacement is not AdamW
applied to a mean gradient. Source labels establish neither ethnicity nor
CCTV/Zamboanga performance; both subsets are now used for TRAIN design.

**Four reset disposable directions at scales1,1/2,1/4,1/8;100 baseline plus
400 trial images. Zero optimizer updates, new gradient queries, committed
training updates, epochs or new checkpoints.** All trials start from the
original DGP. No V32 continuation or historical pilot runs. The original17
PNG group, source, brightness and1%-at50/10%-at800 gates remain. Linear
geometry cannot prove finite PNG preservation or visible structural usefulness.
Every failed check is exported. No automatic training or app replacement.

Require the existing running L4/g2-standard-4, idle GPU and6GiB free. Estimate
2-6minutes for the new probe,1-3minutes export; not yet timed. Worker900s,
external930s plus30s grace; export300s/external330s plus30s grace;20GiB
allocated VRAM;768MiB uncompressed return. No files are deleted.

Protocol SHA256:3c8bf84d50fbeb7705e6ada5d2f386022090834fe96aa292a4642fa4093b4de7
Execution archive SHA256:10f62f7e576881f4570b52ed23763cf8eb6b784ef80cd1f27556e085c98a43e7

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-group-guard-probe-v35-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-group-guard-probe-v35-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-group-guard-probe-v35-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_group_guard_grad_v34_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/cctv_dgp_group_guard_probe_v35_vm &&
tar -xzf cctv-dgp-group-guard-probe-v35-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_probe_v35
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_group_guard_probe_v35_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_group_guard_probe_v35_vm.py --root . --protocol-sha 3c8bf84d50fbeb7705e6ada5d2f386022090834fe96aa292a4642fa4093b4de7 --verify-transfer &&
bash scripts/run_v35_probe.sh 3c8bf84d50fbeb7705e6ada5d2f386022090834fe96aa292a4642fa4093b4de7
```

Detach with **Ctrl+B**, release, then **D**. Step3 reconnects.
Expected probe:raw_outputs:500, candidate_displacement_trials:4,
optimizer_updates:0. Complete:true means diagnostic/export completion; it
does not qualify restoration or training. Download failures too. Do not
edit the frozen protocol, resume a trial, or run follow-on training.

5. Download from **Windows Google Cloud SDK Shell** after export:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-group-guard-probe-v35-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-group-guard-probe-v35-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-group-guard-probe-v35-export.json" "."
```

The returned images require the prospective independent audit and a bounded
actual review before choosing any further pilot. Native CCTV, reserved final
identities, the app and covering-family failures remain separate and unchanged.
