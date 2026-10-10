# V34 preservation-gradient diagnostic: five manual steps

V33 passed its independent return audit but failed preservation qualification.
This diagnostic measures the protected functions before their zero-at-baseline
hinges. It covers the same two 50-case photographic TRAIN subsets, original
DGP state only, same batch context and 23 selected tensors. Source labels do
not establish ethnicity, native CCTV performance or Zamboanga performance.

**300 gradient queries; zero optimizer updates, parameter updates or new
checkpoints.** No V32 continuation, V33 repeat or historical worker is launched.
All 17 group checks, original loss weights, 1% at50 and10% at800 requirements,
source/brightness limits and the current app remain unchanged. Measured raw
derivatives cannot substitute for finite PNG preservation or useful structure.

Use the existing, running **NVIDIA L4/g2-standard-4** with an idle GPU and
**6GiB free**. Estimated diagnostic **1–6 minutes**, export **1–3 minutes**,
based on prior runs; V34 has not been timed. Worker600s; external630s plus30s
kill grace; export300s/external330s plus30s grace; allocated VRAM20GiB;
uncompressed return1.5GiB. Nonfinite gradients, altered dependencies, different
context, competing GPU work, count, time, VRAM or disk violations stop and
retain evidence. No files are deleted. This packet must run manually.

Protocol SHA256: d03050e18e4fd2369a6dd0803632bebfa0c709cc95cc8f7a6c8776785b0923a1
Execution archive SHA256: db822ba0b17f941d8e20783e5ee73ed97586dfdfb0f53a4d4909d392bd4b2656

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-group-guard-grad-v34-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-group-guard-grad-v34-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-group-guard-grad-v34-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_loss_cone_probe_v33_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/cctv_dgp_group_guard_grad_v34_vm &&
tar -xzf cctv-dgp-group-guard-grad-v34-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_guard_v34
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_group_guard_grad_v34_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_group_guard_grad_v34_vm.py --root . --protocol-sha d03050e18e4fd2369a6dd0803632bebfa0c709cc95cc8f7a6c8776785b0923a1 --verify-transfer &&
bash scripts/run_v34_guard.sh d03050e18e4fd2369a6dd0803632bebfa0c709cc95cc8f7a6c8776785b0923a1
```

Detach with **Ctrl+B**, release, then **D**. Step3 reconnects. Expected completed
diagnostic: gradient_queries:300, optimizer_updates:0, parameter_updates:0.
Export complete:true means packaging. Retain and download failures too; do
not change the frozen packet or run follow-on training.

5. Download from **Windows Google Cloud SDK Shell** after export completes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-group-guard-grad-v34-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-group-guard-grad-v34-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-group-guard-grad-v34-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-group-guard-grad-v34-export.json" "."
```

Each SCP command has one remote source for Windows PuTTY. An independent local
audit will verify hashes, all300 saved case gradients, all17 raw group derivative
assemblies, original output parity and frozen CPU inference. No local autograd
or optimizer is allowed. An audited derivative alone cannot approve a training
recipe, final evaluation, app promotion or completion of the thesis goal.
