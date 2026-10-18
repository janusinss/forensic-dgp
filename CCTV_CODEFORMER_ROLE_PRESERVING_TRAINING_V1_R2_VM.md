# Historical R2 training guide — actual 50-update run retained

**R2 ALREADY RAN: GENUINE TRAINED MODEL / PRESERVATION QUALITY STOP.**
Do not rerun this historical training folder unchanged or append epochs.
Use [the local research testing guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_R2_LOCAL_TEST_GUIDE.md>) and
[the actual independently audited return review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R2_RETURN_REVIEW.md>).
Dark retention passed, preservation failed; the trained model remains experimental.

---

# CodeFormer restoration training R2: five manual steps

**VERIFIED TRANSFER PREPARED / NOT RUN — 18 October 2026.**
[R2 preparation and limits](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R2_PREPARATION.md). [Latest R1 return review](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R1_RETURN_REVIEW.md).

Both V1 and R1 stopped before Adam: **zero new updates and no new checkpoint**.
R1 saved exactly equal reconstructed pixels and dictionary selections; only
intermediate logits differed by at most 0.00000762939453125. Their actual files
and failed source packages remain intact. The low-level cause remains unknown.
R2 explicitly changes the initializer's float32 logit numerical test to the
declared PyTorch 2.9 tolerances (atol0.00001/rtol0.0000013), with exact argmax/token
and margin-stability checks. Other arrays/state/RNG and restoration-quality gates
remain exact/unchanged. CPU checks do not prove this prerequisite passes on L4.

This is an actual **restoration-network training pilot**, with at most **50 new
Adam updates / 200 microbatches**. It starts from the retained full-coarse
CodeFormer research checkpoint, uses fresh Adam, and trains no brightness head.
Untouched pretrained CodeFormer remains the benchmark. This is our fine-tuned
adaptation of CodeFormer; restoration improvement remains unproven.

Require **16 GiB free after installation**, retaining a **4 GiB reserve**. Parent
`~/forensic-dgp/cctv_codeformer_adain_gradient_isolation_v1_vm` and the existing venv must remain available. Preflight
verifies **3,362 existing file bindings**. No automatic cleanup, retry or resume.
Preparation did not contact the VM or measure current free space.

Upload: **1,424,772 bytes**. SHA-256:
`c10957cd804d0844e67ecc4ee7990c810677d9b172e069631443654f0f5709ad`. Protocol: `82fcd34cada1eed238ef65a341c22580d12146e3ee30c2ff3094312d3ba1c85a`.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-role-preserving-training-v1-r2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-role-preserving-training-v1-r2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-codeformer-role-preserving-training-v1-r2-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_adain_gradient_isolation_v1_vm &&
test ! -e ~/forensic-dgp/cctv_codeformer_role_preserving_training_v1_r2_vm &&
tar -xzf cctv-codeformer-role-preserving-training-v1-r2-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s codeformer_role_preserving_v1_r2
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_codeformer_role_preserving_training_v1_r2_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
TASK_ROOT="$(pwd -P)" &&
PYTHON_VM="$(python -c 'import sys; print(sys.executable)')" &&
"$PYTHON_VM" -B -u "$TASK_ROOT/scripts/supervise_codeformer_role_preserving_training_v1_r2_vm.py" --root "$TASK_ROOT" --protocol-sha 82fcd34cada1eed238ef65a341c22580d12146e3ee30c2ff3094312d3ba1c85a
```

The supervisor runs preflight, training and export once. Estimated duration:
**15–35 minutes**, unmeasured for this repaired recipe. Enforced limits: worker
**3,000 seconds**, initialization **600 seconds**, gradient witness **180 seconds**,
optimization **900 seconds**, and each 406-case snapshot **600 seconds**; phases
share the worker cap. Preflight/export caps are **300/600 seconds**, with **30
seconds** termination grace. Maximum **1,399 neural forwards / 1,854 reserved
operations**, including **five gradient queries / 200 backward calls / 50 updates**.
Detach with **Ctrl+B, then D**. A competing workload stops this launch untouched.

5. Download the **four return files** from **Windows Google Cloud SDK Shell** after the supervisor finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-role-preserving-training-v1-r2-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-role-preserving-training-v1-r2-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-role-preserving-training-v1-r2-export.json" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-role-preserving-training-v1-r2-supervisor-final.json" "."
```

One remote source per SCP command avoids the Windows PuTTY multiple-source error.
The checksum is ASCII/LF. Export completion means packaging finished and can
include a failed run. Preserve errors, partial outputs and checkpoints; do not
erase or relaunch an existing folder unchanged. Failed preflight prevents training.

After 50 updates and the final comparison bank finish, the genuine checkpoint is:
`~/forensic-dgp/cctv_codeformer_role_preserving_training_v1_r2_vm/outputs/role_preserving_training_v1_r2/snapshot_0050/checkpoint.pt`.
An earlier training failure retains its actual stopped/partial evidence. An
initializer failure before Adam can correctly produce **no new checkpoint**.
A final-state alias can reference an existing verified checkpoint without copying it.

Optional monitoring from another **VM SSH** terminal:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_role_preserving_training_v1_r2_vm/outputs/supervision/diagnostic.log ~/forensic-dgp/cctv_codeformer_role_preserving_training_v1_r2_vm/outputs/supervision/export.log
```

Returned optimizer/state/RNG/sampler/counters, every RAW/PNG and planned visual
sheet require independent audit. Existing normal preservation and dark-retention
gates remain unchanged. User image-quality acceptance, independent FINAL review
and the existing app Auto/override/PNG flow remain pending. No app promotion.
