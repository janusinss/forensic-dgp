# Historical V1 commands: stopped before training — 18 October 2026

**ORIGINAL V1 FAILED INITIALIZER PARITY: ZERO NEW UPDATES / NO NEW CHECKPOINT.**
This document is retained as historical evidence. Do not rerun the failed folder
unchanged. [Actual return review](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_RETURN_REVIEW.md).
Use the separately prepared [R1 five-step guide](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R1_VM.md); R1 is unrun and
keeps the original restoration objective and quality requirements.

---

# CodeFormer restoration training: five manual steps

**VERIFIED TRANSFER PREPARED / NOT RUN — 18 October 2026.**
[Why this is the next model change](CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_PREPARATION.md).

This pilot trains **350 tensors in CodeFormer's restoration network** for up to
**50 new Adam updates / 200 microbatches**. It starts from the retained full-coarse
research checkpoint, uses a fresh optimizer and trains no brightness head.
Both normal exposure paths preserve the untouched model's full RGB/token behavior;
dark paths use clean-reference supervision plus frozen-parent retention. This is
our fine-tuned adaptation of CodeFormer. Its quality has not yet been demonstrated.

Require **16 GiB free after installation**, with a 4 GiB reserve. The immutable
parent `cctv_codeformer_adain_gradient_isolation_v1_vm`
must remain under `~/forensic-dgp`. Preflight verifies **3,362 existing file
bindings**. No automatic cleanup, download, retry or resume is included.
Preparation did not contact the VM or measure its current free space.

Upload: **1,487,646 bytes**. SHA-256:
`ec0ad9b3a815c8e9ee76e52be60f929a99a24a7a5791c0ff3d4180ee5a561fd8`.
Protocol SHA-256: `67d71dbb38de308b3f0b9299bc9f34844c51c8445149ce35cdac8e98004f1afb`.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-role-preserving-training-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-role-preserving-training-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-codeformer-role-preserving-training-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_adain_gradient_isolation_v1_vm &&
test ! -e ~/forensic-dgp/cctv_codeformer_role_preserving_training_v1_vm &&
tar -xzf cctv-codeformer-role-preserving-training-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s codeformer_role_preserving_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_codeformer_role_preserving_training_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
TASK_ROOT="$(pwd -P)" &&
PYTHON_VM="$(python -c 'import sys; print(sys.executable)')" &&
"$PYTHON_VM" -B -u "$TASK_ROOT/scripts/supervise_codeformer_role_preserving_training_v1_vm.py" --root "$TASK_ROOT" --protocol-sha 67d71dbb38de308b3f0b9299bc9f34844c51c8445149ce35cdac8e98004f1afb
```

The supervisor runs preflight, training and export once. Expect approximately
**15–35 minutes**, an estimate for this unrun recipe. Enforced limits: worker
**3,000 seconds**, optimization **900 seconds**, initialization **600 seconds**,
gradient witness **180 seconds** and each 406-case snapshot **600 seconds**.
Preflight/export are capped at **300/600 seconds**, with **30 seconds** termination
grace per stage. Each phase remains within the worker cap. Maximum **50 updates,
200 backward calls and five gradient queries**; **1,397 neural forwards** maximum.
Detach using **Ctrl+B, then D**. An existing competing workload stops this launch.

5. Download the **four return files** from **Windows Google Cloud SDK Shell** after the supervisor finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-role-preserving-training-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-role-preserving-training-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-role-preserving-training-v1-export.json" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-role-preserving-training-v1-supervisor-final.json" "."
```

One remote source per SCP command avoids the Windows PuTTY multiple-source error.
The checksum file is ASCII with LF line endings. An export marked complete only
means the evidence was packaged; it can include a stopped or quality-failed run.
Preserve errors, partial outputs and checkpoints. Do not erase or relaunch the
folder unchanged. Preflight failure leaves its log and does not start training.

The actual model checkpoint is retained at
`~/forensic-dgp/cctv_codeformer_role_preserving_training_v1_vm/outputs/role_preserving_training_v1/snapshot_0050/checkpoint.pt`
after all 50 updates and the final comparison bank finish. Earlier failure keeps
the actual stopped state or partial checkpoint evidence instead. A final-state
alias can point to the same verified checkpoint without another large copy.

Optional monitoring from another **VM SSH** terminal:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_role_preserving_training_v1_vm/outputs/supervision/diagnostic.log ~/forensic-dgp/cctv_codeformer_role_preserving_training_v1_vm/outputs/supervision/export.log
```

Returned state, Adam/RNG/sampler, actual updates, every RAW/PNG, all planned visual
sheets and the unchanged preservation/dark-retention gates require independent
audit. App acceptance, reserved FINAL review and user approval of image quality
remain pending. No automatic app switch or model qualification occurs.
