# Historical AdaIN V1 commands — returned and audited 17 October 2026

**50 UPDATES VERIFIED; QUALITY STOP RETAINED. DO NOT RELAUNCH OR RESUME UNCHANGED.**
[Actual result](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_RETURN_REVIEW.md)
supersedes PREPARED/UNRUN text below. All original guide bytes remain history.
The next design review has no new executable packet or launch command.
The requested usable-input gate is documented prospectively and unimplemented.

---

# CodeFormer AdaIN gradient isolation V1: five manual steps

**17 October 2026. VERIFIED / PREPARED / UNRUN.**
Upload **1,159,959 bytes (1.16 MB)**. No VM command or training was executed during
preparation. Use this new packet once; preserve all earlier packets and results.

This is a new controlled experiment from untouched CodeFormer, with at most
**50 optimizer updates**. It does not resume the stopped FP32 checkpoint. The
estimated runtime is **12–20 minutes, unmeasured for this new experiment**.
Enforced ceilings are 3,000 seconds for the worker and 600 seconds for export,
with a separate 300-second preflight, then termination grace of 30/180/30 seconds
for preflight/worker/export. Initialization is limited to 600 seconds; the witness
to 180 seconds within initialization, actual training to 900 seconds and each
snapshot to 600 seconds. The worker total stays within 3,000 seconds. A quality stop still exports evidence;
an archive marked complete does not mean the model passed.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-adain-gradient-isolation-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-adain-gradient-isolation-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

Each `gcloud compute scp` command has one source. This avoids the Windows PuTTY
multiple-remote-source limitation when downloading later.

## 2. Verify and install in VM SSH

Keep at least **18 GiB free before installing**. After extracting the small
overlay, the installer independently requires at least **18,619,279,802 bytes**
for the exact dependency copies plus a **16 GiB free-space floor**. It retains
the original parent files and creates verified regular copies, without links.

```bash
cd ~ &&
sha256sum -c cctv-codeformer-adain-gradient-isolation-v1-execution.tar.gz.sha256 &&
printf '%s  %s\n' 'b49e476b3570b8dbe5d6160e8ee0225571f767f72a331946f4f428aab412a24e' 'cctv-codeformer-adain-gradient-isolation-v1-execution.tar.gz' | sha256sum -c - &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -d ~/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm &&
printf '%s  %s\n' '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1' "$HOME/forensic-dgp/cctv_codeformer_fp32_comparison_v1_vm/protocol.json" | sha256sum -c - &&
python3 -B -c 'import shutil; available=shutil.disk_usage("/home/janusdominic0/forensic-dgp").free; print({"free_bytes":available,"required_before_install_bytes":18*1024**3}); assert available >= 18*1024**3, "Need at least 18 GiB free before installation"' &&
test ! -e ~/forensic-dgp/cctv_codeformer_adain_gradient_isolation_v1_vm &&
tar -xzf cctv-codeformer-adain-gradient-isolation-v1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_adain_gradient_isolation_v1_vm &&
python3 -B scripts/install_codeformer_adain_gradient_isolation_v1.py --root . --protocol-sha 4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5
```

Expected install receipt: `"complete": true`, `"files": 1909`, and zero neural,
gradient and optimizer calls. Do not relaunch an incomplete installation or
delete its journal; retain the message for diagnosis. The stopped FP32 checkpoint
is not required by this new initializer.

## 3. Open tmux

```bash
tmux new-session -A -s cf_adain_isolation_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_adain_gradient_isolation_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u -m research.codeformer_adain_gradient_isolation_v1.worker --root . --protocol-sha 4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5 --verify-transfer &&
python -B -u scripts/supervise_codeformer_adain_gradient_isolation_v1.py --root . --protocol-sha 4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5
```

The new supervisor checks competing workloads, runs a finite preflight, and
starts the guarded worker. The worker first verifies all 391 regenerated HQ
targets and runs the fixed four-CodeFormer/four-LPIPS/eight-query witness before
constructing Adam. A failed witness or preservation guard must remain a stop.
There is no automatic retry or resume.

To leave training running, press **Ctrl+B**, release both keys, then press **D**.
Reattach with `tmux attach-session -t cf_adain_isolation_v1`.

Wait for export to finish before downloading. The trainer status and the export
status are separate; a successful export can contain a failed experiment.

## 5. Download from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adain-gradient-isolation-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adain-gradient-isolation-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adain-gradient-isolation-v1-export.json" "."
```

Retain all three returned files. Their arrival triggers independent archive,
state, saved-pixel and visual review. Training completion alone does not qualify
this adaptation or change the local application's model.

[Design](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_DESIGN.md).
[Preparation evidence](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_PREPARATION.md).
[Audited preceding measurement](CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_RETURN_REVIEW.md).

Protocol SHA-256: `4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5`.
Archive SHA-256: `b49e476b3570b8dbe5d6160e8ee0225571f767f72a331946f4f428aab412a24e`.
