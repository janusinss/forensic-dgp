# Normal-tone diagnostic stopped; direct training pilot design — 17 October 2026

**CURRENT: RETURN AUDITED / EXACT-GRADIENT CONTROL FAILED / ZERO MODEL UPDATES.**
[Return review](CCTV_CODEFORMER_NORMAL_TONE_V1_RETURN_REVIEW.md).
[New checkpoint-producing training design](CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md).

All199 returned members/91,344,209 payload bytes and four companions verified;
all20 static files exact. Fresh original0 coarse gradients failed strict cached
bit-identity after16 source forwards and24 queries; zero Adam steps/probes/
committed updates or new checkpoint. All16 source image/token/logit/feature
outputs equal the original controls; root reviewed both source-only sheets.
Small recorded gradient differences do not establish harmlessness or exact
physical cause. The failed control and original zero tolerance stay preserved.

The latest user request prioritizes an actual model. The optional eight-step
sequence is prospectively superseded by a NEW50-update training hypothesis:
normal-only coarse luminance centering, fresh full-objective gradients, original
initializer, unchanged dark/other losses/data/optimizer/route and full61-clear/
53-blur gates. Save genuine candidate/stopped states before gate assessment;
never promote from checkpoint/export completion. No cached-gradient training,
failed recipe resume or historical replay. The loss remains an unproven choice.

Only the user's finite manual L4/tmux workflow may launch training after a new
sealed and reviewed packet exists. No new training has run here. Model/app/user/
independent-final qualification, input-only insufficiency threshold and later
separate seven-family completion scope remain required. Native unpaired and
paired synthetic evidence/untouched FINAL stay separate; no identity/ethnicity/
Zamboanga performance claim. Existing app design and original assets/failures
stay intact. Storage offload is complete and was not repeated. Goal incomplete.

Earlier command guides below are historical; do not relaunch the failed test.

---

# CodeFormer normal-tone V1: five manual steps

17 October 2026. **VERIFIED SOURCE PACKET / PREPARED / UNRUN.**

This is the next short objective test toward an improved trained CodeFormer
checkpoint. It keeps useful dark-face supervision and tests whether normal-image
coarse brightness supervision contributes to appearance drift. It commits zero
training updates and exports no new model checkpoint. Eight real disposable Adam
steps are controls, not additional epochs or an accepted model.

Upload 782,091 bytes; 20 regular source members.
Protocol SHA-256: `6a3e162f6383cefac9d14fa11e08032d0149a7583d6ff60ba2ff893056335d22`.
Archive SHA-256: `62e8e8c3d5b520fac1bf65acdbcc51ce2ae97baa4eaa6e84fb2ecbadd24b9e88`.

Estimated runtime **8–18 minutes**, unmeasured and excluding transfers. Enforced
worker limit1,800 seconds, preflight300 and export600, with grace60/30/30 seconds.
Initialization300/source600/probe900 are cumulative phase caps within the worker
total. Require **10 GiB free after installation**, retaining4 GiB reserve. Caps:
RSS8 GiB, reserved CUDA22 GiB, result2 GiB, archive2.25 GiB. These are limits,
not measured peaks or guaranteed completion. Final external closure has30 seconds.

Keep both existing parent folders and the frozen environment. Their present VM
availability was not checked remotely during this preparation. The new preflight
verifies4,630 read-only dependencies:2,011 AdaIN and2,619 loss-response files,
including old source/control outputs and the exact eight metric targets. No
weights or caches are uploaded again. Do not upgrade image/numerical packages.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-normal-tone-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-normal-tone-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-normal-tone-v1-execution.tar.gz.sha256 &&
printf '%s  %s\n' '62e8e8c3d5b520fac1bf65acdbcc51ce2ae97baa4eaa6e84fb2ecbadd24b9e88' 'cctv-codeformer-normal-tone-v1-execution.tar.gz' | sha256sum -c - &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
printf '%s  %s\n' '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5' "$HOME/forensic-dgp/cctv_codeformer_adain_gradient_isolation_v1_vm/protocol.json" | sha256sum -c - &&
printf '%s  %s\n' '23c19c394e34261861b8853cfd60aa60d1cfaefc91b12e92a2f8b8dce85112ab' "$HOME/forensic-dgp/cctv_codeformer_loss_response_v1_vm/protocol.json" | sha256sum -c - &&
python3 -B -c 'import shutil; from pathlib import Path; free=shutil.disk_usage(Path.home()/"forensic-dgp").free; need=10*1024**3+32*1024**2; print({"free_bytes":free,"required_before_install_bytes":need}); assert free>=need, "Need 10 GiB plus 32 MiB free before installation"' &&
test ! -e ~/forensic-dgp/cctv_codeformer_normal_tone_v1_vm &&
tar -xzf cctv-codeformer-normal-tone-v1-execution.tar.gz -C ~/forensic-dgp
```

If a check stops, retain its output. Do not delete previous run folders, replay
historical pilots or recreate missing parent evidence automatically.

## 3. Open tmux

```bash
tmux new-session -A -s cf_normal_tone_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_normal_tone_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/supervise_codeformer_normal_tone_v1.py --root . --protocol-sha 6a3e162f6383cefac9d14fa11e08032d0149a7583d6ff60ba2ff893056335d22
```

Maximum224 new CodeFormer forwards,48 autograd queries and8 completed disposable
Adam steps. There are no teacher/LPIPS/benchmark forwards, committed updates,
checkpoint exports or automatic follow-on training. Each branch restores its
exact base; neither measurement state resumes the failed historical recipe.

The supervisor checks workloads, verifies hashes, runs this one finite test and
exports terminal evidence. A parity/resource/quality stop remains a failure;
no automatic retry follows. Detach with **Ctrl+B**, release, **D**; reattach using
`tmux attach-session -t cf_normal_tone_v1`.

Wait for export and the final supervisor companion to finish. Export reports
`"complete": true` and `"archive_sha256"`; a full measurement additionally has
`"complete_measurement": true` and `"actual_disposable_Adam_steps": 8`.
An exported failure can have fewer steps. Export completion is not model success.

## 5. Download from Windows Google Cloud SDK Shell

Each command has one remote source, compatible with Windows PuTTY.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-normal-tone-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-normal-tone-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-normal-tone-v1-export.json" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-normal-tone-v1-supervisor-final.json" "."
```

The fourth file carries exact final export-log bytes, the parsed finished receipt
and its hash, plus archive/export bindings outside the archive it describes.
Preserve partial/failure evidence if
export or closure fails. Returned files must be independently audited and all
planned visual sheets reviewed before selecting the separate checkpoint-producing
training pilot. The main app is unchanged; no new restoration option is installed.
