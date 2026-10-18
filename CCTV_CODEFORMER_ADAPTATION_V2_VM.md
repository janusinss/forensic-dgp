# CodeFormer V2 return audited: quality-stopped at 50 updates — 15 October 2026

**CURRENT: 50/600 updates; real weight changes; declared quality checks failed.**
The downloaded archive, training state and every returned RAW/PNG output were
independently checked. All 52 sheets / 406 cases and the three failed clear
controls at full 256 pixels were reviewed. V2 has useful-looking improvements
and higher paired SSIM on 50/53 HQ blur cases, but is not an accepted app model.

[V2 results, causes and limitations](CCTV_CODEFORMER_ADAPTATION_V2_RETURN_REVIEW.md).
[All comparison pages](outputs/codeformer_adaptation_v2_return_review/visual_sheets/INDEX.md).
[Latest brightness preference](CCTV_CODEFORMER_ADAPTATION_V2_USER_REVIEW_20261015.md).

The user accepts moderate brightening when it makes a dark face easier to see.
Evaluate that benefit separately from washed-out detail and altered visible
features; brightness change alone is not a verdict on usefulness. All three
failed clear faces also lose SSIM. Preserve
the historical V2 clear and normal-blur photometric failures unchanged. Any
future criteria must be declared prospectively, not fitted to those failures.

All 350 trainable tensors changed; 165 frozen tensors stayed exact. V2 reached
200 TRAIN exposures, not 600 updates or epochs. V1 u200 versus V2 u50 brightness
differences are descriptive, with unequal budgets and jointly changed recipes.
Both stopped states, original checkpoints, splits and gate failures remain
protected. Do not relaunch V1/V2 unchanged or append epochs after these stops.

Next is a bounded zero-optimizer VM diagnostic of training/evaluation precision,
photometry and loss paths before choosing further training. Its design review
is saved under outputs/codeformer_adaptation_v2_return_review/next_diagnostic_review_v1.md;
it is not an execution-ready packet. Actual neural/gradient diagnostic work and
all training remain human-launched on L4 in tmux. No new VM run was launched.

Restoration comes first. Detector/completion development remains deferred until
user-accepted restoration. Native CCTV remains unpaired and unqualified; final
identities, app adoption/flow and independent final review remain outstanding.
The full thesis goal stays active. Earlier PREPARED/UNRUN commands below are
retained historical evidence, superseded by this current return status.

---

# CodeFormer V2: brightness correction — five manual VM steps

**VERIFIED / PREPARED / UNRUN.** This changed fine-tune starts from untouched
CodeFormer and addresses V1's brightness drift. It preserves V1's promising
outputs and its failed clear-preservation evidence. Do not rerun V1.

Require **20 GiB free after installation**; the worker preserves a **4 GiB** reserve.
Planning estimate: **20–40 minutes** including snapshots/export. Actual V2 timing
is unmeasured; first 20 updates measure feasibility. Maximum 600 updates, reviews
at 0/50/200/600; optimization cap 60 minutes, worker cap 120 minutes, export cap
10 minutes. A preservation/resource stop still exports evidence.

## 1. Upload — Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-adaptation-v2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-adaptation-v2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install — VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-adaptation-v2-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_codeformer_adaptation_v2_vm &&
tar -xzf cctv-codeformer-adaptation-v2-execution.tar.gz -C ~/forensic-dgp
```

Expected checksum result: `cctv-codeformer-adaptation-v2-execution.tar.gz: OK`.
The supplied checksum file uses Linux LF endings. If the directory already exists,
retain it and inspect its evidence before any new version; do not overwrite it.

## 3. Open tmux — VM SSH

```bash
tmux new-session -A -s codeformer_adaptation_v2
```

## 4. Launch — inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_adaptation_v2_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
bash scripts/run_adaptation_v2.sh 029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183
```

The supervisor verifies transfer bytes, tmux, idle workloads and finite execution;
the worker checks the L4, packages, space, initial parity, gradients and quality.
An idle tmux session alone is allowed. Existing training/results are not resumed.
Do not start a second run. Detach with **Ctrl+B**, release, then **D**.
Reattach using `tmux attach -t codeformer_adaptation_v2`.

After export, `complete: true` confirms a packaged archive. Check `Trainer exit
code`, `failure_present` and `run_results_present`; export completion alone is
not training success or model acceptance. Preserve the console output and files
even if a guard stops the run. Do not append epochs to a failed recipe.

## 5. Download — Windows Google Cloud SDK Shell, after export

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adaptation-v2-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adaptation-v2-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-adaptation-v2-export.json" "."
```

Each command has one remote source, compatible with Windows PuTTY. The downloaded
files will be in `C:\xampp\htdocs\YEAR 4\Testing\outputs`.
The return must be independently audited before choosing a model or app default.

Execution archive SHA256: `61acb9cd98b74ad1412d1ecbe4e916415e99aa80377d0c94332409fabf2c682b`.
Protocol SHA256: `029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183`.
[Verified preparation](CCTV_CODEFORMER_ADAPTATION_V2_PREPARATION.md).
