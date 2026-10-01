# Existing VM: matched reflection-coverage detector pilot

Actual training runs only on the Google Cloud VM. Two arms fork the same audited
epoch30 detector and saved AdamW moments;252 new updates per arm. The fixed
specification is `REFLECTION_COVERAGE.md`. New CUDA execution is unverified.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM repository: `~/forensic-dgp/`.
Execution root: `~/forensic-dgp/coverage_vm_bundle/`.
Existing source weights, replay, reviewed masks, benchmark and pinned packages
from the completed coverage/continuation/focus runs are required. This is an
incremental package for that existing bundle, not a fresh VM installer.

## 1. Upload two files to VM home

- `C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection-coverage-code.tar.gz`
- `C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection-coverage-code.tar.gz.sha256`

Use Google Cloud SSH's Upload File control. The archive adds the new code,
frozen280-case cache, source-screening lineage, tests and fixed recipe. It
contains no replacement for any previously inventoried file. The checksum is
LF, avoiding the earlier Windows carriage-return filename failure. These new
files have not been pushed; `git pull` alone does not install this package into
the isolated `coverage_vm_bundle` execution directory.

## 2. Paste into SSH

```bash
cd ~
sha256sum -c reflection-coverage-code.tar.gz.sha256 &&
test -d ~/forensic-dgp/coverage_vm_bundle &&
tar --keep-old-files -xzf reflection-coverage-code.tar.gz -C ~/forensic-dgp/coverage_vm_bundle &&
tmux new-session -A -s dgp_training
```

Extraction refuses an existing member. Do not repeat extraction to overwrite
code/evidence. tmux attaches to the existing session or creates it. Use its shell
prompt when the previous run has finished.

## 3. Paste inside tmux

```bash
cd ~/forensic-dgp/coverage_vm_bundle
if [ -f ../feature_vm_bundle/.venv/bin/activate ]; then
  source ../feature_vm_bundle/.venv/bin/activate
elif [ -f ../venv/bin/activate ]; then
  source ../venv/bin/activate
fi
nvidia-smi &&
python3 -u scripts/train_reflection_coverage_vm.py --preflight --output outputs/reflection_coverage_preflight_vm &&
python3 -u scripts/train_reflection_coverage_vm.py
```

Reuse the completed CUDA environment; no new package install is needed. The
runner refuses Windows/CPU before models or optimizers, requires at least4GiB
total GPU memory, verifies all four prior inventories plus the new inventory,
source weights/moments,280-case audit/visual review and fixed schedule. Preflight
restores optimizer states and runs a core8-example forward plus2-example
supplemental forwards for both arms with **zero optimizer updates**. It must print:

```text
CUDA forward/moments/support passed; zero updates
```

Do not add `--dry_run` or `--batch_size`; these are older restoration options.
This matched runner uses `--preflight` and its fixed8+2 batches. The full run
also requires original parent/source CUDA metrics to match audited results
before training. If preflight/source verification fails, preserve exact output;
do not weaken the comparison, remove guards or reset the saved optimizer.

Each independent arm saves global36 and42 under
`~/forensic-dgp/coverage_vm_bundle/outputs/reflection_coverage_vm/{control,reflective}/`.
Final states have882 lifetime model updates and optimizer step672, including
the prior630/420 source history. The recipe adds504 updates in total. Each arm's
`best_detector.pth` exists only if the original real/synthetic guards pass.
Missing best files are a failed eligibility decision, not a canceled run.

The previous420-update L4 pilot reported103 seconds excluding compression. Allow
roughly5–15 minutes for this pilot's verification,504 updates, mask exports and
archive compression; its actual wall time is not measured. Keep roughly3GB free
for checkpoints, moments and the return archive. This is detector adaptation,
separate from restoration epochs27–31 and full Phase5 identity training.

Detach with Ctrl+B, then D. Reattach with:

```bash
tmux attach-session -t dgp_training
```

## 4. Download the completed archive

The runner prints the archive location after writing `complete.json`. In Google
Cloud SSH's Download File control paste:

```text
/home/janusdominic0/forensic-dgp/coverage_vm_bundle/reflection-coverage-results.tar.gz
```

Also download the same path with `.sha256` appended. Save locally as:

```text
C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection-coverage-results.tar.gz
C:\xampp\htdocs\YEAR 4\Testing\outputs\reflection-coverage-results.tar.gz.sha256
```

The archive includes both arms, all504 step records, baseline/source metrics,
initial/final model and optimizer snapshots, raw masks and new source/data/code
inventory. Expected local extraction layout is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\downloaded_reflection_coverage\outputs\reflection_coverage_vm\`.
Do not extract over the local project or previous results; return verification
will check safe member paths and exact file hashes first.

## 5. Next after return

Audit logs, checkpoint/moment bindings, unchanged states, saved masks and original
gates; reproduce predictions without fitting. Inspect the fixed ten-row real
cohort, lens-only zoom and training fixture examples. New fixture metrics are
training diagnostics. Only a candidate passing unchanged gates advances to
reviewed end-to-end completion. Retain generator/application and Phase3 baseline
until that review supports promotion. The project goal stays active and unmet
when a script merely finishes or a detector only improves its training metrics.
