# U96 spatial feasibility: five manual VM steps

**Prepared and unrun. U96 remains the incumbent.** This is a diagnostic before
another training pilot. It starts from exact complete U96 and tests its existing
final spatial residual block plus RGB convolution: 10 tensors / 75,843 scalars;
529 tensors stay frozen. It does not continue the failed decoder-readout run or
overwrite the retained u64 fusion challenger.

Archive: `cctv-codeformer-u96-spatial-feasibility-v1-execution.tar.gz`
(40,009,514 bytes, approximately 40 MB).
Archive SHA256: `84652e91c859874ece30fceb79b84075d712f1af467b6de69c2ed65ac9b5a8d0`.
Protocol SHA256: `6ad2d51f25cc5c7ef6fab95c3ff26f6a646780d5c0e5ba6e118b7d78d16d7f1e`.

The diagnostic measures 32 separate component gradients on 16 previously admitted
TRAIN cases. It can test two independent small parameter changes, both from the
same initializer, with maximum coordinate displacements 0.00001 and 0.00005.
These are **not learning rates or epochs**. There are zero optimizer objects or
steps and at most 218 full forwards. A missing common-descent certificate stops
before either change. A failed first proposal stops the second. Both must pass;
there is no best-of-two selection. Every existing quality requirement remains.

Planning estimate: **12–20 minutes**, unmeasured for this new path. Scientific
work stops internally at 1,200 seconds; the external worker cap is 1,260 seconds,
with at most 20 additional seconds to terminate its process group. Individual
bank phases stop at 180 seconds. Failure capture has at most 30 additional
seconds, followed by at most 5 seconds for its receipt and 5 for RNG restoration,
inside the external limit when the scientific stop returns promptly. Native
code that does not return is subject to external termination. Export has a
300-second internal / 330-second external limit.

Leave **8 GiB free after installation**, plus approximately 377 MB for initializer
copies before installation. The continuing reserve is 4 GiB; output and return
archive each have a 2 GiB cap. Worker limits are 10 GiB RSS and 18 GiB CUDA reserved
memory. Do not run a competing GPU workload. This preparation has not connected
to the VM or checked its current available space.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-spatial-feasibility-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-spatial-feasibility-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Check and install in VM SSH

The installer copies and hashes the two retained R2/U96 files; it does not run the
old smoothanchors-LR experiment. If a check fails, preserve its output and stop.
Never overwrite or delete a partial installation to repeat this command.

```bash
cd ~ &&
sha256sum -c cctv-codeformer-u96-spatial-feasibility-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights/codeformer_r2.pth &&
test -f ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights/codeformer_u96_adapter.pth &&
test ! -e ~/forensic-dgp/cctv_codeformer_u96_spatial_feasibility_v1_vm &&
tar -xzf cctv-codeformer-u96-spatial-feasibility-v1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_u96_spatial_feasibility_v1_vm &&
python3 -B research/codeformer_u96_spatial_feasibility_v1/install_weights.py --root . --source-root ../cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm --protocol-sha 6ad2d51f25cc5c7ef6fab95c3ff26f6a646780d5c0e5ba6e118b7d78d16d7f1e
```

Expected: installer `"complete": true`, two exact copies, zero checkpoint
deserializations and zero neural/training calls. The archive deliberately omits
these two large files. Preserve the separate VM migration backup.

## 3. Open tmux

```bash
tmux new-session -A -s u96_spatial_feasibility_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_u96_spatial_feasibility_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u96_spatial_feasibility_v1.py --root . --protocol-sha 6ad2d51f25cc5c7ef6fab95c3ff26f6a646780d5c0e5ba6e118b7d78d16d7f1e --verify-transfer &&
bash scripts/run_feasibility.sh 6ad2d51f25cc5c7ef6fab95c3ff26f6a646780d5c0e5ba6e118b7d78d16d7f1e
```

The static receipt reports 281 files, 96 retained TRAIN cases, 16 selected gradient
cases, 32 adaptive TRAIN monitors and 24 legacy guards. Static verification makes
zero neural calls; it does not establish actual L4 parity or quality. The
supervisor prints a child PID and log path; worker progress is written to the log.

In a second VM SSH window, view progress with:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_u96_spatial_feasibility_v1_vm/run.log
```

Detach with **Ctrl+B**, then **D**. Reattach with step 3. Do not launch again.
Normal completion requires the initializer and every applied complete 539-tensor
state, separate component gradients, RAW/PNG outputs and attempt/return counters.
On failure, existing and partial evidence remains; additional state capture is
best effort and can reach its time or output cap. Initial quality or parity
failure can stop before any parameter application.

Forward execution uses strict deterministic FP32 settings. Each reserved gradient
query temporarily enables warning-only handling of the installed Torch CUDA
bilinear-upsample backward warning, records it and restores strict settings.
Unexpected nondeterministic gradient operations stop the run. Bitwise repeatable
CUDA gradients are not claimed.

The final supervisor receipt reports `worker_exit`, `export_exit`,
`export_attempted` and `active_writer_outcome_unknown`. Worker exit 2 is a retained
quality stop. Worker exit 0 is diagnostic completion, not model qualification.
Export exit 0 confirms transport only. After a known worker stop, export is
attempted once cleanup is confirmed. If worker cleanup is unknown, export is
skipped; retain the receipts and terminal output instead of retrying. No automatic
retry, resume, extra epochs or promotion is configured.

## 5. Download from Windows Google Cloud SDK Shell

Wait for export to finish. Windows PuTTY requires these three separate downloads.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-spatial-feasibility-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-spatial-feasibility-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-spatial-feasibility-v1-export.json" "."
```

If export fails or is skipped, do not rerun the experiment or overwrite its partial
archive. Preserve `export_failure.json` if present, supervisor/status/journal
records and the terminal output for inspection. An interrupted operation remains
unknown until independently established otherwise.

The project, zone, instance and username above are the existing user-identified
target. No VM connection, installation, CUDA run or training was performed while
preparing this guide. Returned states, metrics, gates and every planned visual
sheet still require independent audit before a further training decision. The
1% full-pilot endpoints are reported separately; diagnostic feasibility does not
qualify a successor. U96, the earlier u64 challenger and all failures remain
retained. Native/wild evidence stays unpaired, reserved FINAL remains unopened,
independent reviewers remain required and completion development stays deferred.

The sealed packet includes preparation-time instruction snapshots marked unsealed;
those snapshots are historical evidence. This guide and the later publication
record supersede their readiness status without modifying the sealed archive.
