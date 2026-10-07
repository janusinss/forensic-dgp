# V30: broader TRAIN coverage in the same mean-centered DGP path

V29 passes the50-case TRAIN capacity audit but fails21 preservation checks on
520 photographic DEV cases and lacks convincing clarity on24 native CCTV crops.
V30 tests the coverage hypothesis using the already approved781 TRAIN references
and3905 existing cases. It starts from the original DGP, with the same selected12
decoder tensors, mean-centered forward path, frozen encoder/head4/buffers,
seven loss weights,50-case loss normalizers and AdamW settings. No V9 or V29
trained candidate, optimizer or historical pilot is resumed. No DEV/native/final
image enters optimization. Source folders do not imply ethnicity.

Exactly800 updates expose all3905 cases once over781 batches plus95 cases in
19 batches of a predetermined second shuffle:4000 total samples, as V29.
Snapshot0/50/400/800 covers the full TRAIN cohort. The initial70-query selected12
proof precedes any optimizer. Early50 structure gain must reach1%; final800 must
reach10%, both source gains must be nonnegative, all17 preservation groups must
pass and brightness-only fraction must stay<=20%. These are necessary TRAIN
gates. Native useful-output review and separate520 DEV checks are still required.

Eight-file packet:746,356 bytes; no images or weights uploaded.
Archive SHA256: `7a3d58db1e6d14a1c5a589269ec6922cd8fbe8d0f1834f4c242fb1d2cf72ac39`.
Protocol SHA256: `b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1`.
The original V27/R2/V28/diagnostic/V29 and mixed V9r2 data must remain in their
existing folders. Transfer verification checks5467 TRAIN files without neural work.

Use the idle existing L4/g2-standard-4 with6GiB free. Estimate15–35 minutes;
broader timing is unmeasured and the timing receipts determine continuation.
Proof cap300s, cache900s, fit3600s, total worker4500s, external4800s+30s grace;
export900s internally and930s+30s externally. Allocated VRAM<=20GiB, export
uncompressed<=3GiB. Cache timing excludes its first GPU batch from steady sampling;
update20 timing reserves three times the measured full snapshot0 duration with
1.25 safety factor. Existing work is never terminated or deleted by this pilot.
No resume, overwrite, unchanged retry, earlier checkpoint selection or automatic
next recipe occurs. Stops retain their logs, partials and export evidence.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-broader-mean-v30-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-broader-mean-v30-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-broader-mean-v30-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test -f ~/forensic-dgp/cctv_dgp_mean_centered_decoder_vm_v29/outputs/results.json &&
test -f ~/forensic-dgp/cctv_dgp_mixed_vm_v9_r2/mixed_protocol_v9.json &&
test ! -e ~/forensic-dgp/cctv_dgp_broader_mean_vm_v30 &&
tar -xzf cctv-dgp-broader-mean-v30-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_v30
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_broader_mean_vm_v30 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_broader_mean_v30_vm.py --root . --protocol-sha b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1 --verify-transfer &&
bash scripts/run_v30.sh b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1
```

Press `Ctrl+B`, then `D` to detach. The process remains in tmux. A stop must be
downloaded and audited; do not repeat the launch or delete its output folder.

5. Download after **Export exit code: 0**, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-broader-mean-v30-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-broader-mean-v30-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-broader-mean-v30-export.json" "."
```

Each PuTTY download has one remote source. Export completion is not training or
restoration acceptance. The prospective independent checker audits all saved
TRAIN PNGs, mean controls, vectors, metrics and gates; verifies the initial70
gradient arrays and all checkpoint frozen partitions; and replays every final
TRAIN case plus the fixed50 previews at each available checkpoint on CPU without
derivatives. Full replay has a finite7200s cap. Final reserved data, the app and
all covering-family qualification remain separate unfinished milestones.
