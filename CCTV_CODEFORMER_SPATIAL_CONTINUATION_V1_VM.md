# Manual comparison: continue both trained branches from update 24 to 96

This new experiment continues the two audited update-24 states. It adds at most 72 Adam updates per branch, 144 total. It keeps their saved optimizer/RNG states, the same learning rate and matched TRAIN sampling. These are updates, not epochs. Early preservation checks run at 30/36; further checks and saved outputs run at 48/72/96. A failing branch stops. Both results remain available for review.

Require **6 GiB free after installation**, with **4 GiB preserved during the run**. Estimated worker time is **8–12 minutes**, based on the previous L4 measurements, with a hard 45-minute worker limit and 52-minute overall supervision/export ceiling plus 40 seconds for bounded termination. This is a finite learning comparison; useful improvement is not guaranteed. Existing R2 and the completed 24-update experiment stay intact. The new package contains its own required assets and states.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-spatial-continuation-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-spatial-continuation-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install in VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-spatial-continuation-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_codeformer_spatial_continuation_v1_vm &&
tar -xzf cctv-codeformer-spatial-continuation-v1-execution.tar.gz -C ~/forensic-dgp
```

## 3. Open tmux

```bash
tmux new-session -A -s codeformer_spatial96_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_spatial_continuation_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_codeformer_spatial_continuation_v1.py --root . --protocol-sha c355679d2475a65e19338a6c4fa7e8682b0c154442070e06b92a073037ea50ed --verify-transfer &&
bash run_spatial_continuation.sh c355679d2475a65e19338a6c4fa7e8682b0c154442070e06b92a073037ea50ed
```

The Bash launcher is at the **installed experiment root**. Use exactly `bash run_spatial_continuation.sh` as shown. The initial checks must reproduce both saved update-24 image and feature outputs before any additional optimization. The terminal reports absolute updates up to 96. To detach while it runs, press **Ctrl+B**, release, then **D**; reattach with the same command from step 3.

Completion of packaging is indicated by an export receipt containing `"complete": true` and `"archive_sha256"`, followed by `export_exit: 0`. A completed export can also contain stopped branches or a failed worker; preserve its results and traceback for review. Do not append more updates, overwrite this experiment or retry it unchanged.

## 5. Download from Windows Google Cloud SDK Shell after export finishes

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-spatial-continuation-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-spatial-continuation-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-spatial-continuation-v1-export.json" "."
```

Send **downloaded** with the final tmux output. The next audit must check actual checkpoint contents, RAW/PNG preservation and visible facial structure. No candidate becomes the app default from training or export completion alone.

The previous findings and limits remain in [the update-24 return review](CCTV_CODEFORMER_SPATIAL_OBJECTIVES_V1_RETURN_REVIEW.md). The paired data here are ten TRAIN observations from six repeated photographs. They do not establish independent native CCTV, final-identity or Zamboanga performance. Completion development remains deferred until restoration is accepted.
