# Completed worker: repair the export only

The user reported `worker_exit: 0`, last visible `S update24`, followed by `export_exit: 1`. Python3.10 rejected `compresslevel=1` passed to the tar stream constructor. The missing download files come from that packaging failure. **Do not run training again.** Model quality remains unreviewed until the actual returned states and images are audited.

This 13 KB repair uses explicit `gzip.GzipFile` around an uncompressed tar stream, compatible with Python 3.10. It leaves the original training sources/protocol, checkpoints, caches, export manifest and failed partial archive intact. It creates separately named `recovery-r1` download files. The script verifies the terminal worker receipt and result, absence of matching active processes, every manifest member, the full archive readback and all original experiment files before/after. It imports no Torch and performs zero training updates.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "recover_codeformer_spatial_objectives_export_v1_r1.py" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "recover_codeformer_spatial_objectives_export_v1_r1.py.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Export in VM SSH or the existing tmux session

```bash
cd ~ &&
sha256sum -c recover_codeformer_spatial_objectives_export_v1_r1.py.sha256 &&
python3 -B ~/recover_codeformer_spatial_objectives_export_v1_r1.py --self-test &&
python3 -B ~/recover_codeformer_spatial_objectives_export_v1_r1.py --expected-sha 6df8b644a2d4669fadf25781bfd1cd92b5804d76859559f36dccf63c5bc59d38
```

No new training command or environment installation is needed. The whole recovery has a 10-minute cap and preserves 4 GiB free plus bounded archive space. It fails without overwriting anything if another matching process, changed evidence, existing recovery file or insufficient space is detected. Retain the error if it stops; do not delete partials or rerun blindly.

Expected final output contains `"complete": true`, `"archive_sha256"`, `"training_restarted": false` and `"new_neural_gradient_optimizer_training_calls": 0`. This establishes export completion only, not accepted model quality.

## 3. Download the NEW filenames from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-spatial-objectives-v1-results-recovery-r1.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-spatial-objectives-v1-results-recovery-r1.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-spatial-objectives-v1-export-recovery-r1.json" "."
```

Report **downloaded** and paste the final recovery output. The original archive contents and `export.json` manifest are retained unchanged inside the new container, so existing array audits still apply after a safe import. Independent checkpoint tensor/Adam/RNG, feature and RAW/PNG review remain required. No new model is promoted by this repair.

Local evidence: [confirmed user traceback](outputs/codeformer_spatial_objectives_export_recovery_r1_review/confirmed_failure_v1.json), [verified transfer](outputs/codeformer_spatial_objectives_export_recovery_r1_review/transfer_v1.json), [independent format fixtures](outputs/codeformer_spatial_objectives_packet_review_v1/recovery_independent_format_fixture_v1.json), [independent source review](outputs/codeformer_spatial_objectives_packet_review_v1/recovery_independent_source_review_v1.json). Local execution tested the tiny format round-trip on Python 3.13, checked Python 3.10 syntax and documented APIs, and imported no Torch. Actual VM recovery has not yet been observed. The original incompatibility is confirmed by the user's Python 3.10 traceback and the [Python 3.10 tarfile documentation](https://docs.python.org/3.10/library/tarfile.html).
