# V40 endpoint learning diagnostic: five manual steps

V40 stopped at50/800 with **0.0106079234% structure gain**, below the retained
**1%** requirement. Its original decoder, stopped checkpoint and failed gate
remain. Export success packaged evidence; it did not qualify the model.

The saved-weight analysis independently confirms all57 tensors changed. A
distinct diagnostic compares initial/stopped50 states on two metadata-selected
50-case TRAIN cohorts: the fixed previews, unused by the first50 updates, and
the first five optimized references per source. Each reference has one clear
control and the same four degradations. Both cohorts are TRAIN, not evaluation.
Source labels do not establish ethnicity. All visible facial features stay in scope.

**280 component-gradient queries; zero optimizer updates, backwards or epochs.**
Architecture, seven losses, weights and initial50 normalizers remain unchanged.
Endpoint gradient norms/conflicts and actual50-update displacement projections
are diagnostic evidence, not a reconstruction of AdamW or a quality pass. No
loss/learning-rate sweep, resumed V40, new checkpoint or follow-on training.
The prospective checker audits saved gradients without local differentiation,
all200 raw compositions/300 PNGs and40 frozen CPU endpoint replay cases.

Require the existing idle **NVIDIA L4/g2-standard-4** and **2GiB free**.
The self-contained packet is **239.5MiB**; only the existing
venv is required. Estimated diagnostic **2–6 minutes**, export **1–3 minutes**;
worker600s, external630s+30s kill grace, export300s/external330s+30s grace,
allocated VRAM20GiB and uncompressed return512MiB. Historical disk readings
are not current availability. The diagnostic deletes no research assets.

Protocol SHA256: `62d090226e737151083fb388af04c99ee1813aaee37f2e65ec5420e852155071`
Execution archive SHA256: `90e9d8ee576819602afcc3817ab70b1f5dd53c5d38fb167fe94bcc73526b38c9`

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-v40-learning-signal-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-v40-learning-signal-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-v40-learning-signal-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_v40_learning_signal_v1_vm &&
tar -xzf cctv-dgp-v40-learning-signal-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_v40_learning_signal_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_v40_learning_signal_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_v40_learning_signal_v1_vm.py --root . --protocol-sha 62d090226e737151083fb388af04c99ee1813aaee37f2e65ec5420e852155071 --verify-transfer &&
bash scripts/run_learning_signal.sh 62d090226e737151083fb388af04c99ee1813aaee37f2e65ec5420e852155071
```

Detach with **Ctrl+B**, release, then **D**. Step3 reconnects. The completed
log ends at `gradient_queries:280`, followed by diagnostic/export exit codes.
An export `complete:true` reports packaging only. Download failures too;
retain the stop and do not edit, resume or repeat the packet unchanged.

5. Download from **Windows Google Cloud SDK Shell** after export completes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v40-learning-signal-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v40-learning-signal-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-v40-learning-signal-v1-export.json" "."
```

Each SCP has one remote source for Windows PuTTY. The independent prospective
audit executes verified local frozen definitions, never returned Python. New
actual training still requires a separate justified finite manual VM packet.
The1%-at50/10%-at800 and all17 preservation gates remain unchanged. Useful native
restoration, automatic/assisted quality for all seven covering families,
independent final review and the qualified DGP-led app flow remain outstanding.
The app, original checkpoints, splits, caches/local backup and failures remain.
Goal active/incomplete.

[V40 audited stop](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FIT_V40_RESULTS.md>)
[Diagnostic basis and limits](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_REVIEW.md>)
