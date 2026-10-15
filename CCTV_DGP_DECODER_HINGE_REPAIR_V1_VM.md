# USED: hinge repair audited; quality stop retained — 15 October 2026

The downloaded manual return is fully imported and audited. The inactive
brightness-gradient correction works in all 20 saved controls. The new model
still fails after **one Adam update**: **0.113913% RAW / 0.128470% PNG** structure
proxy gain, with **30 RAW / 30 PNG preservation failures**. Both smaller
diagnostic changes also fail. All 20 visual sheets / 800 cells are reviewed
and bound to independently audited pixels. No model qualifies or replaces the
original baseline; no extra epoch, unchanged rerun or optimizer resume follows.

The first local audit's one-ULP reserve-metadata stop is preserved. A separate
strict R1 checker completes all 600 compositions without widening tolerances
or changing vectors, radius or quality gates. Six regression tests pass.

The user's diagnostic answer selects the best supported approach. The next
review examines **one-sided preservation-penalty activation and feasible update
directions**: the tested smaller change reduces reconstruction terms but adds
larger regression penalties. This is saved TRAIN evidence, not native CCTV or
final qualification. No new VM execution packet or training is certified.

[Audited result and limitations](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DECODER_HINGE_REPAIR_V1_RESULTS.md>).
[Selected next causal review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_HINGE_REPAIR_REVIEW_V1.md>).
The full five-milestone goal, seven automatic/assisted covering families,
manual-training boundary and original checkpoints/splits/failures remain intact.
The instructions below are retained historical state; this USED status takes
precedence over their PREPARED/UNRUN labels and upload/run commands.

# Decoder brightness-hinge correction — manual L4 commands

**VERIFIED / PREPARED / UNRUN.** This fresh packet corrects the demonstrated
brightness-penalty import regression. It keeps Adam at0.0001 so the intended
loss change is isolated. Actual correction effectiveness and model improvement
remain unverified. The original app checkpoint stays unchanged.
[Completed replay and discovered cause](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_PATH_REPLAY_V1_RESULTS.md>).

Require **8 GiB free after installation** and **8 GiB available RAM** before
models. Reuse the existing three research roots: spatial_readout_v1,
bank_photometric_guard_v1 and spatial_path_replay_v1. Do not remove their assets.
Upload only **122,323 bytes**. Estimate10–30minutes including
saved audit/export; the new CUDA runtime is unmeasured. Enforced maxima including
kill grace total63minutes plus hash-only verification: worker2400/external2430s,
saved audit600/external630s, export600/external630s,30s kill grace per stage.
VRAM20GiB/RSS12GiB/output3GiB/free reserve1GiB; every limit is enforced.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-decoder-hinge-repair-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-decoder-hinge-repair-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-decoder-hinge-repair-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_spatial_readout_v1_vm &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test -d ~/forensic-dgp/cctv_dgp_spatial_path_replay_v1_vm &&
test ! -e ~/forensic-dgp/cctv_dgp_decoder_hinge_repair_v1_vm &&
tar -xzf cctv-dgp-decoder-hinge-repair-v1-execution.tar.gz -C ~/forensic-dgp
```

Expected checksum:`OK`. The sidecar is verified LF UTF-8 without BOM/CRLF.
If a required root is missing or the new destination already exists, stop and
preserve it; do not remove evidence or bypass the guard.

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_decoder_hinge_repair_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_decoder_hinge_repair_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py --root . --protocol-sha a4d57c933214345a6d1cab5639007d15599f85e0db0450a0aad11aea5d715518 --verify-transfer &&
bash scripts/run_decoder_hinge.sh a4d57c933214345a6d1cab5639007d15599f85e0db0450a0aad11aea5d715518
```

The hash-only preflight checks18 packet assets,410 source assets,203 parent
assets and136 saved-replay assets, with zero neural/gradient/optimizer calls.
The supervised phase then verifies exact original/A0/legacy-A1 outputs and all20
corrected inactive penalty gradients before any optimizer. It saves one corrected
update plus two smaller same-direction diagnostic controls. Those controls never
select a model. Continuation permits **at most50updates**, with preservation and
positive source/global structure checks at1,10,20,30,40,50. The final≥1% requirement
stays unchanged. These are small TRAIN-sample updates, not full-corpus epochs.

`Ctrl+B`, then `D`, detaches tmux safely. A retained quality stop returns exit2;
it is not a reason to rerun or add epochs. A technical failure retains its
traceback/state. Saved audit and export still run. Archive `complete:true` means
packaging, never training or model-quality success. Download all three files.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-decoder-hinge-repair-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-decoder-hinge-repair-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-decoder-hinge-repair-v1-export.json" "."
```

Each SCP has one remote source, compatible with Windows PuTTY. Preserve any
partial export, failed logs and all original checkpoints/caches/splits.
[Independent local return workflow](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DECODER_HINGE_REPAIR_V1_LOCAL_RETURN.md>).
