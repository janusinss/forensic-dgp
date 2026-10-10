# Actual DGP parameter-step review: five manual steps

**Prepared and verified; not run. This is inference only, with zero optimizer
updates and zero gradient queries.** V41 remains a failed treatment at50/800:
0.0516710943% structure gain versus the unchanged1% requirement, plus one
delivered preservation failure. This packet investigates that failure before
choosing another training recipe. It does not resume V41 or create a trained
checkpoint.

The review uses ten recorded V41 before-states. At each state it compares the
unchanged parameters, the actual recorded step, and one fixed saved-array cone
proposal. It retains both unchanged preselected50-case TRAIN cohorts and the
five current-batch cases:30 conditions,3,150 output slots,145 unique cases and29
references. The ten probe states were selected after inspecting TRAIN failures;
they are diagnostic cases, not independent evaluation. All seven objective
terms, visible facial features and original preservation allowances remain.
Float32 rounding is recorded separately from the mathematically intended step.
First-order feasibility does not promise useful finite outputs.

Every slot retains raw float32 RGB, raw/delivered embedding vectors, PNG output
and a mean-only control. The unchanged proposal additionally retains original
DGP floats for independent objective checking. Delivered pixels use the same
floor(raw*255) conversion; pixels outside support copy the input exactly. No
display sharpening is added. The existing trained DGP, reference decoder and
recognizer remain frozen. Temporary proposal assignment is inference, not
optimization; the disposable decoder is restored to its initial state at exit.

Use the existing **NVIDIA L4/g2-standard-4** and existing venv. Require **6 GiB
free after installation**. Archive size:197,588,115bytes, about188.43MiB.
Estimated runtime: **10–25 minutes**, plus **1–15 minutes** export. This is an
estimate from V41's measured485-second/3,905-case snapshot, with additional raw
storage and objective overhead; it is not a measured run of this diagnostic.
Enforced limits: cache300s, review1,500s, worker1,800s, external worker1,830s,
export900s/external930s, allocated VRAM20GiB and encoded return3GiB. The first315
slots must project within1,500s with a1.25 safety factor. A stop retains partial
evidence and attempts export. No automatic follow-on or cleanup is performed.

Protocol SHA256: `339c20425b6fa6b4141001aae4ff3010bf9d438acc32eb71ec08f04f001476f5`
Execution archive SHA256: `ff0ef28764cfb9218f4ec752a5e35dc325c6d896386e78dabc579fe7a4374141`

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-actual-step-review-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-actual-step-review-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-actual-step-review-v1-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_actual_step_review_v1_vm &&
tar -xzf cctv-dgp-actual-step-review-v1-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_actual_step_review_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_actual_step_review_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/review_cctv_dgp_actual_steps_v1_vm.py --root . --protocol-sha 339c20425b6fa6b4141001aae4ff3010bf9d438acc32eb71ec08f04f001476f5 --verify-transfer &&
bash scripts/run_actual_step_review.sh 339c20425b6fa6b4141001aae4ff3010bf9d438acc32eb71ec08f04f001476f5
```

Detach with **Ctrl+B**, release, then **D**. Keep the VM running through export.
The supervisor prints child/log information; per-proposal progress is in
`~/forensic-dgp/cctv_dgp_actual_step_review_v1_vm/review.log`.
From another VM SSH terminal, inspect it with:

```bash
tail -F ~/forensic-dgp/cctv_dgp_actual_step_review_v1_vm/review.log
```

`review_exit_code:0` means the finite review finished. Export `complete:true`
means an archive and checksum were produced, including after a failed review.
Neither qualifies a model. Preserve a failed stop; do not rerun unchanged.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-actual-step-review-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-actual-step-review-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-actual-step-review-v1-export.json" "."
```

One remote source per command is required by the Windows PuTTY-backed transfer.
The local prospective checker is frozen before the VM run. It verifies every
raw/PNG/mean-only slot, recomputes metrics and seven objective terms, preserves
exact categorical decisions, and replays150 current-batch cases with frozen CPU
inference. Its finite audit limit is2,400s. Numerical recomputation allowances
are declared in the protocol and do not relax delivered-image gates. A partial
failed return receives a hash-bound import receipt; it cannot receive a complete
scientific-output audit. All returned code remains unexecuted.

Use of earlier "apply the best approach" authorization selects this bounded
diagnostic after the completed applied-step architecture review. Historical
training code and the app remain unchanged; another training recipe requires
the finite evidence first. No native, DEV or reserved-final cases enter this
packet. Paired photographic TRAIN results remain distinct from unpaired CCTV;
source labels do not establish ethnicity or Zamboanga performance.
Useful native restoration, automatic/assisted quality for all seven covering
families and independent final review remain outstanding. The overall goal is
active and incomplete.

The worker disables gradients and freezes all model parameters before inference.
The explicit parameter checks also cover the factory exception described by
[PyTorch's no-grad documentation](https://docs.pytorch.org/docs/2.9/generated/torch.no_grad.html).

[V41 audited failure and design review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V41_OPTIMIZER_REVIEW.md>)
[Preservation coverage trace](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V41_PRESERVATION_TRACE_V1.md>)
[Independent packet audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_actual_step_review_v1_preparation/independent_packet_audit.json>)
