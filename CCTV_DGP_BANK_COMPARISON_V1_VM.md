# Conditioned generative-prior comparison — manual L4 run

Prepared10 October2026. The transfer archive is independently verified and has
not been run on the VM. This is a finite training comparison, not a qualified
replacement or a promise that additional epochs improve CCTV faces.

## What this comparison resolves

Both routes start from the retained checkpoint
SHA256 `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The application and that permanent comparator remain unchanged.

| Route | Learned path | Frozen components |
| --- | --- | --- |
| A | Repaired current reconstruction:15 tensors/609,219 elements; active clear and blur target anchors. | Retained observation pyramid and stored normalization. |
| B | The same reconstruction plus our input-to-style, spatial conditioning and multiscale fusion:37 tensors/3,364,643 elements total. | Retained observation pyramid, normalization and declared external StyleGAN2 face bank. |

B is our proposed hybrid generative-prior restorer. The standalone generator
is pretrained externally; we did not train that generator. No pretrained
restoration encoder or restoration checkpoint is substituted. The design
uses the [GLEAN architecture](https://arxiv.org/abs/2012.00739) as a reference;
it is not an exact reproduction of [published DGP](https://arxiv.org/abs/2003.13659).
See CCTV_DGP_GENERATIVE_PRIOR_METHOD_REVIEW_V2.md for source/licence/overlap limits.

B uses continuous conditioned generator features at16/32/64/128/256 and our
256 RGB reconstruction. It does not subtract generated RGB or use a512 RGB
restoration tail. New fusion projections start at zero to retain initial
appearance. Conditioning therefore has zero initial loss gradients; the VM
must verify finite nonzero conditioning derivatives after the first fusion
update and before update2. The external prior remains frozen throughout.

Both routes use the same fixed objective, Adam learning rate0.0001,50 updates,
and250 distinct TRAIN reference exposures/1,250 profile-image exposures.
Five reference batches, each containing all five profiles, accumulate into
each update. Each source contributes125 references. This is **zero complete
epochs**, approximately0.3201 of the781-reference epoch per route.

Full baseline/A50/B50 raw and delivered PNG checks cover3,905 paired TRAIN
cases. They are TRAIN capacity evidence, not held-out validation. The same24
input-reviewed native CCTV DEV crops are retained separately as unpaired
evidence, without PSNR/SSIM against invented clean targets. Final pixels are
excluded. B's bank-disabled ablation covers the fixed100 TRAIN previews and
24 native DEV cases; it is not claimed to cover all3,905 paired cases.

The original1% early structure requirement, all source/profile preservation
checks, appearance checks and brightness limit remain unchanged in raw and
PNG results. A quality failure preserves A and stops further A updates; B then
starts from its independently declared original initializer. Transfer,
preflight or resource failures stop the comparison. No failed-state resume,
extra epochs, automatic continuation or app promotion is permitted.

## Checks already completed

CPU initialization proof took230.35s and reproduced the retained model exactly
on100 paired TRAIN inputs and24 native DEV inputs. B's internal features
respond to the input, but its new modules remain untrained. Initializers are
not new trained checkpoints.

The20.34s independent checker verified480 files,11 source bindings,124
raw/PNG compositions,66 sheet cells, mean-style recomputation and four fresh
original/A/B CPU replays. Both local gradient-enabled APIs were refused. All
three planned sheets were viewed at original resolution: the inherited
softness remains and there is no quality gain yet.

The84.28s independent transfer check verified all5,611 archive members,
5,610 packet assets and the actual transfer-only CLI. It rejected five adverse
protocol mutations, including lowered gates, final-role leakage, repeated
exposures, removal of the clear anchor and replacement of the baseline.
No local gradients, backward calls or optimizer updates occurred.

Portable return checking uses narrow metadata tolerances for cross-CPU
float32 embedding-dot and float64 mean-shift roundoff. Saved raw/PNG hashes,
roles/counts and original scientific thresholds remain exact; independently
recomputed gate decisions must agree. The original embedded checker is retained.

## Resource and stop limits

Require **16 GiB free after installation**, the existing retained Python
environment, NVIDIA L4/g2-standard-4 and manual tmux. The VM's current free
space and GPU state have not been inspected in this preparation turn.

The worker has a90-minute cap. Initial gradient/measurement work has a5-minute
cap; each route's fit has a10-minute cap and each snapshot a20-minute cap.
The export has a10-minute cap. The supervisor enforces independent deadlines
with30 seconds for termination and a bounded forced-kill fallback.
These are hard limits, not a measured runtime forecast for this new design.

Before any optimizer, the VM checks initial CUDA output parity, a generated
seed's saved CPU/CUDA prior RGB/features, both sources' true gradients, measured
fit/snapshot/whole-worker timing projections, and output/export storage
projection. It stops on a competing GPU process,20 GiB peak allocation,
1 GiB disk reserve or6 GiB result-output cap. The own tmux shell is allowed.

The packet retains inherited terms, source/split metadata, data, old failure
evidence, external licences, code, full-state/RNG/scheduler/optimizer evidence
and negative gates. Emergency termination can leave partial output/state;
an archive or export receipt does not establish successful training or exact
resume. Partial files remain evidence and must be independently checked.

## Upload and launch

Use **Windows Google Cloud SDK Shell** for steps1–2. Use **VM SSH** for steps3–5.

1. Upload the archive:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-comparison-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Upload its checksum:

```bat
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-bank-comparison-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

3. Install the verified new packet in **VM SSH**:

The checksum stream is normalized to Linux line endings, so an already uploaded
Windows CRLF checksum also works. The archive and protocol hashes stay the same.

```bash
cd ~ &&
tr -d '\r' < cctv-dgp-bank-comparison-v1-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_dgp_bank_comparison_v1_vm &&
tar -xzf cctv-dgp-bank-comparison-v1-execution.tar.gz -C ~/forensic-dgp
```

Expected checksum output: `cctv-dgp-bank-comparison-v1-execution.tar.gz: OK`.
The extraction is silent. Continue to step 4 after the command succeeds.

4. Open tmux:

```bash
tmux new-session -A -s dgp_bank_comparison_v1
```

5. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_bank_comparison_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_comparison_v1_vm.py --root . --protocol-sha 0e4f9645abc99f6d980f3b20240b196f65f3e130472795faa2f964df1a5a3b3e --verify-transfer &&
bash scripts/run_bank_comparison.sh 0e4f9645abc99f6d980f3b20240b196f65f3e130472795faa2f964df1a5a3b3e
```

The transfer-only line should report5,610 assets and zero neural/gradient calls.
Training prints route/update or snapshot progress. Exit2 can mean a retained
negative comparison; inspect the exported outcome, not just the shell status.
Use Ctrl+B, then D to detach tmux. Reattach with:

```bash
tmux attach -t dgp_bank_comparison_v1
```

Archive bytes:809,058,558. SHA256:
`ece56acfdf440fc1ced9f5d9d53ba39db8c3a4da48fa0b57bee5346d265297a0`.

## Download after export

Use **Windows Google Cloud SDK Shell**, one remote source per command for PuTTY.
The export receipt must report an archive checksum; `complete:true` only means
the export finished, even when training or a quality gate failed.

1. Download the result archive:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-results.tar.gz" "."
```

2. Download the result checksum:

```bat
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-results.tar.gz.sha256" "."
```

3. Download the export receipt:

```bat
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-bank-comparison-v1-export.json" "."
```

Codex will verify the returned archive/receipt, safely import it without
overwriting the prepared packet, and run the independent portable checker.
The declared review produces20 paired preview sheets and6 native sheets.
Every available planned sheet must be inspected, with missing stopped outputs
declared. Saved metrics, limited CPU model/recognizer replay and bank-disabled
ablation do not replace native visual usefulness or independent final review.

All five goal milestones, seven separately evaluated covering families,
automatic/assisted completion review, downloads and bundled inline Playwright
app verification remain required. This comparison does not complete the goal.
