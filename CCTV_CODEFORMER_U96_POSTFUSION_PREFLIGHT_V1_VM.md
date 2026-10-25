# Manual U96 post-fusion preflight V1

21 October 2026. **Prepared and independently byte-verified; NOT RUN.**

This is a graph/initialization diagnostic, not a training launch or a new trained
model. It preserves the exact full U96 initializer (R2 base plus U96 adapter),
adds zero-output heads, and permits **20 full forwards, 4 gradient queries and
0 optimizer constructions/updates** on four TRAIN photographs. No native, DEV,
FINAL or wild image is used in this diagnostic. U96 remains the incumbent.

Require **6 GiB free on the VM after installation**, an idle **NVIDIA L4**, and
the existing virtual environment. Current VM status and free space were not
queried during preparation. Runtime is not benchmarked: internal worker cap
20 minutes, external worker cap21 minutes, termination grace up to20 seconds;
export has a330-second internal and345-second external cap. Each worker phase
is capped at180 seconds. Hash verification adds file-I/O time. Other limits:
4 GiB continuing disk reserve,10 GiB peak/current host RSS,18 GiB CUDA reserved,
128 MiB diagnostic output,160 MiB returned archive. A limit failure must be kept.

The transfer archive is **351,541,768 bytes** (about335 MiB).
All40 static files and41 archive members were independently checked. The actual
Linux/Python3.10/L4 neural runtime is still unverified.

Archive SHA256: `c01c6772b82dc811a50de5d4178b2d4189a250ab59a58da36d217aafa9f1337b`  
Protocol SHA256: `1ebf080ba7a6bae090df7c776ad17cf145ad43f3c0e090b5b401028b655bda07`

## 1. Upload — Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-postfusion-preflight-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-postfusion-preflight-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install — VM SSH

```bash
cd ~ &&
sha256sum -c cctv-codeformer-u96-postfusion-preflight-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_vm &&
tar -xzf cctv-codeformer-u96-postfusion-preflight-v1-execution.tar.gz -C ~/forensic-dgp
```

Checksum text is generated as ASCII with LF endings. Preserve a failed install
or any pre-existing stage; do not overwrite it or rename old evidence to rerun.

## 3. Open tmux — VM SSH

```bash
tmux new-session -A -s u96_postfusion_preflight_v1
```

## 4. Launch — inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u96_postfusion_preflight_v1.py --root . --protocol-sha 1ebf080ba7a6bae090df7c776ad17cf145ad43f3c0e090b5b401028b655bda07 --verify-transfer &&
bash scripts/run_preflight.sh 1ebf080ba7a6bae090df7c776ad17cf145ad43f3c0e090b5b401028b655bda07
```

The first line of supervisor output reports the child PID and `run.log`.
Detailed progress is written to that log. In a **second VM SSH terminal**, view it:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_vm/run.log
```

Press `Ctrl+C` to stop following the log. Detach from tmux with `Ctrl+B`, then
`D`; the diagnostic remains in tmux. Do not launch a second copy.

Wait for the supervisor's final `worker_exit` and `export_exit` message.
`worker_exit:0` and `export_exit:0` mean the preflight and packing returned
successfully; saved evidence still needs the independent local audit.
`complete:true` in the export receipt means the archive was written, not that
the model improved. A nonzero worker exit, timeout or quality/integrity stop
remains a failure even if export succeeds. Download and retain it; do not add
epochs, relax gates or rerun unchanged.

## 5. Download — Windows Google Cloud SDK Shell

Run each single-source command after export finishes. These separate downloads
avoid the Windows PuTTY limitation on multiple remote sources.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-preflight-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-preflight-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-postfusion-preflight-v1-export.json" "."
```

## What happens after download

The local auditor will verify the archive and protocol, all saved parity arrays
and20 PNGs,144 gradient tensors,72 portable initial/final head arrays,
resources/counters and failure records. It never unpickles the retained `.pt`
files. Exact frozen-weight claims depend on the reviewed runtime and its state
receipts because those large baseline tensors are intentionally omitted from the
return. The original local model files remain retained and hash-bound. Root
visual review of both planned sheets is still required after a real return.

Strict deterministic policy remains active for forwards and parity. Only the
four gradient-query contexts permit the documented CUDA bilinear-backward warning;
all warnings and actual gradients are saved, strict flags are restored, and
unexpected nondeterminism warnings stop the run. RNG equality is not a claim of
bitwise reproducible CUDA gradients.

A passing feasibility preflight does not establish learned correction leverage,
actual optimizer-step preservation, useful restoration, native performance or
FINAL quality. Its return selects whether a separate finite, justified learning
pilot is appropriate; no training is automatically authorized or launched.

[Source/compatibility review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_PREFLIGHT_COMPATIBILITY_V1.md>) ·
[Learning-path decision](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_LEARNING_PATH_REVIEW_V1.md>) ·
[Independent packet readback](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_preflight_v1_packet/independent_packet_audit_v1.json>).
