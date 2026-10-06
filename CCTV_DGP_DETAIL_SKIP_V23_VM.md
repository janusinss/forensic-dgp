# V23 — upload, finite tmux run and download

**Closed 6 October 2026: V23 return independently audited; do not relaunch.**

The original update-50 structure stop is verified: 50 updates/51 backwards,
delivered degraded feature error 0.00223306% worse, no visible gain across all
50 reviewed cases. Packaging succeeded; model capacity did not pass. Original
sources/gates/partial evidence stay intact. The final800 gate was not executed.

V24 prepares a different training-objective experiment with unchanged head,
data, limits and quality gates. Use [V24 commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DEGRADED_DETAIL_V24_VM.md>).
The old V23 steps below are historical. No automatic retry, assistant VM action
or app checkpoint replacement. [Audited result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_RESULTS.md>).


**Latest 6 October 2026 — V23 user-reported early structure stop; download/audit pending:**
The pasted L4 log reports successful source/data/CUDA preflight, exact four-case
original-DGP parity and50 initial cached-DGP checks. Training reaches update50.
Delivered-PNG degraded feature error increases from0.0019965976532523044 to
0.0019966422385438174, approximately0.00223306% worse, instead of the required1%
improvement. The unchanged early condition stops the worker (trainer exit1).
Export exits0 and packages failure evidence; its complete flag is not model success.

Reported return:76,481,832 bytes, SHA256
`b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9`.
The return is not yet present locally. Independently verify hashes, sources,
timing/gradient/state/metric receipts and all images before diagnosing the
underlying model failure. Do not repeat the unchanged V23 launch, relax its gate,
or prepare another recipe from this console log alone. Final800 capacity gate
was not executed; no app acceptance or visible-quality verdict is claimed.

Proceed to step5 in the V23 runbook: three separate Windows gcloud downloads.
Original transfer/protocol/runtime and historical failure gates are unchanged.
Prior document bytes are retained under
`outputs/cctv_dgp_detail_skip_v23_reported_stop_v1/before_docs/`.
No assistant VM action/local training or app change occurs. Goal remains active;
full native restoration, covering-family scope and independent review stay open.
Runbook: [V23](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_VM.md>).


Prepared and checked 6 October 2026. V22 R1 is closed: its stopped outputs show
no visible structure gain. V23 tests a different full-resolution detail path.
This is an exposed-training capacity experiment; useful CCTV restoration and
application acceptance remain unverified.

Archive: **217,943,471 bytes (207.85 MiB), 211 regular members**.
SHA256: `21e5ea32887e99f35e8196f46e862ab102db0ecdeef2285d793ca3432c35fa53`.
Protocol: `4cddb98fb5e6a215cb84f98132cfa5a2146ee157c1cb939919ffbf8c5740e863`.
The bundle includes the original DGP/recognizer weights, cached data and sources.
It needs the existing `cctv_dgp_vm_bundle/.venv`, 3 GiB free disk and an idle L4.
It does not require the old expanded training-cache directories.

These commands require the existing VM to be running. A separate maintenance
handoff records it as stopped on 6 October; availability has not been refreshed
in this restoration turn. No assistant upload, SSH, VM start or training occurred.

1. Upload from **Windows Google Cloud SDK Shell**:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-detail-skip-v23-execution.tar.gz" "cctv-dgp-detail-skip-v23-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-detail-skip-v23-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_detail_skip_vm_v23 &&
tar -xzf cctv-dgp-detail-skip-v23-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_detail_skip_v23
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_detail_skip_vm_v23 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_detail_skip_v23.py --root . --protocol-sha 4cddb98fb5e6a215cb84f98132cfa5a2146ee157c1cb939919ffbf8c5740e863 --preflight &&
bash scripts/run_v23.sh 4cddb98fb5e6a215cb84f98132cfa5a2146ee157c1cb939919ffbf8c5740e863
```

5. Download after the **export receipt** appears, from **Windows Google Cloud SDK Shell**:

```cmd
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-detail-skip-v23-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-detail-skip-v23-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-detail-skip-v23-export.json" "."
```

`Ctrl+B`, then `D` detaches safely. Step 3 reattaches. Progress is displayed and
saved in `~/forensic-dgp/cctv_dgp_detail_skip_vm_v23/trainer.log`.
Run the three downloads separately: the Windows PuTTY backend rejects multiple
remote sources in one command. The upload has local sources and one remote
destination.

The experiment ends at **800 updates/80 epochs**, or earlier at a failed stop.
Preflight has a 5-minute cap, fitting 25 minutes, worker 30 minutes and the outer
supervisor 35 minutes plus 30 seconds to kill a stalled worker. Export has a
2-minute internal cap and a 2.5-minute outer cap plus 30 seconds of kill grace.
These are limits, not a measured V23 runtime estimate. The update-20 projection
must fit the original budget; update 50 must improve degraded feature error by
at least 1%. Final capacity also requires the unchanged appearance safeguards.

An export receipt with `"complete": true` confirms packaging. It does not imply
a training or quality pass. Download failed/partial evidence too; preserve the
traceback, outputs and old checkpoints. Do not relaunch an unchanged failed
recipe. The supervisor exports receipts with timing, step samples, GPU-memory
peak and component forward counts when the worker reaches its receipt stage.
Hard termination before that stage can leave partial evidence.

If preflight rejects the host/GPU, changed files, missing dependencies, another
GPU process, disk space or original-DGP parity, preserve the message. The guard
allows idle tmux sessions and does not stop other tasks or install dependencies.

After return, independently check the archive/source hashes, gradient and timing
receipts, exact saved outputs/metrics, frozen model states and every training
image. Capacity success still needs separately frozen broader development and
canonical app checks, native usefulness and independent final review.

Plan: [V23 detail path](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_PLAN.md>).
Closed result: [V22 R1 audit](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md>).

