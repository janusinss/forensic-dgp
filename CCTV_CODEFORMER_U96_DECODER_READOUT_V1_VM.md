# U96 decoder-readout pilot: five manual steps

**Prepared, unrun.** This is one new finite training experiment, not another run
of the failed post-fusion recipe. It starts from complete U96 and trains only its
two existing final RGB convolution tensors. U96 remains the incumbent until the
returned images, RAW/PNG preservation, broader DEV and human review justify change.

Archive: `cctv-codeformer-u96-decoder-readout-v1-execution.tar.gz` (39,462,364 bytes).
Archive SHA256: `8ef7ceeb18379d98db50c250c84afd6e8a3cae3332ad75932ab0ae138423bed2`.
Protocol SHA256: `a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e`.

At most64optimizer updates, not64epochs. Four role cases per update; maximum712
full forwards. Planning estimate10–20minutes is unmeasured for this new path.
Worker stops at30minutes internally; supervisor at31minutes plus at most20seconds
termination grace. Export stops at5minutes internally/5.5minutes externally.
Leave **8GiB free after installation**, with a continuing4GiB reserve. The
installer additionally needs about377MB for exclusive initializer copies.
No competing GPU job; the worker checks before importing Torch.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-decoder-readout-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-codeformer-u96-decoder-readout-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Check and install in VM SSH

This copies the exact initializer bytes from the retained smoothanchors-LR stage;
it does not execute that historical pilot. If a test or hash check fails, retain
the output and stop. Do not delete or overwrite a partial install to retry.

```bash
cd ~ &&
sha256sum -c cctv-codeformer-u96-decoder-readout-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights/codeformer_r2.pth &&
test -f ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights/codeformer_u96_adapter.pth &&
test ! -e ~/forensic-dgp/cctv_codeformer_u96_decoder_readout_v1_vm &&
tar -xzf cctv-codeformer-u96-decoder-readout-v1-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/cctv_codeformer_u96_decoder_readout_v1_vm &&
python3 -B research/codeformer_u96_decoder_readout_v1/install_weights.py --root . --source-root ../cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm --protocol-sha a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e
```

Expected installer: `"complete": true`, two exact file copies and zero model calls.
The archive omits these large frozen initializer files intentionally. The local
execution archive alone is not a replacement for the retained VM migration backup.

## 3. Open tmux

```bash
tmux new-session -A -s u96_decoder_readout_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/cctv_codeformer_u96_decoder_readout_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u96_decoder_readout_v1.py --root . --protocol-sha a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e --verify-transfer &&
bash scripts/run_learning.sh a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e
```

The first receipt reports261staticfiles,96TRAINcases and24legacy guards with zero
model calls. The supervisor prints a child PID and the `run.log` path, then waits.
Progress goes into that log. To view progress in a second VM SSH window:

```bash
tail -F ~/forensic-dgp/cctv_codeformer_u96_decoder_readout_v1_vm/run.log
```

Detach tmux with **Ctrl+B**, then **D**. Reattach with the command in step3.
Do not launch again while it runs. Snapshot checks can stop it at0/1/4/16/32/64;
update16 also requires a positive spatial trend. A quality stop is a retained
experimental result, not an instruction to add epochs. Initial preservation
failures can stop before training. A completed export is still expected after a
worker failure. Download that evidence for independent audit.

Successful transport prints `"complete": true` and `"archive_sha256"` for export,
followed by `worker_exit` and `export_exit`. Worker exit2means quality stop;
exit124means timeout. Export exit0does not qualify the model.

## 5. Download from Windows Google Cloud SDK Shell

Use three separate commands because Windows PuTTY refuses multiple remote sources.
Wait for the supervisor's export result before downloading.

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-decoder-readout-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-decoder-readout-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-codeformer-u96-decoder-readout-v1-export.json" "."
```

If export fails, retain the terminal output and partial archive. Do not rerun
training or overwrite the partial export; its `export_failure.json`, supervisor
and status files must be inspected first. This guide starts no VM job automatically.
The named project/instance/zone is the existing user-identified L4 target; this
preparation did not connect to the VM or verify its current free space.

After download, independently verify transfer/state/RAW/PNG/gates and inspect all
planned images. No automatic promotion. Native/wild samples remain unpaired;
FINAL input pixels remain reserved and independent reviewers are still required.
