# V42: finite DGP correction-supervision study

Start from the current app DGP, SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Train our own 17,952-parameter spatial reconstruction decoder with direct
mean-centered correction targets before the output clamp. The original DGP,
stored normalization, fixed initial decoder and recognizer remain unchanged.
These are additional epochs of the spatial decoder. The original DGP's
ancestral epoch count remains unconfirmed. No pretrained restoration model is
used as a training target or primary substitute.

**Five epochs maximum, 3,905 updates.** Each epoch covers all 781 photographic
TRAIN references and their five profiles once. Compare snapshots at updates
0, 50, 781 (epoch 1), 1,562 (epoch 2), and 3,905 (epoch 5). The first failed
structure or preservation requirement stops the run. An early stop is the
declared result, not a reason to bypass the assertion or repeat the recipe.
The same 1% early and 10% final structure thresholds, both-source nonregression,
all 17 preservation groups and 20% brightness-only limit apply. Separate raw
and PNG gates are enforced. No DEV, native CCTV or final identity enters training.
Native development outputs are reviewed only after the independent return audit.

The objective has observed RGB, landmark RGB and three-scale RGB correction
losses with weights 1, 1 and 0.25 and frozen initial-40 degraded scales. Clear
cases teach zero correction at multiplier 4; degraded cases use multiplier
1.25. Absolute degraded identity loss has weight 0.1; the existing identity,
pixel and SSIM regression penalties have weights 5, 2 and 5. These penalties
are training signals, not preservation guarantees. AdamW starts at 0.0003,
weight decay 0.01, gradient clipping 1; after update 1,562 the rate becomes
0.00009. These are declared choices, not established optimal hyperparameters.
Before any optimizer, all 57 tensors must have finite nonzero reconstruction
gradients on the fixed 50-case TRAIN cohort. No failed V40/V41 state is resumed.

Require the idle existing NVIDIA L4/g2-standard-4 and **8 GiB free after install**.
The packet is 444,114,939 bytes. It is self-contained for code, inputs, original
weights, recognizer and initial decoder; the existing VM venv is reused. Model
and optimizer/scheduler/RNG/schedule states are saved at snapshots and stops.
They support a later reviewed migration, not automatic resume of a failed gate.
The historical local research-cache backup remains separately retained.

Estimated training 45–110 minutes, export 3–15 minutes; this is a preparation
estimate, not measured V42 runtime. Enforced limits: cache 900 seconds, fit
6,300 seconds, worker 7,200 seconds, external worker 7,230 seconds plus 30-second
kill grace, export 900 seconds/external 930 seconds plus grace, peak allocated
VRAM 20 GiB, return contents 3.5 GiB, protected disk reserve 512 MiB. After
snapshot 0, reserve outputs plus a portable archive. At update 20, project the
remaining updates and four measured snapshots with a 1.25 safety factor. Stop
on a projection or actual-limit failure. This packet deletes no file.

Protocol SHA256: `c19ca1790ae9ebe7f99000a81678c8fc756d70ce2114debb6f81691c741a6f2d`
Execution archive SHA256: `9988d57e0c3733df0b255c93282b284dd13b98e296fe605de101f3c2972ddf4c`

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-residual-epochs-v42-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-dgp-residual-epochs-v42-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c cctv-dgp-residual-epochs-v42-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/cctv_dgp_residual_epochs_vm_v42 &&
tar -xzf cctv-dgp-residual-epochs-v42-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

3. Open tmux:

```bash
tmux new-session -A -s dgp_residual_epochs_v42
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/cctv_dgp_residual_epochs_vm_v42 &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_residual_epochs_v42_vm.py --root . --protocol-sha c19ca1790ae9ebe7f99000a81678c8fc756d70ce2114debb6f81691c741a6f2d --verify-transfer &&
bash scripts/run_v42.sh c19ca1790ae9ebe7f99000a81678c8fc756d70ce2114debb6f81691c741a6f2d
```

The transfer check makes zero neural or training calls. The manual run then
checks hardware/idle state, caches TRAIN, checks correction gradients and
writes snapshot 0 before optimizing. Detach with Ctrl+B, release, D; reattach
with `tmux attach-session -t dgp_residual_epochs_v42`. A failure still exports
its logs and weights. Retain the assertion, stop receipt and original outputs.

5. Download from **Windows Google Cloud SDK Shell**, one remote file per command:

```bat
cd /d "C:\xampp\htdocs\YEAR 4\Testing\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-residual-epochs-v42-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-residual-epochs-v42-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-residual-epochs-v42-export.json" "."
```

Return all three files and the terminal output. Export `complete: true` means
the archive was produced; it does not mean training, preservation or quality
passed. The frozen return checker rechecks every delivered PNG metric, all
saved raw aggregates and 50 raw inference previews per snapshot. Other raw
float arrays are hashed rather than retained, so their pixel values are not
independently recomputed by that checker. CPU/model and embedding replay
tolerances are separate from the unchanged scientific gates. Review every
preview and native development case before any app decision. Five-epoch
completion alone cannot qualify restoration or seven-family completion.

No training, VM connection, app promotion or final evaluation occurred while
preparing this packet. Original checkpoints, splits, caches and failures remain.
No ethnicity, exact hidden identity or Zamboanga performance is inferred.
The broader goal remains active/incomplete.
