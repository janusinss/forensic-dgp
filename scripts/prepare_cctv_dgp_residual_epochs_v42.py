"""Freeze a new finite manual packet, never execute training or contact the VM."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40'
NAME = 'cctv_dgp_residual_epochs_vm_v42'
STEM = 'cctv-dgp-residual-epochs-v42'
OUT = ROOT/'outputs'/NAME
PREP = ROOT/'outputs/cctv_dgp_residual_epochs_v42_preparation'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f: f.write(value)


def write(path, value): text(path, json.dumps(value, indent=2, allow_nan=False)+'\n')


SHELL = '''#!/usr/bin/env bash
set -uo pipefail
PIN="${1:?Pass the exact V42 protocol SHA256}"
ROOT="$HOME/forensic-dgp/cctv_dgp_residual_epochs_vm_v42"
PYTHON="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
WORKER="$ROOT/scripts/cctv_dgp_residual_epochs_v42_vm.py"
test -n "${TMUX:-}" || { printf 'Launch inside tmux.\\n'; exit 1; }
cd "$ROOT" || exit 1
test ! -e outputs && test ! -e trainer.log && test ! -e supervisor_receipt.json && test ! -e export_manifest.json || exit 1
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --verify-transfer || exit 1
START=$("$PYTHON" -c 'import time; print(time.monotonic())')
timeout --signal=TERM --kill-after=30s 7230s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --run 2>&1 | tee trainer.log
CODE=${PIPESTATUS[0]}
printf '%s\\n' "$CODE" > trainer_exit_code.txt
ELAPSED=$("$PYTHON" -c 'import sys,time; print(time.monotonic()-float(sys.argv[1]))' "$START")
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --record-supervision --elapsed "$ELAPSED" --exit-code "$CODE" || exit 1
printf 'Trainer exit code: %s\\n' "$CODE"
timeout --signal=TERM --kill-after=30s 930s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --export
EXPORT_CODE=$?
printf 'Export exit code: %s\\n' "$EXPORT_CODE"
if [ "$CODE" -ne 0 ]; then exit "$CODE"; fi
exit "$EXPORT_CODE"
'''


def guide(pin, digest, size):
    return f'''# V42: finite DGP correction-supervision study

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
The packet is {size:,} bytes. It is self-contained for code, inputs, original
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

Protocol SHA256: `{pin}`
Execution archive SHA256: `{digest}`

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

3. Open tmux:

```bash
tmux new-session -A -s dgp_residual_epochs_v42
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_residual_epochs_v42_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v42.sh {pin}
```

The transfer check makes zero neural or training calls. The manual run then
checks hardware/idle state, caches TRAIN, checks correction gradients and
writes snapshot 0 before optimizing. Detach with Ctrl+B, release, D; reattach
with `tmux attach-session -t dgp_residual_epochs_v42`. A failure still exports
its logs and weights. Retain the assertion, stop receipt and original outputs.

5. Download from **Windows Google Cloud SDK Shell**, one remote file per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
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
'''


def main():
    start = time.monotonic(); assert not OUT.exists() and not PREP.exists()
    old = read(BASE/'protocol.json')
    analysis = read(ROOT/'outputs/cctv_dgp_residual_supervision_v1/analysis.json')
    audit = read(ROOT/'outputs/cctv_dgp_residual_supervision_v1/independent_audit.json')
    assert audit['complete'] and audit['cases_checked'] == 145
    assert audit['analysis_sha256'] == sha(ROOT/'outputs/cctv_dgp_residual_supervision_v1/analysis.json')
    assert sha(ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == old['original_checkpoint_sha256']
    for name, digest in old['assets_sha256'].items(): assert sha(BASE/name) == digest, name
    for name, digest in analysis['source_bindings'].items(): assert sha(ROOT/name) == digest, name
    assert analysis['native_DEV_or_reserved_final_used'] is False and not analysis['model_capacity_or_preservation_pass']
    OUT.mkdir(); PREP.mkdir(); mapping = {}
    def copy(source, name):
        destination = OUT/name; assert destination.resolve().is_relative_to(OUT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, destination)
        assert sha(source) == sha(destination)
        mapping[name] = source.relative_to(ROOT).as_posix()
    for name in old['assets_sha256']:
        if name.startswith('data/') or name.startswith('models/') or name.startswith('weights/') or name in \
            ['cctv_dgp_frozen_norm.py', 'cctv_dgp_pilot.py', 'dgp_face_restoration.py',
             'dgp_frozen_inference_v2.py', 'dgp_mean_centered_inference_v29.py',
             'frozen_definitions.py', 'untrained_initial_decoder.pth']:
            copy(BASE/name, name)
    copy(BASE/'cctv_dgp_spatial_fit_v40_contract.py', 'frozen_capacity_contract.py')
    copy(ROOT/'scripts/cctv_dgp_actual_step_review_v1_metrics.py', 'frozen_raw_metrics.py')
    copy(ROOT/'cctv_dgp_residual_supervision_v1.py', 'cctv_dgp_residual_supervision_v1.py')
    for name in ['cctv_dgp_residual_epochs_v42_contract.py', 'cctv_dgp_residual_epochs_v42_objective.py',
                 'cctv_dgp_residual_epochs_v42_data.py', 'cctv_dgp_spatial_decoder_v42.py']:
        copy(ROOT/'scripts'/name, name)
    copy(ROOT/'scripts/cctv_dgp_residual_epochs_v42_vm.py', 'scripts/cctv_dgp_residual_epochs_v42_vm.py')
    text(OUT/'scripts/run_v42.sh', SHELL)
    refs = {}; cases = old['cases']
    for index, c in enumerate(cases): refs.setdefault(c['source_person_or_reference'], {})[c['profile']] = index
    assert len(refs) == 781
    rng = random.Random(420042); batches = []; orders = []
    profiles = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
    for epoch in range(5):
        ids = sorted(refs); rng.shuffle(ids); orders.append(ids)
        batches.extend([[refs[rid][profile] for profile in profiles] for rid in ids])
    write(OUT/'schedule.json', {'seed': 420042, 'epochs': 5, 'batches': batches,
          'reference_order_per_epoch': orders, 'clear_controls_per_batch': 1,
          'five_complete_passes': True})
    sys.path.insert(0, str(OUT))
    from cctv_dgp_residual_epochs_v42_contract import BUDGETS, LOSS_WEIGHTS, validate_schedule, validate_protocol
    validate_schedule(cases, read(OUT/'schedule.json'))
    for path in OUT.rglob('*.py'): ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    assets = {q.relative_to(OUT).as_posix(): sha(q) for q in OUT.rglob('*') if q.is_file()}
    local_names = set(mapping.values()) | {'scripts/prepare_cctv_dgp_residual_epochs_v42.py',
        'scripts/verify_cctv_dgp_residual_epochs_v42_packet.py',
        'scripts/audit_cctv_dgp_residual_epochs_v42_return.py',
        'CCTV_DGP_RESIDUAL_SUPERVISION_V1_REVIEW.md', 'CCTV_DGP_CURRENT_TRAINING_IMPROVEMENT_PLAN.md',
        'outputs/cctv_dgp_residual_supervision_v1/analysis.json',
        'outputs/cctv_dgp_residual_supervision_v1/independent_audit.json',
        'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json',
        'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'}
    p = {'format': 'own-DGP-direct-residual-supervision-finite-epochs-v42',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'Directly learn mean-centered pre-clamp RGB corrections from approved paired TRAIN; preserve current DGP appearance. Arithmetic is verified, learned capacity is unproven.',
        'cases': cases, 'references': old['references'], 'preview_case_ids': old['preview_case_ids'],
        'updates': 3905, 'epochs_maximum': 5, 'snapshots': [0, 50, 781, 1562, 3905],
        'decoder_parameters': 17952, 'decoder_tensors': 57, 'parameter_layout': old['parameter_layout'],
        'initial_states': old['initial_states'], 'original_checkpoint_sha256': old['original_checkpoint_sha256'],
        'recognizer_weights_sha256': old['recognizer_weights_sha256'],
        'audited_normalizers': analysis['normalizers_initial40_degraded'], 'normalizer_floor': .001,
        'loss_weights': LOSS_WEIGHTS, 'early_gain': .01, 'final_gain': .1,
        'retained_capacity_gates': old['retained_capacity_gates'],
        'optimizer': {'type': 'AdamW', 'learning_rate': .0003, 'weight_decay': .01,
                      'gradient_clip_norm': 1., 'AMP': False, 'EMA': False},
        'scheduler': {'type': 'MultiStepLR', 'milestones_updates': [1562], 'gamma': .3},
        'budgets': BUDGETS, 'timing_safety_factor': 1.25,
        'gradient_preflight': {'case_ids': old['preview_case_ids'], 'batches': 10,
                               'component_queries': 30, 'all57_nonzero_before_optimizer': True},
        'original_DGP_and_initial_reference_frozen': True,
        'clear_teacher': 'Zero mean-centered correction to current DGP',
        'degraded_teacher': 'Audited bounded mean-centered paired RGB correction before clamp',
        'forward_inputs': ['degraded_RGB_256', 'observed_support'],
        'arithmetic_audit_sha256': sha(ROOT/'outputs/cctv_dgp_residual_supervision_v1/independent_audit.json'),
        'source_overlap_limits': 'Unchanged photographic roles; full historical identity overlap remains unestablished. Native development/final IDs do not enter TRAIN.',
        'raw_storage': 'All3905 PNG and mean-only outputs/raw and PNG embeddings/metrics;50 raw float32 previews per snapshot; remaining raw arrays hashed only',
        'migration_states': 'Decoder, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, next schedule index, source hashes, environment and failures',
        'no_failed_recipe_resume': True, 'automatic_follow_on': False, 'manual_tmux_required': True,
        'native_or_reserved_used': False, 'pretrained_restoration_targets_used': False,
        'local_gradient_queries': 0, 'local_optimizer_updates': 0, 'VM_connections': 0,
        'training_launched': False, 'app_promotion': False, 'goal_complete': False,
        'copied_source_mapping': mapping, 'assets_sha256': assets,
        'local_sources_sha256': {name: sha(ROOT/name) for name in sorted(local_names)}}
    validate_protocol(p); write(OUT/'protocol.json', p); pin = sha(OUT/'protocol.json')
    archive_path = ROOT/'outputs'/(STEM+'-execution.tar.gz'); assert not archive_path.exists()
    with tarfile.open(archive_path, 'w:gz', compresslevel=1) as tar:
        for q in sorted(OUT.rglob('*')):
            if not q.is_file(): continue
            info = tar.gettarinfo(str(q), arcname=NAME+'/'+q.relative_to(OUT).as_posix())
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0; info.mode = 0o644
            with q.open('rb') as f: tar.addfile(info, f)
    digest = sha(archive_path); size = archive_path.stat().st_size
    text(Path(str(archive_path)+'.sha256'), digest+'  '+archive_path.name+'\n')
    text(ROOT/'CCTV_DGP_RESIDUAL_EPOCHS_V42_VM.md', guide(pin, digest, size))
    write(PREP/'prepared.json', {'complete': True, 'protocol_sha256': pin,
        'archive_sha256': digest, 'archive_bytes': size, 'assets': len(assets),
        'epochs_maximum': 5, 'updates_maximum': 3905, 'cases': 3905, 'references': 781,
        'seconds': time.monotonic()-start, 'neural_calls': 0, 'gradient_queries': 0,
        'optimizer_updates': 0, 'VM_connections': 0, 'training_launched': False,
        'independent_packet_audit_pending': True, 'goal_complete': False})
    print({'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest,
           'archive_bytes': size, 'neural_or_training_calls': 0}, flush=True)


if __name__ == '__main__': main()
