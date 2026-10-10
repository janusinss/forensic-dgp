"""Freeze distinct V33 proposal/output probe. No model, optimizer or VM execution."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_dgp_loss_cone_probe_v33_vm'
STEM = 'cctv-dgp-loss-cone-probe-v33'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_preparation'
OLD = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_vm'
RETURN = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_return'
OLD_PIN = '61c04aef4ee184adc9830482477bf37ad9c936a8c0247e44d26eefc1a6bf483d'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def text_file(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream: stream.write(value)


def write(path, value): text_file(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


SHELL = '''#!/usr/bin/env bash
set -uo pipefail
PIN="${1:?Pass the exact V33 protocol SHA256}"
ROOT="$HOME/forensic-dgp/cctv_dgp_loss_cone_probe_v33_vm"
PYTHON="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
WORKER="$ROOT/scripts/cctv_dgp_loss_cone_probe_v33_vm.py"
cd "$ROOT" || exit 1
test ! -e outputs && test ! -e probe.log && test ! -e supervisor_receipt.json && test ! -e export_manifest.json || exit 1
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --verify-transfer || exit 1
START=$("$PYTHON" -c 'import time; print(time.monotonic())')
timeout --signal=TERM --kill-after=30s 930s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --run 2>&1 | tee probe.log
CODE=${PIPESTATUS[0]}
printf '%s\\n' "$CODE" > probe_exit_code.txt
ELAPSED=$("$PYTHON" -c 'import sys,time; print(time.monotonic()-float(sys.argv[1]))' "$START")
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --record-supervision --elapsed "$ELAPSED" --exit-code "$CODE" || exit 1
printf 'Probe exit code: %s\\n' "$CODE"
timeout --signal=TERM --kill-after=30s 330s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --export
EXPORT_CODE=$?
printf 'Export exit code: %s\\n' "$EXPORT_CODE"
if [ "$CODE" -ne 0 ]; then exit "$CODE"; fi
exit "$EXPORT_CODE"
'''


def guide(pin, digest):
    return f'''# V33 finite update probe: five manual steps

The returned V32 loss diagnostic passed independent audit. At its stopped state,
the summed raw loss-gradient direction increases the facial-detail term in both
measured TRAIN cohorts. Preservation gradients remain necessary for measured
regressions. This probe changes update formation on disposable model copies.

**Eight fresh AdamW proposal steps; zero new gradient queries; no continuing
training trajectory or new checkpoint.** Each original/stopped state and exposed/
unexposed cohort resets completely. The eight steps use saved gradients, not new
backpropagation. They do change disposable parameters, so run this manually on
the existing L4 VM. No agent launch or automatic follow-on training occurs.

It compares the original summed-loss proposal, a restoration-only diagnostic
control, and the restoration proposal constrained by all seven nonzero loss
gradients at four fixed scales:1,1/2,1/4,1/8. AdamW's actual displacement is
projected after its adaptive transformation. The original seven loss values,
17 delivered preservation groups, source and brightness limits remain. The
1% at50 and10% at800 capacity requirements remain for later finite training.
The unconstrained control cannot qualify a training recipe. Linear projection,
finite probe completion and packaging cannot establish useful restoration.

Require an idle **NVIDIA L4/g2-standard-4** with **6GiB free**. Estimate **3-10
minutes for the probe plus1-4 minutes for export**; this new finite probe has not
been timed on the VM. Enforced worker900s; external930s plus30s kill grace;
export300s/external330s plus30s grace; allocated VRAM20GiB; return2GiB.
There are exactly1,200 trial outputs plus200 before outputs. Nonfinite values,
changed dependencies, missing assets, count violations or time/space violations
stop and retain evidence. Existing checkpoints, caches, splits and failures stay.

Protocol SHA256: {pin}
Archive SHA256: {digest}

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
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_v32_loss_gradient_v1_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_cone_v33
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{NAME}.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v33_probe.sh {pin}
```

Detach with **Ctrl+B**, release, then **D**. Step3 reconnects. Expected completed
probe: optimizer_updates:8, committed_trajectory_updates:0, raw_outputs:1400.
Export complete:true means packaging. Download retained failures too. Preserve
the run and its assertions; do not rerun it or launch an800-update continuation.

5. Download from **Windows Google Cloud SDK Shell**, after export completes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Each download uses one remote source, as required by Windows PuTTY. Returned
files need independent hash, displacement/moment/projection and output audits
before another training design is justified. This is paired photographic TRAIN
evidence; source labels are not ethnicity, native CCTV or Zamboanga performance.
The application and all seven completion-family requirements remain unchanged.
'''


def main():
    assert not BUNDLE.exists() and not PREP.exists()
    assert sha(OLD / 'protocol.json') == OLD_PIN
    basis = read(OLD / 'protocol.json')
    audit = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_independent_audit.json'
    arithmetic = ROOT / 'outputs/cctv_dgp_v32_restoration_cone_v1/independent_arithmetic_audit.json'
    assert read(audit)['complete'] and read(audit)['diagnostic_complete'] and read(arithmetic)['complete']
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    assets = ['scripts/' + NAME + '.py', 'cctv_dgp_loss_cone_v33.py', 'cctv_dgp_loss_cone_probe_v33_metrics.py']
    for target in assets:
        source = ROOT / 'scripts' / Path(target).name
        text = source.read_text(encoding='utf-8'); ast.parse(text, feature_version=(3, 10)); text_file(BUNDLE / target, text)
    text_file(BUNDLE / 'scripts/run_v33_probe.sh', SHELL); assets.append('scripts/run_v33_probe.sh')
    names = set(basis['assets_sha256']) | {'protocol.json', 'outputs/results.json', 'supervisor_receipt.json', 'diagnostic_exit_code.txt'}
    for state in [0, 50]:
        for cohort in basis['cohorts']:
            folder = f'outputs/state{state}_{cohort["name"]}/'
            names.update(folder + name for name in ['receipt.json', 'gradient_components.npy'])
            for case in cohort['cases']: names.update(folder + case['id'] + suffix for suffix in ['.npy', '.png'])
    mapping = {}
    for name in sorted(names):
        source = OLD / name if name == 'protocol.json' or name in basis['assets_sha256'] else RETURN / name
        mapping[name] = sha(source)
    replays = {}
    for cohort in basis['cohorts']:
        chosen = {}
        for case in cohort['cases']: chosen.setdefault(case['source'], case['source_person_or_reference'])
        ids = [c['id'] for c in cohort['cases'] if c['source_person_or_reference'] in chosen.values()]
        assert len(ids) == 10 and len(chosen) == 2
        replays[cohort['name']] = ids
    p = {'format': 'own-DGP-saved-gradient-finite-loss-cone-probe-v33', 'frozen_UTC': datetime.now(timezone.utc).isoformat(),
         'purpose': 'Test finite output consequences of restoration-priority AdamW displacement projected against existing seven losses.',
         'hypothesis': 'At stopped50, preserving the restoration proposal while constraining all seven loss gradients can improve actual visible structure without violating retained deliverable guards.',
         'basis_protocol_sha256': OLD_PIN, 'basis_VM_sha256': mapping, 'cohorts': basis['cohorts'], 'states': [0, 50],
         'selection_policy': 'Exact two previously frozen50-case source/profile-matched TRAIN cohorts; no new ranking, DEV or final exposure.',
         'terms': basis['terms'], 'parameter_layout': basis['parameter_layout'], 'normalizers': basis['normalizers'],
         'optimizer_updates': 8, 'committed_trajectory_updates': 0, 'new_gradient_queries': 0, 'backwards': 0, 'epochs': 0,
         'AdamW': {'learning_rate': .00003, 'weight_decay': .01, 'betas': [.9, .999], 'epsilon': 1e-8, 'clip_norm': 1.,
                   'fresh_moments_for_each_proposal': True, 'foreach': False, 'fused': False},
         'variants': [{'name': 'summed_adamw', 'proposal': 'summed', 'scale': 1.},
                      {'name': 'restoration_adamw_unconstrained', 'proposal': 'restoration', 'scale': 1.},
                      *[{'name': name, 'proposal': 'projected_restoration', 'scale': scale} for name, scale in
                        [('cone_1', 1.), ('cone_half', .5), ('cone_quarter', .25), ('cone_eighth', .125)]]],
         'trial_outputs': 1200, 'before_outputs': 200, 'fresh_AdamW_control_is_not_saved_V32_moment_history': True,
         'zero_gradient_protection_may_activate_after_finite_step': True, 'float32_roundoff_may_change_linear_feasibility': True,
         'finite_probe_not_training_capacity': True, 'loss_weights_or_normalizers_changed': False,
         'retained_capacity_gates': {'early_at50_minimum': .01, 'final_at800_minimum': .1, 'preservation_groups': 17,
                                    'MSE_tolerance': 1e-12, 'SSIM_ArcFace_tolerance': 1e-6, 'source_gains_minimum': 0., 'brightness_fraction_maximum': .2},
         'forward_call_limits': {'reference_DGP_forwards': 20, 'candidate_DGP_forwards': 280, 'recognizer_forwards': 860},
         'same_VM_raw_tolerance': 1e-5, 'CPU_raw_absolute_tolerance': 1e-5, 'CPU_PNG_byte_tolerance': 1,
         'CPU_vector_absolute_tolerance': 5e-5, 'CPU_component_value_tolerance': 1e-4,
         'CPU_replay_case_ids': replays, 'CPU_replay_outputs': 280,
         'budgets': {'worker_seconds': 900, 'external_seconds': 930, 'kill_grace_seconds': 30, 'export_seconds': 300,
                     'external_export_seconds': 330, 'peak_vram_bytes': 20 * 1024**3, 'minimum_free_disk_bytes': 6 * 1024**3,
                     'export_uncompressed_bytes': 2 * 1024**3, 'local_audit_seconds': 1800, 'return_files_maximum': 7500},
         'assets_sha256': {name: sha(BUNDLE / name) for name in assets},
         'local_basis_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in [
             Path(__file__), audit, arithmetic, ROOT / 'outputs/cctv_dgp_v32_restoration_cone_v1/analysis.json',
             ROOT / 'outputs/cctv_dgp_v32_loss_directions_v1/analysis.json', ROOT / 'scripts/cctv_dgp_loss_cone_v33.py',
             ROOT / 'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py', ROOT / 'scripts/audit_cctv_dgp_loss_cone_probe_v33_return.py',
             ROOT / 'scripts/audit_cctv_dgp_v32_loss_gradient_v1_return.py', ROOT / 'scripts/cctv_dgp_loss_cone_probe_v33_vm.py',
             ROOT / 'scripts/verify_cctv_dgp_loss_cone_probe_v33_packet.py',
             ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_preparation_failure_v1/failure_record.json',
             ROOT / 'outputs/cctv_dgp_post_v32_architecture_review_v1/user_decision.json',
             ROOT / 'SYSTEM_WORKFLOW_AND_GOAL.md', ROOT / 'PRACTICAL_OUTPUT_SCOPE.md']},
         'VM_execution_started': False, 'checkpoint_created': False, 'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False}
    write(BUNDLE / 'protocol.json', p); pin = sha(BUNDLE / 'protocol.json')
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz'); assert not archive.exists()
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file(): tar.add(path, arcname=NAME + '/' + path.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive); text_file(Path(str(archive) + '.sha256'), digest + '  ' + archive.name + '\n')
    text_file(ROOT / 'CCTV_DGP_LOSS_CONE_PROBE_V33_VM.md', guide(pin, digest))
    write(PREP / 'preparation_receipt.json', {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest,
          'archive_bytes': archive.stat().st_size, 'assets': len(assets), 'basis_VM_files': len(mapping),
          'source_sha256': sha(Path(__file__)), 'VM_launches': 0, 'local_neural_calls': 0, 'local_optimizer_updates': 0,
          'goal_complete': False})
    print(json.dumps(read(PREP / 'preparation_receipt.json'), indent=2))


if __name__ == '__main__': main()
