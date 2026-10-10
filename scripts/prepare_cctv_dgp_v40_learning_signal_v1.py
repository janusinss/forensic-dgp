"""Freeze a portable endpoint diagnostic after the independently retained V40 stop."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40'
RETURN = ROOT/'outputs/cctv_dgp_spatial_fit_v40_return'
ANALYSIS = ROOT/'outputs/cctv_dgp_v40_saved_learning_analysis'
NAME = 'cctv_dgp_v40_learning_signal_v1_vm'
STEM = 'cctv-dgp-v40-learning-signal-v1'
BUNDLE = ROOT/'outputs'/NAME
PREP = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_preparation'
GUIDE = ROOT/'CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM.md'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def text(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream: stream.write(value)


def write(path, value): text(path, json.dumps(value, indent=2, allow_nan=False)+'\n')


SHELL = '''#!/usr/bin/env bash
set -uo pipefail
PIN="${1:?Pass the exact endpoint diagnostic protocol SHA256}"
ROOT="$HOME/forensic-dgp/cctv_dgp_v40_learning_signal_v1_vm"
PYTHON="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
WORKER="$ROOT/scripts/cctv_dgp_v40_learning_signal_v1_vm.py"
cd "$ROOT" || exit 1
test ! -e outputs && test ! -e diagnostic.log && test ! -e supervisor_receipt.json && test ! -e export_manifest.json || exit 1
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --verify-transfer || exit 1
START=$("$PYTHON" -c 'import time; print(time.monotonic())')
timeout --signal=TERM --kill-after=30s 630s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --run 2>&1 | tee diagnostic.log
CODE=${PIPESTATUS[0]}
printf '%s\\n' "$CODE" > diagnostic_exit_code.txt
ELAPSED=$("$PYTHON" -c 'import sys,time; print(time.monotonic()-float(sys.argv[1]))' "$START")
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --record-supervision --elapsed "$ELAPSED" --exit-code "$CODE" || exit 1
printf 'Diagnostic exit code: %s\\n' "$CODE"
timeout --signal=TERM --kill-after=30s 330s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --export
EXPORT_CODE=$?
printf 'Export exit code: %s\\n' "$EXPORT_CODE"
if [ "$CODE" -ne 0 ]; then exit "$CODE"; fi
exit "$EXPORT_CODE"
'''


def guide(pin, digest, byte_count):
    return f'''# V40 endpoint learning diagnostic: five manual steps

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
The self-contained packet is **{byte_count/1024**2:.1f}MiB**; only the existing
venv is required. Estimated diagnostic **2–6 minutes**, export **1–3 minutes**;
worker600s, external630s+30s kill grace, export300s/external330s+30s grace,
allocated VRAM20GiB and uncompressed return512MiB. Historical disk readings
are not current availability. The diagnostic deletes no research assets.

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
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_v40_learning_signal_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_v40_learning_signal_v1_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_learning_signal.sh {pin}
```

Detach with **Ctrl+B**, release, then **D**. Step3 reconnects. The completed
log ends at `gradient_queries:280`, followed by diagnostic/export exit codes.
An export `complete:true` reports packaging only. Download failures too;
retain the stop and do not edit, resume or repeat the packet unchanged.

5. Download from **Windows Google Cloud SDK Shell** after export completes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
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
'''


def main():
    assert not BUNDLE.exists() and not PREP.exists() and not GUIDE.exists(), 'Preserve all prior/partial preparations'
    parent = read(PARENT/'protocol.json'); audited = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json')
    analysis = read(ANALYSIS/'results.json'); checked = read(ANALYSIS/'independent_analysis_audit.json')
    assert audited['complete'] and audited['failure_retained'] and not audited['necessary_capacity_pass']
    assert analysis['complete'] and checked['complete'] and analysis['tensors_changed'] == 57 and analysis['stop_and_snapshot50_tensors_exact']
    assert analysis['normalizers_exact'] and analysis['local_gradient_calls'] == analysis['local_optimizer_updates'] == 0
    for name, value in analysis['sources_sha256'].items(): assert sha(ROOT/name) == value, name
    cases = {c['id']: c for c in parent['cases']}; refs = {r['id']: r for r in parent['references']}
    schedule = read(PARENT/'schedule.json')['batches']; first50 = schedule[:50]
    ordered = [parent['cases'][ids[0]]['source_person_or_reference'] for ids in first50]
    chosen = []; counts = {source: 0 for source in sorted({c['source'] for c in parent['cases']})}
    for rid in ordered:
        source = refs[rid]['source']
        if counts[source] < 5: chosen.append(rid); counts[source] += 1
    assert len(chosen) == 10 and all(n == 5 for n in counts.values())
    optimized = [parent['cases'][i]['id'] for ids in first50 if parent['cases'][ids[0]]['source_person_or_reference'] in chosen for i in ids]
    assert len(optimized) == len(set(optimized)) == 50
    fixed = parent['preview_case_ids']; assert len(fixed) == 50
    all_exposed = {parent['cases'][i]['id'] for ids in first50 for i in ids}; assert not set(fixed) & all_exposed
    selected = [dict(cases[cid]) for cid in fixed+optimized]
    selected_refs = [refs[rid] for rid in dict.fromkeys(c['source_person_or_reference'] for c in selected)]
    assert len(selected_refs) == 20 and all(c['role'] == 'train' for c in selected)
    BUNDLE.mkdir(); (BUNDLE/'scripts').mkdir(); PREP.mkdir(); mapping = {}
    def copy(source, name):
        dest = BUNDLE/name; dest.parent.mkdir(parents=True, exist_ok=True); assert not dest.exists()
        shutil.copyfile(source, dest); assert sha(source) == sha(dest); mapping[name] = source.relative_to(ROOT).as_posix()
    for path in sorted(PARENT.rglob('*.py')):
        name = path.relative_to(PARENT).as_posix()
        if name.startswith('scripts/') or name == 'cctv_dgp_spatial_decoder_v40.py': continue
        assert sha(path) == parent['assets_sha256'][name]; copy(path, name)
    for name in ['weights/dgp_v2.pth', 'weights/w600k_r50.onnx', 'untrained_initial_decoder.pth']:
        assert sha(PARENT/name) == parent['assets_sha256'][name]; copy(PARENT/name, name)
    copy(RETURN/'outputs/stopped_spatial_decoder.pth', 'stopped_decoder_input.pth')
    copy(RETURN/'outputs/cohort_loss_setup.json', 'historical_cohort_loss_setup.json')
    copy(ANALYSIS/'parameter_delta.npy', 'saved_parameter_delta.npy')
    for name in sorted({c[k] for c in selected for k in ['input', 'target', 'observed']}):
        assert sha(PARENT/name) == parent['assets_sha256'][name]; copy(PARENT/name, name)
    for c in selected[:50]:
        for update, key in [(0, 'historical_initial_raw'), (50, 'historical_stopped_raw')]:
            name = 'historical/update'+str(update)+'/'+c['id']+'.npy'
            copy(RETURN/('outputs/update'+str(update)+'/'+c['id']+'.npy'), name); c[key] = name
    original_source = (PARENT/'cctv_dgp_spatial_decoder_v40.py').read_text(encoding='utf-8')
    transformed = original_source.replace('cctv_dgp_spatial_fit_vm_v40', NAME).replace('Distinct V40 VM root', 'Distinct endpoint diagnostic VM root').replace('SpatialDGPCandidateV40', 'SpatialDGPCandidateV40LearningSignalV1')
    text(BUNDLE/'cctv_dgp_v40_learning_signal_v1_decoder.py', transformed)
    copy(ROOT/'scripts/cctv_dgp_v40_learning_signal_v1_vm.py', 'scripts/cctv_dgp_v40_learning_signal_v1_vm.py')
    text(BUNDLE/'scripts/run_learning_signal.sh', SHELL)
    for path in BUNDLE.rglob('*.py'): ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    basis_names = ['scripts/prepare_cctv_dgp_v40_learning_signal_v1.py', 'scripts/verify_cctv_dgp_v40_learning_signal_v1_packet.py',
        'scripts/audit_cctv_dgp_v40_learning_signal_v1_return.py', 'scripts/analyze_cctv_dgp_v40_saved_learning.py',
        'scripts/verify_cctv_dgp_v40_saved_learning.py', 'CCTV_DGP_V40_LEARNING_SIGNAL_V1_REVIEW.md',
        'outputs/cctv_dgp_v40_return_completion_milestone/milestone.json', 'outputs/cctv_dgp_v40_return_completion_milestone/independent_closure_audit.json',
        'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json', 'outputs/cctv_dgp_spatial_fit_vm_v40/schedule.json',
        'outputs/cctv_dgp_spatial_fit_vm_v40/cctv_dgp_spatial_decoder_v40.py', 'outputs/cctv_dgp_v40_saved_learning_analysis/results.json',
        'outputs/cctv_dgp_v40_saved_learning_analysis/independent_analysis_audit.json']
    basis = {name: sha(ROOT/name) for name in basis_names}; basis.update(analysis['sources_sha256'])
    for source in mapping.values(): basis[source] = sha(ROOT/source)
    assets = {path.relative_to(BUNDLE).as_posix(): sha(path) for path in BUNDLE.rglob('*') if path.is_file()}
    states = dict(parent['initial_states'])
    replay_batches = []
    for ids in [fixed, optimized]:
        for source in sorted(counts):
            rid = next(cases[cid]['source_person_or_reference'] for cid in ids if cases[cid]['source'] == source)
            replay_batches.append([cid for cid in ids if cases[cid]['source_person_or_reference'] == rid])
    p = {'format': 'own-DGP-V40-endpoint-learning-signal-v1', 'UTC': datetime.now(timezone.utc).isoformat(),
         'hypothesis': 'Compare endpoint learning and preservation signals on optimized and not-yet-optimized TRAIN cohorts before choosing a new recipe; no causal or quality claim.',
         'cases': selected, 'references': selected_refs, 'cohorts': {'not_yet_optimized': fixed, 'optimized': optimized},
         'selection': 'Fixed V40 preview50 plus firstfive references per source in first50 frozen paired schedule batches, allfive profiles, metadata only.',
         'first50_reference_order': ordered, 'selected_optimized_references': chosen,
         'parameter_layout': parent['parameter_layout'], 'decoder_parameters': 17952, 'decoder_tensors': 57,
         'expected_initial_states': states, 'decoder_states': {'initial': analysis['initial_state'], 'stopped50': analysis['stopped_state']},
         'original_checkpoint_sha256': assets['weights/dgp_v2.pth'], 'recognizer_weights_sha256': assets['weights/w600k_r50.onnx'],
         'terms': parent['terms'], 'retained_capacity_gates': parent['retained_capacity_gates'],
         'assets_sha256': assets, 'local_basis_sha256': basis, 'copied_source_mapping': mapping,
         'architecture_equivalent_to_V40_except_guard_and_class_name': True,
         'normalizers_and_objective_exact_V40': True, 'component_gradient_calls': 280,
         'optimizer_updates': 0, 'parameter_updates': 0, 'backwards': 0, 'epochs': 0,
         'forward_call_limits': {'original_DGP_forwards': 60, 'decoder_forwards': 40, 'reference_decoder_forwards': 40, 'recognizer_forwards': 60},
         'budgets': {'worker_seconds': 600, 'external_seconds': 630, 'kill_grace_seconds': 30, 'export_seconds': 300,
             'export_external_seconds': 330, 'peak_vram_bytes': 20*1024**3, 'minimum_free_disk_bytes': 2*1024**3, 'export_uncompressed_bytes': 512*1024**2},
         'historical_raw_tolerance': 1e-5, 'CPU_raw_replay_tolerance': 1e-5, 'CPU_PNG_byte_tolerance': 1,
         'CPU_scalar_replay_tolerance': 5e-5, 'CPU_embedding_absolute_tolerance': 1e-4, 'CPU_replay_batches': replay_batches,
         'source_labels_not_ethnicity': True, 'exposed_photographic_TRAIN_only': True, 'native_or_reserved_used': False,
         'optimizer_moments_not_reconstructed': True, 'new_learning_recipe_or_checkpoint': False,
         'manual_transfer_and_launch_required': True, 'automatic_follow_on': False, 'VM_calls_here': 0,
         'local_gradient_calls': 0, 'app_promotion': False, 'independent_final_review': False, 'goal_complete': False}
    write(BUNDLE/'protocol.json', p); pin = sha(BUNDLE/'protocol.json')
    archive = ROOT/'outputs'/(STEM+'-execution.tar.gz'); assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file(): tar.add(path, arcname=NAME+'/'+path.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive); text(Path(str(archive)+'.sha256'), digest+'  '+archive.name+'\n')
    text(GUIDE, guide(pin, digest, archive.stat().st_size))
    write(PREP/'prepared.json', {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest,
          'archive_bytes': archive.stat().st_size, 'files': len(assets)+1, 'VM_calls': 0, 'neural_calls': 0,
          'local_gradient_calls': 0, 'optimizer_updates': 0, 'manual_diagnostic_prepared_only': True,
          'independent_packet_audit_pending': True, 'app_promotion': False, 'goal_complete': False})
    print({'prepared': True, 'files': len(assets)+1, 'bytes': archive.stat().st_size, 'protocol_sha256': pin, 'archive_sha256': digest}, flush=True)


if __name__ == '__main__': main()
