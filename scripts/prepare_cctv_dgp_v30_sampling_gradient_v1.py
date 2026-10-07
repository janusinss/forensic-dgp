"""Prepare metadata and pinned source only; all new gradient queries remain manual VM work."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile
import time
from cctv_dgp_v30_sampling_gradient_v1_vm import matched_unexposed_cases

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30'
RETURNED = ROOT / 'outputs/cctv_dgp_broader_mean_v30_return'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
ACTIVE = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
NAME = 'cctv_dgp_v30_sampling_gradient_v1_vm'
STEM = 'cctv-dgp-v30-sampling-gradient-v1'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_preparation'
WORKER = 'scripts/cctv_dgp_v30_sampling_gradient_v1_vm.py'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


SHELL = '''#!/usr/bin/env bash
set -uo pipefail
PIN="${1:?Pass the exact diagnostic protocol SHA256}"
ROOT="$HOME/forensic-dgp/cctv_dgp_v30_sampling_gradient_v1_vm"
PYTHON="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
WORKER="$ROOT/scripts/cctv_dgp_v30_sampling_gradient_v1_vm.py"
cd "$ROOT" || exit 1
test ! -e outputs && test ! -e diagnostic.log && test ! -e supervisor_receipt.json && test ! -e export_manifest.json || exit 1
"$PYTHON" -B "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --verify-transfer || exit 1
START=$("$PYTHON" -c 'import time; print(time.monotonic())')
timeout --signal=TERM --kill-after=30s 900s "$PYTHON" -B -u "$WORKER" --root "$ROOT" --protocol-sha "$PIN" --run 2>&1 | tee diagnostic.log
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


def guide(pin, archive_sha):
    return f'''# V30 sampling-gradient diagnostic: five manual steps

V30 stopped at 50/800 with 0.805717% structure improvement against the unchanged 1% early requirement.
Do not resume or rerun V30. This distinct diagnostic measures 280 gradients with
zero optimizer updates. It compares the original and stopped update50 decoder on the
first 50 actually exposed cases and 50 unexposed TRAIN cases matched by source
and degradation profile. Reference repetition is also matched one to one.
Selection uses the frozen schedule and input metadata.
All visible regions remain in scope; no held-out/native/final cases are used.

Use the existing NVIDIA L4/g2-standard-4 at ~/forensic-dgp. Require 4 GiB free and
no other GPU workload. Estimated 2-8 minutes of measurements plus archive export.
Enforced worker limit: 600s; external limit: 900s + 30s grace; export limit: 300s
(external export: 330s + 30s grace), allocated VRAM <=20 GiB, return <=1.5 GiB.
No optimizer, backward,
checkpoint writer, loss change, app promotion or automatic training is present.
Read the new return locally before selecting any later finite training pilot.
Original checkpoints, splits, logs and failed gates remain preserved.

Protocol SHA256: {pin}
Execution archive SHA256: {archive_sha}

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
test -d ~/forensic-dgp/cctv_dgp_broader_mean_vm_v30/outputs/update50 &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_v30_sampling
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_v30_sampling_gradient_v1_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_sampling.sh {pin}
```

The final diagnostic should report 280 queries and 0 updates. Export `complete:true`
confirms packaging only; retain any `failure_present:true` result. Detach using
Ctrl+B, release both keys, then D. The finite worker continues inside tmux.

5. Download from **Windows Google Cloud SDK Shell** using three separate calls:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Three separate remote-source calls avoid Windows PuTTY's multiple-source error.
The local prospective checker verifies archive boundaries, source/data provenance,
all saved derivative arithmetic and 100 cases at both states through CPU inference
only. Derivative computation and any future training remain manual VM work.
The CPU check replays outputs and loss values; it does not recompute gradients.
'''


def main():
    start = time.monotonic(); assert not BUNDLE.exists() and not PREP.exists()
    audit_path = ROOT / 'outputs/cctv_dgp_broader_mean_v30_independent_audit_r1.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['VM_failure_retained'] and not audit['training_completed800']
    assert not audit['necessary_capacity_pass'] and [r['update'] for r in audit['snapshots_audited']] == [0, 50]
    review_path = ROOT / 'outputs/cctv_dgp_broader_mean_v30_failure_review_v1/visual_review.json'
    review = read(review_path); assert review['complete'] and review['cases_reviewed'] == 50 and not review['app_promotion']
    old = read(OLD / 'protocol.json'); assert sha(OLD / 'protocol.json') == 'b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1'
    schedule = read(OLD / 'schedule.json')['batches']
    exposed = [old['case_rows'][i] for batch in schedule[:10] for i in batch]
    touched = {old['case_rows'][i]['source_person_or_reference'] for batch in schedule[:50] for i in batch}
    assert len(touched) == 218 and len({r['id'] for r in exposed}) == 50
    matched, pairs = matched_unexposed_cases(exposed, old['case_rows'], touched)
    assert len(matched) == 50 and set(c['id'] for c in exposed).isdisjoint(c['id'] for c in matched)
    cohorts = [{'name': 'exposed', 'selection': 'First10 unchanged V30 optimizer batches in exact recorded order', 'cases': exposed},
               {'name': 'unexposed', 'selection': 'First available source/profile match on a distinct untouched reference for each unique exposed reference; reuse its matched reference for repeated profiles. Exclude all218 touched references and duplicate cases.', 'cases': matched}]
    dependency_names = ['protocol.json', *old['assets_sha256'], 'outputs/failure.json', 'outputs/early_structure_stop.json',
                        'outputs/cohort_loss_setup.json', 'outputs/execution_receipt.json', 'outputs/update50/dgp_candidate_v30.pth',
                        'outputs/update0/metrics.json', 'outputs/update50/metrics.json']
    for case in exposed + matched:
        for update in [0, 50]: dependency_names.extend(f'outputs/update{update}/{case["id"]}{suffix}' for suffix in ['.png', '_embedding.npy'])
        dependency_names.append('outputs/update0/' + case['id'] + '_target_embedding.npy')
    closed_hashes = {name: sha(OLD / name if name == 'protocol.json' or name in old['assets_sha256'] else RETURNED / name) for name in sorted(set(dependency_names))}
    parent = read(PARENT / 'protocol.json')
    parent_hashes = {'protocol.json': sha(PARENT / 'protocol.json'), **parent['assets_sha256']}
    for name, digest in parent_hashes.items(): assert sha(PARENT / name) == digest
    active_names = ['cctv_dgp_active_original_decoder_v28.py', 'cctv_dgp_original_decoder_candidate_v1.py']
    active_hashes = {name: sha(ACTIVE / name) for name in active_names}
    data_names = {c[k] for group in cohorts for c in group['cases'] for k in ['input', 'target', 'observed']}
    data_hashes = {name: sha(MIXED / name) for name in sorted(data_names)}
    assert all(old['mixed_TRAIN_assets_sha256'][name] == digest for name, digest in data_hashes.items())
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    (BUNDLE / WORKER).write_bytes((ROOT / WORKER).read_bytes())
    (BUNDLE / 'scripts/run_sampling.sh').write_text(SHELL, encoding='ascii', newline='\n')
    assert ast.parse((BUNDLE / WORKER).read_text(), feature_version=(3, 10))
    normalizers = read(RETURNED / 'outputs/cohort_loss_setup.json')
    p = {'format': 'V30-sampling-matched-zero-update-gradient-diagnostic-v1', 'date': '2026-10-07',
         'purpose': 'Distinguish conflicting sampled descent directions and active preservation terms from insufficient decoder learning before any new recipe. The stopped V30 result cannot establish the unseen800-update outcome.',
         'hypotheses': ['Broad per-batch improvement directions may cancel more than repeated10-reference fitting.',
                        'The stopped50 preservation gradients may oppose useful detail descent even while aggregate PNG preservation gates pass.'],
         'selection_independent_of_output_or_gradient': True, 'cohorts': cohorts, 'states': [0, 50],
         'matched_reference_pairs': pairs, 'reference_repetition_matched': True,
         'normalizers': [normalizers['feature_normalizer'], normalizers['interior_normalizer']],
         'normalizer_policy': 'Reuse exact independently audited frozen initial50 V30 scalars; no rescaling',
         'terms': old['terms'], 'parameter_layout': old['parameter_layout'], 'selected_parameters': 498627, 'selected_tensors': 12,
         'optimizer_updates': 0, 'backwards': 0, 'epochs': 0, 'gradient_queries': 280,
         'original_checkpoint_sha256': old['original_checkpoint_sha256'], 'original_DGP_state': old['original_DGP_state'],
         'stopped50_DGP_state': read(RETURNED / 'outputs/update50/metrics.json')['candidate_DGP_state'],
         'recognizer_state': old['frozen_recognizer_state'], 'V30_protocol_sha256': sha(OLD / 'protocol.json'),
         'V30_dependencies_sha256': closed_hashes, 'parent_dependencies_sha256': parent_hashes,
         'active_decoder_dependencies_sha256': active_hashes, 'TRAIN_assets_sha256': data_hashes,
         'local_basis_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in [audit_path, review_path, ROOT / WORKER, Path(__file__),
                               ROOT / 'outputs/cctv_dgp_broader_mean_v30_failure_review_v1/preparation.json',
                               ROOT / 'scripts/cctv_dgp_v30_sampling_gradient_v1_return_audit_template.py',
                               ROOT / 'scripts/verify_cctv_dgp_v30_sampling_gradient_v1_packet.py',
                               ROOT / 'scripts/record_cctv_dgp_broader_mean_v30_failure_review.py',
                               ROOT / 'tests/test_cctv_dgp_v30_sampling_gradient_v1.py']},
         'budgets': {'worker_seconds': 600, 'external_seconds': 900, 'kill_grace_seconds': 30, 'export_seconds': 300,
                     'external_export_seconds': 330, 'peak_vram_bytes': 20 * 1024**3, 'minimum_free_disk_bytes': 4 * 1024**3,
                     'export_uncompressed_bytes': 1536 * 1024**2, 'local_forward_audit_seconds': 1800},
         'CPU_raw_absolute_tolerance': 1e-5, 'CPU_PNG_byte_tolerance': 1, 'CPU_vector_absolute_tolerance': 5e-5,
         'CPU_batch_component_value_tolerance': 1e-4,
         'limits': 'These matched TRAIN subsets are not held-out data. Gradient cancellation/directional derivatives do not reconstruct AdamW steps, identify a unique historical cause or prove finite-step/real-CCTV utility.',
         'retained_gates': old['retained_capacity_gates'], 'V30_failure_must_remain': True, 'no_resume_or_new_checkpoint': True,
         'new_training_recipe_created': False, 'human_manual_VM_execution_required': True,
         'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False}
    p['assets_sha256'] = {f.relative_to(BUNDLE).as_posix(): sha(f) for f in sorted(BUNDLE.rglob('*')) if f.is_file()}
    write(BUNDLE / 'protocol.json', p); pin = sha(BUNDLE / 'protocol.json')
    template = (ROOT / 'scripts/cctv_dgp_v30_sampling_gradient_v1_return_audit_template.py').read_text(encoding='utf-8')
    assert template.count('PROTOCOL_PIN_TO_FILL') == 1
    prospective = ROOT / 'scripts/audit_cctv_dgp_v30_sampling_gradient_v1_return.py'
    with prospective.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(template.replace('PROTOCOL_PIN_TO_FILL', pin))
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for f in sorted(BUNDLE.rglob('*')):
            if f.is_file(): tar.add(f, arcname=NAME + '/' + f.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(digest + '  ' + archive.name + '\n')
    guide_path = ROOT / 'CCTV_DGP_V30_SAMPLING_GRADIENT_V1_VM.md'
    with guide_path.open('x', encoding='ascii', newline='\n') as stream: stream.write(guide(pin, digest))
    receipt = {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
               'packet_files': len(p['assets_sha256']) + 1, 'cases': 100, 'states': 2, 'gradient_queries_bound': 280,
               'optimizer_updates': 0, 'TRAIN_assets_verified': len(data_hashes), 'V30_dependency_hashes': len(closed_hashes),
               'prospective_return_auditor_sha256': sha(prospective),
               'original_data_or_weights_uploaded': False, 'local_neural_or_gradient_calls': 0,
               'human_manual_VM_execution_required': True, 'actual_VM_run_started': False, 'app_promotion': False,
               'goal_complete': False, 'seconds': time.monotonic() - start}
    write(PREP / 'preparation.json', receipt); print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
