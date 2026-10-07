"""Bind the audited fixed-state return and one finite reference-processing correction."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone'
PREVIOUS = ROOT / 'outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json'
DOCS = [
    'PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
    'CCTV_DGP_DEGRADED_DETAIL_V24_PLAN.md', 'CCTV_DGP_DEGRADED_DETAIL_V24_VM.md',
    'CCTV_DGP_SPATIAL_FEATURES_V25_PLAN.md', 'CCTV_DGP_SPATIAL_FEATURES_V25_VM.md',
    'CCTV_DGP_SPATIAL_FEATURES_V25_RESULTS.md',
    'CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_PLAN.md', 'CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_VM.md',
]
SOURCES = [
    'scripts/import_cctv_dgp_v25_gradient_diagnostic_v1.py',
    'scripts/audit_cctv_dgp_v25_gradient_diagnostic_v1_return.py',
    'scripts/cctv_dgp_batchmatched_identity_v26.py',
    'scripts/cctv_dgp_batchmatched_identity_v26_preflight.py',
    'scripts/install_cctv_dgp_batchmatched_identity_v26_vm.py',
    'scripts/prepare_cctv_dgp_batchmatched_identity_v26.py',
    'scripts/cctv_dgp_batchmatched_identity_v26_vm.py',
    'scripts/verify_cctv_dgp_batchmatched_identity_v26.py',
    'scripts/audit_cctv_dgp_batchmatched_identity_v26_preparation.py',
    'tests/test_cctv_dgp_gradient_return_and_batchmatched_v26.py',
    'scripts/record_cctv_dgp_v25_gradient_v26_milestone.py',
    'scripts/verify_cctv_dgp_v25_gradient_v26_milestone.py',
]
NEW_DOCS = [
    'CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_RESULTS.md',
    'CCTV_DGP_BATCHMATCHED_IDENTITY_V26_PLAN.md', 'CCTV_DGP_BATCHMATCHED_IDENTITY_V26_VM.md',
]
LATEST = '''**Latest research milestone — 6 October 2026: V25 gradient return audited; V26 reference-processing correction verified for manual L4 execution.**

The 1,695,519-byte fixed-state diagnostic return is verified and safely imported:
ten regular files, two 7×53,781 gradient matrices, 14 component rows and 26 parameter
groups per state. Independent arithmetic/source/state/count/timing checks pass.
The L4 measurement takes 18.535 seconds with 140 gradient queries and zero optimizer
updates. This readback checks saved derivatives; it does not independently recompute
L4 derivatives or establish the cause of the entire optimizer trajectory.

All 50 initial raw outputs equal their cached own-DGP baseline exactly. The legacy
identity reference nevertheless incurs a 1.3932586e-7 mean penalty and a gradient
1.45139 times the landmark-detail gradient. Baseline scores are computed one case
at a time without gradient tracking, while predictions use five-case gradient
calls. The zero-margin penalty responds to that numerical difference. A demonstrated
reference-processing correction is justified; the original V25 structure failure
(0.0282235213% against 1% at update 50) remains a failure. Real preservation penalties
are retained, and useful restoration is still unproven.

V26 changes only the identity reference calculation: baseline and prediction share
one frozen recognizer call, with the baseline embedding detached. The original
zero margin, coefficient 5, other six terms, own-DGP spatial head, 50 exposed TRAIN
cases, schedule, optimizer, quality gates and finite limits remain unchanged.
The L4 preflight must reproduce the legacy discrepancy and prove exactly zero
matched penalty and all 26 matched parameter gradients on every initial baseline
batch before creating an optimizer. No epsilon or preservation waiver is added.

The new transfer is 49,537 bytes, seven regular files and five new source assets.
Its guarded 60-second installer verifies and copies 235 original VM assets into a
fresh directory, preserving failed V25. No original data or weights are reuploaded.
The actual proof has a 180-second limit inside the original 300-second preflight.
The pilot retains at most 800 updates/80 epochs, the 1% early structure gate at 50
and the 10% final structure gate at 800, all 17 preservation groups and the brightness
limit. Fit 1,500 seconds; worker 1,800; supervisor 2,100 plus 30 seconds grace.
Any proof, timing or quality stop is retained; no unchanged failed recipe is rerun.

Eleven saved-matrix/archive/reference contracts, Python 3.10 parsing, actual Windows
rejection guards and Bash syntax checks pass. Tests use an arithmetic mock recognizer;
no local learned-model forwards, gradients, backwards or optimization occur. No
assistant VM/cloud action occurs. Actual V26 L4 proof/training, independent return
audit and whole-face review remain pending. Human execution uses the five manual
upload/install/tmux/run/download steps; PuTTY downloads use three separate calls.

Prior 697 evidence bindings and ten full document bodies are preserved, along with
the app's 22 bindings and concurrent completed VM maintenance. No app promotion.
The user-selected own-DGP spatial/feature direction and all visible facial features
together remain required; previous useful native inputs stay usable despite model
softness. Native CCTV remains unpaired, and paired photographic TRAIN metrics remain
separate. No native/reserved-final/new covering pixels, ethnicity or Zamboanga
performance claim is introduced. Useful native outputs, candidate app parity/full
flow, insufficient-information handling, all seven automatic/assisted covering
families and independent final review remain required. Goal active/incomplete.

[Audited diagnostic](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_RESULTS.md>) ·
[Finite V26 design](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BATCHMATCHED_IDENTITY_V26_PLAN.md>) ·
[Current five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BATCHMATCHED_IDENTITY_V26_VM.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json>)

Previous bodies below are preserved history. Diagnostic-pending statements and old
V22–V25/diagnostic launch commands are superseded by the completed diagnostic audit
and the guarded fresh V26 correction. Do not repeat immutable historical pilots.
'''


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_bindings(bindings):
    for name, digest in bindings.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve the current milestone; do not overwrite it'
    assert sha(PREVIOUS) == 'd9fa6cbd35b3b89b5aa986d8e0f20ccf5dd04b9499ded074be6202bfdd4e8b68'
    previous = read(PREVIOUS)
    assert len(previous['new_evidence_sha256']) == 697
    verify_bindings(previous['new_evidence_sha256'])
    audit = read(ROOT / 'outputs/cctv_dgp_v25_gradient_diagnostic_v1_independent_audit.json')
    assert audit['complete'] and audit['identity_baseline_numerical_discrepancy_confirmed']
    assert audit['initial_raw_equals_cached_baseline_cases'] == 50
    assert audit['VM_optimizer_updates_verified'] == 0 and audit['VM_gradient_calls_verified'] == 140
    assert audit['gradient_arrays_verified'] == 2 and audit['parameter_groups_per_state'] == 26
    assert audit['original_V25_quality_failure_retained'] and not audit['gradient_trajectory_causality_proven']
    verify_bindings(audit['source_bindings_sha256'])
    gradient_archive = ROOT / 'outputs/cctv-dgp-v25-gradient-diagnostic-v1-results.tar.gz'
    assert sha(gradient_archive) == audit['archive_sha256'] and gradient_archive.stat().st_size == 1695519
    prep = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_preparation'
    packet = read(prep / 'preparation.json')
    check = read(prep / 'independent_packet_source_audit.json')
    readback = read(prep / 'independent_preparation_readback.json')
    assert all(item['complete'] for item in [packet, check, readback])
    assert packet['protocol_sha256'] == check['protocol_sha256'] == readback['protocol_sha256']
    assert packet['assets'] == 240 and packet['inherited_assets'] == 235 and packet['archive_bytes'] == 49537
    assert readback['new_tests_passed'] == 11 and readback['Bash_n_exit_code'] == 0
    verify_bindings(readback['source_bindings_sha256'])
    bundle = ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26'
    assert sha(bundle / 'protocol.json') == packet['protocol_sha256']
    protocol = read(bundle / 'protocol.json')
    for name, digest in protocol['assets_sha256'].items():
        assert sha(bundle / name) == digest, name
    v25 = read(ROOT / 'outputs/cctv_dgp_spatial_features_vm_v25/protocol.json')
    for key in ['cases', 'references', 'design', 'budgets', 'prospective_gates']:
        assert protocol[key] == v25[key], key
    assert protocol['identity_preflight_policy']['matched_identity_penalty_exactly_zero']
    assert protocol['identity_preflight_policy']['all26_matched_parameter_gradients_exactly_zero']
    assert protocol['actual_VM_processing_correction_and_learning_pending']
    apppath = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(apppath) == 'e744f3708f248012469a111e204222f66261af4ca28bbaf9521ee4c5464b5e7a'
    app = read(apppath)
    appbindings = {**app['sources_sha256'], **app['evidence_sha256']}
    assert len(appbindings) == 22
    verify_bindings(appbindings)
    for name in SOURCES:
        ast.parse((ROOT / name).read_text(encoding='utf-8'), feature_version=(3, 10))
    for name in DOCS + NEW_DOCS:
        assert (ROOT / name).is_file() and b'\n' in (ROOT / name).read_bytes(), name
    OUT.mkdir()
    (OUT / 'before_docs').mkdir()
    before = {}
    for name in DOCS:
        shutil.copy2(ROOT / name, OUT / 'before_docs' / name)
        before[name] = sha(OUT / 'before_docs' / name)
        assert before[name] == previous['new_evidence_sha256'][name], name
    assert b'VM storage cleanup completed; future VM use preserved' in (OUT / 'before_docs/PROJECT_HANDOFF.md').read_bytes()
    locations = {name: (OUT / 'before_docs' / name).relative_to(ROOT).as_posix() for name in DOCS}
    write(OUT / 'before_update_plan.json', {
        'date': '2026-10-06', 'before_docs_sha256': before,
        'previous_milestone_sha256': sha(PREVIOUS), 'previous697_original_locations': locations,
        'app_record_sha256': sha(apppath), 'runner_sha256': sha(Path(__file__)),
        'actual_V26_L4_proof_and_training_pending': True,
    })
    for name in DOCS:
        assert sha(ROOT / name) == before[name], 'Concurrent document update: ' + name
        old = (OUT / 'before_docs' / name).read_bytes()
        split = old.index(b'\n') + 1
        (ROOT / name).write_bytes(old[:split] + b'\n' + LATEST.encode('utf-8') + b'\n' + old[split:])
        assert (ROOT / name).read_bytes().endswith(old[split:]), name
    files = DOCS + SOURCES + NEW_DOCS + [
        'outputs/cctv-dgp-v25-gradient-diagnostic-v1-results.tar.gz',
        'outputs/cctv-dgp-v25-gradient-diagnostic-v1-results.tar.gz.sha256',
        'outputs/cctv-dgp-v25-gradient-diagnostic-v1-export.json',
        'outputs/cctv_dgp_v25_gradient_diagnostic_v1_return_import.json',
        'outputs/cctv_dgp_v25_gradient_diagnostic_v1_independent_audit.json',
        'outputs/cctv-dgp-batchmatched-identity-v26-execution.tar.gz',
        'outputs/cctv-dgp-batchmatched-identity-v26-execution.tar.gz.sha256',
    ]
    paths = {ROOT / name for name in files}
    for name in ['cctv_dgp_v25_gradient_diagnostic_v1_return',
                 'cctv_dgp_batchmatched_identity_vm_v26', 'cctv_dgp_batchmatched_identity_v26_preparation']:
        paths.update(path for path in (ROOT / 'outputs' / name).rglob('*') if path.is_file())
    paths.update(path for path in OUT.rglob('*') if path.is_file())
    bindings = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}
    record = {
        'complete': True, 'date': '2026-10-06',
        'scope': 'Audited fixed-state L4 diagnostic and single verified finite reference-processing correction; Goal incomplete',
        'new_evidence_sha256': bindings, 'previous_milestone_sha256': sha(PREVIOUS),
        'previous697_original_locations': locations, 'app_record_sha256': sha(apppath),
        'actual_gradient_diagnostic_return_present': True, 'gradient_return_archive_bytes': 1695519,
        'gradient_return_archive_sha256': audit['archive_sha256'],
        'gradient_matrices_verified': 2, 'fixed_state_gradient_entries_checked': 752934,
        'VM_diagnostic_gradient_calls': 140, 'VM_diagnostic_optimizer_updates': 0,
        'diagnostic_worker_seconds': audit['diagnostic_worker_seconds'],
        'initial_exact_baseline_cases': 50, 'initial_identity_penalty': audit['snapshots'][0]['component_values'][6],
        'initial_identity_to_landmark_gradient_norm_ratio': audit['snapshots'][0]['ArcFace_to_landmark_gradient_norm_ratio'],
        'reference_numerical_discrepancy_confirmed': True, 'gradient_trajectory_causality_proven': False,
        'original_V25_failure_retained': True, 'final800_of_V25_executed': False,
        'selected_route': 'A', 'single_processing_correction': 'Same-call detached baseline identity reference',
        'V26_protocol_sha256': packet['protocol_sha256'], 'V26_archive_sha256': packet['archive_sha256'],
        'V26_archive_bytes': 49537, 'V26_transfer_members': 7, 'V26_reconstructed_assets': 240,
        'unchanged_original_data_head_objective_weights_quality_gates_and_finite_limits': True,
        'all26_exact_zero_matched_gradients_required_before_optimizer': True,
        'actual_V26_L4_proof_and_training_pending': True, 'V26_return_and_whole_face_review_pending': True,
        'new_regressions_passed': 11, 'Bash_syntax_passed': True,
        'local_model_calls': 0, 'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'VM_actions': False, 'app_unchanged': True, 'app_promotion': False, 'native_or_reserved_used': False,
        'independent_final_review_complete': False, 'goal_status': 'active', 'goal_complete': False,
        'seconds': time.monotonic() - started,
    }
    write(OUT / 'milestone.json', record)
    print(json.dumps({key: value for key, value in record.items()
                      if key not in ['new_evidence_sha256', 'previous697_original_locations']}, indent=2))
    print('Bound files:', len(bindings))


if __name__ == '__main__':
    main()
