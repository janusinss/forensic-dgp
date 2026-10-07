"""Close the actual V26 failure with immutable evidence and preserved history."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_batchmatched_identity_v26_audit_milestone'
PREVIOUS = ROOT / 'outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json'
DOCS = [
    'PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
    'CCTV_DGP_DEGRADED_DETAIL_V24_PLAN.md', 'CCTV_DGP_DEGRADED_DETAIL_V24_VM.md',
    'CCTV_DGP_SPATIAL_FEATURES_V25_PLAN.md', 'CCTV_DGP_SPATIAL_FEATURES_V25_VM.md',
    'CCTV_DGP_SPATIAL_FEATURES_V25_RESULTS.md',
    'CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_PLAN.md', 'CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_VM.md',
    'CCTV_DGP_BATCHMATCHED_IDENTITY_V26_PLAN.md', 'CCTV_DGP_BATCHMATCHED_IDENTITY_V26_VM.md',
]
SOURCES = [
    'scripts/cctv_dgp_batchmatched_identity_v26_return_rules.py',
    'scripts/prepare_cctv_dgp_batchmatched_identity_v26_return_audit.py',
    'scripts/import_cctv_dgp_batchmatched_identity_v26.py',
    'scripts/audit_cctv_dgp_batchmatched_identity_v26.py',
    'scripts/audit_cctv_dgp_batchmatched_identity_v26_execution.py',
    'scripts/verify_cctv_dgp_batchmatched_identity_v26_return_audit.py',
    'tests/test_cctv_dgp_batchmatched_identity_v26_return_audit.py',
    'scripts/prepare_cctv_dgp_batchmatched_identity_v26_diagnostic.py',
    'scripts/diagnose_cctv_dgp_batchmatched_identity_v26.py',
    'scripts/verify_cctv_dgp_batchmatched_identity_v26_diagnostic.py',
    'scripts/record_cctv_dgp_batchmatched_identity_v26_visual_review.py',
    'scripts/record_cctv_dgp_batchmatched_identity_v26_audit_milestone.py',
    'scripts/verify_cctv_dgp_batchmatched_identity_v26_audit_milestone.py',
]
LATEST = '''**Latest research milestone — 6 October 2026: V26 return audited; identity-reference correction proved, structure failure retained.**

The downloaded 325,764,761-byte V26 archive matches SHA256
`9ea6afad52b70a27681630b8b09f96dfe560f6756664b627037198fe9e47ae53`.
Safe import verifies 631 regular files. Independent CPU inference and saved-source/
arithmetic checks pass: 240 assets, two snapshots at 0/50, 100 raw/PNG pairs and
metric rows, 50 original own-DGP forwards and 250 feature arrays. All original
numerical and quality bounds remain unchanged. Twenty-five tamper regressions pass.

Actual L4 preflight confirms exact zero matched identity penalty and all 26 matched
parameter-gradient assertions on all 50 cases, before optimization. The legacy
discrepancy is reproduced. The reference-processing fix therefore works, but it
does not resolve the structure failure: **0.0335635587% delivered degraded gain
against the unchanged 1% requirement at update 50**. Training stops after 50 updates/
51 backwards; final 800 never runs. Export complete:true packages the retained
failure and does not imply training success. Do not repeat V26 unchanged or relax
the stop. Original V22–V25 failures and the separate processing evidence stay intact.

All ten original-size sheets/50 paired photographic TRAIN cases are reviewed;
200 exact cells are independently checked. Eyes, nose, mouth, face outline and
overall visible appearance show no convincing V26 gain. All five own-feature
projection gradients are active and all 26 learned tensors change. Corrections
reach output, but the median degraded raw change is only 0.1348669090 byte level.
Raw degraded structure gain is 0.0151795995%, below even the delivered PNG gain;
quantization alone does not explain the failure. Worker 29.077s, fit 9.197s and
supervisor 35.001s pass timing bounds. Torch allocated peak VRAM is 1,806,989,312
bytes. There is no time-cap or CUDA failure behind the reported structure stop.

The corrected identity calculation is insufficient to establish useful restoration.
Review the corrected objective and spatial response at saved states before choosing
another training recipe. The complete optimizer-trajectory cause is not proved.
No V27 packet, unchanged retry, local gradients/optimization or assistant VM/cloud
action occurs. The user-selected own-DGP spatial/feature direction remains active.

The previous 299 bindings, deeper 697/513 bindings, twelve complete document bodies,
app 22 bindings and concurrent completed VM maintenance are preserved. No candidate
is promoted. Native CCTV stays unpaired; exposed paired TRAIN metrics are separate.
No native/reserved-final/new covering pixels or ethnicity/Zamboanga performance
claim enters this milestone. Previously useful native inputs remain usable despite
model softness. Useful native development output, input-only insufficient-information
handling, candidate app parity/full flow, all seven automatic/assisted covering
families and independent final review remain required. Goal active/incomplete.

[V26 verified result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BATCHMATCHED_IDENTITY_V26_RESULTS.md>) ·
[Milestone](<C:/xampp/htdocs/YEAR 4/Testing/outputs/dgp_batchmatched_identity_v26_audit_milestone/milestone.json>)

Previous bodies below are preserved history. Their V26-pending statements and manual
V22–V26/diagnostic launch commands are historical. V26 is closed as a failure;
do not use those commands to repeat immutable historical pilots.
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


def verify_bindings(bindings, locations=None):
    for name, digest in bindings.items():
        path = (ROOT / (locations or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve this closure; do not overwrite it'
    assert sha(PREVIOUS) == 'e7dc3ae9a6d7501fc9fc3e048e0d0b50a4a0f16543e35d5bcba9b9943617957a'
    previous = read(PREVIOUS)
    assert len(previous['new_evidence_sha256']) == 299
    verify_bindings(previous['new_evidence_sha256'])
    oldpath = ROOT / 'outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json'
    assert sha(oldpath) == previous['previous_milestone_sha256']
    old = read(oldpath)
    assert len(old['new_evidence_sha256']) == 697
    verify_bindings(old['new_evidence_sha256'], previous['previous697_original_locations'])
    deep_path = ROOT / 'outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json'
    assert sha(deep_path) == old['previous_milestone_sha256']
    deep = read(deep_path)
    assert len(deep['new_evidence_sha256']) == 513
    verify_bindings(deep['new_evidence_sha256'], old['previous513_original_locations'])
    audit = read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json')
    assert audit['complete'] and audit['returned_files_verified'] == 631
    assert audit['VM_failure_present'] and not audit['VM_training_result_present']
    assert audit['complete_snapshot_updates'] == [0, 50]
    assert audit['early_structure_stop'] == {'update': 50,
        'relative_feature_error_gain': 0.0003356355871739769, 'minimum': .01, 'pass': False}
    assert not audit['necessary_capacity_pass'] and not audit['actual_local_derivative_replay']
    verify_bindings({
        'outputs/cctv_dgp_batchmatched_identity_vm_v26/protocol.json': audit['protocol_sha256'],
        'scripts/audit_cctv_dgp_batchmatched_identity_v26.py': audit['checker_sha256'],
        'scripts/audit_cctv_dgp_batchmatched_identity_v26_execution.py': audit['execution_checker_sha256'],
        'scripts/cctv_dgp_batchmatched_identity_v26_return_rules.py': audit['identity_receipt_checker_sha256'],
        'scripts/audit_cctv_dgp_spatial_features_v25_features.py': audit['feature_checker_sha256'],
        'scripts/audit_cctv_dgp_degraded_detail_v24_cohort.py': audit['cohort_checker_sha256'],
    })
    identity = audit['identity_preflight_audit']
    assert identity['completed_main_preflight'] and identity['VM_gradient_queries_checked'] == 20
    assert identity['cases'] == 50 and identity['batches'] == 10
    assert identity['batchmatched_component_value'] == identity['batchmatched_component_gradient_norm'] == 0
    assert identity['all26_exact_zero_gradient_assertions_source_bound'] and identity['optimizer_updates'] == 0
    execution = audit['execution_audit']
    assert execution['updates'] == 50 and execution['backwards'] == 51
    assert execution['five_projection_update2_gradients_verified']
    archive = ROOT / 'outputs/cctv-dgp-batchmatched-identity-v26-results.tar.gz'
    assert sha(archive) == '9ea6afad52b70a27681630b8b09f96dfe560f6756664b627037198fe9e47ae53'
    assert archive.stat().st_size == 325764761
    imported = read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return_import.json')
    assert imported['complete'] and imported['members'] == 631 and imported['archive_sha256'] == sha(archive)
    verify_bindings({('outputs/cctv_dgp_batchmatched_identity_v26_return/' + k): v
                     for k, v in imported['files_sha256'].items()})
    folder = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_diagnostic'
    diagnostic, check, review = [read(folder / name) for name in [
        'results.json', 'independent_saved_diagnostic_audit.json', 'visual_review.json']]
    assert diagnostic['complete'] and diagnostic['head_exact_layer_traces'] == 50
    assert diagnostic['backward_calls'] == diagnostic['optimizer_updates'] == 0
    assert check['complete'] and check['exact_review_cells'] == 200 and check['source_bindings_verified'] == 625
    assert check['results_sha256'] == sha(folder / 'results.json')
    verify_bindings(diagnostic['source_bindings_sha256'])
    assert review['complete'] and review['cases_reviewed'] == 50 and review['exact_cells'] == 200
    assert review['reference_notes_recorded_from_actual_V26_images']
    assert not any(r['convincing_visible_structure_gain'] for r in review['rows'])
    assert review['diagnostic_sha256'] == sha(folder / 'results.json')
    sourcecheck = read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return_audit_preparation/independent_source_audit.json')
    assert sourcecheck['complete'] and sourcecheck['tamper_tests_passed'] == 25
    verify_bindings(sourcecheck['source_bindings_sha256'])
    apppath = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(apppath) == 'e744f3708f248012469a111e204222f66261af4ca28bbaf9521ee4c5464b5e7a'
    app = read(apppath)
    appbindings = {**app['sources_sha256'], **app['evidence_sha256']}
    assert len(appbindings) == 22
    verify_bindings(appbindings)
    for name in SOURCES:
        ast.parse((ROOT / name).read_text(encoding='utf-8'), feature_version=(3, 10))
    for name in DOCS:
        assert sha(ROOT / name) == previous['new_evidence_sha256'][name], name
    OUT.mkdir()
    (OUT / 'before_docs').mkdir()
    before = {}
    for name in DOCS:
        shutil.copy2(ROOT / name, OUT / 'before_docs' / name)
        before[name] = sha(OUT / 'before_docs' / name)
        assert before[name] == previous['new_evidence_sha256'][name], name
    assert b'VM storage cleanup completed; future VM use preserved' in (OUT / 'before_docs/PROJECT_HANDOFF.md').read_bytes()
    locations = {name: (OUT / 'before_docs' / name).relative_to(ROOT).as_posix() for name in DOCS}
    write(OUT / 'before_update_plan.json', {'date': '2026-10-06', 'before_docs_sha256': before,
        'previous_milestone_sha256': sha(PREVIOUS), 'previous299_original_locations': locations,
        'app_record_sha256': sha(apppath), 'runner_sha256': sha(Path(__file__)),
        'V26_closed_failure': True, 'original_historical_bodies_retained': True})
    for name in DOCS:
        assert sha(ROOT / name) == before[name], 'Concurrent document update: ' + name
        original = (OUT / 'before_docs' / name).read_bytes()
        split = original.index(b'\n') + 1
        (ROOT / name).write_bytes(original[:split] + b'\n' + LATEST.encode('utf-8') + b'\n' + original[split:])
        assert (ROOT / name).read_bytes().endswith(original[split:]), name
    files = DOCS + SOURCES + ['CCTV_DGP_BATCHMATCHED_IDENTITY_V26_RESULTS.md',
        'outputs/cctv-dgp-batchmatched-identity-v26-results.tar.gz',
        'outputs/cctv-dgp-batchmatched-identity-v26-results.tar.gz.sha256',
        'outputs/cctv-dgp-batchmatched-identity-v26-export.json',
        'outputs/cctv_dgp_batchmatched_identity_v26_return_import.json',
        'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json']
    paths = {ROOT / name for name in files}
    for name in ['cctv_dgp_batchmatched_identity_v26_return',
                 'cctv_dgp_batchmatched_identity_v26_return_audit_preparation',
                 'cctv_dgp_batchmatched_identity_v26_diagnostic_preparation',
                 'cctv_dgp_batchmatched_identity_v26_diagnostic']:
        paths.update(p for p in (ROOT / 'outputs' / name).rglob('*') if p.is_file())
    paths.update(p for p in OUT.rglob('*') if p.is_file())
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
    learned = {k: v for k, v in diagnostic['parameter_changes'].items() if k.endswith(('.weight', '.bias'))}
    assert len(learned) == 26 and all(v['changed_values'] > 0 for v in learned.values())
    record = {'complete': True, 'date': '2026-10-06',
        'scope': 'Actual L4 V26 processing proof, failed training, independent CPU audit and whole-face development review; Goal incomplete',
        'new_evidence_sha256': bindings, 'previous_milestone_sha256': sha(PREVIOUS),
        'previous299_original_locations': locations, 'app_record_sha256': sha(apppath),
        'archive_sha256': sha(archive), 'archive_bytes': 325764761, 'returned_files': 631,
        'protocol_sha256': audit['protocol_sha256'], 'identity_reference_correction_verified': True,
        'identity_exact_zero_assertions_source_bound': True, 'actual_VM_derivatives_replayed_locally': False,
        'VM_updates': 50, 'VM_training_backwards': 51, 'complete_snapshots': [0, 50],
        'early_structure_gain_fraction': audit['early_structure_stop']['relative_feature_error_gain'],
        'early_structure_minimum_fraction': .01, 'early_structure_pass': False,
        'final800_executed': False, 'V26_closed_failure': True, 'necessary_capacity_pass': False,
        'unchanged_original_gates_and_finite_limits': True, 'original_V22_to_V25_failures_retained': True,
        'five_FPN_projection_gradients_active': True, 'learned_tensors_changed': 26,
        'degraded_raw_gain_percent': 100 * (1 - diagnostic['groups']['degraded']['raw_update50_feature_MSE'] /
                                         diagnostic['groups']['degraded']['raw_baseline_feature_MSE']),
        'degraded_median_correction_in_byte_levels': 255 * diagnostic['groups']['degraded']['median_saved_correction_RMS'],
        'reviewed_training_cases': 50, 'exact_review_cells': 200, 'convincing_visible_gain_cases': 0,
        'independent_final_review_complete': False, 'complete_optimizer_trajectory_cause_proven': False,
        'new_regressions_passed': 25, 'new_recipe_prepared': False, 'selected_route': 'A',
        'local_inference_only_counts': {'saved_head_forwards': 100, 'recognizer_forwards': 110,
                                      'original_DGP_forwards': 50, 'exact_head_layer_traces': 50},
        'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'VM_actions': False, 'app_unchanged': True, 'app_promotion': False, 'native_or_reserved_used': False,
        'goal_status': 'active', 'goal_complete': False, 'seconds': time.monotonic() - started}
    write(OUT / 'milestone.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in [
        'new_evidence_sha256', 'previous299_original_locations']}, indent=2))
    print('Bound files:', len(bindings))


if __name__ == '__main__':
    main()
