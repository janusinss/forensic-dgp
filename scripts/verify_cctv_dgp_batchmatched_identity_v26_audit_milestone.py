"""Independently read back V26 closure, preserved history and unchanged app."""
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_batchmatched_identity_v26_audit_milestone'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_bindings(bindings, locations=None):
    for name, digest in bindings.items():
        path = (ROOT / (locations or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    return len(bindings)


def main():
    started = time.monotonic()
    m = read(OUT / 'milestone.json')
    newcount = verify_bindings(m['new_evidence_sha256'])
    priorpath = ROOT / 'outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json'
    assert sha(priorpath) == m['previous_milestone_sha256']
    prior = read(priorpath)
    priorcount = verify_bindings(prior['new_evidence_sha256'], m['previous299_original_locations'])
    oldpath = ROOT / 'outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json'
    assert sha(oldpath) == prior['previous_milestone_sha256']
    old = read(oldpath)
    oldcount = verify_bindings(old['new_evidence_sha256'], prior['previous697_original_locations'])
    deeppath = ROOT / 'outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json'
    assert sha(deeppath) == old['previous_milestone_sha256']
    deep = read(deeppath)
    deepcount = verify_bindings(deep['new_evidence_sha256'], old['previous513_original_locations'])
    assert (priorcount, oldcount, deepcount) == (299, 697, 513)
    history = 0
    for name, before in m['previous299_original_locations'].items():
        original = (ROOT / before).read_bytes()
        current = (ROOT / name).read_bytes()
        split = original.index(b'\n') + 1
        assert current.startswith(original[:split]) and current.endswith(original[split:]), name
        prefix = current[split:len(current) - len(original[split:])].decode('utf-8')
        for token in ['V26 return audited', '325,764,761-byte', '0.0335635587%',
                      'unchanged 1% requirement', 'final 800 never runs', 'No V27 packet',
                      'Goal active/incomplete', 'historical pilots']:
            assert token in prefix, (name, token)
        history += 1
    assert history == 12
    assert b'VM storage cleanup completed; future VM use preserved' in (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    apppath = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(apppath) == m['app_record_sha256']
    app = read(apppath)
    appcount = verify_bindings({**app['sources_sha256'], **app['evidence_sha256']})
    assert appcount == 22
    imported = read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return_import.json')
    assert imported['complete'] and imported['members'] == 631
    assert imported['archive_sha256'] == m['archive_sha256']
    archive = ROOT / 'outputs/cctv-dgp-batchmatched-identity-v26-results.tar.gz'
    assert sha(archive) == m['archive_sha256'] and archive.stat().st_size == m['archive_bytes'] == 325764761
    returncount = verify_bindings({'outputs/cctv_dgp_batchmatched_identity_v26_return/' + name: value
                                  for name, value in imported['files_sha256'].items()})
    assert returncount == 631
    audit = read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json')
    assert audit['complete'] and audit['VM_failure_present'] and not audit['VM_training_result_present']
    verify_bindings({
        'outputs/cctv_dgp_batchmatched_identity_vm_v26/protocol.json': audit['protocol_sha256'],
        'scripts/audit_cctv_dgp_batchmatched_identity_v26.py': audit['checker_sha256'],
        'scripts/audit_cctv_dgp_batchmatched_identity_v26_execution.py': audit['execution_checker_sha256'],
        'scripts/cctv_dgp_batchmatched_identity_v26_return_rules.py': audit['identity_receipt_checker_sha256'],
        'scripts/audit_cctv_dgp_spatial_features_v25_features.py': audit['feature_checker_sha256'],
        'scripts/audit_cctv_dgp_degraded_detail_v24_cohort.py': audit['cohort_checker_sha256'],
    })
    returned = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return'
    failure = read(returned / 'outputs/failure.json')
    assert failure['updates'] == 50 and failure['backwards'] == 51 and not failure['resume_permitted']
    assert failure['cause'] == 'No one-percent early structural gain; retain stop'
    assert not (returned / 'outputs/results.json').exists()
    assert not (returned / 'outputs/update800').exists()
    baseline = read(returned / 'outputs/update0/metrics.json')['groups']['degraded']['landmark_high_frequency_MSE']
    stopped = read(returned / 'outputs/update50/metrics.json')['groups']['degraded']['landmark_high_frequency_MSE']
    gain = 1 - stopped / baseline
    early = read(returned / 'outputs/early_structure_stop.json')
    assert gain == early['relative_feature_error_gain'] == m['early_structure_gain_fraction']
    assert early['minimum'] == m['early_structure_minimum_fraction'] == .01
    assert gain < .01 and not early['pass'] and not m['early_structure_pass']
    assert audit['complete_snapshot_updates'] == m['complete_snapshots'] == [0, 50]
    proof = audit['identity_preflight_audit']
    assert proof['completed_main_preflight'] and proof['cases'] == 50 and proof['batches'] == 10
    assert proof['batchmatched_component_value'] == proof['batchmatched_component_gradient_norm'] == 0
    assert proof['all26_exact_zero_gradient_assertions_source_bound'] and proof['VM_gradient_queries_checked'] == 20
    assert proof['legacy_component_value'] > 0 and proof['legacy_component_gradient_norm'] > 0
    assert m['identity_reference_correction_verified'] and not m['actual_VM_derivatives_replayed_locally']
    folder = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_diagnostic'
    diag = read(folder / 'results.json')
    check = read(folder / 'independent_saved_diagnostic_audit.json')
    review = read(folder / 'visual_review.json')
    assert check['complete'] and check['exact_review_cells'] == review['exact_cells'] == 200
    assert check['results_sha256'] == review['diagnostic_sha256'] == sha(folder / 'results.json')
    assert review['saved_array_check_sha256'] == sha(folder / 'independent_saved_diagnostic_audit.json')
    assert review['complete'] and review['cases_reviewed'] == 50 and len(review['rows']) == 50
    assert {r['id'] for r in review['rows']} == {r['id'] for r in diag['rows']}
    assert all(r['regions_reviewed'] == ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance']
               and r['reference_note'] and not r['convincing_visible_structure_gain'] for r in review['rows'])
    assert not review['independent_final_review'] and not review['native_or_reserved_used']
    for name, digest in review['sheets_sha256'].items():
        assert sha(folder / name) == digest
    assert len(review['sheets_sha256']) == 10 and check['source_bindings_verified'] == 625
    verify_bindings(diag['source_bindings_sha256'])
    learned = {k: v for k, v in diag['parameter_changes'].items() if k.endswith(('.weight', '.bias'))}
    assert len(learned) == m['learned_tensors_changed'] == 26
    assert all(r['changed_values'] > 0 for r in learned.values())
    assert diag['parameter_changes']['kernel']['changed_values'] == 0
    assert diag['parameter_changes']['reflect_indices']['changed_values'] == 0
    g = diag['groups']['degraded']
    raw_gain = 100 * (1 - g['raw_update50_feature_MSE'] / g['raw_baseline_feature_MSE'])
    assert raw_gain == m['degraded_raw_gain_percent'] and raw_gain < gain * 100
    assert 255 * g['median_saved_correction_RMS'] == m['degraded_median_correction_in_byte_levels']
    assert g['identical_PNGs'] == 0 and g['maximum_PNG_byte_change'] == 1
    sourcecheck = read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return_audit_preparation/independent_source_audit.json')
    assert sourcecheck['complete'] and sourcecheck['tamper_tests_passed'] == m['new_regressions_passed'] == 25
    verify_bindings(sourcecheck['source_bindings_sha256'])
    v26 = read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26/protocol.json')
    v25 = read(ROOT / 'outputs/cctv_dgp_spatial_features_vm_v25/protocol.json')
    assert sha(ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26/protocol.json') == m['protocol_sha256']
    for key in ['cases', 'references', 'design', 'budgets', 'prospective_gates']:
        assert v26[key] == v25[key], key
    for name in m['new_evidence_sha256']:
        if name.endswith('.py'):
            ast.parse((ROOT / name).read_text(encoding='utf-8'), feature_version=(3, 10))
    assert m['V26_closed_failure'] and not m['final800_executed'] and not m['necessary_capacity_pass']
    assert not m['new_recipe_prepared'] and not m['complete_optimizer_trajectory_cause_proven']
    assert m['local_gradient_calls'] == m['local_backward_calls'] == m['local_optimizer_updates'] == 0
    assert not m['VM_actions'] and not m['native_or_reserved_used'] and not m['app_promotion']
    assert m['goal_status'] == 'active' and not m['goal_complete']
    result = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'milestone_sha256': sha(OUT / 'milestone.json'), 'new_bindings_verified': newcount,
        'previous299_bindings_verified': priorcount, 'deeper697_bindings_verified': oldcount,
        'deeper513_bindings_verified': deepcount, 'historical_document_bodies_preserved': history,
        'app22_bindings_verified': appcount, 'returned_files_verified': returncount,
        'whole_face_training_cases_reviewed': 50, 'exact_review_cells': 200,
        'identity_correction_and_original_structure_stop_separately_verified': True,
        'concurrent_completed_maintenance_preserved': True, 'original_gates_and_finite_limits_retained': True,
        'local_neural_calls': 0, 'local_gradient_calls': 0, 'local_backward_calls': 0,
        'local_optimizer_updates': 0, 'VM_actions': False, 'app_promotion': False,
        'independent_final_review_complete': False, 'goal_status': 'active', 'goal_complete': False,
        'seconds': time.monotonic() - started}
    with (OUT / 'independent_readback.json').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
