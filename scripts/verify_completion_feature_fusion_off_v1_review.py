"""Independent source/AST, layout-diagnosis, saved-review scope and binding checks."""
import ast
from pathlib import Path
import sys
import time
from completion_feature_fusion_off_v1_common import ROOT, OUT, sha, read, write, verify_bindings


def main():
    start = time.monotonic()
    p, r, a, v, fix, diagnostic = [read(OUT/name) for name in [
        'protocol.json', 'results.json', 'independent_saved_output_audit_r1.json', 'visual_review.json',
        'audit_r1_contract.json', 'replay_layout_diagnostic/results.json']]
    verify_bindings(p)
    original_path = ROOT/'scripts/audit_completion_feature_fusion_off_v1.py'
    revised_path = ROOT/'scripts/audit_completion_feature_fusion_off_v1_r1.py'
    assert sha(original_path) == fix['original_checker_sha256'] == p['sources_sha256'][original_path.relative_to(ROOT).as_posix()]
    assert sha(revised_path) == fix['R1_checker_sha256'] == a['checker_sha256']
    old = "torch.from_numpy(replay_source['neural_input512']), w=0, adain=False"
    new = "torch.from_numpy(replay_source['neural_input512']).to(memory_format=torch.channels_last), w=0, adain=False"
    old_receipt = "OUT/'independent_saved_output_audit.json'"; new_receipt = "OUT/'independent_saved_output_audit_r1.json'"
    original = original_path.read_text(); revised = revised_path.read_text()
    assert revised.count(new) == revised.count(new_receipt) == 1
    restored = revised.replace(new, old).replace(new_receipt, old_receipt)
    assert restored == original and ast.dump(ast.parse(restored)) == ast.dump(ast.parse(original))
    assert fix['original_failure_sha256'] == sha(OUT/'replay_layout_diagnostic/original_checker_failure.json')
    assert fix['diagnostic_contract_sha256'] == sha(OUT/'replay_layout_diagnostic/contract.json')
    assert fix['diagnostic_results_sha256'] == sha(OUT/'replay_layout_diagnostic/results.json')
    failed = read(OUT/'replay_layout_diagnostic/original_checker_failure.json')
    assert not failed['complete'] and failed['exit_code'] == 1 and failed['mismatched_elements'] == 131004
    assert not (OUT/'independent_saved_output_audit.json').exists()
    assert diagnostic['complete'] and diagnostic['actual_network_forwards'] == 2 and diagnostic['seconds'] < 120
    d0, d1 = diagnostic['rows']; assert d0['stride'] == [786432, 262144, 512, 1] and d1['stride'] == [786432, 1, 1536, 3]
    assert d0['input_sha256'] == d1['input_sha256'] and d0['code_indices_exact'] and d1['code_indices_exact']
    assert d0['stages']['internal512']['exact'] and d0['stages']['logits']['mismatched_elements'] == 131004
    assert all(stage['exact'] and stage['maximum_absolute_error'] == 0 for stage in d1['stages'].values())
    assert diagnostic['gradient_calls'] == diagnostic['optimizer_updates'] == diagnostic['outputs_regenerated'] == 0
    assert not diagnostic['tolerances_changed'] and not fix['new_tolerance'] and fix['all_exact_comparisons_unchanged']
    assert a['complete'] and a['results_sha256'] == v['results_sha256'] == sha(OUT/'results.json')
    assert a['protocol_sha256'] == v['protocol_sha256'] == fix['original_protocol_sha256'] == sha(OUT/'protocol.json')
    assert v['independent_saved_output_audit_r1_sha256'] == sha(OUT/'independent_saved_output_audit_r1.json')
    assert a['delivered_outputs'] == 32 and a['artifacts_verified'] == 108 and a['all160_page_cells_exact']
    assert a['visible_source_bytes_exact'] == 5483088 and a['protected_source_bytes_exact'] == 1075314
    assert a['independent_frozen_w0_network_replay']['network_raw_logits_and_encoder_exact']
    assert r['forwards'] == p['max_forwards'] and r['candidate_fusion_calls'] == p['expected_candidate_fusion_calls']
    assert v['complete'] and v['all32_outputs_and32_w1_baselines_actually_viewed'] and v['all8_pages_actually_viewed_at_original256_cell_detail']
    assert v['quality_criteria_unchanged'] == p['quality_criteria'] and v['pages'] == r['pages']
    assert [row['id'] for row in v['rows']] == [c['id'] for c in p['cases']]
    eligible = []; controls = exclusions = 0
    for row, c in zip(v['rows'], p['cases']):
        assert row['condition'] == c['condition'] and row['family'] == c['family'] and row['excluded_before_generation'] == c['rejected']
        if c['rejected']:
            assert row['input_review'] == c['input_review'] and not row['new_output_viewed']; exclusions += 1
        else:
            assert row['new_output_and_w1_baseline_actually_viewed'] and row['observed_comparison'] and row['visible_outside_final_and_protected_bytes_exact']
            assert row['hidden_reference'] is None and row['hidden_accuracy_decision'] is None and not row['independent_final_quality_review']
            controls += row['empty_bypass']; eligible.append(c['id'])
    viewed = [entry['id'] for page in v['pages'] for entry in page['entries']]
    assert len(viewed) == len(set(viewed)) == len(eligible) == 32 and set(viewed) == set(eligible) and controls == exclusions == 4
    for page in v['pages']: assert sha(OUT/page['path']) == page['sha256']
    assert v['new_automatic_outputs'] == 0 and v['all7_covering_families_remain_in_scope'] and v['no_delivered_mask_expansion']
    assert not v['automatic_quality_qualification'] and not v['assisted_quality_qualification'] and not v['app_adoption'] and not v['independent_final_review']
    assert not any(n in sys.modules for n in ['torch', 'pretrained_completion', 'dgp_face_workflow_v3'])
    assert time.monotonic()-start < 180
    write(OUT/'independent_review_audit.json', {
        'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': sha(OUT/'protocol.json'),
        'visual_review_sha256': sha(OUT/'visual_review.json'), 'R1_audit_sha256': sha(OUT/'independent_saved_output_audit_r1.json'),
        'R1_inverse_source_and_AST_exact': True, 'original_checker_and_failure_retained': True,
        'replay_layout_root_cause_two_fixed_calls': True, 'all_exact_rules_and_quality_criteria_unchanged': True,
        'all32_case_notes_and8_actual_page_records_verified': True, 'all4_controls_and4_exclusions_retained': True,
        'all14_app_bindings_unchanged': True, 'model_forwards': 0, 'gradient_calls': 0, 'optimizer_updates': 0,
        'VM_calls': 0, 'app_adoption': False, 'automatic_quality_qualification': False,
        'assisted_quality_qualification': False, 'independent_final_review': False,
        'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 180})
    print({'complete': True, 'case_notes': 32, 'exact_rules_unchanged': True, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
