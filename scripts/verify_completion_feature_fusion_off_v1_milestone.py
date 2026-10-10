"""Close negative architecture evidence while preserving every earlier milestone."""
import hashlib
from pathlib import Path
import sys
import time
from completion_feature_fusion_off_v1_common import ROOT, OUT as COMPARISON, sha, read, write

OUT = ROOT/'outputs/completion_feature_fusion_off_v1_milestone'


def verify(bindings, start, handoff=None, overrides=None):
    for name, digest in bindings.items():
        path = ROOT/handoff if handoff and name == 'PROJECT_HANDOFF.md' else ROOT/(overrides or {}).get(name, name)
        assert path.resolve().is_relative_to(ROOT) and sha(path) == digest, name
        assert time.monotonic()-start < 600


def main():
    start = time.monotonic(); m = read(OUT/'milestone.json'); assert m['complete']
    verify(m['new_evidence_sha256'], start)
    before = (ROOT/m['previous_handoff_path']).read_bytes(); current = (ROOT/'PROJECT_HANDOFF.md').read_bytes()
    at = before.index(b'\n')+1
    assert current[:at] == before[:at] and current[at+m['document']['addition_bytes']:] == before[at:]
    assert hashlib.sha256(before).hexdigest() == m['document']['before_sha256']
    folders = ['outputs/cctv_dgp_v39_return_v40_prepared_milestone', 'outputs/cctv_dgp_v39_prepared_milestone_v1',
               'outputs/cctv_dgp_v38_return_development_milestone_v1', 'outputs/cctv_dgp_v37_return_v38_probe_milestone',
               'outputs/cctv_dgp_v36_return_v37_diagnostic_milestone', 'outputs/completion_conditioning_union_v1_milestone',
               'outputs/cctv_dgp_v35_return_v36_probe_milestone', 'outputs/cctv_dgp_v34_return_v35_probe_milestone',
               'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone']
    child = m; historical = []
    for folder in folders:
        parent = read(ROOT/folder/'milestone.json')
        assert child['previous_milestone_sha256'] == sha(ROOT/folder/'milestone.json')
        verify(parent['new_evidence_sha256'], start, child['previous_handoff_path'], child.get('previous_file_overrides'))
        closed = read(ROOT/folder/'independent_closure_audit.json')
        assert closed['complete'] and closed['milestone_sha256'] == sha(ROOT/folder/'milestone.json')
        historical.append(len(parent['new_evidence_sha256'])); child = parent
    assert historical == [11283, 536, 9157, 734, 2562, 539, 2230, 316, 5961]
    p, r, a, v, checked = [read(COMPARISON/name) for name in ['protocol.json', 'results.json',
        'independent_saved_output_audit_r1.json', 'visual_review.json', 'independent_review_audit.json']]
    assert r['complete'] and a['complete'] and v['complete'] and checked['complete']
    assert checked['visual_review_sha256'] == sha(COMPARISON/'visual_review.json') and checked['R1_audit_sha256'] == sha(COMPARISON/'independent_saved_output_audit_r1.json')
    assert checked['R1_inverse_source_and_AST_exact'] and checked['original_checker_and_failure_retained']
    assert checked['all_exact_rules_and_quality_criteria_unchanged'] and checked['all32_case_notes_and8_actual_page_records_verified']
    assert len([row for row in v['rows'] if not row['excluded_before_generation']]) == 32
    assert v['all32_outputs_and32_w1_baselines_actually_viewed'] and v['all8_pages_actually_viewed_at_original256_cell_detail']
    assert a['all160_page_cells_exact'] and a['visible_source_bytes_exact'] == 5483088 and a['protected_source_bytes_exact'] == 1075314
    assert r['forwards'] == {'completion': 29, 'internal512': 29, 'codebook': 29, 'DGP': 0, 'detector': 0}
    assert all(value == 0 for value in r['candidate_fusion_calls'].values()) and r['state_before'] == r['state_after'] == p['parent_model_state']
    assert r['optimizer_updates'] == r['gradient_calls'] == r['backward_calls'] == v['new_automatic_outputs'] == 0
    assert not v['app_adoption'] and not v['automatic_quality_qualification'] and not v['assisted_quality_qualification'] and not v['independent_final_review']
    assert len(p['app_preservation_sha256']) == 14; verify(p['app_preservation_sha256'], start)
    previous_capacity = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_preparation/independent_packet_audit.json')
    assert previous_capacity['complete'] and previous_capacity['TRAIN_inputs'] == 3905
    assert not (ROOT/'outputs/cctv-dgp-spatial-fit-v40-results.tar.gz').exists() and not m['V40_return_present_locally'] and not m['VM_status_observed']
    paired = read(ROOT/'outputs/cctv_dgp_v38_quarter_paired_development_v1/saved_output_audit.json')
    assert len(paired['diagnostic_preservation_failures']) == 1
    assert m['original_audit_failure_retained_and_layout_only_R1_passed'] and m['exact_criteria_unchanged'] and m['DGP_primary_app_bindings_unchanged']
    assert m['local_optimizer_updates'] == m['local_gradient_calls'] == m['VM_calls'] == 0 and not m['training_started_here']
    assert not m['goal_complete'] and not m['app_promotion'] and not any(n in sys.modules for n in ['torch', 'pretrained_completion', 'dgp_face_workflow_v3'])
    assert time.monotonic()-start < 600
    write(OUT/'independent_closure_audit.json', {'complete': True, 'milestone_sha256': sha(OUT/'milestone.json'),
          'checker_sha256': sha(Path(__file__)), 'new_bindings_verified': len(m['new_evidence_sha256']),
          'historical_bindings_preserved': historical, 'entire_previous_handoff_preserved': True,
          'all32_development_output_reviews_bound': True, 'R1_layout_only_exact_audit_pass_original_failure_retained': True,
          'all14_app_bindings_unchanged': True, 'V39_audit_and_manual_V40_packet_preserved': True,
          'V38_failed_development_gate_preserved': True, 'VM_status_observed': False, 'VM_calls': 0,
          'local_neural_calls_in_closure': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
          'app_promotion': False, 'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
          'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 600})
    print({'complete': True, 'bindings': len(m['new_evidence_sha256']), 'historical': historical, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
