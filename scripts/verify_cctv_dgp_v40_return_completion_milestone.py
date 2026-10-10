"""Independent full history closure for audited negative restoration/completion."""
from pathlib import Path
import hashlib
import sys
import time
from completion_feature_fusion_off_v1_common import ROOT, OUT as COMPLETION, sha, read, write
OUT = ROOT/'outputs/cctv_dgp_v40_return_completion_milestone'


def verify(bindings, start, handoff=None, overrides=None):
    for name, digest in bindings.items():
        path = ROOT/handoff if handoff and name == 'PROJECT_HANDOFF.md' else ROOT/(overrides or {}).get(name, name)
        assert path.resolve().is_relative_to(ROOT) and sha(path) == digest, name
        assert time.monotonic()-start < 600


def main():
    start = time.monotonic(); m = read(OUT/'milestone.json'); assert m['complete']; verify(m['new_evidence_sha256'], start)
    before = (ROOT/m['previous_handoff_path']).read_bytes(); current = (ROOT/'PROJECT_HANDOFF.md').read_bytes(); at = before.index(b'\n')+1
    assert hashlib.sha256(before).hexdigest() == m['document']['before_sha256']
    assert current[:at] == before[:at] and current[at+m['document']['addition_bytes']:] == before[at:]
    folders = ['outputs/cctv_dgp_v39_return_v40_prepared_milestone', 'outputs/cctv_dgp_v39_prepared_milestone_v1',
               'outputs/cctv_dgp_v38_return_development_milestone_v1', 'outputs/cctv_dgp_v37_return_v38_probe_milestone',
               'outputs/cctv_dgp_v36_return_v37_diagnostic_milestone', 'outputs/completion_conditioning_union_v1_milestone',
               'outputs/cctv_dgp_v35_return_v36_probe_milestone', 'outputs/cctv_dgp_v34_return_v35_probe_milestone',
               'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone']
    child = m; counts = []
    for folder in folders:
        parent = read(ROOT/folder/'milestone.json'); assert child['previous_milestone_sha256'] == sha(ROOT/folder/'milestone.json')
        verify(parent['new_evidence_sha256'], start, child['previous_handoff_path'], child.get('previous_file_overrides'))
        closed = read(ROOT/folder/'independent_closure_audit.json'); assert closed['complete'] and closed['milestone_sha256'] == sha(ROOT/folder/'milestone.json')
        counts.append(len(parent['new_evidence_sha256'])); child = parent
    assert counts == [11283, 536, 9157, 734, 2562, 539, 2230, 316, 5961]
    a = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json')
    v = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_failure_review/visual_review.json')
    va = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_failure_review/independent_review_audit.json')
    assert a['complete'] and a['members_verified'] == 27451 and a['failure_retained'] and not a['necessary_capacity_pass'] and not a['training_finished800']
    assert [s['update'] for s in a['snapshots']] == [0, 50] and [s['PNG_cases'] for s in a['snapshots']] == [3905, 3905]
    assert [s['raw_previews_composed_exact'] for s in a['snapshots']] == [50, 50] and [r['cases'] for r in a['CPU_replays']] == [50, 50]
    assert not a['gates'][0]['pass'] and a['gates'][0]['minimum_gain'] == .01 and not a['gates'][0]['preservation_failures']
    assert va['complete'] and va['visual_review_sha256'] == sha(ROOT/'outputs/cctv_dgp_spatial_fit_v40_failure_review/visual_review.json')
    assert v['all50_previews_and250_comparison_cells_actually_viewed'] and len(v['rows']) == 50 and not v['quality_qualification']
    ca, cv, checked = [read(COMPLETION/name) for name in ['independent_saved_output_audit_r1.json', 'visual_review.json', 'independent_review_audit.json']]
    assert ca['complete'] and checked['complete'] and checked['R1_inverse_source_and_AST_exact'] and checked['original_checker_and_failure_retained']
    assert checked['visual_review_sha256'] == sha(COMPLETION/'visual_review.json') and checked['R1_audit_sha256'] == sha(COMPLETION/'independent_saved_output_audit_r1.json')
    assert ca['all160_page_cells_exact'] and ca['visible_source_bytes_exact'] == 5483088 and ca['protected_source_bytes_exact'] == 1075314
    assert cv['all32_outputs_and32_w1_baselines_actually_viewed'] and not cv['app_adoption'] and not cv['automatic_quality_qualification'] and not cv['assisted_quality_qualification']
    stop = read(COMPLETION/'milestone_record_stop.json'); assert stop['handoff_unchanged_verified'] and stop['milestone_folder_not_created']
    assert stop['recorder_sha256'] == sha(ROOT/'scripts/record_completion_feature_fusion_off_v1_milestone.py')
    assert not (ROOT/'outputs/completion_feature_fusion_off_v1_milestone').exists()
    app = read(COMPLETION/'protocol.json')['app_preservation_sha256']; assert len(app) == 14; verify(app, start)
    paired = read(ROOT/'outputs/cctv_dgp_v38_quarter_paired_development_v1/saved_output_audit.json'); assert len(paired['diagnostic_preservation_failures']) == 1
    assert m['V40_updates'] == 50 and m['V40_structure_requirement_failed'] and m['DGP_primary_app_bindings_unchanged']
    assert m['VM_calls'] == m['local_gradient_calls'] == m['local_optimizer_updates'] == 0 and not m['training_started_here']
    assert not m['goal_complete'] and not m['app_promotion'] and not any(n in sys.modules for n in ['torch', 'pretrained_completion', 'dgp_face_workflow_v3'])
    assert time.monotonic()-start < 600
    write(OUT/'independent_closure_audit.json', {'complete': True, 'milestone_sha256': sha(OUT/'milestone.json'), 'checker_sha256': sha(Path(__file__)),
          'new_bindings_verified': len(m['new_evidence_sha256']), 'historical_bindings_preserved': counts,
          'entire_previous_handoff_preserved': True, 'all50_TRAIN_and32_completion_reviews_bound': True,
          'V40_original_failed_gate_and_all_checkpoints_retained': True, 'R1_layout_only_exact_audit_and_original_failure_preserved': True,
          'all14_app_bindings_unchanged': True, 'V38_failed_development_gate_preserved': True,
          'VM_calls': 0, 'local_neural_calls_in_closure': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
          'app_promotion': False, 'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
          'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 600})
    print({'complete': True, 'bindings': len(m['new_evidence_sha256']), 'historical': counts, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
