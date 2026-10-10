"""Close new completion evidence while retaining the DGP and earlier lineage."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_conditioning_union_v1_milestone'
RUN = ROOT / 'outputs/completion_conditioning_union_v1'
PREVIOUS = ROOT / 'outputs/cctv_dgp_v35_return_v36_probe_milestone'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_bindings(bindings, handoff=None, overrides=None):
    for name, digest in bindings.items():
        path = ROOT / handoff if handoff and name == 'PROJECT_HANDOFF.md' else ROOT / (overrides or {}).get(name, name)
        assert sha(path) == digest, name


def main():
    started = time.monotonic()
    m, old = read(OUT / 'milestone.json'), read(PREVIOUS / 'milestone.json')
    assert m['complete'] and m['previous_milestone_sha256'] == sha(PREVIOUS / 'milestone.json')
    verify_bindings(m['new_evidence_sha256'])
    before, current = (ROOT / m['previous_handoff_path']).read_bytes(), (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    at, amount = before.index(b'\n') + 1, m['document']['addition_bytes']
    assert current[:at] == before[:at] and current[at + amount:] == before[at:]
    assert hashlib.sha256(before).hexdigest() == m['document']['before_sha256']
    verify_bindings(old['new_evidence_sha256'], m['previous_handoff_path'])
    earlier_path = ROOT / 'outputs/cctv_dgp_v34_return_v35_probe_milestone/milestone.json'
    earlier = read(earlier_path)
    assert old['previous_milestone_sha256'] == sha(earlier_path)
    verify_bindings(earlier['new_evidence_sha256'], old['previous_handoff_path'])
    historical_path = ROOT / 'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone/milestone.json'
    historical = read(historical_path)
    assert earlier['previous_milestone_sha256'] == sha(historical_path)
    verify_bindings(historical['new_evidence_sha256'], earlier['previous_handoff_path'], earlier['previous_file_overrides'])
    p, r, a, v, review_audit, outer = [read(RUN / name) for name in ['protocol.json', 'results.json',
                          'independent_saved_output_audit.json', 'visual_review.json', 'independent_review_support_audit.json', 'external_receipt.json']]
    assert r['complete'] and a['complete'] and v['complete'] and review_audit['complete'] and outer['complete'] and not outer['timeout']
    assert a['exact_trial_outputs'] == 32 and a['exact_union_neural_inputs_and_raw_compositions'] == 28
    assert a['visible_source_bytes_exact'] == 5483088 and a['protected_source_bytes_exact'] == 1075314
    assert a['exact_page_cells'] == 160 and a['same_final_support_as_baseline'] and a['same_pipeline_parity_PNG_and_raw_exact']
    assert a['exact_empty_bypasses'] == a['input_exclusions_retained'] == 4
    assert v['all32_outputs_and_same_final_baselines_actually_viewed'] and v['all8_pages_actually_viewed_at_original256_cell_detail']
    assert review_audit['post_output_processing_windows'] == 8 and review_audit['windows_not_new_masks_or_semantic_truth']
    assert r['state_before'] == r['state_after'] == p['parent_model_state'] and r['forwards'] == p['max_forwards']
    assert r['forwards'] == {'completion': 29, 'internal512': 29, 'DGP': 0, 'detector': 0}
    assert r['optimizer_updates'] == r['gradient_calls'] == r['backward_calls'] == 0 and not r['new_checkpoint']
    assert p['cap_seconds'] == 600 and p['external_timeout_seconds'] == 630 and r['artifact_bytes'] <= p['artifact_cap_bytes'] == 268435456
    for key in ['automatic_quality_qualification', 'assisted_quality_qualification', 'app_adoption', 'independent_final_review', 'goal_complete']:
        assert not v[key]
    assert len(p['app_preservation_sha256']) == 14
    verify_bindings(p['app_preservation_sha256'])
    prior_cover = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/visual_review.json')
    assert not prior_cover['app_adoption'] and not prior_cover['automatic_quality_qualification'] and not prior_cover['assisted_quality_qualification']
    V35 = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1/analysis.json')
    assert not V35['jointly_eligible_subset_variants'] and all(row['preservation_against_original']['failures'] for row in V35['variants'])
    V36 = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_preparation/independent_packet_audit.json')
    assert V36['complete'] and V36['all108_projection_rows_verified'] and V36['all65_empirical_clearances_independently_recalibrated']
    assert V36['VM_launches'] == V36['neural_calls'] == V36['gradient_calls'] == V36['optimizer_updates'] == 0
    assert not (ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return').exists()
    assert not any(name in sys.modules for name in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    receipt = {'complete': True, 'milestone_sha256': sha(OUT / 'milestone.json'), 'checker_sha256': sha(Path(__file__)),
               'new_bindings_verified': len(m['new_evidence_sha256']), 'previous_bindings_preserved': len(old['new_evidence_sha256']),
               'earlier_bindings_preserved': len(earlier['new_evidence_sha256']), 'historical_bindings_preserved': len(historical['new_evidence_sha256']),
               'entire_previous_handoff_preserved': True, 'all32_outputs_and8_pages_reviewed': True,
               'all28_actual_neural_inputs_and_raw_compositions_verified': True, 'all14_app_bindings_unchanged': True,
               'current_same_mask_parity_exact': True, 'zero_gradient_or_optimizer_updates': True,
               'all4_controls_and4_exclusions_retained': True, 'V35_failures_retained': True, 'V36_packet_preserved': True,
               'V36_return_present_locally': False, 'VM_calls_here': 0, 'neural_calls_in_closure': 0,
               'independent_final_review': False, 'automatic_quality_qualification': False,
               'assisted_quality_qualification': False, 'app_adoption': False, 'goal_complete': False,
               'seconds': time.monotonic() - started, 'cap_seconds': 300}
    assert receipt['seconds'] < 300
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: receipt[key] for key in ['complete', 'new_bindings_verified', 'previous_bindings_preserved',
                      'earlier_bindings_preserved', 'historical_bindings_preserved', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
