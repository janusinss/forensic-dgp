"""Independent closure of V38, development rejection and completion display evidence."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v38_return_development_milestone_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(mapping, start, handoff=None, overrides=None):
    for name, digest in mapping.items():
        path = ROOT / handoff if handoff and name == 'PROJECT_HANDOFF.md' else ROOT / (overrides or {}).get(name, name)
        assert path.resolve().is_relative_to(ROOT) and sha(path) == digest, name
        assert time.monotonic() - start < 300


def main():
    start = time.monotonic(); m = read(OUT / 'milestone.json')
    assert m['complete']; verify(m['new_evidence_sha256'], start)
    before = (ROOT / m['previous_handoff_path']).read_bytes(); current = (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    at = before.index(b'\n') + 1; amount = m['document']['addition_bytes']
    assert current[:at] == before[:at] and current[at + amount:] == before[at:]
    assert hashlib.sha256(before).hexdigest() == m['document']['before_sha256']
    paths = ['outputs/cctv_dgp_v37_return_v38_probe_milestone',
             'outputs/cctv_dgp_v36_return_v37_diagnostic_milestone',
             'outputs/completion_conditioning_union_v1_milestone',
             'outputs/cctv_dgp_v35_return_v36_probe_milestone',
             'outputs/cctv_dgp_v34_return_v35_probe_milestone',
             'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone']
    child = m; counts = []
    for folder in paths:
        old = read(ROOT / folder / 'milestone.json')
        assert child['previous_milestone_sha256'] == sha(ROOT / folder / 'milestone.json')
        verify(old['new_evidence_sha256'], start, child['previous_handoff_path'], child.get('previous_file_overrides'))
        prior = read(ROOT / folder / 'independent_closure_audit.json')
        assert prior['complete'] and prior['milestone_sha256'] == sha(ROOT / folder / 'milestone.json')
        counts.append(len(old['new_evidence_sha256'])); child = old
    assert counts == [734, 2562, 539, 2230, 316, 5961]
    imp = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return_import.json')
    audit = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_independent_audit.json')
    outer = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_audit_run_v1/external_receipt.json')
    result = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return/outputs/results.json')
    assert imp['complete'] and imp['members'] == audit['members_verified'] == 2135
    assert not imp['returned_code_executed'] and audit['finite_probe_complete'] and not audit['failure_retained']
    assert audit['complete'] and outer['complete'] and not outer['timeout']
    assert result['complete'] and result['candidate_displacement_trials'] == 4 and result['raw_outputs'] == 500
    for key in ['optimizer_updates', 'committed_trajectory_updates', 'new_gradient_queries', 'backwards', 'epochs']:
        assert result[key] == 0
    assert not result['new_checkpoint_created'] and result['all_trial_states_reset']
    assert audit['CPU_replay']['outputs'] == 100 and audit['CPU_replay']['all_states_restored']
    a = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1/analysis.json')
    av = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1/visual_review.json')
    ac = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1/independent_analysis_page_audit.json')
    assert a['complete'] and a['all100_V36_original_PNG_and_raw_files_exact'] and ac['complete'] and av['complete']
    assert a['jointly_eligible_subset_variants'] == ['margin_quarter', 'margin_eighth']
    assert a['visual_cases'] == 100 and a['model_output_cells'] == 500 and a['input_target_output_cells'] == 700
    parity = read(ROOT / 'outputs/cctv_dgp_v38_quarter_single_input_parity_v1/results.json')
    assert parity['complete'] and parity['cases'] == 100 and parity['raw_maximum_error'] <= 1e-5
    assert parity['PNG_maximum_byte_difference'] <= 1
    native = ROOT / 'outputs/cctv_dgp_v38_quarter_native_development_v1'
    nr, nv, nc = [read(native / n) for n in ['results.json', 'visual_review.json', 'saved_output_audit.json']]
    assert nr['complete'] and nc['complete'] and nv['complete'] and nv['cases_reviewed'] == 24
    assert nc['all144_sheet_cells_exact'] and nc['exact_Auto_aliases'] == 24
    assert len(nr['rows']) == 24 and nr['native_evidence_unpaired'] and nv['input_only_usable_count_retained'] == 24
    assert nr['PSNR'] is None and nr['SSIM'] is None and nr['identity_accuracy'] is None
    assert not nv['restoration_qualified'] and not nr['restoration_qualified']
    paired = ROOT / 'outputs/cctv_dgp_v38_quarter_paired_development_v1'
    pr, pv, pc = [read(paired / n) for n in ['results.json', 'visual_review.json', 'saved_output_audit.json']]
    po = read(ROOT / 'outputs/cctv_dgp_v38_quarter_paired_development_run_v1/external_receipt.json')
    assert pr['complete'] and pc['complete'] and pv['complete'] and po['complete'] and not po['timeout']
    assert pc['cases_recomputed'] == len(pr['rows']) == 520 and pc['all17_groups_recomputed'] and pc['all200_preview_cells_exact']
    assert pv['cases_reviewed'] == 50 and not pv['all520_images_visually_reviewed']
    assert pr['diagnostic_preservation_failures'] == pc['diagnostic_preservation_failures']
    assert len(pr['diagnostic_preservation_failures']) == 1
    failure = pr['diagnostic_preservation_failures'][0]
    assert failure['group'] == 'dataset/thumbnails128x128/compound_lr24' and failure['metric'] == 'ArcFace_observed_fixed'
    assert failure['original'] - failure['v38_quarter'] > 1e-6
    assert not pv['restoration_qualified']
    completion = ROOT / 'outputs/completion_margin_feather_v1'
    cp, cr, cc, cv = [read(completion / n) for n in ['protocol.json', 'results.json', 'independent_saved_output_audit.json', 'visual_review.json']]
    assert cr['complete'] and cc['complete'] and cv['complete']
    assert cc['exact_outputs'] == 32 and cc['exact_bypasses'] == 4 and cc['exact_page_cells'] == 160
    assert cc['old_raw_to_PNG_compositions_reverified'] == 28 and cc['boundary_jump_decreased_cases'] == 28
    assert cc['changed_pixels_confined_to_existing_margin'] and cv['all8_pages_actually_viewed_at_original256_cell_detail']
    assert cp['automatic_outputs'] == cv['automatic_outputs'] == 0
    for key in ['automatic_quality_qualification', 'assisted_quality_qualification', 'app_adoption']:
        assert not cv[key] and not cc[key]
    assert (completion / 'audit_default_runtime_failure.json').is_file()
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14; verify(app, start)
    for receipt in [nr, pr]:
        assert receipt['local_gradient_calls'] == receipt['optimizer_updates'] == 0
        assert not receipt['reserved_final_used'] and not receipt['app_changed'] and not receipt['goal_complete']
    for key in ['VM_calls_here', 'local_gradient_calls', 'local_optimizer_updates', 'V38_optimizer_updates']:
        assert m[key] == 0
    assert m['V38_independently_audited'] and m['V36_and_V38_failed_gates_retained']
    assert not m['training_capacity_pass'] and not m['app_adoption'] and not m['goal_complete']
    assert not any(name in sys.modules for name in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    receipt = {'complete': True, 'milestone_sha256': sha(OUT / 'milestone.json'), 'checker_sha256': sha(Path(__file__)),
               'new_bindings_verified': len(m['new_evidence_sha256']), 'historical_bindings_preserved': counts,
               'entire_previous_handoff_preserved': True, 'all2135_V38_return_files_bound': True,
               'all500_TRAIN_measurements_and700_comparison_cells_audited': True,
               'quarter_and_eighth_TRAIN_only_passes': True, 'all100_single_input_replays_retained': True,
               'all24_native_faces_actually_reviewed': True, 'all520_paired_metrics_recomputed': True,
               'paired_visual_cases_reviewed': 50, 'paired_ArcFace_failure_retained': True,
               'all32_completion_variants_actually_reviewed': True, 'completion_core_visible_protected_bytes_exact': True,
               'completion_edge_metric_does_not_qualify_quality': True, 'all14_app_bindings_unchanged': True,
               'VM_calls_here': 0, 'neural_calls_in_closure': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
               'independent_final_review': False, 'automatic_quality_qualification': False,
               'assisted_quality_qualification': False, 'app_adoption': False, 'goal_complete': False,
               'seconds': time.monotonic() - start, 'cap_seconds': 300}
    assert receipt['seconds'] < 300
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: receipt[key] for key in ['complete', 'new_bindings_verified', 'historical_bindings_preserved', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
