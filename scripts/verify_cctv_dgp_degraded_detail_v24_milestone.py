"""Independent current-binding/history/arithmetic readback; zero model calls."""
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_degraded_detail_v24_audit_and_architecture_milestone'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    milestone = read(OUT / 'milestone.json')
    for name, expected in milestone['new_evidence_sha256'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == expected, name
    previous_path = ROOT / 'outputs/dgp_detail_skip_v23_audit_and_degraded_detail_v24_milestone/milestone.json'
    assert sha(previous_path) == milestone['previous_milestone_sha256']
    previous = read(previous_path)
    for name, expected in previous['new_evidence_sha256'].items():
        path = ROOT / milestone['previous664_original_locations'].get(name, name)
        assert sha(path) == expected, name
    status_path = ROOT / 'outputs/cctv_dgp_degraded_detail_v24_reported_stop_v1/status_update_receipt.json'
    status = read(status_path)
    for name, expected in status['bindings_sha256'].items():
        path = ROOT / milestone['reported_stop_original_doc_locations'].get(name, name)
        assert sha(path) == expected, name
    histories = 0
    for name, path in milestone['reported_stop_original_doc_locations'].items():
        before, current = (ROOT / path).read_bytes(), (ROOT / name).read_bytes()
        split = before.index(b'\n') + 1
        assert current.startswith(before[:split]) and current.endswith(before[split:])
        latest = current[split:len(current) - len(before[split:])].decode('utf-8')
        assert 'V24 failure audited/reviewed' in latest and 'Goal active/incomplete' in latest
        assert 'No fourth recipe' in latest and 'No further V24 download' in latest
        histories += 1
    assert b'VM storage cleanup completed; future VM use preserved' in (ROOT / 'PROJECT_HANDOFF.md').read_bytes()
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    assert sha(app_path) == milestone['app_record_sha256']
    app = read(app_path); app_count = 0
    for name, expected in {**app['sources_sha256'], **app['evidence_sha256']}.items():
        assert sha(ROOT / name) == expected, name
        app_count += 1
    architecture_path = ROOT / 'outputs/cctv_dgp_post_v24_architecture_review_v1/review.json'
    architecture = read(architecture_path)
    for name, expected in architecture['source_bindings_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for row in architecture['three_pilots']:
        pilot = row['pilot']
        audit = read(ROOT / 'outputs' / (pilot + '_independent_audit.json'))
        trace = read(ROOT / 'outputs' / (pilot + '_diagnostic') / 'results.json')
        assert not audit['early_structure_stop']['pass'] and row['required_early_percent'] == 1
        assert row['delivered_feature_gain_percent'] == audit['early_structure_stop']['relative_feature_error_gain'] * 100
        assert row['degraded_median_observed_correction_RMS'] == trace['groups']['degraded']['median_saved_correction_RMS']
        assert row['degraded_median_observed_correction_RMS_in_byte_units'] == row['degraded_median_observed_correction_RMS'] * 255
    trace = read(ROOT / 'outputs/cctv_dgp_degraded_detail_v24_diagnostic/results.json')
    selected = [row for row in trace['rows'] if row['profile'] != 'clear']
    ratio = statistics.median(row['stages']['band_after_projection']['RMS'] / row['stages']['q_before_projection']['spatial_RMS'] for row in selected)
    assert ratio == architecture['V24_median_interior_band_RMS_over_preprojection_spatial_RMS']
    # Independent complex Fourier expression for the declared illustrative filter.
    kernel = [math.exp(-k * k / 8) for k in range(-6, 7)]
    for period, gain in architecture['declared_Gaussian_1D_highpass_sinusoid_gain'].items():
        theta = 2 * math.pi / int(period)
        value = sum(w * complex(math.cos(theta * k), math.sin(theta * k)) for k, w in zip(range(-6, 7), kernel)) / sum(kernel)
        assert abs(value.imag) < 1e-15 and abs(1 - value.real - gain) < 1e-15
    loss_check = read(ROOT / 'outputs/cctv_dgp_degraded_detail_v24_loss_audit_v1/independent_saved_loss_audit.json')
    assert architecture['corrected_loss']['clear_weighted_contribution_to_net_reduction'] == loss_check['clear_contribution']
    assert architecture['corrected_loss']['degraded_weighted_contribution_to_net_reduction'] == loss_check['degraded_contribution']
    assert architecture['corrected_loss']['net_objective_reduction'] == loss_check['overall_reduction']
    visual = read(ROOT / 'outputs/cctv_dgp_degraded_detail_v24_diagnostic/visual_review.json')
    assert visual['complete'] and len(visual['rows']) == 50 and len(visual['sheets_sha256']) == 10
    for row in visual['rows']:
        assert row['regions_reviewed'] == architecture['all_facial_features_required']
        assert not row['convincing_visible_structure_gain']
    for name in ['CCTV_DGP_DEGRADED_DETAIL_V24_RESULTS.md', 'CCTV_DGP_POST_V24_ARCHITECTURE_REVIEW.md']:
        assert (ROOT / name).is_file()
    assert milestone['goal_status'] == 'active' and not milestone['goal_complete'] and not milestone['necessary_capacity_pass']
    assert milestone['local_backward_calls'] == milestone['local_optimizer_updates'] == 0
    decision = read(ROOT / 'outputs/cctv_dgp_post_v24_architecture_review_v1/user_architecture_decision.json')
    assert decision['selected_route'] == 'A' and not milestone['architecture_choice_pending'] and not milestone['fourth_recipe_prepared']
    receipt = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'milestone_sha256': sha(OUT / 'milestone.json'), 'current_bindings_verified': len(milestone['new_evidence_sha256']),
        'previous664_bindings_verified': len(previous['new_evidence_sha256']),
        'reported_status_bindings_verified': len(status['bindings_sha256']),
        'history_bodies_exact': histories, 'concurrent_maintenance_entry_preserved': True,
        'app22_bindings_verified': app_count, 'architecture_saved_arithmetic_verified': True,
        'all50_visual_records_bound': True, 'no_fourth_recipe_prepared': True,
        'neural_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'VM_actions': False,
        'scope_limit': 'Verifies bytes, saved arithmetic and review-ledger completeness; does not supply a second human visual review, independent stage recomputation or final quality acceptance.',
        'goal_status': 'active', 'goal_complete': False, 'seconds': time.monotonic() - started}
    with (OUT / 'independent_readback.json').open('x', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
