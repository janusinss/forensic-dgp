"""Independent saved-arithmetic and review binding check; zero neural calls."""
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from import_cctv_dgp_bank_group_step_check_v1_return import ROOT, RETURN, ANALYSIS, PARENT, read, sha, write
from cctv_dgp_signed_step_audit import signed_differences


def main():
    receipt = ANALYSIS/'review_binding_audit.json'
    assert not receipt.exists()
    audited = read(ANALYSIS/'independent_audit_r1.json')
    finite = read(ANALYSIS/'finite_metrics.json')
    signed = read(ANALYSIS/'signed_derivative_review.json')
    actual = read(ANALYSIS/'actual_step_direction_review.json')
    pixel = read(ANALYSIS/'quantization_review.json')
    visual = read(ANALYSIS/'visual_review.json')
    contract = read(ANALYSIS/'r1_checker_contract.json')
    original = ROOT/'scripts/audit_cctv_dgp_bank_group_step_check_v1_return.py'
    repaired = ROOT/'scripts/audit_cctv_dgp_bank_group_step_check_v1_return_r1.py'
    assert contract['original_sha256'] == sha(original)
    assert contract['r1_sha256'] == audited['checker_sha256'] == sha(repaired)
    assert contract['helper_sha256'] == sha(ROOT/'scripts/cctv_dgp_parameter_summary_roundoff_r1.py')
    assert all(record['complete'] for record in [audited, finite, signed, actual, pixel, visual, contract])
    for record in [actual, pixel]:
        assert record['independent_audit_sha256'] == sha(ANALYSIS/'independent_audit_r1.json')
        assert record['finite_metrics_sha256'] == sha(ANALYSIS/'finite_metrics.json')
    predictions = read(RETURN/'direction_review.json')['float32_group_predictions']
    recomputed = signed_differences(finite['proposals'], finite['baseline_groups'], predictions)
    assert signed['groups'] == recomputed['groups']
    assert len(signed['groups']) == len(actual['groups']) == 32
    summary = []
    for label, proposal in finite['proposals'].items():
        assert not proposal['all_preview_requirements_pass']
        summary.append(dict(proposal=label, negative_step_diagnostic_only=proposal['negative_step_diagnostic_only'],
            raw_structure_gain_percent=100*proposal['gates']['raw']['relative_feature_gain'],
            PNG_structure_gain_percent=100*proposal['gates']['png']['relative_feature_gain'],
            raw_preservation_failure_count=len(proposal['gates']['raw']['preservation_failures']),
            PNG_preservation_failure_count=len(proposal['gates']['png']['preservation_failures']),
            raw_brightness_gain_fraction=proposal['gates']['raw']['brightness_gain_fraction'],
            PNG_brightness_gain_fraction=proposal['gates']['png']['brightness_gain_fraction'],
            source_structure_gains=proposal['source_structure_gains'], all_preview_requirements_pass=False))
    initial = torch.load(PARENT/'initializers/initial_B.pth', map_location='cpu', weights_only=True)
    p = read(RETURN/'protocol.json')
    steps = {}
    with np.load(RETURN/'directions.npz', allow_pickle=False) as d:
        for proposal in p['proposals']:
            steps[proposal['label']] = np.concatenate([
                ((initial[name].numpy()+np.float32(proposal['rate'])*d[f'balanced_p{i:02d}']).astype(np.float64)
                    -initial[name].numpy().astype(np.float64)).reshape(-1)
                for i, name in enumerate(p['learning_names'])])
    groups = ROOT/'outputs/cctv_dgp_bank_group_gradients_v1_r1_return/outputs/group_gradients_v1_r1'
    source_rows = {row['key']: row for row in read(groups/'run_results.json')['groups']}
    dots_checked = 0
    for row in actual['groups']:
        path = groups/source_rows[row['key']]['file']
        assert sha(path) == row['saved_gradient_sha256'] == source_rows[row['key']]['sha256']
        g = np.load(path, allow_pickle=False, mmap_mode='r')
        for label, delta in steps.items():
            # Independent chunked reduction, rather than the review's single BLAS dot.
            value = math.fsum(float(np.dot(g[i:i+65536], delta[i:i+65536])) for i in range(0, len(g), 65536))
            assert math.isclose(value, row['first_order_change_for_actual_saved_steps'][label], rel_tol=1e-12, abs_tol=1e-16)
            dots_checked += 1
        for index, h in enumerate([.001, .0001]):
            pair = row['pairs'][index]
            assert pair['radius'] == h and pair['measured_central_slope'] == signed['groups'][list(source_rows).index(row['key'])]['pairs'][index]['central_directional_derivative']
            assert pair['measured_central_slope'] < 0
    assert dots_checked == 128
    sys.path.insert(0, str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    old = read(PARENT/'protocol.json')
    refs = {r['id']: r for r in old['references']}
    metrics = {label: {r['id']: r for r in trial['rows']} for label, trial in finite['proposals'].items()}
    baselines = {r['id']: r for r in finite['baseline_rows']}
    recorded = {(r['proposal'], r['id']): r for r in pixel['rows']}
    assert len(recorded) == 400
    recount = 0
    for control in read(RETURN/'outputs/group_step_check_v1/run_results.json')['controls']:
        begin = control['begin']
        cases = old['cases'][begin:begin+5]
        raw_before, _ = unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz', [c['id'] for c in cases])
        for label in finite['proposals']:
            after = np.load(RETURN/f'outputs/group_step_check_v1/{label}/b{begin//5:03d}.npy', allow_pickle=False)
            for i, case in enumerate(cases):
                row = recorded[label, case['id']]
                with Image.open(PARENT/refs[case['source_person_or_reference']]['observed']) as im:
                    area = np.asarray(im.convert('L')) > 0
                difference = (np.floor(after[i]*np.float32(255)).astype(np.int16)
                              -np.floor(raw_before[i]*np.float32(255)).astype(np.int16))[area]
                assert row['PNG_changed_channels'] == np.count_nonzero(difference)
                assert row['PNG_changed_pixels'] == np.count_nonzero(np.any(difference != 0, axis=1))
                assert row['maximum_absolute_PNG_channel_change'] == abs(difference).max()
                assert row['raw_changed_channels'] == np.count_nonzero(after[i][area] != raw_before[i][area])
                assert row['maximum_absolute_raw_channel_change'] == abs(after[i][area].astype(np.float64)-raw_before[i][area]).max()
                for stage, prefix in [('raw', 'raw'), ('png', 'PNG')]:
                    assert row[prefix+'_MSE_change'] == metrics[label][case['id']][stage]['MSE']-baselines[case['id']][stage]['MSE']
                recount += 1
    assert recount == 400
    expected_sheets = {r['file']: r for r in audited['comparison_sheets']}
    assert visual['all20_sheets_and800_cells_inspected'] and visual['cells_inspected'] == 800
    assert len(visual['sheets']) == len(expected_sheets) == 20
    for row in visual['sheets']:
        assert row['sha256'] == expected_sheets[row['file']]['sha256'] == sha(ROOT/row['file'])
        assert row['all40_cells_inspected'] and row['observations']
        assert not row['convincing_useful_gain_over_retained_DGP']
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name) == digest for name, digest in protected.items())
    stats = audited['parameter_summary_reduction_checks']
    assert len(stats) == 444 and sum(row['float64_ULP_distance'] > 0 for row in stats) == 53
    assert all(row['float64_ULP_distance'] <= 2 for row in stats)
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        original_checker_and_original_stop_retained=True, R1_only_reduction_summary_roundoff=True,
        independent_audit_sha256=sha(ANALYSIS/'independent_audit_r1.json'),
        signed_derivative_review_sha256=sha(ANALYSIS/'signed_derivative_review.json'),
        actual_step_direction_review_sha256=sha(ANALYSIS/'actual_step_direction_review.json'),
        quantization_review_sha256=sha(ANALYSIS/'quantization_review.json'), visual_review_sha256=sha(ANALYSIS/'visual_review.json'),
        signed_pairs_recomputed=64, actual_rounded_step_gradient_dots_recomputed=128,
        quantization_case_recounts=400, sheet_hashes=20, exact_sheet_cells=800,
        summary=summary, all_four_proposals_rejected_for_adoption=True,
        no_convincing_useful_detail_gain_demonstrated=True,
        no_neural_gradients_training_or_VM_connections=True,
        protected_checkpoints_app_and_examples_exact=True, more_epochs_qualified=False,
        model_qualified=False, goal_complete=False))
    print(dict(complete=True, signed_pairs=64, rounded_step_dots=128, quantization_cases=400, visual_sheets=20))


if __name__ == '__main__':
    main()
