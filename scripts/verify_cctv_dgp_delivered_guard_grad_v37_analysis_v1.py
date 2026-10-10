"""Independently assemble casewise PNG guard means and recount finite misses."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_analysis_v1'
RETURN = ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_return'
V36 = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return'


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    import numpy as np
    a = read(OUT / 'analysis.json'); assert a['complete']
    for name, digest in a['bindings_sha256'].items(): assert sha(ROOT / name) == digest, name
    for name, digest in a['artifact_sha256'].items(): assert sha(OUT / name) == digest, name
    p = read(RETURN / 'protocol.json')
    assert p['parameter_layout'] == read(ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_return/protocol.json')['parameter_layout']
    matrix = np.load(OUT / 'PNG_group_guard_matrix.npy', allow_pickle=False, mmap_mode='r')
    assert matrix.shape == (102, 978243) and matrix.dtype == np.float64
    raw = np.load(ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/group_guard_matrix.npy', allow_pickle=False, mmap_mode='r')
    theta = np.load(ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/theta_before.npy', allow_pickle=False)
    direction = np.load(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/projected_displacement.npy', allow_pickle=False)
    worst, compared, actual_failures, flagged, byte_files = 0., 0, 0, 0, 0
    for ci, cohort in enumerate(p['cohorts']):
        label, cases = cohort['name'], cohort['cases']; folder = RETURN / 'outputs' / label
        labels = a['constraint_labels'][51 * ci:51 * (ci + 1)]
        selectors = []
        for row in labels:
            group = row['group']
            if group == 'all': selected = np.ones(50, bool)
            elif group == 'clear': selected = np.array([c['profile'] == 'clear' for c in cases])
            elif group == 'degraded': selected = np.array([c['profile'] != 'clear' for c in cases])
            else:
                source, kind = group.rsplit('/', 1)
                selected = np.array([c['source'] == source and (kind == 'all' or c['profile'] == kind or
                                       kind == 'degraded' and c['profile'] != 'clear') for c in cases])
            assert selected.any() and row['cohort'] == label
            selectors.append((selected, p['guard_metrics'].index(row['metric'])))
        rebuilt = np.zeros((51, 978243), np.float64)
        for batch in range(10):
            g = np.load(folder / f'batch{batch}_guard_gradients.npy', allow_pickle=False, mmap_mode='r')
            for slot in range(5):
                index = batch * 5 + slot
                for row_index, (selected, metric) in enumerate(selectors):
                    if selected[index]: rebuilt[row_index] += g[slot * 3 + metric].astype(np.float64) / int(selected.sum())
        current = matrix[51 * ci:51 * (ci + 1)]
        error = float(np.abs(rebuilt - current).max()); assert error <= 2e-12; worst = max(worst, error)
        for index in range(51):
            left, right = current[index], raw[51 * ci + index]
            alignment = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
            assert abs(alignment - a['raw_to_coarse_alignment'][51 * ci + index]['raw_to_coarse_cosine']) <= 2e-12
        before_folder = V36 / f'outputs/state0_{label}/before'
        before = read(before_folder / 'receipt.json')
        for case in cases:
            for suffix in ['.npy', '.png']:
                assert sha(folder / (case['id'] + suffix)) == sha(before_folder / (case['id'] + suffix)); byte_files += 1
        for variant, scale in [('clearance_1', 1.), ('clearance_half', .5), ('clearance_quarter', .25), ('clearance_eighth', .125)]:
            path = V36 / f'outputs/state0_{label}/{variant}'
            receipt, comparison = read(path / 'receipt.json'), read(path / 'comparison.json')
            effective = theta.astype(np.float64) - (theta.astype(np.float64) - scale * direction).astype(np.float32).astype(np.float64)
            changes = -(rebuilt @ effective)
            selected_rows = [r for r in a['V36_all408_group_metric_scale_comparisons'] if r['cohort'] == label and r['variant'] == variant]
            assert len(selected_rows) == 51
            for i, row in enumerate(selected_rows):
                assert row['metric'] == labels[i]['metric'] and row['group'] == labels[i]['group'] and row['scale'] == scale
                key = {'PNG_MSE': 'MSE', 'one_minus_PNG_SSIM': 'SSIM', 'one_minus_PNG_ArcFace': 'ArcFace_observed_fixed'}[row['metric']]
                old, new = before['groups'][row['group']][key], receipt['groups'][row['group']][key]
                finite = new - old if key == 'MSE' else old - new
                fail = (row['group'], key) in {(r['group'], r['metric']) for r in comparison['preservation_against_original']['failures']}
                threshold = 1e-12 if key == 'MSE' else 1e-6
                predicted_bad = bool(changes[i] > threshold)
                assert abs(float(changes[i]) - row['coarse_linear_loss_change']) <= 1e-12
                assert finite == row['finite_PNG_loss_change'] and fail == row['actual_preservation_failure']
                assert abs(finite - float(changes[i]) - row['finite_minus_coarse']) <= 1e-12
                assert predicted_bad == row['coarse_prediction_exceeds_gate_tolerance']
                actual_failures += fail; flagged += fail and predicted_bad; compared += 1
        del rebuilt, current, g
        assert time.monotonic() - started < 300
    assert byte_files == 200 and compared == 408 and actual_failures == 11 and flagged == a['V36_failures_flagged_by_new_coarse_prediction'] == 0
    assert not any(name in sys.modules for name in ['torch', 'dgp_frozen_inference_v2', 'cctv_dgp_pilot'])
    result = {'complete': True, 'analysis_sha256': sha(OUT / 'analysis.json'), 'checker_sha256': sha(Path(__file__)),
              'all102_coarse_groups_rebuilt_casewise': True, 'maximum_group_element_error': worst,
              'all408_finite_comparisons_recounted': True, 'V36_PNG_failures': actual_failures,
              'V36_failures_flagged_by_coarse_predictions': flagged, 'all100_original_raw_and_PNG_files_exact': True,
              'coarse_only_feasibility_is_insufficient': True, 'local_gradients': 0, 'neural_calls': 0,
              'optimizer_updates': 0, 'VM_calls': 0, 'parameter_assignments': 0,
              'finite_preservation_implied': False, 'app_promotion': False, 'goal_complete': False,
              'seconds': time.monotonic() - started, 'cap_seconds': 300}
    with (OUT / 'independent_analysis_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['complete', 'maximum_group_element_error', 'V36_failures_flagged_by_coarse_predictions', 'seconds']}), flush=True)


if __name__ == '__main__': main()
