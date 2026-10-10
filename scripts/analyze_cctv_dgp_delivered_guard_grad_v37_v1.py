"""Assemble saved coarse guards and compare with failed finite V36 outputs.

Array arithmetic only. No model, local differentiation, parameter assignment,
optimizer, VM connection or neural training is permitted here.
"""
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
BASIS = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    import numpy as np
    assert not OUT.exists()
    audit_path = ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['diagnostic_complete'] and not audit['failure_retained']
    assert audit['fresh_CPU_replay']['outputs'] == 20 and audit['fresh_CPU_replay']['state_unchanged']
    assert read(BASIS / 'independent_analysis_audit.json')['complete']
    p = read(RETURN / 'protocol.json'); old_a = read(BASIS / 'analysis.json')
    assert old_a['parameter_layout'] == p['parameter_layout'] if 'parameter_layout' in old_a else True
    original = np.load(BASIS / 'group_guard_matrix.npy', allow_pickle=False, mmap_mode='r')
    assert original.shape == (102, 978243) and original.dtype == np.float64
    theta = np.load(BASIS / 'theta_before.npy', allow_pickle=False)
    mean = np.load(BASIS / 'mean_original_displacement.npy', allow_pickle=False)
    old_direction = np.load(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/projected_displacement.npy', allow_pickle=False)
    assert theta.dtype == np.float32 and mean.dtype == old_direction.dtype == np.float64
    assert theta.shape == mean.shape == old_direction.shape == (978243,)
    OUT.mkdir()
    destination = OUT / 'PNG_group_guard_matrix.npy'
    saved = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float64, shape=(102, 978243))
    bindings = {path.relative_to(ROOT).as_posix(): sha(path) for path in [
        audit_path, RETURN / 'protocol.json', RETURN / 'outputs/results.json', BASIS / 'analysis.json',
        BASIS / 'independent_analysis_audit.json', BASIS / 'group_guard_matrix.npy', BASIS / 'theta_before.npy',
        BASIS / 'mean_original_displacement.npy', ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/projected_displacement.npy',
        ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_independent_audit.json', Path(__file__)]}
    labels, alignments, comparisons = [], [], []
    exact_files, meaningful_new_constraints = 0, 0
    for ci, cohort in enumerate(p['cohorts']):
        label = cohort['name']; cases = cohort['cases']; folder = RETURN / 'outputs' / label
        summary = audit['coarse_gradient_summaries'][ci]; assert summary['cohort'] == label
        current_labels = summary['labels']; assert len(current_labels) == 51
        weights = np.zeros((51, 150), np.float64)
        for row_index, row in enumerate(current_labels):
            group = row['group']
            if group == 'all': ids = list(range(50))
            elif group in ['clear', 'degraded']:
                ids = [i for i, c in enumerate(cases) if (c['profile'] == 'clear') == (group == 'clear')]
            else:
                source, kind = group.rsplit('/', 1)
                ids = [i for i, c in enumerate(cases) if c['source'] == source and
                       (kind == 'all' or kind == c['profile'] or kind == 'degraded' and c['profile'] != 'clear')]
            assert ids
            metric = p['guard_metrics'].index(row['metric'])
            weights[row_index, 3 * np.asarray(ids) + metric] = 1. / len(ids)
        matrix = np.zeros((51, 978243), np.float64)
        for batch in range(10):
            assert time.monotonic() - started < 300
            path = folder / f'batch{batch}_guard_gradients.npy'
            g = np.load(path, allow_pickle=False, mmap_mode='r')
            assert g.shape == (15, 978243) and g.dtype == np.float32 and np.isfinite(g).all()
            matrix += weights[:, 15 * batch:15 * (batch + 1)] @ g.astype(np.float64)
            bindings[path.relative_to(ROOT).as_posix()] = sha(path)
        saved[51 * ci:51 * (ci + 1)] = matrix
        norms = np.linalg.norm(matrix, axis=1); assert (norms > 0).all()
        assert np.allclose(norms, summary['group_gradient_norms'], rtol=2e-10, atol=1e-11)
        assert np.allclose(matrix @ matrix.T, summary['group_gradient_gram'], rtol=2e-10, atol=1e-11)
        prior_matrix = original[51 * ci:51 * (ci + 1)]
        prior_norm = np.linalg.norm(prior_matrix, axis=1)
        cosine = np.einsum('ij,ij->i', matrix, prior_matrix) / (norms * prior_norm)
        for index, row in enumerate(current_labels):
            old = old_a['constraint_labels'][51 * ci + index]
            assert row['group'] == old['group'] and old['cohort'] == label
            expected = {'PNG_MSE': 'raw_MSE', 'one_minus_PNG_SSIM': 'one_minus_raw_SSIM',
                        'one_minus_PNG_ArcFace': 'one_minus_raw_ArcFace'}[row['metric']]
            assert old['metric'] == expected
            labels.append({'cohort': label, **row})
            alignments.append({'cohort': label, **row, 'raw_metric': expected, 'raw_gradient_norm': float(prior_norm[index]),
                               'coarse_PNG_gradient_norm': float(norms[index]), 'raw_to_coarse_cosine': float(cosine[index])})
        before_folder = V36 / f'outputs/state0_{label}/before'
        before = read(before_folder / 'receipt.json')
        bindings[(before_folder / 'receipt.json').relative_to(ROOT).as_posix()] = sha(before_folder / 'receipt.json')
        for case in cases:
            for suffix in ['.npy', '.png']:
                new, old = folder / (case['id'] + suffix), before_folder / (case['id'] + suffix)
                assert sha(new) == sha(old), 'Original image must remain exact; no restoration improvement in a gradient diagnostic'
                bindings[new.relative_to(ROOT).as_posix()] = sha(new)
                bindings[old.relative_to(ROOT).as_posix()] = sha(old)
                exact_files += 1
        for variant, scale in [('clearance_1', 1.), ('clearance_half', .5), ('clearance_quarter', .25), ('clearance_eighth', .125)]:
            path = V36 / f'outputs/state0_{label}/{variant}'
            receipt, comparison = read(path / 'receipt.json'), read(path / 'comparison.json')
            bindings[(path / 'receipt.json').relative_to(ROOT).as_posix()] = sha(path / 'receipt.json')
            bindings[(path / 'comparison.json').relative_to(ROOT).as_posix()] = sha(path / 'comparison.json')
            # Reconstruct float32-copy displacement arithmetically; no model parameters assigned.
            delta = theta.astype(np.float64) - (theta.astype(np.float64) - scale * old_direction).astype(np.float32).astype(np.float64)
            predicted = -(matrix @ delta)
            for index, row in enumerate(current_labels):
                key = {'PNG_MSE': 'MSE', 'one_minus_PNG_SSIM': 'SSIM', 'one_minus_PNG_ArcFace': 'ArcFace_observed_fixed'}[row['metric']]
                a, b = before['groups'][row['group']][key], receipt['groups'][row['group']][key]
                actual = b - a if key == 'MSE' else a - b
                fail = any(f['group'] == row['group'] and f['metric'] == key for f in comparison['preservation_against_original']['failures'])
                tolerance = 1e-12 if key == 'MSE' else 1e-6
                comparisons.append({'cohort': label, 'variant': variant, 'scale': scale, **row,
                    'coarse_linear_loss_change': float(predicted[index]), 'finite_PNG_loss_change': actual,
                    'finite_minus_coarse': float(actual - predicted[index]), 'actual_preservation_failure': fail,
                    'coarse_prediction_exceeds_gate_tolerance': bool(predicted[index] > tolerance)})
                meaningful_new_constraints += bool(fail and predicted[index] > tolerance)
        print(json.dumps({'saved_coarse_groups': 51 * (ci + 1), 'of': 102, 'cohort': label}), flush=True)
        del matrix, prior_matrix, g
    saved.flush(); del saved
    assert exact_files == 200 and len(labels) == 102 and len(comparisons) == 408
    failures = [row for row in comparisons if row['actual_preservation_failure']]
    assert len(failures) == 11
    result = {'complete': True, 'protocol_sha256': sha(RETURN / 'protocol.json'), 'coarse_group_rows': 102,
              'all100_original_raw_and_PNG_files_byte_exact': True, 'assembled_from_saved_case_gradients': 300,
              'constraint_labels': labels, 'raw_to_coarse_alignment': alignments,
              'V36_all408_group_metric_scale_comparisons': comparisons, 'V36_actual_PNG_failures': failures,
              'V36_failures_flagged_by_new_coarse_prediction': meaningful_new_constraints,
              'existing_raw_rows_and_six_restoration_rows_must_be_retained': True,
              'derivative_is_coarse_not_true_PNG_gradient': True, 'finite_output_preservation_implied': False,
              'TRAIN_cohorts_are_design_data': True, 'native_or_reserved_used': False,
              'new_model_outputs': 0, 'neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0, 'VM_calls': 0,
              'parameter_assignments': 0, 'training_capacity_pass': False, 'app_promotion': False, 'goal_complete': False,
              'bindings_sha256': bindings, 'artifact_sha256': {'PNG_group_guard_matrix.npy': sha(destination)},
              'seconds': time.monotonic() - started, 'cap_seconds': 300}
    assert not any(name in sys.modules for name in ['torch', 'dgp_frozen_inference_v2', 'cctv_dgp_pilot'])
    assert result['seconds'] < 300
    write(OUT / 'analysis.json', result)
    print(json.dumps({k: result[k] for k in ['complete', 'coarse_group_rows', 'V36_failures_flagged_by_new_coarse_prediction', 'seconds']}), flush=True)


if __name__ == '__main__': main()
