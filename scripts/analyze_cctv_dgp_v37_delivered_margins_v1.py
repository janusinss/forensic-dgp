"""Calibrate matched PNG coarse residuals, retaining V36's108 raw/loss rows."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v37_delivered_margins_v1'
PNG = ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_analysis_v1'
RAW = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1'


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    import numpy as np
    from cctv_dgp_delivered_margin_geometry_v38 import project_affine
    assert not OUT.exists()
    a = read(PNG / 'analysis.json')
    checked = read(PNG / 'independent_analysis_audit.json')
    assert checked['complete'] and checked['V36_failures_flagged_by_coarse_predictions'] == 0
    assert a['V36_failures_flagged_by_new_coarse_prediction'] == 0
    p = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/protocol.json')
    raw = np.load(RAW / 'group_guard_matrix.npy', allow_pickle=False, mmap_mode='r')
    png = np.load(PNG / 'PNG_group_guard_matrix.npy', allow_pickle=False, mmap_mode='r')
    extras = [np.load(ROOT / f'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/state0_{c["name"]}/gradient_components.npy',
                     allow_pickle=False, mmap_mode='r')[:3] for c in p['cohorts']]
    gradients = np.concatenate([raw, *extras, png], axis=0)
    assert gradients.shape == (210, 978243) and gradients.dtype == np.float64 and np.isfinite(gradients).all()
    old_targets = np.asarray(read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/projection.json')['clearance_targets'], np.float64)
    assert old_targets.shape == (108,) and np.count_nonzero(old_targets) == 65 and np.count_nonzero(old_targets[102:]) == 0
    targets = np.concatenate([old_targets, np.zeros(102, np.float64)])
    labels = p['constraint_labels'] + a['constraint_labels']; assert len(labels) == 210
    calibration = []
    for index, label in enumerate(a['constraint_labels']):
        observations = [row for row in a['V36_all408_group_metric_scale_comparisons'] if all(row[key] == label[key] for key in ['cohort', 'group', 'metric'])]
        assert len(observations) == 4 and {r['scale'] for r in observations} == {1., .5, .25, .125}
        residuals = [max(0., r['finite_minus_coarse'] / r['scale']) for r in observations]
        maximum = max(residuals); targets[108 + index] = 2 * maximum
        calibration.append({**label, 'observations': observations, 'maximum_scale_normalized_positive_residual': maximum,
                            'empirical_margin_multiplier': 2., 'required_linear_clearance': float(targets[108 + index])})
    proposal = np.load(RAW / 'mean_original_displacement.npy', allow_pickle=False)
    theta = np.load(RAW / 'theta_before.npy', allow_pickle=False)
    candidate, proof = project_affine(gradients, proposal, targets, cap_seconds=120)
    OUT.mkdir(); np.save(OUT / 'clearance_targets.npy', targets, allow_pickle=False)
    copies = []
    if candidate is not None:
        np.save(OUT / 'candidate_displacement.npy', candidate, allow_pickle=False)
        for scale in [1., .5, .25, .125]:
            effective = theta.astype(np.float64) - (theta.astype(np.float64) - scale * candidate).astype(np.float32).astype(np.float64)
            changes = -(gradients @ effective)
            copies.append({'scale': scale, 'linear_guard_loss_changes': changes[:102].tolist(),
                           'linear_restoration_loss_changes': changes[102:108].tolist(),
                           'coarse_PNG_loss_changes': changes[108:].tolist(),
                           'minimum_raw_clearance_residual': float((gradients @ effective - scale * targets).min())})
    files = [PNG / 'analysis.json', PNG / 'independent_analysis_audit.json', PNG / 'PNG_group_guard_matrix.npy',
             RAW / 'analysis.json', RAW / 'independent_analysis_audit.json', RAW / 'group_guard_matrix.npy',
             RAW / 'mean_original_displacement.npy', RAW / 'theta_before.npy',
             ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/projection.json',
             ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/protocol.json',
             ROOT / 'scripts/cctv_dgp_delivered_margin_geometry_v38.py', Path(__file__)]
    files += [ROOT / f'outputs/cctv_dgp_v32_loss_gradient_v1_return/outputs/state0_{c["name"]}/gradient_components.npy' for c in p['cohorts']]
    result = {'complete': True, 'policy': 'Measured PNG coarse residual envelopes with retained108 V36 rows; not validated bounds',
              'original_V36_rows_retained': 108, 'new_PNG_rows': 102, 'full_rows': 210,
              'original65_clearances_byte_value_unchanged': bool(np.array_equal(targets[:108], old_targets)),
              'coarse_residual_calibration': calibration, 'nonzero_new_PNG_clearances': int(np.count_nonzero(targets[108:])),
              'constraint_labels': labels, 'projection': proof, 'float32_copy_checks': copies,
              'geometry_qualified_for_disposable_probe': proof['geometry_qualified_for_disposable_probe'],
              'all17_finite_preservation_groups_still_required': True, 'empirical_margins_not_validated_bounds': True,
              'finite_preservation_implied': False, 'no_new_model_or_gradient_calls': True, 'optimizer_updates': 0,
              'parameter_assignments': 0, 'VM_calls': 0, 'native_or_reserved_used': False,
              'training_capacity_pass': False, 'app_promotion': False, 'goal_complete': False,
              'basis_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in files},
              'artifact_sha256': {path.name: sha(path) for path in OUT.glob('*.npy')},
              'seconds': time.monotonic() - started, 'cap_seconds': 300}
    assert not any(name in sys.modules for name in ['torch', 'dgp_frozen_inference_v2', 'cctv_dgp_pilot'])
    assert result['seconds'] < 300
    with (OUT / 'analysis.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['complete', 'full_rows', 'nonzero_new_PNG_clearances', 'geometry_qualified_for_disposable_probe', 'seconds']}), flush=True)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__': main()
