"""Predeclared loss contrasts on saved derivatives. No new differentiation."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1'
RETURNED = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1_return/outputs'
OUT = ROOT / 'outputs/cctv_dgp_multiscale_active_anchor_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    start = time.monotonic()
    assert not OUT.exists()
    p = read(PACKET/'protocol.json')
    baseline = read(RETURNED/'baseline/metrics.json')
    audit_path = ROOT/'outputs/cctv_dgp_multiscale_calibration_v1_independent_audit_r2.json'
    audit = read(audit_path)
    assert audit['complete'] and sha(PACKET/'protocol.json') == audit['protocol_sha256']
    paths = [Path(__file__), ROOT/'scripts/audit_cctv_dgp_multiscale_active_anchor_v1.py',
             PACKET/'protocol.json', RETURNED/'baseline/metrics.json',
             RETURNED/'initial_gradients.npy', RETURNED/'gradient_preflight.json', audit_path]
    bindings = {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}
    clear = baseline['groups']['raw']['clear']['MSE']
    blur_rows = [r['raw']['MSE'] for r in baseline['rows'] if r['profile']=='blur_lr24']
    blur = float(np.mean(blur_rows))
    current = [.2/n for n in p['normalizers']]
    current[1:] = [1/n for n in p['normalizers'][1:]]
    coefficients = {}
    for name, extra in [('current', {}), ('plus_active_clear_RGB', {4: 1/clear}),
                        ('plus_active_clear_and_blur_RGB', {4: 1/clear, 7: 1/blur})]:
        weights = [0.]*14
        weights[:4] = current
        for index, value in extra.items():
            weights[index] = value
        coefficients[name] = weights
    OUT.mkdir()
    write(OUT/'plan.json', dict(complete=True, frozen_before_new_measurements=True,
        bindings=bindings, baseline_clear_RGB_MSE=clear, baseline_blur_RGB_MSE=blur,
        all3_fixed_contrasts=coefficients, pools=p['pools'],
        purpose='Test the specific inactive-first-update clear/blur anchor hypothesis in the complete RGB decoder',
        normalization='negative weighted saved gradient, divided by its L2 norm',
        not_a_selected_training_recipe=True, no_coefficient_search=True,
        no_Adam_or_step_assignment=True, worker_seconds=180,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        mathematical_optimizer_calls=0, final_pixels_decoded=0, model_qualification=False))
    matrix = np.load(RETURNED/'initial_gradients.npy', mmap_mode='r', allow_pickle=False)
    assert matrix.dtype == np.float32 and matrix.shape == (20,14,609219)
    contrasts = []
    for pool, begins in enumerate(p['pools']):
        indices = [i//5 for i in begins]
        averaged = np.zeros((14,609219),np.float64)
        for index in indices:
            averaged += matrix[index]
        averaged /= len(indices)
        for name, weights in coefficients.items():
            direction = -np.einsum('n,nk->k', np.asarray(weights), averaged, optimize=False)
            norm = float(np.linalg.norm(direction))
            assert norm > 0 and np.isfinite(direction).all()
            direction /= norm
            file = OUT/f'{name}_pool{pool}.npy'
            np.save(file, direction, allow_pickle=False)
            rows = []
            for index in range(20):
                prediction = np.einsum('nk,k->n', np.asarray(matrix[index],np.float64), direction, optimize=False)
                rows.append(dict(reference=p['references'][index]['id'],
                    source=p['cases'][index*5]['source'], fitting_reference=index in indices,
                    predictions=dict(zip(p['signals'],prediction.tolist()))))
            fitting = [row for row in rows if row['fitting_reference']]
            contrasts.append(dict(name=name, pool=pool, unnormalized_direction_L2=norm,
                direction=file.relative_to(ROOT).as_posix(), direction_sha256=sha(file), rows=rows,
                fitting_means={key:float(np.mean([r['predictions'][key] for r in fitting])) for key in p['signals']},
                all_reference_means={key:float(np.mean([r['predictions'][key] for r in rows])) for key in p['signals']},
                positive_clear_MSE_reference_slopes=sum(r['predictions']['MSE_clear']>0 for r in rows),
                positive_HF_reference_slopes=sum(r['predictions']['HF_degraded']>0 for r in rows)))
            assert time.monotonic()-start < 180
    for name,digest in bindings.items():
        assert sha(ROOT/name) == digest, name
    write(OUT/'results.json',dict(complete=True, plan_sha256=sha(OUT/'plan.json'),
        contrasts=contrasts, scalar_predictions=1680, normalized_stored_directions=6,
        first_order_only=True, nonlinear_finite_Adam_and_PNG_preservation_unproven=True,
        no_local_model_parameter_assignment=True, local_neural_calls=0,
        local_gradient_queries=0, local_optimizer_updates=0, mathematical_optimizer_calls=0,
        app_changed=False, model_qualification=False, goal_complete=False, seconds=time.monotonic()-start))
    print(dict(complete=True, fixed_contrasts=6, scalar_predictions=1680, seconds=time.monotonic()-start),flush=True)


if __name__ == '__main__':
    main()
