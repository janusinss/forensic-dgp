"""Independent saved-array projection arithmetic and exact unscaled sheet audit."""
from pathlib import Path
import hashlib
import time
import sys
import numpy as np
from PIL import Image
from cctv_dgp_spatial_fit_v40_contract import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_analysis'
RETURN = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_return'


def independent(a, number):
    vectors = [v.astype(np.float64).copy() for v in a]; untouched = [v.copy() for v in vectors]
    rng = np.random.default_rng(20261008+number-1); orders = []; conflicts = 0
    for i in range(7):
        order = [int(j) for j in rng.permutation(7) if j != i]; orders.append(order)
        for j in order:
            inner = float(np.dot(vectors[i], untouched[j])); length = float(np.dot(untouched[j], untouched[j]))
            if length and inner < 0:
                vectors[i] = vectors[i] - untouched[j] * (inner/length); conflicts += 1
    merged = np.sum(np.stack(vectors), axis=0)
    return (a.astype(np.float64) @ -merged), orders, conflicts


def main():
    start = time.monotonic(); report = read(OUT/'analysis.json'); assert report['complete']
    for name, digest in report['sources_sha256'].items(): assert sha(ROOT/name) == digest, name
    original = read(RETURN/'outputs/gradient_summary.json'); cases = read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm/protocol.json')['cohorts']['optimized']
    for state, cohorts in report['reports'].items():
        for cohort, row in cohorts.items():
            a = np.load(RETURN/('outputs/'+state+'/'+cohort+'_gradient_components.npy'), allow_pickle=False)
            derivatives, orders, conflicts = independent(a, 1); saved = row['fixed_PCGrad_arithmetic']
            assert orders == saved['orders'] and conflicts == saved['conflict_projections']
            np.testing.assert_allclose(derivatives, saved['projected_direction_derivatives'], rtol=2e-12, atol=1e-14)
            norms = np.linalg.norm(a, axis=1)
            np.testing.assert_allclose(norms, row['component_norms'], rtol=2e-12, atol=1e-14)
            assert row['identity_to_landmark_gradient_norm_ratio'] == norms[6]/norms[0]
            assert row['seven_weighted_mean_losses'] == original['states'][state][cohort]['values']
            assert row['landmark_loss_gain_fraction_from_initial'] == 1-row['seven_weighted_mean_losses'][0]/original['states']['initial'][cohort]['values'][0]
    assert len(report['batches']) == 40
    for row in report['batches']:
        a = np.load(RETURN/('outputs/'+row['state']+'/gradients/'+row['cohort']+'_batch'+str(row['batch'])+'.npy'), allow_pickle=False)
        derivative, orders, conflicts = independent(a, row['batch']+1)
        assert orders == row['orders'] and conflicts == row['conflict_projections'] and max(derivative) <= 1e-12
        np.testing.assert_allclose(derivative, row['projected_direction_derivatives'], rtol=2e-12, atol=1e-14)
    actual_cases = []; cells = 0
    for page in report['pages']:
        assert sha(OUT/page['path']) == page['sha256']; actual_cases.extend(page['ids'])
        with Image.open(OUT/page['path']) as im: image = np.asarray(im).copy()
        for cell in page['cells']:
            with Image.open(ROOT/cell['source']) as im: expected = np.asarray(im).copy()
            x, y = cell['xy']; assert np.array_equal(image[y:y+256, x:x+256], expected)
            assert hashlib.sha256(expected.tobytes()).hexdigest() == cell['RGB_sha256']; cells += 1
    assert actual_cases == cases and cells == 200
    assert report['local_neural_calls'] == report['local_gradient_calls'] == report['local_optimizer_updates'] == report['VM_calls'] == 0
    assert not report['app_promotion'] and not report['goal_complete'] and 'torch' not in sys.modules
    assert time.monotonic()-start < 180
    write(OUT/'independent_analysis_audit.json', {'complete': True, 'analysis_sha256': sha(OUT/'analysis.json'), 'checker_sha256': sha(Path(__file__)),
        'all44_projection_derivative_sets_verified': True, 'saved_batches': 40, 'exact_unscaled_sheet_cells': 200,
        'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'first_order_not_a_quality_pass': True,
        'seconds': time.monotonic()-start, 'cap_seconds': 180})
    print({'complete': True, 'saved_batches': 40, 'exact_cells': 200, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
