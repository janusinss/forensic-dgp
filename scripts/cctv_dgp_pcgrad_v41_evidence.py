"""Independent NumPy readback of saved VM steps; never constructs an optimizer."""
from pathlib import Path
import zipfile
import numpy as np

FIELDS = ['components', 'combined', 'applied', 'before', 'after', 'exp_avg', 'exp_avg_sq']


def projected_gradient(a, update):
    # Separate implementation from the worker's combine(): original rows remain immutable.
    originals = [v.astype(np.float64).copy() for v in a]
    result = []; orders = []; rng = np.random.default_rng(20261008+update-1); conflicts = 0
    for i, original in enumerate(originals):
        value = original.copy(); order = [int(j) for j in rng.permutation(7) if j != i]; orders.append(order)
        for j in order:
            length = np.dot(originals[j], originals[j]); inner = np.dot(value, originals[j])
            if length > 0 and inner < 0:
                value = value - (inner/length)*originals[j]; conflicts += 1
        result.append(value)
    return np.sum(result, axis=0).astype(np.float32), orders, conflicts


def read_arrays(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        assert {v.filename for v in infos} == {v+'.npy' for v in FIELDS} and len(infos) == 7
        assert sum(v.file_size for v in infos) <= 14*17952*4+4096
        assert all(v.file_size <= 7*17952*4+1024 for v in infos)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in FIELDS}
    for name, value in values.items():
        assert value.dtype == np.float32 and value.shape == ((7, 17952) if name == 'components' else (17952,))
        assert np.isfinite(value).all()
    return values


def verify_step(arrays, record, update, previous, first, second):
    a = arrays; assert record['update'] == update and record['component_queries'] == 7
    assert np.array_equal(a['before'], previous)
    expected, orders, conflicts = projected_gradient(a['components'], update)
    # CPU BLAS reduction order may straddle one float32 rounding boundary.
    allowance = 2*np.abs(np.spacing(expected)).astype(np.float64)+1e-12
    assert np.all(np.abs(expected.astype(np.float64)-a['combined'].astype(np.float64)) <= allowance)
    assert record['projection']['orders'] == orders and record['projection']['conflict_projections'] == conflicts
    norms = np.linalg.norm(a['components'].astype(np.float64), axis=1)
    np.testing.assert_allclose(norms, record['projection']['component_norms'], rtol=2e-12, atol=1e-14)
    # Retained torch clipping uses float32 tensor reductions; allow only reduction roundoff.
    scale = min(1., 1./(float(np.linalg.norm(expected.astype(np.float64)))+1e-6))
    np.testing.assert_allclose(a['applied'], (expected.astype(np.float64)*scale).astype(np.float32), rtol=3e-6, atol=1e-9)
    gradient = a['applied'].astype(np.float64)
    m = .9*first.astype(np.float64)+.1*gradient
    v = .999*second.astype(np.float64)+.001*np.square(gradient)
    np.testing.assert_allclose(a['exp_avg'], m, rtol=3e-6, atol=1e-9)
    np.testing.assert_allclose(a['exp_avg_sq'], v, rtol=3e-6, atol=1e-12)
    assert a['exp_avg_sq'].min() >= 0
    # Readback of AdamW arithmetic, not a local optimizer step or reconstructed unknown trajectory.
    predicted = a['before'].astype(np.float64)*(1-.0003*.01)
    predicted -= (.0003/(1-.9**update))*a['exp_avg'].astype(np.float64)/(np.sqrt(a['exp_avg_sq'].astype(np.float64))/(np.sqrt(1-.999**update))+1e-8)
    error = float(np.max(np.abs(predicted-a['after'].astype(np.float64))))
    assert error <= 3e-7
    assert record['optimizer_steps_all_equal_update'] and record['optimizer_type'] == 'AdamW'
    assert record['loss_weights_or_definitions_changed'] is False
    assert record['backward_calls'] == 0
    return error
