"""Synthetic nearly correlated RAW32 constraints must retain float64 rank logic."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import numpy as np

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--record', required=True)
    args = parser.parse_args()
    contract = Path(args.contract).resolve()
    record = Path(args.record).resolve()
    assert contract.is_relative_to(ROOT) and record.is_relative_to(OUT)
    spec = importlib.util.spec_from_file_location('raw32_precision_under_test', contract)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = np.eye(32, dtype=np.float64)
    rows[-1] = np.sqrt(1 - .001**2) * rows[-2] + .001 * rows[-1]
    losses = [dict(value=.01, metric='feature_MSE' if i >= 30 else 'MSE') for i in range(32)]
    smallest = float(np.linalg.eigvalsh(rows @ rows.T)[0])
    assert smallest > 32 * np.finfo(np.float64).eps * 2
    error = None
    try:
        direction, certificate = module.solve_direction([r.copy() for r in rows], losses)
        assert certificate['smallest_eigenvalue'] == smallest
        assert certificate['rank_bound'] < smallest
        assert certificate['all32_RAW_loss_dimensions_retained']
        np.testing.assert_allclose(certificate['normalized_Gram'], rows @ rows.T, rtol=0, atol=1e-14)
        fractions = np.array([1. if i >= 30 else .1 for i in range(32)])
        assert np.all(-(rows @ direction.astype(np.float64)) / .01 >= fractions - 1e-6)
    except AssertionError as failure:
        error = str(failure)
    write(record, dict(complete=error is None, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), contract_sha256=sha(contract), failure=error,
        synthetic_Gram_smallest_eigenvalue=smallest,
        declared_synthetic_full_rank_correlated_fixture_not_model_gradient_or_training=True,
        local_neural_or_actual_model_direction_or_weight_displacement_calls=0))
    print(dict(complete=error is None, synthetic_smallest_eigenvalue=smallest, failure=error), flush=True)
    assert error is None, 'RAW32 precision regression failed'


if __name__ == '__main__':
    main()
