"""Declared synthetic RAW32 solver regression, never a model-training result."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import numpy as np

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, read, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--record', required=True)
    args = parser.parse_args()
    source = Path(args.contract).resolve()
    destination = Path(args.record).resolve()
    assert source.is_relative_to(ROOT) and destination.is_relative_to(OUT)
    spec = importlib.util.spec_from_file_location('raw32_contract_under_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [dict(value=.01 + i * .0001, metric='feature_MSE' if i >= 30 else 'MSE') for i in range(32)]
    vectors = [v.copy() for v in np.eye(32)]
    rows = []
    failures = []
    try:
        direction, certificate = module.solve_direction(vectors, records)
        assert direction.shape == (32,) and direction.dtype == np.float32 and np.isfinite(direction).all()
        assert certificate['all32_RAW_loss_dimensions_retained'] and certificate['no_ridge_or_loss_removal']
        dots = np.asarray([v @ direction.astype(np.float64) for v in vectors])
        fractions = np.array([1 if i >= 30 else .1 for i in range(32)])
        values = np.asarray([r['value'] for r in records])
        assert np.all(-dots / values >= fractions - 1e-6)
        initial = np.zeros(32, np.float32)
        previous = -np.ones(32, np.float32) * .001
        rate = module.bounded_rate(initial, previous, direction, 10.)
        next_fixture = previous + np.float32(rate) * direction
        assert np.linalg.norm(next_fixture.astype(np.float64) - initial) < .01 and rate <= .02
        rows.append(dict(fixture='declared_orthonormal_RAW32', all32_constraints=True,
                         cumulative_distance=True, not_a_neural_model_or_parameter_proposal=True))
    except (AssertionError, KeyError) as error:
        failures.append(dict(fixture='declared_orthonormal_RAW32', reason=str(error), type=type(error).__name__))
    if not failures:
        rank_deficient = [v.copy() for v in vectors]
        rank_deficient[-1] = rank_deficient[-2].copy()
        try:
            module.solve_direction(rank_deficient, records)
        except AssertionError:
            rows.append(dict(fixture='declared_rank_deficiency', rejected=True))
        else:
            failures.append(dict(fixture='declared_rank_deficiency', reason='Rank-deficient constraints accepted'))
        for value in [float('nan'), float('inf')]:
            bad = [dict(v) for v in records]
            bad[0]['value'] = value
            try:
                module.solve_direction(vectors, bad)
            except AssertionError:
                rows.append(dict(fixture='declared_nonfinite_loss', value=str(value), rejected=True))
            else:
                failures.append(dict(fixture='declared_nonfinite_loss', reason='Nonfinite loss accepted'))
    report = dict(complete=not failures, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), contract_sha256=sha(source), fixtures=rows, failures=failures,
        all_fixtures_explicitly_synthetic_not_research_model_evidence=True,
        neural_calls_or_actual_model_gradients_updates_or_displacements=0,
        actual_saved_gradient_direction_not_computed_locally=True, model_qualified=False, goal_complete=False)
    write(destination, report)
    print(dict(complete=report['complete'], synthetic_fixtures=len(rows), local_neural_calls=0), flush=True)
    assert not failures, 'RAW32 finite solver fixture failed'


if __name__ == '__main__':
    main()
