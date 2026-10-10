"""Trace derived ratio roundoff from stored rows; do not change any gate."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1'
RETURNED = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1_return'
OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_gate_roundoff'
sys.path.insert(0, str(PACKET))
from frozen_raw_metrics import review_groups
from frozen_capacity_contract import capacity


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def different(a, b, path=''):
    if isinstance(a, dict):
        assert set(a) == set(b)
        return [row for key in a for row in different(a[key], b[key], path+'/'+key)]
    if isinstance(a, list):
        assert len(a) == len(b)
        return [row for i, (x, y) in enumerate(zip(a, b)) for row in different(x, y, path+'/'+str(i))]
    if isinstance(a, (float, int)) and not isinstance(a, bool):
        return [] if a == b else [dict(path=path, computed=a, saved=b, absolute_difference=abs(a-b))]
    assert a == b, (path, a, b)
    return []


def main():
    assert not OUT.exists()
    OUT.mkdir()
    p = read(PACKET / 'protocol.json')
    initial = read(RETURNED / 'outputs/baseline/metrics.json')
    base = {stage: review_groups(initial['rows'], stage) for stage in ['raw', 'png']}
    entries = []
    for arm in p['arms']:
        folder = RETURNED / 'outputs' / arm['id']
        metrics = read(folder / 'metrics.json')
        gate = read(folder / 'quality_gate.json')
        for stage in ['raw', 'png']:
            groups = review_groups(metrics['rows'], stage)
            decision = capacity(base[stage], groups, .01)
            assert decision['pass'] == gate['comparisons'][stage]['pass']
            # All serialized decisions and exact failing group/metric names stay fixed.
            names = lambda g: [(row['group'], row['metric']) for row in g['preservation_failures']]
            assert names(decision) == names(gate['comparisons'][stage])
            diffs = different(decision, gate['comparisons'][stage])
            b = base[stage]['degraded']
            c = groups['degraded']
            denominator = max(b['MSE']-c['MSE'], 1e-12)
            numerator = max(0, b['MSE']-c['constant_mean_shift_only_MSE'])
            with localcontext() as ctx:
                ctx.prec = 60
                exact_ratio = Decimal(numerator)/Decimal(denominator)
            entries.append(dict(arm=arm['id'], stage=stage, saved_gate_differences=diffs,
                group_differences=different(groups, metrics['groups'][stage]),
                denominator=denominator, numerator=numerator,
                computed_brightness_fraction=decision['brightness_gain_fraction'],
                high_precision_ratio=str(exact_ratio),
                brightness_rounding_error=float(abs(exact_ratio-Decimal(decision['brightness_gain_fraction']))),
                sampled_capacity_pass=decision['pass'], preservation_failure_names=names(decision)))
    hypothesis = dict(complete=True, original_failure_sha256=sha(ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_return_audit/stderr.log'),
        original_checker_sha256=sha(ROOT / 'scripts/audit_cctv_dgp_multiscale_calibration_return_v1.py'),
        failure_location='recomputed derived capacity brightness ratio, not pixels or saved rows',
        failed_values=[0.9264908100902142,0.9264908100876813],
        difference=abs(0.9264908100902142-0.9264908100876813),
        sampled_decisions_and_failure_names_unchanged=True, entries=entries,
        proposed_repair_scope='derived brightness ratio only; leave all pixel/row/group tolerances and scientific gates unchanged',
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0, model_qualification=False)
    with (OUT / 'diagnostic.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(hypothesis, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, comparisons=24, maximum_stored_row_gate_difference=max((d['absolute_difference'] for row in entries for d in row['saved_gate_differences']), default=0),
               all_sampled_decisions_unchanged=True, original_failed_difference=hypothesis['difference']))


if __name__ == '__main__':
    main()
