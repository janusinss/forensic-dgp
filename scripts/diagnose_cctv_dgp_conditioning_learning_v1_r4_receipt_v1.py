"""Reproduce the saved decision-receipt equality stop without neural calls."""
from datetime import datetime, timezone
import math
import time

import numpy as np

from audit_cctv_dgp_conditioning_learning_v1_r4_return import (
    ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write,
    schema, audit_snapshot, decision,
)


def differences(actual, expected, location=''):
    if isinstance(actual, dict):
        assert isinstance(expected, dict) and actual.keys() == expected.keys(), location
        return [row for key in actual
                for row in differences(actual[key], expected[key], location + '/' + key)]
    if isinstance(actual, list):
        assert isinstance(expected, list) and len(actual) == len(expected), location
        return [row for index, (a, e) in enumerate(zip(actual, expected))
                for row in differences(a, e, location + '/' + str(index))]
    assert type(actual) == type(expected), (location, type(actual), type(expected))
    if actual == expected:
        return []
    assert isinstance(actual, float) and math.isfinite(actual) and math.isfinite(expected), location
    return [dict(location=location, recomputed=actual, saved=expected,
                 absolute_difference=abs(actual - expected))]


def main():
    start = time.monotonic()
    assert not (ANALYSIS / 'receipt_equality_red.json').exists()
    imported = read(ANALYSIS / 'import.json')
    assert imported['complete'] and imported['protocol_sha256'] == PIN
    write(ANALYSIS / 'original_checker_stop.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(ROOT / 'scripts/audit_cctv_dgp_conditioning_learning_v1_r4_return.py'),
        observed_tool_chunk_id='e32665', exit_code=1,
        location='scripts/audit_cctv_dgp_conditioning_learning_v1_r4_return.py:210',
        failed_assertion="assert checked==trial['decision']",
        original_checker_retained_unchanged=True, audit_not_complete=True))
    p, old = read(RETURN / 'protocol.json'), read(PARENT / 'protocol.json')
    out = RETURN / 'outputs/conditioning_learning_v1_r4'
    virtual = schema(RETURN)
    names = p['conditioning_names']
    edges = np.cumsum([0] + [int(np.prod(p['conditioning_shapes'][n])) for n in names])
    groups = {}
    for label in ['initial', 'r00_trial00', 'r00_trial01', 'r00_trial02']:
        assert time.monotonic() - start < 240
        _, groups[label], _ = audit_snapshot(out / label, p, old, virtual, names, edges)
        print(dict(recomputed_snapshot=label, model_inference_calls=0), flush=True)
    rows = []
    for label in ['r00_trial00', 'r00_trial01', 'r00_trial02']:
        stored = read(out / label / 'decision.json')['decision']
        checked = decision(groups['initial'], groups['initial'], groups[label])
        delta = differences(checked, stored)
        rows.append(dict(trial=label, independently_recomputed=checked, saved=stored,
                         bitwise_receipt_equal=checked == stored, differences=delta,
                         maximum_absolute_difference=max((v['absolute_difference'] for v in delta), default=0),
                         all_boolean_string_integer_layout_and_failure_members_exact=True))
    maximum = max(row['maximum_absolute_difference'] for row in rows)
    exact = all(row['bitwise_receipt_equal'] for row in rows)
    write(ANALYSIS / 'receipt_equality_red.json', dict(
        complete=exact, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), protocol_sha256=PIN, import_sha256=sha(ANALYSIS / 'import.json'),
        original_checker_stop_sha256=sha(ANALYSIS / 'original_checker_stop.json'),
        actual_four_snapshot_pixel_and_saved_embedding_metrics_recomputed=True,
        trials=rows, maximum_absolute_receipt_difference=maximum,
        scientific_group_metric_replay_bound=1e-13, actual_scientific_gates_unchanged=True,
        no_local_neural_gradient_optimizer_or_parameter_displacement_calls=0,
        seconds=time.monotonic() - start, model_qualified=False, goal_complete=False))
    print(dict(exact_receipt_equality=exact, maximum_difference=maximum,
               unchanged_scientific_replay_bound=1e-13), flush=True)
    assert exact, 'Exact equality of independently replayed decision float receipts is invalid'


if __name__ == '__main__':
    main()
