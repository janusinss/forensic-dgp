"""Bind receipts exactly to saved scores; keep independently replayed gate branches exact.

The caller independently recomputes all underlying image scores at the original
1e-13 replay bound before invoking this checker. Derived receipt arithmetic can
amplify those floating differences. This adds no score or quality tolerance.
"""
import math


def _shape_and_finite(actual, expected):
    assert type(actual) == type(expected)
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _shape_and_finite(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _shape_and_finite(a, e)
    elif isinstance(actual, float):
        assert math.isfinite(actual) and math.isfinite(expected)
    else:
        assert actual == expected


def _branches(value):
    stages = {}
    for stage, row in value['stages'].items():
        stages[stage] = dict(
            accept_step=row['accept_step'], one_percent_checkpoint_pass=row['one_percent_checkpoint_pass'],
            brightness_preserved=row['brightness_gain_fraction'] <= .2,
            global_one_percent=row['relative_feature_gain'] >= .01,
            source_one_percent={k: v >= .01 for k, v in row['source_feature_gains'].items()},
            source_and_global_positive={k: v > 0 for k, v in row['step_source_and_global_progress'].items()},
            initial_failure_members=[(v['group'], v['metric']) for v in row['initial_preservation_failures']],
            current_failure_members=[(v['group'], v['metric']) for v in row['current_preservation_failures']])
    return dict(accept_step=value['accept_step'],
                one_percent_checkpoint_pass=value['one_percent_checkpoint_pass'], stages=stages)


def validate_receipt(independently_recomputed, saved_receipt, saved_score_groups):
    from cctv_dgp_raw_conditioning_learning_v1_r1_contract import decision
    exact_saved_decision = decision(*saved_score_groups)
    _shape_and_finite(independently_recomputed, saved_receipt)
    assert exact_saved_decision == saved_receipt, 'Receipt is not exactly bound to its saved scientific scores'
    assert _branches(independently_recomputed) == _branches(saved_receipt), 'An independent scientific gate branch differs'
    return dict(exact_saved_score_receipt_binding=True,
                independently_recomputed_gate_branches_and_failure_members_exact=True,
                new_scientific_or_numeric_replay_tolerance_introduced=False)
