"""Real saved receipt regression; scientific acceptance flags must stay exact."""
import argparse
import copy
from datetime import datetime, timezone

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, RETURN, ANALYSIS, read, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', action='store_true')
    args = parser.parse_args()
    red = read(ANALYSIS / 'receipt_equality_red.json')
    assert not red['complete'] and red['actual_four_snapshot_pixel_and_saved_embedding_metrics_recomputed']
    if args.original:
        # Exercise the original assertion on actual independently replayed data.
        validate = lambda actual, receipt, score_groups: actual == receipt
    else:
        from conditioning_decision_receipt_check_v1 import validate_receipt
        validate = validate_receipt
    stage = RETURN / 'outputs/conditioning_learning_v1_r4'
    initial = read(stage / 'initial/snapshot.json')['groups']
    failures = []
    verified = []
    rejected_corruptions = []
    for row in red['trials']:
        saved_groups = read(stage / row['trial'] / 'snapshot.json')['groups']
        args_tuple = (row['independently_recomputed'], row['saved'], (initial, initial, saved_groups))
        try:
            result = validate(*args_tuple)
            if result is False:
                raise AssertionError('Original exact-float receipt comparison rejects the verified image replay')
            verified.append(row['trial'])
        except AssertionError as error:
            failures.append(dict(trial=row['trial'], reason=str(error)))
        if not args.original:
            cases = {}
            bad = copy.deepcopy(row['saved'])
            bad['accept_step'] = True
            cases['changed_acceptance_flag'] = (row['independently_recomputed'], bad)
            bad = copy.deepcopy(row['saved'])
            bad['stages']['raw']['initial_preservation_failures'].pop()
            cases['removed_preservation_failure'] = (row['independently_recomputed'], bad)
            bad = copy.deepcopy(row['saved'])
            bad['stages']['raw']['relative_feature_gain'] += 1e-12
            cases['unbound_saved_numeric_field'] = (row['independently_recomputed'], bad)
            bad = copy.deepcopy(row['independently_recomputed'])
            bad['stages']['raw']['brightness_gain_fraction'] = .2000000001
            cases['recomputed_brightness_branch_changed'] = (bad, row['saved'])
            bad = copy.deepcopy(row['independently_recomputed'])
            bad['stages']['png']['source_feature_gains'][next(iter(bad['stages']['png']['source_feature_gains']))] = .0100000001
            cases['recomputed_source_one_percent_branch_changed'] = (bad, row['saved'])
            bad = copy.deepcopy(row['saved'])
            bad['stages']['raw']['brightness_gain_fraction'] = float('nan')
            cases['nonfinite_receipt'] = (row['independently_recomputed'], bad)
            for label, (actual, receipt) in cases.items():
                try:
                    validate(actual, receipt, (initial, initial, saved_groups))
                except AssertionError:
                    rejected_corruptions.append(dict(trial=row['trial'], corruption=label))
                else:
                    failures.append(dict(trial=row['trial'], corruption=label, reason='Corrupt receipt accepted'))
    name = 'receipt_check_red.json' if args.original else 'receipt_check_green.json'
    value = dict(complete=not failures, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), input_red_sha256=sha(ANALYSIS / 'receipt_equality_red.json'),
        original_exact_equality=args.original, real_replayed_trials_verified=verified,
        corruptions_rejected=rejected_corruptions, failures=failures,
        new_scientific_or_numeric_replay_tolerance_introduced=False,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,
        model_qualified=False, goal_complete=False)
    if not args.original:
        value['helper_sha256'] = sha(ROOT / 'scripts/conditioning_decision_receipt_check_v1.py')
        assert len(rejected_corruptions) == 18
    write(ANALYSIS / name, value)
    print(dict(complete=value['complete'], real_trials_verified=len(verified),
               corruptions_rejected=len(rejected_corruptions)), flush=True)
    assert not failures, 'Decision receipt regression failed'


if __name__ == '__main__':
    main()
