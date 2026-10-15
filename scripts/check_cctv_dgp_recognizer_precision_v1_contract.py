"""Saved-array regressions for precision projections and frozen reference selection."""
import importlib.util
import numpy as np


def main():
    assert importlib.util.find_spec('cctv_dgp_recognizer_precision_v1_contract'), 'Precision contract is not implemented'
    from cctv_dgp_recognizer_precision_v1_contract import project, select_references
    before = np.full((256, 256, 3), .5, dtype=np.float32)
    after = before.copy(); after[0, 0, 0] += np.float32(.0001)
    g = np.zeros_like(before); g[0, 0, 0] = 2
    expected = 2 * (float(after[0, 0, 0]) - .5)
    assert project(g, before, after) == project(g.astype(np.float64), before, after) == expected
    assert project(g, after, before) == -expected
    assert project(g, before, before) == 0
    import cctv_dgp_recognizer_precision_v1_contract as contract
    assert callable(getattr(contract, 'decode_sign_map', None)), 'Activation bit-map contract is not implemented'
    values = np.asarray([-1, 0, 1] * 10, dtype=np.int8).reshape(5, 2, 3)
    flags = np.stack([values > 0, values == 0])
    packed = np.packbits(flags.reshape(-1), bitorder='little')
    np.testing.assert_array_equal(contract.decode_sign_map(packed, values.shape), values)
    corrupt = flags.copy(); corrupt[1][values > 0] = True
    try:
        contract.decode_sign_map(np.packbits(corrupt.reshape(-1), bitorder='little'), values.shape)
    except AssertionError:
        pass
    else:
        raise AssertionError('Contradictory activation signs accepted')
    rejected = 0
    for x, b, a in [(g.astype(np.float16), before, after), (g, before.astype(np.float64), after),
                    (g[0], before, after), (g, before, after * 3), (g + np.nan, before, after)]:
        try:
            project(x, b, a)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('Unsafe projection input accepted')
    # A large residual that already passes is not the failure used for selection.
    rows = [dict(repeat=0, pair=2, id=f'{s}_{r}_blur', key=f'RAW/{s}/blur/ArcFace_loss',
                 saved_double_reporting=dict(central_prediction_residual=err,
                     central_comparison_within_diagnostic_bound=passed))
            for s in ['s1', 's2'] for r, err, passed in [('r1', .5, False), ('r2', 4., True)]]
    cases = [dict(id=r['id'], source=r['key'].split('/')[1], role='train',
                  source_person_or_reference=r['id'].rsplit('_', 1)[0]) for r in rows]
    selected = select_references(rows, cases, ['s1', 's2'])
    assert [v['reference'] for v in selected] == ['s1_r1', 's2_r1']
    cases[0]['role'] = 'final'
    try:
        select_references(rows, cases, ['s1', 's2'])
    except AssertionError:
        rejected += 1
    else:
        raise AssertionError('Final identity accepted for diagnostic tuning')
    print(dict(complete=True, both_gradient_precisions=True, failure_bound_selection=True,
               unsafe_inputs_rejected=rejected, neural_calls=0))


if __name__ == '__main__':
    main()
