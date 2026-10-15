"""Pure saved-array contracts; no model, autograd, solve or training."""
import numpy as np


def decode_sign_map(packed, shape):
    assert packed.dtype == np.uint8 and packed.ndim == 1
    assert len(shape) > 0 and all(isinstance(v, (int, np.integer)) and v > 0 for v in shape)
    units = int(np.prod(shape)); bits = np.unpackbits(packed, bitorder='little')
    assert len(packed) == (2 * units + 7) // 8 and not bits[2 * units:].any()
    positive, zero = bits[:2 * units].reshape(2, units).astype(bool)
    assert not (positive & zero).any(), 'Contradictory activation signs'
    return np.where(positive, 1, np.where(zero, 0, -1)).astype(np.int8).reshape(shape)


def project(gradient, before, after):
    assert gradient.shape == before.shape == after.shape == (256, 256, 3)
    assert gradient.dtype in [np.dtype('float32'), np.dtype('float64')]
    assert before.dtype == after.dtype == np.float32
    assert all(np.isfinite(v).all() for v in [gradient, before, after])
    assert all(v.min() >= 0 and v.max() <= 1 for v in [before, after])
    return float(np.dot(gradient.astype(np.float64).reshape(-1),
        (after.astype(np.float64) - before.astype(np.float64)).reshape(-1)))


def select_references(rows, cases, sources):
    case_map = {c['id']: c for c in cases}
    assert len(case_map) == len(cases) and all(c['role'] == 'train' for c in cases)
    selected = []
    for source in sources:
        candidates = [r for r in rows if r['repeat'] == 0 and r['pair'] == 2
                      and case_map[r['id']]['source'] == source and '/clear/' not in r['key']
                      and r['saved_double_reporting']['central_comparison_within_diagnostic_bound'] is False]
        assert candidates, 'No diagnosed failed case for source'
        candidate = sorted(candidates, key=lambda r: (-abs(r['saved_double_reporting']['central_prediction_residual']), r['id']))[0]
        selected.append(dict(source=source, reference=case_map[candidate['id']]['source_person_or_reference'],
            failure_case=candidate['id'], selection_absolute_residual=abs(candidate['saved_double_reporting']['central_prediction_residual']),
            rule='Largest absolute smallest-step saved-double residual among failed degraded TRAIN cases; reference id breaks ties'))
    assert len({r['reference'] for r in selected}) == len(sources)
    return selected
