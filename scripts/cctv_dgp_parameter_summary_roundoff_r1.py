"""Two-ULP reduction-only checker for this returned NumPy summary diagnostic."""
import math
import struct

SUM_FIELDS = {'squared_L2', 'nominal_squared_L2', 'rounding_error_squared_L2'}
EXACT_FIELDS = {'changed_values', 'values', 'maximum_rounding_error'}
RECORDS = []


def ordered_bits(value):
    assert math.isfinite(value) and value >= 0
    return struct.unpack('>Q', struct.pack('>d', float(value)))[0]


def verify_summary(saved, recomputed, proposal, parameter):
    assert set(saved) == set(recomputed) == SUM_FIELDS | EXACT_FIELDS
    assert all(saved[key] == recomputed[key] for key in EXACT_FIELDS)
    for key in sorted(SUM_FIELDS):
        a, b = saved[key], recomputed[key]
        assert math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0
        if a == 0 or b == 0:
            assert a == b, 'A zero squared sum must remain exact'
        distance = abs(ordered_bits(a) - ordered_bits(b))
        assert distance <= 2, (proposal, parameter, key, a, b, distance)
        RECORDS.append(dict(proposal=proposal, parameter=parameter, field=key,
            VM=a, local=b, float64_ULP_distance=distance,
            allowance_float64_ULPs=2, applies_only_to_reduction_summary=True))


def fixtures():
    base = dict(changed_values=1, values=1, maximum_rounding_error=0.,
                squared_L2=1., nominal_squared_L2=1., rounding_error_squared_L2=0.)
    accepted = rejected = 0
    for steps in [0, 1, 2, 3]:
        candidate = dict(base)
        for _ in range(steps):
            candidate['squared_L2'] = math.nextafter(candidate['squared_L2'], math.inf)
        try:
            verify_summary(base, candidate, 'fixture', str(steps))
        except AssertionError:
            assert steps == 3
            rejected += 1
        else:
            assert steps <= 2
            accepted += 1
    for field, value in [('values', 2), ('changed_values', 2),
                         ('maximum_rounding_error', 1e-20),
                         ('rounding_error_squared_L2', math.ulp(0.)),
                         ('squared_L2', math.nan), ('squared_L2', -1.)]:
        candidate = dict(base)
        candidate[field] = value
        try:
            verify_summary(base, candidate, 'fixture', field)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('Unsafe summary fixture accepted: ' + field)
    RECORDS.clear()
    return dict(complete=True, accepted_0_1_2_ULP_fixtures=accepted,
        rejected_3_ULP_count_max_zero_nonfinite_and_negative_fixtures=rejected,
        numerical_fixtures_not_model_evidence=True)
