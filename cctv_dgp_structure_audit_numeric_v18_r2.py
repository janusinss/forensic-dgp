"""External V18 audit: only portable log10 rounding, never quality tolerances."""
import math


PSNR_LOG_ULPS = 4


def _bound(value):
    return PSNR_LOG_ULPS * math.ulp(value)


def verify_psnr_summaries(local, recorded):
    """All means/counts remain exact; verify PSNR against independent math.log10.

    NumPy log10 differs by at most two binary64 ULPs in the returned V18 data.
    Four ULPs cover the two libraries' transcendental rounding. This applies
    only to derived PSNR, not MSE/SSIM/cosine, stop decisions or neural replay.
    """
    if set(local) != set(recorded):
        raise ValueError('Summary group schema differs')
    rows = {}
    for group, expected in local.items():
        actual = recorded[group]
        if set(actual) != set(expected):
            raise ValueError('Summary metric schema differs')
        if {k: v for k, v in actual.items() if k != 'PSNR'} != {
                k: v for k, v in expected.items() if k != 'PSNR'}:
            raise ValueError('Exact non-PSNR summary/count differs:' + group)
        mse = expected['MSE']
        if not math.isfinite(mse) or mse < 0:
            raise ValueError('Nonfinite/negative summary MSE')
        if mse == 0:
            if actual['PSNR'] is not None or expected['PSNR'] is not None:
                raise ValueError('Zero-error PSNR must remain None')
            rows[group] = {'absolute_difference': 0., 'log10_bound': 0.}
            continue
        reference = -10 * math.log10(mse)
        bound = _bound(reference)
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) or abs(x - reference) > bound
               for x in [actual['PSNR'], expected['PSNR']]):
            raise ValueError('Derived PSNR exceeds binary64 log10 rounding bound:' + group)
        rows[group] = {'absolute_difference': abs(actual['PSNR'] - expected['PSNR']),
                       'reference_math_log10_PSNR': reference, 'log10_bound': bound}
    return {'complete': True, 'exact_non_PSNR_means_and_counts': True,
            'log10_ULPs': PSNR_LOG_ULPS, 'groups': rows}


def verify_preservation_report(expected, recorded, candidate, baseline):
    """All decision/list fields stay exact; only derived PSNR gain may round."""
    gain_key = 'degraded_PSNR_gain_dB'
    if set(recorded) != set(expected) or {
            k: v for k, v in recorded.items() if k != gain_key} != {
            k: v for k, v in expected.items() if k != gain_key}:
        raise ValueError('Exact preservation decisions/failures/MSE gain differ')
    a = -10 * math.log10(candidate['degraded']['MSE'])
    b = -10 * math.log10(baseline['degraded']['MSE'])
    reference = a - b
    # Each log-derived PSNR can round by four ULPs; subtraction adds one.
    bound = _bound(a) + _bound(b) + math.ulp(reference)
    if any(not math.isfinite(x) or abs(x - reference) > bound
           for x in [expected[gain_key], recorded[gain_key]]):
        raise ValueError('PSNR gain exceeds propagated binary64 log10 rounding bound')
    return {'complete': True, 'exact_decisions_failures_MSE_gain': True,
            'absolute_PSNR_gain_difference': abs(expected[gain_key] - recorded[gain_key]),
            'reference_math_log10_PSNR_gain': reference, 'propagated_log10_bound': bound}
