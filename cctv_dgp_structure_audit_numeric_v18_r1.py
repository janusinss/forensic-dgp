"""Portable accumulation check for an external audit; no training/quality edits."""
import math

import numpy as np

RELATIVE_ACCUMULATION_TOLERANCE = 2 * float(np.finfo(np.float32).eps)


def verify_training_weighting(weighting, local_means, reference_means, normalize):
    """Check float32 reduction roundoff; retain exact coefficient derivation.

    reference_means accumulates the original pointwise float32 squared errors
    in float64. This tolerance applies only to these error means, never image
    metrics, preservation gates, changed weights or network parity.
    """
    recorded = weighting['raw_MSE_means']
    keys = set(reference_means)
    if len(keys) != 10 or set(local_means) != keys or set(recorded) != keys:
        raise ValueError('Every original ten training-only group is required')
    if not all(math.isfinite(row[key]) and row[key] > 0
               for row in [reference_means, local_means, recorded] for key in keys):
        raise ValueError('Require finite positive training-only error means')
    deltas = {key: {'recorded_relative_error': abs(recorded[key] - reference_means[key]) / reference_means[key],
                    'local_relative_error': abs(local_means[key] - reference_means[key]) / reference_means[key],
                    'recorded_local_absolute_difference': abs(recorded[key] - local_means[key])}
              for key in keys}
    if any(value['recorded_relative_error'] > RELATIVE_ACCUMULATION_TOLERANCE
           or value['local_relative_error'] > RELATIVE_ACCUMULATION_TOLERANCE for value in deltas.values()):
        raise ValueError('Training-only mean exceeds float32 accumulation tolerance')
    # The frozen recipe derives Python float coefficients from the recorded
    # float32 group means. Those coefficients must still match exactly.
    expected = normalize(recorded)
    if weighting['weights'] != expected:
        raise ValueError('Recorded training coefficients differ from exact frozen derivation')
    return expected, {'complete': True, 'relative_accumulation_tolerance': RELATIVE_ACCUMULATION_TOLERANCE,
        'scope': 'Float32 training-error accumulation only; quality/parity/stop thresholds unchanged',
        'exact_recorded_coefficient_derivation': True, 'groups': deltas}
