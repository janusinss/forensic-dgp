"""Pure finite-loss arithmetic and a declared affine-residual activation model.

No model, autograd, optimizer, parameter assignment or acceptance gate lives in
this module.  A right derivative of this scalar model is not a certified right
derivative of a neural network or of literal float32/PNG execution.
"""
import numpy as np

PROFILES = ('clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24')
RECONSTRUCTION_WEIGHTS = (.2, 1., 1., 1.)
REGRESSION_OFFSETS = {'MSE': 1e-12, 'SSIM_loss': 1e-6, 'ArcFace_loss': 1e-6}
METRIC_DEFINITIONS = {
    'structure_error': 'landmark-supported high-frequency luminance MSE; RAW and PNG are separate',
    'structure_gain': '1 - candidate_degraded_mean_landmark_high_frequency_MSE / baseline_degraded_mean_landmark_high_frequency_MSE',
    'ArcFace_observed_fixed': 'fixed recognition similarity, a separate preservation metric; not structure gain',
    'training_landmark_structure': 'retained differentiable training term; scientific RAW/PNG gain is independently recomputed',
}


def _array(value, shape, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or array.shape != shape:
        raise ValueError(name+' must be a finite numeric array with shape '+str(shape))
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(name+' must be finite')
    return array


def _parameters(normalizers, anchor_normalizers, reconstruction_weights):
    normalizers = _array(normalizers, (4,), 'normalizers')
    anchors = _array(anchor_normalizers, (2,), 'anchor_normalizers')
    weights = _array(reconstruction_weights, (4,), 'reconstruction_weights')
    if (normalizers <= 0).any() or (anchors <= 0).any():
        raise ValueError('All normalization values must be positive')
    if not np.array_equal(weights, np.asarray(RECONSTRUCTION_WEIGHTS)):
        raise ValueError('Retain the frozen reconstruction weights; this is not a weight search')
    return normalizers, anchors, weights


def _terms(values, structure=True, nonnegative=False, prefix='terms'):
    keys = ['MSE', 'SSIM_loss', 'ArcFace_loss']+(['landmark_structure'] if structure else [])
    try:
        result = {key: _array(values[key], (5,), prefix+'.'+key) for key in keys}
    except (KeyError, TypeError) as error:
        raise ValueError('Missing named five-profile '+prefix) from error
    if nonnegative and ((result['MSE'] < 0).any() or
                        (structure and (result['landmark_structure'] < 0).any())):
        raise ValueError('MSE and landmark error values cannot be negative')
    return result


def _photo(values, prefix='photometric', nonnegative=True):
    try:
        result = {key: _array(values[key], (5,), prefix+'.'+key)
                  for key in ['old_MSE', 'actual_MSE', 'mean_only_MSE']}
    except (KeyError, TypeError) as error:
        raise ValueError('Provide explicit photometric old/actual/mean-only MSE arrays') from error
    if nonnegative and any((value < 0).any() for value in result.values()):
        raise ValueError('Photometric MSE values cannot be negative')
    return result


def relu_right_derivative(residual, slope):
    """Exact right rule for max(0, residual + alpha*slope), alpha -> 0+."""
    left, right = np.asarray(residual), np.asarray(slope)
    if left.shape != right.shape or left.dtype.kind not in 'fiu' or right.dtype.kind not in 'fiu':
        raise ValueError('Residual and slope require identical numeric shapes')
    left, right = left.astype(np.float64), right.astype(np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Residual and slope must be finite')
    answer = np.where(left > 0., right, np.where(left == 0., np.maximum(right, 0.), 0.))
    return float(answer) if answer.ndim == 0 else answer


def affine_hinge_events(residual, slope):
    """Finite crossing distances in an affine model; no network-validity claim."""
    left, right = np.asarray(residual), np.asarray(slope)
    derivative = np.asarray(relu_right_derivative(left, right))
    if left.ndim != 1:
        raise ValueError('Event lists require one-dimensional residuals and slopes')
    rows = []
    for index, (value, direction, chosen) in enumerate(zip(left, right, derivative)):
        value, direction = float(value), float(direction)
        distance = -value/direction if value < 0. and direction > 0. else 0. if value == 0. and direction > 0. else None
        leaving = value/-direction if value > 0. and direction < 0. else None
        rows.append(dict(index=index, residual=value, slack=max(0., -value), slope=direction,
            state='active' if value > 0. else 'equality' if value == 0. else 'inactive_with_slack',
            right_derivative=float(chosen), activation_distance=distance, deactivation_distance=leaving))
    return rows


def finite_objective(current, baseline, normalizers, anchor_normalizers, *,
                     photometric, reconstruction_weights=RECONSTRUCTION_WEIGHTS):
    """Reproduce the frozen scalar formula from explicit saved loss values.

    current: MSE, SSIM_loss, ArcFace_loss, landmark_structure, each shape (5,).
    baseline: frozen MSE, SSIM_loss, ArcFace_loss, each shape (5,).
    photometric: explicit old_MSE, actual_MSE, mean_only_MSE from the guard.
    Float64 arithmetic implements the formula; CUDA reduction rounding is not a
    scientific tolerance or a claim of bit-exact cross-device reductions.
    """
    terms = _terms(current, nonnegative=True)
    old = _terms(baseline, structure=False, nonnegative=True)
    photo = _photo(photometric)
    ns, an, weights = _parameters(normalizers, anchor_normalizers, reconstruction_weights)
    averaged = np.asarray([terms['MSE'].mean(), terms['SSIM_loss'].mean(),
        terms['ArcFace_loss'].mean(), terms['landmark_structure'][1:].mean()])
    weighted = weights*averaged/ns
    residuals = {key: terms[key]-old[key]-offset for key, offset in REGRESSION_OFFSETS.items()}
    barriers = np.asarray([5.*np.maximum(residuals[key], 0.).mean()/ns[index]
                           for index, key in enumerate(REGRESSION_OFFSETS)])
    anchors = np.asarray([terms['MSE'][0]/an[0], terms['MSE'][1]/an[1]])
    mean_gain = np.maximum(photo['old_MSE']-photo['mean_only_MSE'], 0.)
    actual_gain = np.maximum(photo['old_MSE']-photo['actual_MSE'], 0.)
    excess = np.maximum(mean_gain-.2*actual_gain, 0.)
    penalty = float(10.*excess[1:].mean()/ns[0])
    objective = float(sum(float(v) for v in weighted)+sum(float(v) for v in barriers)+sum(float(v) for v in anchors)+penalty)
    if not np.isfinite(objective):
        raise ValueError('Finite inputs produced a nonfinite objective')
    detail = dict(photometric_guard=penalty, degraded_mean_gain=float(mean_gain[1:].mean()),
        degraded_actual_gain=float(actual_gain[1:].mean()), degraded_excess=float(excess[1:].mean()),
        **{key: value.tolist() for key, value in photo.items()},
        per_image_guard_stronger_than_aggregate_gate=True, clear_profile_in_guard=False)
    return dict(objective=objective,
        normalized_weighted_RGB_SSIM_recognizer_structure=weighted.tolist(),
        regression_barriers_RGB_SSIM_recognizer=barriers.tolist(),
        clear_blur_RGB_anchors=anchors.tolist(), photometric_guard=penalty,
        per_image_photometric=detail,
        regression_residuals={key: value.tolist() for key, value in residuals.items()},
        raw_RGB_MSE=terms['MSE'].tolist(), raw_SSIM_loss=terms['SSIM_loss'].tolist(),
        raw_recognizer_loss=terms['ArcFace_loss'].tolist(), raw_structure_loss=terms['landmark_structure'].tolist(),
        metric_definitions=dict(METRIC_DEFINITIONS), model_qualified=False)


def one_sided_objective(current, baseline, slopes, normalizers, anchor_normalizers, *,
                        photometric, photometric_slopes, mean_control_derivative_kind='scalar_autograd',
                        reconstruction_weights=RECONSTRUCTION_WEIGHTS):
    """Right slope of a declared affine-residual model, not of literal execution.

    slopes contains the same four five-profile keys as current.
    photometric_slopes contains actual_MSE and mean_only_MSE.  Frozen old_MSE
    has zero slope.  Scalar-autograd slopes through the clipped float32 control
    are not certified right slopes, so the full prediction is then unsupported.
    """
    finite = finite_objective(current, baseline, normalizers, anchor_normalizers,
        photometric=photometric, reconstruction_weights=reconstruction_weights)
    derivatives = _terms(slopes, prefix='slopes')
    photo = _photo(photometric)
    try:
        actual_slope = _array(photometric_slopes['actual_MSE'], (5,), 'photometric_slopes.actual_MSE')
        mean_slope = _array(photometric_slopes['mean_only_MSE'], (5,), 'photometric_slopes.mean_only_MSE')
    except (KeyError, TypeError) as error:
        raise ValueError('Provide explicit photometric slopes') from error
    ns, an, weights = _parameters(normalizers, anchor_normalizers, reconstruction_weights)
    weighted = weights*np.asarray([derivatives['MSE'].mean(), derivatives['SSIM_loss'].mean(),
        derivatives['ArcFace_loss'].mean(), derivatives['landmark_structure'][1:].mean()])/ns
    barriers = np.asarray([5.*np.asarray(relu_right_derivative(
        np.asarray(finite['regression_residuals'][key]), derivatives[key])).mean()/ns[index]
        for index, key in enumerate(REGRESSION_OFFSETS)])
    anchors = np.asarray([derivatives['MSE'][0]/an[0], derivatives['MSE'][1]/an[1]])
    partial = float(weighted.sum()+barriers.sum()+anchors.sum())
    supported = mean_control_derivative_kind in ['analytic_scalar_fixture', 'idealized_pixel_clip']
    nested = None
    penalty = None
    if supported:
        mean_residual = photo['old_MSE']-photo['mean_only_MSE']
        actual_residual = photo['old_MSE']-photo['actual_MSE']
        mg, ag = np.maximum(mean_residual, 0.), np.maximum(actual_residual, 0.)
        d_mg = relu_right_derivative(mean_residual, -mean_slope)
        d_ag = relu_right_derivative(actual_residual, -actual_slope)
        outer_residual, outer_slope = mg-.2*ag, d_mg-.2*d_ag
        d_excess = relu_right_derivative(outer_residual, outer_slope)
        penalty = float(10.*d_excess[1:].mean()/ns[0])
        nested = dict(mean_gain_residual=mean_residual.tolist(), actual_gain_residual=actual_residual.tolist(),
            mean_gain_right_slopes=d_mg.tolist(), actual_gain_right_slopes=d_ag.tolist(),
            outer_residual=outer_residual.tolist(), outer_right_slopes=outer_slope.tolist(),
            excess_right_slopes=d_excess.tolist(),
            mean_gain_events=affine_hinge_events(mean_residual, -mean_slope),
            actual_gain_events=affine_hinge_events(actual_residual, -actual_slope),
            outer_events_for_initial_inner_branches=affine_hinge_events(outer_residual, outer_slope),
            outer_event_scope='Outer affine crossing distances apply only while the initial inner hinge branches remain valid; they are not a global finite clipping/network prediction.')
    if not np.isfinite(partial) or (penalty is not None and not np.isfinite(penalty)):
        raise ValueError('Nonfinite affine-model derivative')
    return dict(objective_directional=None if penalty is None else partial+penalty,
        known_nonphotometric_directional=partial,
        normalized_weighted_reconstruction_directional=weighted.tolist(),
        regression_barriers_directional=barriers.tolist(), clear_blur_anchors_directional=anchors.tolist(),
        photometric_guard_directional=penalty, photometric_directional_supported=supported,
        mean_control_derivative_kind=mean_control_derivative_kind,
        unsupported_reason=None if supported else 'Clipped mean-control and float32 cast require an explicit idealized pixel-clip direction or analytic scalar fixture; scalar autograd alone is not a certified right derivative.',
        regression_activation_events={key: affine_hinge_events(np.asarray(value), derivatives[key])
                                      for key, value in finite['regression_residuals'].items()},
        nested_photometric_model=nested,
        scope='Affine residual model using supplied slopes and actual offsets; finite network execution remains authoritative.',
        selected_neural_branch_must_be_validated=True, literal_float32_program_derivative_certified=False,
        PNG_directional_derivative_certified=False, metric_definitions=dict(METRIC_DEFINITIONS), model_qualified=False)


def mean_control_right_mse_at_initial(baseline, target, observed, channel_mean_slopes):
    """Idealized real-valued clip direction at the unchanged initial baseline.

    Inputs are prepared float32 NCHW (5,3,256,256), binary N1HW observed mask,
    and selected decoder-branch observed channel-mean slopes (5,3).  At zero,
    outward negative shifts are blocked; at one, outward positive shifts are
    blocked.  This does not differentiate the literal float32 cast/program or
    certify a network branch.  A second frozen-float32-residual proxy exposes
    the difference from interpreting the prepared inputs as real numbers.
    """
    baseline, target, observed = np.asarray(baseline), np.asarray(target), np.asarray(observed)
    if baseline.dtype != np.float32 or target.dtype != np.float32 or baseline.shape != (5, 3, 256, 256) or target.shape != baseline.shape:
        raise ValueError('Prepared baseline and target must be float32 NCHW five-profile arrays')
    if not np.isfinite(baseline).all() or not np.isfinite(target).all() or baseline.min() < 0 or baseline.max() > 1 or target.min() < 0 or target.max() > 1:
        raise ValueError('Prepared pixels must be finite and within [0,1]')
    if observed.shape != (5, 1, 256, 256) or observed.dtype.kind not in 'bifu' or not np.isfinite(observed).all() or not np.isin(observed, [0, 1]).all():
        raise ValueError('Observed mask must be binary N1HW')
    count = observed.astype(np.float64).sum((1, 2, 3))
    if (count <= 0).any(): raise ValueError('Each profile requires observed pixels')
    means = _array(channel_mean_slopes, (5, 3), 'channel_mean_slopes')
    direction = means[:, :, None, None]
    clip_direction = np.where(baseline == 0., np.maximum(direction, 0.),
                              np.where(baseline == 1., np.minimum(direction, 0.), direction))
    mask = observed.astype(np.float64)
    error = baseline.astype(np.float64)-target.astype(np.float64)
    frozen_error = (baseline-target).astype(np.float64)
    slope = (2*error*clip_direction*mask).sum((1, 2, 3))/(3*count)
    frozen_slope = (2*frozen_error*clip_direction*mask).sum((1, 2, 3))/(3*count)
    if not np.isfinite(slope).all() or not np.isfinite(frozen_slope).all():
        raise ValueError('Nonfinite idealized mean-control direction')
    return dict(mean_only_MSE_slopes=slope.tolist(),
        frozen_float32_residual_MSE_slopes=frozen_slope.tolist(),
        maximum_rounding_proxy_slope_difference=float(np.abs(slope-frozen_slope).max()),
        channel_mean_slopes=means.tolist(), mean_control_derivative_kind='idealized_pixel_clip',
        observed_zero_pixels_per_channel=((baseline == 0.) & observed.astype(bool)).sum((2, 3)).tolist(),
        observed_one_pixels_per_channel=((baseline == 1.) & observed.astype(bool)).sum((2, 3)).tolist(),
        observed_pixels_per_profile=count.astype(np.int64).tolist(),
        scope='Selected decoder-channel-mean branch with a real-valued piecewise clip model at the initial baseline.',
        literal_float32_program_derivative_certified=False, selected_neural_branch_must_be_validated=True,
        finite_retained_state_comparisons_required=True, model_qualified=False)
