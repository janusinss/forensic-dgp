"""Saved-array signed finite differences; no model, optimizer or neural imports."""
import math


def group_loss(groups, prediction):
    value = groups[prediction['source']+'/'+prediction['profile']]
    metric = prediction['metric']
    if metric == 'MSE':
        return value['MSE']
    if metric == 'SSIM_loss':
        return 1-value['SSIM']
    if metric == 'ArcFace_loss':
        return 1-value['ArcFace_observed_fixed']
    assert metric == 'feature_MSE'
    return value['landmark_high_frequency_MSE']


def signed_differences(proposals, baseline_groups, predictions):
    """Use raw losses only. PNG gates and pixel changes remain separate."""
    assert len(predictions) == 32 and len({r['key'] for r in predictions}) == 32
    assert set(proposals) == {'balanced_-0.001', 'balanced_-0.0001',
                              'balanced_0.0001', 'balanced_0.001'}
    rows = []
    for prediction in predictions:
        base = group_loss(baseline_groups['raw'], prediction)
        derivative = prediction['dot']
        assert all(math.isfinite(v) for v in [base, derivative]) and base > 0
        pairs = []
        for h, rate in [(0.001, '0.001'), (0.0001, '0.0001')]:
            plus = group_loss(proposals['balanced_'+rate]['groups']['raw'], prediction)
            minus = group_loss(proposals['balanced_-'+rate]['groups']['raw'], prediction)
            assert math.isfinite(plus) and math.isfinite(minus)
            central = (plus-minus)/(2*h)
            error = central-derivative
            pairs.append(dict(radius=h, positive_loss=plus, negative_loss=minus,
                central_directional_derivative=central,
                forward_directional_derivative=(plus-base)/h,
                backward_directional_derivative=(base-minus)/h,
                saved_directional_derivative=derivative,
                absolute_derivative_error=abs(error),
                relative_derivative_error=abs(error)/max(abs(derivative), 1e-30),
                finite_difference_signal=plus-minus,
                predicted_signed_pair_signal=2*h*derivative,
                midpoint_minus_baseline=(plus+minus)/2-base,
                second_difference_per_squared_radius=(plus-2*base+minus)/(h*h),
                positive_loss_change=plus-base, negative_loss_change=minus-base))
        rows.append(dict(key=prediction['key'], source=prediction['source'],
            profile=prediction['profile'], metric=prediction['metric'],
            raw_gradient_baseline_loss=prediction['loss'],
            independent_raw_baseline_loss=base,
            gradient_vs_audit_baseline_loss_difference=base-prediction['loss'],
            pairs=pairs, central_slope_change_on_tenfold_smaller_radius=
                pairs[1]['central_directional_derivative']-pairs[0]['central_directional_derivative']))
    return dict(complete=True, groups=rows, groups_checked=32,
        raw_loss_derivative_only=True, PNG_is_not_a_differentiable_gradient_test=True,
        central_difference_is_about_the_pinned_initial_state=True,
        numerical_noise_rounding_and_metric_definition_differences_require_review=True,
        no_arbitrary_derivative_threshold_or_model_selection=True,
        negative_steps_diagnostic_only=True, optimizer_updates=0, model_qualified=False,
        more_epochs_qualified=False, goal_complete=False)
