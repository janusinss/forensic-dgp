"""Finite scalar response reporting; no network, solve or quality-gate change."""
import math
import sys


def response_receipt(before, plus, minus, magnitude, predicted_plus_delta, predicted_minus_delta, repeat_spread):
    values=[before,plus,minus,magnitude,predicted_plus_delta,predicted_minus_delta,repeat_spread]
    assert all(math.isfinite(float(v)) for v in values)
    assert magnitude>0 and repeat_spread>=0
    difference=plus-minus
    predicted=predicted_plus_delta-predicted_minus_delta
    noise=max(repeat_spread,8*sys.float_info.epsilon*max(1.,abs(before),abs(plus),abs(minus)))
    resolved=abs(difference)>10*noise and abs(predicted)>10*noise
    bound=max(.05*abs(predicted),10*noise)
    return dict(magnitude=magnitude,observed_central_slope=difference/(2*magnitude),
        actual_rounded_prediction_central_slope=predicted/(2*magnitude),
        observed_two_sided_difference=difference,predicted_two_sided_difference=predicted,
        even_response=(plus+minus)/2-before,even_linear_prediction=(predicted_plus_delta+predicted_minus_delta)/2,
        central_prediction_residual=difference-predicted,noise_floor=noise,resolved=resolved,
        central_comparison_fraction=.05,central_comparison_absolute_bound=bound,
        central_comparison_within_diagnostic_bound=abs(difference-predicted)<=bound if resolved else None,
        diagnostic_bound_is_not_a_scientific_preservation_or_qualification_gate=True,
        preservation_gate_relaxed=False,model_qualified=False)
