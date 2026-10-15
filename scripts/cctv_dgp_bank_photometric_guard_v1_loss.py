"""TRAIN-only brightness-gain guard; no model, inference filter or gate change.

The existing return gate permits at most 20% brightness-only MSE improvement.
This conservative per-image hinge aligns a training signal with that gate,
without dividing by a small/negative total improvement. Profiles must be the
frozen clear/blur/lowlight/motion/compound order; only degraded profiles enter.
Differentiation and optimizers remain in the guarded manual VM worker.
"""


def brightness_guard(prediction, baseline, target, observed, normalizer,
                     coefficient=10., fraction_maximum=.2):
    import torch
    assert prediction.shape == baseline.shape == target.shape == (5, 3, 256, 256)
    assert observed.shape == (5, 1, 256, 256)
    assert normalizer > 0 and coefficient == 10. and fraction_maximum == .2
    old = baseline.detach()
    mask = observed.detach().double()
    count = mask.sum(dim=(2, 3), keepdim=True)
    assert bool((count > 0).all())
    # Match the retained raw mean-only control: subtract in float32, then
    # compute the observed channel means in float64, clip, cast to float32.
    shift = ((prediction-old).double()*mask).sum(dim=(2, 3), keepdim=True)/count
    control = (old.double()+shift).clamp(0, 1).float()
    control = torch.where(observed.bool(), control, old)

    def mse(value):
        error = (value-target).square().double()
        return (error*mask).sum(dim=(1, 2, 3))/(3*count.flatten())

    old_mse = mse(old)
    actual_mse = mse(prediction)
    mean_mse = mse(control)
    mean_gain = (old_mse-mean_mse).clamp_min(0)
    actual_gain = (old_mse-actual_mse).clamp_min(0)
    excess = (mean_gain-fraction_maximum*actual_gain).clamp_min(0)
    penalty = coefficient*excess[1:].mean()/normalizer
    assert bool(torch.isfinite(penalty))
    return penalty, {
        'photometric_guard': float(penalty.detach()),
        'degraded_mean_gain': float(mean_gain[1:].mean().detach()),
        'degraded_actual_gain': float(actual_gain[1:].mean().detach()),
        'degraded_excess': float(excess[1:].mean().detach()),
        'old_MSE': old_mse.detach().cpu().tolist(),
        'actual_MSE': actual_mse.detach().cpu().tolist(),
        'mean_only_MSE': mean_mse.detach().cpu().tolist(),
        'per_image_guard_stronger_than_aggregate_gate': True,
        'clear_profile_in_guard': False,
    }
