"""One-sided replay-referenced projection; no optimizer or training policy."""
import torch


def _validate(a, b):
    if a.ndim != 1 or a.numel() == 0 or a.shape != b.shape:
        raise ValueError('Nonempty matching flat vectors required')
    if a.device != b.device or a.dtype != b.dtype or not a.is_floating_point():
        raise ValueError('Matching floating dtype/device required')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite gradient/update')


def project_real_against_replay(real, replay):
    """Return projected-real + replay. Replay itself is not projected.

    Accumulate inner products in float64. A strictly zero replay vector leaves
    the real gradient unchanged. This is not symmetric PCGrad and does not
    constrain an AdamW step after momentum, preconditioning and weight decay.
    """
    _validate(real, replay)
    a, b = real.double(), replay.double()
    dot = torch.dot(a, b)
    norm2 = torch.dot(b, b)
    applied = bool(dot < 0 and norm2 > 0)
    projected = a - dot / norm2 * b if applied else a
    combined = (projected + b).to(real.dtype)
    if not torch.isfinite(combined).all():
        raise ValueError('Projection overflow')
    return combined, {
        'applied': applied,
        'real_replay_dot_before': float(dot),
        'real_replay_dot_after': float(torch.dot(projected,b)),
        'combined_replay_dot': float(torch.dot(combined.double(),b)),
        'real_norm': float(a.norm()), 'replay_norm': float(b.norm()),
        'removed_real_norm': float((a-projected).norm()),
    }


def step_alignment(before, after, replay):
    """First-order replay loss change g_replay dot (theta_after-theta_before).

    Positive means a local ascent direction. This is a diagnostic, not a
    finite-step loss or retention guarantee. Compute before/after subtraction
    in float64 to faithfully measure the stored parameter change.
    """
    _validate(before, after)
    _validate(before, replay)
    delta = after.double() - before.double()
    reference = replay.double()
    dot = torch.dot(delta,reference)
    denom = delta.norm()*reference.norm()
    return {'replay_first_order_loss_change':float(dot),
            'replay_descent_cosine':float(-dot/denom) if denom > 0 else None,
            'parameter_step_norm':float(delta.norm())}
