"""Transactional AdamW trials with fixed replay-loss ceilings.

The caller supplies gradients and a deterministic, training-only evaluator.
No backward pass or model training is started by importing this module.
"""
import copy
import math

import torch


def guarded_step(module, optimizer, evaluate_replay, ceilings, *,
                 factors=(1., .5, .25, .125), atol=1e-6):
    """Try scaled learning rates; reject atomically, including optimizer moments.

    Evaluate both covered and clear losses against fixed parent ceilings. The
    supplied evaluator must be deterministic and must not write external state.
    Gradients must already be finite and clipped. Accepted optimizer state is
    kept, but the nominal learning rate is restored for the next scheduled batch.
    """
    if set(ceilings) != {'covered', 'clear'} or any(
            not math.isfinite(v) or v < 0 for v in ceilings.values()):
        raise ValueError('Require finite nonnegative covered and clear ceilings')
    if not math.isfinite(atol) or atol < 0:
        raise ValueError('Invalid tolerance')
    if not factors or any(not math.isfinite(f) or not 0 < f <= 1 for f in factors):
        raise ValueError('Invalid trial factors')
    if any(a <= b for a, b in zip(factors, factors[1:])):
        raise ValueError('Trial factors must strictly decrease')
    params = list(module.parameters())
    owned = {id(p) for p in params}
    optimized = [p for g in optimizer.param_groups for p in g['params']]
    if any(id(p) not in owned for p in optimized):
        raise ValueError('Optimizer owns parameters outside supplied module')
    if not any(p.grad is not None for p in optimized) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in optimized):
        raise ValueError('Require finite supplied gradients')
    nominal = [g['lr'] for g in optimizer.param_groups]
    if any(not math.isfinite(lr) or lr <= 0 for lr in nominal):
        raise ValueError('Invalid nominal learning rate')
    state = copy.deepcopy(module.state_dict())
    opt_state = copy.deepcopy(optimizer.state_dict())
    grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
    modes = [(m, m.training) for m in module.modules()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None

    def restore_context():
        for p, grad in zip(params, grads):
            p.grad = None if grad is None else grad.clone()
        for m, training in modes:
            m.training = training
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    def rollback():
        module.load_state_dict(state)
        optimizer.load_state_dict(copy.deepcopy(opt_state))
        restore_context()

    attempts = []
    try:
        for factor in factors:
            rollback()
            for group, lr in zip(optimizer.param_groups, nominal):
                group['lr'] = lr * factor
            optimizer.step()
            module.eval()
            with torch.inference_mode():
                losses = {k: float(v) for k, v in evaluate_replay().items()}
            if set(losses) != set(ceilings):
                raise ValueError('Evaluator must return covered and clear losses')
            finite_state = all(torch.isfinite(v).all() for v in module.state_dict().values())
            accepted = bool(finite_state) and all(
                math.isfinite(v) and v >= 0 and v <= ceilings[k] + atol
                for k, v in losses.items())
            attempts.append({'factor': factor, 'losses': losses, 'accepted': accepted})
            if accepted:
                for group, lr in zip(optimizer.param_groups, nominal):
                    group['lr'] = lr
                restore_context()
                return {'accepted': True, 'factor': factor, 'attempts': attempts}
        rollback()
        return {'accepted': False, 'factor': None, 'attempts': attempts}
    except BaseException:
        rollback()
        raise
