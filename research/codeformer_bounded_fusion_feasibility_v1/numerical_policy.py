"""Prospective CPU-only internal feature replay bound; not a quality gate."""
import math


def canary_replay(base, other, gate):
    """Return finite float32 replay and a predeclared elementwise audit bound.

    This computes no image restoration and calls no neural framework. Different
    CPU/CUDA reductions are not declared bit exact. A discrepancy beyond this
    recorded rounding allowance retains an audit failure; no fallback or retry.
    """
    import numpy as np
    if (not isinstance(base, np.ndarray) or not isinstance(other, np.ndarray) or
            base.shape != other.shape or base.ndim != 4 or base.shape[0] != 1 or
            base.shape[1:] not in ((256, 32, 32), (256, 64, 64)) or
            base.dtype != np.float32 or other.dtype != np.float32 or
            not np.isfinite(base).all() or not np.isfinite(other).all() or
            type(gate) not in (float, int) or not math.isfinite(gate) or not .25 <= gate <= 1):
        raise ValueError('Exact saved FP32 canary geometry, finite values and declared gate required')
    scale = np.float32(.10) * np.sqrt(np.square(base).mean(axis=(-2, -1),
        keepdims=True, dtype=np.float32) + np.float32(1e-6))
    used = np.tanh((other - base) / scale) * scale
    used = used * np.float32(gate)
    n = base.shape[-1] * base.shape[-2]
    eps = float(np.finfo(np.float32).eps)
    unit = eps / 2
    gamma = n * unit / (1 - n * unit)
    # The positive-square reduction allowance is scaled by D. The second term
    # bounds low-level pointwise arithmetic at the source feature magnitudes.
    # This is an explicitly conservative audit policy, not empirical tuning.
    tolerance = (8 * gamma * scale.astype(np.float64) + 64 * eps *
        (1 + np.abs(base.astype(np.float64)) + np.abs(other.astype(np.float64)) +
         np.abs(used.astype(np.float64))))
    if not (np.isfinite(scale).all() and np.isfinite(used).all() and np.isfinite(tolerance).all()):
        raise ValueError('Canary arithmetic overflow or nonfinite bound')
    return {'scale': scale, 'used_delta': used, 'tolerance': tolerance,
        'n': n, 'float32_epsilon': eps, 'gamma_n': gamma,
        'policy': 'pointwise abs(returned_delta - CPU_replay) <= 8*gamma_n*D + 64*eps32*(1+abs(F0)+abs(FS)+abs(CPU_delta)); no threshold refitting'}
