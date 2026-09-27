"""Explicit, reproducible mask policies; boundary correction is experimental."""
import torch
from torch.nn import functional as F


def completion_mask(probability,policy='baseline'):
    if probability.ndim!=4 or probability.shape[1]!=1 or not probability.numel():
        raise ValueError('Expected nonempty N1HW probabilities')
    if not torch.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
        raise ValueError('Expected finite probabilities in [0,1]')
    seed=probability>=.5
    if policy=='baseline':return seed.to(probability.dtype)
    if policy!='boundary035':raise ValueError('Unknown mask policy')
    # Single non-iterative step; never grow disconnected low-confidence regions.
    near=F.max_pool2d(seed.float(),3,1,1)>0
    return (seed|(near&(probability>=.35))).to(probability.dtype)
