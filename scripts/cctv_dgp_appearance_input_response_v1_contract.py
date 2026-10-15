"""Exact saved float32 image displacement projected with float64 arithmetic."""
import numpy as np


def input_projection(gradient,before,after):
    for a in [gradient,before,after]:
        assert isinstance(a,np.ndarray) and a.dtype==np.float32 and a.shape==(256,256,3)
        assert np.isfinite(a).all()
    assert 0<=before.min()<=before.max()<=1 and 0<=after.min()<=after.max()<=1
    # Subtract after promoting; a float32 subtraction discards part of the actual displacement.
    return float(np.sum(gradient.astype(np.float64)*(after.astype(np.float64)-before.astype(np.float64)),dtype=np.float64))
