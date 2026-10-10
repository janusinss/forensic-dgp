"""Display-only blend confined to the previously documented two-pixel margin."""
import numpy as np
from scipy.ndimage import distance_transform_edt


def feather(source, estimate, core, final):
    if source.dtype != np.uint8 or estimate.dtype != np.uint8 or source.shape != estimate.shape or source.shape != (256, 256, 3):
        raise ValueError('Matching RGB256 byte arrays required')
    if core.dtype != bool or final.dtype != bool or core.shape != (256, 256) or final.shape != core.shape or (core & ~final).any():
        raise ValueError('Nested binary256 core/final masks required')
    weights = np.zeros((256, 256), np.float32)
    if not core.any():
        if final.any(): raise ValueError('An empty core must be an empty control')
        return source.copy(), weights
    distance = distance_transform_edt(~core)
    if (distance[final & ~core] > 2.).any():
        raise ValueError('Existing final support exceeds the documented two-pixel Euclidean margin')
    weights[final] = np.float32(1.) - distance[final].astype(np.float32) / np.float32(3.)
    values = weights[..., None] * estimate.astype(np.float32) + (np.float32(1.) - weights[..., None]) * source.astype(np.float32)
    output = np.floor(values + np.float32(.5)).astype(np.uint8)
    # Preserve source and the already generated core byte-for-byte.
    output[~final] = source[~final]; output[core] = estimate[core]
    return output, weights
