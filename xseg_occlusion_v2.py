"""Numerically bounded XSeg output; V1 spatial conversion remains unchanged."""
import numpy as np

from xseg_occlusion import XSegVisibleFace

POLICY = "xseg-frontal-ellipse-proposal-v2"


def clip_visible_probability(probability):
    if probability.shape != (256, 256) or not np.isfinite(probability).all():
        raise FloatingPointError("Expected finite 256-square probability")
    # ORT produced one value 1 + 1 float32 ULP. Reject substantive range errors.
    if probability.min() < -1e-6 or probability.max() > 1 + 1e-6:
        raise FloatingPointError("Substantive XSeg probability range error")
    return np.clip(probability, 0, 1).astype(np.float32)


class XSegVisibleFaceClipped(XSegVisibleFace):
    def __call__(self, rgb):
        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
            raise ValueError("Research adapter requires uint8 RGB 256-square crops")
        sample = np.ascontiguousarray(rgb[..., ::-1][None], dtype=np.float32) / 255
        self.forward_count += 1
        result = self.session.run(None, {"input": sample})[0]
        if result.shape != (1, 256, 256, 1):
            raise ValueError("Unexpected XSeg result shape")
        return clip_visible_probability(result[0, ..., 0])
