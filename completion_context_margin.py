"""Experimental inference context suppression; reviewed output support stays fixed."""
import torch
from torch.nn import functional as F

from completion import validate_mask


def conditioning_mask(mask, radius_at256):
    if type(radius_at256) is not int or not 0 <= radius_at256 <= 6:
        raise ValueError("Use a context radius from zero to six pixels at 256 scale")
    if mask.ndim != 4 or mask.shape[1] != 1 or mask.dtype != torch.float32 or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Use finite float32 binary N1HW context support")
    radius = round(radius_at256 * mask.shape[-1] / 256)
    return F.max_pool2d(mask, 2 * radius + 1, 1, radius) if radius else mask.clone()


class ContextMarginCompletion(torch.nn.Module):
    """Suppress a small context ring, then discard generated pixels in that ring.

    The wrapped pretrained completion model and its weights stay unchanged.
    This experiment is not connected to the application or detector training.
    """
    def __init__(self, backend, radius_at256):
        super().__init__()
        if type(radius_at256) is not int or not 0 <= radius_at256 <= 6:
            raise ValueError("Use a context radius from zero to six pixels at 256 scale")
        self.backend = backend
        self.radius_at256 = radius_at256
        self.last_context_mask = None

    @torch.inference_mode()
    def forward(self, image, mask):
        validate_mask(image, mask)
        context = conditioning_mask(mask, self.radius_at256)
        if (context.mean((1, 2, 3)) >= .85).any():
            raise ValueError("Context suppression leaves too little visible face")
        self.last_context_mask = context.detach().cpu().clone()
        if not mask.any():
            return image.clone()
        generated = self.backend(image, context)
        if generated.shape != image.shape or not torch.isfinite(generated).all():
            raise FloatingPointError("Invalid contextual completion output")
        return torch.where(mask.bool(), generated, image)
