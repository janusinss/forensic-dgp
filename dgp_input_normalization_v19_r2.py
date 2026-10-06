"""Explicit historical float encodings; same RGB bytes, no model or fitting."""
import numpy as np
import torch

POLICY = {
    'retained_DGP': 'NumPy float32 division by255 before device transfer; exact V15 comparison encoding',
    'spatial_pipeline': 'PyTorch float32 scalar division by255 after device transfer; unchanged V18 encoding',
    'CPU_cached_spatial_replay': 'NumPy float32 multiplication by float32 reciprocal255, matching measured L4 scalar encoding',
    'source': 'Same prepared RGB256 bytes/canvas support for all comparison arms',
    'display_processing': 'none', 'quantization': 'unchanged floor(raw*255), input padding retained',
    'reason': 'Audited three-case L4 diagnostic confirms both historical encodings require preservation',
}


def checked(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (256, 256, 3):
        raise ValueError('Require exact prepared uint8 RGB256')
    return rgb


def retained_input(rgb, device='cpu'):
    """Match V15's input tensor exactly; division happens before transfer."""
    array = checked(rgb).astype(np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1)[None].to(device)


def spatial_input(rgb, device='cpu'):
    """Preserve V18's model input; CUDA scalar division is intentionally retained."""
    return torch.from_numpy(checked(rgb).copy()).permute(2, 0, 1).float()[None].to(device) / 255


def spatial_cpu_replay_input(rgb):
    """Use the measured GPU input floats for CPU head arithmetic, no forwards."""
    array = checked(rgb).astype(np.float32) * np.float32(1 / 255)
    return torch.from_numpy(array).permute(2, 0, 1)[None]
