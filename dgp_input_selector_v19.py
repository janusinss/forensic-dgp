"""Training-calibrated input-only selector; not an information/face detector."""
import numpy as np

ABSOLUTE_DETAIL_THRESHOLD = 5e-5
RELATIVE_DETAIL_THRESHOLD = .002
DESIGN = {'format': 'dgp-input-detail-selector-v19',
    'absolute_laplacian_MSE_threshold': ABSOLUTE_DETAIL_THRESHOLD,
    'relative_laplacian_variance_threshold': RELATIVE_DETAIL_THRESHOLD,
    'retain_condition': 'absolute >= threshold AND relative >= threshold',
    'luminance_coefficients': [.299, .587, .114],
    'arithmetic': 'float64; four-neighbour Laplacian on fully supported inner pixels',
    'flat_input_policy': 'exact constant observed luminance retains DGP; external information review still required',
    'fallback': 'retained_dgp_v2', 'correction': 'fixed_terminal_structure_v18_update600',
    'calibration': 'input features on existing fifty synthetic training cases only; no target/output metric search',
    'qualification': 'None. This selector does not detect a face, pose, covering or usable identity structure.'}


def select_restoration(camera, support):
    """No labels, identities, targets, model predictions or filenames accepted."""
    if camera.shape != (256, 256, 3) or camera.dtype != np.uint8:
        raise ValueError('Require RGB uint8 256x256 camera canvas')
    if support.shape != (256, 256) or support.dtype != np.bool_:
        raise ValueError('Require boolean support from input canvas preparation')
    inner = (support[1:-1, 1:-1] & support[:-2, 1:-1] & support[2:, 1:-1]
             & support[1:-1, :-2] & support[1:-1, 2:])
    if not support.any() or not inner.any():
        raise ValueError('No usable input canvas support')
    rgb = camera.astype(np.float64) / 255
    y = .299 * rgb[..., 0] + .587 * rgb[..., 1] + .114 * rgb[..., 2]
    lap = (4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1]
           - y[1:-1, :-2] - y[1:-1, 2:])
    variance = float(np.var(y[support], dtype=np.float64))
    flat_input = bool(np.ptp(y[support]) == 0)
    energy = float(np.mean(np.square(lap[inner]), dtype=np.float64))
    relative = energy / max(variance, 1e-12)
    high_detail = energy >= ABSOLUTE_DETAIL_THRESHOLD and relative >= RELATIVE_DETAIL_THRESHOLD
    # Flat input must retain the base. Insufficient-face rejection is a separate
    # required gate; a selected branch is never permission to invent a face.
    branch = 'retained_dgp_v2' if high_detail or flat_input else 'structure_v18_update600'
    return {'branch': branch, 'laplacian_MSE': energy, 'luminance_variance': variance,
            'laplacian_over_variance': relative, 'supported_pixels': int(support.sum()),
            'inner_pixels': int(inner.sum()), 'high_detail': bool(high_detail),
            'flat_input': flat_input}
