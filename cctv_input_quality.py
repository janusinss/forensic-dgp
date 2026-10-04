"""Observed-support quality signals for the future DGP256 route, not a face gate.

These input-only heuristics suggest restoration. They do not establish readable
facial structure, pose, model usefulness or permission to generate a hidden face.
The existing application policy and immutable comparison sources remain intact.
"""
import numpy as np

from dgp_face_restoration import prepare_crop
from face_workflow import quality_signals

POLICY = 'dgp256-observed-quality-v1'


def observed_quality_signals(canvas, observed, removal):
    """Score only visible captured support after its filter-neighborhood erosion."""
    if (not isinstance(canvas, np.ndarray) or canvas.dtype != np.uint8
            or canvas.shape != (256, 256, 3)):
        raise ValueError('Use a prepared 256 x256 uint8 RGB canvas')
    for name, mask in [('observed', observed), ('removal', removal)]:
        if not isinstance(mask, np.ndarray) or mask.shape != (256, 256) or not np.isin(mask, (0, 1)).all():
            raise ValueError('Use a matching binary ' + name + ' mask')
    observed, removal = observed.astype(bool), removal.astype(bool)
    if np.any(removal & ~observed):
        raise ValueError('Removal cannot extend outside observed support')
    ignored = (~observed | removal).astype(np.uint8)
    signals = quality_signals(canvas, ignored)
    return {**signals, 'policy': POLICY,
            'support_policy': 'Captured pixels minus reviewed removal and filter border; no square padding',
            'observed_pixels': int(observed.sum()), 'removed_pixels': int(removal.sum()),
            'structure_qualification_established': False, 'usable_face_confirmed': False,
            'scope': 'Developmental restoration suggestion only; facial structure and pose require a separate qualification policy'}


def quality_for_crop(rgb, removal_mask=None):
    """Use the frozen DGP canvas preparation, including bilinear enlargement."""
    canvas, observed, removal, geometry = prepare_crop(rgb, removal_mask)
    return {**observed_quality_signals(canvas, observed, removal), 'geometry': geometry}
