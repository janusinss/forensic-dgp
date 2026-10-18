"""Deterministic RGB256 corruptions; no image-dependent choices or models."""
import hashlib
import json

import numpy as np
from scipy.ndimage import gaussian_filter


LOCAL_RECT = (64, 64, 192, 192)
PROFILES = (
    {'id': 'clear', 'operator': 'identity', 'parameters': {}, 'strong_control': False},
    {'id': 'gaussian_2p4', 'operator': 'gaussian', 'parameters': {'sigma_x': 2.4, 'sigma_y': 2.4}, 'strong_control': False},
    {'id': 'gaussian_6p0', 'operator': 'gaussian', 'parameters': {'sigma_x': 6.0, 'sigma_y': 6.0}, 'strong_control': True},
    {'id': 'anisotropic_h5_v0p7', 'operator': 'gaussian', 'parameters': {'sigma_x': 5.0, 'sigma_y': .7}, 'strong_control': False},
    {'id': 'anisotropic_h0p7_v5', 'operator': 'gaussian', 'parameters': {'sigma_x': .7, 'sigma_y': 5.0}, 'strong_control': False},
    {'id': 'repeated_rows_4_blend0p75', 'operator': 'repeated_rows', 'parameters': {'block_rows': 4, 'repeat_weight': .75}, 'strong_control': True},
    {'id': 'global_mosaic_8', 'operator': 'mosaic', 'parameters': {'cell': 8}, 'strong_control': False},
    {'id': 'global_mosaic_16', 'operator': 'mosaic', 'parameters': {'cell': 16}, 'strong_control': True},
    {'id': 'local_gaussian_3p0', 'operator': 'local_gaussian', 'parameters': {'sigma_x': 3.0, 'sigma_y': 3.0}, 'strong_control': False},
    {'id': 'local_gaussian_10p0', 'operator': 'local_gaussian', 'parameters': {'sigma_x': 10.0, 'sigma_y': 10.0}, 'strong_control': True},
    {'id': 'local_mosaic_8', 'operator': 'local_mosaic', 'parameters': {'cell': 8}, 'strong_control': False},
    {'id': 'local_mosaic_16', 'operator': 'local_mosaic', 'parameters': {'cell': 16}, 'strong_control': True},
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def profiles_document():
    # Return a fresh JSON tree, so a caller cannot mutate the fixed module list.
    return json.loads(json.dumps({'format': 'codeformer-fixed-filter-profiles-v1',
        'root_prospective_approval': '12 fixed profiles approved before generation; no trained model selected',
        'input_size': [256, 256], 'local_rectangle_xyxy_half_open': list(LOCAL_RECT),
        'profiles': PROFILES, 'input_geometry': 'direct RGB1024 to Pillow BILINEAR256',
        'target512_geometry': 'direct RGB1024 to Pillow LANCZOS512',
        'target256_geometry': 'retained exact target256; verify direct RGB1024 to Pillow LANCZOS256',
        'Gaussian': {'arithmetic': 'float64', 'implementation': 'scipy.ndimage.gaussian_filter',
            'sigma_order': '[sigma_y,sigma_x,0]', 'edge_mode': 'reflect', 'truncate': 3.0},
        'mosaic': 'nonoverlapping cell means in RGB255 float64; round half up; repeat each cell without interpolation',
        'repeated_rows': 'each block copies its first row; blend original*(1-weight)+repeated*weight; round half up',
        'local_edges': 'hard replacement within fixed rectangle; no feather; local Gaussian filtered on complete canvas',
        'rounding': 'floor(clip(float64RGB255,0,255)+0.5) to uint8',
        'seeds': 'recorded per case; no stochastic operation or RNG call in these fixed profiles',
        'geometry_alignment_warp': False, 'brightness_contrast_or_colour_operation': False,
        'corruption_footprint_is_not_completion_or_removal_mask': True,
        'training_eligibility': 'strong controls excluded initially; all other cases PENDING input-only review',
        'model_or_output_based_parameter_selection': False}, allow_nan=False))


def check_profile(profile):
    approved = {row['id']: row for row in PROFILES}
    require(isinstance(profile, dict) and profile.get('id') in approved and profile == approved[profile['id']],
        'Exactly a prospectively approved fixed profile required')


def rgb256(image):
    require(isinstance(image, np.ndarray) and image.shape == (256, 256, 3) and image.dtype == np.uint8,
        'Exact RGB256 uint8 input required')


def rounded(image):
    require(np.isfinite(image).all(), 'Finite filter arithmetic required')
    return np.floor(np.clip(image, 0, 255) + .5).astype(np.uint8)


def gaussian(image, sigma_x, sigma_y):
    return rounded(gaussian_filter(image.astype(np.float64), sigma=(sigma_y, sigma_x, 0),
        mode='reflect', truncate=3.0))


def mosaic(image, cell):
    height, width = image.shape[:2]
    require(type(cell) is int and cell > 0 and height % cell == 0 and width % cell == 0,
        'Mosaic cell must divide both image dimensions')
    means = image.astype(np.float64).reshape(height // cell, cell, width // cell, cell, 3).mean(axis=(1, 3))
    return rounded(means).repeat(cell, axis=0).repeat(cell, axis=1)


def apply_profile(image, profile):
    rgb256(image); check_profile(profile)
    operation, p = profile['operator'], profile['parameters']
    footprint = np.zeros((256, 256), dtype=np.uint8)
    original = image.copy()
    if operation == 'identity':
        result = original.copy()
    elif operation in ('gaussian', 'local_gaussian'):
        result = gaussian(original, p['sigma_x'], p['sigma_y'])
    elif operation == 'repeated_rows':
        rows = np.arange(256) // p['block_rows'] * p['block_rows']
        weight = p['repeat_weight']
        result = rounded(original.astype(np.float64) * (1 - weight) + original[rows].astype(np.float64) * weight)
    elif operation in ('mosaic', 'local_mosaic'):
        if operation == 'mosaic':
            result = mosaic(original, p['cell'])
        else:
            x0, y0, x1, y1 = LOCAL_RECT
            result = original.copy()
            result[y0:y1, x0:x1] = mosaic(original[y0:y1, x0:x1], p['cell'])
    else:
        raise ValueError('Unsupported fixed operator')
    if operation.startswith('local_'):
        x0, y0, x1, y1 = LOCAL_RECT
        footprint[y0:y1, x0:x1] = 255
        result[footprint == 0] = original[footprint == 0]
    elif operation != 'identity':
        footprint[:] = 255
    require(np.array_equal(image, original), 'Filter mutated caller input')
    return np.ascontiguousarray(result), footprint


def seed_record(reference_id, profile_id):
    payload = ('codeformer-filter-pairs-v1/' + reference_id + '/' + profile_id).encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')
