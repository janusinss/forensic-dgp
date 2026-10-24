"""Pure optional Protect/Apply preview; no model, HTTP or training entrypoint.

Protect always wins. Exact-pixel guarantees concern preprocessing only. Blur
reads neighboring original context but never writes protected/outside-Apply RGB.
"""
import base64
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps, __version__ as PILLOW_VERSION
import scipy
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 10 * 1024 ** 2
MAX_MASK_BYTES = 3 * 1024 ** 2
MAX_SIDE = 1600
MAX_PIXELS = 2_560_000
POLICY = 'reviewed-native-Protect-Apply-brush-preview-v1'
PREPARE_SOURCE = 'implementation/runtime.py'
# Runtime geometry source is bound into each preview identity below.


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def png(array):
    stream = io.BytesIO(); Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


def url(raw): return 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def decode_original(contents):
    need(type(contents) is bytes and 0 < len(contents) <= MAX_UPLOAD_BYTES, 'Upload a PNG/JPEG face crop up to10MiB.')
    try:
        with Image.open(io.BytesIO(contents)) as image:
            need(image.format in ('PNG', 'JPEG') and getattr(image, 'n_frames', 1) == 1,
                 'One nonanimated PNG or JPEG required.')
            need(0 < image.width <= MAX_SIDE and 0 < image.height <= MAX_SIDE
                 and image.width * image.height <= MAX_PIXELS, 'Native crop exceeds1600side/2,560,000pixels.')
            return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError('Cannot decode the original crop.') from error


def decode_mask(raw, size, label):
    need(type(raw) is bytes and 0 < len(raw) <= MAX_MASK_BYTES, label + ' mask must be a PNG up to3MiB.')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            need(image.format == 'PNG' and image.mode == 'L' and getattr(image, 'n_frames', 1) == 1,
                 label + ' mask must be single-frame8-bit L PNG.')
            need(image.size == size, label + ' mask must exactly match decoded oriented native dimensions.')
            # Masks are in decoded image coordinates; EXIF tags may not rotate a mask silently.
            need(image.getexif().get(274, 1) == 1, label + ' mask may not contain an orientation transform.')
            array = np.asarray(image).copy()
        need(np.all((array == 0) | (array == 255)), label + ' mask must contain only0/255 pixels.')
        return array == 255
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError('Cannot decode ' + label + ' mask.') from error


def parameter(value, lower, upper, label):
    need(type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper,
         label + ' must be a finite number in[' + str(lower) + ',' + str(upper) + '].')
    return float(value)


def mask_weights(protect, apply, guard, feather):
    """Distance feather is external to protection and entirely inside Apply.

    A zero padded Apply border specifies native frame-edge behavior. Empty
    Protect has infinite protection distance, rather than SciPy's implicit edge.
    """
    if protect.any():
        protection_distance = distance_transform_edt(~protect)
        guarded = protection_distance <= guard
        outside_distance = distance_transform_edt(~guarded) if guarded.any() else np.full(protect.shape, np.inf)
    else:
        guarded = np.zeros(protect.shape, bool)
        outside_distance = np.full(protect.shape, np.inf)
    inside_apply_distance = distance_transform_edt(np.pad(apply, 1, constant_values=False))[1:-1, 1:-1]
    effective = apply & ~guarded
    t = np.clip(np.minimum(outside_distance, inside_apply_distance) / feather, 0., 1.)
    alpha = (t * t * (3 - 2 * t)).astype(np.float32)
    alpha[~effective] = 0
    need(not alpha[guarded | ~apply].any(), 'Protected/outside-Apply alpha must be exactlyzero.')
    return guarded, effective, alpha


def conservative_256(native_unchanged, observed):
    """Conservative same-bilinear-footprint interior, never a boundary claim.

    Positive float-mask influence excludes any256 pixel that reads a potentially
    changed native pixel. Original center-padding is always unchanged. Exact RGB
    equality is checked independently after actual RGB preparation.
    """
    height, width = native_unchanged.shape
    side = max(height, width); left, top = (side - width) // 2, (side - height) // 2
    influence = Image.new('F', (side, side), 0.)
    influence.paste(Image.fromarray((~native_unchanged).astype(np.float32)), (left, top))
    resized = np.asarray(influence.resize((256, 256), Image.Resampling.BILINEAR))
    return (resized == 0) & observed


def build_preview(contents, protect_png_bytes, apply_png_bytes, nominal_radius=6, guard=8, feather=16):
    """Return (JSON-safe response, fixed-name byte payloads), without model calls.

    All input arrays are decoded anew. Caller must rebuild and compare preview_id
    again before generation; the module stores no mutable preview/original cache.
    """
    radius = parameter(nominal_radius, 0, 12, 'Nominal radius at256')
    guard = parameter(guard, 0, 64, 'Native protection guard')
    feather = parameter(feather, 1, 128, 'Native transition distance')
    rgb = decode_original(contents); height, width = rgb.shape[:2]
    protect = decode_mask(protect_png_bytes, (width, height), 'Protect')
    apply = decode_mask(apply_png_bytes, (width, height), 'Apply')
    guarded, effective, alpha = mask_weights(protect, apply, guard, feather)
    native_radius = radius * max(width, height) / 256.
    # Filtering is computed from the original once. It can read protected context;
    # this is not a no-sampling or unknown-kernel-isolation guarantee.
    blurred = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=native_radius)))
    mixed = np.rint(rgb.astype(np.float64) * (1 - alpha[..., None].astype(np.float64))
                    + blurred.astype(np.float64) * alpha[..., None].astype(np.float64)).clip(0, 255).astype(np.uint8)
    unchanged = guarded | ~apply
    mixed[unchanged] = rgb[unchanged]
    need(np.array_equal(mixed[guarded], rgb[guarded]) and np.array_equal(mixed[~apply], rgb[~apply]),
         'Every protected and outside-Apply native RGB pixel must stay exact.')

    source = ROOT / PREPARE_SOURCE
    prepare_sha = sha(source.read_bytes())
    from implementation.runtime import prepare_crop
    original256, observed, _, original_geometry = prepare_crop(rgb)
    derived256, derived_observed, _, derived_geometry = prepare_crop(mixed)
    need(np.array_equal(observed, derived_observed), 'Prepared observed support changed.')
    interiors = {'protected': conservative_256(guarded, observed),
                 'outside_apply': conservative_256(~apply, observed),
                 'unchanged_union': conservative_256(unchanged, observed)}
    for interior in interiors.values():
        need(np.array_equal(original256[interior], derived256[interior]), 'Conservative prepared256 interior changed.')

    algorithm = {'policy': POLICY, 'nominal_Gaussian_radius_at256': radius, 'native_Gaussian_radius': native_radius,
                 'outward_Protect_guard_native_pixels': guard, 'external_transition_native_pixels': feather,
                 'transition': 'smoothstep(min(distanceOutsideGuard,distanceInsideApply)/feather); alpha0outsideApplyandinsideGuard',
                 'Apply_edge_boundary': 'one-zero-pixel pad before Euclidean distance transform',
                 'empty_Protect_distance': 'infinite; no implicit image-edge protection',
                 'quantization': 'float64 blend with float32 alpha; np.rint ties-to-even clipRGB8; exact native unchanged assignment',
                 'kernel': 'Pillow GaussianBlur approximation; original context may be sampled',
                 'Pillow': PILLOW_VERSION, 'NumPy': np.__version__, 'SciPy': scipy.__version__}
    sources = {'preview_module': {'path': Path(__file__).relative_to(ROOT).as_posix(), 'sha256': sha(Path(__file__).read_bytes())},
               'prepare_helper': {'path': PREPARE_SOURCE, 'sha256': prepare_sha}}
    identity = {'original_upload_sha256': sha(contents), 'original_RGB_sha256': sha(rgb.tobytes()),
                'native_size': [width, height], 'Protect_upload_sha256': sha(protect_png_bytes),
                'Apply_upload_sha256': sha(apply_png_bytes), 'Protect_binary_sha256': sha(protect.tobytes()),
                'Apply_binary_sha256': sha(apply.tobytes()), 'effective_Protect_sha256': sha(guarded.tobytes()),
                'effective_Apply_sha256': sha(effective.tobytes()), 'alpha_float32_sha256': sha(alpha.tobytes()),
                'algorithm': algorithm, 'sources': sources}
    preview_id = sha(json_bytes(identity))
    metadata = {'policy': POLICY, 'preview_id': preview_id, **identity,
                'protected_native_pixels': int(guarded.sum()), 'Apply_native_pixels': int(apply.sum()),
                'effective_Apply_native_pixels': int(effective.sum()), 'positive_alpha_pixels': int((alpha > 0).sum()),
                'changed_native_pixels': int(np.any(mixed != rgb, axis=2).sum()),
                'no_effect': bool(np.array_equal(mixed, rgb)), 'Protect_wins_overlap': True,
                'every_protected_native_RGB_exact': True, 'every_outside_Apply_native_RGB_exact': True,
                'conservative_256_exact_counts': {k: int(v.sum()) for k, v in interiors.items()},
                'prepared_boundary_pixels_not_claimed_exact': True,
                'original_geometry': original_geometry, 'derived_geometry': derived_geometry,
                'derived_native_RGB_sha256': sha(mixed.tobytes()), 'manual_assisted_masks': True,
                'automatic_detection_or_adaptive_matching': False, 'blur_matching_is_approximate': True,
                'filter_may_read_protected_context': True, 'protection_scope': 'preprocessing input only; generated appearance may change',
                'original_rebuilt_each_preview': True, 'server_preview_cache': False,
                'preprocessing_neural_optimizer_training_VM_calls': 0, 'model_qualified': False,
                'main_app_default_changed': False}
    mask = lambda value: png(value.astype(np.uint8) * 255)
    stream = io.BytesIO(); np.save(stream, alpha, allow_pickle=False)
    colors = rgb.copy(); colors[apply] = (55, 125, 220); colors[(alpha > 0) & (alpha < 1)] = (245, 180, 45)
    colors[guarded] = (45, 185, 90)
    overlay = np.rint(rgb.astype(np.float64) * .60 + colors.astype(np.float64) * .40).astype(np.uint8)
    payloads = {'uploaded-original.bin': contents, 'original-decoded-rgb.png': png(rgb),
                'Protect-native.png': mask(protect), 'Apply-native.png': mask(apply),
                'effective-Protect-native.png': mask(guarded), 'effective-Apply-native.png': mask(effective),
                'alpha-float32.npy': stream.getvalue(), 'derived-native.png': png(mixed),
                'prepared-original-256.png': png(original256), 'prepared-derived-256.png': png(derived256),
                'observed-support-256.png': mask(observed), 'protection-overlay.png': png(overlay),
                **{key + '-interior-256.png': mask(value) for key, value in interiors.items()}}
    metadata['payload_pins'] = {name: {'bytes': len(raw), 'sha256': sha(raw)} for name, raw in payloads.items()}
    payloads['processing.json'] = json_bytes(metadata)
    response = {'preview_id': preview_id, 'width': width, 'height': height, 'metadata': metadata,
                'original': url(payloads['original-decoded-rgb.png']), 'prepared_input': url(payloads['derived-native.png']),
                'protection_overlay': url(payloads['protection-overlay.png'])}
    return response, payloads
