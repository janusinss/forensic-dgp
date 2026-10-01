"""One fixed, input-derived face crop; standalone CPU diagnostic, no fitting."""
import math
import cv2
import numpy as np
import torch

SIZE = 256
CONFIDENCE = .6
CONTEXT = 1.25


def require(condition, message):
    if not condition: raise ValueError(message)


def make_crop(rgb, boxes):
    """Targets never enter detection, crop choice, margin or fallback decisions."""
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3
            and rgb.shape[2] == 3 and min(rgb.shape[:2]) >= 32, 'Require byte RGB image >=32 pixels')
    height, width = rgb.shape[:2]
    identity = np.array([[1., 0., 0.], [0., 1., 0.]])
    result = {'cropped': False, 'reason': 'missing_multiple_or_invalid_face', 'rgb': rgb.copy(),
              'affine': identity, 'inverse': identity.copy(), 'roi': np.ones((height, width), bool),
              'source_shape': [height, width], 'bounds': [0, 0, width, height], 'face_confidence': None}
    boxes = np.asarray(boxes)
    if boxes.shape != (1, 5) or not np.issubdtype(boxes.dtype, np.number) or not np.isfinite(boxes).all(): return result
    x0, y0, x1, y1, score = (float(v) for v in boxes[0])
    result['face_confidence'] = score
    if not CONFIDENCE <= score <= 1: result['reason'] = 'low_confidence'; return result
    cx, cy = (x0+x1)/2, (y0+y1)/2
    if min(x1-x0, y1-y0) < 16 or not (0 <= cx < width and 0 <= cy < height):
        result['reason'] = 'invalid_or_tiny_box'; return result
    side = math.ceil(CONTEXT*max(x1-x0, y1-y0))
    if side < 32 or side >= max(height, width): result['reason'] = 'no_reliable_zoom'; return result
    left, top = math.floor(cx-(side-1)/2), math.floor(cy-(side-1)/2)
    scale = (SIZE-1)/(side-1)
    affine = np.array([[scale, 0., -scale*left], [0., scale, -scale*top]], np.float64)
    cropped = cv2.warpAffine(rgb, affine, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(96, 96, 96))
    yy, xx = np.mgrid[:height, :width]
    roi = (xx >= left) & (xx < left+side) & (yy >= top) & (yy < top+side)
    require(roi.any(), 'Crop has no original image support')
    result.update(cropped=True, reason='single_reliable_face', rgb=cropped, affine=affine,
                  inverse=cv2.invertAffineTransform(affine), roi=roi,
                  bounds=[left, top, left+side, top+side])
    return result


def warp_target(target, transform):
    """Diagnostic target mapping after the input-only crop is already frozen."""
    require(isinstance(target, np.ndarray) and target.dtype == np.bool_ and target.ndim == 2
            and list(target.shape) == transform['source_shape'], 'Require matching binary target')
    if not transform['cropped']: return target.copy()
    return cv2.warpAffine(target.astype('uint8'), transform['affine'], (SIZE, SIZE),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def map_prediction(prediction, original, transform):
    require(isinstance(prediction, np.ndarray) and prediction.dtype == np.bool_
            and prediction.shape == (SIZE, SIZE) and isinstance(original, np.ndarray)
            and original.dtype == np.bool_ and list(original.shape) == transform['source_shape'],
            'Require binary crop prediction and original mask with matching geometry')
    if not transform['cropped']: return original.copy()
    height, width = original.shape
    mapped = cv2.warpAffine(prediction.astype('uint8'), transform['inverse'], (width, height),
                            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return np.where(transform['roi'], mapped, original)


def infer_crop_masks(model, items, *, progress=True):
    require(items and len({r['id'] for r in items}) == len(items) and
            all(v.device.type == 'cpu' for v in model.state_dict().values()), 'Require unique cases and CPU model')
    model.requires_grad_(False).eval()
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    predictions = {}; forwards = 0
    with torch.inference_mode():
        for i, row in enumerate(items, 1):
            transform, original = row['transform'], row['original']
            require(original.dtype == np.bool_ and list(original.shape) == transform['source_shape'],
                    'Original saved mask shape/type differs')
            if transform['cropped']:
                image = transform['rgb']
                require(image.dtype == np.uint8 and image.shape == (SIZE, SIZE, 3), 'Invalid crop RGB input')
                x = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float()/255
                logits = model.detect(x)
                require(isinstance(logits, torch.Tensor) and logits.device.type == 'cpu' and
                        logits.shape == (1, 1, SIZE, SIZE) and torch.isfinite(logits).all(), 'Invalid crop logits')
                p = (logits.sigmoid()[0, 0] >= .5).numpy(); forwards += 1
                predictions[row['id']] = map_prediction(p, original, transform)
            else: predictions[row['id']] = original.copy()
            if progress and i % 50 == 0: print('Frozen crop inference', i, len(items), flush=True)
    require(all(torch.equal(v.cpu(), before[k]) for k, v in model.state_dict().items())
            and all(p.grad is None for p in model.parameters()), 'Read-only crop inference changed state or gradients')
    return predictions, {'cases': len(items), 'forward_count': forwards, 'model_state_unchanged': True,
                         'optimizer_constructed': False, 'optimizer_updates_locally': 0}
