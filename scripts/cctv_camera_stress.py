"""Versioned, deterministic RGB camera-stress proxy; not a calibrated CCTV sensor."""
import cv2
import numpy as np
from PIL import Image


PROFILES = [
    {"id": "clear", "sigma": 0.0, "motion": 0, "long_edge": 256,
     "exposure": 1.0, "shot": 0.0, "read": 0.0, "jpeg": None},
    {"id": "blur_lr24", "sigma": 1.2, "motion": 0, "long_edge": 24,
     "exposure": 1.0, "shot": 0.0, "read": 0.0, "jpeg": 45},
    {"id": "lowlight_lr32", "sigma": 0.8, "motion": 0, "long_edge": 32,
     "exposure": 0.35, "shot": 0.0008, "read": 0.002, "jpeg": 40},
    {"id": "motion_lr48", "sigma": 0.0, "motion": 9, "long_edge": 48,
     "exposure": 1.0, "shot": 0.0, "read": 0.0, "jpeg": 30},
    {"id": "compound_lr24", "sigma": 1.2, "motion": 5, "long_edge": 24,
     "exposure": 0.20, "shot": 0.0012, "read": 0.003, "jpeg": 22},
]


def reference_canvas(path):
    """Native center-pad, then bilinear256; preserve source aspect and field of view."""
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        width, height = rgb.size
        side = max(width, height)
        offset = ((side-width)//2, (side-height)//2)
        canvas = Image.new("RGB", (side, side), (128, 128, 128))
        canvas.paste(rgb, offset)
        mask = Image.new("L", (side, side), 0)
        mask.paste(255, (*offset, offset[0]+width, offset[1]+height))
    target = np.asarray(canvas.resize((256, 256), Image.Resampling.BILINEAR)).copy()
    observed = np.asarray(mask.resize((256, 256), Image.Resampling.NEAREST)).copy() > 0
    yy, xx = np.nonzero(observed)
    return target, observed, [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]


def degrade(target, bounds, profile, seed):
    """Stress the observed rectangle; retain the original padding outside it."""
    if profile["id"] == "clear":
        return target.copy(), {"working_size": [bounds[2]-bounds[0], bounds[3]-bounds[1]],
                               "low_resolution_size": None, "seed": seed}
    x0, y0, x1, y1 = bounds
    rgb = target[y0:y1, x0:x1].astype(np.float32)/255
    if profile["sigma"]:
        rgb = cv2.GaussianBlur(rgb, (0, 0), profile["sigma"], borderType=cv2.BORDER_REPLICATE)
    if profile["motion"]:
        k = profile["motion"]
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k//2] = 1/k  # Fixed horizontal motion, in 256px reference coordinates.
        rgb = cv2.filter2D(rgb, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    height, width = rgb.shape[:2]
    scale = profile["long_edge"]/max(height, width)
    low = (max(1, round(width*scale)), max(1, round(height*scale)))
    rgb = cv2.resize(rgb, low, interpolation=cv2.INTER_AREA)
    if profile["exposure"] != 1 or profile["shot"] or profile["read"]:
        # An approximate processed-RGB inverse transfer, not RAW unprocessing/ISP calibration.
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        linear *= profile["exposure"]
        rng = np.random.default_rng(seed)
        deviation = np.sqrt(profile["shot"]*linear + profile["read"]**2)
        linear = np.clip(linear + rng.normal(0, 1, linear.shape)*deviation, 0, 1)
        rgb = np.where(linear <= .0031308, linear*12.92, 1.055*linear**(1/2.4)-.055)
    rgb8 = (np.clip(rgb, 0, 1)*255).astype(np.uint8)
    if profile["jpeg"] is not None:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, profile["jpeg"]])
        if not ok:
            raise ValueError("JPEG proxy encoding failed")
        rgb8 = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    enlarged = cv2.resize(rgb8, (width, height), interpolation=cv2.INTER_CUBIC)
    result = target.copy()
    result[y0:y1, x0:x1] = enlarged
    return result, {"working_size": [width, height], "low_resolution_size": list(low), "seed": seed}
