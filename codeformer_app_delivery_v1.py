"""Pure delivery boundary for a future CodeFormer app integration.

Consumes the existing reviewed worker's bytes; does not load or select a model,
run inference, change the app default, validate Auto routing, or complete a face.
The controller must separately enforce input review and worker source ownership.
"""
from dataclasses import dataclass
import hashlib
import io

import numpy as np
from PIL import Image, ImageOps

ARMS = ("incumbent_U96", "postfusion_LR_u64")
UPLOAD_LIMIT = 10 * 1024 ** 2
MODEL_FILES = {
    "r2": "d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896",
    "u96": "958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0",
    "heads": "77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def png_array(contents, mode, shape):
    need(type(contents) is bytes and 0 < len(contents) <= 1024 ** 2,
         "Bounded PNG bytes required.")
    with Image.open(io.BytesIO(contents)) as image:
        need(image.format == "PNG" and image.mode == mode and image.size == (256, 256),
             "Exact 256 PNG channels required.")
        value = np.array(image)
    need(value.shape == shape and value.dtype == np.uint8, "PNG array shape differs.")
    return value


def prepare_native(contents):
    """Pure mirror of retained prepare_crop geometry; no Torch parent import.

    Neutral 128 center padding, bilinear RGB and nearest observed support. This
    does not judge whether a face contains enough usable facial information.
    """
    need(type(contents) is bytes and 0 < len(contents) <= UPLOAD_LIMIT,
         "Upload a PNG or JPEG smaller than 10 MB.")
    with Image.open(io.BytesIO(contents)) as image:
        need(image.format in ("PNG", "JPEG"), "Use a PNG or JPEG face crop.")
        need(max(image.size) <= 4096 and image.width * image.height <= 16_000_000,
             "Face crop exceeds the image bounds.")
        rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    h, w = rgb.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(Image.fromarray(rgb), (left, top))
    support = Image.new("L", (side, side), 0)
    support.paste(255, (left, top, left + w, top + h))
    canvas = np.array(canvas.resize((256, 256), Image.Resampling.BILINEAR))
    observed = np.array(support.resize((256, 256), Image.Resampling.NEAREST)) > 0
    need(observed.any(), "No observed input pixels survive the 256 canvas; use a face crop.")
    yy, xx = np.nonzero(observed)
    geometry = {
        "native_size": [w, h], "native_center_offset": [left, top],
        "observed_bounds_256": [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)],
        "output_size": [256, 256], "policy": "dgp256-observed-canvas-v1",
        "input_rgb_sha256": sha(rgb.tobytes()), "canvas_rgb_sha256": sha(canvas.tobytes()),
        "observed_sha256": sha(observed.tobytes()),
        "removal_mask_sha256": sha(np.zeros((256, 256), dtype=bool).tobytes()),
    }
    return canvas, observed, geometry


@dataclass(frozen=True)
class FrozenDelivery:
    prepared: np.ndarray
    observed: np.ndarray
    raw: np.ndarray
    pixels: np.ndarray
    png: bytes
    selection: str
    upload_sha256: str


def decode_worker_delivery(value, contents, selection, expected_models):
    """Reject delivery drift before a future controller accepts worker output.

    Validates saved transport/delivery, not the model's neural relationship or
    useful restoration. Source manifests and process counters remain separate.
    """
    need(selection in ARMS, "Choose a retained restoration arm.")
    need(type(value) is dict and set(value) == {"PNG", "raw", "prepared", "support", "metadata"},
         "Exact worker result fields required.")
    metadata = value["metadata"]
    need(type(metadata) is dict and metadata.get("selection") == selection,
         "Worker model selection differs.")
    need(type(expected_models) is dict and set(expected_models) == set(MODEL_FILES),
         "Explicit reviewed model file bindings required.")
    for name, digest in MODEL_FILES.items():
        row = expected_models[name]
        need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
             and type(row["path"]) is str and bool(row["path"])
             and type(row["bytes"]) is int and row["bytes"] > 0 and row["sha256"] == digest,
             "Reviewed model binding differs.")
    need(metadata.get("models") == expected_models, "Worker model file bindings differ.")
    expected_parameters = {"CPU_FP32": True, "w": 1.0, "AdaIN": True, "detach16": True,
                           "internal_resolution": 512, "delivery_resolution": 256,
                           "display_processing": "none", "model_qualified": False,
                           "main_app_default_changed": False}
    need(all(type(metadata.get(k)) is type(v) and metadata[k] == v for k, v in expected_parameters.items()),
         "Frozen delivery parameters differ.")
    canvas, observed, geometry = prepare_native(contents)
    need(metadata.get("uploaded_bytes_sha256") == sha(contents) and metadata.get("geometry") == geometry,
         "Worker input or geometry binding differs.")
    prepared = png_array(value["prepared"], "RGB", (256, 256, 3))
    support = png_array(value["support"], "L", (256, 256))
    pixels = png_array(value["PNG"], "RGB", (256, 256, 3))
    need(np.array_equal(prepared, canvas) and np.array_equal(support, observed.astype(np.uint8) * 255),
         "Worker prepared input or observed support differs.")
    raw_bytes = value["raw"]
    need(type(raw_bytes) is bytes and 0 < len(raw_bytes) <= 1024 ** 2, "Bounded raw NPY required.")
    stream = io.BytesIO(raw_bytes)
    need(np.lib.format.read_magic(stream) == (1, 0), "Worker NPY v1 format required.")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
    need(shape == (256, 256, 3) and dtype == np.dtype(np.float32) and fortran is False
         and len(raw_bytes) - stream.tell() == 256 * 256 * 3 * 4,
         "Exact bounded raw NPY header and payload required.")
    raw = np.frombuffer(raw_bytes, dtype=np.float32, offset=stream.tell()).reshape(shape).copy()
    need(np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1,
         "Finite float32 256 RGB delivery required.")
    need(metadata.get("raw_RGB_sha256") == sha(raw.tobytes())
         and metadata.get("PNG_RGB_sha256") == sha(pixels.tobytes()), "Worker output hashes differ.")
    need(np.array_equal(raw[~observed], canvas[~observed].astype(np.float32) / np.float32(255))
         and np.array_equal(pixels[~observed], canvas[~observed]), "Padding delivery differs.")
    expected_png = canvas.copy()
    expected_png[observed] = np.floor(raw[observed] * np.float32(255)).astype(np.uint8)
    need(np.array_equal(pixels, expected_png), "RAW to floor PNG delivery differs.")
    arrays = [a.copy() for a in (canvas, observed, raw, pixels)]
    for array in arrays:
        array.flags.writeable = False
    return FrozenDelivery(*arrays, value["PNG"], selection, sha(contents))


def compose_visible(delivery, removal_mask):
    """Keep a later completion component's region out of visible restoration.

    No completion is generated here. Masked pixels retain prepared input until
    that separate component supplies an explicitly reviewed plausible estimate.
    """
    need(isinstance(delivery, FrozenDelivery), "Validated frozen delivery required.")
    need(isinstance(removal_mask, np.ndarray) and removal_mask.shape == (256, 256)
         and removal_mask.dtype in (np.dtype(bool), np.dtype(np.uint8))
         and np.isin(removal_mask, (0, 1)).all(), "Matching binary removal mask required.")
    visible = delivery.observed & ~removal_mask.astype(bool)
    output = delivery.prepared.copy()
    output[visible] = delivery.pixels[visible]
    return output
