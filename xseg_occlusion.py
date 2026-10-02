"""Pinned XSeg1 research inference; not selected by the main application.

XSeg predicts visible face, not a covering class. The explicit crop ellipse is
a developmental conversion policy; background errors require evaluation.
Official model/preprocessing attribution is retained in the acquisition record.
No FaceFusion implementation is imported or copied.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np

XSEG_SHA256 = "c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3"
POLICY = "xseg-frontal-ellipse-proposal-v1"


def facial_domain():
    yy, xx = np.mgrid[:256, :256]
    return (((xx - 127.5) / 83) ** 2 + ((yy - 137) / 104) ** 2 <= 1)


def covering_proposal(probability):
    if probability.shape != (256, 256) or not np.isfinite(probability).all():
        raise ValueError("Expected finite 256-square visible-face probability")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Visible-face probability outside [0, 1]")
    domain = facial_domain()
    raw = ((probability < .5) & domain).astype(np.uint8)
    proposed = cv2.dilate(raw, np.ones((7, 7), np.uint8)) * domain
    return raw, proposed.astype(np.uint8)


class XSegVisibleFace:
    def __init__(self, weights, threads=4):
        import onnxruntime as ort
        path = Path(weights)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != XSEG_SHA256:
            raise ValueError("XSeg1 weights differ from verified acquisition")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "input" or inputs[0].shape[1:] != [256, 256, 3] or inputs[0].type != "tensor(float)":
            raise ValueError("Unexpected XSeg NHWC input contract")
        if len(outputs) != 1 or outputs[0].shape[1:] != [256, 256, 1] or outputs[0].type != "tensor(float)":
            raise ValueError("Unexpected XSeg NHWC output contract")
        self.forward_count = 0

    def __call__(self, rgb):
        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
            raise ValueError("Research adapter requires uint8 RGB 256-square face crops")
        sample = np.ascontiguousarray(rgb[..., ::-1][None], dtype=np.float32) / 255
        self.forward_count += 1
        result = self.session.run(None, {"input": sample})[0]
        if result.shape != (1, 256, 256, 1):
            raise ValueError("Unexpected XSeg result shape")
        probability = result[0, ..., 0]
        if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
            raise FloatingPointError("Invalid visible-face probability")
        return probability.copy()
