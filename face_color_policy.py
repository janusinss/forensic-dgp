"""Preserve black-and-white presentation using only the uploaded visible pixels."""
import cv2
import numpy as np

POLICY = "visible-input-grayscale-v2"


def grayscale_signal(rgb, mask):
    sample = cv2.resize(rgb, (256,256), interpolation=cv2.INTER_AREA)
    region = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)
    support = cv2.erode((1-region).astype(np.uint8), np.ones((9,9),np.uint8)) != 0
    blurred = cv2.GaussianBlur(sample.astype(np.float32), (5,5), 1)
    chroma = blurred.max(axis=-1)-blurred.min(axis=-1)
    count = int(support.sum())
    measured = float(chroma[support].mean()) if count >= 512 else None
    return {"policy":POLICY, "grayscale_input":measured is not None and measured <= 4,
            "mean_visible_channel_range_255":measured, "threshold":4,
            "visible_signal_pixels":count, "selection_uses":"Uploaded RGB outside removal area only"}


def preserve_input_palette(original, mask, output, visible_restored=False):
    if original.shape != output.shape or mask.shape != original.shape[:2]:
        raise ValueError("Palette policy requires matching original/mask/output dimensions")
    if original.dtype != np.uint8 or output.dtype != np.uint8 or not np.isin(mask,(0,1)).all():
        raise ValueError("Palette policy requires uint8 RGB and a binary removal mask")
    signal = grayscale_signal(original, mask)
    result = output.copy()
    if signal["grayscale_input"]:
        gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        if visible_restored:
            result = np.repeat(gray[...,None],3,-1)
        else:
            result[mask==1] = np.repeat(gray[...,None],3,-1)[mask==1]
    return result, {**signal, "applied":signal["grayscale_input"],
                    "scope":"All restored/output pixels" if visible_restored else "Generated region only; observed RGB exact"}
