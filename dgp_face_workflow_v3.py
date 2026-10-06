"""DGP-led research app, distinct from immutable historical experiments.

Functional integration is not qualification for native CCTV or hidden identity.
Input usability is an operator decision; blur/noise only suggests restoration.
"""
import hashlib
import io
import json
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image, ImageOps

from cctv_input_quality import observed_quality_signals
from dgp_face_restoration import prepare_crop
from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
from face_color_policy import preserve_input_palette
from face_workflow import FaceWorkflow, ROOT, UPLOAD_LIMIT, png_bytes, visibility_check

POLICY = "reviewed-dgp256-research-workflow-v3"
DGP_SHA256 = "646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b"
DEFAULT_DGP = ROOT / "outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth"


def array_sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def decode_crop(contents):
    """Accept native crop sizes without treating their dimensions as usability."""
    if not contents or len(contents) > UPLOAD_LIMIT:
        raise ValueError("Upload a PNG or JPEG smaller than 10 MB.")
    try:
        with Image.open(io.BytesIO(contents)) as source:
            if source.format not in ("PNG", "JPEG"):
                raise ValueError("Use a PNG or JPEG image.")
            if max(source.size) > 4096 or source.width * source.height > 16_000_000:
                raise ValueError("Use a face crop no larger than 4096 pixels per side.")
            return np.array(ImageOps.exif_transpose(source).convert("RGB"))
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Cannot read this image. Use a PNG or JPEG face crop.") from exc


def decode_mask(contents, shape):
    rgb = decode_crop(contents)
    if rgb.shape[:2] != tuple(shape):
        raise ValueError("The removal mask must match the uploaded face dimensions.")
    return (rgb.mean(axis=-1) >= 127.5).astype(np.uint8)


def review_input(review):
    if review == "needs_clearer":
        raise ValueError("Facial structure is insufficient. Upload a clearer face crop.")
    if review == "out_of_scope":
        raise ValueError("Upload one already cropped frontal or mildly turned face.")
    if review != "usable":
        raise ValueError("Review the input first: one frontal or mildly turned face with readable visible features.")


def canonical_tensor(canvas, device):
    # Audited V15 convention: float32 NumPy division BEFORE device transfer.
    values = canvas.astype(np.float32) / np.float32(255)
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(device)


def float_rgb(tensor, expected):
    if (tensor.shape != expected.shape or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1):
        raise FloatingPointError("Invalid model output; generation stopped.")
    return tensor[0].permute(1, 2, 0).cpu().numpy().copy()


def make_dgp_bundle(original, native_mask, prepared, mask, output, metadata, raw):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, image in (("original.png", original), ("removal-mask-original.png", native_mask * 255),
                            ("input-256.png", prepared), ("removal-mask.png", mask * 255),
                            ("estimate.png", output)):
            archive.writestr(name, png_bytes(image))
        if raw is not None:
            raw_stream = io.BytesIO()
            np.save(raw_stream, raw, allow_pickle=False)
            archive.writestr("dgp-raw-float32.npy", raw_stream.getvalue())
        archive.writestr("processing.json", json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        archive.writestr("README.txt", "Original is the decoded EXIF-oriented upload. Input, mask and estimate are aligned 256x256 canvases; neutral padding is not captured face evidence. White mask marks estimated regions. Raw DGP floats precede mask compositing and grayscale display processing. Hidden regions are plausible estimates, not exact hidden identity. This research app is not qualified for useful native CCTV restoration.\n")
    return stream.getvalue()


class DGPFaceWorkflow(FaceWorkflow):
    def __init__(self, device=None, detector_path=None, completion_path=None, restoration_path=None):
        super().__init__(device, detector_path, completion_path, restoration_path or DEFAULT_DGP)
        self.proposals = OrderedDict()

    def configuration(self):
        return {**super().configuration(), "policy": POLICY, "output_size": [256, 256],
                "primary_restorer": "retained own trained DGP identity-v2",
                "restoration_sha256": DGP_SHA256,
                "input_review": "Operator input-only usability review required; no automatic face qualification",
                "native_usefulness_qualified": False,
                "automatic_restoration": "Observed-support blur/noise suggestion; Auto/On/Off override",
                "completion_status": "Separate pretrained inpainting component; covering-family qualification incomplete"}

    def _restorer(self):
        if self.restorer is None:
            self.restorer, self.restorer_provenance = load_frozen_dgp_restorer(
                self.restoration_path, expected_sha256=DGP_SHA256, device=self.device)
            self.restorer_provenance = {**self.restorer_provenance,
                "checkpoint_role": "retained identity-supervised v2 research baseline",
                "native_usefulness_qualified": False}
        return self.restorer

    def review_mask(self, rgb):
        result = super().review_mask(rgb)
        # Historical minimum-one-native-pixel dilation is large on tiny CCTV crops.
        # Bound the new proposal margin to three pixels on the 256 canvas instead.
        radius = (3 * max(rgb.shape[:2])) // 256
        raw = result["raw_mask"]
        result["mask"] = cv2.dilate(raw, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) if radius else raw.copy()
        result["proposal_margin_pixels"] = radius
        result["proposal_margin_policy"] = "At most 3 pixels at 256 scale; zero native margin for crops under 86 pixels"
        # Compare the final reviewed native mask, rather than trusting a client label.
        key = (rgb.shape, array_sha(rgb))
        with self.lock:
            self.proposals[key] = {"sha256": array_sha(result["mask"]),
                                   "margin_native_pixels": result["proposal_margin_pixels"]}
            self.proposals.move_to_end(key)
            while len(self.proposals) > 128:
                self.proposals.popitem(last=False)
        return result

    def generate(self, rgb, native_mask, restoration="auto", input_review="", mask_reviewed=False):
        review_input(input_review)
        if not mask_reviewed:
            raise ValueError("Review and confirm the removal area before generation.")
        if restoration not in ("auto", "on", "off"):
            raise ValueError("Choose automatic, off or on restoration.")
        canvas, observed, mask, geometry = prepare_crop(rgb, native_mask)
        visible = observed & ~mask
        native_visible = rgb[native_mask == 0]
        if not visible.any() or not len(native_visible) or np.ptp(native_visible.astype(np.int16), axis=0).max() == 0:
            raise ValueError("No visible structure remains. Upload a clearer, less-covered face crop.")
        # Padding is not evidence when evaluating near-total covering.
        visibility = visibility_check((~observed | mask).astype(np.uint8))
        if visibility["rejected"]:
            raise ValueError("Too little face remains. Upload a less-covered face image.")
        signals = observed_quality_signals(canvas, observed, mask)
        if restoration == "auto" and signals["blur_variance"] is None:
            raise ValueError("Auto has too little visible area to assess. Review the crop and choose On or Off, or upload a less-covered image.")
        use_restoration = restoration == "on" or (restoration == "auto" and signals["suggest_restoration"])
        started = time.perf_counter()
        raw = None
        with self.lock, torch.inference_mode():
            self._runtime()
            x = canonical_tensor(canvas, self.device)
            m = torch.from_numpy(mask.copy()).unsqueeze(0).unsqueeze(0).to(self.device)
            completed = self._generator()(x, m.float()) if mask.any() else x
            completed_rgb = float_rgb(completed, x)
            output = canvas.copy()
            if use_restoration:
                raw = float_rgb(self._restorer()(x), x)
                output[visible] = np.floor(raw[visible] * np.float32(255)).astype(np.uint8)
            output[mask] = np.floor(completed_rgb[mask] * np.float32(255)).astype(np.uint8)
            # Off retains prepared observed RGB byte-for-byte, including mask boundaries.
            ignored = (~observed | mask).astype(np.uint8)
            output, colour = preserve_input_palette(canvas, ignored, output, visible_restored=use_restoration)
            output[~observed] = canvas[~observed]
            proposal = self.proposals.get((rgb.shape, array_sha(rgb)))
            automatic = proposal is not None and proposal["sha256"] == array_sha(native_mask)
        metadata = {"policy": POLICY, "device": self.device,
                    "elapsed_seconds": time.perf_counter() - started,
                    "width": 256, "height": 256, "geometry": geometry,
                    "input_review": {"decision": input_review, "source": "operator input-only review",
                                     "automatic_structure_qualification": False},
                    "restoration_requested": restoration, "restoration_applied": bool(use_restoration),
                    "quality_signals": signals, "visibility": visibility,
                    "mask_source": "automatic_reviewed" if automatic else "assisted_reviewed",
                    "mask_source_basis": "Final native mask equals this server's input-bound proposal" if automatic else "No matching automatic proposal; manual/imported/edited or cache expired",
                    "proposal_margin_native_pixels": proposal["margin_native_pixels"] if proposal else None,
                    "additional_mask_expansion": 0,
                    "completion": self.generator_provenance if mask.any() else {"bypassed": "empty mask"},
                    "visible_restoration": self.restorer_provenance if use_restoration else None,
                    "display_processing": {"float_to_png": "floor(float32 *255)", "colour_policy": colour,
                                           "composition": "DGP on observed visible support; completion only inside reviewed mask; original padding"},
                    "raw_dgp": {"sha256": array_sha(raw), "dtype": "float32", "shape": [256, 256, 3]} if raw is not None else None,
                    "original_rgb_sha256": array_sha(rgb), "native_mask_sha256": array_sha(native_mask),
                    "output_rgb_sha256": array_sha(output), "optimizer_updates": 0, "backward_calls": 0,
                    "native_usefulness_qualified": False, "covering_families_qualified": False,
                    "meaning": "One plausible estimate. Hidden identity is not verified."}
        return {"original": canvas, "native_original": rgb, "mask": mask.astype(np.uint8),
                "output": output, "metadata": metadata, "raw": raw}
