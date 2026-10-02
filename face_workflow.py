"""Local, inference-only face workflow. Reviewed masks are never expanded again."""
import hashlib
import io
import json
import os
import threading
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
UPLOAD_LIMIT = 10 * 1024 * 1024
POLICY = "reviewed-face-workflow-v1"
PARENT_SHA256 = "c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93"
DEFAULT_DETECTOR = ROOT / "outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth"
DEFAULT_COMPLETION = ROOT / "checkpoints/codeformer_inpainting.pth"
DEFAULT_RESTORATION = ROOT / "outputs/codeformer_restoration_pretrained_v1/codeformer.pth"


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode_image(contents):
    if not contents or len(contents) > UPLOAD_LIMIT:
        raise ValueError("Upload a PNG or JPEG smaller than 10 MB.")
    try:
        with Image.open(io.BytesIO(contents)) as source:
            if source.format not in ("PNG", "JPEG"):
                raise ValueError("Use a PNG or JPEG image.")
            if min(source.size) < 32 or max(source.size) > 4096 or source.width * source.height > 16_000_000:
                raise ValueError("Use a cropped face between 32 and 4096 pixels per side.")
            return np.array(ImageOps.exif_transpose(source).convert("RGB"))
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Cannot read this image. Use a PNG or JPEG face crop.") from exc


def decode_removal_mask(contents, shape):
    rgb = decode_image(contents)
    if rgb.shape[:2] != tuple(shape):
        raise ValueError("The removal mask must match the uploaded face dimensions.")
    return (rgb.mean(axis=-1) >= 127.5).astype(np.uint8)


def png_bytes(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def square_pad(rgb, mask=None):
    """Neutral padding avoids reflecting a covering into the visible context."""
    h, w = rgb.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    padded = np.full((side, side, 3), 127, np.uint8)
    padded[top:top+h, left:left+w] = rgb
    region = np.zeros((side, side), np.uint8)
    if mask is not None:
        region[top:top+h, left:left+w] = mask
    return padded, region, (top, left, h, w)


def visibility_check(square_mask):
    """Conservative crop geometry heuristic, conditional on a reviewed mask.

    This is not a face/landmark detector. For cropped frontal/mildly turned faces,
    reject near-total covering of an approximate facial ellipse or all four
    central feature bands. Never equate remaining background with visible face.
    """
    m = cv2.resize(square_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST) != 0
    yy, xx = np.mgrid[:256, :256]
    face = ((xx - 127.5) / 83) ** 2 + ((yy - 137) / 104) ** 2 <= 1
    bands = {"left_eye": (60, 84, 119, 133), "right_eye": (137, 84, 196, 133),
             "nose": (108, 119, 149, 166), "mouth": (82, 165, 174, 208)}
    fractions = {name: float(m[y1:y2, x1:x2].mean())
                 for name, (x1, y1, x2, y2) in bands.items()}
    face_covered = float(m[face].mean())
    all_features = (fractions["left_eye"] >= .85 and fractions["right_eye"] >= .85
                    and fractions["nose"] >= .9 and fractions["mouth"] >= .9)
    rejected = face_covered >= .80 or all_features or m.mean() >= .85
    return {"rejected": bool(rejected), "face_covered_fraction": face_covered,
            "feature_covered_fractions": fractions,
            "scope": "Approximate frontal-crop geometry, conditional on the reviewed removal area"}


def quality_signals(square_rgb, square_mask):
    """Input-only blur/noise signals. Thresholds are developmental, overridable."""
    rgb = cv2.resize(square_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(square_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    support = cv2.erode(1-mask, np.ones((9, 9), np.uint8)).astype(bool)
    support[:4] = False
    support[-4:] = False
    support[:, :4] = False
    support[:, -4:] = False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    enough = int(support.sum()) >= 512
    if not enough:
        return {"blur_variance": None, "noise_sigma_255": None, "suggest_restoration": False,
                "reason": "Too little visible area for the automatic quality signal; choose an override.",
                "visible_signal_pixels": int(support.sum()), "blur_threshold": 24, "noise_threshold": 8}
    smoothed = cv2.GaussianBlur(gray, (3, 3), .6)
    blur = float(cv2.Laplacian(smoothed, cv2.CV_32F)[support].var())
    low = cv2.GaussianBlur(gray, (7, 7), 1.5)
    gradient = np.hypot(cv2.Sobel(low, cv2.CV_32F, 1, 0), cv2.Sobel(low, cv2.CV_32F, 0, 1))
    flat = support & (gradient <= np.percentile(gradient[support], 35))
    response = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32))
    noise = float(np.median(np.abs(response[flat])) / (6 * .67448975))
    reasons = []
    if blur < 24:
        reasons.append("blur signal")
    if noise >= 8:
        reasons.append("noise signal")
    return {"blur_variance": blur, "noise_sigma_255": noise, "suggest_restoration": bool(reasons),
            "reason": ", ".join(reasons) if reasons else "Visible detail does not trigger restoration.",
            "visible_signal_pixels": int(support.sum()), "blur_threshold": 24, "noise_threshold": 8}


def make_review_bundle(original, mask, output, metadata):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("original.png", png_bytes(original))
        archive.writestr("removal-mask.png", png_bytes(mask * 255))
        archive.writestr("estimate.png", png_bytes(output))
        archive.writestr("processing.json", json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        archive.writestr("README.txt", "White mask pixels mark generated facial regions. Hidden features are plausible estimates, not verified identity. Original is the decoded, EXIF-oriented upload.\n")
    return stream.getvalue()


class FaceWorkflow:
    """Lazy, serialized pretrained inference. No training or random fallback."""
    def __init__(self, device=None, detector_path=None, completion_path=None, restoration_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_path = Path(detector_path or os.environ.get("FACE_DETECTOR_CHECKPOINT", DEFAULT_DETECTOR))
        self.completion_path = Path(completion_path or os.environ.get("FACE_COMPLETION_CHECKPOINT", DEFAULT_COMPLETION))
        self.restoration_path = Path(restoration_path or os.environ.get("FACE_RESTORATION_CHECKPOINT", DEFAULT_RESTORATION))
        self.lock = threading.Lock()
        self.detector = self.generator = self.restorer = None
        self.detector_state = self.generator_provenance = self.restorer_provenance = None

    def configuration(self):
        return {"device": self.device, "policy": POLICY,
                "detector_configured": self.detector_path.is_file(),
                "completion_configured": self.completion_path.is_file(),
                "restoration_configured": self.restoration_path.is_file(),
                "automatic_mask_status": "Development baseline; review every removal area and correct missed coverings.",
                "output_kind": "One plausible estimate; hidden identity is not verified"}

    def _runtime(self):
        if self.device == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))

    def _detector(self):
        if self.detector is None:
            from completion_inference import load_completion
            if file_sha(self.detector_path) != PARENT_SHA256:
                raise ValueError("The configured detector is not the retained baseline. Review a candidate before selecting it.")
            model, state = load_completion(self.detector_path, self.device)
            self.detector = model.eval().requires_grad_(False)
            self.detector_state = state
        return self.detector

    def _generator(self):
        if self.generator is None:
            from pretrained_completion import load_codeformer
            self.generator, self.generator_provenance = load_codeformer(self.completion_path, self.device)
        return self.generator

    def _restorer(self):
        if self.restorer is None:
            from pretrained_face_restoration import load_face_restorer
            self.restorer, self.restorer_provenance = load_face_restorer(self.restoration_path, self.device)
        return self.restorer

    def review_mask(self, rgb):
        with self.lock, torch.inference_mode():
            self._runtime()
            padded, _, (top, left, h, w) = square_pad(rgb)
            detector = self._detector()
            size = self.detector_state["size"]
            sample = cv2.resize(padded, (size, size), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(sample.copy()).permute(2, 0, 1).float()[None].to(self.device)/255
            probability = detector.detect(x).sigmoid()[0, 0].cpu().numpy()
            if not np.isfinite(probability).all():
                raise FloatingPointError("Invalid region detector output")
            raw = (cv2.resize(probability, (len(padded), len(padded)), interpolation=cv2.INTER_LINEAR) >= .5).astype(np.uint8)
            radius = max(1, round(3 * len(padded)/256))
            proposed = cv2.dilate(raw, np.ones((radius*2+1, radius*2+1), np.uint8))
            return {"original": rgb, "raw_mask": raw[top:top+h, left:left+w].copy(),
                    "mask": proposed[top:top+h, left:left+w].copy(),
                    "proposal_margin_pixels": radius, "threshold": .5,
                    "detector_sha256": PARENT_SHA256,
                    "message": "Review the marked area. Paint missed coverings; erase clear glasses and visible features. The detector is a development baseline."}

    def generate(self, rgb, mask, restoration="auto"):
        if restoration not in ("auto", "off", "on"):
            raise ValueError("Choose automatic, off or on restoration.")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected decoded RGB pixels.")
        if mask.shape != rgb.shape[:2] or not np.isin(mask, (0, 1)).all():
            raise ValueError("Use a matching binary removal mask.")
        padded, square_mask, bounds = square_pad(rgb, mask)
        visibility = visibility_check(square_mask)
        if visibility["rejected"]:
            raise ValueError("Too much of the face is hidden. Upload a less-covered face image; do not generate an identity from almost no visible features.")
        signals = quality_signals(padded, square_mask)
        use_restoration = restoration == "on" or (restoration == "auto" and signals["suggest_restoration"])
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            self._runtime()
            x = torch.from_numpy(padded.copy()).permute(2, 0, 1).float()[None].to(self.device)/255
            m = torch.from_numpy(square_mask.copy()).float()[None, None].to(self.device)
            completed = self._generator()(x, m) if mask.any() else x.clone()
            if completed.shape != x.shape or not torch.isfinite(completed).all():
                raise FloatingPointError("Invalid completion output")
            completed = torch.where(m.bool(), completed, x)
            output = completed
            if use_restoration:
                restored = self._restorer()(completed, fidelity=1.0)
                if restored.shape != x.shape or not torch.isfinite(restored).all():
                    raise FloatingPointError("Invalid restoration output")
                output = torch.where(m.bool(), completed, .5*restored + .5*completed)
            output = (output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
        top, left, h, w = bounds
        output = output[top:top+h, left:left+w].copy()
        if not use_restoration:
            output[mask == 0] = rgb[mask == 0]
        metadata = {"policy": POLICY, "device": self.device, "elapsed_seconds": time.perf_counter()-started,
                    "restoration_requested": restoration, "restoration_applied": use_restoration,
                    "restoration_fidelity": 1.0 if use_restoration else None,
                    "visible_restoration_blend": .5 if use_restoration else 0,
                    "quality_signals": signals, "visibility": visibility,
                    "mask_source": "reviewed operator mask", "additional_mask_expansion": 0,
                    "padding": "neutral gray; original dimensions restored",
                    "completion": self.generator_provenance if mask.any() else {"bypassed": "empty mask"},
                    "visible_restoration": self.restorer_provenance if use_restoration else None,
                    "width": w, "height": h, "optimizer_updates": 0,
                    "original_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "removal_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                    "output_rgb_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                    "meaning": "Hidden facial features are plausible estimates, not verified identity."}
        return output, metadata
