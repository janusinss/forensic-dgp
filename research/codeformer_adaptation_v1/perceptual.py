"""Frozen Torch-only LPIPS v0.1/VGG16, with standard RGB [-1,1] scaling.

The feature slicing and calibrated distance follow richzhang/PerceptualSimilarity
(BSD-style license) and torchvision VGG16 (BSD-3-Clause). Exact notices, source
references, original weights and conversion bindings accompany the asset pack.
This module has no dependency on torchvision/lpips and performs no downloading.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ASSET_SCHEMA = "codeformer-lpips-vgg16-assets-v1"
FEATURE_FILE = "vgg16_features.pth"
CALIBRATION_FILE = "lpips_vgg_v01.pth"
MANIFEST_FILE = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_asset(root: Path, entry: dict) -> Path:
    path = (root / entry["path"]).resolve()
    if path.parent != root or path.is_symlink() or not path.is_file():
        raise ValueError("Perceptual asset is not a regular direct child")
    if path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
        raise ValueError(f"Perceptual asset fingerprint differs: {path.name}")
    return path


def _feature_stack() -> nn.Sequential:
    # Official VGG16 feature indices; the final max-pool is not used by LPIPS.
    cfg = (64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
           512, 512, 512, "M", 512, 512, 512)
    layers: list[nn.Module] = []
    incoming = 3
    for outgoing in cfg:
        if outgoing == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.extend((nn.Conv2d(incoming, outgoing, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)))
            incoming = outgoing
    return nn.Sequential(*layers)


class FrozenLPIPSVGG16(nn.Module):
    """Scalar mean LPIPS distance; frozen parameters, differentiable prediction.

    Prediction is the raw CodeFormer normalized-domain output (nominally [-1,1],
    allowed outside that range). Target is RGB [-1,1]. No clipping, external
    ImageNet normalization or implicit resizing is performed here.
    """

    def __init__(self):
        super().__init__()
        self.features = _feature_stack()
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188]).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.tensor([.458, .448, .450]).view(1, 3, 1, 1))
        self.calibration = nn.ParameterList([
            nn.Parameter(torch.empty(1, channels, 1, 1), requires_grad=False)
            for channels in (64, 128, 256, 512, 512)
        ])
        self.provenance: dict = {}
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True):
        # A parent restoration worker's .train() must not activate this network.
        return super().train(False)

    def _features(self, image: torch.Tensor) -> list[torch.Tensor]:
        image = (image - self.shift) / self.scale
        outputs = []
        for index, layer in enumerate(self.features):
            image = layer(image)
            if index in (3, 8, 15, 22, 29):
                # Match lpips.normalize_tensor: epsilon is OUTSIDE the square root.
                norm = torch.sqrt(torch.sum(image ** 2, dim=1, keepdim=True))
                outputs.append(image / (norm + 1e-10))
        return outputs

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("LPIPS needs matching NCHW images")
        if prediction.shape[0] < 1 or prediction.shape[1] != 3 or min(prediction.shape[2:]) < 32:
            raise ValueError("LPIPS needs nonempty RGB images of at least 32 pixels")
        if prediction.device != target.device or prediction.device != self.shift.device:
            raise ValueError("LPIPS input/target/module devices differ")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError("LPIPS inputs must be floating point")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("LPIPS inputs must be finite")
        if target.min() < -1 or target.max() > 1:
            raise ValueError("LPIPS clean target must use RGB [-1,1]")
        # Stable FP32 perceptual objective, including under an outer AMP context.
        # No no_grad/inference decorator: gradients must reach the prediction.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            left = self._features(prediction.float())
            right = self._features(target.float())
            distances = [F.conv2d((a - b) ** 2, weight).mean(dim=(2, 3), keepdim=True)
                         for a, b, weight in zip(left, right, self.calibration)]
            return sum(distances).mean()


def load_perceptual(asset_root: Path, device: str | torch.device = "cpu") -> FrozenLPIPSVGG16:
    """Load only verified compact assets; never download, install or use random fallback."""
    root = Path(asset_root).resolve()
    manifest_path = root / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ASSET_SCHEMA:
        raise ValueError("Unsupported perceptual asset manifest")
    feature_path = _checked_asset(root, manifest["features"])
    calibration_path = _checked_asset(root, manifest["calibration"])
    if feature_path.name != FEATURE_FILE or calibration_path.name != CALIBRATION_FILE:
        raise ValueError("Unexpected perceptual asset filenames")
    payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    if payload.get("source_sha256") != manifest["original_vgg16"]["sha256"]:
        raise ValueError("Compact VGG16 source binding differs")
    features = payload["params"]
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=True)
    expected = {f"lin{index}.model.1.weight" for index in range(5)}
    if set(calibration) != expected:
        raise ValueError("Unexpected LPIPS v0.1 VGG calibration keys")
    model = FrozenLPIPSVGG16()
    model.features.load_state_dict({key.removeprefix("features."): value
                                    for key, value in features.items()}, strict=True)
    with torch.no_grad():
        for index, weight in enumerate(model.calibration):
            source = calibration[f"lin{index}.model.1.weight"]
            if source.shape != weight.shape or source.dtype != torch.float32:
                raise ValueError("LPIPS calibration shape/dtype differs")
            weight.copy_(source)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Nonfinite perceptual state")
    model.provenance = {"schema": ASSET_SCHEMA, "lpips_version": "0.1",
                        "package_version": "0.1.4", "backbone": "VGG16_IMAGENET1K_V1",
                        "manifest_sha256": _sha256(manifest_path),
                        "features_sha256": manifest["features"]["sha256"],
                        "calibration_sha256": manifest["calibration"]["sha256"],
                        "input": "RGB raw prediction / clean target [-1,1]; standard LPIPS scaling",
                        "normalization": "No external ImageNet normalization; FP32; no clipping",
                        "loss_reduction": "sum over five layers, spatial mean, batch mean"}
    return model.to(device).eval().requires_grad_(False)
