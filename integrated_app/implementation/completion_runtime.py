"""Portable frozen R1 completion on an already prepared/restored RGB256 canvas.

Importing this module loads no Torch model, detector, restoration or research
service. Completion preserves the supplied canvas outside the reviewed mask.
If that canvas was restored, the preservation reference is the restored canvas,
not the earlier upload. Original/prepared/restored evidence belongs to the app.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from . import completion_assets as assets

ROOT = Path(__file__).resolve().parents[1]
LOGICAL_SHA256 = "55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6"
ORIGINAL_SHA256 = "b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797"
PROTOCOL_SHA256 = "8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d"
ARCHITECTURE_SOURCES = {
    "third_party.codeformer_completion.codeformer_arch": ("codeformer_arch.py", "29c40cee39414c1117d1ae55fc15d6cdd42aed09e03019522c7f55771b1b84af"),
    "third_party.codeformer_completion.vqgan_arch": ("vqgan_arch.py", "01d7d9a796489716439cbd59fbd9e305278d762d97e6582aafd54b9279e73265"),
    "third_party.codeformer_completion.registry": ("registry.py", "21c23bd3bf727eb2decd99035120fc47316db92486d73116cca18e9fa8b6c854"),
}


class CompletionRefused(ValueError):
    """Insufficient reviewed facial support; ask for a clearer/less-covered crop."""


def need(ok, message):
    if not ok:
        raise ValueError(message)


def model_contract():
    entry = assets.load_manifest()["assets"]["completion_r1"]
    return {"model_id": "completion_r1", "model_label": "Trained CodeFormer completion R1",
            "backend": "codeformer-inpainting-finetune-r1", "weights_sha256": entry["sha256"],
            "original_checkpoint_sha256": ORIGINAL_SHA256, "logical_state_sha256": LOGICAL_SHA256,
            "training_protocol_sha256": PROTOCOL_SHA256, "source_revision": entry["source_revision"],
            "device": "cpu", "precision": "float32", "internal_resolution": 512,
            "delivered_resolution": 256, "input_policy": "visible-normalized-resize-v1",
            "compositing_policy": "mask-only-v1", "additional_mask_expansion": 0,
            "model_qualified": False, "hidden_identity_verified": False,
            "claim": "One plausible estimate in the reviewed removal area; hidden identity is not verified."}


def validate_arrays(rgb, mask, prepared_bounds=None):
    need(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (256, 256, 3),
         "Expected the exact prepared/restored RGB256 uint8 canvas.")
    need(isinstance(mask, np.ndarray) and mask.dtype in (np.dtype("uint8"), np.dtype("bool"))
         and mask.shape == (256, 256), "Expected a matching binary256 removal mask.")
    values = set(np.unique(mask).tolist())
    need(values.issubset({0, 1}) or values.issubset({0, 255}), "Use only binary0/1 or0/255 removal values.")
    removal = (mask != 0).astype(np.uint8)
    bounds = [0, 0, 256, 256] if prepared_bounds is None else prepared_bounds
    need(isinstance(bounds, (list, tuple)) and len(bounds) == 4 and all(type(v) is int for v in bounds),
         "prepared_bounds must contain four integer content coordinates.")
    left, top, right, bottom = bounds
    need(0 <= left < right <= 256 and 0 <= top < bottom <= 256, "Invalid prepared content bounds.")
    unavailable = np.ones((256, 256), np.uint8)
    unavailable[top:bottom, left:right] = 0
    need(not removal[unavailable != 0].any(), "Keep prepared padding outside the removal area.")
    unavailable = np.maximum(unavailable, removal)
    review = visibility_check256(unavailable)
    if review["rejected"]:
        raise CompletionRefused("Too little visible facial structure remains. Use a clearer or less-covered face crop.")
    return removal, list(bounds), review


def visibility_check256(unavailable):
    """Retained frontal-crop geometry heuristic; not a face/quality detector."""
    need(isinstance(unavailable, np.ndarray) and unavailable.shape == (256, 256), "Expected256 visibility support.")
    m = unavailable != 0
    yy, xx = np.mgrid[:256, :256]
    face = ((xx - 127.5) / 83) ** 2 + ((yy - 137) / 104) ** 2 <= 1
    bands = {"left_eye": (60, 84, 119, 133), "right_eye": (137, 84, 196, 133),
             "nose": (108, 119, 149, 166), "mouth": (82, 165, 174, 208)}
    fractions = {name: float(m[y1:y2, x1:x2].mean()) for name, (x1, y1, x2, y2) in bands.items()}
    covered = float(m[face].mean())
    all_features = (fractions["left_eye"] >= .85 and fractions["right_eye"] >= .85
                    and fractions["nose"] >= .9 and fractions["mouth"] >= .9)
    return {"rejected": bool(covered >= .80 or all_features or m.mean() >= .85),
            "face_covered_fraction": covered, "feature_covered_fractions": fractions,
            "padding_counts_as_unavailable": True,
            "scope": "Approximate frontal/mild-turn crop geometry conditional on the reviewed mask; not a detector or proof of adequate detail."}


def encode_output(rgb, removal, completed):
    need(isinstance(completed, np.ndarray) and completed.dtype == np.float32
         and completed.shape == rgb.shape and np.isfinite(completed).all(), "Invalid completion float output.")
    need(float(completed.min()) >= 0 and float(completed.max()) <= 1, "Completion output must lie in[0,1].")
    result = np.rint(completed * np.float32(255)).clip(0, 255).astype(np.uint8)
    result[removal == 0] = rgb[removal == 0]
    return result


def architecture_file_sha256(path):
    raw = assets.ordinary_file(path).read_bytes().replace(b"\r\n", b"\n")
    need(b"\r" not in raw, "Bare CR in completion source refused.")
    return hashlib.sha256(raw).hexdigest()


def verify_architecture_sources():
    for name, (filename, expected) in ARCHITECTURE_SOURCES.items():
        path = ROOT / "third_party" / "codeformer_completion" / filename
        need(architecture_file_sha256(path) == expected, "Completion architecture source changed: " + filename)
        module = sys.modules.get(name)
        need(module is None or Path(module.__file__).resolve() == path.resolve(), "Noncanonical completion architecture loaded.")


class CompletionAdapter:
    """Owned wrapper with unchanged R1 adapter math; no shared restoration hooks."""
    def __init__(self, net):
        self.net = net.eval().requires_grad_(False)
        self.raw512 = None
        self.input512 = None

    def __call__(self, image, mask, forward_event=None):
        import torch
        from torch.nn import functional as F
        need(image.ndim == 4 and image.shape[1:] == (3, 256, 256)
             and mask.shape == (image.shape[0], 1, 256, 256), "Expected NCHW256 RGB and N1HW mask.")
        need(torch.isfinite(image).all().item() and image.min().item() >= 0 and image.max().item() <= 1,
             "Expected finite RGB values in[0,1].")
        need(((mask == 0) | (mask == 1)).all().item(), "Expected a binary mask.")
        if (mask.mean((1, 2, 3)) >= .85).any().item():
            raise CompletionRefused("Too little visible face remains.")
        with torch.inference_mode():
            result = image.clone()
            active = mask.sum((1, 2, 3)) > 0
            if not active.any().item():
                return result
            visible = 1 - mask[active]
            support = F.interpolate(visible, size=(512, 512), mode="bilinear", align_corners=False)
            x = F.interpolate(image[active] * visible, size=(512, 512), mode="bilinear", align_corners=False)
            x = x / support.clamp_min(1e-8)
            m = F.interpolate(mask[active], size=(512, 512), mode="nearest")
            x = (x * (1 - m) + m) * 2 - 1
            self.input512 = x.detach().cpu().numpy().copy()
            if forward_event:
                forward_event("entered")
            raw = self.net(x, w=1, adain=False)[0]
            if forward_event:
                forward_event("returned")
            need(raw.shape == x.shape and torch.isfinite(raw).all().item(), "Invalid R1 generator output.")
            self.raw512 = raw.detach().cpu().numpy().copy()
            generated = ((raw + 1) / 2).clamp(0, 1)
            generated = F.interpolate(generated, size=image.shape[-2:], mode="bilinear", align_corners=False)
            result[active] = image[active] * (1 - mask[active]) + generated * mask[active]
            return result


def load_frozen_cpu_completion():
    verify_architecture_sources()
    checkpoint = assets.asset_path("completion_r1")
    import torch
    from third_party.codeformer_completion.codeformer_arch import CodeFormer
    verify_architecture_sources()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    need(type(state) is dict and set(state) == {"params_ema"} and isinstance(state["params_ema"], dict),
         "Completion asset must contain only params_ema.")
    params = state["params_ema"]
    need(len(params) == 497, "Expected complete497-tensor R1 state.")
    logical = hashlib.sha256()
    for name, value in sorted(params.items()):
        need(isinstance(name, str) and isinstance(value, torch.Tensor) and value.device.type == "cpu"
             and value.dtype == torch.float32 and torch.isfinite(value).all().item(), "Invalid completion state tensor.")
        value = value.detach().contiguous()
        logical.update(name.encode()); logical.update(str(tuple(value.shape)).encode())
        logical.update(str(value.dtype).encode()); logical.update(value.numpy().tobytes())
    need(logical.hexdigest() == LOGICAL_SHA256, "Completion logical state differs.")
    net = CodeFormer(dim_embd=512, codebook_size=512, n_head=8, n_layers=9, connect_list=["32", "64", "128"])
    net.load_state_dict(params, strict=True)
    adapter = CompletionAdapter(net.cpu().float())
    need(not net.training and not any(p.requires_grad for p in net.parameters()), "Completion model must remain frozen.")
    assets.asset_path("completion_r1")
    return adapter


def complete(rgb, mask, prepared_bounds=None, *, forward_event=None):
    """Return one plausible completion; no restoration, detection or extra mask margin."""
    started = time.monotonic()
    removal, bounds, visibility = validate_arrays(rgb, mask, prepared_bounds)
    contract = model_contract()
    if not removal.any():
        return {"output_rgb": rgb.copy(), "completed_rgb": rgb.astype(np.float32) / np.float32(255),
                "raw512": None, "input512": None, "metadata": {**contract, "bypassed": True,
                "neural_calls": 0, "checkpoint_loaded": False, "prepared_bounds": bounds,
                "visibility": visibility, "seconds": time.monotonic() - started}}
    import torch
    torch.set_num_threads(4)
    adapter = load_frozen_cpu_completion()
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255
    tensor_mask = torch.from_numpy(removal.copy())[None, None].float()
    calls = {"entered": 0, "returned": 0}

    def counted_event(event):
        calls[event] += 1
        need(calls[event] == 1, "One completion forward maximum.")
        if forward_event:
            forward_event(event)

    result = adapter(image, tensor_mask, forward_event=counted_event)
    need(result.shape == image.shape and result.dtype == torch.float32 and torch.isfinite(result).all().item(),
         "Invalid R1 completion result.")
    outside = (tensor_mask == 0).expand_as(image)
    need(torch.equal(result[outside], image[outside]), "R1 changed supplied canvas outside the reviewed mask.")
    need(calls == {"entered": 1, "returned": 1}, "Exactly one successful completion forward required.")
    completed = result[0].permute(1, 2, 0).detach().cpu().numpy().copy()
    output = encode_output(rgb, removal, completed)
    need(np.array_equal(output[removal == 0], rgb[removal == 0]), "Completion changed outside-mask RGB bytes.")
    return {"output_rgb": output, "completed_rgb": completed, "raw512": adapter.raw512,
            "input512": adapter.input512, "metadata": {**contract, "bypassed": False, "neural_calls": 1,
            "checkpoint_loaded": True, "prepared_bounds": bounds, "visibility": visibility,
            "seconds": time.monotonic() - started, "exact_outside_reviewed_mask": True,
            "preservation_reference": "The exact supplied prepared/restored RGB256 canvas.",
            "png_conversion": "clamp[0,1], multiply255, np.rint, uint8; supplied bytes recopied outside reviewed support."}}
