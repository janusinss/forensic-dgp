"""Author-provided CelebA-HQ AOT-GAN inference with reviewed removal masks."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from completion import validate_mask

SOURCE_REVISION = "2cd1afd8fdfabb101c678f6062d14bc7d302509e"
WEIGHTS_SHA256 = "5cfbf8e545e75adc1be66682741d22b4e5288f734971cc896b795e7e137f42f9"
WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=1T7Xkv09pvf6gy-R2Cn0RN5V-Pq0vSFPw"


class AOTCompletion(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, image, mask):
        validate_mask(image, mask)
        if min(image.shape[-2:]) < 32 or not image.is_floating_point():
            raise ValueError("Expected floating RGB of at least 32 pixels")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Expected finite RGB in [0,1]")
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Use a binary explicit removal mask")
        if (mask.mean((1, 2, 3)) >= .85).any():
            raise ValueError("Too little visible image remains; request a less-covered face")
        result = image.clone()
        active = mask.sum((1, 2, 3)) > 0
        if not active.any():
            return result
        m = mask[active]
        visible = 1 - m
        support = F.interpolate(visible, (512, 512), mode="bilinear", align_corners=False)
        x = F.interpolate(image[active] * visible, (512, 512), mode="bilinear", align_corners=False)
        x = (x / support.clamp_min(1e-8)).clamp(0, 1) * 2 - 1
        resized_mask = F.interpolate(m, (512, 512), mode="nearest")
        # Official test path: white fill in normalized RGB, plus an explicit mask.
        network_image = x * (1 - resized_mask) + resized_mask
        predicted = self.net(network_image, resized_mask)
        if predicted.shape != network_image.shape or not torch.isfinite(predicted).all():
            raise FloatingPointError("Invalid AOT-GAN output")
        generated = F.interpolate(((predicted + 1) / 2).clamp(0, 1), image.shape[-2:], mode="bilinear", align_corners=False)
        result[active] = torch.where(m.bool(), generated, image[active])
        return result


def load_aot(path, device="cpu"):
    path = Path(path)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError("AOT-GAN checkpoint SHA256 mismatch")
    from third_party.aot_gan.aotgan import InpaintGenerator
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = InpaintGenerator(SimpleNamespace(rates=[1, 2, 4, 8], block_num=8))
    net.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in net.state_dict().values()):
        raise ValueError("Non-finite AOT-GAN weights")
    model = AOTCompletion(net).to(device).eval()
    return model, {"backend": "aot-gan-celebahq", "source_revision": SOURCE_REVISION,
                   "weights_url": WEIGHTS_URL, "weights_sha256": digest,
                   "checksum_scope": "Pinned observed acquisition fingerprint, not a publisher-supplied checksum",
                   "mask_conditioning": "White-filled normalized RGB plus explicit binary mask",
                   "input_policy": "visible-normalized-resize512-v1", "compositing_policy": "mask-only-v1",
                   "face_specific_pretraining": True}
