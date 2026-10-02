"""Verified Big-LaMa inference with explicit binary masks and visible preservation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from completion import validate_mask

SOURCE_REVISION = "786f5936b27fb3dacd2b1ad799e4de968ea697e7"
ARCHIVE_SHA256 = "f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6"
ORIGINAL_SHA256 = "fccb7adffd53ec0974ee5503c3731c2c2f1e7e07856fd9228cdcc0b46fd5d423"
GENERATOR_SHA256 = "54fbd7b0b7eaee1ad6c90ae3fe1f660f8fd50c9c8a88eba37d589be24f68f05a"


class LaMaCompletion(torch.nn.Module):
    """RGB [0,1], N1HW mask=1 for removal; no facial identity guarantee."""

    def __init__(self, net):
        super().__init__()
        self.net = net.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, image, mask):
        validate_mask(image, mask)
        if min(image.shape[-2:]) < 32 or not image.is_floating_point():
            raise ValueError("Expected floating RGB images of at least 32 pixels")
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
        x, m = image[active], mask[active]
        h, w = image.shape[-2:]
        padding = (0, (-w) % 32, 0, (-h) % 32)
        # Official tensor inference uses reflection padding to a network modulo.
        # At the frozen 256px benchmark size this is a no-op. No image resize.
        if any(padding):
            x, m = F.pad(x, padding, mode="reflect"), F.pad(m, padding, mode="reflect")
        network_input = torch.cat((x * (1 - m), m), dim=1)
        generated = self.net(network_input)
        if generated.shape != x.shape or not torch.isfinite(generated).all():
            raise FloatingPointError("Invalid LaMa generator output")
        generated = generated[:, :, :h, :w].clamp(0, 1)
        result[active] = torch.where(mask[active].bool(), generated, image[active])
        return result


def load_lama(path, device="cpu"):
    path = Path(path)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != GENERATOR_SHA256:
        raise ValueError("Prepared LaMa generator SHA256 mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("format") != "dgp-lama-inference-v1" or
            state.get("source_revision") != SOURCE_REVISION or
            state.get("original_checkpoint_sha256") != ORIGINAL_SHA256 or
            state.get("archive_sha256") != ARCHIVE_SHA256):
        raise ValueError("LaMa generator provenance mismatch")
    from third_party.lama.ffc import FFCResNetGenerator
    # Resolved generator section of the verified official big-lama/config.yaml.
    net = FFCResNetGenerator(
        input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18,
        add_out_act="sigmoid",
        init_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        downsample_conv_kwargs={"ratio_gin": 0, "ratio_gout": 0, "enable_lfu": False},
        resnet_conv_kwargs={"ratio_gin": .75, "ratio_gout": .75, "enable_lfu": False},
    )
    net.load_state_dict(state["generator"], strict=True)
    if not all(torch.isfinite(value).all() for value in net.state_dict().values()):
        raise ValueError("Non-finite LaMa generator state")
    model = LaMaCompletion(net).to(device).eval()
    return model, {
        "backend": "big-lama-places2", "source_revision": SOURCE_REVISION,
        "archive_sha256": ARCHIVE_SHA256, "original_checkpoint_sha256": ORIGINAL_SHA256,
        "generator_sha256": digest, "mask_conditioning": "RGB*(1-mask) concatenated with binary removal mask",
        "input_policy": "native resolution; modulo32 reflection padding; no resize",
        "compositing_policy": "mask-only-v1", "face_specific_pretraining": False,
    }
