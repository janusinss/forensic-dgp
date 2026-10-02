"""Separate official CodeFormer restoration weights; no inpainting substitution."""
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

REVISION = "b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b"
WEIGHTS_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
WEIGHTS_SHA256 = "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7"


class CodeFormerRestoration(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, image, fidelity=1.0):
        if image.ndim != 4 or image.shape[1] != 3 or not len(image):
            raise ValueError("Expected a nonempty batch of RGB crops")
        if image.shape[-2] != image.shape[-1] or image.shape[-1] < 32:
            raise ValueError("Use aligned square face crops of at least 32 pixels")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Expected finite RGB values in [0,1]")
        if not 0 <= fidelity <= 1:
            raise ValueError("Fidelity must be in [0,1]")
        x = F.interpolate(image, size=(512, 512), mode="bilinear", align_corners=False) * 2 - 1
        output = self.net(x, w=fidelity, adain=True)[0]
        if output.shape != x.shape or not torch.isfinite(output).all():
            raise FloatingPointError("Invalid pretrained restoration output")
        output = ((output + 1) / 2).clamp(0, 1)
        return F.interpolate(output, size=image.shape[-2:], mode="bilinear", align_corners=False)


def load_face_restorer(path, device="cpu"):
    path = Path(path)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError("CodeFormer restoration checkpoint fingerprint differs")
    from third_party.codeformer.codeformer_arch import CodeFormer
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                     connect_list=["32", "64", "128", "256"])
    net.load_state_dict(state["params_ema"], strict=True)
    if not all(torch.isfinite(value).all() for value in net.state_dict().values()):
        raise ValueError("Non-finite restoration weights")
    model = CodeFormerRestoration(net).to(device).eval()
    provenance = {"backend": "codeformer-restoration", "source_revision": REVISION,
                  "weights_url": WEIGHTS_URL, "weights_sha256": digest,
                  "checksum_scope": "Observed acquisition fingerprint, not a publisher-supplied checksum",
                  "codebook_size": 1024, "input_policy": "RGB bilinear512 normalized [-1,1]",
                  "adain": True, "output_policy": "Bilinear back to input dimensions",
                  "license": "S-Lab License 1.0; retained in third_party/codeformer/LICENSE"}
    return model, provenance
