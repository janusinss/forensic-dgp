"""Stage-III-inspired supervised adaptation, separate from the frozen benchmark.

CodeFormer architecture is unchanged and credited in third_party/codeformer.
Only a manually invoked, VM-guarded worker may call backward/optimizer methods.
These functions themselves neither run an optimizer nor change app defaults.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .common import require, sha256, tensor_state_digest

INITIALIZER_SHA256 = "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7"
TEACHER_SHA256 = "4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f"
TRAINABLE_GROUPS = ("encoder", "position_emb", "feat_emb", "ft_layers",
                    "idx_pred_layer", "fuse_convs_dict")
FROZEN_GROUPS = ("quantize", "generator")
LOSS_WEIGHTS = {"token_ce": 0.5, "latent_mse": 1.0,
                "pixel_l1_rgb": 1.0, "lpips_vgg": 1.0}


def parameter_group(name):
    prefix = name.split(".", 1)[0]
    require(prefix in TRAINABLE_GROUPS + FROZEN_GROUPS,
            "Undeclared CodeFormer parameter: " + name)
    return prefix


def parameter_census(net):
    groups = {}
    for name, parameter in net.named_parameters():
        group = parameter_group(name)
        expected = group in TRAINABLE_GROUPS
        require(parameter.requires_grad == expected,
                "Trainability differs: " + name)
        row = groups.setdefault(group, {"tensors": 0, "parameters": 0,
                                        "trainable": expected})
        row["tensors"] += 1
        row["parameters"] += parameter.numel()
    require(set(groups) == set(TRAINABLE_GROUPS + FROZEN_GROUPS),
            "Missing declared parameter group")
    return {"groups": groups,
            "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in net.parameters() if not p.requires_grad),
            "trainable_names": [n for n, p in net.named_parameters() if p.requires_grad],
            "frozen_names": [n for n, p in net.named_parameters() if not p.requires_grad]}


def _load_state(path, expected_sha):
    require(sha256(path) == expected_sha, "Checkpoint file fingerprint differs")
    loaded = torch.load(Path(path), map_location="cpu", weights_only=True)
    require(isinstance(loaded, dict) and "params_ema" in loaded,
            "Expected the official params_ema state")
    state = loaded["params_ema"]
    require(all(isinstance(x, torch.Tensor) and torch.isfinite(x).all().item()
                for x in state.values()), "Nonfinite or nontensor initializer")
    return state


def load_trainable(path, device="cpu"):
    from third_party.codeformer.codeformer_arch import CodeFormer
    state = _load_state(path, INITIALIZER_SHA256)
    net = CodeFormer(dim_embd=512, n_head=8, n_layers=9, codebook_size=1024,
                     connect_list=["32", "64", "128", "256"],
                     fix_modules=["quantize", "generator"])
    net.load_state_dict(state, strict=True)
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(parameter_group(name) in TRAINABLE_GROUPS)
    census = parameter_census(net)
    return net.to(device).eval(), census


def load_teacher(path, device="cpu"):
    from third_party.codeformer.vqgan_arch import VQAutoEncoder
    state = _load_state(path, TEACHER_SHA256)
    net = VQAutoEncoder(img_size=512, nf=64, ch_mult=[1, 2, 2, 4, 4, 8],
                        quantizer="nearest", res_blocks=2,
                        attn_resolutions=[16], codebook_size=1024)
    net.load_state_dict(state, strict=True)
    return net.to(device).eval().requires_grad_(False)


def assert_shared_codebook(net, teacher):
    """Verify ordered dictionary compatibility, not falsely claim byte equality.

    The two fingerprinted official release assets have tiny floating-point
    differences. Their 1024 dictionary rows retain identical nearest indices.
    Labels come from the HQ teacher, then feature targets use the restoration
    dictionary at those indices, as in the upstream joint model. Both original
    files stay immutable; the failed exact-equality assumption is retained.
    This compatibility bound is unrelated to a restoration quality threshold.
    """
    restoration = net.quantize.embedding.weight.detach().cpu().float()
    hq = teacher.quantize.embedding.weight.detach().cpu().float()
    require(restoration.shape == hq.shape == (1024, 256), "Dictionary geometry differs")
    distances = torch.cdist(restoration, hq)
    ordered = torch.arange(1024)
    nearest = distances.argmin(1)
    difference = float((restoration - hq).abs().max())
    require(difference <= 1e-4 and torch.equal(nearest, ordered),
            "Official teacher/restorer token dictionary order is incompatible")
    return {"exact_tensor_equality": torch.equal(restoration, hq),
            "maximum_absolute_difference": difference,
            "same_index_nearest_rows": int((nearest == ordered).sum()),
            "feature_target_policy": "HQ teacher indices; frozen restoration dictionary features"}


def state_digest(module):
    return tensor_state_digest(module.state_dict())


def frozen_state_digest(net):
    return tensor_state_digest({n: p for n, p in net.state_dict().items()
                                if n.split(".", 1)[0] in FROZEN_GROUPS})


def _rgb(image, size):
    require(image.ndim == 4 and image.shape[1:] == (3, size, size) and len(image) > 0,
            "Expected nonempty RGB NCHW %dx%d" % (size, size))
    require(torch.isfinite(image).all().item() and image.min().item() >= 0
            and image.max().item() <= 1, "Expected finite RGB in [0,1]")


def prepare_input(input256):
    _rgb(input256, 256)
    return F.interpolate(input256, size=(512, 512), mode="bilinear",
                         align_corners=False) * 2 - 1


def delivered_rgb(output512):
    require(output512.ndim == 4 and output512.shape[1:] == (3, 512, 512),
            "Invalid restored geometry")
    require(torch.isfinite(output512).all().item(), "Nonfinite model output")
    return F.interpolate(((output512.float() + 1) / 2).clamp(0, 1),
                         size=(256, 256), mode="bilinear", align_corners=False)


def infer(net, input256):
    """Return native512 prediction; caller chooses inference_mode for evaluation."""
    return net(prepare_input(input256), w=1.0, detach_16=True, adain=True)[0]


@dataclass(frozen=True)
class CleanCodes:
    indices: torch.Tensor
    features: torch.Tensor


@torch.no_grad()
def hq_targets(teacher, clean512, codebook=None):
    _rgb(clean512, 512)
    require(not teacher.training and not any(p.requires_grad for p in teacher.parameters()),
            "The clean token teacher must be frozen and in eval mode")
    # FP32 nearest-code distances must not inherit the restoration BF16 context.
    with torch.autocast(device_type=clean512.device.type, enabled=False):
        encoded = teacher.encoder(clean512.float() * 2 - 1)
        _, _, stats = teacher.quantize(encoded)
        indices = stats["min_encoding_indices"].view(len(clean512), -1)
        target_codebook = teacher.quantize if codebook is None else codebook
        require(not any(p.requires_grad for p in target_codebook.parameters()),
                "Target dictionary must remain frozen")
        features = target_codebook.get_codebook_feat(
            indices, shape=[len(clean512), 16, 16, 256])
    require(indices.shape[1] == 256 and features.shape[1:] == (256, 16, 16),
            "Clean code geometry differs")
    require(torch.isfinite(features).all().item(), "Nonfinite HQ code features")
    return CleanCodes(indices.detach(), features.detach())


def forward_training_terms(net, teacher, perceptual, input256, clean512,
                           image_eligible=True, codes=None):
    """No backward call here. The worker guards all actual gradient execution.

    Prediction and clean target use native512 supervision. Pixel L1 is in RGB
    [0,1] units without clipping predictions. LPIPS v0.1 takes raw [-1,1].
    This is a declared adaptation, not a reproduction of upstream loss scaling.
    """
    _rgb(clean512, 512)
    codes = hq_targets(teacher, clean512, codebook=net.quantize) if codes is None else codes
    output, logits, feature = net(prepare_input(input256), w=1.0,
                                  detach_16=True, adain=True)
    require(torch.isfinite(logits).all().item() and torch.isfinite(output).all().item(),
            "Nonfinite restoration forward")
    terms = {
        "token_ce": F.cross_entropy(logits.float().transpose(1, 2), codes.indices),
        "latent_mse": F.mse_loss(feature.float(), codes.features.float()),
    }
    if image_eligible:
        terms["pixel_l1_rgb"] = F.l1_loss((output.float() + 1) / 2, clean512.float())
        # The fixed perceptual weights stay frozen; its prediction forward must
        # remain differentiable so image supervision reaches restoration paths.
        with torch.autocast(device_type=output.device.type, enabled=False):
            terms["lpips_vgg"] = perceptual(output.float(), clean512.float() * 2 - 1)
    total = sum(LOSS_WEIGHTS[name] * value for name, value in terms.items())
    require(torch.isfinite(total).item(), "Nonfinite adaptation objective")
    return total, terms, output
