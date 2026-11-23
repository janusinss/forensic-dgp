"""CodeFormer Restoration V3 — 3-Stage Resumable L4 Trainer & Pareto Model Soup.

Stages:
  --stage 1    Photometric, Degradation, Codebook & Clean-SFT alignment (stage1_only=True, EMA 0.999)
  --stage 2    End-to-end Multi-Objective Fidelity + Identity + Sharpness (EMA 0.999, w_arcface=0.35)
  --stage 3    Light Degraded-Only PatchGAN-R1 Texture Polish (EMA 0.9995, w_arcface=0.45, lambda_gan=0.01)
  --stage soup Pareto Model Soup over alpha in {0.0, 0.25, 0.5, 0.75, 1.0} between Stage-2 & Stage-3 EMA

Features:
- `torch.amp.autocast('cuda', dtype=torch.bfloat16)` under `--bf16` on CUDA (clean FP32 fallback on CPU)
- Zero-leakage split guard via `datasets_v3.ZeroLeakageSplitGuard`
- Stratified balanced batches via `StratifiedRoleBatchSampler`
- Gradient accumulation, gradient clipping (1.0), EMA, atomic checkpoints (`latest.pt`,
  `best_ema.pt`, `v3_modules_ema.npz`), and structured `journal.jsonl` logging.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from research.codeformer_restoration_v3.architecture_v3 import (
    CodeFormerV3Restorer,
    V3FreqHead,
    detect_v3_npz_variant,
    export_v3_npz,
    load_v3_npz,
)
from research.codeformer_restoration_v3.datasets_v3 import (
    V3RestorationDataset,
    ZeroLeakageSplitGuard,
    make_stratified_dataloader,
)
from research.codeformer_restoration_v3.eval_v3 import (
    _load_canvas_and_observed,
    evaluate_runner_on_cases,
    evaluate_v3_on_manifest,
    load_dev_native_cctv,
    load_dev_syn,
    make_v3_runner,
)
from research.codeformer_restoration_v3.losses_v3 import (
    DifferentiableArcFaceLoss,
    PatchGANDiscriminator,
    Stage1Loss,
    Stage2Loss,
    Stage3GANHelper,
    V34Stage1Loss,
    V34Stage2Loss,
    load_frozen_lpips,
)

ROOT = Path(__file__).resolve().parents[2]
SOUP_ALPHAS: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_BLUR_ROLES: Tuple[str, ...] = (
    "global_defocus_blur",
    "directional_motion_blur",
    "compound_cctv_codec",
    "severe_lr_survface",
    "joint_lowlight_blur_codec",
)


# ---------------------------------------------------------------------------
# Atomic I/O & Structured JSONL Journal
# ---------------------------------------------------------------------------


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_export_v3_npz(model: CodeFormerV3Restorer, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    sha = export_v3_npz(model, tmp)
    os.replace(tmp, path)
    return sha


def append_journal(journal_path: Path, record: Dict[str, Any]) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, allow_nan=False) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# EMA Helper over model.v3
# ---------------------------------------------------------------------------


class V3EMA:
    """FP32 Exponential Moving Average shadow copy of `model.v3` state dict."""

    def __init__(self, model: CodeFormerV3Restorer, decay: float = 0.999):
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().float() for k, v in model.v3.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: CodeFormerV3Restorer) -> None:
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for k, v in model.v3.state_dict().items():
            src = v.detach().float()
            if k not in self.shadow:
                self.shadow[k] = src.clone()
            else:
                if self.shadow[k].device != src.device:
                    self.shadow[k] = self.shadow[k].to(src.device)
                self.shadow[k].mul_(d).add_(src, alpha=1.0 - d)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, sd: Mapping[str, Any], device: Union[str, torch.device]) -> None:
        self.decay = float(sd.get("decay", self.decay))
        self.num_updates = int(sd.get("num_updates", 0))
        self.shadow = {k: v.to(device=device, dtype=torch.float32).clone() for k, v in sd["shadow"].items()}

    def copy_to(self, model: CodeFormerV3Restorer) -> Dict[str, torch.Tensor]:
        """Swap EMA weights into `model.v3` and return backup of original weights."""
        backup = {k: v.detach().clone() for k, v in model.v3.state_dict().items()}
        target_dev = next(model.v3.parameters()).device
        ema_sd = {k: v.to(device=target_dev, dtype=backup[k].dtype) for k, v in self.shadow.items()}
        model.v3.load_state_dict(ema_sd, strict=True)
        return backup

    @staticmethod
    def restore(model: CodeFormerV3Restorer, backup: Mapping[str, torch.Tensor]) -> None:
        model.v3.load_state_dict(backup, strict=True)


# ---------------------------------------------------------------------------
# Stage Configuration, Freezing & Optimizer / Scheduler Builders
# ---------------------------------------------------------------------------


def reset_freq_head(model: CodeFormerV3Restorer) -> None:
    """Re-initialize model.v3['freq_head'] fresh matching model.arch_variant."""
    ref_param = next(model.v3["freq_head"].parameters())
    fresh = V3FreqHead(arch_variant=model.arch_variant).to(device=ref_param.device, dtype=ref_param.dtype)
    model.v3["freq_head"].load_state_dict(fresh.state_dict(), strict=True)


def configure_stage_trainability(model: CodeFormerV3Restorer, stage: str) -> None:
    """Enforce exact frozen/trainable module boundaries per stage (#333, #387, #407, #450)."""
    # Official CodeFormer base is ALWAYS 100% frozen
    for p in model.base.parameters():
        p.requires_grad = False
    model.base.eval()

    if stage in ("3b", "3c", "3d"):
        # V3.1 Stage-3B: pyramid_sft, freq_head, output_composer, deg_encoder.w_heads
        # V3.2 Stage-3C / V3.3 Stage-3D: + deg_encoder.alpha_adain_head
        for p in model.v3.parameters():
            p.requires_grad = False
        for name in ("pyramid_sft", "freq_head", "output_composer"):
            for p in model.v3[name].parameters():
                p.requires_grad = True
        for p in model.v3["deg_encoder"].w_heads.parameters():
            p.requires_grad = True
        if stage in ("3c", "3d"):
            for p in model.v3["deg_encoder"].alpha_adain_head.parameters():
                p.requires_grad = True
        return

    if stage == "1":
        trainable = {"deg_encoder", "photo_net", "transformer_lora", "pyramid_sft"}
    elif stage in ("2", "2b", "2c", "2d", "1v4", "2v4"):
        trainable = set(model.v3.keys())
    elif stage == "3":
        trainable = {"freq_head", "output_composer"}
    else:
        trainable = set()

    for name, mod in model.v3.items():
        req = name in trainable
        for p in mod.parameters():
            p.requires_grad = req


def _make_warmup_cosine_lambda(warmup_steps: int, total_steps: int, min_ratio: float):
    w_steps = max(0, int(warmup_steps))
    tot = max(1, int(total_steps))
    mr = float(min_ratio)

    def _lr_lambda(step: int) -> float:
        if w_steps > 0 and step < w_steps:
            return 0.1 + 0.9 * (float(step) / float(w_steps))
        if tot <= w_steps:
            return 1.0
        progress = min(1.0, float(step - w_steps) / float(max(1, tot - w_steps)))
        return mr + 0.5 * (1.0 - mr) * (1.0 + math.cos(math.pi * progress))

    return _lr_lambda


def _dedup_param_groups(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate parameters by id(p) and assert zero overlap across optimizer groups."""
    seen: set[int] = set()
    cleaned: List[Dict[str, Any]] = []
    for g in groups:
        unique_params = []
        group_ids: set[int] = set()
        for p in g["params"]:
            pid = id(p)
            if pid in group_ids:
                continue
            if pid in seen:
                raise AssertionError(f"Duplicate parameter across optimizer param groups: id={pid}")
            group_ids.add(pid)
            seen.add(pid)
            unique_params.append(p)
        if unique_params:
            cleaned.append({**g, "params": unique_params})
    return cleaned


def build_optimizer_and_scheduler(
    model: CodeFormerV3Restorer,
    stage: str,
    max_steps: int,
    lr_override: Optional[float] = None,
    warmup_steps: int = 0,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    use_ratio_cosine = False
    min_ratio = 0.1
    group_min_ratios: Optional[List[float]] = None
    lr_min = 1e-5

    if stage == "1":
        lr_max = lr_override or 3e-4
        lr_min = lr_max * 0.1
        groups = [{"params": [p for p in model.v3.parameters() if p.requires_grad], "lr": lr_max}]
    elif stage == "2":
        lr_hi = lr_override or 1.5e-4
        lr_lo = lr_hi / 3.0
        lr_min = 1e-5
        lo_names = {"deg_encoder", "photo_net", "transformer_lora"}
        hi_params, lo_params = [], []
        for name, mod in model.v3.items():
            target = lo_params if name in lo_names else hi_params
            target.extend([p for p in mod.parameters() if p.requires_grad])
        groups = []
        if hi_params:
            groups.append({"params": hi_params, "lr": lr_hi})
        if lo_params:
            groups.append({"params": lo_params, "lr": lr_lo})
    elif stage == "2b":
        # V3.1 Stage-2B: hi (pyramid_sft, freq_head, output_composer, deg_encoder.w_heads) 1.2e-4 -> 1.5e-5;
        #                lo (deg_encoder non-w_heads, photo_net, transformer_lora) 4.0e-5 -> 5.0e-6.
        lr_hi = lr_override or 1.2e-4
        lr_lo = (lr_override / 3.0) if lr_override is not None else 4.0e-5
        min_ratio = 1.5e-5 / 1.2e-4  # exact 0.125 for both groups
        use_ratio_cosine = True

        w_head_ids = {id(p) for p in model.v3["deg_encoder"].w_heads.parameters()}
        hi_params, lo_params = [], []
        for name in ("pyramid_sft", "freq_head", "output_composer"):
            hi_params.extend([p for p in model.v3[name].parameters() if p.requires_grad])
        hi_params.extend([p for p in model.v3["deg_encoder"].w_heads.parameters() if p.requires_grad])
        for p in model.v3["deg_encoder"].parameters():
            if p.requires_grad and id(p) not in w_head_ids:
                lo_params.append(p)
        for name in ("photo_net", "transformer_lora"):
            lo_params.extend([p for p in model.v3[name].parameters() if p.requires_grad])
        groups = [
            {"params": hi_params, "lr": lr_hi},
            {"params": lo_params, "lr": lr_lo},
        ]
    elif stage in ("2c", "2d"):
        # V3.2 Stage-2C / V3.3 Stage-2D:
        #   hi (freq_head, pyramid_sft, output_composer, deg_encoder.w_heads,
        #       deg_encoder.alpha_adain_head) 1.5e-4 -> 1.5e-5;
        #   lo (deg_encoder rest, photo_net, [transformer_lora in v30]) 3.0e-5 -> 4.0e-6.
        lr_hi = lr_override or 1.5e-4
        lr_lo = (lr_override * 0.2) if lr_override is not None else 3.0e-5
        group_min_ratios = [1.5e-5 / 1.5e-4, 4.0e-6 / 3.0e-5]
        use_ratio_cosine = True

        hi_deg_ids = (
            {id(p) for p in model.v3["deg_encoder"].w_heads.parameters()}
            | {id(p) for p in model.v3["deg_encoder"].alpha_adain_head.parameters()}
        )
        hi_params, lo_params = [], []
        for name in ("freq_head", "pyramid_sft", "output_composer"):
            hi_params.extend([p for p in model.v3[name].parameters() if p.requires_grad])
        hi_params.extend([p for p in model.v3["deg_encoder"].w_heads.parameters() if p.requires_grad])
        hi_params.extend([p for p in model.v3["deg_encoder"].alpha_adain_head.parameters() if p.requires_grad])
        for p in model.v3["deg_encoder"].parameters():
            if p.requires_grad and id(p) not in hi_deg_ids:
                lo_params.append(p)
        for name in ("photo_net", "transformer_lora"):
            if name in model.v3:
                lo_params.extend([p for p in model.v3[name].parameters() if p.requires_grad])
        groups = [
            {"params": hi_params, "lr": lr_hi},
            {"params": lo_params, "lr": lr_lo},
        ]
    elif stage == "3":
        lr_max = lr_override or 2e-5
        lr_min = lr_max * 0.1
        groups = [{"params": [p for p in model.v3.parameters() if p.requires_grad], "lr": lr_max}]
    elif stage == "3b":
        # V3.1 Stage-3B: hi (freq_head, output_composer, deg_encoder.w_heads) 5.0e-5 -> 5.0e-6;
        #                mid (pyramid_sft) 3.5e-5 -> 3.5e-6.
        lr_hi = lr_override or 5.0e-5
        lr_mid = (lr_override * 0.7) if lr_override is not None else 3.5e-5
        min_ratio = 0.1  # exact 5.0e-6 and 3.5e-6
        use_ratio_cosine = True

        hi_params = []
        for name in ("freq_head", "output_composer"):
            hi_params.extend([p for p in model.v3[name].parameters() if p.requires_grad])
        hi_params.extend([p for p in model.v3["deg_encoder"].w_heads.parameters() if p.requires_grad])
        mid_params = [p for p in model.v3["pyramid_sft"].parameters() if p.requires_grad]
        groups = [
            {"params": hi_params, "lr": lr_hi},
            {"params": mid_params, "lr": lr_mid},
        ]
    elif stage in ("3c", "3d"):
        # V3.2 Stage-3C / V3.3 Stage-3D:
        #   hi (freq_head, output_composer, deg_encoder.w_heads,
        #       deg_encoder.alpha_adain_head) 5.0e-5 -> 5.0e-6;
        #   mid (pyramid_sft) 3.0e-5 -> 3.0e-6.
        lr_hi = lr_override or 5.0e-5
        lr_mid = (lr_override * 0.6) if lr_override is not None else 3.0e-5
        min_ratio = 0.1  # exact 5.0e-6 and 3.0e-6
        use_ratio_cosine = True

        hi_params = []
        for name in ("freq_head", "output_composer"):
            hi_params.extend([p for p in model.v3[name].parameters() if p.requires_grad])
        hi_params.extend([p for p in model.v3["deg_encoder"].w_heads.parameters() if p.requires_grad])
        hi_params.extend([p for p in model.v3["deg_encoder"].alpha_adain_head.parameters() if p.requires_grad])
        mid_params = [p for p in model.v3["pyramid_sft"].parameters() if p.requires_grad]
        groups = [
            {"params": hi_params, "lr": lr_hi},
            {"params": mid_params, "lr": lr_mid},
        ]
    elif stage == "1v4":
        # V3.4 Stage-1v4: all 3 front-end modules 3.0e-4 -> 3.0e-5
        lr_max = lr_override or 3.0e-4
        min_ratio = 0.1
        use_ratio_cosine = True
        groups = [{"params": [p for p in model.v3.parameters() if p.requires_grad], "lr": lr_max}]
    elif stage == "2v4":
        # V3.4 Stage-2v4:
        #   pre/photo/deg_encoder rest: 1.2e-4 -> 1.2e-5
        #   deg_encoder.{w_head, clean_head}: 2.0e-4 -> 2.0e-5
        lr_pre = lr_override or 1.2e-4
        lr_wc = (lr_override * (2.0 / 1.2)) if lr_override is not None else 2.0e-4
        min_ratio = 0.1
        use_ratio_cosine = True
        deg_enc = model.v3["deg_encoder"]
        wc_ids = {id(p) for p in deg_enc.w_head.parameters()} | {id(p) for p in deg_enc.clean_head.parameters()}
        wc_params = [p for p in deg_enc.parameters() if p.requires_grad and id(p) in wc_ids]
        pre_params = [p for p in deg_enc.parameters() if p.requires_grad and id(p) not in wc_ids]
        for name in ("photo_net", "pre_net"):
            if name in model.v3:
                pre_params.extend([p for p in model.v3[name].parameters() if p.requires_grad])
        groups = [
            {"params": pre_params, "lr": lr_pre},
            {"params": wc_params, "lr": lr_wc},
        ]
    else:
        raise ValueError(f"No optimizer for stage={stage}")

    groups = _dedup_param_groups(groups)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=1e-4)
    w_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(max_steps))
    if use_ratio_cosine:
        if group_min_ratios is not None and len(group_min_ratios) == len(groups):
            lambdas = [_make_warmup_cosine_lambda(w_steps, total_steps, mr) for mr in group_min_ratios]
            sched: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambdas)
        else:
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=_make_warmup_cosine_lambda(w_steps, total_steps, min_ratio)
            )
    elif w_steps > 0 and total_steps > w_steps:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=w_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps - w_steps, eta_min=lr_min
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[w_steps]
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_min)
    return opt, sched


@torch.no_grad()
def evaluate_stage1_exit_gates(
    model: CodeFormerV3Restorer,
    dev_manifest: Path,
    device: torch.device,
    max_cases: Optional[int] = None,
) -> Dict[str, float]:
    """Compute Stage-1 exit-gate metrics on DEV-Syn manifest using current (EMA) weights."""
    from PIL import Image
    from torch.nn import functional as F

    cases = load_dev_syn(dev_manifest)
    if max_cases is not None:
        cases = cases[: int(max_cases)]

    was_training = model.training
    model.eval()
    deg_v3_accs: List[float] = []
    deg_cf_accs: List[float] = []
    lowlight_maes: List[float] = []
    clear_drifts: List[float] = []

    try:
        for case in cases:
            if not case.get("paired") or not case.get("target_path"):
                continue
            canvas_u8, obs = _load_canvas_and_observed(case)
            with Image.open(case["target_path"]) as tm:
                gt_u8 = np.asarray(tm.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8)

            x_in_256 = torch.from_numpy(canvas_u8.transpose(2, 0, 1).copy()).float()[None].to(device) / 255.0
            gt_256 = torch.from_numpy(gt_u8.transpose(2, 0, 1).copy()).float()[None].to(device) / 255.0
            obs_t = torch.from_numpy(obs.astype(np.float32))[None, None].to(device)
            gt_512 = F.interpolate(gt_256, size=(512, 512), mode="bilinear", align_corners=False)
            in_512 = F.interpolate(x_in_256, size=(512, 512), mode="bilinear", align_corners=False)

            gt_teacher = model.encode_gt_teacher(gt_512, observed_mask=obs_t)
            cf_base = model.encode_gt_teacher(in_512, observed_mask=obs_t)
            v3_out = model(x_in_256, obs_t, stage1_only=True)

            gt_idx = gt_teacher["code_indices_gt"]
            cf_idx = cf_base["code_indices_gt"]
            v3_idx = v3_out["code_logits"].argmax(dim=2)
            x_photo = v3_out["x_photo_256"].float()

            role = str(case.get("role", ""))
            denom_rgb = float(obs_t.sum().clamp_min(1.0).item()) * 3.0
            if role == "clear":
                drift = float(((x_photo - x_in_256).abs() * obs_t).sum().item() / denom_rgb)
                clear_drifts.append(drift)
            else:
                deg_v3_accs.append(float((v3_idx == gt_idx).float().mean().item()))
                deg_cf_accs.append(float((cf_idx == gt_idx).float().mean().item()))
                if "lowlight" in role:
                    mae = float(((x_photo - gt_256).abs() * obs_t).sum().item() / denom_rgb)
                    lowlight_maes.append(mae)
    finally:
        if was_training:
            model.train(True)

    v3_acc = float(np.mean(deg_v3_accs)) if deg_v3_accs else 0.0
    cf_acc = float(np.mean(deg_cf_accs)) if deg_cf_accs else 0.0
    ll_mae = float(np.mean(lowlight_maes)) if lowlight_maes else 0.0
    cl_drift = float(np.mean(clear_drifts)) if clear_drifts else 0.0
    return {
        "degraded_code_top1_acc": v3_acc,
        "cf_base_code_top1_acc": cf_acc,
        "delta_top1_acc": v3_acc - cf_acc,
        "lowlight_photo_mae": ll_mae,
        "clear_photo_drift": cl_drift,
        "cases_evaluated": len(cases),
    }


def _autocast_ctx(device: torch.device, bf16: bool):
    if bf16 and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


V33_PARTIAL_SKIP_PREFIXES: Tuple[str, ...] = (
    "transformer_lora.",
    "freq_head.",
    "pyramid_sft.",
)


def _load_v3_weights_from_path(
    model: CodeFormerV3Restorer,
    path: Path,
    device: torch.device,
    *,
    partial: bool = False,
    skip_prefixes: Tuple[str, ...] = (),
) -> Dict[str, int]:
    if path.suffix.lower() == ".npz":
        if partial:
            arrays = load_v3_npz(path, model=model, partial=True, skip_prefixes=skip_prefixes)
            cur_sd = model.v3.state_dict()
            loaded = sum(
                1
                for k, v in arrays.items()
                if not (skip_prefixes and any(k.startswith(pref) for pref in skip_prefixes))
                and k in cur_sd
                and tuple(cur_sd[k].shape) == tuple(v.shape)
            )
            return {"loaded": loaded, "skipped": len(arrays) - loaded}
        arrays = load_v3_npz(path, model=model)
        return {"loaded": len(arrays), "skipped": 0}
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_sd: Optional[Mapping[str, torch.Tensor]] = None
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "shadow" in ckpt["ema"]:
            raw_sd = {k: v.to(device=device, dtype=torch.float32) for k, v in ckpt["ema"]["shadow"].items()}
        elif "v3_state_dict" in ckpt:
            raw_sd = {k: v.to(device=device, dtype=torch.float32) for k, v in ckpt["v3_state_dict"].items()}
    if raw_sd is None:
        raise ValueError(f"Unrecognized V3 checkpoint format: {path}")
    if partial:
        cur_sd = model.v3.state_dict()
        load_sd = {
            k: v
            for k, v in raw_sd.items()
            if not (skip_prefixes and any(k.startswith(pref) for pref in skip_prefixes))
            and k in cur_sd
            and tuple(cur_sd[k].shape) == tuple(v.shape)
        }
        model.v3.load_state_dict(load_sd, strict=False)
        return {"loaded": len(load_sd), "skipped": len(raw_sd) - len(load_sd)}
    model.v3.load_state_dict(raw_sd, strict=True)
    return {"loaded": len(raw_sd), "skipped": 0}


# ---------------------------------------------------------------------------
# Single-Step Forward/Backward Primitives (reusable in tests & main loop)
# ---------------------------------------------------------------------------


def train_step_stage1(
    model: CodeFormerV3Restorer,
    loss_fn: Stage1Loss,
    batch: Dict[str, Any],
    device: torch.device,
    bf16: bool = False,
    grad_accum: int = 1,
) -> Dict[str, float]:
    x_in_256 = batch["x_in_256"].to(device, non_blocking=True)
    gt_256 = batch["gt_256"].to(device, non_blocking=True)
    gt_512 = batch["gt_512"].to(device, non_blocking=True)
    obs_mask = batch["observed_mask"].to(device, non_blocking=True)
    is_clear = batch["is_clear"].to(device, non_blocking=True)
    deg_targets = batch["deg_targets"].to(device, non_blocking=True)

    with torch.no_grad():
        with _autocast_ctx(device, bf16):
            teacher = model.encode_gt_teacher(gt_512, observed_mask=obs_mask)

    with _autocast_ctx(device, bf16):
        outputs = model(x_in_256, obs_mask, stage1_only=True)

    loss, metrics = loss_fn(
        outputs=outputs,
        teacher=teacher,
        x_in=x_in_256,
        gt=gt_256,
        observed_mask=obs_mask,
        is_clear=is_clear,
        target_deg=deg_targets,
    )
    (loss / float(grad_accum)).backward()
    return metrics


def train_step_stage2(
    model: CodeFormerV3Restorer,
    loss_fn: Stage2Loss,
    batch: Dict[str, Any],
    device: torch.device,
    bf16: bool = False,
    grad_accum: int = 1,
) -> Dict[str, float]:
    x_in_256 = batch["x_in_256"].to(device, non_blocking=True)
    gt_256 = batch["gt_256"].to(device, non_blocking=True)
    gt_512 = batch["gt_512"].to(device, non_blocking=True)
    obs_mask = batch["observed_mask"].to(device, non_blocking=True)
    is_clear = batch["is_clear"].to(device, non_blocking=True)
    landmarks = batch["landmarks"].to(device, non_blocking=True) if "landmarks" in batch and batch["landmarks"] is not None else None
    deg_targets = batch["deg_targets"].to(device, non_blocking=True) if "deg_targets" in batch and batch["deg_targets"] is not None else None
    blur_mask = batch["blur_mask"].to(device, non_blocking=True) if "blur_mask" in batch and batch["blur_mask"] is not None else None

    needs_teacher = (loss_fn.w_ret_code > 0.0) or (loss_fn.w_ret_sft > 0.0)
    teacher = None
    if needs_teacher:
        with torch.no_grad():
            with _autocast_ctx(device, bf16):
                teacher = model.encode_gt_teacher(gt_512, observed_mask=obs_mask)

    with _autocast_ctx(device, bf16):
        outputs = model(x_in_256, obs_mask, stage1_only=False)

    loss, metrics = loss_fn(
        outputs=outputs,
        gt_256=gt_256,
        gt_512=gt_512,
        x_in_512=x_in_256,
        observed_mask=obs_mask,
        is_clear=is_clear,
        landmarks=landmarks,
        teacher=teacher,
        deg_targets=deg_targets,
        blur_mask=blur_mask,
    )
    (loss / float(grad_accum)).backward()
    return metrics


def train_step_stage3(
    model: CodeFormerV3Restorer,
    loss_fn: Stage2Loss,
    gan_helper: Stage3GANHelper,
    opt_d: torch.optim.Optimizer,
    batch: Dict[str, Any],
    device: torch.device,
    step: int,
    bf16: bool = False,
    grad_accum: int = 1,
    do_d_step: bool = True,
    lambda_scale: float = 1.0,
) -> Dict[str, float]:
    x_in_256 = batch["x_in_256"].to(device, non_blocking=True)
    gt_256 = batch["gt_256"].to(device, non_blocking=True)
    gt_512 = batch["gt_512"].to(device, non_blocking=True)
    obs_mask = batch["observed_mask"].to(device, non_blocking=True)
    is_clear = batch["is_clear"].to(device, non_blocking=True)
    landmarks = batch["landmarks"].to(device, non_blocking=True) if "landmarks" in batch and batch["landmarks"] is not None else None
    deg_targets = batch["deg_targets"].to(device, non_blocking=True) if "deg_targets" in batch and batch["deg_targets"] is not None else None
    blur_mask = batch["blur_mask"].to(device, non_blocking=True) if "blur_mask" in batch and batch["blur_mask"] is not None else None

    # 1. Generator forward & loss (GAN strictly on degraded samples; skip frozen teacher/retention in S3/S3B)
    for p in gan_helper.discriminator.parameters():
        p.requires_grad = False

    with _autocast_ctx(device, bf16):
        outputs = model(x_in_256, obs_mask, stage1_only=False)

    g_rec_loss, metrics = loss_fn(
        outputs=outputs,
        gt_256=gt_256,
        gt_512=gt_512,
        x_in_512=None,
        observed_mask=obs_mask,
        is_clear=is_clear,
        landmarks=landmarks,
        teacher=None,
        deg_targets=deg_targets,
        blur_mask=blur_mask,
    )
    g_gan_loss = gan_helper.generator_gan_loss(
        outputs["y_256"], gt_256, obs_mask, is_clear, lambda_scale=lambda_scale
    )
    g_total = g_rec_loss + g_gan_loss
    (g_total / float(grad_accum)).backward()
    metrics["l_gan_g"] = float(g_gan_loss.detach().item())
    metrics["gan_lambda_eff"] = float(gan_helper.lambda_gan * float(lambda_scale))
    metrics["loss_total"] = float(g_total.detach().item())

    # 2. Discriminator update + lazy R1 (strictly on degraded samples)
    for p in gan_helper.discriminator.parameters():
        p.requires_grad = True

    d_loss, d_metrics = gan_helper.discriminator_step_loss(
        outputs["y_256"].detach(),
        gt_256,
        obs_mask,
        is_clear,
        step=step,
    )
    if d_loss is not None:
        (d_loss / float(grad_accum)).backward()
        if do_d_step:
            nn.utils.clip_grad_norm_(gan_helper.discriminator.parameters(), 1.0)
            opt_d.step()
            opt_d.zero_grad(set_to_none=True)
    metrics.update(d_metrics)
    return metrics


def train_step_stage1_v34(
    model: CodeFormerV3Restorer,
    loss_fn: V34Stage1Loss,
    batch: Dict[str, Any],
    device: torch.device,
    bf16: bool = False,
    grad_accum: int = 1,
) -> Dict[str, float]:
    x_in_256 = batch["x_in_256"].to(device, non_blocking=True)
    gt_256 = batch["gt_256"].to(device, non_blocking=True)
    gt_512 = batch["gt_512"].to(device, non_blocking=True)
    x_struct_clean = (
        batch["x_struct_clean_256"].to(device, non_blocking=True)
        if "x_struct_clean_256" in batch and batch["x_struct_clean_256"] is not None
        else gt_256
    )
    x_photo_tgt = (
        batch["x_photo_tgt_256"].to(device, non_blocking=True)
        if "x_photo_tgt_256" in batch and batch["x_photo_tgt_256"] is not None
        else gt_256
    )
    obs_mask = batch["observed_mask"].to(device, non_blocking=True)
    is_clear = batch["is_clear"].to(device, non_blocking=True)
    deg_targets = batch["deg_targets"].to(device, non_blocking=True)
    regimes = batch.get("regime")
    is_nn_pix = (
        batch["is_nn_pixelated"].to(device, non_blocking=True)
        if "is_nn_pixelated" in batch and batch["is_nn_pixelated"] is not None
        else None
    )

    teacher = None
    if loss_fn.w_code > 0.0:
        with torch.no_grad():
            with _autocast_ctx(device, bf16):
                teacher = model.encode_gt_teacher(gt_512, observed_mask=obs_mask)

    with _autocast_ctx(device, bf16):
        outputs = model(x_in_256, obs_mask, skip_decoder=True)

    loss, metrics = loss_fn(
        outputs=outputs,
        teacher=teacher,
        x_struct_clean=x_struct_clean,
        x_photo_tgt=x_photo_tgt,
        observed_mask=obs_mask,
        is_clear=is_clear,
        deg_targets=deg_targets,
        regimes=regimes,
        is_nn_pixelated=is_nn_pix,
    )
    (loss / float(grad_accum)).backward()
    return metrics


def train_step_stage2_v34(
    model: CodeFormerV3Restorer,
    loss_fn: V34Stage2Loss,
    batch: Dict[str, Any],
    device: torch.device,
    bf16: bool = False,
    grad_accum: int = 1,
) -> Dict[str, float]:
    x_in_256 = batch["x_in_256"].to(device, non_blocking=True)
    gt_256 = batch["gt_256"].to(device, non_blocking=True)
    gt_512 = batch["gt_512"].to(device, non_blocking=True)
    x_struct_clean = (
        batch["x_struct_clean_256"].to(device, non_blocking=True)
        if "x_struct_clean_256" in batch and batch["x_struct_clean_256"] is not None
        else gt_256
    )
    obs_mask = batch["observed_mask"].to(device, non_blocking=True)
    is_clear = batch["is_clear"].to(device, non_blocking=True)
    landmarks = (
        batch["landmarks"].to(device, non_blocking=True)
        if "landmarks" in batch and batch["landmarks"] is not None
        else None
    )
    deg_targets = (
        batch["deg_targets"].to(device, non_blocking=True)
        if "deg_targets" in batch and batch["deg_targets"] is not None
        else None
    )
    regimes = batch.get("regime")
    is_nn_pix = (
        batch["is_nn_pixelated"].to(device, non_blocking=True)
        if "is_nn_pixelated" in batch and batch["is_nn_pixelated"] is not None
        else None
    )

    teacher = None
    if loss_fn.w_code > 0.0:
        with torch.no_grad():
            with _autocast_ctx(device, bf16):
                teacher = model.encode_gt_teacher(gt_512, observed_mask=obs_mask)

    with _autocast_ctx(device, bf16):
        outputs = model(x_in_256, obs_mask, skip_decoder=False)

    loss, metrics = loss_fn(
        outputs=outputs,
        gt_256=gt_256,
        x_struct_clean=x_struct_clean,
        observed_mask=obs_mask,
        is_clear=is_clear,
        landmarks=landmarks,
        teacher=teacher,
        deg_targets=deg_targets,
        regimes=regimes,
        is_nn_pixelated=is_nn_pix,
    )
    (loss / float(grad_accum)).backward()
    return metrics


# ---------------------------------------------------------------------------
# Pareto Model Soup (--stage soup)
# ---------------------------------------------------------------------------


def _extract_v3_state(path: Path, model: CodeFormerV3Restorer) -> Dict[str, torch.Tensor]:
    if path.suffix.lower() == ".npz":
        arrs = load_v3_npz(path, model=None)
        return {k: torch.from_numpy(v.copy()).float() for k, v in arrs.items()}
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "ema" in ckpt and "shadow" in ckpt["ema"]:
        return {k: v.detach().cpu().float() for k, v in ckpt["ema"]["shadow"].items()}
    if "v3_state_dict" in ckpt:
        return {k: v.detach().cpu().float() for k, v in ckpt["v3_state_dict"].items()}
    raise ValueError(f"Cannot extract V3 weights from {path}")


def _load_native_cctv_cases_for_train(
    native_cctv_dir: Optional[Path] = None,
    qmul_json: Optional[Path] = None,
    chokepoint_json: Optional[Path] = None,
    qmul_crops_dir: Optional[Path] = None,
    chokepoint_crops_dir: Optional[Path] = None,
    manifests_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Resolve DEV native CCTV cases (QMUL-SurvFace + ChokePoint development ONLY; never FINAL)."""
    if native_cctv_dir is None and qmul_crops_dir is None and chokepoint_crops_dir is None:
        return []

    def _pick(cands: Sequence[Optional[Path]]) -> Optional[Path]:
        for c in cands:
            if c is not None and c.exists():
                return c
        return None

    nd = Path(native_cctv_dir) if native_cctv_dir is not None else None
    md = Path(manifests_dir) if manifests_dir is not None else None

    eff_qjson = _pick([
        qmul_json,
        (nd / "qmul_frozen_subset.json") if nd else None,
        (nd / "cctv_native_development_v2" / "frozen_subset.json") if nd else None,
        (md / "cctv_native_development_v2" / "frozen_subset.json") if md else None,
        ROOT / "outputs/cctv_native_development_v2/frozen_subset.json",
    ])
    eff_cjson = _pick([
        chokepoint_json,
        (nd / "chokepoint_frozen_subset.json") if nd else None,
        (nd / "cctv_chokepoint_native_development_v1" / "frozen_subset.json") if nd else None,
        (md / "cctv_chokepoint_native_development_v1" / "frozen_subset.json") if md else None,
        ROOT / "outputs/cctv_chokepoint_native_development_v1/frozen_subset.json",
    ])
    eff_qcrops = _pick([
        qmul_crops_dir,
        (nd / "qmul_crops") if nd else None,
        (nd / "qmul") if nd else None,
        (nd / "cctv_native_development_v2") if nd else None,
        nd,
    ])
    eff_ccrops = _pick([
        chokepoint_crops_dir,
        (nd / "chokepoint_crops") if nd else None,
        (nd / "chokepoint") if nd else None,
        (nd / "cctv_chokepoint_native_development_v1") if nd else None,
        nd,
    ])
    return load_dev_native_cctv(
        qmul_json=eff_qjson if eff_qjson is not None else Path("__missing_qmul__.json"),
        chokepoint_json=eff_cjson if eff_cjson is not None else Path("__missing_choke__.json"),
        qmul_crops_dir=eff_qcrops,
        chokepoint_crops_dir=eff_ccrops,
    )


def run_model_soup(
    model: CodeFormerV3Restorer,
    s2_path: Path,
    s3_path: Path,
    dev_manifest: Path,
    out_dir: Path,
    lpips_model: Optional[nn.Module] = None,
    arcface_model: Optional[DifferentiableArcFaceLoss] = None,
    device: torch.device = torch.device("cpu"),
    alphas: Sequence[float] = SOUP_ALPHAS,
    max_cases: Optional[int] = None,
    native_cctv_cases: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Evaluate alpha in SOUP_ALPHAS between Stage-2 & Stage-3 EMA on DEV (+ optional DEV Native CCTV)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = out_dir / "journal.jsonl"
    sd_s2 = _extract_v3_state(s2_path, model)
    sd_s3 = _extract_v3_state(s3_path, model)

    if set(sd_s2.keys()) != set(sd_s3.keys()):
        raise ValueError("Stage-2 and Stage-3 V3 state dict keys differ")

    results: List[Dict[str, Any]] = []
    best_score = -1e9
    best_alpha = 0.0
    best_sd: Optional[Dict[str, torch.Tensor]] = None

    for alpha in alphas:
        blended = {
            k: ((1.0 - float(alpha)) * sd_s2[k] + float(alpha) * sd_s3[k]).to(device=device, dtype=torch.float32)
            for k in sd_s2
        }
        model.v3.load_state_dict(blended, strict=True)
        summary = evaluate_v3_on_manifest(
            model,
            dev_manifest,
            lpips_model=lpips_model,
            arcface_model=arcface_model,
            device=device,
            max_cases=max_cases,
        )
        native_summary = None
        if native_cctv_cases:
            native_summary, _ = evaluate_runner_on_cases(
                make_v3_runner(model),
                native_cctv_cases,
                lpips_model=None,
                arcface_model=arcface_model,
                device=device,
            )
        ov = summary["png_floor"]["overall"]
        score = _pareto_eval_score_v31(summary["png_floor"], native_summary=native_summary)
        nat_arc = native_summary.get("same_id_arcface_cos_mean") if native_summary else None
        nat_lap = native_summary["png_floor"]["overall"].get("central_laplacian") if native_summary else None
        entry = {
            "alpha": float(alpha),
            "score": score,
            "overall_png_floor": ov,
            "native_same_id_arc": nat_arc,
            "native_central_laplacian": nat_lap,
            "summary": summary,
        }
        results.append(entry)
        append_journal(
            journal_path,
            {
                "event": "soup_eval",
                "alpha": float(alpha),
                "score": score,
                "overall": ov,
                "native_same_id_arc": nat_arc,
                "native_central_laplacian": nat_lap,
            },
        )

        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_sd = {k: v.detach().cpu().clone() for k, v in blended.items()}

    assert best_sd is not None
    model.v3.load_state_dict({k: v.to(device) for k, v in best_sd.items()}, strict=True)
    npz_path = out_dir / "v3_modules_ema.npz"
    sha = atomic_export_v3_npz(model, npz_path)
    atomic_torch_save({"stage": "soup", "best_alpha": best_alpha, "v3_state_dict": best_sd}, out_dir / "best_ema.pt")

    soup_report = {
        "best_alpha": best_alpha,
        "best_score": best_score,
        "v3_modules_ema_npz": str(npz_path),
        "v3_state_sha": sha,
        "candidates": [
            {
                "alpha": r["alpha"],
                "score": r["score"],
                "overall": r["overall_png_floor"],
                "native_same_id_arc": r["native_same_id_arc"],
                "native_central_laplacian": r["native_central_laplacian"],
            }
            for r in results
        ],
    }
    (out_dir / "soup_summary.json").write_text(
        json.dumps(soup_report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return soup_report


# ---------------------------------------------------------------------------
# CLI & Main Training Loop
# ---------------------------------------------------------------------------


def _pareto_eval_score(ov: Mapping[str, Any]) -> float:
    psnr = float(ov.get("psnr_db") or 0.0)
    ssim7 = float(ov.get("ssim7") or 0.0)
    lpips_v = float(ov.get("lpips") or 0.0)
    arc_v = float(ov.get("arcface_cos") or 0.0)
    return psnr + 10.0 * ssim7 + 5.0 * arc_v - 10.0 * lpips_v


def _mean_blur_lap_ratio(by_role: Mapping[str, Any]) -> Optional[float]:
    blur_laps = [
        float(by_role[r]["central_laplacian_ratio"])
        for r in _BLUR_ROLES
        if r in by_role and by_role[r].get("central_laplacian_ratio") is not None
    ]
    if not blur_laps:
        return None
    return float(sum(blur_laps) / len(blur_laps))


def _pareto_eval_score_v31(
    png_floor: Mapping[str, Any],
    native_summary: Optional[Mapping[str, Any]] = None,
) -> float:
    ov = png_floor.get("overall", png_floor)
    by_role = png_floor.get("by_role", {})
    psnr = min(float(ov.get("psnr_db") or 0.0), 34.0)
    ssim7 = float(ov.get("ssim7") or 0.0)
    lpips_v = float(ov.get("lpips") or 0.0)
    arc_v = float(ov.get("arcface_cos") or 0.0)
    lap_r = float(ov.get("central_laplacian_ratio") or 1.0)
    flips = int(ov.get("catastrophic_id_flips") or 0)

    # Sharpness band centered at LapR=1.0, strongly punishing blur (<0.85) & halos (>1.15)
    lap_pen = (
        5.0 * abs(lap_r - 1.0)
        + 4.0 * max(0.0, 0.85 - lap_r)
        + 6.0 * max(0.0, lap_r - 1.15)
    )
    score = 0.45 * psnr + 5.0 * ssim7 + 18.0 * arc_v - 30.0 * lpips_v - lap_pen - 5.0 * flips

    # Per-role guardrails (prevents hiding blur behind clear/haze or regressing clear/lowlight/haze)
    blur_lap_mean = _mean_blur_lap_ratio(by_role)
    if blur_lap_mean is not None:
        score -= 5.0 * max(0.0, 0.85 - blur_lap_mean)
    if "clear" in by_role and by_role["clear"].get("psnr_db") is not None:
        score -= 2.0 * max(0.0, 45.0 - float(by_role["clear"]["psnr_db"]))
    if "lowlight_sensor_noise" in by_role and by_role["lowlight_sensor_noise"].get("psnr_db") is not None:
        score -= 1.5 * max(0.0, 24.5 - float(by_role["lowlight_sensor_noise"]["psnr_db"]))
    if "haze_colorcast_glare" in by_role and by_role["haze_colorcast_glare"].get("psnr_db") is not None:
        score -= 1.5 * max(0.0, 22.5 - float(by_role["haze_colorcast_glare"]["psnr_db"]))

    # Native CCTV Pareto term (prioritize same_id_arc >= 0.650 & healthy CentralLap)
    if native_summary is not None:
        nat_arc = native_summary.get("same_id_arcface_cos_mean")
        if nat_arc is not None:
            nat_arc_f = float(nat_arc)
            score += 20.0 * (nat_arc_f - 0.650) - 40.0 * max(0.0, 0.650 - nat_arc_f)
        nat_lap = native_summary.get("png_floor", {}).get("overall", {}).get("central_laplacian")
        if nat_lap is not None:
            nat_lap_f = float(nat_lap)
            score -= 50.0 * max(0.0, 0.065 - nat_lap_f) + 50.0 * max(0.0, nat_lap_f - 0.115)
    return float(score)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CodeFormer Restoration V3 / V3.1 / V3.2 / V3.3 / V3.4 — Multi-Stage L4 Trainer")
    p.add_argument("--stage", choices=["1", "2", "3", "2b", "3b", "2c", "3c", "2d", "3d", "1v4", "2v4", "soup"], required=True)
    p.add_argument(
        "--arch-variant",
        choices=["v30", "v33", "v34"],
        default=None,
        help="Architecture variant (default: v34 for 1v4/2v4, v33 for 2d/3d, auto-detected for soup, v30 otherwise)",
    )
    p.add_argument(
        "--calibration",
        choices=["v30", "v31", "v32", "v33", "v34"],
        default=None,
        help="Degradation/sampler calibration profile (default: v34 for 1v4/2v4, v33 for 2d/3d, v32 for 2c/3c, v31 for 2b/3b, v30 otherwise)",
    )
    p.add_argument(
        "--partial-init",
        action="store_true",
        help="Allow partial warm-start from incompatible .npz/.pt (auto-enabled for stage 2d from V3.0/3.1/3.2)",
    )
    p.add_argument(
        "--reset-freq-head",
        action="store_true",
        help="Re-initialize V3FreqHead fresh after --init-from (skipped on --resume)",
    )
    p.add_argument("--ema-decay", type=float, default=None, help="Optional EMA decay override")
    p.add_argument(
        "--gan-warmup-steps",
        type=int,
        default=None,
        help="Stage-3B/3C/3D linear lambda_gan warmup steps (default: 400 for 3d, 500 otherwise)",
    )
    p.add_argument("--bf16", action="store_true", help="Enable bf16 autocast on CUDA (FP32 fallback on CPU)")
    p.add_argument("--grad-checkpoint", action="store_true", help="Enable activation checkpointing on frozen CF blocks")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optimizer steps (default: 2500 for 1v4, 4500 for 2v4, 3000 for 2d, 1200 for 3d, 1000 otherwise)",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Linear warmup steps (0.1->1.0) before cosine decay (default: 150 for 1v4, 200 otherwise)",
    )
    p.add_argument("--log-every", type=int, default=25, help="Print concise stdout progress every N optimizer steps")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument(
        "--mid-npz-every",
        type=int,
        default=None,
        help="Save periodic mid-checkpoint EMA .npz every N steps (default: 1500 for 2v4, 0 otherwise)",
    )
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--soup-s2", type=Path, default=None, help="Stage-2 EMA checkpoint/npz for --stage soup")
    p.add_argument("--soup-s3", type=Path, default=None, help="Stage-3 EMA checkpoint/npz for --stage soup")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--manifests-dir", type=Path, default=None, help="Optional flat VM manifests/ directory")
    p.add_argument("--native-cctv-dir", type=Path, default=None, help="Optional DEV native CCTV crops/manifests dir for live eval & Soup")
    p.add_argument("--qmul-json", type=Path, default=None)
    p.add_argument("--qmul-crops-dir", type=Path, default=None)
    p.add_argument("--chokepoint-json", type=Path, default=None)
    p.add_argument("--chokepoint-crops-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--codeformer-path", type=Path, default=ROOT / "outputs/codeformer_restoration_pretrained_v1/codeformer.pth")
    p.add_argument("--perceptual-dir", type=Path, default=ROOT / "outputs/codeformer_perceptual_assets_v1")
    p.add_argument("--arcface-onnx", type=Path, default=Path("~/.insightface/models/buffalo_l/w600k_r50.onnx").expanduser())
    p.add_argument("--dev-manifest", type=Path, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-lpips", action="store_true", help="Explicitly disable LPIPS perceptual loss")
    p.add_argument("--no-arcface", action="store_true", help="Explicitly disable ArcFace identity loss")
    p.add_argument("--allow-random-base", action="store_true", help="Allow random CodeFormer base (unit tests only)")
    p.add_argument("--allow-missing-manifests", action="store_true", help="Allow missing split manifests (unit tests only)")
    p.add_argument("--no-split-guard", action="store_true", help="Disable workspace split.json guard (unit tests only)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_bf16 = bool(args.bf16 and device.type == "cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = args.out_dir / "journal.jsonl"

    # Fail-fast official CodeFormer weights check (unless --allow-random-base)
    if args.allow_random_base and (args.codeformer_path is None or not args.codeformer_path.is_file()):
        cf_path = None
    else:
        if args.codeformer_path is None or not args.codeformer_path.is_file():
            raise FileNotFoundError(
                f"Official CodeFormer weights not found at --codeformer-path={args.codeformer_path}. "
                "Pass a valid codeformer.pth (or --allow-random-base for synthetic unit tests)."
            )
        cf_path = args.codeformer_path

    # Resolve arch_variant (auto-set to v34 for 1v4/2v4, v33 for 2d/3d; auto-detect for soup)
    if args.arch_variant is not None:
        arch_variant = args.arch_variant
    elif args.stage in ("1v4", "2v4"):
        arch_variant = "v34"
    elif args.stage in ("2d", "3d"):
        arch_variant = "v33"
    elif args.stage == "soup":
        s2_probe = args.soup_s2 or args.init_from
        if s2_probe is not None and s2_probe.suffix.lower() == ".npz" and s2_probe.is_file():
            arch_variant = detect_v3_npz_variant(s2_probe)
        else:
            arch_variant = "v30"
    else:
        arch_variant = "v30"

    if args.max_steps is None:
        if args.stage == "1v4":
            args.max_steps = 2500
        elif args.stage == "2v4":
            args.max_steps = 4500
        elif args.stage == "2d":
            args.max_steps = 3000
        elif args.stage == "3d":
            args.max_steps = 1200
        else:
            args.max_steps = 1000
    if args.warmup_steps is None:
        args.warmup_steps = 150 if args.stage == "1v4" else 200
    if args.mid_npz_every is None:
        args.mid_npz_every = 1500 if args.stage == "2v4" else 0
    if args.gan_warmup_steps is None:
        args.gan_warmup_steps = 400 if args.stage == "3d" else 500

    model = CodeFormerV3Restorer(
        codeformer_weights_path=cf_path,
        grad_checkpoint=bool(args.grad_checkpoint),
        arch_variant=arch_variant,
    ).to(device)

    # Fail-fast LPIPS & ArcFace checks in perceptual stages & soup unless explicitly disabled
    lpips_model = None
    if args.stage in ("2", "3", "2b", "3b", "2c", "3c", "2d", "3d", "2v4", "soup") and not args.no_lpips:
        if (
            args.perceptual_dir is None
            or not (args.perceptual_dir / "vgg16_features.pth").is_file()
            or not (args.perceptual_dir / "lpips_vgg_v01.pth").is_file()
        ):
            raise FileNotFoundError(
                f"LPIPS weights (vgg16_features.pth, lpips_vgg_v01.pth) missing in --perceptual-dir={args.perceptual_dir}. "
                "Pass a valid --perceptual-dir or explicitly disable with --no-lpips."
            )
        lpips_model = load_frozen_lpips(args.perceptual_dir, device=device)

    arcface_model = None
    if args.stage in ("2", "3", "2b", "3b", "2c", "3c", "2d", "3d", "2v4", "soup") and not args.no_arcface:
        if args.arcface_onnx is None or not args.arcface_onnx.is_file():
            raise FileNotFoundError(
                f"ArcFace ONNX weights missing at --arcface-onnx={args.arcface_onnx}. "
                "Pass a valid w600k_r50.onnx or explicitly disable with --no-arcface."
            )
        arcface_model = DifferentiableArcFaceLoss(onnx_path=args.arcface_onnx, device=device)

    native_cctv_cases = _load_native_cctv_cases_for_train(
        native_cctv_dir=args.native_cctv_dir,
        qmul_json=args.qmul_json,
        chokepoint_json=args.chokepoint_json,
        qmul_crops_dir=args.qmul_crops_dir,
        chokepoint_crops_dir=args.chokepoint_crops_dir,
        manifests_dir=args.manifests_dir,
    )

    # Handle --stage soup
    if args.stage == "soup":
        s2_path = args.soup_s2 or args.init_from
        s3_path = args.soup_s3 or args.resume
        if not s2_path or not s3_path or not args.dev_manifest:
            raise ValueError("--stage soup requires --soup-s2 (or --init-from), --soup-s3 (or --resume), and --dev-manifest")
        run_model_soup(
            model=model,
            s2_path=s2_path,
            s3_path=s3_path,
            dev_manifest=args.dev_manifest,
            out_dir=args.out_dir,
            lpips_model=lpips_model,
            arcface_model=arcface_model,
            device=device,
            native_cctv_cases=native_cctv_cases if native_cctv_cases else None,
        )
        return

    if args.data_dir is None:
        raise ValueError("--data-dir is required for stages 1, 2, 3, 2b, 3b, 2c, 3c, 2d, and 3d")

    init_report: Dict[str, int] = {"loaded": 0, "skipped": 0}
    if args.init_from is not None:
        use_partial = bool(args.partial_init or args.stage == "2d")
        skip_prefs = V33_PARTIAL_SKIP_PREFIXES if (use_partial and arch_variant == "v33") else ()
        init_report = _load_v3_weights_from_path(
            model,
            args.init_from,
            device,
            partial=use_partial,
            skip_prefixes=skip_prefs,
        )

    is_resuming = bool(args.resume is not None and args.resume.is_file())
    if args.reset_freq_head and not is_resuming:
        reset_freq_head(model)

    if args.calibration is not None:
        calibration = args.calibration
    elif args.stage in ("1v4", "2v4"):
        calibration = "v34"
    elif args.stage in ("2d", "3d"):
        calibration = "v33"
    elif args.stage in ("2c", "3c"):
        calibration = "v32"
    elif args.stage in ("2b", "3b"):
        calibration = "v31"
    else:
        calibration = "v30"

    configure_stage_trainability(model, args.stage)
    opt_g, sched_g = build_optimizer_and_scheduler(
        model, args.stage, args.max_steps, lr_override=args.lr, warmup_steps=args.warmup_steps
    )
    if args.ema_decay is not None:
        ema_decay = float(args.ema_decay)
    elif args.stage in ("2b", "2c", "2d"):
        ema_decay = 0.9985
    elif args.stage == "3":
        ema_decay = 0.9995
    else:
        # stages 1, 2, 3b, 3c, 3d
        ema_decay = 0.999
    ema = V3EMA(model, decay=ema_decay)

    disc: Optional[PatchGANDiscriminator] = None
    gan_helper: Optional[Stage3GANHelper] = None
    opt_d: Optional[torch.optim.Optimizer] = None
    if args.stage in ("3", "3b", "3c", "3d"):
        disc = PatchGANDiscriminator().to(device)
        if args.stage == "3d":
            lambda_gan_max = 0.03
        elif args.stage == "3c":
            lambda_gan_max = 0.08
        elif args.stage == "3b":
            lambda_gan_max = 0.06
        else:
            lambda_gan_max = 0.01
        gan_helper = Stage3GANHelper(disc, lambda_gan=lambda_gan_max, r1_gamma=10.0, r1_every=16, gan_mode="hinge")
        opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.0, 0.99), weight_decay=0.0)

    start_step = 0
    epoch = 0
    best_metric = -1e9
    if args.resume is not None and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.v3.load_state_dict(ckpt["v3_state_dict"], strict=True)
        opt_g.load_state_dict(ckpt["opt_g"])
        sched_g.load_state_dict(ckpt["sched_g"])
        ema.load_state_dict(ckpt["ema"], device=device)
        if disc is not None and "disc" in ckpt:
            disc.load_state_dict(ckpt["disc"], strict=True)
        if opt_d is not None and "opt_d" in ckpt:
            opt_d.load_state_dict(ckpt["opt_d"])
        start_step = int(ckpt.get("step", 0))
        epoch = int(ckpt.get("epoch", 0))
        best_metric = float(ckpt.get("best_metric", -1e9))

    split_guard = (
        None
        if args.no_split_guard
        else ZeroLeakageSplitGuard(
            workspace_root=ROOT,
            manifests_dir=args.manifests_dir,
            strict_files=not args.allow_missing_manifests,
        )
    )
    dataset = V3RestorationDataset(
        data_dir=args.data_dir,
        split_guard=split_guard,
        split="train",
        seed=args.seed,
        calibration=calibration,
    )
    loader, sampler = make_stratified_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        seed=args.seed,
        drop_last=len(dataset) >= args.batch_size,
        calibration=calibration,
    )

    stage1_loss = Stage1Loss()
    v34_stage1_loss = V34Stage1Loss()
    v34_stage2_loss = (
        V34Stage2Loss(lpips_model=lpips_model, arcface_model=arcface_model, device=device)
        if args.stage == "2v4"
        else None
    )
    if args.stage == "2b":
        stage2_loss = Stage2Loss(
            lpips_model=lpips_model,
            arcface_model=arcface_model,
            device=device,
            w_charb=0.55,
            w_ssim=0.20,
            w_chroma=0.15,
            w_lpips=0.65,
            w_vgg_gram=0.20,
            w_hf_energy=0.18,
            w_arcface=0.45,
            w_lap=0.10,
            w_fft=0.12,
            w_aux512=0.12,
            w_wmap=0.30,
            w_ret_code=0.12,
            w_ret_sft=0.03,
            w_ret_photo=0.10,
            w_ret_deg=0.15,
            w_clear_l1=2.0,
            v31_clear_anchor=True,
            v31_aux512=True,
        )
    elif args.stage == "3b":
        stage2_loss = Stage2Loss(
            lpips_model=lpips_model,
            arcface_model=arcface_model,
            device=device,
            w_charb=0.40,
            w_ssim=0.15,
            w_chroma=0.15,
            w_lpips=0.75,
            w_vgg_gram=0.25,
            w_hf_energy=0.20,
            w_arcface=0.55,
            w_lap=0.05,
            w_fft=0.12,
            w_aux512=0.10,
            w_wmap=0.25,
            w_ret_code=0.0,
            w_ret_sft=0.0,
            w_ret_photo=0.0,
            w_ret_deg=0.0,
            w_clear_l1=2.5,
            v31_clear_anchor=True,
            v31_aux512=True,
        )
    elif args.stage == "2c":
        stage2_loss = Stage2Loss(
            lpips_model=lpips_model,
            arcface_model=arcface_model,
            device=device,
            w_charb=0.35,
            w_ssim=0.12,
            w_chroma=0.15,
            w_lpips=0.85,
            w_vgg_gram=0.30,
            w_hf_energy=0.25,
            w_arcface=0.60,
            w_lap=0.05,
            w_fft=0.15,
            w_aux512=0.10,
            w_wmap=0.30,
            w_ret_code=0.10,
            w_ret_sft=0.0,
            w_ret_photo=0.10,
            w_ret_deg=0.15,
            w_adain_floor=0.10,
            w_pos_gate=0.15,
            w_clear_l1=2.5,
            v31_clear_anchor=True,
            v31_aux512=True,
        )
    elif args.stage == "3c":
        stage2_loss = Stage2Loss(
            lpips_model=lpips_model,
            arcface_model=arcface_model,
            device=device,
            w_charb=0.25,
            w_ssim=0.08,
            w_chroma=0.15,
            w_lpips=0.90,
            w_vgg_gram=0.35,
            w_hf_energy=0.25,
            w_arcface=0.65,
            w_lap=0.03,
            w_fft=0.15,
            w_aux512=0.08,
            w_wmap=0.25,
            w_ret_code=0.0,
            w_ret_sft=0.0,
            w_ret_photo=0.0,
            w_ret_deg=0.0,
            w_adain_floor=0.10,
            w_pos_gate=0.15,
            w_clear_l1=3.0,
            v31_clear_anchor=True,
            v31_aux512=True,
        )
    elif args.stage == "2d":
        stage2_loss = Stage2Loss(
            lpips_model=lpips_model,
            arcface_model=arcface_model,
            device=device,
            w_charb=0.35,
            w_ssim=0.12,
            w_chroma=0.15,
            w_lpips=0.85,
            w_vgg_gram=0.25,
            w_hf_energy=0.04,
            w_arcface=0.60,
            w_lap=0.05,
            w_fft=0.10,
            w_aux512=0.08,
            w_wmap=0.35,
            w_ret_code=0.0,
            w_ret_sft=0.0,
            w_ret_photo=0.10,
            w_ret_deg=0.15,
            w_adain_floor=0.10,
            w_pos_gate=0.0,
            w_clear_l1=1.5,
            v31_clear_anchor=False,
            v31_aux512=False,
            v33_mode=True,
        )
    elif args.stage == "3d":
        stage2_loss = Stage2Loss(
            lpips_model=lpips_model,
            arcface_model=arcface_model,
            device=device,
            w_charb=0.25,
            w_ssim=0.08,
            w_chroma=0.15,
            w_lpips=0.85,
            w_vgg_gram=0.25,
            w_hf_energy=0.04,
            w_arcface=0.60,
            w_lap=0.03,
            w_fft=0.10,
            w_aux512=0.06,
            w_wmap=0.30,
            w_ret_code=0.0,
            w_ret_sft=0.0,
            w_ret_photo=0.0,
            w_ret_deg=0.0,
            w_adain_floor=0.10,
            w_pos_gate=0.0,
            w_clear_l1=1.5,
            v31_clear_anchor=False,
            v31_aux512=False,
            v33_mode=True,
        )
    else:
        stage2_loss = Stage2Loss(
            lpips_model=lpips_model,
            arcface_model=arcface_model,
            device=device,
            w_arcface=0.45 if args.stage == "3" else 0.35,
        )

    append_journal(
        journal_path,
        {
            "event": "train_start",
            "stage": args.stage,
            "arch_variant": arch_variant,
            "calibration": calibration,
            "init_loaded": init_report["loaded"],
            "init_skipped": init_report["skipped"],
            "native_cctv_cases": len(native_cctv_cases),
            "reset_freq_head": bool(args.reset_freq_head and not is_resuming),
            "ema_decay": ema_decay,
            "device": str(device),
            "bf16": use_bf16,
            "start_step": start_step,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "dataset_size": len(dataset),
            "trainable_params": model.trainable_parameter_count(),
        },
    )

    model.train()
    opt_g.zero_grad(set_to_none=True)
    if opt_d is not None:
        opt_d.zero_grad(set_to_none=True)

    step = start_step
    accum_count = 0
    t_start = time.monotonic()

    while step < args.max_steps:
        sampler.set_epoch(epoch)
        epoch += 1
        last_metrics: Dict[str, float] = {}

        for batch in loader:
            if step >= args.max_steps:
                break
            accum_count += 1
            is_update_boundary = accum_count % args.grad_accum == 0

            if args.stage == "1":
                last_metrics = train_step_stage1(
                    model, stage1_loss, batch, device, bf16=use_bf16, grad_accum=args.grad_accum
                )
            elif args.stage == "1v4":
                last_metrics = train_step_stage1_v34(
                    model, v34_stage1_loss, batch, device, bf16=use_bf16, grad_accum=args.grad_accum
                )
            elif args.stage == "2v4":
                assert v34_stage2_loss is not None
                last_metrics = train_step_stage2_v34(
                    model, v34_stage2_loss, batch, device, bf16=use_bf16, grad_accum=args.grad_accum
                )
            elif args.stage in ("2", "2b", "2c", "2d"):
                last_metrics = train_step_stage2(
                    model, stage2_loss, batch, device, bf16=use_bf16, grad_accum=args.grad_accum
                )
            else:
                assert gan_helper is not None and opt_d is not None
                next_step = step + 1
                if args.stage in ("3b", "3c", "3d") and args.gan_warmup_steps > 0:
                    gan_scale = min(1.0, float(next_step) / float(args.gan_warmup_steps))
                else:
                    gan_scale = 1.0
                last_metrics = train_step_stage3(
                    model,
                    stage2_loss,
                    gan_helper,
                    opt_d,
                    batch,
                    device,
                    step=next_step,
                    bf16=use_bf16,
                    grad_accum=args.grad_accum,
                    do_d_step=is_update_boundary,
                    lambda_scale=gan_scale,
                )

            if not is_update_boundary:
                continue

            nn.utils.clip_grad_norm_([p for p in model.v3.parameters() if p.requires_grad], 1.0)
            opt_g.step()
            opt_g.zero_grad(set_to_none=True)
            sched_g.step()
            ema.update(model)
            step += 1

            lr_now = float(opt_g.param_groups[0]["lr"])
            elapsed_s = time.monotonic() - t_start
            steps_done = max(1, step - start_step)
            sec_per_step = elapsed_s / float(steps_done)
            eta_min = max(0.0, (args.max_steps - step) * sec_per_step / 60.0)
            append_journal(
                journal_path,
                {
                    "event": "step",
                    "stage": args.stage,
                    "step": step,
                    "lr": lr_now,
                    "elapsed_s": round(elapsed_s, 2),
                    **last_metrics,
                },
            )
            if args.log_every > 0 and (step % args.log_every == 0 or step == args.max_steps or step == start_step + 1):
                print(
                    f"[S{args.stage} {step}/{args.max_steps}] "
                    f"loss={last_metrics.get('loss_total', 0.0):.4f} "
                    f"lr={lr_now:.2e} s/step={sec_per_step:.2f} eta={eta_min:.1f}m",
                    flush=True,
                )

            # Periodic DEV evaluation
            if args.dev_manifest is not None and args.eval_every > 0 and step % args.eval_every == 0:
                backup = ema.copy_to(model)
                try:
                    if args.stage in ("1", "1v4"):
                        s1_ev = evaluate_stage1_exit_gates(model, args.dev_manifest, device=device)
                        score = (
                            s1_ev["delta_top1_acc"]
                            - 2.0 * s1_ev["lowlight_photo_mae"]
                            - 5.0 * s1_ev["clear_photo_drift"]
                        )
                        append_journal(
                            journal_path,
                            {"event": "eval", "stage": args.stage, "step": step, "epoch": epoch, "score": score, **s1_ev},
                        )
                        print(
                            f"[EVAL S{args.stage} step={step}] top1={s1_ev['degraded_code_top1_acc']:.4f} "
                            f"(base={s1_ev['cf_base_code_top1_acc']:.4f}, delta={s1_ev['delta_top1_acc']:+.4f}) "
                            f"ll_photo_mae={s1_ev['lowlight_photo_mae']:.4f} "
                            f"clear_drift={s1_ev['clear_photo_drift']:.4f} score={score:+.4f}",
                            flush=True,
                        )
                    else:
                        ev = evaluate_v3_on_manifest(
                            model,
                            args.dev_manifest,
                            lpips_model=lpips_model,
                            arcface_model=arcface_model,
                            device=device,
                        )
                        native_summary = None
                        if native_cctv_cases:
                            native_summary, _ = evaluate_runner_on_cases(
                                make_v3_runner(model),
                                native_cctv_cases,
                                lpips_model=None,
                                arcface_model=arcface_model,
                                device=device,
                            )
                        png_floor = ev["png_floor"]
                        ov = png_floor["overall"]
                        blur_lap = _mean_blur_lap_ratio(png_floor.get("by_role", {}))
                        lap_r = float(ov.get("central_laplacian_ratio") or 0.0)
                        nat_arc = native_summary.get("same_id_arcface_cos_mean") if native_summary else None
                        nat_lap = native_summary["png_floor"]["overall"].get("central_laplacian") if native_summary else None
                        score = (
                            _pareto_eval_score_v31(png_floor, native_summary=native_summary)
                            if args.stage in ("2b", "3b", "2c", "3c", "2d", "3d", "2v4")
                            else _pareto_eval_score(ov)
                        )
                        append_journal(
                            journal_path,
                            {
                                "event": "eval",
                                "stage": args.stage,
                                "step": step,
                                "epoch": epoch,
                                "score": score,
                                "lap_r": lap_r,
                                "blur_lap": blur_lap,
                                "native_same_id_arc": nat_arc,
                                "native_central_laplacian": nat_lap,
                                "overall": ov,
                            },
                        )
                        blur_lap_str = f"{blur_lap:.3f}" if blur_lap is not None else "n/a"
                        nat_str = (
                            f" nat_arc={nat_arc:.4f} nat_lap={nat_lap:.5f}"
                            if (nat_arc is not None and nat_lap is not None)
                            else (f" nat_lap={nat_lap:.5f}" if nat_lap is not None else "")
                        )
                        print(
                            f"[EVAL S{args.stage} step={step}] psnr={ov.get('psnr_db') or 0.0:.2f} "
                            f"mae={ov.get('mae') or 0.0:.5f} ssim7={ov.get('ssim7') or 0.0:.4f} "
                            f"lpips={ov.get('lpips') or 0.0:.4f} arc={ov.get('arcface_cos') or 0.0:.4f} "
                            f"lap_r={lap_r:.3f} blur_lap={blur_lap_str}{nat_str} score={score:.3f}",
                            flush=True,
                        )
                    if score > best_metric:
                        best_metric = score
                        atomic_torch_save(
                            {
                                "stage": args.stage,
                                "step": step,
                                "epoch": epoch,
                                "best_metric": best_metric,
                                "v3_state_dict": {k: v.detach().cpu() for k, v in model.v3.state_dict().items()},
                                "ema": ema.state_dict(),
                            },
                            args.out_dir / "best_ema.pt",
                        )
                        atomic_export_v3_npz(model, args.out_dir / "best_v3_modules_ema.npz")
                        atomic_export_v3_npz(model, args.out_dir / "v3_modules_ema.npz")
                finally:
                    ema.restore(model, backup)
                    model.train()

            # Periodic atomic checkpoint (never overwrites best_v3_modules_ema.npz or DEV-selected v3_modules_ema.npz)
            if args.save_every > 0 and (step % args.save_every == 0 or step == args.max_steps):
                ckpt_payload: Dict[str, Any] = {
                    "stage": args.stage,
                    "step": step,
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "v3_state_dict": {k: v.detach().cpu() for k, v in model.v3.state_dict().items()},
                    "opt_g": opt_g.state_dict(),
                    "sched_g": sched_g.state_dict(),
                    "ema": ema.state_dict(),
                }
                if disc is not None and opt_d is not None:
                    ckpt_payload["disc"] = disc.state_dict()
                    ckpt_payload["opt_d"] = opt_d.state_dict()
                atomic_torch_save(ckpt_payload, args.out_dir / "latest.pt")

                backup = ema.copy_to(model)
                try:
                    atomic_export_v3_npz(model, args.out_dir / "latest_v3_modules_ema.npz")
                    if args.dev_manifest is None:
                        atomic_export_v3_npz(model, args.out_dir / "v3_modules_ema.npz")
                finally:
                    ema.restore(model, backup)
                    model.train()

            # Periodic mid-checkpoint EMA .npz snapshot (for optional Soup blending)
            if args.mid_npz_every > 0 and step % args.mid_npz_every == 0:
                backup = ema.copy_to(model)
                try:
                    atomic_export_v3_npz(model, args.out_dir / f"ema_step_{step:05d}.npz")
                finally:
                    ema.restore(model, backup)
                    model.train()


if __name__ == "__main__":
    main()
