"""CodeFormer Restoration V3 — Comprehensive Multi-Metric Benchmark Harness.

Compares apples-to-apples:
- Untouched CodeFormer (w=1.0, w=0.7)
- Production V2 (conditioned_sft_v2_p2_u0224)
- CodeFormer Restoration V3 (v3_modules.npz + untouched codeformer.pth)

Across cohorts:
- DEV-Legacy-48 (monitor_v2 from v2_data_manifest.json)
- DEV-Syn (frozen unseen-ID synthetic manifest from datasets_v3.generate_dev_syn)
- DEV-Native-CCTV (ChokePoint + QMUL-SurvFace development cases ONLY; FINAL never touched)

Computes per-role and overall metrics on BOTH FP32 RAW and delivered uint8 PNG
(`floor` AND `rint` quantization comparison):
  MAE, PSNR (dB), Accuracy ((1-MAE)*100%), SSIM7, LPIPS (VGG16), ArcFace cosine,
  Central Laplacian + GT ratio, catastrophic ID flips (cos < 0.25 when input > 0.45),
  ~observed bit-exactness, and CPU latency.

Writes `summary.json`, `summary.md`, and visual side-by-side contact sheets
(`Input | CF | V2 | V3 | GT`).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from research.codeformer_restoration_v3.architecture_v3 import (
    CodeFormerV3Restorer,
    delivered_views_v3,
    detect_v3_npz_variant,
    load_v3_npz,
)
from research.codeformer_restoration_v3.losses_v3 import DifferentiableArcFaceLoss, load_frozen_lpips

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Exact Paired & No-Reference Metric Math (matching adviser_three_model_comparison.py)
# ---------------------------------------------------------------------------


def central_laplacian_energy(rgb_f64: np.ndarray) -> float:
    """Mean absolute 3x3 Laplacian energy on central [32:224, 32:224] luma region."""
    h, w = rgb_f64.shape[:2]
    y0, y1 = min(32, h // 8), max(h - 32, h - h // 8)
    x0, x1 = min(32, w // 8), max(w - 32, w - w // 8)
    patch = rgb_f64[y0:y1, x0:x1]
    luma = 0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2]
    lap = cv2.Laplacian(luma.astype(np.float64), cv2.CV_64F, ksize=3)
    return float(np.abs(lap).mean())


def estimate_noise_sigma(rgb_f64: np.ndarray) -> float:
    """Robust Immerse/Immerkaer-style MAD high-frequency noise estimator on central [32:224, 32:224] luma."""
    h, w = rgb_f64.shape[:2]
    y0, y1 = min(32, h // 8), max(h - 32, h - h // 8)
    x0, x1 = min(32, w // 8), max(w - 32, w - w // 8)
    patch = rgb_f64[y0:y1, x0:x1]
    luma = (0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2]).astype(np.float64)
    # 2D diagonal Haar wavelet detail HH = 0.5 * (I[0::2,0::2] - I[0::2,1::2] - I[1::2,0::2] + I[1::2,1::2])
    h2 = (luma.shape[0] // 2) * 2
    w2 = (luma.shape[1] // 2) * 2
    if h2 < 2 or w2 < 2:
        return 0.0
    sub = luma[:h2, :w2]
    hh = 0.5 * (sub[0::2, 0::2] - sub[0::2, 1::2] - sub[1::2, 0::2] + sub[1::2, 1::2])
    return float(np.median(np.abs(hh)) / 0.6745)


def compute_paired_pixel_metrics(
    target_f64: np.ndarray,
    pred_f64: np.ndarray,
    observed: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Exact MAE, MSE, PSNR (dB), Accuracy (%), SSIM7, and Central Laplacian + GT ratio.

    `target_f64` and `pred_f64` are float64 (H, W, 3) in [0, 1].
    """
    a = np.asarray(target_f64, dtype=np.float64)
    b = np.asarray(pred_f64, dtype=np.float64)
    error = a - b
    if observed is not None and not observed.all():
        obs = observed.astype(bool)
        mae = float(np.abs(error[obs]).mean())
        mse = float(np.mean(error[obs] ** 2))
    else:
        mae = float(np.abs(error).mean())
        mse = float(np.mean(error ** 2))

    psnr_db = float(-10.0 * math.log10(mse)) if mse > 0.0 else None

    def _mean7(val: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(val, -1, (7, 7), normalize=True)[3:-3, 3:-3]

    mu_a, mu_b = _mean7(a), _mean7(b)
    var_a = np.maximum(0.0, _mean7(a * a) - mu_a * mu_a)
    var_b = np.maximum(0.0, _mean7(b * b) - mu_b * mu_b)
    cov = _mean7(a * b) - mu_a * mu_b
    ssim_map = ((2.0 * mu_a * mu_b + 0.01 ** 2) * (2.0 * cov + 0.03 ** 2)) / (
        (mu_a ** 2 + mu_b ** 2 + 0.01 ** 2) * (var_a + var_b + 0.03 ** 2)
    )
    if observed is not None and not observed.all():
        obs_inner = observed[3:-3, 3:-3]
        ssim7 = float(ssim_map[obs_inner].mean()) if obs_inner.any() else float(ssim_map.mean())
    else:
        ssim7 = float(ssim_map.mean())

    lap_pred = central_laplacian_energy(b)
    lap_gt = central_laplacian_energy(a)
    lap_ratio = float(lap_pred / max(lap_gt, 1e-8))
    noise_pred = estimate_noise_sigma(b)

    return {
        "mae": mae,
        "mse": mse,
        "psnr_db": psnr_db,
        "psnr_infinite": mse == 0.0,
        "accuracy_pct": float((1.0 - mae) * 100.0),
        "ssim7": ssim7,
        "central_laplacian": lap_pred,
        "central_laplacian_gt": lap_gt,
        "central_laplacian_ratio": lap_ratio,
        "noise_sigma": noise_pred,
    }


def compute_perceptual_and_id_metrics(
    target_f32: np.ndarray,
    pred_f32: np.ndarray,
    input_f32: Optional[np.ndarray] = None,
    observed: Optional[np.ndarray] = None,
    lpips_model: Optional[torch.nn.Module] = None,
    arcface_model: Optional[DifferentiableArcFaceLoss] = None,
    device: Union[str, torch.device] = "cpu",
    cached_ref_emb: Optional[torch.Tensor] = None,
    cached_input_cos: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute LPIPS (VGG16), ArcFace cosine, and catastrophic ID flip flag."""
    out: Dict[str, Any] = {
        "lpips": None,
        "arcface_cos": None,
        "input_arcface_cos": cached_input_cos,
        "catastrophic_id_flip": False,
    }
    t_t = torch.from_numpy(target_f32.transpose(2, 0, 1).copy()).float()[None].to(device)
    p_t = torch.from_numpy(pred_f32.transpose(2, 0, 1).copy()).float()[None].to(device)
    m_t = (
        torch.from_numpy(observed.astype(np.float32))[None, None].to(device)
        if observed is not None
        else None
    )

    with torch.no_grad():
        if lpips_model is not None:
            if m_t is not None:
                p_use = p_t * m_t + t_t * (1.0 - m_t)
            else:
                p_use = p_t
            out["lpips"] = float(lpips_model(p_use * 2.0 - 1.0, t_t * 2.0 - 1.0).item())

        if arcface_model is not None:
            if cached_ref_emb is not None:
                p_clean = p_t * m_t + t_t * (1.0 - m_t) if m_t is not None else p_t
                emb_p, valid_p = arcface_model.embed(p_clean)
                if valid_p.any():
                    out["arcface_cos"] = float((emb_p[0] * cached_ref_emb).sum().clamp(-1.0, 1.0).item())
            else:
                cos_p, valid_p = arcface_model.cosine_similarity(p_t, t_t, mask=m_t)
                if valid_p.any():
                    out["arcface_cos"] = float(cos_p[0].item())
            if out["input_arcface_cos"] is None and input_f32 is not None:
                in_t = torch.from_numpy(input_f32.transpose(2, 0, 1).copy()).float()[None].to(device)
                cos_in, valid_in = arcface_model.cosine_similarity(in_t, t_t, mask=m_t)
                if valid_in.any():
                    out["input_arcface_cos"] = float(cos_in[0].item())
            if (
                out["arcface_cos"] is not None
                and out["input_arcface_cos"] is not None
                and out["input_arcface_cos"] > 0.45
                and out["arcface_cos"] < 0.25
            ):
                out["catastrophic_id_flip"] = True
    return out


def aggregate_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-case metric dicts into mean/sum summary statistics."""
    if not records:
        return {"count": 0}

    def _mean(key: str) -> Optional[float]:
        vals = [r[key] for r in records if r.get(key) is not None and math.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else None

    return {
        "count": len(records),
        "mae": _mean("mae"),
        "psnr_db": _mean("psnr_db"),
        "psnr_infinite_count": int(sum(1 for r in records if r.get("psnr_infinite"))),
        "accuracy_pct": _mean("accuracy_pct"),
        "ssim7": _mean("ssim7"),
        "lpips": _mean("lpips"),
        "arcface_cos": _mean("arcface_cos"),
        "central_laplacian": _mean("central_laplacian"),
        "central_laplacian_ratio": _mean("central_laplacian_ratio"),
        "noise_sigma": _mean("noise_sigma"),
        "catastrophic_id_flips": int(sum(1 for r in records if r.get("catastrophic_id_flip"))),
        "unobserved_exact_all": bool(all(r.get("unobserved_exact", True) for r in records)),
        "cpu_latency_s_mean": _mean("cpu_latency_s"),
    }


# ---------------------------------------------------------------------------
# Cohort Loaders (DEV-Legacy-48, DEV-Syn, DEV-Native-CCTV)
# ---------------------------------------------------------------------------


def _resolve_file(candidates: Sequence[Optional[Path]]) -> Optional[Path]:
    for cand in candidates:
        if cand is not None and cand.is_file():
            return cand
    return None


def load_dev_legacy_48(
    manifest_path: Union[str, Path] = ROOT / "outputs/codeformer_conditioned_sft_learning_v2_preparation/v2_data_manifest.json",
    workspace_root: Path = ROOT,
    v2_data_root: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    mp = Path(manifest_path)
    if not mp.is_file():
        raise FileNotFoundError(f"DEV-Legacy-48 v2_data_manifest.json not found: {mp}")
    data = json.loads(mp.read_text(encoding="utf-8"))
    dr = Path(v2_data_root) if v2_data_root is not None else mp.parent
    cases = []
    for row in data["monitor_v2_cases"]:
        in_rel = row["input"]["path"]
        tgt_rel = row["target"]["path"]
        in_p = _resolve_file([
            workspace_root / in_rel,
            dr / in_rel,
            dr / "inputs" / Path(in_rel).name,
            dr / Path(in_rel).name,
        ])
        tgt_p = _resolve_file([
            workspace_root / tgt_rel,
            dr / tgt_rel,
            dr / "targets" / Path(tgt_rel).name,
            dr / Path(tgt_rel).name,
        ])
        if in_p is None or tgt_p is None:
            raise FileNotFoundError(f"Missing DEV-Legacy-48 case files for {row['id']} (v2_data_root={dr})")
        cases.append(
            {
                "id": row["id"],
                "reference_id": row["reference_id"],
                "role": row.get("role", row.get("profile", "unknown")),
                "input_path": in_p,
                "target_path": tgt_p,
                "observed_path": None,
                "paired": True,
            }
        )
    return cases


def load_dev_syn(manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
    mp = Path(manifest_path)
    base = mp.parent
    data = json.loads(mp.read_text(encoding="utf-8"))
    cases = []
    for row in data["cases"]:
        cases.append(
            {
                "id": row["id"],
                "reference_id": row["reference_id"],
                "role": row["regime"],
                "input_path": base / row["input"],
                "target_path": base / row["target"],
                "observed_path": (base / row["observed"]) if row.get("observed") else None,
                "paired": True,
            }
        )
    return cases


def load_dev_native_cctv(
    qmul_json: Union[str, Path] = ROOT / "outputs/cctv_native_development_v2/frozen_subset.json",
    chokepoint_json: Union[str, Path] = ROOT / "outputs/cctv_chokepoint_native_development_v1/frozen_subset.json",
    qmul_crops_dir: Optional[Union[str, Path]] = None,
    chokepoint_crops_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Load ONLY role=='development' native CCTV cases (NEVER reserved_evaluation)."""
    cases: List[Dict[str, Any]] = []

    qp = Path(qmul_json)
    if qp.is_file():
        qdata = json.loads(qp.read_text(encoding="utf-8"))
        qdir = qp.parent
        qcd = Path(qmul_crops_dir) if qmul_crops_dir is not None else None
        for row in qdata.get("cases", []):
            if row.get("role") != "development":
                continue
            cid = row["id"]
            src_name = Path(row.get("source_file", "")).name
            cand_in = _resolve_file([
                (qcd / cid / "input.png") if qcd else None,
                (qcd / f"{cid}_input.png") if qcd else None,
                (qcd / "inputs" / f"{cid}_input.png") if qcd else None,
                (qcd / f"{cid}.png") if qcd else None,
                (qcd / src_name) if (qcd and src_name) else None,
                (qcd / row["source_file"]) if (qcd and row.get("source_file")) else None,
                qdir / "inputs" / f"{cid}_input.png",
                qdir / row["source_file"] if row.get("source_file") else None,
            ])
            cand_obs = _resolve_file([
                (qcd / cid / "observed.png") if qcd else None,
                (qcd / f"{cid}_observed.png") if qcd else None,
                (qcd / "observed" / f"{cid}_observed.png") if qcd else None,
                qdir / "observed" / f"{cid}_observed.png",
            ])
            if cand_in is not None:
                cases.append(
                    {
                        "id": f"qmul_{cid}",
                        "reference_id": f"qmul_{row['global_person_id']}",
                        "role": f"qmul_{row.get('size_bin', 'dev')}",
                        "input_path": cand_in,
                        "target_path": None,
                        "observed_path": cand_obs,
                        "paired": False,
                    }
                )

    cp = Path(chokepoint_json)
    if cp.is_file():
        cdata = json.loads(cp.read_text(encoding="utf-8"))
        cdir = cp.parent
        ccd = Path(chokepoint_crops_dir) if chokepoint_crops_dir is not None else None
        for row in cdata.get("cases", []):
            if row.get("role") != "development":
                continue
            in_rel = row.get("input", "")
            in_name = Path(in_rel).name if in_rel else f"{row['id']}.png"
            obs_rel = row.get("observed", "")
            obs_name = Path(obs_rel).name if obs_rel else f"{row['id']}.png"
            in_path = _resolve_file([
                (ccd / row["id"] / "input.png") if ccd else None,
                (ccd / in_name) if ccd else None,
                (ccd / in_rel) if (ccd and in_rel) else None,
                (ccd / "inputs" / in_name) if ccd else None,
                cdir / in_rel if in_rel else None,
            ])
            obs_path = _resolve_file([
                (ccd / row["id"] / "observed.png") if ccd else None,
                (ccd / "observed" / obs_name) if ccd else None,
                (ccd / obs_rel) if (ccd and obs_rel) else None,
                cdir / obs_rel if obs_rel else None,
            ])
            if in_path is not None:
                cases.append(
                    {
                        "id": row["id"],
                        "reference_id": f"choke_{row['source_person_id']}",
                        "role": "chokepoint_dev",
                        "input_path": in_path,
                        "target_path": None,
                        "observed_path": obs_path,
                        "paired": False,
                    }
                )
    return cases


def load_app_test_dir(app_test_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load staged v0.2.14 Direct + Step 2 `test_images/` cohort from `manifest.json`.

    Supports both `{"cases": [...]}` and top-level `[...]` JSON formats, with optional
    aligned GT (`gt` / `target`) and observed mask (`observed` / `observed_mask`).
    """
    base = Path(app_test_dir)
    mp = base / "manifest.json"
    if not mp.is_file():
        raise FileNotFoundError(f"App test cohort manifest.json not found: {mp}")
    data = json.loads(mp.read_text(encoding="utf-8"))
    rows = data["cases"] if isinstance(data, dict) and "cases" in data else data
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported app test manifest schema in {mp}")

    cases: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        cid = str(row.get("case_id") or row.get("id") or f"app_case_{idx:02d}")
        in_rel = row.get("input") or row.get("input_path")
        if not in_rel:
            raise ValueError(f"Missing 'input' path for app test case {cid}")
        in_p = Path(in_rel) if Path(in_rel).is_absolute() else (base / in_rel)
        gt_rel = row.get("gt") if row.get("gt") is not None else row.get("target")
        gt_p = (Path(gt_rel) if Path(gt_rel).is_absolute() else (base / gt_rel)) if gt_rel else None
        obs_rel = row.get("observed") if row.get("observed") is not None else row.get("observed_mask")
        obs_p = (Path(obs_rel) if Path(obs_rel).is_absolute() else (base / obs_rel)) if obs_rel else None
        has_gt = bool(row.get("has_gt", gt_p is not None)) and (gt_p is not None and gt_p.is_file())
        role = str(row.get("role") or row.get("route") or ("with_gt" if has_gt else "no_gt"))
        cases.append(
            {
                "id": cid,
                "reference_id": str(row.get("reference_id") or cid),
                "role": role,
                "route": str(row.get("route") or ""),
                "input_path": in_p,
                "target_path": gt_p if has_gt else None,
                "observed_path": obs_p if (obs_p is not None and obs_p.is_file()) else None,
                "paired": has_gt,
            }
        )
    return cases


# ---------------------------------------------------------------------------
# Model Runners (returns raw_f32 (256, 256, 3) in [0, 1])
# ---------------------------------------------------------------------------


def make_codeformer_runner(
    codeformer_path: Union[str, Path],
    fidelity: float = 1.0,
    device: str = "cpu",
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from pretrained_face_restoration import load_face_restorer

    net, _ = load_face_restorer(Path(codeformer_path), device=device)

    @torch.no_grad()
    def _run(canvas_u8: np.ndarray, observed: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(canvas_u8.transpose(2, 0, 1).copy()).float()[None].to(device) / 255.0
        out = net(t, fidelity=fidelity)[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        return np.clip(out, 0.0, 1.0)

    return _run


def make_v2_runner(
    v2_weights_dir: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    app_root = str(ROOT / "integrated_app")
    if app_root not in sys.path:
        sys.path.insert(0, app_root)
    from implementation import conditioning_v2 as cond_v2
    from implementation import runtime as rt
    from implementation.runtime import SITES, load_frozen_cpu_restorer
    from torch.nn import functional as F

    # Isolate integrated_app/third_party when present so runtime.py's exact
    # architecture origin check succeeds even after top-level third_party was imported.
    rt_root = Path(rt.__file__).resolve().parents[1]
    saved_tp: Dict[str, Any] = {}
    swapped = False
    if (rt_root / "third_party").is_dir():
        swapped = True
        for k in list(sys.modules.keys()):
            if k == "third_party" or k.startswith("third_party."):
                saved_tp[k] = sys.modules.pop(k)
        if sys.path[0] != str(rt_root):
            sys.path.insert(0, str(rt_root))
    try:
        if v2_weights_dir is not None:
            net = load_frozen_cpu_restorer(model_dir=Path(v2_weights_dir))
        else:
            net = load_frozen_cpu_restorer()
    finally:
        if swapped:
            for k in list(sys.modules.keys()):
                if k == "third_party" or k.startswith("third_party."):
                    sys.modules.pop(k, None)
            sys.modules.update(saved_tp)

    dev = torch.device(device)
    if dev.type != "cuda":
        @torch.no_grad()
        def _run_cpu(canvas_u8: np.ndarray, observed: np.ndarray) -> np.ndarray:
            raw = net.restore_canvas(canvas_u8)
            return np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)

        return _run_cpu

    # Move verified SHA-256-checked frozen V2 modules to CUDA in strict FP32
    net.base.to(dev).float().eval()
    net.adapters.to(dev).float().eval()
    net.heads.to(dev).float().eval()
    net.v2_bundle.to(dev).float().eval()

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 256, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, 256, dtype=torch.float32),
        indexing="ij",
    )
    r2_grid = (xx / 0.64) ** 2 + ((yy - 0.08) / 0.70) ** 2
    face_raw = torch.clamp(1.28 - r2_grid, 0.0, 1.0)[None, None]
    for _ in range(2):
        face_raw = F.avg_pool2d(F.pad(face_raw, (7, 7, 7, 7), mode="reflect"), kernel_size=15, stride=1)
    fmask_np = face_raw[0].detach().cpu().numpy().transpose(1, 2, 0).copy()

    def _gauss_blur(z: torch.Tensor, k: int) -> torch.Tensor:
        p = k // 2
        out = F.avg_pool2d(F.pad(z, (p, p, p, p), mode="reflect"), kernel_size=k, stride=1)
        out = F.avg_pool2d(F.pad(out, (p, p, p, p), mode="reflect"), kernel_size=k, stride=1)
        return out

    @torch.no_grad()
    def _run_cuda(canvas_u8: np.ndarray, observed: np.ndarray) -> np.ndarray:
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                inp_f32 = canvas_u8.astype(np.float32) / np.float32(255.0)
                inp_t = (
                    torch.from_numpy(canvas_u8.transpose(2, 0, 1).copy())
                    .to(device=dev, dtype=torch.float32)[None]
                    / 255.0
                )
                sample_512 = F.interpolate(inp_t, (512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0
                quant_feat, enc_feat_dict, _, _ = net._encode_and_quantize(sample_512)
                u96_dict = {s: net.adapters[s](enc_feat_dict[s].detach()) for s in SITES}
                u64_raw = net._decode_sft(quant_feat, u96_dict)[0].detach().cpu().numpy().transpose(1, 2, 0).copy()
                cond_t = cond_v2.compute_v2_conditioning_torch(inp_t, torch=torch)
                used_p2 = net.v2_bundle.refine_sft_features(u96_dict, cond_t)
                p2_t = net._decode_sft(quant_feat, used_p2, cond_t=cond_t, use_head_v2=True)
                p2_raw = p2_t[0].detach().cpu().numpy().transpose(1, 2, 0).copy()
                p2_c, cond_info = cond_v2.condition_output_v2_numpy(
                    p2_raw,
                    inp_f32,
                    np=np,
                    baseline_rgb=u64_raw,
                    beta=cond_v2.DEFAULT_ANCHOR_BETA,
                    gamma=net.v2_gamma,
                    dark_drift_align=True,
                )
                m256 = cond_t["site_maps"]["256"]
                out_iso = cond_info["outer_ring_laplacian"]
                w_global = min(1.0, max(0.0, (0.0130 - out_iso) / 0.0025))
                if float(m256.max().item()) > 0.0 and w_global > 0.0:
                    g_dark = cond_info["g_dark"]
                    g_blur = cond_info["g_blur"]
                    p2_lap = cond_v2.central_laplacian_sharpness_numpy(p2_c, np=np)
                    s_in = min(1.0, max(0.0, (0.0052 - cond_info["min_directional_laplacian"]) / 0.0020))
                    s_out = min(1.0, max(0.0, (0.0155 - p2_lap) / 0.0045))
                    sev = max(s_in, s_out) if g_dark == 0.0 else min(1.0, max(0.0, (g_blur - 0.15) / 0.35))
                    a_usm = float(net.v2_gamma) * w_global * (0.05 + sev * (0.90 - 0.05))
                    a_vq = float(net.v2_gamma) * w_global * (0.08 + sev * (1.15 - 0.08))
                    vq_seed = u96_dict if g_dark > 0.0 else used_p2
                    vq_t = net._decode_sft(
                        quant_feat,
                        vq_seed,
                        w_by_site={"32": 0.50, "64": 0.30, "128": 0.08, "256": 0.04},
                    )
                    p2_c_t = torch.from_numpy(p2_c.transpose(2, 0, 1).copy()).to(device=dev, dtype=torch.float32)[None]
                    hp3_p2 = (p2_c_t - _gauss_blur(p2_c_t, 3))[0].detach().cpu().numpy().transpose(1, 2, 0).copy()
                    hp7_diff = (
                        ((vq_t - _gauss_blur(vq_t, 7)) - (p2_c_t - _gauss_blur(p2_c_t, 7)))[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .transpose(1, 2, 0)
                        .copy()
                    )
                    m_np = m256[0].detach().cpu().numpy().transpose(1, 2, 0).copy()
                    v2_raw = np.clip(
                        p2_c + m_np * (a_usm * hp3_p2 + a_vq * fmask_np * hp7_diff), 0.0, 1.0
                    ).astype(np.float32)
                    if g_dark > 0.0:
                        f64 = v2_raw.astype(np.float64)
                        b64 = u64_raw.astype(np.float64)
                        v2_raw = np.clip(
                            f64 - g_dark * cond_v2.smooth_coarse_256_numpy(f64 - b64, np=np), 0.0, 1.0
                        ).astype(np.float32)
                else:
                    v2_raw = p2_c
                return np.clip(np.asarray(v2_raw, dtype=np.float32), 0.0, 1.0)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    return _run_cuda


def make_v3_runner(
    model: CodeFormerV3Restorer,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    model.eval()

    @torch.no_grad()
    def _run(canvas_u8: np.ndarray, observed: np.ndarray) -> np.ndarray:
        raw = model.restore_canvas(canvas_u8, observed_bool=observed, return_delivered=False)
        return np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)

    return _run


# ---------------------------------------------------------------------------
# Core Evaluation Engine
# ---------------------------------------------------------------------------


def _load_canvas_and_observed(case: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    with Image.open(case["input_path"]) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != (256, 256):
        # Apply exact neutral-128 center pad + bilinear 256 if native crop is non-256
        h, w = rgb.shape[:2]
        side = max(h, w)
        top, left = (side - h) // 2, (side - w) // 2
        canvas = Image.new("RGB", (side, side), (128, 128, 128))
        canvas.paste(Image.fromarray(rgb), (left, top))
        rgb = np.asarray(canvas.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8)
        support = Image.new("L", (side, side), 0)
        support.paste(255, (left, top, left + w, top + h))
        obs = np.asarray(support.resize((256, 256), Image.Resampling.NEAREST)) > 0
    elif case.get("observed_path") and Path(case["observed_path"]).is_file():
        with Image.open(case["observed_path"]) as om:
            obs = np.asarray(om.convert("L")) > 0
    else:
        obs = np.ones((256, 256), dtype=bool)
    return rgb, obs


def evaluate_runner_on_cases(
    runner: Callable[[np.ndarray, np.ndarray], np.ndarray],
    cases: Sequence[Dict[str, Any]],
    lpips_model: Optional[torch.nn.Module] = None,
    arcface_model: Optional[DifferentiableArcFaceLoss] = None,
    device: Union[str, torch.device] = "cpu",
    save_outputs_for_sheet: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Evaluate one model runner on a cohort of paired or unpaired cases.

    Computes metrics for `raw_fp32`, `png_floor`, and `png_rint` views!
    """
    views = ("raw_fp32", "png_floor", "png_rint")
    per_view_all: Dict[str, List[Dict[str, Any]]] = {v: [] for v in views}
    per_view_role: Dict[str, Dict[str, List[Dict[str, Any]]]] = {v: defaultdict(list) for v in views}
    saved_imgs: Dict[str, np.ndarray] = {}

    # For unpaired native CCTV same-ID ArcFace verification
    unpaired_embs: Dict[str, List[np.ndarray]] = defaultdict(list)

    for case in cases:
        canvas_u8, obs = _load_canvas_and_observed(case)
        t0 = time.monotonic()
        raw_pred = runner(canvas_u8, obs)
        elapsed = time.monotonic() - t0

        raw_floor, png_floor = delivered_views_v3(raw_pred, canvas_u8, obs, rounding="floor")
        _, png_rint = delivered_views_v3(raw_pred, canvas_u8, obs, rounding="rint")

        unobs_exact_floor = bool((png_floor[~obs] == canvas_u8[~obs]).all()) if (~obs).any() else True
        unobs_exact_rint = bool((png_rint[~obs] == canvas_u8[~obs]).all()) if (~obs).any() else True

        if save_outputs_for_sheet:
            saved_imgs[case["id"]] = png_floor

        if case["paired"] and case.get("target_path"):
            with Image.open(case["target_path"]) as tm:
                target_u8 = np.asarray(tm.convert("RGB"), dtype=np.uint8)
            target_f64 = target_u8.astype(np.float64) / 255.0
            target_f32 = target_u8.astype(np.float32) / 255.0
            input_f32 = canvas_u8.astype(np.float32) / 255.0

            # Cache per-case GT and input ArcFace embeddings once across all 3 views
            cached_ref_emb = None
            cached_input_cos = None
            if arcface_model is not None:
                with torch.no_grad():
                    t_t = torch.from_numpy(target_f32.transpose(2, 0, 1).copy()).float()[None].to(device)
                    in_t = torch.from_numpy(input_f32.transpose(2, 0, 1).copy()).float()[None].to(device)
                    m_t = torch.from_numpy(obs.astype(np.float32))[None, None].to(device)
                    ref_embs, valid_ref = arcface_model.embed(t_t)
                    if valid_ref.any():
                        cached_ref_emb = ref_embs[0]
                        in_clean = in_t * m_t + t_t * (1.0 - m_t)
                        in_embs, valid_in = arcface_model.embed(in_clean)
                        if valid_in.any():
                            cached_input_cos = float((in_embs[0] * cached_ref_emb).sum().clamp(-1.0, 1.0).item())

            view_arrays = {
                "raw_fp32": (raw_floor.astype(np.float64), raw_floor.astype(np.float32), unobs_exact_floor),
                "png_floor": (png_floor.astype(np.float64) / 255.0, png_floor.astype(np.float32) / 255.0, unobs_exact_floor),
                "png_rint": (png_rint.astype(np.float64) / 255.0, png_rint.astype(np.float32) / 255.0, unobs_exact_rint),
            }
            for vname, (arr_f64, arr_f32, exact_flag) in view_arrays.items():
                pm = compute_paired_pixel_metrics(target_f64, arr_f64, observed=obs)
                pid = compute_perceptual_and_id_metrics(
                    target_f32,
                    arr_f32,
                    input_f32=input_f32,
                    observed=obs,
                    lpips_model=lpips_model,
                    arcface_model=arcface_model,
                    device=device,
                    cached_ref_emb=cached_ref_emb,
                    cached_input_cos=cached_input_cos,
                )
                rec = {
                    "id": case["id"],
                    "role": case["role"],
                    **pm,
                    **pid,
                    "unobserved_exact": exact_flag,
                    "cpu_latency_s": elapsed,
                }
                per_view_all[vname].append(rec)
                per_view_role[vname][case["role"]].append(rec)
        else:
            # Unpaired native CCTV / no-GT app test images
            f_floor = png_floor.astype(np.float64) / 255.0
            f_rint = png_rint.astype(np.float64) / 255.0
            f_raw = raw_floor.astype(np.float64)
            lap_floor = central_laplacian_energy(f_floor)
            lap_rint = central_laplacian_energy(f_rint)
            lap_raw = central_laplacian_energy(f_raw)
            noise_floor = estimate_noise_sigma(f_floor)
            noise_rint = estimate_noise_sigma(f_rint)
            noise_raw = estimate_noise_sigma(f_raw)
            if arcface_model is not None:
                p_t = torch.from_numpy(png_floor.transpose(2, 0, 1).copy()).float()[None].to(device) / 255.0
                with torch.no_grad():
                    emb, valid = arcface_model.embed(p_t)
                if valid.any():
                    unpaired_embs[case["reference_id"]].append(emb[0].detach().cpu().numpy())

            for vname, lap_val, noise_val, exact_flag in (
                ("raw_fp32", lap_raw, noise_raw, unobs_exact_floor),
                ("png_floor", lap_floor, noise_floor, unobs_exact_floor),
                ("png_rint", lap_rint, noise_rint, unobs_exact_rint),
            ):
                rec = {
                    "id": case["id"],
                    "role": case["role"],
                    "central_laplacian": lap_val,
                    "noise_sigma": noise_val,
                    "unobserved_exact": exact_flag,
                    "cpu_latency_s": elapsed,
                }
                per_view_all[vname].append(rec)
                per_view_role[vname][case["role"]].append(rec)

    summary: Dict[str, Any] = {}
    for vname in views:
        summary[vname] = {
            "overall": aggregate_records(per_view_all[vname]),
            "by_role": {r: aggregate_records(recs) for r, recs in sorted(per_view_role[vname].items())},
        }

    # Same-ID cosine on unpaired Native CCTV (e.g., ChokePoint multi-frame pairs)
    same_id_cosines: List[float] = []
    for ref_id, embs in unpaired_embs.items():
        if len(embs) >= 2:
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    same_id_cosines.append(float(np.dot(embs[i], embs[j])))
    if same_id_cosines:
        summary["same_id_arcface_cos_mean"] = float(np.mean(same_id_cosines))
        summary["same_id_pair_count"] = len(same_id_cosines)

    return summary, saved_imgs


def evaluate_v3_on_manifest(
    model: CodeFormerV3Restorer,
    manifest_path: Union[str, Path],
    lpips_model: Optional[torch.nn.Module] = None,
    arcface_model: Optional[DifferentiableArcFaceLoss] = None,
    device: Union[str, torch.device] = "cpu",
    max_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """Convenience entry point used by `train_v3.py` (including `--stage soup`)."""
    cases = load_dev_syn(manifest_path)
    if max_cases is not None:
        cases = cases[: int(max_cases)]
    runner = make_v3_runner(model)
    summary, _ = evaluate_runner_on_cases(
        runner, cases, lpips_model=lpips_model, arcface_model=arcface_model, device=device
    )
    return summary


# ---------------------------------------------------------------------------
# Visual Contact Sheets & Markdown Table Rendering
# ---------------------------------------------------------------------------


def save_contact_sheet(
    cases: Sequence[Dict[str, Any]],
    model_outputs: Dict[str, Dict[str, np.ndarray]],
    out_path: Union[str, Path],
    max_rows: int = 12,
) -> Path:
    """Save side-by-side visual contact sheet: Input | <models...> | GT (if paired)."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    selected = list(cases[:max_rows])
    if not selected:
        return out_p

    model_names = list(model_outputs.keys())
    has_gt = any(c.get("paired") and c.get("target_path") for c in selected)
    col_labels = ["Input"] + model_names + (["GT"] if has_gt else [])

    tile = 256
    header_h = 28
    w = tile * len(col_labels)
    h = header_h + tile * len(selected)
    sheet = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    for c_idx, label in enumerate(col_labels):
        draw.text((c_idx * tile + 8, 6), label, fill=(240, 240, 240))

    for r_idx, case in enumerate(selected):
        y = header_h + r_idx * tile
        canvas_u8, _ = _load_canvas_and_observed(case)
        sheet.paste(Image.fromarray(canvas_u8), (0, y))
        for m_idx, mname in enumerate(model_names):
            img_arr = model_outputs[mname].get(case["id"])
            if img_arr is not None:
                sheet.paste(Image.fromarray(img_arr), ((m_idx + 1) * tile, y))
        if has_gt and case.get("target_path"):
            with Image.open(case["target_path"]) as tm:
                gt_im = tm.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            sheet.paste(gt_im, ((len(model_names) + 1) * tile, y))

    sheet.save(out_p)
    return out_p


def save_app_test_contact_sheet(
    cases: Sequence[Dict[str, Any]],
    model_outputs: Dict[str, Dict[str, np.ndarray]],
    out_path: Union[str, Path],
    max_rows: int = 20,
) -> Path:
    """Save app test contact sheet with BOTH full 256px AND 2x center-face [64:192, 64:192] eye zoom
    across `Input | <models...> | GT`.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    selected = list(cases[:max_rows])
    if not selected:
        return out_p

    model_names = list(model_outputs.keys())
    has_gt = any(c.get("paired") and c.get("target_path") for c in selected)
    col_labels = ["Input"] + model_names + (["GT"] if has_gt else [])

    tile = 256
    header_h = 28
    row_h = tile * 2  # top: full 256px, bottom: 2x center-face [64:192, 64:192] eye zoom
    w = tile * len(col_labels)
    h = header_h + row_h * len(selected)
    sheet = Image.new("RGB", (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    for c_idx, label in enumerate(col_labels):
        draw.text((c_idx * tile + 8, 6), label, fill=(240, 240, 240))

    def _eye_zoom(arr_u8: np.ndarray) -> Image.Image:
        crop = arr_u8[64:192, 64:192]
        return Image.fromarray(crop).resize((tile, tile), Image.Resampling.NEAREST)

    for r_idx, case in enumerate(selected):
        y_full = header_h + r_idx * row_h
        y_zoom = y_full + tile
        canvas_u8, _ = _load_canvas_and_observed(case)
        sheet.paste(Image.fromarray(canvas_u8), (0, y_full))
        sheet.paste(_eye_zoom(canvas_u8), (0, y_zoom))
        draw.rectangle([0, y_full, tile - 1, y_full + 16], fill=(0, 0, 0))
        draw.text((4, y_full + 2), f"{case['id'][:30]} [256]", fill=(255, 255, 160))
        draw.rectangle([0, y_zoom, tile - 1, y_zoom + 16], fill=(0, 0, 0))
        draw.text((4, y_zoom + 2), f"{case['id'][:26]} [2x eye]", fill=(160, 255, 255))

        for m_idx, mname in enumerate(model_names):
            img_arr = model_outputs[mname].get(case["id"])
            if img_arr is not None:
                x_col = (m_idx + 1) * tile
                sheet.paste(Image.fromarray(img_arr), (x_col, y_full))
                sheet.paste(_eye_zoom(img_arr), (x_col, y_zoom))

        if has_gt:
            x_gt = (len(model_names) + 1) * tile
            if case.get("paired") and case.get("target_path"):
                with Image.open(case["target_path"]) as tm:
                    gt_arr = np.asarray(tm.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS), dtype=np.uint8)
                sheet.paste(Image.fromarray(gt_arr), (x_gt, y_full))
                sheet.paste(_eye_zoom(gt_arr), (x_gt, y_zoom))
            else:
                draw.text((x_gt + 80, y_full + 120), "NO GT", fill=(160, 160, 160))
                draw.text((x_gt + 80, y_zoom + 120), "NO GT", fill=(160, 160, 160))

    sheet.save(out_p)
    return out_p


def format_markdown_summary(results: Dict[str, Any]) -> str:
    """Format multi-cohort, multi-model, multi-view results as a GitHub Markdown report."""
    lines: List[str] = ["# CodeFormer Restoration V3 — Benchmark Summary", ""]
    for cohort_name, cohort_res in results.items():
        lines.append(f"## Cohort: `{cohort_name}`")
        lines.append("")
        lines.append(
            "| Model | View | MAE | PSNR (dB) | Acc (%) | SSIM7 | LPIPS | ArcFace | Lap Ratio | ID Flips | ~Obs Exact | Latency (s) |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|")
        for mname, mres in cohort_res.items():
            for view in ("raw_fp32", "png_floor", "png_rint"):
                ov = mres.get(view, {}).get("overall", {})
                if not ov or ov.get("count", 0) == 0:
                    continue

                def _fmt(val: Any, fmt: str = ".4f") -> str:
                    return "-" if val is None else format(val, fmt)

                lines.append(
                    f"| `{mname}` | `{view}` | {_fmt(ov.get('mae'), '.5f')} | {_fmt(ov.get('psnr_db'), '.2f')} "
                    f"| {_fmt(ov.get('accuracy_pct'), '.2f')} | {_fmt(ov.get('ssim7'), '.4f')} "
                    f"| {_fmt(ov.get('lpips'), '.4f')} | {_fmt(ov.get('arcface_cos'), '.4f')} "
                    f"| {_fmt(ov.get('central_laplacian_ratio'), '.3f')} | {ov.get('catastrophic_id_flips', 0)} "
                    f"| {'OK' if ov.get('unobserved_exact_all', True) else 'FAIL'} | {_fmt(ov.get('cpu_latency_s_mean'), '.2f')} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CodeFormer Restoration V3 Comprehensive Evaluator")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--codeformer-path", type=Path, default=ROOT / "outputs/codeformer_restoration_pretrained_v1/codeformer.pth")
    p.add_argument("--v3-npz", type=Path, default=None, help="Path to v3_modules_ema.npz")
    p.add_argument("--v2-weights-dir", type=Path, default=None, help="Optional directory for V2 model weights")
    p.add_argument("--v2-manifest", type=Path, default=ROOT / "outputs/codeformer_conditioned_sft_learning_v2_preparation/v2_data_manifest.json")
    p.add_argument("--v2-data-root", type=Path, default=None, help="Root directory for DEV-Legacy-48 inputs/targets")
    p.add_argument("--qmul-json", type=Path, default=ROOT / "outputs/cctv_native_development_v2/frozen_subset.json")
    p.add_argument("--qmul-crops-dir", type=Path, default=None, help="Directory with QMUL-SurvFace DEV crops")
    p.add_argument("--chokepoint-json", type=Path, default=ROOT / "outputs/cctv_chokepoint_native_development_v1/frozen_subset.json")
    p.add_argument("--chokepoint-crops-dir", type=Path, default=None, help="Directory with ChokePoint DEV crops")
    p.add_argument("--perceptual-dir", type=Path, default=ROOT / "outputs/codeformer_perceptual_assets_v1")
    p.add_argument("--arcface-onnx", type=Path, default=Path("~/.insightface/models/buffalo_l/w600k_r50.onnx").expanduser())
    p.add_argument("--dev-syn-manifest", type=Path, default=None)
    p.add_argument(
        "--app-test-dir",
        type=Path,
        default=None,
        help="Optional staged v0.2.14 Direct + Step 2 test_images/ directory with manifest.json",
    )
    p.add_argument("--arch-variant", choices=["v30", "v33", "v34"], default=None, help="Optional explicit V3 arch_variant (auto-detected from --v3-npz)")
    p.add_argument("--models", nargs="+", default=["cf_w10", "cf_w07", "v2", "v3"])
    p.add_argument("--cohorts", nargs="+", default=["dev_legacy_48", "dev_syn", "dev_native_cctv"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--no-lpips", action="store_true", help="Explicitly disable LPIPS evaluation")
    p.add_argument("--no-arcface", action="store_true", help="Explicitly disable ArcFace evaluation")
    p.add_argument("--allow-random-base", action="store_true", help="Allow random CodeFormer base (unit tests only)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Fail-fast LPIPS & ArcFace checks unless explicitly disabled
    lpips_model = None
    if not args.no_lpips:
        if (
            args.perceptual_dir is None
            or not (args.perceptual_dir / "vgg16_features.pth").is_file()
            or not (args.perceptual_dir / "lpips_vgg_v01.pth").is_file()
        ):
            raise FileNotFoundError(
                f"LPIPS weights (vgg16_features.pth, lpips_vgg_v01.pth) missing in --perceptual-dir={args.perceptual_dir}. "
                "Pass --perceptual-dir or explicitly disable with --no-lpips."
            )
        lpips_model = load_frozen_lpips(args.perceptual_dir, device=args.device)

    arcface_model = None
    if not args.no_arcface:
        if args.arcface_onnx is None or not args.arcface_onnx.is_file():
            raise FileNotFoundError(
                f"ArcFace ONNX weights missing at --arcface-onnx={args.arcface_onnx}. "
                "Pass --arcface-onnx or explicitly disable with --no-arcface."
            )
        arcface_model = DifferentiableArcFaceLoss(onnx_path=args.arcface_onnx, device=args.device)

    needs_cf = any(m in ("cf_w10", "cf_w07", "v3") for m in args.models)
    if needs_cf and not args.allow_random_base:
        if args.codeformer_path is None or not args.codeformer_path.is_file():
            raise FileNotFoundError(
                f"Official CodeFormer weights not found at --codeformer-path={args.codeformer_path}."
            )

    runners: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {}
    for mname in args.models:
        if mname == "cf_w10":
            runners[mname] = make_codeformer_runner(args.codeformer_path, fidelity=1.0, device=args.device)
        elif mname == "cf_w07":
            runners[mname] = make_codeformer_runner(args.codeformer_path, fidelity=0.7, device=args.device)
        elif mname == "v2":
            runners[mname] = make_v2_runner(v2_weights_dir=args.v2_weights_dir, device=args.device)
        elif mname == "v3":
            cf_path = args.codeformer_path if (args.codeformer_path and args.codeformer_path.is_file()) else None
            if args.arch_variant is not None:
                v3_variant = args.arch_variant
            elif args.v3_npz is not None:
                v3_variant = detect_v3_npz_variant(args.v3_npz)
            else:
                v3_variant = "v30"
            v3_model = CodeFormerV3Restorer(codeformer_weights_path=cf_path, arch_variant=v3_variant).to(args.device)
            if args.v3_npz is not None:
                load_v3_npz(args.v3_npz, model=v3_model)
            runners[mname] = make_v3_runner(v3_model)
        else:
            raise ValueError(f"Unknown model arm: {mname}")

    cohorts: Dict[str, List[Dict[str, Any]]] = {}
    if "dev_legacy_48" in args.cohorts:
        cohorts["dev_legacy_48"] = load_dev_legacy_48(
            manifest_path=args.v2_manifest,
            v2_data_root=args.v2_data_root,
        )
    if "dev_syn" in args.cohorts:
        if args.dev_syn_manifest is None or not args.dev_syn_manifest.is_file():
            raise FileNotFoundError(f"Requested cohort 'dev_syn' requires valid --dev-syn-manifest (got {args.dev_syn_manifest})")
        cohorts["dev_syn"] = load_dev_syn(args.dev_syn_manifest)
    if "dev_native_cctv" in args.cohorts:
        cohorts["dev_native_cctv"] = load_dev_native_cctv(
            qmul_json=args.qmul_json,
            chokepoint_json=args.chokepoint_json,
            qmul_crops_dir=args.qmul_crops_dir,
            chokepoint_crops_dir=args.chokepoint_crops_dir,
        )
    if args.app_test_dir is not None or "app_test_images" in args.cohorts:
        if args.app_test_dir is None or not (args.app_test_dir / "manifest.json").is_file():
            raise FileNotFoundError(
                f"Requested cohort 'app_test_images' requires valid --app-test-dir with manifest.json (got {args.app_test_dir})"
            )
        cohorts["app_test_images"] = load_app_test_dir(args.app_test_dir)

    for cname, cases in cohorts.items():
        if not cases:
            raise RuntimeError(
                f"Explicitly requested cohort '{cname}' loaded 0 cases! "
                "Verify manifest and crop directory CLI paths."
            )

    all_results: Dict[str, Any] = {}
    for cname, cases in cohorts.items():
        if args.max_cases is not None:
            cases = cases[: args.max_cases]
        if not cases:
            continue
        all_results[cname] = {}
        sheet_outputs: Dict[str, Dict[str, np.ndarray]] = {}
        for mname, runner in runners.items():
            summary, imgs = evaluate_runner_on_cases(
                runner,
                cases,
                lpips_model=lpips_model,
                arcface_model=arcface_model,
                device=args.device,
                save_outputs_for_sheet=True,
            )
            all_results[cname][mname] = summary
            sheet_outputs[mname] = imgs

        if cname == "app_test_images":
            save_app_test_contact_sheet(cases, sheet_outputs, args.out_dir / "contact_sheets" / "app_test_images.png")
        else:
            save_contact_sheet(cases, sheet_outputs, args.out_dir / "contact_sheets" / f"{cname}.png")

    (args.out_dir / "summary.json").write_text(
        json.dumps(all_results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "summary.md").write_text(format_markdown_summary(all_results), encoding="utf-8")


if __name__ == "__main__":
    main()
