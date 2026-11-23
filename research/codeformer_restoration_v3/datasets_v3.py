"""CodeFormer Restoration V3 — Zero-Leakage Split Guard, High-Throughput Datasets & DEV-Syn Generator.

Guarantees:
1. Strict Zero-Leakage Split Guard asserting ZERO overlap between:
   - TRAIN IDs (subset of split['train'])
   - DEV-Syn unseen IDs (subset of split['validation'])
   - DEV-Legacy-48 (monitor_v2, 8 IDs)
   - V1 monitors / old guards (protocol.json validation + evaluation_bundle_v1)
   - All 45 reserved FINAL CCTV IDs (32 QMUL-SurvFace + 13 ChokePoint) & 36 Native CCTV DEV IDs
2. High-throughput PyTorch Dataset reading 512x512 HQ faces from local directories,
   indexed WebDataset .tar shards, .zip archives, or LMDB with num_workers=4,
   pin_memory=True, persistent_workers=True.
3. StratifiedRoleBatchSampler guaranteeing every batch balances `clear`,
   `normal_degraded`, and `lowlight_joint` regimes (eliminating dark gradient dominance).
4. Deterministic frozen DEV-Syn-240 generator (N unseen val IDs x 12 regimes) saved as
   deterministic PNGs + JSON manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

# Canonical 12 V3 CCTV regimes matching #333
REGIMES_12: Tuple[str, ...] = (
    "clear",
    "global_defocus_blur",
    "directional_motion_blur",
    "local_partial_blur",
    "compound_cctv_codec",
    "severe_lr_survface",
    "lowlight_sensor_noise",
    "joint_lowlight_blur_codec",
    "daylight_sensor_noise",
    "h264_macroblock_interlace",
    "haze_colorcast_glare",
    "mild_transition",
)

REGIME_GROUPS: Dict[str, Tuple[str, ...]] = {
    "clear": ("clear",),
    "normal_degraded": (
        "global_defocus_blur",
        "directional_motion_blur",
        "local_partial_blur",
        "compound_cctv_codec",
        "severe_lr_survface",
        "daylight_sensor_noise",
        "h264_macroblock_interlace",
        "haze_colorcast_glare",
        "mild_transition",
    ),
    "lowlight_joint": (
        "lowlight_sensor_noise",
        "joint_lowlight_blur_codec",
    ),
}

# Exact 8 monitor_v2 IDs from v2_data_manifest.json (fallback if manifest omitted)
DEFAULT_MONITOR_V2_IDS: Tuple[str, ...] = (
    "ffhq_08698",
    "ffhq_02147",
    "ffhq_05179",
    "ffhq_08949",
    "ffhq_11039",
    "ffhq_11154",
    "ffhq_08064",
    "ffhq_08094",
)

FORBIDDEN_TRAIN_PATH_TOKENS: Tuple[str, ...] = (
    "chokepoint",
    "survface",
    "qmul",
    "reserved_evaluation",
    "final_holdout",
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_ASIAN_RE = re.compile(r"(?:asian_face_|(?:tr|va|val)_asian_|asian_)(\d{5})", re.IGNORECASE)
_FFHQ_EXPLICIT_RE = re.compile(r"(?:(?:tr|va|val)_ffhq_|ffhq_|thumbnails128x128/|images1024/)(\d{5})", re.IGNORECASE)


def canonicalize_face_id(identifier: str) -> str:
    """Normalize FFHQ / Asian / CCTV identifiers into canonical namespaced IDs."""
    s = str(identifier).replace("\\", "/").strip()
    low = s.lower()
    if low.startswith("qmul-survface/") or low.startswith("chokepoint/"):
        return s
    m_asian = _ASIAN_RE.search(low)
    if m_asian:
        return f"asian_{int(m_asian.group(1)):05d}"
    m_ffhq = _FFHQ_EXPLICIT_RE.search(low)
    if m_ffhq:
        return f"ffhq_{int(m_ffhq.group(1)):05d}"
    stem = Path(s).stem
    if stem.isdigit() and len(stem) <= 5:
        return f"ffhq_{int(stem):05d}"
    return stem


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Strict Zero-Leakage Split Guard
# ---------------------------------------------------------------------------


@dataclass
class SplitGuardReport:
    train_pool_count: int
    val_pool_count: int
    monitor_v2_count: int
    protocol_val_count: int
    v1_guard_count: int
    native_cctv_dev_count: int
    reserved_final_cctv_count: int
    excluded_from_train_count: int


class ZeroLeakageSplitGuard:
    """Loads all canonical split/protocol/monitor/FINAL manifests and enforces zero overlap."""

    @staticmethod
    def _resolve_manifest(
        explicit: Optional[Union[str, Path]],
        manifests_dir: Optional[Path],
        vm_rel: str,
        local_rel: str,
        root: Path,
    ) -> Path:
        if explicit is not None:
            return Path(explicit)
        if manifests_dir is not None:
            cand = manifests_dir / vm_rel
            if cand.is_file():
                return cand
        cand_vm = root / "manifests" / vm_rel
        if cand_vm.is_file():
            return cand_vm
        return root / local_rel

    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        manifests_dir: Optional[Union[str, Path]] = None,
        split_json: Optional[Union[str, Path]] = None,
        protocol_json: Optional[Union[str, Path]] = None,
        v2_manifest_json: Optional[Union[str, Path]] = None,
        eval_bundle_v1_json: Optional[Union[str, Path]] = None,
        qmul_subset_json: Optional[Union[str, Path]] = None,
        chokepoint_subset_json: Optional[Union[str, Path]] = None,
        strict_files: bool = False,
    ):
        root = Path(workspace_root) if workspace_root is not None else Path(__file__).resolve().parents[2]
        mdir = Path(manifests_dir) if manifests_dir is not None else None
        self.root = root

        split_path = self._resolve_manifest(
            split_json, mdir, "split.json",
            "outputs/downloaded_phase4/outputs/phase4_with_progress/split.json", root
        )
        proto_path = self._resolve_manifest(
            protocol_json, mdir, "cctv_dgp_vm_bundle_protocol.json",
            "outputs/cctv_dgp_vm_bundle_v1/protocol.json", root
        )
        v2_path = self._resolve_manifest(
            v2_manifest_json, mdir, "v2_data_manifest.json",
            "outputs/codeformer_conditioned_sft_learning_v2_preparation/v2_data_manifest.json", root
        )
        eval_v1_path = self._resolve_manifest(
            eval_bundle_v1_json, mdir, "evaluation_bundle_v1.json",
            "outputs/codeformer_filter_fusion_training_v1_preparation/evaluation_bundle_v1.json", root
        )
        qmul_path = self._resolve_manifest(
            qmul_subset_json, mdir, "cctv_native_development_v2/frozen_subset.json",
            "outputs/cctv_native_development_v2/frozen_subset.json", root
        )
        choke_path = self._resolve_manifest(
            chokepoint_subset_json, mdir, "cctv_chokepoint_native_development_v1/frozen_subset.json",
            "outputs/cctv_chokepoint_native_development_v1/frozen_subset.json", root
        )

        self.raw_train_ids: Set[str] = set()
        self.val_ids: Set[str] = set()
        self.monitor_v2_ids: Set[str] = set(DEFAULT_MONITOR_V2_IDS)
        self.protocol_val_ids: Set[str] = set()
        self.v1_guard_ids: Set[str] = set()
        self.native_cctv_dev_ids: Set[str] = set()
        self.reserved_final_cctv_ids: Set[str] = set()

        # 1. Historical split.json (76,000 train / 4,000 validation)
        if split_path.is_file():
            split = json.loads(split_path.read_text(encoding="utf-8"))
            self.raw_train_ids = {canonicalize_face_id(p) for p in split.get("train", [])}
            self.val_ids = {canonicalize_face_id(p) for p in split.get("validation", [])}
            overlap_sv = self.raw_train_ids & self.val_ids
            if overlap_sv:
                raise AssertionError(f"split.json train/validation overlap detected: {len(overlap_sv)} IDs")
        elif strict_files:
            raise FileNotFoundError(f"Required split.json missing: {split_path}")

        # 2. protocol.json validation references (59 FFHQ + 51 Asian = 110)
        if proto_path.is_file():
            proto = json.loads(proto_path.read_text(encoding="utf-8"))
            for ref in proto.get("references", []):
                if ref.get("role") == "validation":
                    src = ref.get("source_file") or ref.get("id", "")
                    self.protocol_val_ids.add(canonicalize_face_id(src))
        elif strict_files:
            raise FileNotFoundError(f"Required protocol.json missing: {proto_path}")

        # 3. v2_data_manifest.json monitor_v2 (8 IDs)
        if v2_path.is_file():
            v2m = json.loads(v2_path.read_text(encoding="utf-8"))
            mon_order = v2m.get("monitor_reference_order", [])
            if mon_order:
                self.monitor_v2_ids = {canonicalize_face_id(r) for r in mon_order}
        elif strict_files:
            raise FileNotFoundError(f"Required v2_data_manifest.json missing: {v2_path}")

        # 4. evaluation_bundle_v1.json (v1_monitors / old_guards)
        if eval_v1_path.is_file():
            ev1 = json.loads(eval_v1_path.read_text(encoding="utf-8"))
            for case in ev1.get("cases", []):
                for field in ("original_id", "id"):
                    val = str(case.get(field, ""))
                    if "ffhq" in val.lower() or "asian" in val.lower():
                        self.v1_guard_ids.add(canonicalize_face_id(val))

        # 5. QMUL-SurvFace frozen_subset.json (24 DEV / 32 reserved FINAL)
        if qmul_path.is_file():
            qm = json.loads(qmul_path.read_text(encoding="utf-8"))
            for c in qm.get("cases", []):
                ns_id = f"QMUL-SurvFace/global/{c['global_person_id']}"
                if c.get("role") == "reserved_evaluation":
                    self.reserved_final_cctv_ids.add(ns_id)
                elif c.get("role") == "development":
                    self.native_cctv_dev_ids.add(ns_id)
        elif strict_files:
            raise FileNotFoundError(f"Required QMUL frozen_subset.json missing: {qmul_path}")

        # 6. ChokePoint frozen_subset.json (12 DEV / 13 reserved FINAL)
        if choke_path.is_file():
            cp = json.loads(choke_path.read_text(encoding="utf-8"))
            for c in cp.get("cases", []):
                ns_id = f"ChokePoint/P1/{c['source_person_id']}"
                if c.get("role") == "reserved_evaluation":
                    self.reserved_final_cctv_ids.add(ns_id)
                elif c.get("role") == "development":
                    self.native_cctv_dev_ids.add(ns_id)
        elif strict_files:
            raise FileNotFoundError(f"Required ChokePoint frozen_subset.json missing: {choke_path}")

        # Verify DEV vs FINAL CCTV disjointness
        cctv_overlap = self.native_cctv_dev_ids & self.reserved_final_cctv_ids
        if cctv_overlap:
            raise AssertionError(f"Native CCTV DEV/FINAL overlap detected: {sorted(cctv_overlap)[:5]}")

        # Compute clean excluded set and allowed train set
        self.excluded_from_train: Set[str] = (
            self.val_ids
            | self.monitor_v2_ids
            | self.protocol_val_ids
            | self.v1_guard_ids
            | self.native_cctv_dev_ids
            | self.reserved_final_cctv_ids
        )
        self.allowed_train_ids: Set[str] = self.raw_train_ids - self.excluded_from_train
        if strict_files:
            if not self.raw_train_ids or not self.val_ids or not self.reserved_final_cctv_ids:
                raise ValueError(
                    "ZeroLeakageSplitGuard strict_files=True requires non-empty raw_train_ids, "
                    f"val_ids, and reserved_final_cctv_ids (got {len(self.raw_train_ids)}, "
                    f"{len(self.val_ids)}, {len(self.reserved_final_cctv_ids)})."
                )

    def report(self) -> SplitGuardReport:
        return SplitGuardReport(
            train_pool_count=len(self.allowed_train_ids),
            val_pool_count=len(self.val_ids),
            monitor_v2_count=len(self.monitor_v2_ids),
            protocol_val_count=len(self.protocol_val_ids),
            v1_guard_count=len(self.v1_guard_ids),
            native_cctv_dev_count=len(self.native_cctv_dev_ids),
            reserved_final_cctv_count=len(self.reserved_final_cctv_ids),
            excluded_from_train_count=len(self.excluded_from_train),
        )

    def is_train_allowed(self, identifier: str, path_hint: str = "") -> bool:
        low_path = f"{identifier} {path_hint}".lower()
        if any(tok in low_path for tok in FORBIDDEN_TRAIN_PATH_TOKENS):
            return False
        cid = canonicalize_face_id(identifier)
        if cid in self.excluded_from_train:
            return False
        if self.raw_train_ids and (cid.startswith("ffhq_") or cid.startswith("asian_")):
            return cid in self.allowed_train_ids
        return True

    def is_dev_syn_candidate(self, identifier: str) -> bool:
        cid = canonicalize_face_id(identifier)
        if cid in self.monitor_v2_ids or cid in self.reserved_final_cctv_ids:
            return False
        if self.val_ids:
            return cid in self.val_ids
        return cid not in self.allowed_train_ids

    def assert_zero_leakage(
        self,
        train_ids: Iterable[str],
        dev_syn_ids: Optional[Iterable[str]] = None,
    ) -> None:
        """Hard assertion: ZERO overlap across TRAIN, DEV-Syn, DEV-Legacy-48, and FINAL."""
        tr_set = {canonicalize_face_id(x) for x in train_ids}
        bad_val = tr_set & self.val_ids
        bad_mon = tr_set & self.monitor_v2_ids
        bad_pvl = tr_set & self.protocol_val_ids
        bad_v1g = tr_set & self.v1_guard_ids
        bad_dev = tr_set & self.native_cctv_dev_ids
        bad_fin = tr_set & self.reserved_final_cctv_ids

        if bad_val:
            raise AssertionError(f"LEAKAGE: {len(bad_val)} split['validation'] IDs in TRAIN: {sorted(bad_val)[:5]}")
        if bad_mon:
            raise AssertionError(f"LEAKAGE: {len(bad_mon)} monitor_v2 IDs in TRAIN: {sorted(bad_mon)[:5]}")
        if bad_pvl:
            raise AssertionError(f"LEAKAGE: {len(bad_pvl)} protocol validation IDs in TRAIN: {sorted(bad_pvl)[:5]}")
        if bad_v1g:
            raise AssertionError(f"LEAKAGE: {len(bad_v1g)} v1 monitor/guard IDs in TRAIN: {sorted(bad_v1g)[:5]}")
        if bad_dev:
            raise AssertionError(f"LEAKAGE: {len(bad_dev)} native CCTV DEV IDs in TRAIN: {sorted(bad_dev)[:5]}")
        if bad_fin:
            raise AssertionError(f"LEAKAGE: {len(bad_fin)} reserved FINAL CCTV IDs in TRAIN: {sorted(bad_fin)[:5]}")

        if dev_syn_ids is not None:
            ds_set = {canonicalize_face_id(x) for x in dev_syn_ids}
            ov_td = tr_set & ds_set
            ov_dm = ds_set & self.monitor_v2_ids
            ov_df = ds_set & self.reserved_final_cctv_ids
            if ov_td:
                raise AssertionError(f"LEAKAGE: {len(ov_td)} DEV-Syn IDs overlap TRAIN: {sorted(ov_td)[:5]}")
            if ov_dm:
                raise AssertionError(f"LEAKAGE: {len(ov_dm)} DEV-Syn IDs overlap DEV-Legacy-48: {sorted(ov_dm)[:5]}")
            if ov_df:
                raise AssertionError(f"LEAKAGE: {len(ov_df)} DEV-Syn IDs overlap FINAL CCTV: {sorted(ov_df)[:5]}")


# ---------------------------------------------------------------------------
# Fast Multi-Backend Image Store (Directory / WebDataset .tar / .zip / LMDB)
# ---------------------------------------------------------------------------


class MultiBackendImageStore:
    """Worker-safe random-access reader for local dir, .tar shards, .zip, or LMDB."""

    def __init__(self, source_path: Union[str, Path]):
        self.source_path = Path(source_path)
        self.backend: str = ""
        # Each entry is (canonical_id, display_name, locator)
        self.entries: List[Tuple[str, str, Any]] = []
        self._handles: Dict[str, Any] = {}
        self._handles_pid: int = os.getpid()

        if self.source_path.is_dir():
            if (self.source_path / "data.mdb").is_file():
                self._index_lmdb()
            else:
                tars = sorted(self.source_path.glob("*.tar"))
                if tars:
                    self._index_tars(tars)
                else:
                    self._index_dir()
        elif self.source_path.is_file():
            suf = self.source_path.suffix.lower()
            if suf == ".tar":
                self._index_tars([self.source_path])
            elif suf == ".zip":
                self._index_zip()
            else:
                raise ValueError(f"Unsupported single-file dataset source: {self.source_path}")
        else:
            raise FileNotFoundError(f"Dataset path not found: {self.source_path}")

    def _index_dir(self) -> None:
        self.backend = "dir"
        for p in sorted(self.source_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rel = p.relative_to(self.source_path).as_posix()
                self.entries.append((canonicalize_face_id(rel), rel, str(p)))

    def _index_tars(self, tar_paths: Sequence[Path]) -> None:
        self.backend = "tar"
        for tp in tar_paths:
            with tarfile.open(tp, "r") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    ext = Path(member.name).suffix.lower()
                    if ext in IMAGE_EXTS:
                        cid = canonicalize_face_id(member.name)
                        # Store (tar_path, offset_data, size) for O(1) seek in workers
                        self.entries.append((cid, member.name, (str(tp), member.offset_data, member.size)))

    def _index_zip(self) -> None:
        self.backend = "zip"
        with zipfile.ZipFile(self.source_path, "r") as zf:
            for name in sorted(zf.namelist()):
                if Path(name).suffix.lower() in IMAGE_EXTS:
                    self.entries.append((canonicalize_face_id(name), name, (str(self.source_path), name)))

    def _index_lmdb(self) -> None:
        import lmdb

        self.backend = "lmdb"
        env = lmdb.open(str(self.source_path), readonly=True, lock=False, readahead=False)
        try:
            with env.begin(write=False) as txn:
                for key_bytes in txn.cursor().iternext(keys=True, values=False):
                    key = key_bytes.decode("utf-8", errors="ignore")
                    if key.startswith("__"):
                        continue
                    self.entries.append((canonicalize_face_id(key), key, key_bytes))
        finally:
            env.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def close(self) -> None:
        for handle in list(self._handles.values()):
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()

    def read_rgb(self, locator: Any) -> Image.Image:
        # Fork-safety on Linux: drop inherited parent handles without closing parent fds
        pid = os.getpid()
        if pid != self._handles_pid:
            self._handles = {}
            self._handles_pid = pid
        if self.backend == "dir":
            with Image.open(locator) as im:
                return im.convert("RGB")
        if self.backend == "tar":
            tar_path, offset, size = locator
            fh = self._handles.get(tar_path)
            if fh is None:
                fh = open(tar_path, "rb")
                self._handles[tar_path] = fh
            fh.seek(offset)
            raw = fh.read(size)
            with Image.open(io.BytesIO(raw)) as im:
                return im.convert("RGB")
        if self.backend == "zip":
            zip_path, member_name = locator
            zf = self._handles.get(zip_path)
            if zf is None:
                zf = zipfile.ZipFile(zip_path, "r")
                self._handles[zip_path] = zf
            raw = zf.read(member_name)
            with Image.open(io.BytesIO(raw)) as im:
                return im.convert("RGB")
        if self.backend == "lmdb":
            import lmdb

            env = self._handles.get("lmdb")
            if env is None:
                env = lmdb.open(str(self.source_path), readonly=True, lock=False, readahead=False)
                self._handles["lmdb"] = env
            with env.begin(write=False) as txn:
                raw = txn.get(locator)
            with Image.open(io.BytesIO(raw)) as im:
                return im.convert("RGB")
        raise RuntimeError(f"Unknown backend: {self.backend}")


# ---------------------------------------------------------------------------
# Degradation Adapter (connects seamlessly to degradations_v3.py)
# ---------------------------------------------------------------------------


def _default_degrade_adapter(
    gt_512_u8: np.ndarray,
    regime: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Invoke degradations_v3 if available, else apply a lightweight deterministic fallback."""
    try:
        from research.codeformer_restoration_v3 import degradations_v3 as deg_mod

        if hasattr(deg_mod, "degrade_image"):
            res = deg_mod.degrade_image(gt_512_u8, regime=regime, rng=rng)
            if isinstance(res, tuple) and len(res) >= 2:
                img_out, meta = res[0], res[1]
                targets = _extract_deg_targets(meta, regime)
                return img_out, targets
        if hasattr(deg_mod, "apply_regime"):
            res = deg_mod.apply_regime(gt_512_u8, regime=regime, rng=rng)
            if isinstance(res, tuple) and len(res) >= 2:
                return res[0], _extract_deg_targets(res[1], regime)
    except ImportError:
        pass

    # Built-in fallback for standalone/unit-test execution
    img = gt_512_u8.astype(np.float32) / 255.0
    exposure, blur, noise, codec_lr = 0.0, 0.0, 0.0, 0.0
    if regime == "clear":
        pass
    elif regime in ("lowlight_sensor_noise", "joint_lowlight_blur_codec"):
        gain = float(rng.uniform(0.22, 0.55))
        gamma = float(rng.uniform(1.15, 2.0))
        img = np.power(np.clip(img, 0.0, 1.0), gamma) * gain
        sigma = float(rng.uniform(0.008, 0.03))
        img = np.clip(img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32), 0.0, 1.0)
        exposure = float(np.clip(1.0 - gain, 0.0, 1.0))
        noise = float(np.clip(sigma / 0.05, 0.0, 1.0))
        if regime == "joint_lowlight_blur_codec":
            blur = 0.6
            codec_lr = 0.6
            pil_tmp = Image.fromarray(np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8))
            pil_tmp = pil_tmp.resize((48, 48), Image.Resampling.BILINEAR).resize((512, 512), Image.Resampling.BILINEAR)
            img = np.asarray(pil_tmp, dtype=np.float32) / 255.0
    elif regime == "daylight_sensor_noise":
        sigma = float(rng.uniform(0.015, 0.045))
        img = np.clip(img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32), 0.0, 1.0)
        noise = float(np.clip(sigma / 0.05, 0.0, 1.0))
    else:
        # Blur / LR / Codec / Haze fallback
        small = int(rng.integers(24, 96)) if "lr" in regime or "codec" in regime else int(rng.integers(96, 192))
        pil_tmp = Image.fromarray(np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8))
        pil_tmp = pil_tmp.resize((small, small), Image.Resampling.BILINEAR).resize((512, 512), Image.Resampling.BILINEAR)
        img = np.asarray(pil_tmp, dtype=np.float32) / 255.0
        blur = 0.5
        codec_lr = float(np.clip(1.0 - small / 256.0, 0.0, 1.0))

    out_u8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
    return out_u8, {"exposure": exposure, "blur": blur, "noise": noise, "codec_lr": codec_lr}


def _extract_deg_targets(meta: Any, regime: str) -> Dict[str, float]:
    if isinstance(meta, dict):
        if all(k in meta for k in ("exposure", "blur", "noise", "codec_lr")):
            return {k: float(meta[k]) for k in ("exposure", "blur", "noise", "codec_lr")}
        if "targets" in meta and isinstance(meta["targets"], dict):
            t = meta["targets"]
            return {
                "exposure": float(t.get("exposure", 0.0)),
                "blur": float(t.get("blur", 0.0)),
                "noise": float(t.get("noise", 0.0)),
                "codec_lr": float(t.get("codec_lr", t.get("codec", 0.0))),
            }
        return {
            "exposure": float(meta.get("exposure", 0.0)),
            "blur": float(meta.get("blur", 0.0)),
            "noise": float(meta.get("noise", 0.0)),
            "codec_lr": float(meta.get("codec_lr", meta.get("codec", 0.0))),
        }
    return {"exposure": 0.0, "blur": 0.0, "noise": 0.0, "codec_lr": 0.0}


def simulate_prepare_crop_pair(
    lq_u8: np.ndarray,
    gt_u8: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact live `prepare_crop` geometry simulation on both LQ and GT (50% of train batches).

    Applies matched random scale (0.75-1.15x), shift (±12%), rotation (±10°), and
    non-square aspect ratio (0.60-1.50), then neutral-128 center pad -> 256x256 bilinear
    + nearest observed_mask.
    """
    h, w = lq_u8.shape[:2]
    angle = float(rng.uniform(-10.0, 10.0))
    scale = float(rng.uniform(0.75, 1.15))
    ar = float(rng.uniform(0.60, 1.50))
    shift_x = float(rng.uniform(-0.12, 0.12)) * w
    shift_y = float(rng.uniform(-0.12, 0.12)) * h

    crop_w = int(np.clip(round(w / scale * math.sqrt(ar)), 64, w))
    crop_h = int(np.clip(round(h / scale / math.sqrt(ar)), 64, h))
    cx = w * 0.5 + shift_x
    cy = h * 0.5 + shift_y
    x0 = int(np.clip(round(cx - crop_w * 0.5), 0, max(0, w - crop_w)))
    y0 = int(np.clip(round(cy - crop_h * 0.5), 0, max(0, h - crop_h)))

    def _transform(arr: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(arr)
        if abs(angle) > 0.25:
            pil = pil.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
        return np.asarray(pil.crop((x0, y0, x0 + crop_w, y0 + crop_h)))

    lq_crop = _transform(lq_u8)
    gt_crop = _transform(gt_u8)

    ch, cw = lq_crop.shape[:2]
    side = max(ch, cw)
    top, left = (side - ch) // 2, (side - cw) // 2

    def _pad_and_resize(crop: np.ndarray) -> np.ndarray:
        canvas = Image.new("RGB", (side, side), (128, 128, 128))
        canvas.paste(Image.fromarray(crop), (left, top))
        return np.asarray(canvas.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.uint8)

    support = Image.new("L", (side, side), 0)
    support.paste(255, (left, top, left + cw, top + ch))
    observed = np.asarray(support.resize((256, 256), Image.Resampling.NEAREST)) > 0

    return _pad_and_resize(lq_crop), _pad_and_resize(gt_crop), observed


# ---------------------------------------------------------------------------
# StratifiedRoleBatchSampler (eliminates dark gradient dominance!)
# ---------------------------------------------------------------------------


V31_SHARPNESS_CRITICAL_NORMAL_REGIMES: Tuple[str, ...] = (
    "severe_lr_survface",
    "compound_cctv_codec",
    "global_defocus_blur",
    "directional_motion_blur",
)

V34_PRIORITY_NORMAL_REGIMES: Tuple[str, ...] = (
    "severe_lr_survface",
    "daylight_sensor_noise",
    "compound_cctv_codec",
    "global_defocus_blur",
    "local_partial_blur",
)


class StratifiedRoleBatchSampler(Sampler[List[Tuple[int, str]]]):
    """Guarantees every batch contains balanced `clear`, `normal_degraded`, and `lowlight_joint`.

    Yields batches of `(dataset_index, regime_name)` tuples consumed by `V3RestorationDataset`.
    Under `calibration="v31"`, the 4 sharpness-critical `normal_degraded` regimes
    (`severe_lr_survface`, `compound_cctv_codec`, `global_defocus_blur`,
    `directional_motion_blur`) are weighted 2x vs the other 5.
    Under `calibration="v34"`, batches use `12.5% clear / 25% lowlight_joint / 62.5% normal_degraded`
    with 2x weight on `V34_PRIORITY_NORMAL_REGIMES`.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        clear_frac: Optional[float] = None,
        lowlight_frac: float = 0.25,
        seed: int = 42,
        drop_last: bool = True,
        calibration: str = "v30",
    ):
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive")
        if calibration not in ("v30", "v31", "v32", "v33", "v34"):
            raise ValueError(f"Unknown sampler calibration '{calibration}'. Expected 'v30', 'v31', 'v32', 'v33', or 'v34'.")
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        if clear_frac is None:
            self.clear_frac = 0.125 if calibration == "v34" else 0.15
        else:
            self.clear_frac = float(clear_frac)
        self.lowlight_frac = float(lowlight_frac)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.calibration = str(calibration)
        self.epoch = 0

        # Compute exact per-batch quotas (guaranteed >= 1 per group when batch_size >= 3)
        if self.batch_size >= 3:
            self.n_clear = max(1, int(round(self.batch_size * self.clear_frac)))
            self.n_lowlight = max(1, int(round(self.batch_size * self.lowlight_frac)))
            self.n_normal = max(1, self.batch_size - self.n_clear - self.n_lowlight)
            while self.n_clear + self.n_lowlight + self.n_normal > self.batch_size:
                if self.n_normal > 1:
                    self.n_normal -= 1
                elif self.n_lowlight > 1:
                    self.n_lowlight -= 1
                else:
                    self.n_clear -= 1
        elif self.batch_size == 2:
            self.n_clear, self.n_normal, self.n_lowlight = 1, 1, 0
        else:
            self.n_clear, self.n_normal, self.n_lowlight = 0, 1, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self) -> Iterator[List[Tuple[int, str]]]:
        rng = random.Random(self.seed + self.epoch * 9973)
        indices = list(range(self.num_samples))
        rng.shuffle(indices)

        if self.calibration == "v34":
            normal_pool: Tuple[str, ...] = REGIME_GROUPS["normal_degraded"] + V34_PRIORITY_NORMAL_REGIMES
        elif self.calibration in ("v31", "v32", "v33"):
            normal_pool = REGIME_GROUPS["normal_degraded"] + V31_SHARPNESS_CRITICAL_NORMAL_REGIMES
            if self.calibration == "v33":
                normal_pool = normal_pool + ("local_partial_blur",)
        else:
            normal_pool = REGIME_GROUPS["normal_degraded"]
        lowlight_pool = REGIME_GROUPS["lowlight_joint"]

        n_batches = len(self)
        for b_idx in range(n_batches):
            start = b_idx * self.batch_size
            chunk = indices[start : start + self.batch_size]
            if len(chunk) < self.batch_size:
                if self.drop_last:
                    break
                # Pad roles proportionally for final partial batch
                roles = (["clear"] * self.n_clear +
                         [rng.choice(lowlight_pool) for _ in range(self.n_lowlight)] +
                         [rng.choice(normal_pool) for _ in range(self.n_normal)])[: len(chunk)]
            else:
                roles = (
                    ["clear"] * self.n_clear
                    + [rng.choice(lowlight_pool) for _ in range(self.n_lowlight)]
                    + [rng.choice(normal_pool) for _ in range(self.n_normal)]
                )
            rng.shuffle(roles)
            yield list(zip(chunk, roles))


# ---------------------------------------------------------------------------
# High-Throughput PyTorch Dataset
# ---------------------------------------------------------------------------


class V3RestorationDataset(Dataset):
    """High-throughput 512x512 HQ face dataset with zero-leakage filtering & 12-regime degradation."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        split_guard: Optional[ZeroLeakageSplitGuard] = None,
        split: str = "train",
        prepare_crop_prob: float = 0.5,
        degrade_fn: Optional[Callable[[np.ndarray, str, np.random.Generator], Tuple[np.ndarray, Dict[str, float]]]] = None,
        seed: int = 20261119,
        max_samples: Optional[int] = None,
        calibration: str = "v30",
    ):
        if calibration not in ("v30", "v31", "v32", "v33", "v34"):
            raise ValueError(f"Unknown dataset calibration '{calibration}'. Expected 'v30', 'v31', 'v32', 'v33', or 'v34'.")
        self.store = MultiBackendImageStore(data_dir)
        self.split = split
        self.prepare_crop_prob = float(prepare_crop_prob)
        self.degrade_fn = degrade_fn
        self.seed = int(seed)
        self.calibration = str(calibration)

        filtered: List[Tuple[str, str, Any]] = []
        for cid, name, loc in self.store.entries:
            if split_guard is not None:
                if split == "train" and not split_guard.is_train_allowed(cid, name):
                    continue
                if split == "val" and not split_guard.is_dev_syn_candidate(cid):
                    continue
            filtered.append((cid, name, loc))

        if max_samples is not None:
            filtered = filtered[: int(max_samples)]
        if not filtered:
            raise ValueError(f"Zero valid samples found in {data_dir} for split={split!r}")

        self.samples = filtered
        if split_guard is not None and split == "train":
            split_guard.assert_zero_leakage([s[0] for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def sample_ids(self) -> List[str]:
        return [s[0] for s in self.samples]

    def __getitem__(self, item: Union[int, Tuple[int, str]]) -> Dict[str, Any]:
        if isinstance(item, tuple):
            idx, regime = int(item[0]), str(item[1])
        else:
            idx = int(item)
            regime = REGIMES_12[idx % len(REGIMES_12)]

        cid, _, loc = self.samples[idx]
        # Deterministic per-worker stochastic seed mixed with index and OS entropy for train
        if self.split == "train":
            seed_val = (self.seed ^ (idx * 2654435761) ^ int.from_bytes(os.urandom(4), "little")) & 0xFFFFFFFF
        else:
            seed_val = int(hashlib.sha256(f"{self.seed}:{cid}:{regime}".encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_val)

        pil = self.store.read_rgb(loc)
        if pil.size != (512, 512):
            pil = pil.resize((512, 512), Image.Resampling.LANCZOS)
        gt_512_u8 = np.asarray(pil, dtype=np.uint8)

        blur_mask_np: Optional[np.ndarray] = None
        x_struct_clean_u8: Optional[np.ndarray] = None
        x_photo_tgt_u8: Optional[np.ndarray] = None
        is_nn_pixelated = False
        if self.degrade_fn is None:
            from research.codeformer_restoration_v3.degradations_v3 import (
                FFHQ_CANONICAL_5PT_NORM,
                generate_cctv_pair,
            )

            lq_256_u8, gt_256_u8, gt_512_u8, observed, raw_targets = generate_cctv_pair(
                gt_512_u8,
                regime=regime,
                seed=seed_val,
                geometry_prob=self.prepare_crop_prob if self.split == "train" else 0.0,
                calibration=self.calibration if self.split == "train" else "v30",
            )
            landmarks_np = raw_targets.get("landmarks_256")
            if landmarks_np is None:
                landmarks_np = (FFHQ_CANONICAL_5PT_NORM * 256.0).astype(np.float32)
            deg_targets = _extract_deg_targets(raw_targets, regime)
            if "blur_mask" in raw_targets and raw_targets["blur_mask"] is not None:
                blur_mask_np = np.asarray(raw_targets["blur_mask"], dtype=np.float32)
            if "x_struct_clean_256" in raw_targets and raw_targets["x_struct_clean_256"] is not None:
                x_struct_clean_u8 = np.asarray(raw_targets["x_struct_clean_256"], dtype=np.uint8)
            if "x_photo_tgt_256" in raw_targets and raw_targets["x_photo_tgt_256"] is not None:
                x_photo_tgt_u8 = np.asarray(raw_targets["x_photo_tgt_256"], dtype=np.uint8)
            is_nn_pixelated = bool(raw_targets.get("is_nn_pixelated", False))
        else:
            from research.codeformer_restoration_v3.degradations_v3 import FFHQ_CANONICAL_5PT_NORM

            landmarks_np = (FFHQ_CANONICAL_5PT_NORM * 256.0).astype(np.float32)
            lq_512_u8, deg_targets = self.degrade_fn(gt_512_u8, regime, rng)
            if lq_512_u8.shape[:2] != (512, 512):
                lq_512_u8 = np.asarray(
                    Image.fromarray(lq_512_u8).resize((512, 512), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
            if self.split == "train" and rng.random() < self.prepare_crop_prob:
                lq_256_u8, gt_256_u8, observed = simulate_prepare_crop_pair(lq_512_u8, gt_512_u8, rng)
            else:
                lq_256_u8 = np.asarray(
                    Image.fromarray(lq_512_u8).resize((256, 256), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
                gt_256_u8 = np.asarray(
                    Image.fromarray(gt_512_u8).resize((256, 256), Image.Resampling.LANCZOS),
                    dtype=np.uint8,
                )
                observed = np.ones((256, 256), dtype=bool)

        if blur_mask_np is None:
            b_val = float(
                np.clip(
                    max(float(deg_targets.get("blur", 0.0)), 0.85 * float(deg_targets.get("codec_lr", 0.0))),
                    0.0,
                    1.0,
                )
            )
            blur_mask_np = np.where(observed, b_val, 0.0).astype(np.float32)

        x_in_256 = torch.from_numpy(lq_256_u8.transpose(2, 0, 1).copy()).float() / 255.0
        gt_256 = torch.from_numpy(gt_256_u8.transpose(2, 0, 1).copy()).float() / 255.0
        gt_512 = torch.from_numpy(gt_512_u8.transpose(2, 0, 1).copy()).float() / 255.0
        x_struct_clean_256 = (
            torch.from_numpy(x_struct_clean_u8.transpose(2, 0, 1).copy()).float() / 255.0
            if x_struct_clean_u8 is not None
            else gt_256.clone()
        )
        x_photo_tgt_256 = (
            torch.from_numpy(x_photo_tgt_u8.transpose(2, 0, 1).copy()).float() / 255.0
            if x_photo_tgt_u8 is not None
            else gt_256.clone()
        )
        obs_t = torch.from_numpy(observed.astype(np.float32)).unsqueeze(0)
        blur_mask_t = torch.from_numpy(blur_mask_np.copy()).unsqueeze(0)
        landmarks_t = torch.from_numpy(np.asarray(landmarks_np, dtype=np.float32).copy())

        deg_vec = torch.tensor(
            [
                float(deg_targets.get("exposure", 0.0)),
                float(deg_targets.get("blur", 0.0)),
                float(deg_targets.get("noise", 0.0)),
                float(deg_targets.get("codec_lr", 0.0)),
            ],
            dtype=torch.float32,
        )

        return {
            "x_in_256": x_in_256,
            "gt_256": gt_256,
            "gt_512": gt_512,
            "x_struct_clean_256": x_struct_clean_256,
            "x_photo_tgt_256": x_photo_tgt_256,
            "observed_mask": obs_t,
            "blur_mask": blur_mask_t,
            "landmarks": landmarks_t,
            "is_clear": torch.tensor(regime == "clear", dtype=torch.bool),
            "is_nn_pixelated": torch.tensor(is_nn_pixelated, dtype=torch.bool),
            "deg_targets": deg_vec,
            "regime": regime,
            "sample_id": cid,
        }


def make_stratified_dataloader(
    dataset: V3RestorationDataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int = 42,
    drop_last: bool = True,
    calibration: Optional[str] = None,
) -> Tuple[DataLoader, StratifiedRoleBatchSampler]:
    """Build a high-throughput DataLoader backed by StratifiedRoleBatchSampler."""
    eff_cal = str(calibration) if calibration is not None else str(getattr(dataset, "calibration", "v30"))
    sampler = StratifiedRoleBatchSampler(
        num_samples=len(dataset),
        batch_size=batch_size,
        seed=seed,
        drop_last=drop_last,
        calibration=eff_cal,
    )
    use_persistent = bool(persistent_workers and num_workers > 0)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory and torch.cuda.is_available()),
        persistent_workers=use_persistent,
    )
    return loader, sampler


# ---------------------------------------------------------------------------
# Deterministic Frozen DEV-Syn-240 Generator
# ---------------------------------------------------------------------------


def generate_dev_syn(
    source_dir: Union[str, Path],
    out_dir: Union[str, Path],
    split_guard: ZeroLeakageSplitGuard,
    num_ids: int = 20,
    seed: int = 20261119,
    degrade_fn: Optional[Callable[[np.ndarray, str, np.random.Generator], Tuple[np.ndarray, Dict[str, float]]]] = None,
    calibration: str = "v30",
) -> Path:
    """Generate deterministic frozen DEV-Syn pairs (num_ids unseen val IDs x 12 regimes).

    Saves deterministic PNGs under `out_dir/{inputs,targets,observed}` and writes
    `out_dir/dev_syn_manifest.json` with SHA-256 bindings and zero-leakage receipt.
    """
    if calibration not in ("v30", "v31", "v32", "v33", "v34"):
        raise ValueError(f"Unknown DEV-Syn calibration '{calibration}'. Expected 'v30', 'v31', 'v32', 'v33', or 'v34'.")
    out_path = Path(out_dir)
    for sub in ("inputs", "targets", "observed"):
        (out_path / sub).mkdir(parents=True, exist_ok=True)

    store = MultiBackendImageStore(source_dir)
    candidates = [e for e in store.entries if split_guard.is_dev_syn_candidate(e[0])]
    # Deterministic sort by SHA-256 of seed + canonical_id
    candidates.sort(key=lambda e: _sha256_bytes(f"{seed}:{e[0]}".encode("utf-8")))
    selected = candidates[: int(num_ids)]
    if len(selected) < int(num_ids):
        raise ValueError(f"Requested {num_ids} unseen val IDs, only found {len(selected)} in {source_dir}")

    selected_ids = [e[0] for e in selected]
    split_guard.assert_zero_leakage(train_ids=[], dev_syn_ids=selected_ids)
    # Also ensure zero overlap with allowed_train_ids
    overlap_tr = set(selected_ids) & split_guard.allowed_train_ids
    if overlap_tr:
        raise AssertionError(f"DEV-Syn selected IDs overlap TRAIN pool: {sorted(overlap_tr)[:5]}")

    deg = degrade_fn or _default_degrade_adapter
    cases: List[Dict[str, Any]] = []

    for cid, _, loc in selected:
        pil = store.read_rgb(loc)
        if pil.size != (512, 512):
            pil = pil.resize((512, 512), Image.Resampling.LANCZOS)
        gt_512_u8 = np.asarray(pil, dtype=np.uint8)

        tgt_rel = f"targets/{cid}.png"
        tgt_path = out_path / tgt_rel
        tgt_sha: Optional[str] = None

        for regime in REGIMES_12:
            case_id = f"{cid}__{regime}"
            case_seed = int(_sha256_bytes(f"{seed}:{case_id}".encode("utf-8"))[:8], 16)
            if degrade_fn is None:
                from research.codeformer_restoration_v3.degradations_v3 import generate_cctv_pair

                lq_256_u8, gt_256_u8, _, obs_bool, raw_targets = generate_cctv_pair(
                    gt_512_u8, regime=regime, seed=case_seed, apply_geometry=False, calibration=calibration
                )
                deg_targets = _extract_deg_targets(raw_targets, regime)
                obs_u8 = (obs_bool.astype(np.uint8)) * 255
            else:
                rng = np.random.default_rng(case_seed)
                lq_512_u8, deg_targets = degrade_fn(gt_512_u8, regime, rng)
                lq_256_u8 = np.asarray(
                    Image.fromarray(lq_512_u8).resize((256, 256), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
                gt_resample = Image.Resampling.LANCZOS if calibration in ("v32", "v33", "v34") else Image.Resampling.BILINEAR
                gt_256_u8 = np.asarray(
                    Image.fromarray(gt_512_u8).resize((256, 256), gt_resample),
                    dtype=np.uint8,
                )
                obs_u8 = np.full((256, 256), 255, dtype=np.uint8)

            if tgt_sha is None:
                Image.fromarray(gt_256_u8).save(tgt_path)
                tgt_sha = _sha256_bytes(tgt_path.read_bytes())

            in_rel = f"inputs/{case_id}.png"
            obs_rel = f"observed/{case_id}.png"
            in_path = out_path / in_rel
            obs_path = out_path / obs_rel
            Image.fromarray(lq_256_u8).save(in_path)
            Image.fromarray(obs_u8).save(obs_path)

            cases.append(
                {
                    "id": case_id,
                    "reference_id": cid,
                    "regime": regime,
                    "input": in_rel,
                    "input_sha256": _sha256_bytes(in_path.read_bytes()),
                    "target": tgt_rel,
                    "target_sha256": tgt_sha,
                    "observed": obs_rel,
                    "deg_targets": deg_targets,
                }
            )

    manifest = {
        "schema": "codeformer-restoration-v3-dev-syn-v1",
        "calibration": str(calibration),
        "seed": int(seed),
        "num_ids": len(selected_ids),
        "regimes": list(REGIMES_12),
        "num_cases": len(cases),
        "selected_ids": selected_ids,
        "zero_leakage_verified": True,
        "cases": cases,
    }
    manifest_path = out_path / "dev_syn_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest_path
