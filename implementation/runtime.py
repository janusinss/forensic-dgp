"""Frozen CPU inference for Pre-Conditioned Frozen CodeFormer V3.4.

CodeFormer remains attributed to S-Lab in third_party/codeformer, including its license.
This module imports no research service, training helper, or historical output.
Importing it constructs no model and performs no network or device discovery.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import threading
import weakref
import zipfile

from .model_store import model_directory

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = model_directory(ROOT, 'implementation')
SITES = ("32", "64", "128", "256")
MAX_IMAGE_BYTES = 10 * 1024**2
MAX_IMAGE_SIDE = 1600
CODEFORMER_BASE_FILE = {
    "bytes": 376637898,
    "sha256": "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
}
V34_FILE = {
    "bytes": 9065704,
    "sha256": "65541ac31b50becafec3db34cca8936ab92c5958b0afd75244d85ecd0e0e88b1",
}
CODEFORMER_BASE_STATE = "6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd"
V34_STATE = "cc679bd6ffd919a21a92a2f07de0c2f1f2602d9449a4c714e3ec82cd5dce61bd"
ARCHITECTURE_SOURCES = {
    "third_party.codeformer.codeformer_arch": {
        "path": "third_party/codeformer/codeformer_arch.py",
        "sha256": "29c40cee39414c1117d1ae55fc15d6cdd42aed09e03019522c7f55771b1b84af",
    },
    "third_party.codeformer.vqgan_arch": {
        "path": "third_party/codeformer/vqgan_arch.py",
        "sha256": "01d7d9a796489716439cbd59fbd9e305278d762d97e6582aafd54b9279e73265",
    },
    "third_party.codeformer.registry": {
        "path": "third_party/codeformer/registry.py",
        "sha256": "21c23bd3bf727eb2decd99035120fc47316db92486d73116cca18e9fa8b6c854",
    },
    "implementation.architecture_v34": {
        "path": "implementation/architecture_v34.py",
        "sha256": "d243bb8abb8d9379c5f9d95bdb786909235b0269194688a8d62182d9c1a28cf2",
    },
}
_OWNER_ATTRIBUTE = "_portable_codeformer_exclusive_owner"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def ordinary_file(path, maximum=512 * 1024**2):
    """Require bounded ordinary project assets, with no links/reparse points."""
    path = Path(os.path.abspath(path))
    allowed_models = model_directory(ROOT, 'implementation')
    need(path.is_relative_to(ROOT) or path.is_relative_to(allowed_models), "Model/source must stay inside this installation.")
    for parent in path.parents:
        info = parent.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
             and not getattr(info, "st_file_attributes", 0) & 1024,
             "Ordinary model/source directory required.")
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_nlink == 1
         and not getattr(info, "st_file_attributes", 0) & 1024 and 0 < info.st_size <= maximum,
         "Bounded ordinary model/source file required.")
    return path


def file_pin(path, maximum=512 * 1024**2):
    path = ordinary_file(path, maximum)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino)
         == (after.st_size, after.st_mtime_ns, after.st_ino),
         "Model/source changed during verification.")
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def architecture_source_sha256(raw):
    """Hash source with CRLF converted to LF; preserve every other source byte."""
    need(type(raw) is bytes, "Architecture source bytes required.")
    canonical = raw.replace(b"\r\n", b"\n")
    need(b"\r" not in canonical, "Bare CR in architecture source refused.")
    return hashlib.sha256(canonical).hexdigest()


def architecture_file_sha256(path):
    path = ordinary_file(path, maximum=1024**2)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino)
         == (after.st_size, after.st_mtime_ns, after.st_ino),
         "Architecture source changed during verification.")
    return architecture_source_sha256(raw)


def _add_mbconv_shapes(shapes, prefix, in_ch, out_ch):
    mid = in_ch * 2
    shapes[prefix + "pw1.weight"] = (mid, in_ch, 1, 1)
    shapes[prefix + "gn1.weight"] = (mid,)
    shapes[prefix + "gn1.bias"] = (mid,)
    shapes[prefix + "dw.weight"] = (mid, 1, 3, 3)
    shapes[prefix + "gn2.weight"] = (mid,)
    shapes[prefix + "gn2.bias"] = (mid,)
    shapes[prefix + "pw2.weight"] = (out_ch, mid, 1, 1)
    shapes[prefix + "gn3.weight"] = (out_ch,)
    shapes[prefix + "gn3.bias"] = (out_ch,)


def _add_naf_shapes(shapes, prefix, c):
    dw_ch = c * 2
    ffn_ch = c * 2
    shapes[prefix + "norm1.weight"] = (1, c, 1, 1)
    shapes[prefix + "norm1.bias"] = (1, c, 1, 1)
    shapes[prefix + "conv1.weight"] = (dw_ch, c, 1, 1)
    shapes[prefix + "conv1.bias"] = (dw_ch,)
    shapes[prefix + "conv2.weight"] = (dw_ch, 1, 3, 3)
    shapes[prefix + "conv2.bias"] = (dw_ch,)
    shapes[prefix + "sca.weight"] = (c, c, 1, 1)
    shapes[prefix + "sca.bias"] = (c,)
    shapes[prefix + "conv3.weight"] = (c, c, 1, 1)
    shapes[prefix + "conv3.bias"] = (c,)
    shapes[prefix + "norm2.weight"] = (1, c, 1, 1)
    shapes[prefix + "norm2.bias"] = (1, c, 1, 1)
    shapes[prefix + "conv4.weight"] = (ffn_ch, c, 1, 1)
    shapes[prefix + "conv4.bias"] = (ffn_ch,)
    shapes[prefix + "conv5.weight"] = (c, c, 1, 1)
    shapes[prefix + "conv5.bias"] = (c,)
    shapes[prefix + "beta"] = (1, c, 1, 1)
    shapes[prefix + "gamma"] = (1, c, 1, 1)


def v34_shapes():
    """Pure-Python exact tensor shape census for V3.4 front-end modules (344 tensors)."""
    shapes = {}
    # 1. deg_encoder
    p = "deg_encoder."
    shapes[p + "stem.0.weight"] = (24, 7, 3, 3)
    shapes[p + "stem.1.weight"] = (24,)
    shapes[p + "stem.1.bias"] = (24,)
    for idx, (in_c, out_c) in enumerate(((24, 40), (40, 56), (56, 80), (80, 112)), start=1):
        _add_mbconv_shapes(shapes, f"{p}stage{idx}.0.", in_c, out_c)
        _add_mbconv_shapes(shapes, f"{p}stage{idx}.1.", out_c, out_c)
    shapes[p + "z_proj.0.weight"] = (128, 224)
    shapes[p + "z_proj.0.bias"] = (128,)
    shapes[p + "z_proj.1.weight"] = (128,)
    shapes[p + "z_proj.1.bias"] = (128,)
    shapes[p + "z_proj.3.weight"] = (128, 128)
    shapes[p + "z_proj.3.bias"] = (128,)
    shapes[p + "deg_head.0.weight"] = (64, 128)
    shapes[p + "deg_head.0.bias"] = (64,)
    shapes[p + "deg_head.2.weight"] = (4, 64)
    shapes[p + "deg_head.2.bias"] = (4,)
    for head in ("g_photo_head", "g_pre_head", "w_head", "clean_head"):
        shapes[f"{p}{head}.0.weight"] = (32, 128)
        shapes[f"{p}{head}.0.bias"] = (32,)
        shapes[f"{p}{head}.2.weight"] = (1, 32)
        shapes[f"{p}{head}.2.bias"] = (1,)

    # 2. photo_net
    p = "photo_net."
    shapes[p + "z_proj.weight"] = (32, 128)
    shapes[p + "z_proj.bias"] = (32,)
    shapes[p + "conv.0.weight"] = (24, 144, 3, 3)
    shapes[p + "conv.1.weight"] = (24,)
    shapes[p + "conv.1.bias"] = (24,)
    shapes[p + "conv.3.weight"] = (24, 24, 3, 3)
    shapes[p + "conv.4.weight"] = (24,)
    shapes[p + "conv.4.bias"] = (24,)
    shapes[p + "param_head.weight"] = (5, 24, 1, 1)
    shapes[p + "param_head.bias"] = (5,)
    shapes[p + "wb_head.weight"] = (3, 128)
    shapes[p + "wb_head.bias"] = (3,)

    # 3. pre_net
    p = "pre_net."
    shapes[p + "intro.weight"] = (32, 6, 3, 3)
    shapes[p + "intro.bias"] = (32,)
    ch = 32
    for level, n_blk in enumerate((1, 2, 2)):
        for b in range(n_blk):
            _add_naf_shapes(shapes, f"{p}encoders.{level}.{b}.", ch)
        shapes[f"{p}downs.{level}.weight"] = (ch * 2, ch, 2, 2)
        shapes[f"{p}downs.{level}.bias"] = (ch * 2,)
        ch *= 2
    for b in range(2):
        _add_naf_shapes(shapes, f"{p}middle.{b}.", ch)
    shapes[p + "mid_film.weight"] = (ch * 2, 128)
    shapes[p + "mid_film.bias"] = (ch * 2,)
    for level, n_blk in enumerate((2, 2, 1)):
        shapes[f"{p}ups.{level}.0.weight"] = (ch * 2, ch, 1, 1)
        ch //= 2
        for b in range(n_blk):
            _add_naf_shapes(shapes, f"{p}decoders.{level}.{b}.", ch)
    shapes[p + "out_conv.weight"] = (3, 32, 3, 3)
    shapes[p + "out_conv.bias"] = (3,)
    return shapes


def _inspect_npz_schema(raw, shapes, *, max_bytes, label):
    need(type(raw) is bytes and 0 < len(raw) < max_bytes, f"Bounded {label} NPZ bytes required.")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        items, names = archive.infolist(), archive.namelist()
        need(len(items) == len(set(names)) == len(shapes) and {n[:-4] for n in names} == set(shapes)
             and all("/" not in n and "\\" not in n and ":" not in n and n.endswith(".npy") for n in names),
             f"Expected exactly {len(shapes)} unique flat {label} arrays.")
        need(all(not item.flag_bits & 1 and not stat.S_ISLNK(item.external_attr >> 16) for item in items)
             and sum(item.file_size for item in items) < max_bytes,
             f"Bounded ordinary expanded {label} NPZ required.")
        import numpy as np
        for member in items:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                need(version in ((1, 0), (2, 0)), f"Unsupported {label} NPY version.")
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(stream, max_header_size=4096)
                need(shape == shapes[member.filename[:-4]] and dtype == np.dtype("float32")
                     and not dtype.hasobject and not fortran, f"Exact float32 {label} schema required.")
                need(member.file_size == stream.tell() + math.prod(shape) * 4, f"Incorrect {label} array size.")
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            result = {}
            for name, shape in shapes.items():
                array = saved[name]
                need(array.shape == shape and array.dtype == np.float32 and np.isfinite(array).all(),
                     f"Finite exact {label} values required.")
                result[name] = np.ascontiguousarray(array).copy()
            return result


def inspect_v34_npz(raw):
    """Validate exact bounded float32 V3.4 pre-conditioning array headers before NumPy allocation."""
    return _inspect_npz_schema(raw, v34_shapes(), max_bytes=16 * 1024**2, label="V3.4")


def numpy_state_sha(state):
    """Canonical tensor-name/dtype/shape/value digest used by retained exports."""
    digest = hashlib.sha256()
    for name, array in sorted(state.items()):
        header = json.dumps([name, "torch.float32", list(array.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def model_contract(model="v3"):
    need(model == "v3", "Only the V3.4 (preconditioned_v34_codeformer) model is supported.")
    assets = {
        "codeformer.pth": dict(CODEFORMER_BASE_FILE),
        "v34_precond_modules.npz": dict(V34_FILE),
    }
    states = {
        "codeformer_base_515": CODEFORMER_BASE_STATE,
        "v34_precond_modules_344": V34_STATE,
    }
    return {
        "model": "v3",
        "arch_variant": "v34",
        "v34_state_sha": V34_STATE,
        "assets": assets,
        "canonical_states": states,
        "frozen_tensors": 859,
        "policy": "CPU FP32 singleton512, adaptive w_pred, MaskedAdaIN, Pre-Conditioned Frozen CodeFormer V3.4, torch.no_grad",
        "architecture": (
            "Official frozen CodeFormer by S-Lab (515 tensors) with V3.4 pre-conditioning front-end "
            "(V34DegradationEncoder, V34PhotoCorrectNet, V34NAFPreConditionNet; 344 tensors, 2,241,395 params)"
        ),
        "license": "S-Lab License 1.0; third_party/codeformer/LICENSE",
        "internal_resolution": 512,
        "output_resolution": 256,
        "adain": True,
        "inference_only": True,
        "model_qualified": False,
    }


def load_frozen_cpu_restorer(model="v3", model_dir=None):
    """Construct an exclusive V3.4 restorer after verifying every fixed artifact."""
    contract = model_contract(model)
    directory = MODEL_DIR if model_dir is None else Path(model_dir)
    if directory == MODEL_DIR:
        from .setup_models import _stage_bundled_asset_if_needed, load_manifest
        manifest = load_manifest()
        for asset in manifest["assets"].values():
            _stage_bundled_asset_if_needed(asset, directory)
    paths = {name: ordinary_file(directory / name) for name in contract["assets"]}
    for name, expected in contract["assets"].items():
        need(file_pin(paths[name]) == expected, "Exact retained model bytes required: " + name)
    for binding in ARCHITECTURE_SOURCES.values():
        need(architecture_file_sha256(ROOT / binding["path"]) == binding["sha256"],
             "Retained CodeFormer/V3.4 architecture differs.")
    arrays_v34 = inspect_v34_npz(paths["v34_precond_modules.npz"].read_bytes())
    need(numpy_state_sha(arrays_v34) == V34_STATE, "Exact V3.4 pre-conditioning module values required.")

    import numpy as np
    import torch
    from torch.nn import functional as F
    from . import architecture_v34 as arch_v34
    from third_party.codeformer.codeformer_arch import CodeFormer
    need(torch.get_default_dtype() == torch.float32, "FP32 default dtype required.")
    for name, binding in ARCHITECTURE_SOURCES.items():
        loaded = sys.modules.get(name)
        need(loaded is not None and getattr(loaded, "__file__", None)
             and Path(loaded.__file__).resolve() == (ROOT / binding["path"]).resolve()
             and architecture_file_sha256(loaded.__file__) == binding["sha256"],
             "Exact loaded CodeFormer/V3.4 architecture origin required.")
    saved_cf = torch.load(paths["codeformer.pth"], map_location="cpu", weights_only=True, mmap=True)
    need(type(saved_cf) is dict and set(saved_cf) == {"params_ema"}, "Exact official CodeFormer inference format required.")
    cf_sd = saved_cf["params_ema"]
    need(isinstance(cf_sd, dict) and len(cf_sd) == 515, "Exact 515-tensor CodeFormer backbone census required.")

    def state_sha(state):
        need(isinstance(state, dict) and all(
            isinstance(value, torch.Tensor) and value.dtype == torch.float32
            and value.device.type == "cpu" and not value.requires_grad and value.grad is None
            and torch.isfinite(value).all().item() for value in state.values()),
            "Detached finite CPU FP32 state required.")
        return numpy_state_sha({name: value.detach().contiguous().numpy() for name, value in state.items()})

    need(state_sha(cf_sd) == CODEFORMER_BASE_STATE, "Exact official CodeFormer canonical state required.")

    class FrozenCPURestorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=list(SITES), fix_modules=None)
            base.load_state_dict(cf_sd, strict=True)
            self.v34 = arch_v34.PreConditionedCodeFormerV34(base=base)
            self.v34.v3.load_state_dict(
                {name: torch.from_numpy(value.copy()) for name, value in arrays_v34.items()},
                strict=True,
            )
            self._forward_lock = threading.Lock()
            self.last_forward_sites = ()
            self.last_base_calls = 0
            self.last_telemetry = {}
            self.eval()
            self.requires_grad_(False)
            setattr(self.base, _OWNER_ATTRIBUTE, weakref.ref(self))

        @property
        def base(self):
            return self.v34.base

        def train(self, mode=True):
            need(mode is False, "Training mode is unavailable in this frozen runtime.")
            return super().train(False)

        def validate_invariants(self):
            named, state = dict(self.named_parameters()), self.state_dict()
            need(len(named) == len(state) == contract["frozen_tensors"] and len(self.base.state_dict()) == 515
                 and self.base.connect_list == list(SITES) and set(self.base.fuse_convs_dict) == set(SITES)
                 and self.base.fuse_generator_block == {"16": 6, "32": 9, "64": 12, "128": 15, "256": 18, "512": 21},
                 "Exact 859-tensor V3.4 census and four-site CodeFormer architecture required.")
            need(all(not module.training for module in self.modules())
                 and all(value.device.type == "cpu" and value.dtype == torch.float32 and not value.requires_grad
                         and value.grad is None and torch.isfinite(value).all().item() for value in named.values()),
                 "All tensors must remain finite frozen/eval CPU FP32 with no gradients.")
            v34_sd = dict(self.v34.v3.state_dict().items())
            need(state_sha(self.base.state_dict()) == CODEFORMER_BASE_STATE,
                 "Frozen CodeFormer base canonical state changed.")
            need(set(v34_sd) == set(v34_shapes())
                 and all(tuple(value.shape) == v34_shapes()[name] for name, value in v34_sd.items())
                 and state_sha(v34_sd) == V34_STATE,
                 "Frozen V3.4 pre-conditioning canonical state changed.")
            owner = getattr(self.base, _OWNER_ATTRIBUTE, None)
            need(isinstance(owner, weakref.ReferenceType) and owner() is self, "Exclusive owned base required.")
            return {"pass": True, "frozen_tensors": contract["frozen_tensors"], "model": "v3"}

        def restore_canvas(self, canvas, observed=None):
            """Run one frozen Pre-Conditioned CodeFormer V3.4 restoration on a 256x256 RGB8 canvas."""
            need(not torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
                 "Use torch.no_grad; gradient/inference_mode runtime refused.")
            need(isinstance(canvas, np.ndarray) and canvas.shape == (256, 256, 3) and canvas.dtype == np.uint8,
                 "256x256 uint8 RGB canvas required.")
            if observed is None:
                obs_np = np.ones((256, 256), dtype=bool)
            else:
                need(isinstance(observed, np.ndarray) and observed.shape == (256, 256) and observed.dtype == bool,
                     "256x256 bool observed mask required.")
                obs_np = observed
            need(self._forward_lock.acquire(blocking=False), "Nested/concurrent forward refused.")
            self.last_forward_sites, self.last_base_calls, self.last_telemetry = (), 0, {}
            try:
                self.validate_invariants()
                with torch.autocast(device_type="cpu", enabled=False):
                    inp_t = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).to(dtype=torch.float32)[None] / 255.0
                    obs_t = torch.from_numpy(obs_np.astype(np.float32))[None, None]
                    out = self.v34(inp_t, obs_t)
                    v34_raw = out["y_256"][0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32, copy=True)
                    telemetry = {
                        "arch_variant": "v34",
                        "w_pred": float(out["w_pred"].detach().cpu().item()),
                        "g_photo": float(out["g_photo"].detach().cpu().item()),
                        "g_pre": float(out["g_pre"].detach().cpu().item()),
                        "alpha_clean": float(out["alpha_clean"].detach().cpu().item()),
                        "v34_state_sha": V34_STATE,
                    }
                need(v34_raw.shape == (256, 256, 3) and v34_raw.dtype == np.float32
                     and np.isfinite(v34_raw).all() and 0.0 <= float(v34_raw.min()) <= float(v34_raw.max()) <= 1.0,
                     "Invalid V3.4 output schema.")
                self.last_forward_sites, self.last_base_calls, self.last_telemetry = SITES, 1, telemetry
                self.validate_invariants()
                return v34_raw
            finally:
                self._forward_lock.release()

        def forward(self, sample, *, adain=True):
            need(not torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
                 "Use torch.no_grad; gradient/inference_mode runtime refused.")
            need(adain is True, "Fixed MaskedAdaIN policy required.")
            need(isinstance(sample, torch.Tensor) and tuple(sample.shape) == (1, 3, 512, 512)
                 and sample.dtype == torch.float32 and sample.device.type == "cpu" and not sample.requires_grad
                 and torch.isfinite(sample).all().item() and sample.min().item() >= -1.0 and sample.max().item() <= 1.0,
                 "Finite singleton CPU FP32 prepared512 input in [-1,1] required.")
            inp_256 = F.interpolate(((sample + 1.0) / 2.0).clamp(0.0, 1.0), (256, 256), mode="bilinear", align_corners=False)
            canvas_u8 = np.floor(inp_256[0].permute(1, 2, 0).contiguous().numpy() * np.float32(255.0)).astype(np.uint8)
            v34_raw = self.restore_canvas(canvas_u8)
            out_256 = torch.from_numpy(v34_raw.transpose(2, 0, 1).copy()).to(dtype=torch.float32)[None]
            out_512 = F.interpolate(out_256, (512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0
            return (
                out_512,
                torch.zeros((1, 256, 1024), dtype=torch.float32),
                torch.zeros((1, 256, 16, 16), dtype=torch.float32),
            )

    with torch.no_grad(), torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261021)
        net = FrozenCPURestorer()
    net.validate_invariants()
    need(all(value.untyped_storage().data_ptr() != cf_sd[name].untyped_storage().data_ptr()
             for name, value in net.base.state_dict().items()), "Base must own independent tensor storage.")
    for name, expected in contract["assets"].items():
        need(file_pin(paths[name]) == expected, "Model changed during loading: " + name)
    net.provenance = contract
    return net


def decode_crop(contents):
    """Decode a bounded PNG/JPEG face crop without loading any neural modules."""
    import numpy as np
    from PIL import Image, ImageOps
    need(type(contents) is bytes and 0 < len(contents) <= MAX_IMAGE_BYTES,
         "Upload a PNG or JPEG no larger than 10 MB.")
    try:
        with Image.open(io.BytesIO(contents)) as source:
            need(source.format in ("PNG", "JPEG"), "Use a PNG or JPEG image.")
            need(max(source.size) <= MAX_IMAGE_SIDE, "Use a face crop no larger than 1600 pixels per side.")
            return np.array(ImageOps.exif_transpose(source).convert("RGB"))
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Cannot read this image. Use a PNG or JPEG face crop.") from exc


def prepare_crop(rgb):
    """Exact neutral128 center padding, bilinear256 and nearest support geometry."""
    import numpy as np
    from PIL import Image
    need(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3
         and rgb.shape[2] == 3 and min(rgb.shape[:2]) > 0, "Expected nonempty uint8 RGB pixels.")
    height, width = rgb.shape[:2]
    need(max(height, width) <= MAX_IMAGE_SIDE, "Face crop exceeds 1600 pixels per side.")
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(Image.fromarray(rgb), (left, top))
    support = Image.new("L", (side, side), 0)
    support.paste(255, (left, top, left + width, top + height))
    target = np.asarray(canvas.resize((256, 256), Image.Resampling.BILINEAR)).copy()
    observed = np.asarray(support.resize((256, 256), Image.Resampling.NEAREST)).copy() > 0
    mask = np.zeros((256, 256), dtype=bool)
    yy, xx = np.nonzero(observed)
    need(len(xx) > 0, "Use a crop with readable facial width and height.")
    geometry = {
        "native_size": [width, height], "native_center_offset": [left, top],
        "observed_bounds_256": [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)],
        "output_size": [256, 256], "policy": "dgp256-observed-canvas-v1",
        "input_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "canvas_rgb_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "observed_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
        "removal_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
    }
    return target, observed, mask, geometry


def delivered_views(raw, canvas, observed):
    """Compose exact prepared pixels outside support for both RAW and PNG."""
    import numpy as np
    need(isinstance(raw, np.ndarray) and raw.shape == (256, 256, 3) and raw.dtype == np.float32
         and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
         and isinstance(canvas, np.ndarray) and canvas.shape == raw.shape and canvas.dtype == np.uint8
         and isinstance(observed, np.ndarray) and observed.shape == (256, 256) and observed.dtype == bool,
         "Invalid restoration pixels or support.")
    raw = raw.copy()
    raw[~observed] = canvas[~observed].astype(np.float32) / np.float32(255)
    pixels = np.floor(raw * np.float32(255)).astype(np.uint8)
    pixels[~observed] = canvas[~observed]
    return raw, pixels
