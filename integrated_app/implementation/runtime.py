"""Frozen CPU inference for the Conditioned SFT V2 (p2_u0224) restoration model.

Numerics are preserved from the Conditioned SFT V2 return-audited implementation.
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

from . import conditioning_v2 as cond_v2
from .managed_paths import data_path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = data_path(ROOT, 'models', 'implementation')
SITES = ("32", "64", "128", "256")
CHANNELS = {"32": 256, "64": 256, "128": 128, "256": 128}
WIDTH = 32
MAX_IMAGE_BYTES = 10 * 1024**2
MAX_IMAGE_SIDE = 1600
R2_FILE = {"bytes": 376609465, "sha256": "d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896"}
U96_FILE = {"bytes": 212827, "sha256": "958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0"}
HEAD_FILE = {"bytes": 685818, "sha256": "77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17"}
V2_FILE = {"bytes": 3831362, "sha256": "1aee2b840397925049a2c821edd7395c3a9e4b48ce1552e06c5ffd627a33e31f"}
R2_STATE = "a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f"
U96_STATE = "fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a"
HEAD_STATE = "6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be"
V2_STATE = "3faa78e15c2fb50bfaa523b5dc7ee64fa4c19a13c490760a7401ea43ab0269d7"
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
}
_OWNER_ATTRIBUTE = "_portable_codeformer_exclusive_owner"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def ordinary_file(path, maximum=512 * 1024**2):
    """Require bounded ordinary project assets, with no links/reparse points."""
    path = Path(os.path.abspath(path))
    allowed_models = data_path(ROOT, 'models', 'implementation')
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


def adapter_shapes():
    return {
        "adapters." + site + "." + name: shape
        for site, channels in CHANNELS.items()
        for name, shape in {
            "down.weight": (32, channels, 1, 1), "down.bias": (32,),
            "depthwise.weight": (32, 1, 3, 3), "depthwise.bias": (32,),
            "up.weight": (channels, 32, 1, 1), "up.bias": (channels,),
        }.items()
    }


def head_shapes():
    return {
        "heads." + site + "." + name: shape
        for site, channels in CHANNELS.items()
        for name, shape in {
            "encoder.weight": (32, channels, 1, 1), "encoder.bias": (32,),
            "decoder.weight": (32, channels, 1, 1), "fused.weight": (32, channels, 1, 1),
            "coarse.weight": (32, 256, 1, 1), "spatial.weight": (32, 32, 3, 3),
            "spatial.bias": (32,), "output.weight": (channels, 32, 1, 1),
            "output.bias": (channels,),
        }.items()
    }


def v2_shapes():
    shapes = {}
    for site, ch in CHANNELS.items():
        in_ch = ch + 1 + (96 if site != "32" else 0)
        p = f"pyramid.stages.{site}."
        shapes[p + "in_proj.weight"] = (96, in_ch, 1, 1)
        shapes[p + "in_proj.bias"] = (96,)
        shapes[p + "out_proj.weight"] = (ch, 96, 1, 1)
        shapes[p + "out_proj.bias"] = (ch,)
        for blk in ("block1", "block2"):
            b = f"{p}{blk}."
            for dw in ("dw1", "dw2", "dw4"):
                shapes[b + dw + ".weight"] = (32, 1, 3, 3)
                shapes[b + dw + ".bias"] = (32,)
            shapes[b + "pw.weight"] = (96, 96, 1, 1)
            shapes[b + "pw.bias"] = (96,)
            shapes[b + "se1.weight"] = (24, 96, 1, 1)
            shapes[b + "se1.bias"] = (24,)
            shapes[b + "se2.weight"] = (96, 24, 1, 1)
            shapes[b + "se2.bias"] = (96,)
        d = f"dark_refiner.stages.{site}."
        shapes[d + "in_proj.weight"] = (48, ch, 1, 1)
        shapes[d + "in_proj.bias"] = (48,)
        for dw in ("dw1", "dw2"):
            shapes[d + dw + ".weight"] = (48, 1, 3, 3)
            shapes[d + dw + ".bias"] = (48,)
        shapes[d + "pw.weight"] = (48, 96, 1, 1)
        shapes[d + "pw.bias"] = (48,)
        shapes[d + "out_proj.weight"] = (ch, 48, 1, 1)
        shapes[d + "out_proj.bias"] = (ch,)
        h = f"head_v2.{site}."
        shapes[h + "freq_gate.weight"] = (ch, 2 * ch, 1, 1)
        shapes[h + "freq_gate.bias"] = (ch,)
        shapes[h + "enc_p.weight"] = (48, ch, 1, 1)
        shapes[h + "enc_p.bias"] = (48,)
        shapes[h + "dec_p.weight"] = (48, ch, 1, 1)
        shapes[h + "fus_p.weight"] = (48, ch, 1, 1)
        shapes[h + "coa_p.weight"] = (48, 256, 1, 1)
        for dw in ("dw1", "dw2"):
            shapes[h + dw + ".weight"] = (48, 1, 3, 3)
            shapes[h + dw + ".bias"] = (48,)
        shapes[h + "pw.weight"] = (48, 96, 1, 1)
        shapes[h + "pw.bias"] = (48,)
        shapes[h + "out_proj.weight"] = (ch, 48, 1, 1)
        shapes[h + "out_proj.bias"] = (ch,)
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


def inspect_head_npz(raw):
    """Validate exact bounded float32 array headers before NumPy allocation."""
    return _inspect_npz_schema(raw, head_shapes(), max_bytes=1024**2, label="head")


def inspect_v2_npz(raw):
    """Validate exact bounded float32 V2 p2_u0224 array headers before NumPy allocation."""
    return _inspect_npz_schema(raw, v2_shapes(), max_bytes=8 * 1024**2, label="V2")


def numpy_state_sha(state):
    """Canonical tensor-name/dtype/shape/value digest used by retained exports."""
    digest = hashlib.sha256()
    for name, array in sorted(state.items()):
        header = json.dumps([name, "torch.float32", list(array.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def model_contract(model="v2"):
    need(model == "v2", "Only the V2 (p2_u0224) model is supported.")
    assets = {
        "codeformer_r2.pth": dict(R2_FILE),
        "codeformer_u96_adapter.pth": dict(U96_FILE),
        "terminal_u0064_heads.npz": dict(HEAD_FILE),
        "v2_p2_u0224_heads.npz": dict(V2_FILE),
    }
    states = {
        "R2_515": R2_STATE,
        "U96_24": U96_STATE,
        "terminal_heads_36": HEAD_STATE,
        "V2_p2_u0224_212": V2_STATE,
    }
    return {
        "model": "v2",
        "assets": assets,
        "canonical_states": states,
        "frozen_tensors": 787,
        "policy": "CPU FP32 singleton512, w1, detach16, AdaIN, Conditioned SFT V2 (p2_u0224), torch.no_grad",
        "architecture": (
            "CodeFormer by S-Lab; U96 encoder adapters; terminal-u64 spatial heads; "
            "Conditioned SFT V2 top-down dilated pyramid and frequency-selective VQGAN heads (p2_u0224)"
        ),
        "license": "S-Lab License 1.0; third_party/codeformer/LICENSE",
        "internal_resolution": 512,
        "output_resolution": 256,
        "fidelity": 1.0,
        "adain": True,
        "detach_16": True,
        "inference_only": True,
        "model_qualified": False,
    }


def load_frozen_cpu_restorer(model="v2", model_dir=None):
    """Construct an exclusive V2 (p2_u0224) restorer after verifying every fixed artifact."""
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
             "Retained CodeFormer architecture differs.")
    arrays_u64 = inspect_head_npz(paths["terminal_u0064_heads.npz"].read_bytes())
    need(numpy_state_sha(arrays_u64) == HEAD_STATE, "Original terminal-u64 head values required.")
    arrays_v2 = inspect_v2_npz(paths["v2_p2_u0224_heads.npz"].read_bytes())
    need(numpy_state_sha(arrays_v2) == V2_STATE, "Exact V2 (p2_u0224) head values required.")

    import numpy as np
    import torch
    from torch.nn import functional as F
    from third_party.codeformer.codeformer_arch import CodeFormer, adaptive_instance_normalization
    need(torch.get_default_dtype() == torch.float32, "FP32 default dtype required.")
    for name, binding in ARCHITECTURE_SOURCES.items():
        loaded = sys.modules.get(name)
        need(loaded is not None and getattr(loaded, "__file__", None)
             and Path(loaded.__file__).resolve() == (ROOT / binding["path"]).resolve()
             and architecture_file_sha256(loaded.__file__) == binding["sha256"],
             "Exact loaded CodeFormer architecture origin required.")
    saved_r2 = torch.load(paths["codeformer_r2.pth"], map_location="cpu", weights_only=True, mmap=True)
    saved_u96 = torch.load(paths["codeformer_u96_adapter.pth"], map_location="cpu", weights_only=True, mmap=True)
    need(type(saved_r2) is dict and set(saved_r2) == {"params_ema"}, "Exact R2 inference format required.")
    need(type(saved_u96) is dict and set(saved_u96) == {"format", "adapter_state"}
         and saved_u96["format"] == "codeformer-spatial-adapter-inference-v1", "Exact U96 inference format required.")
    r2, u96 = saved_r2["params_ema"], saved_u96["adapter_state"]
    need(isinstance(r2, dict) and len(r2) == 515 and type(u96) is dict and set(u96) == set(adapter_shapes()),
         "Exact 515/24 tensor census required.")

    def state_sha(state):
        need(isinstance(state, dict) and all(
            isinstance(value, torch.Tensor) and value.dtype == torch.float32
            and value.device.type == "cpu" and not value.requires_grad and value.grad is None
            and torch.isfinite(value).all().item() for value in state.values()),
            "Detached finite CPU FP32 state required.")
        return numpy_state_sha({name: value.detach().contiguous().numpy() for name, value in state.items()})

    need(state_sha(r2) == R2_STATE and state_sha(u96) == U96_STATE
         and all(tuple(value.shape) == adapter_shapes()[name] for name, value in u96.items()),
         "Exact retained canonical model values/schema required.")

    class ResidualSkipAdapter(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            options = {"device": "cpu", "dtype": torch.float32}
            self.down = torch.nn.Conv2d(channels, 32, 1, bias=True, **options)
            self.depthwise = torch.nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=True, **options)
            self.up = torch.nn.Conv2d(32, channels, 1, bias=True, **options)
            self.activation = torch.nn.SiLU(inplace=False)

        def forward(self, observed):
            return observed + self.up(self.activation(self.depthwise(self.activation(self.down(observed)))))

    class SpatialHead(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            options = {"device": "cpu", "dtype": torch.float32}
            self.encoder = torch.nn.Conv2d(channels, WIDTH, 1, bias=True, **options)
            self.decoder = torch.nn.Conv2d(channels, WIDTH, 1, bias=False, **options)
            self.fused = torch.nn.Conv2d(channels, WIDTH, 1, bias=False, **options)
            self.coarse = torch.nn.Conv2d(256, WIDTH, 1, bias=False, **options)
            self.spatial = torch.nn.Conv2d(WIDTH, WIDTH, 3, padding=1, bias=True, **options)
            self.output = torch.nn.Conv2d(WIDTH, channels, 1, bias=True, **options)
            self.activation = torch.nn.GELU()

        def forward(self, used, decoder, fused, coarse):
            context = self.coarse(coarse.detach())
            context = F.interpolate(context, size=used.shape[-2:], mode="bilinear", align_corners=False)
            mixed = (self.encoder(used.detach()) + self.decoder(decoder) + self.fused(fused) + context) / 2.0
            logits = self.output(self.activation(self.spatial(self.activation(mixed))))
            return fused + torch.tanh(logits) * (fused - decoder)

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

    def _gauss_blur(z, k):
        p = k // 2
        out = F.avg_pool2d(F.pad(z, (p, p, p, p), mode="reflect"), kernel_size=k, stride=1)
        out = F.avg_pool2d(F.pad(out, (p, p, p, p), mode="reflect"), kernel_size=k, stride=1)
        return out

    class FrozenCPURestorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=list(SITES))
            self.base.load_state_dict(r2, strict=True)
            self.adapters = torch.nn.ModuleDict({site: ResidualSkipAdapter(CHANNELS[site]) for site in SITES})
            self.adapters.load_state_dict({name.removeprefix("adapters."): value for name, value in u96.items()}, strict=True)
            self.heads = torch.nn.ModuleDict({site: SpatialHead(CHANNELS[site]) for site in SITES})
            self.heads.load_state_dict({name.removeprefix("heads."): torch.from_numpy(value.copy())
                                       for name, value in arrays_u64.items()}, strict=True)
            self.v2_bundle = cond_v2.build_v2_trainable_modules(torch=torch)
            self.v2_bundle.load_state_dict({name: torch.from_numpy(value.copy())
                                           for name, value in arrays_v2.items()}, strict=True)
            self.v2_gamma = 1.0
            self._out_list = set(self.base.fuse_encoder_block[s] for s in self.base.connect_list)
            self._fuse_map = {self.base.fuse_generator_block[s]: s for s in self.base.connect_list}
            self._forward_lock = threading.Lock()
            self.last_forward_sites = ()
            self.last_base_calls = 0
            self.eval()
            self.requires_grad_(False)
            setattr(self.base, _OWNER_ATTRIBUTE, weakref.ref(self))

        def train(self, mode=True):
            need(mode is False, "Training mode is unavailable in this frozen runtime.")
            return super().train(False)

        def validate_invariants(self):
            named, state = dict(self.named_parameters()), self.state_dict()
            need(len(named) == len(state) == contract["frozen_tensors"] and len(self.base.state_dict()) == 515
                 and self.base.connect_list == list(SITES) and set(self.base.fuse_convs_dict) == set(SITES)
                 and self.base.fuse_generator_block == {"16": 6, "32": 9, "64": 12, "128": 15, "256": 18, "512": 21},
                 "Exact tensor census/four-site architecture required.")
            need(all(not module.training for module in self.modules())
                 and all(value.device.type == "cpu" and value.dtype == torch.float32 and not value.requires_grad
                         and value.grad is None and torch.isfinite(value).all().item() for value in named.values()),
                 "All tensors must remain finite frozen/eval CPU FP32 with no gradients.")
            adapters = {"adapters." + name: value for name, value in self.adapters.state_dict().items()}
            heads = {"heads." + name: value for name, value in self.heads.state_dict().items()}
            v2_sd = dict(self.v2_bundle.state_dict().items())
            need(set(adapters) == set(adapter_shapes())
                 and all(tuple(value.shape) == adapter_shapes()[name] for name, value in adapters.items())
                 and state_sha(self.base.state_dict()) == R2_STATE and state_sha(adapters) == U96_STATE,
                 "Frozen base/adapter canonical state changed.")
            need(set(heads) == set(head_shapes())
                 and all(tuple(value.shape) == head_shapes()[name] for name, value in heads.items())
                 and state_sha(heads) == HEAD_STATE, "Frozen terminal-u64 canonical state changed.")
            need(set(v2_sd) == set(v2_shapes())
                 and all(tuple(value.shape) == v2_shapes()[name] for name, value in v2_sd.items())
                 and state_sha(v2_sd) == V2_STATE, "Frozen V2 (p2_u0224) canonical state changed.")
            owner = getattr(self.base, _OWNER_ATTRIBUTE, None)
            need(isinstance(owner, weakref.ReferenceType) and owner() is self, "Exclusive owned base required.")
            return {"pass": True, "frozen_tensors": contract["frozen_tensors"], "model": "v2"}

        def _encode_and_quantize(self, sample_512):
            enc_feat_dict = {}
            x = sample_512
            for i, block in enumerate(self.base.encoder.blocks):
                x = block(x)
                if i in self._out_list:
                    enc_feat_dict[str(x.shape[-1])] = x.clone()
            lq_feat = x
            pos_emb = self.base.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
            query_emb = self.base.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
            for layer in self.base.ft_layers:
                query_emb = layer(query_emb, query_pos=pos_emb)
            logits = self.base.idx_pred_layer(query_emb).permute(1, 0, 2)
            soft_one_hot = F.softmax(logits, dim=2)
            _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
            quant_feat = self.base.quantize.get_codebook_feat(top_idx, shape=[x.shape[0], 16, 16, 256]).detach()
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)
            return quant_feat, enc_feat_dict, logits, lq_feat

        def _decode_sft(self, quant_feat, used_dict, *, cond_t=None, use_head_v2=False, w_by_site=None):
            x = quant_feat
            coarse = used_dict["32"].detach()
            for i, block in enumerate(self.base.generator.blocks):
                x = block(x)
                if i in self._fuse_map:
                    s = self._fuse_map[i]
                    used = used_dict[s]
                    dec = x
                    w_eff = 1.0 if w_by_site is None else w_by_site[s]
                    fused = self.base.fuse_convs_dict[s](used, dec, w_eff)
                    post = self.heads[s](used, dec, fused, coarse)
                    if use_head_v2 and cond_t is not None:
                        post = self.v2_bundle.refine_post_sft(s, post, used, dec, fused, coarse, cond_t)
                    x = post
            norm = (x + 1.0) / 2.0
            return F.interpolate(norm.clamp(0.0, 1.0), (256, 256), mode="bilinear", align_corners=False)

        def restore_canvas(self, canvas):
            """Run one frozen Conditioned SFT V2 (p2_u0224) restoration on a 256x256 RGB8 canvas."""
            need(not torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
                 "Use torch.no_grad; gradient/inference_mode runtime refused.")
            need(isinstance(canvas, np.ndarray) and canvas.shape == (256, 256, 3) and canvas.dtype == np.uint8,
                 "256x256 uint8 RGB canvas required.")
            need(self._forward_lock.acquire(blocking=False), "Nested/concurrent forward refused.")
            self.last_forward_sites, self.last_base_calls = (), 0
            try:
                self.validate_invariants()
                with torch.autocast(device_type="cpu", enabled=False):
                    inp_f32 = canvas.astype(np.float32) / np.float32(255.0)
                    inp_t = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).to(dtype=torch.float32)[None] / 255.0
                    sample_512 = F.interpolate(inp_t, (512, 512), mode="bilinear", align_corners=False) * 2.0 - 1.0
                    quant_feat, enc_feat_dict, _, _ = self._encode_and_quantize(sample_512)
                    u96_dict = {s: self.adapters[s](enc_feat_dict[s].detach()) for s in SITES}
                    u64_raw = self._decode_sft(quant_feat, u96_dict)[0].detach().cpu().numpy().transpose(1, 2, 0).copy()
                    cond_t = cond_v2.compute_v2_conditioning_torch(inp_t, torch=torch)
                    used_p2 = self.v2_bundle.refine_sft_features(u96_dict, cond_t)
                    p2_t = self._decode_sft(quant_feat, used_p2, cond_t=cond_t, use_head_v2=True)
                    p2_raw = p2_t[0].detach().cpu().numpy().transpose(1, 2, 0).copy()
                    p2_c, cond_info = cond_v2.condition_output_v2_numpy(
                        p2_raw, inp_f32, np=np, baseline_rgb=u64_raw,
                        beta=cond_v2.DEFAULT_ANCHOR_BETA, gamma=self.v2_gamma, dark_drift_align=True,
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
                        a_usm = float(self.v2_gamma) * w_global * (0.05 + sev * (0.90 - 0.05))
                        a_vq = float(self.v2_gamma) * w_global * (0.08 + sev * (1.15 - 0.08))
                        vq_seed = u96_dict if g_dark > 0.0 else used_p2
                        vq_t = self._decode_sft(
                            quant_feat, vq_seed,
                            w_by_site={"32": 0.50, "64": 0.30, "128": 0.08, "256": 0.04},
                        )
                        p2_c_t = torch.from_numpy(p2_c.transpose(2, 0, 1).copy()).to(dtype=torch.float32)[None]
                        hp3_p2 = (p2_c_t - _gauss_blur(p2_c_t, 3))[0].detach().cpu().numpy().transpose(1, 2, 0).copy()
                        hp7_diff = ((vq_t - _gauss_blur(vq_t, 7)) - (p2_c_t - _gauss_blur(p2_c_t, 7)))[0].detach().cpu().numpy().transpose(1, 2, 0).copy()
                        m_np = m256[0].detach().cpu().numpy().transpose(1, 2, 0).copy()
                        v2_raw = np.clip(p2_c + m_np * (a_usm * hp3_p2 + a_vq * fmask_np * hp7_diff), 0.0, 1.0).astype(np.float32)
                        if g_dark > 0.0:
                            f64 = v2_raw.astype(np.float64)
                            b64 = u64_raw.astype(np.float64)
                            v2_raw = np.clip(f64 - g_dark * cond_v2.smooth_coarse_256_numpy(f64 - b64, np=np), 0.0, 1.0).astype(np.float32)
                    else:
                        v2_raw = p2_c
                need(v2_raw.shape == (256, 256, 3) and v2_raw.dtype == np.float32
                     and np.isfinite(v2_raw).all() and 0.0 <= float(v2_raw.min()) <= float(v2_raw.max()) <= 1.0,
                     "Invalid V2 output schema.")
                self.last_forward_sites, self.last_base_calls = SITES, 1
                self.validate_invariants()
                return v2_raw
            finally:
                self._forward_lock.release()

        def forward(self, sample, *, w=1.0, detach_16=True, adain=True):
            need(not torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
                 "Use torch.no_grad; gradient/inference_mode runtime refused.")
            need(w == 1.0 and detach_16 is True and adain is True, "Fixed w1/detach16/AdaIN policy required.")
            need(isinstance(sample, torch.Tensor) and tuple(sample.shape) == (1, 3, 512, 512)
                 and sample.dtype == torch.float32 and sample.device.type == "cpu" and not sample.requires_grad
                 and torch.isfinite(sample).all().item() and sample.min().item() >= -1.0 and sample.max().item() <= 1.0,
                 "Finite singleton CPU FP32 prepared512 input in [-1,1] required.")
            inp_256 = F.interpolate(((sample + 1.0) / 2.0).clamp(0.0, 1.0), (256, 256), mode="bilinear", align_corners=False)
            canvas_u8 = np.floor(inp_256[0].permute(1, 2, 0).contiguous().numpy() * np.float32(255.0)).astype(np.uint8)
            v2_raw = self.restore_canvas(canvas_u8)
            out_256 = torch.from_numpy(v2_raw.transpose(2, 0, 1).copy()).to(dtype=torch.float32)[None]
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
    need(all(value.untyped_storage().data_ptr() != r2[name].untyped_storage().data_ptr()
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
