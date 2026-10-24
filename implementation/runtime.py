"""Frozen CPU inference for the retained U96 and original terminal-u64 models.

Numerics are preserved from the research inference implementations. CodeFormer
remains attributed to S-Lab in third_party/codeformer, including its license.
This module imports no research service, training helper, or historical output.
Importing it constructs no model and performs no network or device discovery.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import stat
import sys
import threading
import weakref
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "implementation"
SITES = ("32", "64", "128", "256")
CHANNELS = {"32": 256, "64": 256, "128": 128, "256": 128}
WIDTH = 32
MAX_IMAGE_BYTES = 10 * 1024**2
MAX_IMAGE_SIDE = 1600
R2_FILE = {"bytes": 376609465, "sha256": "d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896"}
U96_FILE = {"bytes": 212827, "sha256": "958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0"}
HEAD_FILE = {"bytes": 685818, "sha256": "77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17"}
R2_STATE = "a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f"
U96_STATE = "fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a"
HEAD_STATE = "6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be"
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
    path = Path(path).absolute()
    need(path.is_relative_to(ROOT), "Model/source must stay inside this installation.")
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
    """Hash source with CRLF converted to LF; preserve every other source byte.

    Architecture-only policy supports Git's Windows line-ending conversion.
    Bare CR is refused. Model artifacts always use exact unmodified file_pin.
    """
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


def inspect_head_npz(raw):
    """Validate exact bounded float32 array headers before NumPy allocation."""
    need(type(raw) is bytes and 0 < len(raw) < 1024**2, "Bounded head NPZ bytes required.")
    shapes = head_shapes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        items, names = archive.infolist(), archive.namelist()
        need(len(items) == len(set(names)) == 36 and {n[:-4] for n in names} == set(shapes)
             and all("/" not in n and "\\" not in n and ":" not in n and n.endswith(".npy") for n in names),
             "Expected exactly 36 unique flat head arrays.")
        need(all(not item.flag_bits & 1 and not stat.S_ISLNK(item.external_attr >> 16) for item in items)
             and sum(item.file_size for item in items) < 1024**2, "Bounded ordinary expanded NPZ required.")
        import numpy as np
        for member in items:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                need(version in ((1, 0), (2, 0)), "Unsupported head NPY version.")
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(stream, max_header_size=4096)
                need(shape == shapes[member.filename[:-4]] and dtype == np.dtype("float32")
                     and not dtype.hasobject and not fortran, "Exact float32 head schema required.")
                need(member.file_size == stream.tell() + math.prod(shape) * 4, "Incorrect head array size.")
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            result = {}
            for name, shape in shapes.items():
                array = saved[name]
                need(array.shape == shape and array.dtype == np.float32 and np.isfinite(array).all(),
                     "Finite exact head values required.")
                result[name] = np.ascontiguousarray(array).copy()
            return result


def numpy_state_sha(state):
    """Canonical tensor-name/dtype/shape/value digest used by retained exports."""
    digest = hashlib.sha256()
    for name, array in sorted(state.items()):
        header = json.dumps([name, "torch.float32", list(array.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def model_contract(model="u96"):
    need(model in ("u96", "u64"), "Choose u96 or u64.")
    assets = {"codeformer_r2.pth": dict(R2_FILE), "codeformer_u96_adapter.pth": dict(U96_FILE)}
    states = {"R2_515": R2_STATE, "U96_24": U96_STATE}
    if model == "u64":
        assets["terminal_u0064_heads.npz"] = dict(HEAD_FILE)
        states["terminal_heads_36"] = HEAD_STATE
    return {
        "model": model, "assets": assets, "canonical_states": states,
        "frozen_tensors": 575 if model == "u64" else 539,
        "policy": "CPU FP32 singleton512, w1, detach16, AdaIN, torch.no_grad",
        "architecture": "CodeFormer by S-Lab; four learned encoder-feature residual adapters"
                        + ("; original terminal-u64 post-fusion heads" if model == "u64" else ""),
        "license": "S-Lab License 1.0; third_party/codeformer/LICENSE",
        "internal_resolution": 512, "output_resolution": 256,
        "fidelity": 1.0, "adain": True, "detach_16": True,
        "inference_only": True, "model_qualified": False,
    }


def load_frozen_cpu_restorer(model="u96", model_dir=None):
    """Construct an exclusive model after verifying every required fixed artifact."""
    contract = model_contract(model)
    directory = MODEL_DIR if model_dir is None else Path(model_dir)
    paths = {name: ordinary_file(directory / name) for name in contract["assets"]}
    for name, expected in contract["assets"].items():
        need(file_pin(paths[name]) == expected, "Exact retained model bytes required: " + name)
    for binding in ARCHITECTURE_SOURCES.values():
        need(architecture_file_sha256(ROOT / binding["path"]) == binding["sha256"],
             "Retained CodeFormer architecture differs.")
    arrays = inspect_head_npz(paths["terminal_u0064_heads.npz"].read_bytes()) if model == "u64" else None
    if arrays is not None:
        need(numpy_state_sha(arrays) == HEAD_STATE, "Original terminal-u64 head values required.")

    import torch
    from torch.nn import functional as F
    from third_party.codeformer.codeformer_arch import CodeFormer
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

    class FrozenCPURestorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=list(SITES))
            self.base.load_state_dict(r2, strict=True)
            self.adapters = torch.nn.ModuleDict({site: ResidualSkipAdapter(CHANNELS[site]) for site in SITES})
            self.adapters.load_state_dict({name.removeprefix("adapters."): value for name, value in u96.items()}, strict=True)
            if model == "u64":
                self.heads = torch.nn.ModuleDict({site: SpatialHead(CHANNELS[site]) for site in SITES})
                self.heads.load_state_dict({name.removeprefix("heads."): torch.from_numpy(value.copy())
                                           for name, value in arrays.items()}, strict=True)
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
            need(set(adapters) == set(adapter_shapes())
                 and all(tuple(value.shape) == adapter_shapes()[name] for name, value in adapters.items())
                 and state_sha(self.base.state_dict()) == R2_STATE and state_sha(adapters) == U96_STATE,
                 "Frozen base/adapter canonical state changed.")
            if model == "u64":
                heads = {"heads." + name: value for name, value in self.heads.state_dict().items()}
                need(set(heads) == set(head_shapes())
                     and all(tuple(value.shape) == head_shapes()[name] for name, value in heads.items())
                     and state_sha(heads) == HEAD_STATE, "Frozen terminal-u64 canonical state changed.")
            owner = getattr(self.base, _OWNER_ATTRIBUTE, None)
            need(isinstance(owner, weakref.ReferenceType) and owner() is self, "Exclusive owned base required.")
            return {"pass": True, "frozen_tensors": contract["frozen_tensors"], "model": model}

        def forward(self, sample, *, w=1.0, detach_16=True, adain=True):
            need(not torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
                 "Use torch.no_grad; gradient/inference_mode runtime refused.")
            need(w == 1.0 and detach_16 is True and adain is True, "Fixed w1/detach16/AdaIN policy required.")
            need(isinstance(sample, torch.Tensor) and tuple(sample.shape) == (1, 3, 512, 512)
                 and sample.dtype == torch.float32 and sample.device.type == "cpu" and not sample.requires_grad
                 and torch.isfinite(sample).all().item() and sample.min().item() >= -1.0 and sample.max().item() <= 1.0,
                 "Finite singleton CPU FP32 prepared512 input in [-1,1] required.")
            need(self._forward_lock.acquire(blocking=False), "Nested/concurrent forward refused.")
            handles, pending, completed = [], {}, []
            coarse, base_calls = None, 0
            thread_id = threading.get_ident()
            self.last_forward_sites, self.last_base_calls = (), 0

            def guard_thread():
                need(threading.get_ident() == thread_id, "Foreign-thread base/SFT call refused.")

            def base_pre_hook(module, args):
                nonlocal base_calls
                guard_thread()
                need(module is self.base and base_calls == 0 and len(args) == 1 and args[0] is sample,
                     "Exactly one explicit base call required.")
                base_calls += 1

            def pre_hook(site):
                def adapt_and_capture(module, args):
                    nonlocal coarse
                    guard_thread()
                    need(module is self.base.fuse_convs_dict[site] and base_calls == 1 and len(args) == 3
                         and args[2] == 1.0 and len(completed) < 4 and site == SITES[len(completed)] and not pending,
                         "One ordered U96 pre-SFT adaptation required.")
                    encoder, decoder, _ = args
                    shape = (1, CHANNELS[site], int(site), int(site))
                    need(encoder.shape == decoder.shape == shape and encoder.dtype == decoder.dtype == torch.float32
                         and encoder.device.type == decoder.device.type == "cpu" and not encoder.requires_grad
                         and not decoder.requires_grad and torch.isfinite(encoder).all().item()
                         and torch.isfinite(decoder).all().item(), "Finite no-grad CPU SFT inputs required.")
                    used = self.adapters[site](encoder.detach())
                    need(used.shape == shape and used.dtype == torch.float32 and used.device.type == "cpu"
                         and not used.requires_grad and torch.isfinite(used).all().item(), "Invalid frozen U96 feature.")
                    if site == "32":
                        need(coarse is None, "Capture coarse32 exactly once.")
                        coarse = used.detach()
                    pending[site] = (used, decoder)
                    return used, decoder, args[2]
                return adapt_and_capture

            def post_hook(site):
                def correct(module, args, fused):
                    guard_thread()
                    need(module is self.base.fuse_convs_dict[site] and set(pending) == {site}
                         and len(completed) < 4 and site == SITES[len(completed)], "Ordered post-SFT capture required.")
                    used, decoder = pending.pop(site)
                    need(len(args) == 3 and args[0] is used and args[1] is decoder and args[2] == 1.0
                         and isinstance(fused, torch.Tensor) and fused.shape == used.shape
                         and fused.dtype == torch.float32 and fused.device.type == "cpu"
                         and not fused.requires_grad and torch.isfinite(fused).all().item(), "Invalid stock SFT result.")
                    corrected = self.heads[site](used, decoder, fused, coarse) if model == "u64" else fused
                    need(corrected.shape == fused.shape and corrected.dtype == torch.float32
                         and corrected.device.type == "cpu" and not corrected.requires_grad
                         and torch.isfinite(corrected).all().item(), "Invalid same-geometry corrected result.")
                    completed.append(site)
                    return corrected
                return correct

            try:
                self.validate_invariants()
                need(all(not module._forward_pre_hooks and not module._forward_hooks for module in self.modules()),
                     "Clean owned hook namespace required.")
                handles.append(self.base.register_forward_pre_hook(base_pre_hook))
                for site in SITES:
                    module = self.base.fuse_convs_dict[site]
                    handles.append(module.register_forward_pre_hook(pre_hook(site)))
                    handles.append(module.register_forward_hook(post_hook(site)))
                with torch.autocast(device_type="cpu", enabled=False):
                    result = self.base(sample, w=w, detach_16=detach_16, adain=adain)
                need(base_calls == 1 and tuple(completed) == SITES and not pending,
                     "One complete ordered four-site forward required.")
                need(isinstance(result, tuple) and len(result) == 3, "Stock native/logits/LQ tuple required.")
                for value, shape in zip(result, ((1, 3, 512, 512), (1, 256, 1024), (1, 256, 16, 16))):
                    need(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and value.dtype == torch.float32
                         and value.device.type == "cpu" and not value.requires_grad and torch.isfinite(value).all().item(),
                         "Invalid stock output schema.")
                self.validate_invariants()
                return result
            finally:
                try:
                    for handle in reversed(handles):
                        handle.remove()
                    pending.clear()
                    coarse = None
                    self.last_forward_sites, self.last_base_calls = tuple(completed), base_calls
                    need(all(not module._forward_pre_hooks and not module._forward_hooks for module in self.modules()),
                         "Hooks must clean up on every exit.")
                finally:
                    self._forward_lock.release()

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
