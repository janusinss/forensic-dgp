"""Guarded paired spatial-objective adapters; standard-library import only.

All original R2 tensors remain frozen. Four new residual translators run on
detached encoder arguments immediately before unchanged SFT modules. The worker
owns prospective call, backward and optimizer reservations and actual launch.
Training is feature-only: this source exposes no optimizing decoder forward.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys

VM_NAME = 'cctv_codeformer_spatial_continuation_v1_vm'
SITES = ('32', '64', '128', '256')
CHANNELS = {'32': 256, '64': 256, '128': 128, '256': 128}
WIDTH = 32
INITIALIZATION_SEED = 20261019
ADAPTER_PREFIX = 'adapters.'
R2_DIGEST = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
MODEL_DIGESTS = {'r2':R2_DIGEST}
ASSETS = {'r2': {'path':'weights/codeformer_r2.pth','bytes':376609465,
    'sha256':'d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'}}
INITIAL_ADAPTER_DIGEST = '20398c0179f55436cc03333ed439c3f61e9a33e026274af548423e2753fd6b4b'
ARCHITECTURE_SOURCES = {
    'third_party.codeformer.codeformer_arch': ('third_party/codeformer/codeformer_arch.py',
        'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb'),
    'third_party.codeformer.vqgan_arch': ('third_party/codeformer/vqgan_arch.py',
        'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf'),
    'third_party.codeformer.registry': ('third_party/codeformer/registry.py',
        'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e')}
SLOT_DIVISOR = 3
BASE_LR = 1e-4
WARMUP_UPDATES = 10
MAX_UPDATES = 96
GRADIENT_CLIP_NORM = 1.
_TORCH = _ROOT = _PROTOCOL_SHA = _ARCHITECTURE = None
_CANDIDATE_CLASS = _ADAPTER_CLASS = None
_CANDIDATE_CREATED = False


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            row = part.lstat()
            require(not stat.S_ISLNK(row.st_mode) and not getattr(row, 'st_file_attributes', 0) & 1024,
                    'Linked/reparse model, source or root refused')
    row = path.stat()
    require(stat.S_ISREG(row.st_mode) and row.st_nlink == 1, 'Unique regular model/source required')
    return path.resolve(strict=True)


def read_json(path):
    require(regular(path).stat().st_size <= 8 * 1024**2, 'Bounded protocol metadata required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate metadata field refused'); result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite metadata refused')))


def expected_adapter_shapes():
    result = {}
    for site in SITES:
        channels = CHANNELS[site]
        for name, shape in {
            'down.weight': (WIDTH, channels, 1, 1), 'down.bias': (WIDTH,),
            'depthwise.weight': (WIDTH, 1, 3, 3), 'depthwise.bias': (WIDTH,),
            'up.weight': (channels, WIDTH, 1, 1), 'up.bias': (channels,)}.items():
            result[ADAPTER_PREFIX + site + '.' + name] = shape
    require(len(result) == 24 and sum(math.prod(shape) for shape in result.values()) == 51328,
            'Exact24-tensor/51328-parameter schema differs')
    return result


def learning_rate(update):
    require(type(update) is int and 1 <= update <= MAX_UPDATES, 'Finite1..96 absolute update required')
    return BASE_LR * min(1., update / WARMUP_UPDATES)


def validate_readout_difference(error, magnitude):
    require(isinstance(error, (float, int)) and isinstance(magnitude, (float, int)) and
            math.isfinite(error) and math.isfinite(magnitude) and error >= 0 and magnitude >= 0,
            'Finite nonnegative readout evidence required')
    bound = 8 * 2**-23 * max(1., magnitude)
    require(error <= bound, 'Explicit readout rounding witness exceeded')
    return {'pass': True, 'maximum_absolute_error': error, 'bound': bound,
            'bound_formula': '8*float32_epsilon*max(1,maxabs(native_RGB))',
            'same_native_reference_only': True, 'quality_gate_tolerance_changed': False}


def numpy_feature_loss(used, clean, scales, *, np):
    require(set(used) == set(clean) == set(scales) == set(SITES), 'All four feature sites required')
    values = {}
    for site in SITES:
        shape = (1, CHANNELS[site], int(site), int(site))
        require(used[site].shape == clean[site].shape == shape and
                used[site].dtype == clean[site].dtype == np.float32 and
                np.isfinite(used[site]).all() and np.isfinite(clean[site]).all() and
                isinstance(scales[site], (float, int)) and math.isfinite(scales[site]) and scales[site] >= 1e-6,
                'Finite exact feature arrays/fixed scale required')
        values[site] = float(np.abs(used[site].astype(np.float64) - clean[site].astype(np.float64)).mean() / scales[site])
    return sum(values.values()) / 4., values


def manual_runtime_guard(root, protocol_sha):
    """Read-only exact Linux/tmux/path/protocol check, before writes or Torch."""
    require(platform.system() == 'Linux' and bool(os.environ.get('TMUX')), 'Manual Linux tmux runtime required')
    root = Path(root).absolute(); expected = Path.home() / 'forensic-dgp' / VM_NAME
    for part in (root, *root.parents):
        row = part.lstat()
        require(not stat.S_ISLNK(row.st_mode) and not getattr(row, 'st_file_attributes', 0) & 1024,
                'Linked/reparse runtime path refused')
    require(root.resolve(strict=True) == expected.resolve(strict=True), 'Exact new manual VM root required')
    require(isinstance(protocol_sha, str) and re.fullmatch('[0-9a-f]{64}', protocol_sha), 'Frozen protocol SHA required')
    protocol_path = regular(root / 'protocol.json')
    require(sha256(protocol_path) == protocol_sha, 'Exact protocol bytes differ')
    protocol = read_json(protocol_path); file_pins = protocol['files']
    require(protocol.get('vm_name') == VM_NAME, 'Protocol must name this new pilot')
    return root,protocol,file_pins


def activate_runtime(root, protocol_sha):
    """Only actual exact manual tmux/L4 runtime may import Torch."""
    global _TORCH, _ROOT, _PROTOCOL_SHA
    root,protocol,file_pins=manual_runtime_guard(root,protocol_sha)
    def packet_file(name, expected_sha=None):
        pin = file_pins[name]; path = regular(root / name)
        require(isinstance(pin, dict) and set(pin) == {'bytes', 'sha256'} and
                type(pin['bytes']) is int and pin['bytes'] > 0 and path.stat().st_size == pin['bytes'] and
                isinstance(pin['sha256'], str) and re.fullmatch('[0-9a-f]{64}', pin['sha256']) and
                sha256(path) == pin['sha256'] and (expected_sha is None or pin['sha256'] == expected_sha),
                'Frozen packet source/asset bytes or SHA differ')
        return path
    for _, (name, digest) in ARCHITECTURE_SOURCES.items():
        packet_file(name, digest)
    for name in ('third_party/__init__.py', 'third_party/codeformer/__init__.py',
                 'research/codeformer_spatial_continuation_v1/__init__.py',
                 'research/codeformer_spatial_continuation_v1/model.py'):
        packet_file(name)
    require(regular(__file__) == regular(root / 'research/codeformer_spatial_continuation_v1/model.py'),
            'Model module imported outside exact packet')
    for item in ASSETS.values():
        path = packet_file(item['path'], item['sha256'])
        require(path.stat().st_size == item['bytes'],
                'Exact retained R2 initializer asset required')
    if _TORCH is not None:
        require(_ROOT == root.resolve(strict=True) and _PROTOCOL_SHA == protocol_sha, 'Runtime cannot rebind')
        return _TORCH
    require(not any(name in sys.modules for name in ('third_party', 'third_party.codeformer', *ARCHITECTURE_SOURCES)),
            'Architecture must import fresh after source checks')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == 'NVIDIA L4', 'Exactly one NVIDIA L4 required')
    require(not torch.is_inference_mode_enabled(), 'Ordinary autograd runtime required')
    _TORCH, _ROOT, _PROTOCOL_SHA = torch, root.resolve(strict=True), protocol_sha
    return torch


def _runtime():
    require(_TORCH is not None and _ROOT is not None, 'Activate exact guarded manual L4 runtime first')
    return _TORCH


def tensor_state_digest(state):
    torch = _runtime(); digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor), 'Tensor-only state required')
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_digest(net):
    return tensor_state_digest(net.state_dict())


def _base(net):
    return net.base if getattr(net, '_spatial_continuation_candidate_v1', False) else net


def frozen_state_digest(net):
    return state_digest(_base(net))


def cpu_state(net):
    _runtime(); return {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}


def adapter_state(net):
    parameter_census(net)
    return {name: value.detach().cpu().clone() for name, value in net.state_dict().items() if name.startswith(ADAPTER_PREFIX)}


def _frozen_teacher(net):
    torch = _runtime()
    require(len(net.state_dict()) == len(tuple(net.parameters())) == 515 and
            all(not m.training for m in net.modules()) and all(not p.requires_grad and p.grad is None and
            p.dtype == torch.float32 and p.device.type == 'cuda' for p in net.parameters()), 'Exact frozen515 CUDA FP32 eval model required')


def _architecture():
    global _ARCHITECTURE
    _runtime()
    if _ARCHITECTURE is None: _ARCHITECTURE = importlib.import_module('third_party.codeformer.codeformer_arch')
    for module, (name, digest) in ARCHITECTURE_SOURCES.items():
        require(Path(sys.modules[module].__file__).resolve(strict=True) == regular(_ROOT / name) and
                sha256(regular(_ROOT / name)) == digest, 'Architecture origin or bytes changed')
    return _ARCHITECTURE


def load_frozen_baseline(name):
    torch = _runtime(); require(name in ASSETS, 'Known baseline required')
    asset = ASSETS[name]; path = regular(_ROOT / asset['path'])
    require(path.stat().st_size == asset['bytes'] and sha256(path) == asset['sha256'], 'Model asset fingerprint differs')
    architecture = _architecture(); cpu_rng = torch.get_rng_state().clone(); gpu_rng = torch.cuda.get_rng_state(0).clone()
    with torch.random.fork_rng(devices=[0]):
        net = architecture.CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=list(SITES)).float().eval()
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        key = 'model' if name == 'parent' else 'params_ema'
        require(isinstance(saved, dict) and key in saved, 'Exact input model export key required')
        state = saved[key]
        require(len(state) == 515 and set(state) == set(net.state_dict()) and all(isinstance(v, torch.Tensor) and
                v.dtype == torch.float32 and torch.isfinite(v).all().item() for v in state.values()), 'Finite exact515 input tensors required')
        net.load_state_dict(state, strict=True); del saved, state
        net = net.to('cuda').float().eval().requires_grad_(False)
    _frozen_teacher(net); digest = state_digest(net)
    require(digest == MODEL_DIGESTS[name] and sha256(path) == asset['sha256'], 'Actual frozen baseline differs')
    require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state(0)), 'Baseline factory changed RNG')
    return net, {'arm': name, 'canonical515_sha256': digest, 'weights': dict(asset), 'factory_RNG_preserved': True,
                 'neural_calls': 0, 'optimizer_state_installed': False}


def _classes():
    global _ADAPTER_CLASS, _CANDIDATE_CLASS
    torch = _runtime()
    if _CANDIDATE_CLASS is not None: return _ADAPTER_CLASS, _CANDIDATE_CLASS
    class ResidualSkipAdapter(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.down = torch.nn.Conv2d(channels, WIDTH, 1, bias=True)
            self.depthwise = torch.nn.Conv2d(WIDTH, WIDTH, 3, padding=1, groups=WIDTH, bias=True)
            self.up = torch.nn.Conv2d(WIDTH, channels, 1, bias=True)
            self.activation = torch.nn.SiLU(inplace=False)
            torch.nn.init.zeros_(self.up.weight); torch.nn.init.zeros_(self.up.bias)
        def forward(self, value):
            return value + self.up(self.activation(self.depthwise(self.activation(self.down(value)))))
    class SkipFeatureCandidate(torch.nn.Module):
        _spatial_continuation_candidate_v1 = True
        def __init__(self, frozen_base):
            super().__init__(); self.base = frozen_base
            self.adapters = torch.nn.ModuleDict({site: ResidualSkipAdapter(CHANNELS[site]) for site in SITES})
        def train(self, mode=True):
            return super().train(False)
        def forward(self, *args, **kwargs):
            raise ValueError('Worker must reserve a full call and use forward_record/forward_candidate; no hidden wrapper call')
    _ADAPTER_CLASS, _CANDIDATE_CLASS = ResidualSkipAdapter, SkipFeatureCandidate
    return _ADAPTER_CLASS, _CANDIDATE_CLASS


def parameter_census(net):
    torch = _runtime(); require(getattr(net, '_spatial_continuation_candidate_v1', False), 'New skip candidate required')
    _frozen_teacher(net.base); named = dict(net.named_parameters()); shapes = expected_adapter_shapes()
    require(len(net.state_dict()) == len(named) == 539 and len(net.base.state_dict()) == 515, 'Exact515base+24adapter tensor census required')
    trainable = {}
    for name, value in named.items():
        is_adapter = name in shapes
        require(name.startswith('base.') or is_adapter, 'Unexpected candidate parameter namespace')
        require(value.dtype == torch.float32 and value.device.type == 'cuda' and value.requires_grad == is_adapter,
                'Exact adapter-only FP32 CUDA trainability differs: ' + name)
        if is_adapter:
            require(tuple(value.shape) == shapes[name], 'Adapter tensor shape differs'); trainable[name] = value.numel()
        else: require(value.grad is None, 'Original frozen parameter acquired a gradient')
    require(len(trainable) == 24 and sum(trainable.values()) == 51328 and all(not m.training for m in net.modules()),
            'Exact24/51328 eval adapter census required')
    return {'trainable_names': sorted(trainable), 'trainable_tensors': 24, 'frozen_tensors': 515,
            'trainable_scalar_parameters': 51328, 'candidate_state_tensors': 539,
            'frozen_scalar_parameters': sum(v.numel() for v in net.base.parameters()),
            'sites': {site: 6 for site in SITES}}


def make_candidate(r2):
    global _CANDIDATE_CREATED
    torch = _runtime(); require(not _CANDIDATE_CREATED, 'Construct this finite candidate once')
    _frozen_teacher(r2); require(state_digest(r2) == R2_DIGEST, 'Exact R2 initializer required')
    _, candidate_class = _classes(); cpu_rng = torch.get_rng_state().clone(); gpu_rng = torch.cuda.get_rng_state(0).clone()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(INITIALIZATION_SEED); torch.cuda.manual_seed_all(INITIALIZATION_SEED)
        candidate = candidate_class(deepcopy(r2).eval().requires_grad_(False)).to('cuda').float().eval()
    census = parameter_census(candidate)
    require(frozen_state_digest(candidate) == R2_DIGEST and all(p.grad is None for p in candidate.parameters()), 'Fresh base/gradient state differs')
    require(all(torch.count_nonzero(candidate.adapters[s].up.weight).item() == 0 and
                torch.count_nonzero(candidate.adapters[s].up.bias).item() == 0 for s in SITES), 'Final adapter projection must be zero')
    require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state(0)), 'Candidate factory changed RNG')
    require(tensor_state_digest(adapter_state(candidate)) == INITIAL_ADAPTER_DIGEST, 'Exact original identity24 initializer differs')
    _CANDIDATE_CREATED = True
    return candidate, census, {'complete': True, 'base_canonical515_sha256': R2_DIGEST,
        'new_adapter_sha256': tensor_state_digest(candidate.adapters.state_dict()), 'initialization_seed': INITIALIZATION_SEED,
        'zero_only_final_up_projection': True, 'candidate_all_eval': True, 'factory_RNG_preserved': True,
        'original_all515_frozen': True, 'census': census, 'historical_optimizer_RNG_sampler_installed': False,
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_constructions': 0}


def trainable_parameters(net):
    census = parameter_census(net); named = dict(net.named_parameters())
    return [named[name] for name in census['trainable_names']]


def optimizer_groups(net):
    return [{'name': 'skip_adapters', 'params': trainable_parameters(net), 'lr': learning_rate(1), 'base_lr': BASE_LR}]


def make_optimizer(net):
    """Worker must reserve construction; this helper is never called on import."""
    torch = _runtime()
    optimizer = torch.optim.Adam(optimizer_groups(net), betas=(.9, .99), eps=1e-8, weight_decay=0.)
    require(len(optimizer.state) == 0, 'Fresh empty Adam state required')
    return optimizer


def _rgb(value, size, bounded=True):
    torch = _runtime()
    require(isinstance(value, torch.Tensor) and value.shape == (1, 3, size, size) and value.dtype == torch.float32 and
            value.device.type == 'cuda' and torch.isfinite(value).all().item(), 'Finite singleton FP32 CUDA RGB required')
    if bounded: require(value.min().item() >= 0 and value.max().item() <= 1, 'RGB must be within[0,1]')


def prepare_input(value):
    torch = _runtime(); _rgb(value, 256)
    require(not value.requires_grad and not torch.is_inference(value), 'Ordinary fixed RGB input required')
    return torch.nn.functional.interpolate(value, (512, 512), mode='bilinear', align_corners=False) * 2 - 1


def unclipped_rgb(native):
    _rgb(native, 512, bounded=False); value = (native.float() + 1) / 2
    horizontal = value[..., 0::2] * .5 + value[..., 1::2] * .5
    return horizontal[..., 0::2, :] * .5 + horizontal[..., 1::2, :] * .5


def delivered_rgb(native):
    torch = _runtime(); _rgb(native, 512, bounded=False)
    with torch.no_grad():
        return torch.nn.functional.interpolate(((native.detach().float() + 1) / 2).clamp(0, 1),
                                               (256, 256), mode='bilinear', align_corners=False)


def png_tensor(delivered):
    torch = _runtime(); _rgb(delivered, 256)
    return (delivered.detach() * 255).floor().to(torch.uint8)


@dataclass(frozen=True)
class ForwardRecord:
    native: object
    unclipped256: object
    delivered256: object
    logits: object
    lq: object
    tokens: object
    post_adain: object
    observed_features: object
    used_features: object
    U_readout_parity: object


@dataclass(frozen=True)
class TeacherTarget:
    unclipped256: object
    kind: str


def forward_record(net, input256, *, ledger=None, context=None):
    """Exactly one base call; all recording/intervention hooks are scoped."""
    torch = _runtime(); base = _base(net)
    if getattr(net, '_spatial_continuation_candidate_v1', False): parameter_census(net)
    else: _frozen_teacher(base)
    require(not torch.is_inference_mode_enabled(), 'Use ordinary no_grad or autograd, never inference_mode')
    require(all(not m._forward_pre_hooks and not m._forward_hooks for m in base.modules()) and
            'get_codebook_feat' not in base.quantize.__dict__, 'Clean hook/method namespace required')
    handles = []; observed = {}; used = {}; tokens = []; post = []
    original_lookup = base.quantize.get_codebook_feat
    def lookup(indices, shape):
        tokens.append(indices.detach().clone().reshape(1, 256))
        return original_lookup(indices, shape)
    def site_hook(site):
        def capture(module, args):
            require(len(args) == 3 and args[2] == 1. and site not in observed, 'One exact w1 SFT call per site required')
            feature, decoder, weight = args
            require(feature.shape == (1, CHANNELS[site], int(site), int(site)) and feature.dtype == torch.float32 and
                    feature.device.type == 'cuda' and not feature.requires_grad and torch.isfinite(feature).all().item(),
                    'Exact detached observed encoder-feature argument required')
            observed[site] = feature
            if getattr(net, '_spatial_continuation_candidate_v1', False):
                require(ledger is not None, 'Each full-site adapter call needs a durable reservation')
                token=ledger.reserve('adapter_full',site=site,**(context or {}))
                transformed=net.adapters[site](feature)
                ledger.returned(token)
            else: transformed=feature
            require(transformed.shape == feature.shape and transformed.dtype == feature.dtype and torch.isfinite(transformed).all().item(),
                    'Finite unchanged-geometry adapter output required')
            used[site] = transformed
            return (transformed, decoder, weight) if transformed is not feature else None
        return capture
    try:
        for site in SITES: handles.append(base.fuse_convs_dict[site].register_forward_pre_hook(site_hook(site)))
        handles.append(base.generator.blocks[0].register_forward_pre_hook(lambda module, args: post.append(args[0].detach().clone())))
        base.quantize.get_codebook_feat = lookup
        with torch.autocast(device_type='cuda', enabled=False):
            native, logits, lq = base(prepare_input(input256), w=1., detach_16=True, adain=True)
    finally:
        for handle in handles: handle.remove()
        if 'get_codebook_feat' in base.quantize.__dict__: del base.quantize.__dict__['get_codebook_feat']
    require(set(observed) == set(used) == set(SITES) and len(tokens) == len(post) == 1, 'Exact four sites and one actual code/postAdaIN capture required')
    require(logits.shape == (1, 256, 1024) and lq.shape == post[0].shape == (1, 256, 16, 16) and
            all(t.dtype == torch.float32 and not t.requires_grad and torch.isfinite(t).all().item() for t in (logits, lq, post[0])) and
            tokens[0].dtype == torch.int64 and tokens[0].min().item() >= 0 and tokens[0].max().item() < 1024,
            'Frozen finite upstream readouts differ')
    U = unclipped_rgb(native)
    with torch.no_grad():
        rgb = (native.detach().float() + 1) / 2
        reference = torch.nn.functional.interpolate(rgb, (256, 256), mode='bilinear', align_corners=False)
        parity = validate_readout_difference(float((U.detach().double() - reference.double()).abs().max().item()),
                                            float(rgb.abs().max().item()))
    return ForwardRecord(native, U, delivered_rgb(native), logits, lq, tokens[0], post[0], observed, used, parity)


def teacher_feature_cpu(record):
    _runtime(); require(isinstance(record, ForwardRecord), 'Actual forward record required')
    return {site: value.detach().cpu().clone() for site, value in record.observed_features.items()}


def feature_scales_from_clean(value):
    torch = _runtime(); values = value.observed_features if isinstance(value, ForwardRecord) else value
    require(set(values) == set(SITES), 'All four clean feature banks required')
    scales = {}
    for site, feature in values.items():
        require(isinstance(feature, torch.Tensor) and feature.shape == (1, CHANNELS[site], int(site), int(site)) and
                feature.dtype == torch.float32 and not feature.requires_grad and not torch.is_inference(feature) and
                torch.isfinite(feature).all().item(), 'Ordinary finite clean feature target required')
        scales[site] = max(1e-6, float(feature.detach().double().square().mean().sqrt().item()))
    return scales


def teacher_target_from_record(record, kind):
    torch = _runtime(); require(kind in ('original', 'r2') and isinstance(record, ForwardRecord), 'Explicit original/R2 teacher kind required')
    value = record.unclipped256.detach().clone()
    require(not value.requires_grad and not torch.is_inference(value), 'Ordinary frozen teacher tensor required')
    return TeacherTarget(value, kind)


def feature_terms(record, clean_features, scales):
    torch = _runtime(); require(isinstance(record, ForwardRecord) and set(clean_features) == set(scales) == set(SITES), 'Four fixed clean targets/scales required')
    site_terms = {}
    for site in SITES:
        target = clean_features[site]
        require(isinstance(target, torch.Tensor) and target.shape == record.used_features[site].shape and
                target.dtype == torch.float32 and not target.requires_grad and not torch.is_inference(target) and
                torch.isfinite(target).all().item() and isinstance(scales[site], (float, int)) and
                math.isfinite(scales[site]) and scales[site] >= 1e-6, 'Frozen finite clean target and prior fixed RMS scale required')
        target = target.to(device=record.used_features[site].device, dtype=torch.float32)
        site_terms[site] = (record.used_features[site] - target).abs().mean() / scales[site]
    return sum(site_terms.values()) / 4., site_terms


def readout_fingerprints(record):
    _runtime(); require(isinstance(record, ForwardRecord), 'Forward record required')
    return {name: tensor_state_digest({name: getattr(record, name)}) for name in ('logits', 'lq', 'tokens', 'post_adain')}


def validate_upstream_readouts(before, after):
    require(set(before) == set(after) == {'logits', 'lq', 'tokens', 'post_adain'} and before == after, 'Frozen upstream readouts changed')
    return {'pass': True, 'logits_LQ_actual_codes_postAdaIN_exact': True, 'identity_confidence_claim': False}


def validate_initial_parity(candidate, teacher, *, np=None):
    torch = _runtime(); require(isinstance(candidate, ForwardRecord) and isinstance(teacher, ForwardRecord), 'Actual paired records required')
    fields = ('native', 'unclipped256', 'delivered256', 'logits', 'lq', 'tokens', 'post_adain')
    require(all(torch.equal(getattr(candidate, key).detach(), getattr(teacher, key).detach()) for key in fields), 'Exact same-runtime initial R2 parity failed')
    require(torch.equal(png_tensor(candidate.delivered256), png_tensor(teacher.delivered256)) and
            all(torch.equal(candidate.used_features[s].detach(), teacher.observed_features[s].detach()) for s in SITES), 'Initial branch/PNG parity failed')
    return {'pass': True, 'fields': list(fields), 'used_features_exact': True, 'PNG_exact': True, 'tolerance': 0.}


def clipping_statistics(record):
    _runtime(); rgb = (record.native.detach() + 1) / 2; U = record.unclipped256.detach()
    return {'native512_below0_fraction': float((rgb < 0).double().mean().item()),
            'native512_above1_fraction': float((rgb > 1).double().mean().item()),
            'U256_below0_fraction': float((U < 0).double().mean().item()),
            'U256_above1_fraction': float((U > 1).double().mean().item()),
            'native_RGB_min': float(rgb.min().item()), 'native_RGB_max': float(rgb.max().item()),
            'U_RGB_min': float(U.min().item()), 'U_RGB_max': float(U.max().item())}


def summarize_gradients(names, gradients, *, np):
    torch = _runtime(); shapes = expected_adapter_shapes()
    require(len(names) == len(gradients) == 24 and set(names) == set(shapes), 'Query exactly24 adapter tensors')
    rows = {}; sites = {site: {'tensors': 0, 'none': 0, 'nonzero': 0, 'squared_l2': 0., 'up_squared_l2': 0.} for site in SITES}
    for name, gradient in zip(names, gradients):
        site = name.split('.')[1]; stats = sites[site]; stats['tensors'] += 1
        if gradient is None:
            stats['none'] += 1; rows[name] = {'is_none': True, 'l2': 0., 'maximum_absolute': 0., 'sha256': None}; continue
        require(gradient.dtype == torch.float32 and tuple(gradient.shape) == shapes[name], 'Exact finite adapter gradient schema required')
        values = np.ascontiguousarray(gradient.detach().cpu().numpy()); require(np.isfinite(values).all(), 'Nonfinite adapter gradient')
        norm = float(np.square(values.astype(np.float64)).sum()); maximum = float(np.abs(values).max())
        stats['squared_l2'] += norm; stats['nonzero'] += int(maximum > 0)
        if '.up.' in name: stats['up_squared_l2'] += norm
        rows[name] = {'is_none': False, 'l2': math.sqrt(norm), 'maximum_absolute': maximum,
                      'sha256': hashlib.sha256(values.tobytes()).hexdigest()}
    for stats in sites.values():
        stats['l2'] = math.sqrt(stats.pop('squared_l2')); stats['up_l2'] = math.sqrt(stats.pop('up_squared_l2'))
    return {'parameters': rows, 'sites': sites, 'queried_tensors': 24,
            'all_new_tensor_nonzero_required': False, 'zero_last_projection_upstream_zero_initially_expected': True,
            'source_bound_gradient_readout_not_independent_replay': True}


def validate_gradient_summary(summary, *, allow_all_zero=False):
    require(type(allow_all_zero) is bool and summary['queried_tensors'] == 24 and set(summary['sites']) == set(SITES), 'Exact24/four-site gradient query required')
    for site, row in summary['sites'].items():
        require(row['tensors'] == 6 and row['none'] == 0 and math.isfinite(row['l2']) and math.isfinite(row['up_l2']),
                'Disconnected/nonfinite adapter graph: ' + site)
        require(allow_all_zero or row['up_l2'] > 0, 'Each zero-final projection needs nonzero improving gradient: ' + site)
    return {'pass': True, 'all4sites_connected': True, 'up_projection_nonzero_required': not allow_all_zero,
            'earlier_layer_zero_initial_gradient_is_valid': True}


def prerequisite_context(net, teachers):
    """Capture actual witness state/RNG/callable context; creates no optimizer."""
    torch = _runtime(); parameter_census(net)
    require(_ARCHITECTURE is not None and all(p.grad is None for p in net.parameters()), 'Fresh no-gradient witness context required')
    for teacher in teachers: _frozen_teacher(teacher)
    return {'candidate_state': state_digest(net), 'frozen_state': frozen_state_digest(net),
        'teacher_states': [state_digest(teacher) for teacher in teachers],
        'module_modes': tuple(m.training for m in net.modules()),
        'CPU_RNG': torch.get_rng_state().clone(), 'CUDA_RNG': torch.cuda.get_rng_state(0).clone(),
        'style_callable': _ARCHITECTURE.adaptive_instance_normalization}


def validate_prerequisite_restoration(before, net, teachers):
    torch = _runtime(); parameter_census(net)
    require(all(p.grad is None for p in net.parameters()), 'Clear witness gradient fields to None')
    for teacher in teachers: _frozen_teacher(teacher)
    require(before['candidate_state'] == state_digest(net) and before['frozen_state'] == frozen_state_digest(net) and
            before['teacher_states'] == [state_digest(teacher) for teacher in teachers] and
            before['module_modes'] == tuple(m.training for m in net.modules()) and
            torch.equal(before['CPU_RNG'], torch.get_rng_state()) and torch.equal(before['CUDA_RNG'], torch.cuda.get_rng_state(0)) and
            before['style_callable'] is _ARCHITECTURE.adaptive_instance_normalization and
            all(not module._forward_pre_hooks and not module._forward_hooks for module in net.base.modules()) and
            'get_codebook_feat' not in net.base.quantize.__dict__, 'Witness altered state, RNG, modes, style or hooks')
    return {'pass': True, 'all_states_modes_RNG_style_callable_exact': True,
            'candidate_teacher_grad_fields_None': True, 'temporary_hooks_and_lookup_restored': True,
            'optimizer_constructed_by_model_helper': False}


def reset_adapters(net, initial):
    """Restore one candidate, never construct a second model or reuse Adam."""
    torch=_runtime();parameter_census(net);shapes=expected_adapter_shapes()
    require(set(initial)==set(shapes) and tensor_state_digest(initial)==INITIAL_ADAPTER_DIGEST,
            'Exact identity24 state required before each arm')
    with torch.no_grad():
        named=dict(net.named_parameters())
        for name,value in initial.items():
            require(value.dtype==torch.float32 and tuple(value.shape)==shapes[name],'Initial adapter schema')
            named[name].copy_(value.to(device='cuda',dtype=torch.float32));named[name].grad=None
    require(tensor_state_digest(adapter_state(net))==INITIAL_ADAPTER_DIGEST and frozen_state_digest(net)==R2_DIGEST,
            'Arm reset changed frozen base or identity24')


def prefix_features(net, input256):
    """One frozen encoder prefix, stopping after official block14; no decoder."""
    torch=_runtime();parameter_census(net)
    require(not torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),'Ordinary no_grad prefix required')
    require(net.base.fuse_encoder_block=={'512':2,'256':5,'128':8,'64':11,'32':14,'16':18},'Official tap map changed')
    x=prepare_input(input256);features={};taps={5:'256',8:'128',11:'64',14:'32'}
    for index,block in enumerate(net.base.encoder.blocks):
        require(index<=14,'Never execute later encoder, predictor or decoder in feature cache')
        x=block(x)
        if index in taps:
            site=taps[index];require(x.shape==(1,CHANNELS[site],int(site),int(site)) and
                x.dtype==torch.float32 and not x.requires_grad and torch.isfinite(x).all().item(),'Exact frozen prefix feature')
            features[site]=x.detach().clone()
        if index==14:break
    require(set(features)==set(SITES),'Four exact prefix taps required')
    return features


def adapter_features(net, observed, ledger, kind, context):
    torch=_runtime();parameter_census(net)
    require(kind in ('adapter_train','adapter_stats') and set(observed)==set(SITES),'Explicit feature-only call kind')
    result={}
    for site in SITES:
        value=observed[site]
        require(value.shape==(1,CHANNELS[site],int(site),int(site)) and value.dtype==torch.float32 and
                value.device.type=='cuda' and not value.requires_grad and torch.isfinite(value).all().item(),
                'Frozen observed CUDA FP32 feature required')
        token=ledger.reserve(kind,site=site,**context)
        result[site]=net.adapters[site](value)
        ledger.returned(token)
        require(result[site].shape==value.shape and torch.isfinite(result[site]).all().item(),'Finite adapter result')
    return result


def feature_objective(arm, used, target, scales):
    """Only U/S feature objectives; same fixed total-clean RMS and four sites."""
    torch=_runtime();require(arm in ('U','S') and torch.is_grad_enabled(),'Explicit optimizing U/S context required')
    require(set(used)==set(target)==set(scales)==set(SITES),'Four used/clean/RMS joins required')
    terms={}
    for site in SITES:
        z,t=used[site],target[site]
        require(z.shape==t.shape==(1,CHANNELS[site],int(site),int(site)) and z.dtype==t.dtype==torch.float32 and
                z.device.type==t.device.type=='cuda' and z.requires_grad and not t.requires_grad and
                math.isfinite(scales[site]) and scales[site]>=1e-6,'Connected adapter/frozen target/scale required')
        if arm=='S':
            z=z-z.mean(dim=(2,3),keepdim=True);t=t-t.mean(dim=(2,3),keepdim=True)
        terms[site]=(z-t).abs().mean()/scales[site]
    scalar=sum(terms.values())/4.
    require(torch.isfinite(scalar).item(),'Nonfinite feature objective')
    return scalar/SLOT_DIVISOR,terms,scalar


def value_digest(value):
    """Canonical exact checkpoint tree digest; device-independent, no forward."""
    torch=_runtime();h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            a=x.detach().cpu().contiguous();header=['tensor',str(a.dtype),list(a.shape)]
            b=json.dumps(header,separators=(',',':')).encode();h.update(len(b).to_bytes(8,'big'));h.update(b)
            h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes());return
        if isinstance(x,dict):
            h.update(b'dict[')
            for key in sorted(x,key=lambda k:(type(k).__name__,str(k))):visit(key);visit(x[key])
            h.update(b']');return
        if isinstance(x,(list,tuple)):
            h.update(('tuple[' if isinstance(x,tuple) else 'list[').encode())
            for item in x:visit(item)
            h.update(b']');return
        require(x is None or type(x) in (str,int,float,bool),'Unsupported resume checkpoint tree')
        if isinstance(x,float):require(math.isfinite(x),'Nonfinite resume scalar')
        b=json.dumps([type(x).__name__,x],separators=(',',':'),allow_nan=False).encode()
        h.update(len(b).to_bytes(8,'big'));h.update(b)
    visit(value);return h.hexdigest()


def load_resume_state(root, protocol, arm):
    """Content-bound weights_only CPU read, allowed only in guarded runtime."""
    torch=_runtime();require(arm in ('U','S'),'Own U/S resume arm required')
    binding=protocol['resume'][arm];item=binding['checkpoint'];path=regular(Path(root)/item['path'])
    require(path.stat().st_size==item['bytes'] and sha256(path)==item['sha256'],'Own24 checkpoint bytes changed')
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    require(saved['format']=='codeformer-spatial-objective-adapter-state-v1' and saved['arm']==arm and
        saved['protocol_sha256']==protocol['parent_protocol_sha256'] and saved['completed_updates']==24 and
        saved['base515_sha256']==R2_DIGEST and saved['sampler_committed_rows']==72 and not saved['pending'],
        'Original own24 complete state/protocol/commit binding differs')
    shapes=expected_adapter_shapes();state=saved['adapter_state']
    require(set(state)==set(shapes) and all(isinstance(v,torch.Tensor) and v.device.type=='cpu' and
        v.dtype==torch.float32 and tuple(v.shape)==shapes[k] and torch.isfinite(v).all().item()
        for k,v in state.items()),'Exact finite own24 adapter state required')
    require(tensor_state_digest(state)==saved['adapter_sha256']==binding['adapter_sha256'],
        'Own24 canonical adapter state differs')
    adam=saved['optimizer'];require(len(adam['param_groups'])==1 and len(adam['state'])==24,'Own24 Adam state census')
    group=adam['param_groups'][0]
    require(group['params']==list(range(24)) and group['lr']==BASE_LR and group['base_lr']==BASE_LR and
        group['betas']==(.9,.99) and group['eps']==1e-8 and group['weight_decay']==0. and
        group.get('amsgrad') is False,'Exact original Adam schedule/hyperparameters required')
    names=sorted(shapes)
    for index,name in enumerate(names):
        a=adam['state'][index]
        require(set(a)=={'step','exp_avg','exp_avg_sq'} and float(a['step'].item())==24 and
            all(a[k].shape==state[name].shape and a[k].dtype==torch.float32 and torch.isfinite(a[k]).all().item()
                for k in ('exp_avg','exp_avg_sq')),'Exact own24 finite Adam moments/step required')
    require(saved['CPU_RNG'].dtype==saved['CUDA_RNG'].dtype==torch.uint8 and
        saved['CPU_RNG'].ndim==saved['CUDA_RNG'].ndim==1 and
        set(saved['NumPy_RNG'])=={'algorithm','state','position','has_gauss','cached_gaussian'},'Saved RNG schema')
    return saved,{'source_checkpoint':dict(item),'arm':arm,'completed_updates':24,
        'adapter_sha256':binding['adapter_sha256'],'optimizer_tree_sha256':value_digest(adam),
        'RNG_tree_sha256':value_digest({k:saved[k] for k in ('CPU_RNG','CUDA_RNG','NumPy_RNG','python_RNG')}),
        'optimizer_steps':{str(k):float(v['step'].item()) for k,v in adam['state'].items()},
        'base515_sha256':R2_DIGEST,'checkpoint_pickle_independent_review_still_separate':True}


def install_resume(net, saved, binding):
    """Restore one own24 adapter and Adam. No forward/backward/step performed."""
    torch=_runtime();parameter_census(net);state=saved['adapter_state'];shapes=expected_adapter_shapes()
    require(tensor_state_digest(state)==binding['adapter_sha256'] and set(state)==set(shapes),'Resume adapter binding')
    with torch.no_grad():
        named=dict(net.named_parameters())
        for name,value in state.items():named[name].copy_(value.to('cuda'));named[name].grad=None
    optimizer=make_optimizer(net);optimizer.load_state_dict(deepcopy(saved['optimizer']))
    require(value_digest(optimizer.state_dict())==value_digest(saved['optimizer']),
        'Installed Adam parameters/moments/counters differ from own24')
    require(tensor_state_digest(adapter_state(net))==binding['adapter_sha256'] and frozen_state_digest(net)==R2_DIGEST,
        'Installed adapter/base515 differs')
    require(all(v['exp_avg'].device.type=='cuda' and v['exp_avg_sq'].device.type=='cuda'
        for v in optimizer.state.values()),'Adam moments must follow CUDA parameters')
    return optimizer
