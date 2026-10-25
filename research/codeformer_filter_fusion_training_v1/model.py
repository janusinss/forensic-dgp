"""Fusion-only CodeFormer helpers for the finite manual L4 filter pilot.

Import is standard-library only. No optimizer, backward, gradient query, neural
call or architecture import occurs at import time. The worker owns all finite
operation reservations. The candidate's frozen decoder remains under ordinary
autograd: frozen weights are not permission to detach decoder activations.
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


VM_NAME = 'cctv_codeformer_filter_fusion_training_v1_vm'
SITES = ('32', '64', '128', '256')
FUSION_PREFIX = 'fuse_convs_dict.'
ORIGINAL_DIGEST = '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd'
PARENT_DIGEST = '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b'
R2_DIGEST = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
ASSETS = {
    'original': {'path': 'weights/codeformer_original.pth', 'bytes': 376637898,
                 'sha256': '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'},
    'r2': {'path': 'weights/codeformer_r2.pth', 'bytes': 376609465,
           'sha256': 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'},
    'parent': {'path': 'weights/codeformer_parent.pth', 'bytes': 376643953,
               'sha256': '53778f54589697a14b067033eff8cffc2985985e8a33021092de3c14fbf443bc'}}
MODEL_DIGESTS = {'original': ORIGINAL_DIGEST, 'parent': PARENT_DIGEST, 'r2': R2_DIGEST}
ARCHITECTURE_SOURCES = {
    'third_party.codeformer.codeformer_arch': ('third_party/codeformer/codeformer_arch.py',
        'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb'),
    'third_party.codeformer.vqgan_arch': ('third_party/codeformer/vqgan_arch.py',
        'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf'),
    'third_party.codeformer.registry': ('third_party/codeformer/registry.py',
        'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e')}
FILTER_TERMS = ('spatial_rgb_l1', 'adjacent_residual_l1')
ROLE_TERMS = {'filter': FILTER_TERMS, 'clear': ('teacher_rgb_l1',),
              'dark': ('teacher_rgb_l1',)}
SLOT_DIVISOR = 4
U_READOUT_POLICY = 'unclipped512-to256-separable-disjoint-half-pairs-v1'
U_REFERENCE_EPS_MULTIPLIER = 8
_TORCH = _ROOT = _PROTOCOL_SHA = _ARCHITECTURE = None
_ORIGINAL = _R2 = None
_FACTORY_DONE = False


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, 'Duplicate metadata key')
            result[name] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite metadata')))


def regular(path):
    path = Path(path).absolute()
    for current in (path, *path.parents):
        if current.exists() or current.is_symlink():
            state = current.lstat()
            require(not stat.S_ISLNK(state.st_mode) and
                    not getattr(state, 'st_file_attributes', 0) & 1024,
                    'Linked/reparse model, source or root forbidden')
    state = path.lstat()
    require(stat.S_ISREG(state.st_mode) and state.st_nlink == 1,
            'Single-link regular packet source/asset required')
    return path.resolve(strict=True)


def expected_fusion_shapes():
    """Exact72 named tensors, derived from the unchanged four-block schema."""
    result = {}
    for site, channels in zip(SITES, (256, 256, 128, 128)):
        shapes = {
            'encode_enc.norm1.weight': (2 * channels,),
            'encode_enc.norm1.bias': (2 * channels,),
            'encode_enc.conv1.weight': (channels, 2 * channels, 3, 3),
            'encode_enc.conv1.bias': (channels,),
            'encode_enc.norm2.weight': (channels,),
            'encode_enc.norm2.bias': (channels,),
            'encode_enc.conv2.weight': (channels, channels, 3, 3),
            'encode_enc.conv2.bias': (channels,),
            'encode_enc.conv_out.weight': (channels, 2 * channels, 1, 1),
            'encode_enc.conv_out.bias': (channels,)}
        for branch in ('scale', 'shift'):
            for layer in (0, 2):
                shapes[f'{branch}.{layer}.weight'] = (channels, channels, 3, 3)
                shapes[f'{branch}.{layer}.bias'] = (channels,)
        for name, shape in shapes.items():
            result[f'{FUSION_PREFIX}{site}.{name}'] = shape
    require(len(result) == 72 and sum(math.prod(v) for v in result.values()) == 10659584,
            'Exact fusion tensor/parameter schema differs')
    return result


def fusion_site(name):
    require(isinstance(name, str) and name in expected_fusion_shapes(),
            'Name is not an exact declared fusion tensor: ' + str(name))
    return name.split('.')[1]


def role_term_names(role):
    require(role in ROLE_TERMS, 'Role must be filter, clear or dark')
    return ROLE_TERMS[role]


def term_weights(role):
    role_term_names(role)
    return {'spatial_rgb_l1': 1., 'adjacent_residual_l1': .25} if role == 'filter' else {'teacher_rgb_l1': 2.}


def learning_rate(update):
    require(type(update) is int and 1 <= update <= 151, 'Sealed finite update1..151 required')
    return 1e-5 * min(update, 10) / 10


def validate_footprint(footprint, width=256, height=256):
    """Half-open synthetic corruption support, never an inference/removal mask."""
    require(type(width) is int and type(height) is int and width >= 2 and height >= 2,
            'Valid image dimensions required')
    require(type(footprint) in (tuple, list) and len(footprint) == 4 and
            all(type(v) is int for v in footprint), 'Exact integer[x0,y0,x1,y1] footprint required')
    x0, y0, x1, y1 = footprint
    require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height,
            'Footprint outside image or empty')
    inside = (x1 - x0) * (y1 - y0)
    require(0 < inside < width * height, 'Inside and outside must both be nonempty')
    return tuple(footprint)


def numpy_loss_terms(prediction, target, *, role, footprint=None, np):
    """Independent exported-array readout; float64 NumPy, no neural framework.

    This does not establish gradient connectivity or cross-runtime FP32 parity.
    It permits CPU fixtures/audits of coefficients, region counts and boundaries.
    Arrays are CHW/NCHW RGB with arbitrary nondegenerate spatial sizes for tests.
    """
    role_term_names(role)
    p, q = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    require(p.shape == q.shape and p.ndim in (3, 4) and p.shape[-3] == 3 and
            min(p.shape[-2:]) >= 2 and np.isfinite(p).all() and np.isfinite(q).all(),
            'Matching finite CHW/NCHW RGB arrays required')
    residual = p - q
    if role == 'filter':
        if footprint is None:
            spatial = float(np.abs(residual).mean(dtype=np.float64))
        else:
            x0, y0, x1, y1 = validate_footprint(footprint, p.shape[-1], p.shape[-2])
            inside = np.zeros(p.shape[-2:], dtype=bool); inside[y0:y1, x0:x1] = True
            spatial = .5 * float(np.abs(residual[..., inside]).mean(dtype=np.float64)) + \
                      .5 * float(np.abs(residual[..., ~inside]).mean(dtype=np.float64))
        adjacent = .5 * float(np.abs(residual[..., :, 1:] - residual[..., :, :-1]).mean(dtype=np.float64)) + \
                   .5 * float(np.abs(residual[..., 1:, :] - residual[..., :-1, :]).mean(dtype=np.float64))
        terms = {'spatial_rgb_l1': spatial, 'adjacent_residual_l1': adjacent}
    else:
        require(footprint is None, 'Teacher roles cannot receive corruption support')
        terms = {'teacher_rgb_l1': float(np.abs(residual).mean(dtype=np.float64))}
    weighted = sum(term_weights(role)[name] * value for name, value in terms.items())
    return {'unweighted_terms': terms, 'slot_scalar': weighted,
            'divided_slot_scalar': weighted / SLOT_DIVISOR,
            'domain': 'unclipped256_U', 'slot_divisor': SLOT_DIVISOR}


def numpy_unclipped_readout(native, *, np):
    """FP32 pure-array oracle of the declared separable optimizing readout.

    Even CHW/NCHW sizes are allowed for small independent math fixtures. The
    manual Torch helper separately requires exact native512 geometry. This
    never imports Torch and makes no autograd/gradient-execution claim.
    """
    native = np.asarray(native)
    require(native.dtype == np.float32 and native.ndim in (3, 4) and native.shape[-3] == 3 and
            all(size >= 2 and size % 2 == 0 for size in native.shape[-2:]) and np.isfinite(native).all(),
            'Finite even-size native FP32 RGB array required')
    rgb = (native + np.float32(1)) / np.float32(2)
    horizontal = rgb[..., 0::2] * np.float32(.5) + rgb[..., 1::2] * np.float32(.5)
    return np.ascontiguousarray(horizontal[..., 0::2, :] * np.float32(.5) +
                                horizontal[..., 1::2, :] * np.float32(.5))


def readout_parity_bound(maximum_absolute_native_rgb):
    """Prospective readout roundoff bound, never a delivered quality epsilon."""
    value = float(maximum_absolute_native_rgb)
    require(math.isfinite(value) and value >= 0., 'Finite nonnegative RGB magnitude required')
    return U_REFERENCE_EPS_MULTIPLIER * (2. ** -23) * max(1., value)


def validate_readout_difference(maximum_error, maximum_absolute_native_rgb):
    error = float(maximum_error); limit = readout_parity_bound(maximum_absolute_native_rgb)
    require(math.isfinite(error) and error >= 0. and error <= limit,
            'Deterministic U exceeds prospectively fixed same-native stock readout roundoff bound')
    return {'pass': True, 'policy': U_READOUT_POLICY, 'maximum_absolute_difference': error,
            'maximum_absolute_native_RGB': float(maximum_absolute_native_rgb), 'absolute_bound': limit,
            'bound_formula': '8*float32_eps*max(1,maxabs(native_RGB))',
            'same_native_stock_bilinear_reference_no_grad': True,
            'new_neural_or_gradient_queries': 0, 'delivered_quality_gate_tolerance_changed': False}


def activate_runtime(root, protocol_sha):
    """The sole Torch import follows exact manual-root/protocol/source guards."""
    global _TORCH, _ROOT, _PROTOCOL_SHA
    require(platform.system() == 'Linux', 'Neural/gradient work is manual Linux L4 only')
    root = Path(root).absolute()
    expected = Path.home() / 'forensic-dgp' / VM_NAME
    for item in (root, *root.parents):
        state = item.lstat()
        require(not stat.S_ISLNK(state.st_mode) and
                not getattr(state, 'st_file_attributes', 0) & 1024,
                'Linked/reparse runtime directory forbidden')
    require(root.resolve(strict=True) == expected.resolve(strict=True), 'Exact fresh manual VM root required')
    require(bool(os.environ.get('TMUX')), 'Manual tmux required before Torch import')
    require(isinstance(protocol_sha, str) and re.fullmatch('[0-9a-f]{64}', protocol_sha), 'Protocol SHA required')
    protocol_path = regular(root / 'protocol.json')
    require(protocol_path.stat().st_size <= 8 * 1024**2 and sha256(protocol_path) == protocol_sha,
            'Frozen protocol differs')
    protocol = read_json(protocol_path)
    pins = protocol['files_sha256']
    for module_name, (relative, known_sha) in ARCHITECTURE_SOURCES.items():
        require(pins.get(relative) == known_sha and sha256(regular(root / relative)) == known_sha,
                'Exact unchanged architecture bytes required: ' + module_name)
    for relative in ('third_party/__init__.py', 'third_party/codeformer/__init__.py'):
        require(pins.get(relative) == sha256(regular(root / relative)), 'Package initializer differs')
    if _TORCH is not None:
        require(_ROOT == root.resolve(strict=True) and _PROTOCOL_SHA == protocol_sha,
                'No model runtime rebinding')
        return _TORCH
    require(not any(name in sys.modules for name in ('third_party', 'third_party.codeformer', *ARCHITECTURE_SOURCES)),
            'Architecture module cache must be fresh after verification')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == 'NVIDIA L4', 'Exactly one NVIDIA L4 required')
    require(not torch.is_inference_mode_enabled(), 'No inference_mode activation')
    _TORCH, _ROOT, _PROTOCOL_SHA = torch, root.resolve(strict=True), protocol_sha
    return torch


def _runtime():
    require(_TORCH is not None and _ROOT is not None, 'Activate manual guarded VM runtime first')
    return _TORCH


def tensor_state_digest(state):
    torch = _runtime(); digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor), 'Tensor state required')
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_digest(net):
    return tensor_state_digest(net.state_dict())


def frozen_state_digest(net):
    return tensor_state_digest({name: value for name, value in net.state_dict().items()
                                if not name.startswith(FUSION_PREFIX)})


def cpu_state(net):
    _runtime()
    return {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}


def parameter_census(net):
    torch = _runtime(); named = dict(net.named_parameters()); state = net.state_dict()
    shapes = expected_fusion_shapes()
    require(len(named) == len(state) == 515 and set(named) == set(state), 'Exact515 parameter-only state required')
    trainable, frozen = {}, {}
    for name, parameter in named.items():
        is_fusion = name in shapes
        require(not name.startswith(FUSION_PREFIX) or is_fusion, 'Unexpected fusion tensor')
        require(parameter.dtype == torch.float32 and parameter.device.type == 'cuda' and
                parameter.requires_grad == is_fusion, 'Exact fusion-only FP32 CUDA trainability differs: ' + name)
        if is_fusion:
            require(tuple(parameter.shape) == shapes[name], 'Exact fusion tensor shape differs: ' + name)
        else:
            require(parameter.grad is None, 'Frozen candidate gradient must remain None: ' + name)
        (trainable if is_fusion else frozen)[name] = parameter.numel()
    require(len(trainable) == 72 and len(frozen) == 443 and sum(trainable.values()) == 10659584,
            'Exact72/443/10659584 census differs')
    require(all(not child.training for child in net.modules()), 'Every candidate module remains eval')
    return {'trainable_names': sorted(trainable), 'frozen_names': sorted(frozen),
            'trainable_tensors': len(trainable), 'frozen_tensors': len(frozen),
            'trainable_scalar_parameters': sum(trainable.values()),
            'frozen_scalar_parameters': sum(frozen.values()),
            'fusion_sites': {site: sum(fusion_site(name) == site for name in trainable) for site in SITES}}


def _frozen_teacher(net):
    torch = _runtime()
    require(len(tuple(net.parameters())) == len(net.state_dict()) == 515 and
            all(not child.training for child in net.modules()) and
            all(not value.requires_grad and value.grad is None and value.dtype == torch.float32 and
                value.device.type == 'cuda' for value in net.parameters()), 'Exact frozen eval FP32 CUDA teacher required')


def _verified_architecture():
    global _ARCHITECTURE
    _runtime()
    if _ARCHITECTURE is None:
        _ARCHITECTURE = importlib.import_module('third_party.codeformer.codeformer_arch')
    for name, (relative, known_sha) in ARCHITECTURE_SOURCES.items():
        require(Path(sys.modules[name].__file__).resolve(strict=True) == (_ROOT / relative).resolve(strict=True) and
                sha256(regular(_ROOT / relative)) == known_sha,
                'Architecture source/origin changed after verification')
    return _ARCHITECTURE


def _pinned_asset(name, supplied=None):
    _runtime(); require(name in ASSETS, 'Undeclared packet model asset')
    pin = ASSETS[name]; path = regular(_ROOT / pin['path'])
    require(supplied is None or Path(supplied).resolve(strict=True) == path,
            'Model outside exact packet weight path')
    before = path.stat()
    require(before.st_size == pin['bytes'] and sha256(path) == pin['sha256'], 'Model file hash/size differs')
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Model changed during hash check')
    return path


def _fresh_frozen_model(path, *, key):
    torch = _runtime(); architecture = _verified_architecture()
    net = architecture.CodeFormer(dim_embd=512, codebook_size=1024, n_head=8,
        n_layers=9, connect_list=list(SITES)).float().eval()
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    require(type(saved) is dict and key in saved, 'Exact model state export key required')
    values = saved[key]
    require(len(values) == 515 and set(values) == set(net.state_dict()) and all(
        isinstance(v, torch.Tensor) and v.dtype == torch.float32 and torch.isfinite(v).all().item()
        for v in values.values()), 'Finite FP32 exact515 model schema required')
    net.load_state_dict(values, strict=True)
    del values, saved
    return net.to('cuda').eval().requires_grad_(False)


def load_frozen_baseline(name):
    """Fresh current-CUDA original/parent/R2, no forward; worker frees each.

    Parent bytes are the retained R2 snapshot0000 model state, never its older
    optimizer-bearing historical checkpoint. This cannot install optimizer/RNG.
    """
    torch = _runtime(); require(name in ASSETS, 'Unknown frozen baseline arm')
    path = _pinned_asset(name)
    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    with torch.random.fork_rng(devices=[0]):
        net = _fresh_frozen_model(path, key='model' if name == 'parent' else 'params_ema')
    _frozen_teacher(net)
    require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state(0)),
            'Baseline construction changed caller RNG')
    digest = state_digest(net)
    require(digest == MODEL_DIGESTS[name], 'Frozen baseline canonical515 differs')
    return net, {'arm': name, 'canonical515_sha256': digest, 'factory_RNG_preserved': True,
                 'neural_calls': 0, 'optimizer_state_installed': False}


def load_models(original_path=None, r2_path=None):
    """One factory, model values only: candidate, original, R2, census, receipt.

    No historical optimizer, RNG, sampler or update state is loaded/installed.
    All three full-model canonical digests are checked after device construction.
    """
    global _ORIGINAL, _R2, _FACTORY_DONE
    torch = _runtime()
    require(not _FACTORY_DONE and _ORIGINAL is None and _R2 is None, 'Construct finite training factory once only')
    paths = {'original': _pinned_asset('original', original_path), 'r2': _pinned_asset('r2', r2_path)}
    cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    with torch.random.fork_rng(devices=[0]):
        original, r2 = (_fresh_frozen_model(paths[name], key='params_ema') for name in ('original', 'r2'))
        candidate = deepcopy(r2).eval()
        for name, parameter in candidate.named_parameters():
            parameter.requires_grad_(name.startswith(FUSION_PREFIX))
    require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state(0)),
            'Factory changed caller CPU/CUDA RNG')
    census = parameter_census(candidate); _frozen_teacher(original); _frozen_teacher(r2)
    require(all(p.grad is None for p in candidate.parameters()), 'Fresh candidate gradients must be None')
    digests = {'original': state_digest(original), 'r2': state_digest(r2), 'candidate': state_digest(candidate)}
    require(digests == {'original': ORIGINAL_DIGEST, 'r2': R2_DIGEST, 'candidate': R2_DIGEST},
            'Exact original/R2/candidate canonical states differ')
    _ORIGINAL, _R2, _FACTORY_DONE = original, r2, True
    receipt = {'complete': True, 'canonical515_sha256': digests, 'parameter_census': census,
        'candidate_decoder_ordinary_autograd_required': True, 'candidate_all_eval': True,
        'teacher_all_eval_frozen': True, 'style_callable_monkeypatch': False,
        'historical_optimizer_RNG_sampler_installed': False, 'factory_RNG_preserved': True,
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_constructions': 0, 'optimizer_steps': 0}
    return candidate, original, r2, census, receipt


def optimizer_groups(net):
    """One exact72-tensor group; creates no optimizer and installs no state."""
    _runtime(); census = parameter_census(net)
    selected = [value for name, value in net.named_parameters() if name in census['trainable_names']]
    require(len(selected) == 72 and sum(v.numel() for v in selected) == 10659584,
            'Exact optimizer membership differs')
    return [{'name': 'fusion', 'params': selected, 'lr': learning_rate(1), 'base_lr': 1e-5}]


def _rgb(value, size, *, bounded):
    torch = _runtime()
    require(isinstance(value, torch.Tensor) and value.shape == (1, 3, size, size) and
            value.dtype == torch.float32 and value.device.type == 'cuda' and torch.isfinite(value).all().item(),
            'Finite FP32 CUDA singleton RGB geometry differs')
    if bounded:
        require(value.min().item() >= 0 and value.max().item() <= 1, 'Input/clean target RGB must be[0,1]')


def prepare_input(input256):
    torch = _runtime(); _rgb(input256, 256, bounded=True)
    require(not input256.requires_grad and not torch.is_inference(input256), 'Ordinary non-gradient image input required')
    return torch.nn.functional.interpolate(input256, size=(512, 512), mode='bilinear', align_corners=False) * 2 - 1


def unclipped_rgb(output512):
    torch = _runtime(); _rgb(output512, 512, bounded=False)
    # Exact2:1 align_corners=False samples(2*y+.5,2*x+.5). Four disjoint
    # contributors have RGB weights1/4 (native signed-output weights1/8).
    # Basic slicing/add/multiply preserves ordinary candidate decoder autograd
    # without invoking nondeterministic CUDA bilinear interpolation backward.
    rgb = (output512.float() + 1) / 2
    horizontal = rgb[..., 0::2] * .5 + rgb[..., 1::2] * .5
    return horizontal[..., 0::2, :] * .5 + horizontal[..., 1::2, :] * .5


def unclipped_reference_parity(output512, deterministic_U):
    """Same-native CUDA forward arithmetic witness; no second network call."""
    torch = _runtime(); _rgb(output512, 512, bounded=False)
    with torch.no_grad():
        rgb = (output512.detach().float() + 1) / 2
        reference = torch.nn.functional.interpolate(rgb, size=(256, 256), mode='bilinear', align_corners=False)
        error = float((deterministic_U.detach().double() - reference.double()).abs().max().item())
        magnitude = float(rgb.abs().max().item())
    return validate_readout_difference(error, magnitude)


def delivered_rgb(output512):
    torch = _runtime(); _rgb(output512, 512, bounded=False)
    # Native clamp occurs BEFORE resizing, exactly as the retained RAW policy.
    # Delivery is numerical evidence, not an optimizing graph. Stock native
    # clamp-before-bilinear RAW/PNG arithmetic is unchanged.
    with torch.no_grad():
        rgb = ((output512.detach().float() + 1) / 2).clamp(0, 1)
        return torch.nn.functional.interpolate(rgb, size=(256, 256), mode='bilinear', align_corners=False)


def png_tensor(delivered256):
    torch = _runtime(); _rgb(delivered256, 256, bounded=True)
    return (delivered256.detach() * 255).floor().to(torch.uint8)


@dataclass(frozen=True)
class ForwardRecord:
    native: object
    unclipped256: object
    delivered256: object
    logits: object
    lq: object
    tokens: object
    U_readout_parity: object


@dataclass(frozen=True)
class TeacherTarget:
    unclipped256: object
    kind: str


def forward_record(net, input256):
    """Exactly one ordinary unchanged forward; caller chooses grad context.

    Snapshot use may surround this in no_grad. Training must call the stricter
    forward_candidate wrapper. No decoder callable, activation or style rewrite.
    """
    torch = _runtime()
    require(not torch.is_inference_mode_enabled() and all(not m.training for m in net.modules()),
            'Ordinary eval forward outside inference_mode required')
    with torch.autocast(device_type='cuda', enabled=False):
        native, logits, lq = net(prepare_input(input256), w=1.0, detach_16=True, adain=True)
        require(logits.shape == (1, 256, 1024) and lq.shape == (1, 256, 16, 16) and
                logits.dtype == lq.dtype == torch.float32 and torch.isfinite(logits).all().item() and
                torch.isfinite(lq).all().item(), 'Finite unchanged CodeFormer readouts required')
        tokens = torch.topk(torch.softmax(logits.detach(), dim=2), 1, dim=2).indices.squeeze(2)
        U = unclipped_rgb(native)
        parity = unclipped_reference_parity(native, U)
        return ForwardRecord(native, U, delivered_rgb(native), logits, lq, tokens, parity)


def forward_candidate(net, input256):
    torch = _runtime(); parameter_census(net)
    require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
            'Candidate ordinary autograd must span frozen decoder blocks')
    value = forward_record(net, input256)
    require(value.native.requires_grad and value.unclipped256.requires_grad and
            not value.logits.requires_grad and not value.lq.requires_grad,
            'Fusion output gradient route/upstream freeze differs')
    return value


def teacher_target(teacher, input256, *, kind):
    """One frozen original-clear or R2-dark teacher forward under no_grad."""
    torch = _runtime(); _frozen_teacher(teacher)
    require((kind == 'original' and teacher is _ORIGINAL) or (kind == 'r2' and teacher is _R2),
            'Target kind must name the exact factory teacher')
    require(not torch.is_inference_mode_enabled(), 'Teacher targets must be ordinary tensors')
    with torch.no_grad():
        value = forward_record(teacher, input256).unclipped256.detach().clone()
    require(not value.requires_grad and not torch.is_inference(value), 'Ordinary detached teacher U required')
    return TeacherTarget(value, kind)


def slot_terms(record, *, role, target256, footprint=None):
    """Complete scalar divided by4, unweighted terms, undivided scalar.

    Filter target is the known clean RGB256 tensor. Clear/dark require the exact
    typed original/R2 U teacher target; neither role silently receives clean GT.
    """
    torch = _runtime(); role_term_names(role)
    require(isinstance(record, ForwardRecord), 'Candidate ForwardRecord required')
    prediction = record.unclipped256; _rgb(prediction, 256, bounded=False)
    require(prediction.requires_grad and torch.is_grad_enabled(), 'Connected ordinary candidate U required')
    if role == 'filter':
        require(not isinstance(target256, TeacherTarget), 'Filter needs paired clean target, not behavior teacher')
        _rgb(target256, 256, bounded=True); target = target256
    else:
        require(isinstance(target256, TeacherTarget) and target256.kind == ('original' if role == 'clear' else 'r2'),
                'Role and exact frozen behavior teacher kind differ')
        require(footprint is None, 'Clear/dark target cannot receive filter footprint')
        target = target256.unclipped256; _rgb(target, 256, bounded=False)
    require(not target.requires_grad and not torch.is_inference(target), 'Ordinary detached target required')
    residual = prediction - target
    with torch.autocast(device_type='cuda', enabled=False):
        if role == 'filter':
            if footprint is None:
                spatial = residual.abs().mean()
            else:
                x0, y0, x1, y1 = validate_footprint(footprint)
                inside = torch.zeros((256, 256), device=prediction.device, dtype=torch.bool)
                inside[y0:y1, x0:x1] = True
                spatial = .5 * residual[..., inside].abs().mean() + .5 * residual[..., ~inside].abs().mean()
            adjacent = .5 * (residual[..., :, 1:] - residual[..., :, :-1]).abs().mean() + \
                       .5 * (residual[..., 1:, :] - residual[..., :-1, :]).abs().mean()
            terms = {'spatial_rgb_l1': spatial, 'adjacent_residual_l1': adjacent}
        else:
            terms = {'teacher_rgb_l1': residual.abs().mean()}
        require(tuple(terms) == role_term_names(role), 'Actual role loss graph differs')
        scalar = sum(term_weights(role)[name] * value for name, value in terms.items())
        require(all(value.ndim == 0 and torch.isfinite(value).item() for value in (*terms.values(), scalar)),
                'Finite scalar loss terms required')
        return scalar / SLOT_DIVISOR, terms, scalar


def clipping_statistics(record):
    torch = _runtime(); require(isinstance(record, ForwardRecord), 'ForwardRecord required')
    rgb = (record.native.detach() + 1) / 2
    u = record.unclipped256.detach()
    return {'native512_below0_fraction': float((rgb < 0).double().mean().item()),
            'native512_above1_fraction': float((rgb > 1).double().mean().item()),
            'U256_below0_fraction': float((u < 0).double().mean().item()),
            'U256_above1_fraction': float((u > 1).double().mean().item()),
            'native_RGB_min': float(rgb.min().item()), 'native_RGB_max': float(rgb.max().item()),
            'U_RGB_min': float(u.min().item()), 'U_RGB_max': float(u.max().item())}


def summarize_gradients(names, gradients, *, np):
    """Finite source-bound VM readouts, not independent gradient replay."""
    torch = _runtime(); shapes = expected_fusion_shapes()
    require(len(names) == len(gradients) == 72 and set(names) == set(shapes), 'Query all exact72 fusion tensors')
    rows = {}; sites = {site: {'tensors': 0, 'none': 0, 'nonzero': 0, 'squared_l2': 0.,
                              'maximum_absolute': 0.} for site in SITES}
    for name, gradient in zip(names, gradients):
        group = sites[fusion_site(name)]; group['tensors'] += 1
        if gradient is None:
            group['none'] += 1
            rows[name] = {'is_none': True, 'nonzero': False, 'l2': 0., 'maximum_absolute': 0., 'sha256': None}
            continue
        require(gradient.dtype == torch.float32 and tuple(gradient.shape) == shapes[name], 'Gradient FP32/schema differs')
        array = np.ascontiguousarray(gradient.detach().cpu().numpy())
        require(np.isfinite(array).all(), 'Nonfinite witness gradient')
        squared = float(np.square(array.astype(np.float64)).sum(dtype=np.float64))
        maximum = float(np.abs(array).max())
        group['squared_l2'] += squared; group['nonzero'] += int(maximum > 0)
        group['maximum_absolute'] = max(group['maximum_absolute'], maximum)
        rows[name] = {'is_none': False, 'nonzero': maximum > 0, 'l2': math.sqrt(squared),
                      'maximum_absolute': maximum, 'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
    for group in sites.values():
        group['l2'] = math.sqrt(group.pop('squared_l2'))
        require(group['tensors'] == 18 and math.isfinite(group['l2']), 'Exact18 finite gradients per fusion site required')
    return {'parameters': rows, 'sites': sites, 'queried_tensors': 72,
            'full_gradient_arrays_exported': False, 'source_bound_readout_not_independent_replay': True}


def validate_gradient_summary(summary, *, allow_all_zero=False):
    require(type(allow_all_zero) is bool and summary['queried_tensors'] == 72 and
            set(summary['sites']) == set(SITES), 'Exact fusion gradient summary required')
    for site, row in summary['sites'].items():
        require(row['tensors'] == 18 and row['none'] == 0 and math.isfinite(row['l2']) and
                math.isfinite(row['maximum_absolute']), 'Disconnected/nonfinite fusion gradient site: ' + site)
        require(allow_all_zero or (row['nonzero'] > 0 and row['l2'] > 0),
                'Every fusion site needs nonzero improvement gradient: ' + site)
    return {'pass': True, 'all_four_sites_finite_connected': True,
            'all_four_sites_nonzero_required': not allow_all_zero,
            'zero_initial_identical_R2_anchor_permitted': allow_all_zero}


def validate_initial_parity(candidate, teacher, *, np=None):
    """Exact same-context R2 candidate/teacher before any Adam construction."""
    torch = _runtime()
    require(isinstance(candidate, ForwardRecord) and isinstance(teacher, ForwardRecord), 'Full forward records required')
    fields = ('native', 'unclipped256', 'delivered256', 'logits', 'lq', 'tokens')
    differences = {}
    for name in fields:
        left, right = getattr(candidate, name).detach(), getattr(teacher, name).detach()
        require(left.shape == right.shape and left.dtype == right.dtype, 'Initial parity schema differs: ' + name)
        exact = bool(torch.equal(left, right))
        differences[name] = {'exact': exact, 'maximum_absolute': float((left.double() - right.double()).abs().max().item())}
        require(exact, 'Initial same-context exact R2 parity failed: ' + name)
    require(torch.equal(png_tensor(candidate.delivered256), png_tensor(teacher.delivered256)), 'Initial floored PNG parity differs')
    return {'pass': True, 'fields': differences, 'floored_PNG_exact': True,
            'context': 'same runtime, same input, eval FP32, w1,detach16,AdaINon', 'floating_tolerance': 0.}


def readout_fingerprints(record):
    """Source-bound frozen upstream readouts for before/after fixed inputs."""
    _runtime(); require(isinstance(record, ForwardRecord), 'ForwardRecord required')
    return {name: tensor_state_digest({name: getattr(record, name)})
            for name in ('logits', 'lq', 'tokens')}


def validate_upstream_readouts(before, after):
    require(set(before) == set(after) == {'logits', 'lq', 'tokens'} and before == after,
            'Fixed-input upstream logits/LQ/hard-token readouts changed')
    return {'pass': True, 'fixed_input_logits_LQ_hard_tokens_exact': True,
            'not_an_identity_accuracy_measure': True}


def component_sum_witness(whole, components, *, role, np):
    """Compare VM-computed gradients of complete divided slot to components.

    Each component is the gradient of its UNWEIGHTED term. Query/mode/RNG/grad
    restoration is the worker's responsibility. Fixed pre-outcome FP32 tolerance
    handles addition ordering only; this is not a quality or preservation gate.
    """
    _runtime(); names = set(expected_fusion_shapes()); weights = term_weights(role)
    require(set(whole) == names and set(components) == set(weights) and
            all(set(values) == names for values in components.values()), 'Complete exact72 gradient maps required')
    maximum_error = 0.; maximum_relative = 0.
    for name in sorted(names):
        actual = np.asarray(whole[name].detach().cpu().numpy(), dtype=np.float64)
        expected = sum(weights[term] * np.asarray(components[term][name].detach().cpu().numpy(), dtype=np.float64)
                       for term in weights) / SLOT_DIVISOR
        require(actual.shape == expected.shape == expected_fusion_shapes()[name] and
                np.isfinite(actual).all() and np.isfinite(expected).all(), 'Finite component witness schema required')
        error = np.abs(actual - expected)
        maximum_error = max(maximum_error, float(error.max()))
        maximum_relative = max(maximum_relative, float((error / np.maximum(np.abs(expected), 1e-30)).max()))
        require(np.all(error <= 1e-7 + 1e-5 * np.abs(expected)), 'Whole divided-slot/component gradient mismatch: ' + name)
    return {'pass': True, 'tensors': 72, 'maximum_absolute_error': maximum_error,
            'maximum_relative_error': maximum_relative, 'absolute_tolerance': 1e-7,
            'relative_tolerance': 1e-5, 'slot_divisor': SLOT_DIVISOR,
            'not_a_quality_gate': True, 'source_bound_VM_gradient_readout': True}


def prerequisite_context(net, teachers):
    """Capture modes, fixed model state, RNG and exact style callable identity."""
    torch = _runtime(); parameter_census(net)
    require(_ARCHITECTURE is not None and all(p.grad is None for p in net.parameters()), 'Fresh no-gradient witness context required')
    for teacher in teachers:
        _frozen_teacher(teacher)
    return {'candidate_state': state_digest(net), 'frozen_state': frozen_state_digest(net),
            'teacher_states': [state_digest(teacher) for teacher in teachers],
            'module_modes': tuple(m.training for m in net.modules()),
            'CPU_RNG': torch.get_rng_state().clone(), 'CUDA_RNG': torch.cuda.get_rng_state(0).clone(),
            'style_callable': _ARCHITECTURE.adaptive_instance_normalization}


def validate_prerequisite_restoration(before, net, teachers):
    torch = _runtime(); parameter_census(net)
    require(all(p.grad is None for p in net.parameters()), 'Clear all candidate grad fields to None after witness')
    for teacher in teachers:
        _frozen_teacher(teacher)
    require(before['candidate_state'] == state_digest(net) and before['frozen_state'] == frozen_state_digest(net) and
            before['teacher_states'] == [state_digest(teacher) for teacher in teachers] and
            before['module_modes'] == tuple(m.training for m in net.modules()) and
            torch.equal(before['CPU_RNG'], torch.get_rng_state()) and
            torch.equal(before['CUDA_RNG'], torch.cuda.get_rng_state(0)) and
            before['style_callable'] is _ARCHITECTURE.adaptive_instance_normalization,
            'Witness changed model values, modes, RNG or AdaIN callable')
    return {'pass': True, 'all_states_modes_RNG_style_callable_exact': True,
            'all_candidate_and_teacher_grad_fields_None': True, 'optimizer_constructed_by_model_helper': False}
