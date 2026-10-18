"""Guarded role-preserving CodeFormer objective; unchanged inference architecture.

Importing this file uses the standard library only. The candidate starts from
the real historical full-coarse fifty-update state. The r1 namespace repairs
initializer/witness measurement contexts; it changes no architecture or loss.
Normal observations have
only original-teacher RGB/token distillation; dark observations retain the
existing clean objective plus full-coarse-parent distillation. The worker owns
the scoped AdaIN style detach, every gradient/Adam call and finite counters.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys


VM_NAME = 'cctv_codeformer_role_preserving_training_v1_r1_vm'
PARENT_NAME = 'cctv_codeformer_adain_gradient_isolation_v1_vm'
PARENT_PROTOCOL = '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
ORIGINAL_DIGEST = '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd'
BASE_DIGEST = '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b'
ASSETS = {
    'initializer': {'root': 'adain', 'path': 'weights/codeformer.pth',
        'sha256': '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7', 'bytes': 376637898},
    'full_coarse_checkpoint': {'root': 'adain',
        'path': 'outputs/adain_gradient_isolation_v1/stopped_checkpoint.pt',
        'sha256': '43c0f313365e13849ddcd852f401511e4c7ae52cbac55fdd98a6b032f06ef37b', 'bytes': 855230072},
    'hq_teacher': {'root': 'adain', 'path': 'weights/vqgan_code1024.pth',
        'sha256': '4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f', 'bytes': 255078528}}
HISTORICAL_MODULES = {
    'research.codeformer_adaptation_v1': 'research/codeformer_adaptation_v1/__init__.py',
    'research.codeformer_adaptation_v1.common': 'research/codeformer_adaptation_v1/common.py',
    'research.codeformer_adaptation_v1.perceptual': 'research/codeformer_adaptation_v1/perceptual.py',
    'research.codeformer_adaptation_v2': 'research/codeformer_adaptation_v2/__init__.py',
    'research.codeformer_adaptation_v2.model': 'research/codeformer_adaptation_v2/model.py',
    'research.codeformer_adain_gradient_isolation_v1': 'research/codeformer_adain_gradient_isolation_v1/__init__.py',
    'research.codeformer_adain_gradient_isolation_v1.gradient_route': 'research/codeformer_adain_gradient_isolation_v1/gradient_route.py',
    'third_party': 'third_party/__init__.py',
    'third_party.codeformer': 'third_party/codeformer/__init__.py',
    'third_party.codeformer.registry': 'third_party/codeformer/registry.py',
    'third_party.codeformer.vqgan_arch': 'third_party/codeformer/vqgan_arch.py',
    'third_party.codeformer.codeformer_arch': 'third_party/codeformer/codeformer_arch.py'}
LOSS_WEIGHTS = {'token_ce': .5, 'latent_mse': 1., 'coarse_rgb_l1': 4.,
    'pixel_l1_rgb': 1., 'lpips_vgg': 1., 'behavior_rgb_l1': 2., 'behavior_token_kl': .1}
NORMAL_TERMS = ('behavior_rgb_l1', 'behavior_token_kl')
DARK_TERMS = ('token_ce', 'latent_mse', 'coarse_rgb_l1', 'pixel_l1_rgb', 'lpips_vgg',
              'behavior_rgb_l1', 'behavior_token_kl')
IMAGE_TERMS = frozenset(('coarse_rgb_l1', 'pixel_l1_rgb', 'lpips_vgg', 'behavior_rgb_l1'))
TOKEN_TERMS = frozenset(('token_ce', 'latent_mse', 'behavior_token_kl'))
TRAINABLE_PREFIXES = ('encoder', 'position_emb', 'feat_emb', 'ft_layers', 'idx_pred_layer', 'fuse_convs_dict')
FROZEN_PREFIXES = ('quantize', 'generator')
_TORCH = _ROOT = _PROTOCOL_SHA = _BASE = _ARCHITECTURE = _ROUTE = _PERCEPTUAL_MODULE = None
_ORIGINAL = _PARENT = None


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def regular(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            s = item.lstat()
            require(not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024,
                    'Linked/reparse model or source path forbidden')
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Exact unlinked regular source required')
    return path.resolve(strict=True)


def read_json(path):
    def unique(items):
        out = {}
        for name, value in items:
            require(name not in out, 'Duplicate metadata key')
            out[name] = value
        return out
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite metadata')))


def activate_runtime(root, protocol_sha):
    """Strict new Linux/home/tmux/protocol guards precede the sole Torch import."""
    global _TORCH, _ROOT, _PROTOCOL_SHA
    require(platform.system() == 'Linux', 'Actual neural calls are manual Linux L4 VM only')
    root = Path(root).absolute()
    expected = Path.home() / 'forensic-dgp' / VM_NAME
    for path in (root, expected):
        for item in (path, *path.parents):
            if item.exists() or item.is_symlink():
                s = item.lstat()
                require(not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024,
                        'Linked/reparse manual runtime root forbidden')
    require(root.resolve(strict=True) == expected.resolve(strict=True) and root.is_dir(), 'Exact new manual VM root required')
    require(bool(os.environ.get('TMUX')), 'Manual tmux required before Torch import')
    require(isinstance(protocol_sha, str) and re.fullmatch('[0-9a-f]{64}', protocol_sha), 'Exact protocol SHA required')
    pp = regular(root / 'protocol.json')
    require(pp.stat().st_size <= 8*1024**2 and sha256(pp) == protocol_sha, 'Frozen new protocol differs')
    from . import contract as c
    c.manual_root(root)
    c.verify_protocol(read_json(pp))
    if _TORCH is not None:
        require(_ROOT == root.resolve(strict=True) and _PROTOCOL_SHA == protocol_sha, 'Do not rebind an activated model runtime')
        return _TORCH
    require(not any(name in sys.modules for name in HISTORICAL_MODULES), 'Historical model modules must be fresh before activation')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == 'NVIDIA L4', 'Exactly one actual NVIDIA L4 required')
    _TORCH, _ROOT, _PROTOCOL_SHA = torch, root.resolve(strict=True), protocol_sha
    return torch


def _runtime():
    require(_TORCH is not None and _ROOT is not None, 'Activate guarded new manual VM runtime first')
    return _TORCH


def role_term_names(role, image_eligible=True):
    """Pure standard-library role dispatch; the network receives no role input."""
    require(role in ('normal', 'dark') and type(image_eligible) is bool, 'Explicit role and image-eligibility boolean required')
    if role == 'normal':
        return NORMAL_TERMS
    return DARK_TERMS if image_eligible else tuple(n for n in DARK_TERMS if n not in ('pixel_l1_rgb', 'lpips_vgg'))


def parameter_group(name):
    prefix = name.split('.', 1)[0]
    require(prefix in TRAINABLE_PREFIXES + FROZEN_PREFIXES, 'Undeclared CodeFormer parameter group')
    return prefix


def tensor_state_digest(state):
    torch = _runtime(); h = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor), 'Tensor state required')
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode('utf-8')
        h.update(len(header).to_bytes(8, 'big')); h.update(header)
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_digest(module):
    _runtime()
    return tensor_state_digest(module.state_dict())


def frozen_state_digest(net):
    _runtime()
    return tensor_state_digest({n: p for n, p in net.state_dict().items() if parameter_group(n) in FROZEN_PREFIXES})


def cpu_state(module):
    _runtime()
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def parameter_census(net):
    torch = _runtime()
    require(len(tuple(net.parameters())) == len(net.state_dict()) == 515, 'Exact515 CodeFormer parameter/state tensors required')
    for name, value in net.named_parameters():
        require(value.requires_grad == (parameter_group(name) in TRAINABLE_PREFIXES) and value.dtype == torch.float32
            and value.device.type == 'cuda', 'Candidate trainability/FP32/CUDA differs: ' + name)
    census = _BASE.parameter_census(net)
    require(len(census['trainable_names']) == 350 and len(census['frozen_names']) == 165, 'Exact350/165 trainability required')
    require(sum(n.startswith('fuse_convs_dict.') for n in census['trainable_names']) == 72, 'Exact72 fusion trainable tensors required')
    return census


def set_training_mode(net):
    _runtime(); parameter_census(net)
    net.train(); net.quantize.eval(); net.generator.eval()
    return net


def optimizer_groups(net):
    """Group objects only; no optimizer construction, step, schedule or resume."""
    _runtime(); parameter_census(net)
    rows = []
    for name, lr in (('encoder_token', 1e-5), ('fusion', 2e-5)):
        selected = [p for n, p in net.named_parameters() if p.requires_grad and
                    (n.startswith('fuse_convs_dict.') == (name == 'fusion'))]
        require(len(selected) == (72 if name == 'fusion' else 278), 'Optimizer group tensor census differs')
        rows.append({'name': name, 'params': selected, 'lr': lr, 'base_lr': lr})
    return rows


def _check_pin(path, pin):
    path = regular(path); before = path.stat()
    require(before.st_size == pin['bytes'] and sha256(path) == pin['sha256'], 'Exact asset hash/size differs')
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino),
            'Asset changed during hashing')
    return path


def _historical_imports(parent):
    global _BASE, _ARCHITECTURE, _ROUTE, _PERCEPTUAL_MODULE
    protocol = regular(parent / 'protocol.json')
    require(sha256(protocol) == PARENT_PROTOCOL, 'Exact immutable AdaIN parent protocol required')
    old = read_json(protocol)
    require(not any(name in sys.modules for name in HISTORICAL_MODULES), 'Historical factory/architecture modules already cached')
    for name, relative in HISTORICAL_MODULES.items():
        path = regular(parent / relative)
        require(sha256(path) == old['files_sha256'][relative], 'Immutable historical source differs: ' + relative)
    # find_spec may load only the already byte-checked package initializers.
    for name, relative in HISTORICAL_MODULES.items():
        spec = importlib.util.find_spec(name)
        require(spec is not None and spec.origin is not None and Path(spec.origin).resolve(strict=True) == (parent / relative).resolve(strict=True),
                'Historical module resolves outside exact parent: ' + name)
    from research.codeformer_adaptation_v2 import model as base_model
    from research.codeformer_adaptation_v1 import perceptual
    from research.codeformer_adain_gradient_isolation_v1 import gradient_route
    from third_party.codeformer import codeformer_arch
    _BASE, _PERCEPTUAL_MODULE, _ROUTE, _ARCHITECTURE = base_model, perceptual, gradient_route, codeformer_arch
    return old


def historical_modules():
    _runtime()
    require(_BASE is not None and _ARCHITECTURE is not None and _ROUTE is not None, 'Load verified model sources first')
    return _BASE, _ARCHITECTURE, _ROUTE


def _frozen(net, count=None):
    torch = _runtime()
    require(all(not m.training for m in net.modules()) and
            all(not p.requires_grad and p.grad is None and p.dtype == torch.float32 and p.device.type == 'cuda' for p in net.parameters()),
            'Frozen eval FP32 CUDA teacher required')
    if count is not None:
        require(len(tuple(net.parameters())) == len(net.state_dict()) == count, 'Frozen model tensor census differs')


def load_role_models(initializer, full_coarse_checkpoint, hq_teacher):
    """Return candidate,HQ,original,parent,census,receipt; install model weights only."""
    global _ORIGINAL, _PARENT
    torch = _runtime()
    require(_BASE is None and _ORIGINAL is None and _PARENT is None, 'Construct this finite model factory once only')
    parent = (_ROOT.parent / PARENT_NAME).resolve(strict=True)
    require(parent.parent == _ROOT.parent and parent != _ROOT, 'Exact immutable parent sibling required')
    from . import contract as c
    supplied = {'initializer': initializer, 'full_coarse_checkpoint': full_coarse_checkpoint, 'hq_teacher': hq_teacher}
    paths = {}
    for name, pin in ASSETS.items():
        require(c.ASSETS[name] == pin, 'Model/contract asset policy differs')
        expected = parent / pin['path']
        require(Path(supplied[name]).resolve(strict=True) == expected.resolve(strict=True), 'Asset outside exact declared parent namespace')
        paths[name] = _check_pin(supplied[name], pin)
    _historical_imports(parent)
    before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    with torch.random.fork_rng(devices=[0]):
        original, _ = _BASE.load_trainable(paths['initializer'], device='cuda')
        original.eval().requires_grad_(False)
        candidate = deepcopy(original)
        saved = torch.load(paths['full_coarse_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
        require(saved.get('format') == 'codeformer-adain-gradient-isolation-state-v1' and
            saved.get('protocol_sha256') == PARENT_PROTOCOL and saved.get('completed_updates') == 50 and
            saved.get('partial_microbatches_discarded') == 0 and saved.get('resume_safe') is True,
            'Expected actual historical committed full-coarse50 state')
        state = saved['model']
        require(list(state) == list(candidate.state_dict()) and len(state) == 515 and all(
            isinstance(p, torch.Tensor) and p.dtype == torch.float32 and torch.isfinite(p).all().item() for p in state.values()),
            'Historical515 model state geometry/precision/values differ')
        candidate.load_state_dict(state, strict=True)
        del state, saved  # Historical Adam/RNG/sampler fields are never installed.
        parent_teacher = deepcopy(candidate).eval().requires_grad_(False)
        for name, parameter in candidate.named_parameters():
            parameter.requires_grad_(parameter_group(name) in TRAINABLE_PREFIXES)
        candidate.eval()
        clean_teacher = _BASE.load_teacher(paths['hq_teacher'], device='cuda').float().eval().requires_grad_(False)
    require(torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(before_cuda, torch.cuda.get_rng_state(0)),
            'Model construction changed pre-call CPU/CUDA RNG')
    census = parameter_census(candidate)
    _frozen(original, 515); _frozen(parent_teacher, 515); _frozen(clean_teacher)
    require(all(p.grad is None for p in candidate.parameters()), 'Fresh candidate has parameter gradients')
    digests = {'original': state_digest(original), 'base': state_digest(parent_teacher), 'candidate': state_digest(candidate)}
    require(digests == {'original': ORIGINAL_DIGEST, 'base': BASE_DIGEST, 'candidate': BASE_DIGEST}, 'Actual original/base/candidate identities differ')
    require(c.MODEL_DIGESTS['original'] == ORIGINAL_DIGEST and c.MODEL_DIGESTS['base'] == BASE_DIGEST, 'Contract model identities differ')
    _ORIGINAL, _PARENT = original, parent_teacher
    receipt = {'complete': True, 'asset_sha256': {n: p['sha256'] for n, p in ASSETS.items()},
        'canonical_tensor_state_sha256': {**digests, 'clean_teacher': state_digest(clean_teacher)},
        'candidate_parameter_tensors': 515, 'candidate_trainable_tensors': 350, 'candidate_frozen_tensors': 165,
        'candidate_parameter_census': census, 'original_and_parent_frozen_tensors_each': 515,
        'candidate_starts_from_actual_historical_full_coarse50': True, 'fresh_optimizer_required': True,
        'historical_optimizer_RNG_sampler_schedule_installed': False, 'construction_RNG_preserved_exact': True,
        'all_teachers_eval_frozen_FP32_CUDA_grad_None': True, 'neural_forwards': 0,
        'gradient_queries': 0, 'backward_calls': 0, 'optimizer_constructions': 0, 'optimizer_steps': 0,
        'forward_architecture_and_delivery_unchanged': True,
        'candidate_gradient_route_scope_owned_by_worker': 'pinned candidate-AdaIN-style-detach-only-v1 context'}
    return candidate, clean_teacher, original, parent_teacher, census, receipt


def load_perceptual(asset_directory):
    """Exact immutable old LPIPS loader; no network/download or new normalization."""
    _runtime()
    require(_PERCEPTUAL_MODULE is not None, 'Load verified historical modules before perceptual factory')
    directory = Path(asset_directory).resolve(strict=True)
    parent = (_ROOT.parent / PARENT_NAME).resolve(strict=True)
    require(directory == parent / 'perceptual', 'Exact declared parent perceptual directory required')
    old = read_json(parent / 'protocol.json')
    manifest = regular(directory / 'manifest.json')
    require(sha256(manifest) == old['files_sha256']['perceptual/manifest.json'], 'Exact parent perceptual manifest differs')
    for key in ('perceptual_features', 'perceptual_calibration'):
        pin = old['assets'][key]
        _check_pin(parent / pin['path'], pin)
    result = _PERCEPTUAL_MODULE.load_perceptual(directory, device='cuda').float().eval().requires_grad_(False)
    _frozen(result)
    return result


def prepare_input(input256):
    _runtime(); require(_BASE is not None, 'Load verified base helper first')
    return _BASE.prepare_input(input256)


def delivered_rgb(output512):
    _runtime(); require(_BASE is not None, 'Load verified base helper first')
    return _BASE.delivered_rgb(output512)


def infer(net, input256):
    """One unchanged CodeFormer forward; caller selects inference mode."""
    _runtime()
    return net(prepare_input(input256), w=1.0, detach_16=True, adain=True)[0]


@dataclass(frozen=True)
class CleanCodes:
    indices: object
    features: object


@dataclass(frozen=True)
class BehaviorTargets:
    delivered256: object
    logits: object
    teacher_kind: str


def hq_targets(teacher, clean512, codebook=None):
    _runtime(); require(_BASE is not None, 'Load verified HQ helper first')
    value = _BASE.hq_targets(teacher, clean512, codebook=codebook)
    return CleanCodes(value.indices, value.features)


def assert_shared_codebook(net, teacher):
    _runtime(); require(_BASE is not None, 'Load verified codebook helper first')
    return _BASE.assert_shared_codebook(net, teacher)


def behavior_targets(teacher, input256):
    """One exact original/parent teacher forward with ordinary detached clones."""
    torch = _runtime(); _frozen(teacher, 515)
    require(teacher is _ORIGINAL or teacher is _PARENT, 'Role target must use an exact factory teacher object')
    teacher_kind = 'original' if teacher is _ORIGINAL else 'full_coarse_parent'
    require(not torch.is_inference_mode_enabled(), 'Behavior targets must be built outside inference_mode')
    with torch.no_grad(), torch.autocast(device_type=input256.device.type, enabled=False):
        native, logits, _ = teacher(prepare_input(input256.float()), w=1.0, detach_16=True, adain=True)
        value = BehaviorTargets(delivered_rgb(native).detach().float().clone(), logits.detach().float().clone(), teacher_kind)
    _behavior(value, len(input256), teacher_kind)
    return value


def _behavior(value, batch, expected_kind):
    torch = _runtime()
    require(isinstance(value, BehaviorTargets) and value.teacher_kind == expected_kind, 'Behavior teacher does not match TRAIN objective role')
    require(value.delivered256.shape == (batch, 3, 256, 256) and value.logits.shape == (batch, 256, 1024), 'Behavior target geometry differs')
    for array in (value.delivered256, value.logits):
        require(isinstance(array, torch.Tensor) and array.dtype == torch.float32 and array.device.type == 'cuda'
            and torch.isfinite(array).all().item() and not array.requires_grad and not torch.is_inference(array),
            'Detached ordinary finite FP32 CUDA behavior targets required')
    require(value.delivered256.min().item() >= 0 and value.delivered256.max().item() <= 1, 'Delivered teacher RGB must be[0,1]')


def token_distillation(candidate_logits, target_logits):
    _runtime(); require(_BASE is not None, 'Load verified token-distillation helper first')
    return _BASE.token_distillation(candidate_logits, target_logits)


def forward_role_terms(net, clean_teacher, perceptual, input256, clean512, *, role,
                       image_eligible, codes=None, behavior):
    """One candidate forward; worker wraps it in its pinned style-detach scope.

    No normal clean-target graph is built, including an unused scalar graph.
    Dark native512 RGB is unclipped for L1/pooling and raw signed LPIPS. The
    behavior RGB target uses the unchanged clipped delivered256 policy.
    """
    torch = _runtime(); require(_BASE is not None, 'Load verified base helper first')
    expected = role_term_names(role, image_eligible)
    _behavior(behavior, len(input256), 'original' if role == 'normal' else 'full_coarse_parent')
    if role == 'normal':
        require(clean512 is None and codes is None, 'Normal role must not receive clean-supervised data/codes')
    else:
        _BASE._rgb(clean512, 512)
        if codes is None:
            require(clean_teacher is not None, 'Dark clean codes require the frozen HQ teacher')
            codes = hq_targets(clean_teacher, clean512, codebook=net.quantize)
        require(codes.indices.shape == (len(input256), 256) and codes.features.shape == (len(input256), 256, 16, 16)
            and codes.indices.dtype == torch.int64 and codes.features.dtype == torch.float32
            and codes.indices.device.type == codes.features.device.type == 'cuda'
            and codes.indices.min().item() >= 0 and codes.indices.max().item() < 1024
            and torch.isfinite(codes.features).all().item()
            and not codes.indices.requires_grad and not codes.features.requires_grad
            and not torch.is_inference(codes.indices) and not torch.is_inference(codes.features), 'Ordinary detached dark HQ codes required')
    native, logits, feature = net(prepare_input(input256), w=1.0, detach_16=True, adain=True)
    require(native.dtype == logits.dtype == feature.dtype == torch.float32 and torch.isfinite(native).all().item()
        and torch.isfinite(logits).all().item() and torch.isfinite(feature).all().item(), 'Finite candidate FP32 forward required')
    terms = {}
    with torch.autocast(device_type=native.device.type, enabled=False):
        if role == 'dark':
            rgb = (native.float() + 1) / 2
            terms['token_ce'] = _BASE.F.cross_entropy(logits.float().transpose(1, 2), codes.indices)
            terms['latent_mse'] = _BASE.F.mse_loss(feature.float(), codes.features.float())
            terms['coarse_rgb_l1'] = _BASE.F.l1_loss(_BASE.F.avg_pool2d(rgb, 32, 32), _BASE.F.avg_pool2d(clean512.float(), 32, 32))
            if image_eligible:
                require(perceptual is not None, 'Eligible dark image objective requires frozen LPIPS')
                _frozen(perceptual)
                terms['pixel_l1_rgb'] = _BASE.F.l1_loss(rgb, clean512.float())
                terms['lpips_vgg'] = perceptual(native.float(), clean512.float() * 2 - 1)
        terms['behavior_rgb_l1'] = _BASE.F.l1_loss(delivered_rgb(native), behavior.delivered256.float())
        terms['behavior_token_kl'] = token_distillation(logits, behavior.logits)
        require(tuple(terms) == expected, 'Actual role-specific term graph differs from declared objective')
        total = sum(LOSS_WEIGHTS[name] * value for name, value in terms.items())
    require(all(value.ndim == 0 and torch.isfinite(value).item() for value in terms.values())
        and torch.isfinite(total).item(), 'Finite scalar role objective required')
    return total, terms, native
