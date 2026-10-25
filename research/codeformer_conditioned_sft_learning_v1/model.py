"""Inert Conditioned SFT V1 factory; only the exact manual L4 stage may build.

Both arms start from all 575 retained original trained-u64 tensors. Arm A trains
the 36 existing spatial-head tensors. Arm B additionally trains the 24 existing
U96 pre-SFT adapter tensors with multi-site Clean-SFT feature distillation at
sites 32/64/128/256 and input-only exposure/degradation conditioning.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import threading
import weakref

from . import conditioning


STATUS = 'INERT_CONDITIONED_SFT_LEARNING_SOURCE_ONLY_NOT_A_WORKER'
VM_NAME = 'cctv_codeformer_conditioned_sft_learning_v1_vm'
ROOT = Path(__file__).resolve().parents[2]
SITES = ('32', '64', '128', '256')
CHANNELS = {'32': 256, '64': 256, '128': 128, '256': 128}
WIDTH = 32
INITIALIZATION_SEED = 20261021
BASE_TENSORS = 515
ADAPTER_TENSORS = 24
ADAPTER_PARAMETERS = 51328
HEAD_TENSORS = 36
HEAD_PARAMETERS = 168960
TOTAL_TENSORS = 575
TERMINAL_HEADS_CANONICAL_SHA256 = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
FULL_STATE_CANONICAL_SHA256 = '47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a'
R2_CANONICAL_SHA256 = 'a3b0986c793cb7c937ca82efde6c9e23e705557d76b93406e0f53e51b938f9e7f'
# Exact 64-hex R2 digest from retained u64/matched factory
R2_CANONICAL_SHA256 = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_CANONICAL_SHA256 = 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
ARTIFACTS = {
    'r2': {'path': 'weights/codeformer_r2.pth', 'bytes': 376609465,
           'sha256': 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'},
    'u96': {'path': 'weights/codeformer_u96_adapter.pth', 'bytes': 212827,
            'sha256': '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'},
    'heads': {'path': 'weights/terminal_u0064_heads.npz', 'bytes': 685818,
              'sha256': '77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17'},
}
RETAINED_FACTORY_SOURCES = {
    'research/codeformer_u96_postfusion_v1/model.py': {
        'bytes': 23240,
        'sha256': 'ca2b466161932068c01980bb41fcc08e12ee8f0120f17c8fdf47898f35934aa8'},
    'research/codeformer_u64_cleananchors_continuation_v1/model.py': {
        'bytes': 21294,
        'sha256': '6a67468fd6b94e938efe8548205282c048635fcc63e8e4efd8389a2910129829'},
}
ARCHITECTURE_SOURCES = {
    'third_party.codeformer.codeformer_arch': {
        'path': 'third_party/codeformer/codeformer_arch.py',
        'sha256': 'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb'},
    'third_party.codeformer.vqgan_arch': {
        'path': 'third_party/codeformer/vqgan_arch.py',
        'sha256': 'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf'},
    'third_party.codeformer.registry': {
        'path': 'third_party/codeformer/registry.py',
        'sha256': 'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e'},
}
_OWNERS = weakref.WeakKeyDictionary()
_OWNERS_LOCK = threading.Lock()
_OWNER_ATTRIBUTE = '_forensic_codeformer_exclusive_owner'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_adapter_shapes():
    """Pure retained 24-tensor / 51,328-coordinate U96 schema."""
    result = {}
    for site, channels in CHANNELS.items():
        shapes = {
            'down.weight': (WIDTH, channels, 1, 1), 'down.bias': (WIDTH,),
            'depthwise.weight': (WIDTH, 1, 3, 3), 'depthwise.bias': (WIDTH,),
            'up.weight': (channels, WIDTH, 1, 1), 'up.bias': (channels,),
        }
        result.update({'adapters.' + site + '.' + name: shape for name, shape in shapes.items()})
    require(len(result) == ADAPTER_TENSORS and
            sum(math.prod(shape) for shape in result.values()) == ADAPTER_PARAMETERS,
            'Exact24/51328 retained U96 schema required')
    return result


def expected_head_shapes():
    """Pure retained 36-tensor / 168,960-coordinate learned-head schema."""
    result = {}
    for site, channels in CHANNELS.items():
        shapes = {
            'encoder.weight': (WIDTH, channels, 1, 1), 'encoder.bias': (WIDTH,),
            'decoder.weight': (WIDTH, channels, 1, 1), 'fused.weight': (WIDTH, channels, 1, 1),
            'coarse.weight': (WIDTH, 256, 1, 1), 'spatial.weight': (WIDTH, WIDTH, 3, 3),
            'spatial.bias': (WIDTH,), 'output.weight': (channels, WIDTH, 1, 1),
            'output.bias': (channels,),
        }
        result.update({'heads.' + site + '.' + name: shape for name, shape in shapes.items()})
    require(len(result) == HEAD_TENSORS and
            sum(math.prod(shape) for shape in result.values()) == HEAD_PARAMETERS,
            'Exact36/168960 retained learned-head schema required')
    return result


def arm_census(arm):
    """Static parameter permissions, independent of any model construction."""
    require(type(arm) is str and arm in ('A', 'B'), 'Explicit matched arm A or B required')
    adapters_trainable = arm == 'B'
    return {
        'arm': arm, 'total_tensors': TOTAL_TENSORS, 'base_tensors': BASE_TENSORS,
        'adapter_tensors': ADAPTER_TENSORS, 'head_tensors': HEAD_TENSORS,
        'adapter_parameters': ADAPTER_PARAMETERS, 'head_parameters': HEAD_PARAMETERS,
        'trainable_tensors': HEAD_TENSORS + (ADAPTER_TENSORS if adapters_trainable else 0),
        'trainable_parameters': HEAD_PARAMETERS + (ADAPTER_PARAMETERS if adapters_trainable else 0),
        'frozen_tensors': BASE_TENSORS + (0 if adapters_trainable else ADAPTER_TENSORS),
        'trainable_U96_adapters': adapters_trainable,
        'frozen_digest_targets': ['R2'] if adapters_trainable else ['R2', 'U96'],
    }


def _ordinary(path, *, directory=False):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        is_directory = item != path or directory
        expected = stat.S_ISDIR(info.st_mode) if is_directory else stat.S_ISREG(info.st_mode)
        require(expected and not stat.S_ISLNK(info.st_mode) and
                not getattr(info, 'st_file_attributes', 0) & 1024,
                'Ordinary unlinked path ancestry required')
        if item == path and not directory:
            require(info.st_nlink == 1 and 0 < info.st_size <= 512 * 1024 ** 2,
                    'Bounded single-link nonempty artifact required')
    return path.resolve(strict=True)


def file_pin(path):
    """Read-only byte binding; no checkpoint decoder or framework import."""
    path = _ordinary(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            'Artifact changed during byte hashing')
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def validate_manual_context(*, system, root, home, cwd, prefix, device, tmux, gpu_names):
    """Pure positive/negative context schema; performs no runtime discovery."""
    require(system == 'Linux', 'Manual Linux L4 VM only; no local/app model execution')
    home_path = PurePosixPath(home)
    expected = home_path / 'forensic-dgp' / VM_NAME
    require(type(home) is str and home_path.is_absolute() and '..' not in home_path.parts and
            type(root) is str and root == str(expected) and '..' not in PurePosixPath(root).parts,
            'Exact future manual conditioned-sft-learning stage required')
    require(cwd == root and prefix == str(home_path / 'forensic-dgp/cctv_dgp_vm_bundle/.venv'),
            'Exact staged cwd and retained VM virtual environment required')
    require(device == 'cuda:0' and type(tmux) is str and bool(tmux.strip()),
            'Explicit cuda:0 and manual tmux required')
    require(type(gpu_names) is tuple and gpu_names == ('NVIDIA L4',),
            'Exactly one NVIDIA L4 required')
    return True


def _assert_manual_location(root):
    require(platform.system() == 'Linux', 'Manual Linux L4 VM only; no local/app model execution')
    expected = Path.home() / 'forensic-dgp' / VM_NAME
    require(root == ROOT == expected and _ordinary(root, directory=True) == expected and
            _ordinary(Path.cwd(), directory=True) == expected,
            'Factory source and cwd must belong to the exact future manual stage')
    require(_ordinary(sys.prefix, directory=True) ==
            Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv' and
            bool(os.environ.get('TMUX', '').strip()),
            'Retained manual VM virtual environment and tmux required')


def _guard_manual_vm():
    """Finish every standard-library runtime guard before importing Torch."""
    require(platform.system() == 'Linux', 'Manual Linux L4 VM only; no local/app model execution')
    require(sys.version_info >= (3, 10), 'Manual Python3.10+ required')
    root = ROOT
    _assert_manual_location(root)
    for relative, expected in RETAINED_FACTORY_SOURCES.items():
        require(file_pin(root / relative) == expected, 'Exact unchanged retained factory source required')
    for binding in ARCHITECTURE_SOURCES.values():
        require(file_pin(root / binding['path'])['sha256'] == binding['sha256'],
                'Exact unchanged retained architecture source required')
    executable = shutil.which('nvidia-smi')
    require(executable is not None, 'Bounded GPU-name probe required before framework import')
    _ordinary(executable)
    probe = subprocess.run([executable, '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=6, check=False)
    require(probe.returncode == 0 and not probe.stderr.strip(), 'Finite GPU-name probe failed')
    names = tuple(line.strip() for line in probe.stdout.splitlines() if line.strip())
    validate_manual_context(system=platform.system(), root=str(root), home=str(Path.home()),
                            cwd=str(Path.cwd()), prefix=str(Path(sys.prefix)), device='cuda:0',
                            tmux=os.environ.get('TMUX', ''), gpu_names=names)
    return root


@contextmanager
def _preserve_runtime_rng(torch):
    cpu = torch.get_rng_state().clone()
    cuda = [value.clone() for value in torch.cuda.get_rng_state_all()]
    receipt = {'CPU_restored_exact': False, 'CUDA_restored_exact': False}
    try:
        yield receipt
    finally:
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)
        receipt['CPU_restored_exact'] = bool(torch.equal(cpu, torch.get_rng_state()))
        observed = torch.cuda.get_rng_state_all()
        receipt['CUDA_restored_exact'] = len(cuda) == len(observed) and all(
            torch.equal(before, after) for before, after in zip(cuda, observed))
        require(receipt['CPU_restored_exact'] and receipt['CUDA_restored_exact'],
                'Constructor must restore actual CPU and CUDA RNG exactly')


def _tensor_state_sha256(state, *, torch):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and
                torch.isfinite(value).all().item(), 'Finite actual FP32 tensor state required')
        detached = value.detach().cpu().contiguous()
        header = json.dumps([name, 'torch.float32', list(detached.shape)],
                            separators=(',', ':')).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def architecture_contract():
    """Source scope only; no numerical parity, gradient or quality certification."""
    expected_adapter_shapes()
    expected_head_shapes()
    return {
        'status': STATUS, 'initializer': 'Exact original trained-u64 complete575; no head reset',
        'artifacts': {name: dict(binding) for name, binding in ARTIFACTS.items()},
        'retained_factory_sources': {name: dict(binding) for name, binding in RETAINED_FACTORY_SOURCES.items()},
        'R2_canonical515_sha256': R2_CANONICAL_SHA256,
        'U96_canonical24_sha256': U96_CANONICAL_SHA256,
        'head_canonical36_sha256': TERMINAL_HEADS_CANONICAL_SHA256,
        'complete575_canonical_sha256': FULL_STATE_CANONICAL_SHA256,
        'arms': {arm: arm_census(arm) for arm in ('A', 'B')},
        'sites': list(SITES), 'width': WIDTH,
        'U96_pre_SFT': 'used=e+up(SiLU(depthwise(SiLU(down(e)))))',
        'stock': 'z=stock_SFT(used,d,w=1)',
        'head': 'h=output(GELU(spatial(GELU((E(used)+D(d)+Z(z)+resize(C(coarse32_used)))/2))))',
        'post_SFT': 'z_new=z+tanh(h)*(z-d)',
        'conditioning': conditioning.conditioning_contract(),
        'head_encoder_branch': {'A': 'used.detach()', 'B': 'used'},
        'sft_distillation_branch': {'A': 'used.detach()', 'B': 'used'},
        'coarse32_context_detached_both_arms': True,
        'stock_observed_encoder_detach_unchanged': True,
        'decoder_and_fused_graph_not_detached': True,
        'stock_token_path_and_w1_detach16_AdaIN_policy_unchanged': True,
        'base_frozen_eval_both_arms': True, 'adapters_eval_both_arms': True,
        'new_layers_or_token_training': False,
        'forward_math_intended_identical_at_zero_update': True,
        'zero_update_parity_measured': False, 'gradients_measured': False,
        'manual_finite_L4_runtime_required': True, 'quality_or_training_verified': False,
    }


def build_restorer(r2_base, u96_adapter_state, u64_head_state, *, arm,
                  r2_artifact, u96_artifact, head_artifact):
    """Construct only in the exact future manual Linux/tmux/L4 staged runtime."""
    root = _guard_manual_vm()
    census = arm_census(arm)
    paths = {'r2': r2_artifact, 'u96': u96_artifact, 'heads': head_artifact}
    for name, value in paths.items():
        path = _ordinary(value)
        require(path == root / ARTIFACTS[name]['path'] and
                file_pin(path) == {key: ARTIFACTS[name][key] for key in ('bytes', 'sha256')},
                'Exact separate staged R2/U96/original-trained-u64 initializer artifacts required')

    import torch
    from torch.nn import functional as F
    from .graph import inspect_adapter_reachability

    require(torch.cuda.is_initialized() and torch.cuda.is_available() and
            torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == 'NVIDIA L4' and
            torch.cuda.current_device() == 0,
            'Supplied base must already belong to the sole guarded NVIDIA L4')
    require(torch.get_default_dtype() == torch.float32 and not torch.is_inference_mode_enabled(),
            'Ordinary FP32 autograd-capable runtime required')
    require(isinstance(r2_base, torch.nn.Module) and
            type(r2_base).__module__ == 'third_party.codeformer.codeformer_arch' and
            type(r2_base).__name__ == 'CodeFormer', 'Exact supplied raw R2 CodeFormer class required')
    for name, expected in ARCHITECTURE_SOURCES.items():
        module = sys.modules.get(name)
        require(module is not None and getattr(module, '__file__', None),
                'Retained architecture module must already be loaded: ' + name)
        path = _ordinary(module.__file__)
        require(path == root / expected['path'] and file_pin(path)['sha256'] == expected['sha256'],
                'Exact retained loaded module origin and source bytes required: ' + name)
    require(r2_base.connect_list == list(SITES) and set(r2_base.fuse_convs_dict) == set(SITES) and
            r2_base.fuse_generator_block == {'16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21},
            'Exact retained four-site SFT CodeFormer architecture required')
    require(all(not module._forward_pre_hooks and not module._forward_hooks
                for module in r2_base.modules()), 'Base must have a clean hook namespace')
    initial = r2_base.state_dict()
    base_named = dict(r2_base.named_parameters())
    require(len(initial) == len(base_named) == BASE_TENSORS and set(initial) == set(base_named),
            'Exactly515 supplied R2 parameter tensors without extra buffers required')
    device = torch.device('cuda:0')
    require(all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and
                value.device == device and torch.isfinite(value).all().item()
                for value in initial.values()), 'Finite colocated CUDA:0 FP32 R2 tensors required')
    require(all(value.grad is None for value in base_named.values()), 'No retained R2 gradient fields allowed')
    base_shapes = {name: tuple(value.shape) for name, value in initial.items()}
    adapter_shapes = expected_adapter_shapes()
    head_shapes = expected_head_shapes()

    def check_supplied_state(state, schema, label):
        require(type(state) is dict and set(state) == set(schema),
                'Exact explicit ' + label + ' state dictionary required')
        require(all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and
                    tuple(value.shape) == schema[name] and value.device.type in ('cpu', 'cuda') and
                    not value.requires_grad and value.grad is None and torch.isfinite(value).all().item()
                    for name, value in state.items()), 'Finite detached exact FP32 ' + label + ' schema required')

    check_supplied_state(u96_adapter_state, adapter_shapes, '24-key U96 adapter')
    check_supplied_state(u64_head_state, head_shapes, '36-key retained trained-u64 head')
    require(_tensor_state_sha256(initial, torch=torch) == R2_CANONICAL_SHA256,
            'Exact canonical515 R2 content required')
    require(_tensor_state_sha256(u96_adapter_state, torch=torch) == U96_CANONICAL_SHA256,
            'Exact canonical24 U96 content required; partial or other adapter state refused')
    require(_tensor_state_sha256(u64_head_state, torch=torch) == TERMINAL_HEADS_CANONICAL_SHA256,
            'Exact original trained-u64 canonical36 required; reset/failed-new heads refused')
    train_adapters = arm == 'B'
    complete_shapes = {'base.' + name: shape for name, shape in base_shapes.items()}
    complete_shapes.update(adapter_shapes)
    complete_shapes.update(head_shapes)
    trainable_names = set(head_shapes) | (set(adapter_shapes) if train_adapters else set())
    frozen_names = set(complete_shapes) - trainable_names

    class ResidualSkipAdapter(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            options = {'device': 'cpu', 'dtype': torch.float32}
            self.down = torch.nn.Conv2d(channels, WIDTH, 1, bias=True, **options)
            self.depthwise = torch.nn.Conv2d(WIDTH, WIDTH, 3, padding=1, groups=WIDTH, bias=True, **options)
            self.up = torch.nn.Conv2d(WIDTH, channels, 1, bias=True, **options)
            self.activation = torch.nn.SiLU(inplace=False)

        def forward(self, observed):
            return observed + self.up(self.activation(self.depthwise(self.activation(self.down(observed)))))

    class SpatialHead(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            options = {'device': 'cpu', 'dtype': torch.float32}
            self.encoder = torch.nn.Conv2d(channels, WIDTH, 1, bias=True, **options)
            self.decoder = torch.nn.Conv2d(channels, WIDTH, 1, bias=False, **options)
            self.fused = torch.nn.Conv2d(channels, WIDTH, 1, bias=False, **options)
            self.coarse = torch.nn.Conv2d(256, WIDTH, 1, bias=False, **options)
            self.spatial = torch.nn.Conv2d(WIDTH, WIDTH, 3, padding=1, bias=True, **options)
            self.output = torch.nn.Conv2d(WIDTH, channels, 1, bias=True, **options)
            self.activation = torch.nn.GELU()

        def forward(self, used, decoder, fused, coarse, *, encoder_graph_witness):
            context = self.coarse(coarse.detach())
            context = F.interpolate(context, size=used.shape[-2:], mode='bilinear', align_corners=False)
            observed = used if train_adapters else used.detach()
            encoded = self.encoder(observed)
            encoder_graph_witness(encoded.grad_fn)
            mixed = (encoded + self.decoder(decoder) + self.fused(fused) + context) / 2.
            logits = self.output(self.activation(self.spatial(self.activation(mixed))))
            return fused + torch.tanh(logits) * (fused - decoder)

    class ConditionedSFTLearningRestorer(torch.nn.Module):
        def __init__(self, frozen_base):
            super().__init__()
            self.base = frozen_base
            self.adapters = torch.nn.ModuleDict({site: ResidualSkipAdapter(CHANNELS[site]) for site in SITES})
            self.heads = torch.nn.ModuleDict({site: SpatialHead(CHANNELS[site]) for site in SITES})
            self.adapters.load_state_dict({name.removeprefix('adapters.'): value
                                           for name, value in u96_adapter_state.items()}, strict=True)
            self.heads.load_state_dict({name.removeprefix('heads.'): value
                                        for name, value in u64_head_state.items()}, strict=True)
            self.adapters.to(device=device, dtype=torch.float32).eval().requires_grad_(train_adapters)
            self.heads.to(device=device, dtype=torch.float32).requires_grad_(True)
            self.base.eval().requires_grad_(False)
            self.arm = arm
            self._forward_lock = threading.Lock()
            self.last_forward_sites = ()
            self.last_base_calls = 0
            self.last_graph_witness = {'status': 'UNWITNESSED_NO_FORWARD', 'complete': False, 'sites': {}}
            self.last_sft_features = {}
            named = dict(self.named_parameters())
            self._initial_trainable = {
                name: named[name].detach().clone()
                for name in sorted(trainable_names)
            }

        def train(self, mode=True):
            super().train(mode)
            self.base.eval()
            self.adapters.eval()
            return self

        def clear_sft_features(self):
            self.last_sft_features = {}

        def max_trainable_displacement(self):
            named = dict(self.named_parameters())
            return max(
                float((named[name].detach() - ref).abs().max().item())
                for name, ref in self._initial_trainable.items()
            )

        def conditioning_gamma(self, *, force_active=False):
            if force_active:
                return 1.0
            return conditioning.displacement_gamma(self.max_trainable_displacement())

        def validate_invariants(self, *, check_frozen_digests=False):
            require(type(check_frozen_digests) is bool, 'Explicit frozen digest-check boolean required')
            _assert_manual_location(root)
            require(self.arm == arm, 'Matched arm is immutable after construction')
            named = dict(self.named_parameters())
            state = self.state_dict()
            require(len(named) == len(state) == TOTAL_TENSORS and
                    set(named) == set(state) == set(complete_shapes) and
                    len(self.base.state_dict()) == BASE_TENSORS and
                    {name for name, value in named.items() if value.requires_grad} == trainable_names,
                    'Exact575 full keys and arm-specific trainable/frozen permissions required')
            require(all(isinstance(value, torch.Tensor) and tuple(value.shape) == complete_shapes[name] and
                        value.dtype == torch.float32 and value.device == device and
                        torch.isfinite(value).all().item() for name, value in state.items()),
                    'All575 tensors must retain exact finite colocated FP32 schemas')
            require(sum(named[name].numel() for name in trainable_names) == census['trainable_parameters'] and
                    len(trainable_names) == census['trainable_tensors'] and
                    len(frozen_names) == census['frozen_tensors'] and
                    all(not named[name].requires_grad and named[name].grad is None for name in frozen_names),
                    'Arm-specific exact trainable coordinate census and frozen grad=None required')
            require(all(not module.training for module in self.base.modules()) and
                    all(not module.training for module in self.adapters.modules()),
                    'R2 base and U96 adapters must stay eval in both arms')
            require(all(value.grad is None or (value.grad.dtype == torch.float32 and
                        value.grad.device == device and tuple(value.grad.shape) == tuple(value.shape) and
                        torch.isfinite(value.grad).all().item())
                        for name, value in named.items() if name in trainable_names),
                    'Any retained trainable gradient must have exact finite FP32 schema')
            digests = None
            if check_frozen_digests:
                digests = {'R2': _tensor_state_sha256(self.base.state_dict(), torch=torch)}
                require(digests['R2'] == R2_CANONICAL_SHA256, 'Frozen515 R2 content changed')
                if not train_adapters:
                    digests['U96'] = _tensor_state_sha256(
                        {'adapters.' + name: value for name, value in self.adapters.state_dict().items()},
                        torch=torch)
                    require(digests['U96'] == U96_CANONICAL_SHA256, 'Arm A frozen24 U96 content changed')
            return {**arm_census(arm), 'pass': True,
                    'frozen_parameters': sum(named[name].numel() for name in frozen_names),
                    'trainable_names': sorted(trainable_names), 'frozen_names': sorted(frozen_names),
                    'all575_finite_FP32_schema_checked': True,
                    'frozen_digests_checked': check_frozen_digests, 'digests': digests,
                    'runtime_integrity_is_not_parity_gradient_or_quality': True}

        def capture_complete_state_for_audit(self):
            _assert_manual_location(root)
            require(self._forward_lock.acquire(blocking=False), 'No capture during concurrent/nested forward')
            try:
                self.validate_invariants(check_frozen_digests=True)
                require(all(not module._forward_pre_hooks and not module._forward_hooks
                            for module in self.modules()), 'Capture requires a clean hook namespace')
                live = self.state_dict()
                before = _tensor_state_sha256(live, torch=torch)
                captured = {name: value.detach().cpu().clone().contiguous() for name, value in live.items()}
                require(len(captured) == TOTAL_TENSORS and set(captured) == set(complete_shapes) and
                        all(not value.requires_grad and value.grad is None and value.device.type == 'cpu'
                            for value in captured.values()) and
                        before == _tensor_state_sha256(captured, torch=torch) ==
                        _tensor_state_sha256(self.state_dict(), torch=torch),
                        'All actual575 captured tensors must bind to unchanged live complete state')
                return captured
            finally:
                self._forward_lock.release()

        def forward(self, sample, *, w=1., detach_16=True, adain=True, capture_sft_features=False):
            _assert_manual_location(root)
            require(w == 1. and detach_16 is True and adain is True,
                    'Fixed w1/detach16/AdaIN-on policy required')
            require(isinstance(sample, torch.Tensor) and sample.shape == (1, 3, 512, 512) and
                    sample.dtype == torch.float32 and sample.device == device and not sample.requires_grad and
                    torch.isfinite(sample).all().item() and
                    sample.min().item() >= -1. and sample.max().item() <= 1.,
                    'Finite singleton FP32 prepared512[-1,1] input required')
            require(not torch.is_inference_mode_enabled(),
                    'Autograd-capable ordinary runtime required; inference_mode refused')
            self.validate_invariants(check_frozen_digests=False)
            require(self._forward_lock.acquire(blocking=False), 'Nested/concurrent matched-arm forward refused')
            handles = []
            pending = {}
            completed = []
            captured_sft = {}
            coarse = None
            base_calls = 0
            thread_id = threading.get_ident()
            self.last_forward_sites = ()
            self.last_base_calls = 0
            self.last_sft_features = {}
            graph_enabled = torch.is_grad_enabled()
            graph_sites = {}
            self.last_graph_witness = {
                'status': 'INCOMPLETE' if graph_enabled else 'UNWITNESSED_NO_GRAD',
                'arm': arm, 'grad_enabled': graph_enabled, 'complete': False,
                'sites': graph_sites, 'structural_reachability_only': True,
                'nonzero_or_finite_branch_gradient_proven': False,
                'extra_neural_or_autograd_query_or_backward_calls': 0,
                'returned_tensor_or_node_references': False,
            }

            def guard_thread():
                require(threading.get_ident() == thread_id, 'Concurrent/foreign-thread base or SFT call refused')

            def base_pre_hook(module, args):
                nonlocal base_calls
                guard_thread()
                require(module is self.base and base_calls == 0 and len(args) == 1 and args[0] is sample,
                        'One explicit nonnested base call required')
                base_calls += 1

            def witness_branch(site, branch, node):
                guard_thread()
                records = graph_sites.setdefault(site, {})
                require(branch not in records, 'Exactly one witness per existing branch required')
                if not graph_enabled:
                    records[branch] = {'status': 'UNWITNESSED_NO_GRAD', 'pass': None,
                                       'structural_reachability_only': True,
                                       'nonzero_or_finite_branch_gradient_proven': False}
                    return
                try:
                    record = inspect_adapter_reachability(
                        node, dict(self.adapters[site].named_parameters()),
                        expected_present=train_adapters)
                except Exception as exc:
                    records[branch] = {'status': 'FAILED', 'pass': False, 'error': str(exc)}
                    self.last_graph_witness['status'] = 'FAILED'
                    raise
                records[branch] = {'status': 'WITNESSED', **record}

            def pre_hook(site):
                def adapt_and_capture(module, args):
                    nonlocal coarse
                    guard_thread()
                    require(module is self.base.fuse_convs_dict[site] and base_calls == 1 and
                            len(args) == 3 and args[2] == 1. and len(completed) < len(SITES) and
                            site == SITES[len(completed)] and not pending,
                            'Exactly one ordered U96 pre-SFT adaptation required')
                    encoder, decoder, _ = args
                    shape = (1, CHANNELS[site], int(site), int(site))
                    require(encoder.shape == decoder.shape == shape and
                            encoder.dtype == decoder.dtype == torch.float32 and
                            encoder.device == decoder.device == device and not encoder.requires_grad and
                            torch.isfinite(encoder).all().item() and torch.isfinite(decoder).all().item(),
                            'Exact stock-detached encoder and ordinary connected decoder required')
                    used = self.adapters[site](encoder.detach())
                    require(used.shape == shape and used.dtype == torch.float32 and used.device == device and
                            used.requires_grad == (train_adapters and torch.is_grad_enabled()) and
                            torch.isfinite(used).all().item(),
                            'Finite exact U96 result with arm-specific live adapter graph required')
                    if capture_sft_features:
                        captured_sft[site] = used if (train_adapters and torch.is_grad_enabled()) else used.detach().clone()
                    if site == '32':
                        require(coarse is None, 'Capture used coarse32 exactly once')
                        coarse = used.detach()
                    require(coarse is not None and coarse.shape == (1, 256, 32, 32) and
                            not coarse.requires_grad, 'Detached first-site used coarse32 context required')
                    pending[site] = (used, decoder)
                    return used, decoder, args[2]
                return adapt_and_capture

            def post_hook(site):
                def correct(module, args, fused):
                    guard_thread()
                    require(module is self.base.fuse_convs_dict[site] and set(pending) == {site} and
                            len(completed) < len(SITES) and site == SITES[len(completed)],
                            'Matching ordered U96 SFT capture required')
                    used, decoder = pending.pop(site)
                    require(len(args) == 3 and args[0] is used and args[1] is decoder and args[2] == 1. and
                            isinstance(fused, torch.Tensor) and fused.shape == used.shape and
                            fused.dtype == torch.float32 and fused.device == device and
                            torch.isfinite(fused).all().item() and
                            (not (train_adapters and torch.is_grad_enabled()) or fused.requires_grad),
                            'Finite stock fused result with live Arm B SFT graph required')
                    witness_branch(site, 'stock_SFT', fused.grad_fn)
                    corrected = self.heads[site](
                        used, decoder, fused, coarse,
                        encoder_graph_witness=lambda node: witness_branch(site, 'head_encoder', node))
                    require(corrected.shape == fused.shape and corrected.dtype == torch.float32 and
                            corrected.device == device and torch.isfinite(corrected).all().item(),
                            'Finite unchanged-geometry post-fusion correction required')
                    completed.append(site)
                    return corrected
                return correct

            try:
                require(all(not module._forward_pre_hooks and not module._forward_hooks
                            for module in self.modules()), 'Clean matched candidate hook namespace required')
                handles.append(self.base.register_forward_pre_hook(base_pre_hook))
                for site in SITES:
                    module = self.base.fuse_convs_dict[site]
                    handles.append(module.register_forward_pre_hook(pre_hook(site)))
                    handles.append(module.register_forward_hook(post_hook(site)))
                with torch.autocast(device_type=device.type, enabled=False):
                    result = self.base(sample, w=w, detach_16=detach_16, adain=adain)
                require(base_calls == 1 and tuple(completed) == SITES and not pending,
                        'All four adaptations/corrections must execute in one complete base call')
                require(isinstance(result, tuple) and len(result) == 3 and
                        result[0].shape == (1, 3, 512, 512) and result[0].dtype == torch.float32 and
                        result[0].device == device and torch.isfinite(result[0]).all().item(),
                        'Finite native512 plus unchanged stock logits/LQ return structure required')
                require(set(graph_sites) == set(SITES) and all(
                    set(records) == {'stock_SFT', 'head_encoder'} for records in graph_sites.values()),
                    'Every existing SFT and head-encoder branch must receive one witness record')
                if capture_sft_features:
                    require(set(captured_sft) == set(SITES), 'All 4 SFT sites must be captured')
                    self.last_sft_features = dict(captured_sft)
                self.last_graph_witness['complete'] = True
                self.last_graph_witness['status'] = 'WITNESSED' if graph_enabled else 'UNWITNESSED_NO_GRAD'
                return result
            finally:
                try:
                    for handle in reversed(handles):
                        handle.remove()
                    pending.clear()
                    coarse = None
                    self.last_forward_sites = tuple(completed)
                    self.last_base_calls = base_calls
                finally:
                    self._forward_lock.release()

    with _OWNERS_LOCK:
        owner = _OWNERS.get(r2_base)
        shared_owner = getattr(r2_base, _OWNER_ATTRIBUTE, None)
        require((owner is None or owner() is None) and
                (shared_owner is None or isinstance(shared_owner, weakref.ReferenceType) and shared_owner() is None),
                'R2 base already has a live exclusive wrapper owner')
        with _preserve_runtime_rng(torch) as rng_receipt:
            torch.random.default_generator.manual_seed(INITIALIZATION_SEED)
            candidate = ConditionedSFTLearningRestorer(r2_base).eval()
            candidate.validate_invariants(check_frozen_digests=True)
            require(all(value.grad is None for value in candidate.parameters()),
                    'Initialized candidate must have no gradient fields')
            loaded_adapters = {'adapters.' + name: value for name, value in candidate.adapters.state_dict().items()}
            loaded_heads = {'heads.' + name: value for name, value in candidate.heads.state_dict().items()}
            require(_tensor_state_sha256(loaded_adapters, torch=torch) == U96_CANONICAL_SHA256 and
                    _tensor_state_sha256(loaded_heads, torch=torch) == TERMINAL_HEADS_CANONICAL_SHA256,
                    'All loaded U96 and original learned-u64 tensors must match exactly')
            complete_sha = _tensor_state_sha256(candidate.state_dict(), torch=torch)
            require(complete_sha == FULL_STATE_CANONICAL_SHA256,
                    'Both arms must initialize to the exact retained trained-u64 complete575 state')
        candidate.matched_initializer_receipt = {
            **arm_census(arm), 'source_status': STATUS,
            'terminal_head_canonical_sha256': TERMINAL_HEADS_CANONICAL_SHA256,
            'complete575_canonical_sha256': complete_sha,
            'U96_adapter_canonical_sha256': U96_CANONICAL_SHA256,
            'R2_canonical_sha256': R2_CANONICAL_SHA256,
            'learned_heads_loaded_directly': True, 'learned_heads_zero_reset': False,
            'terminal36_and_initial24_readback_exact': True,
            'CPU_CUDA_constructor_RNG': dict(rng_receipt),
            'optimizer_state_loaded': False, 'neural_calls': 0,
            'quality_or_parity_or_gradient_verified': False,
        }
        _OWNERS[r2_base] = weakref.ref(candidate)
        setattr(r2_base, _OWNER_ATTRIBUTE, weakref.ref(candidate))
    return candidate
