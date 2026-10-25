"""UNRUN_PROTOTYPE_NOT_PACKET_TRAINED: existing U96 final residual block22 plus RGB24.

Importing this module performs no framework, model, gradient or optimizer work.
The guarded factory is prospective source for a later independently reviewed,
finite manual L4 packet. It does not constitute that packet or establish parity.
There is deliberately no optimizer, checkpoint loader, trainer or entry point.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import threading
import weakref


STATUS = 'UNRUN_PROTOTYPE_NOT_PACKET_TRAINED'
VM_NAME = 'cctv_codeformer_u96_spatial_feasibility_v1_vm'
MANUAL_ACKNOWLEDGEMENT = 'REVIEWED_FINITE_MANUAL_L4_WORKER_REQUIRED'
ROOT = Path(__file__).resolve().parents[2]
SITES = ('32', '64', '128', '256')
CHANNELS = {'32': 256, '64': 256, '128': 128, '256': 128}
R2_FILE_SHA256 = 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
R2_CANONICAL_SHA256 = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_FILE_SHA256 = '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'
U96_CANONICAL_SHA256 = 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
ARCHITECTURE_SOURCES = {
    'third_party.codeformer.codeformer_arch': (
        'third_party/codeformer/codeformer_arch.py',
        'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb'),
    'third_party.codeformer.vqgan_arch': (
        'third_party/codeformer/vqgan_arch.py',
        'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf'),
    'third_party.codeformer.registry': (
        'third_party/codeformer/registry.py',
        'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e'),
}
BASE_TENSORS = 515
ADAPTER_TENSORS = 24
ADAPTER_PARAMETERS = 51328
TOTAL_TENSORS = 539
FROZEN_TENSORS = 529
TRAINABLE_TENSORS = 10
RESIDUAL_PARAMETERS = 74112
READOUT_PARAMETERS = 1731
TRAINABLE_PARAMETERS = RESIDUAL_PARAMETERS + READOUT_PARAMETERS
_OWNERS = weakref.WeakKeyDictionary()
_OWNERS_LOCK = threading.Lock()
_OWNER_ATTRIBUTE = '_forensic_codeformer_exclusive_owner'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_readout_shapes():
    """Names of the two EXISTING final Conv2d parameters, no new neural head."""
    return {'base.generator.blocks.24.weight': (3, 64, 3, 3),
            'base.generator.blocks.24.bias': (3,)}



def expected_spatial_shapes():
    """Only original ResBlock22 internals plus original RGB24, no new head."""
    shapes = {}
    for norm in ('norm1', 'norm2'):
        shapes['base.generator.blocks.22.' + norm + '.weight'] = (64,)
        shapes['base.generator.blocks.22.' + norm + '.bias'] = (64,)
    for conv in ('conv1', 'conv2'):
        shapes['base.generator.blocks.22.' + conv + '.weight'] = (64, 64, 3, 3)
        shapes['base.generator.blocks.22.' + conv + '.bias'] = (64,)
    shapes.update(expected_readout_shapes())
    require(len(shapes) == TRAINABLE_TENSORS and
            sum(math.prod(shape) for shape in shapes.values()) == TRAINABLE_PARAMETERS,
            'Exact existing block22 plus RGB24 parameter schema differs')
    return shapes


def expected_adapter_shapes():
    result = {}
    for site, channels in CHANNELS.items():
        for name, shape in {
            'down.weight': (32, channels, 1, 1), 'down.bias': (32,),
            'depthwise.weight': (32, 1, 3, 3), 'depthwise.bias': (32,),
            'up.weight': (channels, 32, 1, 1), 'up.bias': (channels,),
        }.items():
            result['adapters.' + site + '.' + name] = shape
    require(len(result) == ADAPTER_TENSORS and
            sum(math.prod(shape) for shape in result.values()) == ADAPTER_PARAMETERS,
            'Exact retained U96 adapter schema differs')
    return result


def architecture_contract():
    """Static prospective scope; every numerical/runtime claim stays unproven."""
    return {
        'status': STATUS, 'format': 'complete-U96-existing-final-residual22-RGB24-feasibility-v1',
        'initializer': {'R2_file_sha256': R2_FILE_SHA256,
                        'R2_canonical515_sha256': R2_CANONICAL_SHA256,
                        'U96_file_sha256': U96_FILE_SHA256,
                        'U96_canonical24_sha256': U96_CANONICAL_SHA256},
        'base_tensors': BASE_TENSORS, 'adapter_tensors': ADAPTER_TENSORS,
        'adapter_parameters': ADAPTER_PARAMETERS, 'total_tensors': TOTAL_TENSORS,
        'frozen_tensors': FROZEN_TENSORS, 'trainable_tensors': TRAINABLE_TENSORS,
        'trainable_parameters': TRAINABLE_PARAMETERS,
        'trainable_shapes': {name: list(shape) for name, shape in expected_spatial_shapes().items()},
        'RGB_readout_control_shapes': {name: list(shape) for name, shape in expected_readout_shapes().items()},
        'existing_residual_block': 22, 'existing_outer_normalization_frozen': 23,
        'existing_RGB_readout': 24, 'new_architecture_modules': False,
        'complete_state_capture': 'Explicit locked detached CPU clone of all539 tensors; no file write',
        'new_head': False, 'all_modules_eval': True,
        'same_four_frozen_pre_SFT_adapters': True,
        'stock_forward': {'w': 1.0, 'adain': True, 'detach_16': True, 'code_only': False},
        'input': {'shape': [1, 3, 512, 512], 'dtype': 'float32', 'range': [-1, 1]},
        'connected_delivery': 'native512 -> (native+1)/2 -> clamp[0,1] -> bilinear256',
        'delivery_floor_or_ST_estimator': False,
        'CPU_constructor_RNG_restored': True, 'historical_optimizer_loaded': False,
        'initial_RAW_PNG_logits_LQ_parity': 'UNMEASURED_REQUIRES_MANUAL_VM',
        'gradient_or_optimizer_or_quality_verified': False,
        'finite_worker_protocol_and_independent_novelty_review': 'REQUIRED_NOT_INCLUDED',
        'quality_gates_changed': False,
    }


def validate_manual_context(*, system, root, home, device, tmux, gpu_names, acknowledgement):
    """Pure context schema, exercised without filesystem or subprocess access."""
    require(system == 'Linux', 'Manual Linux VM only; local/app model execution refused')
    expected = PurePosixPath(home) / 'forensic-dgp' / VM_NAME
    require(PurePosixPath(root).is_absolute() and str(root) == str(expected) and
            '..' not in PurePosixPath(root).parts, 'Exact manual VM root required')
    require(device == 'cuda:0', 'Only explicit cuda:0 runtime permitted')
    require(isinstance(tmux, str) and bool(tmux.strip()), 'Manual tmux required')
    require(type(gpu_names) is tuple and gpu_names == ('NVIDIA L4',),
            'Exactly one NVIDIA L4 required before framework import')
    require(acknowledgement == MANUAL_ACKNOWLEDGEMENT,
            'Explicit reviewed finite manual-worker acknowledgement required')
    return True


def _regular_file(path):
    path = Path(path).absolute()
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
            not getattr(info, 'st_file_attributes', 0) & 1024,
            'Regular nonlinked file required')
    for parent in path.parents:
        state = parent.lstat()
        require(stat.S_ISDIR(state.st_mode) and not stat.S_ISLNK(state.st_mode) and
                not getattr(state, 'st_file_attributes', 0) & 1024,
                'Linked or reparse file ancestry refused')
    return path


def _file_digest(path):
    path = _regular_file(path)
    require(0 < path.stat().st_size < 1024 ** 3, 'Bounded nonempty artifact required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_root(manual_vm_root):
    require(platform.system() == 'Linux', 'Manual Linux VM only; no local/app execution')
    root = Path(manual_vm_root).absolute()
    expected = Path.home() / 'forensic-dgp' / VM_NAME
    require(str(root) == str(expected) and root.resolve(strict=True) == ROOT == expected.resolve(strict=True),
            'Prototype source must live in the exact fresh manual VM root')
    for parent in (root, *root.parents):
        info = parent.lstat()
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
                not getattr(info, 'st_file_attributes', 0) & 1024,
                'Linked runtime directory refused')
    require(bool(os.environ.get('TMUX', '').strip()), 'Explicit manual tmux runtime required')
    return root


def _guard_manual_vm(manual_vm_root, device, acknowledgement, prototype_source_sha256):
    """All these standard-library guards execute BEFORE the sole Torch import."""
    root = _bound_root(manual_vm_root)
    require(device == 'cuda:0' and acknowledgement == MANUAL_ACKNOWLEDGEMENT,
            'Exact explicit manual device/acknowledgement required')
    require(type(prototype_source_sha256) is str and
            re.fullmatch('[0-9a-f]{64}', prototype_source_sha256) and
            _file_digest(__file__) == prototype_source_sha256,
            'Future sealed-worker prototype source pin required')
    for relative, known_sha in ARCHITECTURE_SOURCES.values():
        require(_file_digest(root / relative) == known_sha, 'Retained architecture source differs')
    executable = shutil.which('nvidia-smi')
    require(executable is not None, 'nvidia-smi is required before framework import')
    _regular_file(executable)
    probe = subprocess.run([executable, '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=6, check=False)
    require(probe.returncode == 0 and not probe.stderr.strip(), 'Finite GPU-name probe failed')
    names = tuple(line.strip() for line in probe.stdout.splitlines() if line.strip())
    validate_manual_context(system=platform.system(), root=str(root), home=str(Path.home()),
                            device=device, tmux=os.environ.get('TMUX', ''), gpu_names=names,
                            acknowledgement=acknowledgement)
    return root


def _actual_device(torch):
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == 'NVIDIA L4', 'Actual single Torch NVIDIA L4 required')
    require(torch.cuda.current_device() == 0, 'Actual current CUDA device must be zero')
    return torch.device('cuda:0')


def connected_rgb256(native, *, torch, manual_vm_root):
    """Prospective differentiable actual RAW delivery; no floor or detach surrogate.

    The caller supplies Torch explicitly. Even this helper refuses local/app use.
    Clamp saturation and PNG quantization still need separate reported evidence.
    """
    _bound_root(manual_vm_root)
    device = _actual_device(torch)
    require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
            'Ordinary connected autograd runtime required')
    require(isinstance(native, torch.Tensor) and native.shape == (1, 3, 512, 512) and
            native.device == device and native.dtype == torch.float32 and
            native.requires_grad and native.grad_fn is not None and torch.isfinite(native).all().item(),
            'Finite connected FP32 native readout required')
    rgb = ((native + 1.0) / 2.0).clamp(0.0, 1.0)
    raw = torch.nn.functional.interpolate(rgb, (256, 256), mode='bilinear', align_corners=False)
    require(raw.shape == (1, 3, 256, 256) and raw.dtype == torch.float32 and
            raw.requires_grad and raw.grad_fn is not None and torch.isfinite(raw).all().item(),
            'Connected finite FP32 delivered256 required')
    return raw


def build_restorer(r2_base, u96_adapter_state, *, r2_artifact, u96_artifact,
                   manual_vm_root, prototype_source_sha256,
                   device='cuda:0', acknowledgement=None):
    """UNRUN factory, not a training command or an independently approved packet."""
    root = _guard_manual_vm(manual_vm_root, device, acknowledgement, prototype_source_sha256)
    import torch

    target = _actual_device(torch)
    require(not torch.is_inference_mode_enabled() and torch.is_grad_enabled(),
            'Factory requires ordinary autograd-capable manual runtime')
    require(isinstance(r2_base, torch.nn.Module) and
            type(r2_base).__module__ == 'third_party.codeformer.codeformer_arch' and
            type(r2_base).__name__ == 'CodeFormer', 'Exact raw supplied R2 class required')
    for name, (relative, known_sha) in ARCHITECTURE_SOURCES.items():
        module = sys.modules.get(name)
        require(module is not None and getattr(module, '__file__', None) and
                Path(module.__file__).resolve() == (root / relative).resolve() and
                _file_digest(module.__file__) == known_sha, 'Actual loaded architecture origin differs')
    require(_file_digest(r2_artifact) == R2_FILE_SHA256 and
            _file_digest(u96_artifact) == U96_FILE_SHA256, 'Complete exact initializer artifacts required')
    require(r2_base.connect_list == list(SITES) and set(r2_base.fuse_convs_dict) == set(SITES) and
            r2_base.fuse_generator_block == {'16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21},
            'Exact stock four-SFT-site architecture required')
    require(len(r2_base.generator.blocks) == 25, 'Existing final generator block24 required')
    readout = r2_base.generator.blocks[24]
    require(type(readout) is torch.nn.Conv2d and readout.in_channels == 64 and readout.out_channels == 3 and
            readout.kernel_size == (3, 3) and readout.stride == (1, 1) and readout.padding == (1, 1) and
            readout.dilation == (1, 1) and readout.groups == 1 and readout.padding_mode == 'zeros' and
            readout.bias is not None, 'Existing exact RGB Conv2d required, no replacement head')
    spatial = r2_base.generator.blocks[22]
    final_normalization = r2_base.generator.blocks[23]
    require(type(spatial).__module__ == 'third_party.codeformer.vqgan_arch' and
            type(spatial).__name__ == 'ResBlock' and spatial.in_channels == spatial.out_channels == 64 and
            set(spatial._modules) == {'norm1', 'conv1', 'norm2', 'conv2'} and
            not hasattr(spatial, 'conv_out'),
            'Exact existing64-to64 identity-skip ResBlock22 required')
    for norm in (spatial.norm1, spatial.norm2, final_normalization):
        require(type(norm) is torch.nn.GroupNorm and norm.num_groups == 32 and
                norm.num_channels == 64 and norm.eps == 1e-6 and norm.affine is True and
                norm.weight is not None and norm.bias is not None,
                'Exact existing GroupNorm32x64 eps1e-6 affine layers required')
    for conv in (spatial.conv1, spatial.conv2):
        require(type(conv) is torch.nn.Conv2d and conv.in_channels == conv.out_channels == 64 and
                conv.kernel_size == (3, 3) and conv.stride == (1, 1) and conv.padding == (1, 1) and
                conv.dilation == (1, 1) and conv.groups == 1 and conv.padding_mode == 'zeros' and
                conv.bias is not None, 'Exact existing ResBlock22 convolutions required')
    require(all(not module._forward_pre_hooks and not module._forward_hooks for module in r2_base.modules()),
            'Fresh clean raw-base hook namespace required')
    initial = r2_base.state_dict()
    require(len(initial) == len(dict(r2_base.named_parameters())) == BASE_TENSORS,
            'Complete exact515 R2 tensors required')
    require(all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and value.device == target and
                torch.isfinite(value).all().item() for value in initial.values()) and
            all(value.grad is None for value in r2_base.parameters()),
            'Fresh finite FP32 CUDA R2 base with no gradient fields required; cached CPU/app base refused')
    adapter_schema = expected_adapter_shapes()
    require(type(u96_adapter_state) is dict and set(u96_adapter_state) == set(adapter_schema) and
            all(isinstance(value, torch.Tensor) and tuple(value.shape) == adapter_schema[name] and
                value.dtype == torch.float32 and value.device.type in ('cpu', 'cuda') and
                not value.requires_grad and value.grad is None and torch.isfinite(value).all().item()
                for name, value in u96_adapter_state.items()), 'Exact finite detached24-key U96 state required')

    def state_digest(state):
        digest = hashlib.sha256()
        for name, value in sorted(state.items()):
            value = value.detach().cpu().contiguous()
            header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode('utf-8')
            digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    require(state_digest(initial) == R2_CANONICAL_SHA256 and
            state_digest(u96_adapter_state) == U96_CANONICAL_SHA256,
            'Complete exact R2 plus U96 canonical state required; partial/S96/wrong state refused')
    base_trainable_names = {name.removeprefix('base.') for name in expected_spatial_shapes()}
    initial_frozen_digest = state_digest({name: value for name, value in initial.items()
                                          if name not in base_trainable_names})

    class ResidualSkipAdapter(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            options = {'device': 'cpu', 'dtype': torch.float32}
            self.down = torch.nn.Conv2d(channels, 32, 1, bias=True, **options)
            self.depthwise = torch.nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=True, **options)
            self.up = torch.nn.Conv2d(32, channels, 1, bias=True, **options)
            self.activation = torch.nn.SiLU(inplace=False)

        def forward(self, observed):
            return observed + self.up(self.activation(self.depthwise(self.activation(self.down(observed)))))

    class U96SpatialFeasibility(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = r2_base
            self.adapters = torch.nn.ModuleDict({site: ResidualSkipAdapter(CHANNELS[site]) for site in SITES})
            self.adapters.load_state_dict({name.removeprefix('adapters.'): value
                                           for name, value in u96_adapter_state.items()}, strict=True)
            self.adapters.to(device=target, dtype=torch.float32)
            self.requires_grad_(False)
            self.base.generator.blocks[22].requires_grad_(True)
            self.base.generator.blocks[24].requires_grad_(True)
            self.eval()
            self._forward_lock = threading.Lock()
            self.last_forward_sites = (); self.last_base_calls = 0; self.last_readout_calls = 0
            self.last_spatial_calls = 0; self.last_normalization_calls = 0

        def train(self, mode=True):
            require(type(mode) is bool, 'Explicit boolean mode required')
            super().train(False)
            return self

        def validate_invariants(self, *, check_frozen_digests=True):
            require(type(check_frozen_digests) is bool, 'Explicit digest-check boolean required')
            named = dict(self.named_parameters()); schema = expected_spatial_shapes()
            trainable = {name: value for name, value in named.items() if value.requires_grad}
            adapters = {name: value for name, value in named.items() if name.startswith('adapters.')}
            frozen = {name: value for name, value in named.items() if not value.requires_grad}
            require(len(named) == len(self.state_dict()) == TOTAL_TENSORS and
                    len(self.base.state_dict()) == BASE_TENSORS and len(frozen) == FROZEN_TENSORS and
                    set(trainable) == set(schema) and
                    sum(value.numel() for value in trainable.values()) == TRAINABLE_PARAMETERS and
                    all(tuple(trainable[name].shape) == shape for name, shape in schema.items()) and
                    set(adapters) == set(adapter_schema) and
                    all(tuple(adapters[name].shape) == shape for name, shape in adapter_schema.items()),
                    'Exact539 tensors/529 frozen/ten existing block22+RGB24 tensors/75843 scalars required')
            require(all(value.dtype == torch.float32 and value.device == target and
                        torch.isfinite(value).all().item() for value in named.values()) and
                    all(value.grad is None for value in frozen.values()) and
                    all(not module.training for module in self.modules()),
                    'Every module stays eval; all tensors finite FP32 CUDA; frozen grads stay None')
            digests = None
            if check_frozen_digests:
                digests = {'R2_except_existing_block22_RGB24': state_digest({name: value for name, value
                           in self.base.state_dict().items() if name not in base_trainable_names}),
                           'U96': state_digest({'adapters.' + name: value
                                  for name, value in self.adapters.state_dict().items()})}
                require(digests == {'R2_except_existing_block22_RGB24': initial_frozen_digest,
                                    'U96': U96_CANONICAL_SHA256},
                        'Frozen state changed; preserve failure, never silently rebind')
            return {'pass': True, 'status': STATUS, 'total_tensors': TOTAL_TENSORS,
                    'frozen_tensors': FROZEN_TENSORS, 'trainable_tensors': TRAINABLE_TENSORS,
                    'trainable_parameters': TRAINABLE_PARAMETERS, 'digests': digests,
                    'parity_gradient_training_quality_verified': False}

        def capture_complete_state_for_audit(self):
            """Explicit complete539 CPU clone; no files, loading, or optimizer work.

            The manual worker must call this only at a prospectively declared
            snapshot boundary, with no outstanding graph/step on another thread.
            The lock rejects a simultaneous forward or another capture. Returned
            tensors are detached independent clones, never live parameter views.
            """
            _bound_root(root)
            require(self._forward_lock.acquire(blocking=False),
                    'Concurrent forward/state capture refused')
            try:
                self.validate_invariants(check_frozen_digests=True)
                captured = {name: value.detach().cpu().contiguous().clone()
                            for name, value in self.state_dict().items()}
                require(len(captured) == TOTAL_TENSORS and
                        set(captured) == set(dict(self.named_parameters())) and
                        all(value.dtype == torch.float32 and value.device.type == 'cpu' and
                            not value.requires_grad and value.grad_fn is None and
                            torch.isfinite(value).all().item() for value in captured.values()),
                        'Complete detached finite539-tensor CPU audit capture required')
                return captured
            finally:
                self._forward_lock.release()

        def forward(self, sample, *, w=1.0, detach_16=True, adain=True):
            require(type(w) in (int, float) and w == 1.0 and detach_16 is True and adain is True,
                    'Fixed stock w1/detach16/AdaIN policy required')
            require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
                    'Ordinary connected autograd mode required')
            require(isinstance(sample, torch.Tensor) and sample.shape == (1, 3, 512, 512) and
                    sample.dtype == torch.float32 and sample.device == target and not sample.requires_grad and
                    torch.isfinite(sample).all().item() and sample.min().item() >= -1.0 and
                    sample.max().item() <= 1.0, 'Finite singleton FP32 CUDA prepared512[-1,1] input required')
            require(self._forward_lock.acquire(blocking=False), 'Concurrent/nested wrapper forward refused')
            handles = []; completed = []; base_calls = 0; readout_calls = 0; spatial_calls = 0
            normalization_calls = 0; last_spatial = None; last_normalized = None; last_native = None
            thread_id = threading.get_ident()
            self.last_forward_sites = (); self.last_base_calls = 0; self.last_readout_calls = 0
            self.last_spatial_calls = 0; self.last_normalization_calls = 0

            def same_thread():
                require(threading.get_ident() == thread_id, 'Foreign-thread base execution refused')

            def base_guard(module, args):
                nonlocal base_calls
                same_thread()
                require(module is self.base and base_calls == 0 and len(args) == 1 and args[0] is sample,
                        'Exactly one explicit stock base call required')
                base_calls += 1

            def adapt(site):
                def pre_hook(module, args):
                    same_thread()
                    require(module is self.base.fuse_convs_dict[site] and base_calls == 1 and
                            len(completed) < len(SITES) and site == SITES[len(completed)] and
                            len(args) == 3 and args[2] == 1.0, 'Four ordered exact U96 pre-SFT calls required')
                    observed, decoder, strength = args
                    shape = (1, CHANNELS[site], int(site), int(site))
                    require(observed.shape == decoder.shape == shape and
                            observed.dtype == decoder.dtype == torch.float32 and
                            observed.device == decoder.device == target and not observed.requires_grad and
                            not decoder.requires_grad and torch.isfinite(observed).all().item() and
                            torch.isfinite(decoder).all().item(), 'Finite frozen encoder/decoder SFT inputs required')
                    used = self.adapters[site](observed.detach())
                    require(used.shape == shape and not used.requires_grad and torch.isfinite(used).all().item(),
                            'Exact frozen U96 adapter result required')
                    completed.append(site)
                    return used, decoder, strength
                return pre_hook

            def spatial_guard(module, args):
                nonlocal spatial_calls
                same_thread()
                require(module is spatial and base_calls == 1 and tuple(completed) == SITES and
                        spatial_calls == 0 and len(args) == 1 and args[0].shape == (1, 64, 512, 512) and
                        args[0].dtype == torch.float32 and args[0].device == target and
                        not args[0].requires_grad and torch.isfinite(args[0]).all().item(),
                        'One existing block22 call from frozen upstream features required')
                spatial_calls += 1

            def capture_spatial(module, args, output):
                nonlocal last_spatial
                same_thread()
                require(module is spatial and spatial_calls == 1 and last_spatial is None and
                        isinstance(output, torch.Tensor) and output.shape == (1, 64, 512, 512) and
                        output.dtype == torch.float32 and output.device == target and
                        output.requires_grad and output.grad_fn is not None and torch.isfinite(output).all().item(),
                        'One connected existing block22 output required')
                last_spatial = output

            def normalization_guard(module, args):
                nonlocal normalization_calls
                same_thread()
                require(module is final_normalization and spatial_calls == 1 and last_spatial is not None and
                        normalization_calls == 0 and len(args) == 1 and args[0] is last_spatial,
                        'Frozen outer norm23 must receive exact connected block22 output')
                normalization_calls += 1

            def capture_normalized(module, args, output):
                nonlocal last_normalized
                same_thread()
                require(module is final_normalization and normalization_calls == 1 and last_normalized is None and
                        isinstance(output, torch.Tensor) and output.shape == (1, 64, 512, 512) and
                        output.dtype == torch.float32 and output.device == target and
                        output.requires_grad and output.grad_fn is not None and torch.isfinite(output).all().item(),
                        'One connected frozen-norm23 output required')
                last_normalized = output

            def readout_guard(module, args):
                nonlocal readout_calls
                same_thread()
                require(module is readout and base_calls == 1 and tuple(completed) == SITES and
                        readout_calls == 0 and len(args) == 1 and args[0].shape == (1, 64, 512, 512) and
                        args[0].dtype == torch.float32 and args[0].device == target and
                        spatial_calls == normalization_calls == 1 and last_normalized is not None and
                        args[0] is last_normalized and args[0].requires_grad and args[0].grad_fn is not None and
                        torch.isfinite(args[0]).all().item(),
                        'One existing RGB24 readout from exact connected norm23/block22 path required')
                readout_calls += 1

            def capture_native(module, args, output):
                nonlocal last_native
                same_thread()
                require(module is readout and readout_calls == 1 and last_native is None,
                        'One existing final RGB result required')
                last_native = output

            try:
                self.validate_invariants(check_frozen_digests=True)
                require(all(not module._forward_pre_hooks and not module._forward_hooks
                            for module in self.modules()), 'Clean complete wrapper hook namespace required')
                handles.append(self.base.register_forward_pre_hook(base_guard))
                for site in SITES:
                    handles.append(self.base.fuse_convs_dict[site].register_forward_pre_hook(adapt(site)))
                handles.append(spatial.register_forward_pre_hook(spatial_guard))
                handles.append(spatial.register_forward_hook(capture_spatial))
                handles.append(final_normalization.register_forward_pre_hook(normalization_guard))
                handles.append(final_normalization.register_forward_hook(capture_normalized))
                handles.append(readout.register_forward_pre_hook(readout_guard))
                handles.append(readout.register_forward_hook(capture_native))
                with torch.autocast(device_type='cuda', enabled=False):
                    result = self.base(sample, w=1.0, detach_16=True, code_only=False, adain=True)
                require(base_calls == spatial_calls == normalization_calls == readout_calls == 1 and
                        tuple(completed) == SITES and
                        isinstance(result, tuple) and len(result) == 3 and result[0] is last_native,
                        'Untouched stock native/logits/LQ tuple required')
                for value, shape in zip(result, ((1, 3, 512, 512), (1, 256, 1024), (1, 256, 16, 16))):
                    require(isinstance(value, torch.Tensor) and value.shape == shape and
                            value.dtype == torch.float32 and value.device == target and torch.isfinite(value).all().item(),
                            'Exact finite FP32 native/logits/LQ shapes required')
                require(result[0].requires_grad and result[0].grad_fn is not None and
                        not result[1].requires_grad and not result[2].requires_grad,
                        'Only native RGB connects through the existing block22/norm23/RGB24 path')
                return result
            finally:
                try:
                    for handle in reversed(handles):
                        handle.remove()
                    self.last_forward_sites = tuple(completed)
                    self.last_base_calls = base_calls; self.last_readout_calls = readout_calls
                    self.last_spatial_calls = spatial_calls; self.last_normalization_calls = normalization_calls
                    last_native = None; last_spatial = None; last_normalized = None
                finally:
                    self._forward_lock.release()

    with _OWNERS_LOCK:
        owner = _OWNERS.get(r2_base); shared = getattr(r2_base, _OWNER_ATTRIBUTE, None)
        require((owner is None or owner() is None) and
                (shared is None or isinstance(shared, weakref.ReferenceType) and shared() is None),
                'Supplied raw base already has a live exclusive wrapper owner')
        old_flags = [(value, value.requires_grad) for value in r2_base.parameters()]
        old_modes = [(module, module.training) for module in r2_base.modules()]
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(20261023)
                candidate = U96SpatialFeasibility()
            candidate.validate_invariants(check_frozen_digests=True)
            require(state_digest(candidate.base.state_dict()) == R2_CANONICAL_SHA256,
                    'Factory must leave every original base value exact, including block22 and readout')
        except BaseException:
            for value, flag in old_flags:
                value.requires_grad_(flag)
            for module, mode in old_modes:
                module.training = mode
            raise
        _OWNERS[r2_base] = weakref.ref(candidate)
        setattr(r2_base, _OWNER_ATTRIBUTE, weakref.ref(candidate))
    return candidate
