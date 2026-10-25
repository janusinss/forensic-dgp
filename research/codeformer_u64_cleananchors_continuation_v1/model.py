"""Inert exact-u64 initializer; only a future guarded manual L4 may construct.

The retained postfusion architecture is unchanged. Its original factory must
first pass its seeded zero-head checks, after which all 36 terminal head values
are loaded and read back exactly. No optimizer, trainer or entry point exists.
Local array/source tests establish neither CUDA parity nor gradient feasibility.
"""
from contextlib import contextmanager
import hashlib
import importlib
import io
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
from types import MethodType
import zipfile


STATUS = 'INERT_U64_INITIALIZER_UNRUN_NOT_A_SEALED_WORKER'
VM_NAME = 'cctv_codeformer_u64_cleananchors_continuation_v1_vm'
MANUAL_ACKNOWLEDGEMENT = 'REVIEWED_FINITE_MANUAL_L4_U64_CONTINUATION_REQUIRED'
ROOT = Path(__file__).resolve().parents[2]
SITES = ('32', '64', '128', '256')
CHANNELS = {'32': 256, '64': 256, '128': 128, '256': 128}
TOTAL_TENSORS = 575
FROZEN_TENSORS = 539
TRAINABLE_TENSORS = 36
TRAINABLE_PARAMETERS = 168960
MAX_HEAD_FILE_BYTES = 1024 ** 2
INITIAL_HEADS_CANONICAL_SHA256 = '2e7826c07d31298eecf07d7f4d53286aa86247c00157dea9005242f23902726c'
TERMINAL_HEADS_CANONICAL_SHA256 = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
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
ORIGINAL_FACTORY = {
    'path': 'research/codeformer_u96_postfusion_v1/model.py', 'bytes': 23240,
    'sha256': 'ca2b466161932068c01980bb41fcc08e12ee8f0120f17c8fdf47898f35934aa8',
}
ARCHITECTURE_SOURCES = {
    'third_party/codeformer/codeformer_arch.py': 'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb',
    'third_party/codeformer/vqgan_arch.py': 'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf',
    'third_party/codeformer/registry.py': 'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_head_shapes():
    """Pure exact 36-tensor schema; imports no numerical framework."""
    result = {}
    for site, channels in CHANNELS.items():
        shapes = {
            'encoder.weight': (32, channels, 1, 1), 'encoder.bias': (32,),
            'decoder.weight': (32, channels, 1, 1), 'fused.weight': (32, channels, 1, 1),
            'coarse.weight': (32, 256, 1, 1), 'spatial.weight': (32, 32, 3, 3),
            'spatial.bias': (32,), 'output.weight': (channels, 32, 1, 1),
            'output.bias': (channels,),
        }
        result.update({'heads.' + site + '.' + name: shape for name, shape in shapes.items()})
    require(len(result) == TRAINABLE_TENSORS and
            sum(math.prod(shape) for shape in result.values()) == TRAINABLE_PARAMETERS,
            'Exact terminal36/168960 head schema required')
    return result


def head_arrays_canonical_sha256(arrays, *, np):
    """Canonical supplied-array digest; does not alone authorize terminal state."""
    shapes = expected_head_shapes()
    require(type(arrays) is dict and set(arrays) == set(shapes), 'Exact 36 head keys required')
    digest = hashlib.sha256()
    for name, shape in sorted(shapes.items()):
        value = arrays[name]
        require(isinstance(value, np.ndarray) and value.shape == shape and
                value.dtype == np.dtype('float32') and value.flags.c_contiguous and
                np.isfinite(value).all(), 'Finite C-order exact FP32 head array required')
        header = json.dumps([name, 'torch.float32', list(shape)], separators=(',', ':')).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def decode_head_npz(raw, *, np):
    """Bounded non-pickle schema parser; synthetic values are permitted here.

    Production initialization additionally requires the fixed terminal file and
    canonical pins in read_terminal_head_state. No np.load or pickle is used.
    """
    require(type(raw) is bytes and 0 < len(raw) < MAX_HEAD_FILE_BYTES,
            'Bounded nonempty NPZ bytes required')
    shapes = expected_head_shapes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        require(len(members) == len(set(names)) == TRAINABLE_TENSORS and
                set(names) == {name + '.npy' for name in shapes}, 'Exact unique flat head NPZ keys required')
        require(not archive.comment and all(
            item.compress_type == zipfile.ZIP_STORED and not item.flag_bits & 1 and
            not stat.S_ISLNK(item.external_attr >> 16) and not item.is_dir()
            for item in members) and sum(item.file_size for item in members) < MAX_HEAD_FILE_BYTES,
            'Ordinary unencrypted uncompressed bounded head members required')
        arrays = {}
        for item in members:
            name = item.filename[:-4]
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'Supported bounded NPY header required')
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(stream, max_header_size=4096)
                require(shape == shapes[name] and dtype == np.dtype('float32') and
                        dtype.str == '<f4' and not dtype.hasobject and fortran is False,
                        'Exact C-order nonobject FP32 NPY schema required')
                size = math.prod(shape) * 4
                require(item.file_size == stream.tell() + size, 'Exact bounded NPY payload size required')
                payload = stream.read(size + 1)  # Also checks this ZIP member's CRC.
                require(len(payload) == size, 'Complete exact NPY payload required')
                arrays[name] = np.frombuffer(payload, dtype=dtype).reshape(shape).copy(order='C')
        head_arrays_canonical_sha256(arrays, np=np)
        return arrays


def _ordinary(path, *, directory=False):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        is_directory = item != path or directory
        expected = stat.S_ISDIR(info.st_mode) if is_directory else stat.S_ISREG(info.st_mode)
        require(expected and not stat.S_ISLNK(info.st_mode) and
                not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary unlinked ancestry required')
        if item == path and not directory:
            require(info.st_nlink == 1 and 0 < info.st_size <= 512 * 1024 ** 2,
                    'Bounded single-link nonempty artifact required')
    return path.resolve(strict=True)


def file_pin(path):
    """Read-only byte binding, never a checkpoint decoder."""
    path = _ordinary(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Artifact changed during byte hashing')
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def read_terminal_head_state(path, *, np):
    """Read exact retained values only; does not construct/apply a model."""
    path = _ordinary(path)
    expected = {key: ARTIFACTS['heads'][key] for key in ('bytes', 'sha256')}
    require(file_pin(path) == expected, 'Exact retained terminal-u64 head file required')
    raw = path.read_bytes()
    require(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'],
            'Terminal head file changed before decode')
    arrays = decode_head_npz(raw, np=np)
    require(head_arrays_canonical_sha256(arrays, np=np) == TERMINAL_HEADS_CANONICAL_SHA256,
            'Exact terminal-u64 canonical36 values required; seeded/reset heads refused')
    require(file_pin(path) == expected, 'Terminal head file changed during decode')
    return arrays


def validate_manual_context(*, system, root, home, cwd, prefix, device, tmux,
                            gpu_names, acknowledgement):
    """Pure positive/negative guard schema; performs no runtime discovery."""
    require(system == 'Linux', 'Manual Linux L4 VM only')
    expected = PurePosixPath(home) / 'forensic-dgp' / VM_NAME
    require(PurePosixPath(home).is_absolute() and '..' not in PurePosixPath(home).parts and
            type(root) is str and root == str(expected) and '..' not in PurePosixPath(root).parts,
            'Exact future manual stage root required')
    require(cwd == root and prefix == str(PurePosixPath(home) / 'forensic-dgp/cctv_dgp_vm_bundle/.venv'),
            'Exact staged cwd and retained VM virtual environment required')
    require(device == 'cuda:0' and type(tmux) is str and bool(tmux.strip()), 'Explicit cuda:0 and manual tmux required')
    require(type(gpu_names) is tuple and gpu_names == ('NVIDIA L4',), 'Exactly one NVIDIA L4 required')
    require(acknowledgement == MANUAL_ACKNOWLEDGEMENT, 'Explicit finite manual-u64 acknowledgement required')
    return True


def _guard_manual_vm(manual_vm_root, device, acknowledgement, prototype_source_sha256):
    """Every standard-library guard completes before the factory imports Torch."""
    require(platform.system() == 'Linux', 'Manual Linux L4 VM only; no local/app model execution')
    require(sys.version_info >= (3, 10), 'Manual Python3.10+ required')
    root = _ordinary(manual_vm_root, directory=True)
    expected = Path.home() / 'forensic-dgp' / VM_NAME
    require(str(Path(manual_vm_root).absolute()) == str(expected) and root == ROOT == expected and
            _ordinary(Path.cwd(), directory=True) == root,
            'Initializer source and cwd must belong to the exact future manual stage')
    require(_ordinary(sys.prefix, directory=True) == Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv',
            'Retained manual VM virtual environment required')
    require(device == 'cuda:0' and bool(os.environ.get('TMUX', '').strip()) and
            acknowledgement == MANUAL_ACKNOWLEDGEMENT, 'Explicit manual tmux/device/acknowledgement required')
    require(type(prototype_source_sha256) is str and re.fullmatch('[0-9a-f]{64}', prototype_source_sha256)
            and file_pin(__file__)['sha256'] == prototype_source_sha256,
            'Future independently reviewed initializer-source pin required')
    require(file_pin(root / ORIGINAL_FACTORY['path']) ==
            {key: ORIGINAL_FACTORY[key] for key in ('bytes', 'sha256')}, 'Exact reviewed original postfusion factory required')
    for relative, known_sha in ARCHITECTURE_SOURCES.items():
        require(file_pin(root / relative)['sha256'] == known_sha, 'Exact retained architecture source required')
    executable = shutil.which('nvidia-smi')
    require(executable is not None, 'Bounded GPU-name probe required before framework import')
    _ordinary(executable)
    probe = subprocess.run([executable, '--query-gpu=name', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=6, check=False)
    require(probe.returncode == 0 and not probe.stderr.strip(), 'Finite GPU-name probe failed')
    names = tuple(line.strip() for line in probe.stdout.splitlines() if line.strip())
    validate_manual_context(system=platform.system(), root=str(root), home=str(Path.home()),
                            cwd=str(Path.cwd()), prefix=str(Path(sys.prefix)), device=device,
                            tmux=os.environ.get('TMUX', ''), gpu_names=names, acknowledgement=acknowledgement)
    return root


@contextmanager
def _preserve_runtime_rng(torch):
    """Prospective runtime check; fake-state fixtures need no Torch import."""
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
        header = json.dumps([name, 'torch.float32', list(detached.shape)], separators=(',', ':')).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def architecture_contract():
    """Prospective source scope only; no construction or numerical claim."""
    return {
        'status': STATUS, 'initializer': 'Exact retained complete-u64; no learned head reset',
        'artifacts': {name: dict(binding) for name, binding in ARTIFACTS.items()},
        'original_factory': dict(ORIGINAL_FACTORY), 'terminal_head_canonical_sha256': TERMINAL_HEADS_CANONICAL_SHA256,
        'total_tensors': TOTAL_TENSORS, 'frozen_tensors': FROZEN_TENSORS,
        'trainable_tensors': TRAINABLE_TENSORS, 'trainable_parameters': TRAINABLE_PARAMETERS,
        'trainable_shapes': {name: list(shape) for name, shape in expected_head_shapes().items()},
        'architecture_changed': False, 'optimizer_state_loaded': False,
        'complete_state_capture': 'Locked detached CPU clone of every actual575 tensor, with live/captured binding',
        'runtime_parity_gradient_or_quality_verified': False, 'manual_finite_worker_required': True,
    }


def build_restorer(r2_base, u96_adapter_state, *, r2_artifact, u96_artifact, head_artifact,
                  manual_vm_root, device='cuda:0', acknowledgement=None, prototype_source_sha256=None):
    """Initialize a future manual-L4 candidate; local verification never calls it.

    The guarded worker owns R2/U96 checkpoint decoding/base construction. This
    factory reuses the byte-pinned original architecture, then restores terminal
    heads. It accepts no optimizer state, data, steps, retry or neural call.
    """
    root = _guard_manual_vm(manual_vm_root, device, acknowledgement, prototype_source_sha256)
    paths = {'r2': r2_artifact, 'u96': u96_artifact, 'heads': head_artifact}
    for name, value in paths.items():
        path = _ordinary(value)
        require(path == root / ARTIFACTS[name]['path'] and file_pin(path) ==
                {key: ARTIFACTS[name][key] for key in ('bytes', 'sha256')}, 'Exact staged initializer artifacts required')
    import torch
    import numpy as np
    require(torch.cuda.is_initialized() and torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == 'NVIDIA L4' and torch.cuda.current_device() == 0,
            'Supplied base must already belong to the sole guarded NVIDIA L4')
    require(torch.get_default_dtype() == torch.float32 and not torch.is_inference_mode_enabled(),
            'Ordinary FP32 autograd-capable runtime required')
    arrays = read_terminal_head_state(paths['heads'], np=np)
    original = importlib.import_module('research.codeformer_u96_postfusion_v1.model')
    require(Path(original.__file__).resolve() == root / ORIGINAL_FACTORY['path'] and
            file_pin(original.__file__) == {key: ORIGINAL_FACTORY[key] for key in ('bytes', 'sha256')},
            'Active original factory differs from reviewed staged source')

    def validate(candidate, *, terminal=False):
        candidate.validate_invariants(check_frozen_digests=True)
        named = dict(candidate.named_parameters())
        head_names = set(expected_head_shapes())
        require(len(named) == len(candidate.state_dict()) == TOTAL_TENSORS and
                {name for name, value in named.items() if value.requires_grad} == head_names and
                all(value.device == torch.device('cuda:0') and value.dtype == torch.float32 and
                    torch.isfinite(value).all().item() for value in named.values()) and
                all(not module.training for module in candidate.modules()), 'Exact575/539frozen/36trainable eval FP32 state required')
        if terminal:
            actual = {name: named[name].detach().cpu().contiguous().numpy().copy() for name in head_names}
            require(head_arrays_canonical_sha256(actual, np=np) == TERMINAL_HEADS_CANONICAL_SHA256 and
                    all(np.array_equal(actual[name], arrays[name]) for name in head_names),
                    'All actual loaded terminal36 values must match exactly; reset/partial state refused')
        return named

    with _preserve_runtime_rng(torch) as rng_receipt:
        # Its zero-initialization assertion MUST run unchanged before restoration.
        candidate = original.build_restorer(r2_base, u96_adapter_state,
                                            r2_artifact=r2_artifact, u96_artifact=u96_artifact).eval()
        seeded = {name: value for name, value in candidate.named_parameters() if name.startswith('heads.')}
        require(_tensor_state_sha256(seeded, torch=torch) == INITIAL_HEADS_CANONICAL_SHA256,
                'Original seeded zero-head constructor checks must precede terminal restoration')
        with torch.no_grad():
            candidate.heads.load_state_dict({name.removeprefix('heads.'): torch.from_numpy(array.copy()).to('cuda:0')
                                            for name, array in arrays.items()}, strict=True)
        validate(candidate, terminal=True)
        require(all(value.grad is None for value in candidate.parameters()), 'Initializer must have no gradient fields')
        complete_sha = _tensor_state_sha256(candidate.state_dict(), torch=torch)

    def capture_complete_state_for_audit(self):
        require(_ordinary(Path.cwd(), directory=True) == root and ROOT == root and
                platform.system() == 'Linux', 'Complete-state capture belongs to the exact manual runtime')
        require(self._forward_lock.acquire(blocking=False), 'No capture during a concurrent/nested forward')
        try:
            validate(self)
            require(all(not module._forward_pre_hooks and not module._forward_hooks for module in self.modules()),
                    'Capture requires a clean hook namespace')
            live = self.state_dict()
            before = _tensor_state_sha256(live, torch=torch)
            captured = {name: value.detach().cpu().clone().contiguous() for name, value in live.items()}
            require(len(captured) == TOTAL_TENSORS and set(captured) == set(dict(self.named_parameters())) and
                    all(not value.requires_grad and value.grad is None and value.device.type == 'cpu'
                        for value in captured.values()) and
                    before == _tensor_state_sha256(captured, torch=torch) ==
                    _tensor_state_sha256(self.state_dict(), torch=torch),
                    'Every captured actual575 value must bind to the unchanged live complete state')
            return captured
        finally:
            self._forward_lock.release()

    candidate.capture_complete_state_for_audit = MethodType(capture_complete_state_for_audit, candidate)
    candidate.u64_initializer_receipt = {
        'source_status': STATUS, 'terminal_head_canonical_sha256': TERMINAL_HEADS_CANONICAL_SHA256,
        'complete575_canonical_sha256': complete_sha, 'total_tensors': TOTAL_TENSORS,
        'frozen_tensors': FROZEN_TENSORS, 'trainable_tensors': TRAINABLE_TENSORS,
        'trainable_parameters': TRAINABLE_PARAMETERS, 'original_zero_head_checks_preceded_terminal_load': True,
        'terminal36_readback_exact': True, 'CPU_CUDA_constructor_RNG': dict(rng_receipt),
        'optimizer_state_loaded': False, 'neural_calls': 0, 'quality_or_parity_verified': False,
    }
    return candidate
