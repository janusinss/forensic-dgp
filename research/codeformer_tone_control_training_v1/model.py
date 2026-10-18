"""Guarded frozen-CodeFormer plus scalar-controller helpers.

Importing this module constructs nothing and imports no neural framework.
Only activate_runtime, after the worker's own protocol/manual-root checks, can
import Torch. CPU feature/analytic functions remain available without Torch.
No optimizer is constructed or stepped by this module.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import stat


VM_NAME = 'cctv_codeformer_tone_control_training_v1_vm'
OUTPUTS = 'outputs/tone_control_training_v1'
INITIALIZER_SHA256 = '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'
INITIALIZER_BYTES = 376637898
STOPPED_SHA256 = '43c0f313365e13849ddcd852f401511e4c7ae52cbac55fdd98a6b032f06ef37b'
STOPPED_BYTES = 855230072
PARENT_PROTOCOL = '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
ORIGINAL_MODEL_DIGEST = '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd'
STOPPED_MODEL_DIGEST = '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b'
LUMINANCE_WEIGHTS = (.2126, .7152, .0722)
HEAD_SEED = 20261017
HEAD_PARAMETERS = 8257
OFFSET_SCALE = .25
GRADIENT_ATOL = 2e-7
GRADIENT_RTOL = 2e-5
RESERVE_BYTES = 4 * 1024**3
_RUNTIME = None
_BASE = None
ZERO_HEAD_IDENTITY = {
    'fresh_base_RAW': 'Exact historical full-coarse RAW on all pixels',
    'corrected_RAW': 'Equals fresh base on observed pixels; equals input on unobserved pixels',
    'delivered_PNG': 'Exact historical full-coarse PNG on all pixels after the existing padding restoration',
    'full_frame_corrected_RAW_equals_unprotected_historical_RAW_claimed': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for payload in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(payload)
    return value.hexdigest()


def regular(path):
    path = Path(path).absolute()
    for component in [path, *path.parents]:
        info = component.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                'Model/source path links or reparse points forbidden')
    require(stat.S_ISREG(path.lstat().st_mode), 'Exact regular model/source file required')
    return path


def activate_runtime(root, protocol_sha):
    """Repeat the worker's manual guard before the sole lazy Torch import."""
    global _RUNTIME
    require(platform.system() == 'Linux', 'Actual model/controller calls are manual Linux L4 VM only')
    root = Path(root).absolute()
    for path in [root, *root.parents]:
        require(not path.is_symlink(), 'Manual runtime root must not contain links')
    root = root.resolve(strict=True)
    expected = (Path.home() / 'forensic-dgp' / VM_NAME).resolve(strict=False)
    require(root == expected and bool(os.environ.get('TMUX')), 'Exact new manual home root and tmux required')
    require(isinstance(protocol_sha, str) and len(protocol_sha) == 64 and
            all(c in '0123456789abcdef' for c in protocol_sha), 'Exact protocol SHA256 required')
    protocol = regular(root / 'protocol.json')
    require(sha256(protocol) == protocol_sha, 'Manual runtime protocol changed before Torch import')
    if _RUNTIME is not None:
        require(_RUNTIME['root'] == root and _RUNTIME['protocol_sha256'] == protocol_sha,
                'Do not rebind an activated runtime')
        return _RUNTIME['torch']
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == 'NVIDIA L4', 'Exactly one available NVIDIA L4 required')
    _RUNTIME = {'root': root, 'protocol_sha256': protocol_sha, 'torch': torch}
    return torch


def _runtime():
    require(_RUNTIME is not None, 'Activate the verified manual VM runtime before any Torch helper')
    return _RUNTIME['torch']


def features_numpy(input_rgb, observed):
    """One shared production implementation belongs to the CPU data module."""
    from .data import features
    return features(input_rgb, observed)


def numpy_mean_y(rgb, observed):
    """Independent scalar/fixture arithmetic; no neural imports or calls."""
    import numpy as np
    rgb, observed = np.asarray(rgb), np.asarray(observed)
    require(rgb.ndim == 3 and rgb.shape[2] == 3 and observed.shape == rgb.shape[:2] and
            observed.dtype == np.bool_ and observed.any() and np.isfinite(rgb).all(),
            'Finite HWC RGB and nonempty boolean support required')
    return float((rgb.astype(np.float64)[observed] @ np.asarray(LUMINANCE_WEIGHTS, np.float64)).mean())


def analytic_fixture_gradients(base, observed, target, logit, hidden):
    """Exact real-valued clamp/mean/tanh chain while final-layer W is zero.

    This function never calls autograd. Fixtures avoid clamp/L1 kinks. Hidden
    activations are source-captured values; earlier-layer derivatives are zero.
    """
    import numpy as np
    base = np.asarray(base, dtype=np.float64)
    observed = np.asarray(observed)
    hidden = np.asarray(hidden, dtype=np.float64).reshape(32)
    require(base.ndim == 3 and base.shape[-1] == 3 and observed.shape == base.shape[:2] and
            observed.dtype == np.bool_ and observed.any() and np.isfinite(base).all() and
            np.isfinite(hidden).all() and math.isfinite(float(target)) and math.isfinite(float(logit)),
            'Invalid analytic fixture')
    tanh = math.tanh(float(logit)); delta = OFFSET_SCALE * tanh
    shifted = base + delta
    require(float(np.minimum(np.abs(shifted), np.abs(shifted - 1)).min()) > .001,
            'Fixture must remain away from clamp kinks')
    corrected = np.clip(shifted, 0, 1)
    mean = numpy_mean_y(corrected, observed)
    active = ((shifted > 0) & (shifted < 1)).astype(np.float64)
    d_mean_d_delta = numpy_mean_y(active, observed)
    d_loss_d_delta = 2 * (mean - float(target)) * d_mean_d_delta
    d_loss_d_logit = d_loss_d_delta * OFFSET_SCALE * (1 - tanh**2)
    gradients = {'hidden.weight': np.zeros((32, 256), np.float64),
        'hidden.bias': np.zeros(32, np.float64),
        'output.weight': d_loss_d_logit * hidden.reshape(1, 32),
        'output.bias': np.asarray([d_loss_d_logit], np.float64)}
    return gradients, {'delta': delta, 'mean': mean, 'target': float(target),
        'loss': (mean - float(target))**2, 'd_mean_d_delta': d_mean_d_delta,
        'd_loss_d_delta': d_loss_d_delta, 'd_loss_d_logit': d_loss_d_logit,
        'clipped_scalar_fraction': float(((shifted < 0) | (shifted > 1))[observed].mean())}


def ordinary_tensor(array, kind):
    """New ordinary FP32/bool CUDA tensors; never reuse inference-mode tensors."""
    torch = _runtime()
    import numpy as np
    require(not torch.is_inference_mode_enabled(), 'Construct head tensors outside inference_mode')
    value = np.asarray(array)
    if kind == 'rgb':
        require(value.shape == (256, 256, 3) and value.dtype == np.float32 and
                np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1,
                'Exact finite float32 RGB256 in[0,1] required')
        result = torch.from_numpy(value.copy()).permute(2, 0, 1).unsqueeze(0).to('cuda')
    elif kind == 'mask':
        require(value.shape == (256, 256) and value.dtype == np.bool_ and value.any(),
                'Exact nonempty boolean256 support required')
        result = torch.from_numpy(value.copy()).unsqueeze(0).unsqueeze(0).to('cuda')
    elif kind == 'features':
        require(value.dtype == np.float32 and value.shape in ((256,), (1, 256)) and np.isfinite(value).all(),
                'Exact float32 input feature vector required')
        result = torch.from_numpy(value.reshape(1, 256).copy()).to('cuda')
    elif kind == 'target':
        require(value.size == 1 and np.isfinite(value).all(), 'One finite target mean required')
        result = torch.from_numpy(np.asarray(value, dtype=np.float32).reshape(1).copy()).to('cuda')
    else:
        raise ValueError('Unknown ordinary tensor kind')
    require(not torch.is_inference(result) and not result.requires_grad, 'Unsafe head input tensor')
    return result


def create_head():
    """Return exactly8257FP32 parameters; preserve pre-call Torch CPU/CUDA RNG."""
    torch = _runtime()
    class ToneHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.nn.Linear(256, 32, bias=True)
            self.activation = torch.nn.SiLU()
            self.output = torch.nn.Linear(32, 1, bias=True)
            torch.nn.init.zeros_(self.output.weight)
            torch.nn.init.zeros_(self.output.bias)

        def forward(self, features, return_details=False):
            require(features.ndim == 2 and features.shape[1] == 256 and
                    features.dtype == torch.float32 and features.device.type == 'cuda' and
                    torch.isfinite(features).all().item(), 'Head expects finite FP32 CUDA Bx256 features')
            with torch.autocast(device_type='cuda', enabled=False):
                hidden = self.activation(self.hidden(features))
                logit = self.output(hidden)
                delta = (OFFSET_SCALE * torch.tanh(logit)).reshape(-1)
            require(torch.isfinite(delta).all().item(), 'Nonfinite controller offset')
            return (delta, {'hidden': hidden, 'logit': logit}) if return_details else delta

    before_cpu, before_cuda = torch.get_rng_state().clone(), [s.clone() for s in torch.cuda.get_rng_state_all()]
    with torch.random.fork_rng(devices=[0], enabled=True):
        torch.manual_seed(HEAD_SEED)
        head = ToneHead().float().to('cuda')
    require(torch.equal(before_cpu, torch.get_rng_state()) and len(before_cuda) == 1 and
            all(torch.equal(a, b) for a, b in zip(before_cuda, torch.cuda.get_rng_state_all())),
            'Head construction changed pre-call RNG')
    require(sum(p.numel() for p in head.parameters()) == HEAD_PARAMETERS and
            len(tuple(head.parameters())) == 4 and all(p.dtype == torch.float32 for p in head.parameters()),
            'Controller parameter census/precision differs')
    return head


def tensor_state_digest(state):
    """Canonical old audit/common algorithm:8-byte length+JSON header+bytes."""
    torch = _runtime()
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor), 'State digest requires tensors')
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def legacy_tensor_state_digest(state):
    """Explicitly separate the old exporter's unlength-prefixed concatenation."""
    _runtime()
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(list(value.shape)).encode('ascii')); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_digest(module):
    return tensor_state_digest(module.state_dict())


def cpu_state(module):
    _runtime()
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def load_frozen_parents(initializer, stopped):
    """Load two unchanged CodeFormer models; never construct an optimizer."""
    global _BASE
    torch = _runtime()
    initial_path, stopped_path = regular(initializer), regular(stopped)
    for path, size, expected in ((initial_path, INITIALIZER_BYTES, INITIALIZER_SHA256),
                                 (stopped_path, STOPPED_BYTES, STOPPED_SHA256)):
        before = path.stat()
        require(before.st_size == size and sha256(path) == expected, 'Exact frozen checkpoint bytes differ')
        after = path.stat()
        require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                (after.st_size, after.st_mtime_ns, after.st_ino), 'Checkpoint changed while hashing')
    from research.codeformer_adaptation_v2 import model as base_model
    _BASE = base_model
    original, _ = base_model.load_trainable(initial_path, device='cuda')
    original.eval().requires_grad_(False)
    base = deepcopy(original).eval().requires_grad_(False)
    loaded = torch.load(stopped_path, map_location='cpu', weights_only=True, mmap=True)
    require(loaded.get('format') == 'codeformer-adain-gradient-isolation-state-v1' and
            loaded.get('protocol_sha256') == PARENT_PROTOCOL and loaded.get('completed_updates') == 50 and
            loaded.get('partial_microbatches_discarded') == 0 and loaded.get('resume_safe') is True,
            'Expected committed historical50 model state, not an optimizer resume')
    values = loaded['model']
    require(list(values) == list(original.state_dict()) and len(values) == 515 and all(
        isinstance(value, torch.Tensor) and value.dtype == torch.float32 and torch.isfinite(value).all().item()
        for value in values.values()), 'Historical515tensor model values/precision differ')
    base.load_state_dict(values, strict=True)
    del loaded, values
    for net in (original, base):
        require(len(tuple(net.parameters())) == len(net.state_dict()) == 515 and not net.training and
                all(not p.requires_grad and p.grad is None and p.dtype == torch.float32 for p in net.parameters()),
                'Frozen parent parameter/mode/gradient status differs')
    canonical = {'original': state_digest(original), 'restorer': state_digest(base)}
    require(canonical == {'original': ORIGINAL_MODEL_DIGEST, 'restorer': STOPPED_MODEL_DIGEST},
            'Exact frozen parent model tensor identities differ')
    receipt = {'complete': True, 'original_checkpoint_sha256': INITIALIZER_SHA256,
        'restorer_checkpoint_sha256': STOPPED_SHA256, 'restorer_completed_updates': 50,
        'canonical_tensor_state_sha256': canonical,
        'canonical_digest_algorithm': 'sortedname;8byte-big-endian JSON[name,dtype,shape]length;header;contiguousbytes',
        'legacy_export_concat_tensor_sha256': {'original': legacy_tensor_state_digest(original.state_dict()),
                                             'restorer': legacy_tensor_state_digest(base.state_dict())},
        'model_tensors_each': 515, 'all_parameters_FP32_eval_frozen_empty_grad': True,
        'historical_optimizer_RNG_sampler_loaded_into_runtime': False, 'new_optimizer_constructed': False,
        'model_architecture_changed': False, 'neural_forwards': 0}
    return original, base, receipt


def infer_rgb(net, input_rgb, *, return_receipt=False):
    """One frozen CodeFormer call; return an ordinary independent HWC ndarray."""
    torch = _runtime()
    require(_BASE is not None and not net.training and all(not p.requires_grad and p.grad is None
        for p in net.parameters()), 'Frozen eval parent required')
    value = ordinary_tensor(input_rgb, 'rgb')
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        native, logits, feature = net(_BASE.prepare_input(value), w=1.0, detach_16=True, adain=True)
        delivered = _BASE.delivered_rgb(native)
        require(native.dtype == logits.dtype == feature.dtype == delivered.dtype == torch.float32,
                'Frozen forward must retain actual FP32 native/logits/features/delivery')
        require(torch.isfinite(logits).all().item() and torch.isfinite(feature).all().item(),
                'Nonfinite frozen logits/features')
        cpu = delivered[0].permute(1, 2, 0).detach().cpu()
        receipt = {'native_dtype': str(native.dtype), 'logits_dtype': str(logits.dtype),
            'features_dtype': str(feature.dtype), 'delivered_dtype': str(delivered.dtype),
            'native_shape': list(native.shape), 'logits_shape': list(logits.shape),
            'features_shape': list(feature.shape), 'delivered_shape': list(delivered.shape),
            'autocast_enabled': bool(torch.is_autocast_enabled('cuda')),
            'softmax_dtype_not_separately_captured': True, 'neural_forwards': 1,
            'all_parent_parameter_gradients_none': all(p.grad is None for p in net.parameters())}
    result = cpu.numpy().copy()
    require(result.shape == (256, 256, 3), 'Frozen delivered geometry differs')
    return (result, receipt) if return_receipt else result


def _rgb_mask(rgb, observed):
    torch = _runtime()
    require(isinstance(rgb, torch.Tensor) and rgb.ndim == 4 and rgb.shape[1] == 3 and len(rgb) > 0 and
            rgb.dtype == torch.float32 and rgb.device.type == 'cuda' and torch.isfinite(rgb).all().item(),
            'Finite FP32 CUDA BCHW RGB required')
    require(isinstance(observed, torch.Tensor) and observed.dtype == torch.bool and
            observed.shape == (len(rgb), 1, rgb.shape[2], rgb.shape[3]) and
            observed.device == rgb.device and observed.flatten(1).any(dim=1).all().item(),
            'Nonempty exact boolean observed support per image required')
    return torch


def mean_y(rgb, observed):
    torch = _rgb_mask(rgb, observed)
    with torch.autocast(device_type='cuda', enabled=False):
        weights = rgb.new_tensor(LUMINANCE_WEIGHTS).view(1, 3, 1, 1)
        luminance = (rgb * weights).sum(dim=1, keepdim=True)
        mask = observed.to(dtype=torch.float32)
        result = (luminance * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1)
    require(torch.isfinite(result).all().item(), 'Nonfinite observed luminance mean')
    return result


def composite_raw(base, input_rgb, observed, delta):
    torch = _rgb_mask(base, observed)
    require(input_rgb.shape == base.shape and input_rgb.dtype == torch.float32 and
            input_rgb.device == base.device and torch.isfinite(input_rgb).all().item() and
            base.min().item() >= 0 and base.max().item() <= 1 and
            input_rgb.min().item() >= 0 and input_rgb.max().item() <= 1 and
            not base.requires_grad and not input_rgb.requires_grad,
            'Ordinary detached bounded base/input tensors required')
    require(delta.shape == (len(base),) and delta.dtype == torch.float32 and delta.device == base.device and
            torch.isfinite(delta).all().item() and delta.min().item() >= -OFFSET_SCALE and
            delta.max().item() <= OFFSET_SCALE, 'One finite bounded FP32 controller offset per image required')
    with torch.autocast(device_type='cuda', enabled=False):
        corrected = (base + delta.reshape(-1, 1, 1, 1)).clamp(0, 1)
        result = torch.where(observed, corrected, input_rgb)
    require(torch.isfinite(result).all().item(), 'Nonfinite composite RAW')
    return result


def apply_head(head, features, base, input_rgb, observed):
    _runtime()
    delta = head(features)
    result = composite_raw(base, input_rgb, observed, delta)
    return result, delta, mean_y(result, observed)


def training_loss(head, features, base, input_rgb, observed, target):
    torch = _runtime()
    require(not torch.is_inference_mode_enabled() and target.shape == (len(base),) and
            target.dtype == torch.float32 and target.device == base.device and
            not target.requires_grad and torch.isfinite(target).all().item(), 'Ordinary detached target means required')
    for value in (features, base, input_rgb, observed, target):
        require(not torch.is_inference(value), 'Do not save an inference-mode tensor for head backward')
    corrected, delta, predicted_mean = apply_head(head, features, base, input_rgb, observed)
    loss = squared_mean_loss(predicted_mean, target)
    return loss, corrected, delta, predicted_mean


def squared_mean_loss(predicted_mean, target):
    """One scalar loss expression shared by training and both VM fixtures."""
    torch = _runtime()
    require(predicted_mean.ndim == 1 and target.shape == predicted_mean.shape and
            target.dtype == predicted_mean.dtype == torch.float32 and
            not target.requires_grad and torch.isfinite(target).all().item(),
            'Same-shape finite detached target means required')
    loss = (predicted_mean - target).square().mean()
    require(torch.isfinite(loss).item(), 'Nonfinite controller loss')
    return loss


def capture_rng():
    torch = _runtime()
    import numpy as np
    value = np.random.get_state()
    return {'python': random.getstate(), 'numpy': {'algorithm': value[0], 'keys': value[1].tolist(),
        'position': value[2], 'has_gauss': value[3], 'cached_gaussian': value[4]},
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': [state.clone() for state in torch.cuda.get_rng_state_all()]}


def restore_rng(state):
    torch = _runtime()
    import numpy as np
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value['algorithm'], np.asarray(value['keys'], dtype=np.uint32),
                         value['position'], value['has_gauss'], value['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu']); torch.cuda.set_rng_state_all(state['torch_cuda'])


def rng_equal(a, b):
    torch = _runtime()
    return (a['python'] == b['python'] and a['numpy'] == b['numpy'] and
            torch.equal(a['torch_cpu'], b['torch_cpu']) and len(a['torch_cuda']) == len(b['torch_cuda']) == 1 and
            all(torch.equal(x, y) for x, y in zip(a['torch_cuda'], b['torch_cuda'])))


def head_prerequisites(head, out, root, reserve, budget):
    """Two actual VM head forwards/queries; preserve state/RNG/None gradients.

    reserve(name) reserves from the worker's already frozen protocol counters.
    budget() measures/enforces the parent's finite time/resource limits.
    This creates25bounded regular artifacts, never an optimizer or CF forward.
    """
    torch = _runtime()
    import numpy as np
    import shutil
    root, out = Path(root).resolve(strict=True), Path(out).resolve(strict=True)
    require(root == _RUNTIME['root'] and out == root / OUTPUTS and
            all(p.grad is None for p in head.parameters()), 'Exact head prerequisite root/empty gradients required')
    directory = out / 'head_derivative_prerequisites'
    require(not directory.exists(), 'Preserve previous head prerequisite evidence')
    directory.mkdir()
    before_rng, before_state = capture_rng(), cpu_state(head)
    before_digest = tensor_state_digest(before_state)
    before_modes = [(module, module.training) for module in head.modules()]
    before_grad_none = [p.grad is None for p in head.parameters()]
    rows = []
    report = {'complete': False, 'pass': False, 'head_forwards': 0, 'gradient_queries': 0,
        'CodeFormer_forwards': 0, 'CodeFormer_backward': 0, 'optimizer_steps': 0,
        'before_Adam_construction': True, 'head_state_before_sha256': before_digest,
        'fixtures': rows, 'analytic_gradients_are_not_neural_or_optimizer_replay': True,
        'zero_head_identity_policy': ZERO_HEAD_IDENTITY}

    def checked_path(name):
        path = directory / name
        require(path.parent.resolve(strict=True) == directory and not path.exists(), 'Preserve prerequisite artifact')
        return path

    def write_array(name, value):
        budget(); value = np.ascontiguousarray(value)
        require(value.dtype in (np.dtype('float32'), np.dtype('float64'), np.dtype('uint8')) and
                value.nbytes <= 128 * 1024, 'Prerequisite array bound/dtype differs')
        require(shutil.disk_usage(root).free >= RESERVE_BYTES + value.nbytes + 65536, 'Prerequisite disk reserve reached')
        path = checked_path(name)
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        return {'path': path.relative_to(root).as_posix(), 'sha256': sha256(path),
                'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': path.stat().st_size,
                'array_sha256': hashlib.sha256(value.tobytes()).hexdigest()}

    def write_json(name, value):
        budget()
        payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
        require(len(payload) <= 256 * 1024 and shutil.disk_usage(root).free >= RESERVE_BYTES + len(payload) + 65536,
                'Prerequisite metadata/disk bound reached')
        path = checked_path(name)
        with path.open('xb') as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())

    def save_rng(label, state):
        value = {'python': state['python'], 'numpy': state['numpy'], 'torch': {}}
        for key, item in [('CPU', state['torch_cpu']), ('CUDA0', state['torch_cuda'][0])]:
            value['torch'][key] = write_array('rng_' + label + '_' + key + '.npy', item.cpu().numpy())
        write_json('rng_' + label + '.json', value)

    try:
        require(list(dict(head.named_parameters())) == ['hidden.weight', 'hidden.bias', 'output.weight', 'output.bias'] and
                sum(p.numel() for p in head.parameters()) == HEAD_PARAMETERS and
                torch.count_nonzero(head.output.weight).item() == torch.count_nonzero(head.output.bias).item() == 0,
                'Actual pristine zero-output head required')
        save_rng('before', before_rng)
        features = torch.from_numpy(np.linspace(.1, .9, 256, dtype=np.float32).reshape(1, 256)).to('cuda')
        fixtures = [('zero_output_chain', np.full((2, 2, 3), .4, np.float32), .45, 0.),
                    ('active_clipping_chain', np.broadcast_to(np.asarray([.95, .30, .10], np.float32),
                        (2, 2, 3)).copy(), .60, math.atanh(.4))]
        for label, base_np, target_value, bias in fixtures:
            budget(); head.load_state_dict(before_state, strict=True)
            with torch.no_grad(): head.output.bias.fill_(bias)
            require(torch.count_nonzero(head.output.weight).item() == 0, 'Analytic fixture final weight must be zero')
            observed_np = np.ones((2, 2), np.bool_)
            input_np = base_np.copy()
            if label == 'active_clipping_chain':
                observed_np[0, 1] = False
                input_np[0, 1] = np.asarray([.11, .22, .33], np.float32)
            base = torch.from_numpy(base_np.transpose(2, 0, 1).copy()).unsqueeze(0).to('cuda')
            fixture_input = torch.from_numpy(input_np.transpose(2, 0, 1).copy()).unsqueeze(0).to('cuda')
            observed = torch.from_numpy(observed_np.copy()).unsqueeze(0).unsqueeze(0).to('cuda')
            target = torch.tensor([target_value], dtype=torch.float32, device='cuda')
            reserve('controller_math_prerequisites'); report['head_forwards'] += 1
            delta, details = head(features, return_details=True)
            corrected = composite_raw(base, fixture_input, observed, delta)
            predicted = mean_y(corrected, observed)
            loss = squared_mean_loss(predicted, target)
            reserve('head_autograd_queries'); report['gradient_queries'] += 1
            names, parameters = zip(*head.named_parameters())
            actual = torch.autograd.grad(loss, parameters, allow_unused=True, create_graph=False, retain_graph=False)
            require(all(value is not None for value in actual), 'Connected zero gradient must not be recorded as None')
            hidden = details['hidden'].detach().float().cpu().numpy()[0]
            logit = float(details['logit'].detach().float().cpu()[0, 0])
            expected, scalar = analytic_fixture_gradients(base_np, observed_np,
                float(target.detach().cpu()[0]), logit, hidden)
            comparisons = {}
            for name, value in zip(names, actual):
                require(value.dtype == torch.float32 and torch.isfinite(value).all().item(), 'Nonfinite actual fixture gradient')
                array = value.detach().cpu().numpy().copy()
                wanted = expected[name]
                maximum = float(np.abs(array.astype(np.float64) - wanted).max())
                bound = GRADIENT_ATOL + GRADIENT_RTOL * float(np.abs(wanted).max())
                actual_file = write_array(label + '_' + name.replace('.', '_') + '_actual.npy', array)
                expected_file = write_array(label + '_' + name.replace('.', '_') + '_expected.npy', wanted)
                comparisons[name] = {'is_none': False, 'maximum_absolute_error': maximum, 'bound': bound,
                    'actual': actual_file, 'expected': expected_file, 'pass': maximum <= bound,
                    'expected_connected_zero': name.startswith('hidden.'),
                    'connected_zero_exact': not name.startswith('hidden.') or np.count_nonzero(array) == 0}
            require(all(p.grad is None for p in head.parameters()), 'Query unexpectedly populated parameter.grad')
            corrected_np = corrected.detach().cpu().numpy()[0].transpose(1, 2, 0).copy()
            protected_exact = bool(np.array_equal(corrected_np[~observed_np], input_np[~observed_np]))
            require(protected_exact, 'Actual composite must restore protected input pixels exactly')
            fixture_pass = all(item['pass'] and item['connected_zero_exact'] for item in comparisons.values())
            row = {'fixture': label, 'pass': fixture_pass, 'features': features.detach().cpu().numpy()[0].tolist(),
                'base_RGB': base_np.tolist(), 'input_RGB': input_np.tolist(), 'corrected_RAW': corrected_np.tolist(),
                'unobserved_input_exact': protected_exact,
                'observed': observed_np.tolist(), 'target': float(target.detach().cpu()[0]),
                'actual_delta': float(delta.detach().cpu()[0]), 'actual_mean': float(predicted.detach().cpu()[0]),
                'actual_loss': float(loss.detach().cpu()), 'logit': logit, 'analytic_scalars': scalar,
                'hidden': hidden.tolist(), 'gradient_queries': 1, 'head_forwards': 1,
                'absolute_tolerance': GRADIENT_ATOL, 'relative_tolerance': GRADIENT_RTOL,
                'gradient_arrays': comparisons, 'parameter_grad_fields_none': True}
            rows.append(row); write_json(label + '.json', row)
            require(fixture_pass, 'Actual shared head/clamp derivative or connected-zero check fails prospective bound')
            del actual, loss, corrected, predicted, details, delta
        require(report['head_forwards'] == report['gradient_queries'] == len(rows) == 2,
                'Exactly two actual head forwards/queries required')
        report['complete'] = report['pass'] = True
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        observed_grad_none = all(p.grad is None for p in head.parameters())
        head.load_state_dict(before_state, strict=True)
        for module, mode in before_modes: module.training = mode
        for parameter, was_none in zip(head.parameters(), before_grad_none):
            if was_none: parameter.grad = None
        restore_rng(before_rng); after_rng = capture_rng()
        report.update(head_state_restored_sha256=state_digest(head),
            head_state_restored_exact=state_digest(head) == before_digest,
            RNG_restored_exact=rng_equal(before_rng, after_rng),
            parameter_grad_fields_none_before_cleanup=observed_grad_none,
            parameter_grad_fields_none_restored=all(p.grad is None for p in head.parameters()),
            modes_restored_exact=all(module.training == mode for module, mode in before_modes))
        save_rng('restored', after_rng)
        write_json('prerequisites.json', report)
        require(report['head_state_restored_exact'] and report['RNG_restored_exact'] and
                report['parameter_grad_fields_none_restored'] and report['modes_restored_exact'] and observed_grad_none,
                'Head prerequisites failed exact state/RNG/mode/None restoration')
    return report
