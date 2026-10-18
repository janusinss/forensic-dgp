"""Frozen manual-L4 response helpers; importing this module never imports Torch.

No optimizer is constructed, no historical optimizer/RNG is restored, and no
gradient is queried. The original and adapted CodeFormer plus both actual heads
are frozen eval models. A response is always the unchanged clipped-RAW policy.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys

VM_NAME = 'cctv_codeformer_fixed_state_response_v1_vm'
ADAIN_NAME = 'cctv_codeformer_adain_gradient_isolation_v1_vm'
TONE_NAME = 'cctv_codeformer_tone_control_training_v1_vm'
TONE_PROTOCOL = 'f10ce6d52c2ec259632750af2d3afbdf97c5c925b99849fae2a2c4d9cd04a5b1'
ADAIN_PROTOCOL = '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
ORIGINAL_DIGEST = '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd'
BASE_DIGEST = '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b'
HEAD_DIGESTS = {'head0000': '9a3f78f85d1bc9b70878cabad61545ec808248566c7b13220052c3ed02918509',
                'head0050': 'd6ec3096958fc53c3e067de3c44ff398252a2bd9bd25568dcf89997f68f3b20b'}
ASSETS = {
    'initializer': {'root': 'adain', 'path': 'weights/codeformer.pth',
        'sha256': '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7', 'bytes': 376637898},
    'initial_composite': {'root': 'tone_training',
        'path': 'outputs/tone_control_training_v1/composite_head0000/checkpoint.pt',
        'sha256': 'dc3f161ff94fe7f5d851d7610fdab403211e0bba9a1d4b36ada7328a10130da7', 'bytes': 376665005},
    'stopped_composite': {'root': 'tone_training',
        'path': 'outputs/tone_control_training_v1/stopped_composite/checkpoint.pt',
        'sha256': 'f492d13cd63f9652407ca2eb4b90ae2b9a5fc49a528c4a62d552f47d1aec4a2f', 'bytes': 376734485}}
HISTORICAL_MODULES = {
    'research.codeformer_adaptation_v1': 'research/codeformer_adaptation_v1/__init__.py',
    'research.codeformer_adaptation_v1.common': 'research/codeformer_adaptation_v1/common.py',
    'research.codeformer_adaptation_v2': 'research/codeformer_adaptation_v2/__init__.py',
    'research.codeformer_adaptation_v2.model': 'research/codeformer_adaptation_v2/model.py',
    'third_party': 'third_party/__init__.py',
    'third_party.codeformer': 'third_party/codeformer/__init__.py',
    'third_party.codeformer.registry': 'third_party/codeformer/registry.py',
    'third_party.codeformer.vqgan_arch': 'third_party/codeformer/vqgan_arch.py',
    'third_party.codeformer.codeformer_arch': 'third_party/codeformer/codeformer_arch.py'}
HEAD_SHAPES = {'hidden.weight': (32, 256), 'hidden.bias': (32,),
               'output.weight': (1, 32), 'output.bias': (1,)}
_TORCH = _ROOT = _BASE = None


def require(condition, message):
    if not condition:
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
            info = item.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                    'Linked/reparse model source forbidden')
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Exact unlinked regular model source required')
    return path.resolve(strict=True)


def _read_json(path):
    def unique(items):
        result = {}
        for k, v in items:
            require(k not in result, 'Duplicate protocol key')
            result[k] = v
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite protocol')))


def activate_runtime(root, protocol_sha):
    """Repeat the strict manual-root/transfer prerequisite before lazy Torch import."""
    global _TORCH, _ROOT
    require(platform.system() == 'Linux', 'Manual NVIDIA L4 VM only; local neural calls forbidden')
    root = Path(root).absolute()
    expected = Path.home() / 'forensic-dgp' / VM_NAME
    for path in (root, expected):
        for item in (path, *path.parents):
            if item.exists() or item.is_symlink():
                st = item.lstat()
                require(not stat.S_ISLNK(st.st_mode) and not getattr(st, 'st_file_attributes', 0) & 1024,
                        'Linked/reparse runtime root forbidden')
    require(root.resolve(strict=True) == expected.resolve(strict=True) and root.is_dir(), 'Exact manual VM root required')
    require(bool(os.environ.get('TMUX')), 'Manual tmux session required before framework import')
    require(isinstance(protocol_sha, str) and re.fullmatch('[0-9a-f]{64}', protocol_sha), 'Exact protocol SHA required')
    protocol = regular(root / 'protocol.json')
    require(protocol.stat().st_size <= 8*1024**2 and sha256(protocol) == protocol_sha, 'Frozen new protocol differs')
    from . import contract as c
    c.manual_root(root)
    c.verify_protocol(_read_json(protocol))
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            torch.cuda.get_device_name(0) == 'NVIDIA L4', 'One actual NVIDIA L4 required')
    _TORCH, _ROOT = torch, root.resolve(strict=True)
    return torch


def _runtime():
    require(_TORCH is not None and _ROOT is not None, 'Activate the guarded manual L4 runtime first')
    return _TORCH


def features_numpy(input_rgb, observed):
    """The sole new production feature implementation is the pinned data helper."""
    from .data import features
    return features(input_rgb, observed)


def tensor_state_digest(state):
    _runtime()
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode('utf-8')
        h.update(len(header).to_bytes(8, 'big')); h.update(header)
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_digest(module):
    _runtime()
    return tensor_state_digest(module.state_dict())


def tensor_digest(value):
    """Canonical historical digest for a module or an explicit tensor-state dict."""
    _runtime()
    return tensor_state_digest(value if isinstance(value, dict) else value.state_dict())


def cpu_state(module):
    _runtime()
    return {name:value.detach().cpu().clone() for name,value in module.state_dict().items()}


def _frozen(module, count):
    torch = _runtime()
    require(all(not m.training for m in module.modules()) and
            len(tuple(module.parameters())) == len(module.state_dict()) == count and
            all(not p.requires_grad and p.grad is None and p.dtype == torch.float32 and
                p.device.type == 'cuda' for p in module.parameters()), 'Frozen eval FP32 CUDA module required')


def _check_pin(path, pin):
    path = regular(path); before = path.stat()
    require(before.st_size == pin['bytes'] and sha256(path) == pin['sha256'], 'Exact checkpoint bytes differ')
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Checkpoint changed while hashing')
    return path


def _historical_imports(adain):
    old_protocol = regular(adain / 'protocol.json')
    require(sha256(old_protocol) == ADAIN_PROTOCOL, 'Exact historical AdaIN protocol required')
    old = _read_json(old_protocol)
    for name in HISTORICAL_MODULES.values():
        path = regular(adain / name)
        require(sha256(path) == old['files_sha256'][name], 'Historical model source changed: ' + name)
    # The worker adds only the verified old research namespace and repository root.
    # Reject a cached module or import resolution pointing at a different checkout.
    for module_name, relative in HISTORICAL_MODULES.items():
        spec = importlib.util.find_spec(module_name)
        require(spec is not None and spec.origin is not None and
                Path(spec.origin).resolve(strict=True) == (adain / relative).resolve(strict=True),
                'Historical module resolves outside exact read-only source: ' + module_name)
    from research.codeformer_adaptation_v2 import model as base_model
    return base_model


def _head(torch):
    require(torch is _runtime(), 'Use the activated runtime for frozen head construction')
    class FixedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.nn.Linear(256, 32)
            self.activation = torch.nn.SiLU()
            self.output = torch.nn.Linear(32, 1)

        def forward(self, features):
            return .25 * torch.tanh(self.output(self.activation(self.hidden(features)))).squeeze(-1)
    return FixedHead().to(device='cuda', dtype=torch.float32).eval().requires_grad_(False)


def load_frozen_parents_and_heads(initializer, initial_composite, stopped_composite):
    """Load model/head tensors only; never construct/restore an optimizer or old RNG."""
    global _BASE
    torch = _runtime()
    roots = {'adain': _ROOT.parent / ADAIN_NAME, 'tone_training': _ROOT.parent / TONE_NAME}
    supplied = {'initializer': initializer, 'initial_composite': initial_composite, 'stopped_composite': stopped_composite}
    paths = {}
    for name, pin in ASSETS.items():
        require(Path(supplied[name]).resolve(strict=True) == (roots[pin['root']] / pin['path']).resolve(strict=True),
                'Checkpoint lies outside its exact declared parent namespace')
        paths[name] = _check_pin(supplied[name], pin)
    from . import contract as c
    require(c.ASSETS == ASSETS, 'Factory and protocol checkpoint assets differ')
    tone_protocol = regular(roots['tone_training'] / 'protocol.json')
    require(sha256(tone_protocol) == TONE_PROTOCOL, 'Exact tone-training parent protocol required')
    _BASE = _historical_imports(roots['adain'])
    before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    with torch.random.fork_rng(devices=[0]):
        original, _ = _BASE.load_trainable(paths['initializer'], device='cuda')
        original.eval().requires_grad_(False)
        base = deepcopy(original).eval().requires_grad_(False)
        heads = {}
        initial_base_digest = None
        for key, asset, completed in [('head0000', 'initial_composite', 0), ('head0050', 'stopped_composite', 50)]:
            saved = torch.load(paths[asset], map_location='cpu', weights_only=True, mmap=True)
            require(saved['format'] == 'codeformer-tone-control-composite-state-v1' and
                    saved['protocol_sha256'] == TONE_PROTOCOL and saved['completed_head_updates'] == completed and
                    saved['restorer_updates_this_run'] == 0 and saved['historical_restorer_updates'] == 50 and
                    saved['partial_microbatches'] == 0 and saved['head_optimizer_inflight'] is False and
                    saved['head_state_safe'] is True and saved['model_qualified'] is False,
                    'Actual committed historical composite state required')
            require(list(saved['base_model']) == list(original.state_dict()) and len(saved['base_model']) == 515 and
                    list(saved['head']) == list(HEAD_SHAPES) and all(
                        tuple(saved['head'][n].shape) == shape for n, shape in HEAD_SHAPES.items()),
                    'Historical base/head tensor key or shape differs')
            for state in (saved['base_model'], saved['head']):
                require(all(isinstance(q, torch.Tensor) and q.dtype == torch.float32 and torch.isfinite(q).all().item()
                            for q in state.values()), 'Historical FP32 finite tensor state required')
            bd, hd = tensor_state_digest(saved['base_model']), tensor_state_digest(saved['head'])
            require(bd == saved['base_model_tensor_digest'] == BASE_DIGEST and
                    hd == saved['head_tensor_digest'] == HEAD_DIGESTS[key], 'Exact actual base/head identities differ')
            if initial_base_digest is None: initial_base_digest = bd
            require(bd == initial_base_digest, 'Head states use different frozen restorers')
            if key == 'head0050':
                base.load_state_dict(saved['base_model'], strict=True)
                require(saved['counts'] == _read_json(tone_protocol)['call_caps'], 'Stopped state must contain final3090reservations')
            head = _head(torch)
            head.load_state_dict(saved['head'], strict=True)
            heads[key] = head
            del saved
    require(torch.equal(torch.get_rng_state(), before_cpu) and torch.equal(torch.cuda.get_rng_state(0), before_cuda),
            'Model construction RNG scope did not restore exactly')
    _frozen(original, 515); _frozen(base, 515)
    for h in heads.values(): _frozen(h, 4)
    require(state_digest(original) == ORIGINAL_DIGEST and state_digest(base) == BASE_DIGEST and
            all(state_digest(heads[k]) == v for k, v in HEAD_DIGESTS.items()), 'Loaded frozen identities differ')
    require(sum(p.numel() for p in heads['head0000'].parameters()) == 8257 and
            torch.count_nonzero(heads['head0000'].output.weight).item() ==
            torch.count_nonzero(heads['head0000'].output.bias).item() == 0, 'Actual initial head zero output differs')
    receipt = {'complete': True, 'asset_sha256': {k:v['sha256'] for k,v in ASSETS.items()},
        'canonical_tensor_state_sha256': {'original': ORIGINAL_DIGEST, 'base': BASE_DIGEST, **HEAD_DIGESTS},
        'frozen_original_and_base_tensors_each': 515, 'frozen_head_tensors_each': 4, 'head_parameters_each': 8257,
        'all_modules_FP32_CUDA_eval_requires_grad_false_grad_None': True,
        'construction_RNG_preserved_exact': True, 'historical_RNG_installed': False,
        'optimizer_constructed_or_state_installed': False, 'composite_container_optimizer_fields_ignored': True,
        'neural_forwards': 0, 'gradient_queries': 0, 'backward_calls': 0, 'optimizer_steps': 0,
        'unchanged_historical_code_and_controller_architecture': True}
    return original, base, heads, receipt


def _tensor(array, kind):
    torch = _runtime()
    import numpy as np
    require(isinstance(array, np.ndarray), 'Ordinary NumPy source required')
    if kind == 'rgb':
        require(array.shape == (256,256,3) and array.dtype == np.float32 and np.isfinite(array).all() and
                array.min() >= 0 and array.max() <= 1, 'Finite RGB256 float32[0,1] required')
        array = array.transpose(2,0,1)[None]
    elif kind == 'features':
        require(array.shape == (256,) and array.dtype == np.float32 and np.isfinite(array).all(), 'Exact256features required')
        array = array[None]
    else:
        require(kind == 'mask' and array.shape == (256,256) and array.dtype == np.bool_ and array.any(),
                'Nonempty exact boolean observed256support required')
        array = array[None,None]
    return torch.from_numpy(np.ascontiguousarray(array).copy()).to('cuda')


def infer_rgb(net, input_rgb):
    """Exactly one frozen CodeFormer invocation and unchanged native512 delivery."""
    torch = _runtime(); _frozen(net, 515)
    require(_BASE is not None, 'Load exact frozen parent sources first')
    value = _tensor(input_rgb, 'rgb')
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        native, logits, feature = net(_BASE.prepare_input(value), w=1.0, detach_16=True, adain=True)
        delivered = _BASE.delivered_rgb(native)
        require(native.dtype == logits.dtype == feature.dtype == delivered.dtype == torch.float32 and
                torch.isfinite(delivered).all().item() and torch.isfinite(logits).all().item() and
                torch.isfinite(feature).all().item(), 'Frozen native/logits/features/delivery precision or values differ')
        cpu = delivered[0].permute(1,2,0).detach().cpu()
    return cpu.numpy().copy()


def mean_target(rgb, observed):
    """The same CUDA FP32 observed mean used by the historical behavioral loss."""
    torch = _runtime()
    value, support = _tensor(rgb,'rgb'), _tensor(observed,'mask')
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        weights = value.new_tensor([.2126,.7152,.0722]).view(1,3,1,1)
        mask = support.to(dtype=torch.float32)
        mean = ((value * weights).sum(dim=1,keepdim=True) * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1)
        require(torch.isfinite(mean).all().item(), 'Finite behavioral observed mean required')
        return float(mean[0])


def head_response(head, features, base, input_rgb, observed, target):
    """One frozen head invocation; actual clipped observed RAW mean and scalar loss."""
    torch = _runtime(); _frozen(head, 4)
    require(type(target) in (int,float) and 0 <= target <= 1, 'Finite behavioral scalar target required')
    import numpy as np
    require(np.isfinite(target) and np.array_equal(features, features_numpy(input_rgb, observed)),
            'Head features must exactly match declared input-only data helper')
    ft, bt, it, mt = [_tensor(a,k) for a,k in [(features,'features'),(base,'rgb'),(input_rgb,'rgb'),(observed,'mask')]]
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
        delta = head(ft)
        require(delta.shape == (1,) and delta.dtype == torch.float32 and torch.isfinite(delta).all().item() and
                -.25 <= float(delta[0]) <= .25, 'One bounded actual FP32 scalar correction required')
        corrected = torch.where(mt, (bt + delta.view(1,1,1,1)).clamp(0,1), it)
        weights = corrected.new_tensor([.2126,.7152,.0722]).view(1,3,1,1)
        mask = mt.to(dtype=torch.float32)
        mean = ((corrected * weights).sum(dim=1,keepdim=True) * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1)
        target_tensor = mean.new_tensor([target])
        loss = (mean - target_tensor).square().mean()
        require(torch.isfinite(corrected).all().item() and torch.isfinite(mean).all().item() and torch.isfinite(loss).item(),
                'Finite corrected RAW/mean/loss required')
        cpu = corrected[0].permute(1,2,0).detach().cpu()
        result = {'delta':float(delta[0]), 'mean_y':float(mean[0]), 'loss':float(loss)}
    result['raw'] = cpu.numpy().copy()
    require(all(q.grad is None for q in head.parameters()), 'Frozen head unexpectedly owns gradients')
    return result


def frozen_state_receipt(original, base, heads):
    """Saved source identities; no neural call or claim to replay CUDA gradients."""
    _frozen(original,515);_frozen(base,515)
    require(set(heads) == set(HEAD_DIGESTS), 'Both exact frozen head states required')
    for head in heads.values():_frozen(head,4)
    actual = {'original':state_digest(original),'base':state_digest(base),
              **{k:state_digest(v) for k,v in heads.items()}}
    require(actual == {'original':ORIGINAL_DIGEST,'base':BASE_DIGEST,**HEAD_DIGESTS}, 'Frozen state changed during measurement')
    return {'complete':True,'canonical_tensor_state_sha256':actual,'all_eval_frozen_grad_None':True,
            'gradient_queries':0,'backward_calls':0,'optimizer_constructions_or_steps':0}
