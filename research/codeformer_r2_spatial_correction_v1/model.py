# R2-bound adaptation of the preserved source-only spatial-correction prototype.
# Latest user decision selects the exact R2 state as initializer and permanent
# reference. This file is UNTRAINED and its factory has not been called.
"""Source-only prototype; importing this module never imports Torch.

The caller owns a future verified manual runtime, all operation reservations,
training, checkpoints and quality gates. No executable training entry point,
optimizer, data loader, model download or automatic launch is provided here.
The factory constructs an UNTRAINED candidate only when explicitly called.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import weakref


SITES = ('32', '64', '128', '256')
CHANNELS = {'32': 256, '64': 256, '128': 128, '256': 128}
WIDTH = 32
INITIALIZATION_SEED = 20261021
R2_STATE_SHA256 = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
BASE_TENSORS = 515
HEAD_TENSORS = 36
HEAD_PARAMETERS = 168960
ARCHITECTURE_SOURCE_SHA256 = {
    'third_party.codeformer.codeformer_arch':
        'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb',
    'third_party.codeformer.vqgan_arch':
        'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf',
    'third_party.codeformer.registry':
        'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e',
}
_OWNERS = weakref.WeakKeyDictionary()
_OWNERS_LOCK = threading.Lock()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_head_shapes():
    """Exact prospective tensor schema, derived without Torch or a model."""
    result = {}
    for site in SITES:
        channels = CHANNELS[site]
        shapes = {
            'encoder.weight': (WIDTH, channels, 1, 1),
            'encoder.bias': (WIDTH,),
            'decoder.weight': (WIDTH, channels, 1, 1),
            'fused.weight': (WIDTH, channels, 1, 1),
            'coarse.weight': (WIDTH, 256, 1, 1),
            'spatial.weight': (WIDTH, WIDTH, 3, 3),
            'spatial.bias': (WIDTH,),
            'output.weight': (channels, WIDTH, 1, 1),
            'output.bias': (channels,),
        }
        require(sum(math.prod(shape) for shape in shapes.values()) == 129 * channels + 17472,
                'Per-head prospective parameter arithmetic differs')
        for name, shape in shapes.items():
            result['heads.' + site + '.' + name] = shape
    require(len(result) == HEAD_TENSORS and
            sum(math.prod(shape) for shape in result.values()) == HEAD_PARAMETERS,
            'Exact36-tensor/168960-parameter head schema differs')
    return result


def architecture_contract():
    """Descriptive source contract, not actual runtime evidence."""
    shapes = expected_head_shapes()
    return {
        'format': 'CodeFormer-R2-spatial-correction-prototype-v1',
        'status': 'UNRUN_PROTOTYPE_NOT_A_TRAINED_MODEL_OR_SEALED_PACKET',
        'base_r2_canonical515_sha256': R2_STATE_SHA256,
        'sites': list(SITES), 'width': WIDTH,
        'head_tensor_count': len(shapes),
        'head_parameters': sum(math.prod(shape) for shape in shapes.values()),
        'total_expected_tensor_count': BASE_TENSORS + HEAD_TENSORS,
        'stock': 'z=stock_SFT(e,d,w=1)',
        'head': 'h=output(GELU(spatial(GELU((E(e)+D(d)+Z(z)+resize(C(coarse32)))/2))))',
        'correction': 'z_new=z+tanh(h)*(z-d)',
        'residual_multiplier': 'q=1+tanh(h), mathematically0<q<2; FP32mayroundendpoints',
        'identity': 'zero output.weight/output.bias implies h=0; actual VM parity unexecuted',
        'encoder_and_coarse_detached': True,
        'decoder_and_fused_graph_not_detached': True,
        'coarse_projection_before_bilinear_resize': True,
        'base_all515_frozen_eval': True,
        'confidence_or_identity_probability': False,
        'training_or_inference_executed_by_import': False,
        'quality_gates_changed': False,
        'loss_LR_dataset_sampler_resources_sealed': False,
        'trained_model_or_useful_restoration_claim': False,
    }


def build_restorer(r2):
    """Explicit UNTRAINED factory; future guarded runtime must call it.

    This is the only dependency-import boundary. Base content is checked once
    here and again only when validate_invariants(check_r2_digest=True)
    is explicitly requested; the caller must bind that check to checkpoint and
    optimizer reservations in any future sealed worker.
    """
    import torch
    from torch.nn import functional as F

    require(isinstance(r2, torch.nn.Module), 'Explicit r2 CodeFormer module required')
    require(type(r2).__module__ == 'third_party.codeformer.codeformer_arch' and
            type(r2).__name__ == 'CodeFormer', 'Exact retained CodeFormer class required')
    for name, expected in ARCHITECTURE_SOURCE_SHA256.items():
        source_module = sys.modules.get(name)
        require(source_module is not None and getattr(source_module, '__file__', None),
                'Retained architecture dependency must already be loaded')
        require(hashlib.sha256(Path(source_module.__file__).read_bytes()).hexdigest() == expected,
                'Retained architecture source bytes differ: ' + name)
    require(r2.connect_list == list(SITES) and
            set(r2.fuse_convs_dict) == set(SITES) and
            r2.fuse_generator_block ==
            {'16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21},
            'R2 four-site SFT architecture required')
    require(all(not module._forward_pre_hooks and not module._forward_hooks
                for module in r2.modules()), 'Base must have a clean hook namespace')
    initial_state = r2.state_dict()
    require(len(initial_state) == len(dict(r2.named_parameters())) == BASE_TENSORS,
            'Exactly515 r2 parameter tensors required')
    first = next(r2.parameters())
    device = first.device
    require(device.type in ('cpu', 'cuda') and all(
        isinstance(value, torch.Tensor) and value.dtype == torch.float32 and
        value.device == device and torch.isfinite(value).all().item()
        for value in initial_state.values()), 'Finite colocated FP32 r2 tensors required')
    require(all(value.grad is None for value in r2.parameters()),
            'R2 base must have no retained gradient fields')

    def state_digest(state):
        digest = hashlib.sha256()
        for name, value in sorted(state.items()):
            value = value.detach().cpu().contiguous()
            header = json.dumps([name, str(value.dtype), list(value.shape)],
                                separators=(',', ':')).encode('utf-8')
            digest.update(len(header).to_bytes(8, 'big'))
            digest.update(header)
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    require(state_digest(initial_state) == R2_STATE_SHA256,
            'Exact retained R2 content required; untouched CodeFormer or another state is not accepted here')

    class SpatialHead(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            # Explicit CPU construction means only CPU RNG is consumed here.
            options = {'device': 'cpu', 'dtype': torch.float32}
            self.encoder = torch.nn.Conv2d(channels, WIDTH, 1, bias=True, **options)
            self.decoder = torch.nn.Conv2d(channels, WIDTH, 1, bias=False, **options)
            self.fused = torch.nn.Conv2d(channels, WIDTH, 1, bias=False, **options)
            self.coarse = torch.nn.Conv2d(256, WIDTH, 1, bias=False, **options)
            self.spatial = torch.nn.Conv2d(WIDTH, WIDTH, 3, padding=1, bias=True, **options)
            self.output = torch.nn.Conv2d(WIDTH, channels, 1, bias=True, **options)
            self.activation = torch.nn.GELU()
            torch.nn.init.zeros_(self.output.weight)
            torch.nn.init.zeros_(self.output.bias)

        def forward(self, encoder, decoder, fused, coarse):
            encoder = encoder.detach()
            coarse = coarse.detach()
            context = self.coarse(coarse)
            context = F.interpolate(context, size=encoder.shape[-2:], mode='bilinear',
                                    align_corners=False)
            mixed = (self.encoder(encoder) + self.decoder(decoder) +
                     self.fused(fused) + context) / 2.
            hidden = self.activation(mixed)
            logits = self.output(self.activation(self.spatial(hidden)))
            return fused + torch.tanh(logits) * (fused - decoder)

    class R2SpatialRestorer(torch.nn.Module):
        def __init__(self, frozen_base):
            super().__init__()
            self.base = frozen_base
            self.heads = torch.nn.ModuleDict({
                site: SpatialHead(CHANNELS[site]) for site in SITES})
            self.heads.to(device=device, dtype=torch.float32)
            self.base.eval().requires_grad_(False)
            self._forward_lock = threading.Lock()
            self.last_forward_sites = ()
            self.last_base_calls = 0

        def train(self, mode=True):
            super().train(mode)
            self.base.eval()
            return self

        def validate_invariants(self, *, check_r2_digest=False):
            require(type(check_r2_digest) is bool, 'Explicit digest-check boolean required')
            named = dict(self.named_parameters())
            shapes = expected_head_shapes()
            heads = {name: value for name, value in named.items() if name.startswith('heads.')}
            require(len(named) == len(self.state_dict()) == BASE_TENSORS + HEAD_TENSORS and
                    set(heads) == set(shapes) and sum(v.numel() for v in heads.values()) == HEAD_PARAMETERS,
                    'Actual551/36/168960 parameter/tensor census differs')
            require(all(tuple(heads[name].shape) == shape and
                        heads[name].dtype == torch.float32 and heads[name].device == device and
                        heads[name].requires_grad for name, shape in shapes.items()),
                    'Exact trainable FP32 spatial-head schema required')
            require(len(self.base.state_dict()) == BASE_TENSORS and all(
                not value.requires_grad and value.grad is None and value.dtype == torch.float32 and
                value.device == device for value in self.base.parameters()) and
                all(not module.training for module in self.base.modules()),
                'All515 r2 tensors must stay frozen/eval with grad=None')
            digest = state_digest(self.base.state_dict()) if check_r2_digest else None
            require(digest is None or digest == R2_STATE_SHA256,
                    'R2 content changed; retain failure, never replace initializer')
            return {'pass': True, 'base_tensors': BASE_TENSORS,
                    'head_tensors': len(heads), 'head_parameters': HEAD_PARAMETERS,
                    'base_digest_checked': check_r2_digest,
                    'base_canonical_sha256': digest,
                    'source_or_runtime_integrity_not_quality': True}

        def forward(self, sample, *, w=1., detach_16=True, adain=True):
            require(w == 1. and detach_16 is True and adain is True,
                    'Fixed w1/detach16/AdaIN-on policy required')
            require(isinstance(sample, torch.Tensor) and sample.shape == (1, 3, 512, 512) and
                    sample.dtype == torch.float32 and sample.device == device and
                    not sample.requires_grad and torch.isfinite(sample).all().item() and
                    sample.min().item() >= -1. and sample.max().item() <= 1.,
                    'Finite fixed singleton FP32 prepared512 sample required')
            require(not torch.is_inference_mode_enabled(),
                    'Ordinary no_grad or autograd runtime required, not inference_mode')
            self.validate_invariants(check_r2_digest=False)
            require(self._forward_lock.acquire(blocking=False),
                    'Nested/concurrent candidate forward refused')
            handles = []
            pending = {}
            completed = []
            coarse = None
            thread_id = threading.get_ident()
            base_calls = 0
            self.last_forward_sites = ()
            self.last_base_calls = 0

            def guard_thread():
                require(threading.get_ident() == thread_id,
                        'Concurrent/foreign-thread base or SFT call refused')

            def base_pre_hook(module, args):
                nonlocal base_calls
                guard_thread()
                require(module is self.base and base_calls == 0 and len(args) == 1 and
                        args[0] is sample, 'One explicit nonnested base call required')
                base_calls += 1

            def pre_hook(site):
                def capture(module, args):
                    nonlocal coarse
                    guard_thread()
                    require(base_calls == 1 and len(args) == 3 and args[2] == 1. and
                            len(completed) < len(SITES) and site == SITES[len(completed)] and not pending,
                            'Exact one ordered SFT call per site required')
                    encoder, decoder, _ = args
                    shape = (1, CHANNELS[site], int(site), int(site))
                    require(encoder.shape == decoder.shape == shape and
                            encoder.dtype == decoder.dtype == torch.float32 and
                            encoder.device == decoder.device == device and
                            not encoder.requires_grad and torch.isfinite(encoder).all().item() and
                            torch.isfinite(decoder).all().item(),
                            'Exact detached encoder and connected decoder arguments required')
                    if site == '32':
                        require(coarse is None, 'Capture coarse32 exactly once')
                        coarse = encoder.detach()
                    require(coarse is not None and coarse.shape == (1, 256, 32, 32),
                            'First-site detached coarse32 context required')
                    pending[site] = (encoder.detach(), decoder)
                return capture

            def post_hook(site):
                def apply(module, args, fused):
                    guard_thread()
                    require(set(pending) == {site} and len(completed) < len(SITES) and
                            site == SITES[len(completed)],
                            'Paired SFT capture missing or out of order')
                    encoder, decoder = pending.pop(site)
                    require(isinstance(fused, torch.Tensor) and fused.shape == encoder.shape and
                            fused.dtype == torch.float32 and fused.device == device and
                            torch.isfinite(fused).all().item(), 'Finite stock SFT result required')
                    corrected = self.heads[site](encoder, decoder, fused, coarse)
                    require(corrected.shape == fused.shape and corrected.dtype == torch.float32 and
                            torch.isfinite(corrected).all().item(), 'Finite unchanged-geometry correction required')
                    completed.append(site)
                    return corrected
                return apply

            try:
                require(all(not module._forward_pre_hooks and not module._forward_hooks
                            for module in self.base.modules()), 'Clean base hook namespace required')
                handles.append(self.base.register_forward_pre_hook(base_pre_hook))
                for site in SITES:
                    module = self.base.fuse_convs_dict[site]
                    handles.append(module.register_forward_pre_hook(pre_hook(site)))
                    handles.append(module.register_forward_hook(post_hook(site)))
                with torch.autocast(device_type=device.type, enabled=False):
                    result = self.base(sample, w=w, detach_16=detach_16, adain=adain)
                require(base_calls == 1 and tuple(completed) == SITES and not pending,
                        'All four corrections must execute in one complete base call')
                require(isinstance(result, tuple) and len(result) == 3 and
                        result[0].shape == (1, 3, 512, 512) and
                        result[0].dtype == torch.float32 and torch.isfinite(result[0]).all().item(),
                        'Finite native512 plus r2 logits/LQ result required')
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

    # One wrapper owns this base. It cannot be shared with another prototype
    # wrapper; the weak references do not keep otherwise unused models alive.
    with _OWNERS_LOCK:
        owner = _OWNERS.get(r2)
        require(owner is None or owner() is None, 'R2 base already owned by another live wrapper')
        # All Conv parameters start on CPU. Preserve the caller's CPU RNG exactly;
        # moving fixed tensors to the base device consumes no CUDA random draws.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(INITIALIZATION_SEED)
            candidate = R2SpatialRestorer(r2)
        candidate.validate_invariants(check_r2_digest=True)
        require(all(torch.count_nonzero(head.output.weight).item() == 0 and
                    torch.count_nonzero(head.output.bias).item() == 0
                    for head in candidate.heads.values()), 'Exact zero final projections required')
        _OWNERS[r2] = weakref.ref(candidate)
    return candidate
