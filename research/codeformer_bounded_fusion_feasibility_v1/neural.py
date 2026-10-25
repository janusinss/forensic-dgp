"""Prospective manual-L4, zero-update bounded-fusion measurement.

Import and pure helpers are model-free. ``execute`` is the only neural entry;
its independent Linux/home/tmux guard and caller guard precede heavy imports.
This module is unsealed preparation, not an instruction to launch a VM job.
"""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re


PACKAGE_DIR = 'cctv_codeformer_bounded_fusion_feasibility_v1_vm'
ORIGINAL_SHA = '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'
STOPPED_SHA = '133c5a1e15e069bdd2abda4ebf818d3f4ff59b966677762753e894ff42e2583d'
STOPPED_STATE_SHA = '51605ee7908c40a457e41ac03c614211e8028cc0386b85d13aa59a3d19a182eb'
PARENT_PROTOCOL_SHA = '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'
SITES = {'32': (256, 32, 32), '64': (256, 64, 64),
         '128': (128, 128, 128), '256': (128, 256, 256)}
MODES = ('original', 'stopped', 'rollback', 'bounded_constant', 'bounded_fixed')
CAPS = {'codeformer_forwards': 170, 'fusion_block_evaluations': 960,
        'hq_teacher_forwards': 0, 'behavior_teacher_forwards': 0, 'lpips_forwards': 0,
        'gradient_queries': 0, 'backward_calls': 0, 'optimizer_attempts': 0, 'optimizer_updates': 0}
GATE = {'q10': 0.21007671428419458, 'q90': 0.5546973290483728,
        'beta': .1, 'epsilon': 1e-6, 'gate_floor': .25, 'gate_ceiling': 1., 'central_support': [32, 224]}
CALLBACK_NAMES = ('require', 'clock', 'reserve', 'write_json', 'write_array',
                  'write_png', 'guard')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def callback_api(callbacks):
    result = {}
    for name in CALLBACK_NAMES:
        value = callbacks.get(name) if isinstance(callbacks, dict) else getattr(callbacks, name, None)
        _require(callable(value), 'Missing callable callback: ' + name)
        result[name] = value
    return result


def manual_root(root, *, system=None, home=None, environment=None):
    system = platform.system() if system is None else system
    environment = os.environ if environment is None else environment
    _require(system == 'Linux', 'Neural measurement is manual Linux L4 only')
    _require(bool(environment.get('TMUX')), 'Human-opened tmux is required')
    root = Path(root).resolve(strict=True)
    expected = (Path.home() if home is None else Path(home)) / 'forensic-dgp' / PACKAGE_DIR
    _require(not expected.is_symlink() and root == expected.resolve(strict=True),
             'Use the exact new feasibility directory')
    return root


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_environment(environment):
    _require(environment.get('CUBLAS_WORKSPACE_CONFIG') in (None, ':4096:8'),
             'Unexpected CUBLAS deterministic workspace policy')
    environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def inside(root, relative):
    _require(isinstance(relative, str) and bool(relative) and '\\' not in relative,
             'Expected a nonempty portable relative path')
    p = Path(relative)
    _require(not p.is_absolute() and '..' not in p.parts and ':' not in relative,
             'Relative path may not escape the installed packet')
    out = (root / p).resolve(strict=True)
    _require(out.is_relative_to(root) and out.is_file() and not (root / p).is_symlink(),
             'Input must be a regular installed file')
    return out


@contextmanager
def temporary_attribute(owner, name, replacement):
    """Restore the exact attribute namespace, including a class-bound method."""
    owned = name in vars(owner)
    previous = vars(owner).get(name)
    try:
        setattr(owner, name, replacement)
        yield
    finally:
        if owned:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


def observed_gate(rgb8, gate=GATE):
    """CPU input arithmetic only; no metadata/clean target is accepted."""
    import numpy as np
    _require(isinstance(rgb8, np.ndarray) and rgb8.dtype == np.uint8 and
             rgb8.shape == (256, 256, 3), 'Gate needs exact RGB256 uint8')
    _require(gate == GATE, 'Unreviewed gate/bound parameter change')
    rgb = rgb8.astype(np.float64) / 255.
    y = rgb @ np.array([.2126, .7152, .0722], dtype=np.float64)
    z = float(y[32:224, 32:224].mean())
    t = min(1., max(0., (gate['q90'] - z) / (gate['q90'] - gate['q10'])))
    g = gate['gate_floor'] + (gate['gate_ceiling'] - gate['gate_floor']) * t * t * (3 - 2 * t)
    _require(math.isfinite(z) and math.isfinite(g) and .25 <= g <= 1.,
             'Invalid observed-input gate')
    return {'central192_sRGB_mean_luminance': z, 'gate': g}


def bounded_numpy_reference(base, other, g, site):
    """Model-free canary replay specification, never a restoration fallback."""
    import numpy as np
    _require(site in SITES and isinstance(base, np.ndarray) and isinstance(other, np.ndarray) and
             base.dtype == other.dtype == np.float32 and base.shape == other.shape == (1, *SITES[site]),
             'Canary replay requires the exact native FP32 feature shape')
    _require(np.isfinite(base).all() and np.isfinite(other).all() and
             isinstance(g, (float, int)) and math.isfinite(g) and .25 <= g <= 1.,
             'Invalid canary feature/gate')
    d = np.float32(GATE['beta']) * np.sqrt(
        np.square(base).mean(axis=(-2, -1), keepdims=True, dtype=np.float32) + np.float32(GATE['epsilon']))
    used = np.tanh((other - base) / d) * d
    used = used * np.float32(g)
    _require(np.isfinite(d).all() and np.isfinite(used).all(), 'Nonfinite canary arithmetic')
    return np.ascontiguousarray(base + used), np.ascontiguousarray(used), np.ascontiguousarray(d)


def expected_plan(case_ids, canary_ids):
    _require(len(case_ids) == 32 and len(set(case_ids)) == 32, 'Exactly32 unique cases required')
    _require(len(canary_ids) == 2 and len(set(canary_ids)) == 2 and
             set(canary_ids) <= set(case_ids), 'Exactly2 existing canaries required')
    plan = [(case, mode, 'paths') for case in case_ids for mode in MODES]
    plan += [(case, mode, 'repeats') for case in canary_ids for mode in ('original', 'stopped')]
    plan += [(case, mode, 'zero_delta') for case in canary_ids
             for mode in ('bounded_constant', 'bounded_fixed')]
    plan += [(case, 'native_identity', 'identity') for case in canary_ids]
    return plan


def fusion_calls_for(mode, section):
    return 8 if mode in ('bounded_constant', 'bounded_fixed', 'native_identity') else 4


def validate_protocol(p):
    _require(p.get('call_caps') == CAPS, 'Finite call caps differ')
    _require(p.get('gate') == GATE, 'Gate formulation/constants differ')
    _require(p.get('seed') == 20261015, 'Frozen seed differs')
    cases = p.get('cases', [])
    _require(len(cases) == 32, 'Exactly32 selected TRAIN observations required')
    for row in cases:
        _require(re.fullmatch(r'[A-Za-z0-9_.-]+', row.get('id', '')) is not None,
                 'Unsafe case identifier')
        _require(type(row.get('update')) is int and 1 <= row['update'] <= 50 and
                 type(row.get('slot')) is int and row['slot'] in (1, 2, 3, 4),
                 'Case is outside retained first50 TRAIN schedule')
        for name in ('input_path', 'input_sha256', 'target512_path', 'target512_sha256',
                     'target256_path', 'target256_sha256', 'reference_id'):
            _require(isinstance(row.get(name), str) and bool(row[name]), 'Case lacks ' + name)
    _require(len(p.get('calibration_inputs', [])) == 200, 'Exactly200 calibration inputs required')
    plan = expected_plan([r['id'] for r in cases], p.get('canary_ids', []))
    _require(len(plan) == CAPS['codeformer_forwards'] and sum(fusion_calls_for(m, s) for _, m, s in plan) ==
             CAPS['fusion_block_evaluations'], 'Prospective call ledger mismatch')
    for name, expected in [('original', ORIGINAL_SHA), ('stopped', STOPPED_SHA)]:
        asset = p.get('assets', {}).get(name, {})
        _require(asset.get('sha256') == expected and type(asset.get('bytes')) is int and
                 asset['bytes'] > 0, 'Unexpected immutable ' + name + ' asset')
    return plan


def execute(root, p, original_path, stopped_path, callbacks):
    """Run only after independent installation/asset guard; no optimizer exists.

    All artifact paths passed to callbacks are relative to the caller's new
    output directory. Callbacks enforce namespace, byte/file caps, exclusive
    writes, fsync, process supervision and failure export. See NEURAL_INTERFACE.
    """
    cb = callback_api(callbacks)
    root = manual_root(root)  # Must precede torch/model/architecture imports.
    cb['guard'](root, p)
    require = cb['require']
    plan = validate_protocol(p)
    cb['clock']('initialization')
    for name, path in [('original', original_path), ('stopped', stopped_path)]:
        require(not Path(path).is_symlink(), 'Neural asset path must not be a symlink')
        path = Path(path).resolve(strict=True)
        asset = p['assets'][name]
        require(path.is_file() and not Path(path).is_symlink() and
                path.stat().st_size == asset['bytes'] and file_sha(path) == asset['sha256'],
                'Immutable neural asset differs: ' + name)
    # Heavy dependencies are deliberately unavailable on the import/test path.
    deterministic_environment(os.environ)  # Before any CUDA/BLAS initialization.
    import numpy as np
    import torch
    import resource
    from PIL import Image
    from research.codeformer_adaptation_v2 import model
    import third_party.codeformer.codeformer_arch as architecture
    require(Path(__file__).resolve() == root / 'research/codeformer_bounded_fusion_feasibility_v1/neural.py',
            'Execute the installed, reviewed neural source')
    require(Path(model.__file__).resolve() == root / 'research/codeformer_adaptation_v2/model.py' and
            Path(architecture.__file__).resolve() == root / 'third_party/codeformer/codeformer_arch.py',
            'No source fallback outside the verified packet')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            'NVIDIA L4' in torch.cuda.get_device_name(0), 'Existing single L4 is required')
    require(not torch.cuda.is_initialized() or torch.cuda.memory_reserved() == 0,
            'Refuse pre-existing allocations in the measurement process')
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(p['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    runtime = {'torch': torch.__version__, 'numpy': np.__version__,
               'CUDA': torch.version.cuda, 'GPU': torch.cuda.get_device_name(0),
               'capability': list(torch.cuda.get_device_capability(0)),
               'autocast': bool(torch.is_autocast_enabled('cuda')),
               'TF32': bool(torch.backends.cuda.matmul.allow_tf32),
               'cudnn_TF32': bool(torch.backends.cudnn.allow_tf32),
               'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
               'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']}
    require(not runtime['autocast'] and not runtime['TF32'] and not runtime['cudnn_TF32'] and
            runtime['deterministic_algorithms'], 'Actual deterministic FP32 runtime policy differs')
    expected_runtime = p.get('runtime_expected', {})
    require(set(expected_runtime) == {'torch', 'numpy', 'CUDA', 'GPU', 'capability'} and
            all(runtime[k] == v for k, v in expected_runtime.items()), 'Frozen neural runtime differs')
    cb['write_json']('runtime.json', runtime)
    counts = {name: 0 for name in CAPS}
    observations, baselines, results = {}, {}, []
    original = stopped = None
    initial_states = None
    stage = 'initialization'

    def clock():
        return cb['clock'](stage)

    def reserve(kind, context):
        clock()
        require(kind in counts and counts[kind] < CAPS[kind], 'Neural call cap reached: ' + kind)
        cb['reserve'](kind, context)
        counts[kind] += 1

    def array_hash(value):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

    def tensor_array(value):
        require(value.dtype == torch.float32 and torch.isfinite(value).all().item(),
                'Captured tensor must remain finite FP32')
        return value.detach().contiguous().cpu().numpy().copy()

    def tensor_record(value):
        a = tensor_array(value)
        return {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': array_hash(a)}

    def open_input(row):
        path = inside(root, row['input_path'])
        require(file_sha(path) == row['input_sha256'], 'Exact retained TRAIN PNG differs')
        with Image.open(path) as im:
            require(im.mode == 'RGB' and im.size == (256, 256), 'Exact RGB256 TRAIN geometry required')
            a = np.asarray(im).copy()
        if 'input256_rgb8_sha256' in row:
            require(array_hash(a) == row['input256_rgb8_sha256'], 'TRAIN RGB8 pixels differ')
        f = np.ascontiguousarray(a.astype(np.float32) / np.float32(255))
        if 'input256_float32_sha256' in row:
            require(array_hash(f) == row['input256_float32_sha256'], 'TRAIN float32 pixels differ')
        return a, f

    def states(label, baseline=None):
        clock()
        require(all(v.grad is None for net in (original, stopped) for v in net.parameters()),
                'Measurement must never populate .grad')
        require(all(not v.requires_grad for net in (original, stopped) for v in net.parameters()) and
                not original.training and not stopped.training, 'Inference state changed')
        value = {name: model.state_digest(net) for name, net in [('original', original), ('stopped', stopped)]}
        if baseline is not None:
            require(value == baseline, 'Read-only model state changed at ' + label)
        cb['write_json']('state_checks/' + label + '.json',
                         {'stage': label, 'states': value, 'all_grad_none': True, 'counts': dict(counts)})
        return value

    def channel_list(tensor):
        require(torch.isfinite(tensor).all().item(), 'Nonfinite captured channel statistic')
        return tensor.detach().float().reshape(-1).cpu().tolist()

    def bound_arithmetic(base, other, g, context, feature_capture=False):
        require(base.dtype == other.dtype == torch.float32 and base.shape == other.shape and
                tuple(base.shape[1:]) == SITES[context['site']], 'Fusion geometry/dtype differs')
        require(torch.isfinite(base).all().item() and torch.isfinite(other).all().item(),
                'Nonfinite unmodified fusion result')
        # Exact reviewed FP32 sequence. Reuse ratio storage only after needed
        # saturation reductions, bounding named arithmetic tensors independently
        # of block-internal convolution workspaces.
        scale = (base.square().mean(dim=(-2, -1), keepdim=True) +
                 torch.tensor(GATE['epsilon'], device=base.device, dtype=torch.float32)).sqrt()
        scale = scale * torch.tensor(GATE['beta'], device=base.device, dtype=torch.float32)
        difference = other - base
        require(torch.isfinite(scale).all().item() and (scale > 0).all().item() and
                torch.isfinite(difference).all().item(), 'Nonfinite pre-tanh difference/scale')
        ratio = difference / scale
        require(torch.isfinite(ratio).all().item(), 'Nonfinite normalized fusion difference')
        ratio.tanh_()
        saturation = (ratio.abs() >= .95).float().mean(dim=(-2, -1), keepdim=True)
        ratio.mul_(scale)
        ratio.mul_(torch.tensor(g, device=base.device, dtype=torch.float32))
        used = ratio
        require(torch.isfinite(scale).all().item() and (scale > 0).all().item() and
                torch.isfinite(used).all().item(), 'Nonfinite bound arithmetic')
        maximum = used.abs().amax(dim=(-2, -1), keepdim=True)
        # This is a local FP32 arithmetic sanity check, not a quality tolerance.
        limit = scale * torch.tensor(g, device=base.device, dtype=torch.float32)
        tolerance = 8 * torch.finfo(torch.float32).eps * limit + torch.finfo(torch.float32).tiny
        require((maximum <= limit + tolerance).all().item(), 'Local feature cap violated')
        stats = {'site': context['site'], 'gate': g, 'dtype': 'torch.float32',
                 'D': channel_list(scale), 'difference_absmax': channel_list(difference.abs().amax((-2, -1))),
                 'difference_RMS': channel_list(difference.square().mean((-2, -1)).sqrt()),
                 'used_delta_absmax': channel_list(maximum),
                 'used_delta_RMS': channel_list(used.square().mean((-2, -1)).sqrt()),
                 'tanh_abs_ge_095_fraction': channel_list(saturation), 'nonfinite_count': 0,
                 'local_cap_with_FP32_roundoff_pass': True}
        if feature_capture:
            prefix = context['directory'] + '/features/' + context['site']
            stats['arrays'] = {name: cb['write_array'](prefix + '/' + name + '.npy', tensor_array(t))
                              for name, t in [('original', base), ('stopped', other), ('used_delta', used)]}
        return base + used, stats

    @contextmanager
    def capture_and_fusion(net, mode, section, case, directory, g):
        captured = {'encoder_skips': {}, 'fusion_stats': []}
        hooks = []
        forward_methods = {label: {site: module.forward for site, module in item.fuse_convs_dict.items()}
                           for label, item in [('original', original), ('stopped', stopped)]}
        own_method_namespace = {site: ('forward' in vars(net.fuse_convs_dict[site])) for site in SITES}

        def encoder_hook(site):
            def hook(module, args, output):
                require(tuple(output.shape) == (1, *SITES[site]), 'Encoder skip geometry differs')
                require(site not in captured['encoder_skips'], 'Repeated encoder skip capture')
                captured['encoder_skips'][site] = tensor_record(output)
            return hook

        def generator_pre(module, args):
            require('pre_generator' not in captured, 'Repeated first generator call')
            value = args[0]
            require(tuple(value.shape) == (1, 256, 16, 16), 'AdaIN/code feature geometry differs')
            captured['pre_generator'] = tensor_record(value)
            mean, std = architecture.calc_mean_std(value)
            captured['pre_generator_mean'] = tensor_record(mean)
            captured['pre_generator_std'] = tensor_record(std)

        lookup = net.quantize.get_codebook_feat
        def lookup_capture(indices, shape):
            require('tokens' not in captured, 'Repeated codebook lookup')
            require(indices.dtype == torch.int64 and indices.numel() == 256, 'Hard-selected token dtype/shape differs')
            captured['tokens'] = indices.detach().cpu().numpy().reshape(256).copy()
            return lookup(indices, shape)

        native_softmax = architecture.F.softmax
        def softmax_capture(value, *args, **kwargs):
            output = native_softmax(value, *args, **kwargs)
            if tuple(value.shape) == (1, 256, 1024):
                require('probabilities_dtype' not in captured, 'Repeated native token softmax')
                captured['probabilities_dtype'] = str(output.dtype)
            return output

        def fusion_wrapper(site):
            def call(e, x, w=1.):
                require(w == 1. and e.dtype == x.dtype == torch.float32 and
                        tuple(e.shape) == tuple(x.shape) == (1, *SITES[site]),
                        'Fusion inputs/weight differ')
                require(not e.requires_grad and not x.requires_grad, 'No gradient tape permitted')
                local = {'case_id': case['id'], 'mode': mode, 'section': section,
                         'site': site, 'directory': directory}
                def block(label, role):
                    reserve('fusion_block_evaluations', {**local, 'weights': label, 'role': role})
                    return forward_methods[label][site](e, x, w)
                if mode in ('original', 'stopped'):
                    label = mode
                    captured['fusion_stats'].append({'site': site, 'route': label, 'block_evaluations': 1})
                    return block(label, 'native')
                if mode == 'rollback':
                    captured['fusion_stats'].append({'site': site, 'route': 'original', 'gate': 0., 'block_evaluations': 1})
                    return block('original', 'rollback')
                first = block('original', 'base')
                second = block('original' if section == 'zero_delta' else 'stopped', 'delta_source')
                if mode == 'native_identity':
                    captured['fusion_stats'].append({'site': site, 'route': 'native_identity', 'block_evaluations': 2})
                    return second
                if section == 'zero_delta':
                    require(torch.equal(first, second), 'Original-original local delta is not exactly zero')
                output, stats = bound_arithmetic(first, second, g, local,
                    feature_capture=(section == 'paths' and case['id'] in p['canary_ids'] and site in ('32', '64')))
                stats['block_evaluations'] = 2
                captured['fusion_stats'].append(stats)
                return output
            return call

        try:
            for site in SITES:
                index = net.fuse_encoder_block[site]
                hooks.append(net.encoder.blocks[index].register_forward_hook(encoder_hook(site)))
            hooks.append(net.generator.blocks[0].register_forward_pre_hook(generator_pre))
            with ExitStack() as stack:
                stack.enter_context(temporary_attribute(net.quantize, 'get_codebook_feat', lookup_capture))
                stack.enter_context(temporary_attribute(architecture.F, 'softmax', softmax_capture))
                for site in SITES:
                    stack.enter_context(temporary_attribute(net.fuse_convs_dict[site], 'forward', fusion_wrapper(site)))
                yield captured
        finally:
            for hook in reversed(hooks):
                hook.remove()
            require(all(('forward' in vars(net.fuse_convs_dict[s])) == own_method_namespace[s] for s in SITES),
                    'Temporary fusion namespace was not restored')

    def forward(case, mode, section):
        net = original if mode == 'original' or section == 'zero_delta' else stopped
        value = observations[case['id']]
        g = (value['gate']['gate'] if mode == 'bounded_fixed' else (0. if mode == 'rollback' else 1.))
        directory = section + '/' + case['id'] + '/' + mode
        reserve('codeformer_forwards', {'case_id': case['id'], 'mode': mode, 'section': section})
        tensor = torch.from_numpy(value['float32']).permute(2, 0, 1).unsqueeze(0).to('cuda', torch.float32)
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            with capture_and_fusion(net, mode, section, case, directory, g) as captured:
                native, logits, latent = net(model.prepare_input(tensor), w=1., detach_16=True, adain=True)
            require(not torch.is_grad_enabled() and torch.is_inference_mode_enabled() and
                    not torch.is_autocast_enabled('cuda') and native.dtype == logits.dtype == latent.dtype == torch.float32,
                    'Actual forward must remain deterministic inference-only FP32')
            require(captured.get('probabilities_dtype') == 'torch.float32' and
                    len(captured['encoder_skips']) == 4 and len(captured['fusion_stats']) == 4 and
                    'tokens' in captured and 'pre_generator' in captured, 'Incomplete actual path capture')
            content = {name: captured[name] for name in ('encoder_skips', 'pre_generator',
                       'pre_generator_mean', 'pre_generator_std')}
            content['logits'] = tensor_record(logits)
            content['latent'] = tensor_record(latent)
            content['selected_tokens_sha256'] = array_hash(captured['tokens'])
            native_rgb = np.ascontiguousarray(((native.float() + 1) / 2)[0].permute(1, 2, 0).cpu().numpy())
            raw = np.ascontiguousarray(model.delivered_rgb(native)[0].permute(1, 2, 0).cpu().numpy())
        require(native_rgb.shape == (512, 512, 3) and raw.shape == (256, 256, 3) and
                native_rgb.dtype == raw.dtype == np.float32 and np.isfinite(native_rgb).all() and
                np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1, 'Invalid native/delivered output')
        if section == 'paths' and mode == 'stopped':
            baselines[case['id'], 'content'] = content
        if mode in ('rollback', 'bounded_constant', 'bounded_fixed', 'native_identity') and section != 'zero_delta':
            require(content == baselines[case['id'], 'content'], 'Stopped content/tokens/moments changed across fusion modes')
        output_hashes = {'native512': array_hash(native_rgb), 'raw256': array_hash(raw),
                         'tokens': content['selected_tokens_sha256']}
        if section == 'paths' and mode in ('original', 'stopped'):
            baselines[case['id'], mode] = {'outputs': output_hashes, 'content': content}
        if section in ('repeats', 'identity', 'zero_delta'):
            expected = 'original' if section == 'zero_delta' else ('stopped' if section == 'identity' else mode)
            require({'outputs': output_hashes, 'content': content} == baselines[case['id'], expected],
                    'Native repeat/initialization/identity parity failed')
        png = np.floor(raw * np.float32(255)).astype(np.uint8)
        artifacts = {
            'native512': cb['write_array'](directory + '/native512.npy', native_rgb),
            'raw256': cb['write_array'](directory + '/raw256.npy', raw),
            'PNG': cb['write_png'](directory + '/native.png', png),
            'tokens': cb['write_array'](directory + '/actual_tokens.npy', captured['tokens']),
        }
        record = {'case_id': case['id'], 'mode': mode, 'section': section,
                  'input_path': case['input_path'], 'input_sha256': case['input_sha256'],
                  'reference_id': case['reference_id'], 'gate_observed_input': value['gate'],
                  'applied_gate': g if mode in ('rollback', 'bounded_constant', 'bounded_fixed') else None,
                  'content': content, 'content_exact_to_stopped': section != 'zero_delta' and mode != 'original',
                  'actual_dtypes': {'native': str(native.dtype), 'logits': str(logits.dtype),
                                   'probabilities': captured['probabilities_dtype'], 'autocast_enabled': False},
                  'artifacts': artifacts, 'pixel_hashes': output_hashes,
                  'targets_for_independent_metrics_only': {k: case[k] for k in ('target512_path', 'target512_sha256',
                                                                            'target256_path', 'target256_sha256')},
                  'neural_state': 'original' if net is original else 'stopped',
                  'read_only_model_state_bindings': initial_states,
                  'elapsed_seconds': clock(), 'model_qualified': False}
        cb['write_json'](directory + '/fusion_stats.json', captured['fusion_stats'])
        record_receipt = cb['write_json'](directory + '/record.json', record)
        # Keep final metadata bounded: evidence is in hash-bound per-path files,
        # not repeated wholesale inside run_outputs and the worker receipt.
        results.append({'case_id': case['id'], 'mode': mode, 'section': section, 'record': record_receipt})
        del tensor, native, logits, latent, captured, native_rgb, raw, png
        clock()

    try:
        # Targets are fingerprinted, never decoded or routed into the network.
        # The independent return metrics module owns all target-based metrics.
        calibration = []
        seen = set()
        for row in p['calibration_inputs']:
            ident = (row['update'], row['slot'])
            require(ident not in seen, 'Repeated calibration observation')
            seen.add(ident)
            pixels, _ = open_input(row)
            gate = observed_gate(pixels, p['gate'])
            calibration.append({'update': row['update'], 'slot': row['slot'],
                                'input_sha256': row['input_sha256'], **gate})
            clock()
        require(seen == {(u, s) for u in range(1, 51) for s in range(1, 5)}, 'Calibration is not exact first200')
        q10, q90 = np.quantile([r['central192_sRGB_mean_luminance'] for r in calibration], [.1, .9], method='linear')
        require(abs(float(q10) - GATE['q10']) <= 1e-12 and abs(float(q90) - GATE['q90']) <= 1e-12,
                'Observed-only calibration quantiles differ')
        cb['write_json']('calibration.json', {'inputs': calibration, 'q10': float(q10), 'q90': float(q90),
                         'profile_or_target_used_in_gate': False, 'roundoff_only_abs_bound': 1e-12})
        for case in p['cases']:
            pixels, f = open_input(case)
            for key in ('target512', 'target256'):
                target = inside(root, case[key + '_path'])
                require(file_sha(target) == case[key + '_sha256'], 'Metric-only clean target fingerprint differs')
            if 'source_path' in case or 'source_sha256' in case:
                require('source_path' in case and 'source_sha256' in case and
                        file_sha(inside(root, case['source_path'])) == case['source_sha256'],
                        'Genuine TRAIN source fingerprint differs')
            observations[case['id']] = {'float32': f, 'gate': observed_gate(pixels, p['gate'])}
        original, census = model.load_trainable(original_path, device='cpu')
        stopped = deepcopy(original)
        loaded = torch.load(Path(stopped_path), map_location='cpu', weights_only=True)
        require(loaded.get('format') == 'codeformer-fp32-comparison-state-v1' and
                loaded.get('protocol_sha256') == PARENT_PROTOCOL_SHA and loaded.get('completed_updates') == 50,
                'Use the actual retained FP32 stopped checkpoint')
        stopped.load_state_dict(loaded['model'], strict=True)
        del loaded  # No optimizer object or resumed state is ever constructed.
        for net in (original, stopped):
            net.eval().float().requires_grad_(False).to('cuda')
        require(model.state_digest(stopped) == STOPPED_STATE_SHA, 'Stopped model-state digest differs')
        require(all(not v.requires_grad for net in (original, stopped) for v in net.parameters()),
                'Every inference parameter must be frozen')
        initial_states = states('initial')
        cb['write_json']('initialization.json', {'census_from_original_loader': census, 'states': initial_states,
                         'asset_sha256': {n: p['assets'][n]['sha256'] for n in ('original', 'stopped')},
                         'inference_frozen_tensor_counts': {name: sum(1 for _ in net.parameters())
                                                           for name, net in [('original', original), ('stopped', stopped)]},
                         'optimizer_constructed': False, 'elapsed_seconds': clock()})
        cases = {c['id']: c for c in p['cases']}
        stage = 'measurement'
        clock()
        for index, (identifier, mode, section) in enumerate(plan, 1):
            forward(cases[identifier], mode, section)
            if index % 5 == 0 or index == len(plan):
                print(json.dumps({'feasibility_forward': index, 'of': len(plan), 'counts': counts}), flush=True)
        states('terminal', initial_states)
        require(counts == CAPS, 'Successful endpoint call counts differ')
        result = {'complete': True, 'counts': counts, 'observations': len(p['cases']),
                  'results': results, 'all_content_and_canaries_pass': True, 'state_before_after_equal': True,
                  'parameter_grad_fields_all_none': True, 'content_preservation_verified': True,
                  'repeat_zero_identity_canaries_pass': True,
                  'elapsed_seconds': clock(), 'optimizer_updates': 0, 'gradient_queries': 0,
                  'memory': {'CUDA_current_reserved_bytes': int(torch.cuda.memory_reserved()),
                             'CUDA_peak_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                             'Linux_host_peak_RSS_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024},
                  'model_qualified': False, 'training_authorized': False,
                  'interpretation': 'Disposable stopped-state TRAIN intervention; not a trained candidate or exposure detector.'}
        cb['write_json']('run_outputs.json', result)
        return result
    except BaseException as error:
        # Main wrapper owns robust terminal failure/export. Preserve this best-
        # effort neural boundary record without hiding the original exception.
        try:
            if original is not None and stopped is not None and initial_states is not None:
                states('failure', initial_states)
            cb['write_json']('neural_failure.json', {'complete': False, 'counts': counts,
                'completed_outputs': len(results), 'error_type': type(error).__name__, 'error': str(error),
                'model_qualified': False, 'optimizer_updates': 0, 'gradient_queries': 0})
        except BaseException:
            pass
        raise
