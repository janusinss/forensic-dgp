"""Finite manual-VM measurement only: zero optimizer construction or updates.

Heavy dependencies are imported only in run after the caller's VM guard and an
independent Linux/home check. Importing this file or its algebra tests does not
load a model, execute a forward, calculate a gradient or contact a VM.
"""
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import time
import traceback


MODES = ('E32', 'E32_grad', 'T32', 'TBF16')
GRADIENT_POSITIONS = {(1, 1), (1, 3), (24, 1), (24, 2), (24, 3), (24, 4)}
REPEAT_POSITIONS = {(1, 1), (1, 3)}
GROUPS = ('encoder', 'position_emb', 'feat_emb', 'ft_layers', 'idx_pred_layer', 'fuse_convs_dict')
WEIGHTS = {'token_ce': .5, 'latent_mse': 1., 'pixel_l1_rgb': 1., 'lpips_vgg': 1.,
           'coarse_rgb_l1': 4., 'behavior_rgb_l1': 2., 'behavior_token_kl': .1}


def expected_terms(eligible, behavior):
    names = ['token_ce', 'latent_mse', 'coarse_rgb_l1']
    if eligible:
        names.extend(['pixel_l1_rgb', 'lpips_vgg'])
    if behavior:
        names.extend(['behavior_rgb_l1', 'behavior_token_kl'])
    return tuple(names)


def cosine_from_sufficient_statistics(dot, norm_a_squared, norm_b_squared):
    """Pure scalar algebra; absent/zero gradients do not become fake cosines."""
    if not all(math.isfinite(value) for value in (dot, norm_a_squared, norm_b_squared)):
        raise ValueError('Nonfinite gradient cosine statistics')
    if norm_a_squared < 0 or norm_b_squared < 0:
        raise ValueError('Negative squared gradient norm')
    if norm_a_squared == 0 or norm_b_squared == 0:
        return None
    result = dot / math.sqrt(norm_a_squared * norm_b_squared)
    if abs(result) > 1 + 1e-6:
        raise ValueError('Invalid gradient cosine beyond numerical tolerance')
    return min(1., max(-1., result))


def weighted_component(value, name):
    if name not in WEIGHTS or not math.isfinite(value):
        raise ValueError('Unknown or nonfinite weighted component')
    return value * WEIGHTS[name] / 4


def check_memory_peaks(current, peak, limit):
    if not all(type(value) is int and value >= 0 for value in (current, peak, limit)):
        raise ValueError('Invalid diagnostic memory counters')
    if current > limit or peak > limit:
        raise ValueError('Diagnostic22GiB current/peak reserved GPU cap reached')


@contextmanager
def temporary_attribute(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield original
    finally:
        setattr(owner, name, original)


def run(root, p):
    """Execute the reviewed measurement; the external worker guards VM launch.

    This function never builds an optimizer, accumulates .grad or saves adapted
    weights. autograd.grad is limited to the prospective finite query count.
    """
    from .contract import bounded, parent_root, read_json, require, sha256, write_new_json
    root = Path(root).resolve(strict=True)
    require(platform.system() == 'Linux', 'Neural diagnostic is manual Linux L4 VM only')
    from .worker import manual_root
    root = manual_root(root)  # Exact packet directory and human-opened tmux.
    base = (Path.home() / 'forensic-dgp').resolve(strict=True)
    require(root != base and root.is_relative_to(base), 'Diagnostic must be inside ~/forensic-dgp')
    parent = parent_root(root, p)
    # The caller already verifies exact transfer/parent bytes. Repeat all model
    # hashes at this boundary before any heavy import or checkpoint read.
    for asset in p['assets'].values():
        path = bounded(parent, asset['path'])
        require(path.stat().st_size == asset['bytes'] and sha256(path) == asset['sha256'],
                'Read-only parent asset changed before diagnostic')
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image
    from research.codeformer_adaptation_v2 import model
    from research.codeformer_adaptation_v1.perceptual import load_perceptual
    import third_party.codeformer.codeformer_arch as architecture

    out = bounded(root, p['outputs'], must_exist=False)
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / 'neural_started.json').exists(), 'No automatic repeated neural diagnostic')
    started = time.monotonic()
    caps = p['call_caps']
    limits = p['limits']
    counts = {name: 0 for name in caps}
    events, summaries, stage_states, gradient_rows = [], [], [], []
    state_objects = {}
    baseline_hashes = None
    original_moments, original_behaviors, stopped_baselines = {}, {}, {}
    raw_comparisons = {}
    parameters = ()
    parameter_names = ()
    progress_index = 0
    write_new_json(out / 'neural_started.json', {'UTC': datetime.now(timezone.utc).isoformat(),
        'counts': counts, 'model_qualification_implied': False, 'optimizer_updates': 0})

    def budget():
        require(time.monotonic() - started <= limits['worker_seconds'], 'Finite diagnostic wall-time cap reached')
        require(shutil.disk_usage(root).free >= limits['reserve_free_bytes'], 'Diagnostic4GiB reserve reached')
        result_bytes = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
        require(result_bytes <= limits['result_bytes'], 'Finite diagnostic256MiB result cap reached')
        check_memory_peaks(torch.cuda.memory_reserved(), torch.cuda.max_memory_reserved(),
                           limits['max_reserved_GPU_bytes'])

    def payload_room(additional_bytes):
        current = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
        require(current + additional_bytes <= limits['result_bytes'],
                'Prospective diagnostic write exceeds256MiB result cap')

    def reserve(name, context):
        nonlocal progress_index
        budget()
        require(name in caps and type(caps[name]) is int and counts[name] < caps[name],
                'Finite diagnostic call cap reached: ' + name)
        counts[name] += 1
        progress_index += 1
        event = {'event': progress_index, 'before_call': name, 'context': context,
                 'counts': dict(counts), 'elapsed_seconds': time.monotonic() - started}
        events.append(event)
        write_new_json(out / 'call_trace' / ('%04d.json' % progress_index), event)

    def assert_clean_gradients():
        require(all(parameter.grad is None for module in state_objects.values()
                    for parameter in module.parameters()), 'Measurement populated a parameter.grad field')

    def stage_hashes(stage, *, require_initial=True):
        budget()
        assert_clean_gradients()
        hashes = {name: model.state_digest(module) for name, module in state_objects.items()}
        if require_initial and baseline_hashes is not None:
            require(hashes == baseline_hashes, 'Read-only diagnostic module state mutated at ' + stage)
        row = {'stage': stage, 'states': hashes, 'counts': dict(counts),
               'parameter_grad_fields_all_none': True, 'elapsed_seconds': time.monotonic() - started}
        stage_states.append(row)
        write_new_json(out / 'state_checks' / ('%02d.json' % len(stage_states)), row)
        return hashes

    def tensor_rgb(array):
        return torch.from_numpy(np.ascontiguousarray(array).copy()).permute(2, 0, 1).unsqueeze(0).to('cuda', torch.float32)

    def save_array(path, value):
        require(not path.exists(), 'Preserve existing diagnostic artifact')
        payload_room(int(value.nbytes) + 4096)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        budget()
        return {'path': path.relative_to(root).as_posix(), 'sha256': sha256(path), 'bytes': path.stat().st_size}

    def array_sha(array):
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    def load_case(case):
        image_path = bounded(root, case['input_path'])
        require(sha256(image_path) == case['input_sha256'], 'Exact saved TRAIN input changed')
        with Image.open(image_path) as image:
            require(image.mode == 'RGB' and image.size == (256, 256), 'TRAIN input must remain RGB256')
            pixels = np.asarray(image).copy()
        input_array = np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255))
        target_path = bounded(root, case['clean512_path'])
        require(sha256(target_path) == case['clean512_sha256'], 'Verified direct clean512 target changed')
        target = np.load(target_path, allow_pickle=False)
        require(target.shape == (512, 512, 3) and target.dtype == np.float32 and np.isfinite(target).all() and
                target.min() >= 0 and target.max() <= 1, 'Invalid clean512 target')
        record_path = bounded(root, case['pair_record_path'])
        require(sha256(record_path) == case['pair_record_sha256'], 'Saved TRAIN record changed')
        record = read_json(record_path)
        require(record['reference_id'] == case['reference_id'] and record['update'] == case['update'] and
                record['slot'] == case['slot'] and record['role'] == 'train' and
                record['input256_rgb8_sha256'] == array_sha(pixels) and
                record['input256_float32_sha256'] == array_sha(input_array), 'Saved TRAIN case pixels/role differ')
        require(record['clean512_float32_sha256'] == array_sha(target), 'Target is not the exact recorded clean512')
        return input_array, target, record

    def load_saved_behavior(case):
        """Exact targets used by the retained V2 run, never a new path target."""
        supplied = case['behavior']
        require(case['slot'] == 3 and supplied is not None, 'Saved behavior target belongs only to normal B')
        tensors = {}
        for name, shape in [('delivered', (1, 3, 256, 256)), ('logits', (1, 256, 1024))]:
            path = bounded(root, supplied[name + '_path'])
            require(sha256(path) == supplied[name + '_sha256'], 'Exact saved V2 behavior target changed')
            value = np.load(path, allow_pickle=False)
            require(value.dtype == np.float32 and value.shape == shape and np.isfinite(value).all(),
                    'Exact saved V2 behavior target shape/dtype/finiteness differs')
            if name == 'delivered':
                require(value.min() >= 0 and value.max() <= 1, 'Saved delivered behavior target outside[0,1]')
            # Construct ordinary tensors outside all inference contexts.
            tensors[name] = torch.from_numpy(np.ascontiguousarray(value).copy()).clone()
            require(not tensors[name].requires_grad and not torch.is_inference(tensors[name]),
                    'Saved behavior target must be an ordinary detached tensor')
        return model.BehaviorTargets(tensors['delivered'], tensors['logits'])

    @contextmanager
    def captured_forward(net):
        captured = {}
        original_lookup = net.quantize.get_codebook_feat
        def lookup(indices, shape):
            require('selected' not in captured, 'Unexpected repeated CodeFormer dictionary lookup')
            captured['selected'] = indices.detach().cpu().clone()
            return original_lookup(indices, shape)
        def encoder_hook(module, arguments, feature):
            # Compute native-dtype moments with the exact architecture helper;
            # retain only256 channel means/stds, not a second encoder forward.
            mean, std = architecture.calc_mean_std(feature)
            captured['moment_mean'] = mean.detach().float().cpu().clone()
            captured['moment_std'] = std.detach().float().cpu().clone()
        hook = net.encoder.blocks[-1].register_forward_hook(encoder_hook)
        try:
            with temporary_attribute(net.quantize, 'get_codebook_feat', lookup):
                yield captured
        finally:
            hook.remove()

    def forward(net, case, value, mode, context):
        net.eval() if mode.startswith('E32') else net.train()
        net.quantize.eval(); net.generator.eval()
        reserve('codeformer_forwards', context)
        with ExitStack() as stack:
            if mode == 'E32':
                stack.enter_context(torch.inference_mode())
            else:
                stack.enter_context(torch.enable_grad())
            stack.enter_context(torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(mode == 'TBF16')))
            with captured_forward(net) as captured:
                native, logits, feature = net(model.prepare_input(value), w=1., detach_16=True, adain=True)
                native_probabilities = F.softmax(logits, dim=2)
                top2, top2_indices = torch.topk(native_probabilities, 2, dim=2)
                replay_selected = torch.topk(native_probabilities, 1, dim=2).indices
                require(torch.equal(replay_selected.cpu(), captured['selected']),
                        'Captured actual hard-selected indices differ from same-context native probability replay')
                captured.update(logits_dtype=str(logits.dtype), probabilities_dtype=str(native_probabilities.dtype),
                    candidate_training=net.training, quantizer_training=net.quantize.training,
                    generator_training=net.generator.training, grad_enabled=torch.is_grad_enabled(),
                    inference_mode=torch.is_inference_mode_enabled(),
                    top2=top2.detach().float().cpu().clone(), top2_indices=top2_indices.detach().cpu().clone(),
                    logits_top2=torch.topk(logits.detach().float(), 2, dim=2).values.cpu().clone())
        budget()
        return native, logits, feature, captured

    def describe(native, captured, target, directory, case, state_label, mode, arm=None):
        # Conversion is the unchanged delivered policy, not an enhancement.
        raw = model.delivered_rgb(native.detach()).detach().cpu().float()[0].permute(1, 2, 0).numpy()
        native_rgb = ((native.detach().float() + 1) / 2).cpu().numpy()
        png = np.floor(np.clip(raw, 0, 1) * 255).astype(np.uint8)
        require(raw.shape == (256, 256, 3) and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1,
                'Nonfinite diagnostic delivered output')
        raw_receipt = save_array(directory / 'raw.npy', raw)
        png_path = directory / 'native.png'
        require(not png_path.exists(), 'Preserve existing diagnostic PNG')
        payload_room(198656)
        Image.fromarray(png).save(png_path, format='PNG')
        require(png_path.stat().st_size <= 198656, 'Ordinary RGB PNG exceeds bound')
        tokens = captured['selected'].reshape(256).numpy().astype(np.int64)
        token_receipt = save_array(directory / 'actual_tokens.npy', tokens)
        top2 = captured['top2'].numpy().reshape(256, 2)
        save_array(directory / 'native_probability_top2.npy', top2)
        save_array(directory / 'native_probability_top2_indices.npy', captured['top2_indices'].numpy().reshape(256, 2))
        save_array(directory / 'FP32_logits_top2.npy', captured['logits_top2'].numpy().reshape(256, 2))
        save_array(directory / 'encoder_mean.npy', captured['moment_mean'].numpy())
        save_array(directory / 'encoder_std.npy', captured['moment_std'].numpy())
        with torch.no_grad():
            target256 = F.interpolate(target, (256, 256), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).cpu().numpy()
        luma = raw.astype(np.float64) @ np.asarray([.2126, .7152, .0722])
        luma_target = target256.astype(np.float64) @ np.asarray([.2126, .7152, .0722])
        error = raw.astype(np.float64) - target256.astype(np.float64)
        derivative_error = .5 * (float(np.abs(np.diff(raw.astype(np.float64), axis=0) -
            np.diff(target256.astype(np.float64), axis=0)).mean()) +
            float(np.abs(np.diff(raw.astype(np.float64), axis=1) - np.diff(target256.astype(np.float64), axis=1)).mean()))
        summary = {'case_id': case['id'], 'state': state_label, 'mode': mode, 'intervention': arm,
            'input_sha256': case['input_sha256'], 'clean512_sha256': case['clean512_sha256'],
            'raw': raw_receipt, 'native_png': {'path': png_path.relative_to(root).as_posix(), 'sha256': sha256(png_path)},
            'tokens': token_receipt, 'logits_dtype': captured['logits_dtype'], 'native_probability_dtype': captured['probabilities_dtype'],
            'execution': {name: captured[name] for name in ['candidate_training', 'quantizer_training',
                          'generator_training', 'grad_enabled', 'inference_mode']},
            'native_probability_top2_margin_minimum': float((top2[:, 0] - top2[:, 1]).min()),
            'native_probability_top2_zero_margin_tokens': int(np.equal(top2[:, 0], top2[:, 1]).sum()),
            'selected_index_not_top2_first_due_to_ties': int((tokens != captured['top2_indices'].numpy().reshape(256, 2)[:, 0]).sum()),
            'preclip_RGB': {'minimum': float(native_rgb.min()), 'maximum': float(native_rgb.max()),
                'mean': float(native_rgb.mean()), 'outside_range_fraction': float(((native_rgb < 0) | (native_rgb > 1)).mean())},
            'delivered_RGB_clip_fraction': float(((raw <= 0) | (raw >= 1)).mean()),
            'luminance_mean': float(luma.mean()), 'luminance_std': float(luma.std()),
            'target_luminance_mean': float(luma_target.mean()), 'target_luminance_std': float(luma_target.std()),
            'signed_luminance_bias': float((luma - luma_target).mean()), 'RGB_MAE': float(np.abs(error).mean()),
            'RGB_MSE': float(np.square(error).mean()), 'mean_absolute_first_difference_error': derivative_error,
            'target256_policy': 'exact retained clean512 float32 -> torch bilinear256 align_cornersFalse',
            'paired_synthetic_only': True, 'brightness_shift_is_not_a_quality_pass_or_fail': True,
            'identity_or_usefulness_claim': False}
        write_new_json(directory / 'summary.json', summary)
        summaries.append(summary)
        return raw.copy(), tokens.copy()

    def loss_terms(native, logits, feature, target, codes, behavior, eligible, context):
        with torch.autocast(device_type='cuda', enabled=False):
            rgb = (native.float() + 1) / 2
            terms = {'token_ce': F.cross_entropy(logits.float().transpose(1, 2), codes.indices),
                'latent_mse': F.mse_loss(feature.float(), codes.features.float()),
                'coarse_rgb_l1': F.l1_loss(F.avg_pool2d(rgb, 32, 32), F.avg_pool2d(target.float(), 32, 32))}
            if eligible:
                terms['pixel_l1_rgb'] = F.l1_loss(rgb, target.float())
                reserve('lpips_forwards', context)
                terms['lpips_vgg'] = perceptual(native.float(), target.float() * 2 - 1)
            if behavior is not None:
                terms['behavior_rgb_l1'] = F.l1_loss(model.delivered_rgb(native), behavior.delivered256.float())
                terms['behavior_token_kl'] = model.token_distillation(logits, behavior.logits)
        require(tuple(terms) == expected_terms(eligible, behavior is not None), 'Diagnostic loss routing differs from V2')
        return terms

    def gradient_summary(terms, case, state_label, vectors):
        names = tuple(terms)
        photo = [None] * len(parameters)
        behavior = [None] * len(parameters)
        for index, name in enumerate(names):
            scalar = terms[name] * WEIGHTS[name] / 4
            require(torch.isfinite(scalar).item(), 'Nonfinite diagnostic weighted loss')
            reserve('gradient_queries', state_label + '/' + case['id'] + '/' + name)
            grads = torch.autograd.grad(scalar, parameters, allow_unused=True,
                                       retain_graph=(index != len(names) - 1), create_graph=False)
            groups = {group: {'parameters': 0, 'absent_tensors': 0, 'zero_tensors': 0,
                              'nonzero_tensors': 0, 'norm_squared': 0., 'maximum_absolute': 0.} for group in GROUPS}
            for position, (parameter_name, grad) in enumerate(zip(parameter_names, grads)):
                group = groups[parameter_name.split('.')[0]]; group['parameters'] += 1
                if grad is None:
                    group['absent_tensors'] += 1
                    continue
                require(torch.isfinite(grad).all().item(), 'Nonfinite component gradient')
                cpu = grad.detach().float().cpu()
                maximum = float(cpu.abs().max())
                group['zero_tensors' if maximum == 0 else 'nonzero_tensors'] += 1
                group['maximum_absolute'] = max(group['maximum_absolute'], maximum)
                group['norm_squared'] += float(cpu.double().square().sum())
                family = photo if name in ('coarse_rgb_l1', 'pixel_l1_rgb') else (
                    behavior if name in ('behavior_rgb_l1', 'behavior_token_kl') else None)
                if family is not None:
                    family[position] = cpu.clone() if family[position] is None else family[position] + cpu
            for group in groups.values():
                group['norm'] = math.sqrt(group.pop('norm_squared'))
            row = {'state': state_label, 'case_id': case['id'], 'term': name,
                'raw_loss': float(terms[name].detach().float().cpu()), 'weight': WEIGHTS[name],
                'accumulation_divisor': 4, 'weighted_loss_divided_by_four': float(scalar.detach().float().cpu()),
                'groups': groups, 'all_groups_nonzero_not_required': True,
                'zero_optimizer_updates': True}
            gradient_rows.append(row)
            write_new_json(out / 'gradients' / state_label / case['id'] / (name + '.json'), row)
            del grads
            assert_clean_gradients(); budget()
        vectors[(case['update'], case['slot'], 'photometric')] = photo
        if case['slot'] == 3:
            vectors[(case['update'], case['slot'], 'behavior')] = behavior

    def gradient_cosines(vectors, state_label):
        comparisons = [((24, 1, 'photometric'), (24, 2, 'photometric')),
                       ((24, 3, 'photometric'), (24, 4, 'photometric')),
                       ((1, 3, 'photometric'), (1, 3, 'behavior')),
                       ((24, 3, 'photometric'), (24, 3, 'behavior')),
                       ((24, 3, 'behavior'), (24, 4, 'photometric'))]
        result = []
        for a, b in comparisons:
            groups = {group: [0., 0., 0.] for group in GROUPS}
            for name, x, y in zip(parameter_names, vectors[a], vectors[b]):
                group = groups[name.split('.')[0]]
                if x is not None:
                    group[1] += float(x.double().square().sum())
                if y is not None:
                    group[2] += float(y.double().square().sum())
                if x is not None and y is not None:
                    group[0] += float((x.double() * y.double()).sum())
            result.append({'a': list(a), 'b': list(b), 'groups': {name:
                {'dot': value[0], 'norm_a': math.sqrt(value[1]), 'norm_b': math.sqrt(value[2]),
                 'cosine': cosine_from_sufficient_statistics(*value)} for name, value in groups.items()}})
        write_new_json(out / 'gradients' / state_label / 'cosines.json',
            {'state': state_label, 'comparisons': result, 'predicted_training_quality_not_implied': True})
        return result

    try:
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
                torch.cuda.get_device_name(0) == 'NVIDIA L4' and torch.cuda.is_bf16_supported(),
                'Finite diagnostic requires one NVIDIA L4 with BF16 support')
        require(shutil.disk_usage(root).free >= limits['initial_free_bytes'], 'Require6GiB free after diagnostic install')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(p['seed']); torch.cuda.manual_seed_all(p['seed'])
        write_new_json(out / 'neural_runtime.json', {'torch': torch.__version__, 'numpy': np.__version__,
            'CUDA': torch.version.cuda, 'GPU': torch.cuda.get_device_name(0),
            'MHA_fastpath_enabled': torch.backends.mha.get_fastpath_enabled(),
            'TF32_matmul': torch.backends.cuda.matmul.allow_tf32, 'TF32_cudnn': torch.backends.cudnn.allow_tf32,
            'candidate_precision': 'FP32 parameters; BF16 autocast only TBF16; FP32 all loss reductions',
            'all_neural_calls_require_manual_guarded_L4_execution': True})
        original, _ = model.load_trainable(bounded(parent, p['assets']['initializer']['path']), device='cuda')
        stopped, _ = model.load_trainable(bounded(parent, p['assets']['initializer']['path']), device='cuda')
        checkpoint_asset = p['assets']['stopped_checkpoint']
        checkpoint = torch.load(bounded(parent, checkpoint_asset['path']), map_location='cpu', weights_only=True)
        require(checkpoint['completed_updates'] == 50 and checkpoint['protocol_sha256'] == p['parent_protocol_sha256'],
                'Diagnostic requires the retained stopped V2 update50 state')
        stopped.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        teacher = model.load_teacher(bounded(parent, p['assets']['teacher']['path']), device='cuda')
        perceptual = load_perceptual(bounded(parent, p['assets']['perceptual_features']['path']).parent, device='cuda')
        state_objects.update(original=original, stopped=stopped, HQ_teacher=teacher, perceptual=perceptual)
        require(model.LOSS_WEIGHTS == WEIGHTS, 'Pinned V2 loss weights differ')
        require(model.frozen_state_digest(original) == model.frozen_state_digest(stopped), 'Retained frozen tensors changed')
        baseline_hashes = stage_hashes('loaded_before_any_forward', require_initial=False)
        cases = p['cases']
        require({(case['update'], case['slot']) for case in cases} ==
                {(update, slot) for update in (1, 24) for slot in range(1, 5)} and len(cases) == 8,
                'Diagnostic fixed TRAIN cohort differs')
        loaded = {case['id']: load_case(case) for case in cases}
        saved_behaviors = {case['id']: load_saved_behavior(case) for case in cases if case.get('behavior')}
        require(len(saved_behaviors) == 2, 'Exactly two original saved V2 behavior targets required')
        clean_codes = {}
        for case in cases:
            identifier = case['reference_id']
            if identifier in clean_codes:
                continue
            reserve('hq_teacher_forwards', identifier)
            target = tensor_rgb(loaded[case['id']][1])
            clean_codes[identifier] = model.hq_targets(teacher, target, codebook=original.quantize)
            del target
        require(len(clean_codes) == 4, 'Four distinct TRAIN reference targets required')
        stage_hashes('HQ_targets_complete')
        all_cosines = {}
        for state_label, net in [('original', original), ('stopped50', stopped)]:
            stage_hashes(state_label + '_before_paths')
            parameters = tuple(parameter for _, parameter in net.named_parameters() if parameter.requires_grad)
            parameter_names = tuple(name for name, parameter in net.named_parameters() if parameter.requires_grad)
            vectors = {}
            for case in cases:
                input_array, target_array, record = loaded[case['id']]
                value, target = tensor_rgb(input_array), tensor_rgb(target_array)
                for mode in MODES:
                    context = state_label + '/' + case['id'] + '/' + mode
                    native, logits, feature, captured = forward(net, case, value, mode, context)
                    if mode == 'E32':
                        if state_label == 'original':
                            original_moments[case['id']] = (captured['moment_mean'].clone(), captured['moment_std'].clone())
                            if case['slot'] == 3:
                                # Clone outside inference context into ordinary
                                # CPU tensors before later differentiable losses.
                                original_behaviors[case['id']] = model.BehaviorTargets(
                                    model.delivered_rgb(native.detach()).detach().cpu().clone(),
                                    logits.detach().float().cpu().clone())
                        else:
                            stopped_baselines[case['id']] = captured['selected'].reshape(256).numpy().copy()
                    raw, tokens = describe(native, captured, target, out / 'paths' / state_label / case['id'] / mode,
                                           case, state_label, mode)
                    raw_comparisons[(state_label, case['id'], mode)] = (raw, tokens)
                    if mode == 'TBF16' and (case['update'], case['slot']) in GRADIENT_POSITIONS:
                        behavior = None
                        if case['slot'] == 3:
                            anchor = saved_behaviors[case['id']]
                            behavior = model.BehaviorTargets(anchor.delivered256.to('cuda'), anchor.logits.to('cuda'))
                        terms = loss_terms(native, logits, feature, target, clean_codes[case['reference_id']], behavior,
                                           case['image_eligible'], context)
                        gradient_summary(terms, case, state_label, vectors)
                        del terms, behavior
                    del native, logits, feature, captured
                if state_label == 'original' and case.get('behavior'):
                    supplied = case['behavior']; anchor = original_behaviors[case['id']]
                    differences = {}
                    for name, actual in [('delivered', anchor.delivered256.numpy()), ('logits', anchor.logits.numpy())]:
                        path = bounded(root, supplied[name + '_path'])
                        require(sha256(path) == supplied[name + '_sha256'], 'Retained teacher anchor file changed')
                        saved = np.load(path, allow_pickle=False)
                        require(saved.shape == actual.shape and saved.dtype == np.float32 and np.isfinite(saved).all(),
                                'Retained teacher anchor shape/dtype/finiteness differs')
                        differences[name] = {'maximum_absolute': float(np.abs(actual.astype(np.float64) - saved.astype(np.float64)).max()),
                                             'mean_absolute': float(np.abs(actual.astype(np.float64) - saved.astype(np.float64)).mean())}
                    write_new_json(out / 'teacher_anchor_comparisons' / (case['id'] + '.json'),
                        {'case_id': case['id'], 'differences': differences,
                         'new_original_E32_vs_retained_original_FP32_no_grad': True,
                         'gradient_targets_are_exact_saved_V2_anchors_not_new_E32': True,
                         'saved_RGB8_to_float32_input_binding_verified': True,
                         'no_grad_inference_context_or_runtime_numerics_may_differ': True,
                         'cause_and_exact_replay_not_claimed': True})
                del value, target
            all_cosines[state_label] = gradient_cosines(vectors, state_label)
            del vectors
            stage_hashes(state_label + '_after_paths_and_gradients')

        # Repeat four declared modes on two fixed inputs at both checkpoints.
        repetitions = []
        for state_label, net in [('original', original), ('stopped50', stopped)]:
            stage_hashes(state_label + '_before_repeats')
            for case in cases:
                if (case['update'], case['slot']) not in REPEAT_POSITIONS:
                    continue
                value, target = tensor_rgb(loaded[case['id']][0]), tensor_rgb(loaded[case['id']][1])
                for mode in MODES:
                    context = state_label + '/' + case['id'] + '/' + mode + '/repeat'
                    native, logits, feature, captured = forward(net, case, value, mode, context)
                    raw, tokens = describe(native, captured, target, out / 'repeats' / state_label / case['id'] / mode,
                                           case, state_label, mode)
                    old_raw, old_tokens = raw_comparisons[(state_label, case['id'], mode)]
                    repetitions.append({'state': state_label, 'case_id': case['id'], 'mode': mode,
                        'tokens_equal': bool(np.array_equal(tokens, old_tokens)),
                        'RAW_maximum_absolute': float(np.abs(raw.astype(np.float64) - old_raw.astype(np.float64)).max())})
                    del native, logits, feature, captured
                del value, target
            stage_hashes(state_label + '_after_repeats')
        write_new_json(out / 'repeat_stability.json', {'comparisons': repetitions,
            'different_paths_need_not_agree_even_if_repeats_are_stable': True})

        # Three missing arms of the stopped-state2x2; current/current exists.
        stage_hashes('before_disposable_interventions')
        original_fusion = {name: value.detach().cpu().clone() for name, value in original.fuse_convs_dict.state_dict().items()}
        stopped_fusion = {name: value.detach().cpu().clone() for name, value in stopped.fuse_convs_dict.state_dict().items()}
        original_adain_function = architecture.adaptive_instance_normalization
        try:
            for case in cases:
                value, target = tensor_rgb(loaded[case['id']][0]), tensor_rgb(loaded[case['id']][1])
                mean, std = original_moments[case['id']]
                def original_moment_adain(content, style):
                    content_mean, content_std = architecture.calc_mean_std(content)
                    return (content - content_mean) / content_std * std.to(content.device, content.dtype) + mean.to(content.device, content.dtype)
                for use_moments, use_fusion in [(True, False), (False, True), (True, True)]:
                    arm = ('original_moments' if use_moments else 'current_moments') + '__' + (
                           'original_SFT' if use_fusion else 'current_SFT')
                    stopped.fuse_convs_dict.load_state_dict(original_fusion if use_fusion else stopped_fusion, strict=True)
                    with temporary_attribute(architecture, 'adaptive_instance_normalization',
                                             original_moment_adain if use_moments else original_adain_function):
                        native, logits, feature, captured = forward(stopped, case, value, 'E32', 'intervention/' + case['id'] + '/' + arm)
                    require(np.array_equal(captured['selected'].reshape(256).numpy(), stopped_baselines[case['id']]),
                            'Intervention unexpectedly changed hard-selected code indices')
                    describe(native, captured, target, out / 'interventions' / case['id'] / arm,
                             case, 'stopped50', 'E32', arm=arm)
                    del native, logits, feature, captured
                del value, target
        finally:
            architecture.adaptive_instance_normalization = original_adain_function
            stopped.fuse_convs_dict.load_state_dict(stopped_fusion, strict=True)
        stage_hashes('after_all_interventions_restored')
        comparisons = []
        for state_label in ('original', 'stopped50'):
            for case in cases:
                baseline, baseline_tokens = raw_comparisons[(state_label, case['id'], 'E32')]
                for mode in MODES[1:]:
                    raw, tokens = raw_comparisons[(state_label, case['id'], mode)]
                    comparisons.append({'state': state_label, 'case_id': case['id'], 'baseline': 'E32', 'mode': mode,
                        'token_disagreements': int((tokens != baseline_tokens).sum()),
                        'RAW_maximum_absolute': float(np.abs(raw.astype(np.float64) - baseline.astype(np.float64)).max()),
                        'RAW_mean_absolute': float(np.abs(raw.astype(np.float64) - baseline.astype(np.float64)).mean())})
        write_new_json(out / 'path_comparisons.json', {'comparisons': comparisons,
            'precision_or_mode_difference_is_not_yet_50step_drift_cause': True})
        require(counts == caps and counts['optimizer_updates'] == 0, 'Diagnostic ended outside the exact prospective call census')
        stage_hashes('completed_all_stages')
        for asset in p['assets'].values():
            require(sha256(bounded(parent, asset['path'])) == asset['sha256'],
                    'Read-only original/stopped parent file changed during diagnostic')
        budget()
        result = {'complete': True, 'counts': counts, 'seconds': time.monotonic() - started,
            'state_before_after_equal': True, 'states': baseline_hashes, 'parameter_grad_fields_all_none': True,
            'output_conditions': len(summaries), 'gradient_component_queries': len(gradient_rows),
            'CUDA_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'no_optimizer_constructed_or_updated': True, 'model_qualified': False,
            'causal_training_quality_claim': False, 'diagnostic_outputs_are_not_model_candidates': True}
        write_new_json(out / 'neural_results.json', result)
        return result
    except BaseException as error:
        failure = {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'counts': counts, 'seconds': time.monotonic() - started,
            'optimizer_updates': 0, 'model_qualified': False, 'partial_outputs_retained': True}
        try:
            if state_objects:
                assert_clean_gradients()
                failure['states_after_failure'] = {name: model.state_digest(module) for name, module in state_objects.items()}
                failure['states_equal_to_initial'] = baseline_hashes == failure['states_after_failure']
            write_new_json(out / 'neural_failure.json', failure)
        except BaseException as evidence_error:
            failure['failure_receipt_error'] = str(evidence_error)
        raise
