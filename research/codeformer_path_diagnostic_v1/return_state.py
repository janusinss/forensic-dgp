"""Independent CPU-only audit of saved diagnostic control/state/scalar evidence.

This module constructs no neural network, runs no forward, gradient or optimizer,
and contacts no VM. Runtime state and gradient vectors were not exported; their
reported fingerprints/statistics are checked for source-bound consistency, not
represented as an independent rerun of those measurements.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import struct

from .contract import bounded, read_json, require, sha256, validate_protocol, write_new_json


MODES = ('E32', 'E32_grad', 'T32', 'TBF16')
STATES = ('original', 'stopped50')
GRADIENT_POSITIONS = {(1, 1), (1, 3), (24, 1), (24, 2), (24, 3), (24, 4)}
REPEAT_POSITIONS = {(1, 1), (1, 3)}
GROUP_COUNTS = {'encoder': 164, 'position_emb': 1, 'feat_emb': 2,
                'ft_layers': 108, 'idx_pred_layer': 3, 'fuse_convs_dict': 72}
WEIGHTS = {'token_ce': .5, 'latent_mse': 1., 'coarse_rgb_l1': 4.,
           'pixel_l1_rgb': 1., 'lpips_vgg': 1., 'behavior_rgb_l1': 2., 'behavior_token_kl': .1}
ARMS = ('original_moments__current_SFT', 'current_moments__original_SFT',
        'original_moments__original_SFT')
COSINE_PAIRS = (((24, 1, 'photometric'), (24, 2, 'photometric')),
                ((24, 3, 'photometric'), (24, 4, 'photometric')),
                ((1, 3, 'photometric'), (1, 3, 'behavior')),
                ((24, 3, 'photometric'), (24, 3, 'behavior')),
                ((24, 3, 'behavior'), (24, 4, 'photometric')))


def finite(value, *, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), 'Expected finite scalar')
    require(not nonnegative or value >= 0, 'Expected nonnegative scalar')
    return value


def near(actual, expected, *, tolerance=1e-12):
    finite(actual); finite(expected)
    require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
            f'Scalar differs: {actual} versus {expected}')


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def weighted_scalar(loss, weight):
    """Independent IEEE float32 multiply/divide replay; no Torch loss execution."""
    finite(loss); finite(weight)
    return f32(f32(f32(loss) * f32(weight)) / f32(4))


def audit_execution(row, mode):
    execution = {'candidate_training': mode.startswith('T'), 'quantizer_training': False,
        'generator_training': False, 'grad_enabled': mode != 'E32', 'inference_mode': mode == 'E32'}
    require(row['execution'] == execution, 'Forward execution mode differs')
    logits_dtype = 'torch.bfloat16' if mode == 'TBF16' else 'torch.float32'
    # CUDA autocast softmax runs in FP32 even when incoming logits are BF16.
    # These are the actual worker/architecture probabilities, not a substituted
    # off-path FP32-logit selection. BF16 logits can still contain ties.
    require(row['logits_dtype'] == logits_dtype and row['native_probability_dtype'] == 'torch.float32',
            'Forward precision differs')


def terms_for(case):
    names = ['token_ce', 'latent_mse', 'coarse_rgb_l1']
    if case['image_eligible']:
        names += ['pixel_l1_rgb', 'lpips_vgg']
    if case['behavior'] is not None:
        names += ['behavior_rgb_l1', 'behavior_token_kl']
    return names


def expected_calls(cases):
    """Enumerate prospective reservations independently of worker execution."""
    events = []
    seen = set()
    for case in cases:
        identifier = case['reference_id']
        if identifier not in seen:
            events.append(('hq_teacher_forwards', identifier)); seen.add(identifier)
    for state in STATES:
        for case in cases:
            for mode in MODES:
                context = f"{state}/{case['id']}/{mode}"
                events.append(('codeformer_forwards', context))
                if mode == 'TBF16' and (case['update'], case['slot']) in GRADIENT_POSITIONS:
                    if case['image_eligible']:
                        events.append(('lpips_forwards', context))
                    events.extend(('gradient_queries', f"{state}/{case['id']}/{term}")
                                  for term in terms_for(case))
    for state in STATES:
        for case in cases:
            if (case['update'], case['slot']) in REPEAT_POSITIONS:
                events.extend(('codeformer_forwards', f"{state}/{case['id']}/{mode}/repeat") for mode in MODES)
    for case in cases:
        events.extend(('codeformer_forwards', f"intervention/{case['id']}/{arm}") for arm in ARMS)
    return events


def audit_trace(rows, expected, caps, max_seconds):
    require(len(rows) == len(expected), 'Call reservation event count differs')
    counts = {name: 0 for name in caps}
    last = 0.
    by_counts = {tuple(counts.values()): 0.}
    for index, (row, (kind, context)) in enumerate(zip(rows, expected), 1):
        require(row['event'] == index and type(row['event']) is int and row['before_call'] == kind
                and row['context'] == context, 'Call reservation order/context differs')
        counts[kind] += 1
        require(row['counts'] == counts and all(type(x) is int for x in row['counts'].values()),
                'Call reservation counter differs')
        elapsed = finite(row['elapsed_seconds'], nonnegative=True)
        require(last <= elapsed <= max_seconds and counts[kind] <= caps[kind], 'Nonmonotone or over-budget call')
        last = elapsed; by_counts[tuple(counts.values())] = elapsed
    require(counts == caps, 'Call census incomplete or optimizer present')
    return {'events': len(rows), 'counts': counts, 'last_reserved_seconds': last}, by_counts


def expected_stage_counts():
    def c(cf, lp=0, gr=0, hq=4):
        return {'codeformer_forwards': cf, 'hq_teacher_forwards': hq,
                'lpips_forwards': lp, 'gradient_queries': gr, 'optimizer_updates': 0}
    return [('loaded_before_any_forward', c(0, hq=0)), ('HQ_targets_complete', c(0)),
            ('original_before_paths', c(0)), ('original_after_paths_and_gradients', c(32, 5, 32)),
            ('stopped50_before_paths', c(32, 5, 32)), ('stopped50_after_paths_and_gradients', c(64, 10, 64)),
            ('original_before_repeats', c(64, 10, 64)), ('original_after_repeats', c(72, 10, 64)),
            ('stopped50_before_repeats', c(72, 10, 64)), ('stopped50_after_repeats', c(80, 10, 64)),
            ('before_disposable_interventions', c(80, 10, 64)),
            ('after_all_interventions_restored', c(104, 10, 64)), ('completed_all_stages', c(104, 10, 64))]


def audit_stages(rows, states, counts_times, max_seconds):
    expected = expected_stage_counts()
    require(len(rows) == len(expected), 'State-check stage census differs')
    require(set(states) == {'original', 'stopped', 'HQ_teacher', 'perceptual'}
            and all(isinstance(x, str) and len(x) == 64 for x in states.values()), 'State fingerprints malformed')
    elapsed = 0.
    for row, (name, counts) in zip(rows, expected):
        require(row['stage'] == name and row['counts'] == counts, 'State-check stage/count differs')
        require(row['states'] == states and row['parameter_grad_fields_all_none'] is True,
                'Reported module mutation or populated .grad')
        current = finite(row['elapsed_seconds'], nonnegative=True)
        require(elapsed <= current <= max_seconds and current >= counts_times[tuple(counts.values())],
                'State-check clock inconsistent with call reservations')
        elapsed = current
    return {'stage_count': len(rows), 'all_fingerprints_equal_within_model': True,
            'all_empty_grad_fields_reported': True, 'last_state_check_seconds': elapsed,
            'runtime_tensor_bytes_not_exported': True}


def audit_gradient(row, state, case, term):
    require(row['state'] == state and row['case_id'] == case['id'] and row['term'] == term,
            'Gradient term identity differs')
    require(row['weight'] == WEIGHTS[term] and row['accumulation_divisor'] == 4
            and row['zero_optimizer_updates'] is True and row['all_groups_nonzero_not_required'] is True,
            'Gradient weight/routing declaration differs')
    near(row['weighted_loss_divided_by_four'], weighted_scalar(row['raw_loss'], WEIGHTS[term]))
    require(set(row['groups']) == set(GROUP_COUNTS), 'Gradient parameter groups differ')
    for group, expected in GROUP_COUNTS.items():
        item = row['groups'][group]
        require(item['parameters'] == expected and type(item['parameters']) is int, 'Gradient group tensor count differs')
        statuses = [item[key] for key in ('absent_tensors', 'zero_tensors', 'nonzero_tensors')]
        require(all(type(x) is int and x >= 0 for x in statuses) and sum(statuses) == expected,
                'Gradient absent/zero/nonzero partition differs')
        norm = finite(item['norm'], nonnegative=True)
        maximum = finite(item['maximum_absolute'], nonnegative=True)
        require((statuses[2] == 0) == (norm == 0) == (maximum == 0), 'Gradient nonzero/norm status contradicts')
        require(maximum <= norm * (1 + 1e-12) + 1e-15, 'Gradient maximum exceeds norm')
    return row


def audit_cosines(row, state):
    require(row['state'] == state and row['predicted_training_quality_not_implied'] is True,
            'Cosine identity/claim differs')
    comparisons = row['comparisons']
    require(len(comparisons) == len(COSINE_PAIRS), 'Cosine comparison census differs')
    checked = []
    for item, (a, b) in zip(comparisons, COSINE_PAIRS):
        require(item['a'] == list(a) and item['b'] == list(b) and set(item['groups']) == set(GROUP_COUNTS),
                'Cosine pair/group order differs')
        for group, stats in item['groups'].items():
            dot = finite(stats['dot']); na = finite(stats['norm_a'], nonnegative=True)
            nb = finite(stats['norm_b'], nonnegative=True)
            if na == 0 or nb == 0:
                require(stats['cosine'] is None and dot == 0, 'Absent gradient has invented cosine/dot')
            else:
                expected = dot / (na * nb)
                require(abs(expected) <= 1 + 1e-6, 'Cosine violates Cauchy bound')
                near(stats['cosine'], max(-1., min(1., expected)))
        checked.append(item)
    return checked


def audit_family_norms(comparisons, components):
    """Check saved sum-vector norms against component triangle bounds.

    Cross-component dot products are absent, so these are bounds rather than an
    invented exact recomputation of a vector sum.
    """
    indexed = {(row['case_id'], row['term']): row for row in components}
    repeated, checked = {}, 0
    for comparison in comparisons:
        for side in ('a', 'b'):
            update, slot, family = comparison[side]
            identifier = f'train_u{update:04d}_slot{slot}'
            names = ('coarse_rgb_l1', 'pixel_l1_rgb') if family == 'photometric' else (
                    'behavior_rgb_l1', 'behavior_token_kl')
            present = [indexed[(identifier, name)] for name in names if (identifier, name) in indexed]
            require(present, 'Cosine vector family has no recorded component')
            for group in GROUP_COUNTS:
                norms = [row['groups'][group]['norm'] for row in present]
                saved = comparison['groups'][group]['norm_' + side]
                low, high = max(0., max(norms) - (sum(norms) - max(norms))), sum(norms)
                require(low - 1e-6 <= saved <= high + 1e-6, 'Saved sum-gradient norm violates component triangle bounds')
                key = (identifier, family, group)
                if key in repeated:
                    near(saved, repeated[key])
                repeated[key] = saved; checked += 1
    return checked


def tensor_digest(state):
    """Exact documented state serialization, independently implemented, CPU only."""
    import torch
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.grad is None,
                'State contains non-CPU/non-tensor/populated-grad value')
        require(torch.isfinite(value).all().item(), 'Nonfinite CPU state')
        value = value.detach().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_cpu_assets(protocol, parent, stopped_path, reported):
    import torch
    torch.set_num_threads(1)
    result, files = {}, {}
    for name in ('initializer', 'stopped_checkpoint', 'teacher', 'perceptual_features', 'perceptual_calibration', 'perceptual_manifest'):
        asset = protocol['assets'][name]
        path = Path(stopped_path).resolve(strict=True) if name == 'stopped_checkpoint' else bounded(parent, asset['path'])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == asset['bytes']
                and sha256(path) == asset['sha256'], 'CPU state asset file hash/size differs: ' + name)
        files[name] = path
    for label, key in (('original', 'initializer'), ('stopped', 'stopped_checkpoint'), ('HQ_teacher', 'teacher')):
        payload = torch.load(files[key], map_location='cpu', weights_only=True)
        if label == 'stopped':
            require(payload['completed_updates'] == 50 and payload['protocol_sha256'] == protocol['parent_protocol_sha256'],
                    'CPU stopped checkpoint parent/update differs')
            state = payload['model']
        else:
            state = payload['params_ema']
        actual = tensor_digest(state)
        require(actual == reported[label], 'CPU state digest differs from runtime evidence: ' + label)
        result[label] = {'state_sha256': actual, 'tensors': len(state), 'file_sha256': protocol['assets'][key]['sha256']}
        del state, payload; gc.collect()
    features = torch.load(files['perceptual_features'], map_location='cpu', weights_only=True)['params']
    calibration = torch.load(files['perceptual_calibration'], map_location='cpu', weights_only=True)
    assembled = dict(features)
    assembled.update({'calibration.' + str(index): calibration[f'lin{index}.model.1.weight'] for index in range(5)})
    assembled.update(shift=torch.tensor([-.030, -.088, -.188], dtype=torch.float32).reshape(1, 3, 1, 1),
                     scale=torch.tensor([.458, .448, .450], dtype=torch.float32).reshape(1, 3, 1, 1))
    actual = tensor_digest(assembled)
    require(actual == reported['perceptual'], 'CPU reconstructed perceptual state digest differs')
    result['perceptual'] = {'state_sha256': actual, 'tensors': len(assembled),
        'assembly': 'Exact VGG feature tensors, five calibrated weights, declared constant shift/scale; no module instantiated'}
    result['limits'] = 'Confirms parent state bytes match reported runtime fingerprints; no post-run runtime tensors or gradient vectors were exported.'
    return result


def audit(root, packet, outputdir, *, parent_assets=None, stopped_checkpoint=None):
    root, packet = Path(root).resolve(strict=True), Path(packet).resolve(strict=True)
    outputdir = Path(outputdir)
    require(not outputdir.exists(), 'Preserve existing state audit')
    protocol = validate_protocol(read_json(bounded(root, 'protocol.json')))
    require(sha256(root / 'protocol.json') == sha256(packet / 'protocol.json'), 'Returned protocol differs from frozen packet')
    out = root / protocol['outputs']
    require(out.is_dir(), 'Missing diagnostic evidence directory')
    evidence_paths = [path for path in out.rglob('*') if path.is_file()]
    require(not any(path.suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors') for path in evidence_paths),
            'Diagnostic unexpectedly exports model/gradient weights')
    require(not (out / 'worker_failure.json').exists() and not (out / 'neural_failure.json').exists(),
            'Diagnostic contains retained execution failure')
    neural = read_json(out / 'neural_results.json'); worker = read_json(out / 'worker_complete.json')
    supervisor = read_json(root / 'outputs/supervision/run_receipt.json')
    require(neural['complete'] is True and neural['counts'] == protocol['call_caps'] and neural['output_conditions'] == 104
            and neural['gradient_component_queries'] == 64 and neural['no_optimizer_constructed_or_updated'] is True,
            'Neural terminal declaration differs')
    require(neural['state_before_after_equal'] is True and neural['parameter_grad_fields_all_none'] is True
            and neural['model_qualified'] is False and neural['causal_training_quality_claim'] is False
            and neural['diagnostic_outputs_are_not_model_candidates'] is True, 'Neural state/quality claim differs')
    seconds = finite(neural['seconds'], nonnegative=True)
    require(seconds <= protocol['limits']['worker_seconds'] and type(neural['CUDA_peak_reserved_bytes']) is int
            and 0 < neural['CUDA_peak_reserved_bytes'] <= protocol['limits']['max_reserved_GPU_bytes'], 'Neural resource cap differs')
    protocol_sha = sha256(root / 'protocol.json')
    require(worker['measurement_result'] == neural and worker['protocol_sha256'] == protocol_sha
            and worker['complete'] is True and worker['manual_tmux'] is True and worker['automatic_training'] is False
            and worker['optimizer_updates'] == 0 and worker['model_qualified'] is False
            and worker['next_training_authorized'] is False and worker['automatic_app_promotion'] is False,
            'Worker terminal evidence inconsistent')
    worker_seconds = finite(worker['seconds'], nonnegative=True)
    require(seconds <= worker_seconds <= protocol['limits']['worker_seconds'], 'Worker timing cap differs')
    launch = read_json(out / 'manual_launch.json')
    require(all(worker[k] == v for k, v in launch.items()), 'Worker launch identity differs')
    require(supervisor['protocol_sha256'] == protocol_sha and supervisor['manual_tmux'] is True
            and supervisor['optimizer_updates'] == 0 and supervisor['neural_measurement_complete'] is True
            and supervisor['no_automatic_retry'] is True and supervisor['model_qualified'] is False
            and supervisor['automatic_app_promotion'] is False, 'Supervisor final evidence differs')
    idle = supervisor['initial_idle_check']
    require(idle['GPU_compute_processes'] == [] and idle['competing_research_commands'] == []
            and idle['idle_tmux_sessions_are_not_workloads'] is True, 'Initial workload check differs')
    require([x['stage'] for x in supervisor['stages']] == ['--verify-transfer', '--run'], 'Supervisor stage sequence differs')
    for item, cap, grace, log in zip(supervisor['stages'], (300, 900), (30, 90), ('preflight.log', 'diagnostic.log')):
        require(item['exit_code'] == 0 and item['external_cap_seconds'] == cap and item['grace_seconds'] == grace
                and item['log'] == 'outputs/supervision/' + log and 0 <= finite(item['seconds']) <= cap,
                'Supervisor stage failed or exceeded cap')
    require(worker_seconds <= supervisor['stages'][1]['seconds'], 'Supervisor run shorter than worker')
    runtime = read_json(out / 'neural_runtime.json')
    require(runtime['GPU'] == 'NVIDIA L4' and runtime['TF32_matmul'] is False and runtime['TF32_cudnn'] is False
            and runtime['all_neural_calls_require_manual_guarded_L4_execution'] is True, 'Runtime execution policy differs')
    expected = expected_calls(protocol['cases'])
    trace_files = sorted((out / 'call_trace').glob('*.json'))
    require([x.name for x in trace_files] == [f'{i:04}.json' for i in range(1, len(expected) + 1)], 'Trace namespace differs')
    trace, count_times = audit_trace([read_json(x) for x in trace_files], expected, protocol['call_caps'], seconds)
    stages_files = sorted((out / 'state_checks').glob('*.json'))
    require([x.name for x in stages_files] == [f'{i:02}.json' for i in range(1, 14)], 'State-check namespace differs')
    stages = audit_stages([read_json(x) for x in stages_files], neural['states'], count_times, seconds)
    gradients, cosines = [], {}
    expected_gradient_paths = set()
    summaries, expected_summaries = [], []
    for state in STATES:
        for case in protocol['cases']:
            for mode in MODES:
                expected_summaries.append((out / 'paths' / state / case['id'] / mode / 'summary.json', state, case, mode, None))
                if (case['update'], case['slot']) in REPEAT_POSITIONS:
                    expected_summaries.append((out / 'repeats' / state / case['id'] / mode / 'summary.json', state, case, mode, None))
            if (case['update'], case['slot']) in GRADIENT_POSITIONS:
                for term in terms_for(case):
                    path = out / 'gradients' / state / case['id'] / (term + '.json')
                    gradients.append(audit_gradient(read_json(path), state, case, term))
                    expected_gradient_paths.add(path)
        cosines[state] = audit_cosines(read_json(out / 'gradients' / state / 'cosines.json'), state)
        audit_family_norms(cosines[state], [row for row in gradients if row['state'] == state])
    expected_gradient_paths.update(out / 'gradients' / state / 'cosines.json' for state in STATES)
    require(set((out / 'gradients').rglob('*.json')) == expected_gradient_paths and len(gradients) == 64,
            'Gradient file census differs')
    for case in protocol['cases']:
        for arm in ARMS:
            expected_summaries.append((out / 'interventions' / case['id'] / arm / 'summary.json', 'stopped50', case, 'E32', arm))
    require(set(out.rglob('summary.json')) == {x[0] for x in expected_summaries} and len(expected_summaries) == 104,
            'Forward summary census differs')
    for path, state, case, mode, arm in expected_summaries:
        row = read_json(path)
        require(row['state'] == state and row['case_id'] == case['id'] and row['mode'] == mode
                and row['intervention'] == arm and row['input_sha256'] == case['input_sha256']
                and row['clean512_sha256'] == case['clean512_sha256'], 'Forward summary input/route identity differs')
        audit_execution(row, mode)
        summaries.append({'path': path.relative_to(root).as_posix(), 'state': state, 'case_id': case['id'], 'mode': mode, 'arm': arm})
    cpu = None
    require((parent_assets is None) == (stopped_checkpoint is None), 'Supply both CPU parent and stopped state paths')
    if parent_assets is not None:
        cpu = audit_cpu_assets(protocol, Path(parent_assets).resolve(strict=True), stopped_checkpoint, neural['states'])
    result = {'complete': True, 'evidence_consistency': True, 'protocol_sha256': protocol_sha,
        'auditor_source_sha256': sha256(Path(__file__)),
        'call_trace': trace, 'state_checks': stages, 'gradient_components': len(gradients),
        'cosine_comparisons': sum(map(len, cosines.values())), 'scalar_cosine_group_checks': 60,
        'family_component_triangle_and_repeated_norm_checks': 120,
        'execution_summaries': len(summaries), 'CPU_asset_state_verification': cpu,
        'timing': {'neural_seconds': seconds, 'worker_seconds': worker_seconds,
                   'supervisor_run_seconds': supervisor['stages'][1]['seconds']},
        'CUDA_peak_reserved_bytes': neural['CUDA_peak_reserved_bytes'], 'runtime': runtime,
        'optimizer_updates': 0, 'model_qualified': False, 'next_training_authorized': False,
        'limits': ['No local neural, gradient, autograd, optimizer, GPU or VM execution.',
            'Reservations and immutable runtime state are source-bound recorded evidence, not independent runtime instrumentation.',
            'Loss reductions and cosine formulas replay scalar sufficient statistics; full gradients were intentionally not archived and cannot be replayed.',
            'No successful Adam update, quality, identity or native CCTV qualification follows from diagnostic completion.']}
    outputdir.mkdir(parents=True)
    write_new_json(outputdir / 'state_audit.json', result)
    write_new_json(outputdir / 'gradient_components.json', gradients)
    write_new_json(outputdir / 'gradient_cosines.json', cosines)
    write_new_json(outputdir / 'execution_conditions.json', summaries)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--outputdir', type=Path, required=True)
    parser.add_argument('--parent-assets', type=Path)
    parser.add_argument('--stopped-checkpoint', type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.packet, args.outputdir, parent_assets=args.parent_assets,
                   stopped_checkpoint=args.stopped_checkpoint)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
