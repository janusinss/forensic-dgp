"""Independent saved-evidence CPU audit. Never constructs or runs a model."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from . import contract as c
from .numerical_policy import canary_replay

PROTOCOL_SHA = 'c3b0f5f0f3ff83b508625df0259ed0c9acbc2d071da2fdde066a2bda429a15eb'
ARCHIVE_SHA = '3e1a0a4bee698b397747ba39d3546cbca3cbd298e9cb450bc1abfc61aa880bb9'
MANIFEST_SHA = 'c9286d80ac6586aa281aa91bb00b6b430521b2d797f79308f450559c244548b3'
PRIOR_SHA = '66fc747b0de090d398058d9a7504c12e9847afb5124277caa6e2aaf58759c231'
NUMERIC_SHA = 'ea81c31527ff852ee5ea768b62aae546864a9e60bd678894a97c21ef4dd7d4c9'
STATES = {'original': '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd',
          'stopped': '51605ee7908c40a457e41ac03c614211e8028cc0386b85d13aa59a3d19a182eb'}
SITES = {'32': (256, 32, 32), '64': (256, 64, 64), '128': (128, 128, 128), '256': (128, 256, 256)}
MODES = ('original', 'stopped', 'rollback', 'bounded_constant', 'bounded_fixed')
HEX = re.compile(r'^[a-f0-9]{64}$')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def observed_gate(pixels, gate):
    require(pixels.dtype == np.uint8 and pixels.shape == (256, 256, 3), 'Exact RGB256 input required')
    y = pixels.astype(np.float64) / 255. @ np.asarray([.2126, .7152, .0722], np.float64)
    z = float(y[32:224, 32:224].mean())
    t = min(1., max(0., (gate['q90'] - z) / (gate['q90'] - gate['q10'])))
    return {'central192_sRGB_mean_luminance': z, 'gate': .25 + .75 * t * t * (3 - 2 * t)}


def plan_rows(p):
    rows = [(case['id'], mode, 'paths') for case in p['cases'] for mode in MODES]
    rows += [(uid, mode, 'repeats') for uid in p['canary_ids'] for mode in ('original', 'stopped')]
    rows += [(uid, mode, 'zero_delta') for uid in p['canary_ids'] for mode in ('bounded_constant', 'bounded_fixed')]
    rows += [(uid, 'native_identity', 'identity') for uid in p['canary_ids']]
    require(len(rows) == 170 and len(set(rows)) == 170, 'Exact independent170 path plan required')
    return rows


def expected_events(p):
    events = []
    for uid, mode, section in plan_rows(p):
        events.append(('codeformer_forwards', {'case_id': uid, 'mode': mode, 'section': section}))
        for site in SITES:
            context = {'case_id': uid, 'mode': mode, 'section': section, 'site': site,
                       'directory': section + '/' + uid + '/' + mode}
            roles = ([(mode, 'native')] if mode in ('original', 'stopped') else
                     [('original', 'rollback')] if mode == 'rollback' else
                     [('original', 'base'), ('original' if section == 'zero_delta' else 'stopped', 'delta_source')])
            for state, role in roles:
                events.append(('fusion_block_evaluations', {**context, 'weights': state, 'role': role}))
    require(len(events) == 1130, 'Exact1130 reservation sequence required')
    return events


def expected_namespace(p):
    out = p['outputs']
    names = set(p['export_files']) | {'protocol.json'}
    names.update('outputs/supervision/' + name for name in ('preflight.log', 'diagnostic.log', 'run_receipt.json'))
    names.update(out + '/' + name for name in ('runtime.json', 'calibration.json', 'initialization.json',
        'state_checks/initial.json', 'state_checks/terminal.json', 'manual_launch.json',
        'run_outputs.json', 'worker_complete.json', 'export_manifest.json'))
    names.update(out + '/call_trace/%04d.json' % i for i in range(1, 1131))
    for uid, mode, section in plan_rows(p):
        directory = out + '/' + section + '/' + uid + '/' + mode
        names.update(directory + '/' + name for name in
                     ('native512.npy', 'raw256.npy', 'native.png', 'actual_tokens.npy', 'record.json', 'fusion_stats.json'))
        if section == 'paths' and uid in p['canary_ids'] and mode in ('bounded_constant', 'bounded_fixed'):
            names.update(directory + '/features/' + site + '/' + name + '.npy'
                         for site in ('32', '64') for name in ('original', 'stopped', 'used_delta'))
    return names


def check_trace(rows, p):
    expected = expected_events(p)
    require(len(rows) == len(expected), 'Reservation count differs')
    counts = {name: 0 for name in p['call_caps']}
    last = -1.
    for i, (row, (kind, context)) in enumerate(zip(rows, expected), 1):
        counts[kind] += 1
        require(type(row['event']) is int and row['event'] == i and row['before_call'] == kind and row['context'] == context,
                'Reservation sequence/context differs at event%d' % i)
        require(c.exact(row['counts'], counts), 'Reservation cumulative counters differ at event%d' % i)
        elapsed = row['elapsed_seconds']
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and last <= elapsed <= p['limits']['worker_seconds'],
                'Reservation monotonic/finite time differs')
        last = elapsed
    require(counts == p['call_caps'], 'Successful endpoint cap census differs')
    return {'events': len(rows), 'counts': counts, 'last_elapsed_seconds': last, 'exact_order_and_context': True}


def check_tensor_capture(row, shape):
    require(set(row) == {'shape', 'dtype', 'sha256'} and row['shape'] == list(shape) and
            row['dtype'] == 'float32' and isinstance(row['sha256'], str) and HEX.fullmatch(row['sha256']),
            'Invalid captured content shape/dtype/hash')


def check_content(row, tokens):
    require(set(row) == {'encoder_skips', 'pre_generator', 'pre_generator_mean', 'pre_generator_std',
                         'logits', 'latent', 'selected_tokens_sha256'} and set(row['encoder_skips']) == set(SITES),
            'Incomplete captured stopped content')
    for site, shape in SITES.items():
        check_tensor_capture(row['encoder_skips'][site], (1, *shape))
    for name, shape in [('pre_generator', (1, 256, 16, 16)), ('latent', (1, 256, 16, 16)),
                        ('pre_generator_mean', (1, 256, 1, 1)), ('pre_generator_std', (1, 256, 1, 1)),
                        ('logits', (1, 256, 1024))]:
        check_tensor_capture(row[name], shape)
    require(row['selected_tokens_sha256'] == digest_array(tokens), 'Actual token array differs from content capture')


def gate_match(reported, expected):
    if expected is None:
        return reported is None
    return (type(reported) in (float, int) and math.isfinite(reported) and
            abs(reported - expected) <= 1e-12 and np.float32(reported) == np.float32(expected))


def channel_stats(row, site, gate, zero=False):
    require(row['site'] == site and gate_match(row['gate'], gate) and row['dtype'] == 'torch.float32' and
            row['block_evaluations'] == 2 and row['nonfinite_count'] == 0 and
            row['local_cap_with_FP32_roundoff_pass'] is True, 'Bounded-site status/gate differs')
    names = ['D', 'difference_absmax', 'difference_RMS', 'used_delta_absmax',
             'used_delta_RMS', 'tanh_abs_ge_095_fraction']
    arrays = {name: np.asarray(row[name], np.float64) for name in names}
    require(all(a.shape == (SITES[site][0],) and np.isfinite(a).all() for a in arrays.values()),
            'Wrong channel census/nonfinite bounded statistics')
    d = arrays['D']; eps = np.finfo(np.float32).eps
    require(np.all(d > 0) and all(np.all(arrays[name] >= 0) for name in names[1:]),
            'Invalid negative bound/statistic')
    tolerance = 8 * eps * d * np.float32(gate) + np.finfo(np.float32).tiny
    require(np.all(arrays['used_delta_absmax'] <= d * np.float32(gate) + tolerance),
            'Returned channel feature cap fails independently')
    for prefix in ('difference', 'used_delta'):
        require(np.all(arrays[prefix + '_RMS'] <= arrays[prefix + '_absmax'] * (1 + 8 * eps) + np.finfo(np.float32).tiny),
                'RMS exceeds channel maximum')
    fractions = arrays['tanh_abs_ge_095_fraction']; n = SITES[site][-1] ** 2
    require(np.all((fractions >= 0) & (fractions <= 1)) and np.all(fractions * n == np.floor(fractions * n)),
            'Saturation fractions are not valid finite pixel counts')
    if zero:
        require(all(np.count_nonzero(arrays[name]) == 0 for name in names[1:]), 'Original-original local delta not zero')
    return arrays


def replay_canary(base, other, used, gate, stats):
    reference = canary_replay(base, other, gate)
    require(used.dtype == np.float32 and used.shape == base.shape and np.isfinite(used).all(), 'Invalid saved used delta')
    error = np.abs(used.astype(np.float64) - reference['used_delta'].astype(np.float64))
    require(np.all(error <= reference['tolerance']), 'Frozen prospective feature replay allowance exceeded')
    eps = np.finfo(np.float32).eps
    n = reference['n']; gamma = reference['gamma_n']
    # Prospective positive-reduction envelope, not fitted to returned numbers.
    # Maxima use the actual returned tensor and are order-independent/exact.
    actual_max = np.max(np.abs(used), axis=(-2, -1)).reshape(-1)
    require(np.array_equal(actual_max, np.asarray(stats['used_delta_absmax'], np.float32)), 'Actual canary maxima differ')
    rms64 = np.sqrt(np.mean(np.square(used.astype(np.float64)), axis=(-2, -1))).reshape(-1)
    returned_rms = np.asarray(stats['used_delta_RMS'], np.float64)
    rms_bound = 8 * gamma * np.maximum(rms64, np.finfo(np.float32).tiny) + 64 * eps * (1 + rms64)
    require(np.all(np.abs(rms64 - returned_rms) <= rms_bound), 'Actual canary RMS reduction differs beyond analytic bound')
    d_cpu = reference['scale'].astype(np.float64).reshape(-1)
    d_returned = np.asarray(stats['D'], np.float64)
    d_bound = 8 * gamma * d_cpu + 64 * eps * (1 + d_cpu)
    require(np.all(np.abs(d_returned - d_cpu) <= d_bound), 'Saved D reduction differs beyond analytic bound')
    return {'elements': int(used.size), 'maximum_replay_error': float(error.max()),
            'maximum_frozen_allowance': float(reference['tolerance'].max()),
            'maximum_error_fraction_of_allowance': float(np.max(error / reference['tolerance'])),
            'frozen_feature_replay_pass': True, 'actual_used_maxima_exact': True,
            'RMS_D_reduction_gamma_envelope_pass': True, 'n': n}


def parent_state_audit(original_path, stopped_path, assets):
    """Fresh raw hashes bind the already closed independent CPU state audit."""
    result = {}
    for label, path in [('original', original_path), ('stopped', stopped_path)]:
        path = c.no_links(path)
        require(path.stat().st_size == assets[label]['bytes'] and c.sha256(path) == assets[label]['sha256'],
                'Retained immutable parent checkpoint bytes differ')
        result[label] = {'path': str(path), 'sha256': assets[label]['sha256'], 'bytes': path.stat().st_size,
                         'prior_CPU_weights_only_state_sha256': STATES[label], 'prior_tensor_count': 515,
                         'prior_closed_CPU_state_audit_sha256': PRIOR_SHA, 'tensors_loaded_again': False}
    return result


def audit(imported, packet, transport_path, prior_path, original_path, stopped_path):
    imported, packet = Path(imported).resolve(strict=True), Path(packet).resolve(strict=True)
    t = c.read_json(transport_path)
    require(t.get('complete') is True and t.get('archive_sha256') == ARCHIVE_SHA and
            t.get('protocol_sha256') == PROTOCOL_SHA and t.get('manifest_sha256') == MANIFEST_SHA,
            'Closed exact strict transport receipt required')
    require(Path(t['import_directory']).resolve(strict=True) == imported,
            'Transport receipt binds another import directory')
    # Exact transport flag/schema keys are finalized with the independent
    # transport auditor before this module is allowed to run on real evidence.
    require(t.get('exact_namespace_verified') is True and
            t.get('whole_archive_and_all_member_hashes_verified_before_import') is True and
            t.get('member_maps_include_manifest') is True, 'Strict import was not closed before destination creation')
    files = t['files_sha256']; sizes = t['files_bytes']
    require(set(files) == set(sizes), 'Closed transport inventory differs')
    require(t['member_maps_include_manifest'] is True and files[t['manifest_member']] == MANIFEST_SHA and
            t['archive_bytes'] == 668557079 and t['members'] == len(files) and
            t['payload_bytes'] == sum(sizes.values()), 'Closed transport byte/member accounting differs')
    require(c.sha256(packet / 'protocol.json') == PROTOCOL_SHA, 'Frozen protocol differs')
    p = c.verify_transfer(packet, PROTOCOL_SHA, verify_parent=False)
    require(set(files) == expected_namespace(p) and
            {path.relative_to(imported).as_posix() for path in imported.rglob('*') if path.is_file()} == set(files),
            'Exact planned successful-result namespace differs')
    metadata_bytes = sum(size for name, size in sizes.items() if Path(name).suffix.lower() in
                         {'.json', '.jsonl', '.log', '.txt', '.md', '.py', '.sh'})
    require(len(files) <= p['limits']['regular_files'] and sum(sizes.values()) <= p['limits']['result_bytes'] and
            metadata_bytes <= p['limits']['metadata_bytes'] and t['archive_bytes'] <= p['limits']['archive_bytes'],
            'Actual return file/payload/archive/metadata resource ceiling exceeded')
    require(c.sha256(Path(__file__).with_name('numerical_policy.py')) == NUMERIC_SHA and
            p['files_sha256']['research/codeformer_bounded_fusion_feasibility_v1/numerical_policy.py'] == NUMERIC_SHA,
            'Frozen independent numerical replay source differs')
    prior = c.read_json(prior_path)
    require(c.sha256(prior_path) == PRIOR_SHA and prior.get('evidence_consistency_pass') is True,
            'Prior independently audited original/stopped dependency evidence differs')
    states = prior['sections']['checkpoints']['states']
    require(states[0]['model_state_sha256'] == STATES['original'] and
            any(row['model_state_sha256'] == STATES['stopped'] and row['completed_updates'] == 50 for row in states),
            'Prior independently verified model-state identities differ')
    parents = parent_state_audit(original_path, stopped_path, p['assets'])

    def bound(name):
        require(name in files and name in sizes, 'Returned artifact absent from strict inventory: ' + name)
        f = c.bounded(imported, name)
        require(f.stat().st_size == sizes[name] and c.sha256(f) == files[name], 'Returned artifact changed: ' + name)
        return f

    def js(name):
        return c.read_json(bound(name))

    def receipt_file(row, expected):
        require(set(row) == {'path', 'sha256', 'bytes'} and row['path'] == expected and
                files[expected] == row['sha256'] and sizes[expected] == row['bytes'], 'Saved artifact receipt/path differs')
        return bound(expected)

    def array(row, expected, shape, dtype):
        f = receipt_file(row, expected)
        a = np.load(f, allow_pickle=False)
        require(a.dtype == dtype and a.shape == shape and a.flags.c_contiguous and np.isfinite(a).all(),
                'Saved array shape/dtype/finiteness differs: ' + expected)
        return a

    out = p['outputs']
    result = js(out + '/run_outputs.json'); worker = js(out + '/worker_complete.json')
    require(result['complete'] is True and worker['complete'] is True and
            c.exact(result['counts'], p['call_caps']) and c.exact(worker['counts'], p['call_caps']) and worker['measurement_result'] == result,
            'Worker/result terminal status or exact counts differ')
    for name in ('state_before_after_equal', 'parameter_grad_fields_all_none', 'content_preservation_verified',
                 'repeat_zero_identity_canaries_pass', 'all_content_and_canaries_pass'):
        require(result.get(name) is True, 'Missing complete control claim: ' + name)
    require(result['observations'] == 32 and result['optimizer_updates'] == result['gradient_queries'] == 0 and
            result['model_qualified'] is False and result['training_authorized'] is False,
            'Measurement is not a qualified/trained replacement')
    require(not any(Path(name).suffix.lower() in {'.pt', '.pth', '.pkl', '.pickle', '.ckpt', '.safetensors'} for name in files),
            'Measurement exported forbidden model/optimizer payload')
    require(not any(name.endswith('/worker_failure.json') for name in files), 'Returned failure cannot be hidden by completion')
    trace_names = sorted(name for name in files if name.startswith(out + '/call_trace/'))
    require(trace_names == [out + '/call_trace/%04d.json' % i for i in range(1, 1131)], 'Exact trace namespace differs')
    trace = check_trace([js(name) for name in trace_names], p)
    initial, terminal = js(out + '/state_checks/initial.json'), js(out + '/state_checks/terminal.json')
    require(initial['states'] == terminal['states'] == STATES and initial['all_grad_none'] is terminal['all_grad_none'] is True,
            'Before/after state or empty-gradient evidence differs')
    require(initial['counts'] == {name: 0 for name in p['call_caps']} and terminal['counts'] == p['call_caps'],
            'State-boundary counter census differs')
    initialization = js(out + '/initialization.json')
    launch = js(out + '/manual_launch.json')
    require(type(launch['pid']) is int and launch['pid'] > 0 and launch['protocol_sha256'] == PROTOCOL_SHA and
            launch['manual_tmux'] is True and launch['automatic_training'] is False and
            all(worker[key] == value for key, value in launch.items()), 'Closed manual worker identity differs')
    require(initialization['states'] == STATES and initialization['optimizer_constructed'] is False and
            initialization['inference_frozen_tensor_counts'] == {'original': 515, 'stopped': 515},
            'All515 inference tensors must be frozen in both states')
    require(initialization['asset_sha256'] == {key: row['sha256'] for key, row in p['assets'].items()}, 'Initialization parent asset binding differs')
    runtime = js(out + '/runtime.json')
    require(all(runtime[k] == v for k, v in p['runtime_expected'].items()) and
            runtime['autocast'] is runtime['TF32'] is runtime['cudnn_TF32'] is False and
            runtime['deterministic_algorithms'] is True and runtime['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8',
            'Actual runtime is not the declared deterministic FP32 path')
    gates = {}
    for row in p['calibration_inputs']:
        f = c.bounded(packet, row['input_path']); require(c.sha256(f) == row['input_sha256'], 'Calibration PNG changed')
        with Image.open(f) as im:
            require(im.mode == 'RGB' and im.size == (256, 256), 'Calibration RGB geometry differs')
            pixels = np.asarray(im).copy()
        gates[row['id']] = observed_gate(pixels, p['gate'])
    q = np.quantile([v['central192_sRGB_mean_luminance'] for v in gates.values()], [.1, .9], method='linear')
    require(abs(float(q[0]) - p['gate']['q10']) <= 1e-12 and abs(float(q[1]) - p['gate']['q90']) <= 1e-12,
            'Input-only calibration quantiles differ')
    saved_cal = js(out + '/calibration.json')
    require(saved_cal['profile_or_target_used_in_gate'] is False and len(saved_cal['inputs']) == 200 and
            saved_cal['roundoff_only_abs_bound'] == 1e-12, 'Calibration source/scope differs')
    for a, b in zip(saved_cal['inputs'], p['calibration_inputs']):
        require((a['update'], a['slot'], a['input_sha256']) == (b['update'], b['slot'], b['input_sha256']) and
                all(abs(a[key] - gates[b['id']][key]) <= 1e-12 for key in gates[b['id']]), 'Returned gate calibration differs')
    expected = plan_rows(p); require(len(result['results']) == 170, 'Path index must contain170 records')
    cases = {r['id']: r for r in p['cases']}
    records, hashes, bounded_rows, canaries = {}, {}, 0, []
    for index, ((uid, mode, section), item) in enumerate(zip(expected, result['results']), 1):
        require((item['case_id'], item['mode'], item['section']) == (uid, mode, section), 'Path index order differs')
        directory = out + '/' + section + '/' + uid + '/' + mode
        rec_path = receipt_file(item['record'], directory + '/record.json'); record = c.read_json(rec_path)
        require((record['case_id'], record['mode'], record['section']) == (uid, mode, section), 'Record route identity differs')
        case = cases[uid]
        require(record['input_sha256'] == case['input_sha256'] and record['input_path'] == case['input_path'] and
                record['reference_id'] == case['reference_id'] and record['read_only_model_state_bindings'] == STATES,
                'Path input/dependency state binding differs')
        require(all(abs(record['gate_observed_input'][key] - gates[uid][key]) <= 1e-12 for key in gates[uid]),
                'Path gate uses wrong observed input')
        applied = gates[uid]['gate'] if mode == 'bounded_fixed' else 1. if mode == 'bounded_constant' else 0. if mode == 'rollback' else None
        require(gate_match(record['applied_gate'], applied), 'Actual gate route differs independently of feature tolerance')
        require(record['actual_dtypes'] == {'native': 'torch.float32', 'logits': 'torch.float32',
                'probabilities': 'torch.float32', 'autocast_enabled': False} and record['model_qualified'] is False,
                'Actual path precision/classification differs')
        artifacts = record['artifacts']
        native = array(artifacts['native512'], directory + '/native512.npy', (512, 512, 3), np.float32)
        raw = array(artifacts['raw256'], directory + '/raw256.npy', (256, 256, 3), np.float32)
        tokens = array(artifacts['tokens'], directory + '/actual_tokens.npy', (256,), np.int64)
        receipt_file(artifacts['PNG'], directory + '/native.png')
        require(np.all((tokens >= 0) & (tokens < 1024)) and raw.min() >= 0 and raw.max() <= 1, 'Invalid actual token/raw range')
        hp = {'native512': digest_array(native), 'raw256': digest_array(raw), 'tokens': digest_array(tokens)}
        require(record['pixel_hashes'] == hp, 'Saved pixel/token hashes differ')
        check_content(record['content'], tokens)
        expected_state = 'original' if mode == 'original' or section == 'zero_delta' else 'stopped'
        require(record['neural_state'] == expected_state and
                record['content_exact_to_stopped'] is (section != 'zero_delta' and mode != 'original'), 'Path content flag/state differs')
        require(record['targets_for_independent_metrics_only'] == {key: case[key] for key in
                ('target512_path', 'target512_sha256', 'target256_path', 'target256_sha256')}, 'Metric-only target binding differs')
        if section == 'paths':
            if mode not in ('original', 'stopped'):
                require(record['content'] == records[uid, 'stopped', 'paths']['content'], 'Stopped content changed across fusion modes')
        else:
            baseline_mode = 'original' if section == 'zero_delta' else 'stopped' if section == 'identity' else mode
            require(record['content'] == records[uid, baseline_mode, 'paths']['content'] and
                    hp == hashes[uid, baseline_mode, 'paths'], 'Exact repeat/zero/identity canary fails')
        site_rows = js(directory + '/fusion_stats.json')
        require(len(site_rows) == 4 and [r['site'] for r in site_rows] == list(SITES), 'Fusion-site order/census differs')
        for site, stats in zip(SITES, site_rows):
            if mode in ('bounded_constant', 'bounded_fixed'):
                channel_stats(stats, site, applied, section == 'zero_delta'); bounded_rows += 1
                capture = section == 'paths' and uid in p['canary_ids'] and site in ('32', '64')
                require(('arrays' in stats) == capture, 'Exact canary feature-array scope differs')
                if capture:
                    fs = stats['arrays']; require(set(fs) == {'original', 'stopped', 'used_delta'}, 'Missing canary feature array')
                    a = {name: array(row, directory + '/features/' + site + '/' + name + '.npy', (1, *SITES[site]), np.float32)
                         for name, row in fs.items()}
                    canaries.append({'case_id': uid, 'mode': mode, 'site': site,
                                     **replay_canary(a['original'], a['stopped'], a['used_delta'], applied, stats)})
            else:
                route = mode if mode in ('original', 'stopped', 'native_identity') else 'original'
                require(stats['route'] == route and stats['block_evaluations'] == (2 if mode == 'native_identity' else 1),
                        'Unbounded/native fusion route differs')
        records[uid, mode, section] = record; hashes[uid, mode, section] = hp
        del native, raw, tokens
    require(bounded_rows == 272 and len(canaries) == 8, 'Bounded/canary proof census differs')
    supervision = js('outputs/supervision/run_receipt.json')
    require(supervision['protocol_sha256'] == PROTOCOL_SHA and supervision['neural_measurement_complete'] is True and
            supervision['optimizer_updates'] == 0 and supervision['model_qualified'] is False and
            supervision['initial_idle_check']['GPU_compute_processes'] == [] and
            supervision['initial_idle_check']['competing_research_commands'] == [], 'Supervisor status/workload evidence differs')
    for row, flag, cap, grace in zip(supervision['stages'], ('--verify-transfer', '--run'), (300, 900), (30, 60)):
        require(row['stage'] == flag and row['exit_code'] == 0 and row['external_cap_seconds'] == cap and
                row['grace_seconds'] == grace and row['log_capped'] is False and 0 <= row['seconds'] <= cap + grace,
                'Finite supervisor stage differs')
    require(len(supervision['stages']) == 2 and worker['seconds'] <= p['limits']['worker_seconds'] and
            worker['stage_seconds']['initialization'] <= 300 and worker['stage_seconds']['measurement'] <= 600,
            'Worker finite timing differs')
    memory = result['memory']; require(0 <= memory['CUDA_current_reserved_bytes'] <= memory['CUDA_peak_reserved_bytes'] <=
            p['limits']['max_reserved_GPU_bytes'] and memory['Linux_host_peak_RSS_bytes'] > 0, 'Memory status differs')
    source_checks = 0
    for name in p['export_files']:
        require(name in files and files[name] == p['files_sha256'][name], 'Returned executing/static source differs')
        bound(name); source_checks += 1
    return {'complete': True, 'evidence_consistency_pass': True, 'protocol_sha256': PROTOCOL_SHA,
            'archive_sha256': ARCHIVE_SHA, 'trace': trace, 'paths_checked': 170, 'bounded_site_statistics_checked': bounded_rows,
            'feature_replay': canaries, 'feature_arrays_replayed': 24, 'exact_canary_paths': 10,
            'all_stopped_content_captures_exact': True, 'all_token_arrays_and_native_raw_hashes_checked': True,
            'all200_gates_recomputed_from_observed_TRAIN_PNG': True, 'states': STATES,
            'prior_dependency_audit_sha256': PRIOR_SHA, 'runtime': runtime, 'worker_seconds': worker['seconds'],
            'independent_CPU_parent_state_audit': parents,
            'worker_stage_seconds': worker['stage_seconds'], 'supervised_stages': supervision['stages'], 'memory': memory,
            'returned_static_sources_verified': source_checks, 'optimizer_updates': 0, 'gradient_queries': 0,
            'actual_resource_accounting': {'members': len(files), 'payload_bytes': sum(sizes.values()),
                                           'metadata_bytes': metadata_bytes, 'archive_bytes': t['archive_bytes']},
            'model_qualified': False, 'training_authorized': False,
            'limits': ['Model-state/empty-gradient and intermediate-content hashes are frozen-source runtime captures, not a new independent live neural probe.',
                       'Fresh raw checkpoint hashes bind prior independent CPU weights_only state digests. No model tensors were reloaded, and the return contains no live terminal model tensor export. Reported unchanged VM state is source-bound evidence, not rerun inference.',
                       'Only24 saved internal feature arrays permit arithmetic replay. Other channel statistics/content hashes do not reconstruct full logits, encoder tensors or all activations.',
                       'Feature replay uses the prospectively frozen CPU/CUDA allowance. Scalar RMS/D comparisons use analytically specified positive-reduction gamma envelopes, not output-quality tolerances.',
                       'No NN, CUDA, gradient, optimizer, model construction or VM call occurred locally. PNG/delivery/paired quality and all visual judgments are a separate audit.']}


def run(imported, packet, transport, prior, original_weights, stopped_weights, outputdir):
    outputdir = Path(outputdir).resolve()
    require(not outputdir.exists(), 'Preserve existing audit receipt/directory')
    outputdir.mkdir(parents=True)
    envelope = {'UTC': datetime.now(timezone.utc).isoformat(), 'auditor_sha256': c.sha256(__file__),
                'local_model_instances': 0, 'local_neural_calls': 0, 'local_gradient_queries': 0,
                'local_optimizer_updates': 0, 'VM_calls': 0}
    try:
        result = {**envelope, **audit(imported, packet, transport, prior, original_weights, stopped_weights), 'issues': []}
        code = 0
    except Exception as error:
        result = {**envelope, 'complete': False, 'evidence_consistency_pass': False,
                  'issues': [{'type': type(error).__name__, 'message': str(error)}],
                  'model_qualified': False, 'training_authorized': False}
        code = 1
    c.write_new_json(outputdir / 'state_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('feature_replay', 'limits', 'runtime', 'supervised_stages')}, indent=2))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('imported', 'packet', 'transport', 'prior', 'original-weights', 'stopped-weights', 'outputdir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.imported, args.packet, args.transport, args.prior,
                         args.original_weights, args.stopped_weights, args.outputdir))
