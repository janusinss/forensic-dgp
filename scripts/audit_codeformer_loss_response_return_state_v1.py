"""Independent saved-state/gradient/ledger audit; never executes returned code.

Only stdlib and NumPy saved-array arithmetic are used. No model, autograd,
optimizer, Torch import, VM call, archive extraction or new image is produced.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = '23c19c394e34261861b8853cfd60aa60d1cfaefc91b12e92a2f8b8dce85112ab'
ARCHIVE_SHA = 'b0e2b6d0aa245fcebe334c21c1b50cfbfde3ca65d494bb1d1fd8a9d17730de86'
RUN = 'outputs/loss_response_v1'
COMPONENTS = ('token_ce', 'latent_mse', 'pixel_l1_rgb', 'lpips_vgg',
              'coarse_rgb_l1', 'behavior_rgb_l1', 'behavior_token_kl')
WEIGHTS = dict(zip(COMPONENTS, (.5, 1., 1., 1., 4., 2., .1)))
CONDITIONS = ('baseline', 'full', *('drop:' + item for item in COMPONENTS), 'momentum_only')
CAPS = {'candidate_source_forwards': 32, 'candidate_probe_forwards': 320,
    'restoration_repeat_forwards': 32, 'behavior_teacher_forwards': 4,
    'initializer_parity_candidate_forwards': 2, 'initializer_parity_benchmark_forwards': 2,
    'HQ_teacher_forwards': 8, 'perceptual_source_forwards': 24,
    'perceptual_prerequisite_forwards': 1, 'component_gradient_queries': 160,
    'full_gradient_queries': 32, 'perceptual_prerequisite_gradient_queries': 1,
    'optimizer_step_attempts': 18, 'disposable_optimizer_steps': 18,
    'committed_training_updates': 0, 'backward_calls': 0}
EXPECTED_PARTITION = {'encoder': 164, 'predictor': 114, 'fusion': 72}
IMAGE_COMPONENTS = {'pixel_l1_rgb', 'lpips_vgg', 'coarse_rgb_l1', 'behavior_rgb_l1'}
HEX = re.compile(r'^[a-f0-9]{64}$')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(value)
    return digest.hexdigest()


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def partition(name):
    if name.startswith('encoder.'):
        return 'encoder'
    if name.startswith('fuse_convs_dict.'):
        return 'fusion'
    require(name.split('.')[0] in {'position_emb', 'feat_emb', 'ft_layers', 'idx_pred_layer'},
            'Undeclared trainable parameter: ' + name)
    return 'predictor'


def close(actual, expected, message, *, rtol=2e-10, atol=1e-13):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol), message)


def expected_terms(case):
    return tuple(item for item in COMPONENTS if item in
        ({'token_ce', 'latent_mse', 'coarse_rgb_l1'} |
         ({'pixel_l1_rgb', 'lpips_vgg'} if case['parameters']['image_eligible'] else set()) |
         ({'behavior_rgb_l1', 'behavior_token_kl'} if case['slot'] == 3 else set())))


def expected_events(cases):
    events = [('perceptual_prerequisite_forwards', None),
              ('perceptual_prerequisite_gradient_queries', None)]
    seen = set()
    for case in cases:
        if case['reference_id'] not in seen:
            seen.add(case['reference_id'])
            events.append(('HQ_teacher_forwards', {'reference_id': case['reference_id']}))
    for case in cases[:2]:
        for name in ('initializer_parity_candidate_forwards', 'initializer_parity_benchmark_forwards'):
            events.append((name, {'case': case['id']}))
    for case in cases:
        if case['slot'] == 3:
            events.append(('behavior_teacher_forwards', {'case': case['id']}))
    for snapshot, base in (('original0', 0), ('stopped50', 50)):
        for case in cases:
            context = {'snapshot': snapshot, 'case': case['id']}
            events.append(('candidate_source_forwards', context))
            if case['parameters']['image_eligible']:
                events.append(('perceptual_source_forwards', context))
            events.extend(('component_gradient_queries', {**context, 'component': component})
                          for component in expected_terms(case))
            events.append(('full_gradient_queries', context))
        for condition in CONDITIONS:
            if condition != 'baseline':
                events.append(('optimizer_step_attempts', {'snapshot': snapshot, 'condition': condition,
                    'base_counter': base, 'disposable_counter_after': base + 1}))
            events.extend(('candidate_probe_forwards', {'snapshot': snapshot, 'condition': condition,
                                                      'case': case['id']}) for case in cases)
        events.extend(('restoration_repeat_forwards', {'snapshot': snapshot, 'case': case['id']})
                      for case in cases)
    return events


def verify_ledger(rows, cases):
    expected = expected_events(cases)
    require(len(rows) == len(expected) == 636, 'Exact636 ordered reservation files required')
    counts = dict.fromkeys(CAPS, 0)
    completed, last_seconds, previous = 0, -1., None
    for index, (row, (name, context)) in enumerate(zip(rows, expected), start=1):
        if previous == 'optimizer_step_attempts':
            completed += 1; counts['disposable_optimizer_steps'] += 1
        counts[name] += 1
        require(row['event'] == index and row['before_call'] == name and row['context'] == context,
                'Call order/context differs at event%d' % index)
        require(row['counts'] == counts and row['actual_completed_Adam_steps'] == completed,
                'Reserved versus completed counters differ at event%d' % index)
        require(row['committed_training_updates'] == row['backward_calls'] == 0,
                'Ledger claims committed training/backward')
        require(type(row['seconds']) in (int, float) and row['seconds'] >= last_seconds,
                'Nonmonotonic ledger time')
        last_seconds, previous = row['seconds'], name
    require(counts == CAPS and completed == 18, 'Terminal exact counters differ')
    return {'events': len(rows), 'counts': counts, 'completed_steps': completed,
            'last_event_seconds': last_seconds, 'order_context_counter_replay_pass': True}


def check_sum_capture(record):
    require(record.get('pass') is True and record['full_gradient_vector_exported'] is False,
            'Full-sum capture status differs')
    require(record['atol'] == 2e-5 and record['rtol'] == 2e-4, 'Frozen gradient-sum tolerance differs')
    bound = 2e-5 + 2e-4 * record['maximum_full_gradient_absolute']
    close(record['bound'], bound, 'Gradient-sum bound arithmetic differs')
    require(0 <= record['maximum_absolute'] <= bound and record['difference_l2'] >= 0,
            'Full-sum capture exceeds declared mechanism bound')


class SavedAudit:
    def __init__(self, root, np):
        self.root, self.np = Path(root).resolve(strict=True), np
        self.manifest = read_json(self.root / RUN / 'export_manifest.json')
        self.hashes = self.manifest['files_sha256']; self.sizes = self.manifest['files_bytes']
        require(self.hashes.keys() == self.sizes.keys(), 'Exact returned manifest hash/size namespace required')
        self.read_metadata, self.read_arrays = {}, 0

    def path(self, name):
        relative = PurePosixPath(name)
        require(isinstance(name, str) and not relative.is_absolute() and relative.as_posix() == name and
                '..' not in relative.parts and '\\' not in name and ':' not in name,
                'Unsafe returned reference path')
        path = self.root.joinpath(*relative.parts)
        require(path.resolve(strict=True).is_relative_to(self.root) and not path.is_symlink() and path.is_file(),
                'Returned reference missing/linked/nonregular')
        return path

    def binding(self, name):
        path = self.path(name)
        require(name in self.hashes and path.stat().st_size == self.sizes[name],
                'Returned artifact not manifest-bound: ' + name)
        return path

    def metadata(self, name):
        path = self.binding(name)
        require(path.stat().st_size <= 8 * 1024**2 and sha256(path) == self.hashes[name],
                'Returned metadata bytes differ: ' + name)
        value = read_json(path)
        self.read_metadata[name] = self.hashes[name]
        return value

    def array(self, binding, shape=None, dtype=None, *, whole=True):
        path = self.binding(binding['path']); np = self.np
        require(binding['sha256'] == self.hashes[binding['path']] and
                binding['bytes'] == self.sizes[binding['path']], 'Array binding differs')
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        require(value.dtype.kind != 'O' and (shape is None or tuple(value.shape) == tuple(shape)) and
                (dtype is None or value.dtype == np.dtype(dtype)), 'Saved array shape/dtype differs')
        require(list(value.shape) == binding['shape'] and str(value.dtype) == binding['dtype'],
                'Declared array shape/dtype differs')
        if whole:
            require(np.isfinite(value).all() and sha256(path) == binding['sha256'] and
                    hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == binding['array_sha256'],
                    'Saved array values/hash differ')
        self.read_arrays += 1
        return value


def empty_stats():
    return {key: {'tensors': 0, 'none': 0, 'nonzero': 0, 'squared_l2': 0., 'maximum_absolute': 0.}
            for key in EXPECTED_PARTITION}


def add_stats(stats, name, array, np):
    group = stats[partition(name)]; group['tensors'] += 1
    if array is None:
        group['none'] += 1
        return
    maximum = float(np.max(np.abs(array)))
    group['nonzero'] += int(maximum > 0)
    group['squared_l2'] += float(np.square(array.astype(np.float64)).sum(dtype=np.float64))
    group['maximum_absolute'] = max(group['maximum_absolute'], maximum)


def finish_stats(stats):
    return {key: {**{name: value for name, value in row.items() if name != 'squared_l2'},
                  'l2': math.sqrt(row['squared_l2'])} for key, row in stats.items()}


def compare_stats(actual, expected):
    require(actual.keys() == expected.keys(), 'Gradient statistic partition differs')
    for key in actual:
        for field in ('tensors', 'none', 'nonzero'):
            require(actual[key][field] == expected[key][field], 'Gradient statistic census differs')
        for field in ('l2', 'maximum_absolute'):
            close(actual[key][field], expected[key][field], 'Gradient statistic reduction differs')


def compose_one(parts, omitted, momentum, np):
    result = None
    for component in COMPONENTS:
        if component == omitted or parts[component] is None:
            continue
        if result is None:
            result = parts[component].copy()
        else:
            np.add(result, parts[component], out=result)
    require(result is not None, 'Deletion composition unexpectedly drops the entire parameter route')
    return np.zeros_like(result) if momentum else result


def audit_gradients(audit, snapshot, cases, source_summary, branch_records):
    np = audit.np
    manifests = {component: audit.metadata(RUN + '/' + snapshot + '/gradients/' + component + '/manifest.json')
                 for component in COMPONENTS}
    names = list(manifests['token_ce']['parameters'])
    require(len(names) == len(set(names)) == 350 and
            Counter(partition(name) for name in names) == Counter(EXPECTED_PARTITION),
            'Exact164/114/72 trainable parameter census required')
    component_stats = {component: empty_stats() for component in COMPONENTS}
    branch_stats = {condition: empty_stats() for condition in CONDITIONS if condition != 'baseline'}
    physical, bytes_read, zero_arrays, composition_checks = 0, 0, 0, 0
    for component, manifest in manifests.items():
        require(list(manifest['parameters']) == names and manifest['component'] == component and
                manifest['weighted'] is True and manifest['source_exposures'] == manifest['aggregation_divisor'] == 16 and
                manifest['accumulation_dtype'] == 'float32' and
                manifest['accumulation_order'] == [case['id'] for case in cases] and
                manifest['None_is_not_zero'] is True and manifest['only_nonNone_physical_arrays_saved'] is True,
                'Aggregate gradient metadata differs')
    for name in names:
        parts = {}
        for component, manifest in manifests.items():
            record = manifest['parameters'][name]
            if record['is_none']:
                require(record.get('sha256') is None and 'file' not in record, 'None gradient has physical array')
                parts[component] = None
            else:
                value = audit.array(record['file'], dtype='float32')
                require(list(value.shape) == record['shape'] ==
                        branch_records['full']['step']['parameter_deltas']['parameters'][name]['shape'] and
                        record['sha256'] == record['file']['array_sha256'], 'Gradient vector row hash/shape differs')
                close(record['l2'], math.sqrt(float(np.square(value.astype(np.float64)).sum(dtype=np.float64))),
                      'Named gradient L2 differs')
                close(record['maximum_absolute'], float(np.abs(value).max()), 'Named gradient max differs')
                parts[component] = value
                physical += 1; bytes_read += value.nbytes
                zero_arrays += int(not np.any(value))
            add_stats(component_stats[component], name, parts[component], np)
            allowed = ({'fusion'} if component in IMAGE_COMPONENTS else
                       {'encoder'} if component == 'latent_mse' else {'encoder', 'predictor'})
            require(parts[component] is None if partition(name) not in allowed else parts[component] is not None,
                    'Exported aggregate gradient connectivity differs')
        for condition in branch_stats:
            omitted = condition.split(':', 1)[1] if condition.startswith('drop:') else None
            assigned = compose_one(parts, omitted, condition == 'momentum_only', np)
            expected = branch_records[condition]['step']['assigned_gradient_sha256_by_tensor'][name]
            require(hashlib.sha256(assigned.tobytes()).hexdigest() == expected,
                    'Actual assigned gradient hash differs from independent composition: ' + condition + '/' + name)
            add_stats(branch_stats[condition], name, assigned, np)
            composition_checks += 1
            del assigned
        del parts
    for component in COMPONENTS:
        actual = finish_stats(component_stats[component]); compare_stats(actual, manifests[component]['groups'])
        require(manifests[component]['groups'] == source_summary['components'][component]['groups'],
                'Source/manifest group summaries differ')
    for condition in branch_stats:
        compare_stats(finish_stats(branch_stats[condition]), branch_records[condition]['step']['assigned_gradient_groups'])
    require(physical == 1008, 'Exactly1008 physical component arrays required per snapshot')
    return {'parameter_names': names, 'physical_arrays': physical, 'FP32_element_bytes': bytes_read,
            'explicit_allzero_arrays': zero_arrays, 'branch_named_compositions_hash_exact': composition_checks,
            'component_groups': {key: finish_stats(value) for key, value in component_stats.items()},
            'independent_saved_vector_reductions_pass': True}


def check_delta(record):
    rows, groups = record['parameters'], record['groups']
    require(record['model_checkpoint_written'] is False and len(rows) == 515,
            'Exact515 parameter/buffer delta rows required')
    derived = {}
    for name, row in rows.items():
        require(all(HEX.fullmatch(row[key]) for key in ('base_sha256', 'after_sha256', 'delta_sha256')),
                'Invalid saved delta digest')
        maximum, norm = row['maximum_absolute_delta'], row['delta_l2']
        require(maximum >= 0 and norm >= 0 and math.isfinite(maximum) and math.isfinite(norm), 'Invalid saved delta norm')
        require((row['base_sha256'] != row['after_sha256']) == (maximum > 0), 'Changed tensor/hash flag differs')
        require((norm == 0) == (maximum == 0), 'Delta norm/max zero flags differ')
        group = derived.setdefault(name.split('.')[0], {'tensors': 0, 'changed_tensors': 0,
            'squared_l2': 0., 'maximum_absolute': 0.})
        group['tensors'] += 1; group['changed_tensors'] += int(maximum > 0)
        group['squared_l2'] += norm**2; group['maximum_absolute'] = max(group['maximum_absolute'], maximum)
    for key, row in derived.items():
        require(groups[key]['tensors'] == row['tensors'] and groups[key]['changed_tensors'] == row['changed_tensors'],
                'Delta group census differs')
        close(groups[key]['l2'], math.sqrt(row['squared_l2']), 'Delta group L2 identity differs')
        close(groups[key]['maximum_absolute'], row['maximum_absolute'], 'Delta group max differs')
    require(sum(groups[key]['tensors'] for key in ('quantize', 'generator')) == 165 and
            all(groups[key]['changed_tensors'] == 0 and groups[key]['l2'] == 0 for key in ('quantize', 'generator')),
            'Frozen165 tensor delta rows changed')
    return {'rows': len(rows), 'frozen_rows_unchanged': 165,
            'trainable_rows_changed': sum(row['changed_tensors'] for key, row in groups.items()
                                         if key not in ('quantize', 'generator')),
            'scope': 'Recorded digests/norm consistency; full new model/delta vectors were not exported'}


def output_bindings(audit, outputs, source=False):
    np = audit.np
    keys = {'native512', 'raw256', 'png256', 'tokens'} | ({'logits', 'features'} if source else set())
    require(keys <= outputs.keys(), 'Required source/probe output binding missing')
    shapes = {'native512': ((1, 3, 512, 512), 'float32'), 'raw256': ((256, 256, 3), 'float32'),
              'tokens': ((256,), 'int64'), 'logits': ((1, 256, 1024), 'float32'),
              'features': ((1, 256, 16, 16), 'float32')}
    for name in keys - {'png256'}:
        binding = outputs[name]
        value = audit.array(binding, *shapes[name], whole=False)
        if name == 'tokens':
            require(int(value.min()) >= 0 and int(value.max()) < 1024, 'Selected code out of range')
    png = outputs['png256']; audit.binding(png['path'])
    require(png['sha256'] == audit.hashes[png['path']] and png['bytes'] == audit.sizes[png['path']],
            'PNG manifest binding differs')
    require(outputs['native_domain'] == 'raw CodeFormer normalized domain; no clipping' and
            outputs['raw_policy'] == 'native(output+1)/2 clamp[0,1], bilinear512to256, align_cornersFalse' and
            outputs['PNG_policy'] == 'floor(clipped RAW*255) RGB8; no display enhancement' and
            all(value == 'torch.float32' for value in outputs['dtypes'].values()),
            'RAW/native/PNG domain declaration differs')


def response_records(branches, cases, np):
    """Readback of source-recorded metrics; independent pixel recomputation is separate."""
    result = []
    for view in ('raw_metrics', 'png_metrics'):
        for condition in CONDITIONS:
            for role, slots in (('A_normal', [1]), ('A_dark', [2]), ('B_normal_anchored', [3]), ('B_dark', [4])):
                indices = [i for i, case in enumerate(cases) if case['slot'] in slots]
                entries = [branches[condition]['probes'][i][view] for i in indices]
                baseline = [branches['baseline']['probes'][i][view] for i in indices]
                full = [branches['full']['probes'][i][view] for i in indices]
                mae = np.asarray([item['mae'] for item in entries], dtype=np.float64)
                signed_bias = np.asarray([np.asarray(item['signed_mean_RGB_error']) @ np.asarray([.2126,.7152,.0722])
                                          for item in entries])
                result.append({'condition': condition, 'view': view, 'stratum': role, 'cases': [cases[i]['id'] for i in indices],
                    'mean_recorded_MAE': float(mae.mean()), 'mean_MAE_delta_vs_same_state_baseline':
                    float((mae - np.asarray([item['mae'] for item in baseline])).mean()),
                    'mean_MAE_delta_vs_full': float((mae - np.asarray([item['mae'] for item in full])).mean()),
                    'worst_MAE_delta_vs_full': float((mae - np.asarray([item['mae'] for item in full])).max()),
                    'mean_signed_luminance_residual': float(signed_bias.mean()),
                    'mean_absolute_luminance_bias': float(np.mean([item['absolute_luminance_bias'] for item in entries]))})
    return result


def audit_run(imported, transport_receipt, np):
    audit = SavedAudit(imported, np)
    imported_receipt = read_json(transport_receipt)
    require(imported_receipt.get('complete') is True and
            Path(imported_receipt['import_root']).resolve(strict=True) == audit.root and
            imported_receipt['archive_sha256'] == ARCHIVE_SHA and imported_receipt['protocol_sha256'] == PROTOCOL_SHA,
            'Completed root hash-bound import receipt required before this independent audit')
    require(set(imported_receipt['files_sha256']) == set(audit.hashes) | {RUN + '/export_manifest.json'} and
            all(imported_receipt['files_sha256'][name] == digest and
                imported_receipt['files_bytes'][name] == audit.sizes[name] for name,digest in audit.hashes.items()) and
            imported_receipt['files_sha256'][RUN + '/export_manifest.json'] ==
            sha256(audit.root / RUN / 'export_manifest.json'), 'Import/export manifest identities disagree')
    protocol = audit.metadata('protocol.json'); require(sha256(audit.root / 'protocol.json') == PROTOCOL_SHA,
                                                       'Exact original protocol required')
    require(protocol['call_caps'] == CAPS and protocol['branches'] == list(CONDITIONS) and
            protocol['source_updates'] == [1,6,15,24] and protocol['loss_weights'] == WEIGHTS,
            'Frozen protocol mathematical controls differ')
    require(audit.manifest['protocol_sha256'] == PROTOCOL_SHA and audit.manifest['model_qualified'] is False,
            'Manifest identity/qualification differs')
    cases = audit.metadata('cases.json')['cases']; require(len(cases) == 16, 'Exact16 cases required')
    completion = audit.metadata(RUN + '/completion.json')
    require(completion['protocol_sha256'] == PROTOCOL_SHA and completion['complete'] is True and
            completion['counts'] == CAPS and completion['actual_completed_disposable_Adam_steps'] ==
            completion['actual_disposable_Adam_steps'] == 18 and completion['committed_training_updates'] ==
            completion['backward_calls'] == 0 and completion['model_checkpoint_written'] is False and
            completion['model_qualified'] is False and completion['resume_authorized'] is False and
            completion['branches_chain'] is False, 'Completed diagnostic evidence/counts differ')
    require(RUN + '/failure.json' not in audit.hashes, 'Failure coexists with claimed complete audit')
    ledger_names = sorted(name for name in audit.hashes if name.startswith(RUN + '/call_trace/'))
    require([Path(name).name for name in ledger_names] == ['%05d.json' % i for i in range(1,637)],
            'Call trace namespace differs')
    ledger = verify_ledger([audit.metadata(name) for name in ledger_names], cases)
    init = audit.metadata(RUN + '/initialization.json')
    require(init['pass'] is True and init['FP32_autocast_disabled'] is True and
            init['source_observations'] == 16 and init['source_photographic_TRAIN_ids'] == 8 and
            init['source_image_eligible'] == 12 and init['schedule_extension'] is False,
            'Initialization source/runtime declaration differs')
    require(init['algorithm_settings'] == completion['algorithm_settings'] and
            completion['final_frozen'] == init['frozen'], 'Terminal frozen/algorithm readback differs')
    algo = init['algorithm_settings']
    require(algo['matmul_allow_tf32'] is False and algo['cudnn_allow_tf32'] is False and
            algo['cudnn_benchmark'] is False, 'Required FP32 backend settings differ')
    require(init['fixed_diagnostic_rates'] == {'encoder_token':1e-5,'fusion':2e-5}, 'Fixed measurement rates differ')
    require(all(item['maximum_absolute_error'] <= item['bound'] == 1e-6 for item in init['initializer_parity']),
            'Official initializer parity bound failed')
    snapshots, responses, global_steps, fullsum_count = [], {}, 0, 0
    for snapshot, counter in (('original0',0),('stopped50',50)):
        source = audit.metadata(RUN + '/' + snapshot + '/source_summary.json')
        end = audit.metadata(RUN + '/' + snapshot + '/snapshot_complete.json')
        require(source['snapshot'] == snapshot and source['fixed_component_order'] == list(COMPONENTS) and
                source['full_query_vectors_not_exported'] is True and source['source_extraction_committed_updates'] == 0,
                'Source snapshot contract differs')
        check_sum_capture(source['aggregate_vector_agreement']); fullsum_count += 1
        require([item['case'] for item in source['sources']] == [case['id'] for case in cases],
                'Source summary case order differs')
        for index, case in enumerate(cases):
            record = audit.metadata(RUN + '/' + snapshot + '/sources/' + case['id'] + '/record.json')
            require(record['snapshot'] == snapshot and record['case'] == case['id'] and
                    record['update'] == case['update'] and record['slot'] == case['slot'] and
                    record['reference_id'] == case['reference_id'] and record['profile'] == case['profile'] and
                    record['image_eligible'] == case['parameters']['image_eligible'] and
                    set(record['terms']) == set(expected_terms(case)) and record['aggregation_divisor'] == 16 and
                    record['parameter_grad_remains_None'] is True, 'Source observation/term routing differs')
            for component, value in record['terms'].items():
                close(record['weighted_terms_before_division16'][component], WEIGHTS[component]*value,
                      'Weighted objective scalar differs')
            check_sum_capture(record['full_component_vector_agreement']); fullsum_count += 1
            require(source['sources'][index]['full_component_vector_agreement'] == record['full_component_vector_agreement'],
                    'Source full-sum record identity differs')
            route = record['route']; require(route['original_callable_restored'] is True and len(route['calls']) == 1,
                                            'Source callable restoration capture differs')
            call = route['calls'][0]
            require(call['style_requires_grad_before'] is True and call['style_requires_grad_after'] is False and
                    call['content_requires_grad'] is False and call['detached_shares_storage'] is True and
                    call['shape_dtype_device_unchanged'] is True, 'Source isolation route capture differs')
            output_bindings(audit, record['outputs'], source=True)
        branches, deltas = {}, []
        for condition in CONDITIONS:
            directory = RUN + '/' + snapshot + '/conditions/' + condition.replace(':','__')
            branch = audit.metadata(directory + '/branch.json'); restoration = audit.metadata(directory + '/restoration.json')
            branches[condition] = branch
            require(branch['snapshot'] == restoration['snapshot'] == snapshot and
                    branch['condition'] == restoration['condition'] == condition and
                    branch['base_model_sha256'] == source['base_model_sha256'] == restoration['model_sha256'] and
                    branch['base_Adam_sha256'] == source['base_Adam_sha256'] == restoration['Adam_sha256'] and
                    branch['base_RNG_sha256'] == source['base_RNG_sha256'] == restoration['RNG_sha256'] and
                    branch['restored_independently_before_branch'] is True and branch['branch_chain'] is False and
                    branch['model_checkpoint_written'] is False and restoration['exact_base_model_Adam_RNG_restored'] is True and
                    restoration['Adam_counter'] == counter and restoration['optimizer_inflight'] is False and
                    branch['committed_training_updates'] == restoration['committed_training_updates'] == 0,
                    'Branch exact base/restoration/source identities differ')
            if condition == 'baseline':
                require(branch['step'] is None and branch['branch_model_sha256'] == source['base_model_sha256'],
                        'Baseline performed a parameter step')
            else:
                global_steps += 1; step = branch['step']
                require(step['actual_Adam_step'] is True and step['step_before'] == counter and
                        step['step_after'] == counter+1 and step['trainable_tensors_with_explicit_gradients'] == 350 and
                        step['fixed_rates'] == {'encoder_token':1e-5,'fusion':2e-5} and step['schedule_advanced'] is False and
                        step['momentum_only_explicit_zero'] == (condition == 'momentum_only') and
                        step['omitted_component'] == (condition.split(':',1)[1] if condition.startswith('drop:') else None),
                        'Real disposable Adam step/counter/rate contract differs')
                deltas.append({'condition':condition, **check_delta(step['parameter_deltas'])})
            require(branch['actual_disposable_step_count_so_far'] == restoration['actual_completed_disposable_Adam_steps'] == global_steps,
                    'Branch completed-step ledger differs')
            require([item['case'] for item in branch['probes']] == [case['id'] for case in cases], 'Probe order differs')
            for index, case in enumerate(cases):
                record = audit.metadata(directory+'/'+case['id']+'/record.json')
                require(record['snapshot'] == snapshot and record['condition'] == condition and record['case'] == case['id'] and
                        record['reference_id'] == case['reference_id'] and record['profile'] == case['profile'],
                        'Probe identity differs')
                require(record['route']['original_callable_restored'] is True and len(record['route']['calls']) == 1,
                        'Probe callable restoration capture differs')
                output_bindings(audit, record['outputs'])
                require(branch['probes'][index]['raw_metrics'] == record['outputs']['raw_metrics'] and
                        branch['probes'][index]['png_metrics'] == record['outputs']['png_metrics'], 'Probe/branch metric record differs')
        gradient = audit_gradients(audit,snapshot,cases,source,branches)
        require(end['base_counter'] == counter and end['actual_disposable_Adam_steps_this_snapshot'] == 9 and
                end['final_exact_model_Adam_RNG_restoration'] is True and end['committed_training_updates'] == 0 and
                end['model_checkpoint_written'] is False and
                end['final_base_model_sha256'] == source['base_model_sha256'] and
                end['final_base_Adam_sha256'] == source['base_Adam_sha256'] and
                end['final_base_RNG_sha256'] == source['base_RNG_sha256'] and
                len(end['final_repeat']) == 16 and all(item['native_RAW_tokens_bit_exact'] is True for item in end['final_repeat']),
                'Snapshot final restoration/repeat contract differs')
        require(end == completion['snapshots'][0 if counter==0 else 1], 'Completion snapshot identity differs')
        snapshots.append({'snapshot':snapshot,'base_counter':counter,'source_summary':source,
            'delta_readback':deltas,'gradient_readback':gradient,'snapshot_restore_pass':True,
            'scope':'model/Adam/RNG/freeze digests are source-bound runtime captures, not new live state inspection'})
        responses[snapshot]=response_records(branches,cases,np)
    require(global_steps == 18 and fullsum_count == 34, 'Final audit state/full-sum census differs')
    return {'format':'codeformer-loss-response-independent-return-state-v1','complete':True,
        'status':'FINITE_DIAGNOSTIC_STATE_GRADIENT_LEDGER_AUDIT_PASS_NO_MODEL_QUALIFICATION',
        'UTC':datetime.now(timezone.utc).isoformat(),'protocol_sha256':PROTOCOL_SHA,
        'imported_root':str(audit.root),'transport_receipt_sha256':sha256(transport_receipt),
        'archive_sha256':ARCHIVE_SHA,'completion':completion,'ledger':ledger,'snapshots':snapshots,
        'full_sum_captures_source_bound':fullsum_count,'assigned_gradient_composition_checks':6300,
        'metadata_files_freshly_verified':audit.read_metadata,'array_header_or_vector_reads':audit.read_arrays,
        'response_table_source_recorded_not_independent_pixels':responses,
        'neural_calls':0,'autograd_queries':0,'optimizer_steps':0,'VM_calls':0,'returned_Python_executed':False,
        'model_or_new_delta_arrays_exported':False,'model_qualified':False,'quality_thresholds_changed':False,
        'limitations':['Model/Adam/RNG/modes/hooks/freeze restorations are source-bound runtime captures; full runtime states were not returned.',
            'Saved aggregate gradient composition/reductions are independently checked; autograd and direct-full vectors are not replayed.',
            'CUDA parameter delta digests/norms cannot establish per-element tolerance against a different CPU implementation.',
            'Output metrics here are metadata readbacks; independent RAW/PNG recomputation and complete visual review remain separate.',
            'Guard success/admission projections do not supply measured RAM/CUDA peak numbers absent explicit captures.',
            'Same16 TRAIN source/probe observations do not evaluate fullDEV61/53 gates, native usefulness, FINAL or app acceptance.']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imported-root',type=Path,default=ROOT/'outputs/codeformer_loss_response_v1_return')
    parser.add_argument('--transport-receipt',type=Path,
        default=ROOT/'outputs/codeformer_loss_response_v1_return_review/return_import_v1.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/codeformer_loss_response_v1_return_review/state_v1/state_audit.json')
    args=parser.parse_args(argv)
    require(not args.output.exists(),'Retain previous audit evidence; use a new attempt output')
    started=time.monotonic()
    try:
        import numpy as np  # Saved array arithmetic only; no neural framework.
        result=audit_run(args.imported_root,args.transport_receipt,np)
        result['audit_seconds']=time.monotonic()-started
        result['source_sha256']=sha256(__file__)
        result['numpy_version']=np.__version__
        write_new(args.output,result)
        print(json.dumps({'complete':True,'output':str(args.output),'events':636,
            'gradient_compositions':6300,'actual_diagnostic_Adam_steps':18,'local_neural_or_optimizer_calls':0}),flush=True)
    except BaseException as error:
        failure=args.output.with_name(args.output.stem+'_failure.json')
        if not failure.exists():
            write_new(failure,{'complete':False,'error_type':type(error).__name__,'error':str(error),
                'traceback':traceback.format_exc(),'seconds':time.monotonic()-started,
                'source_sha256':sha256(__file__),'local_neural_or_optimizer_calls':0,'evidence_retained':True})
        raise


if __name__=='__main__':
    main()
