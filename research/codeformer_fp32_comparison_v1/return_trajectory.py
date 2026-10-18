"""Independent saved-evidence FP32 audit. No model instance/forward/gradient/VM.

The frozen recipe is never edited or resumed. Checkpoints are loaded on CPU
with weights_only=True; scalar losses/teacher outputs are not re-executed.
"""
import argparse
from collections import Counter, defaultdict
import gc
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import traceback


PROTOCOL_SHA256 = '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'
ARCHIVE_SHA256 = '8b2361147a1d12b7d2bad8a7d3a8965e0e90c077bb1cdbb4ad7378ad8703ef78'
MANIFEST_SHA256 = '4562d4961c7862828d2164b96dd6df675f7b09171298109aba29cce44fb8395b'
MANIFEST_MEMBER = 'outputs/fp32_comparison_v1/export_manifest_20261016T132557Z-e00becd6.json'
TRAINABLE = ('encoder', 'position_emb', 'feat_emb', 'ft_layers', 'idx_pred_layer', 'fuse_convs_dict')
FROZEN = ('quantize', 'generator')


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_file(root, relative):
    require(isinstance(relative, str) and relative, 'Missing evidence path')
    relative_path = PurePosixPath(relative)
    require(not relative_path.is_absolute() and relative_path.as_posix() == relative and
            '..' not in relative_path.parts and ':' not in relative and '\\' not in relative,
            'Evidence path is not canonical and relative')
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*relative_path.parts)
    require(path.resolve(strict=True).is_relative_to(root), 'Evidence path escapes imported root')
    for item in [path, *path.parents]:
        if item == root.parent:
            break
        info = item.lstat()
        require(not item.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400,
                'Evidence symlink/reparse point forbidden')
    require(path.is_file(), 'Missing regular evidence file: ' + relative)
    return path


def read_json(path):
    def fields(rows):
        result = {}
        for name, value in rows:
            require(name not in result, 'Duplicate JSON field: ' + name)
            result[name] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=fields,
                      parse_constant=lambda value: (_ for _ in ()).throw(AuditError('Nonfinite JSON constant')))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(actual, expected, *, tolerance=1e-12):
    require(finite(actual) and finite(expected) and
            abs(actual - expected) <= tolerance * max(1.0, abs(expected)),
            'Saved scalar differs from independent arithmetic')


def lr_at(update, base, training):
    if update == 0:
        return base
    warmup = training['warmup_updates']
    if update <= warmup:
        return base * update / warmup
    phase = (update - warmup) / (training['schedule_horizon_updates'] - warmup)
    return training['cosine_end_lr'] + (base - training['cosine_end_lr']) * (1 + math.cos(math.pi * phase)) / 2


def read_history(path, failure=None):
    """Retain truncated tails and recover only explicitly committed pending rows."""
    rows, warnings = [], []
    if Path(path).exists():
        lines = Path(path).read_bytes().splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except (ValueError, UnicodeError):
                require(index == len(lines) - 1, 'Malformed non-final trajectory line')
                warnings.append({'kind': 'truncated_history_tail', 'sha256': hashlib.sha256(line).hexdigest()})
    pending = (failure or {}).get('pending_step_record')
    if pending and pending.get('optimizer_committed') is True:
        if rows and pending['update'] == rows[-1]['update']:
            require(pending == rows[-1], 'Pending committed record disagrees with saved history')
        else:
            require(pending['update'] == len(rows) + 1, 'Pending record cannot fill a non-prefix history gap')
            rows.append(pending)
            warnings.append({'kind': 'committed_row_recovered_from_failure', 'update': pending['update']})
    require([row['update'] for row in rows] == list(range(1, len(rows) + 1)),
            'Trajectory is not a contiguous committed update prefix')
    return rows, warnings


def check_loss(sample, weights):
    expected = {'token_ce', 'latent_mse', 'coarse_rgb_l1'}
    if sample['image_eligible']:
        expected |= {'pixel_l1_rgb', 'lpips_vgg'}
    if sample['slot'] == 3:
        expected |= {'behavior_rgb_l1', 'behavior_token_kl'}
    require(set(sample['terms']) == set(sample['weighted_terms']) == expected,
            'Loss namespace/eligibility differs')
    for name, value in sample['terms'].items():
        require(finite(value), 'Nonfinite saved loss')
        close(sample['weighted_terms'][name], value * weights[name])
    total = sum(sample['weighted_terms'].values())
    # Native total was accumulated in FP32; the separately logged terms/weights
    # are Python numbers. This is numerical reduction tolerance, no quality gate.
    tolerance = 32 * 2**-23 * max(1.0, sum(abs(v) for v in sample['weighted_terms'].values()))
    require(finite(sample['total_before_accumulation_division']) and
            abs(sample['total_before_accumulation_division'] - total) <= tolerance,
            'Reported objective differs from weighted term sum')
    return total / 4


def audit_history(rows, references, protocol):
    from research.codeformer_adaptation_v2 import data
    train = {row['id']: row for row in references if row['historical_role'] == 'train'}
    require(len(train) == 391, 'Only391 approved TRAIN references are expected')
    require(len(rows) <= protocol['training']['max_updates'], 'Trajectory exceeds finite update cap')
    sampler = data.TrainSampler(sorted(train), seed=protocol['seed'])
    states = {0: sampler.state_dict()}
    exposures, profiles, expected_records = Counter(), Counter(), {}
    eligible = 0
    previous_seconds = 0.0
    for row in rows:
        update = row['update']
        selection = sampler.preview_update()
        require(row['TRAIN_selection'] == selection.to_record(), 'TRAIN sampler replay differs')
        require(row.get('optimizer_committed') is True, 'History row is not a committed optimizer update')
        require(len(row['microbatches']) == 4, 'Committed update needs four slots')
        role_pairs = {}
        for slot, sample in enumerate(row['microbatches'], start=1):
            identifier = selection.id_a if slot <= 2 else selection.id_b
            pair_role = 'A' if slot <= 2 else 'B'
            member = 'normal' if slot % 2 else 'dark'
            require(sample['slot'] == slot and sample['id'] == identifier, 'TRAIN slot/reference order differs')
            record = sample['pair_record']
            parameters = record['parameters']
            expected = data.camera_parameters(identifier, pair_role, update)
            require(all(parameters.get(key) == value for key, value in expected.items()),
                    'Prospectively keyed camera parameters differ')
            ref = train[identifier]
            require(record['reference_id'] == identifier and record['role'] == 'train' and
                    record['source_path'] == ref['source_path'] and record['source_sha256'] == ref['source_sha256'] and
                    record['reference_manifest_sha256'] == protocol['files_sha256'][protocol['data_manifest']] and
                    record['clean512_rgb8_sha256'] == ref['proposed_512_rgb_sha256'] and
                    record['update'] == update and record['slot'] == slot and record['pair_role'] == pair_role and
                    record['member'] == member and parameters['member'] == member and
                    parameters['slot'] == slot and parameters['pair_role'] == pair_role and
                    parameters['behavior_anchor'] == (slot == 3), 'TRAIN provenance/pair linkage differs')
            require(record['profile'] == sample['profile'] == expected['family'] + '_' + member and
                    sample['image_eligible'] == parameters['image_eligible'] == expected['image_eligible'],
                    'Camera family or image-loss eligibility differs')
            close(parameters['exposure_linear'], 1.0 if member == 'normal' else expected['dark_exposure_linear'])
            role_pairs.setdefault(pair_role, []).append(record)
            check_loss(sample, protocol['training']['loss_weights'])
            require((sample.get('behavior_anchor') is not None) == (slot == 3), 'Behavior anchor slot differs')
            expected_records[(update, slot)] = sample
            exposures[identifier] += 1; profiles[record['profile']] += 1
            eligible += int(sample['image_eligible'])
        for pair in role_pairs.values():
            require(pair[0]['pair_id'] == pair[1]['pair_id'] and
                    pair[0]['parameters']['noise_fields'] == pair[1]['parameters']['noise_fields'] and
                    pair[0]['clean512_float32_sha256'] == pair[1]['clean512_float32_sha256'],
                    'Normal/dark pair does not share fields/target')
        for name, base_key in [('encoder_token', 'encoder_token_lr'), ('fusion', 'fusion_lr')]:
            close(row['group_lrs'][name], lr_at(update, protocol['training'][base_key], protocol['training']))
        seconds = row['optimization_seconds']
        require(finite(seconds) and seconds >= previous_seconds, 'Optimization timing is nonfinite/nonmonotone')
        previous_seconds = seconds
        sampler.commit(selection); states[update] = sampler.state_dict()
    return {'committed_updates': len(rows), 'committed_exposures': 4 * len(rows),
            'unique_TRAIN_references': len(exposures), 'reference_exposure_counts': dict(exposures),
            'profile_counts': dict(profiles), 'detail_LPIPS_exposures': eligible,
            'normal_exposures': 2 * len(rows), 'dark_exposures': 2 * len(rows),
            'optimization_seconds': previous_seconds, 'sampler_replayed_exactly': True,
            'loss_weighting_and_logged_total_consistent': True,
            'backward_division_by_four_is_source_bound_not_independently_observed': True}, states, expected_records


def tensor_state_digest(state):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tree_digest(value):
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(b'T' + tensor_state_digest({'value': item}).encode())
        elif isinstance(item, dict):
            digest.update(b'D')
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(('L' if isinstance(item, list) else 'U').encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps([type(item).__name__, item], allow_nan=False, separators=(',', ':')).encode())
    visit(value)
    return digest.hexdigest()


def check_rng(rng):
    import torch
    require(set(rng) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'}, 'RNG schema differs')
    require(len(rng['python']) == 3 and rng['python'][0] == 3 and len(rng['python'][1]) == 625,
            'Python committed RNG structure differs')
    numpy = rng['numpy']
    require(numpy['algorithm'] == 'MT19937' and len(numpy['keys']) == 624 and
            0 <= numpy['position'] <= 624, 'NumPy committed RNG structure differs')
    require(isinstance(rng['torch_cpu'], torch.Tensor) and rng['torch_cpu'].dtype == torch.uint8 and
            rng['torch_cpu'].ndim == 1 and rng['torch_cpu'].numel() > 0, 'CPU RNG tensor differs')
    require(len(rng['torch_cuda']) == 1 and isinstance(rng['torch_cuda'][0], torch.Tensor) and
            rng['torch_cuda'][0].dtype == torch.uint8 and rng['torch_cuda'][0].ndim == 1 and
            rng['torch_cuda'][0].numel() > 0, 'Single-L4 committed RNG tensor differs')


def check_checkpoint(state, original, protocol, expected_sampler, expected_update):
    """CPU tensor/value inspection; constructs no module or optimizer."""
    import torch
    require(state['format'] == 'codeformer-fp32-comparison-state-v1' and
            state['protocol_sha256'] == PROTOCOL_SHA256 and state['completed_updates'] == expected_update and
            state['schedule'] == {'completed_updates': expected_update, 'protocol_training': protocol['training']} and
            state['EMA'] is False and state['no_automatic_resume_supported'] is True,
            'Checkpoint protocol/update/schedule metadata differs')
    model = state['model']
    require(set(model) == set(original), 'Checkpoint model key set differs from initializer')
    group_changes = defaultdict(lambda: {'tensors': 0, 'changed_tensors': 0, 'maximum_change': 0.0, 'change_l2_squared': 0.0})
    for name, tensor in model.items():
        before = original[name]
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == before.dtype and tensor.shape == before.shape and
                torch.isfinite(tensor).all().item(), 'Checkpoint tensor geometry/dtype/finiteness differs: ' + name)
        group = name.split('.')[0]
        require(group in TRAINABLE + FROZEN, 'Undeclared checkpoint tensor group')
        same = torch.equal(tensor, before)
        require(group not in FROZEN or same, 'Frozen codebook/generator tensor changed: ' + name)
        require(expected_update != 0 or same, 'Update0 differs from untouched initializer')
        row = group_changes[group]; row['tensors'] += 1; row['changed_tensors'] += int(not same)
        if not same:
            delta = tensor.double() - before.double()
            row['maximum_change'] = max(row['maximum_change'], float(delta.abs().max()))
            row['change_l2_squared'] += float(delta.square().sum())
    require(state['sampler'] == expected_sampler, 'Checkpoint sampler is not the committed replay state')
    check_rng(state['rng'])
    optimizer = state['optimizer']
    groups = optimizer['param_groups']
    require([row['name'] for row in groups] == ['encoder_token', 'fusion'], 'Adam group names/order differs')
    used, steps = [], Counter()
    safe = state['resume_safe'] is True
    require(type(state['resume_safe']) is bool and 0 <= state['partial_microbatches_discarded'] <= 4,
            'Checkpoint safety/partial-gradient metadata differs')
    counts = state['counts']
    for name, cap in protocol['training']['call_caps'].items():
        require(type(counts[name]) is int and 0 <= counts[name] <= cap, 'Checkpoint finite counter differs: ' + name)
    require(counts['optimizer_updates'] == expected_update and
            4 * expected_update <= counts['backward_calls'] <= min(200, 4 * expected_update + 4),
            'Checkpoint committed/partial counters differ')
    if safe:
        require(counts['optimizer_step_attempts'] == expected_update,
                'Safe checkpoint optimizer-attempt count differs')
    for group in groups:
        fusion = group['name'] == 'fusion'
        names = [name for name in original if name.split('.')[0] in TRAINABLE and
                 name.startswith('fuse_convs_dict.') == fusion]
        identifiers = group['params']; used.extend(identifiers)
        require(len(names) == len(identifiers), 'Adam parameter group size differs')
        base = protocol['training']['fusion_lr' if fusion else 'encoder_token_lr']
        close(group['base_lr'], base); close(group['eps'], protocol['training']['eps'])
        require(list(group['betas']) == protocol['training']['betas'] and group['weight_decay'] == 0,
                'Adam hyperparameters differ')
        if safe:
            close(group['lr'], lr_at(expected_update, base, protocol['training']))
        for identifier, name in zip(identifiers, names):
            value = optimizer['state'].get(identifier)
            if expected_update == 0 and value is None:
                continue
            if expected_update == 0:
                require(not safe, 'Safe initial checkpoint already has Adam moments')
            require(value is not None and set(value) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Adam moment schema differs')
            step = float(value['step'])
            require(math.isfinite(step) and step.is_integer() and 0 <= step <= expected_update + int(not safe),
                    'Adam step is outside committed/unsafe update bound')
            if safe:
                require(step == expected_update, 'Safe checkpoint Adam step differs from completed updates')
            steps[int(step)] += 1
            for key in ['exp_avg', 'exp_avg_sq']:
                tensor = value[key]
                require(tensor.shape == original[name].shape and tensor.dtype == original[name].dtype and
                        torch.isfinite(tensor).all().item(), 'Adam moment shape/dtype/finiteness differs')
            require((value['exp_avg_sq'] >= 0).all().item(), 'Adam second moment is negative')
    require(len(used) == len(set(used)) and set(optimizer['state']) <= set(used), 'Adam includes duplicate/frozen IDs')
    for row in group_changes.values():
        row['change_l2'] = math.sqrt(row.pop('change_l2_squared'))
    return {'completed_updates': expected_update, 'reported_resume_safe': safe,
            'resume_authorized': False, 'partial_microbatches_discarded': state['partial_microbatches_discarded'],
            'model_state_sha256': tensor_state_digest(model),
            'frozen_state_sha256': tensor_state_digest({name: value for name, value in model.items() if name.split('.')[0] in FROZEN}),
            'model_Adam_RNG_sampler_schedule_sha256': tree_digest({key: state[key] for key in ['model', 'optimizer', 'rng', 'sampler', 'schedule']}),
            'module_changes': dict(group_changes), 'Adam_parameter_states': len(optimizer['state']),
            'Adam_step_histogram': dict(steps), 'RNG_schema_valid_not_replayed_through_neural_calls': True}


def load_array(path, shape, dtype):
    import numpy as np
    path = Path(path)
    require(path.stat().st_size <= math.prod(shape) * np.dtype(dtype).itemsize + 4096,
            'NPY evidence exceeds planned geometry')
    with path.open('rb') as stream:
        value = np.load(stream, allow_pickle=False)
        require(not stream.read(1), 'Trailing bytes in NPY evidence')
    require(value.shape == shape and value.dtype == np.dtype(dtype) and np.isfinite(value).all(),
            'NPY evidence shape/dtype/finiteness differs')
    return value


def array_digest(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def verified_transport(imported, receipt_path):
    imported = Path(imported).resolve(strict=True)
    receipt = read_json(receipt_path)
    require(receipt.get('complete') is True and receipt.get('archive_sha256') == ARCHIVE_SHA256 and
            receipt.get('protocol_sha256') == PROTOCOL_SHA256 and receipt.get('manifest_member') == MANIFEST_MEMBER and
            receipt.get('manifest_sha256') == MANIFEST_SHA256 and receipt.get('all_members_hash_verified') is True and
            receipt.get('all_members_verified_before_destination_created') is True and receipt.get('exact_namespace') is True and
            receipt.get('members') == 6501 and Path(receipt.get('import_directory', '')).resolve(strict=True) == imported,
            'Closed exact transport verification must bind this imported directory')
    require(sha256(safe_file(imported, MANIFEST_MEMBER)) == MANIFEST_SHA256, 'Imported export manifest changed after strict import')
    return {'receipt_sha256': sha256(receipt_path), 'archive_sha256': ARCHIVE_SHA256,
            'manifest_sha256': MANIFEST_SHA256, 'bound_import_directory': str(imported), 'members': 6501}


def dtype_evidence(value):
    require(value.get('logits_dtype') == value.get('probabilities_dtype') ==
            value.get('native_output_dtype') == 'torch.float32' and
            value.get('autocast_enabled') is False and
            value.get('actual_native_softmax_and_selected_indices_captured') is True,
            'Candidate path was not recorded as FP32 without autocast')


def expected_call_events(protocol):
    """Independent fixed order; no worker helpers or neural execution."""
    counts = {key: 0 for key in ('optimizer_updates', 'microbatches', 'gradient_queries',
        'backward_calls', 'baseline_forwards', 'initializer_parity_candidate_forwards',
        'snapshot_forwards', 'teacher_forwards', 'candidate_training_forwards',
        'perceptual_prerequisite_forwards', 'perceptual_training_forwards',
        'behavior_teacher_forwards', 'optimizer_step_attempts', 'training_input_writes')}
    rows, committed, pending = [], 0, 0
    def reserve(kind):
        counts[kind] += 1
        rows.append({'event': len(rows) + 1, 'before_call': kind, 'committed_update': committed,
                     'pending_microbatches': pending, 'counts': dict(counts)})
    reserve('perceptual_prerequisite_forwards'); reserve('gradient_queries')
    for _ in range(2):
        reserve('baseline_forwards'); reserve('initializer_parity_candidate_forwards')
    for _ in range(391): reserve('teacher_forwards')
    for _ in range(406): reserve('snapshot_forwards')
    for update in range(1, protocol['training']['max_updates'] + 1):
        pending = 0
        for slot in range(1, 5):
            reserve('training_input_writes')
            if slot == 3: reserve('behavior_teacher_forwards')
            if slot >= 3 or (update - 1) % 4 >= 2: reserve('perceptual_training_forwards')
            reserve('candidate_training_forwards'); reserve('backward_calls')
            counts['microbatches'] += 1; pending += 1
        reserve('optimizer_step_attempts')
        committed = update; counts['optimizer_updates'] = update; pending = 0
        if update in protocol['training']['snapshots']:
            for _ in range(406): reserve('snapshot_forwards')
    return rows, counts


def audit_calls(imported, out, protocol, terminal):
    expected, terminal_counts = expected_call_events(protocol)
    files = sorted((out / 'call_trace').glob('*.json'))
    require([path.name for path in files] == [f'{i:05d}.json' for i in range(1, len(files) + 1)],
            'Call trace must be an exact chronological prefix')
    require(len(files) <= len(expected), 'Excess planned calls')
    elapsed, final = 0., None
    for path, control in zip(files, expected):
        row = read_json(safe_file(imported, path.relative_to(imported).as_posix()))
        require({key: row[key] for key in control} == control, 'Call order/count/commit reservation differs')
        require(finite(row['seconds']) and elapsed <= row['seconds'] <= protocol['limits']['worker_seconds'],
                'Call reservation clock is nonmonotone/nonfinite/over cap')
        elapsed = row['seconds']; final = row
    if terminal['completed_updates'] == 50:
        require(len(files) == len(expected) == 2057 and terminal['counts'] == terminal_counts,
                'Full-fifty terminal counts do not match exact prospective census')
    else:
        require(final is not None and all(0 <= terminal['counts'][key] <= terminal_counts[key] for key in terminal_counts),
                'Partial terminal calls exceed finite prefix')
    if 'last_call_reservation' in terminal:
        require(terminal['last_call_reservation'] == final, 'Failure last-call reservation differs from trace')
    return {'exact_call_reservations_verified': len(files), 'full_planned_reservations': len(expected),
        'terminal_counts': terminal['counts'], 'last_reserved_seconds': elapsed,
        'recorded_reservations_not_independent_runtime_instrumentation': True}


def audit_inputs_anchors(imported, packet, out, parity, expected_records, completed, frozen):
    import numpy as np
    from PIL import Image
    cases = {(row['update'], row['slot']): row for row in parity['cases']}
    anchors = {row['update']: row for row in parity['behavior_anchors']}
    input_records, online_rows, exact_rgb, exact_logits = 0, [], 0, 0
    actual_records = set((out / 'training_inputs').glob('update_*/slot_*/pair_record.json'))
    required_records = {out / 'training_inputs' / f'update_{u:04d}' / f'slot_{s}' / 'pair_record.json'
                        for u, s in expected_records}
    require(required_records <= actual_records, 'Committed TRAIN input record missing')
    for path in sorted(actual_records):
        record = read_json(path); key = (record['update'], record['slot'])
        require(key in cases and key[0] <= min(50, completed + 1), 'TRAIN input outside retained trajectory prefix')
        control = cases[key]
        require(record['role'] == 'train' and all(record[name] == control[name] for name in
            ('reference_id', 'source_sha256', 'profile', 'input256_rgb8_sha256', 'input256_float32_sha256',
             'clean512_rgb8_sha256', 'clean512_float32_sha256')), 'Saved actual TRAIN bindings differ from retained V2')
        image_path = safe_file(imported, record['saved_input_path'])
        original_path = safe_file(packet, control['input_path'])
        require(image_path.parent == path.parent and sha256(image_path) == record['saved_input_sha256'] and
                sha256(original_path) == control['input_sha256'], 'Actual/retained TRAIN PNG file binding differs')
        with Image.open(image_path) as image:
            require(image.mode == 'RGB' and image.size == (256, 256), 'Actual TRAIN PNG geometry differs')
            pixels = np.asarray(image).copy()
        with Image.open(original_path) as image: original_pixels = np.asarray(image).copy()
        floats = np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255))
        require(np.array_equal(pixels, original_pixels) and array_digest(pixels) == control['input256_rgb8_sha256'] and
                array_digest(floats) == control['input256_float32_sha256'], 'Actual TRAIN pixel/tensor parity failed')
        receipt = read_json(path.parent / 'parity.json')
        require(receipt['complete'] is True and receipt['update'] == key[0] and receipt['slot'] == key[1] and
                receipt['retained_input_sha256'] == control['input_sha256'] and receipt['exact_input_float32'] is True and
                receipt['exact_clean512_float32'] is True and receipt['before_candidate_training_forward'] is True,
                'Runtime input/clean parity receipt differs')
        if key in expected_records:
            sample = expected_records[key]
            require(sample['pair_record'] == record and sample['input_and_clean_parity'] == receipt,
                    'History/input/parity receipt mismatch')
        dtype = read_json(path.parent / 'dtypes.json'); dtype_evidence(dtype)
        if key in expected_records: require(expected_records[key]['candidate_dtypes'] == dtype, 'History dtype receipt differs')
        tokens = load_array(path.parent / 'actual_tokens.npy', (256,), np.int64)
        require(tokens.min() >= 0 and tokens.max() < 1024, 'Actual TRAIN selected token outside codebook')
        input_records += 1
    for update in range(1, completed + 1):
        control = anchors[update]
        used_dir = out / 'behavior_anchors' / f'update_{update:04d}'
        online_dir = out / 'online_behavior_anchors' / f'update_{update:04d}'
        used_record, online_record = read_json(used_dir / 'record.json'), read_json(online_dir / 'record.json')
        expected_input = cases[(update, 3)]
        require(used_record['update'] == update and used_record['slot'] == 3 and
                used_record['reference_id'] == online_record['reference_id'] == control['reference_id'] and
                used_record['input256_float32_sha256'] == online_record['input256_float32_sha256'] ==
                expected_input['input256_float32_sha256'] and
                used_record['teacher_state_sha256'] == online_record['teacher_state_sha256'] ==
                frozen['behavior_teacher'] == control['teacher_state_sha256'] and
                used_record['teacher_FP32_eval_frozen'] is online_record['teacher_FP32_eval_frozen'] is True and
                used_record['retained_delivered_sha256'] == control['delivered_sha256'] and
                used_record['retained_logits_sha256'] == control['logits_sha256'] and
                online_record['not_used_as_behavior_loss_targets'] is True,
                'Used/online behavior input/state/target identity differs')
        sample = expected_records[(update, 3)]
        require(sample['behavior_anchor'] == used_record and used_record['pair_record'] == sample['pair_record'],
                'Behavior history/actual target receipt differs')
        comparison = read_json(used_dir / 'parity.json')
        require(comparison['complete'] is True and comparison['update'] == update and
                comparison['input256_float32_sha256'] == expected_input['input256_float32_sha256'] and
                comparison['online_numerical_differences_are_reported_not_a_new_quality_gate'] is True,
                'Behavior numerical comparison receipt differs')
        recomputed = {'update': update, 'comparisons': {}}
        for name, shape in [('delivered', (1, 3, 256, 256)), ('logits', (1, 256, 1024))]:
            retained_path = safe_file(packet, control[name + '_path'])
            require(sha256(retained_path) == control[name + '_sha256'], 'Retained target file changed')
            retained = load_array(retained_path, shape, np.float32)
            used_path = safe_file(imported, used_record[name + '_path'])
            online_path = safe_file(imported, online_record[name + '_path'])
            require(used_path.parent == used_dir and online_path.parent == online_dir and
                    sha256(used_path) == used_record[name + '_sha256'] and
                    sha256(online_path) == online_record[name + '_sha256'], 'Used/online array file binding differs')
            used = load_array(used_path, shape, np.float32); online = load_array(online_path, shape, np.float32)
            require(np.array_equal(used, retained), 'Actual used loss target differs from exact retained V2 array')
            if name == 'delivered': require(used.min() >= 0 and used.max() <= 1, 'Behavior RGB outside [0,1]')
            difference = np.abs(online.astype(np.float64) - retained.astype(np.float64))
            values = {'exact_array_equal': bool(np.array_equal(online, retained)),
                'maximum_absolute': float(difference.max()), 'mean_absolute': float(difference.mean()),
                'online_tensor_sha256': array_digest(online), 'used_target_tensor_sha256': array_digest(used),
                'retained_file_sha256': control[name + '_sha256']}
            require(comparison['comparisons'][name].keys() == values.keys(), 'Behavior comparison scalar namespace differs')
            for key, expected in values.items():
                actual = comparison['comparisons'][name][key]
                if type(expected) is float: close(actual, expected)
                else: require(actual == expected, 'Recomputed behavior comparison differs')
            recomputed['comparisons'][name] = values
            if name == 'delivered': exact_rgb += int(values['exact_array_equal'])
            else: exact_logits += int(values['exact_array_equal'])
        online_rows.append(recomputed)
    initial = read_json(out / 'initial_FP32_path_check.json')
    require(initial['update'] == 1 and initial['slot'] == 3 and initial['optimizer_updates_before_check'] == 0 and
            initial['pass'] is True and initial['bound'] == 1e-6 and
            0 <= initial['image_anchor_l1'] <= 1e-6 and abs(initial['token_KL']) <= 1e-6,
            'Initial numerical path gate failed or was too late')
    dtype_evidence(initial['dtypes'])
    require(sha256(safe_file(imported, initial['tokens_file'])) == initial['tokens_sha256'], 'Initial path token hash differs')
    sample = expected_records[(1, 3)]
    close(initial['image_anchor_l1'], sample['terms']['behavior_rgb_l1'])
    close(initial['token_KL'], sample['terms']['behavior_token_kl'])
    return {'actual_TRAIN_inputs_verified': input_records, 'actual_TRAIN_dtype_and_tokens_verified': input_records,
        'exact_retained_behavior_loss_target_pairs': completed, 'new_online_FP32_comparison_pairs': completed,
        'online_RGB_exact_pairs': exact_rgb, 'online_logits_exact_pairs': exact_logits,
        'all_clean512_tensor_hashes_match_retained_V2_records': True,
        'clean512_pixels_not_saved_in_this_return_prospective_preparation_verified_targets': True,
        'initial_FP32_path_gate': initial, 'online_comparisons': online_rows,
        'no_recreated_noise_or_cross_runtime_camera_rendering': True,
        'saved_loss_targets_bound_and_checked_neural_loss_not_recomputed': True}


def audit_hq_cache(imported, out, references, original, init, teacher_asset):
    import numpy as np
    import torch
    teacher = torch.load(teacher_asset, map_location='cpu', weights_only=True)['params_ema']
    require(tensor_state_digest(teacher) == init['frozen']['teacher'], 'HQ teacher state hash differs')
    del teacher; gc.collect()
    train = {row['id']: row for row in references if row['historical_role'] == 'train'}
    records = sorted((out / 'HQ_cache').glob('*/record.json'))
    require(len(records) == len(train) == 391 and {path.parent.name for path in records} == set(train),
            'Complete exact 391 TRAIN HQ-cache namespace required')
    codebook = original['quantize.embedding.weight']
    digest = tensor_state_digest({'embedding.weight': codebook})
    clean_hashes = {}
    for path in records:
        row = read_json(path); identifier = path.parent.name; ref = train[identifier]
        require(row['reference_id'] == identifier and row['role'] == 'train' and
                row['source_path'] == ref['source_path'] and row['source_sha256'] == ref['source_sha256'] and
                row['teacher_state_sha256'] == init['frozen']['teacher'] and
                row['restoration_codebook_state_sha256'] == digest and
                row['prior_V2_HQ_cache_not_exported_and_exact_tensor_parity_unproven'] is True,
                'HQ cache source/state/role binding differs')
        indices_path = safe_file(imported, row['indices_path']); features_path = safe_file(imported, row['features_path'])
        require(indices_path.parent == features_path.parent == path.parent and
                sha256(indices_path) == row['indices_sha256'] and sha256(features_path) == row['features_sha256'],
                'HQ cache array file binding differs')
        indices = load_array(indices_path, (1, 256), np.int64)
        features = load_array(features_path, (1, 256, 16, 16), np.float32)
        require(indices.min() >= 0 and indices.max() < 1024, 'HQ cache index outside frozen codebook')
        expected = codebook[torch.from_numpy(indices.reshape(-1))].reshape(1, 16, 16, 256).permute(0, 3, 1, 2)
        require(np.array_equal(expected.contiguous().numpy(), features), 'Saved HQ features differ from exact frozen codebook lookup')
        clean_hashes[identifier] = row['clean512_float32_sha256']
    return {'verified_HQ_TRAIN_cache_pairs': 391, 'frozen_codebook_lookup_exact': True,
        'HQ_teacher_state_verified_CPU_only': True, 'codebook_state_sha256': digest,
        'clean512_float32_sha256': clean_hashes, 'old_new_HQ_cache_equality_unproven': True,
        'HQ_encoder_or_quantizer_neural_forward_not_reexecuted': True}


def audit_perceptual_state(parent, protocol, init):
    """Assemble frozen feature/calibration/buffer tensors directly, no module."""
    import torch
    features = torch.load(safe_file(parent, protocol['assets']['perceptual_features']['path']),
                          map_location='cpu', weights_only=True)['params']
    calibration = torch.load(safe_file(parent, protocol['assets']['perceptual_calibration']['path']),
                             map_location='cpu', weights_only=True)
    tensors = dict(features)
    tensors.update({'calibration.' + str(i): calibration[f'lin{i}.model.1.weight'] for i in range(5)})
    tensors.update(shift=torch.tensor([-.030, -.088, -.188], dtype=torch.float32).reshape(1, 3, 1, 1),
                   scale=torch.tensor([.458, .448, .450], dtype=torch.float32).reshape(1, 3, 1, 1))
    actual = tensor_state_digest(tensors)
    require(actual == init['frozen']['perceptual'], 'Frozen perceptual state differs from exact CPU assets')
    return {'state_sha256': actual, 'tensors': len(tensors), 'module_instantiated': False}


def audit_return(imported, packet, parent, outputdir, *, transport_receipt=None):
    """Require closed strict transport import before CPU-only scientific auditing."""
    import torch
    torch.set_num_threads(1)
    imported, packet, parent = (Path(x).resolve(strict=True) for x in (imported, packet, parent))
    outputdir = Path(outputdir)
    require(not outputdir.exists(), 'Preserve prior trajectory audit')
    transport_receipt = (imported.parent / (imported.name + '_review') / 'transport_v1.json'
                         if transport_receipt is None else Path(transport_receipt))
    transport = verified_transport(imported, transport_receipt)
    protocol = read_json(safe_file(packet, 'protocol.json'))
    require(sha256(packet / 'protocol.json') == PROTOCOL_SHA256 and
            (imported / 'protocol.json').read_bytes() == (packet / 'protocol.json').read_bytes(), 'Exact frozen protocol required')
    require(protocol['training']['max_updates'] == 50 and protocol['training']['schedule_horizon_updates'] == 600 and
            protocol['training']['data_horizon_updates'] == 600 and protocol['training']['snapshots'] == [0, 50] and
            protocol['training']['autocast'] == 'disabled-FP32', 'Wrong finite comparison controls')
    for relative in protocol['export_files']:
        require(sha256(safe_file(imported, relative)) == protocol['files_sha256'][relative], 'Frozen exported source/metadata changed')
    for name, asset in protocol['assets'].items():
        path = safe_file(parent, asset['path'])
        require(path.stat().st_size == asset['bytes'] and sha256(path) == asset['sha256'], 'Read-only parent asset changed: ' + name)
    out = imported / protocol['outputs']
    failure = read_json(out / 'failure.json') if (out / 'failure.json').exists() else None
    finished = read_json(out / 'run_results.json') if (out / 'run_results.json').exists() else None
    require(bool(failure) != bool(finished), 'One authoritative terminal run/failure receipt required')
    terminal = failure or finished; completed = terminal['completed_updates']
    require(terminal['protocol_sha256'] == PROTOCOL_SHA256 and type(completed) is int and 0 <= completed <= 50,
            'Invalid terminal run update/protocol')
    parity = read_json(safe_file(packet, protocol['training_parity']['path']))
    require(sha256(packet / protocol['training_parity']['path']) == protocol['training_parity']['sha256'],
            'Frozen training parity manifest changed')
    references = read_json(safe_file(packet if (packet / protocol['data_manifest']).exists() else parent,
                                     protocol['data_manifest']))['references']
    rows, warnings = read_history(out / 'training.jsonl', failure)
    init = read_json(out / 'initialization.json')
    report = {'format': 'codeformer-fp32-independent-return-trajectory-v1',
        'UTC': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': PROTOCOL_SHA256,
        'auditor_source_sha256': sha256(Path(__file__)), 'training_completed': finished is not None,
        'verified_transport': transport,
        'completed_updates': completed, 'stop_reason': None if failure is None else failure['error'],
        'history_warnings': warnings, 'model_qualified': False, 'resume_authorized': False,
        'neural_instances': 0, 'neural_forwards': 0, 'gradient_queries': 0, 'optimizer_updates_by_audit': 0,
        'sections': {}, 'issues': [], 'limitations': [
            'Actual gradients and neural losses are not independently recomputed; committed states and scalar reductions are checked.',
            'Runtime reservations and dtype captures are frozen-source recorded evidence, not a separate live runtime probe.',
            'HQ teacher encoder/nearest-code selection is not rerun; saved indices/features and frozen lookup are independently checked.',
            'Pixel/quality-guard replay belongs to the separate metrics audit. Audit consistency cannot qualify restoration.']}
    def section(name, function):
        try: report['sections'][name] = function()
        except (AuditError, KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
            report['issues'].append({'section': name, 'error_type': type(error).__name__, 'error': str(error)})
    require(len(rows) == completed, 'Committed history differs from terminal update count')
    trajectory, states, expected_records = audit_history(rows, references, protocol)
    report['sections']['trajectory'] = trajectory
    for row, control in zip(rows, parity['updates']):
        require(row['group_lrs'] == control['group_lrs'] and row['TRAIN_selection'] == control['TRAIN_selection'],
                'Actual first-fifty LR/sampler parity differs from V2')
    section('call_trace', lambda: audit_calls(imported, out, protocol, terminal))
    section('inputs_and_anchors', lambda: audit_inputs_anchors(imported, packet, out, parity, expected_records,
                                                             completed, init['frozen']))
    original = torch.load(safe_file(parent, protocol['assets']['initializer']['path']),
                          map_location='cpu', weights_only=True)['params_ema']
    require(len(original) == 515 and tensor_state_digest(original) == init['frozen']['behavior_teacher'],
            'Untouched initializer fingerprint/census differs')
    section('HQ_cache', lambda: audit_hq_cache(imported, out, references, original, init,
                                              safe_file(parent, protocol['assets']['teacher']['path'])))
    section('perceptual_state', lambda: audit_perceptual_state(parent, protocol, init))
    if 'HQ_cache' in report['sections']:
        hashes = report['sections']['HQ_cache']['clean512_float32_sha256']
        require(all(hashes[row['reference_id']] == row['clean512_float32_sha256'] for row in parity['cases']),
                'HQ-cache clean input differs from selected exact V2 clean target bindings')
    checkpoint_reports = []
    def checkpoint_section():
        for path in sorted(out.glob('snapshot_*/checkpoint.pt')) + ([out / 'stopped_checkpoint.pt'] if (out / 'stopped_checkpoint.pt').exists() else []):
            state = torch.load(safe_file(imported, path.relative_to(imported).as_posix()), map_location='cpu', weights_only=True)
            at = state['completed_updates']
            require(at in states and (path.name == 'stopped_checkpoint.pt' and at == completed or
                    path.parent.name == f'snapshot_{at:04d}' and at in protocol['training']['snapshots']),
                    'Checkpoint path/update outside committed history')
            value = check_checkpoint(state, original, protocol, states[at], at)
            trainable = sum(item['tensors'] for name, item in value['module_changes'].items() if name in TRAINABLE)
            frozen = sum(item['tensors'] for name, item in value['module_changes'].items() if name in FROZEN)
            require(trainable == 350 and frozen == 165 and value['frozen_state_sha256'] == init['frozen']['candidate_frozen'],
                    'Checkpoint trainable/frozen census or fingerprint differs')
            cap = (384 if not state['optimizer']['state'] else 832) * 1024**2
            require(path.stat().st_size <= cap, 'Checkpoint exceeded prospective strict writer bound')
            if path.name == 'stopped_checkpoint.pt':
                require(not failure.get('optimizer_inflight') or not value['reported_resume_safe'],
                        'In-flight Adam falsely marked safe')
                receipt = failure['checkpoint']
                require(receipt['path'] == path.relative_to(imported).as_posix() and receipt['sha256'] == sha256(path)
                        and receipt['bytes'] == path.stat().st_size, 'Stopped checkpoint receipt differs')
            value.update(path=path.relative_to(imported).as_posix(), sha256=sha256(path), bytes=path.stat().st_size)
            checkpoint_reports.append(value)
            del state; gc.collect()
        require({item['completed_updates'] for item in checkpoint_reports if '/snapshot_' in item['path']} ==
                {at for at in protocol['training']['snapshots'] if at <= completed}, 'Scheduled checkpoint missing')
        same_update = {}
        for at in states:
            items = [item for item in checkpoint_reports if item['completed_updates'] == at]
            if len(items) > 1 and all(item['reported_resume_safe'] for item in items):
                same_update[at] = len({item['model_Adam_RNG_sampler_schedule_sha256'] for item in items}) == 1
                require(same_update[at], 'Same-update safe model/Adam/committed RNG/sampler differ')
        return {'states': checkpoint_reports, 'same_update_safe_states_equal': same_update,
                'original_equals_snapshot0_exact': True, 'trainable_tensors': 350, 'frozen_tensors': 165}
    section('checkpoints', checkpoint_section)
    del original; gc.collect()
    def timing_section():
        benchmark = read_json(out / 'benchmark20.json')
        require(benchmark['complete'] is True and benchmark['updates'] == 20 and benchmark['group_requirement_pass'] is True
                and set(benchmark['group_coverage']) == set(TRAINABLE) and all(
                item['finite_nonzero_grad'] and item['actual_change'] for item in benchmark['group_coverage'].values()),
                'Benchmark gradient/actual-change coverage receipt failed')
        close(benchmark['projected_training_seconds'], benchmark['seconds'] / 20 * 50)
        values = {'worker': (terminal['worker_seconds'], protocol['limits']['worker_seconds']),
                  'optimization': (trajectory['optimization_seconds'], protocol['limits']['training_seconds']),
                  'initialization': (init['seconds'], protocol['limits']['initialization_seconds']),
                  'benchmark20': (benchmark['seconds'], protocol['limits']['benchmark_seconds']),
                  'projected50': (benchmark['projected_training_seconds'], protocol['limits']['projected_training_seconds'])}
        for at in (0, 50):
            path = out / f'snapshot_{at:04d}' / 'metrics.json'
            if path.exists(): values[f'snapshot{at}'] = (read_json(path)['seconds'], protocol['limits']['snapshot_seconds'])
        require(all(finite(seconds) and 0 <= seconds <= cap for seconds, cap in values.values()), 'Recorded duration exceeds finite cap')
        require(0 < benchmark['CUDA_peak_reserved_bytes'] <= protocol['limits']['CUDA_reserved_bytes'], 'Recorded reserved memory exceeded cap')
        run = read_json(safe_file(imported, 'outputs/supervision/run_receipt.json'))
        require(run['protocol_sha256'] == PROTOCOL_SHA256 and run['manual_tmux'] is True and
                run['execution_max_updates'] == 50 and run['schedule_horizon_updates'] == 600 and
                run['no_automatic_retry_or_resume'] is True and run['model_qualified'] is False and
                run['automatic_app_promotion'] is False and
                [stage['stage'] for stage in run['stages']] == ['--verify-transfer', '--run'] and
                run['stages'][0]['exit_code'] == 0 and
                run['stages'][1]['exit_code'] == run['training_process_exit_code'] and
                ((failure is not None and run['training_process_exit_code'] != 0) or
                 (finished is not None and run['training_process_exit_code'] == 0)), 'Supervisor terminal classification differs')
        require(terminal['worker_seconds'] <= run['stages'][1]['seconds'] <= 3000 and
                run['stages'][1]['external_cap_seconds'] == 3000 and run['stages'][1]['grace_seconds'] == 180,
                'Supervisor run timing cap differs')
        actual_runtime = read_json(out / 'runtime.json')
        runtime_match = read_json(out / 'runtime_parity.json')
        version_keys = ('python', 'torch', 'numpy', 'Pillow', 'OpenCV', 'CUDA', 'device', 'capability')
        require(runtime_match['complete'] is True and runtime_match['matched_keys'] == list(version_keys) and all(
                actual_runtime[key] == parity['runtime_expected'][key] == runtime_match['current_values'][key]
                for key in version_keys), 'Recorded computational runtime versions differ from V2')
        return {'durations': {name: {'seconds': seconds, 'cap': cap} for name, (seconds, cap) in values.items()},
                'supervised_run_seconds': run['stages'][1]['seconds'],
                'CUDA_peak_reserved_bytes': benchmark['CUDA_peak_reserved_bytes'], 'gradient_receipts_not_reexecuted': True,
                'computational_runtime_matches_V2': True, 'source_supervisor_exit_code': run['training_process_exit_code']}
    section('finite_timing', timing_section)
    report['audit_finished'] = True
    report['evidence_consistency_pass'] = not report['issues']
    report['authoritative_status'] = 'Stopped quality experiment: retain failure, do not append epochs or auto-resume' if failure else 'Finite training finished; quality/usefulness unproven'
    outputdir.mkdir(parents=True)
    with (outputdir / 'trajectory_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imported', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--outputdir', type=Path, required=True)
    parser.add_argument('--transport-receipt', type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_return(args.imported, args.packet, args.parent, args.outputdir,
                              transport_receipt=args.transport_receipt)
    except BaseException as error:
        if not args.outputdir.exists():
            args.outputdir.mkdir(parents=True)
            with (args.outputdir / 'audit_exception.json').open('x', encoding='utf-8') as stream:
                json.dump({'complete': False, 'kind': 'local_auditor_exception',
                    'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                    'auditor_source_sha256': sha256(Path(__file__)), 'model_qualified': False,
                    'no_VM_or_neural_training_failure_inferred_from_checker_exception': True}, stream, indent=2)
                stream.write('\n')
        raise
    print(json.dumps({key: result[key] for key in ('audit_finished', 'evidence_consistency_pass', 'completed_updates', 'stop_reason', 'issues')}))
    return 0 if result['evidence_consistency_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())


