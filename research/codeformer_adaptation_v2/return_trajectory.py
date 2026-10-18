"""Independent saved-evidence V2 audit. No model instance/forward/gradient/VM.

The frozen recipe is never edited or resumed. Checkpoints are loaded on CPU
with weights_only=True; scalar losses/teacher outputs are not re-executed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


PROTOCOL_SHA256 = '029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'
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
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
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
    phase = (update - warmup) / (training['max_updates'] - warmup)
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
    from . import data
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


def check_noise_fields(parameters, context, recorded_runtime):
    """Active fields must replay exactly; unused discrepancies stay disclosed.

    The exact producer hashes still must match between normal/dark pair members.
    Local/version-dependent unused fields contribute identically zero, and are
    not relabeled as successfully reproduced arrays.
    """
    import numpy as np
    from . import data
    linear, rgb = data._noise_fields(parameters)
    fields = parameters['noise_fields']
    require(fields['shape'] == list(linear.shape) and fields['dtype'] == 'float64',
            'Noise-field geometry/dtype metadata differs')
    mismatches, exact, active_count = [], 0, 0
    specs = [('linear_noise_sha256', linear,
              parameters['shot_coefficient'] != 0 or parameters['read_sigma_linear'] != 0),
             ('rgb_noise_sha256', rgb, parameters['rgb_noise_sigma_255'] != 0)]
    for name, values, active in specs:
        require(np.isfinite(values).all(), 'Recreated finite Gaussian field required')
        actual = hashlib.sha256(values.tobytes()).hexdigest()
        active_count += int(active)
        if actual == fields[name]:
            exact += 1
            continue
        require(not active, 'ACTIVE keyed noise-field fingerprint differs: ' + context + '/' + name)
        coefficients = ({'shot_coefficient': parameters['shot_coefficient'],
                         'read_sigma_linear': parameters['read_sigma_linear']}
                        if name == 'linear_noise_sha256' else
                        {'rgb_noise_sigma_255': parameters['rgb_noise_sigma_255']})
        require(all(value == 0 for value in coefficients.values()), 'Unused-field zero coefficient proof failed')
        mismatches.append({'record': context, 'field': name, 'used_in_rendering': False,
            'recorded_sha256': fields[name], 'local_replay_sha256': actual,
            'recorded_runtime': recorded_runtime, 'local_numpy': np.__version__,
            'coefficients': coefficients,
            'zero_contribution_expression': ('N*sqrt(0*S+0**2)=0' if name == 'linear_noise_sha256' else 'N*0/255=0'),
            'exact_cross_runtime_array_hash_verified': False,
            'precise_version_or_platform_cause_not_proven_without_original_arrays': True})
    return {'fields': 2, 'exact_hash_matches': exact, 'active_fields_verified': active_count,
            'unused_field_replay_warnings': mismatches}


def audit_inputs_anchors(imported, out, references, expected_records, frozen, committed_updates, protocol):
    import numpy as np
    from PIL import Image
    from . import data
    refs = {row['id']: row for row in references if row['historical_role'] == 'train'}
    records, partial, verified_anchors, warnings = {}, [], [], []
    exact_noise_fields, active_fields, total_fields = 0, 0, 0
    for path in sorted((out / 'training_inputs').glob('update_*/slot_*/pair_record.json')):
        relative = path.relative_to(imported).as_posix()
        record = read_json(safe_file(imported, relative))
        update, slot = record['update'], record['slot']
        key = (update, slot)
        require(key not in records and 1 <= update <= min(600, committed_updates + 1) and 1 <= slot <= 4,
                'Duplicate/out-of-prefix TRAIN evidence')
        require(relative == protocol['outputs'] + '/training_inputs/update_%04d/slot_%d/pair_record.json' % key,
                'TRAIN record directory differs')
        require(record['reference_id'] in refs and record['role'] == 'train', 'Unapproved TRAIN input record')
        ref = refs[record['reference_id']]
        require(record['source_path'] == ref['source_path'] and record['source_sha256'] == ref['source_sha256'] and
                record['clean512_rgb8_sha256'] == ref['proposed_512_rgb_sha256'] and
                record['reference_manifest_sha256'] == protocol['files_sha256'][protocol['data_manifest']],
                'Saved TRAIN source/target/manifest binding differs')
        camera = data.camera_parameters(record['reference_id'], record['pair_role'], update)
        require(all(record['parameters'].get(key) == value for key, value in camera.items()),
                'Partial/committed input camera parameters differ from the finite recipe')
        expected = expected_records.get(key)
        if expected:
            require(record == expected['pair_record'], 'Saved TRAIN pair record disagrees with trajectory')
        else:
            partial.append({'update': update, 'slot': slot, 'path': relative})
        input_path = safe_file(imported, record['saved_input_path'])
        require(input_path == path.parent / 'input.png' and sha256(input_path) == record['saved_input_sha256'],
                'Saved TRAIN input file binding differs')
        require(input_path.stat().st_size <= protocol['limits']['PNG_file_bytes'], 'TRAIN PNG byte cap exceeded')
        with Image.open(input_path) as image:
            require(image.size == (256, 256) and image.mode == 'RGB', 'TRAIN input geometry/mode differs')
            pixels = np.asarray(image).copy()
        require(hashlib.sha256(pixels.tobytes()).hexdigest() == record['input256_rgb8_sha256'],
                'TRAIN RGB8 pixel hash differs')
        floats = np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255))
        require(hashlib.sha256(floats.tobytes()).hexdigest() == record['input256_float32_sha256'],
                'TRAIN prepared float32 hash differs')
        parameters = record['parameters']
        # Recreate keyed noise fields only; source photograph/camera pixels are
        # not re-rendered. The actual saved PNG is authoritative across codecs.
        noise = check_noise_fields(parameters, relative, record.get('runtime', {}))
        exact_noise_fields += noise['exact_hash_matches']; active_fields += noise['active_fields_verified']
        total_fields += noise['fields']; warnings.extend(noise['unused_field_replay_warnings'])
        records[key] = record
    require(set(expected_records) <= set(records), 'Committed TRAIN input PNG/record missing')
    for update in {key[0] for key in records}:
        for normal, dark in [(1, 2), (3, 4)]:
            if (update, normal) in records and (update, dark) in records:
                a, b = records[(update, normal)], records[(update, dark)]
                require(a['pair_id'] == b['pair_id'] and a['reference_id'] == b['reference_id'] and
                        a['parameters']['noise_fields'] == b['parameters']['noise_fields'],
                        'Exact producer normal/dark noise-field bindings differ')
    orphan_pngs = []
    for path in sorted((out / 'training_inputs').glob('update_*/slot_*/input.png')):
        if not (path.parent / 'pair_record.json').exists():
            orphan_pngs.append(path.relative_to(imported).as_posix())
    for path in sorted((out / 'behavior_anchors').glob('update_*/record.json')):
        anchor = read_json(safe_file(imported, path.relative_to(imported).as_posix()))
        update = anchor['update']
        require(1 <= update <= min(600, committed_updates + 1) and anchor['slot'] == 3,
                'Out-of-prefix behavior anchor')
        record = records.get((update, 3))
        require(record is not None and anchor['pair_record'] == record and
                anchor['reference_id'] == record['reference_id'] and
                anchor['input256_float32_sha256'] == record['input256_float32_sha256'] and
                anchor['teacher_state_sha256'] == frozen['behavior_teacher'] and
                anchor['teacher_FP32_eval_frozen'] is True and anchor['teacher_is_not_clean_identity_truth'] is True,
                'Behavior teacher input/state binding differs')
        sample = expected_records.get((update, 3))
        if sample:
            require(sample['behavior_anchor'] == anchor, 'Trajectory/anchor receipt differs')
        for name, shape in [('delivered', (1, 3, 256, 256)), ('logits', (1, 256, 1024))]:
            array_path = safe_file(imported, anchor[name + '_path'])
            require(array_path.parent == path.parent and sha256(array_path) == anchor[name + '_sha256'],
                    'Behavior array file fingerprint differs')
            require(array_path.stat().st_size <= math.prod(shape) * 4 + 4096,
                    'Behavior array file exceeds declared geometry payload')
            value = np.load(array_path, allow_pickle=False, mmap_mode='r')
            require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                    'Behavior array shape/dtype/finiteness differs')
            if name == 'delivered':
                require(value.min() >= 0 and value.max() <= 1, 'Delivered teacher RGB is outside[0,1]')
        verified_anchors.append(update)
    require(set(range(1, committed_updates + 1)) <= set(verified_anchors), 'Committed behavior anchor missing')
    return {'verified_TRAIN_input_records': len(records), 'verified_teacher_anchor_updates': verified_anchors,
            'partial_uncommitted_inputs': partial, 'orphan_partial_input_PNGs': orphan_pngs,
            'all_committed_input_and_anchor_bindings_verified': True,
            'noise_fields_checked': total_fields, 'noise_field_exact_local_hash_matches': exact_noise_fields,
            'active_noise_fields_exact_hash_verified': active_fields,
            'unused_noise_field_replay_warnings': warnings,
            'all_noise_fields_claimed_identical_across_runtimes': not warnings,
            'teacher_forward_and_candidate_loss_not_reexecuted': True,
            'source_photograph_pixels_not_read': True}


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
    require(state['format'] == 'codeformer-adaptation-resumable-state-v2' and
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
            4 * expected_update <= counts['backward_calls'] <= min(2400, 4 * expected_update + 4),
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
            if expected_update == 0:
                require(value is None, 'Initial checkpoint already has Adam moments')
                continue
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


def audit_return(imported, packet, outputdir):
    """Audit a separately safe-imported return; emit a new report, never qualify."""
    from . import worker
    imported, packet, outputdir = Path(imported).resolve(strict=True), Path(packet).resolve(strict=True), Path(outputdir)
    require(sha256(safe_file(packet, 'protocol.json')) == PROTOCOL_SHA256, 'Expected immutable V2 protocol')
    protocol, transfer = worker.verify_transfer(packet, PROTOCOL_SHA256)
    require(safe_file(imported, 'protocol.json').read_bytes() == (packet / 'protocol.json').read_bytes(),
            'Returned protocol bytes differ')
    for relative in protocol['export_files']:
        require(sha256(safe_file(imported, relative)) == protocol['files_sha256'][relative], 'Returned bound source/manifest differs')
    out = imported / protocol['outputs']
    require(out.is_dir(), 'Imported V2 run directory missing')
    failure = read_json(out / 'failure.json') if (out / 'failure.json').exists() else None
    finished = read_json(out / 'run_results.json') if (out / 'run_results.json').exists() else None
    require(not (failure and finished), 'Return claims both stopped and completed training')
    terminal = failure or finished
    require(terminal is not None and terminal['protocol_sha256'] == PROTOCOL_SHA256,
            'Terminal run/failure receipt missing or mismatched')
    completed = terminal['completed_updates']
    require(type(completed) is int and 0 <= completed <= 600, 'Invalid completed update count')
    if finished:
        require(finished.get('complete') is True and completed == 600, 'Completed run is not full600 trajectory')
    rows, history_warnings = read_history(out / 'training.jsonl', failure)
    references = read_json(packet / protocol['data_manifest'])['references']
    report = {'format': 'codeformer-v2-independent-return-trajectory-v1', 'UTC': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': PROTOCOL_SHA256, 'auditor_source_sha256': sha256(Path(__file__)),
        'training_completed': finished is not None, 'completed_updates': completed,
        'stop_reason': None if failure is None else failure['error'], 'history_warnings': history_warnings,
        'model_qualified': False, 'automatic_app_promotion': False, 'resume_authorized': False,
        'neural_instances': 0, 'neural_forwards': 0, 'gradient_queries': 0, 'optimizer_updates_by_audit': 0,
        'packet_static_transfer_verified': transfer['complete'], 'sections': {}, 'issues': [],
        'limitations': ['Saved teacher outputs and loss scalars are bound/checked, not reproduced by neural forward.',
                        'Loss reduction arithmetic and frozen source are verified; actual gradients are not recomputed.',
                        'DEV quality/pixel/guard recomputation belongs to the separate independent metrics audit.',
                        'Training/export completion and audit consistency do not qualify restoration quality.']}
    states, expected_records = {}, {}
    try:
        require(len(rows) == completed, 'Terminal completed count differs from recoverable committed history')
        trajectory, states, expected_records = audit_history(rows, references, protocol)
        report['sections']['trajectory'] = trajectory
    except (AuditError, KeyError, TypeError, ValueError) as error:
        report['issues'].append({'section': 'trajectory', 'error': str(error)})
    init = read_json(out / 'initialization.json') if (out / 'initialization.json').exists() else None
    if init:
        try:
            require(init['complete'] is True and len(init['parity']) == 2 and
                    all(row['maximum_raw_error'] <= 1e-6 for row in init['parity']), 'Original same-VM parity failed')
            report['sections']['inputs_and_anchors'] = audit_inputs_anchors(imported, out, references,
                expected_records, init['frozen'], completed, protocol)
        except (AuditError, KeyError, TypeError, ValueError, OSError) as error:
            report['issues'].append({'section': 'inputs_and_anchors', 'error': str(error)})
    else:
        report['issues'].append({'section': 'initialization', 'error': 'Initialization receipt missing; failure may precede model loading'})
    try:
        counts = terminal['counts']
        for name, cap in protocol['training']['call_caps'].items():
            require(type(counts[name]) is int and 0 <= counts[name] <= cap, 'Finite call cap differs: ' + name)
        require(counts['optimizer_updates'] == completed and 4 * completed <= counts['backward_calls'] <=
                counts['candidate_training_forwards'] <= min(2400, 4 * completed + 4), 'Committed/partial training count inconsistency')
        require(4 * completed <= counts['microbatches'] <= counts['backward_calls'] <= counts['microbatches'] + 1,
                'Backward attempt versus completed microbatch counts differ')
        require(completed <= counts['behavior_teacher_forwards'] <= min(600, completed + 1), 'Behavior call prefix differs')
        if not failure or failure.get('pending_step_record') is None and counts['microbatches'] == 4 * completed:
            require(counts['perceptual_training_forwards'] == sum(int(sample['image_eligible']) for row in rows for sample in row['microbatches']),
                    'LPIPS count differs from actual eligible committed exposures')
        if init:
            require(counts['teacher_forwards'] == 391 and counts['baseline_forwards'] == 2 and
                    counts['initializer_parity_candidate_forwards'] == 2 and
                    counts['gradient_queries'] == counts['perceptual_prerequisite_forwards'] == 1,
                    'Completed initialization counter receipts differ')
        snapshot_rows, snapshot_timing = 0, []
        for directory in sorted(out.glob('snapshot_*')):
            metrics_path = directory / 'metrics.json'
            partial_path = directory / 'partial_metrics.json'
            metrics = read_json(metrics_path) if metrics_path.exists() else (
                read_json(partial_path) if partial_path.exists() else None)
            if metrics:
                snapshot_rows += len(metrics['rows'])
                if metrics.get('complete') is True:
                    require(len(metrics['rows']) == 406, 'Completed snapshot cohort count differs')
                    seconds = metrics['seconds']
                    require(finite(seconds) and seconds >= 0, 'Nonfinite snapshot duration')
                    snapshot_timing.append({'update': metrics['update'], 'seconds': seconds,
                        'within_cap': seconds <= protocol['limits']['snapshot_seconds']})
        require(snapshot_rows <= counts['snapshot_forwards'] <= snapshot_rows + 1,
                'Snapshot forward attempts disagree with saved complete/partial rows')
        timing = {'worker': {'seconds': terminal.get('worker_seconds'), 'cap': protocol['limits']['worker_seconds']},
                  'optimization': {'seconds': rows[-1]['optimization_seconds'] if rows else 0,
                                   'cap': protocol['limits']['training_seconds']}}
        if init:
            timing['initialization'] = {'seconds': init['seconds'], 'cap': protocol['limits']['initialization_seconds']}
        for value in timing.values():
            require(finite(value['seconds']) and value['seconds'] >= 0, 'Nonfinite saved duration')
            value['within_cap'] = value['seconds'] <= value['cap']
        benchmark_path = out / 'benchmark20.json'
        benchmark = read_json(benchmark_path) if benchmark_path.exists() else None
        if completed >= 20:
            require(benchmark is not None and benchmark['complete'] is True and benchmark['updates'] == 20 and
                    set(benchmark['group_coverage']) == set(TRAINABLE), 'Benchmark20 receipt/coverage missing')
            close(benchmark['projected_training_seconds'], benchmark['seconds'] / 20 * 600)
            require(all(row['finite_nonzero_grad'] and row['actual_change'] for row in benchmark['group_coverage'].values()) and
                    benchmark['group_requirement_pass'] is True, 'Reported intended module learning coverage failed')
            timing['benchmark20'] = {'seconds': benchmark['seconds'], 'cap': protocol['limits']['benchmark_seconds'],
                'within_cap': benchmark['seconds'] <= protocol['limits']['benchmark_seconds'],
                'projection_seconds': benchmark['projected_training_seconds'],
                'projection_within_cap': benchmark['projected_training_seconds'] <= protocol['limits']['projected_training_seconds'],
                'peak_reserved_bytes': benchmark['CUDA_peak_reserved_bytes'],
                'peak_reserved_within_cap': benchmark['CUDA_peak_reserved_bytes'] <= protocol['limits']['CUDA_reserved_bytes']}
        report['sections']['finite_counts'] = {'counts': counts, 'caps_respected': True,
            'reported_durations': timing, 'snapshot_durations': snapshot_timing,
            'timing_caps_not_quality_gates': True,
            'gradient_coverage_receipt_not_independently_recomputed': True}
    except (AuditError, KeyError, TypeError, ValueError) as error:
        report['issues'].append({'section': 'finite_counts', 'error': str(error)})
    if states and init:
        import torch
        original_path = safe_file(packet, protocol['assets']['initializer']['path'])
        original = torch.load(original_path, map_location='cpu', weights_only=True)['params_ema']
        require(tensor_state_digest(original) == init['frozen']['behavior_teacher'], 'Behavior teacher fingerprint is not initializer state')
        checkpoint_reports = []
        for path in sorted(out.glob('snapshot_*/checkpoint.pt')) + ([out / 'stopped_checkpoint.pt'] if (out / 'stopped_checkpoint.pt').exists() else []):
            relative = path.relative_to(imported).as_posix()
            try:
                state = torch.load(safe_file(imported, relative), map_location='cpu', weights_only=True)
                update = state['completed_updates']
                if path.name == 'stopped_checkpoint.pt':
                    require(update == completed, 'Stopped checkpoint completed count differs')
                else:
                    require(path.parent.name == 'snapshot_%04d' % update and
                            update in protocol['training']['snapshots'], 'Snapshot checkpoint path/update differs')
                require(update in states, 'Checkpoint is outside independently verified committed history')
                item = check_checkpoint(state, original, protocol, states[update], update)
                require(item['frozen_state_sha256'] == init['frozen']['candidate_frozen'], 'Frozen-state receipt differs')
                if failure and path.name == 'stopped_checkpoint.pt':
                    require(not failure.get('optimizer_inflight') or not item['reported_resume_safe'], 'In-flight optimizer falsely marked safe')
                    receipt = failure.get('checkpoint')
                    if receipt:
                        require(receipt['path'] == relative and receipt['sha256'] == sha256(path) and
                                receipt['bytes'] == path.stat().st_size, 'Stopped checkpoint receipt differs')
                item.update(path=relative, sha256=sha256(path), bytes=path.stat().st_size)
                checkpoint_reports.append(item)
                del state
            except (AuditError, KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
                report['issues'].append({'section': 'checkpoint', 'path': relative, 'error': str(error)})
        require(bool(checkpoint_reports), 'No audited checkpoint state exists')
        expected_snapshot_updates = {update for update in protocol['training']['snapshots'] if update <= completed}
        actual_snapshot_updates = {item['completed_updates'] for item in checkpoint_reports
                                   if '/snapshot_' in item['path']}
        if not expected_snapshot_updates <= actual_snapshot_updates:
            report['issues'].append({'section': 'checkpoint', 'error': 'Scheduled checkpoint evidence missing/invalid',
                                     'updates': sorted(expected_snapshot_updates - actual_snapshot_updates)})
        by_update = defaultdict(list)
        for item in checkpoint_reports:
            by_update[item['completed_updates']].append(item)
        equal_at_same_update = {}
        for update, items in by_update.items():
            if len(items) > 1 and all(item['reported_resume_safe'] for item in items):
                same = len({item['model_Adam_RNG_sampler_schedule_sha256'] for item in items}) == 1
                equal_at_same_update[str(update)] = same
                if not same:
                    report['issues'].append({'section': 'checkpoint', 'error': 'Same-update safe states differ', 'update': update})
        report['sections']['checkpoints'] = {'states': checkpoint_reports,
            'same_update_safe_model_Adam_RNG_sampler_equal': equal_at_same_update,
            'initializer_equals_update0_exact': any(item['completed_updates'] == 0 for item in checkpoint_reports)}
        del original
    report['audit_finished'] = True
    report['evidence_consistency_pass'] = not report['issues']
    report['authoritative_status'] = ('Training finished; usefulness/preservation not qualified by this audit' if finished else
                                      'Stopped experiment; keep failure and states; do not append epochs or auto-resume')
    outputdir.mkdir(parents=True, exist_ok=True)
    destination = outputdir / 'trajectory_audit.json'
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imported', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--outputdir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_return(args.imported, args.packet, args.outputdir)
    print(json.dumps({key: result[key] for key in ['audit_finished', 'evidence_consistency_pass', 'completed_updates', 'stop_reason', 'model_qualified']}))
    return 0 if result['evidence_consistency_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
