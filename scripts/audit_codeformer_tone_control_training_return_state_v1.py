"""Independent CPU saved-state audit; never constructs or executes a neural model.

Torch is imported only after the completed import, frozen source, three composite
checkpoint files and the parent file have been hash-checked. Only CPU
weights_only+mmap deserialization and saved-tensor/NumPy arithmetic are used.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'f10ce6d52c2ec259632750af2d3afbdf97c5c925b99849fae2a2c4d9cd04a5b1'
PREFIX = 'outputs/tone_control_training_v1/'
STATE_FORMAT = 'codeformer-tone-control-composite-state-v1'
PARENT_MODEL_DIGEST = '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b'
ORIGINAL_MODEL_DIGEST = '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd'
HEAD_SHAPES = {'hidden.weight': (32, 256), 'hidden.bias': (32,),
               'output.weight': (1, 32), 'output.bias': (1,)}
SLOTS = ('composite_head0000', 'composite_head0050', 'stopped_composite')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(actual, expected, tolerance=1e-12):
    require(finite(actual) and finite(expected) and abs(actual - expected) <= tolerance * max(1., abs(expected)),
            'Scalar readback differs: %r versus %r' % (actual, expected))


class Evidence:
    def __init__(self, receipt_path, archive_sha):
        require(re.fullmatch('[0-9a-f]{64}', archive_sha), 'Explicit expected return archive SHA required')
        self.receipt_path = Path(receipt_path).resolve(strict=True)
        self.receipt_sha = sha(self.receipt_path)
        self.receipt = read(self.receipt_path)
        require(self.receipt.get('complete') is True and self.receipt.get('archive_sha256') == archive_sha and
                self.receipt.get('protocol_sha256') == PROTOCOL, 'Exact completed root import receipt required')
        self.root = Path(self.receipt['import_root']).resolve(strict=True)
        require(self.root == (ROOT / 'outputs/codeformer_tone_control_training_v1_return').resolve(strict=True),
                'Exact imported return root required')
        self.hashes, self.sizes = self.receipt['files_sha256'], self.receipt['files_bytes']
        require(set(self.hashes) == set(self.sizes) and all(type(v) is int and v >= 0 for v in self.sizes.values()),
                'Closed exact-size import map required')
        self.pins, self.parent_pins = {}, {}
        self.p = self.json('protocol.json')
        require(self.hashes['protocol.json'] == PROTOCOL and self.p['status'] == 'PREPARED_UNRUN',
                'Frozen prospective protocol differs')
        local_stage = ROOT / 'outputs/cctv_codeformer_tone_control_training_v1_vm'
        require(sha(local_stage / 'protocol.json') == PROTOCOL, 'Local sealed protocol changed')
        for name, digest in self.p['files_sha256'].items():
            require(self.hashes.get(name) == digest and self.sizes.get(name) == self.p['files_bytes'][name] and
                    sha(self.get(name)) == digest and sha(local_stage / name) == digest,
                    'Returned frozen source/stage binding differs: ' + name)
        manifest_name = PREFIX + 'export_manifest.json'
        self.manifest = self.json(manifest_name)
        require(self.manifest['protocol_sha256'] == PROTOCOL and
                self.manifest['files_sha256'] == {k: v for k, v in self.hashes.items() if k != manifest_name} and
                self.manifest['files_bytes'] == {k: v for k, v in self.sizes.items() if k != manifest_name},
                'Imported map and returned immutable export manifest differ')
        stage_receipt = ROOT / 'outputs/codeformer_tone_control_training_v1_preparation/stage_v1.json'
        self.stage_receipt_sha = sha(stage_receipt)
        self.resolution = read(stage_receipt)['local_dependency_resolution']

    def get(self, name):
        relative = PurePosixPath(name)
        require(name in self.hashes and relative.as_posix() == name and not relative.is_absolute() and
                '..' not in relative.parts and ':' not in name and '\\' not in name, 'Unbound/unsafe return path')
        path = self.root.joinpath(*relative.parts)
        require(path.resolve(strict=True).is_relative_to(self.root), 'Return path escape')
        for item in [path, *path.parents]:
            if item == self.root.parent:
                break
            info = item.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, 'st_file_attributes', 0) & 1024),
                    'Linked/reparse return evidence forbidden')
        info = path.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == self.sizes[name],
                'Return regular-file/size identity differs')
        identity = (info.st_size, info.st_mtime_ns, info.st_ino)
        if name not in self.pins:
            require(sha(path) == self.hashes[name], 'Imported bytes changed before audit: ' + name)
            self.pins[name] = {'sha256': self.hashes[name], 'identity': identity}
        else:
            require(identity == self.pins[name]['identity'], 'Read evidence changed during audit')
        return path

    def json(self, name):
        return read(self.get(name))

    def parent(self, name):
        pin = self.p['dependencies'][name]
        path = Path(self.resolution[name]).resolve(strict=True)
        require(path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and
                path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
                'Exact read-only parent bytes differ')
        self.parent_pins[name] = {'sha256': pin['sha256'], 'bytes': pin['bytes'], 'path': str(path)}
        return path

    def array(self, item, np):
        path = self.get(item['path'])
        require(self.hashes[item['path']] == item['sha256'] and path.stat().st_size == item['bytes'],
                'Saved array manifest binding differs')
        with path.open('rb') as stream:
            value = np.load(stream, allow_pickle=False)
            require(not stream.read(1), 'Trailing NPY bytes')
        require(str(value.dtype) == item['dtype'] and list(value.shape) == item['shape'] and
                value.flags.c_contiguous and np.isfinite(value).all() and
                hashlib.sha256(value.tobytes()).hexdigest() == item['array_sha256'],
                'Saved array shape/dtype/finiteness/digest differs')
        return value

    def finish(self):
        require(sha(self.receipt_path) == self.receipt_sha, 'Root import receipt changed')
        for name, pin in self.pins.items():
            info = (self.root / name).stat()
            require((info.st_size, info.st_mtime_ns, info.st_ino) == pin['identity'], 'Read evidence changed at close')
        return {'import_receipt_sha256': self.receipt_sha, 'stage_receipt_sha256': self.stage_receipt_sha,
                'individually_hashed_return_files': {k: v['sha256'] for k, v in sorted(self.pins.items())},
                'parent_files_read': self.parent_pins, 'read_stat_identity_checked_at_close': True}


def expected_events(caps):
    counts = dict.fromkeys(caps, 0)
    rows, snapshots = [], {}
    committed = pending = 0
    def reserve(kind):
        counts[kind] += 1
        rows.append({'event': len(rows) + 1, 'before_call': kind, 'completed_head_updates': committed,
                     'partial_microbatches': pending, 'head_optimizer_inflight': False, 'counts': dict(counts)})
    for _ in range(4): reserve('CodeFormer_parity')
    for _ in range(2):
        reserve('controller_math_prerequisites'); reserve('head_autograd_queries')
    snapshots[0] = dict(counts)
    for _ in range(406): reserve('CodeFormer_original_DEV_bank')
    for _ in range(406):
        reserve('CodeFormer_zero_head_DEV_bank'); reserve('controller_DEV_zero_and50')
    for _ in range(200):
        reserve('TRAIN_input_writes'); reserve('CodeFormer_TRAIN_original'); reserve('CodeFormer_TRAIN_full_coarse')
    for update in range(1, 51):
        for _ in range(4):
            reserve('controller_TRAIN'); reserve('head_backward_calls'); pending += 1
        reserve('head_optimizer_step_attempts')
        committed, pending = update, 0
    snapshots[50] = dict(counts)
    for _ in range(406):
        reserve('CodeFormer_head50_DEV_bank'); reserve('controller_DEV_zero_and50')
    for _ in range(2): reserve('controller_final_known_stress')
    require(len(rows) == 3090 and counts == caps, 'Independent operation census differs')
    snapshots['stopped'] = dict(counts)
    return rows, snapshots


def tensor_array(value, torch, np):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and not value.requires_grad,
            'Only detached CPU serialized tensors may be examined')
    result = value.detach().contiguous().numpy()
    require(np.isfinite(result).all(), 'Nonfinite saved tensor')
    return result


def tensor_digest(state, torch, np):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = tensor_array(value, torch, np)
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header); digest.update(array.tobytes())
    return digest.hexdigest()


def tree_digest(value, torch, np):
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = tensor_array(item, torch, np)
            return {'tensor': str(item.dtype), 'shape': list(item.shape), 'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
        if isinstance(item, dict):
            return {'dict': [[str(type(k).__name__) + ':' + str(k), visit(v)] for k, v in sorted(item.items(), key=lambda kv: str(kv[0]))]}
        if isinstance(item, (list, tuple)):
            return [visit(v) for v in item]
        require(item is None or type(item) in (int, float, bool, str), 'Unexpected serialized primitive')
        if isinstance(item, float): require(math.isfinite(item), 'Nonfinite serialized scalar')
        return item
    return hashlib.sha256(canonical(visit(value))).hexdigest()


def rng_schema(value, torch, np):
    require(set(value) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'} and len(value['python']) == 3 and
            value['python'][0] == 3 and len(value['python'][1]) == 625 and
            value['numpy']['algorithm'] == 'MT19937' and len(value['numpy']['keys']) == 624 and
            0 <= value['numpy']['position'] <= 624 and len(value['torch_cuda']) == 1,
            'Saved Python/NumPy/CPU/CUDA RNG schema differs')
    arrays = [tensor_array(value['torch_cpu'], torch, np), tensor_array(value['torch_cuda'][0], torch, np)]
    require(all(a.dtype == np.uint8 and a.ndim == 1 and len(a) > 0 for a in arrays), 'RNG byte tensors differ')
    return {'tree_sha256': tree_digest(value, torch, np), 'torch_CPU_bytes': arrays[0].size,
            'torch_CUDA0_bytes': arrays[1].size, 'restored_or_neurally_replayed_by_auditor': False}


def checkpoint_audit(e, paths, parent_path, snapshots, torch, np):
    parent = torch.load(parent_path, map_location='cpu', weights_only=True, mmap=True)
    parent_values = parent['model']
    require(len(parent_values) == 515 and tensor_digest(parent_values, torch, np) == PARENT_MODEL_DIGEST,
            'Historical parent saved model digest differs')
    checkpoints, rows = {}, {}
    for slot, position in zip(SLOTS, (0, 50, 'stopped')):
        state = torch.load(paths[slot], map_location='cpu', weights_only=True, mmap=True)
        at = 0 if position == 0 else 50
        require(state['format'] == STATE_FORMAT and state['protocol_sha256'] == PROTOCOL and
                state['design_sha256'] == e.p['design_sha256'] and state['completed_head_updates'] == at and
                state['restorer_updates_this_run'] == 0 and state['historical_restorer_updates'] == 50 and
                state['partial_microbatches'] == 0 and state['head_optimizer_inflight'] is False and
                state['head_state_safe'] is True and state['no_automatic_resume_supported'] is True and
                state['model_qualified'] is False and state['counts'] == snapshots[position],
                'Composite actual update/pending/inflight/census/status metadata differs')
        require(state['parent_checkpoint_sha256'] == e.p['assets']['base']['sha256'] and
                state['original_checkpoint_sha256'] == e.p['assets']['initializer']['sha256'] and
                state['controller'] == e.p['design']['controller'] and
                state['composite_specification'] == e.p['design']['composite_and_loss'] and
                state['sampler'] == {'next_update': at + 1, 'retained_training_parity_sha256': e.p['training_parity']['sha256']} and
                state['schedule'] == {'completed_updates': at, 'training': e.p['design']['training']},
                'Composite parent/head/feature/output/sampler/schedule policy differs')
        base, head = state['base_model'], state['head']
        require(list(base) == list(parent_values) and len(base) == 515 and
                all(base[k].dtype == parent_values[k].dtype == torch.float32 and
                    np.array_equal(tensor_array(base[k], torch, np), tensor_array(parent_values[k], torch, np)) for k in base),
                'At least one saved frozen base tensor differs from historical parent')
        base_digest = tensor_digest(base, torch, np)
        require(base_digest == state['base_model_tensor_digest'] == PARENT_MODEL_DIGEST and
                list(head) == list(HEAD_SHAPES) and all(tuple(head[k].shape) == shape and head[k].dtype == torch.float32
                    for k, shape in HEAD_SHAPES.items()), 'Saved base/head geometry/precision differs')
        head_digest = tensor_digest(head, torch, np)
        require(head_digest == state['head_tensor_digest'] and sum(v.numel() for v in head.values()) == 8257,
                'Actual head tensor digest/census differs')
        optimizer = state['head_optimizer']
        require(set(optimizer) == {'state', 'param_groups'} and len(optimizer['param_groups']) == 1,
                'Only one head Adam group is permitted')
        group = optimizer['param_groups'][0]
        require(group['params'] == [0, 1, 2, 3] and group['lr'] == .001 and tuple(group['betas']) == (.9, .99) and
                group['eps'] == 1e-8 and group['weight_decay'] == 0 and group['amsgrad'] is False and
                group['maximize'] is False and group['differentiable'] is False,
                'Saved head Adam recipe/parameter-only partition differs')
        adam = []
        require(set(optimizer['state']) == (set() if at == 0 else {0, 1, 2, 3}), 'Head Adam state count differs')
        for index, name in enumerate(head):
            if at == 0: continue
            entry = optimizer['state'][index]
            require(set(entry) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Unexpected saved Adam state')
            step = tensor_array(entry['step'], torch, np)
            m, v = tensor_array(entry['exp_avg'], torch, np), tensor_array(entry['exp_avg_sq'], torch, np)
            require(step.size == 1 and float(step.reshape(-1)[0]) == 50 and m.shape == v.shape == tuple(head[name].shape) and
                    m.dtype == v.dtype == np.float32 and np.all(v >= 0), 'Saved Adam50 moment geometry/counter differs')
            adam.append({'name': name, 'step': 50, 'moment_nonzero': int(np.count_nonzero(m)),
                         'variance_nonzero': int(np.count_nonzero(v)), 'moment_maxabs': float(np.abs(m).max())})
        rows[slot] = {'path': paths[slot].relative_to(e.root).as_posix(), 'sha256': e.hashes[paths[slot].relative_to(e.root).as_posix()],
            'bytes': paths[slot].stat().st_size, 'completed_head_updates': at, 'restorer_updates_this_run': 0,
            'all515base_tensors_exact_parent': True, 'base_parameters': sum(v.numel() for v in base.values()),
            'base_model_tensor_sha256': base_digest, 'head_tensor_sha256': head_digest, 'head_tensors': 4,
            'head_parameters': 8257, 'head_Adam_states': len(optimizer['state']), 'Adam': adam,
            'RNG': rng_schema(state['rng'], torch, np), 'partial_microbatches': 0, 'inflight': False,
            'sampler': state['sampler'], 'head_optimizer_tree_sha256': tree_digest(optimizer, torch, np)}
        checkpoints[slot] = state
    initial, trained, stopped = [checkpoints[k] for k in SLOTS]
    require(np.count_nonzero(tensor_array(initial['head']['output.weight'], torch, np)) == 0 and
            np.count_nonzero(tensor_array(initial['head']['output.bias'], torch, np)) == 0,
            'Saved initial final layer must be exactly zero')
    changes = {}
    for name in HEAD_SHAPES:
        a, b = tensor_array(initial['head'][name], torch, np), tensor_array(trained['head'][name], torch, np)
        delta = b.astype(np.float64) - a.astype(np.float64)
        changes[name] = {'shape': list(a.shape), 'changed_elements': int(np.count_nonzero(a != b)),
                         'elements': a.size, 'maximum_change': float(np.abs(delta).max()),
                         'change_L2': float(np.sqrt(np.square(delta).sum()))}
    require(all(row['changed_elements'] > 0 for row in changes.values()), 'A head tensor did not change; report before claiming four changed tensors')
    same_keys = ('base_model', 'head', 'head_optimizer', 'rng', 'sampler', 'schedule')
    require(all(tree_digest(trained[key], torch, np) == tree_digest(stopped[key], torch, np) for key in same_keys),
            'Stopped composite differs from trained50 model/head/Adam/RNG/sampler/schedule')
    return checkpoints, {'checkpoints': rows, 'head_changes_from_zero_output_initialization': changes,
        'all4head_tensors_changed': True, 'trained50_and_stopped_state_exact': list(same_keys),
        'checkpoint_metadata_is_not_a_live_module_mode_or_gradient_replay': True}


def prerequisite_audit(e, initial, torch, np):
    directory = PREFIX + 'head_derivative_prerequisites/'
    report = e.json(directory + 'prerequisites.json')
    require(report['complete'] is report['pass'] is True and report['head_forwards'] == report['gradient_queries'] == 2 and
            report['CodeFormer_forwards'] == report['CodeFormer_backward'] == report['optimizer_steps'] == 0 and
            all(report[k] is True for k in ('head_state_restored_exact', 'RNG_restored_exact',
                'parameter_grad_fields_none_before_cleanup', 'parameter_grad_fields_none_restored', 'modes_restored_exact')),
            'Actual derivative prerequisites/restoration differ')
    digest = tensor_digest(initial['head'], torch, np)
    require(report['head_state_before_sha256'] == report['head_state_restored_sha256'] == digest,
            'Prerequisite initial/restored head does not bind checkpoint0')
    results = []
    for label in ('zero_output_chain', 'active_clipping_chain'):
        row = e.json(directory + label + '.json')
        base = np.asarray(row['base_RGB'], np.float64)
        observed = np.asarray(row['observed'], bool)
        hidden = np.asarray(row['hidden'], np.float64)
        require(base.shape == (2, 2, 3) and observed.shape == (2, 2) and hidden.shape == (32,) and
                observed.sum() == (4 if label == 'zero_output_chain' else 3), 'Actual prerequisite fixture geometry differs')
        delta = .25 * math.tanh(row['logit'])
        shifted = base + delta
        weights = np.asarray([.2126, .7152, .0722], np.float64)
        mean = float((np.clip(shifted, 0, 1)[observed] @ weights).mean())
        derivative = float((((shifted > 0) & (shifted < 1))[observed] @ weights).mean())
        scalar = 2 * (mean - row['target']) * derivative * .25 * (1 - math.tanh(row['logit'])**2)
        expected = {'hidden.weight': np.zeros((32, 256), np.float64), 'hidden.bias': np.zeros(32, np.float64),
                    'output.weight': scalar * hidden.reshape(1, 32), 'output.bias': np.asarray([scalar], np.float64)}
        arrays = {}
        for name, wanted in expected.items():
            item = row['gradient_arrays'][name]
            actual, exported_expected = e.array(item['actual'], np), e.array(item['expected'], np)
            require(item['is_none'] is False and actual.shape == exported_expected.shape == wanted.shape and
                    actual.dtype == np.float32 and exported_expected.dtype == np.float64 and
                    np.allclose(exported_expected, wanted, rtol=0, atol=1e-15), 'Independent analytic array differs')
            error = float(np.abs(actual.astype(np.float64) - wanted).max())
            bound = 2e-7 + 2e-5 * float(np.abs(wanted).max())
            require(error <= bound and item['pass'] is True and item['connected_zero_exact'] is True,
                    'Actual saved derivative violates prospective analytic bound')
            close(item['maximum_absolute_error'], float(np.abs(actual.astype(np.float64) - exported_expected).max()))
            close(item['bound'], bound)
            if name.startswith('hidden.'):
                require(np.count_nonzero(actual) == 0, 'Connected-zero hidden gradient changed')
            arrays[name] = {'maximum_absolute_error': error, 'prospective_bound': bound,
                            'physical_actual_and_expected_arrays': True, 'reported_is_none': False}
        corrected, pixels = np.asarray(row['corrected_RAW'], np.float32), np.asarray(row['input_RGB'], np.float32)
        require(np.array_equal(corrected[~observed], pixels[~observed]) and row['unobserved_input_exact'] is True,
                'Actual fixture protected input restoration differs')
        results.append({'fixture': label, 'observed_pixels': int(observed.sum()), 'gradients': arrays,
                        'hidden_activations_source_bound_not_locally_forward_replayed': True})
    before, after = [e.json(directory + 'rng_' + label + '.json') for label in ('before', 'restored')]
    require(before['python'] == after['python'] and before['numpy'] == after['numpy'], 'Prerequisite Python/NumPy RNG restoration differs')
    for field, state in [('CPU', initial['rng']['torch_cpu']), ('CUDA0', initial['rng']['torch_cuda'][0])]:
        a, b = e.array(before['torch'][field], np), e.array(after['torch'][field], np)
        require(np.array_equal(a, b) and np.array_equal(a, tensor_array(state, torch, np)), 'Saved prerequisite/checkpoint0 RNG byte state differs')
    require(json.loads(json.dumps(initial['rng']['python'])) == before['python'] and initial['rng']['numpy'] == before['numpy'],
            'Prerequisite and checkpoint0 primitive RNG states differ')
    return {'complete': True, 'actual_saved_head_forwards': 2, 'actual_saved_gradient_queries': 2,
        'gradient_arrays_checked': 16, 'RNG_arrays_checked': 4, 'fixtures': results,
        'analytic_formula_independent': True, 'local_autograd_or_head_forward_calls': 0,
        'topology_scope': 'Physical arrays plus returned is_none/source query checks; not local autograd replay'}


def history_audit(e, checkpoints, torch, np):
    observations, rows = [], []
    for update in range(1, 51):
        a = e.json(PREFIX + 'history/u%04d_attempt.json' % update)
        b = e.json(PREFIX + 'history/u%04d_committed.json' % update)
        require(a['update'] == b['update'] == b['completed_head_updates'] == update and
                a['completed_before'] == update - 1 and a['observations'] == b['observations'] and
                len(a['observations']) == 4, 'Attempt/commit chronology differs')
        for slot, item in enumerate(a['observations'], 1):
            identifier = 'train_u%04d_slot%d' % (update, slot)
            role = 'normal' if slot in (1, 3) else 'dark'
            cache = e.json(PREFIX + 'TRAIN/' + identifier + '/cache.json')
            require(item['id'] == cache['id'] == identifier and item['role'] == cache['role'] == role and
                    cache['update'] == update and cache['slot'] == slot and item['actual_divisor'] == 4 and
                    item['target_mean_y'] == cache['target_mean_y'] and
                    cache['target_source'] == ('original' if role == 'normal' else 'full_coarse_base') and
                    -.25 <= item['delta'] <= .25, 'TRAIN role/target/divisor/input order differs')
            difference = np.float32(np.float32(item['predicted_mean_y']) - np.float32(item['target_mean_y']))
            replay_loss = float(np.float32(difference * difference))
            require(item['loss'] == replay_loss, 'One-observation FP32 squared-mean loss scalar differs')
            require(cache['reachable_mean_interval'][0] <= cache['target_mean_y'] <= cache['reachable_mean_interval'][1],
                    'Returned TRAIN target outside frozen reachable interval')
            observations.append(item)
        rows.append({'update': update, 'head_tensor_digest': b['head_tensor_digest'],
                     'head_optimization_seconds': b['head_optimization_seconds']})
    require(len(observations) == 200 and Counter(r['role'] for r in observations) == {'normal': 100, 'dark': 100} and
            rows[-1]['head_tensor_digest'] == tensor_digest(checkpoints[SLOTS[1]]['head'], torch, np) and
            all(rows[i]['head_optimization_seconds'] >= (rows[i - 1]['head_optimization_seconds'] if i else 0)
                for i in range(50)), 'Head history completion/final model digest differs')
    return {'attempt_records': 50, 'committed_records': 50, 'microbatches': 200,
        'normal': 100, 'dark': 100, 'all_scalar_losses_FP32_replayed_exact': True,
        'all_training_divisors': 4, 'head_optimization_seconds': rows[-1]['head_optimization_seconds'],
        'actual_gradient_or_optimizer_update_sequence_replayed': False}


def gate_audit(e):
    banks = {name: e.json(PREFIX + name + '_metrics.json')['cases'] for name in ('original0', 'head0000', 'head0050')}
    require(all(len(v) == 406 for v in banks.values()) and len({frozenset(v) for v in banks.values()}) == 1,
            'Full406 metric banks differ')
    saved = e.json(PREFIX + 'preservation_guards.json')
    result = {}
    for label, keys, metrics in [('clear', sorted(k for k, r in banks['head0050'].items()
            if r['evidence'] == 'paired_synthetic' and r['profile'] == 'clear'), ['mae']),
            ('normal_blur', sorted(k for k, r in banks['head0050'].items()
            if r['source'] == 'FFHQ validation' and r['profile'] == 'blur'), ['luminance_bias', 'coarse_rgb_mae'])]:
        require(len(keys) == (61 if label == 'clear' else 53), 'Guard cohort changed')
        require(saved[label]['cases'] == len(keys) and saved[label]['mean_limit'] == .01 and
                saved[label]['worst_limit'] == .03, 'Saved prospective guard limits differ')
        item = {'pass': True, 'cases': len(keys), 'views': {}}
        for view in ('raw', 'native_png'):
            item['views'][view] = {}
            for metric in metrics:
                delta = {k: banks['head0050'][k]['metrics'][view][metric] - banks['original0'][k]['metrics'][view][metric] for k in keys}
                mean, worst = sum(delta.values()) / len(keys), max(delta.values())
                passed = mean <= .01 and worst <= .03
                prior = saved[label]['views'][view][metric]
                close(prior['mean_delta'], mean); close(prior['worst_delta'], worst)
                require(prior['deltas'] == delta and prior['pass'] is passed and
                        prior['cases_over_0_03'] == sum(v > .03 for v in delta.values()), 'Saved guard scalar arithmetic differs')
                item['views'][view][metric] = {'mean_delta': mean, 'worst_delta': worst,
                    'cases_over_0_03': sum(v > .03 for v in delta.values()), 'pass': passed}
                item['pass'] = item['pass'] and passed
        require(saved[label]['pass'] is item['pass'], 'Saved cohort guard pass differs')
        result[label] = item
    require(saved['pass'] is (result['clear']['pass'] and result['normal_blur']['pass']), 'Saved joint guard differs')
    dark = e.json(PREFIX + 'dark_retention.json')
    dark_passes = []
    for profile in ('lowlight', 'compound'):
        keys = sorted(k for k, r in banks['head0050'].items() if r['source'] == 'FFHQ validation' and r['profile'] == profile)
        require(len(keys) == 53, 'Dark cohort changed')
        means = {name: {metric: sum(banks[bank][k]['metrics']['native_png'][metric] for k in keys) / 53
            for metric in ('mae', 'ssim')} for name, bank in [('candidate', 'head0050'), ('base', 'head0000')]}
        for name in means:
            for metric in means[name]: close(dark['profiles'][profile][name][metric], means[name][metric])
        passed = means['candidate']['mae'] <= means['base']['mae'] and means['candidate']['ssim'] >= means['base']['ssim']
        require(dark['profiles'][profile]['pass'] is passed and dark['profiles'][profile]['cases'] == keys,
                'Dark requirement arithmetic/cohort differs')
        dark_passes.append(passed)
    require(dark['pass'] is all(dark_passes), 'Dark retention aggregate differs')
    stress = e.json(PREFIX + 'known_overlap_stress.json')
    require(stress['complete'] is True and [r['id'] for r in stress['cases']] ==
            e.p['design']['evaluation']['known_overlap_stress']['cases'], 'Known TRAIN stress namespace differs')
    for row in stress['cases']:
        a, b = row['after']['native_png'], row['before']['native_png']
        passed = a['luminance_bias'] < b['luminance_bias'] if row['role'] == 'normal' else a['mae'] <= b['mae'] and a['ssim'] >= b['ssim']
        require(row['pass'] is passed and row['independent_generalization'] is False, 'Known stress check differs')
    require(stress['pass'] is all(r['pass'] for r in stress['cases']), 'Known stress aggregate differs')
    return {'preservation': result, 'dark_retention': dark, 'known_TRAIN_stress': stress,
        'all_acceptance_checks_pass': saved['pass'] and dark['pass'] and stress['pass'],
        'scope': 'Recomputed from saved metric scalars only; independent saved-pixel readback is separate'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-receipt', type=Path, required=True)
    parser.add_argument('--archive-sha', required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/codeformer_tone_control_training_v1_return_review/state_v1')
    args = parser.parse_args(argv)
    out = args.out.resolve(strict=False)
    require(out.parent == (ROOT / 'outputs/codeformer_tone_control_training_v1_return_review').resolve(strict=True) and
            not out.exists(), 'New unique state-audit directory required')
    e = Evidence(args.import_receipt, args.archive_sha)
    paths = {slot: e.get(PREFIX + slot + '/checkpoint.pt') for slot in SLOTS}
    require(all(p.stat().st_size <= 402653184 for p in paths.values()), 'Composite file cap exceeded')
    parent_path = e.parent(e.p['assets']['base']['path'])
    # This point is reached only after all returned source and checkpoint hashes.
    import numpy as np
    import torch
    started = time.monotonic()
    out.mkdir()
    report = {'format': 'codeformer-tone-control-CPU-saved-state-audit-v1', 'complete': False,
        'protocol_sha256': PROTOCOL, 'archive_sha256': args.archive_sha,
        'source_sha256': sha(__file__), 'model_qualified': False, 'sections': {}}
    try:
        expected, snapshots = expected_events(e.p['call_caps'])
        trace_names = {n for n in e.hashes if n.startswith(PREFIX + 'call_trace/')}
        require(trace_names == {PREFIX + 'call_trace/%05d.json' % (i + 1) for i in range(3090)}, 'Exact3090 trace files required')
        last_time = 0.
        for index, wanted in enumerate(expected, 1):
            actual = e.json(PREFIX + 'call_trace/%05d.json' % index)
            require({k: actual[k] for k in wanted} == wanted and finite(actual['seconds']) and actual['seconds'] >= last_time,
                    'Operation chronology/count/commit/pending differs')
            last_time = actual['seconds']
        report['sections']['operation_census'] = {'trace_events': 3090, 'counts': e.p['call_caps'],
            'full_CodeFormer_forwards': 1622, 'controller_forwards': 1016, 'total_neural_forwards': 2638,
            'head_gradient_queries': 2, 'head_backward_calls': 200, 'head_optimizer_attempts': 50,
            'CodeFormer_gradient_backward_optimizer_calls': 0, 'source_and_trace_bound_not_local_execution': True}
        states, result = checkpoint_audit(e, paths, parent_path, snapshots, torch, np)
        report['sections']['checkpoints'] = result
        report['sections']['prerequisites'] = prerequisite_audit(e, states[SLOTS[0]], torch, np)
        report['sections']['history'] = history_audit(e, states, torch, np)
        report['sections']['saved_scalar_acceptance'] = gate_audit(e)
        failure = e.json(PREFIX + 'failure.json')
        require(failure['complete'] is True and failure['failure'] is True and failure['completed_head_updates'] == 50 and
                failure['restorer_updates_this_run'] == 0 and failure['counts'] == e.p['call_caps'] and
                failure['partial_microbatches'] == 0 and failure['head_optimizer_inflight'] is False and
                failure['model_qualified'] is False and failure['app_promoted'] is False and failure['automatic_resume'] is False and
                not report['sections']['saved_scalar_acceptance']['all_acceptance_checks_pass'] and
                PREFIX + 'run_results.json' not in e.hashes, 'Authoritative failed50-update terminal state differs')
        resource = failure['resources']
        require(resource['peak_RSS_bytes'] <= e.p['limits']['process_RSS_bytes'] and
                resource['peak_CUDA_reserved_bytes'] <= e.p['limits']['CUDA_reserved_bytes'] and
                resource['minimum_sampled_root_free_bytes'] >= e.p['limits']['reserve_free_bytes'] and
                resource['disk_minimum_is_sampled_not_continuous'] is True and
                failure['seconds'] <= e.p['limits']['worker_seconds'] and
                failure['head_optimization_seconds'] <= e.p['limits']['head_optimization_seconds'], 'Reported resource stop bounds differ')
        close(failure['head_optimization_seconds'], report['sections']['history']['head_optimization_seconds'])
        for bank in ('original0', 'head0000', 'head0050'):
            actual = e.json(PREFIX + bank + '_metrics.json')
            cap = e.p['limits']['original_DEV_bank_seconds' if bank == 'original0' else 'each_controller_DEV_bank_seconds']
            require(actual['seconds'] <= cap, 'Reported bank time exceeds cap')
        cache_time = e.json(PREFIX + 'TRAIN_cache_timing.json')
        require(cache_time['complete'] is True and cache_time['cases'] == 200 and
                cache_time['seconds'] <= e.p['limits']['TRAIN_cache_seconds'], 'Reported TRAIN cache bounds differ')
        final = e.json(PREFIX + 'final_frozen_state.json')
        require(final['complete'] is True and final['CodeFormer_updates_this_run'] == 0 and
                final['digests'] == {'original': ORIGINAL_MODEL_DIGEST, 'restorer': PARENT_MODEL_DIGEST}, 'Final frozen digest receipt differs')
        report['sections']['terminal_and_resources'] = {'completed_head_updates': 50, 'restorer_updates_this_run': 0,
            'failure_reason': failure['reason'], 'resources_reported_by_VM': resource,
            'worker_seconds': failure['seconds'], 'head_optimization_seconds': failure['head_optimization_seconds'],
            'local_measured_VM_resources_claimed': False, 'all_reported_caps_satisfied': True,
            'partial_microbatches': 0, 'head_optimizer_inflight': False,
            'independent_pixel_visual_and_user_acceptance_still_required': True}
        report.update(complete=True, evidence_consistency_pass=True,
            authoritative_status='FAILED_ACCEPTANCE_GATES_GENUINE_TRAINED_RESEARCH_COMPOSITE_RETAINED',
            torch_CPU_weights_only_mmap_loads=4, audit_neural_instances=0, audit_neural_forwards=0,
            audit_autograd_queries=0, audit_backward_calls=0, audit_optimizer_constructions_or_steps=0,
            audit_seconds=time.monotonic() - started, evidence_binding=e.finish(),
            limitations=['Saved tensors, chronology and source/receipt checks support50 real head updates; no independent optimizer or full-gradient replay was performed.',
                'RNG states were compared and schema-checked, never restored or replayed through a network.',
                'Live module training modes and per-forward gradients are not reconstructible from state_dict tensors alone.',
                'Analytic derivatives use source-captured hidden activations; no local head/CodeFormer inference was run.',
                'This saved-scalar gate readback is not the separate independent RAW/PNG or root visual review.',
                'A genuine composite requires all515 base tensors,4 head tensors and the exact feature/head/clamp/padding policy; base-only export cannot reproduce it.'])
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error), audit_seconds=time.monotonic() - started)
        with (out / 'failure.json').open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2); stream.write('\n')
        raise
    with (out / 'state_audit_v1.json').open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2); stream.write('\n')
    text = ('# Independent saved-state readback\n\n'
        'The returned run contains genuine initial, head50 and stopped composites. All515 saved base tensors '
        'are exact copies of the frozen historical full-coarse parent. All four controller tensors changed; '
        'the four saved Adam states each reach50. The50 attempt/commit pairs,200 observations and3090 operation '
        'events agree. The stopped model/head/Adam/RNG/sampler/schedule equal head50.\n\n'
        'Both derivative fixtures were checked from all16 saved actual/expected gradient arrays and four RNG '
        'arrays using independent analytic arithmetic. No neural model, forward, autograd or optimizer was run locally.\n\n'
        'Acceptance failed. These are retained research composites, not a qualified model. Saved-scalar guards '
        'and reported VM resource caps were checked; independent pixel/visual review remains separate. '
        'RNG and Adam values were audited, not replayed through training.\n\n'
        'A faithful inference export requires `base_model` plus `head`, with `controller` and '
        '`composite_specification` and the exact shared input-feature implementation. Preserve optimizer/RNG '
        'in the original checkpoints; do not call a base-only params_ema file the trained controller.\n')
    with (out / 'STATE_FINDINGS.md').open('x', encoding='utf-8') as stream: stream.write(text)
    print(json.dumps({'complete': True, 'state_audit_sha256': sha(out / 'state_audit_v1.json'),
        'head_updates': 50, 'restorer_updates': 0, 'model_qualified': False}))


if __name__ == '__main__':
    main()
