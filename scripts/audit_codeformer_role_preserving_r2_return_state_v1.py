"""Independent committed-state audit after strict return import; CPU values only.

Never execute returned Python, construct models/optimizers, forward a network,
query gradients or access a VM. Torch is imported only after hash closure of
the imported checkpoint files and their immutable parent dependency.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = '82fcd34cada1eed238ef65a341c22580d12146e3ee30c2ff3094312d3ba1c85a'
ARCHIVE = 'd47c88c8889b9bda66eeba03f498c7ab81173e4a75ffa2572a75049e73084689'
MANIFEST = '179ecfca4d35f27c24fe42aa9cbd0bdb05e0dfc9577ac199ea22c96c219522d9'
PREFIX = 'outputs/role_preserving_training_v1_r2/'
REVIEW = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_return_review'
DEST = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_return'
PREP = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_preparation'
TRAINABLE = ('encoder', 'position_emb', 'feat_emb', 'ft_layers', 'idx_pred_layer', 'fuse_convs_dict')
FROZEN = ('quantize', 'generator')


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def parse(payload):
    def unique(items):
        result = {}
        for name, value in items:
            need(name not in result, 'Duplicate JSON field')
            result[name] = value
        return result
    return json.loads(payload.decode('utf-8'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def read(path, cap=2 * 1024**2):
    need(Path(path).stat().st_size <= cap, 'Bounded metadata cap exceeded')
    return parse(Path(path).read_bytes())


def regular(path):
    path = Path(path).absolute()
    for ancestor in (path, *path.parents):
        if ancestor.exists() or ancestor.is_symlink():
            info = ancestor.lstat()
            need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                 'Linked/reparse evidence forbidden')
    info = path.stat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and path.resolve().is_relative_to(ROOT),
         'Unlinked regular workspace evidence required')
    return path


class Evidence:
    def __init__(self):
        self.receipt_path = regular(REVIEW / 'return_import_v1.json')
        self.receipt_sha = sha(self.receipt_path)
        imported = read(self.receipt_path, 8 * 1024**2)
        need(imported.get('complete') is True and imported['archive_sha256'] == ARCHIVE and
             imported['protocol_sha256'] == PROTOCOL, 'Strict completed exact R2 import required before state load')
        self.root = Path(imported['import_root']).resolve(strict=True)
        need(self.root == DEST.resolve(strict=True), 'Exact imported R2 namespace required')
        self.hashes, self.sizes = imported['files_sha256'], imported['files_bytes']
        need(set(self.hashes) == set(self.sizes) and len(self.hashes) == 6234,
             'Exact6234-member import closure required')
        self.read_pins = {}
        need(self.hashes[PREFIX + 'export_manifest.json'] == MANIFEST, 'Actual returned manifest differs')
        manifest = self.json(PREFIX + 'export_manifest.json')
        need(manifest['files_sha256'] == {k: v for k, v in self.hashes.items() if k != PREFIX + 'export_manifest.json'} and
             manifest['files_bytes'] == {k: v for k, v in self.sizes.items() if k != PREFIX + 'export_manifest.json'},
             'Actual manifest/import full maps differ')
        self.p = self.json('protocol.json')
        need(self.hashes['protocol.json'] == PROTOCOL and self.p['training']['max_updates'] == 50,
             'Exact finite R2 protocol required')
        for name, digest in self.p['files_sha256'].items():
            need(self.hashes[name] == digest and self.sizes[name] == self.p['files_bytes'][name],
                 'Returned exact frozen source/provenance differs')
        self.stage = read(PREP / 'stage_v1.json', 8 * 1024**2)
        need(self.stage['protocol_sha256'] == PROTOCOL, 'Exact local resolution receipt required')

    def get(self, name):
        relative = PurePosixPath(name)
        need(name in self.hashes and relative.as_posix() == name and not relative.is_absolute() and
             '..' not in relative.parts and '\\' not in name and ':' not in name, 'Unsafe/unbound evidence name')
        path = regular(self.root.joinpath(*relative.parts))
        need(path.resolve().is_relative_to(self.root), 'Imported evidence escaped root')
        before = path.stat()
        need(before.st_size == self.sizes[name] and sha(path) == self.hashes[name], 'Changed imported bytes: ' + name)
        after = path.stat()
        need((before.st_size, before.st_mtime_ns, before.st_ino) ==
             (after.st_size, after.st_mtime_ns, after.st_ino), 'Evidence changed during hashing')
        self.read_pins[name] = {'sha256': self.hashes[name], 'bytes': before.st_size,
            'mtime_ns': before.st_mtime_ns, 'ino': before.st_ino}
        return path

    def json(self, name):
        return read(self.get(name))

    def parent(self, name):
        expected = self.p['dependencies'][name]
        path = regular(Path(self.stage['local_dependency_resolution'][name]))
        need(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'],
             'Immutable parent checkpoint bytes differ')
        return path

    def close(self):
        need(sha(self.receipt_path) == self.receipt_sha, 'Import receipt changed during audit')
        for name, pin in self.read_pins.items():
            st = (self.root / name).stat()
            need((st.st_size, st.st_mtime_ns, st.st_ino) == (pin['bytes'], pin['mtime_ns'], pin['ino']),
                 'Evidence identity changed after read')
        return {'files_freshly_hashed_before_read': len(self.read_pins),
            'all_read_file_stat_identities_rechecked': True, 'import_receipt_sha256': self.receipt_sha}


def event_plan(p):
    counts = {key: 0 for key in p['call_caps']}
    rows, committed, pending = [], 0, 0
    def reserve(name):
        counts[name] += 1
        rows.append({'event': len(rows) + 1, 'before_call': name,
            'completed_training_updates': committed, 'partial_microbatches': pending,
            'optimizer_inflight': False, 'counts': dict(counts)})
    for _ in range(100): reserve('teacher_forwards')
    for _ in range(2):
        reserve('initializer_parity_parent_forwards'); reserve('initializer_parity_candidate_forwards')
    for role in ('normal', 'dark'):
        reserve('route_witness_behavior_teacher_forwards'); reserve('initializer_training_candidate_forwards')
        if role == 'dark': reserve('initializer_parity_perceptual_forwards')
    for role in ('normal', 'dark'):
        reserve('route_witness_forwards')
        if role == 'dark': reserve('route_witness_perceptual_forwards')
        reserve('gradient_queries'); reserve('gradient_queries')
        if role == 'dark': reserve('perceptual_prerequisite_forwards'); reserve('gradient_queries')
    initial_counts = dict(counts)
    for _ in range(406): reserve('snapshot_forwards')
    for update in range(1, 51):
        for slot in range(1, 5):
            reserve('training_input_writes')
            reserve('original_behavior_teacher_forwards' if slot in (1, 3) else 'parent_behavior_teacher_forwards')
            reserve('candidate_training_forwards')
            if slot == 4 or (slot == 2 and (update - 1) % 4 >= 2): reserve('perceptual_training_forwards')
            reserve('backward_calls'); pending += 1
        reserve('optimizer_step_attempts'); committed = update; pending = 0
    for _ in range(406): reserve('snapshot_forwards')
    need(counts == p['call_caps'] and len(rows) == 1854 and
         sum(v for k, v in counts.items() if k.endswith('_forwards')) == 1399,
         'Independent exact finite reservation census differs')
    return rows, counts, initial_counts


def audit_metadata(e):
    p = e.p; failure = e.json(PREFIX + 'failure.json')
    need(failure['complete'] is True and failure['failure'] is True and
         failure['completed_training_updates'] == failure['optimizer_steps_completed'] == 50 and
         failure['optimizer_constructions_completed'] == 1 and failure['partial_microbatches'] == 0 and
         failure['optimizer_inflight'] is False and failure['candidate_state_valid'] is True and
         failure['reason'] == 'ValueError: Preservation or dark-retention quality gate failed; actual trained50state retained' and
         failure['automatic_retry'] is False and failure['automatic_resume'] is False and
         failure['model_qualified'] is False, 'Actual trained50 quality stop differs')
    limits = p['limits']; resources = failure['resources']; phases = failure['phase_seconds']
    need(set(phases) == {'initialization', 'witness', 'training', 'snapshot_0000', 'snapshot_0050'} and
         all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in phases.values()),
         'Actual phase clock namespace/finite values differ')
    for name, value in phases.items():
        limit = limits['snapshot_seconds'] if name.startswith('snapshot_') else limits[name + '_seconds']
        need(value <= limit, 'Recorded phase time cap exceeded: ' + name)
    need(0 <= failure['training_seconds'] <= limits['training_seconds'] and
         0 <= failure['seconds'] <= limits['worker_seconds'] and
         resources['peak_RSS_bytes'] <= limits['RSS_bytes'] and
         resources['peak_CUDA_reserved_bytes'] <= limits['CUDA_reserved_bytes'] and
         resources['minimum_sampled_free_bytes'] >= limits['reserve_free_bytes'] and
         resources['disk_minimum_is_sampled_not_continuous'] is True,
         'Recorded worker/optimization/RSS/CUDA/disk limits exceeded')
    plan, counts, initial_counts = event_plan(p); previous = 0.
    for row in plan:
        actual = e.json(PREFIX + 'call_trace/%05d.json' % row['event'])
        need(all(actual[key] == value for key, value in row.items()) and
             actual['reservation_is_not_completion_evidence'] is True and
             previous <= actual['seconds'] <= p['limits']['worker_seconds'],
             'Exact ordered reservation chronology differs at %d' % row['event'])
        previous = actual['seconds']
    need(failure['counts'] == counts, 'Failure terminal counters differ')
    bundle = e.json(p['data_bundle_file']); cases = {row['id']: row for row in bundle['training']}
    need(len(cases) == 200, 'Exact200 chronological TRAIN cases required')
    retained_parity = read(e.parent(p['training_parity']['path']))
    need(len(retained_parity['updates']) == 50, 'Exact50 retained parent data updates required')
    history_profiles = Counter(); role_counts = Counter(); total_seconds = 0.; committed_digests = {}
    for update in range(1, 51):
        attempt = e.json(PREFIX + 'history/update_%04d_attempt.json' % update)
        commit = e.json(PREFIX + 'history/update_%04d_committed.json' % update)
        need(attempt['update'] == commit['update'] == commit['completed_training_updates'] == update and
             attempt['completed_before'] == update - 1 and attempt['optimizer_committed'] is False and
             commit['optimizer_committed'] is True and commit['partial_microbatches'] == 0 and
             commit['optimizer_inflight'] is False and len(attempt['observations']) == len(commit['observations']) == 4,
             'Actual attempt/committed update chronology differs')
        rates = {'encoder_token': 1e-5 * update / 50, 'fusion': 2e-5 * update / 50}
        need(attempt['group_lrs'] == commit['group_lrs'] == rates, 'Fresh50-update Adam warmup rates differ')
        need(total_seconds <= commit['training_seconds'] <= p['limits']['training_seconds'], 'Actual optimization clock differs')
        total_seconds = commit['training_seconds']; committed_digests[update] = commit['model_tensor_sha256']
        for slot, item in enumerate(attempt['observations'], 1):
            identifier = 'train_u%04d_slot%d' % (update, slot); source = cases[identifier]
            actual = e.json(PREFIX + 'TRAIN/' + identifier + '/record.json')
            need(actual == item and item['id'] == identifier and item['slot'] == slot and item['update'] == update and
                 item['role'] == source['role'] and item['reference_id'] == source['reference_id'] and
                 item['sourcebindings'] == source['sourcebindings'] and
                 item['image_eligible'] == source['image_eligible'], 'TRAIN record/source/input/role join differs')
            role = item['role']; role_counts[role] += 1; history_profiles[source['profile']] += 1
            expected_terms = list(p['objective']['normal_terms']) if role == 'normal' else list(
                p['objective']['dark_eligible_terms' if item['image_eligible'] else 'dark_ineligible_terms'])
            need(list(item['terms']) == expected_terms and item['teacher_kind'] ==
                 ('original' if role == 'normal' else 'full_coarse_parent') and
                 item['actual_accumulation_divisor'] == 4 and item['normal_clean_supervision_called'] is False and
                 item['record_captured_before_optimizer_step'] is True and item['optimizer_committed_at_record_time'] is False,
                 'Role objective or pre-step provenance differs')
            need(all(type(value) in (float, int) and math.isfinite(value) for value in item['terms'].values()),
                 'Nonfinite saved training objective')
            total = sum(p['objective']['weights'][key] * value for key, value in item['terms'].items())
            bound = 32 * 2**-23 * max(1., sum(abs(p['objective']['weights'][key] * value) for key, value in item['terms'].items()))
            need(abs(item['total_loss'] - total) <= bound, 'FP32 objective sum is inconsistent with saved components')
            need(commit['observations'][slot - 1] == {key: item[key] for key in ('id', 'role', 'terms', 'total_loss')},
                 'Committed observations differ from actual pre-step records')
            route = item['route']; need(route['policy'] == 'candidate-AdaIN-style-detach-only-v1' and
                route['original_callable_restored'] is True and len(route['calls']) == 1,
                'Actual scoped candidate route metadata differs')
            call = route['calls'][0]
            need(call['style_shape'] == [1, 256, 16, 16] and call['style_requires_grad_before'] is True and
                 call['style_requires_grad_after'] is False and call['content_requires_grad'] is False and
                 call['detached_shares_storage'] is True and call['shape_dtype_device_unchanged'] is True and
                 call['dtype'] == 'torch.float32' and call['device'] == 'cuda:0', 'Actual AdaIN detach witness differs')
        selection = commit['retained_data_selection']
        selections = {row['slot']: row['reference_id'] for row in bundle['training'] if row['update'] == update}
        need(selection['update'] == update and selection['id_a'] == selections[1] == selections[2] and
             selection['id_b'] == selections[3] == selections[4] and
             selection == retained_parity['updates'][update - 1]['TRAIN_selection'],
             'Retained exact complete data-selection chronology differs')
    need(role_counts == {'normal': 100, 'dark': 100}, 'Normal/dark actual objective coverage differs')
    need(total_seconds == failure['training_seconds'], 'Last committed/terminal actual optimization clock differs')
    witness = e.json(PREFIX + 'gradient_route_witness/report.json')
    need(witness['complete'] is True and witness['pass'] is True and witness['candidate_forwards'] == 2 and
         witness['component_gradient_queries'] == 4 and witness['all_gradient_queries_with_perceptual'] == 5 and
         witness['CPU_CUDA_Python_NumPy_RNG_restored'] is True and witness['all_parameter_grad_None'] is True and
         witness['optimizer_constructed'] is False and len(witness['cases']) == 2, 'Actual before-Adam witness differs')
    for row in witness['cases']:
        need(row['unscoped_full_role_forward_parity_exact'] is True and row['optimizer_constructed'] is False and
             row['connectivity']['pass'] is True and row['connectivity']['queried_tensors'] == 350,
             'Scoped/unscoped complete role witness differs')
        img = row['partitions']['image']['groups']; token = row['partitions']['token_HQ']['groups']
        need(img['encoder']['none'] == img['encoder']['tensors'] and img['predictor']['none'] == img['predictor']['tensors'] and
             img['fusion']['l2'] > 0 and token['encoder']['l2'] > 0 and token['predictor']['l2'] > 0,
             'Actual image-isolation/token-learning partition differs')
    coverage = e.json(PREFIX + 'gradient_coverage.json')
    need(coverage['complete'] is True and coverage['350_trainable_tensors'] is True and coverage['50_actual_Adam_updates'] is True and
         len(coverage['tensors']) == 350 and all(row['gradient_observations'] == 50 and
             0 <= row['nonzero_observations'] <= 50 and math.isfinite(row['max_abs']) for row in coverage['tensors'].values()),
         'Actual350-tensor50-update gradient coverage differs')
    initial_frozen = e.json(PREFIX + 'initial_frozen_state.json')
    final_frozen = e.json(PREFIX + 'final_frozen_state.json')
    need(final_frozen['complete'] is True and final_frozen['same_as_initial'] is True and
         final_frozen['digests'] == initial_frozen, 'Recorded frozen teachers/generator/codebook changed')
    guards = e.json(PREFIX + 'preservation_guards.json'); dark = e.json(PREFIX + 'dark_retention.json')
    need(not (guards['pass'] and dark['pass']), 'Actual authoritative quality failure not present')
    alias = e.json(PREFIX + 'stopped_checkpoint_alias.json')
    need(alias['complete'] is True and alias['completed_training_updates'] == 50 and alias['partial_microbatches'] == 0 and
         alias['optimizer_inflight'] is False and alias['same_committed_state'] is True and
         alias['committed_state_digest_equal'] is True and alias['duplicate_checkpoint_written'] is False and
         alias['reason'] == 'quality_gate_failure_after_actual50', 'Actual stopped alias differs')
    return {'failure': failure, 'counts': counts, 'initial_counts': initial_counts, 'alias': alias,
        'initial_frozen': initial_frozen, 'committed_digests': committed_digests,
        'summary': {'committed_updates': 50, 'attempt_records': 50, 'commit_records': 50, 'TRAIN_records': 200,
            'role_counts': dict(role_counts), 'profiles': dict(history_profiles), 'ordered_reservations_verified': 1854,
            'counter_claim_is_reservations_not_live_profiler': True, 'normal_dark_gradient_witness_source_bound': True,
            'fresh_optimizer_warmup_rates_verified': True, 'normal_clean_supervision_absent_by_saved_source_records': True,
            'gradient_coverage_tensors': 350, 'all350_have50_gradient_observations': True,
            'actual_optimization_seconds': total_seconds, 'recorded_frozen_digests_unchanged': True,
            'recorded_worker_seconds': failure['seconds'], 'recorded_phase_seconds': phases,
            'recorded_resources_within_prospective_caps': resources,
            'preservation_pass': guards['pass'], 'dark_retention_pass': dark['pass'],
            'full_RAW_PNG_metric_recomputation_scope': 'Separate independent pixel audit; state audit reads authoritative gate records only.'}}


def tensor_digest(state, torch):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode('utf-8')
        h.update(len(header).to_bytes(8, 'big')); h.update(header)
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def tree_digest(value, torch):
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            header = json.dumps(['tensor', str(array.dtype), list(array.shape)], separators=(',', ':')).encode()
            h.update(len(header).to_bytes(8, 'big')); h.update(header)
            h.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b'dict[')
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))): visit(key); visit(item[key])
            h.update(b']')
        elif isinstance(item, (tuple, list)):
            h.update((type(item).__name__ + '[').encode())
            for child in item: visit(child)
            h.update(b']')
        else:
            payload = json.dumps([type(item).__name__, item], separators=(',', ':'), allow_nan=False).encode()
            h.update(len(payload).to_bytes(8, 'big')); h.update(payload)
    visit(value)
    return h.hexdigest()


def state_audit(value, parent, update, e, metadata, torch):
    p = e.p
    need(value['format'] == 'codeformer-role-preserving-training-state-v1' and value['protocol_sha256'] == PROTOCOL and
         value['completed_training_updates'] == update and value['historical_parent_updates'] == 50 and
         value['partial_microbatches'] == value['partial_microbatches_discarded'] == 0 and
         value['optimizer_inflight'] is False and value['state_safe'] is True and value['resume_safe'] is True and
         value['automatic_resume_supported'] is False and value['actual_rng_equals_committed_rng'] is True and
         value['objective'] == p['objective'] and value['model_qualified'] is False and value['EMA'] is False,
         'Saved committed-state schema/safety differs')
    need(value['parent_checkpoint_sha256'] == p['assets']['full_coarse_checkpoint']['sha256'] and
         value['original_checkpoint_sha256'] == p['assets']['initializer']['sha256'], 'Checkpoint lineage differs')
    model = value['model']; need(list(model) == list(parent) and len(model) == 515, '515state key/order mismatch')
    groups = defaultdict(lambda: {'tensors': 0, 'changed': 0, 'maximum_absolute_delta': 0., 'squared_l2_delta': 0.})
    for name, tensor in model.items():
        previous = parent[name]; group = name.split('.')[0]
        need(group in TRAINABLE + FROZEN and isinstance(tensor, torch.Tensor) and
             tensor.device.type == previous.device.type == 'cpu' and tensor.dtype == previous.dtype == torch.float32 and
             tensor.shape == previous.shape and torch.isfinite(tensor).all().item(), 'State tensor geometry/finiteness mismatch')
        exact = torch.equal(tensor, previous)
        need((group not in FROZEN or exact) and (update != 0 or exact), 'Initial or frozen model tensor changed')
        row = groups[group]; row['tensors'] += 1; row['changed'] += int(not exact)
        if not exact:
            delta = tensor.double() - previous.double()
            row['maximum_absolute_delta'] = max(row['maximum_absolute_delta'], float(delta.abs().max()))
            row['squared_l2_delta'] += float(delta.square().sum())
    need(sum(row['tensors'] for name, row in groups.items() if name in TRAINABLE) == 350 and
         sum(row['tensors'] for name, row in groups.items() if name in FROZEN) == 165 and
         sum(row['changed'] for name, row in groups.items() if name in TRAINABLE) == (350 if update else 0),
         'Actual350trainable/165frozen change census differs')
    model_sha = tensor_digest(model, torch)
    need(model_sha == value['candidate_tensor_digest'] and (update == 0 or
         model_sha == metadata['committed_digests'][50]), 'Actual model tensor digest differs')
    frozen_sha = tensor_digest({name: tensor for name, tensor in model.items() if name.split('.')[0] in FROZEN}, torch)
    need(frozen_sha == metadata['initial_frozen']['candidate_frozen'] and value['frozen_state'] == metadata['initial_frozen'],
         'Actual frozen165 tensors or teacher digest receipt differs')
    optimizer = value['optimizer']; need(set(optimizer) == {'state', 'param_groups'}, 'Adam saved schema differs')
    need([row['name'] for row in optimizer['param_groups']] == ['encoder_token', 'fusion'], 'Adam group order differs')
    identifiers = []; step_histogram = Counter(); moment_count = 0
    for group in optimizer['param_groups']:
        fusion = group['name'] == 'fusion'; base = 2e-5 if fusion else 1e-5
        names = [name for name in model if name.split('.')[0] in TRAINABLE and name.startswith('fuse_convs_dict.') == fusion]
        need(len(group['params']) == len(names) == (72 if fusion else 278), 'Saved Adam group census differs')
        need(group['base_lr'] == group['lr'] == base and list(group['betas']) == [.9, .99] and
             group['eps'] == 1e-8 and group['weight_decay'] == 0., 'Actual Adam rates/hyperparameters differ')
        identifiers += group['params']
        for identifier, name in zip(group['params'], names):
            entry = optimizer['state'].get(identifier)
            if not update:
                need(entry is None, 'Initial Adam state unexpectedly populated'); continue
            need(isinstance(entry, dict) and set(entry) == {'step', 'exp_avg', 'exp_avg_sq'} and
                 isinstance(entry['step'], torch.Tensor) and float(entry['step']) == 50,
                 'Actual saved Adam step differs')
            step_histogram[int(float(entry['step']))] += 1
            for key in ('exp_avg', 'exp_avg_sq'):
                moment = entry[key]
                need(moment.device.type == 'cpu' and moment.dtype == torch.float32 and moment.shape == model[name].shape and
                     torch.isfinite(moment).all().item(), 'Adam moment finite geometry differs')
                moment_count += 1
            need((entry['exp_avg_sq'] >= 0).all().item(), 'Negative Adam second moment')
    need(len(identifiers) == len(set(identifiers)) == 350 and
         set(optimizer['state']) == (set(identifiers) if update else set()), 'Adam duplicate/frozen/missing ID')
    sampler = {'next_update': update + 1, 'data_bundle_sha256': p['files_sha256'][p['data_bundle_file']],
        'retained_training_parity_sha256': p['training_parity']['sha256'],
        'policy': 'fixed chronological first50 retained TRAIN updates; no random sampler is advanced',
        'completed_data_updates': update, 'historical_sampler_state_installed': False}
    need(value['sampler'] == sampler and value['schedule'] == {'completed_updates': update, 'training': p['training']},
         'Actual committed sampler/schedule differs')
    need(value['counts'] == metadata['counts' if update else 'initial_counts'], 'Saved checkpoint operation counts differ')
    rng = value['rng']; need(set(rng) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'} and
         rng['python'][0] == 3 and len(rng['python'][1]) == 625 and rng['numpy']['algorithm'] == 'MT19937' and
         len(rng['numpy']['keys']) == 624 and len(rng['torch_cuda']) == 1, 'Saved RNG schema differs')
    for tensor in [rng['torch_cpu'], *rng['torch_cuda']]:
        need(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu' and tensor.dtype == torch.uint8 and
             tensor.ndim == 1 and tensor.numel() > 0, 'Saved CPU/CUDA RNG state geometry differs')
    need(tree_digest(rng, torch) == tree_digest(value['actual_rng_at_capture'], torch), 'Actual/committed saved RNG differs')
    committed_keys = ('model', 'optimizer', 'rng', 'sampler', 'schedule', 'counts',
        'completed_training_updates', 'partial_microbatches', 'optimizer_inflight')
    committed_sha = tree_digest({key: value[key] for key in committed_keys}, torch)
    need(committed_sha == value['committed_state_digest'], 'Complete model/Adam/RNG/sampler/schedule/counters digest differs')
    if update:
        alias = metadata['alias']
        need(committed_sha == alias['committed_state_digest'] and model_sha == alias['current_model_tensor_sha256'],
             'Exact retained alias/state digest differs')
    for row in groups.values(): row['delta_l2'] = math.sqrt(row.pop('squared_l2_delta'))
    return {'completed_updates': update, 'model_tensor_sha256': model_sha, 'frozen165_tensor_sha256': frozen_sha,
        'groups': dict(groups), 'Adam_saved_states': len(optimizer['state']), 'Adam_moment_tensors': moment_count,
        'Adam_step_histogram': dict(step_histogram), 'committed_state_digest': committed_sha,
        'optimizer_digest': tree_digest(optimizer, torch), 'RNG_digest': tree_digest(rng, torch),
        'sampler': sampler, 'counts': value['counts'], 'state_safe': True,
        'resume_safe_recorded_not_authorized': True, 'model_qualified': False}


def main():
    started = time.monotonic(); destination = REVIEW / 'state_v1/state_audit_v1.json'
    need(not destination.exists(), 'Preserve previous state audit')
    e = Evidence(); metadata = audit_metadata(e)
    checkpoint_names = [PREFIX + 'snapshot_0000/checkpoint.pt', PREFIX + 'snapshot_0050/checkpoint.pt']
    paths = [e.get(name) for name in checkpoint_names]
    parent_path = e.parent(e.p['assets']['full_coarse_checkpoint']['path'])
    need(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')), 'Framework loaded before checkpoint hash closure')
    import torch
    torch.set_num_threads(2)
    parent_container = torch.load(parent_path, map_location='cpu', weights_only=True, mmap=True)
    need(parent_container['protocol_sha256'] == e.p['parent_protocol_sha256'] and
         parent_container['completed_updates'] == 50, 'Immutable parent checkpoint metadata differs')
    parent = parent_container['model']
    need(tensor_digest(parent, torch) == e.p['model_digests']['base'], 'Actual immutable parent515 digest differs')
    reports = {}
    for label, path, update in zip(('initial', 'update50'), paths, (0, 50)):
        value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        reports[label] = state_audit(value, parent, update, e, metadata, torch)
        cp = metadata['failure']['checkpoints']['snapshot_%04d/checkpoint.pt' % update]
        need(cp['sha256'] == e.hashes[checkpoint_names[0 if update == 0 else 1]] and
             cp['model_tensor_sha256'] == reports[label]['model_tensor_sha256'] and
             cp['committed_state_digest'] == reports[label]['committed_state_digest'], 'Failure/checkpoint pointer differs')
        print(json.dumps({'checkpoint': label, 'complete': True, 'state_tensors': 515,
            'Adam_states': reports[label]['Adam_saved_states'], 'model_sha256': reports[label]['model_tensor_sha256']}), flush=True)
        del value
    alias = metadata['alias']
    need(alias['source_checkpoint'] == checkpoint_names[1] and
         alias['source_sha256'] == e.hashes[checkpoint_names[1]] and
         alias['source_bytes'] == e.sizes[checkpoint_names[1]], 'Actual stopped alias fingerprint differs')
    result = {'format': 'codeformer-role-preserving-r2-independent-state-audit-v1', 'complete': True,
        'UTC': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': PROTOCOL, 'archive_sha256': ARCHIVE,
        'source_sha256': sha(__file__), 'import_root': str(e.root), 'metadata': metadata['summary'],
        'checkpoint_readback': reports, 'initial515_exact_actual_parent': True,
        'all350_trainable_tensors_changed': True, 'all165_frozen_tensors_exact': True,
        'saved_checkpoint50_and_stopped_alias_exact_committed_state': True,
        'checkpoint_load_policy': {'map_location': 'cpu', 'weights_only': True, 'mmap': True,
            'returned_code_executed': False, 'model_instances': 0, 'optimizer_constructions': 0,
            'neural_forwards': 0, 'autograd_queries': 0, 'optimizer_steps': 0, 'VM_calls': 0},
        'evidence_closure': e.close(), 'seconds': time.monotonic() - started,
        'authoritative_outcome': '50 committed new restorer updates and a genuine retained checkpoint; quality gate failure remains research-only.',
        'limitations': ['Saved values/history/source joins establish consistent committed state, not independent replay of all live CUDA gradients or Adam execution.',
            'Frozen teacher digests and live gradient-route readouts remain source-bound; serialized candidate/frozen165 tensors and Adam moments were independently inspected.',
            'Restoration metrics and all planned visual sheets are audited separately. No final review, user acceptance, identity claim or app qualification follows.'],
        'model_qualified': False, 'overall_goal_complete': False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'complete': True, 'receipt': str(destination), 'sha256': sha(destination),
        'seconds': result['seconds'], 'new_committed_updates': 50, 'changed_tensors': 350, 'frozen_exact': 165,
        'model_qualified': False}), flush=True)


if __name__ == '__main__':
    main()
