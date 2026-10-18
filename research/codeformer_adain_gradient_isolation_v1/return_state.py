"""Manifest-bound CPU-only inspection of the returned AdaIN isolation pilot.

No module/model instance, neural forward, autograd query or optimizer is run.
Saved tensor states are loaded with weights_only=True. Gradient captures are
validated as source-bound readouts; they are not independent gradient replay.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import traceback

from research.codeformer_fp32_comparison_v1 import return_trajectory as old

PROTOCOL = '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
ARCHIVE = '7f7fc6b5a1fd58ea1ae5801d5f1edc95e66d11fe5dbc0256e1b4fe88ab7fa633'
MANIFEST = '0091134a5653825fb3a06d63e3907d1e57521f607e67da5d39ea856cdceff805'
MEMBER = 'outputs/adain_gradient_isolation_v1/export_manifest_20261016T170106Z-8446018b.json'
require, sha, read, safe, close = old.require, old.sha256, old.read_json, old.safe_file, old.close


class Evidence:
    def __init__(self, imported, packet, parent, legacy, receipt):
        self.imported, self.packet, self.parent, self.legacy = [Path(x).resolve(strict=True)
                                                              for x in (imported, packet, parent, legacy)]
        self.bindings = {}
        self.receipt = read(receipt)
        require(self.receipt.get('complete') is True and self.receipt.get('archive_sha256') == ARCHIVE and
                self.receipt.get('protocol_sha256') == PROTOCOL and self.receipt.get('manifest_sha256') == MANIFEST and
                self.receipt.get('exact_namespace_verified') is True and
                self.receipt.get('whole_archive_and_all_member_hashes_verified_before_import') is True and
                self.receipt.get('manifest_member') == MEMBER and
                Path(self.receipt.get('import_directory', '')).resolve(strict=True) == self.imported,
                'Exact successful transport receipt required')
        require(sha(safe(self.imported, MEMBER)) == MANIFEST, 'Returned manifest changed')
        self.manifest = read(safe(self.imported, MEMBER))
        require(self.manifest['protocol_sha256'] == PROTOCOL and self.manifest['complete'] is True,
                'Returned manifest identity differs')
        self.returned = dict(self.manifest['files_sha256'], **{MEMBER: MANIFEST})
        require(self.receipt['files_sha256'] == self.returned and self.receipt['members'] == len(self.returned),
                'Transport/manifest exact member map differs')
        require(sha(safe(self.packet, 'protocol.json')) == PROTOCOL, 'Frozen protocol changed')
        self.p = read(safe(self.packet, 'protocol.json'))
        self.files = self.p['files_sha256']
        self.bindings[str(safe(self.packet, 'protocol.json'))] = PROTOCOL
        for name in self.returned:
            self.get(name)
        require((self.imported / 'protocol.json').read_bytes() == (self.packet / 'protocol.json').read_bytes(),
                'Returned/frozen protocol bytes differ')
        for name in self.p['export_files']:
            require(self.returned[name] == self.files[name], 'Returned source differs from frozen protocol')

    def get(self, name):
        path = safe(self.imported, name)
        require(name in self.returned and sha(path) == self.returned[name], 'Unbound/changed returned file: ' + name)
        self.bindings[str(path)] = self.returned[name]
        return path

    def source(self, name):
        require(name in self.files, 'Source is not frozen: ' + name)
        for root in (self.packet, self.parent, self.legacy):
            if (root / name).exists():
                path = safe(root, name)
                require(sha(path) == self.files[name], 'Frozen source/dependency changed: ' + name)
                self.bindings[str(path)] = self.files[name]
                return path
        raise old.AuditError('Missing frozen source/dependency: ' + name)

    def close(self):
        for path, expected in self.bindings.items():
            require(sha(path) == expected, 'Evidence changed during CPU audit: ' + path)
        return {'individually_bound_regular_files': len(self.bindings),
                'returned_manifest_members': len(self.returned), 'all_read_evidence_rehashed_at_close': True,
                'manifest_sha256': MANIFEST, 'transport_receipt_sha256': sha(self.receipt_path)}


def events(p):
    counts = {k: 0 for k in ['optimizer_updates', 'microbatches', 'gradient_queries', 'backward_calls',
        'baseline_forwards', 'initializer_parity_candidate_forwards', 'snapshot_forwards', 'teacher_forwards',
        'candidate_training_forwards', 'perceptual_prerequisite_forwards', 'perceptual_training_forwards',
        'route_witness_forwards', 'route_witness_perceptual_forwards', 'behavior_teacher_forwards',
        'optimizer_step_attempts', 'training_input_writes']}
    rows, committed, pending = [], 0, 0
    def reserve(kind):
        counts[kind] += 1
        rows.append({'event': len(rows) + 1, 'before_call': kind, 'committed_update': committed,
                     'pending_microbatches': pending, 'counts': dict(counts)})
    reserve('perceptual_prerequisite_forwards'); reserve('gradient_queries')
    for _ in range(2):
        reserve('baseline_forwards'); reserve('initializer_parity_candidate_forwards')
    for _ in range(391): reserve('teacher_forwards')
    for _ in range(4):
        reserve('route_witness_perceptual_forwards'); reserve('route_witness_forwards')
        reserve('gradient_queries'); reserve('gradient_queries')
    for _ in range(406): reserve('snapshot_forwards')
    for update in range(1, 51):
        pending = 0
        for slot in range(1, 5):
            reserve('training_input_writes')
            if slot == 3: reserve('behavior_teacher_forwards')
            if slot >= 3 or (update - 1) % 4 >= 2: reserve('perceptual_training_forwards')
            reserve('candidate_training_forwards'); reserve('backward_calls')
            counts['microbatches'] += 1; pending += 1
        reserve('optimizer_step_attempts')
        committed = update; counts['optimizer_updates'] = update; pending = 0
        if update == 50:
            for _ in range(406): reserve('snapshot_forwards')
    require(len(rows) == 2073 and all(counts[k] == v for k, v in p['training']['call_caps'].items()),
            'Independent complete 2073-reservation plan differs from frozen caps')
    return rows, counts


def audit_calls(e, out, terminal):
    expected, counts = events(e.p)
    rows = sorted(out.joinpath('call_trace').glob('*.json'))
    require([p.name for p in rows] == ['%05d.json' % n for n in range(1, len(rows) + 1)],
            'Reservation trace is not an exact chronological prefix')
    previous = 0.
    for path, control in zip(rows, expected):
        row = read(path)
        require({key: row[key] for key in control} == control, 'Reservation/commit order differs: ' + path.name)
        require(old.finite(row['seconds']) and previous <= row['seconds'] <= e.p['limits']['worker_seconds'],
                'Reservation monotonic clock/cap differs')
        previous = row['seconds']
    require(len(rows) == len(expected) and terminal['counts'] == counts and terminal['completed_updates'] == 50,
            'Full-fifty returned reservation count/caps differ')
    if 'last_call_reservation' in terminal:
        require(terminal['last_call_reservation'] == read(rows[-1]), 'Terminal final reservation differs')
    return {'guarded_reservations': len(rows), 'actual_neural_forward_reservations': 1614,
            'gradient_query_reservations': 9, 'backward_reservations': 200,
            'optimizer_attempt_reservations': 50, 'counts': counts, 'exact_order': True,
            'last_reserved_seconds': previous, 'source_recorded_runtime_not_independent_live_instrumentation': True}


def gradients(summary, names):
    require(set(summary) == {'parameters', 'groups', 'full_gradient_vectors_exported',
            'norms_are_source_bound_VM_readouts_not_independent_gradient_replay'} and
            summary['full_gradient_vectors_exported'] is False and
            summary['norms_are_source_bound_VM_readouts_not_independent_gradient_replay'] is True and
            set(summary['parameters']) == set(names), 'Gradient capture schema/census differs')
    groups = {g: {'tensors': 0, 'none': 0, 'nonzero': 0, 'squared_l2': 0., 'maximum_absolute': 0.}
              for g in ('encoder', 'predictor', 'fusion')}
    for name, row in summary['parameters'].items():
        g = 'encoder' if name.startswith('encoder.') else 'fusion' if name.startswith('fuse_convs_dict.') else 'predictor'
        group = groups[g]; group['tensors'] += 1
        require(type(row['is_none']) is bool and type(row['nonzero']) is bool and old.finite(row['l2']) and
                old.finite(row['maximum_absolute']) and row['l2'] >= row['maximum_absolute'] >= 0,
                'Gradient finite norm/None metadata differs')
        if row['is_none']:
            require(row['sha256'] is None and not row['nonzero'] and row['l2'] == row['maximum_absolute'] == 0,
                    'None gradient is confused with a zero-valued connected gradient')
            group['none'] += 1
        else:
            require(row['dtype'] == 'float32' and old.re.fullmatch('[a-f0-9]{64}', row['sha256']) and
                    row['nonzero'] == (row['maximum_absolute'] > 0) and row['shape'] == list(names[name].shape),
                    'Connected gradient geometry/hash/nonzero metadata differs')
        group['nonzero'] += int(row['nonzero']); group['squared_l2'] += row['l2']**2
        group['maximum_absolute'] = max(group['maximum_absolute'], row['maximum_absolute'])
    for key, group in groups.items():
        group['l2'] = math.sqrt(group.pop('squared_l2'))
        for name, value in group.items():
            if isinstance(value, float): close(summary['groups'][key][name], value)
            else: require(summary['groups'][key][name] == value, 'Gradient partition census differs')
    return groups


def audit_witness(e, out, original, init):
    import numpy as np
    from PIL import Image
    directory = out / 'gradient_route_witness'
    witness = read(directory / 'witness.json')
    require(witness == init['gradient_route_witness'] and witness['complete'] is witness['pass'] is True and
            witness['optimizer_constructed'] is False and witness['optimizer_updates'] == witness['backward_calls'] == 0,
            'Successful before-Adam witness is missing/inconsistent')
    require(witness['source_state_sha256'] == witness['terminal_state_sha256'] == old.tensor_state_digest(original) and
            witness['model_state_unchanged'] is witness['CPU_CUDA_Python_NumPy_RNG_restored_exact'] is
            witness['parameter_grad_fields_all_none'] is witness['original_callable_restored'] is True and
            witness['frozen_before'] == witness['frozen_after'] == init['frozen'], 'Witness failed control restoration')
    require(0 < witness['seconds'] <= 180 and len(witness['forwards']) == 4 and len(witness['parity']) == 2,
            'Witness count/time differs')
    before, after = [read(directory / ('rng_' + tag + '.json')) for tag in ('before', 'restored')]
    require(before['python'] == after['python'] and before['numpy'] == after['numpy'] and
            set(before['torch']) == set(after['torch']) == {'CPU', 'CUDA0'}, 'Python/NumPy/single-L4 RNG restoration differs')
    for key in before['torch']:
        lhs, rhs = before['torch'][key], after['torch'][key]
        a, b = np.load(e.get(lhs['path']), allow_pickle=False), np.load(e.get(rhs['path']), allow_pickle=False)
        require(sha(e.get(lhs['path'])) == lhs['sha256'] and sha(e.get(rhs['path'])) == rhs['sha256'] and
                a.dtype == b.dtype == np.uint8 and a.ndim == b.ndim == 1 and a.size > 0 and np.array_equal(a, b),
                'Saved CPU/CUDA RNG arrays do not restore bit-exactly')
    trainable = {n: t for n, t in original.items() if n.split('.')[0] in old.TRAINABLE}
    require(len(trainable) == 350, 'Original trainable tensor census differs')
    groups, arrays, cases = [], {}, []
    expected_pairs = [(1, 'original'), (1, 'isolated'), (2, 'original'), (2, 'isolated')]
    selected = {row['slot']: row for row in read(e.source(e.p['training_parity']['path']))['cases']
                if row['update'] == 3}
    for row, (slot, mode) in zip(witness['forwards'], expected_pairs):
        folder = directory / ('update_0003_slot%d_%s' % (slot, mode))
        require(read(folder / 'record.json') == row and row['slot'] == slot and row['mode'] == mode and
                row['update'] == 3 and row['parameter_grad_fields_all_none'] is True,
                'Witness case/source order differs')
        require(set(row['files_sha256']) == {'native512.npy', 'raw256.npy', 'actual_tokens.npy', 'native.png'} and
                all(sha(folder / name) == digest for name, digest in row['files_sha256'].items()),
                'Exact witness artifact namespace/file hashes differ from row')
        require(row['reference_id'] == selected[slot]['reference_id'] and selected[slot]['image_eligible'] is True and
                row['input_and_clean_parity']['retained_input_sha256'] == selected[slot]['input_sha256'] and
                row['input_and_clean_parity']['update'] == 3 and row['input_and_clean_parity']['slot'] == slot,
                'Witness reference/exposure/eligibility differs from predeclared update3 slots1/2')
        require(row['input_and_clean_parity']['before_candidate_training_forward'] is True and
                row['input_and_clean_parity']['exact_input_float32'] is
                row['input_and_clean_parity']['exact_clean512_float32'] is True, 'Witness input parity differs')
        image, supervised = gradients(row['gradients']['image'], trainable), gradients(row['gradients']['CE_HQ'], trainable)
        require(image['predictor']['none'] == image['predictor']['tensors'] and image['fusion']['l2'] > 0 and
                supervised['encoder']['l2'] > 0 and supervised['predictor']['l2'] > 0,
                'Expected fusion image / encoder-predictor CE-HQ routes missing')
        if mode == 'isolated':
            require(image['encoder']['none'] == image['encoder']['tensors'], 'Isolated encoder image gradient still connected')
            capture = row['route']; calls = capture['calls']
            require(capture['policy'] == 'candidate-AdaIN-style-detach-only-v1' and
                    capture['original_callable_restored'] is True and len(calls) == 1,
                    'One restored AdaIN detach invocation required')
            call = calls[0]
            require(call == {'style_shape': [1, 256, 16, 16], 'content_requires_grad': False,
                'style_requires_grad_before': True, 'style_requires_grad_after': False,
                'dtype': 'torch.float32', 'device': 'cuda:0', 'detached_shares_storage': True,
                'shape_dtype_device_unchanged': True}, 'Detach witness changed geometry/storage/precision')
        else:
            require(image['encoder']['l2'] > 0, 'Original image-to-encoder path was not witnessed')
        require(row['dtypes']['logits_dtype'] == row['dtypes']['probabilities_dtype'] == 'torch.float32',
                'Witness FP32 path differs')
        views = {}
        for name, shape, dtype in [('native512', (512, 512, 3), np.float32),
                                   ('raw256', (256, 256, 3), np.float32), ('actual_tokens', (256,), np.int64)]:
            path = folder / (name + '.npy')
            require(sha(path) == row['files_sha256'][path.name], 'Witness artifact file digest differs')
            value = old.load_array(path, shape, dtype)
            require(name != 'actual_tokens' or value.min() >= 0 and value.max() < 1024, 'Witness token outside dictionary')
            views[name] = value
        with Image.open(folder / 'native.png') as image_file:
            require(image_file.mode == 'RGB' and image_file.size == (256, 256) and
                    np.array_equal(np.asarray(image_file), np.floor(np.clip(views['raw256'], 0, 1) * 255).astype(np.uint8)),
                    'Witness PNG differs from exact raw flooring')
        arrays[slot, mode] = views
        require(row['gradient_check']['pass'] is True, 'Runtime gradient gate failed')
        groups.append({'slot': slot, 'mode': mode, 'image': image, 'CE_HQ': supervised})
        cases.append({'slot': slot, 'mode': mode, 'terms': row['terms'], 'array_sha256': row['array_sha256']})
    for slot in (1, 2):
        for name in ('native512', 'raw256', 'actual_tokens'):
            require(np.array_equal(arrays[slot, 'original'][name], arrays[slot, 'isolated'][name]),
                    'Saved witness forward values differ')
        rows = [row for row in witness['forwards'] if row['slot'] == slot]
        require(rows[0]['terms'] == rows[1]['terms'] and rows[0]['array_sha256'] == rows[1]['array_sha256'],
                'Witness logits/latent/native/raw/token captures or loss terms differ')
    require(all(row['pass'] is True and row['native_raw_logits_latent_tokens_exact'] is True and
                row['all_loss_terms_exact'] is True and row['floating_metadata_tolerance'] == 0
                for row in witness['parity']), 'Fixed-state parity receipt differs')
    return {'forwards': 4, 'gradient_queries': 8, 'before_Adam_construction': True, 'seconds': witness['seconds'],
            'saved_native_raw_tokens_and_PNG_exact': True, 'loss_terms_and_logit_latent_digests_exact': True,
            'CPU_CUDA_Python_NumPy_RNG_saved_equality': True, 'partitions': groups,
            'all_module_modes_restored_by_frozen_source_finally_not_separately_exported': True,
            'full_gradient_logits_latent_arrays_not_exported': True,
            'gradient_norms_hashes_are_source_bound_not_independent_gradient_replay': True}


def audit_hq(e, out, original, init, references):
    import torch
    import numpy as np
    expected = read(e.source(e.p['hq_cache_parity']['path']))
    require(expected['case_count'] == 391 and len(expected['cases']) == 391, 'Prospective HQ reference census differs')
    controls = {row['reference_id']: row for row in expected['cases']}
    train = {row['id']: row for row in references if row['historical_role'] == 'train'}
    require(set(controls) == set(train), 'HQ cache includes non-TRAIN identity')
    codebook = original['quantize.embedding.weight']
    digest = old.tensor_state_digest({'embedding.weight': codebook})
    records = sorted(out.joinpath('HQ_cache').glob('*/record.json'))
    require({p.parent.name for p in records} == set(controls), 'Returned 391 HQ cache namespace differs')
    for path in records:
        row = read(path); expected = controls[path.parent.name]; ref = train[path.parent.name]
        require(all(row[name] == expected[name] for name in ('reference_id', 'role', 'source_path', 'source_sha256',
            'clean512_float32_sha256', 'teacher_state_sha256', 'restoration_codebook_state_sha256')) and
            row['source_path'] == ref['source_path'] and row['source_sha256'] == ref['source_sha256'] and
            row['teacher_state_sha256'] == init['frozen']['teacher'] and row['restoration_codebook_state_sha256'] == digest,
            'HQ source/role/clean/teacher/codebook provenance differs')
        indices = old.load_array(e.get(row['indices_path']), (1, 256), np.int64)
        features = old.load_array(e.get(row['features_path']), (1, 256, 16, 16), np.float32)
        require(sha(e.get(row['indices_path'])) == row['indices_sha256'] and
                sha(e.get(row['features_path'])) == row['features_sha256'] and indices.min() >= 0 and indices.max() < 1024,
                'HQ indices/features files/range differ')
        require(old.array_digest(indices) == expected['indices_array_sha256'] and
                old.array_digest(features) == expected['features_array_sha256'], 'HQ targets differ from retained FP32 arrays')
        lookup = codebook[torch.from_numpy(indices.reshape(-1))].reshape(1, 16, 16, 256).permute(0, 3, 1, 2)
        require(np.array_equal(lookup.contiguous().numpy(), features), 'HQ features differ from frozen codebook lookup')
        captured = row['exact_retained_FP32_target_parity']
        require(captured['pass'] is True and all(captured[k] == v for k, v in expected.items()),
                'Runtime HQ target parity receipt differs')
    return {'TRAIN_targets': 391, 'returned_indices_features_equal_retained_FP32_raw_array_hashes': True,
            'frozen_codebook_lookup_exact_CPU': True, 'clean_source_and_teacher_codebook_hashes_bound': True,
            'neural_HQ_encoder_not_reexecuted': True}


def prior_weight_comparison(e, out, original):
    """Direct CPU tensor comparison against the independently closed FP32 run."""
    import torch
    prior_receipt = Path('outputs/codeformer_fp32_comparison_v1_return_review/trajectory_v1/trajectory_audit.json').resolve(strict=True)
    expected_receipt_sha = '66fc747b0de090d398058d9a7504c12e9847afb5124277caa6e2aaf58759c231'
    require(sha(prior_receipt) == expected_receipt_sha, 'Retained prior trajectory audit changed')
    closed = read(prior_receipt)
    require(closed['evidence_consistency_pass'] is True and closed['completed_updates'] == 50,
            'Retained prior FP32 state was not independently closed')
    before_record = [row for row in closed['sections']['checkpoints']['states'] if row['path'].endswith('/stopped_checkpoint.pt')]
    require(len(before_record) == 1, 'One closed stopped FP32 checkpoint required')
    prior_path = safe(Path('outputs/codeformer_fp32_comparison_v1_return'), before_record[0]['path'])
    require(sha(prior_path) == before_record[0]['sha256'], 'Prior closed FP32 checkpoint bytes changed')
    e.bindings[str(prior_receipt)] = expected_receipt_sha; e.bindings[str(prior_path)] = before_record[0]['sha256']
    previous = torch.load(prior_path, map_location='cpu', weights_only=True)
    prior = previous['model']; del previous; gc.collect()
    require(old.tensor_state_digest(prior) == before_record[0]['model_state_sha256'], 'Prior FP32 tensor state changed')
    current_path = out / 'snapshot_0050/checkpoint.pt'
    current_state = torch.load(current_path, map_location='cpu', weights_only=True)
    current = current_state['model']; del current_state; gc.collect()
    groups = {name: {'tensors': 0, 'new_changed': 0, 'prior_changed': 0, 'new_vs_prior_changed': 0,
                     'new_delta_squared': 0., 'prior_delta_squared': 0., 'between_squared': 0., 'dot': 0.}
              for name in ('encoder', 'predictor', 'fusion', 'frozen')}
    for name, initial in original.items():
        group = ('encoder' if name.startswith('encoder.') else 'fusion' if name.startswith('fuse_convs_dict.') else
                 'frozen' if name.split('.')[0] in old.FROZEN else 'predictor')
        row = groups[group]; a, b = current[name], prior[name]
        require(a.shape == b.shape == initial.shape and a.dtype == b.dtype == initial.dtype and
                torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), 'Compared state geometry/finiteness differs')
        row['tensors'] += 1; row['new_changed'] += int(not torch.equal(a, initial))
        row['prior_changed'] += int(not torch.equal(b, initial)); row['new_vs_prior_changed'] += int(not torch.equal(a, b))
        da, db = a.double() - initial.double(), b.double() - initial.double()
        row['new_delta_squared'] += float(da.square().sum()); row['prior_delta_squared'] += float(db.square().sum())
        row['between_squared'] += float((da - db).square().sum()); row['dot'] += float((da * db).sum())
    for row in groups.values():
        x, y = math.sqrt(row.pop('new_delta_squared')), math.sqrt(row.pop('prior_delta_squared'))
        row.update(new_change_l2=x, prior_change_l2=y, new_vs_prior_l2=math.sqrt(row.pop('between_squared')),
                   new_vs_prior_delta_direction_cosine=row.pop('dot') / (x * y) if x * y else None,
                   new_change_l2_divided_by_prior=x / y if y else None)
    require(groups['frozen']['new_changed'] == groups['frozen']['prior_changed'] == 0,
            'Frozen weights changed across the compared pilots')
    del current, prior; gc.collect()
    return {'prior_closed_trajectory_sha256': expected_receipt_sha,
            'prior_checkpoint_sha256': before_record[0]['sha256'], 'same_initializer_and50updates': True,
            'partitions': groups, 'tensor_difference_is_not_a_causal_or_quality_claim': True}


def audit(imported, packet, parent, legacy, receipt, outputdir):
    import torch
    from research.codeformer_adaptation_v2 import data
    torch.set_num_threads(1)
    outputdir = Path(outputdir)
    require(not outputdir.exists(), 'Retain existing state audit')
    e = Evidence(imported, packet, parent, legacy, receipt); e.receipt_path = Path(receipt)
    p = e.p; out = e.imported / p['outputs']
    require(sha(old.__file__) == 'e92f49393f46c074b872e969c73127bd07f6cc78a80b1e76291ae6913ee529ed' and
            sha(data.__file__) == p['files_sha256']['research/codeformer_adaptation_v2/data.py'],
            'Reused CPU auditor/sampler differs from closed/frozen source')
    e.bindings[str(Path(old.__file__).resolve())] = sha(old.__file__)
    e.bindings[str(Path(data.__file__).resolve())] = sha(data.__file__)
    require(p['training']['max_updates'] == 50 and p['training']['snapshots'] == [0, 50] and
            p['training']['schedule_horizon_updates'] == p['training']['data_horizon_updates'] == 600,
            'Finite one-factor design changed')
    failure = read(out / 'failure.json') if (out / 'failure.json').exists() else None
    completed = read(out / 'run_results.json') if (out / 'run_results.json').exists() else None
    require(bool(failure) != bool(completed), 'Exactly one authoritative terminal state required')
    terminal = failure or completed
    require(terminal['protocol_sha256'] == PROTOCOL and terminal['completed_updates'] == 50,
            'Actual terminal pilot identity/count differs')
    init = read(out / 'initialization.json')
    parity = read(e.source(p['training_parity']['path']))
    references = read(e.source(p['data_manifest']))['references']
    rows, warnings = old.read_history(out / 'training.jsonl', failure)
    trajectory, states, records = old.audit_history(rows, references, p)
    require(len(rows) == terminal['completed_updates'], 'Committed trajectory differs from terminal state')
    for row, control in zip(rows, parity['updates']):
        require(row['group_lrs'] == control['group_lrs'] and row['TRAIN_selection'] == control['TRAIN_selection'],
                'Exact first-fifty LR/sampler controls changed')
    for record in records.values():
        capture = record['candidate_dtypes']['adain_gradient_route']
        require(capture['policy'] == 'candidate-AdaIN-style-detach-only-v1' and
                capture['original_callable_restored'] is True and len(capture['calls']) == 1,
                'Committed TRAIN forward lost the scoped AdaIN gradient route')
        call = capture['calls'][0]
        require(call == {'style_shape': [1, 256, 16, 16], 'content_requires_grad': False,
            'style_requires_grad_before': True, 'style_requires_grad_after': False,
            'dtype': 'torch.float32', 'device': 'cuda:0', 'detached_shares_storage': True,
            'shape_dtype_device_unchanged': True}, 'TRAIN route capture differs from the approved detach')
    report = {'format': 'codeformer-adain-isolation-independent-return-state-v1',
        'UTC': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': PROTOCOL, 'archive_sha256': ARCHIVE,
        'source_sha256': {str(Path(__file__)): sha(__file__), str(Path(old.__file__)): sha(old.__file__)},
        'completed_updates': 50, 'training_finished': bool(completed), 'stop_reason': terminal.get('error'),
        'model_qualified': False, 'resume_authorized': False, 'neural_instances': 0, 'neural_forwards': 0,
        'gradient_queries_by_audit': 0, 'optimizer_updates_by_audit': 0,
        'history_warnings': warnings, 'sections': {'trajectory': trajectory}, 'issues': []}
    def section(name, function):
        try: report['sections'][name] = function()
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
            report['issues'].append({'section': name, 'error_type': type(error).__name__, 'error': str(error)})
    section('call_ledger', lambda: audit_calls(e, out, terminal))
    # Resolve and hash every reused evidence array needed by the retained audit.
    for row in parity['cases']: e.source(row['input_path'])
    for row in parity['behavior_anchors']:
        e.source(row['delivered_path']); e.source(row['logits_path'])
    # Retained helpers accept one packet tree. All these arrays are in the
    # original FP32 overlay; no mutation, synthesis or local renderer is used.
    anchor_root = e.parent
    section('TRAIN_inputs_and_behavior', lambda: old.audit_inputs_anchors(e.imported, anchor_root, out,
        parity, records, 50, init['frozen']))
    original = torch.load(e.source(p['assets']['initializer']['path']), map_location='cpu', weights_only=True)['params_ema']
    require(len(original) == 515 and old.tensor_state_digest(original) == init['frozen']['behavior_teacher'],
            'Untouched CodeFormer initializer/census differs')
    section('route_witness', lambda: audit_witness(e, out, original, init))
    section('HQ_cache', lambda: audit_hq(e, out, original, init, references))
    teacher = torch.load(e.source(p['assets']['teacher']['path']), map_location='cpu', weights_only=True)['params_ema']
    require(old.tensor_state_digest(teacher) == init['frozen']['teacher'], 'Frozen HQ teacher state changed')
    del teacher; gc.collect()
    for key in ('perceptual_features', 'perceptual_calibration'): e.source(p['assets'][key]['path'])
    section('perceptual_state', lambda: old.audit_perceptual_state(e.legacy, p, init))
    def checkpoints():
        result = []
        paths = sorted(out.glob('snapshot_*/checkpoint.pt')) + ([out / 'stopped_checkpoint.pt'] if failure else [])
        require(len(paths) == (3 if failure else 2), 'Exact scheduled/stop checkpoint census differs')
        for path in paths:
            state = torch.load(path, map_location='cpu', weights_only=True)
            at = state['completed_updates']
            require(state['format'] == 'codeformer-adain-gradient-isolation-state-v1' and state['protocol_sha256'] == PROTOCOL,
                    'Actual AdaIN checkpoint binding differs')
            # Reuse exact prior schema/value checks after separately asserting
            # this experiment's identity; no tensor/state/optimizer is changed.
            inspected = dict(state, format='codeformer-fp32-comparison-state-v1', protocol_sha256=old.PROTOCOL_SHA256)
            require(at in states and (path.name == 'stopped_checkpoint.pt' and at == 50 or
                    path.parent.name == 'snapshot_%04d' % at and at in (0, 50)), 'Checkpoint schedule/path mismatch')
            value = old.check_checkpoint(inspected, original, p, states[at], at)
            value.update(path=path.relative_to(e.imported).as_posix(), sha256=sha(path), bytes=path.stat().st_size)
            require(value['frozen_state_sha256'] == init['frozen']['candidate_frozen'], 'Frozen candidate changed')
            require(path.stat().st_size <= (832 if state['optimizer']['state'] else 384) * 1024**2,
                    'Checkpoint exceeded prospective writer cap')
            if path.name == 'stopped_checkpoint.pt':
                require(failure['checkpoint']['sha256'] == value['sha256'] and failure['checkpoint']['bytes'] == value['bytes'] and
                        failure['checkpoint']['path'] == value['path'] and failure['optimizer_inflight'] is False,
                        'Terminal checkpoint receipt/optimizer safety differs')
            result.append(value)
            del state, inspected; gc.collect()
        at50 = [r for r in result if r['completed_updates'] == 50]
        require(len({r['model_Adam_RNG_sampler_schedule_sha256'] for r in at50}) == 1,
                'Snapshot50/stopped model/Adam/committed RNG/sampler differ')
        return {'states': result, 'snapshot50_stopped_state_exact': True,
                'untouched_initializer_equals_snapshot0_exact': True, 'trainable_tensors': 350, 'frozen_tensors': 165}
    section('checkpoints', checkpoints)
    section('prior_FP32_weight_comparison', lambda: prior_weight_comparison(e, out, original))
    del original; gc.collect()
    def timings():
        benchmark, run = read(out / 'benchmark20.json'), read(e.imported / 'outputs/supervision/run_receipt.json')
        require(benchmark['updates'] == 20 and benchmark['group_requirement_pass'] is True and
                set(benchmark['group_coverage']) == set(old.TRAINABLE) and all(
                row['finite_nonzero_grad'] and row['actual_change'] for row in benchmark['group_coverage'].values()),
                'Benchmark intended gradient/change coverage failed')
        close(benchmark['projected_training_seconds'], benchmark['seconds'] / 20 * 50)
        values = {'initialization': (init['seconds'], 600), 'witness': (init['gradient_route_witness']['seconds'], 180),
            'benchmark20': (benchmark['seconds'], 600), 'projected_training': (benchmark['projected_training_seconds'], 900),
            'optimization': (trajectory['optimization_seconds'], 900), 'worker': (terminal['worker_seconds'], 3000)}
        for at in (0, 50): values['snapshot%d' % at] = (read(out / ('snapshot_%04d/metrics.json' % at))['seconds'], 600)
        require(all(old.finite(v) and 0 <= v <= cap for v, cap in values.values()), 'Recorded timing exceeded finite cap')
        require(0 < benchmark['CUDA_peak_reserved_bytes'] <= p['limits']['CUDA_reserved_bytes'], 'GPU reserved memory exceeded cap')
        require(run['protocol_sha256'] == PROTOCOL and run['manual_tmux'] is True and run['execution_max_updates'] == 50 and
                run['training_process_exit_code'] == run['stages'][1]['exit_code'] and
                run['stages'][0]['exit_code'] == 0 and [s['stage'] for s in run['stages']] == ['--verify-transfer', '--run'] and
                run['no_automatic_retry_or_resume'] is True and run['model_qualified'] is False,
                'Supervisor classification/finite manual workflow differs')
        require(all(s['external_cap_seconds'] == cap and s['grace_seconds'] == grace and
                    0 <= s['log_bytes'] <= s['log_byte_cap'] == 2 * 1024**2
                    for s, cap, grace in zip(run['stages'], (300, 3000), (30, 180))),
                'Recorded external/log caps differ from prospective supervisor')
        require(terminal['worker_seconds'] <= run['stages'][1]['seconds'] <= 3180,
                'Supervisor duration differs from worker/cap')
        runtime, match = read(out / 'runtime.json'), read(out / 'runtime_parity.json')
        require(match['complete'] is True and all(runtime[k] == parity['runtime_expected'][k] == match['current_values'][k]
            for k in ('python', 'torch', 'numpy', 'Pillow', 'OpenCV', 'CUDA', 'device', 'capability')), 'Runtime parity changed')
        return {'durations': {k: {'seconds': v, 'cap_seconds': cap} for k, (v, cap) in values.items()},
                'supervised_worker_seconds': run['stages'][1]['seconds'],
                'source_supervisor_exit_code': run['training_process_exit_code'],
                'CUDA_peak_reserved_bytes': benchmark['CUDA_peak_reserved_bytes'], 'runtime_matches_prior_FP32': True}
    section('timing_runtime', timings)
    report['evidence_binding'] = e.close()
    report.update(complete=True, evidence_consistency_pass=not report['issues'],
        authoritative_status='Stopped quality experiment; retain failure, no added epochs or unchanged retry' if failure else
        'Finite pilot ended; preservation/usefulness require separate pixel and visual review',
        limitations=['Neural losses and gradients are saved source-bound captures, not independent neural replay.',
            'Module-mode restoration is source controlled; per-module before/after mode lists were not exported.',
            'Native/raw/token witness arrays were replayed; full logits, latent and gradient arrays were not exported.',
            'Quality metrics and user usefulness belong to the independent pixel and visual audits.'])
    outputdir.mkdir(parents=True)
    with (outputdir / 'state_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('imported', 'packet', 'parent', 'legacy', 'receipt', 'outputdir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(**vars(args))
    except BaseException as error:
        if not args.outputdir.exists(): args.outputdir.mkdir(parents=True)
        path = args.outputdir / 'audit_exception.json'
        if not path.exists(): path.write_text(json.dumps({'complete': False, 'error': str(error),
            'error_type': type(error).__name__, 'traceback': traceback.format_exc(), 'source_sha256': sha(__file__),
            'local_checker_failure_not_VM_training_failure': True, 'model_qualified': False}, indent=2) + '\n', encoding='utf-8')
        raise
    print(json.dumps({k: result[k] for k in ('complete', 'evidence_consistency_pass', 'completed_updates', 'stop_reason', 'issues')}))
    return 0 if result['evidence_consistency_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
