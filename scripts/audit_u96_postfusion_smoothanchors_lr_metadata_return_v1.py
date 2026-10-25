"""Independent saved-JSON/hash readback; no model, Torch, archive extraction or VM.

Consumes this agent's already bounded/CRC-checked single-stream archive inventory.
Numerical quality decisions below are reconstructed from saved scalar metrics;
the separate saved-pixel and state auditors establish their underlying arrays.
"""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_peer'
O = 'outputs/u96_postfusion_smoothanchors_lr_v1/'
SHA = '8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9'
SNAPS = [0, 1, 4, 16, 32, 64]


def pin(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
            'sha256': h.hexdigest()}


def seq(values):
    result = 0.0
    for value in values:
        result += float(value)
    return result


def main():
    checks = 0
    def need(ok, message):
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(message)
    b = json.loads((OUT / 'bounded_stream_readback_v1.json').read_text())
    m, d = b['metadata'], b['documents']
    p = json.loads((STAGE / 'protocol.json').read_text())
    need(pin(STAGE / 'protocol.json')['sha256'] == SHA, 'Exact sealed protocol')
    lim, learn = p['limits'], p['learning']
    files = p['files']
    for name, row in files.items():
        local = pin(STAGE / name)
        need(local['bytes'] == row['bytes'] and local['sha256'] == row['sha256'], 'Local sealed file ' + name)
        if not name.startswith('weights/'):
            need(m[name] == dict(path=name, **row), 'Returned static pin ' + name)
    need(set(name for name in files if name.startswith('weights/')) ==
         {v['path'] for v in p['model'].values()}, 'Exactly two retained initializer weights omitted')
    need(not any(name.startswith('weights/') for name in m), 'No returned initializer weights')

    # Exact full-success namespace, rather than accepting arbitrary owned output paths.
    names = set(name for name in files if not name.startswith('weights/')) | {'protocol.json'}
    names |= {'install.json', 'run.log', 'supervisor.json'}
    names |= {O + name for name in ('controlled_hook_exception.json', 'events.jsonl', 'initial_heads.npz',
        'initial_heads.pt', 'preflight.json', 'reference_bank.json', 'resources.jsonl', 'result.json',
        'sampler.json', 'started.json', 'state_invariants.json', 'status.json')}
    for u in range(65):
        names |= {O + f'checkpoints/u{u:04d}/' + tail for tail in ('heads.npz', 'resume.pt', 'checkpoint.json')}
    reference_rows = p['cases'] + p['guards']
    first4 = {learn['optimization_reference_order'][offset] + '__' + role
              for offset, role in zip(learn['sampler_offsets'], learn['slot_roles'])}
    selected = [r for r in p['cases'] if r['cohort'] == 'monitor'] + p['guards']
    for row in reference_rows:
        names |= {O + 'reference/' + row['id'] + '/' + tail for tail in ('raw.npy', 'unclipped.npy', 'output.png', 'record.json')}
        if row['id'] in first4:
            names |= {O + 'reference/' + row['id'] + '/' + tail for tail in ('native512.npy', 'logits.npy', 'LQ.npy')}
    for u in SNAPS:
        names.add(O + f'snapshots/u{u:04d}/bank_and_gates.json')
        for row in selected:
            names |= {O + f'snapshots/u{u:04d}/' + row['id'] + '/' + tail for tail in
                      ('raw.npy', 'unclipped.npy', 'output.png', 'record.json')}
    for u in range(1, 65):
        names |= {O + f'updates/u{u:04d}/' + tail for tail in ('commit.json', 'postclip_gradients.npz')}
        for slot in range(1, 5):
            names |= {O + f'training/u{u:04d}/slot{slot}/' + tail for tail in ('raw.npy', 'output.png', 'record.json', 'objective.json')}
            if u == 1:
                names |= {O + f'training/u{u:04d}/slot{slot}/' + tail for tail in ('unclipped.npy', 'native512.npy', 'logits.npy', 'LQ.npy')}
    need(names == set(m), 'Complete exact 3469-member full-success namespace')
    need(len(m) == 3469 and b['payload_bytes'] == 1311565858, 'Exact complete payload counts')

    # Every returned runtime direct file pin is checked against the single-stream digest.
    pin_joins = 0
    def join(value):
        nonlocal pin_joins
        if isinstance(value, dict):
            if {'path', 'bytes', 'sha256'} <= value.keys():
                path = value['path']
                wanted = {k: value[k] for k in ('path', 'bytes', 'sha256')}
                if path in m:
                    need(m[path] == wanted, 'Runtime direct pin ' + path)
                elif path.startswith('weights/'):
                    need(wanted in p['model'].values(), 'Copied weight receipt matches fixed initializer')
                else:
                    raise AssertionError('Unbound returned runtime direct pin ' + path)
                pin_joins += 1
            for child in value.values():
                join(child)
        elif isinstance(value, list):
            for child in value:
                join(child)
    for name, value in d.items():
        if name.startswith(O) or name in ('install.json', 'supervisor.json'):
            join(value)

    result, status, sup = d[O + 'result.json'], d[O + 'status.json'], d['supervisor.json']
    pre, state = d[O + 'preflight.json'], d[O + 'state_invariants.json']
    need(result['complete'] and result['status'] == 'FINITE_B64_LEARNING_SCREEN_PASSED_NOT_QUALIFIED', 'Terminal finite TRAIN success status')
    need(not result['model_qualified'] and result['full_DEV_native_FINAL_user_app_qualification_pending'], 'Qualification explicitly pending')
    need(sup['worker_exit'] == 0 and not sup['timeout'] and not sup['optimizer_outcome_unknown'], 'Authoritative supervisor normal exit')
    need(sup['confirmed_optimizer_updates'] == sup['optimizer_update_attempts'] == result['optimizer_updates'] == 64, '64 known commits/attempts')
    need(not status['optimizer_outcome_unknown'] and O + 'failure.json' not in m and O + 'status.pending' not in m, 'No failure/pending/unknown evidence')
    need(pre['GPU'] == 'NVIDIA L4' and pre['host']['system'] == 'Linux' and pre['host']['tmux'], 'Exact L4 Linux manual tmux receipt')
    need(pre['limits'] == lim and pre['seed'] == 20261021 and pre['optimizer_updates'] == 0, 'Original preflight contract')
    need(all(d[name]['protocol_sha256'] == SHA for name in ('install.json', 'supervisor.json', O + 'result.json', O + 'status.json', O + 'preflight.json', O + 'started.json')), 'All protocol joins')
    final_counters = result['counters']
    need(final_counters == status['counters'] == state['counters'], 'Final result/status/state exact counters')
    for key, maximum in lim.items():
        if key in final_counters:
            need(final_counters[key] == maximum, 'Exact final sealed counter ' + key)
    need(final_counters['gradient_queries'] == 0 and final_counters['optimizer_construction_returns'] == 1, 'No extra gradient queries, one constructed optimizer')
    need(final_counters['frozen_U96_adapter_site_returns'] == final_counters['frozen_U96_adapter_site_reservations'] == 712 * 4, 'All frozen adapter site returns')
    need(final_counters['new_head_site_returns'] == final_counters['new_head_site_reservations'] == (336 + 256) * 4, 'All new-head site returns')
    need(state['before']['reference'] == state['after']['reference'], 'Reference canonical state claims unchanged')
    need(state['caller_RNG_before'] == state['caller_RNG_after'], 'Caller RNG receipt exact')
    need(state['all36_head_tensors_only_trainable'] and state['all539_incumbent_tensors_frozen_exact'], 'Trainable/frozen census claims')
    for which in ('before', 'after'):
        candidate = state[which]['candidate']
        need(candidate['frozen_tensors'] == candidate['frozen_grad_fields_None'] == 539 and candidate['full_state_tensors'] == 575 and candidate['head_tensors'] == 36 and candidate['head_parameters'] == 168960, '575/539/36 census ' + which)
        need(candidate['R2_canonical515_sha256'] == pre['initial_states']['candidate']['R2_canonical515_sha256'] and candidate['U96_canonical24_sha256'] == pre['initial_states']['candidate']['U96_canonical24_sha256'], 'Incumbent canonical claims ' + which)
    need(state['before']['candidate']['heads_canonical_sha256'] == learn['initial_heads_canonical_sha256'], 'Fresh original seeded heads')
    need(state['after']['candidate']['heads_canonical_sha256'] != learn['initial_heads_canonical_sha256'], 'Head canonical changed claim')
    probe = d[O + 'controlled_hook_exception.json']
    need(probe['complete'] and probe['all575_states_and_RNG_exact'] and probe['exact_callable_restored'] and probe['owned_hooks_removed'] and probe['original_CodeFormer_forward_calls'] == 0 and probe['base_stub_calls'] == 1, 'Separate safe hook exception probe')

    sampler = d[O + 'sampler.json']
    need(sampler['reference_order'] == learn['optimization_reference_order'] and sampler['role_order'] == learn['slot_roles'] and sampler['exposures'] == 256 and not sampler['dynamic_augmentation'], 'Frozen optimizer data/order')
    prev_head = learn['initial_heads_canonical_sha256']
    for u in range(65):
        cp = d[O + f'checkpoints/u{u:04d}/checkpoint.json']
        need(cp['complete'] and cp['update'] == u and cp['head_canonical_sha256'] == cp['heads']['canonical_sha256'], 'Checkpoint manifest ' + str(u))
        need(cp['counters']['optimizer_updates'] == cp['counters']['optimizer_update_attempts'] == u, 'Checkpoint committed census ' + str(u))
        if u == 0:
            need(cp['head_canonical_sha256'] == prev_head, 'Checkpoint zero initializer')
            continue
        commit = d[O + f'updates/u{u:04d}/commit.json']
        lr = learn['optimizer']['lr'] * min(u / 4, 1)
        expected_slots = [learn['optimization_reference_order'][(u - 1 + offset) % 16] + '__' + role for role, offset in zip(learn['slot_roles'], learn['sampler_offsets'])]
        need(sampler['updates'][u - 1] == {'update': u, 'slots': expected_slots}, 'Independent sampler schedule ' + str(u))
        need(commit['complete'] and commit['update'] == u and commit['LR'] == lr and commit['head_before_sha256'] == prev_head and commit['head_after_sha256'] == cp['head_canonical_sha256'], 'Commit LR/head chain ' + str(u))
        need(commit['all539_frozen_exact'] and commit['all575_grad_fields_None'] and commit['clip_max_norm'] == 1.0 and 0 < commit['clip_coefficient'] <= 1 and commit['preclip_norm'] >= 0, 'Commit gradient/frozen receipts ' + str(u))
        for slot, case_id in enumerate(expected_slots, 1):
            obj = d[O + f'training/u{u:04d}/slot{slot}/objective.json']
            need(obj['complete'] and obj['id'] == case_id and obj['update'] == u and obj['slot'] == slot and obj['LR'] == lr and obj['head_state_before_commit_sha256'] == prev_head and obj['divided_objective'] == obj['objective'] / 4, 'Microbatch scalar/schedule receipt ' + str((u, slot)))
        prev_head = cp['head_canonical_sha256']
    need(prev_head == state['after']['candidate']['heads_canonical_sha256'], 'Final head receipt chain')
    need(result['last_checkpoint'] == status['last_committed_checkpoint'] == sup['last_committed_checkpoint'] == state['last_checkpoint'], 'Last committed checkpoint exact')

    events, resources = d[O + 'events.jsonl'], d[O + 'resources.jsonl']
    prior = dict.fromkeys(final_counters, 0)
    kinds = collections.Counter()
    for event in events:
        kinds[event['kind']] += 1
        counters = event['counters']
        need(set(counters) == set(final_counters), 'Exact event counter schema')
        for k, value in counters.items():
            need(type(value) is int and prior[k] <= value <= final_counters[k], 'Monotonic bounded event ' + k)
        need(counters['full_forward_returns'] <= counters['full_forward_attempts'] and counters['backward_returns'] <= counters['backward_attempts'] and counters['optimizer_updates'] <= counters['optimizer_update_attempts'], 'Event attempts/returns')
        if event['kind'] == 'backward_determinism_scope':
            need(event['restored_exact'] and event['before'] == event['after'] == {'enabled': True, 'warn_only': False} and event['during'] == {'enabled': True, 'warn_only': True} and not event['unexpected_nondeterminism_warnings'], 'Bounded backward warning scope')
        prior = counters
    need(prior == final_counters, 'Last event final counter agreement')
    for name, count in {'forward_reserved': 712, 'forward_returned': 712, 'backward_reserved': 256, 'backward_returned': 256, 'optimizer_reserved': 64, 'optimizer_returned': 64, 'update_fully_saved': 64, 'controlled_hook_probe_reserved': 1, 'controlled_hook_probe_returned': 1, 'optimizer_construction_reserved': 1, 'optimizer_construction_returned': 1}.items():
        need(kinds[name] == count, 'Event ledger exact count ' + name)
    for r in resources:
        need(r['current_RSS_bytes'] <= lim['maximum_RSS_bytes'] and r['peak_RSS_bytes'] <= lim['maximum_RSS_bytes'] and r['CUDA_reserved_bytes'] <= lim['maximum_CUDA_reserved_bytes'] and r['CUDA_peak_reserved_bytes'] <= lim['maximum_CUDA_reserved_bytes'] and r['output_bytes'] <= lim['maximum_output_bytes'] and r['free_disk_bytes'] >= lim['continuing_disk_reserve'] and r['seconds'] <= lim['internal_seconds'], 'Saved resource cap')
    need(result['elapsed_seconds'] <= lim['internal_seconds'] and sup['seconds'] <= lim['external_seconds'], 'Finite runtime caps')

    baseline = {r['id']: r for r in d[O + 'reference_bank.json']['cases']}
    initial_guards = {r['id']: r for r in p['guards']}
    need(set(baseline) == {r['id'] for r in reference_rows} and len(baseline) == 120, 'Reference 120-case bank')
    curves, margins = [], []
    scalar_gates = []
    for u in SNAPS:
        receipt = d[O + f'snapshots/u{u:04d}/bank_and_gates.json']
        bank = {r['id']: r for r in receipt['cases']}
        need(set(bank) == {r['id'] for r in selected} and len(bank) == 56, 'Snapshot 56-case bank')
        legacy = {'pass': True, 'thresholds_reset_to_U96': False, 'groups': {}}
        for role in ('clear', 'dark', 'filter'):
            ids = sorted(k for k in initial_guards if initial_guards[k]['role'] == role)
            group = {'pass': True, 'views': {}}
            for view in ('raw', 'native_png'):
                group['views'][view] = {}
                for field in (('mae', 'adjacent_rgb_error') if role == 'filter' else ('behavior_distance',)):
                    def measurement(rows, k):
                        return float(rows[k]['behavior_distance'][view] if field == 'behavior_distance' else rows[k]['metrics'][view][field])
                    start = {k: measurement(initial_guards, k) for k in ids}
                    now = {k: measurement(bank, k) for k in ids}
                    values = {k: now[k] - start[k] for k in ids} if role == 'filter' else now
                    case_limits = dict.fromkeys(ids, .03) if role == 'filter' else {k: max(.03, start[k]) for k in ids}
                    mean_limit = .01 if role == 'filter' else max(.01, seq(start.values()) / 8)
                    mean = seq(values.values()) / 8
                    failed = [k for k in ids if values[k] > case_limits[k]]
                    passed = not failed and mean <= mean_limit
                    group['views'][view][field] = {'pass': passed, 'initial': start, 'current': now, 'measurements': values, 'case_limits': case_limits, 'mean_limit': mean_limit, 'mean': mean, 'failed_ids': failed}
                    group['pass'] = group['pass'] and passed
                    margins.append({'update': u, 'bank': 'legacy', 'role': role, 'view': view, 'criterion': field, 'mean_margin': mean_limit - mean, 'worst_case_margin': min(case_limits[k] - values[k] for k in ids)})
            legacy['groups'][role] = group
            legacy['pass'] = legacy['pass'] and group['pass']
        monitor = {'pass': True, 'TRAIN_monitor_only_not_DEV': True, 'endpoint_evaluated': u == 64, 'groups': {}}
        for role in learn['slot_roles']:
            ids = sorted(k for k, row in bank.items() if row['role'] == role and row['cohort'] == 'monitor')
            need(len(ids) == 8, 'Eight fixed monitor photos')
            group = {'pass': True, 'views': {}}
            for view in ('raw', 'native_png'):
                criteria = {}
                for region in (('whole', 'central128', 'outside') if role == 'local_gaussian_1p2' else ('whole',)):
                    for field in ('mae', 'adjacent_rgb_error'):
                        start = {k: baseline[k]['regional_metrics'][view][region][field] for k in ids}
                        now = {k: bank[k]['regional_metrics'][view][region][field] for k in ids}
                        deltas = {k: now[k] - start[k] for k in ids}
                        mean = seq(deltas.values()) / 8
                        passed = max(deltas.values()) <= .03 and mean <= .01
                        name = region + '/' + field
                        criteria[name] = {'pass': passed, 'initial': start, 'current': now, 'deltas': deltas, 'mean': mean, 'mean_limit': .01, 'worst_limit': .03}
                        margins.append({'update': u, 'bank': 'monitor', 'role': role, 'view': view, 'criterion': name, 'mean_margin': .01 - mean, 'worst_case_margin': .03 - max(deltas.values())})
                if role in ('clear', 'lowlight_linear_0p35'):
                    name = 'U96_behavior_distance' if role == 'clear' else 'U96_illumination_distance'
                    values = {k: bank[k][name][view] for k in ids}
                    passed = seq(values.values()) / 8 <= .01 and max(values.values()) <= .03
                    criteria[name] = {'pass': passed, 'values': values, 'mean_limit': .01, 'worst_limit': .03}
                    margins.append({'update': u, 'bank': 'monitor', 'role': role, 'view': view, 'criterion': name, 'mean_margin': .01 - seq(values.values()) / 8, 'worst_case_margin': .03 - max(values.values()), 'mean': seq(values.values()) / 8, 'worst': max(values.values())})
                if role in ('local_gaussian_1p2', 'global_gaussian_1p2'):
                    region = 'central128' if role == 'local_gaussian_1p2' else 'whole'
                    start = seq(baseline[k]['regional_metrics'][view][region]['adjacent_rgb_error'] for k in ids) / 8
                    now = seq(bank[k]['regional_metrics'][view][region]['adjacent_rgb_error'] for k in ids) / 8
                    gain = (start - now) / start
                    curves.append({'update': u, 'role': role, 'view': view, 'region': region, 'initial_mean_adjacent_rgb_error': start, 'current_mean_adjacent_rgb_error': now, 'gain_fraction': gain, 'gain_percent': 100 * gain, 'one_percent_requirement_evaluated_only_at64': u == 64, 'endpoint_margin_fraction': gain - .01 if u == 64 else None})
                    if u == 64:
                        criteria['new_1_percent_TRAIN_blur_endpoint'] = {'pass': gain >= .01, 'gain_fraction': gain, 'initial_mean': start, 'current_mean': now, 'region': region, 'not_historical_DGP_feature_gate_or_DEV_qualification': True}
                if u == 64 and role == 'lowlight_linear_0p35':
                    for field in ('mae', 'adjacent_rgb_error'):
                        start = seq(baseline[k]['regional_metrics'][view]['whole'][field] for k in ids) / 8
                        now = seq(bank[k]['regional_metrics'][view]['whole'][field] for k in ids) / 8
                        criteria['endpoint_dark_' + field] = {'pass': now <= start, 'initial_mean': start, 'current_mean': now}
                group['views'][view] = criteria
                group['pass'] = group['pass'] and all(v['pass'] for v in criteria.values())
            monitor['groups'][role] = group
            monitor['pass'] = monitor['pass'] and group['pass']
        need(legacy == receipt['legacy'] and monitor == receipt['monitor'], 'Exact independent scalar gate dictionaries at update ' + str(u))
        need(legacy['pass'] and monitor['pass'], 'All fixed preservation/endpoint gates pass at update ' + str(u))
        scalar_gates.append({'update': u, 'legacy_exact': True, 'monitor_exact': True, 'legacy_pass': legacy['pass'], 'monitor_pass': monitor['pass']})

    cap_headroom = {
        'RSS_bytes': lim['maximum_RSS_bytes'] - max(r['peak_RSS_bytes'] for r in resources),
        'CUDA_reserved_bytes': lim['maximum_CUDA_reserved_bytes'] - max(r['CUDA_peak_reserved_bytes'] for r in resources),
        'output_bytes_from_largest_saved_sample': lim['maximum_output_bytes'] - max(r['output_bytes'] for r in resources),
        'disk_reserve_bytes': min(r['free_disk_bytes'] for r in resources) - lim['continuing_disk_reserve'],
        'internal_seconds': lim['internal_seconds'] - result['elapsed_seconds'],
        'external_seconds': lim['external_seconds'] - sup['seconds']}
    curve_report = {'complete': True, 'archive_sha256': b['archive_sha256'], 'protocol_sha256': SHA,
        'curves': curves, 'preservation_margins': margins, 'resource_headroom': cap_headroom,
        'scope': 'Ordered scalar reconstruction of saved metrics only. All endpoint figures are adaptive TRAIN-monitor means, not DEV/native/FINAL or identity recovery evidence.',
        'model_qualified': False}
    curve_path = OUT / 'endpoint_curve_and_margins_v1.json'
    with curve_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(curve_report, stream, indent=2, allow_nan=False); stream.write('\n')
    review = {'complete': True, 'independent_return_metadata_audit_pass': True, 'semantic_checks': checks,
        'source': pin(Path(__file__).resolve()), 'bounded_stream_receipt': pin(OUT / 'bounded_stream_readback_v1.json'),
        'archive': {'path': 'outputs/cctv-codeformer-u96-postfusion-smoothanchors-lr-v1-results.tar.gz', 'bytes': b['archive_bytes'], 'sha256': b['archive_sha256']},
        'protocol': pin(STAGE / 'protocol.json'), 'payload_members': len(m), 'payload_bytes': b['payload_bytes'],
        'local_sealed_files_rehashed': len(files), 'runtime_direct_pin_joins': pin_joins,
        'authoritative_status': result['status'], 'worker_exit': sup['worker_exit'], 'timeout': sup['timeout'],
        'optimizer_updates': 64, 'optimizer_update_attempts': 64, 'optimizer_outcome_unknown': False,
        'counters': final_counters, 'worker_seconds': result['elapsed_seconds'], 'supervisor_seconds': sup['seconds'],
        'exact_scalar_gate_recomputations': scalar_gates, 'endpoint_gains': [r for r in curves if r['update'] == 64],
        'state_invariant_receipt': m[O + 'state_invariants.json'], 'last_checkpoint': result['last_checkpoint'],
        'events': {'rows': len(events), 'kinds': dict(kinds), 'all_counters_monotonic_and_bounded': True, 'backward_warning_scopes_restored': 256},
        'resource_samples': len(resources), 'resource_headroom': cap_headroom, 'curve_and_margin_receipt': pin(curve_path),
        'visual_sheets_in_archive': 0, 'PNG_to_visual_sheet_creation_owned_by_separate_pixel_auditor': True,
        'limits': ['Independent actual saved-pixel audit and state tensor/Adam recurrence audit are separate.',
                   'State/frozen/RNG claims here are hash-bound runtime receipts; this metadata reader does not decode tensors.',
                   'The same 16 optimization and 8 monitor photographs have adaptive TRAIN reuse; subject/pretraining overlap is unknown.',
                   'Passing a finite TRAIN screen does not qualify the model or authorize automatic continuation/promotion; fixed DEV/native/final/user review remain.'],
        'Torch_neural_optimizer_training_VM_calls': 0, 'model_qualified': False}
    path = OUT / 'independent_return_review_v1.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(review, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'complete': True, 'checks': checks, 'review': pin(path), 'curves': pin(curve_path),
                      'endpoint_gain_percent': [r['gain_percent'] for r in curves if r['update'] == 64]}))


if __name__ == '__main__':
    main()
