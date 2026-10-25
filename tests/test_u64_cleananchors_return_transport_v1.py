"""Saved-data miniature transport/token/quality-stop regressions; no models."""
import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('u64_return_transport_under_test', ROOT / 'scripts/audit_u64_cleananchors_return_transport_v1.py')
A = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(A)
P = json.loads((ROOT / 'outputs' / A.STAGE / 'protocol.json').read_bytes())
PARENT = json.loads((ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm/protocol.json').read_bytes())
PLAN, CASES = A.planned_calls(P, PARENT)


def pin(raw):
    return {'bytes': len(raw), 'sha256': A.sha(raw)}


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def trace(limit=204):
    count, events, token = dict.fromkeys(A.CAPS, 0), [], 0
    def operation(kind, metadata):
        nonlocal token
        token += 1
        attempt, returned, _ = A.OPERATIONS[kind]
        count[attempt] += 1
        if kind == 'forward':
            count[metadata['category']] += 1
        events.append({'kind': kind + '_reserved', 'token': token, 'elapsed_seconds': len(events) * .01,
                       'counters': dict(count), **metadata})
        count[returned] += 1
        events.append({'kind': kind + '_returned', 'token': token, 'elapsed_seconds': len(events) * .01,
                       'counters': dict(count), **metadata,
                       **({} if kind == 'optimizer_construction' else {'synchronized_return': True})})
    for call in PLAN[:limit]:
        if call['slot'] == 61:
            operation('optimizer_construction', {})
        operation('forward', {'id': call['id'], 'category': A.PHASE_CATEGORY[call['phase']],
                              'plan_slot': call['slot'], 'update': call['new_update']})
        if call['backward']:
            operation('backward', {'id': call['id'], 'plan_slot': call['slot'], 'update': call['new_update']})
            if call['optimizer_application_after_this_forward']:
                operation('optimizer_update', {'update': call['new_update']})
    return events


def archive(entries):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as out:
        for name, value in entries:
            member = tarfile.TarInfo(A.STAGE + '/' + name); member.size = len(value)
            out.addfile(member, io.BytesIO(value))
    return gzip.compress(raw.getvalue(), mtime=0)


def omit_operations(rows, predicate):
    """Reseal counters/tokens after deleting an operation; retain actual call IDs."""
    count, token, tokens, result = dict.fromkeys(A.CAPS, 0), 0, {}, []
    for original in rows:
        if predicate(original):
            continue
        row = copy.deepcopy(original); kind, action = row['kind'].rsplit('_', 1)
        attempted, returned, _ = A.OPERATIONS[kind]
        if action == 'reserved':
            token += 1; tokens[row['token']] = token; count[attempted] += 1
            if kind == 'forward': count[row['category']] += 1
        else:
            count[returned] += 1
        row['token'] = tokens[row['token']]; row['counters'] = dict(count); result.append(row)
    return result


def saved_fixture():
    """Synthetic complete u8 stop metadata, with actual sealed TRAIN membership."""
    index = {name: copy.deepcopy(value) for name, value in P['files'].items() if not name.startswith('weights/')}
    index['protocol.json'] = {'bytes': A.PROTOCOL_FIXED[0], 'sha256': A.PROTOCOL_FIXED[1]}
    metadata = {P['parent_protocol']['path']: copy.deepcopy(PARENT)}
    events = trace(); replay = A.event_readback(events, PLAN)
    def store(name, value):
        index[name] = pin(encode(value)); metadata[name] = value
        return {'path': name, **index[name]}
    store(A.RUN + 'events.jsonl', events)
    by = {(row['reference_id'], row['role']): row for row in PARENT['cases']}
    opt, mon = (PARENT['learning'][key] for key in ('optimization_reference_order', 'monitor_reference_order'))
    selection = {'optimization': [by[(ref, role)] for ref in opt for role in A.ROLES],
        'monitors': [by[(ref, role)] for ref in mon for role in A.ROLES], 'guards': PARENT['guards'],
        'optimization_reference_order': opt, 'monitor_reference_order': mon, 'independent_holdout': False,
        'photograph_separation_is_not_identity_or_pretraining_proof': True, 'DEV_native_wild_FINAL_optimized': False}
    store(A.RUN + 'operation_plan.json', {'complete': True, 'calls': PLAN, 'selection': selection, 'optimizer_maximum_new_updates': 16})
    for call in PLAN[:204]:
        source = CASES[call['id']]; base = A.RUN + 'calls/' + str(call['slot']).zfill(4) + '/'
        outputs = {}
        for field, suffix in (('raw', 'raw.npy'), ('png', 'result.png'), ('unclipped', 'unclipped.npy')):
            name = base + suffix; index[name] = pin(name.encode()); outputs[field] = {'path': name, **index[name]}
        row = {key: copy.deepcopy(source[key]) for key in ('id', 'reference_id', 'role', 'split', 'input')}
        row.update({'target': source.get('target'), 'call': call, 'update': call['new_update'],
                    'outputs': outputs, 'model_qualified': False})
        store(base + 'record.json', row)
    for update in (0, 4, 8):
        calls = [c for c in PLAN[:204] if c['new_update'] == update and c['phase'] == ('initial_u64_quality_bank' if update == 0 else 'new_quality_snapshot')]
        records = {}
        for call in calls:
            name = A.RUN + 'calls/' + str(call['slot']).zfill(4) + '/record.json'
            records[call['id']] = {**metadata[name], 'receipt': {'path': name, **index[name]}}
        store(A.RUN + 'banks/u' + str(update).zfill(4) + '/bank.json', {'complete': True, 'records': records, 'gate': {'pass': update != 8}})
    for update in range(9):
        store(A.RUN + 'checkpoints/u' + str(update).zfill(4) + '/checkpoint.json',
              {'complete': True, 'update': update, 'counters': replay['checkpoint_counter_snapshots'][update], 'model_qualified': False})
        if update:
            store(A.RUN + 'updates/u' + str(update).zfill(4) + '/commit.json',
                  {'complete': True, 'new_update': update, 'LR': .0001 * min(1., update / 4.),
                   'microbatches': ['fixture'] * 4, 'frozen539_exact': True})
    common = {'protocol_sha256': A.PROTOCOL_FIXED[1], 'counters': replay['counters'],
              'operation_outcome_unknown': False, 'unknown_reservations': [], 'model_qualified': False}
    store(A.RUN + 'status.json', {**common, 'complete': False, 'quality_stop': True, 'label': 'retained_stop', 'active_update': 8})
    store(A.RUN + 'failure.json', {**common, 'complete': False, 'quality_stop': True, 'error_type': 'QualityStop', 'active_update': 8})
    store('supervisor.worker.json', {'complete': True, 'phase': 'worker', 'spawn_returned': True, 'exit_code': 2,
           'timeout': False, 'seconds': 250., 'cleanup': {'child_reaped': True, 'process_group_absent': True}})
    store(A.RUN + 'resources.jsonl', [{'elapsed_seconds': 200., 'free_disk_bytes': 8 * 1024**3,
          'current_RSS_bytes': 1024, 'peak_RSS_bytes': 2048, 'CUDA_reserved_bytes': 1024,
          'CUDA_peak_reserved_bytes': 2048, 'output_bytes': 100000}])
    export = {'payload_files': len(index), 'payload_bytes': sum(r['bytes'] for r in index.values()),
              'worker_exit': 2, 'confirmed_counters': replay['counters'], 'operation_outcome_unknown': False,
              'status_receipt': {'path': A.RUN + 'status.json', **index[A.RUN + 'status.json']},
              'result_present': False, 'failure_present': True,
              'journal_tail_reconciled': True, 'journal_bytes_observed': index[A.RUN + 'events.jsonl']['bytes'],
              'journal_tail_kind': events[-1]['kind'],
              'quality_stop_exit2': True, 'quality_stop_recorded': True}
    return copy.deepcopy(P), index, metadata, export


class ReturnTransport(unittest.TestCase):
    def test_actual_prospective_schedule_is292_with_exact_training_partition(self):
        self.assertEqual(len(PLAN), 292)
        self.assertEqual(sum(c['backward'] for c in PLAN), 64)
        self.assertEqual(sum(c['optimizer_application_after_this_forward'] for c in PLAN), 16)
        self.assertEqual(len(CASES), 120)

    def test_complete_u8_prefix_reconstructs204_32_8_without_final_caps_assumption(self):
        result = A.event_readback(trace(), PLAN)
        self.assertEqual(result['counters']['full_forward_returns'], 204)
        self.assertEqual(result['counters']['backward_returns'], 32)
        self.assertEqual(result['counters']['optimizer_updates'], 8)
        self.assertFalse(result['operation_outcome_unknown'])

    def test_unreturned_reservation_is_preserved_as_unknown(self):
        result = A.event_readback(trace(1)[:1], PLAN)
        self.assertTrue(result['operation_outcome_unknown'])
        self.assertEqual(result['counters']['full_forward_attempts'], 1)
        self.assertEqual(result['counters']['full_forward_returns'], 0)
        self.assertEqual(result['unknown_reservations'][0]['token'], 1)

    def test_bool_token_duplicate_return_and_skipped_token_refused(self):
        for change in ('bool', 'duplicate', 'skipped'):
            rows = trace(2)
            if change == 'bool': rows[0]['token'] = True
            elif change == 'duplicate': rows.insert(2, copy.deepcopy(rows[1]))
            else: rows[2]['token'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError): A.event_readback(rows, PLAN)

    def test_wrong_case_slot_update_category_and_boolean_metadata_refused(self):
        for field, value in (('id', 'wrong'), ('plan_slot', 2), ('update', 1), ('category', 'candidate_training_forwards'), ('plan_slot', True)):
            rows = trace(1); rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): A.event_readback(rows, PLAN)

    def test_fabricated_counter_unknown_kind_nonmonotone_and_nonfinite_time_refused(self):
        for change in ('counter', 'kind', 'time', 'nan'):
            rows = trace(2)
            if change == 'counter': rows[1]['counters']['full_forward_returns'] = 0
            elif change == 'kind': rows[0]['kind'] = 'extra_neural_call_hidden'
            elif change == 'time': rows[2]['elapsed_seconds'] = 0.
            else: rows[2]['elapsed_seconds'] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError): A.event_readback(rows, PLAN)

    def test_optimizer_requires_four_backward_returns_and_one_construction(self):
        rows = trace(64)
        i = next(i for i, r in enumerate(rows) if r['kind'] == 'optimizer_update_reserved')
        rows[i]['update'] = 2
        with self.assertRaises(ValueError): A.event_readback(rows, PLAN)

    def test_next_update_cannot_begin_when_prior_commit_tokens_are_removed(self):
        rows = omit_operations(trace(65), lambda row: row['kind'].startswith('optimizer_update_') and row['update'] == 1)
        self.assertEqual(rows[-1]['counters']['optimizer_updates'], 0)
        self.assertEqual(rows[-1]['counters']['full_forward_returns'], 65)
        with self.assertRaisesRegex(ValueError, 'committed-update cursor'):
            A.event_readback(rows, PLAN)

    def test_snapshot_requires_its_current_commit_and_next_forward_needs_prior_backward(self):
        rows = omit_operations(trace(77), lambda row: row['kind'].startswith('optimizer_update_') and row['update'] == 4)
        with self.assertRaisesRegex(ValueError, 'committed-update cursor'): A.event_readback(rows, PLAN)
        rows = omit_operations(trace(62), lambda row: row['kind'].startswith('backward_') and row['plan_slot'] == 61)
        with self.assertRaisesRegex(ValueError, 'preceding training forward'): A.event_readback(rows, PLAN)

    def test_single_stream_hash_and_crc_full_tar_footer(self):
        raw = archive([('protocol.json', b'{}\n')])
        index, metadata, kept, framing = A.inspect_stream(io.BytesIO(raw))
        self.assertEqual(framing['compressed_sha256'], A.sha(raw))
        self.assertEqual(framing['compressed_bytes'], len(raw))
        self.assertEqual(index['protocol.json'], pin(b'{}\n'))
        self.assertEqual(metadata['protocol.json'], {})
        self.assertTrue(framing['single_gzip_CRC_ISIZE_complete'])
        self.assertGreaterEqual(framing['tar_zero_EOF_padding_bytes'], 1024)

    def test_extra_frame_truncated_crc_and_hidden_tar_payload_refused(self):
        raw = archive([('file', b'x')]); bad = bytearray(raw); bad[-8] ^= 1
        plain = bytearray(gzip.decompress(raw)); plain[1536] = 1
        for data in (raw + gzip.compress(b'extra'), raw[:-2], bytes(bad), gzip.compress(plain)):
            with self.assertRaises(Exception): A.inspect_stream(io.BytesIO(data))

    def test_regular_member_count_metadata_and_payload_caps(self):
        for patch, value, data in (('MAX_MEMBER', 2, archive([('large', b'123')])),
                                   ('MAX_FILES', 1, archive([('one', b'x'), ('two', b'y')])),
                                   ('MAX_METADATA', 2, archive([('large.json', b'{}\n')]))):
            with mock.patch.object(A, patch, value), self.assertRaises(ValueError): A.inspect_stream(io.BytesIO(data))

    def test_incomplete_jsonl_duplicate_json_and_nonfinite_metadata_refused(self):
        for name, raw in (('events.jsonl', b'{}'), ('proof.json', b'{"x":1,"x":2}'), ('proof.json', b'{"x":1e999}')):
            with self.assertRaises(ValueError): A.inspect_stream(io.BytesIO(archive([(name, raw)])))

    def test_saved_u8_quality_stop_is_transport_pass_without_qualification(self):
        p, index, metadata, export = saved_fixture()
        result = A.validate_return(p, index, metadata, export)
        self.assertEqual(result['terminal_status'], 'KNOWN_QUALITY_STOP_U8_NOT_QUALIFIED')
        self.assertEqual(result['generated_records'], 204)
        self.assertFalse(result['saved_scalar_gates_independently_recomputed'])

    def test_green_export_cannot_replace_worker_exit_or_failed_quality_bank(self):
        for change in ('worker', 'bank', 'result', 'unknown'):
            p, index, metadata, export = saved_fixture()
            if change == 'worker': export['worker_exit'] = 0
            elif change == 'bank': metadata[A.RUN + 'banks/u0008/bank.json']['gate']['pass'] = True
            elif change == 'result': metadata[A.RUN + 'result.json'] = {'complete': True}
            else: export['operation_outcome_unknown'] = True
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)): A.validate_return(p, index, metadata, export)

    def test_quality_stop_cannot_point_to_earlier_bank_after_later_commits(self):
        p, index, metadata, export = saved_fixture()
        metadata[A.RUN + 'failure.json']['active_update'] = 4
        metadata[A.RUN + 'status.json']['active_update'] = 4
        metadata[A.RUN + 'banks/u0004/bank.json']['gate']['pass'] = False
        with self.assertRaisesRegex(ValueError, 'actual committed cursor'):
            A.validate_return(p, index, metadata, export)

    def test_static_byte_drift_and_extra_unreturned_generated_record_refused(self):
        p, index, metadata, export = saved_fixture()
        first = next(iter(p['files'])); index[first] = pin(b'changed')
        with self.assertRaises(ValueError): A.validate_return(p, index, metadata, export)
        p, index, metadata, export = saved_fixture()
        name = A.RUN + 'calls/0292/record.json'; index[name] = pin(b'extra')
        export['payload_files'] = len(index); export['payload_bytes'] = sum(v['bytes'] for v in index.values())
        with self.assertRaises(ValueError): A.validate_return(p, index, metadata, export)

    def test_nested_runtime_pin_must_exist_and_match_actual_index(self):
        value = {'path': A.RUN + 'actual.npy', 'bytes': 1, 'sha256': 'a' * 64}
        with self.assertRaises(ValueError): A.direct_joins({A.RUN + 'proof.json': {'file': value}}, {})

    def test_unexpected_CUDA_warning_cannot_be_hidden_as_allowed_bilinear(self):
        rows = trace(1)
        warning = {'kind': 'backward_warning_scope', 'elapsed_seconds': .02, 'counters': rows[-1]['counters'],
            'warnings': [{'category': 'UserWarning', 'message': 'other_backward has no deterministic implementation'}],
            'unexpected': [], 'strict_flags_restored': True, 'bitwise_CUDA_gradient_repeatability_claimed': False}
        with self.assertRaises(ValueError): A.event_readback(rows + [warning], PLAN)

    def test_checkpoint_counter_or_bank_record_substitution_refused(self):
        for change in ('checkpoint', 'bank'):
            p, index, metadata, export = saved_fixture()
            if change == 'checkpoint': metadata[A.RUN + 'checkpoints/u0004/checkpoint.json']['counters']['optimizer_updates'] = 3
            else:
                records = metadata[A.RUN + 'banks/u0008/bank.json']['records']; key = next(iter(records)); records[key]['update'] = 4
            with self.assertRaises(ValueError): A.validate_return(p, index, metadata, export)


if __name__ == '__main__': unittest.main()
