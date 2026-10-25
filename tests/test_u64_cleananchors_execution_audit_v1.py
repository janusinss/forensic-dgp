"""Miniature archive/approval/cache counterexamples; no model or array decoding."""
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
SPEC = importlib.util.spec_from_file_location('u64_execution_audit_under_test',
    ROOT / 'scripts/audit_u64_cleananchors_execution_v1.py')
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def raw_json(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def pin(raw):
    return {'bytes': len(raw), 'sha256': A.sha(raw)}


def archive_fixture(entries, *, nonzero_tail=False, linked=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, value in entries:
            member = tarfile.TarInfo(A.STAGE + '/' + name)
            member.size = len(value)
            if linked:
                member.type = tarfile.SYMTYPE
                member.linkname = '../outside'
                member.size = 0
            archive.addfile(member, io.BytesIO(value))
    data = bytearray(stream.getvalue())
    if nonzero_tail:
        data[1536] = 1
    return gzip.compress(data, mtime=0)


def source_fixture():
    sources = {A.PACKAGE + '/' + name: (ROOT / A.PACKAGE / name).read_bytes()
               for name in A.CORE_FIXED}
    sources.update({A.PACKAGE + '/worker.py': b'"""fixture worker; never run."""\n',
                    A.PACKAGE + '/script_entry.py': b'"""fixture bridge."""\n',
                    A.PACKAGE + '/run_continuation.sh': b'#!/bin/sh\nexit 1\n'})
    sources.update({name: (ROOT / name).read_bytes() for name in A.SHARED_FIXED})
    approved = {name: {'path': name, **pin(value)} for name, value in sources.items()}
    contents = {A.ENTRY_MAP.get(name, name): value for name, value in sources.items()}
    contents['research/__init__.py'] = b'"""Inert research namespace."""\n'
    members = {name: pin(value) for name, value in contents.items()}
    members[A.HEAD_MEMBER] = dict(zip(('bytes', 'sha256'), A.FIXED[A.HEAD_MEMBER]))
    models = {name: {'path': 'weights/' + name + '.pth', 'bytes': value[0], 'sha256': value[1]}
              for name, value in A.MODEL_FIXED.items()}
    files = copy.deepcopy(members)
    files.update({value['path']: {key: value[key] for key in ('bytes', 'sha256')} for value in models.values()})
    p = {'files': files, 'model': models,
         'u64_head': {'path': A.HEAD_MEMBER, **members[A.HEAD_MEMBER]},
         'prototype': {name[:-3]: {'path': A.PACKAGE + '/' + name,
             'bytes': value[0], 'sha256': value[1]} for name, value in A.CORE_FIXED.items()}}
    return p, members, contents, approved


def cache_fixture():
    """All60 distinct synthetic records; real32/24/4 schema, tiny inert bytes."""
    roles = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight')
    def row(index, role, *, guard=False):
        reference = ('guard_' if guard else 'monitor_') + str(index)
        value = {'id': reference + '__' + role, 'reference_id': reference,
                 'role': role, 'split': 'TRAIN', 'optimization_permitted': False,
                 'input': {'path': 'assets/input_' + reference + '_' + role + '.png', **pin(b'input')},
                 'target': None if role == 'dark' else {'path': 'assets/target_' + reference + '.png', **pin(b'target')}}
        if not guard:
            value['cohort'] = 'monitor'
        return value
    monitors = [row(index, role) for index in range(8) for role in roles]
    guards = [row(index, role, guard=True) for index in range(8) for role in ('gaussian', 'clear', 'dark')]
    selection = {'monitors': monitors, 'guards': guards}
    p = {'files': {}, 'controls': [], 'u64_parity': []}
    members, contents, indexed, measured = {}, {}, {}, []
    for field, expected, kind, update, directory in (
            ('controls', monitors + guards, 'reference', 0, 'reference/'),
            ('u64_parity', monitors[:4], 'snapshot', 64, 'snapshots/u0064/')):
        for reference in expected:
            key = reference['id']
            original = A.OLD_RUN + directory + key + '/'
            installed = 'assets/' + field + '/' + key + '/'
            metrics = {view: {'mae': None if reference['target'] is None else 0.1,
                             'adjacent_rgb_error': None if reference['target'] is None else 0.2}
                       for view in ('raw', 'native_png')}
            regions = {view: None if reference['target'] is None else {'whole': copy.deepcopy(metrics[view])}
                       for view in ('raw', 'native_png')}
            outputs, cache_row = {}, {'id': key}
            for part, old_name, new_name in (('raw', 'raw.npy', 'raw.npy'), ('png', 'output.png', 'result.png')):
                value = ('fixture:' + kind + ':' + key + ':' + part).encode()
                outputs[part] = {'path': original + old_name, **pin(value)}
                cache_row[part] = {'path': installed + new_name, **pin(value)}
                indexed[original + old_name] = outputs[part]
                members[installed + new_name] = pin(value)
            record = {name: copy.deepcopy(reference[name]) for name in ('id', 'reference_id', 'role', 'split', 'input', 'target')}
            record.update({'kind': kind, 'update': update, 'cohort': reference.get('cohort', 'legacy_guard'),
                           'complete': True, 'model_qualified': False, 'outputs': outputs,
                           'metrics': metrics, 'regional_metrics': regions})
            record_raw = raw_json(record)
            cache_row['record'] = {'path': installed + 'record.json', **pin(record_raw)}
            indexed[original + 'record.json'] = {'path': original + 'record.json', **pin(record_raw)}
            members[installed + 'record.json'] = pin(record_raw)
            contents[installed + 'record.json'] = record_raw
            measured.append({'record': original + 'record.json', 'id': key, 'kind': kind, 'update': update,
                'role': reference['role'], 'raw_file': outputs['raw'], 'png_file': outputs['png'],
                'metrics': metrics, 'regional_metrics': regions})
            p[field].append(cache_row)
    pixel = {'complete': True, 'independent_saved_pixel_integrity_pass': True,
             'pilot_quality_gates_pass': True, 'model_qualified': False,
             'protocol': {'sha256': A.OLD_PROTOCOL_SHA}, 'all_saved_pixel_metrics': measured}
    index = {'complete': True, 'archive_bytes': A.OLD_ARCHIVE_FIXED[0],
             'archive_SHA256': A.OLD_ARCHIVE_FIXED[1], 'members': indexed}
    fixed = {}
    for name, field, value in ((A.PIXEL_MEMBER, 'historical_pixel_audit', pixel),
                                (A.INDEX_MEMBER, 'historical_archive_index', index)):
        contents[name] = raw_json(value)
        members[name] = pin(contents[name])
        p[field] = {'path': name, **members[name]}
        fixed[name] = (members[name]['bytes'], members[name]['sha256'])
    p['files'] = copy.deepcopy(members)
    return p, members, contents, selection, fixed


class ExecutionAudit(unittest.TestCase):
    def test_valid_small_archive_has_full_single_frame_and_padding(self):
        raw = archive_fixture([('protocol.json', b'{"fixture":true}\n')])
        members, contents, framing = A.inspect_stream(io.BytesIO(raw))
        self.assertEqual(members['protocol.json'], pin(contents['protocol.json']))
        self.assertTrue(framing['gzip_single_frame_complete'])
        self.assertTrue(framing['gzip_CRC_and_ISIZE_verified'])
        self.assertEqual(framing['complete_zero_tar_padding_bytes'], 9216)

    def test_reader_refuses_second_frame_suffix_bad_crc_and_truncation(self):
        raw = archive_fixture([('file', b'x')])
        bad_crc = bytearray(raw); bad_crc[-8] ^= 1
        for value in (raw + gzip.compress(b'next'), raw + b'x', raw[:-3], bytes(bad_crc)):
            with self.subTest(value=value[-12:]), self.assertRaises(Exception):
                A.inspect_stream(io.BytesIO(value))

    def test_reader_refuses_buffered_tar_tail_links_and_duplicate(self):
        cases = (archive_fixture([('file', b'x')], nonzero_tail=True),
                 archive_fixture([('file', b'')], linked=True),
                 archive_fixture([('file', b'a'), ('file', b'b')]))
        for raw in cases:
            with self.assertRaises(ValueError):
                A.inspect_stream(io.BytesIO(raw))

    def test_exact_prefix_and_canonical_names(self):
        for name in ('other/file', A.STAGE + '/.', A.STAGE + '/../outside',
                     A.STAGE + '/a//b', A.STAGE + '/a\\b', A.STAGE + '/C:outside'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                A.relative(name)

    def test_member_number_and_size_are_bounded(self):
        with mock.patch.object(A, 'MAX_MEMBER', 2):
            with self.assertRaisesRegex(ValueError, 'bounded regular'):
                A.inspect_stream(io.BytesIO(archive_fixture([('file', b'123')])))
        with mock.patch.object(A, 'MAX_MEMBERS', 1):
            with self.assertRaisesRegex(ValueError, 'bounded member'):
                A.inspect_stream(io.BytesIO(archive_fixture([('one', b'x'), ('two', b'y')])))

    def test_expansion_cap_precedes_large_PAX_parsing(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w', format=tarfile.PAX_FORMAT) as archive:
            member = tarfile.TarInfo(A.STAGE + '/small')
            member.pax_headers = {'comment': 'x' * 20000}
            member.size = 1
            archive.addfile(member, io.BytesIO(b'x'))
        with mock.patch.object(A, 'MAX_EXPANDED', 1024), self.assertRaisesRegex(ValueError, 'before tar parsing'):
            A.inspect_stream(io.BytesIO(gzip.compress(stream.getvalue())))

    def test_json_rejects_duplicates_nonfinite_and_loose_local_alias(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.assertRaises(ValueError):
                A._helper.strict_json(raw)
        with self.assertRaisesRegex(ValueError, 'Actual archived JSON'):
            A.bound_json_member('proof.json', {'proof.json': pin(b'{}')}, {'proof.json': b'{"changed":true}'})

    def test_approval_cannot_be_replaced_by_resealed_green_receipt(self):
        review = {'complete': True, 'source_review_pass': True, 'unresolved_blockers': [],
                  'new_continuation_data_reuse_admission': True,
                  'neural_gradient_optimizer_training_VM_calls': 0,
                  'sources': [{'path': A.BUILDER, **pin(b'reviewed builder')}]}
        trusted = raw_json(review)
        contents, members = {A.REVIEW_MEMBER: trusted}, {A.REVIEW_MEMBER: pin(trusted)}
        A.bind_trusted_review(members, contents, trusted)
        review['sources'][0]['sha256'] = A.sha(b'changed builder')
        replacement = raw_json(review)
        with self.assertRaisesRegex(ValueError, 'independently trusted'):
            A.bind_trusted_review({A.REVIEW_MEMBER: pin(replacement)}, {A.REVIEW_MEMBER: replacement}, trusted)

    def test_approval_requires_train_reuse_unique_sources_and_builder(self):
        good = {'complete': True, 'source_review_pass': True, 'unresolved_blockers': [],
                'new_continuation_data_reuse_admission': True,
                'neural_gradient_optimizer_training_VM_calls': 0,
                'sources': [{'path': A.BUILDER, **pin(b'original')}]}
        for changed in ({**good, 'new_continuation_data_reuse_admission': False},
                        {**good, 'sources': good['sources'] * 2}, {**good, 'sources': []}):
            with self.assertRaises(ValueError):
                A.approved_sources(changed)

    def test_parent_fields_include_original_dev_evidence(self):
        parent = {key: {'exact': key} for key in ('model', 'data', 'cases', 'guards', 'quality_policy', 'inherited_DEV_evidence')}
        raw = raw_json(parent)
        p = {**copy.deepcopy(parent), 'parent_protocol': {'path': A.PARENT_MEMBER, **pin(raw)},
             'files': {A.PARENT_MEMBER: pin(raw)}}
        with mock.patch.dict(A.FIXED, {A.PARENT_MEMBER: (len(raw), A.sha(raw))}):
            A.parent_readback(p, {A.PARENT_MEMBER: pin(raw)}, {A.PARENT_MEMBER: raw})
            for key in parent:
                bad = copy.deepcopy(p); bad[key] = {'changed': True}
                with self.assertRaisesRegex(ValueError, 'immutable parent field'):
                    A.parent_readback(bad, {A.PARENT_MEMBER: pin(raw)}, {A.PARENT_MEMBER: raw})

    def test_actual_archived_reuse_and_visual_plans_bind_review(self):
        p, review, contents, members = {'files': {}}, {}, {}, {}
        for key, name in (('continuation_reuse', 'evidence/u64_continuation_reuse.json'),
                          ('visual_plan', 'evidence/u64_planned_visual_review_v1.json')):
            raw = raw_json({'complete': True, 'new_continuation_data_reuse_admission': True})
            p[key] = {'path': name, **pin(raw)}
            review[key] = {'path': 'outputs/prospective/' + key + '.json', **pin(raw)}
            members[name] = p['files'][name] = pin(raw); contents[name] = raw
        A.reviewed_plans_readback(p, members, contents, review)
        name = p['visual_plan']['path']; raw = raw_json({'complete': True, 'changed': True})
        contents[name] = raw; p['visual_plan'] = {'path': name, **pin(raw)}
        members[name] = p['files'][name] = pin(raw)
        with self.assertRaisesRegex(ValueError, 'independently reviewed bytes'):
            A.reviewed_plans_readback(p, members, contents, review)

    def test_archived_root_test_receipt_joins_all_executable_source_pins(self):
        p, members, contents, approved = source_fixture()
        approved[A.BUILDER] = {'path': A.BUILDER, **pin(b'reviewed fixture builder')}
        tests = {'complete': True, 'all_tests_pass': True,
                 'neural_gradient_optimizer_training_VM_calls': 0,
                 'sources': list(approved.values())}
        raw = raw_json(tests)
        approved[A.TESTS_SOURCE] = {'path': A.TESTS_SOURCE, **pin(raw)}
        members[A.TESTS_MEMBER] = p['files'][A.TESTS_MEMBER] = pin(raw)
        contents[A.TESTS_MEMBER] = raw
        self.assertEqual(A.reviewed_tests_readback(p, members, contents, approved),
                         {'path': A.TESTS_MEMBER, **pin(raw)})
        tests['sources'][0]['sha256'] = A.sha(b'changed worker still green')
        changed = raw_json(tests)
        members[A.TESTS_MEMBER] = p['files'][A.TESTS_MEMBER] = pin(changed)
        contents[A.TESTS_MEMBER] = changed
        with self.assertRaisesRegex(ValueError, 'independently reviewed bytes'):
            A.reviewed_tests_readback(p, members, contents, approved)

    def test_test_metadata_must_cover_current_shared_factory_worker_and_builder(self):
        for change in ('omit_factory', 'source_drift', 'nonzero_calls', 'boolean_calls'):
            p, members, contents, approved = source_fixture()
            approved[A.BUILDER] = {'path': A.BUILDER, **pin(b'reviewed builder')}
            tested = copy.deepcopy(list(approved.values()))
            tests = {'complete': True, 'all_tests_pass': True,
                     'neural_gradient_optimizer_training_VM_calls': 0, 'sources': tested}
            if change == 'omit_factory':
                tests['sources'] = [row for row in tested if row['path'] != 'research/codeformer_u96_postfusion_v1/model.py']
            elif change == 'source_drift':
                tested[0]['sha256'] = '0' * 64
            else:
                tests['neural_gradient_optimizer_training_VM_calls'] = False if change == 'boolean_calls' else 1
            raw = raw_json(tests)
            approved[A.TESTS_SOURCE] = {'path': A.TESTS_SOURCE, **pin(raw)}
            members[A.TESTS_MEMBER] = p['files'][A.TESTS_MEMBER] = pin(raw)
            contents[A.TESTS_MEMBER] = raw
            with self.subTest(change=change), self.assertRaises(ValueError):
                A.reviewed_tests_readback(p, members, contents, approved)

    def test_complete_source_and_shared_factory_namespace_and_two_omitted_weights(self):
        p, members, contents, approved = source_fixture()
        omitted, joins = A.reviewed_namespace_readback(p, members, contents, approved)
        self.assertEqual(omitted, {'weights/r2.pth', 'weights/u96.pth'})
        self.assertEqual({r['reviewed_source'] for r in joins}, set(approved))
        for name in (A.PACKAGE + '/worker.py', 'research/codeformer_u96_postfusion_v1/model.py',
                     'research/codeformer_u96_spatial_feasibility_v1/worker.py'):
            bad = dict(approved); bad.pop(name)
            with self.assertRaises(ValueError):
                A.reviewed_namespace_readback(p, members, contents, bad)

    def test_unreviewed_sources_and_duplicate_entry_bridge_are_refused(self):
        p, members, contents, approved = source_fixture()
        for extra in (A.PACKAGE + '/script_entry.py', A.PACKAGE + '/extra.py', 'research/other/__init__.py'):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'complete reviewed installed'):
                A.reviewed_namespace_readback(p, {**members, extra: pin(b'unapproved')}, contents, approved)

    def test_head_cannot_be_treated_as_omitted_or_unbound_extra_weight(self):
        p, members, contents, approved = source_fixture()
        bad = dict(members); bad.pop(A.HEAD_MEMBER)
        with self.assertRaisesRegex(ValueError, 'included immutable binding'):
            A.reviewed_namespace_readback(p, bad, contents, approved)
        p['files']['weights/extra.npz'] = pin(b'not permitted')
        with self.assertRaisesRegex(ValueError, 'retained u64 head'):
            A.reviewed_namespace_readback(p, members, contents, approved)

    def test_protocol_prototype_cannot_alias_other_reviewed_source(self):
        p, members, contents, approved = source_fixture()
        for change in ('alias', 'extra'):
            bad = copy.deepcopy(p)
            if change == 'alias':
                bad['prototype']['model']['path'] = A.PACKAGE + '/worker.py'
            else:
                bad['prototype']['unexpected'] = copy.deepcopy(bad['prototype']['model'])
            with self.assertRaisesRegex(ValueError, 'prototype'):
                A.reviewed_namespace_readback(bad, members, contents, approved)

    def test_all180_cache_members_join_index_pixels_and_actual_legacy_records(self):
        p, members, contents, selection, fixed = cache_fixture()
        with mock.patch.dict(A.FIXED, fixed):
            paths, joins = A.historical_cache_readback(p, members, contents, selection)
        self.assertEqual(len(paths), 180); self.assertEqual(len(joins), 60)
        self.assertEqual(sum(row['kind'] == 'frozen_U96' for row in joins), 56)
        guard = p['controls'][32]['record']['path']
        self.assertEqual(json.loads(contents[guard])['cohort'], 'legacy_guard')
        self.assertNotIn('cohort', selection['guards'][0])
        dark = next(row for row in selection['guards'] if row['role'] == 'dark')
        self.assertIsNone(dark['target'])

    def test_resealed_cached_record_cannot_replace_old_index_bound_bytes(self):
        p, members, contents, selection, fixed = cache_fixture()
        name = p['controls'][0]['record']['path']
        record = json.loads(contents[name]); record['input']['sha256'] = '0' * 64
        raw = raw_json(record); contents[name] = raw
        members[name] = p['files'][name] = pin(raw); p['controls'][0]['record'] = {'path': name, **pin(raw)}
        with mock.patch.dict(A.FIXED, fixed), self.assertRaisesRegex(ValueError, 'independently indexed'):
            A.historical_cache_readback(p, members, contents, selection)

    def test_resealed_cached_raw_cannot_substitute_initializer_for_original_U96(self):
        p, members, contents, selection, fixed = cache_fixture()
        value = p['controls'][0]['raw']; changed = pin(b'new candidate self distance is zero')
        members[value['path']] = p['files'][value['path']] = changed
        value.update(changed)
        with mock.patch.dict(A.FIXED, fixed), self.assertRaisesRegex(ValueError, 'independently indexed'):
            A.historical_cache_readback(p, members, contents, selection)

    def test_monitor_order_first_photo_parity_and_duplicate_ids_are_binding(self):
        for change in ('order', 'second_photo_parity', 'duplicate'):
            p, members, contents, selection, fixed = cache_fixture()
            if change == 'order':
                p['controls'][0], p['controls'][1] = p['controls'][1], p['controls'][0]
            elif change == 'second_photo_parity':
                p['u64_parity'][0]['id'] = selection['monitors'][4]['id']
            else:
                p['controls'][1]['id'] = p['controls'][0]['id']
            with mock.patch.dict(A.FIXED, fixed), self.assertRaisesRegex(ValueError, 'ordered historical'):
                A.historical_cache_readback(p, members, contents, selection)

    def test_replaced_green_pixel_audit_or_index_cannot_self_authorize(self):
        for field in ('historical_pixel_audit', 'historical_archive_index'):
            p, members, contents, selection, fixed = cache_fixture()
            name = p[field]['path']; value = json.loads(contents[name]); value['self_resealed'] = True
            contents[name] = raw_json(value); members[name] = p['files'][name] = pin(contents[name])
            p[field] = {'path': name, **members[name]}
            with mock.patch.dict(A.FIXED, fixed), self.assertRaisesRegex(ValueError, 'included immutable'):
                A.historical_cache_readback(p, members, contents, selection)

    def test_record_case_join_still_refuses_wrong_target_after_index_join(self):
        p, members, contents, selection, fixed = cache_fixture()
        name = p['controls'][0]['record']['path']; record = json.loads(contents[name]); record['target'] = None
        contents[name] = raw_json(record); members[name] = p['files'][name] = pin(contents[name])
        p['controls'][0]['record'] = {'path': name, **members[name]}
        index = json.loads(contents[A.INDEX_MEMBER])
        original = A.OLD_RUN + 'reference/' + record['id'] + '/record.json'
        index['members'][original] = {'path': original, **members[name]}
        contents[A.INDEX_MEMBER] = raw_json(index); members[A.INDEX_MEMBER] = p['files'][A.INDEX_MEMBER] = pin(contents[A.INDEX_MEMBER])
        p['historical_archive_index'] = {'path': A.INDEX_MEMBER, **members[A.INDEX_MEMBER]}
        fixed[A.INDEX_MEMBER] = (len(contents[A.INDEX_MEMBER]), A.sha(contents[A.INDEX_MEMBER]))
        with mock.patch.dict(A.FIXED, fixed), self.assertRaisesRegex(ValueError, 'immutable case'):
            A.historical_cache_readback(p, members, contents, selection)

    def test_byte_auditor_has_no_extract_numpy_or_model_execution_path(self):
        source = (ROOT / 'scripts/audit_u64_cleananchors_execution_v1.py').read_text()
        self.assertNotIn('.extractall(', source)
        self.assertNotIn('.extract(', source)
        self.assertNotIn('import numpy', source)
        self.assertNotIn('import torch', source)
        self.assertNotIn('.run(', source)


if __name__ == '__main__':
    unittest.main()
