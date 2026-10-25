"""Small inert bytes/math/namespace fixtures; no neural framework or VM call."""
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.codeformer_filter_fusion_training_v1 import transport as t


def put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


class TransportTests(unittest.TestCase):
    def setUp(self):
        # Keep exact category/namespace logic but use tiny opaque byte fixtures;
        # no gigabyte model copies or checkpoint deserialization in these tests.
        self.weight_payloads = {n: ('opaque fixture '+n).encode('ascii') for n in t.WEIGHT_ASSETS}
        weights = {n: {'sha256': hashlib.sha256(v).hexdigest(), 'bytes': len(v)}
                   for n, v in self.weight_payloads.items()}
        patcher = mock.patch.object(t, 'WEIGHT_ASSETS', weights)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fixture(self, terminal='failure'):
        scratch = ROOT/'scratch'
        scratch.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix='filter_fusion_transport_', dir=scratch)
        self.assertTrue(Path(temp.name).resolve().is_relative_to(scratch.resolve()))
        self.addCleanup(temp.cleanup)
        base = Path(temp.name)
        root = base/t.VM_NAME
        root.mkdir()
        source = put(root/'worker.py', b'# inert fixture, not executed\n')
        weights = {n: put(root/n, payload) for n, payload in self.weight_payloads.items()}
        dependency = put(base/'historical'/'weights/original.pth', b'opaque preserved weights')
        caps = {'optimizer_step_attempts': 151, 'backward_calls': 604, 'gradient_queries': 7}
        p = {'vm_name': t.VM_NAME, 'outputs': t.OUTPUTS,
            'files_sha256': {'worker.py': source['sha256'], **{n: v['sha256'] for n, v in weights.items()}},
            'files_bytes': {'worker.py': source['bytes'], **{n: v['bytes'] for n, v in weights.items()}},
            'weight_assets': weights,
            'dependencies': {'original': {'root': 'historical', 'path': 'weights/original.pth', **dependency}},
            'allowed_output_names': [t.OUTPUTS+'/'+n for n in ('run_results.json', 'failure.json', 'export_manifest.json')]+list(t.CHECKPOINT_PATHS),
            'required_success_output_names': list(t.CHECKPOINT_PATHS[:3]), 'call_caps': caps}
        sha = put(root/'protocol.json', t.json_bytes(p))['sha256']
        put(root/'outputs/supervision/run_receipt.json', t.json_bytes({'worker_terminal': True, 'protocol_sha256': sha}))
        if terminal:
            value = {'complete': True, 'protocol_sha256': sha, 'failure': terminal == 'failure',
                'completed_training_updates': 3 if terminal == 'failure' else 151,
                'partial_microbatches': 0, 'optimizer_inflight': False,
                'counts': {'optimizer_step_attempts': 3, 'backward_calls': 12, 'gradient_queries': 7} if terminal == 'failure' else caps}
            put(root/t.OUTPUTS/('failure.json' if terminal == 'failure' else 'run_results.json'), t.json_bytes(value))
        return SimpleNamespace(root=root, base=base, protocol=p, sha=sha)

    def archive(self, directory, values, *, extra_frame=False, mutate_padding=False):
        expanded = io.BytesIO()
        expected = {}
        with tarfile.open(fileobj=expanded, mode='w', format=tarfile.USTAR_FORMAT) as tar:
            for name, payload in sorted(values.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(payload), 0o644, 0
                tar.addfile(info, io.BytesIO(payload))
                expected[name] = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
        raw = bytearray(expanded.getvalue())
        if mutate_padding:
            raw[-1] = 1
        path = directory/'fixture.tar.gz'
        path.write_bytes(gzip.compress(raw, mtime=0)+(gzip.compress(b'', mtime=0) if extra_frame else b''))
        return path, expected

    def test_all_schema_math_from_saved_metadata_only(self):
        p = ROOT/'outputs/codeformer_filter_comparison_v1_preparation/model_binding_v1.json'
        if not p.exists():
            self.skipTest('Optional retained schema metadata absent')
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), t.SCHEMA_SHA256)
        schema = json.loads(p.read_text())['full515_tensor_schema']
        self.assertEqual(len(schema), 515)
        self.assertTrue(all(v['dtype'] == 'torch.float32' for v in schema.values()))
        self.assertEqual(sum(v['elements']*4 for v in schema.values()), t.MODEL_FP32_BYTES)
        fusion = {k: v for k, v in schema.items() if k.startswith('fuse_convs_dict.')}
        self.assertEqual(len(fusion), 72)
        self.assertEqual(sum(v['elements']*4 for v in fusion.values()), t.FUSION_FP32_BYTES)
        self.assertEqual(2*t.FUSION_FP32_BYTES, t.ADAM_MOMENT_BYTES)

    def test_complete_resource_disk_envelope(self):
        r = t.resource_ledger()
        self.assertLessEqual(r['projected_files'], 16000)
        self.assertLess(r['projected_payload_bytes'], 7*t.GiB)
        self.assertLess(r['single_gzip_upper_bytes'], 29*t.GiB//4)
        self.assertEqual(r['categories']['exact3_input_weight_copies']['bytes'], 1129891316)
        self.assertEqual(r['trained_tensor_values_bytes_before_serialization'], 461727788)
        self.assertEqual(r['categories']['all3_current_runtime_baseline_RAW']['files'], 1434)
        self.assertEqual(r['categories']['fresh_teacher_U256_arrays']['files'], 110)
        self.assertFalse(r['all604_training_native512_arrays_retained'])
        self.assertGreater(r['serialization_headroom_bytes_per_trained_state'], 8_000_000)

    def test_disk_proof_rejects_bad_admission(self):
        with mock.patch.dict(t.LIMITS, {'initial_free_bytes': 16*t.GiB}):
            with self.assertRaises(ValueError):
                t.resource_ledger()
        for kw in ({'teacher_arrays': 111}, {'static_files': True}, {'static_bytes': 600*t.MiB}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                t.resource_ledger(**kw)

    def test_full_runtime_baselines_and_named_witnesses(self):
        r = t.operation_ledger({'parity_candidate_forwards': 1, 'parity_R2_forwards': 1,
            'filter_witness_forwards': 3, 'clear_witness_forwards': 2, 'dark_witness_forwards': 2})
        self.assertEqual(r['top_level_forwards'], 3639)
        self.assertEqual(r['before_witness_forwards'], 3630)
        self.assertEqual(sum(r['by_counter'][n] for n in ('R2_baseline_forwards', 'original_baseline_forwards', 'parent_baseline_forwards')), 1434)
        self.assertEqual(r['mixed72_split'], {'TRAIN': 36, 'DEV': 36})
        for invalid in ({}, {'unspecified_calls': 18}, {'witness_forwards': True}, {'witness_forwards': -1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                t.operation_ledger(invalid)

    def test_data_only_fresh_transfer_and_changed_dependency(self):
        f = self.fixture()
        v = t.verify_transfer(f.root, f.sha)
        self.assertEqual(v['dependencies_verified'], 1)
        self.assertEqual(v['neural_calls'], 0)
        put(f.base/'historical/weights/original.pth', b'changed')
        with self.assertRaisesRegex(ValueError, 'Read-only dependency differs'):
            t.verify_transfer(f.root, f.sha)

    def test_unbound_importable_source_and_wrong_protocol_refused(self):
        f = self.fixture()
        with self.assertRaises(ValueError):
            t.verify_transfer(f.root, '0'*64)
        put(f.root/'unbound.py', b'pass\n')
        with self.assertRaisesRegex(ValueError, 'Unbound static'):
            t.verify_transfer(f.root, f.sha)

    def test_dependency_traversal_and_case_collision_refused(self):
        f = self.fixture()
        p = deepcopy(f.protocol)
        p['dependencies']['original']['root'] = '../escape'
        with self.assertRaises(ValueError):
            t._bindings_from_protocol(p)
        with self.assertRaises(ValueError):
            t.validate_binding_map({'A': {'sha256': 'a'*64, 'bytes': 1}, 'a': {'sha256': 'b'*64, 'bytes': 1}})

    def test_portable_paths(self):
        for name in ('../escape', '/abs', 'a\\b', 'a/./b', 'a//b', 'a/CON.png', 'a.', 'a:b', 'a/LPT1', 'a/\n'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                t.canonical(name)

    def test_hardlink_rejected(self):
        import os
        import stat
        f = self.fixture()
        try:
            os.link(f.root/'worker.py', f.root/'linked.py')
        except OSError:
            # Windows sandbox blocks creating hardlinks. Exercise the actual
            # rejection against a mocked filesystem link-count observation.
            original = Path.lstat
            def observed(path, *args, **kwargs):
                if path == f.root/'worker.py':
                    return SimpleNamespace(st_mode=stat.S_IFREG | 0o644,
                        st_nlink=2, st_file_attributes=0)
                return original(path, *args, **kwargs)
            with mock.patch.object(Path, 'lstat', observed), self.assertRaises(ValueError):
                t.regular(f.root, 'worker.py')
        else:
            with self.assertRaises(ValueError):
                t.inventory(f.root)

    def test_unknown_activity_stays_unknown(self):
        f = self.fixture(None)
        r = t.terminal_evidence(f.root, t.inventory(f.root), f.sha, f.protocol)
        self.assertTrue(r['activity_unknown_without_worker_receipt'])
        self.assertIsNone(r['completed_training_updates'])
        self.assertIsNone(r['optimizer_inflight'])
        self.assertIsNone(r['worker_recorded_counts'])

    def test_partial_terminal_JSON_preserved_and_counts_unknown(self):
        f = self.fixture(None)
        path = f.root/t.OUTPUTS/'failure.json'
        value = put(path, b'{"complete":true,"completed_training_updates":')
        r = t.terminal_evidence(f.root, t.inventory(f.root), f.sha, f.protocol)
        self.assertTrue(r['activity_unknown_without_worker_receipt'])
        self.assertIsNone(r['completed_training_updates'])
        self.assertIsNone(r['worker_recorded_counts'])
        self.assertEqual(r['malformed_worker_terminal_preserved']['sha256'], value['sha256'])
        self.assertFalse(r['complete_training_process'])

    def test_selfcontained_weight_category_exact_and_not_metadata(self):
        f = self.fixture()
        p = deepcopy(f.protocol)
        p['weight_assets'].pop(next(iter(p['weight_assets'])))
        with self.assertRaises(ValueError):
            t._bindings_from_protocol(p)
        with mock.patch.dict(t.LIMITS, {'metadata_bytes': 0}):
            pins, metadata = t.measure_files(f.root, list(self.weight_payloads))
        self.assertEqual(metadata, 0)
        self.assertEqual(set(pins), set(self.weight_payloads))

    def test_actual_failed_counter_receipt(self):
        f = self.fixture()
        r = t.terminal_evidence(f.root, t.inventory(f.root), f.sha, f.protocol)
        self.assertTrue(r['failure_present'])
        self.assertFalse(r['complete_training_process'])
        self.assertEqual(r['completed_training_updates'], 3)
        self.assertFalse(r['model_qualified'])

    def test_boolean_counters_and_inconsistent_commit_refused(self):
        for change in ('bool', 'overflow', 'commit'):
            f = self.fixture()
            path = f.root/t.OUTPUTS/'failure.json'
            value = t.read_json(path)
            if change == 'bool':
                value['counts']['backward_calls'] = True
            elif change == 'overflow':
                value['counts']['gradient_queries'] = 8
            else:
                value['completed_training_updates'] = 4
            put(path, t.json_bytes(value))
            with self.subTest(change=change), self.assertRaises(ValueError):
                t.terminal_evidence(f.root, t.inventory(f.root), f.sha, f.protocol)

    def test_success_needs_full_three_states_and_required_coverage(self):
        f = self.fixture('success')
        with self.assertRaises(ValueError):
            t.terminal_evidence(f.root, t.inventory(f.root), f.sha, f.protocol)
        for n in t.CHECKPOINT_PATHS[:3]:
            put(f.root/n, b'opaque state')
        path = f.root/t.OUTPUTS/'run_results.json'
        value = t.read_json(path)
        value['state_checkpoint'] = {'path': t.CHECKPOINT_PATHS[2], **t.file_pin(f.root/t.CHECKPOINT_PATHS[2])}
        value['state_checkpoint_complete'] = True
        put(path, t.json_bytes(value))
        r = t.terminal_evidence(f.root, t.inventory(f.root), f.sha, f.protocol)
        self.assertTrue(r['complete_training_process'])

    def test_quality_failure_after151_references_retained_final_without_duplicate(self):
        f = self.fixture('success')
        path = f.root/t.OUTPUTS/'run_results.json'
        value = t.read_json(path)
        path.unlink()
        for name in t.CHECKPOINT_PATHS[:3]:
            put(f.root/name, b'opaque committed state')
        value.update(failure=True, state_checkpoint={'path': t.CHECKPOINT_PATHS[2],
            **t.file_pin(f.root/t.CHECKPOINT_PATHS[2])}, state_checkpoint_complete=True)
        put(f.root/t.OUTPUTS/'failure.json', t.json_bytes(value))
        result = t.terminal_evidence(f.root, t.inventory(f.root), f.sha, f.protocol)
        self.assertTrue(result['failure_present'])
        self.assertFalse(result['complete_training_process'])
        self.assertEqual(result['state_checkpoint']['path'], t.CHECKPOINT_PATHS[2])
        t.checkpoint_accounting({n: 1 for n in t.CHECKPOINT_PATHS[:3]})

    def test_partial_checkpoint_is_retained_but_not_completed_state(self):
        f = self.fixture()
        temp = t.CHECKPOINT_PATHS[3]+'.tmp-'+'a'*32
        pin = put(f.root/temp, b'partial torch-save zip')
        path = f.root/t.OUTPUTS/'failure.json'
        value = t.read_json(path)
        value.update(optimizer_inflight=True, state_checkpoint={'path': temp, **pin},
            state_checkpoint_complete=False)
        put(path, t.json_bytes(value))
        result = t.terminal_evidence(f.root, t.inventory(f.root), f.sha, f.protocol)
        self.assertFalse(result['state_checkpoint_complete'])
        value['state_checkpoint_complete'] = True
        put(path, t.json_bytes(value))
        with self.assertRaises(ValueError):
            t.terminal_evidence(f.root, t.inventory(f.root), f.sha, f.protocol)

    def test_checkpoint_cap_temp_and_duplicate_stopped(self):
        initial, mid, final, stopped = t.CHECKPOINT_PATHS
        self.assertEqual(t.checkpoint_kind(final), 'final')
        temp = final+'.tmp-'+'a'*32
        self.assertEqual(t.checkpoint_kind(temp), 'temporary')
        r = t.checkpoint_accounting({initial: 1, mid: 1, temp: 1, stopped: 1})
        self.assertEqual(r['temporary_bytes'], 1)
        for sizes in ({final: 1, stopped: 1}, {mid: 1, mid+'.tmp-'+'b'*32: 1},
                      {final: t.LIMITS['trained_checkpoint_bytes']+1}):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                t.checkpoint_accounting(sizes)

    def test_arbitrary_checkpoint_outside_exact_slots_refused(self):
        f = self.fixture()
        n = t.OUTPUTS+'/arbitrary.pth'
        put(f.root/n, b'opaque')
        with self.assertRaises(ValueError):
            t.measure_files(f.root, [n])
        p = deepcopy(f.protocol)
        p['allowed_output_names'].append(n)
        with self.assertRaises(ValueError):
            t.allowed_output_names(p)

    def test_canonical_archive_roundtrip_and_exclusive_inert_import(self):
        f = self.fixture()
        archive, expected = self.archive(f.base, {'a/data.npy': b'opaque array', 'b/checkpoint.pt': b'opaque model'})
        with mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*t.GiB)):
            receipt = t.verify_archive(archive, expected, extract_to=f.base/'inert')
            self.assertTrue(receipt['complete'])
            self.assertFalse(receipt['returned_code_or_tensors_executed'])
            with self.assertRaises(ValueError):
                t.verify_archive(archive, expected, extract_to=f.base/'inert')
        self.assertEqual((f.base/'inert/a/data.npy').read_bytes(), b'opaque array')

    def test_corrupt_payload_padding_extra_frame_trailing_and_hash_refused(self):
        for kind in ('padding', 'frame', 'trailing', 'hash', 'truncated'):
            f = self.fixture()
            archive, expected = self.archive(f.base, {'x.txt': b'abc'},
                mutate_padding=kind == 'padding', extra_frame=kind == 'frame')
            if kind == 'trailing':
                archive.write_bytes(archive.read_bytes()+b'junk')
            elif kind == 'hash':
                expected['x.txt']['sha256'] = '0'*64
            elif kind == 'truncated':
                archive.write_bytes(archive.read_bytes()[:-8])
            with self.subTest(kind=kind), self.assertRaises((ValueError, EOFError, zlib.error)):
                t.verify_archive(archive, expected)

    def test_physical_order_and_noncanonical_header_refused(self):
        f = self.fixture()
        archive, expected = self.archive(f.base, {'a.txt': b'a', 'b.txt': b'b'})
        raw = bytearray(gzip.decompress(archive.read_bytes()))
        raw[100:108] = b'0000755\0'
        archive.write_bytes(gzip.compress(raw, mtime=0))
        with self.assertRaises(ValueError):
            t.verify_archive(archive, expected)

    def test_cooperative_deadline_interrupts_physical_readback_preserving_archive(self):
        f = self.fixture()
        archive, expected = self.archive(f.base, {'a.bin': b'x'*1024, 'b.bin': b'y'*1024})
        before = archive.read_bytes()
        calls = []
        def budget():
            calls.append(len(calls)+1)
            if len(calls) == 6:
                raise TimeoutError('fixture finite physical readback budget')
        with self.assertRaises(TimeoutError):
            t.verify_archive(archive, expected, check=budget)
        self.assertEqual(archive.read_bytes(), before)
        self.assertEqual(len(calls), 6)

    def test_decoder_fill_and_final_hash_receive_deadline_callback(self):
        f = self.fixture()
        archive, expected = self.archive(f.base, {'a.txt': b'payload'})
        events = []
        receipt = t.verify_archive(archive, expected, check=lambda: events.append('check'))
        self.assertTrue(receipt['complete'])
        self.assertGreaterEqual(len(events), 9)
        decoder = t.StrictGzip(archive, 10240,
            check=lambda: (_ for _ in ()).throw(TimeoutError('before decode')))
        try:
            with self.assertRaises(TimeoutError):
                decoder.exact(512)
        finally:
            decoder.close()

    def test_LF_checksum_and_single_remote_source_commands(self):
        blob = t.checksum_bytes('sample.tar.gz', 'a'*64)
        self.assertTrue(blob.endswith(b'\n'))
        self.assertNotIn(b'\r', blob)
        guide = t.command_guide('b'*64, 'c'*64)
        self.assertTrue(guide['manual_only'])
        commands = guide['download_Windows_SDK_Shell_one_remote_source_per_command'][1:]
        self.assertEqual(len(commands), 4)
        self.assertTrue(all(v.count('janusdominic0@forensic-dgp-thesis:') == 1 for v in commands))
        self.assertIn('tmux', guide['tmux_VM_SSH'])

    def test_duplicate_JSON_nonfinite_and_framework_absence(self):
        for payload in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(ValueError):
                t.parse_json(payload)
        self.assertFalse(any(n in sys.modules for n in ('torch', 'tensorflow', 'jax')))


import zlib
if __name__ == '__main__':
    unittest.main()
