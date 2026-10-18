"""Pure invariants, pin-shape, exact-parity and instrumentation fixtures; no NN."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('path_isolation', ROOT / 'scripts/run_codeformer_filter_path_isolation_v1.py')
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


class FakeHookSite:
    def __init__(self):
        self.callback = None; self.removed = False

    def register_forward_pre_hook(self, callback, with_kwargs):
        if with_kwargs is not True:
            raise ValueError('Expected kwargs observation')
        self.callback = callback
        return SimpleNamespace(remove=lambda: setattr(self, 'removed', True))


class PathIsolationPureFixtures(unittest.TestCase):
    def test_final_exact_protocol_all_dependency_shapes_without_NN(self):
        self.assertNotEqual(m.PROTOCOL_SHA, 'PENDING_FINAL_PROTOCOL')
        path = ROOT / 'outputs/codeformer_filter_path_isolation_v1_preparation/protocol_v1.json'
        protocol = m.H.json_read(m.H.bound(path, m.PROTOCOL_SHA))
        bindings, runtime, inputs, cached = m.preflight(protocol)
        self.assertEqual(len(bindings['models']), 3)
        self.assertEqual(len(m.shared_decoder_names(bindings['full515_tensor_schema'])), 165)
        self.assertTrue(runtime['torch'])
        self.assertEqual(len(inputs), 5)
        self.assertEqual(len(cached), 15)
        self.assertNotIn('torch', sys.modules)

    def test_shared_decoder_schema_is_exact165_not_arbitrary_parameter_subset(self):
        schema = {**{'generator.test%d' % n: {} for n in range(164)}, 'quantize.embedding.weight': {}, 'encoder.other': {}}
        self.assertEqual(len(m.shared_decoder_names(schema)), 165)
        del schema['generator.test0']
        with self.assertRaisesRegex(ValueError, '164 decoder'):
            m.shared_decoder_names(schema)

    def test_own_caps_do_not_modify_immutable_previous_worker(self):
        previous = dict(m.H.CAPS)
        self.assertEqual(m.H.CAPS['wall_seconds'], 3600)
        self.assertEqual(m.CAPS['wall_seconds'], 600)
        self.assertEqual(m.CAPS['new_forward_calls'], 48)
        self.assertEqual(m.CAPS['journal_records'], 180)
        values = {'process_RSS_bytes': 1024**3, 'available_RAM_bytes': 2 * 1024**3, 'disk_free_bytes': 2 * 1024**3}
        m.enforce(values, 600)
        with self.assertRaisesRegex(ValueError, '600'):
            m.enforce(values, 600.001)
        self.assertEqual(m.H.CAPS, previous)

    def test_all_resource_boundaries_are_hard_stops(self):
        good = {'process_RSS_bytes': 1024**3, 'available_RAM_bytes': 2 * 1024**3, 'disk_free_bytes': 2 * 1024**3}
        for key, value in (('process_RSS_bytes', m.CAPS['process_RSS_bytes'] + 1),
            ('available_RAM_bytes', m.CAPS['minimum_available_RAM_bytes'] - 1),
            ('disk_free_bytes', m.CAPS['minimum_disk_free_bytes'] - 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.enforce(dict(good, **{key: value}), 0)

    def test_readonly_hooks_count_exact_top_and_four_fusion_sites(self):
        net = FakeHookSite(); net.fuse_convs_dict = {key: FakeHookSite() for key in m.FUSIONS}
        observer = m.Observer(net)
        for setting in (*m.SETTINGS, m.BASELINE):
            observer.begin(); net.callback(net, (), {**setting, 'detach_16': True})
            if setting['w'] == 1:
                for site in m.FUSIONS:
                    net.fuse_convs_dict[site].callback(None, (), {})
            m.check_observer(observer.current, setting)
        self.assertEqual(observer.total_top, 4)
        self.assertEqual(observer.total_fusion, dict.fromkeys(m.FUSIONS, 2))
        observer.remove()
        self.assertTrue(net.removed and all(site.removed for site in net.fuse_convs_dict.values()))

    def test_w0_cannot_silently_run_any_fusion_or_extra_top_forward(self):
        setting = m.SETTINGS[0]
        observer = {'top_level_calls': 1, 'top_level_settings': [{'w': 0., 'adain': False, 'detach_16': True}],
            'fusion_calls': dict.fromkeys(m.FUSIONS, 0), 'fusion_order': []}
        bad = copy.deepcopy(observer); bad['fusion_calls']['32'] = 1
        with self.assertRaisesRegex(ValueError, 'fusion'):
            m.check_observer(bad, setting)
        bad = copy.deepcopy(observer); bad['top_level_calls'] = 2
        with self.assertRaisesRegex(ValueError, 'top-level'):
            m.check_observer(bad, setting)

    def test_recomputed_token_record_exact_shape_values_and_lookup_caveat(self):
        tokens = np.arange(256, dtype=np.int64)[None].copy()
        record = m.describe_tokens(tokens)
        self.assertEqual(record['shape'], [1, 256])
        self.assertEqual(record['topk_native_shape'], [1, 256, 1])
        self.assertEqual(record['indices'], list(range(256)))
        self.assertEqual(record['array_sha256'], hashlib.sha256(tokens.tobytes()).hexdigest())
        self.assertFalse(record['actual_internal_codebook_lookup_captured'])
        for bad in (tokens[..., None], tokens.astype(np.int32), tokens + 1024):
            with self.assertRaises(ValueError):
                m.describe_tokens(bad)

    def test_one_ULP_parity_failure_is_not_hidden_by_identical_PNG(self):
        cached = np.full((256, 256, 3), .4, np.float32); raw = cached.copy()
        raw[0, 0, 0] = np.nextafter(raw[0, 0, 0], np.float32(1))
        check = m.exact_parity(raw, m.H.delivered(raw), cached, m.H.delivered(cached))
        self.assertFalse(check['RAW_exact']); self.assertTrue(check['PNG_exact'])
        self.assertGreater(check['RAW_maximum_error'], 0)

    def test_upstream_signature_ignores_delivery_but_never_changed_tokens(self):
        readout = {'logits': {'array_sha256': 'a'}, 'lq_feature': {'array_sha256': 'b'},
            'recomputed_tokens': {'array_sha256': 'c'}, 'top2_probability_margin': {'minimum': .1}}
        second = copy.deepcopy(readout); second['top2_probability_margin']['minimum'] = .2
        self.assertEqual(m.route_signature(readout), m.route_signature(second))
        second['recomputed_tokens']['array_sha256'] = 'different'
        self.assertNotEqual(m.route_signature(readout), m.route_signature(second))

    def test_watchdog_uses_own600second_check_and_own_exit_only(self):
        guard = m.Watchdog(SimpleNamespace(folder=ROOT / 'outputs'), {'reserved': 1})
        guard.stop = Mock(); guard.stop.wait.side_effect = [False, True]
        guard.check = Mock(side_effect=ValueError('own bounded stop')); guard.failure = Mock()
        with patch.object(m.os, '_exit') as own_exit:
            guard.loop()
        own_exit.assert_called_once_with(124)
        self.assertTrue(guard.failure.call_args.kwargs['hard'])

    def test_initial_resource_stop_retains_failure_before_framework_import(self):
        protocol = ROOT / 'outputs/codeformer_filter_path_isolation_v1_preparation/protocol_v1.json'
        with tempfile.TemporaryDirectory(prefix='path-initial-stop-', dir=ROOT / 'outputs') as folder:
            target = Path(folder) / 'partial'
            with patch.object(m, 'preflight', return_value=({}, {}, {}, {})), \
                patch.object(m.Watchdog, 'check', side_effect=ValueError('Available RAM reserve reached')), \
                self.assertRaisesRegex(ValueError, 'RAM reserve'):
                m.run(protocol, target)
            failure = json.loads((target / 'failure.json').read_text(encoding='utf-8'))
            journal = [json.loads(line) for line in (target / 'journal.jsonl').read_text().splitlines()]
            self.assertFalse(failure['complete'])
            self.assertTrue(failure['partial_preserved_do_not_resume'])
            self.assertEqual(failure['progress']['reserved'], 0)
            self.assertEqual(journal[0]['event'], 'failure')
            self.assertNotIn('torch', sys.modules)

    def test_durable_reserve_return_commit_and_exclusive_partial_namespace(self):
        with tempfile.TemporaryDirectory(prefix='path-output-', dir=ROOT / 'outputs') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(ROOT / 'outputs'))
            output = m.Output(Path(folder) / 'new')
            output.record({'event': 'reserve', 'call': 1})
            self.assertEqual(json.loads((output.folder / 'journal.jsonl').read_text())['event'], 'reserve')
            output.record({'event': 'returned', 'call': 1})
            output.put('artifact.bin', b'exact bytes'); output.record({'event': 'commit', 'call': 1})
            output.close_journal(); output.put('manifest.json', m.H.encoded({'complete': True}))
            self.assertEqual(len((output.folder / 'journal.jsonl').read_text().splitlines()), 3)
            self.assertEqual(output.bytes, sum(p.stat().st_size for p in output.folder.rglob('*') if p.is_file()))
            with self.assertRaisesRegex(ValueError, 'Exclusive'):
                m.Output(output.folder)


if __name__ == '__main__':
    unittest.main()
