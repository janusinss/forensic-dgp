"""Pure CPU preparation/guard/serialization tests; no torch or model calls."""
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
SPEC = importlib.util.spec_from_file_location('frozen_comparison', ROOT / 'scripts/run_codeformer_filter_comparison_v1.py')
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def actual_plan():
    data = m.json_read(m.bound(ROOT / 'outputs/codeformer_filter_pairs_v1_data/manifest.json', m.DATA_SHA))
    review = m.json_read(m.bound(ROOT / 'outputs/codeformer_filter_pairs_v1_preparation/root_input_only_review_v1.json', m.REVIEW_SHA))
    return data, review


class ComparisonPureFixtures(unittest.TestCase):
    def test_actual_inert_metadata_pins_and96_plan_without_framework(self):
        data, review = actual_plan(); plan = m.plan_cases(data, review)
        self.assertEqual(sum(r['inference_planned'] for r in plan), 72)
        self.assertEqual(sum(not r['inference_planned'] for r in plan), 24)
        self.assertEqual(sum(r['appearance_review_required'] for r in plan), 12)
        self.assertTrue(all(not r['outputs'] for r in plan))
        m.bound(ROOT / 'outputs/codeformer_filter_pairs_v1_preparation/input_only_review_criteria_v1.json', m.CRITERIA_SHA)
        m.bound(ROOT / 'outputs/codeformer_filter_comparison_v1_preparation/model_binding_v1.json', m.MODELS_SHA)
        self.assertNotIn('torch', sys.modules)

    def test_no_automatic_inference_on_insufficient_input(self):
        data, review = actual_plan()
        altered = copy.deepcopy(review); altered['cases'][0]['decision'] = 'readable'
        with self.assertRaisesRegex(ValueError, 'Exact96'):
            m.plan_cases(data, altered)

    def test_DEV_remains_excluded_even_for_readable_inputs(self):
        data, review = actual_plan()
        row = next(r for r in data['cases'] if r['split'] == 'DEV')
        row['optimization_allowed_by_split'] = True
        with self.assertRaisesRegex(ValueError, 'optimization'):
            m.plan_cases(data, review)

    def test_numpy_FP32_input_matches_explicit_division_policy(self):
        image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
        canonical = m.canonical_input(image)
        self.assertEqual(canonical.dtype, np.float32)
        self.assertTrue(canonical.flags.c_contiguous)
        self.assertEqual(float(canonical[0, 0, 1]), float(np.float32(1) / np.float32(255)))
        self.assertFalse(np.shares_memory(canonical, image))

    def test_floor_output_has_no_rounding_or_enhancement(self):
        raw = np.zeros((256, 256, 3), np.float32)
        raw[0, 0] = [1., .5, 1. / 255]
        self.assertEqual(m.delivered(raw)[0, 0].tolist(), [255, 127, 1])
        raw[1, 1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'Finite'):
            m.delivered(raw)

    def test_resource_guard_all_four_boundaries(self):
        good = {'process_RSS_bytes': 1024**3, 'available_RAM_bytes': 2 * 1024**3, 'disk_free_bytes': 3 * 1024**3}
        m.enforce_resources(good, 0)
        bad = dict(good, process_RSS_bytes=m.CAPS['process_RSS_bytes'] + 1)
        with self.assertRaisesRegex(ValueError, 'RSS'):
            m.enforce_resources(bad, 0)
        bad = dict(good, available_RAM_bytes=m.CAPS['minimum_available_RAM_bytes'] - 1)
        with self.assertRaisesRegex(ValueError, 'RAM'):
            m.enforce_resources(bad, 0)
        bad = dict(good, disk_free_bytes=m.CAPS['minimum_disk_free_bytes'] - 1)
        with self.assertRaisesRegex(ValueError, 'Disk'):
            m.enforce_resources(bad, 0)
        with self.assertRaisesRegex(ValueError, 'wall'):
            m.enforce_resources(good, m.CAPS['wall_seconds'] + .01)

    def test_live_resource_reader_without_any_neural_calls(self):
        values = m.resource_values(ROOT / 'outputs')
        self.assertEqual(set(values), {'process_RSS_bytes', 'available_RAM_bytes', 'disk_free_bytes'})
        self.assertTrue(all(type(value) is int and value > 0 for value in values.values()))
        self.assertNotIn('torch', sys.modules)

    def test_watchdog_persists_stop_then_exits_only_own_worker(self):
        guard = m.Watchdog(SimpleNamespace(folder=ROOT / 'outputs'), {'reserved_forwards': 1})
        guard.stop = Mock(); guard.stop.wait.side_effect = [False, True]
        guard.check = Mock(side_effect=ValueError('mock resource stop'))
        guard.failure = Mock()
        with patch.object(m.os, '_exit') as own_exit:
            guard.loop()
        self.assertEqual(guard.failure.call_count, 1)
        self.assertTrue(guard.failure.call_args.kwargs['watchdog'])
        own_exit.assert_called_once_with(124)
        self.assertNotIn('torch', sys.modules)

    def test_source_mutation_and_path_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='comparison-source-', dir=ROOT / 'scratch') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(ROOT / 'scratch'))
            path = Path(folder) / 'input.bin'; path.write_bytes(b'original')
            fingerprint = hashlib.sha256(b'original').hexdigest()
            self.assertEqual(m.bound(path, fingerprint, 8), b'original')
            path.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                m.bound(path, fingerprint, 8)
        for name in ('../x', '/absolute', 'C:/x', 'dir\\x'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Canonical'):
                m.canonical(name)

    def test_reserve_commit_journal_is_durable_and_partial_preserved(self):
        with tempfile.TemporaryDirectory(prefix='comparison-output-', dir=ROOT / 'outputs') as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(ROOT / 'outputs'))
            output = m.Output(Path(folder) / 'new')
            output.record({'event': 'forward_reserved', 'call': 1})
            journal = output.folder / 'journal.jsonl'
            self.assertEqual(json.loads(journal.read_text()), {'event': 'forward_reserved', 'call': 1})
            self.assertFalse((output.folder / 'manifest.json').exists())
            output.put('raw-evidence.bin', b'committed bytes')
            output.record({'event': 'output_committed', 'call': 1})
            output.close_journal()
            output.put('manifest.json', m.encoded({'complete': True}))
            self.assertEqual(len(journal.read_text().splitlines()), 2)
            self.assertEqual(output.bytes, sum(p.stat().st_size for p in output.folder.rglob('*') if p.is_file()))
            with self.assertRaisesRegex(ValueError, 'New output namespace'):
                m.Output(output.folder)


if __name__ == '__main__':
    unittest.main()
