"""CPU regression checks for matched-call parity and retained failure evidence.

These fixtures use NumPy arrays and ordinary Python descriptor stubs only.
They never import Torch, execute a neural model or construct an optimizer.
"""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_role_preserving_training_v1_r1 import worker as w


def contexts():
    candidate = {
        'module_modes': [['', False], ['encoder', False], ['generator', False]],
        'grad_enabled': False, 'inference_mode': False,
        'autocast_enabled': False, 'requires_grad_tensors': 350}
    parent = deepcopy(candidate)
    parent['requires_grad_tensors'] = 0
    return candidate, parent


def arrays():
    return {name: np.asarray([.25, .5, .75], dtype=np.float32)
            for name in ('native', 'raw', 'logits', 'latent')} | {
        'tokens': np.asarray([1, 7, 12], dtype=np.int64)}


class InitializerParityCPUFixtures(unittest.TestCase):
    def test_context_descriptor_reports_actual_modes_and_grad_state(self):
        class OrdinaryModule:
            def named_modules(self):
                return [('', SimpleNamespace(training=True)),
                        ('generator', SimpleNamespace(training=False))]

            def parameters(self):
                return [SimpleNamespace(requires_grad=x) for x in [True, False, True]]

        runtime = SimpleNamespace(is_grad_enabled=lambda: True,
            is_inference_mode_enabled=lambda: False, is_autocast_enabled=lambda: False)
        actual = w.actual_execution_context(OrdinaryModule(), runtime)
        self.assertEqual(actual['module_modes'], [['', True], ['generator', False]])
        self.assertIs(actual['grad_enabled'], True)
        self.assertIs(actual['inference_mode'], False)
        self.assertIs(actual['autocast_enabled'], False)
        self.assertEqual(actual['requires_grad_tensors'], 2)
        self.assertNotIn('torch', sys.modules)

    def test_inference_contexts_match_without_freezing_candidate(self):
        candidate, parent = contexts()
        self.assertTrue(w.validate_inference_contexts(candidate, parent))
        self.assertEqual(candidate['requires_grad_tensors'], 350)
        self.assertEqual(parent['requires_grad_tensors'], 0)

    def test_each_execution_context_mismatch_is_rejected(self):
        for field in ('grad_enabled', 'inference_mode', 'autocast_enabled'):
            for side in (0, 1):
                with self.subTest(field=field, side=side):
                    values = list(contexts()); values[side][field] = True
                    with self.assertRaises(ValueError):
                        w.validate_inference_contexts(*values)
        for side in (0, 1):
            values = list(contexts()); values[side]['module_modes'][1][1] = True
            with self.assertRaises(ValueError): w.validate_inference_contexts(*values)
        candidate, parent = contexts(); parent['module_modes'][1][0] = 'different_encoder'
        with self.assertRaises(ValueError): w.validate_inference_contexts(candidate, parent)
        candidate, parent = contexts(); candidate['requires_grad_tensors'] = 0
        with self.assertRaises(ValueError): w.validate_inference_contexts(candidate, parent)
        candidate, parent = contexts(); del parent['autocast_enabled']
        with self.assertRaises(ValueError): w.validate_inference_contexts(candidate, parent)

    def test_single_float32_ulp_is_measured_and_never_tolerated(self):
        candidate, parent = arrays(), arrays()
        candidate['latent'][1] = np.nextafter(np.float32(.5), np.float32(1.))
        differences = w.capture_differences(candidate, parent, np)
        self.assertFalse(differences['latent']['exact'])
        self.assertEqual(differences['latent']['different_elements'], 1)
        self.assertEqual(differences['latent']['maximum_absolute_difference'], 2**-24)
        self.assertTrue(differences['latent']['both_arrays_finite'])
        self.assertTrue(all(row['exact'] for key, row in differences.items() if key != 'latent'))

    def test_success_writes_one_exact_report_before_return(self):
        candidate, parent = contexts(); written = []
        result = w.record_exact_parity(arrays(), arrays(), candidate, parent, np, written.append)
        self.assertEqual(written, [result])
        self.assertTrue(result['pass'])
        self.assertFalse(result['exact_gate_relaxed'])
        self.assertTrue(result['evidence_written_before_assertion'])

    def test_failed_float_parity_keeps_difference_before_raising(self):
        candidate, parent = arrays(), arrays(); written = []
        candidate['raw'][0] = np.nextafter(candidate['raw'][0], np.float32(1.))
        with self.assertRaisesRegex(ValueError, 'actual arrays and context retained'):
            w.record_exact_parity(candidate, parent, *contexts(), np, written.append)
        self.assertEqual(len(written), 1)
        self.assertFalse(written[0]['pass'])
        self.assertEqual(written[0]['differences']['raw']['different_elements'], 1)
        self.assertIsNone(written[0]['context_error'])

    def test_failed_context_keeps_report_even_when_arrays_match(self):
        candidate, parent = contexts(); candidate['grad_enabled'] = True; written = []
        with self.assertRaises(ValueError):
            w.record_exact_parity(arrays(), arrays(), candidate, parent, np, written.append)
        self.assertEqual(len(written), 1)
        self.assertFalse(written[0]['pass'])
        self.assertIsInstance(written[0]['context_error'], str)
        self.assertTrue(all(row['exact'] for row in written[0]['differences'].values()))

    def test_matching_infinities_are_retained_and_rejected(self):
        for invalid in (np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                candidate, parent = arrays(), arrays(); written = []
                candidate['native'][0] = parent['native'][0] = invalid
                with self.assertRaises(ValueError):
                    w.record_exact_parity(candidate, parent, *contexts(), np, written.append)
                self.assertEqual(len(written), 1)
                self.assertFalse(written[0]['pass'])
                self.assertTrue(written[0]['differences']['native']['exact'])
                self.assertFalse(written[0]['differences']['native']['both_arrays_finite'])
                self.assertIsNone(written[0]['differences']['native']['maximum_absolute_difference'])

    def test_wrong_shape_and_dtype_do_not_broadcast_into_pass(self):
        for replacement in (np.asarray([[1, 7, 12]], np.int64), np.asarray([1, 7, 12], np.int32)):
            with self.subTest(shape=replacement.shape, dtype=str(replacement.dtype)):
                candidate, parent = arrays(), arrays(); candidate['tokens'] = replacement; written = []
                with self.assertRaises(ValueError):
                    w.record_exact_parity(candidate, parent, *contexts(), np, written.append)
                self.assertEqual(len(written), 1)
                self.assertFalse(written[0]['pass'])
                self.assertIsNone(written[0]['differences']['tokens']['different_elements'])

    def test_capture_duplicate_dictionary_error_restores_method_and_hook(self):
        class Token:
            def detach(self): return self
            def clone(self): return 'actual-token-indices'

        class OrdinaryCaptureStub:
            def __init__(self):
                self.quantize = SimpleNamespace(get_codebook_feat=lambda indices, shape: 'lookup')
                self.hooks = []

            def register_forward_hook(self, fn):
                self.hooks.append(fn)
                return SimpleNamespace(remove=lambda: self.hooks.remove(fn))

        module = OrdinaryCaptureStub(); original = module.quantize.get_codebook_feat
        with self.assertRaisesRegex(ValueError, 'Exactly one candidate dictionary lookup'):
            with w.capture_forward(module) as captured:
                self.assertEqual(module.quantize.get_codebook_feat(Token(), [1]), 'lookup')
                self.assertEqual(captured['tokens'], 'actual-token-indices')
                module.quantize.get_codebook_feat(Token(), [1])
        self.assertIs(module.quantize.get_codebook_feat, original)
        self.assertEqual(module.hooks, [])
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
