"""CPU adversarial fixtures for the declared cross-object initializer repair.

Only saved arrays and plain Python descriptors are used. No Torch import,
network, gradient, optimizer or VM operation is executed.
"""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_role_preserving_training_v1_r2 import contract as c, worker as w


def contexts():
    candidate = {'module_modes': [['', False], ['encoder', False], ['generator', False]],
        'grad_enabled': False, 'inference_mode': False, 'autocast_enabled': False,
        'requires_grad_tensors': 350}
    parent = deepcopy(candidate); parent['requires_grad_tensors'] = 0
    return candidate, parent


def captured():
    logits = np.full((1, 256, 1024), -4., dtype=np.float32)
    logits[:, :, 0] = 2.; logits[:, :, 1] = .5
    return {'native': np.zeros((1, 3, 512, 512), dtype=np.float32),
        'raw': np.zeros((256, 256, 3), dtype=np.float32),
        'latent': np.zeros((1, 256, 16, 16), dtype=np.float32),
        'logits': logits, 'tokens': np.zeros(256, dtype=np.int64)}


class InitializerNumericsCPUFixtures(unittest.TestCase):
    def record(self, actual, expected, values=None):
        written = []
        result = w.record_initializer_parity(actual, expected, *(values or contexts()), np, written.append)
        self.assertEqual(written, [result])
        return result

    def reject(self, actual, expected, values=None):
        written = []
        with self.assertRaisesRegex(ValueError, 'actual arrays and context retained'):
            w.record_initializer_parity(actual, expected, *(values or contexts()), np, written.append)
        self.assertEqual(len(written), 1)
        self.assertFalse(written[0]['pass'])
        self.assertTrue(written[0]['evidence_written_before_assertion'])
        return written[0]

    def test_policy_declares_official_FP32_defaults_and_narrow_scope(self):
        self.assertEqual(c.INITIALIZER_PARITY['logits_rtol'], 1.3e-6)
        self.assertEqual(c.INITIALIZER_PARITY['logits_atol'], 1e-5)
        self.assertEqual(c.INITIALIZER_PARITY['logits_close_source'], 'https://docs.pytorch.org/docs/2.9/testing.html')
        self.assertTrue(c.INITIALIZER_PARITY['numerical_initializer_gate_changed'])
        self.assertFalse(c.INITIALIZER_PARITY['restoration_quality_gates_changed'])
        self.assertEqual(c.INITIALIZER_PARITY['exact_arrays'], ['native', 'raw', 'latent', 'tokens'])

    def test_exact_values_with_robust_margin_pass(self):
        result = self.record(captured(), captured())
        self.assertTrue(result['pass'])
        self.assertEqual(result['numerical_logits']['out_of_bound_elements'], 0)
        self.assertEqual(result['numerical_logits']['unstable_tokens'], 0)
        self.assertFalse(result['quality_gates_changed'])
        self.assertFalse(result['scoped_training_exact_gate_changed'])

    def test_one_ULP_logits_allowed_with_exact_decoded_values_and_stable_tokens(self):
        actual, expected = captured(), captured()
        actual['logits'][0, 0, 1] = np.nextafter(np.float32(.5), np.float32(1.))
        result = self.record(actual, expected)
        self.assertFalse(result['differences']['logits']['exact'])
        self.assertTrue(result['numerical_logits']['pass'])
        self.assertTrue(all(result['differences'][name]['exact'] for name in ('native', 'raw', 'latent', 'tokens')))

    def test_single_ULP_each_decoded_or_latent_array_still_rejected(self):
        for name in ('native', 'raw', 'latent'):
            with self.subTest(array=name):
                actual, expected = captured(), captured()
                actual[name].flat[0] = np.nextafter(np.float32(0.), np.float32(1.))
                result = self.reject(actual, expected)
                self.assertTrue(result['numerical_logits']['pass'])
                self.assertEqual(result['differences'][name]['different_elements'], 1)

    def test_close_logits_with_changed_argmax_rejected(self):
        actual, expected = captured(), captured()
        for item in (actual, expected): item['logits'][0, 0, :2] = [np.float32(1e-6), 0.]
        actual['logits'][0, 0, 0] = np.float32(-1e-6)
        actual['tokens'][0] = 1
        result = self.reject(actual, expected)
        self.assertEqual(result['numerical_logits']['out_of_bound_elements'], 0)
        self.assertFalse(result['numerical_logits']['argmax_matches_saved_tokens'])

    def test_close_logits_unchanged_argmax_but_insufficient_margin_rejected(self):
        actual, expected = captured(), captured()
        for item in (actual, expected): item['logits'][0, 0, :2] = [np.float32(1e-6), 0.]
        actual['logits'][0, 0, 2] = np.float32(-4. + 1e-5)
        result = self.reject(actual, expected)
        self.assertEqual(result['numerical_logits']['out_of_bound_elements'], 0)
        self.assertTrue(result['numerical_logits']['argmax_matches_saved_tokens'])
        self.assertEqual(result['numerical_logits']['unstable_tokens'], 1)

    def test_margin_equal_twice_error_fails_strict_certificate(self):
        actual, expected = captured(), captured()
        for item in (actual, expected):
            item['logits'][0, 0, :3] = [np.float32(2**-20), 0., -1.]
        actual['logits'][0, 0, 2] += np.float32(2**-21)
        result = self.reject(actual, expected)
        self.assertEqual(result['numerical_logits']['minimum_margin_minus_twice_error'], 0.)
        self.assertEqual(result['numerical_logits']['unstable_tokens'], 1)

    def test_absolute_tolerance_boundary_accepts_below_rejects_nextabove(self):
        actual, expected = captured(), captured()
        expected['logits'][0, 0, 2] = actual['logits'][0, 0, 2] = np.float32(0.)
        actual['logits'][0, 0, 2] = np.float32(1e-5)
        self.assertTrue(self.record(actual, expected)['pass'])
        actual['logits'][0, 0, 2] = np.nextafter(np.float32(1e-5), np.float32(1.))
        result = self.reject(actual, expected)
        self.assertEqual(result['numerical_logits']['out_of_bound_elements'], 1)

    def test_relative_tolerance_uses_parent_reference_and_rejects_beyond(self):
        actual, expected = captured(), captured()
        expected['logits'][0, 0, 2] = -20.
        actual['logits'][0, 0, 2] = np.float32(-20. + 3e-5)
        self.assertTrue(self.record(actual, expected)['pass'])
        actual['logits'][0, 0, 2] = np.float32(-20. + 4e-5)
        result = self.reject(actual, expected)
        self.assertEqual(result['numerical_logits']['out_of_bound_elements'], 1)

    def test_saved_tokens_must_match_each_actual_logit_argmax(self):
        actual, expected = captured(), captured()
        actual['tokens'][0] = expected['tokens'][0] = 1
        result = self.reject(actual, expected)
        self.assertFalse(result['numerical_logits']['argmax_matches_saved_tokens'])

    def test_tied_parent_scores_reject_even_when_all_arrays_exact(self):
        actual, expected = captured(), captured()
        actual['logits'][0, 0, 1] = expected['logits'][0, 0, 1] = 2.
        result = self.reject(actual, expected)
        self.assertEqual(result['numerical_logits']['unstable_tokens'], 1)

    def test_nonfinite_logits_or_decoded_values_reject_before_acceptance(self):
        for name in ('logits', 'native', 'raw', 'latent'):
            for invalid in (np.inf, -np.inf, np.nan):
                with self.subTest(array=name, invalid=invalid):
                    actual, expected = captured(), captured()
                    actual[name].flat[0] = expected[name].flat[0] = invalid
                    result = self.reject(actual, expected)
                    self.assertFalse(result['pass'])

    def test_actual_geometry_and_dtypes_required_even_if_both_sides_wrong(self):
        for name in ('logits', 'tokens', 'native', 'raw', 'latent'):
            for kind in ('shape', 'dtype'):
                with self.subTest(array=name, invalid=kind):
                    actual, expected = captured(), captured()
                    value = expected[name]
                    replacement = (value.reshape(-1) if kind == 'shape' else
                        value.astype(np.int32 if name == 'tokens' else np.float64))
                    if replacement.shape == value.shape: replacement = replacement.reshape(1, -1)
                    actual[name] = replacement.copy(); expected[name] = replacement.copy()
                    result = self.reject(actual, expected)
                    self.assertFalse(result['pass'])

    def test_mismatched_execution_context_rejects_with_array_evidence(self):
        for field in ('grad_enabled', 'inference_mode', 'autocast_enabled'):
            with self.subTest(field=field):
                values = list(contexts()); values[0][field] = True
                result = self.reject(captured(), captured(), values)
                self.assertIsInstance(result['context_error'], str)

    def test_saved_R1_actual_arrays_pass_declared_numerical_contract(self):
        archive = ROOT / 'outputs/cctv-codeformer-role-preserving-training-v1-r1-results.tar.gz'
        with archive.open('rb') as stream:
            self.assertEqual(hashlib.file_digest(stream, 'sha256').hexdigest(),
                '075d0fcc9b6dbd35e1ba65050c7837c00f1bd64afb44164a2a45ae991f1ab2e1')
        stem = 'outputs/role_preserving_training_v1_r1/initializer_parity/train_u0003_slot1/'
        actual, expected = {}, {}
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                for target, prefix in ((actual, ''), (expected, 'parent/')):
                    for name in ('native', 'raw', 'logits', 'latent', 'tokens'):
                        if member.name == stem + prefix + name + '.npy':
                            self.assertTrue(member.isfile()); self.assertLess(member.size, 4*1024**2)
                            target[name] = np.load(io.BytesIO(tar.extractfile(member).read()),
                                allow_pickle=False, max_header_size=4096)
        result = self.record(actual, expected)
        self.assertFalse(result['differences']['logits']['exact'])
        self.assertEqual(result['differences']['logits']['different_elements'], 223123)
        self.assertEqual(result['numerical_logits']['out_of_bound_elements'], 0)
        self.assertEqual(result['numerical_logits']['unstable_tokens'], 0)
        self.assertAlmostEqual(result['numerical_logits']['maximum_bound_fraction'], .3754986185808142)
        self.assertNotIn('torch', sys.modules)

    def test_scoped_training_keeps_exact_forward_and_loss_validation(self):
        source = Path(w.__file__).read_text(encoding='utf-8')
        scoped = source.index("next_phase('witness')")
        adam = source.index('optimizer = torch.optim.Adam', scoped)
        witness = source[scoped:adam]
        self.assertIn('route.validate_forward_parity(reference, saved, np)', witness)
        self.assertIn("c.require(float(total.detach()) == initial_totals[row['id']]", witness)
        self.assertIn("c.require(scoped_context == initial_contexts[row['id']]", witness)
        self.assertNotIn('numerical_initializer_policy(', witness)
        self.assertNotIn('record_initializer_parity(', witness)


if __name__ == '__main__':
    unittest.main(verbosity=2)
