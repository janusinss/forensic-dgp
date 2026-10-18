"""Independent scalar and tiny CPU fixture checks for finite preparation."""
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from research.codeformer_fp32_comparison_v1 import preparation_audit as audit


class PreparationAuditTests(unittest.TestCase):
    def test_full50_census_comes_from_fixed_cycle(self):
        counts = audit.first50_counts()
        self.assertEqual(len(counts['eligibility_by_order']), 200)
        self.assertEqual(counts['perceptual_training_forwards'], 148)
        self.assertEqual(sum(counts['eligibility_by_order'][:80]), 60)
        self.assertEqual(counts['candidate_training_forwards'], counts['backward_calls'])
        self.assertEqual(counts['cycles'][0], {'update': 1, 'A_edge': 24, 'B_variant': 'clear256'})
        self.assertEqual(counts['cycles'][23], {'update': 24, 'A_edge': 40, 'B_variant': 'light64'})
        self.assertEqual(counts['cycles'][24]['A_edge'], 24)

    def test_50_warmup_rates_do_not_use_50_cosine_horizon(self):
        self.assertAlmostEqual(audit.first50_lr(1, 'encoder_token'), 2e-7, places=20)
        self.assertEqual(audit.first50_lr(50, 'encoder_token'), 1e-5)
        self.assertAlmostEqual(audit.first50_lr(1, 'fusion'), 4e-7, places=20)
        self.assertEqual(audit.first50_lr(50, 'fusion'), 2e-5)
        for bad in (0, 51, True):
            with self.assertRaises(ValueError):
                audit.first50_lr(bad, 'fusion')

    def test_NPY_header_budget_matches_real_small_serialization(self):
        value = np.zeros((1, 3, 4, 4), dtype=np.float32)
        stream = io.BytesIO(); np.save(stream, value, allow_pickle=False)
        self.assertEqual(audit.npy_file_bytes(value.shape, value.dtype), len(stream.getvalue()))

    def test_finite_resource_envelope_includes_three_states_and_cache(self):
        ledger = audit.resource_ledger()
        self.assertGreater(ledger['payload_margin_bytes'], 0)
        self.assertGreater(ledger['archive_margin_bytes'], 0)
        self.assertGreater(ledger['startup_margin_bytes'], 0)
        self.assertEqual(ledger['components']['HQ_cache_391_index_feature_pairs'], 103399168)
        self.assertEqual(ledger['components']['teacher_anchor_arrays_50pairs'], 91763200)
        self.assertEqual(ledger['components']['exact_retained_behavior_target_arrays_50pairs'], 91763200)
        self.assertEqual(ledger['components']['TRAIN_actual_captured_tokens_200'], 435200)
        self.assertEqual(ledger['regular_file_count_plan']['per_call_trace_records'], 2057)
        self.assertLessEqual(ledger['planned_file_count_upper'], ledger['regular_file_count_cap'])
        self.assertFalse(ledger['extra_safe_checkpoint_copy_allowed'])
        self.assertTrue(ledger['last_safe_checkpoint_must_be_a_reference_not_fourth_serialized_file'])

    def test_fourth_full_state_would_break_prepared_payload_budget(self):
        ledger = audit.resource_ledger()
        self.assertGreater(ledger['conservative_payload_bytes'] + 832 * audit.MIB, ledger['payload_cap_bytes'])

    def test_strict_float_anchor_rejects_dtype_and_trailing_data(self):
        stream = io.BytesIO(); np.save(stream, np.zeros((1, 2), dtype=np.float32), allow_pickle=False)
        data = stream.getvalue()
        with patch.object(Path, 'open', side_effect=lambda *a, **k: io.BytesIO(data)):
            audit.array(Path('anchor.npy'), (1, 2))
            with self.assertRaises(ValueError):
                audit.array(Path('anchor.npy'), (2, 1))
        with patch.object(Path, 'open', side_effect=lambda *a, **k: io.BytesIO(data + b'trailing')):
            with self.assertRaises(ValueError):
                audit.array(Path('anchor.npy'), (1, 2))

    def test_pixel_hash_binds_dtype_and_byte_values(self):
        a = np.array([0, 1, 2], dtype=np.uint8)
        self.assertNotEqual(audit.pixel_sha(a), audit.pixel_sha(a.astype(np.float32)))
        b = a.copy(); b[2] = 3
        self.assertNotEqual(audit.pixel_sha(a), audit.pixel_sha(b))

    def test_only_half_of_all_200_exposures_are_normal(self):
        positions = [(u, s) for u in range(1, 51) for s in range(1, 5)]
        self.assertEqual(sum(s in (1, 3) for _, s in positions), 100)
        self.assertEqual(sum(s == 3 for _, s in positions), 50)
        self.assertEqual(sum(s in (1, 2) for _, s in positions), 100)


if __name__ == '__main__':
    unittest.main()
