"""Proposal processing and request guards; these tests run no detector inference."""
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from implementation import covering_detector as detector
from implementation import covering_detector_worker as worker


class CoveringProposalTests(unittest.TestCase):
    def test_imports_load_no_torch_or_model(self):
        result = subprocess.run([sys.executable, '-B', '-c',
                                 "import sys; import implementation.covering_detector; import implementation.covering_detector_worker; assert 'torch' not in sys.modules"],
                                cwd=detector.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_threshold_boundary_and_exact_3pixel_margin(self):
        probability = np.zeros((256, 256), np.float32)
        probability[100, 100] = np.float32(.5)
        probability[200, 200] = np.nextafter(np.float32(.5), np.float32(0))
        raw, proposed, record = detector.proposal_from_probability(probability, np.ones((256, 256), bool))
        self.assertEqual(int((raw != 0).sum()), 1)
        self.assertEqual(int((proposed != 0).sum()), 49)
        self.assertTrue((proposed[97:104, 97:104] == 255).all())
        self.assertEqual(proposed[96, 100], 0)
        self.assertFalse(record['mask_reviewed'])
        self.assertFalse(record['automatic_confirmation'])
        self.assertFalse(record['automatic_covering_quality_qualified'])

    def test_proposal_never_selects_padding_or_expands_padding_predictions(self):
        probability = np.zeros((256, 256), np.float32)
        observed = np.zeros((256, 256), bool); observed[:, 32:224] = True
        probability[:, :20] = 1
        probability[100, 32] = 1
        raw, proposed, record = detector.proposal_from_probability(probability, observed)
        self.assertFalse(raw[~observed].any())
        self.assertFalse(proposed[~observed].any())
        self.assertEqual(int((proposed != 0).sum()), 28)
        self.assertEqual(record['threshold_positive_padding_pixels_discarded'], 20 * 256)

    def test_empty_proposal_requires_review_and_is_not_uncovered_classification(self):
        raw, proposed, record = detector.proposal_from_probability(np.zeros((256, 256), np.float32),
                                                                  np.ones((256, 256), bool))
        self.assertFalse(raw.any()); self.assertFalse(proposed.any())
        self.assertTrue(record['proposal_empty'])
        self.assertTrue(record['empty_proposal_does_not_prove_uncovered'])
        self.assertTrue(record['mask_review_required'])
        self.assertFalse(record['covering_type_classification'])

    def test_invalid_probability_or_observed_support_is_refused(self):
        good = np.zeros((256, 256), np.float32); observed = np.ones((256, 256), bool)
        for bad in (good.astype(np.float64), np.zeros((255, 256), np.float32),
                    np.full((256, 256), np.nan, np.float32), np.full((256, 256), 1.01, np.float32)):
            with self.assertRaises(ValueError):
                detector.proposal_from_probability(bad, observed)
        for bad in (observed.astype(np.uint8), observed[:255], np.zeros_like(observed)):
            with self.assertRaises(ValueError):
                detector.proposal_from_probability(good, bad)

    def test_bad_request_support_refuses_before_model_load_with_zero_attempts(self):
        scratch = detector.ROOT / 'scratch/detector_plain_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(root / 'input.png')
            Image.fromarray(np.full((256, 256), 1, np.uint8)).save(root / 'observed.png')
            request = {'input_path': str(root / 'input.png'), 'observed_path': str(root / 'observed.png'),
                       'output_dir': str(root / 'output')}
            with patch.object(worker, 'load_frozen_detector', side_effect=AssertionError('model forbidden')):
                with self.assertRaises(ValueError):
                    worker.run_request(request)
            status = json.loads((root / 'output/status.json').read_bytes())
            self.assertEqual(status['forward_attempts'], 0)
            self.assertEqual(status['forward_returns'], 0)
            self.assertEqual(status['phase'], 'failed')


if __name__ == '__main__':
    unittest.main()
