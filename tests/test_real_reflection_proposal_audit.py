"""Saved-label failures: unknown regions cannot quietly become negatives."""
import unittest
import numpy as np
from scripts.audit_real_reflection_proposals import check_support, check_pending


class RealReflectionProposalAuditTests(unittest.TestCase):
    def test_unknown_or_padding_pixels_cannot_receive_supervised_targets(self):
        mask = np.zeros((32, 32), bool); mask[10:15, 10:15] = True
        source = np.ones_like(mask); source[:, :4] = False
        unknown = np.zeros_like(mask); unknown[24:, :] = True
        valid = source & ~unknown
        check_support(mask, valid, source, unknown)
        with self.assertRaises(ValueError):
            check_support(mask, source, source, unknown)
        p = mask.copy(); p[1, 1] = True
        with self.assertRaises(ValueError):
            check_support(p, valid, source, unknown)
        with self.assertRaises(ValueError):
            check_support(mask.astype(float), valid, source, unknown)

    def test_pending_photo_cannot_supply_an_empty_mask_or_image(self):
        row = {'status': 'pending_native_review', 'mask': None, 'valid': None, 'image': None}
        check_pending(row)
        for changed in ({'mask': 'empty.png'}, {'valid': 'valid.png'}, {'image': 'input.png'}):
            with self.assertRaises(ValueError):
                check_pending({**row, **changed})


if __name__ == '__main__':
    unittest.main()
