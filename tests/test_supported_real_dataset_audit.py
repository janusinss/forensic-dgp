"""Original evaluation pixels/support and mannequin identity cannot be diluted."""
import copy
import unittest
import numpy as np
from scripts.audit_supported_real_dataset import check_parent_record, check_original_support


class SupportedDatasetAuditTests(unittest.TestCase):
    def test_outer_labels_or_original_basename_cannot_change_behind_correct_lineage(self):
        original = {'image': 'images/new_covered_40.png', 'mask': 'masks/new_covered_40.png',
                    'split': 'validation', 'kind': 'covered', 'image_sha256': 'a'*64,
                    'mask_sha256': 'b'*64, 'source_sha256': 'c'*64, 'group': 'mannequin-source'}
        row = {**original, 'image': 'images/base/new_covered_40.png',
               'mask': 'masks/base/new_covered_40.png', 'previous_record': copy.deepcopy(original)}
        check_parent_record(original, row)
        for changed in ({'split': 'train'}, {'kind': 'uncovered'}, {'mask_sha256': 'd'*64},
                        {'image': 'images/base/renamed.png'}):
            with self.assertRaises(ValueError):
                check_parent_record(original, {**row, **changed})

    def test_original_difficult_pixels_cannot_be_removed_from_valid_support(self):
        valid = np.ones((32, 32), bool); source = valid.copy()
        check_original_support(valid, source)
        valid[20, 20] = False
        with self.assertRaises(ValueError):
            check_original_support(valid, source)
        with self.assertRaises(ValueError):
            check_original_support(source, source.astype(float))


if __name__ == '__main__':
    unittest.main()
