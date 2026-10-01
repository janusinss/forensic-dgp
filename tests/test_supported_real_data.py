"""Supported-label consumption must preserve context and exclude unknown pixels."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from supported_real_data import FORMAT, SupportedMasks, load_supported_manifest, validate_arrays


class SupportedRealDataTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory); rows = []
        for i in range(6):
            rgb = np.full((32, 32, 3), 80+i, np.uint8)
            source = np.ones((32, 32), np.uint8)*255; source[:, :4] = 0; rgb[:, :4] = 96
            valid = source.copy(); valid[24:, :] = 0
            mask = np.zeros((32, 32), np.uint8)
            if i % 2 == 0:
                mask[10:16, 10:16] = 255
            files = {}
            for key, value in (('image', rgb), ('mask', mask), ('valid', valid), ('source_valid', source)):
                path = root/f'{i}_{key}.png'; Image.fromarray(value).save(path)
                files[key] = path.name; files[key+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({**files, 'reviewed': True, 'group': f'group-{i}', 'source_sha256': f'{i+1:064x}',
                         'split': ('train', 'validation', 'test')[i//2],
                         'kind': 'covered' if i%2 == 0 else 'uncovered', 'support_required': True})
        path = root/'manifest.json'
        data = {'format': FORMAT, 'size': 32, 'dataset_reviewed': True, 'training_recipe_ready': False,
                'support_semantics': 'valid_one_is_supervised', 'supported_records': rows}
        self.save(path, data)
        return path, data

    def save(self, path, data):
        path.write_text(json.dumps(data), encoding='utf-8')

    def test_triple_retains_unknown_rgb_and_legacy_consumer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory)
            dataset = SupportedMasks(path, split='train')
            x, m, valid = dataset[0]
            self.assertEqual(x.shape, (3, 32, 32))
            self.assertEqual(m.shape, valid.shape)
            self.assertEqual(valid.shape, (1, 32, 32))
            self.assertEqual(float(valid[0, 28, 10]), 0)
            self.assertAlmostEqual(float(x[0, 28, 10]), 80/255, places=6)
            self.assertFalse((m.bool() & ~valid.bool()).any())
            from detector_training import load_manifest
            with self.assertRaises(KeyError):
                load_manifest(path)

    def test_unknown_logits_do_not_change_supported_loss_or_receive_gradient(self):
        from reflection_coverage import supported_segmentation_loss
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory)
            _, m, valid = SupportedMasks(path, split='train')[0]
            logits = torch.zeros((1, 1, 32, 32), requires_grad=True)
            target, support = m[None], valid[None]
            loss = supported_segmentation_loss(logits, target, support)
            changed = logits.detach().clone(); changed[support == 0] = 100
            self.assertTrue(torch.equal(loss, supported_segmentation_loss(changed, target, support)))
            loss.backward()
            self.assertEqual(float(logits.grad[support == 0].abs().sum()), 0)
            self.assertGreater(float(logits.grad[support == 1].abs().sum()), 0)

    def test_hash_changes_are_detected_at_load_and_later_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path, data = self.fixture(directory); dataset = SupportedMasks(path, split='train')
            file = Path(directory)/data['supported_records'][0]['valid']
            Image.fromarray(np.ones((32, 32), np.uint8)*255).save(file)
            with self.assertRaises(ValueError):
                dataset[0]
            with self.assertRaises(ValueError):
                load_supported_manifest(path)

    def test_cross_split_duplicate_unreviewed_or_escaping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, data = self.fixture(directory)
            for changes in ({'group': 'group-0'}, {'source_sha256': data['supported_records'][0]['source_sha256']},
                            {'reviewed': False}, {'valid': '../outside.png'}):
                altered = json.loads(json.dumps(data)); altered['supported_records'][2].update(changes)
                self.save(path, altered)
                with self.assertRaises(ValueError):
                    load_supported_manifest(path)

    def test_positive_in_unknown_region_or_mismatched_support_is_rejected(self):
        image = np.full((32, 32, 3), 90, np.uint8)
        target = np.zeros((32, 32), bool); valid = np.ones_like(target); valid[24:] = False
        source = np.ones_like(target); target[28, 10] = True
        with self.assertRaises(ValueError):
            validate_arrays(image, target, valid, source, 32, 'covered')
        with self.assertRaises(ValueError):
            validate_arrays(image, target.astype(float), valid, source, 32, 'covered')


if __name__ == '__main__':
    unittest.main()
