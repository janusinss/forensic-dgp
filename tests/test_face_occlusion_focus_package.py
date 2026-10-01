"""New focus bundle must not overwrite previously executed input files."""
import copy
import unittest
from unittest.mock import patch
import torch
from scripts.package_face_occlusion_focus_vm import validate_new_members, reflection_registry


class FocusPackageTests(unittest.TestCase):
    def test_rejects_any_prior_inventory_overlap_and_escaping_member(self):
        validate_new_members(['face_occlusion_focus.py', 'scripts/train_focus.py'], [{'old.py': 'x'}])
        for files in (['old.py'], ['../escape'], ['C:/escape'], ['scripts\\windows.py'], ['x.py', 'x.py'], []):
            with self.assertRaises(ValueError): validate_new_members(files, [{'old.py': 'x'}])

    def test_all_old_inventories_participate_in_collision_check(self):
        with self.assertRaises(ValueError):
            validate_new_members(['executed_runner.py'], [{}, {}, {'executed_runner.py': 'x'}])

    def reflection_fixture(self):
        rows = [dict(split='train', image_sha256=str(i), mask_sha256=str(i)) for i in range(73)]
        old = {}; reviewed = {}
        for i in (15, 46):
            rows[i]['glare_stratum'] = 'strong_lens_reflection'
            target = torch.zeros(1, 256, 256); target[0, 0, 1:3] = 1
            rows[i]['target'] = target
            old[str(i)] = dict(split='train', mask_sha256='old' + str(i), target=torch.zeros_like(target))
            reviewed[str(i)] = dict(split='train', mask_sha256=str(i), v2_mask_sha256='old' + str(i))
        class Dataset:
            def __init__(self, selected, size): self.rows = selected
            def __getitem__(self, i): return torch.zeros(3, 256, 256), self.rows[i]['target']
        return rows, old, reviewed, Dataset

    def test_priority_derivation_excludes_heldout_and_preserves_original_foreground(self):
        rows, old, reviewed, dataset = self.reflection_fixture()
        before = copy.deepcopy(rows)
        with patch('scripts.package_face_occlusion_focus_vm.ReviewedMasks', dataset):
            registry = reflection_registry(rows, old, reviewed)
            self.assertEqual([r['batch_index'] for r in registry['records']], [15, 46])
            self.assertEqual(registry['records'][0]['flat_indices'], [1, 2])
            self.assertTrue(torch.equal(rows[15]['target'], before[15]['target']))
            for mistake in ('validation', 'lineage', 'removed_pixel'):
                changed_old = copy.deepcopy(old); changed_reviewed = copy.deepcopy(reviewed)
                if mistake == 'validation': changed_old['15']['split'] = 'validation'
                if mistake == 'lineage': changed_reviewed['15']['v2_mask_sha256'] = 'incorrect'
                if mistake == 'removed_pixel': changed_old['15']['target'][0, 100, 100] = 1
                with self.assertRaises(ValueError): reflection_registry(rows, changed_old, changed_reviewed)


if __name__ == '__main__': unittest.main()
