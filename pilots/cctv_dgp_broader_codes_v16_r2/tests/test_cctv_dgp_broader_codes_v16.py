"""Scheduling and cache boundary checks, without a training graph."""
import copy
from pathlib import Path
import unittest

import numpy as np
import cctv_dgp_broader_codes_v16 as v


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = v.read(Path(__file__).resolve().parents[1] / 'outputs/cctv_dgp_mixed_vm_v9_r2/mixed_protocol_v9.json')
        cls.p = {k: cls.p[k] for k in ['references', 'training_cases', 'validation_cases']}
        for role in ['train', 'validation']:
            cls.p[role + '_preview_reference_ids'] = v.previews([r for r in cls.p['references'] if r['role'] == role])
        for key in ['native_used', 'native_reserved_used', 'production_promoted', 'checkpoint_selected']: cls.p[key] = False

    def test_full_balanced_exposure_and_replay(self):
        steps = v.schedule(self.p); v.validate_schedule(self.p, steps)
        self.assertEqual(len(steps), 3128)
        self.assertEqual(sum(len(s['case_ids']) for s in steps), 31280)
        self.assertEqual(steps, v.schedule(self.p))

    def test_overlap_and_heldout_in_training_rejected(self):
        p = copy.deepcopy(self.p)
        train = next(r for r in p['references'] if r['role'] == 'train')
        val = next(r for r in p['references'] if r['role'] == 'validation')
        val['target_rgb_sha256'] = train['target_rgb_sha256']
        with self.assertRaisesRegex(ValueError, 'overlap'): v.validate_cohort(p)
        p = copy.deepcopy(self.p); p['training_cases'][0]['reference_id'] = val['id']
        with self.assertRaisesRegex(ValueError, 'crossed roles'): v.validate_cohort(p)

    def test_missing_profile_or_corrupt_cache_rejected(self):
        p = copy.deepcopy(self.p); p['training_cases'][0]['profile'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'profile'): v.validate_cohort(p)
        d = np.zeros((3,256,256), np.float32); f = np.zeros((256,16,16), np.float32)
        l = np.zeros((256,1024), np.float32); v.cache_arrays(d,f,l)
        l[0,0] = np.nan
        with self.assertRaisesRegex(ValueError, 'cache'): v.cache_arrays(d,f,l)

    def test_escape_or_schedule_tampering_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe'): v.cache_path(Path('.'), '../outside')
        steps = v.schedule(self.p); steps[0]['case_ids'][0] = steps[0]['case_ids'][1]
        with self.assertRaisesRegex(ValueError, 'schedule'): v.validate_schedule(self.p, steps)


if __name__ == '__main__': unittest.main()
