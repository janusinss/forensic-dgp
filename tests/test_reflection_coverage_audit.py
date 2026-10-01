import importlib.util
import unittest

import numpy as np
import torch


class CacheAuditContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.audit_reflection_coverage_data'),
                             'Coverage cache auditor missing')
        target=np.full((64,64,3),100,np.uint8);target[:,:5]=96
        self.ref={'target':torch.from_numpy(target),'valid':torch.ones(64,64,dtype=torch.uint8),
                  'lens_region':torch.zeros(64,64,dtype=torch.uint8)}
        self.ref['valid'][:,:5]=0;self.ref['lens_region'][25:40,25:40]=1
        self.case={'input':self.ref['target'].permute(2,0,1).clone(),
                   'mask':torch.zeros(1,64,64,dtype=torch.uint8),
                   'geometry':torch.zeros(1,64,64,dtype=torch.uint8),'valid':self.ref['valid'][None].clone()}
        self.case['mask'][:,29:34,29:34]=1;self.case['geometry']=self.case['mask'].clone()
        self.case['input'][:,29:34,29:34]=240

    def test_recounts_clean_partial_reflection_and_padding(self):
        from scripts.audit_reflection_coverage_data import audit_case
        r=audit_case(self.case,self.ref,'white_patch',False)
        self.assertEqual(r['hole_pixels'],25);self.assertEqual(r['visible_changed_pixels'],0)

    def test_unmarked_edits_or_nonbinary_targets_fail(self):
        from scripts.audit_reflection_coverage_data import audit_case
        bad={k:v.clone() for k,v in self.case.items()};bad['input'][:,10,10]=0
        with self.assertRaises(ValueError):audit_case(bad,self.ref,'white_patch',False)
        bad={k:v.clone() for k,v in self.case.items()};bad['mask'][0,30,30]=2
        with self.assertRaises(ValueError):audit_case(bad,self.ref,'white_patch',False)

    def test_camera_labels_have_exact_declared_effect_support(self):
        from scripts.audit_reflection_coverage_data import audit_case
        with self.assertRaises(ValueError):audit_case(self.case,self.ref,'white_patch',True)


if __name__=='__main__':unittest.main()
