import importlib.util
import unittest

import numpy as np


class PixelAuditTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.audit_qualified_reflection_review'),
                             'Independent fixture pixel audit missing')
        self.target=np.full((64,64,3),100,np.uint8)
        self.mask=np.zeros((64,64),np.uint8);self.mask[29:34,29:34]=1
        self.valid=np.ones((64,64),np.uint8);self.valid[:3]=0
        self.lens=np.zeros((64,64),np.uint8);self.lens[20:44,20:44]=1
        self.target[:3]=96
        self.input=self.target.copy();self.input[self.mask==1]=240

    def check(self,**changes):
        from scripts.audit_qualified_reflection_review import audit_pixels
        fields={'input':self.input,'target':self.target,'mask':self.mask,
                'valid':self.valid,'lens_region':self.lens}
        fields.update(changes)
        return audit_pixels(fields,'white_patch')

    def test_recounts_partial_hole_and_excludes_padding(self):
        counts=self.check()
        self.assertEqual(counts['hole_pixels'],25)
        self.assertEqual(counts['padding_pixels'],192)
        self.assertEqual(counts['visible_changed_pixels'],0)

    def test_rejects_background_edits_and_padding_or_outside_lens_targets(self):
        changed=self.input.copy();changed[8,8]=0
        with self.assertRaises(ValueError):self.check(input=changed)
        for coordinate in ((1,30),(10,10)):
            bad=self.mask.copy();bad[coordinate]=1
            with self.assertRaises(ValueError):self.check(mask=bad)

    def test_rejects_soft_masks_and_empty_positive_or_nonempty_clear_target(self):
        bad=self.mask.astype(float);bad[30,30]=.5
        with self.assertRaises(ValueError):self.check(mask=bad)
        with self.assertRaises(ValueError):self.check(mask=np.zeros_like(self.mask))
        from scripts.audit_qualified_reflection_review import audit_pixels
        with self.assertRaises(ValueError):audit_pixels({'input':self.input,'target':self.target,
            'mask':self.mask,'valid':self.valid,'lens_region':self.lens},'clear')


if __name__=='__main__':unittest.main()
