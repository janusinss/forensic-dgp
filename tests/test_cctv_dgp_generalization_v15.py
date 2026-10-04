"""Meaningful split/output safety checks; no model construction or training."""
import copy
from pathlib import Path
import unittest

import numpy as np
import cctv_dgp_generalization_v15 as v


def cohort():
    refs=[]
    for source,n in zip(v.SOURCES,[51,53]):
        for i in range(n):
            rid=source+'_'+str(i)
            refs.append({'id':rid,'source':source,'role':'validation','source_sha256':'src_'+rid,'target_rgb_sha256':'rgb_'+rid})
    p={'references':refs,'cases':[{'id':r['id']+'_'+profile,'reference_id':r['id'],'source':r['source'],'profile':profile} for r in refs for profile in v.PROFILES],
       'training_identity_inventory':[{'id':'train','source_sha256':'src_train','target_rgb_sha256':'rgb_train'}],
       'preview_reference_ids':[r['id'] for s in v.SOURCES for r in sorted((r for r in refs if r['source']==s),key=lambda r:r['id'])[:5]],
       'native_used':False,'native_reserved_used':False,'production_promoted':False,'training':False}
    return p


class SafetyTests(unittest.TestCase):
    def test_train_validation_overlap_rejected(self):
        p=cohort();v.validate_cohort(p)
        for key in ['id','source_sha256','target_rgb_sha256']:
            q=copy.deepcopy(p);q['training_identity_inventory'][0][key]=q['references'][0][key]
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'overlap'):v.validate_cohort(q)

    def test_missing_profile_and_changed_previews_rejected(self):
        p=cohort();p['cases'][0]['profile']='blur_lr24'
        with self.assertRaisesRegex(ValueError,'profile'):v.validate_cohort(p)
        p=cohort();p['preview_reference_ids'].reverse()
        with self.assertRaisesRegex(ValueError,'selection'):v.validate_cohort(p)

    def test_path_escape_rejected(self):
        for name in ['../outside','C:/outside','/outside','a\\b']:
            with self.subTest(name=name),self.assertRaises(ValueError):v.safe(Path.cwd(),name)

    def test_delivered_png_keeps_padding_and_rejects_nonfinite(self):
        raw=np.full((256,256,3),.5,dtype=np.float32);camera=np.full((256,256,3),231,dtype=np.uint8)
        mask=np.ones((256,256),dtype=bool);mask[:10]=False
        value=v.png(raw,camera,mask)
        self.assertTrue(np.all(value[:10]==231));self.assertTrue(np.all(value[10:]==127))
        raw[20,20,0]=np.nan
        with self.assertRaisesRegex(ValueError,'raw'):v.png(raw,camera,mask)


if __name__=='__main__':unittest.main()
