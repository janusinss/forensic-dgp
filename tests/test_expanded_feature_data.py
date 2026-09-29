import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch


class ExpandedDataTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('expanded_feature_data'))
        import expanded_feature_data
        return expanded_feature_data

    def fixture(self, root):
        p=root/'dataset/asian_faces/a.png';p.parent.mkdir(parents=True)
        cv2.imwrite(str(p),np.random.default_rng(7).integers(0,256,(256,256,3),dtype=np.uint8))
        split=root/'split.json';split.write_text(json.dumps({'train':['dataset/asian_faces/a.png'],'validation':['heldout.png']}))
        return {'format':'expanded-anatomical-training-v1','partition':'train','seed':42,
                'split_path':'split.json','split_sha256':hashlib.sha256(split.read_bytes()).hexdigest(),
                'excluded_sha256':['0'*64],
                'sources':[{'path':'dataset/asian_faces/a.png','source':'dataset/asian_faces',
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                'landmarks':[[85,150],[170,160],[128,195],[103,230],[151,235]]}]}

    def test_variants_labels_and_generic_compatibility(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root);ds=api.ExpandedCoveringDataset(m,root)
            self.assertEqual(len(ds),10)
            self.assertEqual(ds.rejections,[])
            for i in range(10):
                a=ds[i];b=ds[i]
                self.assertTrue(torch.equal(a['input'],b['input']))
                self.assertEqual(a['input'].shape,(3,256,256))
                self.assertTrue(torch.isfinite(a['input']).all())
                self.assertEqual(set(a['mask'].unique().tolist())- {0.,1.},set())
                if a['kind']=='none':self.assertFalse(a['mask'].any())
                if not a['degraded']:
                    visible=a['mask'][0]==0
                    self.assertTrue(torch.equal(a['input'][:,visible],a['target'][:,visible]))
                if a['kind'] in ('lower','eyes'):self.assertTrue(a['mask'].any())
            self.assertTrue(ds[1]['augmentation']['border_clipped'])
            self.assertGreater(int(ds[6]['mask'].sum()),int(ds[1]['mask'].sum()))

    def test_missing_landmarks_are_recorded_not_empty_positive_examples(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root);m['sources'][0]['landmarks']=None
            ds=api.ExpandedCoveringDataset(m,root)
            self.assertEqual(len(ds),6);self.assertEqual(len(ds.rejections),4)
            self.assertEqual({ds[i]['kind'] for i in range(len(ds))},{'none','object','irregular'})

    def test_matched_control_preserves_membership_and_generic_cases(self):
        api=self.api()
        import inspect
        self.assertIn('placement',inspect.signature(api.ExpandedCoveringDataset).parameters)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root)
            anatomy=api.ExpandedCoveringDataset(m,root)
            fixed=api.ExpandedCoveringDataset(m,root,placement='fixed')
            self.assertEqual(anatomy.cases,fixed.cases)
            self.assertEqual(anatomy.rejections,fixed.rejections)
            changed=0;shared_pixels=0
            for i in range(len(anatomy)):
                a,b=anatomy[i],fixed[i]
                self.assertTrue(torch.equal(a['target'],b['target']))
                if a['kind'] in ('none','object','irregular'):
                    for key in ('input','geometry','mask'):self.assertTrue(torch.equal(a[key],b[key]))
                else:
                    changed+=int(not torch.equal(a['geometry'],b['geometry']))
                    if not a['degraded']:
                        shared=(a['geometry'][0]>0)&(b['geometry'][0]>0)
                        shared_pixels+=int(shared.sum())
                        self.assertTrue(torch.equal(a['input'][:,shared],b['input'][:,shared]))
            self.assertEqual(changed,4)
            self.assertGreater(shared_pixels,0)
            m['sources'][0]['landmarks']=None
            anatomy=api.ExpandedCoveringDataset(m,root)
            fixed=api.ExpandedCoveringDataset(m,root,placement='fixed')
            self.assertEqual(anatomy.cases,fixed.cases)
            self.assertEqual(len(fixed),6)
            with self.assertRaises(ValueError):api.ExpandedCoveringDataset(m,root,placement='unknown')

    def test_leakage_duplicate_path_and_content_mutation_fail_closed(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=self.fixture(root)
            for field,value in [('partition','validation'),('split_sha256','bad')]:
                bad=copy.deepcopy(m);bad[field]=value
                with self.assertRaises(ValueError):api.ExpandedCoveringDataset(bad,root)
            bad=copy.deepcopy(m);bad['excluded_sha256']=[m['sources'][0]['sha256']]
            with self.assertRaises(ValueError):api.ExpandedCoveringDataset(bad,root)
            bad=copy.deepcopy(m);bad['sources']*=2
            with self.assertRaises(ValueError):api.ExpandedCoveringDataset(bad,root)
            bad=copy.deepcopy(m);bad['sources'][0]['path']='../outside.png'
            with self.assertRaises(ValueError):api.ExpandedCoveringDataset(bad,root)
            bad=copy.deepcopy(m)
            (root/'split.json').write_text(json.dumps({'train':[],'validation':[m['sources'][0]['path']]}))
            bad['split_sha256']=hashlib.sha256((root/'split.json').read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'outside training split'):api.ExpandedCoveringDataset(bad,root)
            (root/'split.json').write_text(json.dumps({'train':[m['sources'][0]['path']],'validation':['heldout.png']}))
            ds=api.ExpandedCoveringDataset(m,root)
            (root/m['sources'][0]['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):ds[0]

    def test_sampling_covers_large_groups_with_fixed_budget(self):
        api=self.api();groups=[];start=0
        for n in [43,25,1336,340,1432,364]:
            groups.append(list(range(start,start+n)));start+=n
        schedule=list(api.expanded_balanced_schedule(groups,42,epochs=20,steps_per_epoch=80))
        self.assertEqual(len(schedule),20)
        self.assertEqual(schedule,list(api.expanded_balanced_schedule(groups,42,epochs=20,steps_per_epoch=80)))
        batches=[b for epoch in schedule for b in epoch]
        self.assertEqual(len(batches),1600)
        self.assertEqual(set(i for b in batches for i in b),set(range(start)))
        for b in batches:
            self.assertEqual(len(b),12)
            for g in groups:self.assertEqual(sum(i in g for i in b),2)
        with self.assertRaises(ValueError):list(api.expanded_balanced_schedule(groups,42,epochs=1,steps_per_epoch=80))
        with self.assertRaises(ValueError):list(api.expanded_balanced_schedule([groups[0]]*6,42))
