"""Safety and geometry contracts for the opt-in source-qualified data version."""
import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class QualifiedDataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('completion_data_v2'),
                             'Source-qualified, geometry-preserving data is missing')

    def source(self, root):
        path = root/'dataset/face.png'
        path.parent.mkdir()
        rgb = np.full((100,80,3),130,np.uint8)
        Image.fromarray(rgb).save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row = {'source_id':0,'path':'dataset/face.png','sha256':digest,'split':'train',
               'usage':'paired_unoccluded','review':{'method':'assistant_native_screen',
               'occlusion':'none_observed','reference_quality':'native_low_resolution',
               'rationale':'Both eyes and lower face visible; limited native resolution.'},
               'eyes':[[25.,42.],[55.,42.]],'eyes_reviewed':True}
        return row, {row['path']:digest}, rgb

    def test_circle_shape_and_point_distances_survive_nonsquare_conversion(self):
        from completion_data_v2 import square_preserving_geometry, map_points
        rgb = np.full((81,161,3),60,np.uint8)
        cv2.circle(rgb,(80,40),20,(220,220,220),-1)
        result = square_preserving_geometry(rgb,161)
        y,x = np.where(result['rgb'][:,:,0]>180)
        self.assertLessEqual(abs((x.max()-x.min())-(y.max()-y.min())),1)
        points = np.array([[30.,20.],[70.,20.],[30.,60.]])
        transformed = map_points(points,result['affine'])
        self.assertAlmostEqual(np.linalg.norm(transformed[1]-transformed[0]),
                               np.linalg.norm(transformed[2]-transformed[0]),places=10)
        self.assertTrue(np.array_equal(result['rgb'][result['valid']==0],
                                       np.full((int((result['valid']==0).sum()),3),96,np.uint8)))
        self.assertGreater(result['valid'].sum(),0)
        self.assertLess(result['valid'].sum(),161*161)

    def test_square_at_native_size_is_identity_without_padding(self):
        from completion_data_v2 import square_preserving_geometry
        rgb = np.random.default_rng(4).integers(0,256,(64,64,3),dtype=np.uint8)
        result = square_preserving_geometry(rgb,64)
        self.assertTrue(np.array_equal(result['rgb'],rgb))
        self.assertTrue(np.all(result['valid']==1))
        np.testing.assert_array_equal(result['affine'],[[1.,0.,0.],[0.,1.,0.]])

    def test_rejects_bad_geometry_and_nonfinite_points(self):
        from completion_data_v2 import square_preserving_geometry, map_points
        for rgb in (np.zeros((10,10),np.uint8), np.zeros((1,10,3),np.uint8),
                    np.full((10,10,3),np.nan), np.zeros((10,10,3),np.float32)):
            with self.assertRaises(ValueError):square_preserving_geometry(rgb,64)
        with self.assertRaises(ValueError):map_points([[1.,float('nan')]],np.eye(2,3))

    def test_pending_and_intrinsically_occluded_bases_cannot_become_paired_targets(self):
        from completion_data_v2 import load_source
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); row,training,_=self.source(root)
            for usage,occlusion in [('pending','none_observed'),('unpaired_occlusion','observed'),
                                    ('paired_unoccluded','uncertain'),('paired_unoccluded','observed')]:
                changed=copy.deepcopy(row);changed['usage']=usage;changed['review']['occlusion']=occlusion
                with self.assertRaises(ValueError):load_source(root,changed,training,set())

    def test_exact_training_membership_and_exclusion_hashes_are_required(self):
        from completion_data_v2 import load_source
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); row,training,_=self.source(root)
            for allowed,excluded in [({},set()),(training,{row['sha256']}),
                                     ({row['path']:'0'*64},set())]:
                with self.assertRaises(ValueError):load_source(root,row,allowed,excluded)
            changed=copy.deepcopy(row);changed['split']='validation'
            with self.assertRaises(ValueError):load_source(root,changed,training,set())
            loaded=load_source(root,row,training,set())
            self.assertEqual(loaded.shape,(100,80,3))

    def test_changed_source_and_nonportable_or_escaping_paths_fail(self):
        from completion_data_v2 import load_source
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); row,training,_=self.source(root)
            for name in ['../face.png','/face.png','C:/face.png','dataset\\face.png','dataset/../face.png']:
                changed=copy.deepcopy(row);changed['path']=name
                with self.assertRaises(ValueError):load_source(root,changed,{name:row['sha256']},set())
            (root/row['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):load_source(root,row,training,set())

    def test_review_metadata_is_required_even_for_a_named_clean_folder(self):
        from completion_data_v2 import load_source
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);row,training,_=self.source(root)
            for change in ('review','rationale','eyes_reviewed'):
                bad=copy.deepcopy(row)
                if change=='review':bad.pop('review')
                elif change=='rationale':bad['review']['rationale']=''
                else:bad['eyes_reviewed']=False
                with self.assertRaises(ValueError):load_source(root,bad,training,set())

    def test_clear_and_partial_reflections_preserve_visible_pixels_and_target(self):
        from completion_data_v2 import lens_reflection, REFLECTION_STYLES
        rgb=np.random.default_rng(8).integers(30,210,(128,128,3),dtype=np.uint8)
        original=rgb.copy();eyes=np.array([[42.,54.],[86.,54.]])
        for style in ('clear',*REFLECTION_STYLES):
            a=lens_reflection(rgb,eyes,style,seed=15)
            b=lens_reflection(rgb,eyes,style,seed=15)
            for key in ('input','target','mask','lens_region'):
                self.assertTrue(np.array_equal(a[key],b[key]))
            self.assertTrue(np.array_equal(rgb,original))
            self.assertTrue(np.isin(a['mask'],[0,1]).all())
            self.assertTrue(np.array_equal(a['input'][a['mask']==0],a['target'][a['mask']==0]))
            self.assertFalse(np.any(a['mask'] & ~a['lens_region']))
            if style=='clear':self.assertFalse(a['mask'].any())
            else:
                self.assertGreater(a['mask'].sum(),0)
                self.assertLess(a['mask'].sum(),a['lens_region'].sum())
                self.assertTrue(np.array_equal(a['target'],lens_reflection(rgb,eyes,'clear',seed=15)['target']))

    def test_reflections_reject_unreviewed_or_outside_image_lenses(self):
        from completion_data_v2 import lens_reflection
        rgb=np.full((128,128,3),100,np.uint8)
        for eyes in ([[30.,54.],[30.,54.]],[[2.,54.],[42.,54.]],[[40.,54.],[86.,float('nan')]]):
            with self.assertRaises(ValueError):lens_reflection(rgb,eyes,'white_patch',1)
        valid=np.ones((128,128),np.uint8);valid[:,30:55]=0
        with self.assertRaises(ValueError):lens_reflection(rgb,[[42.,54.],[86.,54.]],'white_patch',1,valid)

    def test_prepared_case_keeps_padding_out_of_supervised_regions(self):
        from completion_data_v2 import prepare_case
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);row,training,_=self.source(root)
            first=prepare_case(root,row,training,set(),'blue_glare',seed=42,size=128)
            second=prepare_case(root,row,training,set(),'blue_glare',seed=42,size=128)
            self.assertTrue(np.array_equal(first['input'],second['input']))
            self.assertTrue(np.all(first['mask'][first['valid']==0]==0))
            self.assertTrue(np.all(first['input'][first['valid']==0]==96))
            self.assertEqual(first['metadata']['source_sha256'],row['sha256'])
            self.assertFalse(first['metadata']['restoration_high_resolution_reference'])


if __name__=='__main__':unittest.main()
