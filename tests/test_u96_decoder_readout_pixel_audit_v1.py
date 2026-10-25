"""Independent saved-array arithmetic fixtures; no model or producer imports."""
import importlib.util
import io
import copy
from pathlib import Path
import unittest

import numpy as np

SPEC=importlib.util.spec_from_file_location('pixel_audit',Path(__file__).resolve().parents[1]/'scripts/audit_u96_decoder_readout_pixels_v1.py')
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)


class SavedPixelArithmetic(unittest.TestCase):
    def test_floor_is_not_round(self):
        value=np.full((256,256,3),.5,dtype=np.float32)
        self.assertTrue((AUDIT.floor_png(value)==127).all())

    def test_floor_rejects_outside_range(self):
        with self.assertRaises(ValueError):AUDIT.floor_png(np.full((256,256,3),1.1,dtype=np.float32))

    def test_constant_residual_has_zero_edges(self):
        a=np.full((256,256,3),.25,dtype=np.float32);b=np.zeros_like(a)
        for mask in AUDIT.MASKS.values():
            self.assertEqual(AUDIT.regional(a,b,mask),{'mae':.25,'adjacent_rgb_error':0.})

    def test_local_seam_not_counted(self):
        a=np.zeros((256,256,3),np.float32);a[64:192,64:192]=.4
        _,terms=AUDIT.objective(a,np.zeros_like(a),'local_gaussian_1p2')
        self.assertEqual(terms['adjacent_inside'],0.);self.assertEqual(terms['adjacent_outside'],0.)
        self.assertGreater(AUDIT.regional(a,np.zeros_like(a),AUDIT.MASKS['whole'])['adjacent_rgb_error'],0)

    def test_local_equal_inside_outside_weight(self):
        a=np.zeros((256,256,3),np.float32);a[64:192,64:192]=.25
        result,_=AUDIT.objective(a,np.zeros_like(a),'local_gaussian_1p2')
        self.assertEqual(result,.125)

    def test_clean_target_used_for_clear_dark(self):
        a=np.full((256,256,3),.25,np.float32);b=np.zeros_like(a)
        for role in AUDIT.ROLES:self.assertEqual(AUDIT.objective(a,b,role)[0],.25)

    def test_unknown_role_refused(self):
        with self.assertRaises(ValueError):AUDIT.objective(np.zeros((256,256,3)),np.zeros((256,256,3)),'guessed')

    def test_array_extent_and_dtype_bounded(self):
        stream=io.BytesIO();np.save(stream,np.zeros((256,256,3),np.float32),allow_pickle=False)
        data=stream.getvalue();self.assertEqual(AUDIT.array(data,'raw.npy').shape,(256,256,3))
        with self.assertRaises(ValueError):AUDIT.array(data+b'x','raw.npy')
        with self.assertRaises(ValueError):AUDIT.array(data,'native512.npy')

    def test_strict_json_refuses_duplicate_nonfinite(self):
        for data in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e400}'):
            with self.assertRaises(ValueError):AUDIT.strict_json(data)

    def test_luminance_blocks_exact_constant(self):
        self.assertTrue(np.allclose(AUDIT.luminance_blocks(np.ones((256,256,3),np.float32)),1,atol=1e-15))

    def monitor_fixture(self):
        protocol={'cases':[]};baseline={}
        for role in AUDIT.ROLES:
            for i in range(8):
                key=role+str(i);protocol['cases'].append({'id':key,'role':role,'cohort':'monitor'})
                baseline[key]={'role':role,'regional_metrics':{view:{region:{'mae':.5,'adjacent_rgb_error':1.}
                    for region in AUDIT.MASKS} for view in AUDIT.VIEWS},
                    'U96_behavior_distance':dict.fromkeys(AUDIT.VIEWS,0.),
                    'U96_illumination_distance':dict.fromkeys(AUDIT.VIEWS,0.)}
        return protocol,baseline,copy.deepcopy(baseline)

    def test_behavior_boundary_has_no_epsilon(self):
        p,base,bank=self.monitor_fixture()
        for i in range(8):bank['clear'+str(i)]['U96_behavior_distance']['raw']=.01
        checks=AUDIT.monitor_checks(bank,base,p,1)
        self.assertTrue(next(x for x in checks if x['role']=='clear' and x['view']=='raw' and x['metric']=='U96_behavior_distance')['pass'])
        for i in range(8):bank['clear'+str(i)]['U96_behavior_distance']['raw']=float(np.nextafter(.01,1.))
        checks=AUDIT.monitor_checks(bank,base,p,1)
        self.assertFalse(next(x for x in checks if x['role']=='clear' and x['view']=='raw' and x['metric']=='U96_behavior_distance')['pass'])

    def test_early_trend_needs_both_views_of_one_role(self):
        p,base,bank=self.monitor_fixture()
        for i in range(8):bank[AUDIT.ROLES[0]+str(i)]['regional_metrics']['raw']['central128']['adjacent_rgb_error']=.9
        self.assertFalse(AUDIT.trend_check(bank,base,p,16)['pass'])
        for i in range(8):bank[AUDIT.ROLES[0]+str(i)]['regional_metrics']['native_png']['central128']['adjacent_rgb_error']=.9
        self.assertTrue(AUDIT.trend_check(bank,base,p,16)['pass'])

    def test_terminal_requires_all_four_blur_views(self):
        p,base,bank=self.monitor_fixture()
        for role in AUDIT.ROLES[:2]:
            region='central128' if role==AUDIT.ROLES[0] else 'whole'
            for i in range(8):
                for view in AUDIT.VIEWS:bank[role+str(i)]['regional_metrics'][view][region]['adjacent_rgb_error']=.98
        checks=AUDIT.monitor_checks(bank,base,p,64)
        self.assertTrue(all(r['pass'] for r in checks if r['metric']=='new_1_percent_TRAIN_blur_endpoint'))
        for i in range(8):bank[AUDIT.ROLES[1]+str(i)]['regional_metrics']['raw']['whole']['adjacent_rgb_error']=.999
        checks=AUDIT.monitor_checks(bank,base,p,64)
        self.assertEqual(sum(not r['pass'] for r in checks if r['metric']=='new_1_percent_TRAIN_blur_endpoint'),1)


if __name__=='__main__':unittest.main()
