"""Pure NumPy/metadata fixtures only; no real return, Torch, or model execution."""
import copy
import importlib.util
import json
import io
from pathlib import Path
import sys
import tarfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('smooth_return_pixels',ROOT/'scripts/audit_u96_postfusion_smoothanchors_return_v1.py')
A=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


class SmoothCriterionFixtures(unittest.TestCase):
    def test_zero_signed_small_boundary_large_formula(self):
        values=np.asarray([0.,-.005,.005,-.01,.01,-.02,.02],dtype=np.float64)
        expected=np.asarray([0.,.00125,.00125,.005,.005,.015,.015])
        np.testing.assert_array_equal(A.smooth_penalty(values),expected)

    def test_continuous_nonflat_derivative_near_zero_and_beta(self):
        step=1e-8
        for value,expected in [(0.,0.),(.001,.1),(-.001,-.1),(.02,1.),(-.02,-1.)]:
            derivative=float((A.smooth_penalty(value+step)-A.smooth_penalty(value-step))/(2*step))
            self.assertAlmostEqual(derivative,expected,places=7)
        self.assertAlmostEqual(float(A.smooth_penalty(.01-1e-9)),float(A.smooth_penalty(.01+1e-9)),places=8)

    def test_clear_uses_smooth_terms_keeps_L1_and_ignores_clean_target(self):
        baseline=np.full((256,256,3),.4,dtype=np.float32)
        raw=baseline+np.float32(.005)
        target=np.zeros_like(raw)
        value,terms=A.objective(raw,target,baseline,'clear',np.float64)
        difference=float(raw[0,0,0]-baseline[0,0,0])
        self.assertAlmostEqual(value,difference*difference/.01,places=14)
        self.assertAlmostEqual(terms['U96_image_anchor_L1'],difference,places=15)
        self.assertEqual(terms['U96_adjacent_anchor_L1'],0.)
        self.assertEqual(terms['U96_adjacent_anchor_SmoothL1'],0.)
        self.assertEqual(set(terms),{'U96_image_anchor_L1','U96_adjacent_anchor_L1','U96_image_anchor_SmoothL1','U96_adjacent_anchor_SmoothL1'})

    def test_clear_adjacent_penalizes_signed_residual_differences(self):
        base=np.zeros((256,256,3),dtype=np.float32)
        raw=base.copy();raw[:,1::2]=np.float32(.02)
        value,terms=A.objective(raw,base,base,'clear',np.float64)
        #255horizontal edges each .02,255vertical edges each0, equal counts.
        self.assertAlmostEqual(terms['U96_adjacent_anchor_SmoothL1'],.00749999977648258,places=14)
        self.assertGreater(value,0.)

    def test_dark_preserves_paired_L1_plus_only_smooth_block_anchor(self):
        baseline=np.full((256,256,3),.2,dtype=np.float32)
        raw=baseline+np.float32(.005);target=raw.copy()
        value,terms=A.objective(raw,target,baseline,'lowlight_linear_0p35',np.float64)
        d=float(raw[0,0,0]-baseline[0,0,0])
        self.assertEqual(terms['target_image_L1'],0.)
        self.assertEqual(terms['target_adjacent_L1'],0.)
        self.assertAlmostEqual(terms['U96_block16_luminance_anchor_L1'],d,places=14)
        self.assertAlmostEqual(value,d*d/.01,places=14)
        self.assertAlmostEqual(terms['U96_block16_luminance_anchor_SmoothL1'],d*d/.02,places=14)

    def test_local_and_global_keep_original_paired_losses(self):
        target=np.zeros((256,256,3),dtype=np.float32)
        raw=np.full_like(target,.02);base=np.ones_like(target)
        local,lt=A.objective(raw,target,base,'local_gaussian_1p2',np.float64)
        glob,gt=A.objective(raw,target,base,'global_gaussian_1p2',np.float64)
        self.assertAlmostEqual(local,float(np.float32(.02)),places=15)
        self.assertEqual(local,glob)
        self.assertTrue(all('Smooth' not in k for k in lt|gt))


class PixelAndEdgeFixtures(unittest.TestCase):
    def test_floor_uses_float32_multiply_no_round_or_rint(self):
        raw=np.zeros((256,256,3),dtype=np.float32)
        raw[0,0]=[.5,1.,np.nextafter(np.float32(1/255),np.float32(0))]
        q=A.floor_rgb8(raw)
        np.testing.assert_array_equal(q[0,0],[127,255,0])
        for value in [np.nan,-.01,1.01]:
            bad=raw.copy();bad[0,0,0]=value
            with self.assertRaises(AssertionError):A.floor_rgb8(bad)

    def test_whole_mean_of_HVmeans_distinct_from_joint_unequal_edgecounts(self):
        y,x=np.mgrid[:3,:4]
        value=np.repeat((x+4*y)[:,:,None],3,axis=2).astype(np.float64)
        target=np.zeros_like(value);mask=np.ones((3,4),dtype=bool)
        metric=A.region_metrics(value,target,mask)['adjacent_rgb_error']
        objective=A.l1_adj(value,target,mask,np.float64)[1]
        self.assertEqual(metric,2.5)
        self.assertEqual(objective,41/17)
        self.assertNotEqual(metric,objective)

    def test_outside_excludes_every_central_cross_boundary_edge(self):
        zero=np.zeros((256,256,3),dtype=np.float32);value=zero.copy()
        value[64:192,64:192]=1.
        result=A.metrics(value,zero)
        self.assertEqual(result['outside'],{'mae':0.,'adjacent_rgb_error':0.})
        self.assertEqual(result['central128'],{'mae':1.,'adjacent_rgb_error':0.})
        self.assertGreater(result['whole']['adjacent_rgb_error'],0.)
        self.assertEqual(A.EDGE_COUNTS['central128']['horizontal_edges'],128*127)
        self.assertEqual(A.EDGE_COUNTS['outside']['horizontal_edges'],256*255-128*127-256)

    def test_masked_joint_edges_match_explicit_loop_with_unequal_HVcounts(self):
        y,x=np.mgrid[:3,:4];v=np.repeat((x+4*y)[:,:,None],3,axis=2).astype(np.float64)
        mask=np.array([[True,True,True,True],[True,False,False,False],[True,False,False,False]])
        total=0.;count=0
        for row in range(3):
            for col in range(4):
                for dr,dc in [(0,1),(1,0)]:
                    rr,cc=row+dr,col+dc
                    if rr<3 and cc<4 and mask[row,col] and mask[rr,cc]:
                        total+=float(np.abs(v[row,col]-v[rr,cc]).sum());count+=3
        self.assertEqual(A.region_metrics(v,np.zeros_like(v),mask)['adjacent_rgb_error'],total/count)


def scalar_fixture():
    protocol={'cases':[],'guards':[],'learning':{'slot_roles':['local_gaussian_1p2','global_gaussian_1p2','clear','lowlight_linear_0p35']}}
    baseline={};bank={}
    for role in protocol['learning']['slot_roles']:
        for i in range(8):
            key=f'{role}_{i}'
            protocol['cases'].append({'id':key,'cohort':'monitor','role':role})
            row={'id':key,'role':role,'regional_metrics':{view:{region:{'mae':0.,'adjacent_rgb_error':1.} for region in A.REGIONS} for view in ['raw','native_png']},'U96_behavior_distance':{'raw':0.,'native_png':0.},'U96_illumination_distance':{'raw':0.,'native_png':0.}}
            baseline[key]=copy.deepcopy(row);bank[key]=copy.deepcopy(row)
    return protocol,baseline,bank


class GateAndPartialFixtures(unittest.TestCase):
    def test_ordered_float_sum_old_VM_order(self):
        self.assertEqual(A.ordered_float_sum([1e16,1.,-1e16]),0.)

    def test_exact_preservation_case_boundary_no_epsilon(self):
        p,b,k=scalar_fixture();cid='global_gaussian_1p2_0'
        k[cid]['regional_metrics']['raw']['whole']['mae']=.03
        self.assertTrue(A.monitor_gate(k,b,p,1)['pass'])
        k[cid]['regional_metrics']['raw']['whole']['mae']=float(np.nextafter(.03,np.inf))
        self.assertFalse(A.monitor_gate(k,b,p,1)['pass'])

    def test_endpoint_requires_both_blur_roles_views(self):
        p,b,k=scalar_fixture()
        for row in k.values():
            if row['role'] in ['local_gaussian_1p2','global_gaussian_1p2']:
                for view in ['raw','native_png']:
                    for region in A.REGIONS:row['regional_metrics'][view][region]['adjacent_rgb_error']=.98
        self.assertTrue(A.monitor_gate(k,b,p,64)['pass'])
        for row in k.values():
            if row['role']=='global_gaussian_1p2':row['regional_metrics']['native_png']['whole']['adjacent_rgb_error']=.999
        self.assertFalse(A.monitor_gate(k,b,p,64)['pass'])

    def test_legacy_limits_remain_historical_never_rebased(self):
        p={'guards':[]};bank={}
        for role in ['clear','dark','filter']:
            for i in range(8):
                cid=f'{role}_{i}'
                row={'id':cid,'role':role,'behavior_distance':{'raw':0.,'native_png':0.},'metrics':{v:{'mae':0.,'adjacent_rgb_error':0.} for v in ['raw','native_png']}}
                p['guards'].append(copy.deepcopy(row));bank[cid]=copy.deepcopy(row)
        bank['clear_0']['behavior_distance']['raw']=.03
        self.assertTrue(A.legacy_gate(bank,p)['pass'])
        bank['clear_0']['behavior_distance']['raw']=float(np.nextafter(.03,np.inf))
        self.assertFalse(A.legacy_gate(bank,p)['pass'])

    def test_absent_or_initial_partial_never_fabricates64(self):
        p={'cases':[{'id':'c','cohort':'monitor'}],'guards':[]}
        d=A.classify_return({},p)
        self.assertFalse(d['full64_pixel_phase_supported']);self.assertIsNone(d['confirmed_optimizer_updates_observed'])
        docs={A.RUN+'/status.json':{'counters':{'optimizer_updates':0,'optimizer_update_attempts':0},'optimizer_outcome_unknown':False},A.RUN+'/failure.json':{'status':'QUALITY_STOP','phase':'snapshot_u0_quality_gates'}}
        d=A.classify_return(docs,p)
        self.assertEqual(d['classification'],'UNSUPPORTED_PARTIAL_RETURN')
        self.assertEqual(d['confirmed_optimizer_updates_observed'],0)
        self.assertFalse(d['complete_success_receipt_present'])

    def test_uncertain_partial_commit_status_stays_unknown(self):
        p={'cases':[],'guards':[]}
        docs={A.RUN+'/status.json':{'counters':{'optimizer_updates':3,'optimizer_update_attempts':4},'optimizer_outcome_unknown':False},A.RUN+'/status.pending.json':{}}
        result=A.classify_return(docs,p)
        self.assertTrue(result['optimizer_outcome_unknown_observed'])
        self.assertEqual(result['confirmed_optimizer_updates_observed'],3)
        self.assertTrue(A.classify_return({},p)['optimizer_outcome_unknown_observed'])

    def test_sparse_and_unexpected_archive_namespace_rejected(self):
        protocol={'files':{'assets/a.png':{}}}
        regular=tarfile.TarInfo(A.STAGE.name+'/assets/a.png');regular.size=1
        A.member_allowed(regular,'assets/a.png',protocol)
        with self.assertRaises(AssertionError):A.member_allowed(regular,'unexpected/payload.bin',protocol)
        sparse=tarfile.TarInfo(A.STAGE.name+'/assets/a.png');sparse.type=tarfile.GNUTYPE_SPARSE
        with self.assertRaises(AssertionError):A.member_allowed(sparse,'assets/a.png',protocol)
        linked=tarfile.TarInfo(A.STAGE.name+'/assets/a.png');linked.type=tarfile.SYMTYPE
        with self.assertRaises(AssertionError):A.member_allowed(linked,'assets/a.png',protocol)

    def test_NPY_header_bounds_and_exact_extent_before_array_creation(self):
        good=io.BytesIO();np.save(good,np.zeros((256,256,3),dtype=np.float32),allow_pickle=False)
        value=A.array(good.getvalue(),'assets/teacher.npy')
        self.assertEqual(value.shape,(256,256,3));self.assertFalse(value.flags.owndata)
        for shape,dtype in [((10**9,10**9),np.dtype('float32')),((256,256,3),np.dtype('O'))]:
            bad=io.BytesIO();np.lib.format.write_array_header_1_0(bad,{'descr':np.lib.format.dtype_to_descr(dtype),'fortran_order':False,'shape':shape})
            with self.assertRaises(AssertionError):A.array(bad.getvalue(),'assets/teacher.npy')
        with self.assertRaises(AssertionError):A.array(good.getvalue()[:-1],'assets/teacher.npy')
        with self.assertRaises(AssertionError):A.array(good.getvalue()+b'junk','assets/teacher.npy')

    def test_path_and_JSON_guards(self):
        for name in ['elsewhere/a',A.STAGE.name+'/../x',A.STAGE.name+'/a//b',A.STAGE.name+'/C:/x',A.STAGE.name+'/a\\b']:
            with self.assertRaises(AssertionError):A.canonical_member(name)
        for data in [b'{"x":1,"x":2}',b'{"x":NaN}']:
            with self.assertRaises((AssertionError,ValueError)):A.strict_json(data)

    def test_inert_import_has_no_torch_and_protocol_pin_exact(self):
        self.assertFalse(any(n=='torch' or n.startswith('torch.') for n in sys.modules))
        self.assertEqual(A.digest((A.STAGE/'protocol.json').read_bytes()),A.PROTOCOL_SHA)


if __name__=='__main__':
    unittest.main()
