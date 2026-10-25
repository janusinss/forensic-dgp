"""Pure saved-array/parser/gate fixtures; no Torch, models or returned code."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('u64_independent_pixels', ROOT/'scripts/audit_u64_cleananchors_pixels_v1.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def npy(value):
    data = io.BytesIO(); np.save(data,value,allow_pickle=False); return data.getvalue()


def fixture():
    parent={'cases':[],'guards':[]};frozen={};initial={}
    def row(cid,role,monitor=True):
        result={'id':cid,'role':role,'cohort':'monitor' if monitor else 'guard'}
        result['metrics']={v:{'mae':.1,'adjacent_rgb_error':.1} for v in a.VIEWS}
        result['behavior_distance']=dict.fromkeys(a.VIEWS,.005)
        return result
    def measured(role,legacy=False):
        mae,edge=((.2,.2) if role in a.ROLES[:2] else (.11,.025) if role==a.ROLES[-1]
                  else (.02,.02) if role=='clear' and not legacy else (.1,.1))
        r={'metrics':{v:{'mae':mae,'adjacent_rgb_error':edge} for v in a.VIEWS},
           'regional_metrics':{v:{region:{'mae':mae,'adjacent_rgb_error':edge} for region in a.MASKS} for v in a.VIEWS},
           'U96_behavior_distance':dict.fromkeys(a.VIEWS,.001),
           'U96_illumination_distance':dict.fromkeys(a.VIEWS,.001)}
        if legacy:r['behavior_distance']=dict.fromkeys(a.VIEWS,.005)
        return r
    for role in a.ROLES:
        for n in range(8):
            cid=str(n)+'__'+role;parent['cases'].append(row(cid,role));frozen[cid]=measured(role)
    for role in ('clear','dark','filter'):
        for n in range(8):
            cid='guard'+str(n)+'__'+role;parent['guards'].append(row(cid,role,False));frozen[cid]=measured(role,True)
    initial=copy.deepcopy(frozen)
    for r in parent['cases']:
        for view in a.VIEWS:
            if r['role'] in a.ROLES[:2]:
                for region in a.MASKS:initial[r['id']]['regional_metrics'][view][region]={'mae':.19,'adjacent_rgb_error':.196}
            elif r['role']==a.ROLES[-1]:
                for region in a.MASKS:initial[r['id']]['regional_metrics'][view][region]={'mae':.1,'adjacent_rgb_error':.02}
    return parent,frozen,initial


def set_role(bank,parent,role,field,value,region=None,view=None):
    for r in parent['cases']:
        if r['role']==role:
            for v in ([view] if view else a.VIEWS):
                for area in ([region] if region else a.MASKS):bank[r['id']]['regional_metrics'][v][area][field]=value


class Parsers(unittest.TestCase):
    def test_duplicate_nonfinite_or_overflow_JSON_refused(self):
        for raw in [b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}']:
            with self.assertRaises(ValueError):a.strict_json(raw)

    def test_relative_paths_are_canonical_and_confined(self):
        for name in ['../x','/x','a//b','a/./b','C:/x','a\\b','x/']:
            with self.assertRaises(ValueError):a.relative(name)
        self.assertEqual(a.relative('outputs/calls/0001/raw.npy').as_posix(),'outputs/calls/0001/raw.npy')

    def test_read_requires_exact_regular_file(self):
        (ROOT/'scratch').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'scratch') as directory:
            root=Path(directory);(root/'small').write_bytes(b'x')
            self.assertEqual(a.read_regular(root,'small',1),b'x')
            with self.assertRaises(ValueError):a.read_regular(root,'small',0)
            with self.assertRaises(ValueError):a.read_regular(root,'../small',100)

    def test_npy_strict_shape_dtype_extent_and_finiteness(self):
        v=np.zeros((256,256,3),np.float32);raw=npy(v)
        self.assertTrue(np.array_equal(a.npy(raw,'raw.npy'),v))
        for data,name in [(raw+b'x','raw.npy'),(npy(v.astype(np.float64)),'raw.npy'),
                          (npy(v[:128]),'raw.npy'),(raw,'unclipped.npy')]:
            with self.assertRaises(ValueError):a.npy(data,name)
        v[0,0,0]=np.nan
        with self.assertRaises(ValueError):a.npy(npy(v),'raw.npy')

    def test_unclipped_negative_values_allowed_and_delivery_range_enforced(self):
        v=np.full((1,3,256,256),-2,np.float32)
        self.assertEqual(a.npy(npy(v),'unclipped.npy').min(),-2)
        with self.assertRaises(ValueError):a.npy(npy(v[0].transpose(1,2,0)),'raw.npy')

    def test_bound_static_hash_named_teacher_has_explicit_RAW_role(self):
        value=np.zeros((256,256,3),np.float32);raw=npy(value)
        row={'path':'assets/teachers/abcdef0123456789.npy',**a.binding(raw)}
        self.assertTrue(np.array_equal(a.static_raw(raw,row),value))
        with self.assertRaises(ValueError):a.npy(raw,row['path'])
        with self.assertRaises(ValueError):a.static_raw(raw+b'x',row)
        wrong=npy(value.astype(np.float64))
        with self.assertRaises(ValueError):a.static_raw(wrong,{'path':row['path'],**a.binding(wrong)})
        outside=npy(value-np.float32(1))
        with self.assertRaises(ValueError):a.static_raw(outside,{'path':row['path'],**a.binding(outside)})

    def test_PNG_single_RGB256_only(self):
        for size,mode in [((256,256),'RGB'),((128,256),'RGB'),((256,256),'RGBA')]:
            out=io.BytesIO();Image.new(mode,size).save(out,format='PNG')
            if mode=='RGB' and size==(256,256):self.assertEqual(a.png(out.getvalue()).shape,(256,256,3))
            else:
                with self.assertRaises(ValueError):a.png(out.getvalue())


class Arithmetic(unittest.TestCase):
    def test_floor_png_uses_saved_FP32_not_rounding(self):
        v=np.full((256,256,3),np.float32(.5),np.float32)
        self.assertEqual(int(a.floor_png(v)[0,0,0]),127)

    def test_nonwhole_edges_exclude_ROI_seam(self):
        p=np.zeros((256,256,3),np.float32);t=p.copy();p[64:192,64:192]=1
        self.assertEqual(a.regional(p,t,a.MASKS['central128'])['adjacent_rgb_error'],0)
        self.assertEqual(a.regional(p,t,a.MASKS['outside'])['adjacent_rgb_error'],0)
        self.assertGreater(a.regional(p,t,a.MASKS['whole'])['adjacent_rgb_error'],0)

    def test_training_movement_unavailable_not_zero(self):
        v=np.zeros((256,256,3),np.float32);q=np.zeros_like(v,dtype=np.uint8)
        result=a.pixel_metrics(v,q,v)
        self.assertIs(result['behavior_movement_reference_available'],False)
        self.assertNotIn('U96_behavior_distance',result)
        paired=a.pixel_metrics(v,q,v,control=(v,q))
        self.assertEqual(paired['U96_behavior_distance'],dict.fromkeys(a.VIEWS,0.0))

    def test_dark_null_remains_null(self):
        v=np.zeros((256,256,3),np.float32);q=np.zeros_like(v,dtype=np.uint8)
        result=a.pixel_metrics(v,q,None,control=(v,q),teacher=(v,q))
        self.assertEqual(result['metrics']['raw'],dict.fromkeys(a.FIELDS,None))
        self.assertIsNone(result['regional_metrics']['native_png'])

    def test_four_independent_objectives_match_local_inert_float64_reference(self):
        # Only the known local NumPy function executes; connected Torch is never called.
        from research.codeformer_u64_cleananchors_continuation_v1.objective import numpy_objective
        rng=np.random.default_rng(918);p=rng.random((256,256,3),dtype=np.float32);t=rng.random((256,256,3),dtype=np.float32)
        for role in a.ROLES:
            actual,terms=a.objective(p,t,role);expected,reference=numpy_objective(p,t,role,np=np)
            self.assertAlmostEqual(actual,expected,places=14);self.assertEqual(set(terms),set(reference))
            for key in terms:self.assertAlmostEqual(terms[key],reference[key],places=14)

    def test_clear_uses_SmoothL1_photograph_and_dark_block_anchor(self):
        p=np.full((256,256,3),.1,np.float32);t=np.zeros_like(p)
        clear,terms=a.objective(p,t,'clear');dark,dterms=a.objective(p,t,a.ROLES[-1])
        self.assertAlmostEqual(clear,2*(float(np.float32(.1))-.005),places=12)
        self.assertAlmostEqual(dark,3*float(np.float32(.1))-.01,places=12)
        self.assertIn('photographic_image_anchor_SmoothL1',terms)
        self.assertIn('photographic_block16_luminance_anchor_SmoothL1',dterms)

    def test_arithmetic_tolerance_never_changes_boolean_gate(self):
        c=a.Comparison();c.same(.1,.1+5e-16,'scalar');self.assertFalse(c.failures)
        c.same(True,False,'gate');self.assertEqual(len(c.failures),1)


class Gates(unittest.TestCase):
    def test_initial_u64_retains_original_one_percent(self):
        p,f,i=fixture();self.assertTrue(a.gates(i,i,f,p,0)['pass'])
        current=copy.deepcopy(i);set_role(current,p,a.ROLES[0],'adjacent_rgb_error',.199)
        self.assertFalse(a.gates(current,current,f,p,0)['pass'])

    def test_u8_strict_clear_both_views_and_dark_nonincrease(self):
        p,f,i=fixture();current=copy.deepcopy(i)
        set_role(current,p,'clear','mae',.019)
        self.assertTrue(a.gates(current,i,f,p,8)['pass'])
        set_role(current,p,'clear','mae',.02,view='native_png')
        result=a.gates(current,i,f,p,8)
        self.assertFalse(result['pass']);self.assertFalse(result['additional_retention']['native_png/clear_target_MAE_improves'])
        set_role(current,p,'clear','mae',.019);set_role(current,p,a.ROLES[-1],'mae',.101)
        self.assertFalse(a.gates(current,i,f,p,8)['pass'])

    def test_u4_does_not_apply_future_u8_trend(self):
        p,f,i=fixture();self.assertTrue(a.gates(i,i,f,p,4)['pass'])
        self.assertEqual(a.gates(i,i,f,p,4)['additional_retention'],{})

    def test_original_U96_monitor_baseline_not_rebased_to_u64(self):
        p,f,i=fixture();current=copy.deepcopy(i)
        set_role(i,p,'clear','mae',.029);set_role(current,p,'clear','mae',.031)
        # Only +.002 vs u64, but +.011 vs the sealed U96 target metric.
        result=a.gates(current,i,f,p,4)
        self.assertFalse(result['ordinary_monitor_pass'])

    def test_original_legacy_teacher_limits_remain_binding(self):
        p,f,i=fixture();current=copy.deepcopy(i)
        for row in p['guards']:
            if row['role']=='clear':current[row['id']]['behavior_distance']['raw']=.011
        result=a.gates(current,i,f,p,4)
        self.assertFalse(result['legacy_pass'])

    def test_terminal_four_structure_gains_and_blur_MAE_nonincrease(self):
        p,f,i=fixture();current=copy.deepcopy(i)
        set_role(current,p,'clear','mae',.019);set_role(current,p,a.ROLES[-1],'mae',.099)
        self.assertTrue(a.gates(current,i,f,p,16)['pass'])
        set_role(current,p,a.ROLES[0],'adjacent_rgb_error',.197)
        result=a.gates(current,i,f,p,16)
        self.assertTrue(result['original_terminal_pass']);self.assertFalse(result['additional_retention_pass'])
        set_role(current,p,a.ROLES[0],'adjacent_rgb_error',.196);set_role(current,p,a.ROLES[1],'mae',.191)
        self.assertFalse(a.gates(current,i,f,p,16)['pass'])

    def test_missing_bank_case_refused_instead_of_averaging_subset(self):
        p,f,i=fixture();current=copy.deepcopy(i);current.pop(next(iter(current)))
        with self.assertRaises(ValueError):a.gates(current,i,f,p,8)

    def test_one_percent_exact_boundary_has_no_epsilon(self):
        p,f,i=fixture();set_role(i,p,a.ROLES[0],'adjacent_rgb_error',.1980000000001)
        self.assertFalse(a.gates(i,i,f,p,0)['pass'])

    def test_exact292_plan_has204_calls_through_stop8(self):
        train=[f'tr{n}' for n in range(16)];monitor=[f'mon{n}' for n in range(8)]
        parent={'learning':{'optimization_reference_order':train,'monitor_reference_order':monitor},'cases':[],'guards':[]}
        for cohort,refs in [('optimize',train),('monitor',monitor)]:
            for ref in refs:
                for role in a.ROLES:parent['cases'].append({'id':ref+'__'+role,'reference_id':ref,'role':role,'cohort':cohort})
        for n in range(8):
            for role in ('clear','dark','filter'):parent['guards'].append({'id':f'guard{n}__{role}','reference_id':f'guard{n}','role':role})
        plan=a.operation_plan(parent)
        self.assertEqual(len(plan),292);self.assertEqual(sum(c['backward'] for c in plan),64)
        prefix=[c for c in plan if c['new_update']<=8]
        self.assertEqual(len(prefix),204);self.assertEqual(plan[204]['new_update'],9)
        self.assertEqual([c['slot'] for c in prefix],list(range(1,205)))


if __name__=='__main__':unittest.main()
