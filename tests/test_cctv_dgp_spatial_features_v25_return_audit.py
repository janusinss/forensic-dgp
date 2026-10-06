"""Feature/count/gradient/gate tamper regressions, without training or prediction."""
import ast
import copy
from pathlib import Path
import sys
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_cctv_dgp_spatial_features_v25 as audit
import audit_cctv_dgp_degraded_detail_v24 as old
from audit_cctv_dgp_spatial_features_v25_features import check_rows,check_CPU_rows,load_arrays
from audit_cctv_dgp_spatial_features_v25_execution import check_execution
from import_cctv_dgp_spatial_features_v25 import read,write
RETURN=ROOT/'outputs/cctv_dgp_spatial_features_v25_return'


class ReturnedEvidence(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/'scratch'/('v25_audit_'+uuid.uuid4().hex);self.root.mkdir(parents=True)
        self.p=read(audit.BUNDLE/'protocol.json')
        self.cache=read(RETURN/'outputs/frozen_DGP_features.json')
        self.rows=copy.deepcopy(self.cache['CUDA_feature_rows'])
    def fixture(self,change=None):
        out=self.root/'return';(out/'outputs').mkdir(parents=True)
        e=read(RETURN/'outputs/execution_receipt.json');g=read(RETURN/'outputs/one_batch_gradient_preflight.json')
        t=read(RETURN/'outputs/timing_update20.json');s=read(RETURN/'supervisor_receipt.json')
        q=read(RETURN/'outputs/feature_path_gradient_update2.json');f=read(RETURN/'outputs/failure.json')
        if change:change(e,g,t,s,q)
        for name,data in [('outputs/execution_receipt.json',e),('outputs/one_batch_gradient_preflight.json',g),
            ('outputs/timing_update20.json',t),('supervisor_receipt.json',s),('outputs/feature_path_gradient_update2.json',q)]:write(out/name,data)
        (out/'trainer_exit_code.txt').write_text('1\n',encoding='ascii')
        preflight=read(next(RETURN.glob('preflight_*.json')))
        return out,[preflight],{0:{},50:{}},None,f,{**{k:None for k in g['per_tensor_gradient_sum_squares']},'kernel':None,'reflect_indices':None},e['head_state']
    def test_actual_source_bound_execution_receipts_pass_without_capacity_claim(self):
        r=check_execution(*self.fixture());self.assertTrue(r['five_projection_update2_gradients_verified']);self.assertEqual(r['updates'],50)
        self.assertNotIn('necessary_capacity_pass',r)
    def test_zero_feature_projection_gradient_rejected(self):
        def change(e,g,t,s,q):q['per_projection_gradient_sum_squares']['4']=0
        with self.assertRaisesRegex(ValueError,'projection gradients'):check_execution(*self.fixture(change))
    def test_changed_frozen_DGP_state_rejected(self):
        def change(e,g,t,s,q):q['DGP_state_unchanged']='0'*64
        with self.assertRaisesRegex(ValueError,'gradient proof'):check_execution(*self.fixture(change))
    def test_new_DGP_CPU_counter_cannot_disappear(self):
        def change(e,g,t,s,q):e['neural_forward_counts'].pop('DGP_CPU')
        with self.assertRaisesRegex(ValueError,'neural counts'):check_execution(*self.fixture(change))
    def test_legacy_four_DGP_count_cannot_replace_50_feature_forwards(self):
        def change(e,g,t,s,q):g['neural_forward_counts']['DGP']=4
        with self.assertRaisesRegex(ValueError,'neural counts'):check_execution(*self.fixture(change))
    def test_original_projection_sample_arithmetic_preserved(self):
        def change(e,g,t,s,q):t['steady_sample_seconds'][0]+=.001
        with self.assertRaisesRegex(ValueError,'timing samples'):check_execution(*self.fixture(change))
    def test_real_feature_row_schema_matches50frozen_inputs(self):check_rows(self.rows,self.p)
    def test_feature_label_shape_and_inference_flags_rejected(self):
        for change in [lambda r:r[0].__setitem__('id','target'),lambda r:r[0]['features'][0].__setitem__('shape',[1,64,256,256]),
            lambda r:r[0]['features'][0].__setitem__('inference_tensor',True),lambda r:r[0]['features'][0].__setitem__('requires_grad',True)]:
            rows=copy.deepcopy(self.rows);change(rows)
            with self.assertRaises(ValueError):check_rows(rows,self.p)
    def test_all50_CUDA_cache_bound_cannot_relax(self):
        self.rows[0]['fresh_cached_raw_maximum']=2.0001e-6
        with self.assertRaisesRegex(ValueError,'CUDA cache parity'):check_rows(self.rows,self.p)
    def test_VM_CPU_feature_bound_cannot_relax(self):
        pre=read(next(RETURN.glob('preflight_*.json')));r=pre['CPU_DGP_comparison_rows'];check_CPU_rows(r,self.p)
        r[0]['CPU_DGP_feature_maxima'][0]=5.0001e-5
        with self.assertRaisesRegex(ValueError,'feature comparison'):check_CPU_rows(r,self.p)
    def test_cache_requires_all250_declared_files_before_loading(self):
        c=copy.deepcopy(self.cache);c['files_sha256'].pop(next(iter(c['files_sha256'])))
        with self.assertRaisesRegex(ValueError,'cache receipt'):load_arrays(RETURN,self.p,self.rows,c)
    def test_fixed_original_metric_capacity_state_schema_and_tolerances_exact(self):
        a=ast.parse((ROOT/'scripts/audit_cctv_dgp_spatial_features_v25.py').read_text())
        b=ast.parse((ROOT/'scripts/audit_cctv_dgp_degraded_detail_v24.py').read_text())
        for name in ['rgb','raw_rgb','vector','observed','interior','feature_mask','pixel_metrics','detail_metric','png','aggregate','capacity','numeric_tree','state_hash','verify_initial_state','fixed_grid']:
            fn=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(fn(a)),ast.dump(fn(b)),name)
        for name in ['HEAD_RAW_TOLERANCE','RECOGNIZER_VECTOR_TOLERANCE','RECOGNIZER_COSINE_TOLERANCE']:self.assertEqual(getattr(audit,name),getattr(old,name))
    def test_saved_initial_new_head_zero_tails_and_fixed_buffers(self):
        import torch
        head=audit.make_head(audit.BUNDLE);state=torch.load(RETURN/'outputs/update0/head.pth',map_location='cpu',weights_only=True)
        p=read(next(RETURN.glob('preflight_*.json')))
        self.assertLessEqual(audit.verify_initial_state(head.state_dict(),state,p['initial_head_state']),1e-8)
        state['tail.weight'][0,0,0,0]=1e-9
        with self.assertRaisesRegex(ValueError,'zero initial tail'):audit.verify_initial_state(head.state_dict(),state,audit.state_hash(state))
        self.assertFalse(any(v.grad is not None or v.requires_grad for v in head.parameters()))


if __name__=='__main__':unittest.main()
