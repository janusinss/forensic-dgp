"""Meaningful CPU protocol, geometry, source-state and pre-neural boundary tests."""
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest
import uuid

import numpy as np

from cctv_dgp_spatial_path_replay_v1_contract import (
    read,sha,validate,selected_vector,expected_state_hash,array_sha,NAME,UPDATES,
)
from prepare_cctv_dgp_spatial_path_replay_v1 import make_protocol

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'
DESIGN=ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_design'


@contextmanager
def fixture_directory():
    # Retain small numerical fixtures with inherited workspace permissions.
    base=DESIGN/'cpu_fixtures'
    base.mkdir(exist_ok=True)
    path=base/uuid.uuid4().hex
    assert path.resolve().is_relative_to(DESIGN.resolve())
    path.mkdir()
    yield str(path)


class Boundaries(unittest.TestCase):
    def setUp(self):
        scratch=ROOT/'scratch'
        scratch.mkdir(exist_ok=True)
        assert not scratch.is_symlink() and scratch.resolve().is_relative_to(ROOT.resolve())
        self.d=read(DESIGN/'design.json')
        self.p=make_protocol(copy.deepcopy(self.d),{})

    def test_frozen_exact_routes_order_and_100_TRAIN_cases(self):
        self.assertEqual([(s['route'],s['update']) for s in self.p['states']],[(r,u) for r in ['A','B'] for u in UPDATES])
        self.assertEqual(len(set(self.p['case_ids'])),100)
        self.assertEqual(len(self.p['reference_blocks']),20)

    def test_training_gradient_or_local_model_permission_rejected(self):
        for name,value in [('maximum_new_optimizer_updates',1),('full_corpus_epochs',1),
            ('no_training_or_gradients',False),('no_historical_train_call',False),
            ('local_neural_construction_allowed',True),('new_incumbent_selection',True),('automatic_launch',True)]:
            p=copy.deepcopy(self.p);p[name]=value
            with self.subTest(name=name),self.assertRaises(AssertionError):validate(p,self.d)

    def test_structure_appearance_brightness_and_radius_weakening_rejected(self):
        for name,value in [('early_structure_gain',.009),('MSE_regression_tolerance',1e-6),
            ('SSIM_ArcFace_regression_tolerance',1e-4),('brightness_fraction_maximum',.3)]:
            p=copy.deepcopy(self.p);p['scientific_thresholds'][name]=value
            with self.subTest(name=name),self.assertRaises(AssertionError):validate(p,self.d)
        p=copy.deepcopy(self.p);p['common_radius_L2']*=2
        with self.assertRaises(AssertionError):validate(p,self.d)

    def test_time_storage_neural_count_caps_cannot_be_expanded(self):
        for name in ['worker_seconds','decoder_batch_calls','recognizer_batch_calls','output_byte_cap','maximum_backwards']:
            p=copy.deepcopy(self.p);p['budgets'][name]+=1
            with self.subTest(name=name),self.assertRaises(AssertionError):validate(p,self.d)

    def test_snapshot_or_case_selection_mutation_rejected(self):
        for key in ['states','case_ids','reference_blocks']:
            p=copy.deepcopy(self.p);p[key]=list(reversed(p[key]))
            with self.subTest(key=key),self.assertRaises(AssertionError):validate(p,self.d)

    def test_source_fingerprint_cannot_be_substituted(self):
        p=copy.deepcopy(self.p);p['source_protocol_sha256']='0'*64
        with self.assertRaises(AssertionError):validate(p,self.d)

    def vector_fixture(self,path,value,radius=.1):
        np.save(path,value,allow_pickle=False)
        initial=np.zeros(value.shape,np.float32)
        s=dict(source_file=path.name,npz_key=None,file_sha256=sha(path),vector_sha256=array_sha(value),elements=value.size,
            scientific_radius_L2=radius,original_distance_L2=float(np.linalg.norm(value.astype(np.float64))))
        return initial,s

    def test_vector_bound_exact_values_before_parameter_replay(self):
        with fixture_directory() as directory:
            path=Path(directory)/'fixture.npy';a=np.array([.01,.02,0,0],np.float32)
            initial,s=self.vector_fixture(path,a)
            self.assertTrue(np.array_equal(selected_vector(Path(directory),s,initial),a))
            s['scientific_radius_L2']=.001
            with self.assertRaises(AssertionError):selected_vector(Path(directory),s,initial)

    def test_nonfinite_or_wrong_dtype_or_wrong_shape_vectors_rejected(self):
        (ROOT/'scratch').mkdir(exist_ok=True)
        for a in [np.array([np.nan,0],np.float32),np.zeros(2,np.float64),np.zeros((2,2),np.float32)]:
            with self.subTest(dtype=str(a.dtype),shape=a.shape),fixture_directory() as directory:
                path=Path(directory)/'fixture.npy';initial,s=self.vector_fixture(path,a)
                with self.assertRaises(AssertionError):selected_vector(Path(directory),s,initial)

    def test_vector_bytes_hash_rejects_wrong_declared_journal(self):
        with fixture_directory() as directory:
            path=Path(directory)/'fixture.npy';initial,s=self.vector_fixture(path,np.zeros(4,np.float32))
            s['vector_sha256']='0'*64
            with self.assertRaises(AssertionError):selected_vector(Path(directory),s,initial)

    def test_read_only_hash_stream_matches_four_actual_initial_endpoint_states(self):
        import torch  # existing serialized source tensors only
        old=read(SOURCE/'protocol.json')
        for route in ['A','B']:
            frozen=torch.load(SOURCE/('outputs/spatial_readout_v1/state_'+route+'0/full_state.pth'),map_location='cpu',weights_only=True)['model']
            layout=read(SOURCE/('outputs/spatial_readout_v1/'+route+'_parameter_layout.json'))
            initial=np.load(SOURCE/('outputs/spatial_readout_v1/'+route+'_initial_vector.npy'),allow_pickle=False)
            for state in [s for s in self.d['states'] if s['route']==route and s['update'] in [0,50]]:
                vector=selected_vector(SOURCE,state,initial)
                h=expected_state_hash(frozen,old['trainable_parameter_names'][route],layout,vector)
                self.assertEqual(h,read(SOURCE/('outputs/spatial_readout_v1/state_'+route+str(state['update'])+'/receipt.json'))['model_state'])

    def retained_calibration_cases(self):
        from PIL import Image
        parent=ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
        sys.path.insert(1,str(parent))
        from cctv_dgp_head4_lossless_v1 import unpack
        from frozen_capacity_contract import feature_support
        data=read(parent/'protocol.json');refs={r['id']:r for r in data['references']};cases={c['id']:c for c in data['cases']}
        def pixels(path,mode='RGB'):
            with Image.open(path) as im:return np.asarray(im.convert(mode)).copy()
        for serial in [0,16]:
            block=self.d['reference_blocks'][serial];ref=refs[block['reference']]
            baseline,ev=unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz",block['case_ids'])
            target=pixels(parent/ref['target']);mask=pixels(parent/ref['observed'],'L')>0
            for label in ['baseline','A50','B50']:
                out=SOURCE/'outputs/spatial_readout_v1'/label
                saved={r['case']:r for r in read(out/'metrics.json')['rows']}
                with np.load(out/'packs'/f'r{serial:02d}.npz',allow_pickle=False) as pack:
                    raw,er,ep=pack['raw'].copy(),pack['embeddings_raw'].copy(),pack['embeddings_png'].copy()
                for i,cid in enumerate(block['case_ids']):
                    c=cases[cid];camera=pixels(parent/c['input']);png=pixels(out/'png'/(cid+'.png'))
                    support=feature_support(mask,c['landmarks5_canvas_xy'])
                    args=(raw[i],png,camera,target,mask,support,baseline[i],ev,er,ep,i,label=='baseline')
                    yield args,saved[cid]

    def test_future_return_scorer_matches_30_actual_RAW_PNG_source_controls(self):
        from audit_cctv_dgp_spatial_path_replay_v1_return import recompute_metrics
        count=0
        for args,saved in self.retained_calibration_cases():
            rm,pm=recompute_metrics(*args)
            for mode,metrics in [('raw',rm),('png',pm)]:
                for key,value in metrics.items():
                    if value is None or isinstance(value,bool):self.assertEqual(value,saved[mode][key])
                    else:self.assertLessEqual(abs(value-saved[mode][key]),1e-13)
            count+=1
        self.assertEqual(count,30)

    def test_future_return_scorer_rejects_changed_PNG_and_protected_exterior(self):
        from audit_cctv_dgp_spatial_path_replay_v1_return import recompute_metrics
        args,_=next(self.retained_calibration_cases());args=list(args)
        args[1]=args[1].copy();args[1][0,0,0]^=np.uint8(1)
        with self.assertRaises(AssertionError):recompute_metrics(*args)
        for args,_ in self.retained_calibration_cases():
            if (~args[4]).any():
                a=list(args);a[0]=a[0].copy();pos=np.argwhere(~a[4])[0];a[0][pos[0],pos[1],0]=.123
                with self.assertRaises(AssertionError):recompute_metrics(*a)
                break
        else:self.fail('Expected an actual letterboxed calibration input')

    def test_new_worker_has_no_learning_optimizer_or_historical_training_call(self):
        tree=ast.parse((ROOT/'scripts/cctv_dgp_spatial_path_replay_v1_vm.py').read_text(encoding='utf-8'))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        forbidden=['backward','step','enable_learning','train','supervise','replay']
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in calls))
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='optim' for n in ast.walk(tree)))

    @unittest.skipUnless(platform.system()=='Windows','Actual Windows pre-neural guard')
    def test_actual_review_export_supervise_commands_reject_Windows_before_files_or_models(self):
        root=ROOT/'scratch'/NAME
        self.assertFalse(root.exists())
        for flag in ['--review','--export','--supervise']:
            command=[sys.executable,'-B',str(ROOT/'scripts/cctv_dgp_spatial_path_replay_v1_vm.py'),
                '--root',str(root),'--protocol-sha','0'*64,flag]
            result=subprocess.run(command,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Human manual Linux L4 tmux replay only',result.stderr)
            self.assertFalse(root.exists())

    def test_all_new_source_has_Python310_syntax(self):
        for name in ['cctv_dgp_spatial_path_replay_v1_contract.py','cctv_dgp_spatial_path_replay_v1_vm.py','audit_cctv_dgp_spatial_path_replay_v1_return.py']:
            ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'),feature_version=(3,10))


if __name__=='__main__':unittest.main(verbosity=2)
