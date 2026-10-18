"""CPU-only data/contract/resource falsifiers; never import a neural framework."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from research.codeformer_fixed_state_response_v1 import contract as c,data,budget


def synthetic():
    refs={}
    for role,count,start in [('train',391,1),('validation',53,1000)]:
        for i in range(start,start+count):
            rid=('tr'if role=='train'else'va')+'_ffhq_%05d'%i
            refs[rid]={'id':rid,'historical_role':role,'source_sha256':'%064x'%i,
                'source_path':'data/sources/'+rid+'.png','decision':'accept_clean_restoration',
                'native_size':[1024,1024],'suitable_as_hq_train_candidate':role=='train'}
    ids=[r for r in refs if r.startswith('tr_')][:100];cases=[]
    for u in range(1,51):
        for s,rid in enumerate([ids[2*(u-1)],ids[2*(u-1)],ids[2*(u-1)+1],ids[2*(u-1)+1]],1):
            cases.append({'update':u,'slot':s,'reference_id':rid,'source_sha256':refs[rid]['source_sha256']})
    return refs,{'TRAIN_only':True,'cases':cases},{'reference_ids':[],'source_sha256':[]}


def held_doc():
    rows=[]
    for i in range(101,133):
        rid='tr_ffhq_%05d'%i
        for role in ('normal','dark'):
            stem='inputs/heldout_'+rid+'/'+role+'/'
            rows.append({'id':'heldout_'+rid+'_'+role,'reference_id':rid,'role':role,
                'input_path':stem+'input.png','pair_record_path':stem+'pair_record.json'})
    return {'format':'codeformer-fixed-state-heldout-pairs-v1','cases':rows}


class DataContractTests(unittest.TestCase):
    def test_canonical_selection_excludes_fitted_and_metadata(self):
        refs,p,e=synthetic();e['reference_ids']=['tr_ffhq_00101'];e['source_sha256']=[refs['tr_ffhq_00102']['source_sha256']]
        chosen=data.select_heldout(refs,p,e)
        self.assertEqual(chosen[0]['reference_id'],'tr_ffhq_00103');self.assertEqual(len(chosen),32)
        self.assertEqual([r['camera_update']for r in chosen[::2]],list(range(1,47,3)))
        self.assertEqual([r['pair_role']for r in chosen],['A','B']*16)

    def test_source_hash_exclusion_not_id_only(self):
        refs,p,e=synthetic();refs['tr_ffhq_00101']['source_sha256']=p['cases'][0]['source_sha256']
        chosen=data.select_heldout(refs,p,e)
        self.assertNotIn('tr_ffhq_00101',[r['reference_id']for r in chosen])

    def test_stale_parity_order_refused(self):
        refs,p,e=synthetic();p['cases'][1]['slot']=3
        with self.assertRaises(ValueError):data.select_heldout(refs,p,e)

    def test_unapproved_reference_or_dev_overlap_refused(self):
        refs,_,_=synthetic();doc={'format':'codeformer-adaptation-reference-manifest-v1','references':list(refs.values())}
        data.reference_index(doc)
        doc['references'][0]['decision']='reject'
        with self.assertRaises(ValueError):data.reference_index(doc)
        doc['references'][0]['decision']='accept_clean_restoration'
        doc['references'][-1]['source_sha256']=doc['references'][0]['source_sha256']
        with self.assertRaises(ValueError):data.reference_index(doc)

    def test_exact_static_and_row_census(self):
        doc=held_doc();fitted={'format':'codeformer-fixed-state-fitted-cases-v1','cases':[{'id':'train_u%04d_slot%d'%(u,s)}for u in range(1,51)for s in range(1,5)]}
        self.assertEqual(len(c.expected_pair_paths(doc)),128);self.assertEqual(len(c.expected_input_pngs(doc)),64)
        self.assertEqual(len(c.expected_row_ids(fitted,doc)['fitted']),200)
        doc['cases'][-1]=deepcopy(doc['cases'][0])
        with self.assertRaises(ValueError):c.expected_pair_paths(doc)

    def test_features_support_normalization_and_plane_order(self):
        a=np.zeros((256,256,3),np.float32);a[:32,:32]=[.25,.5,.75];mask=np.ones((256,256),bool);mask[32:64,:32]=False
        feature=data.features(a,mask)
        self.assertEqual(feature.dtype,np.float32);self.assertEqual(feature.shape,(256,))
        self.assertEqual([feature[0],feature[64],feature[128],feature[192]],[.25,.5,.75,1.])
        self.assertEqual(feature[200],0)
        with self.assertRaises(ValueError):data.features(a,np.zeros((256,256),bool))

    def test_features_equal_previous_shared_CPU_implementation(self):
        from research.codeformer_tone_control_training_v1.data import features as old_features
        rng=np.random.default_rng(42);a=rng.random((256,256,3),dtype=np.float32);mask=rng.random((256,256))>.2
        self.assertTrue(np.array_equal(data.features(a,mask),old_features(a,mask)))

    def test_raw_trailing_nonfinite_and_RGB_geometry_refusals(self):
        parent=Path(__file__).resolve().parents[1]/'scratch/fixed_state_response_data_tests'
        parent.mkdir(parents=True,exist_ok=True)
        self.assertTrue(parent.resolve().is_relative_to(Path(__file__).resolve().parents[1]))
        with tempfile.TemporaryDirectory(dir=parent)as folder:
            p=Path(folder)/'a.npy';a=np.zeros((256,256,3),np.float32);np.save(p,a)
            self.assertEqual(data.load_raw(p).shape,a.shape)
            with p.open('ab')as f:f.write(b'x')
            with self.assertRaises(ValueError):data.load_raw(p)
            a[0,0,0]=np.nan;np.save(p,a)
            with self.assertRaises(ValueError):data.load_raw(p)
            q=Path(folder)/'a.png';Image.new('RGB',(128,256)).save(q)
            with self.assertRaises(ValueError):data.load_rgb(q)

    def test_root_and_unsafe_path_guards_before_framework(self):
        with self.assertRaises(ValueError):c.manual_root(Path.cwd())
        for name in ('../x','x//y','C:/x','a\\b','/absolute','x./q'):
            with self.assertRaises(ValueError):c.safe_relative(name)
        self.assertFalse(any(x in sys.modules for x in ('torch','tensorflow','jax')))

    def test_resource_projection_and_mutated_admission(self):
        ledger=budget.resource_ledger();self.assertLess(ledger['projected_files'],3000)
        self.assertLess(ledger['projected_payload_bytes'],1024**3)
        self.assertLess(ledger['single_default_gzip_bound_bytes'],c.LIMITS['archive_bytes'])
        with patch.dict(c.LIMITS,{'initial_free_bytes_after_install':5*1024**3}):
            with self.assertRaises(ValueError):budget.resource_ledger()

    def test_no_neural_or_optimizer_calls_in_data_builder(self):
        root=Path(__file__).resolve().parents[1]
        for name in ('research/codeformer_fixed_state_response_v1/data.py','scripts/prepare_codeformer_fixed_state_response_v1.py'):
            tree=ast.parse((root/name).read_text(),feature_version=(3,10))
            for n in ast.walk(tree):
                if isinstance(n,(ast.Import,ast.ImportFrom)):
                    module=getattr(n,'module','')or''
                    self.assertFalse(module.startswith(('torch','tensorflow','jax')))
                if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,('backward','step','forward'))

    def test_design_counts_and_no_identity_claim(self):
        self.assertEqual(sum(c.COUNTS.values()),656)
        self.assertEqual(c.COUNTS['optimizer_steps'],0);self.assertEqual(c.COUNTS['gradient_queries'],0)
        self.assertIn('not proven human identities',c.DESIGN['data']['source_overlap_limit'])
        self.assertEqual(c.DESIGN['data']['heldout_observations'],64)


if __name__=='__main__':unittest.main()
