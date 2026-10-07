"""Transfer/matrix tamper and same-call identity contracts; no model or gradients."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest
import uuid

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/cctv_dgp_spatial_features_vm_v25'))
sys.path.insert(0,str(ROOT/'scripts'))
import import_cctv_dgp_v25_gradient_diagnostic_v1 as transfer
import audit_cctv_dgp_v25_gradient_diagnostic_v1_return as audit
from verify_cctv_dgp_v25_gradient_diagnostic_v1 import expected_layout
from cctv_dgp_batchmatched_identity_v26 import batchmatched_scores,objective_terms
from cctv_dgp_degraded_objective_v24 import assemble_terms


class GradientReturn(unittest.TestCase):
    def setUp(self):
        self.folder=ROOT/'scratch'/('gradient_return_v26_'+uuid.uuid4().hex);self.folder.mkdir(parents=True)
    def fixture(self,alter_source=False,extra_checkpoint=False):
        bundle=ROOT/'outputs/cctv_dgp_v25_gradient_diagnostic_v1_vm'
        files={n:(bundle/n).read_bytes() for n in ['protocol.json','scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py','scripts/run_gradient.sh']}
        if alter_source:files['scripts/run_gradient.sh']+=b'\n# changed\n'
        if extra_checkpoint:files['outputs/new_weights.pth']=b'not permitted'
        manifest={'complete':True,'protocol_sha256':transfer.PIN,'files_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}
        files['export_manifest.json']=json.dumps(manifest).encode()
        file=self.folder/'fixture.tar.gz'
        with tarfile.open(file,'x:gz') as tar:
            for n,data in files.items():
                info=tarfile.TarInfo(transfer.PREFIX+'/'+n);info.size=len(data);tar.addfile(info,io.BytesIO(data))
        return file
    def test_changed_source_rejected_despite_consistent_manifest(self):
        with self.assertRaisesRegex(ValueError,'Frozen diagnostic source'):transfer.inspect_archive(self.fixture(alter_source=True))
    def test_new_checkpoint_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unexpected diagnostic'):transfer.inspect_archive(self.fixture(extra_checkpoint=True))
    def test_duplicate_json_key_rejected(self):
        p=self.folder/'receipt.json';p.write_text('{"complete":true,"complete":false}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):transfer.read(p)
    def test_nonfinite_matrix_rejected(self):
        matrix=np.zeros((7,53781),np.float64);matrix[6,10]=np.nan
        with self.assertRaisesRegex(ValueError,'Finite'):audit.matrix_statistics(matrix,expected_layout())
    def test_parameter_layout_relabeling_rejected(self):
        layout=expected_layout();layout[-1]['name']='other.bias'
        with self.assertRaisesRegex(ValueError,'parameter'):audit.matrix_statistics(np.zeros((7,53781),np.float64),layout)
    def test_modified_reported_norm_rejected(self):
        stats=audit.matrix_statistics(np.zeros((7,53781),np.float64),expected_layout())
        changed=copy.deepcopy(stats);changed['component_norms'][0]=1e-5
        with self.assertRaisesRegex(ValueError,'Numeric'):audit.numeric(changed,stats,1e-12)


class MockEmbedding:
    """A deliberately batch-sensitive arithmetic stand-in; no neural parameters."""
    def __init__(self):self.calls=[]
    def embedding(self,image,mask,grid):
        self.calls.append((image.detach().clone(),mask.detach().clone(),grid.detach().clone()))
        return torch.stack((image.mean((1,2,3))-image.shape[0]*1e-6,torch.ones(image.shape[0])),1)


class BatchReference(unittest.TestCase):
    def batch(self,n=2):
        base=torch.full((n,3,4,4),.5);mask=torch.ones((n,1,4,4));mask[:,:,0]=0
        return {'base':base,'x':torch.full_like(base,.2),'mask':mask,'grid':torch.zeros((n,2,2,2)),
            'truth':torch.tensor([[1.,0.]]).repeat(n,1),'base_cosine':torch.full((n,),123.),'target':torch.full_like(base,.7),
            'feature':mask,'interior':mask,'valid7':mask,'degraded_weight':torch.full((n,),1.25),'clear_weight':torch.zeros(n)}
    def test_batch_reference_removes_mock_identical_image_false_cost(self):
        with torch.inference_mode():
            b=self.batch();identity=MockEmbedding();composite=b['base']*b['mask']+b['x']*(1-b['mask'])
            cached=torch.cat([identity.embedding(composite[i:i+1],b['mask'][i:i+1],b['grid'][i:i+1]) for i in range(2)])[:,0]
            legacy=identity.embedding(composite,b['mask'],b['grid'])[:,0]
            self.assertTrue(torch.all(5*F.relu(cached-legacy)>0))
            reference,current=batchmatched_scores(b,b['base'].clone(),identity)
            self.assertTrue(torch.equal(reference,current));self.assertEqual(torch.count_nonzero(5*F.relu(reference-current)),0)
    def test_same_call_composes_visible_support_and_repeats_geometry(self):
        with torch.inference_mode():
            b=self.batch();identity=MockEmbedding();batchmatched_scores(b,b['base'],identity)
            self.assertEqual(len(identity.calls),1);image,mask,grid=identity.calls[0]
            expected=b['base']*b['mask']+b['x']*(1-b['mask'])
            self.assertTrue(torch.equal(image[:2],expected));self.assertTrue(torch.equal(image[2:],expected))
            self.assertTrue(torch.equal(mask,torch.cat((b['mask'],b['mask']))));self.assertTrue(torch.equal(grid,torch.cat((b['grid'],b['grid']))))
    def test_actual_mock_identity_drop_still_receives_weight5(self):
        with torch.inference_mode():
            b=self.batch();reference,current=batchmatched_scores(b,b['base']-.1,MockEmbedding())
            self.assertTrue(torch.allclose(5*F.relu(reference-current),torch.full((2,),.375),atol=1e-6))
    def test_cached_per_case_score_does_not_drive_new_reference(self):
        with torch.inference_mode():
            b=self.batch();first=batchmatched_scores(b,b['base'],MockEmbedding());b['base_cosine'].fill_(-123)
            second=batchmatched_scores(b,b['base'],MockEmbedding())
            self.assertTrue(all(torch.equal(a,c) for a,c in zip(first,second)))
    def test_original_other_six_terms_preserved(self):
        with torch.inference_mode():
            b=self.batch();pred=b['base']-.1
            mean=lambda error,mask:(error*mask).sum((1,2,3))/(mask.sum((1,2,3))*error.shape[1])
            feature_errors=lambda a,t,f,i:(mean((a-t).square(),f),mean((a-t).square(),i))
            ssim=lambda a,t,v:1-mean((a-t).abs(),v)
            normals=(torch.tensor(.1),torch.tensor(.2));identity=MockEmbedding()
            reference,current=batchmatched_scores(b,pred,identity)
            feature,interior=feature_errors(pred,b['target'],b['feature'],b['interior'])
            expected=assemble_terms(feature,interior,mean((pred-b['target']).square(),b['mask']),
                mean((b['base']-b['target']).square(),b['mask']),ssim(pred,b['target'],b['valid7']),
                ssim(b['base'],b['target'],b['valid7']),current,reference,mean((pred-b['base']).square(),b['mask']),
                b['degraded_weight'],b['clear_weight'],normals)
            actual=objective_terms(b,pred,identity,mean,feature_errors,ssim,normals)
            self.assertEqual(set(actual),set(expected));self.assertTrue(all(torch.equal(actual[k],expected[k]) for k in actual))


if __name__=='__main__':unittest.main()
