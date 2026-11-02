"""Pure CPU tensor/layout checks; fake callable only, no model/load/gradient."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'research/completion_codeformer_app_parity_v1/app_contract.py'
spec=importlib.util.spec_from_file_location('completion_app_contract_repair',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
app_source=ROOT/'pretrained_completion.py'
if hashlib.sha256(app_source.read_bytes()).hexdigest()!=helper.APP_SOURCE_SHA256:
    raise RuntimeError('Admitted app source changed; expression comparison invalid')

# Lift the ACTUAL original method instead of rewriting its reference expressions.
module=ast.parse(app_source.read_text(encoding='utf-8'))
cls=next(node for node in module.body if isinstance(node,ast.ClassDef) and node.name=='CodeFormerCompletion')
forward=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='forward')
namespace={'torch':torch,'F':F,'validate_mask':lambda image,mask:None,
           'compose':lambda observed,generated,alpha:observed*(1-alpha)+generated*alpha}
exec(compile(ast.Module(body=[forward],type_ignores=[]),str(app_source),'exec'),namespace)

class FakeRawCallable:
    def __init__(self):self.calls=[]
    def __call__(self,value,**kwargs):
        self.calls.append({'input':value.clone(),'layout':helper.tensor_layout(value),'kwargs':kwargs})
        return (value*.75,)

class TensorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def source(self,batch=1,channels_last=True):
        rng=np.random.default_rng(240)
        rows=[torch.from_numpy(rng.integers(0,256,(256,256,3),dtype=np.uint8)).permute(2,0,1).unsqueeze(0).to(torch.float32)/255 for _ in range(batch)]
        image=rows[0] if batch==1 else torch.cat(rows,dim=0)
        if not channels_last:image=image.contiguous()
        mask=torch.zeros((batch,1,256,256));mask[-1,:,78:132,48:208]=1
        return image,mask
    def compare(self,image,mask):
        fake=FakeRawCallable();holder=type('Holder',(),{'net':fake})()
        reference=namespace['forward'](holder,image,mask)
        active,value,m512=helper.prepare_active(image,mask)
        raw=None if value is None else value*.75
        own=helper.compose_active(image,mask,active,raw)
        self.assertTrue(torch.equal(reference,own))
        if fake.calls:
            self.assertTrue(torch.equal(value,fake.calls[0]['input']))
            self.assertEqual(helper.tensor_layout(value),fake.calls[0]['layout'])
            self.assertEqual(fake.calls[0]['kwargs'],{'w':1,'adain':False})
        self.assertTrue(torch.equal(own*(1-mask),image*(1-mask)))
        return value,m512
    @torch.inference_mode()
    def test_actual_HWC_loader_matches_active_wrapper_values_and_strides(self):
        image,mask=self.source();value,_=self.compare(image,mask)
        visible=1-mask
        old=F.interpolate(image*visible,size=(512,512),mode='bilinear',align_corners=False)/F.interpolate(visible,size=(512,512),mode='bilinear',align_corners=False).clamp_min(1e-8)
        m=F.interpolate(mask,size=(512,512),mode='nearest');old=(old*(1-m)+m)*2-1
        self.assertTrue(torch.equal(old,value))
        self.assertNotEqual(old.stride(),value.stride())
        self.assertEqual(old.stride(),(786432,262144,512,1))
        self.assertEqual(value.stride(),(786432,1,1536,3))
    @torch.inference_mode()
    def test_contiguous_loader_matches_canonical_wrapper_too(self):
        image,mask=self.source(channels_last=False);self.compare(image,mask)
    @torch.inference_mode()
    def test_mixed_batch_leaves_empty_member_unchanged(self):
        image,mask=self.source(batch=2);value,_=self.compare(image,mask)
        self.assertEqual(value.shape[0],1)
    @torch.inference_mode()
    def test_empty_batch_avoids_callable(self):
        image,mask=self.source();mask.zero_();value,m=self.compare(image,mask)
        self.assertIsNone(value);self.assertIsNone(m)
    @torch.inference_mode()
    def test_preparation_ignores_covered_source_values(self):
        image,mask=self.source();active,x,_=helper.prepare_active(image,mask)
        changed=image*(1-mask)+(1-image)*mask
        _,other,_=helper.prepare_active(changed,mask)
        self.assertTrue(torch.equal(x,other))

if __name__=='__main__':unittest.main(verbosity=2)
