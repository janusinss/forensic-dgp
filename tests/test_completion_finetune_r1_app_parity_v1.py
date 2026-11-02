"""Actual R1 nested tensor expressions versus admitted app, with a fake callable.

No pretrained model/load, neural network, gradient computation or VM execution.
"""
import ast
import hashlib
from pathlib import Path
import unittest
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'pretrained_completion.py'
OLD=ROOT/'scripts/cctv_completion_codeformer_finetune_v1_vm.py'
NEW=ROOT/'scripts/cctv_completion_codeformer_finetune_v1_r1_vm.py'

def need(ok,message):
    if not ok:raise ValueError(message)
def lifted(path,name,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

class FakeCallable:
    def __init__(self):self.input=None;self.stride=None;self.calls=0
    def __call__(self,x,**kwargs):
        self.calls+=1;self.input=x.clone();self.stride=x.stride()
        return (x*.75,)

class FreshR1TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.env={'torch':torch,'F':F,'need':need}
        cls.prepare=staticmethod(lifted(NEW,'app_input',cls.env))
        cls.deliver=staticmethod(lifted(NEW,'delivered',cls.env))
        cls.app=staticmethod(lifted(APP,'forward',{'torch':torch,'F':F,'validate_mask':lambda a,b:None,'compose':lambda a,b,m:a*(1-m)+b*m}))
    def fixture(self,kind,contiguous=False):
        rgb=torch.from_numpy(np.random.default_rng(240).integers(0,256,(256,256,3),dtype=np.uint8)).permute(2,0,1).unsqueeze(0).to(torch.float32)/255
        if contiguous:rgb=rgb.contiguous()
        mask=torch.zeros((1,1,256,256))
        shapes={'lower_face':(140,232,38,218),'eye_band':(78,132,48,208),'image_left_lens':(82,127,54,117),'cheek_object':(126,188,148,208)}
        y1,y2,x1,x2=shapes[kind];mask[:,:,y1:y2,x1:x2]=1
        return rgb,mask
    @torch.inference_mode()
    def compare(self,kind,contiguous):
        rgb,mask=self.fixture(kind,contiguous);fake=FakeCallable();holder=type('Holder',(),{'net':fake})()
        reference=self.app(holder,rgb,mask)
        x,m512=self.prepare(rgb,mask);own=self.deliver(rgb,mask,x*.75)
        self.assertTrue(torch.equal(own,reference));self.assertTrue(torch.equal(x,fake.input));self.assertEqual(x.stride(),fake.stride)
        self.assertTrue(torch.equal(own*(1-mask),rgb*(1-mask)));self.assertEqual(fake.calls,1)
    def test_lower_face_HWC(self):self.compare('lower_face',False)
    def test_eye_band_HWC(self):self.compare('eye_band',False)
    def test_left_lens_HWC(self):self.compare('image_left_lens',False)
    def test_cheek_HWC(self):self.compare('cheek_object',False)
    def test_lower_face_NCHW(self):self.compare('lower_face',True)
    def test_eye_band_NCHW(self):self.compare('eye_band',True)
    def test_left_lens_NCHW(self):self.compare('image_left_lens',True)
    def test_cheek_NCHW(self):self.compare('cheek_object',True)
    @torch.inference_mode()
    def test_empty_delivery_is_exact_and_preparation_refuses_missing_bypass(self):
        rgb,mask=self.fixture('lower_face');mask.zero_()
        self.assertTrue(torch.equal(self.deliver(rgb,mask,None),rgb))
        with self.assertRaisesRegex(ValueError,'caller bypasses empty masks'):self.prepare(rgb,mask)
    def test_failed_original_and_training_contract_are_unchanged(self):
        self.assertEqual(hashlib.sha256(OLD.read_bytes()).hexdigest(),'8f3a69bc3209590a6919adfa67afa9dbd3da4857e9180c337123a6ee65e332cb')
        def contract(path):
            tree=ast.parse(path.read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONTRACT' for t in n.targets))
            namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<contract>','exec'),namespace);return namespace['CONTRACT']
        self.assertEqual(contract(OLD),contract(NEW))
        ast.parse(NEW.read_text(),feature_version=(3,10))
        self.assertNotIn('file_digest',NEW.read_text())

if __name__=='__main__':unittest.main(verbosity=2)
