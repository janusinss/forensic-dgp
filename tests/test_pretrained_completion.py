import importlib.util
import unittest
import torch
import tempfile
import json
from pathlib import Path
from PIL import Image


class PretrainedCompletionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def adapter(self, net):
        self.assertIsNotNone(importlib.util.find_spec('pretrained_completion'))
        from pretrained_completion import CodeFormerCompletion
        return CodeFormerCompletion(net)

    def test_explicit_mask_preserves_white_visible_pixels(self):
        class Net(torch.nn.Module):
            def forward(self,x,w,adain):
                self.last=x
                assert w==1 and adain is False
                return (torch.zeros_like(x),)
        net=Net(); adapter=self.adapter(net)
        x=torch.ones(1,3,64,64); m=torch.zeros(1,1,64,64); m[:,:,24:48,16:48]=1
        result=adapter(x,m)
        self.assertTrue(torch.equal(result[m.expand_as(x)==0],x[m.expand_as(x)==0]))
        self.assertTrue(torch.all(result[m.expand_as(x)==1]==.5))
        self.assertEqual(net.last.shape,(1,3,512,512))

    def test_empty_mask_bypasses_generation(self):
        class Net(torch.nn.Module):
            def forward(self,*args,**kwargs):
                raise AssertionError('Empty mask should not invoke the generator')
        adapter=self.adapter(Net()); x=torch.rand(1,3,64,64)
        self.assertTrue(torch.equal(adapter(x,torch.zeros(1,1,64,64)),x))

    def test_resizing_cannot_leak_covered_colors_into_generator_context(self):
        class Net(torch.nn.Module):
            def forward(self,x,**kwargs):
                self.last=x.clone()
                return (torch.zeros_like(x),)
        net=Net(); adapter=self.adapter(net)
        x=torch.rand(1,3,64,64); m=torch.zeros(1,1,64,64); m[:,:,23:47,17:49]=1
        adapter(x,m); first=net.last
        other=x*(1-m)+(1-x)*m
        adapter(other,m)
        self.assertTrue(torch.equal(first,net.last))

    def test_resizing_does_not_bleach_visible_context_with_white_fill(self):
        class Net(torch.nn.Module):
            def forward(self,x,**kwargs):
                self.last=x.clone()
                return (torch.zeros_like(x),)
        net=Net(); adapter=self.adapter(net)
        x=torch.full((1,3,64,64),.25); m=torch.zeros(1,1,64,64); m[:,:,23:47,17:49]=1
        adapter(x,m)
        resized=torch.nn.functional.interpolate(m,size=(512,512),mode='nearest')
        outside=(resized==0).expand_as(net.last)
        self.assertTrue(torch.allclose(net.last[outside],torch.full_like(net.last[outside],-.5)))

    def test_invalid_or_unsupported_masks_and_nonfinite_outputs_rejected(self):
        class Net(torch.nn.Module):
            def forward(self,x,**kwargs): return (torch.full_like(x,float('nan')),)
        adapter=self.adapter(Net()); x=torch.rand(1,3,64,64)
        for m in (torch.ones(1,1,64,64),torch.full((1,1,64,64),.5)):
            with self.assertRaises(ValueError): adapter(x,m)
        m=torch.zeros(1,1,64,64); m[:,:,24:48,16:48]=1
        with self.assertRaises(FloatingPointError): adapter(x,m)

    def test_runner_records_failures_without_writing_fake_predictions(self):
        self.assertIsNotNone(importlib.util.find_spec('run_completion_benchmark'))
        from run_completion_benchmark import run_cases
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); bench=root/'bench'
            for d in ('input','mask'): (bench/d).mkdir(parents=True)
            Image.new('RGB',(32,32),(128,128,128)).save(bench/'input/000000.png')
            Image.new('RGB',(32,32)).save(bench/'mask/000000.png')
            (bench/'manifest.json').write_text(json.dumps({'cases':[{'file':'000000.png','degraded':False}]}))
            def broken(x,m,degraded): raise RuntimeError('deliberate failure')
            report=run_cases(bench,root/'predictions',broken,{'test':True},'cpu')
            self.assertEqual(report['failures'],1)
            self.assertFalse((root/'predictions/000000.png').exists())

    def test_runner_never_reads_targets_and_preserves_case_names(self):
        self.assertIsNotNone(importlib.util.find_spec('run_completion_benchmark'))
        from run_completion_benchmark import run_cases
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); bench=root/'bench'
            for d in ('input','mask'): (bench/d).mkdir(parents=True)
            Image.new('RGB',(32,32),(128,128,128)).save(bench/'input/000007.png')
            Image.new('RGB',(32,32)).save(bench/'mask/000007.png')
            (bench/'manifest.json').write_text(json.dumps({'cases':[{'file':'000007.png','degraded':False}]}))
            report=run_cases(bench,root/'predictions',lambda x,m,d:x,{},'cpu')
            self.assertEqual(report['failures'],0)
            self.assertTrue((root/'predictions/000007.png').exists())
            with self.assertRaises(ValueError):
                run_cases(bench,root/'predictions',lambda x,m,d:x,{},'cpu')

    def test_web_can_complete_using_pretrained_backend_without_detector(self):
        import io, base64
        import numpy as np
        from unittest.mock import patch
        from completion_web import execute, decode_upload
        class Net(torch.nn.Module):
            def forward(self,x,**kwargs): return (torch.zeros_like(x),)
        generator=self.adapter(Net())
        engine={'backend':'codeformer','generator':generator,'model':None,
                'state':{'size':64,'epoch':None},'device':'cpu','restorer':None}
        im=Image.new('RGB',(64,64),(255,255,255)); image=io.BytesIO(); im.save(image,format='PNG')
        m=np.zeros((64,64),dtype=np.uint8); m[24:48,16:48]=255
        mask=io.BytesIO(); Image.fromarray(m).save(mask,format='PNG')
        with patch('completion_web.engine',engine):
            result=execute(image.getvalue(),mask.getvalue(),False)
        out=decode_upload(base64.b64decode(result['output'].split(',')[1]))
        self.assertTrue(np.all(out[m==0]==255))
        self.assertTrue(np.all(out[m==255]==128))
        self.assertEqual(result['backend'],'codeformer')

    def test_checkpoint_integrity_is_checked_before_deserializing(self):
        from pretrained_completion import load_codeformer
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.pth'; path.write_bytes(b'wrong checkpoint')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                load_codeformer(path)

    def test_web_pretrained_path_erases_covering_before_any_resize(self):
        import io
        import numpy as np
        from unittest.mock import patch
        from completion_web import execute
        class Net(torch.nn.Module):
            def forward(self,x,**kwargs):
                self.last=x.clone()
                return (torch.zeros_like(x),)
        net=Net(); generator=self.adapter(net)
        engine={'backend':'codeformer','generator':generator,'model':None,
                'state':{'size':512,'epoch':None},'device':'cpu','restorer':None}
        m=np.zeros((64,64),dtype=np.uint8); m[23:47,17:49]=255
        def png(a):
            b=io.BytesIO(); Image.fromarray(a).save(b,format='PNG'); return b.getvalue()
        x=np.full((64,64,3),100,dtype=np.uint8); other=x.copy(); other[m==255]=220
        with patch('completion_web.engine',engine):
            execute(png(x),png(m),False); first=net.last.clone()
            execute(png(other),png(m),False)
        self.assertTrue(torch.equal(first,net.last))
