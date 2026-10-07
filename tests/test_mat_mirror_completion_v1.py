"""Meaningful tensor-format and support-preservation checks without pretrained forwards."""
import io
import json
import struct
import unittest

import numpy as np
import torch
from mat_mirror_completion_v1 import MATMirrorCompletion,inspect_header


class MockGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__();self.calls=[]
    def forward(self,image,keep,z,label,**options):
        self.calls.append((image.clone(),keep.clone(),options))
        return image*.25 + torch.nn.functional.dropout(torch.ones_like(image)*.025,training=True)


class CompletionChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.net=MockGenerator();self.model=MATMirrorCompletion(self.net)
        array=np.arange(256*256*3,dtype=np.uint32).reshape(256,256,3)%256
        self.image=torch.from_numpy(array.astype(np.float32)/255).permute(2,0,1)[None]
        self.mask=torch.zeros((1,1,256,256));self.mask[:,:,88:172,96:160]=1

    def test_visible_pixels_exact_after_png_quantization(self):
        result=self.model(self.image,self.mask)
        selected=self.mask.expand_as(self.image).bool()
        self.assertTrue(torch.equal(result[~selected],self.image[~selected]))
        self.assertTrue(torch.equal((result*255).round()[~selected],(self.image*255).round()[~selected]))
        self.assertEqual(self.net.calls[0][0].shape,(1,3,512,512))

    def test_covered_input_colors_cannot_influence_generation(self):
        changed=self.image.clone();changed[self.mask.expand_as(changed).bool()]=.731
        self.assertTrue(torch.equal(self.model(self.image,self.mask),self.model(changed,self.mask)))
        self.assertTrue(torch.equal(self.net.calls[0][0],self.net.calls[1][0]))
        image,keep,_=self.net.calls[0]
        self.assertTrue(torch.equal(image[~keep.expand_as(image).bool()],torch.zeros_like(image[~keep.expand_as(image).bool()])))

    def test_one_fixed_estimate_and_rng_restoration(self):
        torch.manual_seed(29);state=torch.random.get_rng_state().clone()
        first=self.model(self.image,self.mask);second=self.model(self.image,self.mask)
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertEqual(self.net.calls[0][2],{'truncation_psi':1,'noise_mode':'const'})

    def test_empty_clear_glasses_control_bypasses_generator(self):
        result,raw=self.model(self.image,torch.zeros_like(self.mask),return_raw=True)
        self.assertTrue(torch.equal(result,self.image));self.assertIsNone(raw);self.assertEqual(self.net.calls,[])

    def test_nonbinary_mask_rejected_before_model(self):
        self.mask[:,:,0,0]=.5
        with self.assertRaises(ValueError):self.model(self.image,self.mask)
        self.assertEqual(self.net.calls,[])

    def test_fully_hidden_face_rejected_before_model(self):
        with self.assertRaises(ValueError):self.model(self.image,torch.ones_like(self.mask))
        self.assertEqual(self.net.calls,[])

    def test_invalid_generator_output_rejected(self):
        class Bad(MockGenerator):
            def forward(self,*args,**kwargs):return torch.full((1,3,512,512),float('nan'))
        with self.assertRaises(FloatingPointError):MATMirrorCompletion(Bad())(self.image,self.mask)


class HeaderChecks(unittest.TestCase):
    @staticmethod
    def stream(header,payload=b'\x00\x00\x00\x00'):
        raw=header if isinstance(header,bytes) else json.dumps(header).encode()
        file=struct.pack('<Q',len(raw))+raw+payload
        return io.BytesIO(file),len(file)

    def test_valid_scalar_and_tensor_layout(self):
        stream,size=self.stream({'a':{'dtype':'F16','shape':[],'data_offsets':[0,2]},
                                 'b':{'dtype':'F16','shape':[1],'data_offsets':[2,4]}})
        rows,_=inspect_header(stream,size)
        self.assertEqual([r['elements'] for r in rows],[1,1])

    def test_duplicate_header_fields_rejected(self):
        stream,size=self.stream(b'{"a":{"dtype":"F16","dtype":"F32","shape":[2],"data_offsets":[0,4]}}')
        with self.assertRaises(ValueError):inspect_header(stream,size)

    def test_overlap_and_trailing_bytes_rejected(self):
        for header in [
            {'a':{'dtype':'F16','shape':[2],'data_offsets':[0,4]},'b':{'dtype':'F16','shape':[1],'data_offsets':[2,4]}},
            {'a':{'dtype':'F16','shape':[1],'data_offsets':[0,2]}}]:
            stream,size=self.stream(header)
            with self.assertRaises(ValueError):inspect_header(stream,size)

    def test_invalid_dimensions_and_dtypes_rejected(self):
        for shape,dtype in [([-1],'F16'),([True],'F16'),([2],'F32'),([50000001],'F16')]:
            stream,size=self.stream({'a':{'dtype':dtype,'shape':shape,'data_offsets':[0,4]}})
            with self.assertRaises(ValueError):inspect_header(stream,size)


if __name__=='__main__':
    unittest.main()
