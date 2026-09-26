"""Contracts for masked completion, independent of downloaded face models."""
import importlib.util
import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path
import cv2


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('completion'),
                             'The completion pipeline is not implemented yet')
        torch.set_num_threads(2)

    def test_composition_preserves_visible_pixels_and_empty_mask(self):
        from completion import compose
        visible, generated = torch.rand(2,3,32,32), torch.rand(2,3,32,32)
        mask = torch.zeros(2,1,32,32)
        mask[0,:,10:20,5:25] = 1
        result = compose(visible, generated, mask)
        self.assertTrue(torch.equal(result[mask.expand_as(result)==0], visible[mask.expand_as(result)==0]))
        self.assertTrue(torch.equal(result[mask.expand_as(result)==1], generated[mask.expand_as(result)==1]))
        self.assertTrue(torch.equal(result[1], visible[1]))

    def test_invalid_masks_are_rejected(self):
        from completion import compose
        x = torch.zeros(1,3,32,32)
        for m in (torch.full((1,1,32,32),float('nan')), torch.full((1,1,32,32),2.), torch.zeros(1,3,32,32)):
            with self.assertRaises(ValueError):
                compose(x,x,m)

    def test_region_error_is_area_normalized_and_empty_is_excluded(self):
        from completion import region_error
        x = torch.zeros(2,3,32,32,requires_grad=True)
        y = torch.ones_like(x)
        m = torch.zeros(2,1,32,32)
        m[0,:,0,0] = 1
        error, valid = region_error(x,y,m)
        self.assertEqual(valid.tolist(), [True,False])
        self.assertEqual(error[0].item(),1.)
        error.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_covering_is_reproducible_and_does_not_modify_target(self):
        from completion import synthetic_covering
        target = np.full((64,64,3),128,np.uint8)
        original = target.copy()
        first = synthetic_covering(target,seed=91,kind='lower')
        second = synthetic_covering(target,seed=91,kind='lower')
        self.assertTrue(np.array_equal(first[0],second[0]))
        self.assertTrue(np.array_equal(first[1],second[1]))
        self.assertTrue(np.array_equal(target,original))
        self.assertGreater(first[1].sum(),0)
        self.assertLess(first[1].mean(),.8)
        self.assertTrue(np.array_equal(first[0][first[1]==0],target[first[1]==0]))

    def test_model_ignores_contents_inside_supplied_hole(self):
        from completion import CompletionNet
        model = CompletionNet(width=8).eval()
        x = torch.rand(1,3,64,64)
        mask = torch.zeros(1,1,64,64)
        mask[:,:,25:50,10:54] = 1
        other = x*(1-mask) + torch.rand_like(x)*mask
        with torch.no_grad():
            a = model.complete(x,mask)
            b = model.complete(other,mask)
        self.assertTrue(torch.equal(a,b))
        loss = model.complete(x,mask).mean() + model.detect(x).mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_blend_support_covers_hole_and_is_bounded(self):
        from completion import blend_mask
        mask = torch.zeros(1,1,64,64)
        mask[:,:,20:40,20:40] = 1
        alpha = blend_mask(mask,radius=3)
        self.assertTrue(torch.all(alpha[mask==1]==1))
        self.assertTrue(torch.all(alpha[:,:,:15]==0))
        self.assertTrue(torch.all((alpha>=0)&(alpha<=1)))

    def test_default_blending_never_modifies_pixels_outside_the_mask(self):
        from completion import blend_mask, compose
        x = torch.rand(1,3,64,64)
        m = torch.zeros(1,1,64,64)
        m[:,:,20:40,20:40] = 1
        result = compose(x,torch.zeros_like(x),blend_mask(m))
        outside = (m==0).expand_as(x)
        self.assertTrue(torch.equal(result[outside],x[outside]))

    def test_default_inference_preserves_visible_pixels_with_supplied_mask(self):
        from completion import CompletionNet
        from completion_inference import predict
        x = torch.rand(1,3,64,64)
        m = torch.zeros(1,1,64,64)
        m[:,:,20:40,20:40] = 1
        result = predict(CompletionNet(8).eval(),x,mask=m)
        outside = (m==0).expand_as(x)
        self.assertTrue(torch.equal(result['output'][outside],x[outside]))

    def test_dataset_covers_before_degrading_and_validation_is_fixed(self):
        self.assertIsNotNone(importlib.util.find_spec('completion_data'))
        from completion_data import CompletionDataset
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'face.png'
            cv2.imwrite(str(path),np.full((64,64,3),150,np.uint8))
            data = CompletionDataset([str(path)],size=64,seed=42,validation=True)
            a = data[1]  # fixed lower-face covering, clear condition
            self.assertGreater(a['mask'].sum().item(),0)
            self.assertFalse(a['degraded'])
            self.assertTrue(torch.equal(a['input']*(1-a['mask']),a['target']*(1-a['mask'])))
            data.epoch = 19
            b = data[1]
            self.assertTrue(torch.equal(a['input'],b['input']))
            self.assertEqual(len(data),10)  # five coverings x clear/degraded
            degraded = data[6]
            self.assertTrue(degraded['degraded'])
            self.assertTrue(torch.all(degraded['mask'] >= a['mask']))

    def test_selection_rejects_visible_regression(self):
        self.assertIsNotNone(importlib.util.find_spec('train_completion'))
        from train_completion import qualifies
        baseline = {'hole_mae':.3,'visible_mae':.1,'mask_iou':.4}
        self.assertTrue(qualifies({'hole_mae':.2,'visible_mae':.1,'mask_iou':.6},baseline,baseline))
        self.assertFalse(qualifies({'hole_mae':.1,'visible_mae':.2,'mask_iou':.9},baseline,baseline))

    def test_smoke_checkpoint_cannot_be_used_as_trained_inference(self):
        self.assertIsNotNone(importlib.util.find_spec('completion_inference'))
        from completion_inference import load_completion
        from completion import CompletionNet, FORMAT
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'smoke.pth'
            torch.save({'format':FORMAT,'width':8,'model':CompletionNet(8).state_dict(),
                        'dry_run':True,'epoch':1},path)
            with self.assertRaisesRegex(ValueError,'smoke'):
                load_completion(path,'cpu')

    def test_inference_uses_only_input_and_keeps_empty_mask_unchanged(self):
        from completion import CompletionNet
        from completion_inference import predict
        x = torch.rand(1,3,64,64)
        result = predict(CompletionNet(8).eval(),x,mask=torch.zeros(1,1,64,64))
        self.assertTrue(torch.equal(result['output'],x))
        with self.assertRaisesRegex(ValueError,'Too little'):
            predict(CompletionNet(8).eval(),x,mask=torch.ones(1,1,64,64))

    def test_web_decode_rejects_invalid_image_and_mismatched_mask(self):
        self.assertIsNotNone(importlib.util.find_spec('completion_web'))
        from completion_web import decode_upload, decode_mask
        with self.assertRaises(ValueError):
            decode_upload(b'not an image')
        ok,encoded = cv2.imencode('.png',np.zeros((32,32),np.uint8))
        self.assertTrue(ok)
        with self.assertRaises(ValueError):
            decode_mask(encoded.tobytes(),(64,64))

    def test_benchmark_reports_missing_predictions_instead_of_skipping(self):
        self.assertIsNotNone(importlib.util.find_spec('completion_benchmark'))
        from completion_benchmark import prepare, score
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root/'images'
            images.mkdir()
            cv2.imwrite(str(images/'face.png'),np.full((64,64,3),150,np.uint8))
            prepare([str(images/'face.png')],root/'benchmark',size=64)
            report = score(root/'benchmark',root/'missing_predictions')
            self.assertEqual(report['expected'],10)
            self.assertEqual(report['failures'],10)
            self.assertIsNone(report['hole_mae'])

    def test_balanced_benchmark_samples_each_dataset_and_rejects_shortfall(self):
        import completion_benchmark as benchmark
        self.assertTrue(hasattr(benchmark,'balanced_sample'))
        paths=[f'dataset/ffhq/{i}.png' for i in range(10)]+[f'dataset/asian/{i}.png' for i in range(3)]
        roots=['dataset/ffhq','dataset/asian']
        a=benchmark.balanced_sample(paths,roots,2,42)
        self.assertEqual(a,benchmark.balanced_sample(list(reversed(paths)),roots,2,42))
        self.assertEqual(sum('/asian/' in p for p in a),2)
        self.assertEqual(sum('/ffhq/' in p for p in a),2)
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            benchmark.balanced_sample(paths,roots,4,42)

    def test_benchmark_source_reports_keep_failures(self):
        import json
        from completion_benchmark import score
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'manifest.json').write_text(json.dumps({'cases':[
                {'file':'a.png','source':'dataset/asian/a.png','kind':'lower','degraded':False},
                {'file':'b.png','source':'dataset/ffhq/b.png','kind':'eyes','degraded':True}]}))
            result=score(root,root/'missing')
            self.assertIn('sources',result)
            self.assertEqual(result['sources']['asian']['failures'],1)
            self.assertFalse(result['sources']['ffhq']['complete'])

    def test_web_inference_preserves_clear_input_with_explicit_empty_region(self):
        from unittest.mock import patch
        from completion import CompletionNet
        from completion_web import execute, decode_upload
        import base64
        image = np.full((64,64,3),137,np.uint8)
        _,encoded = cv2.imencode('.png',image)
        _,empty = cv2.imencode('.png',np.zeros((64,64),np.uint8))
        # In-memory dependency injection tests serialization, not checkpoint quality.
        engine = {'model':CompletionNet(8).eval(),'state':{'size':64,'epoch':1},'device':'cpu','restorer':None}
        with patch('completion_web.engine',engine):
            result = execute(encoded.tobytes(),empty.tobytes(),False)
        output = decode_upload(base64.b64decode(result['output'].split(',')[1]))
        self.assertTrue(np.array_equal(output,image))


if __name__ == '__main__':
    unittest.main()
