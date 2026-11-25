"""Stage binding and preservation tests. Fixtures are not quality evidence."""
import asyncio
import base64
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from implementation import completion_api as api


class CompletionAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store_patch = patch.object(api, 'STORE', Path(self.temp.name))
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)
        api.ADMISSIONS.clear(); api.SESSIONS.clear()
        yy, xx = np.mgrid[:256,:256]
        self.rgb = np.stack((xx,yy,(xx+yy)//2),axis=2).astype(np.uint8)
        self.raw = api.png(self.rgb)
        _, _, _, geometry = api.restoration.prepare_crop(self.rgb)
        self.metadata = {'model': 'v3', 'selection': 'preconditioned_v34_codeformer',
                         'uploaded_bytes_sha256': api.sha(self.raw), 'result_file_sha256': api.sha(self.raw),
                         'restoration_applied': False, 'geometry': geometry}
        self.response = {'metadata': self.metadata, 'original_input': api.restoration.url(self.raw),
                         'prepared_input': api.restoration.url(self.raw), 'result': api.restoration.url(self.raw)}

    def prepare(self):
        api.admit_restoration(api.canonical(self.response))
        return api.prepare(self.raw, self.raw, api.canonical(self.metadata))['session_id']

    def test_completion_status_reports_r2_even_when_assets_are_missing(self):
        from fastapi import FastAPI
        from implementation import completion_assets
        app = FastAPI()
        api.register(app)
        endpoint = next(route.endpoint for route in app.routes
                        if getattr(route, 'path', '') == '/face/completion/status')
        with patch.object(completion_assets, 'verify_assets',
                          side_effect=completion_assets.CompletionAssetError('missing R2')):
            response = endpoint()
        self.assertEqual(response['model'], 'completion_r2')
        self.assertFalse(response['available'])
        self.assertIn('missing R2', response['error'])
        self.assertFalse(response['model_qualified'])

    def prepared_padding_fixture(self, height=620, width=503):
        from implementation.preparation import build_preview
        yy, xx = np.mgrid[:height, :width]
        native = np.stack(((xx % 2) * 255, (yy % 2) * 255,
                           ((xx + yy) % 3) * 127), axis=2).astype(np.uint8)
        uploaded = api.png(native)
        _, payloads = build_preview(uploaded, api.png(np.zeros((height, width), np.uint8)),
                                    api.png(np.full((height, width), 255, np.uint8)), 3)
        original, observed, _, geometry = api.restoration.prepare_crop(native)
        selected = api.read_png(payloads['prepared-derived-256.png'], 'RGB')
        self.assertTrue(np.any(selected[~observed] != original[~observed]),
                        'Fixture must exercise the bilinear fringe outside nearest support.')
        result = selected.copy()
        result[128, 128] = (7, 11, 13)
        metadata = {**self.metadata, 'uploaded_bytes_sha256': api.sha(uploaded),
                    'restoration_applied': True, 'preparation_applied': True, 'geometry': geometry}
        return uploaded, api.png(original), api.png(selected), result, observed, metadata

    def admit_prepared_padding_fixture(self, original, selected, result, metadata):
        result_png = api.png(result)
        metadata['result_file_sha256'] = api.sha(result_png)
        response = {'metadata': metadata, 'original_input': api.restoration.url(original),
                    'prepared_input': api.restoration.url(selected), 'result': api.restoration.url(result_png)}
        api.admit_restoration(api.canonical(response))
        return result_png

    def test_manual_preparation_fringe_is_valid_for_completion_review(self):
        for height, width in ((620, 503), (503, 620)):
            with self.subTest(height=height, width=width):
                uploaded, original, selected, result, _, metadata = self.prepared_padding_fixture(height, width)
                result_png = self.admit_prepared_padding_fixture(original, selected, result, metadata)
                with patch.object(api, 'run_worker') as worker:
                    response = api.prepare(uploaded, result_png, api.canonical(metadata))
                worker.assert_not_called()
                record = api.session(response['session_id'])
                self.assertEqual((record['path'] / 'input.png').read_bytes(), result_png)
                self.assertEqual((record['path'] / 'selected-input-256.png').read_bytes(), selected)

    def test_restoration_padding_drift_still_rejected_after_admission(self):
        uploaded, original, selected, result, observed, metadata = self.prepared_padding_fixture()
        y, x = np.argwhere(~observed)[0]
        result[y, x, 0] ^= 1
        result_png = self.admit_prepared_padding_fixture(original, selected, result, metadata)
        with self.assertRaisesRegex(ValueError, 'Restoration padding changed'):
            api.prepare(uploaded, result_png, api.canonical(metadata))

    def test_changed_retained_selected_input_is_rejected(self):
        uploaded, original, selected, result, _, metadata = self.prepared_padding_fixture()
        result_png = self.admit_prepared_padding_fixture(original, selected, result, metadata)
        record = api.ADMISSIONS[api.metadata_key(metadata)]
        (record['path'] / 'selected-input-256.png').write_bytes(original)
        with self.assertRaisesRegex(ValueError, 'Retained restoration evidence changed'):
            api.prepare(uploaded, result_png, api.canonical(metadata))


    def test_client_asserted_restoration_is_not_admitted(self):
        with self.assertRaisesRegex(ValueError, 'not produced'):
            api.prepare(self.raw,self.raw,api.canonical(self.metadata))

    def test_changed_original_or_result_refused(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'changed'):
            api.prepare(self.raw+b'x',self.raw,api.canonical(self.metadata))
        altered = self.rgb.copy(); altered[128,128] = 0
        with self.assertRaisesRegex(ValueError, 'changed'):
            api.prepare(self.raw,api.png(altered),api.canonical(self.metadata))

    def test_browser_integer_float_encoding_preserves_exact_numeric_binding(self):
        self.metadata['numeric_example'] = {'whole':1.0,'zero':0.0,'fraction':0.125,'flag':True}
        self.prepare()
        browser = copy.deepcopy(self.metadata)
        browser['numeric_example'].update(whole=1,zero=0)
        response = api.prepare(self.raw,self.raw,api.canonical(browser))
        self.assertIn('session_id',response)
        browser['numeric_example']['fraction'] = 0.126
        with self.assertRaisesRegex(ValueError,'not produced'):
            api.prepare(self.raw,self.raw,api.canonical(browser))
        self.assertNotEqual(api.metadata_key({'flag':True}),api.metadata_key({'flag':1}))

    def test_review_rejects_nonbinary_and_insufficient_mask(self):
        ident = self.prepare()
        with self.assertRaisesRegex(ValueError, 'black and white'):
            api.review(ident,api.png(np.full((256,256),127,np.uint8)))
        with self.assertRaisesRegex(ValueError, 'Too little'):
            api.review(ident,api.png(np.full((256,256),255,np.uint8)))

    def test_input_evidence_drift_refused(self):
        ident = self.prepare()
        (api.SESSIONS[ident]['path']/'input.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            api.session(ident)

    def test_confirmation_and_changed_mask_block_worker(self):
        ident = self.prepare()
        mask = api.png(np.zeros((256,256),np.uint8))
        approved = api.review(ident,mask)
        changed = np.zeros((256,256),np.uint8); changed[190,128] = 255
        with patch.object(api,'run_worker') as worker:
            with self.assertRaisesRegex(ValueError, 'Confirm'):
                api.generate(ident,mask,approved['review_id'],False,True)
            with self.assertRaisesRegex(ValueError, 'changed'):
                api.generate(ident,api.png(changed),approved['review_id'],True,True)
            worker.assert_not_called()

    def test_bad_session_id_rejected_without_path_access(self):
        for value in ('../outside', '0'*32, '', 'x'*32):
            with self.assertRaises(ValueError): api.session(value)

    def test_passive_middleware_preserves_every_response_message(self):
        self.prepare()
        messages = [{'type':'http.response.start','status':200,'headers':[(b'content-type',b'application/json')]},
                    {'type':'http.response.body','body':api.canonical(self.response)}]
        seen=[]
        async def source(scope,receive,send):
            for message in messages: await send(message)
        async def send(message): seen.append(copy.deepcopy(message))
        async def receive(): return {'type':'http.request','body':b''}
        asyncio.run(api.RestorationReceiptMiddleware(source)({'type':'http','path':'/face/restoration/restore','method':'POST'},receive,send))
        self.assertEqual(messages,seen)

    def test_receipt_failure_does_not_modify_restoration_response(self):
        messages=[{'type':'http.response.start','status':200,'headers':[]},
                  {'type':'http.response.body','body':b'not-json'}]
        seen=[]
        async def source(scope,receive,send):
            for message in messages: await send(message)
        async def send(message): seen.append(copy.deepcopy(message))
        async def receive(): return {'type':'http.request','body':b''}
        asyncio.run(api.RestorationReceiptMiddleware(source)({'type':'http','path':'/face/restoration/restore','method':'POST'},receive,send))
        self.assertEqual(messages,seen)
        self.assertTrue(api.STATE.get('admission_error'))

    def test_large_response_delivery_preserves_headers_content_and_final_boundary(self):
        body = bytes(range(256))*8500
        start = {'type':'http.response.start','status':200,'headers':[(b'content-length',str(len(body)).encode())]}
        messages=[start,{'type':'http.response.body','body':body}]
        seen=[]
        async def source(scope,receive,send):
            for message in messages: await send(message)
        async def send(message): seen.append(copy.deepcopy(message))
        async def receive(): return {'type':'http.request','body':b''}
        asyncio.run(api.BoundedResponseMiddleware(source)({'type':'http'},receive,send))
        self.assertEqual(seen[0],start)
        self.assertEqual(b''.join(m['body'] for m in seen[1:]),body)
        self.assertTrue(all(len(m['body'])<=65536 for m in seen[1:]))
        self.assertTrue(all(m['more_body'] for m in seen[1:-1]))
        self.assertFalse(seen[-1]['more_body'])

    def test_empty_result_bundle_is_replayable_and_source_preserved(self):
        ident = self.prepare()
        mask = api.png(np.zeros((256,256),np.uint8))
        approved = api.review(ident,mask)
        out = Path(self.temp.name)/'fake-output';out.mkdir()
        raw = self.rgb.astype(np.float32)/np.float32(255)
        (out/'completion.png').write_bytes(self.raw)
        np.save(out/'raw.npy',raw,allow_pickle=False)
        np.savez_compressed(out/'stages.npz',input_rgb=self.rgb,removal_mask=np.zeros((256,256),np.uint8),completed_rgb=raw,output_rgb=self.rgb)
        result={key:str(out/name) for key,name in [('png_path','completion.png'),('raw_path','raw.npy'),('stages_path','stages.npz')]}
        from implementation.completion_runtime import model_contract
        result['metadata']={**model_contract(),'neural_calls':0,'output_pins':{},
                            'source_pins':{'input':{'sha256':api.sha(self.raw)},'mask':{'sha256':api.sha(mask)}}}
        (out/'receipt.json').write_bytes(api.canonical(result['metadata']))
        with patch.object(api,'run_worker',return_value=(out,result,{'seconds':.01,'active_owned_processes':0})):
            response=api.generate(ident,mask,approved['review_id'],True,True)
        self.assertEqual(response['neural_calls'],0)
        self.assertEqual(response['metadata']['model_id'], 'completion_r2')
        self.assertEqual(response['metadata']['weights_sha256'],
                         'ed9f92baf06a73090343ad00cdc23b8493c9dde738a90cde9bdb650e69fee07c')
        self.assertEqual(response['metadata']['outside_reviewed_support_exact_to'],'restoration-result.png')
        with zipfile.ZipFile(out/'evidence.zip') as archive:
            manifest=json.loads(archive.read('manifest.json'))['sha256']
            for name,digest in manifest.items(): self.assertEqual(api.sha(archive.read(name)),digest)
            self.assertEqual(archive.read('restoration-result.png'),archive.read('completion-result.png'))
            self.assertIn(b'R2', archive.read('README.txt'))
            self.assertNotIn(b'R1 changes', archive.read('README.txt'))
        self.assertIsNone(api.SESSIONS[ident]['review'])

    def _test_images_dir(self):
        here = Path(__file__).resolve()
        for parent in (here.parents[1], here.parents[2], *here.parents[3:]):
            candidate = parent / 'test_images'
            if candidate.is_dir():
                return candidate
        raise unittest.SkipTest('test_images fixture directory is not present in this checkout')

    def _load_prepared_test_image(self, filename):
        path = self._test_images_dir() / filename
        with Image.open(path) as img:
            native = np.array(img.convert('RGB'))
        rgb256, observed, _, _ = api.restoration.prepare_crop(native)
        return rgb256, observed

    def test_detect_face_covering_uncovered_test_images_return_zero_pixels(self):
        from implementation.automatic_covering import detect_face_covering
        zero_prob = np.zeros((256, 256), np.float32)
        zero_base = np.zeros((256, 256), np.uint8)
        root = self._test_images_dir()
        covered_benchmarks = {'synthetic_data_6.png', 'synthetic_data_10.png'}
        files = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in ('.png', '.jpg') and p.name not in covered_benchmarks
        )
        self.assertGreaterEqual(len(files), 15)
        for path in files:
            with self.subTest(image=path.name):
                with Image.open(path) as img:
                    native = np.array(img.convert('RGB'))
                rgb256, observed, _, _ = api.restoration.prepare_crop(native)
                mask, meta = detect_face_covering(rgb256, zero_prob, zero_base, observed)
                self.assertEqual(mask.dtype, np.uint8)
                self.assertEqual(mask.shape, (256, 256))
                self.assertEqual(int((mask > 0).sum()), 0)
                self.assertEqual(meta['auto_detected_pixels'], 0)
                self.assertEqual(meta['families_triggered'], [])
                self.assertFalse(meta['visibility_trimmed'])

        review_zips = sorted((root / 'review files').glob('*.zip'))
        self.assertGreaterEqual(len(review_zips), 1)
        full_obs = np.ones((256, 256), bool)
        for zp in review_zips:
            with zipfile.ZipFile(zp) as archive:
                for member in ('result.png', 'original-256.png', 'prepared-256.png'):
                    if member not in archive.namelist():
                        continue
                    with self.subTest(archive=zp.name, member=member):
                        with Image.open(io.BytesIO(archive.read(member))) as img:
                            rgb256 = np.array(img.convert('RGB'))
                        mask, meta = detect_face_covering(rgb256, zero_prob, zero_base, full_obs)
                        self.assertEqual(int((mask > 0).sum()), 0)
                        self.assertEqual(meta['auto_detected_pixels'], 0)
                        self.assertEqual(meta['families_triggered'], [])

    def test_detect_face_covering_real_covered_benchmark_images(self):
        from implementation.automatic_covering import detect_face_covering
        from implementation.completion_runtime import visibility_check256
        zero_prob = np.zeros((256, 256), np.float32)
        zero_base = np.zeros((256, 256), np.uint8)

        syn6_rgb, syn6_obs = self._load_prepared_test_image('synthetic_data_6.png')
        mask6, meta6 = detect_face_covering(syn6_rgb, zero_prob, zero_base, syn6_obs)
        self.assertTrue(meta6['onnx_used'])
        self.assertIn('eyewear_or_glare', meta6['families_triggered'])
        self.assertIn('hand_or_object', meta6['families_triggered'])
        self.assertGreater(int((mask6[80:125, 55:200] > 0).sum()), 700)
        # Clear prescription lenses leave visible eyes unmasked while frames are selected
        self.assertGreaterEqual(int((mask6[94:108, 74:108] == 0).sum()), 200)
        self.assertGreaterEqual(int((mask6[94:108, 146:180] == 0).sum()), 200)
        self.assertGreater(int((mask6[150:245, 80:185] > 0).sum()), 2500)
        self.assertEqual(int((mask6[200:256, 0:78] > 0).sum()), 0)
        self.assertFalse(mask6[~syn6_obs].any())
        self.assertFalse(visibility_check256(np.where(syn6_obs, mask6, 255))['rejected'])

        # Even if the raw covering_detector proposes pixels on the bottom-left orange collar,
        # BiSeNet exterior suppression zeroes them out.
        collar_base = np.zeros((256, 256), np.uint8)
        collar_base[205:240, 25:75] = 255
        mask6_collar, _ = detect_face_covering(syn6_rgb, zero_prob, collar_base, syn6_obs)
        self.assertEqual(int((mask6_collar[200:256, 0:78] > 0).sum()), 0)

        syn10_rgb, syn10_obs = self._load_prepared_test_image('synthetic_data_10.png')
        mask10, meta10 = detect_face_covering(syn10_rgb, zero_prob, zero_base, syn10_obs)
        self.assertTrue(meta10['onnx_used'])
        self.assertIn('eyewear_or_glare', meta10['families_triggered'])
        self.assertGreater(int((mask10[84:130, 56:200] > 0).sum()), 1200)
        self.assertFalse(mask10[~syn10_obs].any())
        self.assertFalse(visibility_check256(np.where(syn10_obs, mask10, 255))['rejected'])

    def test_detect_face_covering_rejects_invalid_inputs(self):
        from implementation.automatic_covering import detect_face_covering
        rgb = np.full((256, 256, 3), 128, np.uint8)
        prob = np.zeros((256, 256), np.float32)
        base = np.zeros((256, 256), np.uint8)
        obs = np.ones((256, 256), bool)

        for bad_prob in (
            np.full((256, 256), -0.01, np.float32),
            np.full((256, 256), 1.01, np.float32),
            np.full((256, 256), np.nan, np.float32),
            np.full((256, 256), np.inf, np.float32),
            np.full((256, 256), -np.inf, np.float32),
            np.zeros((255, 256), np.float32),
        ):
            with self.assertRaises(ValueError):
                detect_face_covering(rgb, bad_prob, base, obs)

        mixed_base = np.zeros((256, 256), np.uint8)
        mixed_base[10, 10] = 1
        mixed_base[10, 11] = 255
        for bad_base in (
            np.zeros((256, 256), np.float32),
            np.full((256, 256), np.nan, np.float32),
            np.full((256, 256), 2, np.uint8),
            mixed_base,
            np.zeros((255, 256), np.uint8),
        ):
            with self.assertRaises(ValueError):
                detect_face_covering(rgb, prob, bad_base, obs)

        with self.assertRaises(ValueError):
            detect_face_covering(rgb.astype(np.float32), prob, base, obs)
        with self.assertRaises(ValueError):
            detect_face_covering(rgb, prob, base, obs.astype(np.uint8))

    def test_onnx_asset_verification_and_fallback_safety(self):
        import builtins
        import hashlib
        import stat
        import types
        from implementation import automatic_covering as ac

        sample_dir = Path(self.temp.name) / 'onnx_check'
        sample_dir.mkdir()
        payload = b'onnx-fixture-bytes-for-hash-check'
        good_file = sample_dir / 'fixture.onnx'
        good_file.write_bytes(payload)
        good_sha = hashlib.sha256(payload).hexdigest()

        self.assertTrue(ac._verify_onnx_file(good_file, len(payload), good_sha))
        self.assertFalse(ac._verify_onnx_file(good_file, len(payload) + 1, good_sha))
        self.assertFalse(ac._verify_onnx_file(good_file, len(payload), '0' * 64))

        fake_symlink = types.SimpleNamespace(st_mode=stat.S_IFLNK | 0o644, st_file_attributes=0)
        with patch.object(Path, 'lstat', return_value=fake_symlink):
            self.assertFalse(ac._verify_onnx_file(good_file, len(payload), good_sha))
        fake_reparse = types.SimpleNamespace(st_mode=stat.S_IFREG | 0o644, st_file_attributes=1024)
        with patch.object(Path, 'lstat', return_value=fake_reparse):
            self.assertFalse(ac._verify_onnx_file(good_file, len(payload), good_sha))

        saved_cache = dict(ac._ONNX_CACHE)
        base_rgb, observed = self._load_prepared_test_image('face 1 - default.jpg')
        masked = base_rgb.copy()
        masked[135:210, 80:176] = (140, 195, 225)
        zero_prob = np.zeros((256, 256), np.float32)
        zero_base = np.zeros((256, 256), np.uint8)
        try:
            ac._ONNX_CACHE.clear()
            with patch.object(ac, 'data_path', side_effect=ac.ManagedPathError('bad managed root')), \
                 patch.object(ac, '_verify_onnx_file', return_value=False):
                self.assertIsNone(ac._resolve_occlusion_model_dir())

            ac._ONNX_CACHE.clear()
            with patch.object(ac, '_resolve_occlusion_model_dir', return_value=None):
                mask, meta = ac.detect_face_covering(masked, zero_prob, zero_base, observed)
                self.assertFalse(meta['onnx_used'])
                self.assertIn('mask_or_scarf', meta['families_triggered'])
                self.assertGreater(int((mask > 0).sum()), 3000)

            real_import = builtins.__import__
            def failing_import(name, *args, **kwargs):
                if name == 'onnxruntime':
                    raise OSError('broken onnxruntime dll')
                return real_import(name, *args, **kwargs)

            ac._ONNX_CACHE.clear()
            with patch.object(ac, '_resolve_occlusion_model_dir', return_value=sample_dir), \
                 patch.object(builtins, '__import__', side_effect=failing_import):
                self.assertIsNone(ac._load_onnx_sessions())

            ac._ONNX_CACHE.clear()
            with patch.object(ac, '_resolve_occlusion_model_dir', return_value=sample_dir), \
                 patch('onnxruntime.InferenceSession', side_effect=RuntimeError('session init boom')):
                self.assertIsNone(ac._load_onnx_sessions())

            from onnxruntime.capi import onnxruntime_pybind11_state as ort_capi
            ac._ONNX_CACHE.clear()
            with patch.object(ac, '_resolve_occlusion_model_dir', return_value=sample_dir), \
                 patch('onnxruntime.InferenceSession', side_effect=ort_capi.NoSuchFile('ort pybind11 missing file')):
                self.assertIsNone(ac._load_onnx_sessions())

            class FakeIO:
                def __init__(self, name, shape):
                    self.name, self.shape = name, shape
            class BadRunSess:
                def __init__(self, mode):
                    self.mode = mode
                def get_inputs(self):
                    return [FakeIO('in', [1, 3, 256, 256])]
                def get_outputs(self):
                    return [FakeIO('out', [1, 19, 512, 512])]
                def run(self, *_args):
                    if self.mode == 'raise':
                        raise RuntimeError('inference failure')
                    if self.mode == 'ort_fail':
                        raise ort_capi.Fail('ort pybind11 inference failure')
                    return [np.full((1, 19, 512, 512), np.nan, np.float32)]

            ac._ONNX_CACHE.clear()
            ac._ONNX_CACHE['sessions'] = (BadRunSess('raise'), 'in', 'out', BadRunSess('raise'), 'in', False)
            mask_err, meta_err = ac.detect_face_covering(masked, zero_prob, zero_base, observed)
            self.assertFalse(meta_err['onnx_used'])
            self.assertIn('mask_or_scarf', meta_err['families_triggered'])

            ac._ONNX_CACHE.clear()
            ac._ONNX_CACHE['sessions'] = (BadRunSess('ort_fail'), 'in', 'out', BadRunSess('ort_fail'), 'in', False)
            mask_ort_err, meta_ort_err = ac.detect_face_covering(masked, zero_prob, zero_base, observed)
            self.assertFalse(meta_ort_err['onnx_used'])
            self.assertIn('mask_or_scarf', meta_ort_err['families_triggered'])

            nan_bise = BadRunSess('nan')
            x_ok = BadRunSess('nan')
            x_ok.run = lambda *_a: [np.ones((1, 256, 256, 1), np.float32)]
            ac._ONNX_CACHE['sessions'] = (nan_bise, 'in', 'out', x_ok, 'in', False)
            out_pair = ac._run_onnx_models(base_rgb)
            self.assertIsNotNone(out_pair)
            self.assertTrue(np.isfinite(out_pair[0]).all())
            self.assertTrue(np.isfinite(out_pair[1]).all())
        finally:
            ac._ONNX_CACHE.clear()
            ac._ONNX_CACHE.update(saved_cache)

    def test_detect_face_covering_detects_all_occlusion_families_and_neural_seeds(self):
        from implementation.automatic_covering import detect_face_covering
        base_rgb, observed = self._load_prepared_test_image('face 1 - default.jpg')
        zero_prob = np.zeros((256, 256), np.float32)
        zero_base = np.zeros((256, 256), np.uint8)

        # Surgical / cloth mask
        masked = base_rgb.copy()
        masked[135:210, 80:176] = (140, 195, 225)
        mask, meta = detect_face_covering(masked, zero_prob, zero_base, observed)
        self.assertIn('mask_or_scarf', meta['families_triggered'])
        self.assertGreater(int((mask[135:210, 80:176] > 0).sum()), 3000)
        self.assertFalse(mask[~observed].any())

        # Dark sunglasses
        sunglasses = base_rgb.copy()
        sunglasses[90:118, 70:116] = (15, 15, 15)
        sunglasses[90:118, 140:186] = (15, 15, 15)
        mask, meta = detect_face_covering(sunglasses, zero_prob, zero_base, observed)
        self.assertIn('eyewear_or_glare', meta['families_triggered'])
        self.assertGreater(int((mask[90:118, 70:116] > 0).sum()), 800)
        self.assertGreater(int((mask[90:118, 140:186] > 0).sum()), 800)

        # Specular lens glare
        glare = base_rgb.copy()
        glare[96:112, 85:105] = (250, 250, 250)
        mask, meta = detect_face_covering(glare, zero_prob, zero_base, observed)
        self.assertIn('eyewear_or_glare', meta['families_triggered'])
        self.assertGreater(int((mask[96:112, 85:105] > 0).sum()), 200)

        # Foreign non-skin object occluder
        foreign = base_rgb.copy()
        foreign[115:155, 110:150] = (40, 190, 55)
        mask, meta = detect_face_covering(foreign, zero_prob, zero_base, observed)
        self.assertIn('foreign_object', meta['families_triggered'])
        self.assertGreater(int((mask[115:155, 110:150] > 0).sum()), 1000)

        # Neural seed expansion (sub-0.5 probability seed growing into Lab-similar skin-toned patch)
        seeded_img = base_rgb.copy()
        seeded_img[125:175, 100:156] = (195, 150, 130)
        prob = np.zeros((256, 256), np.float32)
        prob[140:152, 118:138] = np.float32(0.35)
        mask, meta = detect_face_covering(seeded_img, prob, zero_base, observed)
        self.assertIn('neural_seed', meta['families_triggered'])
        self.assertGreater(meta['neural_seed_pixels'], 0)
        self.assertGreater(int((mask > 0).sum()), meta['neural_seed_pixels'])

    def test_detect_face_covering_clamps_overcoverage_to_pass_visibility_check(self):
        from implementation.automatic_covering import detect_face_covering
        from implementation.completion_runtime import visibility_check256
        base_rgb, observed = self._load_prepared_test_image('face 1 - default.jpg')
        giant_prob = np.full((256, 256), 0.9, np.float32)
        giant_base = np.where(observed, 255, 0).astype(np.uint8)
        mask, meta = detect_face_covering(base_rgb, giant_prob, giant_base, observed)
        self.assertTrue(meta['visibility_trimmed'])
        self.assertFalse(mask[~observed].any())
        self.assertGreater(int((mask > 0).sum()), 0)
        vis = visibility_check256(np.where(observed, mask, 255))
        self.assertFalse(vis['rejected'])

    def _run_propose_review_generate(self, canvas_rgb, expected_neural_calls):
        from implementation.covering_detector import proposal_from_probability
        from implementation.completion_runtime import model_contract

        canvas_raw = api.png(canvas_rgb)
        native_png = api.png(canvas_rgb)
        prepared_rgb, observed, _, geometry = api.restoration.prepare_crop(canvas_rgb)
        self.assertTrue(np.array_equal(prepared_rgb, canvas_rgb))
        meta = {
            'model': 'v3',
            'selection': 'preconditioned_v34_codeformer',
            'uploaded_bytes_sha256': api.sha(native_png),
            'result_file_sha256': api.sha(canvas_raw),
            'restoration_applied': False,
            'geometry': geometry,
        }
        resp = {
            'metadata': meta,
            'original_input': api.restoration.url(canvas_raw),
            'prepared_input': api.restoration.url(canvas_raw),
            'result': api.restoration.url(canvas_raw),
        }
        api.admit_restoration(api.canonical(resp))
        ident = api.prepare(native_png, canvas_raw, api.canonical(meta))['session_id']

        det_out = Path(self.temp.name) / ('det-out-' + ident[:8])
        det_out.mkdir()
        prob = np.zeros((256, 256), np.float32)
        det_raw, det_prop, det_meta = proposal_from_probability(prob, observed)
        np.save(det_out / 'detector-probability.npy', prob, allow_pickle=False)
        (det_out / 'detector-raw-mask.png').write_bytes(api.png(det_raw))
        (det_out / 'detector-proposed-mask.png').write_bytes(api.png(det_prop))
        (det_out / 'receipt.json').write_bytes(api.canonical(det_meta))
        det_result = {
            'probability_path': str(det_out / 'detector-probability.npy'),
            'raw_mask_path': str(det_out / 'detector-raw-mask.png'),
            'proposed_mask_path': str(det_out / 'detector-proposed-mask.png'),
            'metadata': det_meta,
        }
        with patch.object(api, 'run_worker', return_value=(det_out, det_result, {'seconds': .01, 'active_owned_processes': 0})):
            proposal = api.propose(ident)

        prefix = 'data:image/png;base64,'
        self.assertTrue(proposal['proposal_png'].startswith(prefix))
        proposed_png_bytes = base64.b64decode(proposal['proposal_png'][len(prefix):])
        proposed_mask = api.read_png(proposed_png_bytes, 'L')
        reviewed = api.review(ident, proposed_png_bytes, proposal['proposal_id'])
        self.assertFalse(reviewed['visibility']['rejected'])

        comp_out = Path(self.temp.name) / ('comp-out-' + ident[:8])
        comp_out.mkdir()
        completed_rgb = canvas_rgb.copy()
        if proposed_mask.any():
            completed_rgb[proposed_mask > 0] = (180, 145, 130)
        comp_raw = completed_rgb.astype(np.float32) / np.float32(255)
        completed_rgb_replay = np.rint(comp_raw * np.float32(255)).clip(0, 255).astype(np.uint8)
        completed_rgb_replay[proposed_mask == 0] = canvas_rgb[proposed_mask == 0]
        comp_png_bytes = api.png(completed_rgb_replay)
        (comp_out / 'completion.png').write_bytes(comp_png_bytes)
        np.save(comp_out / 'raw.npy', comp_raw, allow_pickle=False)
        np.savez_compressed(
            comp_out / 'stages.npz',
            input_rgb=canvas_rgb,
            removal_mask=(proposed_mask // 255).astype(np.uint8),
            completed_rgb=comp_raw,
            output_rgb=completed_rgb_replay,
        )
        comp_result = {
            'png_path': str(comp_out / 'completion.png'),
            'raw_path': str(comp_out / 'raw.npy'),
            'stages_path': str(comp_out / 'stages.npz'),
            'metadata': {
                **model_contract(),
                'neural_calls': expected_neural_calls,
                'output_pins': {},
                'source_pins': {
                    'input': {'sha256': api.sha(canvas_raw)},
                    'mask': {'sha256': api.sha(proposed_png_bytes)},
                },
            },
        }
        (comp_out / 'receipt.json').write_bytes(api.canonical(comp_result['metadata']))

        with patch.object(api, 'run_worker', return_value=(comp_out, comp_result, {'seconds': .02, 'active_owned_processes': 0})):
            generated = api.generate(ident, proposed_png_bytes, reviewed['review_id'], True, True, proposal['proposal_id'])
        return proposal, generated

    def test_propose_review_generate_end_to_end_returns_output_diagnostics(self):
        base_rgb, _ = self._load_prepared_test_image('face 1 - default.jpg')
        masked_rgb = base_rgb.copy()
        masked_rgb[140:205, 86:170] = (135, 195, 225)
        proposal, generated = self._run_propose_review_generate(masked_rgb, expected_neural_calls=1)

        self.assertIn('auto_covering', proposal['metadata'])
        self.assertIn('mask_or_scarf', proposal['metadata']['auto_covering']['families_triggered'])
        self.assertFalse(proposal['metadata']['proposal_empty'])
        self.assertGreater(proposal['metadata']['proposed_mask_pixels'], 0)
        self.assertEqual(generated['neural_calls'], 1)
        self.assertEqual(generated['metadata']['source'], 'automatic_reviewed')
        diag = generated['metadata']['output_diagnostics']
        self.assertEqual(set(diag), {'before', 'after'})
        for phase in ('before', 'after'):
            self.assertEqual(set(diag[phase]), {'blur_variance', 'noise_sigma_255'})
            self.assertTrue(math.isfinite(diag[phase]['blur_variance']))
            self.assertTrue(math.isfinite(diag[phase]['noise_sigma_255']))

    def test_propose_review_generate_empty_mask_runs_zero_neural_calls(self):
        clean_rgb, _ = self._load_prepared_test_image('face 1 - default.jpg')
        proposal, generated = self._run_propose_review_generate(clean_rgb, expected_neural_calls=0)

        self.assertTrue(proposal['metadata']['proposal_empty'])
        self.assertEqual(proposal['metadata']['proposed_mask_pixels'], 0)
        self.assertEqual(proposal['metadata']['auto_covering']['auto_detected_pixels'], 0)
        self.assertEqual(generated['neural_calls'], 0)
        self.assertEqual(generated['metadata']['source'], 'automatic_reviewed')
        diag = generated['metadata']['output_diagnostics']
        self.assertEqual(set(diag), {'before', 'after'})
        self.assertEqual(diag['before'], diag['after'])

    def test_propose_precheck_and_evidence_zip_includes_auto_proposed_mask(self):
        base_rgb, _ = self._load_prepared_test_image('face 1 - default.jpg')
        masked_rgb = base_rgb.copy()
        masked_rgb[140:205, 86:170] = (135, 195, 225)
        proposal, generated = self._run_propose_review_generate(masked_rgb, expected_neural_calls=1)
        self.assertIsInstance(generated['metadata'].get('auto_covering'), dict)
        self.assertGreater(generated['metadata']['auto_covering']['auto_detected_pixels'], 0)
        bundle_prefix = 'data:application/zip;base64,'
        self.assertTrue(generated['bundle'].startswith(bundle_prefix))
        bundle_bytes = base64.b64decode(generated['bundle'][len(bundle_prefix):])
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
            names = set(archive.namelist())
            self.assertIn('proposal/auto-proposed-mask.png', names)
            self.assertIn('proposal/detector-proposed-mask.png', names)
            manifest = json.loads(archive.read('manifest.json'))['sha256']
            self.assertEqual(api.sha(archive.read('proposal/auto-proposed-mask.png')),
                             manifest['proposal/auto-proposed-mask.png'])
            combined = json.loads(archive.read('combined-workflow.json'))
            self.assertEqual(combined['auto_covering'], proposal['metadata']['auto_covering'])

        # Tiny observed sliver must refuse inside propose() before spawning detector worker.
        sliver = np.full((256, 20, 3), 120, dtype=np.uint8)
        sliver_raw = api.png(sliver)
        orig256, _, _, geom = api.restoration.prepare_crop(sliver)
        orig256_raw = api.png(orig256)
        meta = {
            'model': 'v3',
            'selection': 'preconditioned_v34_codeformer',
            'uploaded_bytes_sha256': api.sha(sliver_raw),
            'result_file_sha256': api.sha(orig256_raw),
            'restoration_applied': False,
            'geometry': geom,
        }
        resp = {
            'metadata': meta,
            'original_input': api.restoration.url(orig256_raw),
            'prepared_input': api.restoration.url(orig256_raw),
            'result': api.restoration.url(orig256_raw),
        }
        api.admit_restoration(api.canonical(resp))
        sliver_sid = api.prepare(sliver_raw, orig256_raw, api.canonical(meta))['session_id']
        with patch.object(api, 'run_worker') as worker:
            with self.assertRaisesRegex(ValueError, 'Too little visible face'):
                api.propose(sliver_sid)
            worker.assert_not_called()

    def test_real_frozen_detector_zero_fp_on_review_zips_and_internet_controls(self):
        import torch
        from implementation.automatic_covering import detect_face_covering
        from implementation.covering_detector import load_frozen_detector, proposal_from_probability

        root = self._test_images_dir()
        review_zips = sorted((root / 'review files').glob('*.zip'))
        self.assertGreaterEqual(len(review_zips), 1)
        detector = load_frozen_detector()

        def run_real(rgb, obs):
            tensor = torch.from_numpy(rgb.astype(np.float32) / np.float32(255.0)).permute(2, 0, 1).unsqueeze(0)
            with torch.inference_mode():
                prob = detector(tensor).sigmoid()[0, 0].cpu().numpy().copy()
            _, prop, _ = proposal_from_probability(prob, obs)
            return detect_face_covering(rgb, prob, prop, obs)

        full_obs = np.ones((256, 256), dtype=bool)
        for zp in review_zips:
            with zipfile.ZipFile(zp) as archive:
                if 'result.png' not in archive.namelist():
                    continue
                with self.subTest(archive=zp.name):
                    with Image.open(io.BytesIO(archive.read('result.png'))) as img:
                        rgb256 = np.array(img.convert('RGB'))
                    mask, meta = run_real(rgb256, full_obs)
                    self.assertTrue(meta['onnx_used'])
                    self.assertEqual(int((mask > 0).sum()), 0)
                    self.assertEqual(meta['auto_detected_pixels'], 0)

        covering_dir = root / 'internet_samples' / 'covering'
        if covering_dir.is_dir():
            controls = sorted(covering_dir.glob('control_*.jpg'))
            self.assertGreaterEqual(len(controls), 4)
            control_crops = {
                'control_01_thick_white_beard.jpg': (0, 31, 960, 1388),
                'control_02_darkskin_beard.jpg': (0, 0, 441, 712),
                'control_03_heavy_stage_makeup.jpg': (0, 212, 845, 1091),
                'control_04_makeup_dark_shadow.jpg': (198, 140, 333, 388),
            }
            for path in controls:
                with self.subTest(control=path.name):
                    with Image.open(path) as img:
                        native = np.array(img.convert('RGB'))
                    crop = control_crops.get(path.name)
                    if crop is not None:
                        x, y, w, h = crop
                        native = native[y:y + h, x:x + w]
                    rgb256, obs, _, _ = api.restoration.prepare_crop(native)
                    mask, meta = run_real(rgb256, obs)
                    self.assertTrue(meta['onnx_used'])
                    self.assertEqual(int((mask > 0).sum()), 0)
                    self.assertEqual(meta['auto_detected_pixels'], 0)

            bearded_eyewear = {
                'eyewear_01_dark_opaque_beard.jpg': (220, 220, 412, 477),
                'eyewear_05_clear_prescription.jpg': (0, 0, 744, 1154),
                'eyewear_06_round_tinted.jpg': (0, 0, 960, 880),
            }
            for fn, (x, y, w, h) in bearded_eyewear.items():
                path = covering_dir / fn
                if not path.is_file():
                    continue
                with self.subTest(bearded_eyewear=fn):
                    with Image.open(path) as img:
                        native = np.array(img.convert('RGB'))[y:y + h, x:x + w]
                    rgb256, obs, _, _ = api.restoration.prepare_crop(native)
                    mask, meta = run_real(rgb256, obs)
                    self.assertIn('eyewear_or_glare', meta['families_triggered'])
                    self.assertGreater(int((mask[:145, :] > 0).sum()), 800)
                    self.assertEqual(int((mask[150:, :] > 0).sum()), 0)


if __name__=='__main__': unittest.main()
