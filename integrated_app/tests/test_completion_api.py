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

    def test_propose_skips_cold_subprocess_when_onnx_ready_and_falls_back_when_unavailable(self):
        from implementation import automatic_covering as ac
        base_rgb, observed = self._load_prepared_test_image('face 1 - default.jpg')
        masked_rgb = base_rgb.copy()
        masked_rgb[140:205, 86:170] = (135, 195, 225)
        canvas_raw = api.png(masked_rgb)
        _, _, _, geometry = api.restoration.prepare_crop(masked_rgb)
        meta = {
            'model': 'v3',
            'selection': 'preconditioned_v34_codeformer',
            'uploaded_bytes_sha256': api.sha(canvas_raw),
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
        sid = api.prepare(canvas_raw, canvas_raw, api.canonical(meta))['session_id']

        # Fast in-process ONNX path must NOT invoke run_worker('detector').
        with patch.object(api, 'run_worker', side_effect=AssertionError('subprocess must be skipped')) as worker:
            proposal = api.propose(sid)
            worker.assert_not_called()
        self.assertTrue(proposal['metadata'].get('subprocess_skipped'))
        self.assertEqual(proposal['metadata'].get('detector_backend'), 'onnx_in_process_fast_path')
        self.assertGreater(proposal['metadata']['proposed_mask_pixels'], 1500)

        # When ONNX sessions are unavailable, propose() falls back cleanly to run_worker('detector').
        from implementation.covering_detector import proposal_from_probability
        det_out = Path(self.temp.name) / 'fallback-det-out'
        det_out.mkdir()
        prob = np.zeros((256, 256), np.float32)
        det_raw, det_prop, det_meta = proposal_from_probability(prob, observed)
        np.save(det_out / 'detector-probability.npy', prob, allow_pickle=False)
        (det_out / 'detector-raw-mask.png').write_bytes(api.png(det_raw))
        (det_out / 'detector-proposed-mask.png').write_bytes(api.png(det_prop))
        (det_out / 'receipt.json').write_bytes(api.canonical(det_meta))
        det_res = {
            'probability_path': str(det_out / 'detector-probability.npy'),
            'raw_mask_path': str(det_out / 'detector-raw-mask.png'),
            'proposed_mask_path': str(det_out / 'detector-proposed-mask.png'),
            'metadata': det_meta,
        }
        with patch.object(ac, '_load_onnx_sessions', return_value=None), \
             patch.object(api, 'run_worker', return_value=(det_out, det_res, {'seconds': .01, 'active_owned_processes': 0})) as worker:
            fb_prop = api.propose(sid)
            worker.assert_called_once()
        self.assertFalse(fb_prop['metadata']['auto_covering']['onnx_used'])

    def test_vnext_empirical_covering_fixes_and_quality_gate(self):
        import cv2
        from implementation import automatic_covering as ac

        root = self._test_images_dir()
        covering_dir = root / 'internet_samples' / 'covering'
        if not covering_dir.is_dir():
            self.skipTest('internet_samples/covering corpus not present')

        z_p = np.zeros((256, 256), np.float32)
        z_b = np.zeros((256, 256), np.uint8)
        crops = {
            'compound_05_cctv_mask_hood.jpg': (399, 163, 261, 292),
            'compound_04_cctv_black_mask.jpg': (163, 245, 438, 488),
            'mask_01_surgical_bw.jpg': (0, 0, 960, 960),
            'eyewear_06_round_tinted.jpg': (0, 0, 960, 880),
            'scarf_02_gray_balaclava.jpg': (0, 29, 960, 1171),
            'mask_02_white_n95_glasses.jpg': (143, 203, 704, 821),
        }

        def run_img(fn):
            with Image.open(covering_dir / fn) as im:
                nat = np.array(im.convert('RGB'))
            x, y, w, h = crops[fn]
            rgb, obs, _, _ = api.restoration.prepare_crop(nat[y:y + h, x:x + w])
            mask, meta = ac.detect_face_covering(rgb, z_p, z_b, obs)
            return rgb, obs, mask, meta

        # 1. compound_05: no longer 0% FN.
        _, _, m5, meta5 = run_img('compound_05_cctv_mask_hood.jpg')
        self.assertGreater(int((m5[116:235, :] > 0).sum()), 2000)
        self.assertIn('mask_or_scarf', meta5['families_triggered'])

        # 2. compound_04: detects lower black chin mask (clamped to chin_y+10, zero hoodie/forehead spill).
        _, _, m4, _ = run_img('compound_04_cctv_black_mask.jpg')
        m4_lower = int((m4[165:240, :] > 0).sum())
        self.assertGreaterEqual(m4_lower, 1200)
        self.assertLessEqual(m4_lower, 1700)
        self.assertEqual(int((m4[:120, :] > 0).sum()), 0)

        # 3. mask_01: visible open eye must NOT trigger eyewear_or_glare.
        _, _, m1, meta1 = run_img('mask_01_surgical_bw.jpg')
        self.assertNotIn('eyewear_or_glare', meta1['families_triggered'])
        self.assertGreater(int((m1[120:, :] > 0).sum()), 10000)

        # 4. eyewear_06: coherent 1-CC eyewear with zero forehead fin above y=55.
        _, _, m6, _ = run_img('eyewear_06_round_tinted.jpg')
        self.assertEqual(int((m6[:55, :] > 0).sum()), 0)
        n_cc6 = cv2.connectedComponentsWithStats((m6 > 0).astype(np.uint8), 8)[0] - 1
        self.assertEqual(n_cc6, 1)

        # 5. scarf_02: coherent single-component balaclava (eliminated 6-CC Swiss-cheese holes).
        _, _, ms2, _ = run_img('scarf_02_gray_balaclava.jpg')
        n_ccs2 = cv2.connectedComponentsWithStats((ms2 > 0).astype(np.uint8), 8)[0] - 1
        self.assertLessEqual(n_ccs2, 2)
        self.assertGreater(int((ms2 > 0).sum()), 8000)

        # 6. mask_02: ear-strap hair/bg wings slashed (< 5% hair/bg spill vs 18.7% in v0.3.3).
        rgb2, _, m2, _ = run_img('mask_02_white_n95_glasses.jpg')
        bprob2, _ = ac._run_onnx_models(rgb2)
        seg2 = bprob2.argmax(axis=0)
        hb_frac = float(((m2 > 0) & np.isin(seg2, (0, 17))).sum()) / float(max(1, int((m2 > 0).sum())))
        self.assertLess(hb_frac, 0.05)

        # 7. Zero-weight 5-factor post-completion quality gate & /generate integration.
        base_rgb, base_obs = self._load_prepared_test_image('face 1 - default.jpg')
        masked_rgb = base_rgb.copy()
        masked_rgb[140:205, 86:170] = (135, 195, 225)
        removal = np.zeros((256, 256), dtype=np.uint8)
        removal[140:205, 86:170] = 255
        q_good = ac.evaluate_completion_quality(masked_rgb, base_rgb, removal, base_obs)
        self.assertIsNotNone(q_good)
        self.assertGreaterEqual(q_good['score_0_100'], 70)
        self.assertEqual(set(q_good['factors']), {'occluder_clear', 'anatomy', 'symmetry', 'seam', 'texture'})
        self.assertIn(q_good['grade'], ('high', 'moderate', 'low'))
        self.assertIsNone(ac.evaluate_completion_quality(masked_rgb, base_rgb, np.zeros_like(removal), base_obs))
        with patch.object(ac, '_run_onnx_models', return_value=None):
            self.assertIsNone(ac.evaluate_completion_quality(masked_rgb, base_rgb, removal, base_obs))

        _, generated = self._run_propose_review_generate(masked_rgb, expected_neural_calls=1)
        cq = generated['metadata'].get('completion_quality')
        self.assertIsInstance(cq, dict)
        self.assertIn('score_0_100', cq)
        self.assertEqual(set(cq['factors']), {'occluder_clear', 'anatomy', 'symmetry', 'seam', 'texture'})

        # 8. Non-uint8 dtype rejected by evaluate_completion_quality, and cv2.error degrades cleanly in generate().
        self.assertIsNone(
            ac.evaluate_completion_quality(
                masked_rgb.astype(np.float32),
                base_rgb,
                removal,
                base_obs,
            )
        )
        self.assertIsNone(
            ac.evaluate_completion_quality(
                masked_rgb,
                base_rgb.astype(np.float32),
                removal,
                base_obs,
            )
        )
        with patch.object(ac, 'evaluate_completion_quality', side_effect=cv2.error('simulated opencv error')):
            _, gen_fallback = self._run_propose_review_generate(masked_rgb, expected_neural_calls=1)
        self.assertIsNone(gen_fallback['metadata'].get('completion_quality'))
        self.assertEqual(gen_fallback['neural_calls'], 1)


class Phase2CoveringOnnxTests(unittest.TestCase):
    def _test_images_dir(self):
        here = Path(__file__).resolve()
        for parent in (here.parents[1], here.parents[2], *here.parents[3:]):
            candidate = parent / 'test_images'
            if candidate.is_dir():
                return candidate
        raise unittest.SkipTest('test_images fixture directory is not present in this checkout')

    def test_phase2_qdq_ofiq_and_2d106det_active_nan_sha_and_ort_fallbacks(self):
        import builtins
        import stat
        import types
        from onnxruntime.capi import onnxruntime_pybind11_state as ort_capi
        from implementation import automatic_covering as ac

        root = self._test_images_dir()
        with Image.open(root / 'face 1 - default.jpg') as im:
            base_rgb, base_obs, _, _ = api.restoration.prepare_crop(np.array(im.convert('RGB')))

        saved_cache = dict(ac._ONNX_CACHE)
        try:
            # 1. Verify active staged Phase 2 models (BiSeNet-QDQ + OFIQ-QDQ + 2d106det) load and execute.
            ac._ONNX_CACHE.clear()
            model_dir = ac._resolve_occlusion_model_dir()
            self.assertIsNotNone(model_dir)
            self.assertEqual(ac._select_parser_file(model_dir).name, ac._BISENET_QDQ_FILENAME)
            occ_choice = ac._select_occlusion_file(model_dir)
            self.assertIsNotNone(occ_choice)
            self.assertEqual(occ_choice[0].name, ac._OFIQ_QDQ_FILENAME)
            self.assertEqual(occ_choice[1], 'ofiq')
            lms_path = ac._resolve_landmarks106_path()
            self.assertIsNotNone(lms_path)
            self.assertEqual(lms_path.name, ac._LMS106_FILENAME)

            sessions = ac._load_onnx_sessions()
            self.assertIsNotNone(sessions)
            self.assertEqual(len(sessions), 7)
            self.assertEqual(sessions[5], 'ofiq')
            self.assertEqual(sessions[6], 224)

            lms_entry = ac._load_landmarks106_session()
            self.assertIsNotNone(lms_entry)
            self.assertEqual(len(lms_entry), 3)

            bprob, px = ac._run_onnx_models(base_rgb)
            self.assertEqual(bprob.shape, (19, 256, 256))
            self.assertEqual(px.shape, (256, 256))
            self.assertTrue(np.isfinite(bprob).all())
            self.assertTrue(np.isfinite(px).all())
            self.assertGreaterEqual(float(px.min()), 0.0)
            self.assertLessEqual(float(px.max()), 1.0)

            pts106 = ac._run_landmarks_106(base_rgb)
            self.assertIsNotNone(pts106)
            self.assertEqual(pts106.shape, (106, 2))
            self.assertEqual(pts106.dtype, np.float32)
            self.assertTrue(np.isfinite(pts106).all())
            pts5 = ac.landmarks_5pt_from_106(pts106)
            self.assertIsNotNone(pts5)
            self.assertEqual(pts5.shape, (5, 2))
            self.assertEqual(pts5.dtype, np.float32)
            hull = ac.face_hull_from_106(pts106, dilate_px=6, forehead_lift=0.25)
            self.assertEqual(hull.shape, (256, 256))
            self.assertEqual(hull.dtype, np.bool_)
            self.assertGreater(int(hull.sum()), 8000)

            # Evaluate completion quality under BiSeNet-QDQ + OFIQ-QDQ.
            removal = np.zeros((256, 256), dtype=np.uint8)
            removal[140:205, 86:170] = 255
            masked_rgb = base_rgb.copy()
            masked_rgb[removal > 0] = (135, 195, 225)
            q = ac.evaluate_completion_quality(masked_rgb, base_rgb, removal, base_obs)
            self.assertIsNotNone(q)
            self.assertGreaterEqual(q['score_0_100'], 70)

            # 2. NaN output sanitization for 7-tuple OFIQ session and 2d106det session.
            class FakeSess:
                def __init__(self, fn):
                    self._fn = fn
                def run(self, *_args):
                    return self._fn()

            ac._ONNX_CACHE.clear()
            ac._ONNX_CACHE['sessions'] = (
                FakeSess(lambda: [np.full((1, 19, 512, 512), np.nan, np.float32)]),
                'input',
                'output',
                FakeSess(lambda: [np.full((1, 1, 224, 224), np.nan, np.float32)]),
                'input',
                'ofiq',
                224,
            )
            nan_out = ac._run_onnx_models(base_rgb)
            self.assertIsNotNone(nan_out)
            self.assertTrue(np.isfinite(nan_out[0]).all())
            self.assertTrue(np.isfinite(nan_out[1]).all())
            self.assertGreaterEqual(float(nan_out[1].min()), 0.0)
            self.assertLessEqual(float(nan_out[1].max()), 1.0)

            ac._ONNX_CACHE['lms106'] = (
                FakeSess(lambda: [np.full((1, 212), np.nan, np.float32)]),
                'data',
                'fc1',
            )
            self.assertIsNone(ac._run_landmarks_106(base_rgb))

            # 3. Real onnxruntime_pybind11_state (NoSuchFile & Fail) fallback for BOTH occlusion and 2d106det.
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                ac._ONNX_CACHE.clear()
                with patch.object(ac, '_resolve_occlusion_model_dir', return_value=tmp_dir), \
                     patch('onnxruntime.InferenceSession', side_effect=ort_capi.NoSuchFile('missing phase2 occ')):
                    self.assertIsNone(ac._load_onnx_sessions())

                ac._ONNX_CACHE.clear()
                with patch.object(ac, '_resolve_landmarks106_path', return_value=tmp_dir / ac._LMS106_FILENAME), \
                     patch('onnxruntime.InferenceSession', side_effect=ort_capi.NoSuchFile('missing 2d106det')):
                    self.assertIsNone(ac._load_landmarks106_session())

                def raise_ort_fail():
                    raise ort_capi.Fail('simulated pybind11 Fail')

                ac._ONNX_CACHE.clear()
                ac._ONNX_CACHE['sessions'] = (
                    FakeSess(raise_ort_fail), 'input', 'output', FakeSess(raise_ort_fail), 'input', 'ofiq', 224
                )
                self.assertIsNone(ac._run_onnx_models(base_rgb))

                ac._ONNX_CACHE['lms106'] = (FakeSess(raise_ort_fail), 'data', 'fc1')
                self.assertIsNone(ac._run_landmarks_106(base_rgb))

                import cv2 as _cv2
                self.assertIn(_cv2.error, ac._onnx_runtime_errors())
                with patch.object(_cv2, 'resize', side_effect=_cv2.error('resize fail')):
                    self.assertIsNone(ac._run_landmarks_106(base_rgb))

                # 4. SHA-256 mismatch and ordinary-file (symlink/reparse) refusal for 2d106det and QDQ models.
                tampered = tmp_dir / ac._LMS106_FILENAME
                tampered.write_bytes(b'corrupt-2d106det')
                self.assertFalse(ac._verify_onnx_file(tampered, ac._LMS106_BYTES, ac._LMS106_SHA256))
                self.assertFalse(ac._verify_onnx_file(tampered, len(b'corrupt-2d106det'), ac._LMS106_SHA256))
                fake_link = types.SimpleNamespace(st_mode=stat.S_IFLNK | 0o644, st_file_attributes=0)
                with patch.object(Path, 'lstat', return_value=fake_link):
                    self.assertIsNone(ac._resolve_landmarks106_path())
                    self.assertIsNone(ac._resolve_occlusion_model_dir())

                real_import = builtins.__import__
                def fail_ort(name, *args, **kwargs):
                    if name == 'onnxruntime':
                        raise ImportError('ort absent')
                    return real_import(name, *args, **kwargs)
                ac._ONNX_CACHE.clear()
                with patch.object(ac, '_resolve_landmarks106_path', return_value=tampered), \
                     patch.object(builtins, '__import__', side_effect=fail_ort):
                    self.assertIsNone(ac._load_landmarks106_session())
        finally:
            ac._ONNX_CACHE.clear()
            ac._ONNX_CACHE.update(saved_cache)

    def test_phase2_49_image_covering_recall_zero_fp_and_visual_gates(self):
        import cv2
        from implementation import app as app_mod
        from implementation import automatic_covering as ac
        from implementation import capture_worker, runtime

        root = self._test_images_dir()
        cov_root = root / 'internet_samples' / 'covering'
        review_dir = root / 'review files'
        if not cov_root.is_dir() or not review_dir.is_dir():
            self.skipTest('49-image covering + review corpus not present')

        zp = np.zeros((256, 256), dtype=np.float32)
        zb = np.zeros((256, 256), dtype=np.uint8)

        cov_files = sorted(
            p for p in cov_root.iterdir()
            if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')
        )
        pos_files = [p for p in cov_files if not p.name.startswith('control_')]
        ctrl_files = [p for p in cov_files if p.name.startswith('control_')]
        self.assertEqual(len(pos_files), 25)
        self.assertEqual(len(ctrl_files), 4)

        def prepare_covering(path: Path):
            with Image.open(path) as im:
                rgb_full = np.asarray(im.convert('RGB'))
            recs = capture_worker.scan_face_records(cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR))
            if recs:
                rec_idx = app_mod._recommend_face_index(rgb_full, recs, rgb_full.shape[1], rgb_full.shape[0])
                cx, cy, cw, ch = recs[rec_idx]['crop']
                crop_rgb = rgb_full[cy:cy + ch, cx:cx + cw]
            else:
                crop_rgb = rgb_full
            rgb_256, obs_256, _, _ = runtime.prepare_crop(crop_rgb)
            return rgb_256, obs_256

        # 25/25 positive recall + targeted visual quality gates
        recall = 0
        for p in pos_files:
            rgb256, obs256 = prepare_covering(p)
            mask, meta = ac.detect_face_covering(rgb256, zp, zb, obs256)
            self.assertTrue(meta['onnx_used'])
            self.assertFalse(mask[~obs256].any())
            px_cnt = int((mask > 0).sum())
            if px_cnt > 0:
                recall += 1
            if p.name == 'compound_02_hand_chin.jpg':
                self.assertGreaterEqual(px_cnt, 2500)
            elif p.name == 'compound_03_hand_over_mouth.jpg':
                bprob, _ = ac._run_onnx_models(rgb256)
                seg = bprob.argmax(axis=0)
                hb_pct = float(((mask > 0) & np.isin(seg, (0, 17))).sum()) / float(max(1, px_cnt)) * 100.0
                self.assertLess(hb_pct, 22.0)
            elif p.name == 'compound_04_cctv_black_mask.jpg':
                self.assertGreaterEqual(px_cnt, 1200)
                self.assertLessEqual(px_cnt, 1700)
                self.assertEqual(int((mask[:115, :] > 0).sum()), 0)
            elif p.name == 'compound_05_cctv_mask_hood.jpg':
                self.assertGreaterEqual(px_cnt, 3000)
            elif p.name == 'mask_09_mask_and_shield.jpg':
                self.assertGreaterEqual(px_cnt, 10000)
                unmasked_ctr = (mask == 0) & obs256 & (ac._YY >= 140) & (ac._YY <= 190) & (ac._XX >= 90) & (ac._XX <= 165)
                self.assertLess(int(unmasked_ctr.sum()), 50)
            elif p.name == 'eyewear_05_clear_prescription.jpg':
                self.assertIn('eyewear_or_glare', meta['families_triggered'])
                bprob, _ = ac._run_onnx_models(rgb256)
                seg = bprob.argmax(axis=0)
                eyes_m = np.isin(seg, (4, 5)) & obs256
                if eyes_m.any():
                    eye_cov = float(((mask > 0) & eyes_m).sum()) / float(int(eyes_m.sum()))
                    self.assertLess(eye_cov, 0.15)
        self.assertEqual(recall, 25)

        # 0/4 FP on beard/makeup controls
        ctrl_fp = 0
        for p in ctrl_files:
            rgb256, obs256 = prepare_covering(p)
            mask, meta = ac.detect_face_covering(rgb256, zp, zb, obs256)
            if int((mask > 0).sum()) > 0:
                ctrl_fp += 1
        self.assertEqual(ctrl_fp, 0)

        # 0/20 FP across ALL review files/*.zip
        zips = sorted(review_dir.glob('*.zip'))
        self.assertEqual(len(zips), 20)
        zip_fp = 0
        for zp_path in zips:
            with zipfile.ZipFile(zp_path) as zf:
                with Image.open(io.BytesIO(zf.read('result.png'))) as im:
                    rgb256 = np.asarray(im.convert('RGB'))
                with Image.open(io.BytesIO(zf.read('original-native.png'))) as im:
                    orig_native = np.asarray(im.convert('RGB'))
            _, obs256, _, _ = runtime.prepare_crop(orig_native)
            mask, _ = ac.detect_face_covering(rgb256, zp, zb, obs256)
            if int((mask > 0).sum()) > 0:
                zip_fp += 1
        self.assertEqual(zip_fp, 0)


class Phase3MaskRefinerAndAdaptiveRerankApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store_patch = patch.object(api, 'STORE', Path(self.temp.name))
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)
        api.ADMISSIONS.clear()
        api.SESSIONS.clear()
        self.saved_state = dict(api.STATE)
        api.STATE['uncertain'] = False
        api.STATE['active_worker'] = False
        self.addCleanup(lambda: api.STATE.update(self.saved_state))

    def _test_images_dir(self):
        here = Path(__file__).resolve()
        for parent in (here.parents[1], here.parents[2], *here.parents[3:]):
            candidate = parent / 'test_images'
            if candidate.is_dir():
                return candidate
        raise unittest.SkipTest('test_images fixture directory is not present in this checkout')

    def test_phase3_49_image_mask_refiner_metrics_and_gates(self):
        import statistics
        import cv2
        from implementation import app as app_mod
        from implementation import automatic_covering as ac
        from implementation import capture_worker, runtime

        root = self._test_images_dir()
        cov_root = root / 'internet_samples' / 'covering'
        review_dir = root / 'review files'
        if not cov_root.is_dir() or not review_dir.is_dir():
            self.skipTest('49-image covering + review corpus not present')

        zp = np.zeros((256, 256), dtype=np.float32)
        zb = np.zeros((256, 256), dtype=np.uint8)
        cov_files = sorted(
            p for p in cov_root.iterdir()
            if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')
        )
        pos_files = [p for p in cov_files if not p.name.startswith('control_')]
        ctrl_files = [p for p in cov_files if p.name.startswith('control_')]
        self.assertEqual(len(pos_files), 25)
        self.assertEqual(len(ctrl_files), 4)

        def prepare_covering(path: Path):
            with Image.open(path) as im:
                rgb_full = np.asarray(im.convert('RGB'))
            recs = capture_worker.scan_face_records(cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR))
            if recs:
                rec_idx = app_mod._recommend_face_index(rgb_full, recs, rgb_full.shape[1], rgb_full.shape[0])
                cx, cy, cw, ch = recs[rec_idx]['crop']
                crop_rgb = rgb_full[cy:cy + ch, cx:cx + cw]
            else:
                crop_rgb = rgb_full
            rgb_256, obs_256, _, _ = runtime.prepare_crop(crop_rgb)
            return rgb_256, obs_256

        recall = 0
        hb_24: list[float] = []
        cc_all: list[int] = []
        for p in pos_files:
            rgb256, obs256 = prepare_covering(p)
            mask, meta = ac.detect_face_covering(rgb256, zp, zb, obs256)
            self.assertTrue(meta['onnx_used'])
            self.assertIn('p3_refiner_actions', meta)
            self.assertFalse(mask[~obs256].any())
            m_bool = mask > 0
            px_cnt = int(m_bool.sum())
            if px_cnt > 0:
                recall += 1
            n_cc = cv2.connectedComponentsWithStats(m_bool.astype(np.uint8), connectivity=8)[0] - 1
            cc_all.append(n_cc)
            bprob, _ = ac._run_onnx_models(rgb256)
            seg = bprob.argmax(axis=0)
            hb_pct = 100.0 * float(int((m_bool & np.isin(seg, (0, 17))).sum())) / float(max(1, px_cnt))
            if 'compound_05' not in p.name:
                hb_24.append(hb_pct)

            if p.name == 'compound_02_hand_chin.jpg':
                self.assertEqual(hb_pct, 0.0)
                self.assertEqual(n_cc, 1)
                self.assertIn('drop_bg_crescent_cc', meta['p3_refiner_actions'])
            elif p.name == 'compound_04_cctv_black_mask.jpg':
                self.assertGreaterEqual(px_cnt, 800)
                self.assertLessEqual(px_cnt, 1700)
                self.assertEqual(n_cc, 1)
                self.assertEqual(hb_pct, 0.0)
                self.assertTrue(any(a.startswith('clamp_chin_hoodie(') for a in meta['p3_refiner_actions']))
            elif p.name == 'eyewear_05_clear_prescription.jpg':
                self.assertEqual(n_cc, 2)
                self.assertIn('eyewear_or_glare', meta['families_triggered'])
            elif p.name in ('mask_09_mask_and_shield.jpg', 'scarf_01_gray_gaiter_veil.jpg'):
                self.assertEqual(n_cc, 1)

        self.assertEqual(recall, 25)
        self.assertLess(statistics.mean(hb_24), 1.0)
        self.assertLessEqual(statistics.mean(cc_all), 1.08)

        ctrl_fp = 0
        for p in ctrl_files:
            rgb256, obs256 = prepare_covering(p)
            mask, _ = ac.detect_face_covering(rgb256, zp, zb, obs256)
            if int((mask > 0).sum()) > 0:
                ctrl_fp += 1
        self.assertEqual(ctrl_fp, 0)

        zips = sorted(review_dir.glob('*.zip'))
        self.assertEqual(len(zips), 20)
        zip_fp = 0
        for zp_path in zips:
            with zipfile.ZipFile(zp_path) as zf:
                with Image.open(io.BytesIO(zf.read('result.png'))) as im:
                    rgb256 = np.asarray(im.convert('RGB'))
                with Image.open(io.BytesIO(zf.read('original-native.png'))) as im:
                    orig_native = np.asarray(im.convert('RGB'))
            _, obs256, _, _ = runtime.prepare_crop(orig_native)
            mask, _ = ac.detect_face_covering(rgb256, zp, zb, obs256)
            if int((mask > 0).sum()) > 0:
                zip_fp += 1
        self.assertEqual(zip_fp, 0)

    def test_run_worker_and_generate_multipass_receipt_propagation_and_validation(self):
        from implementation.completion_runtime import model_contract

        yy, xx = np.mgrid[:256, :256]
        rgb = np.stack((xx, yy, (xx + yy) // 2), axis=2).astype(np.uint8)
        raw_png = api.png(rgb)
        _, _, _, geometry = api.restoration.prepare_crop(rgb)
        meta = {
            'model': 'v3',
            'selection': 'preconditioned_v34_codeformer',
            'uploaded_bytes_sha256': api.sha(raw_png),
            'result_file_sha256': api.sha(raw_png),
            'restoration_applied': False,
            'geometry': geometry,
        }
        resp = {
            'metadata': meta,
            'original_input': api.restoration.url(raw_png),
            'prepared_input': api.restoration.url(raw_png),
            'result': api.restoration.url(raw_png),
        }
        api.admit_restoration(api.canonical(resp))
        sid = api.prepare(raw_png, raw_png, api.canonical(meta))['session_id']
        record = api.session(sid)

        non_empty_mask_arr = np.zeros((256, 256), dtype=np.uint8)
        non_empty_mask_arr[140:190, 100:156] = 255
        non_empty_mask = api.png(non_empty_mask_arr)
        empty_mask = api.png(np.zeros((256, 256), dtype=np.uint8))

        def run_fake_worker(mask_bytes, status_payload, *, result_neural_calls='match'):
            captured_req = {}

            class FakeOwnedWorker:
                def __init__(self, cmd, **_kwargs):
                    req_path = Path(cmd[cmd.index('--request') + 1])
                    res_path = Path(cmd[cmd.index('--result') + 1])
                    captured_req.update(json.loads(req_path.read_text(encoding='utf-8')))
                    out_dir = Path(captured_req['output_dir'])
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / 'status.json').write_bytes(api.canonical(status_payload))
                    res_calls = (
                        status_payload.get('neural_returns', 0)
                        if result_neural_calls == 'match'
                        else result_neural_calls
                    )
                    res_path.write_bytes(api.canonical({'complete': True, 'metadata': {'neural_calls': res_calls}}))

                def __enter__(self): return self
                def __exit__(self, *_a): return False
                def wait_tree(self, timeout=None): return 0
                def poll(self): return 0
                def active_processes(self): return 0
                def memory_usage(self): return {'peak_job_private_bytes': 1024}
                def kill(self): pass

            import types
            fake_mem = types.SimpleNamespace(available=8 * 1024**3)
            with patch.object(api, 'OwnedWorker', FakeOwnedWorker), \
                 patch.object(api.psutil, 'virtual_memory', return_value=fake_mem):
                return captured_req, api.run_worker(record, 'completion', mask_bytes)

        # Valid 2-pass and 3-pass receipts succeed and increment STATE['completion_calls'] accurately.
        before_calls = api.STATE['completion_calls']
        req2, _ = run_fake_worker(non_empty_mask, {'phase': 'complete', 'neural_attempts': 2, 'neural_returns': 2})
        self.assertEqual(req2['max_passes'], 3)
        self.assertEqual(api.STATE['completion_calls'], before_calls + 2)

        req3, _ = run_fake_worker(non_empty_mask, {'phase': 'complete', 'neural_attempts': 3, 'neural_returns': 3})
        self.assertEqual(req3['max_passes'], 3)
        self.assertEqual(api.STATE['completion_calls'], before_calls + 5)

        # Invalid receipts must be refused (mismatched attempts!=returns, bool, negative, >3, 0 on non-empty, >0 on empty, desync).
        for mask_b, bad_status, res_calls in (
            (non_empty_mask, {'phase': 'complete', 'neural_attempts': 2, 'neural_returns': 1}, 'match'),
            (non_empty_mask, {'phase': 'complete', 'neural_attempts': 4, 'neural_returns': 4}, 'match'),
            (non_empty_mask, {'phase': 'complete', 'neural_attempts': 0, 'neural_returns': 0}, 'match'),
            (non_empty_mask, {'phase': 'complete', 'neural_attempts': -1, 'neural_returns': -1}, 'match'),
            (non_empty_mask, {'phase': 'complete', 'neural_attempts': True, 'neural_returns': True}, 'match'),
            (empty_mask, {'phase': 'complete', 'neural_attempts': False, 'neural_returns': False}, 'match'),
            (empty_mask, {'phase': 'complete', 'neural_attempts': 1, 'neural_returns': 1}, 'match'),
            (non_empty_mask, {'phase': 'complete', 'neural_attempts': 1, 'neural_returns': 1, 'neural_calls': 2}, 1),
            (non_empty_mask, {'phase': 'complete', 'neural_attempts': 1, 'neural_returns': 1}, 999),
            (non_empty_mask, {'phase': 'complete', 'neural_attempts': 1, 'neural_returns': 1}, True),
        ):
            with self.subTest(bad_status=bad_status, res_calls=res_calls):
                with self.assertRaisesRegex(ValueError, 'Worker forward receipt differs'):
                    run_fake_worker(mask_b, bad_status, result_neural_calls=res_calls)

        # generate() propagates multi-pass neural_calls (e.g. 3) into both metadata and top-level response.
        reviewed = api.review(sid, non_empty_mask)
        comp_out = Path(self.temp.name) / 'mp-comp-out'
        comp_out.mkdir()
        comp_rgb = rgb.copy()
        comp_rgb[non_empty_mask_arr > 0] = (170, 140, 125)
        comp_f = comp_rgb.astype(np.float32) / np.float32(255.0)
        comp_replay = np.rint(comp_f * np.float32(255.0)).clip(0, 255).astype(np.uint8)
        comp_replay[non_empty_mask_arr == 0] = rgb[non_empty_mask_arr == 0]
        (comp_out / 'completion.png').write_bytes(api.png(comp_replay))
        np.save(comp_out / 'raw.npy', comp_f, allow_pickle=False)
        np.savez_compressed(
            comp_out / 'stages.npz',
            input_rgb=rgb,
            removal_mask=(non_empty_mask_arr // 255).astype(np.uint8),
            completed_rgb=comp_f,
            output_rgb=comp_replay,
        )
        comp_result = {
            'png_path': str(comp_out / 'completion.png'),
            'raw_path': str(comp_out / 'raw.npy'),
            'stages_path': str(comp_out / 'stages.npz'),
            'metadata': {
                **model_contract(),
                'neural_calls': 3,
                'adaptive_rerank': {'enabled': True, 'passes': 3, 'early_stopped': False, 'chosen': 'mild_sw07_d3'},
                'output_pins': {},
                'source_pins': {
                    'input': {'sha256': api.sha(raw_png)},
                    'mask': {'sha256': api.sha(non_empty_mask)},
                },
            },
        }
        (comp_out / 'receipt.json').write_bytes(api.canonical(comp_result['metadata']))
        with patch.object(api, 'run_worker', return_value=(comp_out, comp_result, {'seconds': .05, 'active_owned_processes': 0})):
            generated = api.generate(sid, non_empty_mask, reviewed['review_id'], True, True)
        self.assertEqual(generated['neural_calls'], 3)
        self.assertEqual(generated['metadata']['neural_calls'], 3)
        self.assertEqual(generated['metadata']['adaptive_rerank']['passes'], 3)

        # generate() strictly refuses tampered/missing/bool/OOB receipt['neural_calls'] on non-empty and empty masks.
        for idx, (mask_b, mask_arr, bad_calls) in enumerate((
            (non_empty_mask, non_empty_mask_arr, True),
            (non_empty_mask, non_empty_mask_arr, -5),
            (non_empty_mask, non_empty_mask_arr, 0),
            (non_empty_mask, non_empty_mask_arr, 4),
            (non_empty_mask, non_empty_mask_arr, None),
            (non_empty_mask, non_empty_mask_arr, '__missing__'),
            (empty_mask, np.zeros((256, 256), dtype=np.uint8), 1),
            (empty_mask, np.zeros((256, 256), dtype=np.uint8), True),
        )):
            with self.subTest(bad_receipt_calls=bad_calls, empty=not bool(mask_arr.any())):
                rev_bad = api.review(sid, mask_b)
                bad_dir = Path(self.temp.name) / f'bad-rc-{idx}'
                bad_dir.mkdir()
                out_arr = rgb.copy()
                if mask_arr.any():
                    out_arr[mask_arr > 0] = (170, 140, 125)
                out_f = out_arr.astype(np.float32) / np.float32(255.0)
                out_rep = np.rint(out_f * np.float32(255.0)).clip(0, 255).astype(np.uint8)
                out_rep[mask_arr == 0] = rgb[mask_arr == 0]
                (bad_dir / 'completion.png').write_bytes(api.png(out_rep))
                np.save(bad_dir / 'raw.npy', out_f, allow_pickle=False)
                np.savez_compressed(
                    bad_dir / 'stages.npz',
                    input_rgb=rgb,
                    removal_mask=(mask_arr // 255).astype(np.uint8),
                    completed_rgb=out_f,
                    output_rgb=out_rep,
                )
                bad_meta = {
                    **model_contract(),
                    'output_pins': {},
                    'source_pins': {
                        'input': {'sha256': api.sha(raw_png)},
                        'mask': {'sha256': api.sha(mask_b)},
                    },
                }
                if bad_calls != '__missing__':
                    bad_meta['neural_calls'] = bad_calls
                bad_res = {
                    'png_path': str(bad_dir / 'completion.png'),
                    'raw_path': str(bad_dir / 'raw.npy'),
                    'stages_path': str(bad_dir / 'stages.npz'),
                    'metadata': bad_meta,
                }
                (bad_dir / 'receipt.json').write_bytes(api.canonical(bad_meta))
                with patch.object(api, 'run_worker', return_value=(bad_dir, bad_res, {'seconds': .02, 'active_owned_processes': 0})):
                    with self.assertRaisesRegex(ValueError, 'Invalid completion neural_calls receipt'):
                        api.generate(sid, mask_b, rev_bad['review_id'], True, True)


if __name__=='__main__': unittest.main()
