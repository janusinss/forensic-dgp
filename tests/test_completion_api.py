"""Stage binding and preservation tests. Fixtures are not quality evidence."""
import asyncio
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

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
        self.metadata = {'model': 'v2', 'selection': 'conditioned_sft_v2_p2_u0224',
                         'uploaded_bytes_sha256': api.sha(self.raw), 'result_file_sha256': api.sha(self.raw),
                         'restoration_applied': False, 'geometry': geometry}
        self.response = {'metadata': self.metadata, 'original_input': api.restoration.url(self.raw),
                         'prepared_input': api.restoration.url(self.raw), 'result': api.restoration.url(self.raw)}

    def prepare(self):
        api.admit_restoration(api.canonical(self.response))
        return api.prepare(self.raw, self.raw, api.canonical(self.metadata))['session_id']

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
        self.assertEqual(response['metadata']['outside_reviewed_support_exact_to'],'restoration-result.png')
        with zipfile.ZipFile(out/'evidence.zip') as archive:
            manifest=json.loads(archive.read('manifest.json'))['sha256']
            for name,digest in manifest.items(): self.assertEqual(api.sha(archive.read(name)),digest)
            self.assertEqual(archive.read('restoration-result.png'),archive.read('completion-result.png'))
        self.assertIsNone(api.SESSIONS[ident]['review'])


if __name__=='__main__': unittest.main()
