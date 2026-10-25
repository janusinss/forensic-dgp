"""Real pixel preparation and HTTP flow with a non-neural restorer stub."""
import base64
import asyncio
import builtins
import concurrent.futures
import io
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ORIGINAL_IMPORT = builtins.__import__
TORCH_PREEXISTING = 'torch' in sys.modules


def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'): raise AssertionError('No Torch/model call in stub flow tests')
    return ORIGINAL_IMPORT(name, *args, **kwargs)


with patch('builtins.__import__', blocked):
    import numpy as np
    from fastapi import FastAPI
    import research_brush_restoration_v1 as M


def encode(array):
    return M.preparation.png(array)


def blob(value): return base64.b64decode(value.split(',', 1)[1], validate=True)


class ASGIClient:
    """Bounded real multipart/ASGI transport without optional httpx packages."""
    def __init__(self, app): self.app = app
    def close(self): pass
    def post(self, path, files, data=None):
        boundary = 'pure-brush-fixture-20973b441dd7'
        parts = []
        for name, value in (data or {}).items():
            parts.append(('--' + boundary + '\r\nContent-Disposition: form-data; name="' + name + '"\r\n\r\n' + str(value) + '\r\n').encode())
        for name, (filename, raw, mime) in files.items():
            parts.append(('--' + boundary + '\r\nContent-Disposition: form-data; name="' + name + '"; filename="' + filename + '"\r\nContent-Type: ' + mime + '\r\n\r\n').encode() + raw + b'\r\n')
        body = b''.join(parts) + ('--' + boundary + '--\r\n').encode()
        async def request():
            messages = []; received = False
            async def receive():
                nonlocal received
                if not received:
                    received = True; return {'type': 'http.request', 'body': body, 'more_body': False}
                await asyncio.Event().wait()
            async def send(message): messages.append(message)
            await self.app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1', 'method': 'POST',
                'scheme': 'http', 'path': path, 'raw_path': path.encode(), 'query_string': b'', 'root_path': '',
                'headers': [(b'content-type', ('multipart/form-data; boundary=' + boundary).encode()),
                            (b'content-length', str(len(body)).encode())],
                'client': ('testclient', 123), 'server': ('testserver', 80)}, receive, send)
            return messages
        messages = asyncio.run(request())
        status = next(row['status'] for row in messages if row['type'] == 'http.response.start')
        raw = b''.join(row.get('body', b'') for row in messages if row['type'] == 'http.response.body')
        class Response:
            status_code = status
            text = raw.decode('utf-8')
            def json(self): return json.loads(raw)
        return Response()


class StubRestorer:
    def __init__(self):
        self.counters = {'neural_forward_attempts': 0, 'neural_forward_returns': 0,
                         'completed_comparisons': 0, 'operator_review_refusals': 0}
        self.calls = []; self.releases = 0; self.fail = False; self.fail_cleanup = False
        self.entered = None; self.resume = None

    def count(self, name): self.counters[name] += 1
    def measured_status(self): return dict(self.counters)
    def release_models(self):
        self.releases += 1
        if self.fail_cleanup: raise RuntimeError('synthetic owned cleanup failure')

    def run_one(self, contents, selection):
        self.calls.append((contents, selection)); self.count('neural_forward_attempts')
        if self.entered is not None:
            self.entered.set()
            if not self.resume.wait(5): raise RuntimeError('bounded test barrier timeout')
        if self.fail: raise RuntimeError('synthetic inference failure')
        rgb = M.preparation.decode_original(contents)
        canvas, support, _, geometry = M.P.prepare_crop(rgb)
        raw = canvas.astype(np.float32) / np.float32(255)
        stream = io.BytesIO(); np.save(stream, raw, allow_pickle=False)
        self.count('neural_forward_returns')
        return {'PNG': encode(canvas), 'raw': stream.getvalue(), 'prepared': encode(canvas),
                'support': encode(support.astype(np.uint8) * 255),
                'metadata': {'selection': selection, 'geometry': geometry,
                             'uploaded_bytes_sha256': M.preparation.sha(contents), 'test_stub_not_model': True}}


class BrushFlowFixtures(unittest.TestCase):
    def setUp(self):
        guard = patch('builtins.__import__', blocked); guard.start(); self.addCleanup(guard.stop)
        # Scoped in-memory source configuration only; no fake receipt is saved.
        source = patch.object(M, 'source_check', return_value={'manifest': {'test_only': 'scoped-no-production-receipt'}})
        source.start(); self.addCleanup(source.stop)
        parent = patch.object(M.P, '_configured', {'test_only': 'scoped-parent-binding'})
        parent.start(); self.addCleanup(parent.stop)
        self.stub = StubRestorer(); self.flow = M.BrushFlow(self.stub)
        engine = patch.object(M, 'engine', self.flow); engine.start(); self.addCleanup(engine.stop)
        app = FastAPI(); app.include_router(M.router)
        self.client = ASGIClient(app); self.addCleanup(self.client.close)
        y, x = np.mgrid[:49, :65]
        self.rgb = np.stack([(x * 31 + y * 17) % 256, (x * 19 + y * 11) % 256, (x * 7 + y * 13) % 256], -1).astype(np.uint8)
        self.file = encode(self.rgb)
        self.protect = np.zeros((49, 65), np.uint8); self.protect[16:31, 22:39] = 255
        self.apply = np.full_like(self.protect, 255)
        self.p = encode(self.protect); self.a = encode(self.apply)

    def files(self, contents=None, protect=None, apply=None):
        return {'file': ('crop.png', self.file if contents is None else contents, 'image/png'),
                'protect_mask': ('protect.png', self.p if protect is None else protect, 'image/png'),
                'apply_mask': ('apply.png', self.a if apply is None else apply, 'image/png')}

    def preview(self):
        response = self.client.post(M.ROUTE + '/preview', files=self.files(), data={'radius': '6'})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def compare(self, preview=None, **changes):
        preview = self.preview() if preview is None else preview
        data = {'radius': '6', 'preview_id': preview['preview_id'], 'confirmed': 'true',
                'input_review': 'usable', 'selection': M.P.ARMS[1]}
        data.update(changes.pop('data', {}))
        return self.client.post(M.ROUTE + '/compare', files=self.files(**changes), data=data)

    def test_01_inspect_and_preview_are_real_pixels_and_non_neural(self):
        inspected = self.client.post(M.ROUTE + '/inspect', files={'file': ('crop.png', self.file, 'image/png')})
        self.assertEqual(inspected.status_code, 200); self.assertEqual(inspected.json()['neural_calls'], 0)
        result = self.preview(); self.assertTrue(result['metadata']['every_protected_native_RGB_exact'])
        self.assertEqual(self.stub.calls, []); self.assertEqual(self.stub.releases, 0)
        self.assertEqual('torch' in sys.modules, TORCH_PREEXISTING)

    def test_02_actual_multipart_input_review_refusals_precede_image_and_model(self):
        for review in ('', 'needs_clearer', 'out_of_scope'):
            response = self.client.post(M.ROUTE + '/compare', files=self.files(contents=b'invalid'),
                                        data={'input_review': review, 'confirmed': 'false'})
            self.assertEqual(response.status_code, 422)
        self.assertEqual(self.stub.counters['operator_review_refusals'], 3); self.assertEqual(self.stub.calls, [])

    def test_03_missing_confirmation_refuses_before_model(self):
        response = self.compare(data={'confirmed': 'false'})
        self.assertEqual(response.status_code, 422); self.assertIn('Confirm', response.json()['detail'])
        self.assertEqual(self.stub.calls, [])

    def test_04_original_both_masks_and_radius_edits_make_preview_stale(self):
        preview = self.preview(); changed_rgb = self.rgb.copy(); changed_rgb[0, 0] ^= 1
        protect = self.protect.copy(); protect[0, 0] = 255
        apply = self.apply.copy(); apply[0, 0] = 0
        for change in ({'contents': encode(changed_rgb)}, {'protect': encode(protect)}, {'apply': encode(apply)}, {'data': {'radius': '5'}}):
            response = self.compare(preview, **change)
            self.assertEqual(response.status_code, 422, response.text); self.assertIn('stale', response.json()['detail'])
        self.assertEqual(self.stub.calls, [])

    def test_05_nonnoop_emptyProtect_generation_refuses_but_preview_is_allowed(self):
        zero = encode(np.zeros_like(self.protect))
        response = self.client.post(M.ROUTE + '/preview', files=self.files(protect=zero), data={'radius': '6'})
        self.assertEqual(response.status_code, 200); preview = response.json()
        self.assertFalse(preview['metadata']['no_effect'])
        refused = self.compare(preview, protect=zero)
        self.assertEqual(refused.status_code, 422); self.assertIn('Protect the face', refused.json()['detail'])
        self.assertEqual(self.stub.calls, [])

    def test_06_compare_has_two_same_selected_calls_and_exact_bundle_bindings(self):
        response = self.compare(); self.assertEqual(response.status_code, 200, response.text); value = response.json()
        self.assertEqual(len(self.stub.calls), 2); self.assertEqual([r[1] for r in self.stub.calls], [M.P.ARMS[1]] * 2)
        self.assertEqual(self.stub.calls[0][0], self.file); self.assertNotEqual(self.stub.calls[1][0], self.file)
        self.assertEqual(value['metadata']['neural_forward_attempts'], 2); self.assertEqual(self.stub.releases, 1)
        import zipfile
        with zipfile.ZipFile(io.BytesIO(blob(value['bundle']))) as archive:
            self.assertEqual(archive.read('uploaded-original.bin'), self.file)
            self.assertEqual(archive.read('derived-native.png'), self.stub.calls[1][0])
            self.assertEqual(archive.read('original-restoration.png'), blob(value['original_result']))
            self.assertEqual(archive.read('prepared-restoration.png'), blob(value['prepared_result']))
            metadata = json.loads(archive.read('restoration.json'))
            self.assertEqual(metadata, value['metadata'])
            for name, row in metadata['preparation']['payload_pins'].items():
                raw = archive.read(name); self.assertEqual(row, {'bytes': len(raw), 'sha256': M.preparation.sha(raw)})
            raw = np.load(io.BytesIO(archive.read('prepared-restoration-raw.npy')), allow_pickle=False)
            self.assertEqual(raw.dtype, np.float32); self.assertEqual(raw.shape, (256, 256, 3))

    def test_07_bypass_is_one_call_and_bundle_has_truthful_original_no_masks(self):
        response = self.client.post(M.ROUTE + '/bypass', files={'file': ('crop.png', self.file, 'image/png')},
                                    data={'input_review': 'usable', 'selection': M.P.ARMS[0]})
        self.assertEqual(response.status_code, 200, response.text); value = response.json()
        self.assertEqual(len(self.stub.calls), 1); self.assertEqual(self.stub.releases, 1)
        self.assertEqual(value['metadata']['mode'], 'original_bypass'); self.assertIsNone(value['metadata']['preparation'])
        self.assertIsNone(value['metadata']['preview_id']); self.assertEqual(value['original_result'], value['prepared_result'])
        import zipfile
        with zipfile.ZipFile(io.BytesIO(blob(value['bundle']))) as archive:
            self.assertIn('original-decoded-rgb.png', archive.namelist()); self.assertNotIn('Apply-native.png', archive.namelist())
            self.assertEqual(archive.read('uploaded-original.bin'), self.file)

    def test_08_forward_and_cleanup_failures_release_outer_lock(self):
        self.stub.fail = True
        response = self.client.post(M.ROUTE + '/bypass', files={'file': ('crop.png', self.file, 'image/png')}, data={'input_review': 'usable'})
        self.assertEqual(response.status_code, 409); self.assertEqual(self.stub.releases, 1)
        self.assertTrue(self.flow.lock.acquire(blocking=False)); self.flow.lock.release()
        self.stub.fail = False; self.stub.fail_cleanup = True
        response = self.client.post(M.ROUTE + '/bypass', files={'file': ('crop.png', self.file, 'image/png')}, data={'input_review': 'usable'})
        self.assertEqual(response.status_code, 409)
        self.assertTrue(self.flow.lock.acquire(blocking=False)); self.flow.lock.release()

    def test_09_real_barrier_busy_request_gets409_without_second_stub_call(self):
        self.stub.entered = threading.Event(); self.stub.resume = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(self.flow.compare, self.file, b'', b'', 0., '', False, 'usable', M.P.ARMS[0], True)
            self.assertTrue(self.stub.entered.wait(3))
            try:
                second = self.client.post(M.ROUTE + '/bypass', files={'file': ('crop.png', self.file, 'image/png')}, data={'input_review': 'usable'})
                self.assertEqual(second.status_code, 409); self.assertIn('already running', second.json()['detail'])
                self.assertEqual(len(self.stub.calls), 1); self.assertEqual(self.stub.releases, 0)
            finally: self.stub.resume.set()
            self.assertEqual(first.result(timeout=5)['metadata']['neural_forward_returns'], 1)
        self.assertEqual(self.stub.releases, 1); self.assertFalse(self.flow.lock.locked())

    def test_10_measured_status_failure_also_cleans_up_and_unlocks(self):
        with patch.object(self.stub, 'measured_status', side_effect=RuntimeError('synthetic counter failure')):
            response = self.client.post(M.ROUTE + '/bypass', files={'file': ('crop.png', self.file, 'image/png')}, data={'input_review': 'usable'})
        self.assertEqual(response.status_code, 409); self.assertEqual(self.stub.calls, [])
        self.assertEqual(self.stub.releases, 1); self.assertFalse(self.flow.lock.locked())


if __name__ == '__main__': unittest.main()
