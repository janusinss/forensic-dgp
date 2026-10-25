"""HTTP semantics with counted fake workers and actual retained transport replay.

No app/model import, process worker or application server is required. Requests
use direct ASGI transport and real multipart parsing. Windows asyncio's internal
loopback socketpair is allowed; all external network/process work is blocked.
"""
import base64
import asyncio
import copy
import importlib.util
import io
import json
import socket
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    def barrier(event, args):
        if event == 'import' and (args[0] == 'torch' or args[0].startswith('torch.')
                                 or args[0] in ('app', 'face_workflow_web')):
            raise AssertionError('No app or Torch for pure HTTP boundary tests.')
        if event in ('socket.connect', 'socket.bind'):
            frame = sys._getframe(1)
            internal_pair = False
            while frame:
                if frame.f_code.co_name == '_fallback_socketpair' and frame.f_code.co_filename == socket.__file__:
                    internal_pair = True
                    break
                frame = frame.f_back
            if internal_pair and args[1][0] in ('127.0.0.1', '::1'):
                return  # Windows event-loop self-pipe, never an application connection.
        if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'):
            raise PermissionError('No process/network/model in HTTP boundary tests.')
    sys.addaudithook(barrier)

from fastapi import FastAPI
import numpy as np
from PIL import Image
import codeformer_restoration_web_v1 as HTTP
import codeformer_restoration_workflow_v1 as W

spec = importlib.util.spec_from_file_location('web_controller_fixture', ROOT / 'tests/test_codeformer_restoration_workflow_v1.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


class ASGIClient:
    """Minimal in-process HTTP transport; no optional httpx installation."""
    def __init__(self, app):
        self.app = app

    def get(self, path):
        return self.request('GET', path)

    def post(self, path, data, files):
        boundary = 'boundary-for-reviewed-restoration-test'
        chunks = []
        for name, value in data.items():
            chunks.append(('--' + boundary + '\r\nContent-Disposition: form-data; name="' + name
                           + '"\r\n\r\n' + str(value) + '\r\n').encode())
        for name, value in files.items():
            filename, raw, *kind = value
            chunks.extend([('--' + boundary + '\r\nContent-Disposition: form-data; name="' + name
                            + '"; filename="' + filename + '"\r\nContent-Type: '
                            + (kind[0] if kind else 'application/octet-stream') + '\r\n\r\n').encode(),
                           raw, b'\r\n'])
        chunks.append(('--' + boundary + '--\r\n').encode())
        return self.request('POST', path, b''.join(chunks), 'multipart/form-data; boundary=' + boundary)

    def request(self, method, path, body=b'', kind=''):
        sent = []
        async def run():
            consumed = False
            async def receive():
                nonlocal consumed
                if consumed:
                    return {'type':'http.disconnect'}
                consumed = True
                return {'type':'http.request','body':body,'more_body':False}
            async def send(message):
                sent.append(message)
            headers = [(b'host', b'testserver'), (b'content-length', str(len(body)).encode())]
            if kind: headers.append((b'content-type', kind.encode()))
            scope = {'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},
                     'http_version':'1.1','method':method,'scheme':'http', 'path':path,
                     'raw_path':path.encode(),'root_path':'','query_string':b'', 'headers':headers,
                     'server':('testserver',80),'client':('127.0.0.1',1)}
            await self.app(scope, receive, send)
        asyncio.run(run())
        status = next(item['status'] for item in sent if item['type'] == 'http.response.start')
        payload = b''.join(item.get('body',b'') for item in sent if item['type'] == 'http.response.body')
        class Response:
            status_code, text = status, payload.decode()
            def json(self): return json.loads(payload)
        return Response()


class WebTests(unittest.TestCase):
    def setUp(self):
        self.worker = F.Worker(saved=True)
        self.controller = W.RestorationController(self.worker, lambda: copy.deepcopy(F.SOURCE), F.SOURCE, F.MODELS)
        self.ready_calls = 0
        def ready():
            self.ready_calls += 1
        app = FastAPI()
        app.include_router(HTTP.create_router(self.controller, ready))
        self.client = ASGIClient(app)
        self.file = {'file': ('crop.png', F.CONTENTS, 'image/png')}

    def post(self, data=None, files=None, endpoint='restore'):
        return self.client.post('/face/restoration/' + endpoint, data=data or {}, files=files or self.file)

    def test_factory_and_status_do_not_generate_or_detect_covering(self):
        self.assertEqual(self.ready_calls, 0)
        response = self.client.get('/face/restoration/status')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['maximum_restore_forwards'], 1)
        self.assertFalse(response.json()['automatic_preparation'])
        self.assertEqual(self.worker.calls, [])
        self.assertEqual(self.ready_calls, 1)

    def test_inspect_oriented_native_with_zero_calls_and_invalid_upload(self):
        response = self.post(endpoint='inspect')
        self.assertEqual(response.status_code, 200)
        with Image.open(io.BytesIO(F.CONTENTS)) as image:
            self.assertEqual((response.json()['width'], response.json()['height']), image.size)
        self.assertTrue(response.json()['input_review_required'])
        self.assertEqual(self.post(endpoint='inspect', files={'file': ('bad.png', b'bad')}).status_code, 422)
        self.assertEqual(self.worker.calls, [])

    def test_off_bundle_and_png_are_exact_and_no_forward(self):
        response = self.post({'input_review': 'usable', 'restoration': 'off'})
        self.assertEqual(response.status_code, 200, response.text)
        value = response.json()
        self.assertEqual(value['original_input'], value['prepared_input'])
        self.assertEqual(value['original_input'], value['result'])
        self.assertFalse(value['metadata']['restoration_applied'])
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(value['bundle'].split(',')[1]))) as archive:
            self.assertEqual(archive.read('uploaded-original.bin'), F.CONTENTS)
            self.assertNotIn('restoration-raw-float32.npy', archive.namelist())
        self.assertEqual(self.worker.calls, [])

    def test_on_replays_one_actual_saved_delivery_and_download(self):
        response = self.post({'input_review': 'usable', 'restoration': 'on', 'include_bundle': 'false'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.worker.calls), 1)
        self.assertEqual(self.worker.releases, 1)
        self.assertEqual(self.worker.completed, 1)
        self.assertNotIn('bundle', response.json())
        pixels = W.delivery.png_array(base64.b64decode(response.json()['result'].split(',')[1]), 'RGB', (256, 256, 3))
        self.assertTrue(np.array_equal(pixels, W.delivery.png_array(F.SAVED['PNG'], 'RGB', (256,256,3))))

    def test_refusals_ambiguous_modes_and_missing_masks_zero_calls(self):
        for data in ({'input_review':'needs_clearer'}, {'input_review':'out_of_scope'},
                     {'input_review':'usable','preparation_mode':'manual'},
                     {'input_review':'usable','preparation_mode':'unknown'},
                     {'input_review':'usable','confirmed':'true'},
                     {'input_review':'usable','restoration':'unknown'}):
            self.assertEqual(self.post(data).status_code, 422)
        self.assertEqual(self.worker.calls, [])

    def test_preview_confirmation_and_stale_radius_invalidate_before_forward(self):
        contents = F.image_bytes()
        with Image.open(io.BytesIO(contents)) as image: width, height = image.size
        protect = np.zeros((height,width),np.uint8); protect[32:-32,32:-32] = 255
        files = {'file':('crop.png',contents,'image/png'),
                 'protect_mask':('protect.png',F.F.png(protect),'image/png'),
                 'apply_mask':('apply.png',F.F.png(np.full((height,width),255,np.uint8)),'image/png')}
        response = self.post({'input_review':'usable','radius':'1'},files,'preview')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['neural_calls'],0)
        self.assertTrue(response.json()['metadata']['every_protected_native_RGB_exact'])
        data = {'input_review':'usable','restoration':'on','preparation_mode':'manual',
                'preview_id':response.json()['preview_id'],'confirmed':'true','radius':'2'}
        self.assertEqual(self.post(data,files).status_code,422)
        self.assertEqual(self.worker.calls,[])

    def test_busy_and_unavailable_preserve_zero_calls(self):
        self.controller.lock.acquire()
        try: self.assertEqual(self.post({'input_review':'usable','restoration':'on'}).status_code,409)
        finally: self.controller.lock.release()
        with patch.object(self.controller,'status',side_effect=OSError('missing')):
            self.assertEqual(self.client.get('/face/restoration/status').status_code,503)
        self.assertEqual(self.worker.calls,[])

    def test_confirmed_manual_HTTP_request_forwards_only_derived_input(self):
        self.worker.saved = False  # Counted synthetic transport, not a model forward.
        contents = F.image_bytes()
        with Image.open(io.BytesIO(contents)) as image: width, height = image.size
        protect = np.zeros((height,width),np.uint8); protect[32:-32,32:-32] = 255
        protect_png = F.F.png(protect)
        apply_png = F.F.png(np.full((height,width),255,np.uint8))
        files = {'file':('crop.png',contents,'image/png'),
                 'protect_mask':('protect.png',protect_png,'image/png'),
                 'apply_mask':('apply.png',apply_png,'image/png')}
        preview = self.post({'input_review':'usable','radius':'1'}, files, 'preview')
        self.assertEqual(preview.status_code,200,preview.text)
        response = self.post({'input_review':'usable','restoration':'on','preparation_mode':'manual',
                              'radius':'1','preview_id':preview.json()['preview_id'],'confirmed':'true'},files)
        self.assertEqual(response.status_code,200,response.text)
        value = response.json()
        self.assertEqual(self.worker.calls,[(base64.b64decode(preview.json()['prepared_input'].split(',')[1]),W.SELECTION)])
        self.assertEqual(self.worker.completed,1)
        self.assertTrue(value['metadata']['preparation_applied'])
        self.assertEqual(value['metadata']['neural_forward_returns'],1)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(value['bundle'].split(',')[1]))) as archive:
            self.assertEqual(archive.read('uploaded-original.bin'),contents)
            self.assertEqual(archive.read('Protect-native.png'),protect_png)
            self.assertEqual(archive.read('Apply-native.png'),apply_png)
            self.assertIn('restoration-raw-float32.npy',archive.namelist())

    def test_status_never_advertises_active_uncertain_or_bad_worker_as_ready(self):
        original = self.worker.measured_status
        for changes in ({'active_worker':True}, {'uncertain_requests':1},
                        {'counter_integrity':False}, {'worker_cleanup_failures':1},
                        {'neural_forward_attempts':1,'neural_forward_returns':0}):
            with patch.object(self.worker,'measured_status',return_value={**original(),**changes}):
                response = self.client.get('/face/restoration/status')
                self.assertEqual(response.status_code,409,response.text)
                self.assertNotIn('available',response.json())
        self.assertEqual(self.worker.calls,[])


if __name__ == '__main__':
    unittest.main(verbosity=2)
