"""Controller and delivery checks using fake disposable workers; no inference."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
from fastapi import HTTPException, UploadFile

from implementation import app as controller
from implementation.preparation import build_preview, png
from implementation.runtime import prepare_crop


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.old_state = dict(controller.STATE)
        controller.STATE.update(neural_forward_attempts=0, neural_forward_returns=0,
                                active_worker=False, uncertain=False, last_worker_pid=None)
        fixture_root = Path(os.environ.get('IMPLEMENTATION_TEST_SCRATCH', 'scratch/implementation_app_tests')).resolve()
        fixture_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=fixture_root)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(fixture_root))
        self.root_patch = patch.object(controller, 'ROOT', self.root)
        self.root_patch.start()
        y, x = np.indices((96, 128))
        self.rgb = np.stack(((x * 17 + y * 11) % 256, (x * 5 + y * 23) % 256,
                             (x * 31 + y * 3) % 256), axis=2).astype(np.uint8)
        self.contents = png(self.rgb)

    def tearDown(self):
        self.root_patch.stop()
        controller.STATE.clear()
        controller.STATE.update(self.old_state)
        self.temporary.cleanup()

    @staticmethod
    def upload(contents, filename='crop.png'):
        return UploadFile(filename=filename, file=io.BytesIO(contents))

    def restore(self, **overrides):
        values = dict(file=self.upload(self.contents), input_review='usable', restoration='on',
                      preparation_mode='bypass', protect_mask=None, apply_mask=None, radius=1,
                      preview_id='', confirmed=False, include_bundle=True, model='u96')
        values.update(overrides)
        return asyncio.run(controller.restore(**values))

    def test_inspect_and_off_do_not_call_a_worker(self):
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            inspection = asyncio.run(controller.inspect(self.upload(self.contents)))
            response = self.restore(restoration='off', model='u64')
        self.assertEqual(inspection['neural_calls'], 0)
        self.assertEqual((inspection['width'], inspection['height']), (128, 96))
        self.assertEqual(response['metadata']['model'], 'u64')
        self.assertEqual(response['metadata']['neural_forward_attempts'], 0)
        self.assertEqual(response['metadata']['neural_forward_returns'], 0)
        self.assertEqual(response['original_input'], response['result'])
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(response['bundle'].split(',', 1)[1]))) as archive:
            self.assertEqual(archive.read('uploaded-original.bin'), self.contents)
            self.assertNotIn('restoration-raw-float32.npy', archive.namelist())

    def test_input_review_and_busy_reject_before_worker(self):
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            with self.assertRaises(HTTPException) as error:
                self.restore(input_review='needs_clearer')
            self.assertEqual(error.exception.status_code, 422)
            controller.LOCK.acquire()
            try:
                with self.assertRaises(HTTPException) as error:
                    self.restore()
                self.assertEqual(error.exception.status_code, 409)
            finally:
                controller.LOCK.release()
        self.assertEqual(controller.STATE['neural_forward_attempts'], 0)

    def test_manual_stale_and_unprotected_effect_reject_before_worker(self):
        protect = np.zeros(self.rgb.shape[:2], np.uint8)
        apply = np.full(self.rgb.shape[:2], 255, np.uint8)
        preview, _ = build_preview(self.contents, png(protect), png(apply), nominal_radius=1)
        self.assertFalse(preview['metadata']['no_effect'])
        def manual(identity):
            return self.restore(preparation_mode='manual', protect_mask=self.upload(png(protect)),
                                apply_mask=self.upload(png(apply)), preview_id=identity, confirmed=True)
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            with self.assertRaises(HTTPException) as stale:
                manual('stale')
            self.assertEqual(stale.exception.status_code, 422)
            self.assertIn('stale', stale.exception.detail)
            with self.assertRaises(HTTPException) as unprotected:
                manual(preview['preview_id'])
            self.assertEqual(unprotected.exception.status_code, 422)
            self.assertIn('Protect', unprotected.exception.detail)

    def test_manual_preparation_preserves_protect_and_outside_apply(self):
        protect = np.zeros(self.rgb.shape[:2], np.uint8); protect[32:64, 40:88] = 255
        apply = np.zeros(self.rgb.shape[:2], np.uint8); apply[:, :24] = 255
        preview, payloads = build_preview(self.contents, png(protect), png(apply), nominal_radius=3)
        with Image.open(io.BytesIO(payloads['derived-native.png'])) as image:
            derived = np.asarray(image)
        self.assertTrue(np.array_equal(derived[protect > 0], self.rgb[protect > 0]))
        self.assertTrue(np.array_equal(derived[apply == 0], self.rgb[apply == 0]))
        self.assertEqual(preview['metadata']['preprocessing_neural_optimizer_training_VM_calls'], 0)
        rebuilt, again = build_preview(self.contents, png(protect), png(apply), nominal_radius=3)
        self.assertEqual(preview['preview_id'], rebuilt['preview_id'])
        self.assertEqual(payloads['derived-native.png'], again['derived-native.png'])

    def fake_worker(self, command, **kwargs):
        request_path = Path(command[command.index('--request') + 1])
        result_path = Path(command[command.index('--result') + 1])
        request = json.loads(request_path.read_bytes())
        output = Path(request['output_dir']); output.mkdir()
        contents = Path(request['input_path']).read_bytes()
        with Image.open(io.BytesIO(contents)) as image:
            canvas, observed, _, geometry = prepare_crop(np.asarray(image.convert('RGB')))
        raw = canvas.astype(np.float32) / np.float32(255)
        raw[observed] = np.float32(.371)
        pixels = np.floor(raw * np.float32(255)).astype(np.uint8)
        pixels[~observed] = canvas[~observed]
        if getattr(self, 'corrupt_padding', False):
            pixels[0, 0, 0] ^= 1
        raw_stream = io.BytesIO(); np.save(raw_stream, raw, allow_pickle=False)
        (output / 'raw.npy').write_bytes(raw_stream.getvalue())
        png_bytes = png(pixels); (output / 'restoration.png').write_bytes(png_bytes)
        status = getattr(self, 'worker_status', {'phase': 'complete', 'forward_attempts': 1, 'forward_returns': 1})
        (output / 'status.json').write_text(json.dumps(status))
        result = {'raw_path': str(output / 'raw.npy'), 'png_path': str(output / 'restoration.png'),
                  'metadata': {'uploaded_bytes_sha256': controller.digest(contents), 'geometry': geometry,
                               'raw_rgb_sha256': controller.digest(raw.tobytes()),
                               'PNG_RGB8_sha256': controller.digest(pixels.tobytes()),
                               'PNG_file_sha256': controller.digest(png_bytes)}}
        result_path.write_text(json.dumps(result))
        self.process = types.SimpleNamespace(pid=98765, returncode=getattr(self, 'worker_returncode', 0))
        self.process.poll = lambda: self.process.returncode
        self.process.terminate = lambda: setattr(self.process, 'returncode', -15)
        self.process.kill = lambda: setattr(self.process, 'returncode', -9)
        self.process.wait = lambda timeout: self.process.returncode
        return self.process

    def worker_patches(self):
        return (patch.object(controller, 'verify_models'),
                patch.object(controller.psutil, 'virtual_memory', return_value=types.SimpleNamespace(available=8 * 1024**3)),
                patch.object(controller.subprocess, 'Popen', side_effect=self.fake_worker))

    def test_fake_completed_worker_checks_raw_png_hashes_and_counts(self):
        verify, memory, worker = self.worker_patches()
        with verify, memory, worker:
            raw, pixels, metadata = controller.one_forward(self.contents, 'u64')
        self.assertGreater(len(raw), 0); self.assertGreater(len(pixels), 0)
        self.assertEqual(metadata['uploaded_bytes_sha256'], controller.digest(self.contents))
        self.assertEqual(controller.STATE['neural_forward_attempts'], 1)
        self.assertEqual(controller.STATE['neural_forward_returns'], 1)
        self.assertFalse(controller.STATE['active_worker']); self.assertFalse(controller.STATE['uncertain'])

    def test_delivery_padding_corruption_latches_with_actual_completed_counts(self):
        self.corrupt_padding = True
        verify, memory, worker = self.worker_patches()
        with verify, memory, worker, self.assertRaises(ValueError):
            controller.one_forward(self.contents, 'u96')
        self.assertTrue(controller.STATE['uncertain'])
        self.assertEqual(controller.STATE['neural_forward_attempts'], 1)
        self.assertEqual(controller.STATE['neural_forward_returns'], 1)
        self.assertFalse(controller.STATE['active_worker'])
        with self.assertRaises(HTTPException) as blocked:
            controller.invoke(lambda: None)
        self.assertEqual(blocked.exception.status_code, 409)

    def test_failed_worker_retains_entered_forward_count(self):
        self.worker_returncode = 2
        self.worker_status = {'phase': 'forward_entered', 'forward_attempts': 1, 'forward_returns': 0}
        verify, memory, worker = self.worker_patches()
        with verify, memory, worker, self.assertRaises(RuntimeError):
            controller.one_forward(self.contents, 'u96')
        self.assertTrue(controller.STATE['uncertain'])
        self.assertEqual((controller.STATE['neural_forward_attempts'], controller.STATE['neural_forward_returns']), (1, 0))

    def test_timeout_terminates_only_own_worker_and_retains_actual_status(self):
        self.worker_returncode = None
        self.worker_status = {'phase': 'forward_entered', 'forward_attempts': 1, 'forward_returns': 0}
        verify, memory, worker = self.worker_patches()
        with verify, memory, worker, patch.object(controller.time, 'monotonic', side_effect=[0, 361]):
            with self.assertRaisesRegex(RuntimeError, '360-second'):
                controller.one_forward(self.contents, 'u96')
        self.assertEqual(self.process.returncode, -15)
        self.assertTrue(controller.STATE['uncertain']); self.assertFalse(controller.STATE['active_worker'])
        self.assertEqual((controller.STATE['neural_forward_attempts'], controller.STATE['neural_forward_returns']), (1, 0))

    def test_termination_denial_still_reconciles_persisted_forward_counts(self):
        self.worker_returncode = None
        self.worker_status = {'phase': 'forward_entered', 'forward_attempts': 1, 'forward_returns': 0}
        def denied_worker(command, **kwargs):
            process = self.fake_worker(command, **kwargs)
            def denied():
                raise PermissionError('simulated direct-PID termination denial')
            process.terminate = denied
            return process
        verify, memory, _ = self.worker_patches()
        with verify, memory, patch.object(controller.subprocess, 'Popen', side_effect=denied_worker), \
                patch.object(controller.time, 'monotonic', side_effect=[0, 361]):
            with self.assertRaises((OSError, RuntimeError)):
                controller.one_forward(self.contents, 'u96')
        self.assertTrue(controller.STATE['uncertain'])
        self.assertEqual((controller.STATE['neural_forward_attempts'], controller.STATE['neural_forward_returns']), (1, 0))


if __name__ == '__main__':
    unittest.main()
