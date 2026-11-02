"""Independent hostile-artifact and request-contract tests; no model or worker call."""
import base64
import importlib.util
import io
import json
import os
import subprocess
import time
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SERVER = ROOT / 'completion_codeformer_web_v1.py'
SPEC = importlib.util.spec_from_file_location('manual_completion_web_contract', SERVER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TEMP_ROOT = Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')


def data_url(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def fixture(path):
    yy, xx = np.mgrid[:256, :256]
    original = np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype(np.uint8)
    mask = np.zeros((256, 256), np.uint8)
    mask[120:136, 120:136] = 255
    completed = original.astype(np.float32) / np.float32(255)
    completed[mask != 0] = np.float32(.5)
    result = original.copy()
    result[mask != 0] = 128
    Image.fromarray(original).save(path / 'input.png')
    Image.fromarray(mask).save(path / 'mask.png')
    Image.fromarray(result).save(path / 'result.png')
    (path / 'request.json').write_text('{"prepared_bounds":[0,0,256,256]}', encoding='utf-8')
    np.savez_compressed(path / 'stages.npz', input_rgb=original, removal_mask=mask // 255,
                        completed_rgb=completed, output_rgb=result)
    expected_inputs = {name: MODULE.sha(path / name) for name in ('input.png', 'mask.png', 'request.json')}
    receipt = {'complete': True, 'neural_calls': 1, 'inputs_sha256': expected_inputs,
               'outputs_sha256': {name: MODULE.sha(path / name) for name in ('result.png', 'stages.npz')}}
    (path / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
    return original, mask, completed, result, expected_inputs


def reseal_outputs(path):
    receipt = json.loads((path / 'receipt.json').read_text())
    receipt['outputs_sha256'] = {name: MODULE.sha(path / name) for name in ('result.png', 'stages.npz')}
    (path / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')


def app_without_model():
    app = MODULE.CompletionApp.__new__(MODULE.CompletionApp)
    app.generation_lock = threading.Lock()
    ident = '1' * 24
    app.sessions = {ident: {'prepared': True, 'metadata': {'prepared_bounds': [0, 0, 256, 256]}}}
    return app, ident


class CompletionWebContract(unittest.TestCase):
    def test_import_and_audit_need_no_torch(self):
        self.assertNotIn('torch', sys.modules)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            path = Path(temporary)
            _, mask, _, _, expected = fixture(path)
            receipt = MODULE.audit_worker_result(path, expected, mask)
            self.assertTrue(receipt['complete'])
        self.assertNotIn('torch', sys.modules)

    def test_nonbinary_colored_transparent_and_wrong_size_masks_rejected(self):
        images = [Image.new('L', (256, 256), 127), Image.new('RGB', (256, 256), (255, 0, 0)),
                  Image.new('RGBA', (256, 256), (0, 0, 0, 0)), Image.new('L', (128, 128), 0)]
        for image in images:
            with self.subTest(mode=image.mode, size=image.size, color=image.getpixel((0, 0))):
                with self.assertRaises(ValueError):
                    MODULE.decode_data_url(data_url(image), mask=True)
        _, accepted = MODULE.decode_data_url(data_url(Image.new('L', (256, 256), 0)), mask=True)
        self.assertEqual(accepted.shape, (256, 256))
        self.assertFalse(accepted.any())

    def test_confirmations_padding_and_insufficient_structure_stop_before_spawn(self):
        app, ident = app_without_model()
        good = {'id': ident, 'mask_reviewed': True, 'visible_structure_confirmed': True,
                'mask': data_url(Image.new('L', (256, 256), 0))}
        with patch.object(MODULE.subprocess, 'Popen', side_effect=AssertionError('No worker allowed')):
            for key in ('mask_reviewed', 'visible_structure_confirmed'):
                with self.subTest(confirmation=key), self.assertRaisesRegex(ValueError, 'Confirm'):
                    MODULE.CompletionApp.generate(app, {**good, key: False})
            with self.assertRaisesRegex(ValueError, 'visible face'):
                MODULE.CompletionApp.generate(app, {**good, 'mask': data_url(Image.new('L', (256, 256), 255))})
            app.sessions[ident]['metadata']['prepared_bounds'] = [16, 0, 240, 256]
            mask = np.zeros((256, 256), np.uint8)
            mask[100, 2] = 255
            with self.assertRaisesRegex(ValueError, 'padding'):
                MODULE.CompletionApp.generate(app, {**good, 'mask': data_url(Image.fromarray(mask))})
            app.sessions[ident]['metadata']['prepared_bounds'] = [0, 125, 256, 131]
            with self.assertRaisesRegex(ValueError, 'visible face'):
                MODULE.CompletionApp.generate(app, good)

    def test_concurrency_refuses_second_estimate_before_spawn(self):
        app, ident = app_without_model()
        app.generation_lock.acquire()
        try:
            with patch.object(MODULE.subprocess, 'Popen', side_effect=AssertionError('No worker allowed')):
                with self.assertRaises(BlockingIOError):
                    MODULE.CompletionApp.generate(app, {'id': ident, 'mask_reviewed': True,
                        'visible_structure_confirmed': True, 'mask': data_url(Image.new('L', (256, 256), 0))})
        finally:
            app.generation_lock.release()

    def test_raw_prediction_change_rejected_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            path = Path(temporary)
            original, mask, completed, result, expected = fixture(path)
            completed[125, 125] = np.float32(.25)
            np.savez_compressed(path / 'stages.npz', input_rgb=original, removal_mask=mask // 255,
                                completed_rgb=completed, output_rgb=result)
            reseal_outputs(path)
            with self.assertRaisesRegex(ValueError, 'replay'):
                MODULE.audit_worker_result(path, expected, mask)

    def test_png_change_rejected_even_when_stage_output_and_hash_are_resealed(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            path = Path(temporary)
            original, mask, completed, result, expected = fixture(path)
            result[125, 125] = 0
            Image.fromarray(result).save(path / 'result.png')
            np.savez_compressed(path / 'stages.npz', input_rgb=original, removal_mask=mask // 255,
                                completed_rgb=completed, output_rgb=result)
            reseal_outputs(path)
            with self.assertRaisesRegex(ValueError, 'replay'):
                MODULE.audit_worker_result(path, expected, mask)

    def test_changed_request_or_forged_input_receipt_rejected(self):
        for mutation in ('input', 'receipt'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
                path = Path(temporary)
                _, mask, _, _, expected = fixture(path)
                if mutation == 'input':
                    Image.new('RGB', (256, 256), 'white').save(path / 'input.png')
                else:
                    receipt = json.loads((path / 'receipt.json').read_text())
                    receipt['inputs_sha256']['input.png'] = '0' * 64
                    (path / 'receipt.json').write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, 'input|submitted'):
                    MODULE.audit_worker_result(path, expected, mask)

    def test_visible_raw_float_mutation_rejected_after_visible_byte_copy(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            path = Path(temporary)
            original, mask, completed, result, expected = fixture(path)
            completed[4, 4] = np.float32(.5)
            np.savez_compressed(path / 'stages.npz', input_rgb=original, removal_mask=mask // 255,
                                completed_rgb=completed, output_rgb=result)
            reseal_outputs(path)
            with self.assertRaisesRegex(ValueError, 'visible pixels'):
                MODULE.audit_worker_result(path, expected, mask)


    @unittest.skipUnless(os.name == 'nt', 'Actual Windows process ownership')
    def test_shutdown_event_terminates_owned_venv_tree_on_request_thread(self):
        import completion_process_guard_v1 as guard
        import ctypes
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temporary:
            ready = Path(temporary) / 'ready.json'
            code = "import json,os,pathlib,subprocess,sys,time;child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)']);pathlib.Path(sys.argv[1]).write_text(json.dumps({'actual':os.getpid(),'child':child.pid}));time.sleep(60)"
            with MODULE.OwnedWorker([sys.executable, '-B', '-c', code, str(ready)]) as worker:
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists())
                ids = json.loads(ready.read_text())
                shutdown = threading.Event()
                setter = threading.Timer(.05, shutdown.set)
                setter.start()
                try:
                    with self.assertRaisesRegex(RuntimeError, 'shutting down'):
                        MODULE.wait_owned_worker(worker, shutdown, 10)
                finally:
                    setter.join(timeout=2)
                self.assertEqual(worker.active_processes(), 0)
                for pid in ids.values():
                    handle = guard._K.OpenProcess(0x00100000, False, pid)
                    if handle:
                        try:
                            self.assertEqual(guard._K.WaitForSingleObject(handle, 0), 0)
                        finally:
                            guard._K.CloseHandle(handle)
                self.assertNotIn('torch', sys.modules)

    @unittest.skipUnless(os.name == 'nt', 'Actual Windows process ownership')
    def test_full_wait_deadline_requires_callers_tree_termination(self):
        with MODULE.OwnedWorker([sys.executable, '-B', '-c', 'import time;time.sleep(60)']) as worker:
            with self.assertRaises(subprocess.TimeoutExpired):
                MODULE.wait_owned_worker(worker, threading.Event(), .05)
            worker.kill()
            worker.wait_tree(timeout=5)
            self.assertEqual(worker.active_processes(), 0)
            self.assertNotEqual(worker.returncode, 0)

    def test_server_close_only_sets_event_and_never_touches_worker_handles(self):
        app = MODULE.CompletionApp.__new__(MODULE.CompletionApp)
        app.shutdown_event = threading.Event()
        app.owned_processes = {object()}
        MODULE.CompletionApp.close(app)
        self.assertTrue(app.shutdown_event.is_set())

if __name__ == '__main__':
    unittest.main()
