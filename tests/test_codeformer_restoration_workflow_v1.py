"""Unwired controller tests: real saved transport, pure pixels and counted fake workers."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def barrier(event, args):
    if event == 'import' and (args[0] == 'torch' or args[0].startswith('torch.') or
                              args[0] in ('app', 'face_workflow', 'face_workflow_web', 'dgp_face_workflow_v3')):
        raise AssertionError('No app or Torch in pure controller tests')
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'):
        raise PermissionError('No process/network/model work in pure controller tests')


STANDALONE = __name__ == '__main__'
if STANDALONE:
    sys.addaudithook(barrier)
import cv2
import numpy as np
from PIL import Image
import codeformer_restoration_workflow_v1 as W

fixture_spec = importlib.util.spec_from_file_location('saved_delivery_fixture', ROOT / 'tests/test_codeformer_app_delivery_v1.py')
F = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(F)
CONTENTS, SAVED, ARM, MODELS = F.replay(0)
SOURCE = {'reviewed_sources': 'synthetic source-validator identity; no deployment claim'}


def image_bytes(shape=(192, 160)):
    yy, xx = np.indices(shape)
    values = np.stack(((xx * 7 + yy) % 256, (yy * 11 + xx) % 256, (xx + 3 * yy) % 256), 2).astype(np.uint8)
    return F.png(values)


def synthetic_delivery(contents):
    canvas, support, geometry = W.delivery.prepare_native(contents)
    raw = (canvas.astype(np.float32) / np.float32(255) * np.float32(.6) + np.float32(.2)).astype(np.float32)
    raw[~support] = canvas[~support].astype(np.float32) / np.float32(255)
    pixels = canvas.copy(); pixels[support] = np.floor(raw[support] * np.float32(255)).astype(np.uint8)
    value = copy.deepcopy(SAVED)
    value.update(PNG=F.png(pixels), raw=F.npy(raw), prepared=F.png(canvas), support=F.png(support.astype(np.uint8)*255))
    value['metadata'].update(geometry=geometry, uploaded_bytes_sha256=W.delivery.sha(contents),
                             raw_RGB_sha256=W.delivery.sha(raw.tobytes()), PNG_RGB_sha256=W.delivery.sha(pixels.tobytes()))
    return value


class Worker:
    def __init__(self, saved=False, malformed=False, bad_counts=False, cleanup_error=False):
        self.saved, self.malformed, self.bad_counts, self.cleanup_error = saved, malformed, bad_counts, cleanup_error
        self.calls, self.releases, self.attempts, self.returns, self.completed = [], 0, 0, 0, 0
        self.active, self.pending = False, False

    def measured_status(self):
        return {'neural_forward_attempts': self.attempts, 'neural_forward_returns': self.returns,
                'completed_comparisons': self.completed, 'operator_review_refusals': 0,
                'counter_integrity': True, 'uncertain_requests': 0, 'uncertain_forward_attempts_upper_bound': 0,
                'active_worker': self.active, 'active_forward_counts_known': True, 'worker_cleanup_failures': 0}

    def run_one(self, contents, selection):
        self.calls.append((contents, selection)); self.active = True
        if not self.bad_counts:
            self.attempts += 1; self.returns += 1
        value = copy.deepcopy(SAVED) if self.saved else synthetic_delivery(contents)
        if self.malformed:
            value['raw'] += b'bad'
        return value

    def count(self, name):
        if name != 'completed_comparisons': raise AssertionError('Unexpected counter')
        self.pending = True

    def release_models(self):
        self.releases += 1
        self.active = False
        if self.cleanup_error: raise RuntimeError('Synthetic cleanup failure')
        if self.pending: self.completed += 1; self.pending = False


class ControllerTests(unittest.TestCase):
    def controller(self, worker=None, validator=None):
        worker = worker or Worker()
        return W.RestorationController(worker, validator or (lambda: copy.deepcopy(SOURCE)), SOURCE, MODELS), worker

    def preparation(self, controller, contents):
        with Image.open(io.BytesIO(contents)) as image: width, height = image.size
        protect = np.zeros((height, width), np.uint8); protect[32:height-32, 32:width-32] = 255
        apply = np.full((height, width), 255, np.uint8)
        preview, _ = controller.preview(contents, F.png(protect), F.png(apply), 1., 'usable')
        return {'mode': 'manual', 'protect': F.png(protect), 'apply': F.png(apply), 'radius': 1.,
                'preview_id': preview['preview_id'], 'confirmed': True}

    def test_01_real_saved_U96_transport_one_call_exact_payload(self):
        controller, worker = self.controller(Worker(saved=True))
        result = controller.restore(CONTENTS, 'usable', 'on')
        self.assertEqual(len(worker.calls), 1); self.assertEqual(worker.calls[0], (CONTENTS, W.SELECTION))
        self.assertEqual(worker.releases, 1)
        self.assertTrue(np.array_equal(result['result'], W.delivery.png_array(SAVED['PNG'], 'RGB', (256,256,3))))
        self.assertTrue(np.array_equal(result['raw'], np.load(io.BytesIO(SAVED['raw']), allow_pickle=False)))
        self.assertFalse(result['result'].flags.writeable)
        with zipfile.ZipFile(io.BytesIO(result['bundle'])) as archive:
            self.assertEqual(archive.read('uploaded-original.bin'), CONTENTS)
            self.assertEqual(archive.read('result.png'), result['png'])
            self.assertEqual(json.loads(archive.read('restoration.json'))['selection'], W.SELECTION)
        result['metadata']['models']['u96']['sha256'] = '0'*64
        self.assertEqual(controller.models['u96']['sha256'], W.delivery.MODEL_FILES['u96'])

    def test_02_off_exact_original_and_ignored_stale_preparation_zero_calls(self):
        controller, worker = self.controller(); contents = image_bytes()
        option = self.preparation(controller, contents); option['preview_id'] = 'stale'; option['confirmed'] = False
        result = controller.restore(contents, 'usable', 'off', option)
        self.assertEqual(worker.calls, [])
        self.assertTrue(np.array_equal(result['original256'], result['prepared256']))
        self.assertTrue(np.array_equal(result['result'], result['original256']))
        self.assertIsNone(result['raw']); self.assertFalse(result['metadata']['preparation_applied'])
        self.assertEqual(result['metadata']['preparation_bypass_reason'], 'restoration_off')

    def test_03_input_refusal_uniform_and_nonempty_covering_zero_calls(self):
        cases = [(CONTENTS, 'needs_clearer', None), (CONTENTS, 'out_of_scope', None),
                 (CONTENTS, '', None), (F.png(np.full((64,64,3),128,np.uint8)), 'usable', None)]
        rgb = W.delivery.prepare_native(CONTENTS)
        with Image.open(io.BytesIO(CONTENTS)) as image: shape = (image.height, image.width)
        cases.append((CONTENTS, 'usable', np.ones(shape,np.uint8)))
        for contents, review, mask in cases:
            controller, worker = self.controller()
            with self.assertRaises(ValueError): controller.restore(contents, review, 'on', completion_mask=mask)
            self.assertEqual(worker.calls, []); self.assertEqual(worker.releases, 1)

    def test_04_stale_or_unconfirmed_or_edited_preview_zero_calls(self):
        contents = image_bytes(); controller, worker = self.controller()
        option = self.preparation(controller, contents)
        for bad in ({**option, 'preview_id':'stale'}, {**option,'confirmed':False}, {**option,'radius':2.}):
            with self.assertRaises(ValueError): controller.restore(contents,'usable','on',bad)
        self.assertEqual(worker.calls, []); self.assertEqual(worker.releases, 3)

    def test_05_preparation_immutable_rebuild_and_single_derived_forward(self):
        controller, worker = self.controller(); contents = image_bytes()
        option = self.preparation(controller, contents)
        first = controller.preview(contents, option['protect'],option['apply'],1.,'usable')
        second = controller.preview(contents, option['protect'],option['apply'],1.,'usable')
        self.assertEqual(first[0]['preview_id'],second[0]['preview_id']); self.assertEqual(first[1],second[1])
        result = controller.restore(contents,'usable','on',option)
        self.assertEqual(worker.calls,[(first[1]['derived-native.png'],W.SELECTION)])
        self.assertTrue(result['metadata']['preparation_applied'])
        with zipfile.ZipFile(io.BytesIO(result['bundle'])) as archive:
            self.assertEqual(archive.read('uploaded-original.bin'),contents)
            for name in ('Protect-native.png','Apply-native.png','alpha-float32.npy','effective-Protect-native.png'):
                self.assertEqual(archive.read(name),first[1][name])

    def test_06_Auto_only_original_signals_before_preparation(self):
        controller, worker = self.controller(); contents = image_bytes(); option = self.preparation(controller,contents)
        canvas, _, _ = W.delivery.prepare_native(contents)
        real = W.quality_signals(canvas,np.zeros((256,256),np.uint8))
        seen=[]
        def assess(actual, mask):
            seen.append(actual.copy()); self.assertTrue(np.array_equal(actual,canvas))
            return {**real,'suggest_restoration':True}
        with patch.object(W,'quality_signals',side_effect=assess):
            result=controller.restore(contents,'usable','auto',option)
        self.assertEqual(len(seen),1); self.assertEqual(len(worker.calls),1)
        self.assertTrue(result['metadata']['Auto_assessed_original_canvas_before_preparation'])
        controller,worker=self.controller()
        with patch.object(W,'quality_signals',return_value={**real,'suggest_restoration':False}):
            result=controller.restore(contents,'usable','auto',option)
        self.assertEqual(worker.calls,[]); self.assertFalse(result['metadata']['preparation_applied'])
        self.assertTrue(np.array_equal(result['result'],canvas))

    def test_07_exact_retained_quality_function_parity_without_Torch_import(self):
        path=ROOT/'face_workflow.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),'ed043f9819e4655c297f092037af156cbfa9e32506f6b15cca0c5d5dfe4cacd1')
        tree=ast.parse(path.read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='quality_signals')
        env={'cv2':cv2,'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+':pure-function-only','exec'),env)
        rng=np.random.default_rng(7)
        for shape in ((256,256),(31,57)):
            for covered in (False,True):
                rgb=rng.integers(0,256,(*shape,3),np.uint8);mask=np.zeros(shape,np.uint8)
                if covered:mask[:]=1;mask[:2,:2]=0
                self.assertEqual(W.quality_signals(rgb,mask),env['quality_signals'](rgb,mask))

    def test_08_busy_does_not_cleanup_another_owned_request(self):
        controller,worker=self.controller();controller.lock.acquire()
        try:
            with self.assertRaises(W.RestorationBusy):controller.restore(CONTENTS,'usable','on')
        finally:controller.lock.release()
        self.assertEqual(worker.releases,0);self.assertEqual(worker.calls,[])

    def test_09_cleanup_parser_failure_and_counter_errors_do_not_retry(self):
        for worker in (Worker(malformed=True),Worker(bad_counts=True),Worker(cleanup_error=True)):
            controller,_=self.controller(worker)
            with self.assertRaises((ValueError,RuntimeError)):controller.restore(CONTENTS,'usable','on')
            self.assertEqual(len(worker.calls),1);self.assertEqual(worker.releases,1)
            self.assertFalse(controller.lock.locked())

    def test_10_source_before_after_and_rejection_still_cleanup(self):
        count=[0]
        def source():
            count[0]+=1
            return copy.deepcopy(SOURCE)if count[0]==1 else {'changed':True}
        controller,worker=self.controller(validator=source)
        with self.assertRaisesRegex(ValueError,'source identity'):controller.restore(CONTENTS,'usable','on')
        self.assertEqual(count[0],2);self.assertEqual(len(worker.calls),1);self.assertEqual(worker.releases,1)

    def test_11_uncertain_worker_refuses_before_forward(self):
        worker=Worker();real=worker.measured_status
        worker.measured_status=lambda:{**real(),'uncertain_requests':1,'counter_integrity':False}
        controller,_=self.controller(worker)
        with self.assertRaises(ValueError):controller.restore(CONTENTS,'usable','on')
        self.assertEqual(worker.calls,[]);self.assertEqual(worker.releases,1)

    def test_12_bounded_bundle_optional_and_no_completion_or_activation(self):
        controller,worker=self.controller()
        result=controller.restore(CONTENTS,'usable','off',include_bundle=False)
        self.assertIsNone(result['bundle']);self.assertFalse(result['metadata']['model_qualified'])
        self.assertFalse(result['metadata']['main_app_default_changed'])
        with patch.object(W,'BUNDLE_LIMIT',5),self.assertRaises(ValueError):W.bundle_bytes({'a':b'123456'})
        if STANDALONE:
            for name in ('torch','app','face_workflow','face_workflow_web','dgp_face_workflow_v3'):
                self.assertNotIn(name,sys.modules)

    def test_13_explicit_staging_identity_never_claims_main_promotion(self):
        worker = Worker()
        controller = W.RestorationController(worker, lambda: copy.deepcopy(SOURCE), SOURCE, MODELS,
                                             deployment='opt_in_U96_staging')
        status = controller.status()
        self.assertFalse(status['unwired'])
        self.assertEqual(status['deployment'], 'opt_in_U96_staging')
        self.assertFalse(status['main_app_default_changed'])
        result = controller.restore(CONTENTS, 'usable', 'off')
        self.assertFalse(result['metadata']['unwired'])
        self.assertFalse(result['metadata']['main_app_default_changed'])
        self.assertFalse(result['metadata']['model_qualified'])
        self.assertEqual(worker.calls, [])
        with self.assertRaises(ValueError):
            W.RestorationController(worker, lambda: SOURCE, SOURCE, MODELS, deployment='production-promoted')


if __name__=='__main__':unittest.main(verbosity=2)
