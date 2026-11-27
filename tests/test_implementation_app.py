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
from PIL import Image, ImageFilter
from fastapi import HTTPException, UploadFile

from implementation import app as controller
from implementation import blur_match, capture_api
from implementation.preparation import build_preview, png
from implementation.runtime import prepare_crop
from tests.test_capture_api import asgi_request


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
                      preview_id='', confirmed=False, include_bundle=True, model='v3')
        values.update(overrides)
        return asyncio.run(controller.restore(**values))

    def make_video_crop(self):
        self.capture_owner = 'capture-owner'
        self.other_owner = 'other-capture-owner'
        for key, value in {'STORE': self.root / 'capture', 'FRAMES': {}, 'CROPS': {},
                           'TOKENS': {self.capture_owner, self.other_owner}, 'detect_faces': lambda image: []}.items():
            setting = patch.object(capture_api, key, value)
            setting.start()
            self.addCleanup(setting.stop)
        frame = capture_api._freeze(self.contents, 'video', 1.25, self.capture_owner)
        rectangle = capture_api.Rectangle(x=4, y=5, width=40, height=32)
        crop = capture_api._crop(frame['frame_id'], rectangle, self.capture_owner)
        self.capture_id = crop['capture_id']
        self.capture_contents = base64.b64decode(crop['image'].split(',', 1)[1])

    def video_restore(self, token, *, contents=None, **overrides):
        fields = dict(input_review='video_capture', restoration='off', preparation_mode='bypass',
                      capture_id=self.capture_id, model='v3', include_bundle='true')
        fields.update(overrides)
        boundary = b'controller-test-boundary'
        body = b''.join(b'--' + boundary + f'\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
                        for name, value in fields.items())
        body += b'--' + boundary + b'\r\nContent-Disposition: form-data; name="file"; filename="crop.png"\r\nContent-Type: image/png\r\n\r\n'
        body += self.capture_contents if contents is None else contents
        body += b'\r\n--' + boundary + b'--\r\n'
        headers = {'host': '127.0.0.1', 'content-type': f'multipart/form-data; boundary={boundary.decode()}'}
        if token is not None:
            headers['x-capture-token'] = token
        status, response = asyncio.run(asgi_request(controller.app, 'POST', '/face/restoration/restore', body, headers))
        return status, json.loads(response)

    def test_video_capture_restoration_delivers_bound_frame_and_crop(self):
        self.make_video_crop()
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            status, response = self.video_restore(self.capture_owner)
        self.assertEqual(status, 200, response.get('detail'))
        self.assertEqual(response['metadata']['input_review'], 'video_capture')
        self.assertEqual(response['metadata']['capture_id'], self.capture_id)
        self.assertFalse(response['metadata']['restoration_applied'])
        self.assertEqual(response['original_input'], response['result'])
        bundle_bytes = base64.b64decode(response['bundle'].split(',', 1)[1])
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
            self.assertEqual(archive.read('capture/frame.png'), self.contents)
            self.assertEqual(archive.read('capture/crop.png'), self.capture_contents)
            capture_metadata = json.loads(archive.read('capture/metadata.json'))
            self.assertEqual(capture_metadata['rectangle'], [4, 5, 40, 32])
            self.assertEqual(capture_metadata['source_time'], 1.25)

    def test_video_capture_restoration_requires_matching_crop_and_owner_before_inference(self):
        self.make_video_crop()
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            for token, overrides, expected in (
                (None, {}, 403),
                (self.other_owner, {}, 404),
                (None, {'input_review': 'usable'}, 403),
                (self.other_owner, {'input_review': 'usable'}, 404),
                (self.capture_owner, {'capture_id': ''}, 422),
                (self.capture_owner, {'contents': self.contents}, 422),
                (self.capture_owner, {'preparation_mode': 'manual'}, 422),
                (self.capture_owner, {'restoration': 'auto'}, 422),
            ):
                with self.subTest(token=token, overrides=tuple(overrides)):
                    status, response = self.video_restore(token, **{'restoration': 'on', **overrides})
                    self.assertEqual(status, expected, response.get('detail'))
        self.assertEqual(controller.STATE['neural_forward_attempts'], 0)

    def test_inspect_and_off_do_not_call_a_worker(self):
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            inspection = asyncio.run(controller.inspect(self.upload(self.contents)))
            response = self.restore(restoration='off', model='v3', input_review='picture_upload')
        self.assertEqual(inspection['neural_calls'], 0)
        self.assertFalse(inspection['input_review_required'])
        self.assertEqual((inspection['width'], inspection['height']), (128, 96))
        self.assertEqual(response['metadata']['model'], 'v3')
        self.assertEqual(response['metadata']['input_review'], 'picture_upload')
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

    def test_picture_face_scan_and_crop_preserve_the_original_upload(self):
        with patch.object(capture_api, 'detect_faces', return_value=[[42, 24, 30, 40]]):
            inspection = asyncio.run(controller.inspect(self.upload(self.contents), scan_faces=True))
        rectangle = [29, 11, 56, 66]
        self.assertEqual(inspection['face_crops'], [rectangle])
        self.assertEqual(inspection['face_boxes'], [[42, 24, 30, 40]])
        self.assertEqual(inspection['neural_calls'], 0)
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            result = self.restore(restoration='off', input_review='picture_upload', picture_crop=json.dumps(rectangle))
        self.assertEqual(result['metadata']['picture_crop'], rectangle)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(result['bundle'].split(',', 1)[1]))) as archive:
            self.assertEqual(archive.read('uploaded-original.bin'), self.contents)
            self.assertEqual(archive.read('original-native.png'), png(self.rgb[11:77, 29:85]))

    def test_picture_crop_validation_and_detector_fallback(self):
        with patch.object(capture_api, 'detect_faces', side_effect=RuntimeError):
            inspection = asyncio.run(controller.inspect(self.upload(self.contents), scan_faces=True))
        self.assertEqual(inspection['face_crops'], [])
        self.assertEqual(inspection['face_boxes'], [])
        self.assertTrue(inspection['face_scan_unavailable'])
        for invalid in ('[0,0,2,2]', '[0,0,129,96]', '[0,0,40,40.0]', 'bad'):
            with self.subTest(invalid=invalid), self.assertRaises(HTTPException) as error:
                self.restore(restoration='off', input_review='picture_upload', picture_crop=invalid)
            self.assertEqual(error.exception.status_code, 422)

    def test_inspect_uses_detect_face_details_and_sorts_by_position(self):
        details = [
            {'box': [60, 20, 30, 36], 'crop': [50, 10, 50, 50], 'score': 0.90},
            {'box': [12, 16, 28, 34], 'crop': [4, 8, 44, 44], 'score': 0.88},
        ]
        with patch.object(capture_api, 'detect_face_details', return_value=details) as detail_call:
            inspection = asyncio.run(controller.inspect(self.upload(self.contents), scan_faces=True))
        detail_call.assert_called_once()
        self.assertEqual(inspection['face_boxes'], [[12, 16, 28, 34], [60, 20, 30, 36]])
        self.assertEqual(inspection['face_crops'], [[4, 8, 44, 44], [50, 10, 50, 50]])
        self.assertFalse(inspection['face_scan_unavailable'])

    @unittest.skipUnless(
        all((Path(__file__).resolve().parents[2] / 'test_images' / name).exists() for name in ('wild3.png', 'wild4.png', 'wild5.png')),
        'wild3/wild4/wild5 fixtures missing',
    )
    def test_inspect_real_wild3_wild4_wild5_produce_auto_crop_margin(self):
        from implementation.capture_worker import AUTO_CROP_MARGIN_RATIO
        test_dir = Path(__file__).resolve().parents[2] / 'test_images'
        for name in ('wild3.png', 'wild4.png', 'wild5.png'):
            with self.subTest(image=name):
                raw = (test_dir / name).read_bytes()
                inspection = asyncio.run(controller.inspect(self.upload(raw, filename=name), scan_faces=True))
                self.assertFalse(inspection['face_scan_unavailable'])
                self.assertEqual(len(inspection['face_boxes']), 1)
                self.assertEqual(len(inspection['face_crops']), 1)
                cx, cy, cw, ch = inspection['face_crops'][0]
                bx, by, bw, bh = inspection['face_boxes'][0]
                margin = int(max(bw, bh) * AUTO_CROP_MARGIN_RATIO + .5)
                expected_left, expected_top = max(0, bx - margin), max(0, by - margin)
                self.assertGreaterEqual(bw, 200)
                self.assertEqual(cy, 0)
                self.assertEqual([cx, cy, cw, ch], [
                    expected_left, expected_top,
                    min(inspection['width'], bx + bw + margin) - expected_left,
                    min(inspection['height'], by + bh + margin) - expected_top,
                ])

    def test_picture_crop_preparation_keeps_masks_and_preview_bound_to_selection(self):
        rectangle = [38, 20, 38, 48]
        mask = png(np.zeros((48, 38), np.uint8))
        response = asyncio.run(controller.make_preview(
            self.upload(self.contents), self.upload(mask), self.upload(mask), radius=1,
            input_review='picture_upload', picture_crop=json.dumps(rectangle)))
        self.assertEqual((response['width'], response['height']), (38, 48))
        self.assertEqual(response['metadata']['original_upload_sha256'], controller.digest(self.contents))
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(response['bundle'].split(',', 1)[1]))) as archive:
            self.assertEqual(archive.read('uploaded-original.bin'), self.contents)
            self.assertEqual(controller.digest(archive.read('crop-input.png')), response['metadata']['crop_input_sha256'])
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            with self.assertRaises(HTTPException) as error:
                self.restore(input_review='picture_upload', picture_crop='[39,20,38,48]', preparation_mode='manual',
                             protect_mask=self.upload(mask), apply_mask=self.upload(mask),
                             preview_id=response['preview_id'], confirmed=True)
        self.assertEqual(error.exception.status_code, 422)

    def test_manual_unconfirmed_or_stale_preview_rejects_before_worker(self):
        protect = np.zeros(self.rgb.shape[:2], np.uint8)
        apply = np.zeros(self.rgb.shape[:2], np.uint8); apply[10:80, 15:100] = 255
        preview, _ = build_preview(self.contents, png(protect), png(apply), nominal_radius=1)
        self.assertFalse(preview['metadata']['no_effect'])
        def manual(identity, confirmed=True):
            return self.restore(preparation_mode='manual', protect_mask=self.upload(png(protect)),
                                apply_mask=self.upload(png(apply)), preview_id=identity, confirmed=confirmed)
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            for identity, confirmed in [('stale', True), (preview['preview_id'], False)]:
                with self.subTest(identity=identity, confirmed=confirmed):
                    with self.assertRaises(HTTPException) as error:
                        manual(identity, confirmed)
                    self.assertEqual(error.exception.status_code, 422)
            changed_apply = apply.copy(); changed_apply[85:90, 110:120] = 255
            with self.assertRaises(HTTPException) as error:
                self.restore(preparation_mode='manual', protect_mask=self.upload(png(protect)),
                             apply_mask=self.upload(png(changed_apply)), preview_id=preview['preview_id'], confirmed=True)
            self.assertEqual(error.exception.status_code, 422)

    def test_protect_only_selects_surroundings_and_preserves_protected_area(self):
        protect = np.zeros(self.rgb.shape[:2], np.uint8); protect[32:64, 40:88] = 255
        apply = np.zeros_like(protect)
        preview, payload = build_preview(self.contents, png(protect), png(apply), nominal_radius=3)
        def array(name):
            with Image.open(io.BytesIO(payload[name])) as image:
                return np.asarray(image).copy()
        derived = array('derived-native.png')
        submitted = array('Apply-native.png')
        resolved = array('resolved-Apply-native.png')
        guarded = array('effective-Protect-native.png') > 0
        effective = array('effective-Apply-native.png') > 0
        self.assertFalse(submitted.any())
        self.assertTrue(np.array_equal(resolved > 0, protect == 0))
        self.assertEqual(preview['metadata']['Apply_selection_mode'], 'automatic_surroundings')
        self.assertTrue(np.array_equal(derived[guarded], self.rgb[guarded]))
        self.assertTrue(np.any(derived[effective] != self.rgb[effective]))
        self.assertFalse(np.any(effective & guarded))
        rebuilt, again = build_preview(self.contents, png(protect), png(apply), nominal_radius=3)
        self.assertEqual(preview['preview_id'], rebuilt['preview_id'])
        self.assertEqual(payload['derived-native.png'], again['derived-native.png'])
        self.assertEqual(preview['metadata']['preprocessing_neural_optimizer_training_VM_calls'], 0)
        def fake_forward(selected, model):
            self.assertEqual(selected, payload['derived-native.png'])
            return None, png(np.zeros((256, 256, 3), dtype=np.uint8)), {'fake': True}
        with patch.object(controller, 'one_forward', side_effect=fake_forward) as worker:
            result = self.restore(preparation_mode='manual', protect_mask=self.upload(png(protect)),
                                  apply_mask=self.upload(png(apply)), radius=3,
                                  preview_id=preview['preview_id'], confirmed=True)
        worker.assert_called_once()
        self.assertEqual(result['metadata']['preparation']['preview_id'], preview['preview_id'])
        self.assertEqual(result['metadata']['preparation']['Apply_selection_mode'], 'automatic_surroundings')
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(result['bundle'].split(',', 1)[1]))) as archive:
            self.assertEqual(archive.read('Apply-native.png'), payload['Apply-native.png'])
            self.assertEqual(archive.read('resolved-Apply-native.png'), payload['resolved-Apply-native.png'])
            self.assertEqual(archive.read('derived-native.png'), payload['derived-native.png'])

    def test_apply_only_preview_and_restore_use_reviewed_blur(self):
        protect = np.zeros(self.rgb.shape[:2], np.uint8)
        apply = np.zeros_like(protect); apply[5:90, 5:90] = 255
        preview, payload = build_preview(self.contents, png(protect), png(apply), nominal_radius=3)
        with Image.open(io.BytesIO(payload['derived-native.png'])) as image:
            derived = np.asarray(image).copy()
        self.assertEqual(preview['metadata']['Apply_selection_mode'], 'manual')
        self.assertTrue(np.array_equal(derived[apply == 0], self.rgb[apply == 0]))
        self.assertTrue(np.any(derived[apply > 0] != self.rgb[apply > 0]))
        def fake_forward(selected, model):
            self.assertEqual(selected, payload['derived-native.png'])
            return None, png(np.zeros((256, 256, 3), dtype=np.uint8)), {'fake': True}
        with patch.object(controller, 'one_forward', side_effect=fake_forward) as worker:
            result = self.restore(preparation_mode='manual', protect_mask=self.upload(png(protect)),
                                  apply_mask=self.upload(png(apply)), radius=3,
                                  preview_id=preview['preview_id'], confirmed=True)
        worker.assert_called_once()
        self.assertEqual(result['metadata']['preparation']['preview_id'], preview['preview_id'])
        self.assertEqual(result['metadata']['preparation']['Apply_selection_mode'], 'manual')
        self.assertTrue(result['metadata']['preparation_applied'])
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(result['bundle'].split(',', 1)[1]))) as archive:
            self.assertEqual(archive.read('Apply-native.png'), payload['Apply-native.png'])
            self.assertEqual(archive.read('derived-native.png'), payload['derived-native.png'])

    def test_empty_masks_and_zero_radius_leave_image_unchanged(self):
        empty = np.zeros(self.rgb.shape[:2], np.uint8)
        preview, payload = build_preview(self.contents, png(empty), png(empty), nominal_radius=3)
        self.assertTrue(preview['metadata']['no_effect'])
        with Image.open(io.BytesIO(payload['derived-native.png'])) as image:
            self.assertTrue(np.array_equal(np.asarray(image), self.rgb))
        protect = empty.copy(); protect[32:64, 40:88] = 255
        preview, payload = build_preview(self.contents, png(protect), png(empty), nominal_radius=0)
        self.assertTrue(preview['metadata']['no_effect'])
        with Image.open(io.BytesIO(payload['derived-native.png'])) as image:
            self.assertTrue(np.array_equal(np.asarray(image), self.rgb))

    def test_blur_strength_changes_preview_and_keeps_protected_pixels(self):
        protect = np.zeros(self.rgb.shape[:2], np.uint8); protect[32:64, 40:88] = 255
        apply = np.full_like(protect, 255)
        outputs = []
        for radius in (0, 1, 4, 12):
            preview, payload = build_preview(self.contents, png(protect), png(apply), nominal_radius=radius)
            with Image.open(io.BytesIO(payload['derived-native.png'])) as image:
                derived = np.asarray(image).copy()
            self.assertEqual(preview['metadata']['algorithm']['nominal_Gaussian_radius_at256'], radius)
            self.assertTrue(np.array_equal(derived[protect > 0], self.rgb[protect > 0]))
            self.assertEqual(preview['metadata']['no_effect'], radius == 0)
            outputs.append(derived)
        self.assertTrue(np.array_equal(outputs[0], self.rgb))
        self.assertTrue(all(not np.array_equal(a, b) for a, b in zip(outputs, outputs[1:])))

    def test_explicit_apply_limits_blur_and_protect_wins_overlap(self):
        protect = np.zeros(self.rgb.shape[:2], np.uint8); protect[32:64, 40:88] = 255
        apply = np.zeros_like(protect); apply[5:90, 5:110] = 255
        preview, payload = build_preview(self.contents, png(protect), png(apply), nominal_radius=3)
        with Image.open(io.BytesIO(payload['derived-native.png'])) as image:
            derived = np.asarray(image).copy()
        with Image.open(io.BytesIO(payload['effective-Protect-native.png'])) as image:
            guarded = np.asarray(image) > 0
        self.assertEqual(preview['metadata']['Apply_selection_mode'], 'manual')
        self.assertTrue(np.array_equal(derived[guarded], self.rgb[guarded]))
        self.assertTrue(np.array_equal(derived[apply == 0], self.rgb[apply == 0]))
        self.assertTrue(np.any(derived[(apply > 0) & ~guarded] != self.rgb[(apply > 0) & ~guarded]))

    def test_manual_preparation_preserves_protect_and_outside_apply(self):
        protect = np.zeros(self.rgb.shape[:2], np.uint8); protect[32:64, 40:88] = 255
        apply = np.zeros(self.rgb.shape[:2], np.uint8); apply[:, :24] = 255
        preview, payloads = build_preview(self.contents, png(protect), png(apply), nominal_radius=3)
        with Image.open(io.BytesIO(payloads['derived-native.png'])) as image:
            derived = np.asarray(image)
        self.assertTrue(np.array_equal(derived[protect > 0], self.rgb[protect > 0]))
        self.assertTrue(np.array_equal(derived[apply == 0], self.rgb[apply == 0]))
        alpha = np.load(io.BytesIO(payloads['alpha-float32.npy']))
        transition = (alpha > 0) & (alpha < 1)
        self.assertTrue(transition.any())
        with Image.open(io.BytesIO(payloads['protection-overlay.png'])) as image:
            expected_blue = np.rint(self.rgb[transition] * .60 + np.array([55, 125, 220]) * .40)
            np.testing.assert_array_equal(np.asarray(image)[transition], expected_blue)
        self.assertEqual(preview['metadata']['preprocessing_neural_optimizer_training_VM_calls'], 0)
        rebuilt, again = build_preview(self.contents, png(protect), png(apply), nominal_radius=3)
        self.assertEqual(preview['preview_id'], rebuilt['preview_id'])
        self.assertEqual(payloads['derived-native.png'], again['derived-native.png'])

    def test_picture_preparation_uses_exact_painted_pixels(self):
        protect = np.zeros(self.rgb.shape[:2], np.uint8)
        apply = np.zeros_like(protect)
        protect[25:40, 0:12] = 255
        apply[24:41, 0:13] = 255

        def prepare(blue, radius=6):
            return asyncio.run(controller.make_preview(
                self.upload(self.contents), self.upload(png(protect)), self.upload(png(blue)),
                radius=radius, input_review='picture_upload'))

        response = prepare(apply)
        effective = (apply != 0) & (protect == 0)
        blurred = np.asarray(Image.fromarray(self.rgb).filter(ImageFilter.GaussianBlur(radius=3)))
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(response['bundle'].split(',', 1)[1]))) as archive:
            alpha = np.load(io.BytesIO(archive.read('alpha-float32.npy')))
            np.testing.assert_array_equal(alpha, effective.astype(np.float32))
            prepared_bytes = archive.read('derived-native.png')
            with Image.open(io.BytesIO(prepared_bytes)) as image:
                derived = np.asarray(image)
                np.testing.assert_array_equal(derived[~effective], self.rgb[~effective])
                np.testing.assert_array_equal(derived[effective], blurred[effective])

        empty = np.zeros_like(apply)
        surroundings = prepare(empty)
        self.assertEqual(surroundings['metadata']['Apply_selection_mode'], 'automatic_surroundings')
        self.assertEqual(surroundings['metadata']['algorithm']['nominal_Gaussian_radius_at256'], 6)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(surroundings['bundle'].split(',', 1)[1]))) as archive:
            with Image.open(io.BytesIO(archive.read('derived-native.png'))) as image:
                derived = np.asarray(image)
                np.testing.assert_array_equal(derived[protect != 0], self.rgb[protect != 0])
                np.testing.assert_array_equal(derived[protect == 0], blurred[protect == 0])
        self.assertTrue(prepare(empty, radius=0)['metadata']['no_effect'])
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            with self.assertRaises(HTTPException) as error:
                self.restore(input_review='usable', preparation_mode='manual', radius=6,
                             protect_mask=self.upload(png(protect)), apply_mask=self.upload(png(apply)),
                             preview_id=response['preview_id'], confirmed=True)
        self.assertEqual(error.exception.status_code, 422)
        with patch.object(controller, 'one_forward', return_value=(None, png(np.zeros((256, 256, 3), np.uint8)), {})) as worker:
            self.restore(input_review='picture_upload', preparation_mode='manual', radius=6,
                         protect_mask=self.upload(png(protect)), apply_mask=self.upload(png(apply)),
                         preview_id=response['preview_id'], confirmed=True)
        worker.assert_called_once_with(prepared_bytes, 'v3')

    def test_auto_blur_uses_full_upload_and_preserves_the_already_blurry_region(self):
        stripes = np.tile(np.array([30] * 32 + [220] * 32, np.uint8), 6)
        sharp = Image.fromarray(np.tile(stripes, (384, 1))).convert('RGB')
        original = np.asarray(sharp).copy()
        original[256:] = np.asarray(sharp.filter(ImageFilter.GaussianBlur(5)))[256:]

        def preview(source, protect, rectangle='', blue=None):
            if blue is None:
                blue = np.zeros_like(protect)
            return asyncio.run(controller.make_preview(
                self.upload(png(source)), self.upload(png(protect)), self.upload(png(blue)), radius=1,
                input_review='picture_upload', picture_crop=rectangle, match_blur=True))

        protect = np.zeros((384, 384), np.uint8)
        protect[10:70, 30:100] = 255
        protect[300:350, 30:100] = 255
        full = preview(original, protect)
        matching = full['metadata']['algorithm']['blur_matching']
        self.assertEqual(matching['scope'], 'full_upload_including_protect')
        self.assertGreater(matching['target_radius_at256'], 2)
        with Image.open(io.BytesIO(base64.b64decode(full['prepared_input'].split(',', 1)[1]))) as image:
            derived = np.asarray(image)
            np.testing.assert_array_equal(derived[protect != 0], original[protect != 0])
            np.testing.assert_array_equal(derived[300:], original[300:])
            self.assertTrue(np.any(derived[80:160] != original[80:160]))

        crop = '[0,0,384,256]'
        cropped = preview(original, protect[:256], crop)
        alternate = preview(np.asarray(sharp), protect[:256], crop)
        cropped_matching = cropped['metadata']['algorithm']['blur_matching']
        for key in ('scope', 'analysis_size', 'reliable_tiles', 'target_radius_at256'):
            self.assertEqual(cropped_matching[key], matching[key])
        self.assertNotEqual(cropped['preview_id'], alternate['preview_id'])
        self.assertTrue(alternate['metadata']['no_effect'])
        with patch.object(controller, 'one_forward', return_value=(None, png(np.zeros((256, 256, 3), np.uint8)), {})) as worker:
            self.restore(file=self.upload(png(original)), input_review='picture_upload', picture_crop=crop,
                         preparation_mode='manual', match_blur=True, radius=1, confirmed=True,
                         protect_mask=self.upload(png(protect[:256])), apply_mask=self.upload(png(np.zeros_like(protect[:256]))),
                         preview_id=cropped['preview_id'])
        worker.assert_called_once_with(base64.b64decode(cropped['prepared_input'].split(',', 1)[1]), 'v3')

        flat = preview(np.zeros_like(original), protect)
        self.assertEqual(flat['metadata']['algorithm']['blur_matching']['status'], 'insufficient_edges')
        self.assertTrue(flat['metadata']['no_effect'])
        manual = preview(original, protect, blue=np.full_like(protect, 255))
        self.assertFalse(manual['metadata']['automatic_detection_or_adaptive_matching'])
        self.assertEqual(manual['metadata']['algorithm']['nominal_Gaussian_radius_at256'], 1)

    def test_auto_blur_interpolates_a_gap_without_reblurring_matched_tiles(self):
        rgb = np.tile(np.repeat(self.rgb, 2, axis=0), (1, 2, 1))
        effective = np.ones(rgb.shape[:2], bool)
        effective[70:120, 110:145] = False
        tiles = np.array([[1, np.nan, 1], [1, np.nan, 1], [5, 5, 5]], np.float32)
        with patch.object(blur_match, 'estimate_tiles', return_value=tiles):
            prepared, applied, summary = blur_match.match_surroundings(rgb, rgb, None, effective)
        self.assertEqual(summary['inferred_tiles'], 2)
        self.assertEqual(summary['reliable_tiles'], 7)
        self.assertEqual(summary['target_radius_at256'], 5)
        self.assertTrue(applied[50, 128])
        self.assertTrue(np.any(prepared[40:60, 120:136] != rgb[40:60, 120:136]))
        np.testing.assert_array_equal(prepared[~effective], rgb[~effective])
        np.testing.assert_array_equal(prepared[128:], rgb[128:])

    def test_auto_setting_restores_directly_when_detection_is_uncertain(self):
        output = png(np.zeros((256, 256, 3), np.uint8))
        with patch.object(capture_api, 'detect_faces', return_value=[]), patch.object(controller, 'one_forward', return_value=(None, output, {})) as worker:
            result = self.restore(restoration='auto', preparation_mode='automatic', input_review='picture_upload')
        worker.assert_called_once_with(self.contents, 'v3')
        self.assertFalse(result['metadata']['preparation_applied'])
        self.assertEqual(result['metadata']['automatic_preparation']['route'], 'direct')
        self.assertEqual(result['metadata']['neural_forward_returns'], 1)

    def test_auto_setting_retains_original_and_uses_generated_protection(self):
        from implementation import automatic_preparation
        face = [40, 20, 40, 50]
        decision = {'route': 'protect_auto', 'reason': 'face_blur_exceeds_surroundings'}
        self.rgb[14:64, 40:80] = np.asarray(Image.fromarray(self.rgb[14:64, 40:80]).filter(ImageFilter.GaussianBlur(6)))
        self.contents = png(self.rgb)
        derived = self.rgb.copy()
        derived[:10] = 0
        observed_masks = []
        def fake_matching(rgb, source, rectangle, effective):
            np.testing.assert_array_equal(source, self.rgb)
            self.assertIsNone(rectangle)
            # Face center stays protected, but neck and rectangular lower
            # corners must be eligible for automatic surrounding blur.
            self.assertFalse(effective[40, 60])
            self.assertTrue(effective[68, 60])
            self.assertTrue(effective[68, 42])
            self.assertTrue(effective[-1].all())
            observed_masks.append(effective.copy())
            return derived, effective, {'status': 'estimated', 'target_radius_at256': 3}
        with patch.object(automatic_preparation, 'picture_pixelation', return_value=(None, None, {})), patch.object(capture_api, 'detect_faces', return_value=[face]), patch.object(automatic_preparation, 'decide', return_value=decision), patch('implementation.preparation.match_surroundings', side_effect=fake_matching), patch.object(controller, 'one_forward', return_value=(None, png(np.zeros((256, 256, 3), np.uint8)), {})) as worker:
            result = self.restore(restoration='auto', preparation_mode='automatic', input_review='picture_upload')
        worker.assert_called_once_with(png(derived), 'v3')
        self.assertTrue(result['metadata']['preparation_applied'])
        self.assertEqual(result['metadata']['automatic_preparation']['route'], 'protect_auto')
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(result['bundle'].split(',', 1)[1]))) as archive:
            self.assertEqual(archive.read('uploaded-original.bin'), self.contents)
            self.assertEqual(archive.read('derived-native.png'), png(derived))
            mask = np.asarray(Image.open(io.BytesIO(archive.read('Protect-native.png')))) != 0
            np.testing.assert_array_equal(mask, ~observed_masks[0])
            self.assertEqual(result['metadata']['automatic_preparation']['protection_shape'], 'blur-supported-component-v1')

    def test_auto_protection_is_consistent_after_cropping_and_blurs_below_face(self):
        from implementation import automatic_preparation
        source = np.random.default_rng(7).integers(0, 256, (523, 455, 3), dtype=np.uint8)
        source[62:458, 76:370] = np.asarray(Image.fromarray(source[62:458, 76:370]).filter(ImageFilter.GaussianBlur(12)))
        source[200:230, 180:210] = np.random.default_rng(8).integers(0, 256, (30, 30, 3), dtype=np.uint8)
        face, rectangle = [76, 99, 294, 408], [35, 58, 376, 465]
        x, y, width, height = rectangle
        native = source[y:y + height, x:x + width]
        def matching(rgb, full, crop, effective):
            return np.where(effective[..., None], 255 - rgb, rgb), effective, {'status': 'estimated'}
        decision = {'route': 'protect_auto', 'reason': 'face_blur_exceeds_surroundings'}
        with patch.object(automatic_preparation, 'decide', return_value=decision), patch('implementation.preparation.match_surroundings', side_effect=matching):
            _, _, full_evidence, _ = automatic_preparation.prepare(source, source, face, None)
            _, metadata, evidence, _ = automatic_preparation.prepare(native, source, face, rectangle)
        decode = lambda name: np.asarray(Image.open(io.BytesIO(evidence[name])))
        mask = decode('Protect-native.png') != 0
        self.assertTrue(mask[180, 180])
        self.assertFalse(mask[0, 180])  # Sharp hair above the blur patch.
        self.assertFalse(mask[157, 160])  # Clear detail inside the face is excluded.
        full_mask = np.asarray(Image.open(io.BytesIO(full_evidence['Protect-native.png']))) != 0
        np.testing.assert_array_equal(mask, full_mask[y:y + height, x:x + width])
        prepared = decode('derived-native.png')
        np.testing.assert_array_equal(prepared[mask], native[mask])
        # The bottom strip was previously partly protected by the detector
        # rectangle. Its textured neck/clothing must now receive added blur.
        self.assertFalse(mask[-40:].any())
        self.assertTrue((decode('effective-Apply-native.png')[-40:] > 0).all())
        self.assertGreater(np.any(prepared[-40:] != native[-40:], axis=2).mean(), .95)
        self.assertTrue(metadata['every_protected_native_RGB_exact'])

    def test_video_auto_uses_verified_capture_context(self):
        from implementation import automatic_preparation
        self.make_video_crop()
        with patch.object(automatic_preparation, 'prepare', return_value=(None, None, {}, {'route': 'direct', 'reason': 'insufficient_edges'})) as prepare, patch.object(capture_api, 'detect_faces', return_value=[[8, 8, 24, 20]]), patch.object(controller, 'one_forward', return_value=(None, png(np.zeros((256, 256, 3), np.uint8)), {})):
            status, result = self.video_restore(self.capture_owner, restoration='auto', preparation_mode='automatic')
        self.assertEqual(status, 200, result)
        self.assertEqual(result['metadata']['automatic_preparation']['route'], 'direct')
        source = prepare.call_args.args[1]
        np.testing.assert_array_equal(source, self.rgb)
        self.assertEqual(prepare.call_args.args[3], [4, 5, 40, 32])

    def test_video_scanline_cleanup_uses_full_frame_before_extracting_crop(self):
        from implementation.automatic_preparation import prepare_source

        y, x = np.indices(self.rgb.shape[:2])
        values = 125 + 20 * np.sin(x / 18) + 26 * np.cos(2 * np.pi * y / 5)
        source = np.repeat(np.rint(values)[..., None].astype(np.uint8), 3, axis=2)
        self.contents = png(source)
        prepared, report, _ = prepare_source(source)
        self.make_video_crop()
        output = png(np.zeros((256, 256, 3), np.uint8))
        with patch.object(controller, 'one_forward', return_value=(None, output, {})) as worker:
            status, result = self.video_restore(self.capture_owner, restoration='auto', preparation_mode='automatic')
        self.assertEqual(status, 200)
        self.assertEqual(result['metadata']['automatic_preparation']['route'], 'scanline_cleanup')
        self.assertNotIn('Gaussian_radius_at256', result['metadata']['scanline_cleanup'])
        self.assertIn('comb_ela_masked_pixels', result['metadata']['scanline_cleanup'])
        self.assertIn('chroma_cleanup_pixels', result['metadata']['scanline_cleanup'])
        self.assertEqual(result['metadata']['scanline_cleanup']['full_image_sha256'], report['full_image_sha256'])
        self.assertEqual(base64.b64decode(result['scanline_cleaned_full_image'].split(',', 1)[1]), png(prepared))
        worker.assert_called_once_with(png(prepared[5:37, 4:44]), 'v3')

    def test_video_auto_keeps_scanline_mosaic_mask_with_frozen_crop(self):
        from implementation import automatic_preparation

        self.make_video_crop()
        prepared = self.rgb // 2
        mask = png(np.full(self.rgb.shape[:2], 255, np.uint8))
        report = {'spacing_native_pixels': 5., 'mosaic': {'radius_at256': 2.}}
        evidence = {'scanline-cleaned.png': png(prepared), 'scanline-mosaic-mask.png': mask}
        output = png(np.zeros((256, 256, 3), np.uint8))
        with (
            patch.object(automatic_preparation, 'prepare_source', return_value=(prepared, report, evidence)),
            patch.object(controller, 'one_forward', return_value=(None, output, {})) as worker,
        ):
            status, result = self.video_restore(self.capture_owner, restoration='auto', preparation_mode='automatic')
        self.assertEqual(status, 200)
        auto = result['metadata']['automatic_preparation']
        self.assertEqual(auto['route'], 'scanline_mosaic_coordinated')
        self.assertTrue(auto['uniform_crop_smoothing'])
        expected_radius = automatic_preparation.compute_uniform_crop_gaussian_radius(
            prepared[5:37, 4:44], None, report, source_size=(self.rgb.shape[1], self.rgb.shape[0])
        )
        expected_crop = automatic_preparation.apply_uniform_crop_gaussian(prepared[5:37, 4:44], expected_radius)
        worker.assert_called_once_with(png(expected_crop), 'v3')
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(result['bundle'].split(',', 1)[1]))) as archive:
            self.assertEqual(archive.read('capture/frame-scanline-mosaic-mask.png'), mask)

    def test_video_auto_reuses_verified_frame_scan_for_queued_crops(self):
        from implementation import automatic_preparation

        self.make_video_crop()
        frame_id = capture_api.CROPS[self.capture_id]['frame']
        second_rectangle = capture_api.Rectangle(x=50, y=18, width=30, height=32)
        second = capture_api._crop(frame_id, second_rectangle, self.capture_owner)
        second_bytes = base64.b64decode(second['image'].split(',', 1)[1])
        smoothed = 255 - self.rgb
        full_png = png(smoothed)
        mask_png = png(np.full(self.rgb.shape[:2], 255, dtype=np.uint8))
        report = {'native_size': [128, 96], 'Gaussian_radius_at256': 3.0}
        evidence = {'full-image-smoothed.png': full_png, 'full-image-pixelation-mask.png': mask_png}
        output = png(np.zeros((256, 256, 3), np.uint8))
        controller._frozen_frame_pixelation.cache_clear()
        self.addCleanup(controller._frozen_frame_pixelation.cache_clear)
        with patch.object(automatic_preparation, 'picture_pixelation', return_value=(smoothed, report, evidence)) as scan, \
             patch.object(controller, 'one_forward', return_value=(None, output, {})) as worker:
            status, first_result = self.video_restore(self.capture_owner, restoration='auto', preparation_mode='automatic')
            second_status, second_result = self.video_restore(self.capture_owner, contents=second_bytes,
                capture_id=second['capture_id'], restoration='auto', preparation_mode='automatic', include_smoothed_frame='false')
        self.assertEqual((status, second_status), (200, 200))
        scan.assert_called_once()
        np.testing.assert_array_equal(scan.call_args.args[0], self.rgb)
        crop1 = self.rgb[5:37, 4:44]
        crop2 = self.rgb[18:50, 50:80]
        r1 = automatic_preparation.compute_uniform_crop_gaussian_radius(crop1, None, report, source_size=(128, 96))
        r2 = automatic_preparation.compute_uniform_crop_gaussian_radius(crop2, None, report, source_size=(128, 96))
        self.assertEqual([call.args for call in worker.call_args_list],
                         [(png(automatic_preparation.apply_uniform_crop_gaussian(crop1, r1)), 'v3'),
                          (png(automatic_preparation.apply_uniform_crop_gaussian(crop2, r2)), 'v3')])
        self.assertEqual(first_result['metadata']['automatic_preparation']['route'], 'frame_region_gaussian')
        self.assertTrue(first_result['metadata']['automatic_preparation']['uniform_crop_smoothing'])
        self.assertTrue(second_result['metadata']['automatic_preparation']['uniform_crop_smoothing'])
        self.assertEqual(first_result['metadata']['frame_pixelation'], second_result['metadata']['frame_pixelation'])
        self.assertEqual(base64.b64decode(first_result['smoothed_full_frame'].split(',', 1)[1]), full_png)
        self.assertNotIn('smoothed_full_frame', second_result)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(second_result['bundle'].split(',', 1)[1]))) as archive:
            self.assertEqual(archive.read('capture/frame.png'), self.contents)
            self.assertEqual(archive.read('capture/frame-pixelation-mask.png'), mask_png)

    def test_auto_detection_copy_never_replaces_the_original_model_input(self):
        from implementation import automatic_preparation
        output = png(np.zeros((256, 256, 3), np.uint8))
        with patch.object(automatic_preparation, 'picture_pixelation', return_value=(None, None, {})), patch.object(capture_api, 'detect_faces', side_effect=[[], [], [[20, 10, 80, 70]]]), patch.object(automatic_preparation, 'pixelation_signal', return_value={'suspected': True}), patch.object(automatic_preparation, 'prepare', return_value=(None, None, {}, {'route': 'direct'})) as prepare, patch.object(controller, 'one_forward', return_value=(None, output, {})) as worker:
            result = self.restore(restoration='auto', preparation_mode='automatic', input_review='picture_upload')
        np.testing.assert_array_equal(prepare.call_args.args[1], self.rgb)
        worker.assert_called_once_with(self.contents, 'v3')
        self.assertEqual(result['metadata']['automatic_preparation']['face_detection'], 'smoothed_detection_copy_with_block_signal')

    def test_picture_auto_delivers_jpeg_preparation_and_runs_v2_once(self):
        from implementation import automatic_preparation, jpeg_deblocking

        deblocked = self.rgb.copy()
        deblocked[:, :, 0] = 90
        report = {'preparation_model': 'FBCNN RGB', 'preparation_forward_returns': 1}
        regional = {'native_size': [128, 96], 'Gaussian_radius_at256': 3.0}
        full = png(self.rgb // 2)
        artifacts = {'full-image-smoothed.png': full}
        output = png(np.zeros((256, 256, 3), np.uint8))
        with (
            patch.object(automatic_preparation, 'prepare_source', return_value=(self.rgb // 2, regional, artifacts)),
            patch.object(jpeg_deblocking, 'prepare_jpeg', return_value=(deblocked, report)),
            patch.object(controller, 'one_forward', return_value=(None, output, {})) as worker,
        ):
            result = self.restore(restoration='auto', preparation_mode='automatic', input_review='picture_upload')
        worker.assert_called_once_with(png(deblocked), 'v3')
        self.assertEqual(result['metadata']['automatic_preparation']['route'], 'jpeg_deblocking')
        self.assertEqual(result['metadata']['jpeg_deblocking'], report)
        self.assertEqual(result['metadata']['neural_forward_returns'], 1)
        self.assertEqual(base64.b64decode(result['smoothed_full_picture'].split(',', 1)[1]), full)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(result['bundle'].split(',', 1)[1]))) as archive:
            self.assertEqual(archive.read('uploaded-original.bin'), self.contents)
            self.assertEqual(archive.read('derived-native.png'), png(deblocked))

    def test_picture_auto_uses_hybrid_bytes_and_skips_separate_jpeg_pass(self):
        from implementation import automatic_preparation, jpeg_deblocking, mixed_jpeg

        combined = self.rgb // 3
        combined_png = png(combined)
        mosaic_mask = np.zeros(self.rgb.shape[:2], dtype=bool)
        mosaic_mask[16:80, 24:104] = True
        mask_png = png(mosaic_mask.astype(np.uint8) * 255)
        report = {
            'mixed_jpeg_mosaic': True,
            'mosaic_grid_period_native': 4,
            'outside_deblock_fraction': .35,
            'native_Gaussian_radius': 1.0,
            'preparation_forward_returns': 1,
        }
        evidence = {
            'full-image-smoothed.png': combined_png,
            'full-image-pixelation-mask.png': mask_png,
        }
        restored = png(np.zeros((256, 256, 3), np.uint8))
        with (
            patch.object(automatic_preparation, 'prepare_source', return_value=(None, None, {})),
            patch.object(mixed_jpeg, 'prepare_mixed_jpeg', return_value=(combined, report, evidence)),
            patch.object(jpeg_deblocking, 'prepare_jpeg') as standalone,
            patch.object(jpeg_deblocking, 'run_deblock', side_effect=AssertionError('duplicate FBCNN call forbidden')) as dup_deblock,
            patch.object(controller, 'one_forward', return_value=(None, restored, {})) as worker,
        ):
            result = self.restore(restoration='auto', preparation_mode='automatic', input_review='picture_upload')
        standalone.assert_not_called()
        dup_deblock.assert_not_called()
        expected_base = combined.copy()
        expected_base[mosaic_mask] = self.rgb[mosaic_mask]
        expected_uniform = automatic_preparation.apply_uniform_crop_gaussian(expected_base, 1.0)
        worker.assert_called_once_with(png(expected_uniform), 'v3')
        self.assertEqual(result['metadata']['automatic_preparation']['route'], 'jpeg_mosaic_coordinated')
        self.assertTrue(result['metadata']['automatic_preparation']['uniform_crop_smoothing'])
        self.assertAlmostEqual(result['metadata']['automatic_preparation']['crop_Gaussian_radius_native'], 1.0, places=4)
        self.assertEqual(result['metadata']['preparation']['preparation_forward_returns'], 1)
        self.assertEqual(base64.b64decode(result['smoothed_full_picture'].split(',', 1)[1]), combined_png)

    def test_softened_mosaic_uses_direct_v2_and_retains_the_diagnostic_mask(self):
        from implementation import automatic_preparation

        mask = png(np.full(self.rgb.shape[:2], 255, np.uint8))
        report = {'policy': 'softened-mosaic-spectrum-v1', 'native_size': [128, 96]}
        output = png(np.zeros((256, 256, 3), np.uint8))
        with (
            patch.object(automatic_preparation, 'prepare_source',
                         return_value=(None, report, {'softened-mosaic-mask.png': mask})),
            patch.object(capture_api, 'detect_faces', side_effect=AssertionError('Skip face preparation')),
            patch.object(controller, 'one_forward', return_value=(None, output, {})) as worker,
        ):
            result = self.restore(restoration='auto', preparation_mode='automatic', input_review='picture_upload')
        worker.assert_called_once_with(self.contents, 'v3')
        self.assertFalse(result['metadata']['preparation_applied'])
        self.assertEqual(result['metadata']['automatic_preparation'],
                         {'route': 'direct', 'reason': 'suspected_softened_mosaic'})
        self.assertTrue(result['metadata']['softened_mosaic_detection']['crop_intersects'])
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(result['bundle'].split(',', 1)[1]))) as archive:
            self.assertEqual(archive.read('softened-mosaic-mask.png'), mask)

    def test_picture_auto_exports_full_frame_but_restores_only_selected_crop(self):
        from implementation import automatic_preparation

        rng = np.random.default_rng(28)
        source = rng.integers(0, 256, (384, 420, 3), dtype=np.uint8)
        expected = np.zeros(source.shape[:2], bool)
        for x, y, width, height in ((42, 50, 143, 128), (249, 196, 145, 150)):
            samples = rng.integers(0, 256, (10, 12, 3), dtype=np.uint8)
            mosaic = Image.fromarray(samples).resize((width + 8, height + 12), Image.Resampling.NEAREST)
            source[y:y + height, x:x + width] = mosaic.resize((width, height), Image.Resampling.BILINEAR)
            expected[y:y + height, x:x + width] = True
        self.contents = png(source)
        crop = [40, 40, 160, 160]
        output = png(np.zeros((256, 256, 3), np.uint8))
        with patch.object(controller, 'one_forward', return_value=(None, output, {})) as worker:
            result = self.restore(restoration='auto', preparation_mode='automatic', input_review='picture_upload',
                                  picture_crop=json.dumps(crop))
        self.assertEqual(result['metadata']['automatic_preparation']['route'], 'picture_region_gaussian')
        self.assertEqual(result['metadata']['picture_crop'], crop)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(result['bundle'].split(',', 1)[1]))) as archive:
            self.assertEqual(archive.read('uploaded-original.bin'), self.contents)
            full = base64.b64decode(result['smoothed_full_picture'].split(',', 1)[1])
            self.assertEqual(archive.read('full-image-smoothed.png'), full)
            prepared = np.asarray(Image.open(io.BytesIO(full)))
            mask = np.asarray(Image.open(io.BytesIO(archive.read('full-image-pixelation-mask.png')))) > 0
            self.assertLess(np.count_nonzero(mask & ~expected) / np.count_nonzero(mask), .01)
            self.assertGreater(np.count_nonzero(mask & expected) / np.count_nonzero(expected), .70)
            self.assertTrue(mask[50:178, 42:185].any())
            self.assertTrue(mask[196:346, 249:394].any())
            np.testing.assert_array_equal(prepared[~mask], source[~mask])
            auto_dec = result['metadata']['automatic_preparation']
            self.assertTrue(auto_dec['uniform_crop_smoothing'])
            crop_radius = auto_dec['crop_Gaussian_radius_native']
            expected_uniform_crop = automatic_preparation.apply_uniform_crop_gaussian(source[40:200, 40:200], crop_radius)
            worker.assert_called_once_with(png(expected_uniform_crop), 'v3')
            derived_crop = np.asarray(Image.open(io.BytesIO(archive.read('derived-native.png'))))
            np.testing.assert_array_equal(derived_crop, expected_uniform_crop)
        self.assertEqual(result['metadata']['picture_pixelation']['Gaussian_radius_at256'],
                         automatic_preparation.PIXELATION_RADIUS_AT256)

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
            raw, pixels, metadata = controller.one_forward(self.contents, 'v3')
        self.assertGreater(len(raw), 0); self.assertGreater(len(pixels), 0)
        self.assertEqual(metadata['uploaded_bytes_sha256'], controller.digest(self.contents))
        self.assertEqual(controller.STATE['neural_forward_attempts'], 1)
        self.assertEqual(controller.STATE['neural_forward_returns'], 1)
        self.assertFalse(controller.STATE['active_worker']); self.assertFalse(controller.STATE['uncertain'])

    def test_delivery_padding_corruption_latches_with_actual_completed_counts(self):
        self.corrupt_padding = True
        verify, memory, worker = self.worker_patches()
        with verify, memory, worker, self.assertRaises(ValueError):
            controller.one_forward(self.contents, 'v3')
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
            controller.one_forward(self.contents, 'v3')
        self.assertTrue(controller.STATE['uncertain'])
        self.assertEqual((controller.STATE['neural_forward_attempts'], controller.STATE['neural_forward_returns']), (1, 0))

    def test_timeout_terminates_only_own_worker_and_retains_actual_status(self):
        self.worker_returncode = None
        self.worker_status = {'phase': 'forward_entered', 'forward_attempts': 1, 'forward_returns': 0}
        verify, memory, worker = self.worker_patches()
        with verify, memory, worker, patch.object(controller.time, 'monotonic', side_effect=[0, 361]):
            with self.assertRaisesRegex(RuntimeError, '360-second'):
                controller.one_forward(self.contents, 'v3')
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
                controller.one_forward(self.contents, 'v3')
        self.assertTrue(controller.STATE['uncertain'])
        self.assertEqual((controller.STATE['neural_forward_attempts'], controller.STATE['neural_forward_returns']), (1, 0))

    def test_output_diagnostics_before_native_noise_and_off_equality(self):
        rng = np.random.default_rng(19)
        y, x = np.indices((512, 512))
        base = np.stack(((x * 2) % 256, (y * 2) % 256, ((x + y)) % 256), axis=2).astype(np.float32)
        noisy = np.clip(np.rint(base + rng.normal(0.0, 14.0, base.shape)), 0, 255).astype(np.uint8)
        noisy_png = png(noisy)
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            result = self.restore(file=self.upload(noisy_png), restoration='off', input_review='picture_upload')
        meta = result['metadata']
        diag = meta['output_diagnostics']
        self.assertEqual(set(diag.keys()), {'before', 'after'})
        self.assertIn('noise_sigma_native', diag['before'])
        self.assertEqual(diag['before']['noise_sigma_native'], meta['quality_signals']['noise_sigma_native'])
        self.assertGreater(diag['before']['noise_sigma_native'], diag['before']['noise_sigma_255'] + 2.0)
        self.assertEqual(diag['before']['blur_variance'], diag['after']['blur_variance'])
        self.assertEqual(diag['before']['noise_sigma_255'], diag['after']['noise_sigma_255'])
        self.assertFalse(meta['Auto_assessed_original_canvas_before_preparation'])

    def test_output_diagnostics_includes_prepared_only_when_preparation_ran(self):
        protect = np.zeros(self.rgb.shape[:2], np.uint8)
        apply = np.full(self.rgb.shape[:2], 255, np.uint8)
        preview, payload = build_preview(self.contents, png(protect), png(apply), nominal_radius=4)
        worker_out = png(np.full((256, 256, 3), 128, dtype=np.uint8))

        with patch.object(controller, 'one_forward', return_value=(None, worker_out, {'fake': True})):
            manual_res = self.restore(
                preparation_mode='manual',
                protect_mask=self.upload(png(protect)),
                apply_mask=self.upload(png(apply)),
                radius=4,
                preview_id=preview['preview_id'],
                confirmed=True,
            )
        manual_meta = manual_res['metadata']
        manual_diag = manual_meta['output_diagnostics']
        self.assertEqual(set(manual_diag.keys()), {'before', 'prepared', 'after'})
        self.assertEqual(set(manual_diag['prepared'].keys()), {'blur_variance', 'noise_sigma_255'})
        self.assertLess(manual_diag['prepared']['blur_variance'], manual_diag['before']['blur_variance'])
        self.assertFalse(manual_meta['Auto_assessed_original_canvas_before_preparation'])

        from implementation import automatic_preparation
        with (
            patch.object(automatic_preparation, 'prepare_source', return_value=(None, None, {})),
            patch.object(capture_api, 'detect_faces', return_value=[]),
            patch.object(controller, 'one_forward', return_value=(None, worker_out, {'fake': True})),
        ):
            direct_res = self.restore(restoration='auto', preparation_mode='automatic', input_review='picture_upload')
        direct_meta = direct_res['metadata']
        self.assertFalse(direct_meta['preparation_applied'])
        self.assertEqual(set(direct_meta['output_diagnostics'].keys()), {'before', 'after'})
        self.assertTrue(direct_meta['Auto_assessed_original_canvas_before_preparation'])

        with (
            patch.object(automatic_preparation, 'prepare_source', return_value=(None, None, {})),
            patch.object(capture_api, 'detect_faces', return_value=[[20, 12, 50, 60]]),
            patch.object(automatic_preparation, 'decide', return_value={'route': 'protect_auto', 'reason': 'test'}),
            patch('implementation.preparation.match_surroundings',
                  side_effect=lambda rgb, source, rectangle, effective: (
                      np.where(effective[..., None], np.uint8(0), rgb), effective, {'status': 'estimated'})),
            patch.object(controller, 'one_forward', return_value=(None, worker_out, {'fake': True})),
        ):
            auto_prep_res = self.restore(restoration='auto', preparation_mode='automatic', input_review='picture_upload')
        auto_prep_meta = auto_prep_res['metadata']
        self.assertTrue(auto_prep_meta['preparation_applied'])
        self.assertEqual(set(auto_prep_meta['output_diagnostics'].keys()), {'before', 'prepared', 'after'})
        self.assertTrue(auto_prep_meta['Auto_assessed_original_canvas_before_preparation'])

    def post_reference(self, file_bytes, geometry, *, filename='ref.png', content_type='image/png',
                       original_input=''):
        boundary = b'ref-test-boundary'
        geometry_str = geometry if isinstance(geometry, str) else json.dumps(geometry)
        orig_part = (
            b'--' + boundary
            + b'\r\nContent-Disposition: form-data; name="original_input"\r\n\r\n'
            + original_input.encode('utf-8')
            + b'\r\n'
            if original_input else b''
        )
        body = (
            orig_part
            + b'--' + boundary
            + b'\r\nContent-Disposition: form-data; name="geometry"\r\n\r\n'
            + geometry_str.encode('utf-8')
            + b'\r\n--' + boundary
            + f'\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
              f'Content-Type: {content_type}\r\n\r\n'.encode('utf-8')
            + file_bytes
            + b'\r\n--' + boundary + b'--\r\n'
        )
        headers = {'host': '127.0.0.1', 'content-type': f'multipart/form-data; boundary={boundary.decode()}'}
        status, response = asyncio.run(asgi_request(controller.app, 'POST', '/face/restoration/reference', body, headers))
        return status, json.loads(response)

    @staticmethod
    def decode_data_url_rgb(data_url):
        raw = base64.b64decode(data_url.split(',', 1)[1])
        with Image.open(io.BytesIO(raw)) as image:
            return np.asarray(image.convert('RGB')).copy()

    def test_picture_native_compatibility_and_restore_source_size_metadata(self):
        native_only = controller.picture_native(self.contents, [10, 8, 40, 32])
        np.testing.assert_array_equal(native_only, self.rgb[8:40, 10:50])
        native_pair, source_size = controller.picture_native_with_source_size(self.contents, [10, 8, 40, 32])
        np.testing.assert_array_equal(native_pair, native_only)
        self.assertEqual(source_size, [128, 96])

        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            full_res = self.restore(restoration='off', input_review='picture_upload')
            crop_res = self.restore(restoration='off', input_review='picture_upload', picture_crop='[10,8,40,32]')
        self.assertEqual(full_res['metadata']['source_size'], [128, 96])
        self.assertNotIn('picture_crop', full_res['metadata'])
        self.assertEqual(crop_res['metadata']['source_size'], [128, 96])
        self.assertEqual(crop_res['metadata']['picture_crop'], [10, 8, 40, 32])

    def test_reference_endpoint_full_cropped_and_native_crop_match_prepare_crop_exactly(self):
        rng = np.random.default_rng(42)
        full_rgb = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        full_png = png(full_rgb)
        crop = [50, 30, 120, 160]
        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            restored = self.restore(file=self.upload(full_png), restoration='off',
                                    input_review='picture_upload', picture_crop=json.dumps(crop))
        meta = restored['metadata']
        geometry = {
            'observed_bounds_256': meta['geometry']['observed_bounds_256'],
            'native_size': meta['geometry']['native_size'],
            'picture_crop': meta['picture_crop'],
            'source_size': meta['source_size'],
        }
        expected_256 = self.decode_data_url_rgb(restored['original_input'])

        status_full, body_full = self.post_reference(full_png, geometry)
        self.assertEqual(status_full, 200, body_full)
        self.assertEqual(body_full['source_mode'], 'full_cropped')
        self.assertEqual(body_full['observed_bounds_256'], geometry['observed_bounds_256'])
        np.testing.assert_array_equal(self.decode_data_url_rgb(body_full['reference_256']), expected_256)

        clean_crop_png = png(full_rgb[30:190, 50:170])
        status_crop, body_crop = self.post_reference(clean_crop_png, geometry)
        self.assertEqual(status_crop, 200, body_crop)
        self.assertEqual(body_crop['source_mode'], 'native_crop')
        np.testing.assert_array_equal(self.decode_data_url_rgb(body_crop['reference_256']), expected_256)

        # EXIF-rotated clean crop (Orientation=6) matches upright restore input.
        upright_bytes = io.BytesIO()
        Image.fromarray(full_rgb[30:190, 50:170]).save(upright_bytes, format='PNG')
        rotated_pil = Image.fromarray(full_rgb[30:190, 50:170]).rotate(90, expand=True)
        exif = Image.Exif()
        exif[274] = 6
        exif_buf = io.BytesIO()
        rotated_pil.save(exif_buf, format='JPEG', quality=95, exif=exif)
        status_exif, body_exif = self.post_reference(exif_buf.getvalue(), geometry,
                                                     filename='exif.jpg', content_type='image/jpeg')
        self.assertEqual(status_exif, 200, body_exif)
        self.assertEqual(body_exif['source_mode'], 'native_crop')

    def test_reference_endpoint_256_padded_and_256_source_crop_routing(self):
        rng = np.random.default_rng(77)
        full_256 = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        full_256_png = png(full_256)

        # Non-square crop from a 256x256 source -> padded bounds.
        crop_rect = [40, 20, 120, 160]
        padded_256, _, _, geo_padded = prepare_crop(full_256[20:180, 40:160])
        bounds_padded = geo_padded['observed_bounds_256']
        geometry_padded = {
            'observed_bounds_256': bounds_padded,
            'native_size': [120, 160],
            'picture_crop': crop_rect,
            'source_size': [256, 256],
        }
        status_ok, body_ok = self.post_reference(png(padded_256), geometry_padded)
        self.assertEqual(status_ok, 200, body_ok)
        self.assertEqual(body_ok['source_mode'], 'padded_256')
        np.testing.assert_array_equal(self.decode_data_url_rgb(body_ok['reference_256']), padded_256)

        # Unpadded 256x256 full source matching source_size=[256, 256] slices picture_crop.
        status_slice, body_slice = self.post_reference(full_256_png, geometry_padded)
        self.assertEqual(status_slice, 200, body_slice)
        self.assertEqual(body_slice['source_mode'], 'full_cropped')
        np.testing.assert_array_equal(self.decode_data_url_rgb(body_slice['reference_256']), padded_256)

        # Unpadded 256x256 without matching 256x256 source_size+picture_crop is rejected.
        geometry_no_source = {'observed_bounds_256': bounds_padded, 'native_size': [120, 160]}
        status_bad, body_bad = self.post_reference(full_256_png, geometry_no_source)
        self.assertEqual(status_bad, 400)
        self.assertEqual(
            body_bad['detail'],
            'The 256 × 256 reference is not padded to match this crop. '
            'Upload the clean unpadded crop or a 256 × 256 padded frame.',
        )

        # Square crop (is_padded=False) from 256x256 source still slices picture_crop when source_size==[256,256].
        sq_crop = [20, 20, 180, 180]
        sq_256, _, _, _ = prepare_crop(full_256[20:200, 20:200])
        status_sq, body_sq = self.post_reference(full_256_png, {
            'observed_bounds_256': [0, 0, 256, 256],
            'native_size': [180, 180],
            'picture_crop': sq_crop,
            'source_size': [256, 256],
        })
        self.assertEqual(status_sq, 200, body_sq)
        self.assertEqual(body_sq['source_mode'], 'full_cropped')
        np.testing.assert_array_equal(self.decode_data_url_rgb(body_sq['reference_256']), sq_256)

    def test_reference_endpoint_rejects_invalid_files_transparency_and_geometry(self):
        valid_geo = {'observed_bounds_256': [0, 0, 256, 256], 'native_size': [128, 128]}

        # Transparent RGBA PNG rejected with exact message.
        rgba = np.full((128, 128, 4), 255, dtype=np.uint8)
        rgba[10, 10, 3] = 250
        buf = io.BytesIO()
        Image.fromarray(rgba, mode='RGBA').save(buf, format='PNG')
        status_alpha, body_alpha = self.post_reference(buf.getvalue(), valid_geo)
        self.assertEqual(status_alpha, 400)
        self.assertEqual(body_alpha['detail'], 'The reference must be opaque.')

        # Paletted PNG with tRNS transparency rejected.
        pal = Image.fromarray(np.zeros((128, 128), dtype=np.uint8), mode='P')
        pal_buf = io.BytesIO()
        pal.save(pal_buf, format='PNG', transparency=0)
        status_trns, body_trns = self.post_reference(pal_buf.getvalue(), valid_geo)
        self.assertEqual(status_trns, 400)
        self.assertEqual(body_trns['detail'], 'The reference must be opaque.')

        # Opaque RGBA PNG accepted.
        rgba_opaque = np.full((128, 128, 4), 255, dtype=np.uint8)
        buf_ok = io.BytesIO()
        Image.fromarray(rgba_opaque, mode='RGBA').save(buf_ok, format='PNG')
        status_opaque, body_opaque = self.post_reference(buf_ok.getvalue(), valid_geo)
        self.assertEqual(status_opaque, 200, body_opaque)

        # Oversized > 10 MiB, bad format, > 1600px side.
        status_big, _ = self.post_reference(b'x' * (controller.MAX_UPLOAD_BYTES + 1), valid_geo)
        self.assertEqual(status_big, 400)
        gif_buf = io.BytesIO()
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(gif_buf, format='GIF')
        status_gif, _ = self.post_reference(gif_buf.getvalue(), valid_geo)
        self.assertEqual(status_gif, 400)
        wide_buf = io.BytesIO()
        Image.new('RGB', (1601, 64), (100, 100, 100)).save(wide_buf, format='PNG')
        status_wide, _ = self.post_reference(wide_buf.getvalue(), {'observed_bounds_256': [0, 0, 256, 256]})
        self.assertEqual(status_wide, 400)

        # Bad bounds and mismatched aspect ratio.
        for bad_geo in ('not-json', {'observed_bounds_256': [10, 0, 10, 256]}, {'observed_bounds_256': [-1, 0, 256, 256]}):
            status_bad_geo, _ = self.post_reference(self.contents, bad_geo)
            self.assertEqual(status_bad_geo, 400)

        status_aspect, body_aspect = self.post_reference(
            png(np.zeros((100, 200, 3), dtype=np.uint8)),
            {'observed_bounds_256': [32, 0, 224, 256], 'native_size': [120, 160]},
        )
        self.assertEqual(status_aspect, 400)
        self.assertEqual(
            body_aspect['detail'],
            'Reference aspect ratio does not match the input crop. '
            'Use the full original photo, a clean crop matching the input, or a 256 × 256 padded frame.',
        )

        # Within 5% aspect ratio but prepare_crop produces different observed_bounds_256 -> rejected.
        _, _, _, base_geo = prepare_crop(np.zeros((120, 100, 3), dtype=np.uint8))
        status_drift, _ = self.post_reference(
            png(np.zeros((124, 100, 3), dtype=np.uint8)),
            {'observed_bounds_256': base_geo['observed_bounds_256'], 'native_size': [100, 120]},
        )
        self.assertEqual(status_drift, 400)

    def test_implementation_app_mounts_capture_and_completion_idempotently(self):
        route_paths_before = [getattr(r, 'path', None) for r in controller.app.routes]
        middleware_before = list(controller.app.user_middleware)
        self.assertEqual(route_paths_before.count('/capture/status'), 1)
        self.assertEqual(route_paths_before.count('/face/completion/status'), 1)

        import app as compat_app
        self.assertIs(compat_app.app, controller.app)
        route_paths_after = [getattr(r, 'path', None) for r in controller.app.routes]
        middleware_after = list(controller.app.user_middleware)
        self.assertEqual(route_paths_after.count('/capture/status'), 1)
        self.assertEqual(route_paths_after.count('/face/completion/status'), 1)
        self.assertEqual(len(middleware_after), len(middleware_before))

        with patch.object(capture_api, 'TOKENS', set()):
            cap_status, cap_body = asyncio.run(
                asgi_request(controller.app, 'GET', '/capture/status', headers={'host': '127.0.0.1'})
            )
            self.assertEqual(cap_status, 200)
            cap_json = json.loads(cap_body)
            self.assertTrue(cap_json['available'])
            self.assertTrue(cap_json.get('token'))

        comp_status, comp_body = asyncio.run(
            asgi_request(controller.app, 'GET', '/face/completion/status', headers={'host': '127.0.0.1'})
        )
        self.assertEqual(comp_status, 200)
        comp_json = json.loads(comp_body)
        self.assertEqual(comp_json['model'], 'completion_r2')

    def test_video_restore_records_capture_crop_frame_size_and_slices_full_hd_reference(self):
        rng = np.random.default_rng(1080)
        full_hd_rgb = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        full_hd_png = png(full_hd_rgb)

        self.capture_owner = 'capture-owner-hd'
        for key, value in {
            'STORE': self.root / 'capture_hd',
            'FRAMES': {},
            'CROPS': {},
            'TOKENS': {self.capture_owner},
            'detect_faces': lambda image: [],
        }.items():
            setting = patch.object(capture_api, key, value)
            setting.start()
            self.addCleanup(setting.stop)

        frame = capture_api._freeze(full_hd_png, 'video', 2.5, self.capture_owner)
        rectangle = capture_api.Rectangle(x=640, y=220, width=320, height=400)
        crop = capture_api._crop(frame['frame_id'], rectangle, self.capture_owner)
        self.capture_id = crop['capture_id']
        self.capture_contents = base64.b64decode(crop['image'].split(',', 1)[1])

        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            status, response = self.video_restore(self.capture_owner)
        self.assertEqual(status, 200, response.get('detail'))
        meta = response['metadata']
        self.assertEqual(meta['capture_crop'], [640, 220, 320, 400])
        self.assertEqual(meta['frame_size'], [1920, 1080])
        self.assertEqual(meta['source_size'], [320, 400])
        self.assertEqual(meta['source_time'], 2.5)

        expected_256 = self.decode_data_url_rgb(response['original_input'])
        ref_geo = {
            'observed_bounds_256': meta['geometry']['observed_bounds_256'],
            'native_size': meta['geometry']['native_size'],
            'picture_crop': meta['capture_crop'],
            'source_size': meta['frame_size'],
        }
        status_ref, body_ref = self.post_reference(full_hd_png, ref_geo)
        self.assertEqual(status_ref, 200, body_ref)
        self.assertEqual(body_ref['source_mode'], 'full_cropped')
        np.testing.assert_array_equal(self.decode_data_url_rgb(body_ref['reference_256']), expected_256)

    def test_video_restore_slices_scaled_1920x1080_reference_from_1280x720_frame_size(self):
        rng = np.random.default_rng(720)
        full_hd_rgb = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        full_hd_png = png(full_hd_rgb)
        hd_rgb = np.asarray(Image.fromarray(full_hd_rgb).resize((1280, 720), Image.Resampling.BILINEAR))
        hd_png = png(hd_rgb)

        self.capture_owner = 'capture-owner-720p'
        for key, value in {
            'STORE': self.root / 'capture_720p',
            'FRAMES': {},
            'CROPS': {},
            'TOKENS': {self.capture_owner},
            'detect_faces': lambda image: [],
        }.items():
            setting = patch.object(capture_api, key, value)
            setting.start()
            self.addCleanup(setting.stop)

        frame = capture_api._freeze(hd_png, 'video', 4.75, self.capture_owner)
        rectangle = capture_api.Rectangle(x=420, y=140, width=220, height=280)
        crop = capture_api._crop(frame['frame_id'], rectangle, self.capture_owner)
        self.capture_id = crop['capture_id']
        self.capture_contents = base64.b64decode(crop['image'].split(',', 1)[1])

        with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
            status, response = self.video_restore(self.capture_owner)
        self.assertEqual(status, 200, response.get('detail'))
        meta = response['metadata']
        self.assertEqual(meta['capture_crop'], [420, 140, 220, 280])
        self.assertEqual(meta['frame_size'], [1280, 720])
        self.assertEqual(meta['source_time'], 4.75)

        ref_geo = {
            'observed_bounds_256': meta['geometry']['observed_bounds_256'],
            'native_size': meta['geometry']['native_size'],
            'picture_crop': meta['capture_crop'],
            'source_size': meta['frame_size'],
        }
        status_ref, body_ref = self.post_reference(
            full_hd_png, ref_geo, original_input=response['original_input'],
        )
        self.assertEqual(status_ref, 200, body_ref)
        self.assertEqual(body_ref['source_mode'], 'full_cropped')
        self.assertEqual(body_ref['observed_bounds_256'], meta['geometry']['observed_bounds_256'])

        expected_scaled = np.asarray(Image.fromarray(full_hd_rgb).resize((1280, 720), Image.Resampling.BILINEAR))
        expected_256, _, _, _ = prepare_crop(expected_scaled[140:420, 420:640].copy())
        np.testing.assert_array_equal(self.decode_data_url_rgb(body_ref['reference_256']), expected_256)

    def test_reference_endpoint_real_world_template_and_subcrop_alignment_pairs(self):
        here = Path(__file__).resolve()
        test_images_dir = next(
            (p / 'test_images' for p in (here.parents[1], here.parents[2], *here.parents[3:]) if (p / 'test_images').is_dir()),
            None,
        )
        if test_images_dir is None:
            raise unittest.SkipTest('test_images fixture directory is not present in this checkout')
        b1_path = test_images_dir / '851128671_1090321427166748_3093066005956191228_n.png'
        b2_path = test_images_dir / '851128671_1090321427166748_3093066005956191228_n - Copy.png'
        b1_bytes = b1_path.read_bytes()
        b2_bytes = b2_path.read_bytes()
        b1_rgb = np.asarray(Image.open(io.BytesIO(b1_bytes)).convert('RGB'))
        b2_rgb = np.asarray(Image.open(io.BytesIO(b2_bytes)).convert('RGB'))
        self.assertEqual((b1_rgb.shape[1], b1_rgb.shape[0]), (253, 232))
        self.assertEqual((b2_rgb.shape[1], b2_rgb.shape[0]), (133, 146))

        b1_face1_crop = [88, 51, 122, 141]
        b1_wide_crop = [75, 45, 148, 160]
        b2_face1_crop = [7, 2, 125, 138]

        def restore_pair(src_bytes, crop=None):
            with patch.object(controller, 'one_forward', side_effect=AssertionError('worker forbidden')):
                kwargs = {'file': self.upload(src_bytes), 'restoration': 'off', 'input_review': 'picture_upload'}
                if crop is not None:
                    kwargs['picture_crop'] = json.dumps(crop)
                res = self.restore(**kwargs)
            meta = res['metadata']
            geo = {
                'observed_bounds_256': meta['geometry']['observed_bounds_256'],
                'native_size': meta['geometry']['native_size'],
            }
            if crop is not None:
                geo['picture_crop'] = meta['picture_crop']
                geo['source_size'] = meta['source_size']
            return res, geo

        def assert_aligned(res, geo, ref_bytes, expected_mode, *, expect_eval_bounds=False):
            status, body = self.post_reference(ref_bytes, geo, original_input=res['original_input'])
            self.assertEqual(status, 200, body)
            self.assertEqual(body['source_mode'], expected_mode)
            obs = body['observed_bounds_256']
            self.assertEqual(obs, geo['observed_bounds_256'])
            if expect_eval_bounds:
                eb = body.get('eval_bounds_256')
                self.assertIsInstance(eb, list)
                self.assertEqual(len(eb), 4)
                self.assertGreaterEqual(eb[0], obs[0])
                self.assertGreaterEqual(eb[1], obs[1])
                self.assertLessEqual(eb[2], obs[2])
                self.assertLessEqual(eb[3], obs[3])
                self.assertLess(eb[0], eb[2])
                self.assertLess(eb[1], eb[3])
                l, t, r, b = eb
            else:
                self.assertNotIn('eval_bounds_256', body)
                l, t, r, b = obs
            orig_256 = self.decode_data_url_rgb(res['original_input'])
            ref_256 = self.decode_data_url_rgb(body['reference_256'])
            mae = float(np.abs(orig_256[t:b, l:r].astype(np.float32) - ref_256[t:b, l:r].astype(np.float32)).mean())
            self.assertLess(mae, 2.5)
            return body

        # b1(Face1) <-> b2: wider crop -> subcrop_aligned, tighter Face1 -> template_cropped
        res_b1_wide, geo_b1_wide = restore_pair(b1_bytes, b1_wide_crop)
        assert_aligned(res_b1_wide, geo_b1_wide, b2_bytes, 'subcrop_aligned', expect_eval_bounds=True)

        res_b1_face1, geo_b1_face1 = restore_pair(b1_bytes, b1_face1_crop)
        assert_aligned(res_b1_face1, geo_b1_face1, b2_bytes, 'template_cropped')

        # b2(Face1) -> b1: template_cropped
        res_b2_face1, geo_b2_face1 = restore_pair(b2_bytes, b2_face1_crop)
        assert_aligned(res_b2_face1, geo_b2_face1, b1_bytes, 'template_cropped')

        # b2(Full) -> b1: template_cropped
        res_b2_full, geo_b2_full = restore_pair(b2_bytes)
        assert_aligned(res_b2_full, geo_b2_full, b1_bytes, 'template_cropped')

        # b1(Full) -> b2: subcrop_aligned + eval_bounds_256
        res_b1_full, geo_b1_full = restore_pair(b1_bytes)
        assert_aligned(res_b1_full, geo_b1_full, b2_bytes, 'subcrop_aligned', expect_eval_bounds=True)

        # Multi-scale template match: 0.55x scaled b1 inside mismatched-aspect canvas -> template_cropped
        scaled_small_b1 = Image.fromarray(b1_rgb).resize((139, 128), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (210, 150), (30, 40, 50))
        canvas.paste(scaled_small_b1, (20, 10))
        canvas_buf = io.BytesIO()
        canvas.save(canvas_buf, format='PNG')
        assert_aligned(res_b2_face1, geo_b2_face1, canvas_buf.getvalue(), 'template_cropped')

        # Malformed original_input falls back safely without crashing.
        status_bad_orig, body_bad_orig = self.post_reference(
            b1_bytes, geo_b2_full, original_input='data:image/png;base64,not-valid-base64!!!',
        )
        self.assertEqual(status_bad_orig, 400)
        self.assertIn('Reference aspect ratio does not match', body_bad_orig['detail'])

        # Aspect-matched native_crop fallback with orig_256 handles minor bounds drift within 5% aspect ratio.
        _, _, _, base_geo = prepare_crop(np.zeros((120, 100, 3), dtype=np.uint8))
        orig_drift_256, _, _, _ = prepare_crop(b1_rgb[:120, :100])
        drift_ref = np.asarray(Image.fromarray(b1_rgb[:120, :100]).resize((102, 118), Image.Resampling.BILINEAR))
        status_drift_ok, body_drift_ok = self.post_reference(
            png(drift_ref),
            {'observed_bounds_256': base_geo['observed_bounds_256'], 'native_size': [100, 120]},
            original_input=controller.url(png(orig_drift_256)),
        )
        self.assertEqual(status_drift_ok, 200, body_drift_ok)
        self.assertEqual(body_drift_ok['source_mode'], 'native_crop')
        self.assertEqual(body_drift_ok['observed_bounds_256'], base_geo['observed_bounds_256'])

    def test_auto_blur_composes_after_non_mosaic_stages_and_supports_blur_island_fallback(self):
        from implementation import automatic_preparation

        # Build a 320x320 scene with sharp background grid + heavily blurred face
        rng = np.random.default_rng(707)
        yy, xx = np.mgrid[0:320, 0:320].astype(np.float32)
        scene = np.full((320, 320, 3), 210.0, dtype=np.float32)
        scene[(yy.astype(int) % 16) < 2] = 60.0
        scene[(xx.astype(int) % 24) < 2] = 85.0
        scene += rng.normal(0.0, 12.0, scene.shape)
        face = [100, 90, 120, 140]
        fx, fy, fw, fh = face
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        oval = ((xx - cx) / (fw * 0.48)) ** 2 + ((yy - cy) / (fh * 0.48)) ** 2 <= 1.0
        scene[oval] = np.stack([
            195.0 - 25.0 * np.hypot((xx - cx) / fw, (yy - cy) / fh),
            152.0 - 20.0 * np.hypot((xx - cx) / fw, (yy - cy) / fh),
            128.0 - 16.0 * np.hypot((xx - cx) / fw, (yy - cy) / fh),
        ], axis=-1)[oval]
        scene[fy + 38:fy + 50, fx + 24:fx + 48] = [30, 28, 32]
        scene[fy + 38:fy + 50, fx + 72:fx + 96] = [30, 28, 32]
        scene[fy + 96:fy + 108, fx + 36:fx + 84] = [125, 55, 55]
        scene = np.clip(np.rint(scene), 0, 255).astype(np.uint8)
        blurred = np.asarray(Image.fromarray(scene).filter(ImageFilter.GaussianBlur(12.0)))
        zone = ((xx >= fx - 6) & (xx < fx + fw + 6) & (yy >= fy - 6) & (yy < fy + fh + 6))
        scene[zone] = blurred[zone]

        dummy_out = (None, png(np.zeros((256, 256, 3), np.uint8)), {})
        fake_stage_out = scene.copy()
        fake_stage_out[0, 0] = np.clip(fake_stage_out[0, 0].astype(int) ^ 1, 0, 255).astype(np.uint8)

        # 1) Composition: non-mosaic stage + Auto Blur both run and compose into protect_auto
        with (
            patch.object(automatic_preparation, 'prepare_source', return_value=(None, None, {})),
            patch.object(automatic_preparation, 'prepare_composables', return_value=(fake_stage_out, ['edge_deringing'], {'edge_deringing': {'policy': 'test'}}, {})),
            patch.object(capture_api, 'detect_faces', return_value=[face]),
            patch.object(controller, 'one_forward', return_value=dummy_out),
        ):
            res_composed = self.restore(
                file=self.upload(png(scene)),
                restoration='auto',
                preparation_mode='automatic',
                input_review='picture_upload',
            )
        auto_comp = res_composed['metadata']['automatic_preparation']
        self.assertEqual(auto_comp['route'], 'protect_auto')
        self.assertIn('edge_deringing', auto_comp['applied_stages'])
        self.assertIn('protect_auto', auto_comp['applied_stages'])

        # 2) Blur-island fallback: YuNet returns [] on extreme blur, manual crop still triggers protect_auto
        crop_json = json.dumps([40, 30, 240, 260])
        with (
            patch.object(automatic_preparation, 'prepare_source', return_value=(None, None, {})),
            patch.object(capture_api, 'detect_faces', return_value=[]),
            patch.object(controller, 'one_forward', return_value=dummy_out),
        ):
            res_island = self.restore(
                file=self.upload(png(scene)),
                restoration='auto',
                preparation_mode='automatic',
                input_review='picture_upload',
                picture_crop=crop_json,
            )
        auto_island = res_island['metadata']['automatic_preparation']
        self.assertEqual(auto_island['route'], 'protect_auto')
        self.assertEqual(auto_island['face_detection'], 'blur_island_fallback')

    def test_upright_roll_alignment_pins_derived_native_and_processing_hashes(self):
        """Upright roll alignment must atomically re-pin derived-native.png, processing.json, and selected_input_sha256."""
        from implementation import automatic_preparation

        rng = np.random.default_rng(808)
        yy, xx = np.mgrid[0:320, 0:320].astype(np.float32)
        scene = np.full((320, 320, 3), 210.0, dtype=np.float32)
        scene[(yy.astype(int) % 16) < 2] = 60.0
        scene[(xx.astype(int) % 24) < 2] = 85.0
        scene += rng.normal(0.0, 12.0, scene.shape)
        face = [100, 90, 120, 140]
        fx, fy, fw, fh = face
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        oval = ((xx - cx) / (fw * 0.48)) ** 2 + ((yy - cy) / (fh * 0.48)) ** 2 <= 1.0
        scene[oval] = np.stack([
            195.0 - 25.0 * np.hypot((xx - cx) / fw, (yy - cy) / fh),
            152.0 - 20.0 * np.hypot((xx - cx) / fw, (yy - cy) / fh),
            128.0 - 16.0 * np.hypot((xx - cx) / fw, (yy - cy) / fh),
        ], axis=-1)[oval]
        scene[fy + 38:fy + 50, fx + 24:fx + 48] = [30, 28, 32]
        scene[fy + 38:fy + 50, fx + 72:fx + 96] = [30, 28, 32]
        scene[fy + 96:fy + 108, fx + 36:fx + 84] = [125, 55, 55]
        scene = np.clip(np.rint(scene), 0, 255).astype(np.uint8)
        blurred = np.asarray(Image.fromarray(scene).filter(ImageFilter.GaussianBlur(12.0)))
        zone = ((xx >= fx - 6) & (xx < fx + fw + 6) & (yy >= fy - 6) & (yy < fy + fh + 6))
        scene[zone] = blurred[zone]

        dummy_out = (None, png(np.zeros((256, 256, 3), np.uint8)), {})
        rolled_rec = [{'box': face, 'crop': [55, 45, 210, 230], 'score': 0.91, 'roll_deg': 22.5}]
        fake_stage_out = scene.copy()
        fake_stage_out[0, 0] = np.clip(fake_stage_out[0, 0].astype(int) ^ 1, 0, 255).astype(np.uint8)

        for label_tag, comp_patch in (
            ('standalone_protect_auto', patch.object(automatic_preparation, 'prepare_composables', return_value=(None, [], {}, {}))),
            ('composed_protect_auto', patch.object(automatic_preparation, 'prepare_composables', return_value=(fake_stage_out, ['edge_deringing'], {'edge_deringing': {'policy': 'test'}}, {}))),
        ):
            with self.subTest(branch=label_tag):
                with (
                    patch.object(automatic_preparation, 'prepare_source', return_value=(None, None, {})),
                    patch.object(capture_api, 'detect_face_details', return_value=rolled_rec),
                    comp_patch,
                    patch.object(controller, 'one_forward', return_value=dummy_out),
                ):
                    res = self.restore(
                        file=self.upload(png(scene)),
                        restoration='auto',
                        preparation_mode='automatic',
                        input_review='picture_upload',
                    )
                meta = res['metadata']
                self.assertEqual(meta['alignment']['applied'], True)
                self.assertAlmostEqual(meta['alignment']['roll_deg'], 22.5, places=2)
                bundle_bytes = base64.b64decode(res['bundle'].split(',', 1)[1])
                with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
                    derived_bytes = archive.read('derived-native.png')
                    processing = json.loads(archive.read('processing.json'))
                derived_sha = controller.digest(derived_bytes)
                self.assertEqual(derived_sha, meta['selected_input_sha256'])
                self.assertEqual(derived_sha, processing['derived_native_sha256'])
                if isinstance(processing.get('payload_pins'), dict) and 'derived-native.png' in processing['payload_pins']:
                    self.assertEqual(processing['payload_pins']['derived-native.png']['sha256'], derived_sha)

    def test_policy_a1_localized_face_mosaic_in_sharp_hair_delivers_uniform_crop_blur(self):
        """Localized face mosaic surrounded by sharp hair must deliver 100% uniform crop blur (zero mask holes)."""
        from implementation import automatic_preparation

        rng = np.random.default_rng(909)
        h, w = 256, 256
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Sharp high-frequency hair/background texture everywhere outside center face
        hair = np.stack([
            155.0 + 48.0 * np.sin(0.55 * xx + 0.12 * yy) + rng.normal(0.0, 12.0, (h, w)),
            120.0 + 40.0 * np.sin(0.55 * xx + 0.12 * yy + 0.4) + rng.normal(0.0, 10.0, (h, w)),
            75.0 + 30.0 * np.sin(0.55 * xx + 0.12 * yy + 0.8) + rng.normal(0.0, 8.0, (h, w)),
        ], axis=-1)
        scene = np.clip(np.rint(hair), 0, 255).astype(np.uint8)
        # Localized 10px mosaic inside center face only
        blk = rng.integers(30, 230, (12, 12, 3), dtype=np.uint8)
        mosaic_face = np.asarray(Image.fromarray(blk).resize((120, 120), Image.Resampling.NEAREST))
        scene[68:188, 68:188] = mosaic_face

        dummy_out = (None, png(np.zeros((256, 256, 3), np.uint8)), {})
        with patch.object(controller, 'one_forward', return_value=dummy_out):
            res = self.restore(
                file=self.upload(png(scene)),
                restoration='auto',
                preparation_mode='automatic',
                input_review='picture_upload',
            )
        meta = res['metadata']
        auto = meta['automatic_preparation']
        self.assertEqual(auto['route'], 'picture_region_gaussian')
        self.assertTrue(auto['uniform_crop_smoothing'])
        self.assertGreater(auto['crop_Gaussian_radius_native'], 1.5)
        self.assertNotIn('noise_denoise', auto.get('applied_stages', []))

        with zipfile.ZipFile(io.BytesIO(base64.b64decode(res['bundle'].split(',', 1)[1]))) as archive:
            mask = np.asarray(Image.open(io.BytesIO(archive.read('full-image-pixelation-mask.png')))) > 0
            full_sm = np.asarray(Image.open(io.BytesIO(archive.read('full-image-smoothed.png'))))
            derived = np.asarray(Image.open(io.BytesIO(archive.read('derived-native.png'))))
            processing = json.loads(archive.read('processing.json'))
        self.assertTrue(processing['uniform_crop_smoothing'])
        self.assertEqual(processing['crop_Gaussian_radius_native'], auto['crop_Gaussian_radius_native'])
        # Full-frame localization provenance keeps ~mask exact on full-image-smoothed.png
        self.assertTrue(mask.any() and (~mask).any())
        np.testing.assert_array_equal(full_sm[~mask], scene[~mask])
        # Selected crop sent to V3.4 (derived-native.png) is 100% uniformly smoothed (including exterior hair!)
        expected_uniform = automatic_preparation.apply_uniform_crop_gaussian(scene, auto['crop_Gaussian_radius_native'])
        np.testing.assert_array_equal(derived, expected_uniform)
        self.assertGreater(float(np.any(derived[~mask] != scene[~mask], axis=1).mean()), 0.95)

    def test_policy_a1_degraded_patch_conditioning_surround_matches_sharp_exterior_protecting_face(self):
        """degraded_patch_conditioning keeps conditioned face bit-exact while surround-matching sharp exterior hair."""
        from implementation import automatic_preparation

        rng = np.random.default_rng(919)
        h, w = 320, 320
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Sharp high-frequency hair/background outside patch
        exterior = np.stack([
            150.0 + 55.0 * np.sin(0.65 * xx + 0.15 * yy) + rng.normal(0.0, 14.0, (h, w)),
            120.0 + 45.0 * np.cos(0.60 * xx - 0.18 * yy) + rng.normal(0.0, 12.0, (h, w)),
            85.0 + 35.0 * np.sin(0.50 * xx + 0.22 * yy) + rng.normal(0.0, 10.0, (h, w)),
        ], axis=-1)
        scene = np.clip(np.rint(exterior), 0, 255).astype(np.uint8)
        # Degraded multi-pixel block noise face patch inside [32:288, 32:288] (256x256, 8px blocks)
        blk = rng.integers(35, 220, (32, 32, 3), dtype=np.uint8)
        scene[32:288, 32:288] = np.asarray(Image.fromarray(blk).resize((256, 256), Image.Resampling.NEAREST))

        raw_mask = np.zeros((h, w), dtype=bool)
        raw_mask[32:288, 32:288] = True
        prepared_full, report_full, ev_full = automatic_preparation.apply_pixelation_mask(
            scene, raw_mask, (w, h), 0
        )
        self.assertIsNotNone(prepared_full)
        self.assertIsNotNone(report_full.get('degraded_patch_conditioning'))
        full_mask = np.asarray(Image.open(io.BytesIO(ev_full['full-image-pixelation-mask.png']))) > 0

        dummy_out = (None, png(np.zeros((256, 256, 3), np.uint8)), {})
        with (
            patch.object(automatic_preparation, 'prepare_source', return_value=(prepared_full, report_full, ev_full)),
            patch.object(controller, 'one_forward', return_value=dummy_out),
        ):
            res = self.restore(
                file=self.upload(png(scene)),
                restoration='auto',
                preparation_mode='automatic',
                input_review='picture_upload',
            )
        auto = res['metadata']['automatic_preparation']
        self.assertEqual(auto['route'], 'picture_region_gaussian')
        self.assertFalse(auto.get('uniform_crop_smoothing', False))
        self.assertTrue(auto['surround_matched_after_conditioning'])

        with zipfile.ZipFile(io.BytesIO(base64.b64decode(res['bundle'].split(',', 1)[1]))) as archive:
            derived = np.asarray(Image.open(io.BytesIO(archive.read('derived-native.png'))))
            processing = json.loads(archive.read('processing.json'))
        self.assertTrue(processing['surround_matched_after_conditioning'])
        # Conditioned face interior is 100% bit-exact (never double-blurred)
        np.testing.assert_array_equal(derived[full_mask], prepared_full[full_mask])
        # Sharp exterior hair/background is softened to match the conditioned face
        self.assertGreater(float(np.any(derived[~full_mask] != scene[~full_mask], axis=1).mean()), 0.80)

    def _assert_highres_jpeg_non_blur(self, quality: int):
        from PIL import ImageOps

        test_dir = Path(__file__).resolve().parents[2] / 'test_images'
        clean_path = test_dir / 'face 1 - default.jpg'
        if not clean_path.exists():
            self.skipTest('face 1 - default.jpg fixture missing')
        clean_im = ImageOps.exif_transpose(Image.open(io.BytesIO(clean_path.read_bytes()))).convert('RGB')
        dummy_out = (None, png(np.zeros((256, 256, 3), np.uint8)), {})
        im512 = clean_im.resize((512, 512), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im512.save(buf, format='JPEG', quality=quality, subsampling=2)
        with patch.object(controller, 'one_forward', return_value=dummy_out):
            res_j = self.restore(
                file=self.upload(buf.getvalue(), filename=f'q{quality}.jpg'),
                restoration='auto',
                preparation_mode='automatic',
                input_review='picture_upload',
            )
        route_j = res_j['metadata']['automatic_preparation']['route']
        self.assertNotIn('gaussian', route_j)
        self.assertNotEqual(route_j, 'protect_auto')
        self.assertFalse(res_j['metadata']['automatic_preparation'].get('uniform_crop_smoothing', False))

    def test_negative_controls_clean_portrait_non_degraded(self):
        """Clean sharp portraits never trigger false blur or preparation."""
        test_dir = Path(__file__).resolve().parents[2] / 'test_images'
        clean_path = test_dir / 'face 1 - default.jpg'
        if not clean_path.exists():
            self.skipTest('face 1 - default.jpg fixture missing')
        clean_bytes = clean_path.read_bytes()
        dummy_out = (None, png(np.zeros((256, 256, 3), np.uint8)), {})
        with patch.object(controller, 'one_forward', return_value=dummy_out):
            res_clean = self.restore(
                file=self.upload(clean_bytes, filename='face1.jpg'),
                restoration='auto',
                preparation_mode='automatic',
                input_review='picture_upload',
            )
        self.assertEqual(res_clean['metadata']['automatic_preparation']['route'], 'direct')
        self.assertFalse(res_clean['metadata']['preparation_applied'])

    def test_negative_controls_coarse_mosaic_non_conditioned(self):
        """Coarse 24/32px clean-color mosaics stay uniform Gaussian blur, never degraded_patch_conditioning."""
        import cv2
        from PIL import ImageOps
        from implementation import automatic_preparation

        test_dir = Path(__file__).resolve().parents[2] / 'test_images'
        clean_path = test_dir / 'face 1 - default.jpg'
        if not clean_path.exists():
            self.skipTest('face 1 - default.jpg fixture missing')
        clean_im = ImageOps.exif_transpose(Image.open(io.BytesIO(clean_path.read_bytes()))).convert('RGB')
        clean_rgb = np.asarray(clean_im)
        h, w = clean_rgb.shape[:2]
        y0, y1, x0, x1 = int(h * 0.12), int(h * 0.88), int(w * 0.12), int(w * 0.88)
        for blk_sz in (24, 32):
            rgb_mos = clean_rgb.copy()
            sub = rgb_mos[y0:y1, x0:x1]
            sh, sw = sub.shape[:2]
            sm = cv2.resize(sub, (max(4, sw // blk_sz), max(4, sh // blk_sz)), interpolation=cv2.INTER_AREA)
            rgb_mos[y0:y1, x0:x1] = cv2.resize(sm, (sw, sh), interpolation=cv2.INTER_NEAREST)
            m = np.zeros((h, w), dtype=bool)
            m[y0:y1, x0:x1] = True
            self.assertIsNone(automatic_preparation._condition_degraded_face_patch(rgb_mos, m))

    def test_negative_controls_highres_jpeg_q8_non_blur(self):
        """>=512px real portrait JPEG Q=8 routes to JPEG/macroblock deblocking, never Gaussian blur or protect_auto."""
        self._assert_highres_jpeg_non_blur(8)

    def test_negative_controls_highres_jpeg_q12_non_blur(self):
        """>=512px real portrait JPEG Q=12 routes to JPEG/macroblock deblocking, never Gaussian blur or protect_auto."""
        self._assert_highres_jpeg_non_blur(12)

    def test_recommend_face_index_prefers_foreground_degraded_and_rejects_torso_and_hat_back(self):
        """Step 1 ranking favors larger degraded foreground subjects over tiny bystanders, loose torsos, and hat backs."""
        rng = np.random.default_rng(1212)
        h, w = 440, 640
        scene = np.full((h, w, 3), 120, dtype=np.uint8)
        # 1) Tiny sharp bystander vs larger pixelated foreground officer (mosaic_04 geometry)
        scene[149:180, 150:175] = np.array([175, 138, 118], dtype=np.uint8)
        blk = Image.fromarray(rng.integers(50, 210, (12, 12, 3), dtype=np.uint8)).resize((97, 103), Image.Resampling.NEAREST)
        scene[303:406, 182:279] = np.asarray(blk)
        recs_m04 = [
            {'box': [150, 149, 25, 31], 'crop': [140, 139, 45, 51], 'score': 0.84},
            {'box': [182, 303, 97, 103], 'crop': [149, 270, 163, 169], 'score': 0.47},
        ]
        self.assertEqual(controller._recommend_face_index(scene, recs_m04, w, h), 1)

        # 2) Giant loose low-confidence torso box vs smaller high-confidence archival B&W face (cctv_06 geometry)
        bw_scene = np.full((720, 1280, 3), 110, dtype=np.uint8)
        recs_c06 = [
            {'box': [284, 190, 413, 363], 'crop': [152, 58, 677, 627], 'score': 0.485},
            {'box': [1070, 0, 210, 218], 'crop': [1000, 0, 280, 288], 'score': 0.736},
        ]
        self.assertEqual(controller._recommend_face_index(bw_scene, recs_c06, 1280, 720), 1)

        # 3) Blurry bokeh FP overlapping a sharper high-confidence real face (mask_03 geometry)
        bokeh_scene = np.full((600, 600, 3), [135, 118, 102], dtype=np.uint8)
        bokeh_scene[96:213, 324:416] = rng.integers(30, 230, (117, 92, 3), dtype=np.uint8)
        recs_m03 = [
            {'box': [190, 97, 130, 143], 'crop': [144, 51, 222, 235], 'score': 0.686},
            {'box': [324, 96, 92, 117], 'crop': [289, 61, 162, 187], 'score': 0.910},
        ]
        self.assertEqual(controller._recommend_face_index(bokeh_scene, recs_m03, 600, 600), 1)

    def test_upright_roll_five_to_fifteen_degrees_aligns_mosaic_and_pins_hashes(self):
        """Full-range upright roll (|roll| >= 5 deg) aligns mosaic routes and pins derived-native & processing hashes."""
        rng = np.random.default_rng(1414)
        scene = np.full((256, 256, 3), 120, dtype=np.uint8)
        blk = Image.fromarray(rng.integers(25, 230, (12, 12, 3), dtype=np.uint8)).resize((120, 120), Image.Resampling.NEAREST)
        scene[68:188, 68:188] = np.asarray(blk)
        rolled_rec = [{
            'box': [68, 68, 120, 120],
            'crop': [30, 30, 196, 196],
            'score': 0.82,
            'roll_deg': -8.5,
            'landmarks': [[102.0, 112.0], [154.0, 104.0], [128.0, 134.0], [108.0, 158.0], [148.0, 152.0]],
        }]
        dummy_out = (None, png(np.zeros((256, 256, 3), np.uint8)), {})
        with (
            patch.object(capture_api, 'detect_face_details', return_value=rolled_rec),
            patch.object(controller, 'one_forward', return_value=dummy_out),
        ):
            res = self.restore(
                file=self.upload(png(scene)),
                restoration='auto',
                preparation_mode='automatic',
                input_review='picture_upload',
                picture_crop=json.dumps([30, 30, 196, 196]),
            )
        meta = res['metadata']
        self.assertEqual(meta['alignment']['applied'], True)
        self.assertAlmostEqual(meta['alignment']['roll_deg'], -8.5, places=2)
        self.assertTrue(meta['alignment']['source_context'])
        bundle_bytes = base64.b64decode(res['bundle'].split(',', 1)[1])
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
            derived_bytes = archive.read('derived-native.png')
            processing = json.loads(archive.read('processing.json'))
        derived_sha = controller.digest(derived_bytes)
        self.assertEqual(derived_sha, meta['selected_input_sha256'])
        self.assertEqual(derived_sha, processing['derived_native_sha256'])

    def test_upright_roll_transforms_censor_bar_exclude_mask_coordinates(self):
        """When upright roll runs before prepare_automatically, exclude_mask is warped into the rotated frame."""
        from implementation import automatic_preparation

        bar_img = np.full((192, 192, 3), [172, 142, 122], dtype=np.uint8)
        yy, xx = np.mgrid[0:192, 0:192]
        bar_img[..., 0] = np.clip(172 + (xx // 6) - 16, 0, 255).astype(np.uint8)
        bar_img[68:90, 44:148] = [2, 2, 2]
        unrotated_bar = np.zeros((192, 192), dtype=bool)
        unrotated_bar[68:90, 44:148] = True
        rolled_rec = [{
            'box': [28, 28, 136, 136],
            'crop': [0, 0, 192, 192],
            'score': 0.90,
            'roll_deg': -12.0,
            'landmarks': [[72.0, 82.0], [120.0, 72.0], [96.0, 102.0], [78.0, 128.0], [114.0, 122.0]],
        }]
        captured = {}
        orig_prepare_auto = automatic_preparation.prepare

        def spy_prepare_auto(native_in, source_in, face_box, rect, **kwargs):
            captured['blur_source'] = source_in.copy()
            captured['exclude_mask'] = kwargs.get('exclude_mask')
            return orig_prepare_auto(native_in, source_in, face_box, rect, **kwargs)

        dummy_out = (None, png(np.zeros((256, 256, 3), np.uint8)), {})
        with (
            patch.object(capture_api, 'detect_face_details', return_value=rolled_rec),
            patch.object(automatic_preparation, 'prepare_source', return_value=(None, None, {})),
            patch.object(automatic_preparation, 'prepare', side_effect=spy_prepare_auto),
            patch.object(controller, 'one_forward', return_value=dummy_out),
        ):
            res = self.restore(
                file=self.upload(png(bar_img), filename='rolled_barred.png'),
                restoration='auto',
                preparation_mode='automatic',
                input_review='picture_upload',
            )
        self.assertTrue(res['metadata']['automatic_preparation'].get('synthetic_censor_bar_suspected'))
        excl = captured.get('exclude_mask')
        self.assertIsNotNone(excl)
        self.assertEqual(excl.shape, (192, 192))
        self.assertGreater(int(excl.sum()), 500)
        # Rotated exclude_mask must differ from unrotated bar and track the dark rotated bar pixels.
        self.assertFalse(np.array_equal(excl, unrotated_bar))
        blur_src = captured['blur_source']
        self.assertLess(float(blur_src[excl].mean()), 30.0)

    def test_clean_control_v34_veto_and_censor_bar_metadata_in_restore(self):
        """Clean studio controls route direct with 0% pixel edits; synthetic censor bars surface metadata & suppress debleed."""
        # 1) Clean studio controls (clean_01, clean_03, clean_05 when present) -> direct & 0% modified
        rest_dir = Path(__file__).resolve().parents[2] / 'test_images' / 'internet_samples' / 'restoration'
        dummy_out = (None, png(np.zeros((256, 256, 3), np.uint8)), {})
        for cname in ('clean_01_studio_male.jpg', 'clean_03_uniform_glasses.jpg', 'clean_05_elderly_glasses.jpg'):
            cpath = rest_dir / cname
            if not cpath.exists():
                continue
            with self.subTest(clean=cname):
                raw = cpath.read_bytes()
                with (
                    patch.object(capture_api, 'detect_face_details', return_value=[]),
                    patch.object(controller, 'one_forward', return_value=dummy_out),
                ):
                    res = self.restore(
                        file=self.upload(raw, filename=cname),
                        restoration='auto',
                        preparation_mode='automatic',
                        input_review='picture_upload',
                    )
                self.assertEqual(res['metadata']['automatic_preparation']['route'], 'direct')
                self.assertFalse(res['metadata']['preparation_applied'])

        # 2) Synthetic eye censor bar emits synthetic_censor_bar_suspected=True
        bar_img = np.full((192, 192, 3), [170, 140, 120], dtype=np.uint8)
        yy, xx = np.mgrid[0:192, 0:192]
        bar_img[..., 0] = np.clip(170 + (xx // 6) - 16, 0, 255).astype(np.uint8)
        bar_img[68:90, 44:148] = [2, 2, 2]
        with (
            patch.object(capture_api, 'detect_face_details', return_value=[]),
            patch.object(controller, 'one_forward', return_value=dummy_out),
        ):
            res_bar = self.restore(
                file=self.upload(png(bar_img), filename='barred.png'),
                restoration='auto',
                preparation_mode='automatic',
                input_review='picture_upload',
            )
        self.assertTrue(res_bar['metadata']['automatic_preparation'].get('synthetic_censor_bar_suspected'))
        self.assertNotIn('chroma_debleeding', res_bar['metadata']['automatic_preparation'].get('applied_stages', []))


class Phase2FiqaRankingTests(unittest.TestCase):
    """Phase 2 eDifFIQA-T plausibility integration and graceful None fallback in _recommend_face_index."""

    def test_recommend_face_index_fiqa_active_and_none_fallback_on_spotlights(self):
        import cv2
        from implementation import capture_worker, fiqa_scorer

        root = Path(__file__).resolve().parents[2] / 'test_images' / 'internet_samples'
        cases = [
            ('blur_08', root / 'restoration' / 'blur_08_transit_anon.jpg'),
            ('mosaic_04', root / 'restoration' / 'mosaic_04_police_street.png'),
            ('cctv_06', root / 'restoration' / 'cctv_06_archival_grainy.jpg'),
            ('mask_03', root / 'covering' / 'mask_03_kn95_sunglasses.jpg'),
            ('compound_05', root / 'covering' / 'compound_05_cctv_mask_hood.jpg'),
        ]
        for tag, path in cases:
            if not path.is_file():
                continue
            with self.subTest(case=tag):
                rgb = controller.decode_original(path.read_bytes())
                h, w = rgb.shape[:2]
                recs = sorted(
                    capture_worker.scan_face_records(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
                    key=lambda r: (r['box'][0], r['box'][1]),
                )
                self.assertGreaterEqual(len(recs), 1)
                idx_active = controller._recommend_face_index(rgb, recs, w, h)
                self.assertTrue(0 <= idx_active < len(recs))
                if tag == 'cctv_06':
                    self.assertLess(recs[idx_active]['box'][0], 900)
                elif tag == 'mosaic_04':
                    # Selected box must be a real police face, never the lower bicycle wheel FP (y=303).
                    self.assertLess(recs[idx_active]['box'][1], 240)
                elif tag == 'mask_03':
                    self.assertEqual(len(recs), 1)
                elif tag == 'compound_05':
                    self.assertEqual(len(recs), 1)

                with patch.object(fiqa_scorer, 'score_face_crops_fiqa', return_value=[None] * len(recs)):
                    idx_none = controller._recommend_face_index(rgb, recs, w, h)
                self.assertTrue(0 <= idx_none < len(recs))
                if tag == 'mosaic_04':
                    self.assertLess(recs[idx_none]['box'][1], 240)

    def test_recommend_face_index_calls_fiqa_once_and_prefers_higher_fiqa_plausibility(self):
        from implementation import fiqa_scorer

        scene = np.full((320, 480, 3), 128, dtype=np.uint8)
        # Two symmetric candidates with identical geometry/score; FIQA breaks tie cleanly.
        recs = [
            {'box': [60, 80, 96, 112], 'crop': [24, 44, 168, 184], 'score': 0.80},
            {'box': [324, 80, 96, 112], 'crop': [288, 44, 168, 184], 'score': 0.80},
        ]
        with patch.object(fiqa_scorer, 'score_face_crops_fiqa', return_value=[0.02, 0.52]) as mock_fiqa:
            picked = controller._recommend_face_index(scene, recs, 480, 320)
        self.assertEqual(mock_fiqa.call_count, 1)
        self.assertEqual(len(mock_fiqa.call_args[0][0]), 2)
        self.assertEqual(picked, 1)

        with patch.object(fiqa_scorer, 'score_face_crops_fiqa', return_value=[None, None]):
            picked_fb = controller._recommend_face_index(scene, recs, 480, 320)
        self.assertEqual(picked_fb, 0)


if __name__ == '__main__':
    unittest.main()
