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
        self.assertEqual(result['metadata']['automatic_preparation']['route'], 'scanline_mosaic_coordinated')
        worker.assert_called_once_with(png(prepared[5:37, 4:44]), 'v3')
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
        self.assertEqual([call.args for call in worker.call_args_list],
                         [(png(smoothed[5:37, 4:44]), 'v3'), (png(smoothed[18:50, 50:80]), 'v3')])
        self.assertEqual(first_result['metadata']['automatic_preparation']['route'], 'frame_region_gaussian')
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
        report = {'mixed_jpeg_mosaic': True, 'mosaic_grid_period_native': 4, 'outside_deblock_fraction': .35}
        evidence = {'full-image-smoothed.png': combined_png}
        restored = png(np.zeros((256, 256, 3), np.uint8))
        with (
            patch.object(automatic_preparation, 'prepare_source', return_value=(None, None, {})),
            patch.object(mixed_jpeg, 'prepare_mixed_jpeg', return_value=(combined, report, evidence)),
            patch.object(jpeg_deblocking, 'prepare_jpeg') as standalone,
            patch.object(controller, 'one_forward', return_value=(None, restored, {})) as worker,
        ):
            result = self.restore(restoration='auto', preparation_mode='automatic', input_review='picture_upload')
        standalone.assert_not_called()
        worker.assert_called_once_with(combined_png, 'v3')
        self.assertEqual(result['metadata']['automatic_preparation']['route'], 'jpeg_mosaic_coordinated')
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
            worker.assert_called_once_with(png(prepared[40:200, 40:200]), 'v3')
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


if __name__ == '__main__':
    unittest.main()
