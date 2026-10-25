"""Restoration-only U96 controller; injected owned worker, no model loading.

The existing app/router/defaults are unchanged. Auto is the retained input-only
development heuristic; it does not qualify facial structure or useful output.
"""
import copy
import hashlib
import io
import json
import threading
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps

import codeformer_app_delivery_v1 as delivery
from research.protected_surroundings_brush_v1 import build_preview, png

SELECTION = 'incumbent_U96'
POLICY = 'unwired-reviewed-U96-restoration-controller-v1'
BUNDLE_LIMIT = 64 * 1024 ** 2


class RestorationBusy(RuntimeError):
    """A future HTTP seam maps this explicit busy refusal to409."""


class CoveringWorkflowRequired(ValueError):
    """Nonempty covering removal must use the existing separate workflow."""


class RestorationSourceMismatch(ValueError):
    """Reviewed service sources changed; this is not an input error."""


class RestorationWorkerStateInvalid(ValueError):
    """Owned worker counters/outcome are unavailable; never retry automatically."""


def need(ok, message):
    if not ok:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode()


def quality_signals(square_rgb, square_mask):
    """Exact retained face_workflow.quality_signals math, without its Torch import."""
    rgb = cv2.resize(square_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(square_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    support = cv2.erode(1-mask, np.ones((9, 9), np.uint8)).astype(bool)
    support[:4] = False
    support[-4:] = False
    support[:, :4] = False
    support[:, -4:] = False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    enough = int(support.sum()) >= 512
    if not enough:
        return {'blur_variance': None, 'noise_sigma_255': None, 'suggest_restoration': False,
                'reason': 'Too little visible area for the automatic quality signal; choose an override.',
                'visible_signal_pixels': int(support.sum()), 'blur_threshold': 24, 'noise_threshold': 8}
    smoothed = cv2.GaussianBlur(gray, (3, 3), .6)
    blur = float(cv2.Laplacian(smoothed, cv2.CV_32F)[support].var())
    low = cv2.GaussianBlur(gray, (7, 7), 1.5)
    gradient = np.hypot(cv2.Sobel(low, cv2.CV_32F, 1, 0), cv2.Sobel(low, cv2.CV_32F, 0, 1))
    flat = support & (gradient <= np.percentile(gradient[support], 35))
    response = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32))
    noise = float(np.median(np.abs(response[flat])) / (6 * .67448975))
    reasons = []
    if blur < 24:
        reasons.append('blur signal')
    if noise >= 8:
        reasons.append('noise signal')
    return {'blur_variance': blur, 'noise_sigma_255': noise, 'suggest_restoration': bool(reasons),
            'reason': ', '.join(reasons) if reasons else 'Visible detail does not trigger restoration.',
            'visible_signal_pixels': int(support.sum()), 'blur_threshold': 24, 'noise_threshold': 8}


def reviewed_input(value):
    if value == 'needs_clearer':
        raise ValueError('Facial structure is insufficient. Upload a clearer face crop.')
    if value == 'out_of_scope':
        raise ValueError('Upload one already cropped frontal or mildly turned face.')
    need(value == 'usable', 'Review the input first: one face with readable visible facial features.')


def immutable(value):
    result = value.copy()
    result.flags.writeable = False
    return result


def bundle_bytes(payloads):
    need(sum(len(value) for value in payloads.values()) <= BUNDLE_LIMIT, 'Evidence bundle exceeds64MiB.')
    class Limited(io.BytesIO):
        def write(self, raw):
            need(self.tell() + len(raw) <= BUNDLE_LIMIT, 'Compressed evidence bundle exceeds64MiB.')
            return super().write(raw)
    stream = Limited()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in payloads.items():
            archive.writestr(name, raw)
    return stream.getvalue()


class RestorationController:
    """Own one injected ProcessSelectedRestorer; caller binds reviewed sources.

    source_validator returns the exact declared source_identity and rehashes its
    reviewed files. run_one/release_models/measured_status/count retain the
    established worker ownership, deadline, events and no-retry guarantees.
    """
    def __init__(self, worker, source_validator, source_identity, model_files, preview_builder=build_preview,
                 deployment='unwired'):
        need(callable(source_validator) and callable(preview_builder), 'Reviewed source/preview functions required.')
        need(all(callable(getattr(worker, key, None)) for key in
                 ('run_one', 'release_models', 'measured_status', 'count')), 'Owned worker interface required.')
        need(type(source_identity) is dict and bool(source_identity), 'Explicit reviewed source identity required.')
        need(deployment in ('unwired', 'opt_in_U96_staging'), 'Explicit reviewed deployment required.')
        self.deployment = deployment
        self.source_identity = json.loads(json_bytes(source_identity))
        self.models = json.loads(json_bytes(model_files))
        need(set(self.models) == set(delivery.MODEL_FILES), 'Exact reviewed model pins required.')
        for name, digest in delivery.MODEL_FILES.items():
            row = self.models[name]
            need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}
                 and type(row['path']) is str and bool(row['path']) and type(row['bytes']) is int
                 and row['bytes'] > 0 and row['sha256'] == digest, 'Retained model binding differs.')
        self.worker, self.source_validator, self.preview_builder = worker, source_validator, preview_builder
        self.lock = threading.Lock()

    def _sources(self):
        try:
            identity = self.source_validator()
        except Exception as error:
            raise RestorationSourceMismatch('Reviewed controller/worker sources could not be verified.') from error
        if identity != self.source_identity:
            raise RestorationSourceMismatch('Reviewed controller/worker source identity changed.')

    def _status(self):
        try:
            value = self.worker.measured_status()
        except Exception as error:
            raise RestorationWorkerStateInvalid('Owned worker state could not be verified; do not automatically retry.') from error
        if not (type(value) is dict and all(type(value.get(key)) is int and value[key] >= 0 for key in
                 ('neural_forward_attempts', 'neural_forward_returns', 'completed_comparisons',
                  'uncertain_requests', 'uncertain_forward_attempts_upper_bound', 'worker_cleanup_failures'))):
            raise RestorationWorkerStateInvalid('Exact finite owned-worker counters required.')
        if not (value.get('counter_integrity') is True and value.get('active_worker') is False
             and value.get('active_forward_counts_known') is True and value['uncertain_requests'] == 0
             and value['uncertain_forward_attempts_upper_bound'] == 0 and value['worker_cleanup_failures'] == 0
             and value['neural_forward_attempts'] == value['neural_forward_returns']):
            raise RestorationWorkerStateInvalid('Worker outcome is busy or uncertain; do not automatically retry.')
        return copy.deepcopy(value)

    def _preview(self, contents, protect, apply, radius):
        response, payloads = self.preview_builder(contents, protect, apply, nominal_radius=radius)
        pure_id = response['preview_id']
        preview_id = hashlib.sha256(json_bytes({'pure_preview_id': pure_id,
                                                'sources': self.source_identity})).hexdigest()
        response = copy.deepcopy(response)
        response['preview_id'] = preview_id
        response['metadata']['controller_preview_id'] = preview_id
        return response, payloads

    def preview(self, contents, protect, apply, radius, input_review=''):
        reviewed_input(input_review)
        if not self.lock.acquire(blocking=False):
            raise RestorationBusy('A restoration is already running. Wait for it to finish.')
        try:
            self._sources()
            response, payloads = self._preview(contents, protect, apply, radius)
            self._sources()
            return response, payloads
        finally:
            self.lock.release()

    def status(self):
        # A latched uncertain or active worker must never advertise readiness.
        state = self._status()
        return {'policy': POLICY, 'selection': SELECTION, 'automatic_preparation': False,
                'automatic_restoration': 'Developmental original-input blur/noise suggestion; Auto/On/Off override.',
                'model_qualified': False, 'main_app_default_changed': False,
                'unwired': self.deployment == 'unwired', 'deployment': self.deployment,
                'worker': state}

    def restore(self, contents, input_review='', restoration='auto', preparation=None,
                completion_mask=None, include_bundle=True):
        if not self.lock.acquire(blocking=False):
            raise RestorationBusy('A restoration is already running. Wait for it to finish.')
        started = time.monotonic()
        source_checked = False
        try:
            try:
                reviewed_input(input_review)
                need(restoration in ('auto', 'on', 'off'), 'Choose automatic, on or off restoration.')
                need(type(include_bundle) is bool, 'Explicit evidence download choice required.')
                self._sources(); source_checked = True
                before = self._status()
                canvas, observed, geometry = delivery.prepare_native(contents)
                with Image.open(io.BytesIO(contents)) as image:
                    native = np.array(ImageOps.exif_transpose(image).convert('RGB'))
                if completion_mask is not None:
                    need(isinstance(completion_mask, np.ndarray) and completion_mask.shape == native.shape[:2]
                         and completion_mask.dtype in (np.dtype(bool), np.dtype(np.uint8))
                         and np.isin(completion_mask, (0, 1)).all(), 'Matching binary native removal mask required.')
                    if completion_mask.any():
                        raise CoveringWorkflowRequired('A covering removal area is present. Use the existing reviewed covering workflow; restoration alone does not complete hidden features.')
                need(np.ptp(native.reshape(-1, 3).astype(np.int16), axis=0).max() > 0,
                     'No visible structure remains. Upload a clearer face crop.')
                signals = quality_signals(canvas, (~observed).astype(np.uint8))
                if restoration == 'auto' and signals['blur_variance'] is None:
                    raise ValueError('Auto has too little visible area to assess. Choose On or Off, or upload a clearer crop.')
                applied = restoration == 'on' or (restoration == 'auto' and signals['suggest_restoration'])
                option = {'mode': 'bypass'} if preparation is None else preparation
                need(type(option) is dict and option.get('mode') in ('bypass', 'manual'), 'Explicit preparation mode required.')
                need(set(option) == ({'mode'} if option['mode'] == 'bypass' else
                     {'mode', 'protect', 'apply', 'radius', 'preview_id', 'confirmed'}), 'Exact preparation fields required.')
                payloads = {'uploaded-original.bin': contents, 'original-native.png': png(native),
                            'original-256.png': png(canvas), 'observed-support-256.png': png(observed.astype(np.uint8) * 255)}
                selected, prepared, prep_metadata = contents, canvas, None
                if applied and option['mode'] == 'manual':
                    need(option['confirmed'] is True, 'Confirm the current Protect and Apply preview before restoration.')
                    response, generated = self._preview(contents, option['protect'], option['apply'], option['radius'])
                    need(type(option['preview_id']) is str and option['preview_id'] == response['preview_id'],
                         'The preview is stale. Preview and confirm the current image and both masks again.')
                    need(response['metadata']['no_effect'] or response['metadata']['protected_native_pixels'] > 0,
                         'Protect existing facial blur and soft edges, or bypass preparation.')
                    payloads.update(generated)
                    selected, prep_metadata = generated['derived-native.png'], response['metadata']
                    prepared, derived_support, _ = delivery.prepare_native(selected)
                    need(np.array_equal(observed, derived_support), 'Prepared observed support differs.')
                raw = None
                pixels = canvas.copy()
                worker_metadata = None
                if applied:
                    value = self.worker.run_one(selected, SELECTION)
                    result = delivery.decode_worker_delivery(value, selected, SELECTION, self.models)
                    raw, pixels, prepared = result.raw.copy(), result.pixels.copy(), result.prepared.copy()
                    worker_metadata = copy.deepcopy(value['metadata'])
                    self.worker.count('completed_comparisons')
            finally:
                # Cleanup is mandatory even after input/preview/parser/source validation errors.
                self.worker.release_models()
                if source_checked:
                    self._sources()
            after = self._status()
            expected = 1 if applied else 0
            if not (all(after[key] - before[key] == expected for key in ('neural_forward_attempts', 'neural_forward_returns'))
                    and after['completed_comparisons'] - before['completed_comparisons'] == expected):
                raise RestorationWorkerStateInvalid('Exact0or1 selected-U96 forward and clean completion budget required; do not automatically retry.')
            output_png = png(pixels)
            metadata = {'policy': POLICY, 'selection': SELECTION, 'models': copy.deepcopy(self.models),
                        'sources': copy.deepcopy(self.source_identity), 'input_review': input_review,
                        'restoration_requested': restoration, 'restoration_applied': bool(applied),
                        'quality_signals': signals, 'Auto_assessed_original_canvas_before_preparation': True,
                        'automatic_structure_or_usefulness_qualification': False,
                        'preparation_requested': option['mode'], 'preparation_applied': prep_metadata is not None,
                        'preparation_bypass_reason': ('restoration_off' if restoration == 'off' else
                             'Auto_did_not_suggest_restoration' if not applied else
                             'operator_bypass' if option['mode'] == 'bypass' else None),
                        'preparation': prep_metadata, 'Off_or_Auto_bypass_keeps_original_canvas': not applied,
                        'protection_scope': 'Input preparation only; generated pixels and appearance may change.',
                        'worker_delivery': worker_metadata, 'worker_before': before, 'worker_after_cleanup': after,
                        'neural_forward_attempts': expected, 'neural_forward_returns': expected,
                        'uploaded_bytes_sha256': delivery.sha(contents), 'selected_input_sha256': delivery.sha(selected),
                        'geometry': geometry, 'result_RGB_sha256': delivery.sha(pixels.tobytes()),
                        'raw_RGB_sha256': delivery.sha(raw.tobytes()) if raw is not None else None,
                        'seconds': time.monotonic() - started, 'completion': 'not requested; no nonempty removal area',
                        'model_qualified': False, 'main_app_default_changed': False,
                        'training_optimizer_VM_calls': 0, 'unwired': self.deployment == 'unwired',
                        'deployment': self.deployment}
            payloads.update({'prepared-256.png': png(prepared), 'result.png': output_png})
            if raw is not None:
                stream = io.BytesIO(); np.save(stream, raw, allow_pickle=False)
                payloads['restoration-raw-float32.npy'] = stream.getvalue()
            payloads['restoration.json'] = json_bytes(metadata)
            return {'original_native': immutable(native), 'original256': immutable(canvas),
                    'prepared256': immutable(prepared), 'result': immutable(pixels),
                    'raw': immutable(raw) if raw is not None else None, 'png': output_png,
                    'metadata': metadata, 'bundle': bundle_bytes(payloads) if include_bundle else None}
        finally:
            self.lock.release()
