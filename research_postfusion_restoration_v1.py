"""Separate opt-in U96/u64 CPU comparison; import/status/refusal are non-neural.

No main-app import, default switch, completion, training or automatic context
processing. A reviewed manifest is configured by the separate launcher only.
"""
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import threading
import time
import zipfile

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

import research_restoration_web as base
import research_spatial_restoration_v1 as spatial
from research_filter_restoration_web_v1 import review_input, decode_crop, prepare_crop, delivered_views
from research import codeformer_u96_postfusion_lr_inference_v1 as frozen

ROOT = Path(__file__).resolve().parent
ROUTE = '/restoration-postfusion-test'
ARMS = ('incumbent_U96', 'postfusion_LR_u64')
LABELS = {'incumbent_U96': 'U96 · incumbent · experimental',
          'postfusion_LR_u64': 'Post-fusion LR u64 · experimental'}
QUALITY_STATUS = ('Experimental comparison only. U96 remains the selected development model. '
                  'The new u64 passed a small TRAIN screen; broader usefulness and independent final review remain unresolved.')
LIMIT = 10 * 1024 ** 2
BUNDLE_LIMIT = 64 * 1024 ** 2
INFERENCE_SHA = '56408052cb13616aa3365cd1bb211c70c54826331d1ca2bb0171a7a6334da803'
STATIC = {
    'research/codeformer_u96_postfusion_lr_inference_v1.py': INFERENCE_SHA,
    'research_spatial_restoration_v1.py': '08fa6d4d7d252c0c96d35c87c40ee1035d8e5a68e8ce42d62b1c8b7f721a9437',
    'research_restoration_web.py': 'f27c97e1e8c8db24b3088f73bb8b412083050419742fbe37525938235a62f669',
    'research_filter_restoration_web_v1.py': '98eeb6dc224d986adf792ec9391bd84c3b63a0ed64f3f8ff8b7d31257697b210'}
PROOFS = {
    'export': ('outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_inference_export/inference_export_v1.json', 'fc94a63ac620ed4f7dd54e8c59defbf09f1266bbdc43190ca70c55fe0e9e8d6d'),
    'visual': ('outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_audit/root_visual_review_v1.json', '99072d5b112fa8fceccf2cf24f83c5a2d64b67ef1aa7bba003c73b3434be7cdb'),
    'runtime': ('outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_cpu_runtime_r1/cpu_runtime_witness_v1.json', '4c390272cc01437bde491d0e4d7aa61af57b74bc3e4ca3cfe179febfb32317e0'),
    'runtime_readback': ('outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_cpu_runtime_r1/runtime_independent_readback_v1.json', 'a65bfa4495244406f6eefd76215d3da7861782bb4ccafe4888384d0d681d665f'),
    'runtime_acceptance': ('outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_cpu_runtime_r1/root_runtime_acceptance_v1.json', 'b0f2bdf65838ae728834e685f9e7a1561f24fbdf61730dacf296cc58f7daecfa'),
    'inference_review': ('outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_inference_preparation/independent_source_review_v1_r1.json', '9dee2b5f2a17f4e015ff140538b05d159d9f683edb3fdc8e33c071fc11d620cd')}
OWN_SOURCES = ('research_postfusion_restoration_v1.py', 'serve_codeformer_postfusion_research_comparison_v1.py',
               'templates/research_postfusion_restoration_v1.html', 'static/research_postfusion_restoration_v1.js')
router = APIRouter(prefix=ROUTE, tags=['Experimental post-fusion restoration'])
_configured = None


def need(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'Duplicate comparison metadata key.')
            out[key] = value
        return out
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError('Nonfinite comparison metadata.')))
    def finite(item):
        if isinstance(item, float):
            need(math.isfinite(item), 'Nonfinite comparison metadata exponent.')
        elif isinstance(item, dict):
            for child in item.values(): finite(child)
        elif isinstance(item, list):
            for child in item: finite(child)
    finite(value)
    return value


def pin(path, cap=512 * 1024 ** 2):
    path = Path(path).absolute()
    need(path.is_relative_to(ROOT), 'Comparison evidence must stay in this workspace.')
    for part in (path, *path.parents):
        info = part.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Ordinary comparison evidence paths required.')
    before = path.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= cap,
         'Bounded ordinary comparison evidence required.')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''): h.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Comparison evidence changed during reading.')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': before.st_size, 'sha256': h.hexdigest()}


def bound(row, cap=512 * 1024 ** 2):
    need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'Exact direct comparison evidence pin required.')
    name = row['path']
    need(type(name) is str and name == PurePosixPath(name).as_posix() and not PurePosixPath(name).is_absolute()
         and not any(x in ('', '.', '..') for x in name.split('/')) and '\\' not in name and ':' not in name,
         'Canonical workspace-relative comparison path required.')
    need(type(row['bytes']) is int and row['bytes'] > 0 and type(row['sha256']) is str
         and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None, 'Exact comparison size and digest required.')
    path = ROOT / name
    need(pin(path, cap) == row, 'Reviewed comparison evidence changed: ' + name)
    return path


def read(row, cap=8 * 1024 ** 2):
    return strict_json(bound(row, cap).read_text(encoding='utf-8'))


def manifest():
    need(_configured is not None, 'Experimental page awaits its independent source review and exact test manifest.')
    value = read(_configured, 1024 ** 2)
    need(type(value) is dict and value.get('format') == 'U96-terminal-LR-opt-in-comparison-v1'
         and value.get('complete') is True and value.get('accepted_for_experimental_user_comparison') is True
         and value.get('model_qualified') is False and value.get('main_app_default_changed') is False
         and value.get('training_optimizer_VM_calls') == 0, 'Experimental page authorization must preserve research status.')
    need(value.get('protocol_sha256') == frozen.PROTOCOL_SHA and value.get('archive_sha256') == frozen.ARCHIVE_SHA
         and value.get('canonical_head_sha256') == frozen.HEAD_STATE, 'Exact audited terminal state required.')
    source_rows = value.get('sources')
    need(type(source_rows) is dict and set(source_rows) == set(OWN_SOURCES) | set(STATIC), 'Exact comparison source namespace required.')
    for name, row in source_rows.items():
        need(row.get('path') == name, 'Comparison source path/key mismatch.')
        bound(row, 1024 ** 2)
        if name in STATIC: need(row['sha256'] == STATIC[name], 'Retained inference utility source changed.')
    proofs = value.get('proofs')
    need(type(proofs) is dict and set(proofs) == set(PROOFS) | {'page_source_review'}, 'All completed comparison prerequisites required.')
    actual = {}
    for key, row in proofs.items():
        if key in PROOFS:
            need(row.get('path') == PROOFS[key][0] and row.get('sha256') == PROOFS[key][1], 'Exact completed proof required: ' + key)
        actual[key] = read(row)
        need(actual[key].get('complete') is True, 'Incomplete comparison proof: ' + key)
    need(actual['export'].get('inference_export_pass') is True
         and actual['export'].get('canonical_head_sha256') == frozen.HEAD_STATE
         and actual['export'].get('source_update') == 64, 'Exact exported terminal heads required.')
    need(actual['visual'].get('all_planned_sheets_reviewed') is True
         and actual['visual'].get('sheets_reviewed') == 30
         and actual['visual'].get('generated_PNG_cells_reviewed') == 712, 'All actual TRAIN sheets require root visual review.')
    need(actual['runtime'].get('runtime_integrity_pass') is True
         and actual['runtime_readback'].get('independent_runtime_readback_pass') is True
         and actual['runtime_acceptance'].get('runtime_readback_pass') is True
         and actual['runtime_acceptance'].get('model_qualified') is False
         and actual['runtime_acceptance'].get('measured_witness') == proofs['runtime']
         and actual['runtime_acceptance'].get('independent_readback') == proofs['runtime_readback'], 'Reviewed CPU runtime correspondence required.')
    for key in ('inference_review', 'page_source_review'):
        review = actual[key]
        need(review.get('source_review_pass') is True, 'Independent source review must pass: ' + key)
        rows = {row['path']: row for row in review.get('sources', [])}
        required = ('research/codeformer_u96_postfusion_lr_inference_v1.py',) if key == 'inference_review' else OWN_SOURCES
        need(all(rows.get(name) == source_rows[name] for name in required), 'Independent source review must bind current source bytes.')
    models = value.get('models')
    need(type(models) is dict and set(models) == {'r2', 'u96', 'heads'}, 'Three exact frozen artifact pins required.')
    for name, expected in (('r2', frozen.R2_FILE), ('u96', frozen.U96_FILE), ('heads', frozen.HEAD_FILE)):
        need({k: models[name].get(k) for k in ('bytes', 'sha256')} == expected, 'Exact frozen artifact value: ' + name)
        bound(models[name])
    need(models['heads'] == actual['export']['head_file'], 'Use the unchanged exported head NPZ.')
    return value


def configure_manifest(path, expected_sha):
    """Launcher-only binding; the page cannot select a different checkpoint."""
    global _configured
    need(_configured is None and not engine.models, 'Restart the separate server to change its manifest.')
    row = pin(path, 1024 ** 2)
    need(type(expected_sha) is str and row['sha256'] == expected_sha, 'Exact experimental test-manifest digest required.')
    _configured = row
    try: return manifest()
    except BaseException:
        _configured = None
        raise


class Comparison:
    def __init__(self):
        self.lock = threading.Lock()
        self.counter_lock = threading.Lock()
        self.incumbent_loader = spatial.Comparison()  # Distinct instance; never spatial.engine.
        self.models = {}
        self.counters = {'neural_forward_attempts': 0, 'neural_forward_returns': 0,
                         'completed_comparisons': 0, 'operator_review_refusals': 0}

    def count(self, name):
        with self.counter_lock: self.counters[name] += 1

    def measured_status(self):
        with self.counter_lock: return dict(self.counters)

    def load(self, reviewed, torch):
        if self.models: return self.models
        need(torch.get_default_dtype() == torch.float32, 'FP32 default dtype required.')
        with torch.no_grad(), torch.device('cpu'), torch.random.fork_rng(devices=[]):
            incumbent, _ = self.incumbent_loader.load('spatial_u', torch)
            candidate = frozen.load_frozen_cpu_restorer(*(bound(reviewed['models'][name]) for name in ('r2', 'u96', 'heads')),
                                                        inference_source_sha256=INFERENCE_SHA)
        need(incumbent.base is not candidate.base, 'Separate owned incumbent and candidate bases required.')
        self.models = {ARMS[0]: incumbent, ARMS[1]: candidate}
        self.invariants(torch)
        return self.models

    def invariants(self, torch):
        incumbent = self.models[ARMS[0]]
        need(len(incumbent.state_dict()) == 539 and all(not m.training for m in incumbent.modules())
             and all(not p.requires_grad and p.grad is None and p.device.type == 'cpu' and p.dtype == torch.float32
                     for p in incumbent.parameters()), 'Frozen/eval CPU FP32 complete U96 required.')
        need(base.state_sha(incumbent.base.state_dict(), torch) == frozen.R2_STATE
             and base.state_sha({'adapters.' + k: v for k, v in incumbent.adapters.state_dict().items()}, torch) == frozen.U96_STATE,
             'Incumbent canonical values must remain exact.')
        need(all(not m._forward_hooks and not m._forward_pre_hooks for m in incumbent.modules()), 'Incumbent hooks must clean up.')
        self.models[ARMS[1]].validate_invariants()

    def generate(self, contents, input_review):
        try: review_input(input_review)
        except ValueError:
            self.count('operator_review_refusals')
            raise
        rgb = decode_crop(contents)
        canvas, observed, _, geometry = prepare_crop(rgb)
        if not self.lock.acquire(blocking=False): raise RuntimeError('A comparison is already running. Wait for it to finish.')
        started = time.monotonic()
        try:
            reviewed = manifest()  # Before Torch/model loading; every request rechecks pinned artifacts.
            import torch
            from torch.nn import functional as F
            torch.set_num_threads(4)
            need(torch.get_default_dtype() == torch.float32, 'CPU FP32 inference required.')
            models = self.load(reviewed, torch)
            x = torch.from_numpy((canvas.astype(np.float32) / np.float32(255)).copy()).permute(2, 0, 1)[None]
            outputs = {}
            attempts = returns = 0
            with torch.no_grad(), torch.device('cpu'), torch.autocast(device_type='cpu', enabled=False):
                sample = F.interpolate(x, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
                for arm in ARMS:
                    self.invariants(torch)
                    need(attempts < 2, 'At most two full inference forwards per comparison.')
                    self.count('neural_forward_attempts'); attempts += 1
                    native = models[arm](sample, w=1., detach_16=True, adain=True)[0]
                    self.count('neural_forward_returns'); returns += 1
                    need(tuple(native.shape) == (1, 3, 512, 512) and native.dtype == torch.float32
                         and native.device.type == 'cpu' and not native.requires_grad
                         and torch.isfinite(native).all().item(), 'Finite frozen singleton output required.')
                    self.invariants(torch)
                    need(models[arm].last_forward_sites == spatial.SITES, 'All four ordered feature sites required.')
                    raw = F.interpolate(((native + 1) / 2).clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
                    raw = raw[0].permute(1, 2, 0).contiguous().numpy().copy()
                    raw, pixels = delivered_views(raw, canvas, observed)
                    encoded = base.png(pixels)
                    outputs[arm] = {'raw': raw, 'PNG': encoded, 'label': LABELS[arm],
                                    'raw_rgb_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                                    'PNG_RGB8_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                                    'PNG_file_sha256': hashlib.sha256(encoded).hexdigest()}
            need(attempts == returns == 2 and set(outputs) == set(ARMS), 'Exactly two completed model outputs required.')
            metadata = {'policy': 'opt-in-U96-postfusion-LR-comparison-v1', 'input_review': input_review,
                'geometry': geometry, 'uploaded_bytes_sha256': hashlib.sha256(contents).hexdigest(),
                'test_manifest': dict(_configured), 'quality_status': QUALITY_STATUS, 'device': 'cpu', 'CPU_threads': 4,
                'neural_forward_attempts': attempts, 'neural_forward_returns': returns,
                'model_qualified': False, 'main_app_default_changed': False, 'preferred_development_model': 'U96',
                'inference_only': True, 'training_optimizer_VM_calls': 0, 'no_completion_or_covering_removal': True,
                'automatic_context_blur_or_noise_applied': False, 'display_processing': 'none', 'alignment': 'none',
                'internal_resolution': 512, 'output_resolution': 256, 'fidelity': 1., 'adain': True,
                'manual_review_is_not_a_validated_automatic_information_threshold': True,
                'RAW_float_definition': 'Native RGB clipped to [0,1], bilinear-resized to256; support-composed floats before floor RGB8 quantization.',
                'outside_observed_support_RAW_and_PNG_match_input_exact': True,
                'native_input_has_no_aligned_clean_reference_in_this_flow': True,
                'architecture': 'CodeFormer by S-Lab; our U96 and post-fusion trained adaptations',
                'license': 'S-Lab License1.0; third_party/codeformer/LICENSE',
                'canonical_states': {'R2_515': frozen.R2_STATE, 'U96_24': frozen.U96_STATE, 'u64_heads_36': frozen.HEAD_STATE},
                'models': reviewed['models'], 'seconds': time.monotonic() - started,
                'outputs': {a: {k: v for k, v in row.items() if k not in ('raw', 'PNG')} for a, row in outputs.items()}}
            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('original-upload.png', base.png(rgb)); z.writestr('resize-control.png', base.png(canvas))
                z.writestr('observed-support.png', base.png(observed.astype(np.uint8) * 255))
                for arm, row in outputs.items():
                    z.writestr(arm + '.png', row['PNG'])
                    data = io.BytesIO(); np.save(data, row['raw'], allow_pickle=False)
                    z.writestr(arm + '-raw.npy', data.getvalue())
                z.writestr('processing.json', json.dumps(metadata, indent=2, allow_nan=False) + '\n')
            need(bundle.tell() <= BUNDLE_LIMIT, 'Comparison bundle exceeds64MiB; use a smaller native crop.')
            self.count('completed_comparisons')
            return {'original': base.data_url(base.png(canvas)), 'outputs': {a: base.data_url(row['PNG']) for a, row in outputs.items()},
                    'metadata': metadata, 'bundle': base.data_url(bundle.getvalue(), 'application/zip')}
        finally: self.lock.release()


engine = Comparison()


@router.get('', response_class=HTMLResponse)
def page():
    return HTMLResponse((ROOT / OWN_SOURCES[2]).read_text(encoding='utf-8'), headers={'Cache-Control': 'no-store'})


@router.get('/status')
def status():
    result = {'model_qualified': False, 'main_app_default_changed': False, 'quality_status': QUALITY_STATUS,
              'manual_input_review_required': True, 'route': ROUTE, 'maximum_forwards_per_request': 2,
              'models': {arm: {'label': LABELS[arm], 'available': False} for arm in ARMS}, **engine.measured_status()}
    try:
        manifest()
        for row in result['models'].values(): row['available'] = True
        result['binding_status'] = 'Ready for an experimental comparison of a cropped face with readable visible structure.'
    except (ValueError, OSError) as error: result['binding_status'] = str(error)
    return result


@router.post('/compare')
async def compare(file: UploadFile = File(...), input_review: str = Form('')):
    try: return await run_in_threadpool(engine.generate, await file.read(LIMIT + 1), input_review)
    except ValueError as error: raise HTTPException(422, str(error)) from error
    except OSError as error: raise HTTPException(503, 'The reviewed experimental artifact is unavailable.') from error
    except RuntimeError as error: raise HTTPException(409, str(error)) from error
