"""Explicitly sealed local CPU fixed68 comparison; inert until --run.

Preparation/verification never imports Torch. Actual frozen inference is held
behind pinned TRAIN visual closure, terminal-head export and CPU runtime witness.
No app/server changes, optimization, VM connection, retries, resume or FINAL use.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_preparation'
PLAN = PREP / 'fixed_DEV_plan_v1.json'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_run_v1'
COHORT = ROOT / 'outputs/codeformer_spatial_dev_review_v1/cohort_manifest_v1.json'
REUSE = PREP / 'conditional_fixed_DEV_reuse_v1.json'
STAGE = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm'
COHORT_SHA = 'e42afd973305e42422bcb273dc419ab608bbc1437ab332e62d6eb006e0fe9307'
REUSE_SHA = 'a21ba3285ca90abfd0689c71cec06826653e278e1a2b3ea2686eea5029cb5135'
INFERENCE_SHA = '56408052cb13616aa3365cd1bb211c70c54826331d1ca2bb0171a7a6334da803'
R2_SHA = 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
U96_SHA = '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'
HEAD_SHA = '77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17'
HEAD_STATE = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
ARCHIVE_SHA = '7e5208d2ff69713c834c21fb466020ae54d7ee68b84795e2c044ffc11847041e'
PROTOCOL_SHA = '8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9'
ARMS = ('incumbent_U96', 'postfusion_LR_u64')
LIMITS = {'full_forward_attempts': 136, 'full_forward_returns': 136,
          'per_arm_forward_attempts': 68, 'per_arm_forward_returns': 68,
          'internal_seconds': 2400, 'external_seconds': 2460,
          'individual_forward_watchdog_seconds': 90, 'model_load_watchdog_seconds': 240,
          'maximum_output_bytes': 1024 ** 3, 'minimum_free_disk_bytes': 2 * 1024 ** 3,
          'maximum_worker_RSS_bytes': 6 * 1024 ** 3, 'minimum_available_RAM_bytes': 512 * 1024 ** 2}
STATIC = {
    'research/codeformer_u96_postfusion_lr_inference_v1.py': INFERENCE_SHA,
    'research/codeformer_u96_postfusion_v1/model.py': 'ca2b466161932068c01980bb41fcc08e12ee8f0120f17c8fdf47898f35934aa8',
    'research_spatial_restoration_v1.py': '08fa6d4d7d252c0c96d35c87c40ee1035d8e5a68e8ce42d62b1c8b7f721a9437',
    'research_restoration_web.py': 'f27c97e1e8c8db24b3088f73bb8b412083050419742fbe37525938235a62f669',
    'research_filter_restoration_web_v1.py': '98eeb6dc224d986adf792ec9391bd84c3b63a0ed64f3f8ff8b7d31257697b210',
    'third_party/codeformer/codeformer_arch.py': 'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb',
    'third_party/codeformer/vqgan_arch.py': 'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf',
    'third_party/codeformer/registry.py': 'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def strict_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = item
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    result = json.loads(value, object_pairs_hook=pairs, parse_constant=nonfinite)
    def finite(item):
        if isinstance(item, float):
            need(math.isfinite(item), 'Nonfinite JSON exponent')
        elif isinstance(item, dict):
            for child in item.values(): finite(child)
        elif isinstance(item, list):
            for child in item: finite(child)
    finite(result)
    return result


def ordinary(path, cap=512 * 1024 ** 2):
    path = Path(path).absolute()
    need(path.resolve().is_relative_to(ROOT), 'Source/artifact must stay in workspace')
    for part in (path, *path.parents):
        info = part.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'No linked/reparse evidence path')
    info = path.stat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and 0 < info.st_size <= cap,
         'Ordinary single-link bounded file required')
    return path


def pin(path, cap=512 * 1024 ** 2):
    path = ordinary(path, cap); before = path.stat(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''): h.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Evidence changed during hashing')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': before.st_size, 'sha256': h.hexdigest()}


def bound(row):
    need(type(row) is dict and {'path', 'bytes', 'sha256'} <= row.keys(), 'Direct pin required')
    name = row['path']; rel = PurePosixPath(name)
    need(type(name) is str and name == rel.as_posix() and not rel.is_absolute() and
         not any(piece in ('', '.', '..') for piece in name.split('/')) and '\\' not in name and ':' not in name,
         'Canonical workspace-relative pin')
    path = ROOT / name
    need(pin(path) == {key: row[key] for key in ('path', 'bytes', 'sha256')}, 'Exact bound evidence: ' + name)
    return path


def load(path):
    path = ordinary(path, 16 * 1024 ** 2)
    return strict_json(path.read_text(encoding='utf-8'))


def save_exclusive(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve().is_relative_to(OUT):
        size = sum(item.stat().st_size for item in OUT.rglob('*') if item.is_file())
        need(size + len(data) <= LIMITS['maximum_output_bytes'], 'Reserve owned output bytes before write')
    with path.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return pin(path)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def save_json(path, value):
    return save_exclusive(path, json_bytes(value))


def data_contract():
    need('torch' not in sys.modules, 'Data verification must precede Torch import')
    need(pin(COHORT)['sha256'] == COHORT_SHA and pin(REUSE)['sha256'] == REUSE_SHA,
         'Fixed cohort and independent conditional reuse receipt')
    cohort, reuse = load(COHORT), load(REUSE)
    need(reuse['complete'] and reuse['independent_data_reuse_audit_pass'] and
         not reuse['optimization_permitted'] and not reuse['new_training_authorized'], 'Data-only admission')
    need([r['id'] for r in cohort['cases']] == reuse['case_order'] and len(cohort['cases']) == 68,
         'Exact fixed68 order')
    need(Counter(r['cohort'] for r in cohort['cases']) ==
         {'filter_DEV': 36, 'native_QMUL-SurvFace': 8, 'native_ChokePoint': 24}, 'Source-separated counts')
    need(sum(r.get('appearance_flag', False) for r in cohort['cases']) == 9,
         'All nine08025appearance flags retained')
    for row in cohort['cases']:
        need(row['split'] in ('DEV', 'native_DEV') and row['id'] in reuse['case_order'], 'No TRAIN/FINAL inference case')
        for key in ('upload_source', 'input', 'support', 'target', 'source1024', 'pair_record'):
            if row.get(key) is not None: bound(row[key])
    controls = cohort['native_input_only_controls'] + cohort['filter_insufficient_controls']
    need(len(controls) == 40 and len(cohort['native_input_only_controls']) == 16 and
         len(cohort['filter_insufficient_controls']) == 24, 'Same40refusal controls')
    for row in controls:
        need(row['neural_calls_allowed'] == 0 and row['id'] not in reuse['case_order'], 'Zero-call controls remain outside forward list')
        bound(row['input'])
    for name, sha in STATIC.items(): need(pin(ROOT / name)['sha256'] == sha, 'Exact source ' + name)
    need(pin(STAGE / 'protocol.json')['sha256'] == PROTOCOL_SHA, 'Exact sealed LR source protocol')
    protocol = load(STAGE / 'protocol.json')
    model = {key: dict(row, path=(STAGE / row['path']).relative_to(ROOT).as_posix())
             for key, row in protocol['model'].items()}
    for key, sha in (('r2', R2_SHA), ('u96', U96_SHA)):
        need(pin(bound(model[key]))['sha256'] == sha, 'Exact incumbent model ' + key)
    return cohort, reuse, model


def preparation(args):
    cohort, reuse, model = data_contract()
    need(not PLAN.exists(), 'Plan is exclusive; no overwrite/reselection')
    # Required positive receipts are supplied only after the actual work exists.
    receipts = {name: pin(Path(value).absolute()) for name, value in
                (('TRAIN_visual_closure', args.visual_review), ('head_export', args.head_export),
                 ('CPU_runtime_witness', args.runtime_witness), ('CPU_runtime_root_acceptance', args.runtime_acceptance),
                 ('DEV_runner_source_review', args.runner_review),
                 ('frozen_adapter_source_review', args.source_review))}
    docs = {key: load(bound(row)) for key, row in receipts.items()}
    runner_review = docs['DEV_runner_source_review']
    required_runner_sources = {Path(__file__).resolve().relative_to(ROOT).as_posix(),
                               'tests/test_u96_smoothanchors_lr_fixed_DEV_v1.py'}
    need(runner_review['complete'] and runner_review['source_review_pass'] and
         required_runner_sources <= {row['path'] for row in runner_review['sources']},
         'Independent current DEV runner and fixture source review required')
    for row in runner_review['sources']: bound(row)
    visual = docs['TRAIN_visual_closure']
    need(visual['complete'] and visual['all_planned_sheets_reviewed'] and
         visual['sheets_reviewed'] == 30 and visual['generated_PNG_cells_reviewed'] == 712,
         'Actual root all30TRAIN visual closure required')
    visual_manifest = load(bound(visual['visual_manifest']))
    expected_sheets = {row['file']['path']: row['file'] for row in visual_manifest['sheets']}
    need(visual_manifest['complete'] and visual_manifest['planned_sheets'] == 30 and
         visual_manifest['generated_PNG_cells'] == 712 and visual_manifest['all_generated_PNGs_exactly_once'] and
         len(expected_sheets) == len(visual['observations']) == 30 and
         {row['file']['path'] for row in visual['observations']} == set(expected_sheets) and
         sum(row['generated_cells_reviewed'] for row in visual['observations']) == 712,
         'Root closure covers every actual planned sheet and generated cell')
    for row in visual['observations']:
        need(row['personally_viewed_by_root'] and row['file'] == expected_sheets[row['file']['path']],
             'Exact personal all-sheet observation')
        bound(row['file'])
    source_review = docs['frozen_adapter_source_review']
    need(source_review['complete'] and source_review['source_review_pass'] and
         any(row['path'] == 'research/codeformer_u96_postfusion_lr_inference_v1.py' and
             row['sha256'] == INFERENCE_SHA for row in source_review['sources']), 'Frozen adapter independent source review')
    for row in source_review['sources']: bound(row)
    exported = docs['head_export']; witness = docs['CPU_runtime_witness']
    need(exported['complete'] and exported['inference_export_pass'] and
         exported['root_visual_closure'] == receipts['TRAIN_visual_closure'] and
         exported['canonical_head_sha256'] == HEAD_STATE and
         exported['source_archive_sha256'] == ARCHIVE_SHA and exported['head_file']['sha256'] == HEAD_SHA,
         'Exact actually exported terminal36head NPZ required')
    bound(exported['head_file'])
    need(witness['complete'] and witness['runtime_integrity_pass'] and witness['model_qualified'] is False and
         witness['CPU_threads'] == 4 and witness['forward_calls'] == 4 and
         witness['head_file'] == exported['head_file'], 'Actual successful reviewed four-case CPU witness')
    for row in witness['sources']: bound(row)
    accepted = docs['CPU_runtime_root_acceptance']
    need(accepted['complete'] and accepted['runtime_readback_pass'] and
         accepted['accepted_for_fixed_DEV_inference'] is True and accepted['model_qualified'] is False and
         accepted['measured_witness'] == receipts['CPU_runtime_witness'],
         'Root acceptance of actual measured CPU-vs-L4 differences before DEV')
    sources = [pin(Path(__file__).resolve()), *[pin(ROOT / name) for name in STATIC]]
    plan = {'format': 'fixed68-U96-versus-terminal-smoothanchors-LR-u64-CPU-inference-plan-v1',
            'complete': True, 'status': 'SEALED_PREPARED_UNRUN', 'cohort': pin(COHORT), 'data_reuse': pin(REUSE),
            'prerequisites': receipts, 'sources': sources, 'protocol': pin(STAGE / 'protocol.json'),
            'model': model, 'head_file': exported['head_file'],
            'candidate_head_canonical_sha256': HEAD_STATE, 'source_archive_sha256': ARCHIVE_SHA,
            'arms': list(ARMS), 'cases': cohort['cases'], 'case_order': reuse['case_order'],
            'zero_call_controls': cohort['native_input_only_controls'] + cohort['filter_insufficient_controls'],
            'limits': dict(LIMITS), 'observed_canvas_policy': 'original RGB crop neutral128 square pad/BILINEAR256; support NEAREST256',
            'input_policy': 'same observed RGB256 for both arms; float32 CPU bilinear512[-1,1]; w1/detach16/AdaIN; no surroundings preparation',
            'delivery_policy': 'native [-1,1] to [0,1], clamp at512 before bilinear256; preserve outside observed support exactly; floorRAWfloat32*255 PNG',
            'RAW_policy': 'clipped-and-resized float32 delivery before RGB8floor; separately retain unclipped256/native512/logits/LQ',
            'paired_evidence': '36filterDEV alignedsynthetic; optional paired metrics are saved-output-auditor responsibility',
            'native_evidence': '32unpaired; no clean target/PSNR/SSIM/identityaccuracy or Zamboanga claim',
            'CPU_threads': 4, 'runtime_calls': 0, 'optimization_permitted': False, 'automatic_retry_resume': False,
            'model_qualified': False, 'app_default_changed': False, 'FINAL_identity_or_pixel_review': False}
    result = save_json(PLAN, plan)
    print(json.dumps({'complete': True, 'sealed_plan': result, 'planned_forwards': 136, 'neural_training_VM_calls': 0}))


def verify_plan(expected_sha):
    need(pin(PLAN)['sha256'] == expected_sha, 'Explicit exact sealed plan SHA required')
    plan = load(PLAN); cohort, reuse, model = data_contract()
    need(plan['status'] == 'SEALED_PREPARED_UNRUN' and plan['limits'] == LIMITS and plan['arms'] == list(ARMS) and
         plan['cases'] == cohort['cases'] and plan['case_order'] == reuse['case_order'] and plan['model'] == model and
         plan['zero_call_controls'] == cohort['native_input_only_controls'] + cohort['filter_insufficient_controls'],
         'Frozen plan exact data and resource contract')
    need(not plan['optimization_permitted'] and not plan['automatic_retry_resume'] and
         not plan['model_qualified'] and not plan['app_default_changed'], 'Inference-only unchanged defaults')
    for row in [*plan['sources'], *plan['prerequisites'].values(), plan['protocol'], plan['head_file'], plan['cohort'], plan['data_reuse']]: bound(row)
    return plan


def prepare_pixels(row):
    import numpy as np
    from PIL import Image
    with Image.open(bound(row['upload_source'])) as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    h, w = rgb.shape[:2]; side = max(h, w); top, left = (side - h) // 2, (side - w) // 2
    square = Image.new('RGB', (side, side), (128, 128, 128)); square.paste(Image.fromarray(rgb), (left, top))
    support = Image.new('L', (side, side), 0); support.paste(255, (left, top, left + w, top + h))
    prepared = np.asarray(square.resize((256, 256), Image.Resampling.BILINEAR)).copy()
    observed = np.asarray(support.resize((256, 256), Image.Resampling.NEAREST)).copy() > 0
    with Image.open(bound(row['input'])) as image: retained = np.asarray(image.convert('RGB')).copy()
    with Image.open(bound(row['support'])) as image: retained_support = np.asarray(image.convert('L')).copy() > 0
    need(np.array_equal(prepared, retained) and np.array_equal(observed, retained_support) and
         digest(prepared.tobytes()) == row['input_pixels']['RGB8_sha256'] and
         digest(observed.tobytes()) == row['geometry']['observed_sha256'], 'Exact original-derived frozen input/support')
    return rgb, prepared, observed


def outcome(counters, pending):
    unknown = bool(pending or counters['full_forward_attempts'] > counters['full_forward_returns'])
    return {'forward_outcome_unknown': unknown, 'full_forward_attempts': counters['full_forward_attempts'],
            'full_forward_returns': counters['full_forward_returns'], 'completed_cases': counters['completed_cases']}


def worker(expected_sha):
    plan = verify_plan(expected_sha)
    need(not OUT.exists(), 'Fresh exclusive output namespace; no automatic retry/resume')
    OUT.mkdir()
    counters = {'full_forward_attempts': 0, 'full_forward_returns': 0, 'successful_saved_outputs': 0,
                'completed_cases': 0, 'per_arm': {arm: {'attempts': 0, 'returns': 0, 'saved': 0} for arm in ARMS},
                'gradient_queries': 0, 'backward_calls': 0, 'optimizer_constructions': 0, 'optimizer_updates': 0}
    started = time.monotonic(); pending = False; phase = 'data_before_Torch'; active_case = active_arm = None
    def status(label):
        value = {'complete': label == 'complete', 'label': label, 'phase': phase, 'active_case': active_case,
                 'active_arm': active_arm, 'counters': counters, **outcome(counters, pending),
                 'seconds': time.monotonic() - started, 'plan_sha256': expected_sha,
                 'model_qualified': False, 'no_automatic_retry_resume': True}
        path = OUT / 'status.pending'
        need(not path.exists(), 'Interrupted pending status is immutable failure evidence')
        with path.open('xb') as stream:
            stream.write(json_bytes(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(path, OUT / 'status.json')
    def resource(label):
        import psutil
        rss = psutil.Process().memory_info().rss; available = psutil.virtual_memory().available
        free = shutil.disk_usage(ROOT).free
        size = sum(path.stat().st_size for path in OUT.rglob('*') if path.is_file())
        value = {'label': label, 'seconds': time.monotonic() - started, 'worker_RSS_bytes': rss,
                 'available_RAM_bytes': available, 'free_disk_bytes': free, 'output_bytes': size}
        with (OUT / 'resources.jsonl').open('ab') as stream:
            stream.write((json.dumps(value, allow_nan=False) + '\n').encode()); stream.flush()
        need(value['seconds'] <= LIMITS['internal_seconds'] and rss <= LIMITS['maximum_worker_RSS_bytes'] and
             available >= LIMITS['minimum_available_RAM_bytes'] and free >= LIMITS['minimum_free_disk_bytes'] and
             size <= LIMITS['maximum_output_bytes'], 'Finite local CPU resource stop')
        return value
    def watchdog(seconds):
        # This watchdog terminates only this owned worker. Durable reserved
        # counters remain authoritative if native CPU execution is blocked.
        timer = threading.Timer(seconds, lambda: os._exit(124)); timer.daemon = True; timer.start(); return timer
    try:
        status('claimed_before_Torch')
        inputs = [prepare_pixels(row) for row in plan['cases']]
        save_json(OUT / 'controls_zero_calls.json', {'complete': True, 'controls': plan['zero_call_controls'], 'neural_calls': 0})
        save_json(OUT / 'started.json', {'complete': True, 'plan': pin(PLAN), 'runner': pin(Path(__file__).resolve()),
                    'pid': os.getpid(), 'scope': 'local frozen CPU inference only', 'model_qualified': False})
        resource('before_model_load')
        need(ROOT.resolve() == Path.cwd().resolve() and
             (ROOT / 'third_party/codeformer/codeformer_arch.py').is_file(),
             'Validated project root cwd and retained architecture required before imports')
        sys.path.insert(0, str(ROOT))
        import torch
        import numpy as np
        from PIL import Image
        from torch.nn import functional as F
        need(torch.get_default_dtype() == torch.float32, 'FP32 default dtype')
        torch.set_num_threads(4)
        source = importlib.import_module('research.codeformer_u96_postfusion_lr_inference_v1')
        legacy = importlib.import_module('research_spatial_restoration_v1')
        from third_party.codeformer.codeformer_arch import CodeFormer
        phase = 'load_two_separate_frozen_CPU_models'; status('model_load_before')
        timer = watchdog(LIMITS['model_load_watchdog_seconds'])
        try:
            candidate = source.load_frozen_cpu_restorer(bound(plan['model']['r2']), bound(plan['model']['u96']),
                         bound(plan['head_file']), inference_source_sha256=INFERENCE_SHA)
            with torch.no_grad(), torch.device('cpu'), torch.random.fork_rng(devices=[]):
                base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                  connect_list=['32', '64', '128', '256'])
                saved = torch.load(bound(plan['model']['r2']), map_location='cpu', weights_only=True, mmap=True)
                base.load_state_dict(saved['params_ema'], strict=True)
                incumbent = legacy.inference_class(torch)(base).float().eval().requires_grad_(False)
                adapters = torch.load(bound(plan['model']['u96']), map_location='cpu', weights_only=True, mmap=True)
                incumbent.adapters.load_state_dict({k.removeprefix('adapters.'): v for k, v in adapters['adapter_state'].items()}, strict=True)
                del saved, adapters
            need(incumbent.base is not candidate.base, 'Independent fresh bases; no current app model supplied')
        finally:
            timer.cancel()
        def incumbent_invariants():
            need(len(incumbent.state_dict()) == 539 and all(not m.training for m in incumbent.modules()) and
                 all(v.device.type == 'cpu' and v.dtype == torch.float32 and not v.requires_grad and v.grad is None
                     for v in incumbent.parameters()), 'Frozen/eval539incumbent')
            need(legacy.base.state_sha(incumbent.base.state_dict(), torch) == source.R2_STATE and
                 legacy.base.state_sha({'adapters.' + k: v for k, v in incumbent.adapters.state_dict().items()}, torch) == source.U96_STATE,
                 'Incumbent canonical515+24unchanged')
            need(all(not m._forward_hooks and not m._forward_pre_hooks for m in incumbent.modules()), 'Incumbent hooks clean')
        incumbent_invariants(); candidate.validate_invariants()
        models = {ARMS[0]: incumbent, ARMS[1]: candidate}
        record = {'complete': False, 'plan': pin(PLAN), 'cases': [], 'model_qualified': False,
                  'paired_and_native_evidence_separate': True, 'no_optimization': True, 'app_default_changed': False}
        def array_file(path, array):
            array = np.asarray(array)
            need(array.dtype == np.float32 and np.isfinite(array).all(), 'Finite float32 saved array')
            stream = io.BytesIO(); np.save(stream, array, allow_pickle=False)
            return save_exclusive(path, stream.getvalue())
        def png_file(path, image):
            stream = io.BytesIO(); Image.fromarray(image).save(stream, format='PNG')
            return save_exclusive(path, stream.getvalue())
        for index, (row, (original, canvas, observed)) in enumerate(zip(plan['cases'], inputs)):
            active_case = row['id']; folder = OUT / 'cases' / row['id']; folder.mkdir(parents=True)
            common = {'original': png_file(folder / 'original.png', original),
                      'prepared': png_file(folder / 'prepared.png', canvas),
                      'support': png_file(folder / 'observed_support.png', observed.astype(np.uint8) * 255)}
            if row['target']:
                common['target'] = save_exclusive(folder / 'target.png', bound(row['target']).read_bytes())
            case = {'id': row['id'], 'cohort': row['cohort'], 'split': row['split'], 'input_review': row['input_review'],
                    'appearance_flag': row.get('appearance_flag', False), 'geometry': row['geometry'], 'inputs': common,
                    'paired_target': row['target'], 'outputs': {}, 'native_PSNR_SSIM_identity_metrics': None,
                    'model_qualified': False}
            tensor = torch.from_numpy(canvas.astype(np.float32).transpose(2, 0, 1).copy())[None] / 255
            sample = F.interpolate(tensor, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
            for arm in ARMS:
                active_arm = arm; phase = 'before_forward'; resource(arm + '_before'); status('before_reservation')
                need(counters['full_forward_attempts'] < 136 and counters['per_arm'][arm]['attempts'] < 68,
                     'Bounded exactly136/68forward reservation')
                counters['full_forward_attempts'] += 1; counters['per_arm'][arm]['attempts'] += 1; pending = True
                phase = 'forward_reserved'; status('forward_reserved')
                timer = watchdog(LIMITS['individual_forward_watchdog_seconds'])
                try:
                    with torch.no_grad():
                        triple = models[arm](sample, w=1., detach_16=True, adain=True)
                finally:
                    timer.cancel()
                counters['full_forward_returns'] += 1; counters['per_arm'][arm]['returns'] += 1; pending = False
                phase = 'returned_before_output_validation'; status('forward_returned')
                need(isinstance(triple, tuple) and len(triple) == 3, 'Native/logits/LQ output tuple')
                for value, shape in zip(triple, ((1, 3, 512, 512), (1, 256, 1024), (1, 256, 16, 16))):
                    need(value.shape == shape and value.dtype == torch.float32 and value.device.type == 'cpu' and
                         not value.requires_grad and torch.isfinite(value).all().item(), 'Exact finite CPU output schema')
                with torch.no_grad():
                    native, logits, lq = triple
                    unclipped = F.interpolate((native + 1) / 2, (256, 256), mode='bilinear', align_corners=False)
                    raw = F.interpolate(((native + 1) / 2).clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
                raw_array = raw[0].permute(1, 2, 0).contiguous().numpy().copy()
                # This exact retained observed-support policy matches the local
                # app, preserving padding before floor delivery. Save native and
                # unclipped arrays independently so the auditor can reconstruct.
                raw_array[~observed] = canvas.astype(np.float32)[~observed] / 255
                delivered = np.floor(raw_array * np.float32(255)).astype(np.uint8)
                armfolder = folder / arm; armfolder.mkdir()
                output = {'native512': array_file(armfolder / 'native512.npy', native.numpy()),
                          'logits': array_file(armfolder / 'logits.npy', logits.numpy()),
                          'LQ': array_file(armfolder / 'LQ.npy', lq.numpy()),
                          'unclipped': array_file(armfolder / 'unclipped256.npy', unclipped[0].permute(1, 2, 0).contiguous().numpy()),
                          'raw': array_file(armfolder / 'raw.npy', raw_array),
                          'png': png_file(armfolder / 'output.png', delivered)}
                incumbent_invariants(); candidate.validate_invariants()
                need(models[arm].last_forward_sites == ('32', '64', '128', '256'), 'All four sites executed')
                output['provenance'] = {'arm': arm, 'R2_file_sha256': R2_SHA, 'U96_file_sha256': U96_SHA,
                       'head_file_sha256': HEAD_SHA if arm == ARMS[1] else None, 'head_canonical_sha256': HEAD_STATE if arm == ARMS[1] else None,
                       'w': 1., 'detach16': True, 'AdaIN': True, 'CPU_FP32': True, 'display_processing': 'none',
                       'source_candidate_training_updates': 64 if arm == ARMS[1] else None, 'model_qualified': False,
                       'same_observed_geometry': True, 'outside_support_exact_observed_pixels': True,
                       'forward_sites': list(models[arm].last_forward_sites)}
                case['outputs'][arm] = output
                save_json(armfolder / 'record.json', {'complete': True, 'case_id': row['id'], **output, 'counters': counters})
                counters['successful_saved_outputs'] += 1; counters['per_arm'][arm]['saved'] += 1
                phase = 'output_fully_saved'; status('output_fully_saved'); resource(arm + '_saved')
                del triple, native, logits, lq, raw, unclipped
            active_arm = None
            case['complete'] = True; save_json(folder / 'case.json', case); record['cases'].append(case)
            counters['completed_cases'] += 1
            phase = 'case_complete'; status('case_complete')
            print(json.dumps({'case': index + 1, 'of': 68, 'id': row['id'], 'forwards': counters['full_forward_returns']}), flush=True)
        need(counters['full_forward_attempts'] == counters['full_forward_returns'] == counters['successful_saved_outputs'] == 136 and
             counters['completed_cases'] == 68, 'Exact complete bounded136output trajectory')
        phase = 'final_integrity'; incumbent_invariants(); candidate.validate_invariants(); resource('final')
        for row in [*plan['sources'], plan['head_file'], *plan['model'].values()]: bound(row)
        record.update({'complete': True, 'status': 'SAVED_FIXED_DEV_OUTPUTS_PENDING_INDEPENDENT_AUDIT_VISUAL_REVIEW',
                       'counters': counters, 'seconds': time.monotonic() - started, 'model_qualified': False,
                       'qualification_pending': ['saved_pixel_audit', 'all_visual_sheets', 'source_separated_review', 'full_DEV_FINAL_user_acceptance']})
        save_json(OUT / 'comparison.json', record); phase = 'complete'; status('complete')
        return 0
    except BaseException as error:
        failure = {'complete': True, 'status': 'LOCAL_INFERENCE_STOP_NOT_QUALIFIED', 'phase': phase,
                   'active_case': active_case, 'active_arm': active_arm, 'counters': counters, **outcome(counters, pending),
                   'error': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(),
                   'seconds': time.monotonic() - started, 'automatic_retry_resume': False, 'model_qualified': False}
        if not (OUT / 'failure.json').exists(): save_json(OUT / 'failure.json', failure)
        # A failed atomic write is evidence in its own right. Retain those exact
        # pending bytes rather than overwriting them with a later failure status.
        if not (OUT / 'status.pending').exists():
            status('failed_retaining_partial_evidence')
        raise


def supervise(expected_sha):
    verify_plan(expected_sha)
    need(not OUT.exists(), 'Fresh output namespace; no rerun/resume')
    # Worker alone claims OUT. The supervisor log stays outside that namespace.
    log = PREP / 'fixed_DEV_worker_v1.log'; receipt = PREP / 'fixed_DEV_supervisor_v1.json'
    need(not log.exists() and not receipt.exists(), 'Exclusive supervisor evidence')
    start = time.monotonic(); timed_out = False
    with log.open('xb') as stream:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--plan-sha', expected_sha],
                                   cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=LIMITS['external_seconds'])
        except subprocess.TimeoutExpired:
            timed_out = True; process.terminate()
            try: code = process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); code = process.wait(timeout=10)
    status = load(OUT / 'status.json') if (OUT / 'status.json').exists() else None
    value = {'complete': True, 'worker_exit': code, 'timeout': timed_out, 'seconds': time.monotonic() - start,
             'plan_sha256': expected_sha, 'last_durable_status': pin(OUT / 'status.json') if status else None,
             'status_pending_present': (OUT / 'status.pending').exists(),
             'counters': status['counters'] if status else None,
             'forward_outcome_unknown': True if status is None else
                 bool(status['forward_outcome_unknown'] or (OUT / 'status.pending').exists()),
             'model_qualified': False, 'no_automatic_retry_resume': True}
    save_json(receipt, value)
    print(json.dumps(value))
    return 0 if code == 0 and not timed_out else 2


def self_check():
    for bad in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
        try: strict_json(bad)
        except ValueError: pass
        else: raise AssertionError('Unsafe JSON accepted')
    need(outcome({'full_forward_attempts': 1, 'full_forward_returns': 0, 'completed_cases': 0}, False)['forward_outcome_unknown'], 'Reserved call unknown')
    need(not outcome({'full_forward_attempts': 1, 'full_forward_returns': 1, 'completed_cases': 0}, False)['forward_outcome_unknown'], 'Known returned call')
    need(LIMITS['full_forward_attempts'] == len(ARMS) * 68 == 136 and 'torch' not in sys.modules, 'Pure fixture no model')
    print(json.dumps({'complete': True, 'pure_fixtures_pass': True, 'neural_training_VM_calls': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-plan', action='store_true')
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--worker', action='store_true')
    mode.add_argument('--self-check', action='store_true')
    parser.add_argument('--plan-sha'); parser.add_argument('--visual-review'); parser.add_argument('--head-export')
    parser.add_argument('--runtime-witness'); parser.add_argument('--runtime-acceptance'); parser.add_argument('--source-review')
    parser.add_argument('--runner-review')
    args = parser.parse_args()
    if args.self_check: return self_check()
    if args.prepare_plan:
        need(all((args.visual_review, args.head_export, args.runtime_witness, args.runtime_acceptance, args.source_review, args.runner_review)), 'Six actual prerequisite receipts required')
        return preparation(args)
    need(type(args.plan_sha) is str and len(args.plan_sha) == 64, 'Plan SHA required')
    if args.verify_only:
        verify_plan(args.plan_sha)
        print(json.dumps({'complete': True, 'cases': 68, 'zero_call_controls': 40, 'planned_forwards': 136, 'Torch_model_training_VM_calls': 0}))
        return 0
    if args.worker: return worker(args.plan_sha)
    return supervise(args.plan_sha)


if __name__ == '__main__':
    raise SystemExit(main())
