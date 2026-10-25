"""Finite image-only TRAIN oracle input set; no neural imports or launch.

The plan is written and fsynced before image rendering. New strong inputs need
root input-only review; no training or inference admission is granted here.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import stat
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/codeformer_filter_fusion_training_v1_return'
OUT = ROOT / 'outputs/codeformer_hq_code_oracle_inputs_20261019_v1'
SOURCES = tuple('tr_ffhq_' + number for number in ('03761', '55791', '60978', '34217', '31449', '52422'))
PROFILES = ('clear', 'moderate_existing_gaussian_2p4', 'strong_fixed_blur3p2_lr64_jpeg50')
MAX_SECONDS = 180
MAX_FILES = 38
MAX_BYTES = 32 * 1024**2
START = time.monotonic()


def require(value, message):
    if not value:
        raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    require(path.resolve(strict=False).is_relative_to(ROOT), 'Workspace path required')
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            row = item.lstat()
            require(not stat.S_ISLNK(row.st_mode) and not getattr(row, 'st_file_attributes', 0) & 1024,
                    'Linked/reparse path refused')
    if path.exists():
        row = path.stat()
        require(stat.S_ISREG(row.st_mode) and row.st_nlink == 1, 'Ordinary unique file required')
    return path


def checked(path, expected=None):
    require(time.monotonic() - START < MAX_SECONDS, 'Finite180s image-preparation cap')
    path = regular(path); before = path.stat()
    payload = path.read_bytes(); after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns),
            'Source changed during read')
    pin = {'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'bytes': len(payload),
           'sha256': hashlib.sha256(payload).hexdigest()}
    if expected:
        require(all(pin[key] == expected[key] for key in ('bytes', 'sha256')), 'Frozen source pin differs')
    return payload, pin


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def save_bytes(name, payload):
    path = regular(OUT / name); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return checked(path)[1]


def save_json(name, value):
    return save_bytes(name, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def save_png(name, image):
    stream = io.BytesIO(); image.save(stream, format='PNG')
    pin = save_bytes(name, stream.getvalue())
    pixels = np.asarray(image)
    return {**pin, 'size': list(image.size), 'mode': image.mode,
            'RGB8_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()}


def decoded(payload, size):
    with Image.open(io.BytesIO(payload)) as image:
        require(image.mode == 'RGB' and image.size == size, 'Exact RGB source geometry required')
        return image.copy()


def main():
    require(not OUT.exists(), 'Preserve prior or partial input set; never overwrite or resume')
    require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules), 'Neural-free producer required')
    documents = {}; dependencies = {}
    for name in ('data_bundle.json', 'data_admission_v2.json', 'data_schedule_v2.json', 'evaluation_bundle.json',
                 'provenance/reference_manifest.json', 'provenance/data_source_hold_v2.json',
                 'provenance/candidate_holds_v1.json', 'root_dark_linear_input_review_v2.json'):
        payload, dependencies[name] = checked(RETURN / name)
        documents[name] = json.loads(payload)
    bundle = documents['data_bundle.json']; admission = documents['data_admission_v2.json']
    schedule = documents['data_schedule_v2.json']; evaluation = documents['evaluation_bundle.json']
    refs = {row['id']: row for row in documents['provenance/reference_manifest.json']['references']}
    require(bundle['complete'] and admission['complete'] and admission['root_approved'], 'Existing final admission required')
    require(schedule['anchor_source_order'][:6] == list(SOURCES), 'Exact requested source order must match existing TRAIN guards')
    require(len(schedule['anchor_source_order']) == 55 and all(source in schedule['anchor_source_order'] for source in SOURCES),
            'All six fixed TRAIN photographs must belong to existing55; do not replace')
    require(not set(SOURCES) & set(schedule['whole_source_scope_holds']), 'Held source cannot enter diagnostic')
    admitted = {(row['role'], row['id']) for row in admission['admitted_role_cases']}
    require(evaluation['reserved_FINAL_used'] is False and evaluation['historical_DEV_native_unpaired'] == 48,
            'Preserve FINAL and native evaluation scope')
    dev_photos = {row['sourcebindings']['pair_record']['reference_id'] for row in evaluation['cases']
                  if row['cohort'] == 'historical_DEV406' and 'pair_record' in row['sourcebindings']}
    require(set(SOURCES).isdisjoint(dev_photos), 'TRAIN source intersects historical FFHQ DEV')
    frozen = []; buffers = {}
    for source in SOURCES:
        reference = refs[source]
        require(reference['historical_role'] == 'train', 'Exact project TRAIN role required')
        clear = bundle['case_by_id'].get(source + '__clear')
        moderate = bundle['case_by_id'].get(source + '__gaussian_2p4')
        require(clear is not None and moderate is not None and clear['split'] == moderate['split'] == 'TRAIN',
                'Both exact existing TRAIN profiles required; missing input must stop')
        require(('clear', clear['id']) in admitted and
                any((role, moderate['id']) in admitted for role in ('filter_a', 'filter_b')),
                'Both selected inputs must be separately admitted already')
        pair_payload, pair_pin = checked(ROOT / clear['pair_record']['path'], clear['pair_record'])
        pair = json.loads(pair_payload)
        source_payload, source_pin = checked(ROOT / reference['original_source_path'], pair['source1024'])
        require(source_pin['sha256'] == reference['source_sha256'], 'Exact source photograph fingerprint required')
        clear_payload, clear_pin = checked(RETURN / clear['input']['path'], clear['input'])
        moderate_payload, moderate_pin = checked(RETURN / moderate['input']['path'], moderate['input'])
        target_payload, target_pin = checked(RETURN / clear['target256']['path'], clear['target256'])
        buffers[source] = (source_payload, clear_payload, moderate_payload, target_payload)
        frozen.append({'reference_id': source, 'source1024': source_pin, 'existing_clear': clear_pin,
            'existing_moderate': moderate_pin, 'existing_metrics_target256': target_pin, 'clear_pair': pair_pin,
            'clear_RGB8_sha256': clear['input_pixels']['RGB8_sha256'],
            'moderate_RGB8_sha256': moderate['input_pixels']['RGB8_sha256'],
            'metrics_target_RGB8_sha256': clear['target256_pixels']['RGB8_sha256'],
            'attribution': reference['attribution'], 'project_role': 'TRAIN',
            'official_FFHQ_category': reference['official_category']})
    producer = checked(__file__)[1]
    plan = {'format': 'codeformer-HQ-code-oracle-image-only-plan-v1', 'complete': True,
        'frozen_before_rendering': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'source_order': list(SOURCES), 'profile_order_for_rendering': list(PROFILES),
        'root_input_only_review_order': ['strong_fixed_blur3p2_lr64_jpeg50', 'moderate_existing_gaussian_2p4', 'clear'],
        'dependencies': dependencies, 'frozen_sources': frozen, 'producer': producer,
        'strong_recipe': {'base': 'Exact existing source1024->Pillow BILINEAR256 clear RGB pixels',
            'operations': ['Pillow ImageFilter.GaussianBlur(radius=3.2)', 'Pillow BILINEAR resize64x64',
                'Pillow BILINEAR resize256x256', 'JPEG quality50 subsampling2 optimizeFalse progressiveFalse',
                'Decode JPEG to RGB; save exact PNG'], 'noise': None, 'seed_search': False,
            'fixed_synthetic_proxy_not_calibrated_CCTV': True},
        'HQ_teacher_target': 'Direct sourceRGB1024->Pillow LANCZOS512; no256 upscale',
        'metrics_target': 'Direct sourceRGB1024->Pillow LANCZOS256; must equal retained exact target pixels',
        'planned_counts': {'source_photographs': 6, 'input_cases': 18, 'HQ512_targets': 6,
            'metrics256_targets': 6, 'input_only_sheets': 3, 'source1024_payload_copies': 0},
        'caps': {'seconds': MAX_SECONDS, 'files': MAX_FILES, 'bytes': MAX_BYTES},
        'counters_before_render': {'image_files_written': 0, 'neural_forwards': 0, 'optimizer_updates': 0, 'VM_calls': 0},
        'manual_inference_admission': False, 'training_admission': False,
        'root_input_only_review_pending': True, 'existing55_holds_admission_unchanged': True,
        'reserved_FINAL_used': False, 'DEV_or_native_optimization': False,
        'versions': {'Python': platform.python_version(), 'Pillow': pillow_version, 'NumPy': np.__version__}}
    OUT.mkdir()
    plan_pin = save_json('plan_v1.json', plan)
    cases = []; targets = []; sheets = []; by_profile = {profile: [] for profile in PROFILES}
    for source, source_record in zip(SOURCES, frozen):
        source_payload, clear_payload, moderate_payload, target_payload = buffers[source]
        original = decoded(source_payload, (1024, 1024)); clear_image = decoded(clear_payload, (256, 256))
        moderate_image = decoded(moderate_payload, (256, 256)); old_target = decoded(target_payload, (256, 256))
        direct_input = original.resize((256, 256), Image.Resampling.BILINEAR)
        direct256 = original.resize((256, 256), Image.Resampling.LANCZOS)
        require(np.array_equal(np.asarray(clear_image), np.asarray(direct_input)), 'Clear input is not exact direct source resize')
        require(np.array_equal(np.asarray(old_target), np.asarray(direct256)), 'Metrics target is not exact direct source resize')
        for image, expected_key in ((clear_image, 'clear_RGB8_sha256'), (moderate_image, 'moderate_RGB8_sha256'),
                                    (direct256, 'metrics_target_RGB8_sha256')):
            require(hashlib.sha256(np.asarray(image).tobytes()).hexdigest() == source_record[expected_key], 'Exact existing RGB pixels differ')
        hq = save_png('targets_hq512/' + source + '.png', original.resize((512, 512), Image.Resampling.LANCZOS))
        metric = save_png('targets_metrics256/' + source + '.png', direct256)
        targets.append({'reference_id': source, 'HQ512': hq, 'metrics256': metric,
            'direct1024_derived': True, 'metrics_existing_target_match': True})
        strong = clear_image.filter(ImageFilter.GaussianBlur(radius=3.2)).resize((64, 64), Image.Resampling.BILINEAR)
        strong = strong.resize((256, 256), Image.Resampling.BILINEAR)
        jpeg = io.BytesIO(); strong.save(jpeg, format='JPEG', quality=50, subsampling=2, optimize=False, progressive=False)
        with Image.open(io.BytesIO(jpeg.getvalue())) as image:
            strong = image.convert('RGB').copy()
        images = (clear_image, moderate_image, strong)
        for profile, image in zip(PROFILES, images):
            if profile == 'clear':
                input_pin = save_bytes('inputs/' + source + '__' + profile + '.png', clear_payload)
            elif profile == 'moderate_existing_gaussian_2p4':
                input_pin = save_bytes('inputs/' + source + '__' + profile + '.png', moderate_payload)
            else:
                input_pin = save_png('inputs/' + source + '__' + profile + '.png', image)
            require(decoded((ROOT / input_pin['path']).read_bytes(), (256, 256)).tobytes() == image.tobytes(), 'Saved input pixels differ')
            by_profile[profile].append((source, image.copy()))
            cases.append({'id': source + '__' + profile, 'reference_id': source, 'profile': profile, 'split': 'TRAIN',
                'input': input_pin, 'input_RGB8_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'target_HQ512': hq, 'metrics_target256': metric, 'input_only_review': 'PENDING',
                'inference_admission': False, 'training_admission': False})
    for profile in plan['root_input_only_review_order']:
        sheet = Image.new('RGB', (768, 630), '#f4f1e8'); draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), profile + ' | TRAIN INPUT ONLY | 6 exact256 cells', fill='#171717')
        draw.text((8, 26), 'No clean targets or model outputs. Root review pending; no admission.', fill='#171717')
        panels = []
        for index, (source, image) in enumerate(by_profile[profile]):
            x = (index % 3) * 256; y = 74 + (index // 3) * 278
            draw.text((x + 4, y - 20), source, fill='#171717'); sheet.paste(image, (x, y))
            panels.append({'reference_id': source, 'xyxy_half_open': [x, y, x + 256, y + 256]})
        pin = save_png('input_only_sheets/' + profile + '.png', sheet)
        sheets.append({'profile': profile, 'sheet': pin, 'panels': panels, 'targets_or_model_outputs_shown': False,
            'root_personal_review_complete': False})
    require(len(cases) == 18 and len(targets) == 6 and len(sheets) == 3, 'Exact input/target/sheet counts required')
    for _, frozen_pin in dependencies.items():
        checked(ROOT / frozen_pin['path'], frozen_pin)
    for source in frozen:
        for name in ('source1024', 'existing_clear', 'existing_moderate', 'existing_metrics_target256', 'clear_pair'):
            checked(ROOT / source[name]['path'], source[name])
    checked(__file__, producer)
    require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules), 'Neural framework imported')
    result = {'format': 'codeformer-HQ-code-oracle-image-inputs-v1', 'complete': True,
        'UTC': datetime.now(timezone.utc).isoformat(), 'plan': plan_pin,
        'plan_canonical_sha256': hashlib.sha256(canonical(plan)).hexdigest(), 'source_order': list(SOURCES),
        'case_profiles': list(PROFILES), 'cases': cases, 'targets': targets, 'input_only_sheets': sheets,
        'frozen_source_inputs_reverified_after_render': True, 'source1024_payloads_copied': 0,
        'clear_and_moderate_physical_file_bytes_preserved': True,
        'HQ512_metrics256_direct1024_and_existing256_target_match': True,
        'strong_recipe_fixed_synthetic_not_calibrated_camera': True,
        'root_input_only_review_pending': True, 'inference_admission': False, 'training_admission': False,
        'future_clean_HQ_token_oracle_is_diagnostic_only_not_app_input': True,
        'all55_admission_holds_unchanged': True, 'reserved_FINAL_used': False,
        'DEV_or_native_evaluation_pixels_used': False, 'model_qualified': False, 'app_default_changed': False,
        'execution_boundary': {'image_input_files': 18, 'HQ512_target_files': 6, 'metrics256_target_files': 6,
            'input_only_sheets': 3, 'Torch_imports': 0, 'model_loads': 0, 'neural_forwards': 0,
            'optimizer_updates': 0, 'VM_calls': 0, 'browser_calls': 0, 'root_document_edits': 0},
        'seconds': time.monotonic() - START}
    manifest_pin = save_json('manifest_v1.json', result)
    files = [p for p in OUT.rglob('*') if p.is_file()]
    require(len(files) <= MAX_FILES and sum(p.stat().st_size for p in files) <= MAX_BYTES, 'Input-set physical cap exceeded')
    print(json.dumps({'complete': True, 'manifest': manifest_pin, 'plan': plan_pin, 'image_files': 33,
        'total_files': len(files), 'bytes': sum(p.stat().st_size for p in files), 'neural_forwards': 0,
        'inference_admission': False, 'training_admission': False, 'seconds': time.monotonic() - START}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists() and not (OUT / 'failure_v1.json').exists():
            save_json('failure_v1.json', {'complete': False, 'type': type(exc).__name__, 'message': str(exc),
                'preserve_partial_files': True, 'neural_forwards': 0, 'training_admission': False})
        raise
