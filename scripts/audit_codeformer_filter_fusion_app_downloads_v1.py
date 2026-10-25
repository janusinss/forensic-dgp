"""Audit actual root browser response/download files without neural execution.

Never starts a browser/server, imports Torch, constructs a model, changes an app
default, publishes documents or contacts a VM. Preserve an earlier receipt.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import time
from urllib.parse import urlparse
import zipfile

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_filter_fusion_return_review_20261019_v1'
ARMS = ('original', 'r2', 'fusion50')
PROTOCOL = 'c1f7d8386856bd95ab3af3a8623f33fcecddd4b7e08f0696ea7694a0542c1949'
ARCHIVE = '3d6feaf30d4477fe84bd51c57d560cb10d7b4cd6509db627fc61e03accb35d16'
FUSION_WEIGHTS = 'bc42ad4c55cef0c8e1e3df71e1ae50f6643611e178279fb67b5ffe8bbb511d61'
FUSION_STATE = 'eff91cc6832bdf7ee27cd9f749a8f64a64a2351ab1443b14971b543ae2d86f35'
ORIGINAL_WEIGHTS = '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'
MAX_JSON = 64 * 1024**2


def need(value, message):
    if not value:
        raise ValueError(message)


def safe(path, exists=True, benchmark_hardlinks=False):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    need(path.resolve(strict=exists).is_relative_to(ROOT), 'Workspace evidence required')
    for parent in (path, *path.parents):
        if parent.exists() or parent.is_symlink():
            row = parent.lstat()
            need(not stat.S_ISLNK(row.st_mode) and not getattr(row, 'st_file_attributes', 0) & 1024,
                 'Linked/reparse evidence refused')
    if exists:
        row = path.stat()
        need(stat.S_ISREG(row.st_mode) and (row.st_nlink == 1 or benchmark_hardlinks and
             path == ROOT / 'outputs/codeformer_restoration_pretrained_v1/codeformer.pth'),
             'Ordinary evidence required; only the exact benchmark may retain historical hardlinks')
    return path


def relative(path):
    return str(safe(path, benchmark_hardlinks=True).relative_to(ROOT)).replace('\\', '/')


def pin(path, benchmark_hardlinks=False):
    path = safe(path, benchmark_hardlinks=benchmark_hardlinks); before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Evidence changed during hash')
    return {'bytes': before.st_size, 'sha256': digest}


def load(path):
    path = safe(path); need(path.stat().st_size <= MAX_JSON, 'Bounded response/receipt required')
    def unique(pairs):
        result = {}
        for name, value in pairs:
            need(name not in result, 'Duplicate JSON field refused')
            result[name] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON refused')))


def data_url(value, mime):
    prefix = 'data:' + mime + ';base64,'
    need(isinstance(value, str) and value.startswith(prefix) and len(value) <= MAX_JSON,
         'Expected bounded exact data URL')
    return base64.b64decode(value[len(prefix):], validate=True)


def png_rgb(payload, expected_shape=None):
    with Image.open(io.BytesIO(payload)) as image:
        need(image.format == 'PNG' and image.mode == 'RGB', 'Exact RGB PNG required')
        pixels = np.asarray(image).copy()
    need(pixels.dtype == np.uint8 and (expected_shape is None or pixels.shape == expected_shape),
         'Downloaded PNG size/dtype differs')
    return pixels


def independent_input(path):
    contents = safe(path).read_bytes()
    need(0 < len(contents) <= 10 * 1024**2, 'Bounded real input crop required')
    with Image.open(io.BytesIO(contents)) as image:
        need(image.format in ('PNG', 'JPEG') and max(image.size) <= 4096 and
             image.width * image.height <= 16_000_000, 'Expected real PNG/JPEG crop')
        original = np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
    height, width = original.shape[:2]
    side = max(height, width); left = (side - width) // 2; top = (side - height) // 2
    square = Image.new('RGB', (side, side), (128, 128, 128))
    square.paste(Image.fromarray(original), (left, top))
    observed = Image.new('L', (side, side), 0)
    observed.paste(255, (left, top, left + width, top + height))
    canvas = np.asarray(square.resize((256, 256), Image.Resampling.BILINEAR)).copy()
    support = np.asarray(observed.resize((256, 256), Image.Resampling.NEAREST)).copy() > 0
    yy, xx = np.nonzero(support)
    need(len(xx) > 0, 'No observed face support remains')
    geometry = {'native_size': [width, height], 'native_center_offset': [left, top],
        'observed_bounds_256': [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)],
        'output_size': [256, 256], 'policy': 'dgp256-observed-canvas-v1',
        'input_rgb_sha256': hashlib.sha256(original.tobytes()).hexdigest(),
        'canvas_rgb_sha256': hashlib.sha256(canvas.tobytes()).hexdigest(),
        'observed_sha256': hashlib.sha256(support.tobytes()).hexdigest(),
        'removal_mask_sha256': hashlib.sha256(np.zeros((256, 256), bool).tobytes()).hexdigest()}
    return contents, original, canvas, support, geometry


def statuses(row):
    def one(names):
        for name in names:
            if name in row:
                need(isinstance(row[name], dict), 'Actual status snapshot must be an object')
                return row[name]
        raise ValueError('Measured before/after status snapshots required')
    return one(('before_status', 'status_before', 'before')), one(('after_status', 'status_after', 'after'))


def delta(before, after, arms, completed=0, refusals=0):
    count = len(arms)
    for key in ('neural_forward_attempts', 'neural_forward_returns', 'neural_forward_successes'):
        need(type(before[key]) is int and type(after[key]) is int and after[key] - before[key] == count,
             'Actual measured neural counter differs: ' + key)
    for arm in ARMS:
        expected = arms.count(arm)
        for key in ('attempts', 'returns', 'successes'):
            need(after['per_arm'][arm][key] - before['per_arm'][arm][key] == expected,
                 'Actual measured per-arm counter differs: ' + arm + '/' + key)
    need(after['completed_comparisons'] - before['completed_comparisons'] == completed and
         after['operator_review_refusals'] - before['operator_review_refusals'] == refusals,
         'Actual comparison/refusal counter differs')
    return count


def main(browser_record, output):
    start = time.monotonic()
    output = safe(output, exists=False)
    need(not output.exists(), 'Preserve previous/partial independent download audit')
    browser_record = safe(browser_record); browser_pin = pin(browser_record); browser = load(browser_record)
    need(browser.get('complete') is True, 'Wait for root to finish actual browser collection')
    url = urlparse(browser['url'])
    need(url.scheme == 'http' and url.hostname == '127.0.0.1' and url.port is not None and
         url.path == '/restoration-filter-test' and not url.query and not url.fragment,
         'Actual measured loopback experimental route required')
    export_path = ROOT / 'outputs/codeformer_filter_fusion_training_v1_model/model_export_v1.json'
    export_pin = pin(export_path); exported = load(export_path)
    need(exported['complete'] is True and exported['model_qualified'] is False and
         exported['protocol_sha256'] == PROTOCOL and exported['archive_sha256'] == ARCHIVE and
         exported['export_sha256'] == FUSION_WEIGHTS and exported['model_tensor_sha256'] == FUSION_STATE and
         exported['source_training_updates'] == 50 and exported['app_default_changed'] is False,
         'Exact audited research export required')
    export_closure_path = REVIEW / 'publication_peer_v1/export_closure_v1.json'
    export_closure = load(export_closure_path)
    need(export_closure['complete'] is True and export_closure['audit_pass'] is True and
         export_closure['actual_readback_tensors'] == 515 and export_closure['actual_tensor_sha256'] == FUSION_STATE and
         safe(export_closure['model_export_receipt']['path']) == export_path and
         {key: export_closure['model_export_receipt'][key] for key in ('bytes', 'sha256')} == export_pin and
         export_closure['model_qualified'] is False, 'Actual independent exact515-value CPU readback required')
    state_path = REVIEW / 'state_v1/state_audit_v1.json'; state = load(state_path)
    r2_path = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_model/model_export_v1.json'
    r2_export = load(r2_path)
    need(r2_export['complete'] is True and r2_export['model_qualified'] is False and
         r2_export['model_tensor_sha256'] == state['immutable_input_weights']['r2']['canonical515_tensor_sha256'],
         'Actual retained R2 export binding differs')
    model_paths = {'original': ROOT / 'outputs/codeformer_restoration_pretrained_v1/codeformer.pth',
                   'r2': Path(r2_export['export_path']), 'fusion50': Path(exported['model_path'])}
    model_pins = {arm: pin(path, benchmark_hardlinks=arm == 'original') for arm, path in model_paths.items()}
    need(model_pins['original']['sha256'] == ORIGINAL_WEIGHTS and
         model_pins['r2'] == {'bytes': r2_export['export_bytes'], 'sha256': r2_export['export_sha256']} and
         model_pins['fusion50'] == {'bytes': exported['export_bytes'], 'sha256': FUSION_WEIGHTS},
         'Actual benchmark/R2/fusion50 files differ')
    model_states = {'original': state['immutable_input_weights']['original']['canonical515_tensor_sha256'],
                    'r2': r2_export['model_tensor_sha256'], 'fusion50': FUSION_STATE}
    source_pins = browser['source_pins']
    need(isinstance(source_pins, dict) and
         {'research_filter_restoration_web_v1.py', 'static/research_filter_restoration_v1.js'} <= set(source_pins),
         'Actual serving source pins required')
    for name, value in source_pins.items():
        need(pin(name) == value, 'Serving source differs from root browser collection: ' + name)
    source_pins = dict(source_pins)
    source_pins[relative(__file__)] = pin(__file__)
    physical = []; comparison_records = []; proof_pins = {}
    total_arms = []; calls = browser['comparison_calls']
    need(isinstance(calls, list) and calls and len({row['case_id'] for row in calls}) == len(calls),
         'Actual distinct browser comparisons required')
    for row in calls:
        need(row['status'] == 200 and row['selection'] in ('all', *ARMS), 'Actual successful comparison required')
        arms = list(ARMS) if row['selection'] == 'all' else [row['selection']]
        before, after = statuses(row); delta(before, after, arms, completed=1)
        response_path = safe(row['response_path']); response_pin = pin(response_path); response = load(response_path)
        input_path = safe(row['input_path']); input_pin = pin(input_path)
        contents, original, canvas, support, geometry = independent_input(input_path)
        need(set(response) == {'original', 'outputs', 'metadata', 'bundle'} and
             set(response['outputs']) == set(arms), 'Actual HTTP response output selection differs')
        metadata = response['metadata']
        need(metadata['selection'] == row['selection'] and metadata['input_review'] == 'usable' and
             metadata['geometry'] == geometry and metadata['uploaded_bytes_sha256'] == hashlib.sha256(contents).hexdigest() and
             metadata['device'] == 'cpu' and metadata['inference_only'] is True and
             metadata['model_qualified'] is False and metadata['main_app_default_changed'] is False and
             metadata['RAW_PNG_separate'] is True and
             metadata['outside_observed_support_RAW_and_PNG_match_input_exact'] is True and
             metadata['no_completion_or_covering_removal'] is True and
             metadata['operator_input_review_is_not_a_validated_automatic_threshold'] is True,
             'Actual input/geometry/inference/research boundary metadata differs')
        need(np.array_equal(png_rgb(data_url(response['original'], 'image/png'), (256, 256, 3)), canvas),
             'Actual response resize-control pixels differ')
        need(set(row['download_paths']) == set(arms) | {'bundle'}, 'Actual physical download selection differs')
        bundle_path = safe(row['download_paths']['bundle']); bundle_pin = pin(bundle_path)
        bundle = bundle_path.read_bytes()
        need(bundle_path.suffix.lower() == '.zip' and bundle == data_url(response['bundle'], 'application/zip'),
             'Physical browser-downloaded ZIP differs from actual HTTP response')
        physical.append({'path': relative(bundle_path), **bundle_pin})
        arm_details = {}
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            names = archive.namelist()
            expected = {'original-upload.png', 'resize-control.png', 'observed-support.png', 'processing.json'} | {
                arm + suffix for arm in arms for suffix in ('.png', '-raw.npy')}
            need(len(names) == len(set(names)) and set(names) == expected and
                 sum(info.file_size for info in archive.infolist()) <= 64 * 1024**2 and
                 all(not info.is_dir() and not info.flag_bits & 1 and info.file_size <= 64 * 1024**2
                     for info in archive.infolist()), 'Exact bounded unencrypted comparison ZIP members required')
            need(np.array_equal(png_rgb(archive.read('original-upload.png')), original) and
                 np.array_equal(png_rgb(archive.read('resize-control.png'), (256, 256, 3)), canvas),
                 'ZIP native original or resize-control pixels differ')
            with Image.open(io.BytesIO(archive.read('observed-support.png'))) as mask:
                need(mask.format == 'PNG' and mask.mode == 'L' and
                     np.array_equal(np.asarray(mask), support.astype(np.uint8) * 255), 'ZIP observed support differs')
            need(json.loads(archive.read('processing.json')) == metadata, 'ZIP processing record differs from actual HTTP metadata')
            for arm in arms:
                downloaded_path = safe(row['download_paths'][arm]); downloaded_pin = pin(downloaded_path)
                png = downloaded_path.read_bytes()
                need(downloaded_path.suffix.lower() == '.png' and
                     png == data_url(response['outputs'][arm], 'image/png') == archive.read(arm + '.png'),
                     'Physical browser PNG differs from HTTP/ZIP bytes: ' + arm)
                pixels = png_rgb(png, (256, 256, 3)); arm_metadata = metadata['outputs'][arm]
                provenance = arm_metadata['metadata']
                need(provenance['arm'] == arm and provenance['weights_sha256'] == model_pins[arm]['sha256'] and
                     provenance['model_tensor_sha256'] == model_states[arm] and provenance['model_qualified'] is False and
                     provenance['main_app_default_changed'] is False and provenance['fidelity'] == 1.0 and
                     provenance['adain'] is True and provenance['alignment'] == 'none' and
                     provenance['display_processing'] == 'none' and provenance['internal_resolution'] == 512 and
                     provenance['output_resolution'] == 256,
                     'Actual selected output model/provenance differs: ' + arm)
                if arm == 'fusion50':
                    need(provenance['source_checkpoint_sha256'] == exported['source_checkpoint_sha256'] and
                         provenance['protocol_sha256'] == PROTOCOL and provenance['source_training_updates'] == 50 and
                         provenance['params_ema_key_is_loader_compatibility_not_new_EMA'] is True,
                         'Fusion50 stopped-state provenance differs')
                raw = np.load(io.BytesIO(archive.read(arm + '-raw.npy')), allow_pickle=False)
                need(isinstance(raw, np.ndarray) and raw.shape == (256, 256, 3) and raw.dtype == np.float32 and
                     np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1,
                     'Exact finite float32 RAW required: ' + arm)
                expected_png = np.floor(raw * np.float32(255)).astype(np.uint8)
                expected_png[~support] = canvas[~support]
                need(np.array_equal(expected_png, pixels) and np.array_equal(pixels[~support], canvas[~support]) and
                     np.array_equal(raw[~support], canvas[~support].astype(np.float32) / np.float32(255)),
                     'Exact delivered RAW/PNG conversion or support preservation differs: ' + arm)
                raw_sha = hashlib.sha256(raw.tobytes()).hexdigest(); pixel_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
                need(raw_sha == arm_metadata['raw_rgb_sha256'] and pixel_sha == arm_metadata['PNG_RGB8_sha256'] and
                     downloaded_pin['sha256'] == arm_metadata['PNG_file_sha256'], 'Actual downloaded data hashes differ: ' + arm)
                physical.append({'path': relative(downloaded_path), **downloaded_pin})
                arm_details[arm] = {'model_file': model_pins[arm], 'model_tensor_sha256': model_states[arm],
                    'RAW_rgb_sha256': raw_sha, 'PNG_RGB8_sha256': pixel_sha, 'PNG_file_sha256': downloaded_pin['sha256'],
                    'RAW_dtype_shape_range_pass': True, 'RAW_to_PNG_floor_conversion_pass': True,
                    'outside_observed_support_RAW_and_PNG_match_input_exact': True}
        comparison_records.append({'case_id': row['case_id'], 'selection': row['selection'], 'status': 200,
            'outputs': arms, 'neural_forward_calls': len(arms), 'input_path': relative(input_path), 'input_file': input_pin,
            'response_path': relative(response_path), 'response_file': response_pin,
            'independent_geometry': geometry, 'bundle_file': bundle_pin, 'outputs_audited': arm_details})
        proof_pins[relative(response_path)] = response_pin; proof_pins[relative(input_path)] = input_pin
        total_arms.extend(arms)
    need(any(row['selection'] == 'all' for row in calls), 'At least one actual all-three-model comparison required')
    refusals = []
    for row in browser['refusal_calls']:
        actual_review = row['input_review']
        normalized = actual_review if actual_review in ('needs_clearer', 'out_of_scope') else 'invalid'
        need(actual_review != 'usable' and row['status'] == 422, 'Actual operator input-refusal HTTP422 required')
        before, after = statuses(row); delta(before, after, [], refusals=1)
        refusals.append({'input_review': normalized, 'submitted_input_review': actual_review,
            'status': 422, 'neural_forward_calls': 0})
    need({'needs_clearer', 'out_of_scope', 'invalid'} <= {row['input_review'] for row in refusals},
         'All three actual invalid/insufficient/scope refusal cases required')
    before, after = statuses(browser)
    delta(before, after, total_arms, completed=len(calls), refusals=len(refusals))
    need(len(physical) == len({row['path'] for row in physical}) == len(total_arms) + len(calls),
         'Exact distinct physical PNG/ZIP browser download files required')
    proof_pins[relative(browser_record)] = browser_pin
    for name, value in source_pins.items():
        need(pin(name) == value, 'Source changed during download audit: ' + name)
    for name, value in proof_pins.items():
        need(pin(name) == value, 'Actual browser/input/response evidence changed during audit: ' + name)
    for row in physical:
        need(pin(row['path']) == {key: row[key] for key in ('bytes', 'sha256')}, 'Physical download changed during audit')
    need(pin(export_path) == export_pin, 'Exact research export receipt changed during audit')
    required = set(proof_pins) | set(source_pins) | {row['path'] for row in physical}
    required |= {relative(export_path), relative(export_closure_path), relative(r2_path), relative(state_path)}
    result = {'complete': True, 'audit_pass': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'url': browser['url'], 'archive_sha256': ARCHIVE, 'protocol_sha256': PROTOCOL,
        'model_export_receipt_sha256': export_pin['sha256'], 'model_weights_sha256': FUSION_WEIGHTS,
        'model_tensor_sha256': FUSION_STATE, 'actual_neural_forward_calls': len(total_arms),
        'comparison_calls': comparison_records, 'refusal_calls': refusals,
        'downloaded_PNGs_checked': len(total_arms), 'downloaded_bundles_checked': len(calls),
        'saved_bytes_match_response_hashes': True, 'RAW_PNG_separate': True,
        'outside_observed_support_RAW_and_PNG_match_input_exact': True,
        'physical_files': physical, 'source_browser_evidence': {'path': relative(browser_record), **browser_pin},
        'source_pins': source_pins, 'response_input_proof_pins': proof_pins,
        'model_file_pins': {relative(model_paths[arm]): model_pins[arm] for arm in ARMS},
        'required_evidence_paths': sorted(required), 'independently_reconstructed_geometry': True,
        'exact_ZIP_members_and_processing_metadata_pass': True, 'actual_export515_value_peer_closure_bound': True,
        'model_qualified': False, 'main_app_default_changed': False, 'goal_complete': False,
        'execution_boundary': {'Torch_imports': 0, 'model_constructions': 0, 'neural_forwards': 0,
            'optimizer_updates': 0, 'VM_calls': 0, 'browser_launches': 0, 'server_launches': 0,
            'publication_operations': 0, 'model_promotions': 0}, 'seconds': time.monotonic() - start}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'complete': True, 'audit_pass': True, 'receipt': relative(output), **pin(output),
        'downloaded_PNGs_checked': len(total_arms), 'downloaded_bundles_checked': len(calls),
        'actual_browser_neural_forward_calls': len(total_arms), 'audit_neural_forward_calls': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser-record', type=Path,
        default=ROOT / 'scratch/codeformer_filter_app_audit_20261019_v1/browser_raw_v1.json')
    parser.add_argument('--receipt', type=Path, default=REVIEW / 'app_v1/download_audit_v1.json')
    args = parser.parse_args()
    main(args.browser_record, args.receipt)
