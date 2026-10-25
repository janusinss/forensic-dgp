"""R2 independent saved-only audit of 272 single-arm responses and explicit68-case merges.

No HTTP, Torch, neural model, optimizer, VM or browser entry point exists here.
Image metrics below are reimplemented with NumPy, including uniform7 SSIM.
Passing this audit establishes artifact integrity, never model qualification.
"""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('original', 'r2', 'spatial_u', 'spatial_s')
COHORT = 'outputs/codeformer_spatial_dev_review_v1/cohort_manifest_v1.json'
COHORT_SHA = 'e42afd973305e42422bcb273dc419ab608bbc1437ab332e62d6eb006e0fe9307'
APP_SHA = 'bafc0ec5ef912d2354418623d963e06eebea1289eeb831ab4dd5b6d7ad6bd208'
MODEL_MANIFEST = 'outputs/codeformer_spatial_continuation_v1_models/model_exports_v1.json'
MODEL_MANIFEST_SHA = 'f0b2f5b25c43b7702867833e28ed6a28fe88e6e994b5597698820f3950ad190b'
ORIGINAL_WEIGHTS_SHA = '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'
ORIGINAL_TENSOR_SHA = '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd'
R2_WEIGHTS_SHA = 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
R2_TENSOR_SHA = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
APP_QUALITY_STATUS = ('Experimental restoration comparison. The new adapters are not an accepted replacement. '
    'The inherited R2 preservation failures remain unresolved; repeated TRAIN checks do not establish final quality.')
EXPECTED_ZIP = frozenset(('original-upload.png', 'resize-control.png',
    'observed-support.png', 'processing.json') +
    tuple(name for arm in ARMS for name in (arm + '.png', arm + '-raw.npy')))
EXPECTED_CASE_FILES = frozenset(('app-comparison.zip', 'app-response-metadata.json') +
    tuple(name for arm in ARMS for name in (arm + '.png', arm + '-raw.npy')))
Y = np.array([.2126, .7152, .0722], dtype=np.float64)
ABS_TOLERANCE = 1e-11
REL_TOLERANCE = 1e-10


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def unique_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=reject)


def relative(value):
    need(isinstance(value, str) and value and '\\' not in value and ':' not in value,
         'Canonical relative path required')
    path = PurePosixPath(value)
    need(not path.is_absolute() and path.as_posix() == value and
         all(part not in ('.', '..', '') for part in path.parts), 'Canonical relative path required')
    return path


def checked_bytes(path, expected_sha=None, expected_size=None, cap=25 * 1024**2):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / relative(path.as_posix())
    need(path.resolve(strict=True).is_relative_to(ROOT), 'File escapes workspace')
    for parent in (path, *path.parents):
        if parent == ROOT.parent:
            break
        info = parent.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Linked or reparse evidence is not permitted')
    before = path.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_size <= cap, 'Bounded ordinary file required')
    need(expected_size is None or before.st_size == expected_size, 'File byte count differs')
    value = path.read_bytes()
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'Evidence changed during read')
    need(expected_sha is None or digest(value) == expected_sha, 'File SHA differs: ' + str(path))
    return value


def bound(binding):
    relative(binding['path'])
    return checked_bytes(binding['path'], binding['sha256'], binding['bytes'])


def pin(path):
    path = Path(path)
    value = checked_bytes(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(value), 'sha256': digest(value)}


def rgb(data, exact_size=(256, 256), require_png=True):
    with Image.open(io.BytesIO(data)) as image:
        need(not require_png or image.format == 'PNG', 'PNG required')
        need(image.mode == 'RGB' and getattr(image, 'n_frames', 1) == 1, 'Single-frame RGB required')
        need(exact_size is None or image.size == exact_size, 'Image geometry differs')
        return np.asarray(image).copy()


def upload_rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        need(image.format in ('PNG', 'JPEG') and getattr(image, 'n_frames', 1) == 1,
             'Native upload must be a single PNG/JPEG')
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()


def support_png(data):
    with Image.open(io.BytesIO(data)) as image:
        need(image.format == 'PNG' and image.mode == 'L' and image.size == (256, 256),
             'L256 support PNG required')
        pixels = np.asarray(image).copy()
    need(set(np.unique(pixels)).issubset({0, 255}) and pixels.any(), 'Exact binary nonempty support required')
    return pixels > 0


def prepare_independent(native):
    """Independent transcription of the legacy integer geometry, without app imports."""
    height, width = native.shape[:2]
    side = max(height, width)
    left, top = (side - width) // 2, (side - height) // 2
    padded = Image.new('RGB', (side, side), (128, 128, 128))
    padded.paste(Image.fromarray(native), (left, top))
    native_support = Image.new('L', (side, side), 0)
    native_support.paste(255, (left, top, left + width, top + height))
    canvas = np.asarray(padded.resize((256, 256), Image.Resampling.BILINEAR)).copy()
    support = np.asarray(native_support.resize((256, 256), Image.Resampling.NEAREST)) > 0
    yy, xx = np.nonzero(support)
    geometry = {'native_size': [width, height], 'native_center_offset': [left, top],
        'observed_bounds_256': [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)],
        'output_size': [256, 256], 'policy': 'dgp256-observed-canvas-v1',
        'input_rgb_sha256': digest(native.tobytes()),
        'canvas_rgb_sha256': digest(canvas.tobytes()),
        'observed_sha256': digest(support.tobytes()),
        'removal_mask_sha256': digest(np.zeros((256, 256), dtype=bool).tobytes())}
    return canvas, support, geometry


def raw_array(data):
    stream = io.BytesIO(data)
    array = np.load(stream, allow_pickle=False)
    need(stream.tell() == len(data), 'Trailing bytes after RAW array')
    need(array.dtype == np.float32 and array.shape == (256, 256, 3) and array.flags.c_contiguous,
         'C-contiguous HWC256 FP32 RAW required')
    need(np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1, 'Finite normalized RAW required')
    return array


def uniform_valid7(array):
    """Mean of every fully observed7x7 window using an independent summed-area table."""
    integral = np.pad(array.astype(np.float64).cumsum(0).cumsum(1),
                      ((1, 0), (1, 0), (0, 0)), mode='constant')
    return (integral[7:, 7:] - integral[:-7, 7:] - integral[7:, :-7] + integral[:-7, :-7]) / 49.


def independent_ssim(predicted, target):
    mu_x, mu_y = uniform_valid7(predicted), uniform_valid7(target)
    var_x = (uniform_valid7(predicted * predicted) - mu_x * mu_x) * (49. / 48.)
    var_y = (uniform_valid7(target * target) - mu_y * mu_y) * (49. / 48.)
    covariance = (uniform_valid7(predicted * target) - mu_x * mu_y) * (49. / 48.)
    numerator = (2 * mu_x * mu_y + .01**2) * (2 * covariance + .03**2)
    denominator = (mu_x * mu_x + mu_y * mu_y + .01**2) * (var_x + var_y + .03**2)
    return float((numerator / denominator).mean(dtype=np.float64))


def metrics(predicted, target, support):
    """Independent NumPy implementation, matching the declared metric definitions."""
    predicted, target = predicted.astype(np.float64), target.astype(np.float64)
    residual = predicted - target
    selected = residual[support]
    offset = selected.mean(axis=0, dtype=np.float64)
    centered = selected - offset
    mse = float(np.square(selected).mean(dtype=np.float64))
    centered_mse = float(np.square(centered).mean(dtype=np.float64))
    energy = float(np.square(offset).mean(dtype=np.float64))
    sx, sy = support[:, 1:] & support[:, :-1], support[1:, :] & support[:-1, :]
    nx, ny = int(sx.sum()), int(sy.sum())
    x_error = float(np.abs((residual[:, 1:] - residual[:, :-1])[sx]).sum(dtype=np.float64))
    y_error = float(np.abs((residual[1:, :] - residual[:-1, :])[sy]).sum(dtype=np.float64))
    signed_luma = float((selected @ Y).mean(dtype=np.float64))
    return {'mae': float(np.abs(selected).mean(dtype=np.float64)), 'mse': mse,
        'psnr': None if mse == 0 else -10 * math.log10(mse),
        'PSNR_zero_MSE_policy': 'null denotes mathematically infinite PSNR; no finite fabricated value',
        'ssim': independent_ssim(predicted, target) if support.all() else None,
        'SSIM_policy': 'skimage structural_similarity uniform7x7,sample covariance,data_range1,RGB channel mean; only all-observed case',
        'signed_luminance_bias': signed_luma, 'absolute_luminance_bias': abs(signed_luma),
        'mean_RGB_offset': offset.tolist(),
        'mean_centered_residual_mae': float(np.abs(centered).mean(dtype=np.float64)),
        'mean_centered_residual_mse': centered_mse, 'global_RGB_offset_energy': energy,
        'global_RGB_offset_energy_share': None if mse == 0 else energy / mse,
        'MSE_decomposition_closure_residual': mse - centered_mse - energy,
        'adjacent_RGB_residual_gradient_mae': None if nx + ny == 0 else (x_error + y_error) / (3 * (nx + ny)),
        'adjacent_RGB_residual_gradient_x_mae': None if nx == 0 else x_error / (3 * nx),
        'adjacent_RGB_residual_gradient_y_mae': None if ny == 0 else y_error / (3 * ny),
        'adjacent_x_pairs': nx, 'adjacent_y_pairs': ny, 'observed_pixels': int(support.sum()),
        'offset_diagnostics_are_not_acceptance_or_identity_metrics': True}


def compare_numeric(saved, actual, location, differences):
    if isinstance(actual, dict):
        need(type(saved) is dict and set(saved) == set(actual), 'Metric keys differ: ' + location)
        for key in actual:
            compare_numeric(saved[key], actual[key], location + '/' + key, differences)
    elif isinstance(actual, list):
        need(type(saved) is list and len(saved) == len(actual), 'Metric list differs: ' + location)
        for index, value in enumerate(actual):
            compare_numeric(saved[index], value, location + '/' + str(index), differences)
    elif type(actual) is float:
        need(type(saved) in (int, float) and math.isfinite(saved), 'Finite numeric metric required: ' + location)
        difference = abs(saved - actual)
        need(difference <= ABS_TOLERANCE + REL_TOLERANCE * abs(actual),
             'Independent metric disagreement: ' + location + ' saved=' + repr(saved) + ' own=' + repr(actual))
        differences.append(difference)
    else:
        need(type(saved) is type(actual) and saved == actual, 'Metric literal differs: ' + location)


def scalar_summary(rows):
    fields = ('mae', 'mse', 'psnr', 'ssim', 'mean_centered_residual_mae',
              'adjacent_RGB_residual_gradient_mae', 'absolute_luminance_bias')
    return {name: {'mean': float(np.mean([r[name] for r in rows if r[name] is not None], dtype=np.float64))
                         if any(r[name] is not None for r in rows) else None,
                   'finite_cases': sum(r[name] is not None for r in rows)} for name in fields}


def data_url(value):
    need(type(value) is str and ',' in value, 'Data URL required')
    header, encoded = value.split(',', 1)
    need(header in ('data:image/png;base64', 'data:application/zip;base64'),
         'Exact PNG or ZIP base64 data URL required')
    return base64.b64decode(encoded, validate=True)


def verify_transport(run, row, merged_metadata, upload, canvas, support):
    """Bind all four original server bodies/headers/ZIPs to the explicit client merge."""
    transport_dir = run / 'transport' / digest(upload)
    need(transport_dir.is_dir() and {p.name for p in transport_dir.iterdir()} == set(ARMS),
         'Exact four-arm source transport namespace required')
    declared = merged_metadata.get('transport')
    need(type(declared) is dict and set(declared) == {'version', 'source_responses',
         'each_original_server_ZIP_retained', 'no_retry'} and
         declared['version'] == 'single-arm-persistent-exact-body-r2' and
         declared['each_original_server_ZIP_retained'] is True and declared['no_retry'] is True,
         'Explicit R2 source transport declaration required')
    sources = declared['source_responses']
    need(type(sources) is list and [r['arm'] for r in sources] == list(ARMS),
         'Exactly four ordered source declarations required')
    original_metadata, server_payloads, physical = {}, {}, []
    spatial_status = unique_json(checked_bytes(MODEL_MANIFEST, MODEL_MANIFEST_SHA))['quality_status']
    common = ('original-upload.png', 'resize-control.png', 'observed-support.png')
    for arm, declared_source in zip(ARMS, sources):
        folder = transport_dir / arm
        need({p.name for p in folder.iterdir()} ==
             {'response.body', 'response-headers.json', 'app-comparison.zip', 'metadata.json'},
             'Exact four retained source response files required')
        expected_names = {'response_body': 'response.body', 'headers': 'response-headers.json',
                          'ZIP': 'app-comparison.zip', 'metadata': 'metadata.json'}
        need(set(declared_source) == set(expected_names) | {'arm'}, 'Source response pin schema differs')
        source_bytes = {}
        for key, name in expected_names.items():
            item = declared_source[key]
            need(item['path'] == (folder / name).relative_to(ROOT).as_posix(),
                 'Source response pin points to another arm/input')
            source_bytes[key] = bound(item)
            physical.append(item)
        headers = unique_json(source_bytes['headers'])
        need(type(headers) is dict and all(type(k) is str and type(v) is str for k, v in headers.items()),
             'Saved HTTP header map required')
        lower_headers = {key.lower(): value for key, value in headers.items()}
        need(len(lower_headers) == len(headers), 'Case-insensitive duplicate HTTP header')
        need(lower_headers.get('content-type', '').startswith('application/json'),
             'JSON HTTP response content type required')
        need('content-length' in lower_headers and
             int(lower_headers['content-length']) == len(source_bytes['response_body']),
             'Retained R2 HTTP response body must match exact declared Content-Length')
        response = unique_json(source_bytes['response_body'])
        need(type(response) is dict and set(response) == {'original', 'outputs', 'metadata', 'bundle'} and
             set(response['outputs']) == {arm}, 'Original single-arm response schema differs')
        source_metadata = unique_json(source_bytes['metadata'])
        need(response['metadata'] == source_metadata and source_metadata['selection'] == arm and
             set(source_metadata['outputs']) == {arm}, 'Single-arm metadata/body join differs')
        need(source_metadata['quality_status'] == (spatial_status if arm.startswith('spatial_') else APP_QUALITY_STATUS),
             'Single-arm quality status differs from its exact frozen app/model authority')
        need(source_metadata['uploaded_bytes_sha256'] == digest(upload) and
             source_metadata['geometry'] == row['geometry'] and source_metadata['input_review'] == 'usable',
             'Source response input/geometry join differs')
        need(data_url(response['bundle']) == source_bytes['ZIP'], 'Original server ZIP differs from body')
        with zipfile.ZipFile(io.BytesIO(source_bytes['ZIP'])) as archive:
            expected = set(common) | {arm + '.png', arm + '-raw.npy', 'processing.json'}
            need(len(archive.namelist()) == 6 and set(archive.namelist()) == expected and
                 archive.testzip() is None, 'Exact six-member original server ZIP with valid CRC required')
            need(all(not item.is_dir() and item.file_size <= 2 * 1024**2 and not item.flag_bits & 1
                     for item in archive.infolist()), 'Bounded regular source ZIP members required')
            payloads = {name: archive.read(name) for name in expected}
        need(unique_json(payloads['processing.json']) == source_metadata,
             'Original server ZIP processing differs from body metadata')
        need(data_url(response['outputs'][arm]) == payloads[arm + '.png'],
             'Single-arm body PNG differs from original server ZIP')
        need(np.array_equal(rgb(data_url(response['original'])), canvas),
             'Single-arm body resize control differs from frozen pixels')
        need(np.array_equal(rgb(payloads['original-upload.png'], exact_size=None), upload_rgb(upload)) and
             np.array_equal(rgb(payloads['resize-control.png']), canvas) and
             np.array_equal(support_png(payloads['observed-support.png']), support),
             'Single-arm original/canvas/support differs')
        original_metadata[arm], server_payloads[arm] = source_metadata, payloads
    for arm in ARMS:
        for name in common:
            need(server_payloads[arm][name] == server_payloads['original'][name],
                 'Four single-arm common PNG bytes differ')
        normalized = {k: v for k, v in original_metadata[arm].items()
                      if k not in ('selection', 'outputs', 'seconds', 'quality_status')}
        original_normalized = {k: v for k, v in original_metadata['original'].items()
                               if k not in ('selection', 'outputs', 'seconds', 'quality_status')}
        need(normalized == original_normalized, 'Four original metadata input/policy fields differ')
    expected_metadata = dict(original_metadata['original'])
    expected_metadata['selection'] = 'client_merge_of_four_single_arm_requests'
    expected_metadata['transport'] = declared
    expected_metadata['seconds'] = sum(original_metadata[arm]['seconds'] for arm in ARMS)
    expected_metadata['outputs'] = {arm: original_metadata[arm]['outputs'][arm] for arm in ARMS}
    need(merged_metadata == expected_metadata, 'Client merged metadata is not the declared exact four-source join')
    with zipfile.ZipFile(io.BytesIO(checked_bytes(run / row['id'] / 'app-comparison.zip'))) as archive:
        for name in common:
            need(archive.read(name) == server_payloads['original'][name], 'Merged common PNG differs from server origin')
        for arm in ARMS:
            for name in (arm + '.png', arm + '-raw.npy'):
                need(archive.read(name) == server_payloads[arm][name], 'Merged output differs from its original server ZIP')
    return physical


def audit(run, receipt):
    started = time.monotonic()
    cohort_data = checked_bytes(COHORT, COHORT_SHA)
    cohort = unique_json(cohort_data)
    need(cohort['complete'] is True and cohort['FINAL_identity_or_pixel_review'] is False, 'Frozen DEV authority differs')
    cases = cohort['cases']
    need(len(cases) == 68 and len({r['id'] for r in cases}) == 68, 'Exact68 unique inputs required')
    need(Counter(r['cohort'] for r in cases) == {'filter_DEV': 36, 'native_ChokePoint': 24,
                                               'native_QMUL-SurvFace': 8}, 'Cohort membership differs')
    need(sum(r.get('appearance_flag', False) for r in cases) == 9, 'All9 appearance-flagged cases retained')
    native_controls = cohort['native_input_only_controls']
    filter_controls = cohort['filter_insufficient_controls']
    need(len(native_controls) == 16 and len(filter_controls) == 24 and
         all(r['neural_calls_allowed'] == 0 for r in native_controls + filter_controls),
         'Frozen input-only controls differ')
    controls = {r['id'] for r in native_controls + filter_controls}
    need(not controls & {r['id'] for r in cases}, 'A refusal control entered inference')
    need(not any((run / name).exists() for name in controls), 'Refusal control has a run folder')
    manifest = unique_json(checked_bytes(MODEL_MANIFEST, MODEL_MANIFEST_SHA))
    need(manifest['complete'] is True and manifest['model_qualified'] is False and
         manifest['base_tensor_sha256'] == R2_TENSOR_SHA, 'Model binding differs')
    for arm in ('spatial_u', 'spatial_s'):
        bound(manifest['arms'][arm]['export'])
    checked_bytes('research_spatial_restoration_v1.py', APP_SHA)
    comparison_data = checked_bytes(run / 'comparison.json', cap=25 * 1024**2)
    comparison = unique_json(comparison_data)
    need(comparison['complete'] is True and 'failure' not in comparison, 'Worker is not successfully terminal')
    need(comparison['model_qualified'] is False and comparison['actual_training_optimizer_VM_calls'] == 0,
         'Worker claim exceeds inference-only diagnostic')
    need(comparison['transport_version'] == 'single-arm-persistent-exact-body-r2' and
         comparison['expected_successful_requests'] == 272 and
         comparison['individual_original_API_ZIPs_preserved'] is True and
         comparison['five_completed_but_undelivered_prior_forwards_not_counted_as_own_results'] is True,
         'R2 transport/previous-failure scope differs')
    previous_path = comparison['original_all_arm_run_retained']
    need(previous_path == 'outputs/codeformer_spatial_dev_review_v1/run_v1/comparison.json',
         'Original failed run reference differs')
    previous = unique_json(checked_bytes(previous_path))
    need(previous['complete'] is False and len(previous['cases']) == 0 and
         type(previous.get('failure')) is dict, 'Original undelivered failure evidence changed')
    previous_r1_path = comparison['prior_R1_failure_retained']
    need(previous_r1_path == 'outputs/codeformer_spatial_dev_review_v1/run_v1_r1/comparison.json',
         'Retained R1 failure path differs')
    previous_r1 = unique_json(checked_bytes(previous_r1_path))
    need(previous_r1['complete'] is False and len(previous_r1['cases']) == 0 and
         type(previous_r1.get('failure')) is dict, 'R1 undelivered failure evidence changed')
    need(comparison['cohort']['sha256'] == COHORT_SHA and bound(comparison['cohort']) == cohort_data,
         'Worker cohort pin differs')
    bound(comparison['worker'])
    need(comparison['app_source']['sha256'] == APP_SHA, 'Worker app binding differs')
    bound(comparison['app_source'])
    need(comparison['successful_forwards_in_own_responses'] == 272 and
         len(comparison['cases']) == 68, '68case/272response-forward count differs')
    need([r['id'] for r in comparison['cases']] == [r['id'] for r in cases], 'Worker order/membership differs')
    need({p.name for p in run.iterdir()} == {r['id'] for r in cases} | {'comparison.json', 'transport'},
         'Run top-level namespace differs')
    summary_rows = []
    metric_differences = []
    paired_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    native_groups = defaultdict(list)
    physical = []
    transport_physical = []
    expected_transport_directories = set()
    for index, (row, saved) in enumerate(zip(cases, comparison['cases'])):
        folder = run / row['id']
        need(folder.is_dir() and {p.name for p in folder.iterdir()} == EXPECTED_CASE_FILES,
             'Exact10 case files required: ' + row['id'])
        need(len(saved['files']) == 10 and {x['path'] for x in saved['files']} ==
             {p.relative_to(ROOT).as_posix() for p in folder.iterdir()}, 'Saved case file manifest differs')
        for binding in saved['files']:
            bound(binding)
            physical.append(binding)
        need(saved['input_review_retained'] == row['input_review'] and
             saved['appearance_flag'] is row.get('appearance_flag', False) and
             saved['inference_only_diagnostic_not_user_admission'] is row['inference_only_diagnostic_not_user_admission'] and
             saved['no_optimization'] is True and saved['native_PSNR_SSIM'] is None,
             'Input review/evidence scope changed')
        for key in ('id', 'source', 'profile', 'evidence', 'split', 'reference_id'):
            need(saved[key] == row.get(key), 'Case provenance differs: ' + key)
        upload = bound(row['upload_source'])
        canvas = rgb(bound(row['input']))
        support = support_png(bound(row['support']))
        original = upload_rgb(upload)
        re_canvas, re_support, geometry = prepare_independent(original)
        need(np.array_equal(canvas, re_canvas) and np.array_equal(support, re_support) and
             geometry == row['geometry'], 'Independent native geometry differs')
        need(digest(canvas.tobytes()) == row['input_pixels']['RGB8_sha256'], 'Frozen canvas pixel hash differs')
        target = None if row['target'] is None else rgb(bound(row['target'])).astype(np.float32) / np.float32(255)
        need((target is None) is (row['evidence'] == 'native_unpaired'), 'Native/paired target scope differs')
        metadata = unique_json(checked_bytes(folder / 'app-response-metadata.json'))
        need(metadata['geometry'] == geometry and metadata['uploaded_bytes_sha256'] == digest(upload),
             'API upload/geometry binding differs')
        need(metadata['policy'] == 'research-CodeFormer-spatial-comparison-v1' and
             metadata['device'] == 'cpu' and metadata['input_review'] == 'usable' and
             metadata['selection'] == 'client_merge_of_four_single_arm_requests' and metadata['inference_only'] is True and
             metadata['model_qualified'] is False and metadata['main_app_default_changed'] is False and
             metadata['operator_input_review_is_not_a_validated_automatic_threshold'] is True and
             metadata['no_completion_or_covering_removal'] is True and metadata['RAW_PNG_separate'] is True and
             metadata['outside_observed_support_RAW_and_PNG_match_input_exact'] is True and
             set(metadata['outputs']) == set(ARMS), 'API inference/promotion scope differs')
        transport_physical.extend(verify_transport(run, row, metadata, upload, canvas, support))
        expected_transport_directories.add(digest(upload))
        raw_png = {}
        result_metrics = {}
        with zipfile.ZipFile(io.BytesIO(checked_bytes(folder / 'app-comparison.zip'))) as archive:
            need(len(archive.namelist()) == len(EXPECTED_ZIP) and set(archive.namelist()) == EXPECTED_ZIP,
                 'Exact12 unique ZIP members required')
            need(archive.testzip() is None, 'ZIP CRC failure')
            need(all(not info.is_dir() and info.file_size <= 2 * 1024**2 and not info.flag_bits & 1
                     for info in archive.infolist()), 'Bounded regular unencrypted ZIP members required')
            need(unique_json(archive.read('processing.json')) == metadata, 'ZIP/API metadata differs')
            need(np.array_equal(rgb(archive.read('original-upload.png'), exact_size=None), original),
                 'ZIP original upload pixels differ')
            need(np.array_equal(rgb(archive.read('resize-control.png')), canvas), 'ZIP prepared input differs')
            need(np.array_equal(support_png(archive.read('observed-support.png')), support), 'ZIP observed support differs')
            for arm in ARMS:
                raw_bytes = archive.read(arm + '-raw.npy')
                png_bytes = archive.read(arm + '.png')
                need(raw_bytes == checked_bytes(folder / (arm + '-raw.npy')) and
                     png_bytes == checked_bytes(folder / (arm + '.png')), 'Unpacked evidence differs from ZIP')
                raw = raw_array(raw_bytes)
                image = rgb(png_bytes)
                need(np.array_equal(image, np.floor(raw * np.float32(255)).astype(np.uint8)), 'PNG floor differs')
                input_float = canvas.astype(np.float32) / np.float32(255)
                need(np.array_equal(raw[~support], input_float[~support]) and
                     np.array_equal(image[~support], canvas[~support]), 'Unobserved pixels were changed')
                declaration = metadata['outputs'][arm]
                need(digest(raw.tobytes()) == declaration['raw_rgb_sha256'] and
                     digest(image.tobytes()) == declaration['PNG_RGB8_sha256'] and
                     digest(png_bytes) == declaration['PNG_file_sha256'], 'Output pixel/file hash differs')
                provenance = declaration['metadata']
                need(provenance['arm'] == arm and provenance['fidelity'] == 1 and provenance['adain'] is True and
                     provenance['internal_resolution'] == 512 and provenance['output_resolution'] == 256 and
                     provenance['alignment'] == 'none' and provenance['display_processing'] == 'none' and
                     provenance['model_qualified'] is False and provenance['main_app_default_changed'] is False,
                     'Per-arm runtime/qualification differs')
                if arm == 'original':
                    need(provenance['weights_sha256'] == ORIGINAL_WEIGHTS_SHA and
                         provenance['model_tensor_sha256'] == ORIGINAL_TENSOR_SHA, 'Original benchmark provenance differs')
                elif arm == 'r2':
                    need(provenance['weights_sha256'] == R2_WEIGHTS_SHA and
                         provenance['model_tensor_sha256'] == R2_TENSOR_SHA and provenance['training_updates'] == 50,
                         'R2 provenance differs')
                else:
                    binding = manifest['arms'][arm]
                    need(provenance['weights_sha256'] == binding['export']['sha256'] and
                         provenance['adapter_tensor_sha256'] == binding['adapter_tensor_sha256'] and
                         provenance['source_checkpoint_sha256'] == binding['source_checkpoint_sha256'] and
                         provenance['base_tensor_sha256'] == R2_TENSOR_SHA and provenance['R2_weights_sha256'] == R2_WEIGHTS_SHA and
                         provenance['source_training_updates'] == 96 and provenance['additional_continuation_updates'] == 72 and
                         provenance['protocol_sha256'] == manifest['protocol_sha256'] and
                         provenance['all515_R2_tensors_frozen'] is True and
                         provenance['shared_frozen_R2_base_serial_inference'] is True, 'Spatial adapter provenance differs')
                raw_png[arm] = {'RAW': raw, 'PNG': image.astype(np.float32) / np.float32(255)}
        if target is not None:
            result_metrics['resize_control'] = metrics(canvas.astype(np.float32) / np.float32(255), target, support)
        for arm in ARMS:
            result_metrics[arm] = {kind: metrics(raw_png[arm][kind], target, support) if target is not None else None
                                   for kind in ('RAW', 'PNG')}
            result_metrics[arm]['diagnostic_distance_to_R2'] = {
                kind: float(np.abs(raw_png[arm][kind][support].astype(np.float64) -
                                   raw_png['r2'][kind][support].astype(np.float64)).mean(dtype=np.float64))
                for kind in ('RAW', 'PNG')}
        compare_numeric(saved['target_metrics'], result_metrics, row['id'], metric_differences)
        result = {'id': row['id'], 'cohort': row['cohort'], 'source': row['source'],
                  'reference_id': row['reference_id'], 'profile': row['profile'],
                  'input_review': row['input_review'], 'appearance_flag': row.get('appearance_flag', False),
                  'evidence': row['evidence'], 'target_metrics': result_metrics,
                  'native_aligned_clean_reference': False if target is None else None,
                  'inference_only_diagnostic_not_user_admission': row['inference_only_diagnostic_not_user_admission'],
                  'geometry_and_original_upload_exact': True, 'all_four_RAW_PNG_outside_support_exact': True,
                  'API_usable_field_is_research_execution_override': row['inference_only_diagnostic_not_user_admission']}
        if target is not None:
            groups = ('all36', 'appearance_flag9' if row.get('appearance_flag') else 'unflagged27',
                      'profile_' + row['profile'], 'review_' + row['input_review']['decision'])
            for group in groups:
                for arm in ARMS:
                    for view in ('RAW', 'PNG'):
                        paired_groups[group][arm][view].append(result_metrics[arm][view])
            result['U_S_per_case_error_change_from_R2'] = {
                arm: {view: {key: result_metrics[arm][view][key] - result_metrics['r2'][view][key]
                             for key in ('mae', 'mean_centered_residual_mae',
                                         'adjacent_RGB_residual_gradient_mae', 'absolute_luminance_bias')}
                      for view in ('RAW', 'PNG')} for arm in ('spatial_u', 'spatial_s')}
        else:
            native_groups[row['source']].append(result)
        summary_rows.append(result)
        if index == 0 or (index + 1) % 12 == 0 or index == 67:
            print(json.dumps({'audited_cases': index + 1, 'of': 68,
                              'seconds': round(time.monotonic() - started, 2)}), flush=True)
    need(len(expected_transport_directories) == 68 and
         {p.name for p in (run / 'transport').iterdir()} == expected_transport_directories,
         'Exact68 original-upload transport namespaces required')
    need(len(transport_physical) == 1088, 'Exact1088 retained single-arm transport files required')
    before, after = comparison['server_before'], comparison['server_after']
    counters = ('neural_forward_attempts', 'neural_forward_returns', 'neural_forward_successes', 'completed_comparisons')
    counter_deltas = {key: after[key] - before[key] for key in counters}
    need(counter_deltas == comparison['observed_server_counter_delta'] and
         all(counter_deltas[key] >= 272 for key in counters[:3]) and counter_deltas['completed_comparisons'] >= 272,
         'Saved server counter arithmetic differs')
    expected_deltas = dict(zip(counters, (272, 272, 272, 272)))
    need(comparison['other_server_activity_observed'] is (counter_deltas != expected_deltas),
         'Concurrent server activity status differs')
    for arm in ARMS:
        need(after['per_arm'][arm]['successes'] - before['per_arm'][arm]['successes'] ==
             comparison['per_arm_server_success_deltas'][arm] >= 68, 'Per-arm counter arithmetic differs')
    regression_counts = {}
    paired_results = [row for row in summary_rows if row['evidence'] != 'native_unpaired']
    for arm in ('spatial_u', 'spatial_s'):
        regression_counts[arm] = {}
        for view in ('RAW', 'PNG'):
            regression_counts[arm][view] = {}
            for key in ('mae', 'mean_centered_residual_mae', 'adjacent_RGB_residual_gradient_mae', 'absolute_luminance_bias'):
                changes = [row['U_S_per_case_error_change_from_R2'][arm][view][key] for row in paired_results]
                regression_counts[arm][view][key] = {'lower_error_cases': sum(x < 0 for x in changes),
                    'equal_error_cases': sum(x == 0 for x in changes), 'higher_error_cases': sum(x > 0 for x in changes),
                    'worst_increase': max(changes), 'mean_change': float(np.mean(changes, dtype=np.float64))}
    result = {'format': 'codeformer-spatial-development-independent-saved-output-audit-v1-r2',
        'complete': True, 'audit_pass': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'auditor': pin(Path(__file__)), 'cohort': pin(ROOT / COHORT), 'comparison': pin(run / 'comparison.json'),
        'model_manifest': pin(ROOT / MODEL_MANIFEST), 'runtime': {'python': sys.version, 'numpy': np.__version__},
        'cases': summary_rows, 'counts': {'inputs': 68, 'declared_own_response_forwards': 272,
            'per_arm_outputs': 68, 'RAW_files': 272, 'PNG_files': 272, 'ZIPs': 68,
            'case_files': len(physical), 'native_unpaired': 32, 'paired_filter_DEV': 36,
            'flagged08025_retained': 9, 'native_input_only_controls': 16, 'filter_insufficient_controls': 24,
            'control_run_folders': 0, 'original_single_arm_responses': 272,
            'original_single_arm_ZIPs': 272, 'retained_transport_files': len(transport_physical),
            'total_audited_payload_files_excluding_comparison': len(physical) + len(transport_physical)},
        'transport_version': 'single-arm-persistent-exact-body-r2',
        'all272_original_HTTP_bodies_headers_server_ZIPs_metadata_verified': True,
        'all68_client_merges_byte_join_original_four_server_outputs': True,
        'original_failed_allarm_run_retained': pin(ROOT / previous_path),
        'original_failed_R1_single_arm_run_retained': pin(ROOT / previous_r1_path),
        'five_prior_undelivered_forwards_excluded_from272_response_count': True,
        'observed_server_counter_deltas': counter_deltas,
        'server_concurrency_observed': comparison['other_server_activity_observed'],
        'independently_recomputed_metric_max_absolute_difference': max(metric_differences, default=0),
        'arithmetic_comparison_policy': {'absolute_tolerance': ABS_TOLERANCE, 'relative_tolerance': REL_TOLERANCE,
            'scope': 'independent checker arithmetic only; no image quality threshold changed',
            'SSIM': 'independent NumPy summed-area valid7x7 means, sample49/48 covariance, C1=.01^2,C2=.03^2, RGBchannel mean'},
        'paired_summary': {group: {arm: {view: scalar_summary(values) for view, values in views.items()}
                                         for arm, views in arms.items()} for group, arms in paired_groups.items()},
        'paired_per_case_regression_counts': regression_counts,
        'native_distance_to_R2_summary': {source: {arm: {view: float(np.mean([
            row['target_metrics'][arm]['diagnostic_distance_to_R2'][view] for row in rows], dtype=np.float64))
            for view in ('RAW', 'PNG')} for arm in ARMS} for source, rows in native_groups.items()},
        'input_review_labels_preserved': True, 'research_overrides_are_not_automatic_threshold_acceptance': True,
        'all32_native_originals_reprepare_exact': True, 'all272_RAW_PNG_outside_support_exact': True,
        'original_upload_canvas_support_and_processing_provenance_exact': True,
        'no_aligned_native_clean_target_or_native_PSNR_SSIM': True,
        'model_qualified': False, 'restoration_accepted': False, 'quality_gates_changed': False,
        'main_app_default_changed': False, 'independent_FINAL_review': False, 'goal_complete': False,
        'full406_478_evaluation_remains_required': True,
        'audit_neural_HTTP_gradient_optimizer_VM_browser_calls': 0, 'torch_imported': 'torch' in sys.modules,
        'seconds': time.monotonic() - started}
    need('torch' not in sys.modules, 'This saved-only audit imported Torch')
    need(not receipt.exists(), 'Preserve existing audit; use a fresh receipt')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='outputs/codeformer_spatial_dev_review_v1/run_v1_r2')
    parser.add_argument('--receipt', default='outputs/codeformer_spatial_dev_review_v1/independent_saved_output_audit_v1_r2.json')
    args = parser.parse_args()
    run = ROOT / relative(args.run_dir)
    receipt = ROOT / relative(args.receipt)
    need(not receipt.exists(), 'Existing audit receipt must be preserved')
    try:
        result = audit(run, receipt)
    except BaseException as error:
        if not receipt.exists():
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({'complete': False, 'audit_pass': False,
                'error': {'type': type(error).__name__, 'message': str(error)},
                'neural_HTTP_gradient_optimizer_VM_browser_calls': 0,
                'model_qualified': False, 'quality_gates_changed': False}, indent=2) + '\n',
                encoding='utf-8', newline='\n')
        raise
    print(json.dumps({'complete': result['complete'], 'audit_pass': result['audit_pass'],
                      'counts': result['counts'], 'seconds': result['seconds'],
                      'receipt': str(receipt), 'model_qualified': False}), flush=True)


if __name__ == '__main__':
    main()
