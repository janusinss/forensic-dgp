"""CPU-only audit of explicitly user-supplied, unpaired R2 app examples.

Reads already-recorded HTTP responses and actual downloads. It makes no
requests, imports no model code and performs no inference. Unknown user images
are not CCTV provenance, TRAIN, DEV, FINAL, clean references or identity proof.
"""
import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA = '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'
R2_SHA = 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
R2_STATE_SHA = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
CHECKPOINT_SHA = 'ec115accce587307d005b4e1e0128991f89fb4095b1f70d95b9e065663a8fa93'
PROVENANCE = 'user_supplied_unknown'
EVIDENCE = 'unpaired_user_example'
SOURCE_FILES = ('research_restoration_web.py', 'dgp_face_workflow_v3.py',
    'dgp_face_restoration.py', 'templates/research_restoration.html',
    'static/research_restoration.js')
LIMIT = 32 * 1024**2
Y = np.asarray([.2126, .7152, .0722], dtype=np.float64)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json_bytes(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data.decode('utf-8'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def read_file(pin, cap=LIMIT):
    require(isinstance(pin, dict) and set(pin) == {'path', 'sha256', 'bytes'}, 'Exact file pin required')
    path = Path(pin['path'])
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    for part in (path, *path.parents):
        info = part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
            'Linked/reparse input is forbidden')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'Unlinked regular input required')
    require(type(pin['bytes']) is int and 0 <= pin['bytes'] <= cap and before.st_size == pin['bytes'],
        'Bounded exact file size required')
    data = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
        (after.st_size, after.st_mtime_ns, after.st_ino), 'Input changed during read')
    require(len(data) == pin['bytes'] and digest(data) == pin['sha256'], 'Input fingerprint differs')
    return data


def file_pin(path, fingerprint, size=None):
    actual = Path(path) if Path(path).is_absolute() else ROOT / path
    return {'path': str(path), 'sha256': fingerprint,
        'bytes': actual.stat().st_size if size is None else size}


def source_pin(case):
    return file_pin(case['source_path'], case['source_sha256'], case['source_bytes'])


def response_pin(case):
    # Browser recorded response SHA; byte count is independently observed here.
    request = case['request']
    return file_pin(request['response_path'], request['response_sha256'])


def png_rgb(data, size=None, source=False):
    with Image.open(io.BytesIO(data)) as image:
        require(image.format in (('PNG', 'JPEG') if source else ('PNG',)), 'Unexpected image format')
        require(max(image.size) <= 4096 and image.width * image.height <= 16_000_000, 'Image bounds exceeded')
        if source:
            image = ImageOps.exif_transpose(image).convert('RGB')
        else:
            require(image.mode == 'RGB', 'Exact RGB PNG required')
        require(size is None or image.size == size, 'Unexpected image dimensions')
        return np.asarray(image).copy()


def prepare(rgb):
    """Independent NumPy/Pillow reproduction; never import torch-bearing app."""
    height, width = rgb.shape[:2]
    side = max(height, width); left, top = (side - width) // 2, (side - height) // 2
    square = Image.new('RGB', (side, side), (128, 128, 128))
    square.paste(Image.fromarray(rgb), (left, top))
    support = Image.new('L', (side, side), 0)
    support.paste(255, (left, top, left + width, top + height))
    canvas = np.asarray(square.resize((256, 256), Image.Resampling.BILINEAR)).copy()
    observed = np.asarray(support.resize((256, 256), Image.Resampling.NEAREST)) > 0
    yy, xx = np.nonzero(observed)
    geometry = {'native_size': [width, height], 'native_center_offset': [left, top],
        'observed_bounds_256': [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)],
        'output_size': [256, 256], 'policy': 'dgp256-observed-canvas-v1',
        'input_rgb_sha256': digest(rgb.tobytes()), 'canvas_rgb_sha256': digest(canvas.tobytes()),
        'observed_sha256': digest(observed.tobytes()),
        'removal_mask_sha256': digest(np.zeros((256, 256), bool).tobytes())}
    return canvas, observed, geometry


def raw_array(data):
    require(len(data) <= 3 * 256 * 256 * 4 + 256, 'Bounded RAW file required')
    stream = io.BytesIO(data)
    array = np.load(stream, allow_pickle=False)
    require(not stream.read(1), 'Trailing RAW data')
    require(array.shape == (256, 256, 3) and array.dtype == np.float32 and array.flags.c_contiguous
        and np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1,
        'Finite contiguous float32 RGB256 RAW required')
    return array


def data_url(value, mime):
    prefix = 'data:' + mime + ';base64,'
    require(isinstance(value, str) and value.startswith(prefix) and len(value) <= LIMIT * 2,
        'Expected bounded data URL')
    return base64.b64decode(value[len(prefix):], validate=True)


def zip_members(data):
    expected = {'original-upload.png', 'resize-control.png', 'observed-support.png',
        'original.png', 'r2.png', 'original-raw.npy', 'r2-raw.npy', 'processing.json'}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == expected, 'ZIP exact unique namespace differs')
        require(all(not item.is_dir() and item.file_size <= LIMIT for item in archive.infolist()) and
            sum(item.file_size for item in archive.infolist()) <= LIMIT, 'ZIP payload bounds exceeded')
        require(archive.testzip() is None, 'ZIP CRC mismatch')
        return {name: archive.read(name) for name in names}


def validate_manifest(manifest):
    require(manifest.get('complete') is True and manifest['provenance'] == PROVENANCE
        and manifest['evidence'] == EVIDENCE and manifest['paired_metrics_available'] is False
        and manifest['new_training'] is False and manifest['model_qualified'] is False,
        'Completed unknown-provenance unpaired browser recording required')
    cases = manifest['cases']
    require(len(cases) == 5 and {c['id'] for c in cases} == {'wild1', 'wild2', 'wild3', 'wild4', 'wild5'},
        'Exactly the five separately bound user examples required')
    require(set(manifest['source_pins']) == set(SOURCE_FILES), 'Complete trusted app-source pins required')
    artifact_paths = []
    for case in cases:
        require(case['provenance'] == PROVENANCE and case['evidence'] == EVIDENCE,
            'Unknown user examples may not be relabeled as CCTV/TRAIN/DEV/FINAL or paired evidence')
        require(case['input_review'] == ('usable' if case['id'] in ('wild1', 'wild2', 'wild3') else 'needs_clearer'),
            'Preserve prospectively declared input-only decisions')
        require(isinstance(case['review_reason'], str) and case['review_reason'].strip() and
            isinstance(case['review_uncertainty'], str) and case['review_uncertainty'].strip(),
            'Input-only reason and uncertainty must be explicit')
        request = case['request']
        require(request['method'] == 'POST' and request['endpoint'] == '/restoration-test/compare'
            and request['selection'] == 'both',
            'Recorded request differs from the input-only plan')
        require(request['status'] == (200 if case['input_review'] == 'usable' else 422),
            'Unexpected HTTP status')
        require(isinstance(request['elapsed_seconds'], (int, float)) and
            np.isfinite(request['elapsed_seconds']) and request['elapsed_seconds'] >= 0, 'Invalid elapsed time')
        expected = {'original_png', 'r2_png', 'bundle_zip'} if request['status'] == 200 else set()
        require(set(case['downloads']) == expected, 'Download artifacts disagree with the response status')
        artifact_paths.extend([request['response_path'], *(pin['path'] for pin in case['downloads'].values())])
        if request['status'] == 422:
            require(case['refused_without_result'] is True and case['no_result_download_links'] is True,
                'Browser refusal/result-clear observation required')
    canonical = [str((ROOT / name).absolute()) if not Path(name).is_absolute() else str(Path(name).absolute())
        for name in artifact_paths]
    require(len(canonical) == len(set(canonical)), 'Every response/download needs a unique artifact path')
    return cases


def audit_success(case, response, upload):
    require(set(response) == {'original', 'outputs', 'metadata', 'bundle'} and
        set(response['outputs']) == {'original', 'r2'}, 'Exact both-model response required')
    rgb = png_rgb(upload, source=True)
    canvas, observed, geometry = prepare(rgb)
    downloaded = {key: read_file(pin) for key, pin in case['downloads'].items()}
    require(data_url(response['bundle'], 'application/zip') == downloaded['bundle_zip'], 'Downloaded bundle differs')
    members = zip_members(downloaded['bundle_zip'])
    require(np.array_equal(png_rgb(members['original-upload.png']), rgb), 'ZIP decoded upload differs')
    require(np.array_equal(png_rgb(members['resize-control.png'], (256, 256)), canvas) and
        np.array_equal(png_rgb(data_url(response['original'], 'image/png'), (256, 256)), canvas),
        'Prepared original/resize control differs')
    with Image.open(io.BytesIO(members['observed-support.png'])) as image:
        require(image.mode == 'L' and image.size == (256, 256), 'Exact observed support required')
        require(np.array_equal(np.asarray(image), observed.astype(np.uint8) * 255), 'ZIP observed support differs')
    metadata = response['metadata']
    require(read_json_bytes(members['processing.json']) == metadata, 'UTF8 response/ZIP processing metadata differs')
    require(metadata['policy'] == 'research-CodeFormer-R2-comparison-v1' and metadata['selection'] == 'both'
        and metadata['input_review'] == 'usable' and metadata['geometry'] == geometry and
        metadata['inference_only'] is True and metadata['device'] == 'cpu'
        and metadata['no_completion_or_covering_removal'] is True
        and metadata['operator_input_review_is_not_a_validated_automatic_threshold'] is True
        and metadata['model_qualified'] is False and metadata['main_app_default_changed'] is False,
        'Processing/geometry/qualification policy differs')
    outputs = {}
    for arm in ('original', 'r2'):
        body = data_url(response['outputs'][arm], 'image/png')
        require(body == downloaded[arm + '_png'] == members[arm + '.png'], 'Response/browser/ZIP PNG bytes differ')
        raw = raw_array(members[arm + '-raw.npy'])
        pixels = png_rgb(body, (256, 256))
        expected = np.floor(raw * np.float32(255)).astype(np.uint8)
        expected[~observed] = canvas[~observed]
        require(np.array_equal(pixels, expected), 'RAW floor/padding conversion differs')
        information = metadata['outputs'][arm]
        require(information['raw_rgb_sha256'] == digest(raw.tobytes()) and
            information['PNG_RGB8_sha256'] == digest(pixels.tobytes()), 'Output pixel hash differs')
        model = information['metadata']
        require(model['arm'] == arm and model['weights_sha256'] == (R2_SHA if arm == 'r2' else ORIGINAL_SHA)
            and model['model_qualified'] is False and model['main_app_default_changed'] is False
            and model['internal_resolution'] == 512 and model['output_resolution'] == 256
            and model['fidelity'] == 1.0 and model['adain'] is True and model['alignment'] == 'none'
            and model['display_processing'] == 'none', 'Model/processing pins differ')
        if arm == 'r2':
            require(model['model_tensor_sha256'] == R2_STATE_SHA and
                model['source_checkpoint_sha256'] == CHECKPOINT_SHA and model['training_updates'] == 50,
                'Actual trained R2 lineage differs')
        values = pixels.astype(np.float64) / 255
        input_values = canvas.astype(np.float64) / 255
        outputs[arm] = {'PNG_file_sha256': digest(body), 'RAW_array_sha256': digest(raw.tobytes()),
            'PNG_RGB8_sha256': digest(pixels.tobytes()), 'RAW_shape': list(raw.shape), 'RAW_dtype': str(raw.dtype),
            'RAW_min': float(raw.min()), 'RAW_max': float(raw.max()),
            'RAW_floor_PNG_padding_exact': True, 'model': model,
            'observed_output_mean_luminance': float((values @ Y)[observed].mean()),
            'observed_input_mean_luminance': float((input_values @ Y)[observed].mean()),
            'MAE_change_from_input_NOT_clean_error': float(np.abs(values[observed] - input_values[observed]).mean())}
    return {'id': case['id'], 'status': 200, 'verified': True, 'source': source_pin(case),
        'provenance': PROVENANCE, 'evidence': EVIDENCE, 'input_review': case['input_review'],
        'review_reason': case['review_reason'], 'review_uncertainty': case['review_uncertainty'],
        'native_RGB8_shape': list(rgb.shape), 'native_RGB8_sha256': digest(rgb.tobytes()),
        'geometry': geometry, 'observed_pixels': int(observed.sum()),
        'request': case['request'], 'downloads': case['downloads'], 'outputs': outputs,
        'clean_reference_metrics_computed': False, 'visual_usefulness_review_performed': False}


def audit(manifest_path, out):
    require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')), 'No neural framework allowed')
    manifest_path = Path(manifest_path).resolve(strict=True)
    require(manifest_path.stat().st_size <= LIMIT, 'Manifest bounds exceeded')
    manifest_data = manifest_path.read_bytes()
    manifest = read_json_bytes(manifest_data)
    cases = validate_manifest(manifest)
    require(not out.exists(), 'Preserve prior/partial audit; choose a new output version')
    started = time.monotonic()
    current_source_pins = {}
    for name, fingerprint in manifest['source_pins'].items():
        pin = file_pin(str(ROOT / name), fingerprint)
        read_file(pin)
        current_source_pins[name] = pin
    rows = []
    for case in sorted(cases, key=lambda row: row['id']):
        upload = read_file(source_pin(case), cap=10 * 1024**2)
        response = read_json_bytes(read_file(response_pin(case)))
        if case['request']['status'] == 200:
            rows.append(audit_success(case, response, upload))
        else:
            require(response == {'detail': 'Facial structure is insufficient. Upload a clearer face crop.'},
                'Expected clearer-input refusal without result payload')
            rgb = png_rgb(upload, source=True)
            rows.append({'id': case['id'], 'status': 422, 'verified': True, 'source': source_pin(case),
                'provenance': PROVENANCE, 'evidence': EVIDENCE, 'input_review': 'needs_clearer',
                'review_reason': case['review_reason'], 'review_uncertainty': case['review_uncertainty'],
                'native_RGB8_shape': list(rgb.shape), 'native_RGB8_sha256': digest(rgb.tobytes()),
                'request': case['request'], 'outputs': None, 'downloads': {},
                'automatic_rejection_established': False, 'runtime_forward_counter_measured': False,
                'clean_reference_metrics_computed': False})
    require(Counter(row['status'] for row in rows) == {200: 3, 422: 2}, 'Actual five-case response census differs')
    require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')), 'Neural framework imported')
    result = {'format': 'codeformer-r2-user-wild-download-audit-v1', 'complete': True, 'audit_pass': True,
        'manifest': {'path': str(manifest_path), 'sha256': digest(manifest_data), 'bytes': len(manifest_data)},
        'auditor': {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__).read_bytes())},
        'source_pins': current_source_pins, 'cases': rows, 'cases_count': 5,
        'response_and_source_code_sizes_independently_observed_from_recorded_SHA_bound_files': True,
        'both_model_comparisons': 3, 'operator_review_refusals': 2,
        'unknown_user_image_provenance_not_inferred': True, 'TRAIN_DEV_FINAL_membership_changed': False,
        'CCTV_local_ethnicity_identity_claims': False, 'paired_clean_reference_metrics_computed': False,
        'automatic_threshold_validated': False, 'quality_gates_changed': False,
        'model_qualified': False, 'user_acceptance_established': False,
        'actual_neural_inference_calls_by_auditor': 0, 'actual_training_updates_by_auditor': 0,
        'browser_or_network_calls_by_auditor': 0, 'new_image_outputs_written_by_auditor': 0,
        'CPU': {'Python': sys.version, 'NumPy': np.__version__, 'Pillow': pillow_version},
        'seconds': time.monotonic() - started}
    out.mkdir(parents=True)
    with (out / 'download_audit_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    return {'complete': True, 'audit_pass': True, 'cases': 5, 'both_model_comparisons': 3,
        'operator_review_refusals': 2, 'receipt': str(out / 'download_audit_v1.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest, args.out)), flush=True)
