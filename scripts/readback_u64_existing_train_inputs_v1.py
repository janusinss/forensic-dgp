"""Independent bounded reconstruction and severity readback of retained inputs.

NumPy/PIL saved-input math only; no reviewed script, SciPy, model, output metric,
gradient, parameter application, training, VM or browser operation is invoked.
"""
from datetime import datetime, timezone
import ast
import hashlib
import importlib.abc
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import statistics
import sys


class NoFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'tensorflow', 'jax', 'scipy', 'basicsr', 'facelib'}:
            raise ImportError('Framework/SciPy execution prohibited in independent input readback')
        return None


sys.meta_path.insert(0, NoFrameworks())
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
OUT = ROOT / 'outputs/codeformer_u64_train_evidence_review_v1'
SOURCE = ROOT / 'scripts/audit_u64_existing_train_inputs_v1.py'
PARENT_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
FIELDS = ('input_to_clean_MAE', 'input_to_clean_adjacent_error', 'input_to_clean_PSNR_dB')
ARITHMETIC_BOUND = 5e-12


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def relative(name):
    need(type(name) is str and name and ':' not in name and '\\' not in name, 'Canonical relative input path')
    path = PurePosixPath(name)
    need(not path.is_absolute() and path.as_posix() == name and all(v not in ('', '.', '..') for v in path.parts), 'Confined input path')
    return name


class Reader:
    def __init__(self):
        self.pins = {}

    def read(self, path, expected=None):
        path = Path(path).absolute()
        need(path.is_relative_to(ROOT), 'Workspace-bound readback')
        for current in (path, *path.parents):
            info = current.lstat()
            need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                 'Ordinary unlinked input ancestry')
            need(stat.S_ISREG(info.st_mode) if current == path else stat.S_ISDIR(info.st_mode), 'Ordinary file/directory')
        before = path.stat()
        need(before.st_nlink == 1 and 0 < before.st_size < 16 * 1024**2, 'Bounded single-link non-model artifact')
        raw = path.read_bytes(); after = path.stat()
        need((before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_ino, after.st_size, after.st_mtime_ns), 'File changed during read')
        pin = {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': digest(raw)}
        if expected is not None:
            need(all(pin[key] == expected[key] for key in ('bytes', 'sha256')), 'Bound artifact differs')
        need(pin['path'] not in self.pins or self.pins[pin['path']] == pin, 'Consumed artifact changed')
        self.pins[pin['path']] = pin
        return raw

    def parent(self, row):
        return self.read(PARENT / relative(row['path']), row)

    def after(self):
        before = list(self.pins.values())
        for row in before:
            self.read(ROOT / row['path'], row)
        need(len(before) == len(self.pins), 'Consumed namespace stable')
        return sorted(before, key=lambda row: row['path'])


def decode(raw, side):
    with Image.open(io.BytesIO(raw)) as value:
        need(value.format == 'PNG' and value.mode == 'RGB' and value.size == (side, side)
             and getattr(value, 'n_frames', 1) == 1, 'Exact single-frame RGB PNG shape')
        return np.asarray(value).copy()


def reflected_gaussian_codes(clean):
    """Two indexed half-sample reflected 9-tap convolutions, no np.pad/SciPy."""
    kernel = np.asarray([math.exp(-(offset * offset) / (2 * 1.2 * 1.2)) for offset in range(-4, 5)], np.float64)
    kernel /= math.fsum(kernel)
    value = clean.astype(np.float64)
    for axis in (0, 1):
        length = value.shape[axis]; positions = np.arange(length, dtype=np.int64)
        filtered = np.zeros_like(value)
        for offset, weight in zip(range(-4, 5), kernel):
            wrapped = (positions + offset) % (2 * length)
            reflected = np.where(wrapped < length, wrapped, 2 * length - 1 - wrapped)
            filtered += np.take(value, reflected, axis=axis) * weight
        value = filtered
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def dark_lookup():
    codes = []
    for code in range(256):
        srgb = code / 255.
        linear = srgb / 12.92 if srgb <= .04045 else math.pow((srgb + .055) / 1.055, 2.4)
        exposure = linear * .35
        encoded = 12.92 * exposure if exposure <= .0031308 else 1.055 * math.pow(exposure, 1. / 2.4) - .055
        codes.append(math.floor(min(1., max(0., encoded)) * 255 + .5))
    return np.asarray(codes, np.uint8)


def severity(observed, target):
    # Evaluate integer residual and edge differences before normalization. This
    # avoids reusing the reviewed script's separate normalized-array subtraction.
    residual = observed.astype(np.int32) - target.astype(np.int32)
    squared = residual.astype(np.float64) ** 2
    mse = math.fsum(squared.reshape(-1)) / squared.size / (255. * 255.)
    horizontal = np.abs(residual[:, 1:] - residual[:, :-1])
    vertical = np.abs(residual[1:] - residual[:-1])
    return {'input_to_clean_MAE': float(np.abs(residual).sum(dtype=np.int64)) / residual.size / 255.,
        'input_to_clean_adjacent_error': .5 * (float(horizontal.sum(dtype=np.int64)) / horizontal.size
                                             + float(vertical.sum(dtype=np.int64)) / vertical.size) / 255.,
        'input_to_clean_PSNR_dB': None if mse == 0 else -10 * math.log10(mse),
        'exact_input_equals_target': bool(np.count_nonzero(residual) == 0)}


def compare(actual, expected, at=''):
    if type(expected) is dict:
        need(type(actual) is dict and set(actual) == set(expected), 'Exact readback dictionary: ' + at)
        for key in expected:
            compare(actual[key], expected[key], at + '/' + key)
    elif type(expected) is list:
        need(type(actual) is list and len(actual) == len(expected), 'Exact readback list: ' + at)
        for i, value in enumerate(expected):
            compare(actual[i], value, at + '/' + str(i))
    elif type(expected) is float:
        need(type(actual) in (int, float) and math.isfinite(actual)
             and abs(actual - expected) <= ARITHMETIC_BOUND, 'Input severity arithmetic: ' + at)
    else:
        need(type(actual) is type(expected) and actual == expected, 'Exact readback value: ' + at)


def main():
    reader = Reader(); reader.read(Path(__file__).resolve())
    reviewed_raw = reader.read(OUT / 'inputs_review_v1.json'); reviewed = json.loads(reviewed_raw)
    source_raw = reader.read(SOURCE, reviewed['source'])
    tree = ast.parse(source_raw, filename=str(SOURCE), feature_version=(3, 10))
    need(any(isinstance(node, ast.FunctionDef) and node.name == 'main' for node in tree.body), 'Reviewed script guarded main exists')
    need(not any(isinstance(node, ast.Import) and any(alias.name.split('.')[0] in
             {'torch', 'tensorflow', 'jax', 'basicsr', 'facelib'} for alias in node.names) for node in ast.walk(tree)), 'No direct model imports in reviewed source')
    need(reviewed['complete'] is True and reviewed['existing_train_input_reconstruction_pass'] is True
         and reviewed['unresolved_blockers'] == [] and reviewed['parent_protocol_sha256'] == PARENT_SHA,
         'Complete reviewed input receipt')
    raw = reader.read(PARENT / 'protocol.json')
    need(len(raw) == 279879 and digest(raw) == PARENT_SHA, 'Exact sealed parent protocol')
    protocol = json.loads(raw)
    manifest_raw = reader.parent(protocol['data']['provenance']); manifest = json.loads(manifest_raw)
    need(len(protocol['cases']) == 96 and len(manifest['sources']) == 24 and len({row['reference_id'] for row in manifest['sources']}) == 24,
         'Fixed96/24 unique input evidence')
    operators = manifest['operators']
    gaussian = operators['global_gaussian_1p2']
    need(gaussian['function'] == 'scipy.ndimage.gaussian_filter' and gaussian['dtype'] == 'float64'
         and gaussian['sigma'] == [1.2, 1.2, 0] and gaussian['mode'] == 'reflect'
         and gaussian['truncate'] == 3.0 and gaussian['radius_pixels'] == 4
         and gaussian['quantization'] == 'rint then clip[0,255] then uint8', 'Exact single Gaussian specification')
    need(operators['local_gaussian_1p2']['outside_original_pixels_exact'] is True
         and '[64:192,64:192]' in operators['local_gaussian_1p2']['operator']
         and operators['lowlight_linear_0p35']['exposure'] == 'linear * 0.35'
         and operators['lowlight_linear_0p35']['additional_blur_noise_JPEG_crop_alignment'] is False,
         'Local hard-paste and exposure-only operator specification')
    references, target_bytes, source_rows = {}, {}, []; lookup = dark_lookup()
    for item in manifest['sources']:
        ref = item['reference_id']; source_pin, target_pin = item['source'], item['target256']
        source_data = reader.read(ROOT / relative(source_pin['path']), source_pin)
        target_data = reader.read(ROOT / relative(target_pin['path']), target_pin)
        source, clean = decode(source_data, 1024), decode(target_data, 256)
        need(digest(source.tobytes()) == item['source_RGB8_sha256']
             and digest(clean.tobytes()) == item['target_RGB8_sha256'], 'Photographic source/target RGB pin')
        expected_clean = np.asarray(Image.fromarray(source).resize((256, 256), Image.Resampling.LANCZOS))
        need(np.array_equal(expected_clean, clean), 'Direct1024-to256 LANCZOS target pixels exact')
        need(hashlib.md5(source_data).hexdigest() == item['official_file_MD5']
             and hashlib.md5(source.tobytes()).hexdigest() == item['official_pixel_MD5'], 'Retained official metadata hashes exact')
        blurred = reflected_gaussian_codes(clean)
        local = clean.copy(); local[64:192, 64:192] = blurred[64:192, 64:192]
        references[ref] = {'clear': clean, 'local_gaussian_1p2': local,
            'global_gaussian_1p2': blurred, 'lowlight_linear_0p35': lookup[clean]}
        target_bytes[ref] = target_data
        source_rows.append({'reference_id': ref, 'source1024_RGB8_sha256': digest(source.tobytes()),
            'target256_RGB8_sha256': digest(clean.tobytes()), 'direct_LANCZOS_exact': True,
            'retained_official_file_and_pixel_MD5_match': True,
            'independent_Gaussian_matches_declared_scipy_pixels': True})
    # The independent Gaussian equals all24 actual declared global-blur inputs;
    # the reviewed script's SciPy comparison is source evidence, not rerun here.
    compare(source_rows, reviewed['sources'])
    outside = np.ones((256, 256), bool); outside[64:192, 64:192] = False
    cases, clear_count, outside_count, role_cases = [], 0, 0, {role: 0 for role in ROLES}
    grouped = {}
    for row in protocol['cases']:
        ref, role = row['reference_id'], row['role']
        need(ref in references and role in ROLES and row['split'] == 'TRAIN', 'Only admitted photographic TRAIN roles')
        need(protocol['files'][row['input']['path']] == {key: row['input'][key] for key in ('bytes', 'sha256')}
             and protocol['files'][row['target']['path']] == {key: row['target'][key] for key in ('bytes', 'sha256')}, 'Parent manifest input/target admission')
        input_raw, clean_raw = reader.parent(row['input']), reader.parent(row['target'])
        observed, clean = decode(input_raw, 256), decode(clean_raw, 256)
        need(clean_raw == target_bytes[ref] and digest(observed.tobytes()) == row['input_RGB8_sha256']
             and digest(clean.tobytes()) == row['target_RGB8_sha256'], 'Exact historical target bytes and case RGB pins')
        need(np.array_equal(observed, references[ref][role]), 'Independent operator equals actual saved input')
        if role == 'clear':
            need(input_raw == clean_raw, 'Clear input exact target PNG bytes'); clear_count += 1
        if role == 'local_gaussian_1p2':
            need(np.array_equal(observed[outside], clean[outside]), 'Exact local outside original pixels'); outside_count += 1
        role_cases[role] += 1; grouped.setdefault(row['cohort'], set()).add(ref)
        cases.append({'id': row['id'], 'reference_id': ref, 'role': role, 'cohort': row['cohort'],
            'historical_split': row['split'], 'operator_exact_RGB8_reconstruction': True,
            'stage_target_exact_historical_photographic_bytes': True,
            'local_outside_pixels_unchanged': True if role == 'local_gaussian_1p2' else None,
            'whole256_input_only_severity': severity(observed, clean)})
    compare(cases, reviewed['cases'])
    need(all(count == 24 for count in role_cases.values()) and clear_count == outside_count == 24
         and len(grouped['optimize']) == 16 and len(grouped['monitor']) == 8
         and not grouped['optimize'] & grouped['monitor'], 'Exact four-role24-case disjoint16/8 split census')
    summary = {}
    for role in ROLES:
        rows = [row['whole256_input_only_severity'] for row in cases if row['role'] == role]
        values = {}
        for field in FIELDS:
            numbers = [row[field] for row in rows if row[field] is not None]
            values[field] = None if not numbers else {'minimum': min(numbers), 'maximum': max(numbers),
                'mean': statistics.fmean(numbers), 'median': statistics.median(numbers)}
        summary[role] = {'cases': len(rows), **values}
    compare(summary, reviewed['role_input_only_severity_summary'])
    expected_coverage = {'Gaussian_sigma_at256': [1.2], 'Gaussian_truncate': 3.0, 'Gaussian_radius': 4,
        'local_half_open_ROI': [64, 64, 192, 192], 'local_soft_edge_transition': False,
        'dark_operator': 'linear-sRGB exposure0.35 only',
        'motion_optical_resampling_noise_compression_camera_calibration_in_this96case_operator': False}
    compare(expected_coverage, reviewed['coverage'])
    need(reviewed['model_framework_gradient_optimizer_parameter_VM_browser_calls'] == 0
         and reviewed['model_qualified'] is False
         and any('not model improvement or native CCTV' in text for text in reviewed['limits'])
         and any('not the reviewed Protect/Apply' in text for text in reviewed['limits'])
         and any('No next data/loss/path change, pilot, packet or model promotion is selected.' == text for text in reviewed['limits']),
         'Correct unchanged source-input evidence scope')
    for row in reviewed['consumed_before_after_exact']:
        reader.read(ROOT / relative(row['path']), row)
    need(not any(name.split('.')[0] in {'torch', 'torchvision', 'tensorflow', 'jax', 'scipy', 'basicsr', 'facelib'} for name in sys.modules), 'No model/framework/SciPy execution')
    files = reader.after()
    result = {'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'inputs_readback_pass': True, 'unresolved_blockers': [],
        'reviewed_source': reviewed['source'], 'reviewed_input_receipt': reader.pins[(OUT / 'inputs_review_v1.json').relative_to(ROOT).as_posix()],
        'parent_protocol_sha256': PARENT_SHA,
        'source1024_to_target256_LANCZOS_exact': 24, 'independent_indexed_Gaussian_matches_actual_saved_inputs': 24,
        'independent_scalar_dark_lookup_matches_actual_saved_inputs': 24,
        'all_four_operators_actual_RGB8_exact': 96, 'clear_byte_exact': 24, 'local_outside_pixel_exact': 24,
        'role_case_counts': role_cases, 'optimization_photographs': 16, 'monitor_photographs': 8,
        'input_only_severity_recomputed': True, 'severity_arithmetic_readback_bound_only': ARITHMETIC_BOUND,
        'whole256_input_only_severity_summary': summary, 'recomputed_case_severity': cases,
        'operator_coverage': expected_coverage, 'reviewed_input_source_pins': files,
        'all_consumed_files_before_after_exact': True,
        'source_review_notes': ['Gaussian radius4 is the declared truncate3*sigma1.2 rounded kernel radius; indexed half-sample reflection reproduces actual quantized saved pixels.',
            'Whole256 severity includes local paste seam contributions and is distinct from central128 model-output gate metrics.',
            'Local hard-paste synthetic corruption preserves original outside pixels but is not Protect/Apply preprocessing.',
            'Only Gaussian sigma1.2 and linear-sRGB exposure0.35 are covered; hashes do not establish person identity, pretraining exclusion, calibrated CCTV or recoverability.'],
        'no_new_pilot_recipe_loss_model_or_packet_selected': True,
        'reviewed_script_or_model_executed': False, 'SciPy_comparison_rerun': False,
        'SciPy_claim_scope': 'Reviewed source compares independent Gaussian to SciPy; this readback independently authenticates all24 actual saved Gaussian outputs without importing SciPy.',
        'model_framework_gradient_optimizer_parameter_training_VM_browser_calls': 0,
        'model_qualified': False, 'source': reader.pins[Path(__file__).relative_to(ROOT).as_posix()]}
    destination = OUT / 'inputs_readback_v1.json'
    with destination.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'complete': True, 'inputs_readback_pass': True, 'sources': 24, 'exact_inputs': 96,
        'consumed_files': len(files), 'unresolved_blockers': [], 'new_pilot_selected': False}, indent=2))


if __name__ == '__main__':
    main()
