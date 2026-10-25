"""Read-only reconstruction of retained photographic TRAIN inputs; no model."""
from datetime import datetime, timezone
import hashlib
import importlib.abc
import io
import json
from pathlib import Path, PurePosixPath
import sys


class NoNeuralFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'tensorflow', 'jax', 'basicsr', 'facelib'}:
            raise ImportError('Scientific model/gradient framework prohibited in saved-input audit')


sys.meta_path.insert(0, NoNeuralFrameworks())
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
OUT = ROOT / 'outputs/codeformer_u64_train_evidence_review_v1'
PARENT_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Reader:
    def __init__(self):
        self.pins = {}

    def read(self, path, expected=None):
        path = path.resolve(strict=True)
        need(path.is_relative_to(ROOT) and path.is_file(), 'Existing workspace file required')
        need(path.stat().st_size < 16 * 1024**2, 'Bounded non-model file required')
        data = path.read_bytes()
        pin = {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': sha(data)}
        if expected is not None:
            need(all(pin[k] == expected[k] for k in ('bytes', 'sha256')), 'Exact input pin differs')
        need(pin['path'] not in self.pins or self.pins[pin['path']] == pin, 'Concurrent input change')
        self.pins[pin['path']] = pin
        return data

    def local(self, pin):
        name = PurePosixPath(pin['path'])
        need(not name.is_absolute() and '..' not in name.parts and '\\' not in pin['path'], 'Relative input name required')
        return self.read(PARENT / name, pin)

    def stable(self):
        pins = list(self.pins.values())
        for pin in pins:
            self.read(ROOT / pin['path'], pin)
        return sorted(pins, key=lambda row: row['path'])


def rgb(raw, shape):
    with Image.open(io.BytesIO(raw)) as image:
        need(image.format == 'PNG' and image.mode == 'RGB' and image.size == (shape, shape), 'Exact square RGB PNG required')
        image.load()
        return np.asarray(image).copy()


def independent_gaussian(image):
    """Separable explicit reflected extension, not the historical helper."""
    offsets = np.arange(-4, 5, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / 1.2)**2)
    weights /= weights.sum()
    result = image.astype(np.float64)
    for axis in (0, 1):
        pads = [(0, 0)] * 3
        pads[axis] = (4, 4)
        extended = np.pad(result, pads, mode='symmetric')
        output = np.zeros_like(result)
        for index, weight in enumerate(weights):
            window = [slice(None)] * 3
            window[axis] = slice(index, index + result.shape[axis])
            output += extended[tuple(window)] * weight
        result = output
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def dark(image):
    value = image.astype(np.float64) / 255
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055)**2.4) * 0.35
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear**(1 / 2.4) - 0.055)
    return np.floor(np.clip(encoded, 0, 1) * 255 + 0.5).astype(np.uint8)


def metrics(observed, clean):
    observed, clean = observed.astype(np.float64) / 255, clean.astype(np.float64) / 255
    error = observed - clean
    mse = float(np.mean(error * error))
    adjacent = (float(np.mean(np.abs(np.diff(error, axis=0)))) +
                float(np.mean(np.abs(np.diff(error, axis=1))))) / 2
    return {'input_to_clean_MAE': float(np.mean(np.abs(error))),
            'input_to_clean_adjacent_error': adjacent,
            'input_to_clean_PSNR_dB': None if mse == 0 else float(10 * np.log10(1 / mse)),
            'exact_input_equals_target': bool(np.array_equal(observed, clean))}


def main():
    reader = Reader()
    raw = reader.read(PARENT / 'protocol.json')
    need(sha(raw) == PARENT_SHA and len(raw) == 279879, 'Sealed parent protocol required')
    protocol = json.loads(raw)
    manifest = json.loads(reader.local(protocol['data']['provenance']))
    need(len(manifest['sources']) == 24 and len(protocol['cases']) == 96, 'Fixed24/96 data scope required')
    references, target_bytes, source_rows = {}, {}, []
    for item in manifest['sources']:
        ref = item['reference_id']
        source_raw = reader.read(ROOT / item['source']['path'], item['source'])
        clean_raw = reader.read(ROOT / item['target256']['path'], item['target256'])
        source, clean = rgb(source_raw, 1024), rgb(clean_raw, 256)
        need(sha(source.tobytes()) == item['source_RGB8_sha256'] and
             sha(clean.tobytes()) == item['target_RGB8_sha256'], 'Source/target pixel pin differs')
        direct = np.asarray(Image.fromarray(source).resize((256, 256), Image.Resampling.LANCZOS))
        need(np.array_equal(direct, clean), 'Direct1024to256 photographic target mismatch')
        official_md5 = hashlib.md5(source_raw).hexdigest() == item['official_file_MD5']
        official_pixels = hashlib.md5(source.tobytes()).hexdigest() == item['official_pixel_MD5']
        need(official_md5 and official_pixels, 'Retained official-file/pixel metadata mismatch')
        blurred = independent_gaussian(clean)
        scipy_result = np.clip(np.rint(gaussian_filter(clean.astype(np.float64), sigma=(1.2, 1.2, 0),
                               mode='reflect', truncate=3.0)), 0, 255).astype(np.uint8)
        need(np.array_equal(blurred, scipy_result), 'Independent Gaussian and declared library quantization differ')
        local = clean.copy(); local[64:192, 64:192] = blurred[64:192, 64:192]
        references[ref] = {'clear': clean, 'global_gaussian_1p2': blurred,
                           'local_gaussian_1p2': local, 'lowlight_linear_0p35': dark(clean)}
        target_bytes[ref] = clean_raw
        source_rows.append({'reference_id': ref, 'source1024_RGB8_sha256': sha(source.tobytes()),
            'target256_RGB8_sha256': sha(clean.tobytes()), 'direct_LANCZOS_exact': True,
            'retained_official_file_and_pixel_MD5_match': True,
            'independent_Gaussian_matches_declared_scipy_pixels': True})
    case_rows = []
    outside = np.ones((256, 256), dtype=bool); outside[64:192, 64:192] = False
    for row in protocol['cases']:
        ref, role = row['reference_id'], row['role']
        input_raw, clean_raw = reader.local(row['input']), reader.local(row['target'])
        observed, clean = rgb(input_raw, 256), rgb(clean_raw, 256)
        need(clean_raw == target_bytes[ref], 'Stage photographic target differs from retained original')
        need(sha(observed.tobytes()) == row['input_RGB8_sha256'] and
             sha(clean.tobytes()) == row['target_RGB8_sha256'], 'Case pixel pins differ')
        need(np.array_equal(observed, references[ref][role]), 'Declared operator fails saved-input reconstruction')
        if role == 'clear':
            need(input_raw == clean_raw, 'Clear control not exact photographic target bytes')
        if role == 'local_gaussian_1p2':
            need(np.array_equal(observed[outside], clean[outside]), 'Local blur changed outside original pixels')
        case_rows.append({'id': row['id'], 'reference_id': ref, 'role': role, 'cohort': row['cohort'],
            'historical_split': row['split'], 'operator_exact_RGB8_reconstruction': True,
            'stage_target_exact_historical_photographic_bytes': True,
            'local_outside_pixels_unchanged': True if role == 'local_gaussian_1p2' else None,
            'whole256_input_only_severity': metrics(observed, clean)})
    role_summary = {}
    for role in ROLES:
        rows = [r['whole256_input_only_severity'] for r in case_rows if r['role'] == role]
        values = {}
        for key in ('input_to_clean_MAE', 'input_to_clean_adjacent_error', 'input_to_clean_PSNR_dB'):
            numbers = [r[key] for r in rows if r[key] is not None]
            values[key] = None if not numbers else {'minimum': min(numbers), 'maximum': max(numbers),
                                                   'mean': float(np.mean(numbers)), 'median': float(np.median(numbers))}
        role_summary[role] = {'cases': len(rows), **values}
    sources = reader.stable()
    receipt = {'complete': True, 'existing_train_input_reconstruction_pass': True,
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'unresolved_blockers': [],
        'parent_protocol_sha256': PARENT_SHA, 'source_photographs': 24, 'paired_inputs': 96,
        'photographic_targets': 24, 'optimization_photographs': 16, 'monitor_photographs': 8,
        'exact_source_LANCZOS_targets': 24, 'exact_independent_Gaussian_reconstructions': 24,
        'exact_operator_case_reconstructions': 96, 'byte_exact_clear_controls': 24,
        'local128_hard_paste_outside_exact': 24, 'role_input_only_severity_summary': role_summary,
        'sources': source_rows, 'cases': case_rows, 'consumed_before_after_exact': sources,
        'coverage': {'Gaussian_sigma_at256': [1.2], 'Gaussian_truncate': 3.0, 'Gaussian_radius': 4,
            'local_half_open_ROI': [64, 64, 192, 192], 'local_soft_edge_transition': False,
            'dark_operator': 'linear-sRGB exposure0.35 only',
            'motion_optical_resampling_noise_compression_camera_calibration_in_this96case_operator': False},
        'limits': ['Metrics describe known synthetic input difficulty, not model improvement or native CCTV performance.',
            'No person-identity or CodeFormer-pretraining nonoverlap is established by photograph hashes.',
            'A correct mild synthetic operator does not prove realistic camera coverage or all-feature recoverability.',
            'The hard-paste local corruption is not the reviewed Protect/Apply surroundings preprocessing workflow.',
            'No next data/loss/path change, pilot, packet or model promotion is selected.'],
        'model_framework_gradient_optimizer_parameter_VM_browser_calls': 0, 'model_qualified': False,
        'source': {'path': Path(__file__).relative_to(ROOT).as_posix(),
                   'bytes': Path(__file__).stat().st_size, 'sha256': sha(Path(__file__).read_bytes())}}
    OUT.mkdir(exist_ok=True)
    with (OUT / 'inputs_review_v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print({'complete': True, 'sources': 24, 'exact_targets': 24, 'exact_inputs': 96,
           'new_model_calls': 0, 'next_pilot_selected': False})


if __name__ == '__main__':
    main()
