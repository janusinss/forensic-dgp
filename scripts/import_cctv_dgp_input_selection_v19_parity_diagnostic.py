"""Safe diagnostic return import and independent saved-array audit; no forwards."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_input_selection_vm_v19'
FAILURE = ROOT / 'outputs/cctv_dgp_input_selection_failure_return_v19'
BASELINE = ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(value, message):
    if not value:
        raise ValueError(message)


def rgb(path):
    with Image.open(path) as im:
        require(im.mode == 'RGB' and im.size == (256, 256), 'RGB256 PNG required')
        return np.asarray(im).copy()


def raw(path):
    v = np.load(path, allow_pickle=False)
    require(v.dtype == np.float32 and v.shape == (256, 256, 3) and np.isfinite(v).all()
            and 0 <= v.min() <= v.max() <= 1, 'Finite RGB256 float32 required')
    return v


def collect(archive, receipt, destination):
    started = time.monotonic()
    def clock():
        require(time.monotonic() - started <= 90, 'Diagnostic return audit exceeds90s')
    preparation = read(ROOT / 'outputs/cctv_dgp_input_selection_v19_parity_diagnostic_preparation.json')
    source = ROOT / 'scripts/diagnose_cctv_dgp_input_selection_v19_parity.py'
    require(sha(source) == preparation['script_sha256'], 'Prepared local diagnostic source differs')
    spec = importlib.util.spec_from_file_location('v19_diagnostic_contract', source)
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    terminal = read(receipt)
    require(terminal['complete'] and terminal['archive_sha256'] == sha(archive)
            and terminal['bytes'] == archive.stat().st_size <= 32 * 1024**2
            and terminal['script_sha256'] == preparation['script_sha256']
            and terminal['original_protocol_sha256'] == d.PIN
            and terminal['training'] is False and terminal['V19_resumed'] is False
            and 0 < terminal['seconds'] <= 240
            and archive.name == d.ARCHIVE
            and Path(str(archive) + '.sha256').read_text().split() == [terminal['archive_sha256'], archive.name],
            'Diagnostic export/hash/scope differs')
    require(not destination.exists() and destination.resolve().is_relative_to((ROOT / 'outputs').resolve()),
            'Fresh workspace return required; preserve existing evidence')
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        require(sum(m.size for m in members) <= 32 * 1024**2, 'Diagnostic return exceeds32MiB')
        names = set()
        for member in members:
            p = PurePosixPath(member.name)
            require(member.isfile() and not p.is_absolute() and '..' not in p.parts
                    and str(p) == member.name and member.name not in names, 'Unsafe/duplicate diagnostic member')
            d.safe(destination, member.name)
            names.add(member.name)
        destination.mkdir(parents=True)
        stream.extractall(destination, members=members, filter='data')
    require(sha(destination / d.SCRIPT) == preparation['script_sha256'], 'Returned worker source differs')
    protocol = read(destination / 'diagnostic_protocol.json')
    require(protocol['design'] == d.DESIGN and protocol['script_sha256'] == preparation['script_sha256']
            and protocol['original_protocol_sha256'] == d.PIN, 'Returned diagnostic protocol differs')
    if terminal['diagnostic_succeeded'] is not True:
        require((destination / 'worker_failure.json').is_file(), 'Saved worker failure required')
        result = {'complete': True, 'full_diagnostic_audit': False, 'failure': read(destination / 'worker_failure.json'),
                  'archive_sha256': terminal['archive_sha256'], 'local_neural_forwards': 0,
                  'local_optimizer_updates': 0, 'local_backward_calls': 0, 'VM_repeated': False}
    else:
        r = read(destination / 'results.json')
        require(sha(destination / 'results.json') == terminal['results_sha256'] and r['complete']
                and r['design'] == d.DESIGN and r['script_sha256'] == preparation['script_sha256']
                and r['original_protocol_sha256'] == d.PIN and r['torch'] == '2.9.1+cu129'
                and r['gpu'] == 'NVIDIA L4' and 0 < r['seconds'] <= 120
                and r['forward_counts'] == {'DGP': 8} and r['optimizer_constructed'] is False
                and r['optimizer_updates'] == r['backward_calls'] == 0, 'Diagnostic result/neural scope differs')
        execution = read(FAILURE / 'outputs/input_selection_v19/execution.json')
        require(r['DGP_state_before'] == r['DGP_state_after'] == execution['states_before']['dgp'], 'DGP state differs')
        require(r['protected_original_fingerprints_before'] == r['protected_original_fingerprints_after'],
                'Original scientific files changed')
        p = read(BUNDLE / 'input_selection_protocol_v19.json')
        expected_protected = {**p['assets_sha256'], 'input_selection_protocol_v19.json': d.PIN,
            'supervisor_failure.json': d.FAILURE_SHA, 'inference.log': d.LOG_SHA,
            'outputs/input_selection_v19/fresh_training_parity.json': d.PARITY_SHA}
        require(r['protected_original_fingerprints_before'] == expected_protected, 'Original protected file bindings differ')
        for name, digest in r['artifacts_sha256'].items():
            clock()
            require(sha(d.safe(destination, name)) == digest, 'Diagnostic artifact checksum differs')
        cases = [('failed_development_case', p['cases'][0]),
            ('fixed_clear_preview', next(c for c in p['cases'] if c['reference_id'] == p['preview_reference_ids'][0] and c['profile'] == 'clear')),
            ('training_parity_case', p['training_parity_cases'][0])]
        require([(row['role'], row['id']) for row in r['rows']] == [(role, c['id']) for role, c in cases], 'Diagnostic cohort/order differs')
        refs = {ref['id']: ref for ref in p['references'] + p['training_parity_references']}
        base_rows = {row['id']: row for row in read(BASELINE / 'results.json')['rows']}
        codes = np.arange(256, dtype=np.uint8).astype(np.float32)
        scalar = r['scalar_normalization']
        cuda_codes = np.load(d.safe(destination, scalar['CUDA_values']), allow_pickle=False)
        cpu_codes = np.load(d.safe(destination, scalar['CPU_values']), allow_pickle=False)
        require(cuda_codes.dtype == np.float32 and cuda_codes.shape == (256,) and np.isfinite(cuda_codes).all()
                and np.array_equal(cpu_codes, codes / 255), 'Scalar normalization arrays differ')
        require(scalar['CUDA_matches_reciprocal_simulation'] == bool(np.array_equal(cuda_codes, codes * np.float32(1 / 255)))
                and scalar['different_values_from_CPU_division'] == int(np.count_nonzero(cuda_codes != cpu_codes))
                and scalar['maximum_difference_from_CPU_division'] == float(np.max(np.abs(cuda_codes - cpu_codes))),
                'Scalar normalization arithmetic differs')
        audited_rows, stability = [], {}
        raw_checks = png_checks = normalized_checks = 0
        for row, (role, c) in zip(r['rows'], cases):
            clock(); cid = c['id']; ref = refs[c['reference_id']]
            camera = rgb(MIXED / c['input'])
            with Image.open(MIXED / ref['observed']) as im:
                support = np.asarray(im).copy() > 0
            require(row['camera_sha256'] == sha(MIXED / c['input']) and row['support_sha256'] == sha(MIXED / ref['observed']), 'Input hashes differ')
            if role == 'training_parity_case':
                refpath = BUNDLE / ('parity/dgp_base/' + cid + '.npy')
                refraw = raw(refpath)
                refpng = np.where(support[..., None], np.floor(refraw * 255).astype(np.uint8), camera)
            else:
                previous = base_rows[cid]['arms']['retained_dgp_v2']
                refpath = BASELINE / previous['raw'] if 'raw' in previous else None
                refraw = raw(refpath) if refpath else None
                refpng = rgb(BASELINE / previous['prediction'])
            require(row['reference_raw_sha256'] == (sha(refpath) if refpath else None)
                    and row['reference_PNG_sha256'] == hashlib.sha256(refpng.tobytes()).hexdigest(), 'Reference binding differs')
            routes, inputs, raws = {}, {}, {}
            require(set(row['routes']) == {'numpy_before_GPU', 'CUDA_scalar_division'}, 'Two normalization routes required')
            for route, item in row['routes'].items():
                require(item['input'] == 'inputs/' + route + '/' + cid + '.npy'
                        and item['raw'] == 'raw/' + route + '/' + cid + '.npy'
                        and item['PNG'] == 'predictions/' + route + '/' + cid + '.png', 'Diagnostic artifact paths differ')
                norm = raw(d.safe(destination, item['input']))
                expected_norm = (camera.astype(np.float32) / 255 if route == 'numpy_before_GPU'
                                 else cuda_codes[camera])
                require(np.array_equal(norm, expected_norm), 'Exact normalized input differs')
                value = raw(d.safe(destination, item['raw']))
                png = rgb(d.safe(destination, item['PNG']))
                require(np.array_equal(png, np.where(support[..., None], np.floor(value * 255).astype(np.uint8), camera)), 'Raw/PNG composition differs')
                delta = np.abs(png.astype(np.int16) - refpng.astype(np.int16))
                expected = {**item, 'reference_PNG_equal': bool(np.array_equal(png, refpng)),
                    'reference_PNG_differing_channels': int(np.count_nonzero(delta)),
                    'reference_PNG_max_abs_channel_difference': int(delta.max()),
                    'reference_raw_max_difference': float(np.max(np.abs(value - refraw))) if refraw is not None else None}
                require(expected == item and item['tensor_dtype'] == 'torch.float32', 'Reference delta arithmetic differs')
                routes[route] = expected; inputs[route] = norm; raws[route] = value
                raw_checks += 1; png_checks += 1; normalized_checks += 1
                if role == 'failed_development_case':
                    repeated = raw(destination / ('repeat_raw/' + route + '/' + cid + '.npy'))
                    stability[route] = bool(np.array_equal(value, repeated)); raw_checks += 1
            require(row['maximum_normalized_input_difference'] == float(np.max(np.abs(inputs['numpy_before_GPU'] - inputs['CUDA_scalar_division'])))
                    and row['normalized_input_different_values'] == int(np.count_nonzero(inputs['numpy_before_GPU'] != inputs['CUDA_scalar_division']))
                    and row['maximum_DGP_raw_difference_between_routes'] == float(np.max(np.abs(raws['numpy_before_GPU'] - raws['CUDA_scalar_division']))),
                    'Between-route arithmetic differs')
            audited_rows.append({**row, 'routes': routes})
        confirmed = d.classify_normalization(audited_rows, scalar, stability)
        require(r['repeat_stability'] == stability and r['normalization_hypothesis_confirmed'] == confirmed, 'Diagnostic interpretation differs')
        result = {'complete': True, 'full_diagnostic_audit': True,
            'archive_sha256': terminal['archive_sha256'], 'results_sha256': terminal['results_sha256'],
            'raw_checks': raw_checks, 'normalized_input_checks': normalized_checks, 'raw_PNG_checks': png_checks,
            'protected_original_file_checks': len(expected_protected), 'normalization_hypothesis_confirmed': confirmed,
            'local_neural_forwards': 0, 'local_optimizer_updates': 0, 'local_backward_calls': 0,
            'whole_network_replayed_locally': False, 'quality_or_promotion_claim': False, 'VM_repeated': False}
    clock(); result['seconds'] = time.monotonic() - started
    result['auditor_sha256'] = sha(Path(__file__))
    with (destination / 'local_independent_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['archive', 'receipt', 'extract-to']:
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    collect(a.archive.resolve(), a.receipt.resolve(), a.extract_to.resolve())
