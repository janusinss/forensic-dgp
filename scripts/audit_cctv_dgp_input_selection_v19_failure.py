"""Audit V19's imported partial failure without network inference or training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_input_selection_vm_v19'
RETURN = ROOT / 'outputs/cctv_dgp_input_selection_failure_return_v19'
PIN = '2d707ebba97524867c4ce7610666e3abeb29638eb9054a8025983b6476e628d2'
ARCHIVE_SHA = 'f6847aa742acee19c21a16df7f44dce2456b052793cf7fa154a62cd59abcb16b'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rgb(path):
    with Image.open(path) as im:
        require(im.mode == 'RGB' and im.size == (256, 256), 'RGB256 required')
        return np.asarray(im).copy()


def raw(path):
    value = np.load(path, allow_pickle=False)
    require(value.dtype == np.float32 and value.shape == (256, 256, 3)
            and np.isfinite(value).all() and 0 <= value.min() <= value.max() <= 1,
            'Finite float32 RGB256 required')
    return value


def composed(value, camera, support):
    return np.where(support[..., None], np.floor(value * 255).astype(np.uint8), camera)


def audit(receipt):
    started = time.monotonic()
    def clock():
        require(time.monotonic() - started <= 90, 'Partial audit exceeds90s')
    p = read(BUNDLE / 'input_selection_protocol_v19.json')
    require(sha(BUNDLE / 'input_selection_protocol_v19.json') == PIN ==
            sha(RETURN / 'input_selection_protocol_v19.json'), 'Original protocol differs')
    imported = read(RETURN / 'local_import_and_audit.json')
    require(imported['complete'] and imported['full_output_audit'] is False
            and imported['original_supervisor_success'] is False
            and imported['archive_sha256'] == ARCHIVE_SHA
            and imported['bytes'] == 167982054 and imported['protocol_sha256'] == PIN,
            'Partial import/transfer binding differs')
    for name, digest in p['assets_sha256'].items():
        clock()
        require(sha(BUNDLE / name) == digest == sha(RETURN / name), 'Returned original differs:' + name)
    out = RETURN / 'outputs/input_selection_v19'
    execution = read(out / 'execution.json')
    parent = read(BUNDLE / 'lineage/v18_neural_receipt.json')
    require(execution['protocol_sha256'] == PIN and execution['training'] is False
            and execution['optimizer_constructed'] is False
            and execution['backward_calls'] == execution['optimizer_updates'] == 0
            and execution['torch'] == '2.9.1+cu129' and execution['gpu'] == 'NVIDIA L4',
            'Saved inference scope/runtime differs')
    require({k: execution['states_before'][k] for k in parent['frozen_before']} == parent['frozen_before']
            and execution['states_before']['spatial_decoder'] == p['terminal_state_hash'],
            'Initial frozen states differ')
    failure = read(RETURN / 'supervisor_failure.json')
    log = (RETURN / 'inference.log').read_text(encoding='utf-8')
    require(failure['complete'] is False and failure['resume_permitted'] is False
            and failure['protocol_sha256'] == PIN and 0 < failure['seconds'] < 1200
            and 'V19 fresh-image parity50 cases passed' in log
            and 'Fresh DGP PNG differs from declared V15 baseline' in log,
            'Expected original parity failure differs')
    require(not any((out / name).exists() for name in
                    ['results.json', 'neural_receipt.json', 'inference_timing.json', 'raw', 'predictions'])
            and not (RETURN / 'audit.log').exists(), 'Unexpected development outputs/full audit exist')
    parity = read(out / 'fresh_training_parity.json')
    require(parity['complete'] and parity['training'] is False
            and [i['id'] for i in parity['cases']] == [c['id'] for c in p['training_parity_cases']],
            'All50 ordered parity cases required')
    mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
    trainrefs = {r['id']: r for r in p['training_parity_references']}
    import sys
    sys.path.insert(0, str(BUNDLE))
    from dgp_input_selector_v19 import select_restoration
    control = {r['id']: r for r in read(BUNDLE / 'lineage/processing_results_v19.json')['decisions']}
    maximum = 0.0
    branches = {}
    for c, item in zip(p['training_parity_cases'], parity['cases']):
        clock()
        camera = rgb(mixed / c['input'])
        ref = trainrefs[c['reference_id']]
        with Image.open(mixed / ref['observed']) as im:
            support = np.asarray(im).copy() > 0
        expected = select_restoration(camera, support)
        require(set(expected) == set(item['decision'])
                and expected['branch'] == control[c['id']]['branch'], 'Frozen training input decision differs')
        for key, value in expected.items():
            actual = item['decision'][key]
            if key in ['laplacian_MSE', 'luminance_variance', 'laplacian_over_variance']:
                require(np.isfinite(actual) and abs(actual - value) <= 1e-12, 'Input feature differs:' + key)
            else:
                require(actual == value, 'Input decision differs:' + key)
        branches[expected['branch']] = branches.get(expected['branch'], 0) + 1
        for key, newname, oldfolder in [
            ('retained_dgp_v2', item['base_raw'], 'dgp_base'),
            ('structure_v18_update600', item['terminal_raw'], 'structure_update600')]:
            actual = raw(out / newname)
            previous = raw(BUNDLE / ('parity/' + oldfolder + '/' + c['id'] + '.npy'))
            delta = float(np.max(np.abs(actual - previous)))
            require(delta == item['maximum_float_differences'][key] and delta <= 2e-6
                    and np.array_equal(composed(actual, camera, support), composed(previous, camera, support)),
                    'Saved fresh training raw/PNG parity differs')
            maximum = max(maximum, delta)
    baseline = ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15'
    old_bundle = ROOT / 'outputs/cctv_dgp_generalization_vm_v15'
    bp = read(old_bundle / 'generalization_protocol_v15.json')
    oldcases = {c['id']: c for c in bp['cases']}
    oldrefs = {r['id']: r for r in bp['references']}
    image_checks = 0
    for c in p['cases']:
        clock()
        require(c == oldcases[c['id']], 'Development case metadata differs')
        require(np.array_equal(rgb(mixed / c['input']), rgb(old_bundle / oldcases[c['id']]['input'])),
                'Baseline/current camera pixels differ')
        image_checks += 1
    target_deltas = []
    for ref in p['references']:
        clock()
        for key in ['target', 'observed']:
            with Image.open(mixed / ref[key]) as a, Image.open(old_bundle / oldrefs[ref['id']][key]) as b:
                require(np.array_equal(np.asarray(a), np.asarray(b)), 'Target/support pixels differ')
            image_checks += 1
        value = np.load(out / ('target_embeddings/' + ref['id'] + '.npy'), allow_pickle=False)
        previous = np.load(baseline / ('target_embeddings/' + ref['id'] + '.npy'), allow_pickle=False)
        require(value.dtype == np.float32 and value.shape == (512,) and np.isfinite(value).all()
                and abs(float(np.linalg.norm(value)) - 1) <= 2e-6, 'Fresh embedding schema differs')
        delta = float(np.max(np.abs(value - previous)))
        require(delta <= 2e-6, 'Fresh target embedding parity differs')
        target_deltas.append(delta)
    require(len(list((out / 'target_embeddings').glob('*.npy'))) == 104, 'Target count differs')
    caches = list((out / 'cache_replay').glob('*.npz'))
    require(len(caches) == 4, 'Four training probes required')
    for path in caches:
        with np.load(path, allow_pickle=False) as a:
            require(set(a.files) == {'dgp_base', 'prior64'} and a['dgp_base'].shape == (3, 256, 256)
                    and a['prior64'].shape == (256, 64, 64)
                    and all(a[k].dtype == np.float32 and np.isfinite(a[k]).all() for k in a.files),
                    'Cached probe schema differs')
    # CPU true division is a diagnostic counterexample for the CUDA scalar
    # reciprocal optimization, not a GPU replay or proof of the stopped PNG.
    codes = np.arange(256, dtype=np.uint8).astype(np.float32)
    divide = codes / 255
    reciprocal = codes * np.float32(1 / 255)
    result = {'complete': True, 'scope': 'Independent saved partial evidence; no full development result',
        'protocol_sha256': PIN, 'archive_sha256': ARCHIVE_SHA, 'archive_bytes': 167982054,
        'original_asset_checks': len(p['assets_sha256']), 'training_parity_cases': 50,
        'training_raw_checks': 100, 'training_composition_checks': 100,
        'maximum_training_raw_difference': maximum, 'training_branches': branches,
        'training_cache_schema_checks': 4, 'baseline_input_target_support_checks': image_checks,
        'fresh_target_embeddings': len(target_deltas), 'maximum_target_embedding_difference': max(target_deltas),
        'initial_state_hashes_verified': len(execution['states_before']), 'final_state_receipt_available': False,
        'full_neural_counter_receipt_available': False, 'completed_development_predictions': 0,
        'first_stopped_case': p['cases'][0]['id'], 'supervisor_seconds_before_export': failure['seconds'],
        'original_error': 'Fresh DGP PNG differs from declared V15 baseline',
        'first_failed_raw_or_PNG_saved': False, 'full_output_audit': False,
        'normalization_hypothesis': {'V15': 'NumPy float32 division before GPU transfer',
            'V19': 'PyTorch CUDA scalar division after GPU transfer',
            'CPU_reciprocal_simulation_different_byte_values': int(np.count_nonzero(divide != reciprocal)),
            'CPU_reciprocal_simulation_maximum_difference': float(np.max(np.abs(divide - reciprocal))),
            'primary_source': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu',
            'GPU_cause_verified': False, 'fresh_GPU_diagnostic_required': True},
        'source_hashes': {'auditor': sha(Path(__file__)), 'failure_log': sha(RETURN / 'inference.log'),
            'failure_receipt': sha(RETURN / 'supervisor_failure.json'),
            'training_parity': sha(out / 'fresh_training_parity.json'),
            'V15_runner': sha(old_bundle / 'scripts/run_cctv_dgp_generalization_v15.py'),
            'V19_runner': sha(BUNDLE / 'scripts/run_cctv_dgp_input_selection_v19.py')},
        'seconds': time.monotonic() - started, 'local_neural_forwards': 0,
        'local_optimizer_updates': 0, 'local_backward_calls': 0,
        'production_promoted': False, 'VM_repeated': False}
    clock()
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    audit(parser.parse_args().receipt.resolve())
