"""Exact retained TRAIN/DEV index and shared input-only NumPy features; no Torch."""
from collections import Counter
from io import BytesIO
from pathlib import Path
import re

import numpy as np
from PIL import Image

from . import contract as c

PROFILES = ('clear', 'light', 'blur', 'jpeg', 'lowlight', 'compound')
PARITY_FORMAT = 'codeformer-v2-first50-training-parity-v1'


def pixel_sha256(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_rgb(path):
    with Image.open(path) as image:
        c.require(image.mode == 'RGB' and image.size == (256, 256), 'Exact RGB256 input/target required')
        return np.asarray(image).copy()


def load_mask(path):
    with Image.open(path) as image:
        c.require(image.size == (256, 256), 'Exact observed-mask geometry required')
        pixels = np.asarray(image.convert('L')).copy()
    c.require(np.isin(pixels, [0, 255]).all() and pixels.any(), 'Observed mask must be binary/nonempty')
    return np.ascontiguousarray(pixels > 0)


def metric_target256(source_path, expected_sha256):
    """Direct TRAIN clean target for the two declared stress readouts only; never a head label."""
    payload = Path(source_path).read_bytes()
    import hashlib
    c.require(hashlib.sha256(payload).hexdigest() == expected_sha256, 'Stress clean source changed')
    with Image.open(BytesIO(payload)) as image:
        c.require(image.mode == 'RGB' and image.size == (1024, 1024), 'Stress source must be approved RGB1024')
        pixels = np.asarray(image.resize((256, 256), Image.Resampling.LANCZOS)).copy()
    return input_float32(pixels)


def input_float32(pixels):
    c.require(isinstance(pixels, np.ndarray) and pixels.shape == (256, 256, 3) and
              pixels.dtype == np.uint8, 'Retained RGB8 input required')
    return np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255))


def features(input256, observed):
    """Plane-major support-normalized32px cells; one final float32 cast."""
    c.require(isinstance(input256, np.ndarray) and input256.shape == (256, 256, 3) and
              input256.dtype == np.float32 and np.isfinite(input256).all() and
              input256.min() >= 0 and input256.max() <= 1, 'Features require finite RGB float32[0,1]')
    c.require(isinstance(observed, np.ndarray) and observed.shape == (256, 256) and
              observed.dtype == np.bool_ and observed.any(), 'Features require nonempty boolean support')
    support = observed.astype(np.float64)
    count = support.reshape(8, 32, 8, 32).sum(axis=(1, 3), dtype=np.float64)
    weighted = input256.astype(np.float64) * support[:, :, None]
    total = weighted.reshape(8, 32, 8, 32, 3).sum(axis=(1, 3), dtype=np.float64)
    means = np.zeros_like(total)
    np.divide(total, count[:, :, None], out=means, where=count[:, :, None] > 0)
    planes = np.concatenate((means.transpose(2, 0, 1), (count / 1024)[None]), axis=0)
    result = np.ascontiguousarray(planes.reshape(256).astype(np.float32))
    c.require(result.shape == (256,) and np.isfinite(result).all(), 'Invalid input-only features')
    return result


def _validate_references(rows):
    c.require(isinstance(rows, list) and len(rows) == 444 and
              Counter(row.get('historical_role') for row in rows) == {'train': 391, 'validation': 53},
              'Exact391TRAIN/53validation split required')
    by_id = {}
    for row in rows:
        role = row['historical_role']
        prefix = 'tr' if role == 'train' else 'va'
        c.require(re.fullmatch(prefix + r'_ffhq_[0-9]{5}', row.get('id', '')) and
                  row['id'] not in by_id and row.get('decision') == 'accept_clean_restoration' and
                  row.get('native_size') == [1024, 1024] and row.get('existing_target_size') == [256, 256],
                  'Reference ID/approval/source geometry differs')
        c.safe_relative(row['source_path'])
        c.require(c.HEX.fullmatch(row.get('source_sha256', '')) and
                  c.HEX.fullmatch(row.get('proposed_512_rgb_sha256', '')),
                  'Missing source or direct512 identity')
        if role == 'train':
            c.require(row.get('suitable_as_hq_train_candidate') is True, 'Unapproved TRAIN source')
        by_id[row['id']] = row
    for field in ('source_path', 'source_sha256'):
        train = {r[field] for r in rows if r['historical_role'] == 'train'}
        dev = {r[field] for r in rows if r['historical_role'] == 'validation'}
        c.require(not train & dev, 'TRAIN/DEV source overlap: ' + field)
    return by_id


def _validate_train_record(row, record, ref, reference_sha):
    update, slot = row['update'], row['slot']
    role = 'normal' if slot in (1, 3) else 'dark'
    pair_role = 'A' if slot <= 2 else 'B'
    c.require(record.get('format') == 'codeformer-adaptation-matched-pair-v2' and
              record.get('role') == ref['historical_role'] == 'train' and
              record.get('source_path') == row['clean_source_path'] == ref['source_path'] and
              record.get('source_sha256') == row['source_sha256'] == ref['source_sha256'] and
              record.get('reference_manifest_sha256') == reference_sha and
              record.get('reference_id') == row['reference_id'] == ref['id'] and
              record.get('update') == update and record.get('slot') == slot and
              record.get('member') == role and record.get('pair_role') == pair_role and
              record.get('profile') == row['profile'] and row['profile'].endswith('_' + role),
              'TRAIN record/source/order/normal-dark binding differs')
    for key in ('input256_rgb8_sha256', 'input256_float32_sha256',
                'clean512_rgb8_sha256', 'clean512_float32_sha256'):
        c.require(c.HEX.fullmatch(row.get(key, '')) and record.get(key) == row[key],
                  'TRAIN retained pixel fingerprint differs: ' + key)
    c.require(record['clean512_rgb8_sha256'] == ref['proposed_512_rgb_sha256'],
              'TRAIN direct512 provenance differs')
    parameters = record.get('parameters', {})
    c.require(parameters.get('update') == update and parameters.get('slot') == slot and
              parameters.get('member') == role and parameters.get('pair_role') == pair_role and
              type(row.get('image_eligible')) is bool and parameters.get('image_eligible') == row['image_eligible'],
              'TRAIN retained parameter joins differ')
    return role


def validate_training_data(document, references, record_reader, reference_sha):
    """Data-only chronology/roles; historical optimizer/LR/call caps are not consulted."""
    c.require(document.get('format') == PARITY_FORMAT and document.get('source_protocol_sha256') ==
              c.PARENT_CONTROLS['training_parity']['source_protocol_sha256'] and document.get('TRAIN_only') is True,
              'TRAIN parity provenance differs')
    rows = document.get('cases')
    c.require(isinstance(rows, list) and len(rows) == 200 and
              all(type(r.get('update')) is int and type(r.get('slot')) is int for r in rows) and
              [(r.get('update'), r.get('slot')) for r in rows] ==
              [(u, s) for u in range(1, 51) for s in range(1, 5)], 'Exact chronological200 retained inputs required')
    c.require(len({r['reference_id'] for r in rows}) == 100, 'Exact100 photographic TRAIN IDs required')
    for row in rows:
        ref = references.get(row.get('reference_id'))
        c.require(ref is not None and ref['historical_role'] == 'train', 'DEV/native reference entered TRAIN')
        _validate_train_record(row, record_reader(row), ref, reference_sha)
    updates = document.get('updates')
    c.require(isinstance(updates, list) and len(updates) == 50 and
              [r.get('update') for r in updates] == list(range(1, 51)), 'Exact50 retained selections required')
    for item in updates:
        rows4 = rows[(item['update'] - 1) * 4:item['update'] * 4]
        a, b = rows4[0]['reference_id'], rows4[2]['reference_id']
        selection = item.get('TRAIN_selection', {})
        c.require(a != b and [r['reference_id'] for r in rows4] == [a, a, b, b] and
                  selection.get('update') == item['update'] and selection.get('id_a') == a and
                  selection.get('id_b') == b, 'Retained paired A/B selection differs')
    return rows, updates


def load_index(parent_root, protocol, *, resolver=None):
    """Hash/decode exact saved pixels before NN; resolver supports local split overlays."""
    parent_root = Path(parent_root)
    dependencies = protocol['dependencies']
    checked = {}
    def bound(name, expected=None):
        c.safe_relative(name)
        c.require(name in dependencies and (expected is None or dependencies[name]['sha256'] == expected),
                  'Unbound or stale data dependency: ' + name)
        path = c.safe_path(parent_root, name) if resolver is None else c.no_links(Path(resolver(name)))
        info = path.stat()
        c.require(path.is_file() and info.st_nlink == 1 and info.st_size == dependencies[name]['bytes'],
                  'Data file size/link differs: ' + name)
        if name not in checked:
            c.require(c.sha256(path) == dependencies[name]['sha256'], 'Data file hash differs: ' + name)
            checked[name] = dependencies[name]['sha256']
        return path
    reference_sha = dependencies[protocol['data_manifest']]['sha256']
    manifest = c.read_json(bound(protocol['data_manifest']))
    c.require(manifest.get('format') == 'codeformer-adaptation-reference-manifest-v1', 'Reference format differs')
    references = _validate_references(manifest['references'])
    parity = c.read_json(bound(protocol['training_parity']['path'], protocol['training_parity']['sha256']))
    records = {}
    def train_record(row):
        key = (row['update'], row['slot'])
        records[key] = c.read_json(bound(row['pair_record_path'], row['pair_record_sha256']))
        return records[key]
    parity_rows, updates = validate_training_data(parity, references, train_record, reference_sha)
    train = []
    for row in parity_rows:
        pixels = load_rgb(bound(row['input_path'], row['input_sha256']))
        floats = input_float32(pixels)
        c.require(pixel_sha256(pixels) == row['input256_rgb8_sha256'] and
                  pixel_sha256(floats) == row['input256_float32_sha256'], 'Retained TRAIN RGB8/255 pixels differ')
        source_path = bound(row['clean_source_path'], row['source_sha256'])
        record = records[(row['update'], row['slot'])]
        train.append({'id': 'train_u%04d_slot%d' % (row['update'], row['slot']),
            'update': row['update'], 'slot': row['slot'], 'role': record['member'],
            'reference_id': row['reference_id'], 'input_rgb8': pixels, 'input256': floats,
            'observed': np.ones((256, 256), dtype=bool), 'target': None, 'source': 'FFHQ TRAIN',
            'profile': row['profile'], 'evidence': 'paired_synthetic_TRAIN', 'record': record, 'parity': row,
            'input_path': str(bound(row['input_path'])), 'source_path': str(source_path),
            'sourcebindings': {key: row[key] for key in ('input_path', 'input_sha256', 'pair_record_path',
                'pair_record_sha256', 'clean_source_path', 'source_sha256', 'input256_rgb8_sha256',
                'input256_float32_sha256', 'clean512_rgb8_sha256', 'clean512_float32_sha256')}})
    saved = c.read_json(bound(protocol['validation_saved']['manifest'], protocol['validation_saved']['sha256']))
    c.require(saved.get('case_count') == len(saved.get('items', [])) == 318 and
              saved.get('locally_regenerated_development_pixels') is False, 'Exact318 saved DEV pixels required')
    original = c.read_json(bound(saved['retained_original_manifest_path'], saved['retained_original_manifest_sha256']))
    valid_refs = {key: ref for key, ref in references.items() if ref['historical_role'] == 'validation'}
    expected_ids = {'validation_' + key + '_' + profile for key in valid_refs for profile in PROFILES}
    c.require(len({r['id'] for r in saved['items']}) == 318 and
              {r['id'] for r in saved['items']} == expected_ids, 'Saved validation identity/profile census differs')
    dev = []
    for row in sorted(saved['items'], key=lambda value: value['id']):
        record = c.read_json(bound(row['pair_record_path'], row['pair_record_sha256']))
        ref = valid_refs.get(record.get('reference_id'))
        c.require(ref is not None and record.get('role') == 'validation' and
                  record.get('source_path') == ref['source_path'] and
                  record.get('source_sha256') == ref['source_sha256'] and
                  record.get('reference_manifest_sha256') == reference_sha and
                  record.get('clean512_rgb8_sha256') == ref['proposed_512_rgb_sha256'] and
                  row['id'] == 'validation_' + ref['id'] + '_' + record['profile'] and
                  row['target_path'] == ref['existing_target_path'] and
                  row['target_sha256'] == ref['existing_target_sha256'] and
                  original['cases'][row['id']]['sha256'] == row['input_sha256'],
                  'Saved validation record/input/target/source joins differ')
        bound(ref['source_path'], ref['source_sha256'])
        pixels = load_rgb(bound(row['input_path'], row['input_sha256']))
        floats = input_float32(pixels)
        c.require(pixel_sha256(pixels) == record['input256_rgb8_sha256'] and
                  pixel_sha256(floats) == record['input256_float32_sha256'], 'Saved DEV RGB8/255 pixels differ')
        target_path = bound(row['target_path'], row['target_sha256'])
        dev.append({'id': row['id'], 'update': None, 'slot': None, 'role': 'validation',
            'reference_id': ref['id'], 'input_rgb8': pixels, 'input256': floats,
            'observed': np.ones((256, 256), dtype=bool), 'target': input_float32(load_rgb(target_path)),
            'source': 'FFHQ validation', 'profile': record['profile'], 'evidence': 'paired_synthetic',
            'sourcebindings': dict(row), 'input_path': str(bound(row['input_path'])),
            'target_path': str(target_path), 'record': record})
    comparisons = c.read_json(bound(protocol['comparison_cases']))['cases']
    c.require(len(comparisons) == 88 and len({r['id'] for r in comparisons}) == 88 and
              Counter(r['evidence'] for r in comparisons) == {'paired_synthetic': 40, 'native_unpaired': 48},
              'Exact88 comparison namespace required')
    train_ids = {key for key, ref in references.items() if ref['historical_role'] == 'train'}
    for row in sorted(comparisons, key=lambda value: value['id']):
        c.require(not row['id'].startswith(('tr_', 'final')) and row.get('identity') not in train_ids,
                  'TRAIN/FINAL comparison identity overlap')
        pixels = load_rgb(bound(row['prepared_path'], row['prepared_sha256']))
        mask = load_mask(bound(row['observed_path'], row['observed_sha256']))
        target_path = None
        if row['evidence'] == 'native_unpaired':
            c.require(not row.get('target_path') and not row.get('target_sha256'), 'Native clean target forbidden')
        else:
            target_path = bound(row['target_path'], row['target_sha256'])
        dev.append({'id': 'comparison_' + row['id'], 'update': None, 'slot': None, 'role': 'development',
            'input_rgb8': pixels, 'input256': input_float32(pixels), 'observed': mask,
            'target': input_float32(load_rgb(target_path)) if target_path else None,
            'source': row['source'], 'profile': row['profile'], 'evidence': row['evidence'],
            'sourcebindings': dict(row), 'input_path': str(bound(row['prepared_path'])),
            'target_path': str(target_path) if target_path else None})
    c.require(len(dev) == len({r['id'] for r in dev}) == 406 and
              sum(r['evidence'] == 'paired_synthetic' and r['profile'] == 'clear' for r in dev) == 61 and
              sum(r['source'] == 'FFHQ validation' and r['profile'] == 'blur' for r in dev) == 53,
              'Full406/clear61/normalblur53 census differs')
    retained = protocol['retained_output_dependencies']
    retained_outputs = {}
    for state, update in [('original', '0000'), ('base', '0050')]:
        retained_outputs[state] = {}
        for row in dev:
            item = {}
            for view, filename in [('raw', 'raw.npy'), ('png', 'native.png')]:
                name = 'outputs/adain_gradient_isolation_v1/snapshot_' + update + '/' + row['id'] + '/' + filename
                c.require(name in retained, 'Missing retained full406 output binding')
                path = c.safe_path(parent_root, name) if resolver is None else c.no_links(Path(resolver(name)))
                c.require(path.is_file() and path.stat().st_size == retained[name]['bytes'],
                          'Retained historical output size differs')
                item[view + '_path'] = str(path)
                item[view + '_relative'] = name
                item[view + '_sha256'] = retained[name]['sha256']
            retained_outputs[state][row['id']] = item
    c.require(sum(len(v) * 2 for v in retained_outputs.values()) == len(retained) == 1624,
              'Retained output case census differs')
    return {'train': train, 'train_updates': updates, 'dev': dev, 'binding_receipt': {
        'complete': True, 'TRAIN_cases': 200, 'TRAIN_reference_IDs': 100, 'TRAIN_normal': 100, 'TRAIN_dark': 100,
        'DEV_cases': 406, 'DEV_paired': 358, 'DEV_native_unpaired': 48, 'clear': 61, 'normal_blur': 53,
        'checked_files_sha256': checked, 'input_policy': 'retained RGB8 / float32(255); no camera regeneration',
        'historical_optimizer_caps_validated_as_new_recipe': False, 'neural_calls': 0},
        'retained_outputs': retained_outputs}
