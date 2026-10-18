"""CPU saved-pixel/target loader; no camera regeneration or neural execution."""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from . import contract as c


def pixel_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_rgb(path, size=256):
    with Image.open(path) as image:
        c.require(image.mode == 'RGB' and image.size == (size, size), 'Exact RGB image geometry required')
        return np.asarray(image).copy()


def input_float32(pixels):
    c.require(pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[-1] == 3,
              'Exact RGB8 pixels required')
    return np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255))


def load_mask(path):
    with Image.open(path) as image:
        c.require(image.size == (256, 256), 'Exact observed support geometry required')
        mask = np.asarray(image.convert('L')).copy()
    c.require(np.isin(mask, [0, 255]).all() and mask.any(), 'Nonempty binary observed support required')
    return np.ascontiguousarray(mask > 0)


def clean512(path, expected_rgb8, expected_float32):
    with Image.open(path) as image:
        c.require(image.mode == 'RGB' and image.size == (1024, 1024), 'Approved clean RGB1024 source required')
        pixels = np.asarray(image.resize((512, 512), Image.Resampling.LANCZOS)).copy()
    floats = input_float32(pixels)
    c.require(pixel_sha256(pixels) == expected_rgb8 and pixel_sha256(floats) == expected_float32,
              'Direct clean512 pixels differ from retained target')
    return floats


def load_numeric(path, *, dtype, shape, expected_sha256):
    array = np.load(path, allow_pickle=False)
    c.require(isinstance(array, np.ndarray) and array.dtype == dtype and array.shape == shape and
              array.flags.c_contiguous and np.isfinite(array).all() and
              pixel_sha256(array) == expected_sha256, 'Exact finite retained numeric target required')
    return array


def validate_references(document):
    c.require(document.get('format') == 'codeformer-adaptation-reference-manifest-v1',
              'Exact clean-source reference manifest required')
    rows = document['references']
    c.require(len(rows) == 444 and Counter(r['historical_role'] for r in rows) ==
              {'train': 391, 'validation': 53} and len({r['id'] for r in rows}) == 444,
              'Retained391TRAIN/53DEV metadata split differs')
    by_id = {row['id']: row for row in rows}
    for row in rows:
        c.require(row['decision'] == 'accept_clean_restoration' and row['native_size'] == [1024, 1024]
                  and row['existing_target_size'] == [256, 256], 'Reference approval or geometry differs')
        if row['historical_role'] == 'train':
            c.require(row['suitable_as_hq_train_candidate'] is True, 'Unapproved TRAIN reference')
        for name in ('source_sha256', 'proposed_512_rgb_sha256'):
            c.require(c.HEX.fullmatch(row[name]) is not None, 'Missing exact source/target fingerprint')
        c.safe_relative(row['source_path'])
    for key in ('source_path', 'source_sha256'):
        train = {r[key] for r in rows if r['historical_role'] == 'train'}
        dev = {r[key] for r in rows if r['historical_role'] == 'validation'}
        c.require(not train & dev, 'TRAIN/DEV photographic source overlap')
    return by_id


def load_index(parent_root, protocol, bundle, *, resolver=None):
    """Decode only hash-bound selected TRAIN and DEV pixels; no FINAL images.

    The caller verifies the entire protocol before entering this loader. The
    metadata join verifier is repeated here; it cannot execute a model. Local
    audits provide a resolver over the same exact immutable parent bindings.
    """
    from .transport import verify_data_bundle
    metadata_receipt = verify_data_bundle(None, protocol, parent_root, resolver=resolver, bundle=bundle)
    mappings = {**protocol['dependencies'], **protocol['retained_output_dependencies'],
                **protocol['extra_returned_dependencies']}
    checked = {}

    def bound(name, expected=None):
        c.safe_relative(name)
        c.require(name in mappings and (expected is None or mappings[name]['sha256'] == expected),
                  'Unbound or stale physical data file: ' + name)
        path = c.no_links(Path(resolver(name))) if resolver else c.safe_path(parent_root, name)
        pin = mappings[name]
        before = path.stat()
        c.require(path.is_file() and before.st_nlink == 1 and before.st_size == pin['bytes'],
                  'Exact unlinked regular data file required')
        if name not in checked:
            c.require(c.sha256(path) == pin['sha256'], 'Data bytes changed: ' + name)
            after = path.stat()
            c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                      (after.st_size, after.st_mtime_ns, after.st_ino), 'Data changed while hashing')
            checked[name] = pin['sha256']
        return path

    references = validate_references(c.read_json(bound('references.json')))
    parity = c.read_json(bound('training_parity_v1.json'))
    reference_sha = mappings['references.json']['sha256']
    hq = {}
    for row in bundle['hq']:
        rid = row['reference_id']
        ref = references[rid]
        c.require(ref['historical_role'] == 'train' and ref['source_path'] == row['clean_source_path'] and
                  ref['source_sha256'] == row['source_sha256'] and
                  ref['proposed_512_rgb_sha256'] == row['clean512_rgb8_sha256'], 'HQ source/role join differs')
        clean = clean512(bound(row['clean_source_path'], row['source_sha256']),
                         row['clean512_rgb8_sha256'], row['clean512_float32_sha256'])
        record = c.read_json(bound(row['record_path']))
        c.require(c.equal(record, row['sourcebindings']) and record['role'] == 'train',
                  'Saved HQ source bindings differ')
        indices = load_numeric(bound(row['indices_path'], record['indices_sha256']),
            dtype=np.dtype('int64'), shape=(1, 256), expected_sha256=row['indices_array_sha256'])
        features = load_numeric(bound(row['features_path'], record['features_sha256']),
            dtype=np.dtype('float32'), shape=(1, 256, 16, 16), expected_sha256=row['features_array_sha256'])
        c.require(indices.min() >= 0 and indices.max() < 1024, 'HQ token outside frozen dictionary')
        hq[rid] = {'indices': indices, 'features': features, 'clean512': clean,
                   'sourcebindings': row}
    c.require(len(hq) == 100, 'Exactly100 fresh-selected cached HQ targets required')

    train = []
    for row in bundle['training']:
        record = c.read_json(bound(row['pair_record_path'], row['pair_record_sha256']))
        ref = references[row['reference_id']]
        role = 'normal' if row['slot'] in (1, 3) else 'dark'
        c.require(record['reference_manifest_sha256'] == reference_sha and
                  record['role'] == ref['historical_role'] == 'train' and record['member'] == role == row['role'] and
                  record['source_path'] == row['clean_source_path'] == ref['source_path'] and
                  record['source_sha256'] == row['source_sha256'] == ref['source_sha256'] and
                  record['reference_id'] == row['reference_id'] and
                  record['update'] == row['update'] and record['slot'] == row['slot'] and
                  record['parameters']['image_eligible'] == row['image_eligible'], 'TRAIN source/order/role differs')
        pixels = load_rgb(bound(row['input_path'], row['input_sha256']))
        floats = input_float32(pixels)
        clean = hq[row['reference_id']]['clean512']
        c.require(pixel_sha256(pixels) == row['input256_rgb8_sha256'] == record['input256_rgb8_sha256'] and
                  pixel_sha256(floats) == row['input256_float32_sha256'] == record['input256_float32_sha256'] and
                  pixel_sha256(clean) == row['clean512_float32_sha256'] == record['clean512_float32_sha256'],
                  'Saved TRAIN input/clean-target pixels differ')
        train.append({'id': row['id'], 'update': row['update'], 'slot': row['slot'], 'role': role,
            'reference_id': row['reference_id'], 'input_rgb8': pixels, 'input256': floats,
            'observed': np.ones((256, 256), dtype=bool), 'clean512': clean,
            'target': None, 'source': 'FFHQ TRAIN', 'profile': row['profile'],
            'image_eligible': row['image_eligible'], 'evidence': 'paired_synthetic_TRAIN',
            'record': record, 'parity': row, 'sourcebindings': row['sourcebindings'],
            'input_path': str(bound(row['input_path'])), 'source_path': str(bound(row['clean_source_path']))})
    c.require(len(train) == 200 and len({r['id'] for r in train}) == 200 and
              Counter(r['role'] for r in train) == {'normal': 100, 'dark': 100} and
              sum(r['role'] == 'dark' and r['image_eligible'] for r in train) == 74, 'TRAIN operation census differs')

    dev, retained = [], {'original': {}, 'base': {}}
    for row in bundle['development']:
        pixels = load_rgb(bound(row['input_path']))
        floats = input_float32(pixels)
        support = load_mask(bound(row['observed_path'])) if row['observed_path'] else np.ones((256, 256), dtype=bool)
        target = input_float32(load_rgb(bound(row['target_path']))) if row['target_path'] else None
        if row['evidence'] == 'native_unpaired':
            c.require(target is None, 'Native CCTV has no aligned clean target')
        else:
            c.require(row['evidence'] == 'paired_synthetic' and target is not None, 'Paired target missing')
        if row['source'] == 'FFHQ validation':
            pair = row['sourcebindings']['pair_record']
            ref = references[pair['reference_id']]
            c.require(ref['historical_role'] == pair['role'] == 'validation' and
                      pair['reference_manifest_sha256'] == reference_sha and
                      pair['source_sha256'] == ref['source_sha256'] and
                      pixel_sha256(pixels) == pair['input256_rgb8_sha256'] and
                      pixel_sha256(floats) == pair['input256_float32_sha256'] and
                      row['target_path'] == ref['existing_target_path'], 'Saved DEV source/input/target join differs')
        dev.append({'id': row['id'], 'input_rgb8': pixels, 'input256': floats, 'observed': support,
            'target': target, 'source': row['source'], 'profile': row['profile'], 'evidence': row['evidence'],
            'sourcebindings': row['sourcebindings'], 'input_path': str(bound(row['input_path'])),
            'target_path': str(bound(row['target_path'])) if row['target_path'] else None})
        for state, prefix in [('original', 'original'), ('base', 'parent')]:
            item = {}
            for view in ('raw', 'png'):
                name = row[prefix + '_' + view + '_path']
                item[view + '_path'] = str(bound(name))
                item[view + '_relative'] = name
                item[view + '_sha256'] = mappings[name]['sha256']
            retained[state][row['id']] = item
    c.require(len(dev) == 406 and Counter(r['evidence'] for r in dev) ==
              {'paired_synthetic': 358, 'native_unpaired': 48} and
              sum(r['evidence'] == 'paired_synthetic' and r['profile'] == 'clear' for r in dev) == 61 and
              sum(r['source'] == 'FFHQ validation' and r['profile'] == 'blur' for r in dev) == 53,
              'Exact406 DEV/61clear/53blur census differs')
    return {'train': train, 'train_updates': parity['updates'], 'dev': dev, 'hq': hq,
        'retained_outputs': retained, 'binding_receipt': {'complete': True,
            'metadata': metadata_receipt, 'checked_files_sha256': checked,
            'TRAIN_observations': 200, 'HQ_selected_references': 100, 'DEV_cases': 406,
            'normal_observations': 100, 'dark_observations': 100, 'eligible_dark_LPIPS': 74,
            'input_policy': 'retained RGB8/float32(255); clean direct RGB1024-to-LANCZOS512 parity',
            'unused_TRAIN_source_pixels_loaded': 0, 'FINAL_used': False,
            'source_identifiers_are_not_proven_person_ids': True, 'neural_calls': 0}}
