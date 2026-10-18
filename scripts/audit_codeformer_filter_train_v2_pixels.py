"""Independent saved-pixel/source/namespace audit of the TRAIN64 filter set.

No producer, SciPy, neural runtime, checkpoint or browser is imported. The old
independent mathematical helper is byte-pinned; local rectangles are explicit.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / 'scripts/audit_codeformer_filter_pairs_v1_independent.py'
HELPER_SHA = 'c145831ecaaaba7a917f599198b74c5d300a6440861e8bee4253691d526fa58f'
SELECTION_SHA = 'ffda2fc99e7320af2f9428d41d1c6c1c0d62fff84c5c630a62a1e891fccb9228'
PROFILES_SHA = 'e1e35416742ba28dec7ba555a083836ceede08de9fed2f5097233ebb133d18c9'
CRITERIA_SHA = '9b66c4e0ab4d460a9790b225e470a121cef379f254f4e6cf52d94d58c99045d6'
RECTANGLE_RULE = "SHA256 UTF8('codeformer-filter-train-v2/rectangle/'+source_sha256); x0=byte0%9*16,y0=byte1%9*16; half-open[x0,y0,x0+128,y0+128]"
PROFILE_TUPLES = (
    ('clear', 'identity', {}, False),
    ('gaussian_2p4', 'gaussian', {'sigma_x': 2.4, 'sigma_y': 2.4}, False),
    ('anisotropic_h5_v0p7', 'gaussian', {'sigma_x': 5., 'sigma_y': .7}, False),
    ('anisotropic_h0p7_v5', 'gaussian', {'sigma_x': .7, 'sigma_y': 5.}, False),
    ('repeated_rows_4_blend0p75', 'repeated_rows', {'block_rows': 4, 'repeat_weight': .75}, True),
    ('global_mosaic_8', 'mosaic', {'cell': 8}, False),
    ('local_gaussian_3p0', 'local_gaussian', {'sigma_x': 3., 'sigma_y': 3.}, False),
    ('local_mosaic_8', 'local_mosaic', {'cell': 8}, False),
)
PROFILES = tuple(dict(id=n, operator=o, parameters=p, strong_control=s)
                 for n, o, p, s in PROFILE_TUPLES)
REVIEW_ORDER = ('global_mosaic_8', 'local_mosaic_8', 'local_gaussian_3p0',
    'repeated_rows_4_blend0p75', 'gaussian_2p4', 'anisotropic_h5_v0p7',
    'anisotropic_h0p7_v5', 'clear')
FORBIDDEN_MODULES = ('torch', 'scipy', 'tensorflow', 'jax',
    'research.codeformer_filter_pairs_v1.filters', 'prepare_codeformer_filter_train_v2')


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


check(hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() == HELPER_SHA,
      'Frozen independent mathematical helper differs')
_spec = importlib.util.spec_from_file_location('_independent_filter_v1_math', HELPER_PATH)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def no_producer_or_neural():
    check(not any(name in sys.modules for name in FORBIDDEN_MODULES),
          'Producer/SciPy/neural import is forbidden in independent pixel audit')


def strict_json(data):
    value = H.strict_json(data)
    def finite(item):
        if isinstance(item, float):
            check(math.isfinite(item), 'JSON float overflow/nonfinite forbidden')
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def rectangle_for_source(source_sha):
    check(isinstance(source_sha, str) and re.fullmatch('[0-9a-f]{64}', source_sha),
          'Canonical source SHA256 required')
    hashed = hashlib.sha256(('codeformer-filter-train-v2/rectangle/' + source_sha).encode('utf-8')).digest()
    # The prefix and modulo/grid rule are independently transcribed from approval.
    x0, y0 = 16 * (hashed[0] % 9), 16 * (hashed[1] % 9)
    return [x0, y0, x0 + 128, y0 + 128]


def validate_rectangle(rectangle):
    check(isinstance(rectangle, (list, tuple)) and len(rectangle) == 4 and
          all(type(v) is int for v in rectangle), 'Four integer rectangle coordinates required')
    x0, y0, x1, y1 = rectangle
    check(0 <= x0 <= 128 and 0 <= y0 <= 128 and x0 % 16 == y0 % 16 == 0 and
          x1 == x0 + 128 and y1 == y0 + 128, 'Declared128 square on16-grid required')
    return x0, y0, x1, y1


def reconstruct(clear, profile, rectangle):
    """Own dispatch and moving footprint; do not mutate old helper's RECT."""
    check(clear.dtype == np.uint8 and clear.shape == (256, 256, 3), 'Exact RGB256 required')
    check(profile in PROFILES, 'Literal approved profile required')
    x0, y0, x1, y1 = validate_rectangle(rectangle)
    result, footprint = clear.copy(), np.zeros((256, 256), np.uint8)
    operation, p = profile['operator'], profile['parameters']
    if operation == 'identity':
        return result, footprint
    if operation in ('gaussian', 'local_gaussian'):
        filtered = H.gaussian(clear, p['sigma_x'], p['sigma_y'])
        if operation == 'gaussian':
            result = filtered
        else:
            result[y0:y1, x0:x1] = filtered[y0:y1, x0:x1]
    elif operation == 'repeated_rows':
        for row in range(256):
            first = row // p['block_rows'] * p['block_rows']
            result[row] = H.half_up((1 - p['repeat_weight']) * clear[row].astype(np.float64) +
                                   p['repeat_weight'] * clear[first].astype(np.float64))
    elif operation == 'mosaic':
        result = H.mosaic(clear, p['cell'])
    elif operation == 'local_mosaic':
        result[y0:y1, x0:x1] = H.mosaic(clear[y0:y1, x0:x1], p['cell'])
    else:
        raise ValueError('Unknown independent operator')
    if operation.startswith('local_'):
        footprint[y0:y1, x0:x1] = 255
    else:
        footprint[:] = 255
    return np.ascontiguousarray(result), footprint


def selected_ids(catalog):
    """Independent integer largest-remainder allocation and salted rank."""
    train = [r for r in catalog['references'] if r['historical_role'] == 'train']
    check(len(train) == 391, 'Full391 TRAIN catalog required')
    groups = defaultdict(list)
    for r in train:
        key = r['attribution']['license_url'], r['official_category']
        check(all(isinstance(v, str) and v for v in key), 'Explicit source/license stratum required')
        groups[key].append(r)
    check(len(groups) == 9, 'Nine prospective source/license strata required')
    # One seat/stratum, then55 seats apportioned over382 remaining photos.
    quotas = {key: divmod(55 * (len(rows) - 1), 382) for key, rows in groups.items()}
    allocation = {key: 1 + division[0] for key, division in quotas.items()}
    seats = 64 - sum(allocation.values())
    for key in sorted(groups, key=lambda k: (-quotas[k][1], k))[:seats]:
        allocation[key] += 1
    ids = []
    for key in sorted(groups):
        ranks = sorted(groups[key], key=lambda r: (hashlib.sha256(
            ('codeformer-filter-fusion-train-admission-v1|' + r['id'] + '|' +
             r['source_sha256']).encode('utf-8')).hexdigest(), r['id']))
        ids.extend(r['id'] for r in ranks[:allocation[key]])
    check(len(set(ids)) == len(ids) == 64, 'Exact64 independent selected IDs required')
    return sorted(ids), allocation


def pin_bytes(pin, cap=8 * 1024**2):
    H.relative_name(pin['path'])
    if 'absolute_path' in pin:
        check(Path(pin['absolute_path']).absolute() == (ROOT / pin['path']).absolute(),
              'Logical/physical evidence binding differs')
    return H.bound(ROOT / pin['path'], pin['sha256'], pin['bytes'], cap)


def metadata_pixels(pixels, metadata):
    computed = H.pixel_pin(pixels)
    check(computed['RGB8_sha256'] == metadata['rgb8']['pixel_sha256'],
          'Catalog RGB8 pixels differ')
    if 'float32' in metadata:
        check(computed['float32_sha256'] == metadata['float32']['pixel_sha256'],
              'Catalog float32 pixels differ')


def exact_namespace(folder, files):
    expected_files = set(files) | {'manifest.json'}
    expected_dirs = {parent.as_posix() for name in expected_files
                     for parent in PurePosixPath(name).parents if parent.as_posix() != '.'}
    actual_files, actual_dirs = set(), set()
    for path in folder.rglob('*'):
        if path.is_dir():
            check(not path.is_symlink() and not getattr(path.lstat(), 'st_file_attributes', 0) & 1024,
                  'Linked/reparse output directory')
            actual_dirs.add(path.relative_to(folder).as_posix())
        else:
            H.safe(path)
            actual_files.add(path.relative_to(folder).as_posix())
    check(actual_files == expected_files and actual_dirs == expected_dirs,
          'Closed file/directory namespace differs')


def audit(folder, manifest_sha, out):
    no_producer_or_neural()
    started = time.monotonic()
    folder = Path(folder).resolve(strict=True)
    check(folder.is_dir() and folder.is_relative_to(ROOT / 'outputs'), 'Existing outputs directory required')
    out = Path(out).absolute()
    check(not out.exists() and out.parent.resolve(strict=True).is_relative_to(ROOT / 'outputs'),
          'New exclusive audit directory required')
    raw = H.bound(folder / 'manifest.json', manifest_sha)
    doc = strict_json(raw)
    check(doc['format'] == 'codeformer-filter-train-v2-data' and doc['complete'] is True,
          'Completed TRAIN-v2 data manifest required')
    check(doc['pairs'] == doc['TRAIN_pairs'] == 512 and doc['DEV_pairs'] == 0 and
          doc['reference_photographs'] == 64 and doc['profiles'] == list(PROFILES) and
          doc['historical_pool_counts'] == {'TRAIN': 391, 'DEV': 53}, 'Exact512/64/8 TRAIN census')
    check(doc['selection_sha256'] == SELECTION_SHA and doc['approved_profiles_sha256'] == PROFILES_SHA,
          'Root-frozen selection/profile hashes differ')
    files = doc['files']
    check(len(files) == 1255 and sum(pin['bytes'] for pin in files.values()) < 512 * 1024**2,
          'Expected finite1255 payload files below512MiB')
    cache = {}
    for name, pin in files.items():
        H.relative_name(name)
        check(set(pin) == {'path', 'sha256', 'bytes'} and pin['path'] == name,
              'Exact canonical output pin schema required')
        cache[name] = H.bound(folder / name, pin['sha256'], pin['bytes'])
    exact_namespace(folder, files)

    def get(pin):
        check(pin == files[pin['path']], 'Record pin differs from closed manifest')
        return cache[pin['path']]

    check(H.sha(cache['cohort_selection.json']) == SELECTION_SHA and
          H.sha(cache['approved_profiles.json']) == PROFILES_SHA, 'Saved prospective documents differ')
    selection = strict_json(cache['cohort_selection.json'])
    config = strict_json(cache['approved_profiles.json'])
    criteria = strict_json(cache['input_only_review_criteria.json'])
    check(H.sha(cache['input_only_review_criteria.json']) == CRITERIA_SHA ==
          doc['input_only_criteria']['sha256'] and
          pin_bytes(doc['input_only_criteria']) == cache['input_only_review_criteria.json'] and
          criteria['frozen_before_new_inputs_viewed'] is True and
          set(criteria['decisions']) == {'readable', 'borderline', 'insufficient', 'out_of_scope'} and
          criteria['optimization_permitted_by_data_generation'] is False and
          criteria['existing_quality_gate_or_scope_changed'] is False,
          'Root-frozen current input-only criteria copy/hash differs')
    check(config['profiles'] == list(PROFILES) and config['profile_order'] == list(REVIEW_ORDER) and
          config['local_rectangle_rule'] == RECTANGLE_RULE and config['random_number_generator_calls'] == 0 and
          config['initially_excluded_strong_profiles'] == ['repeated_rows_4_blend0p75'] and
          config['corruption_footprint_is_not_completion_or_removal_mask'] is True,
          'Literal moving-footprint/profile/no-admission contract differs')
    catalog = strict_json(pin_bytes(selection['catalog']))
    refs = catalog['references']
    check(len(refs) == 444 and Counter(r['historical_role'] for r in refs) ==
          {'train': 391, 'validation': 53}, 'Full historical TRAIN/DEV catalog differs')
    roles = {role: {r['source_sha256'] for r in refs if r['historical_role'] == role}
             for role in ('train', 'validation')}
    check(len(roles['train']) == 391 and len(roles['validation']) == 53 and
          not roles['train'] & roles['validation'], 'Exact source-byte split overlaps')
    ids, allocations = selected_ids(catalog)
    check([r['reference_id'] for r in selection['cohort']] == ids, 'Independent64 selection/order differs')
    check(selection['catalog_counts'] == {'TRAIN': 391, 'DEV': 53, 'total': 444} and
          selection['selected_counts'] == {'TRAIN': 64, 'DEV': 0, 'FINAL': 0, 'native_CCTV': 0, 'wild': 0},
          'Prospective cohort pool/role census differs')
    expected_strata = [{'license_url': k[0], 'official_category': k[1], 'count': allocations[k]}
                       for k in sorted(allocations)]
    check(doc['selected_strata'] == expected_strata, 'Recorded independent allocation differs')
    for pin in selection['supporting_audit_and_approval_manifests']:
        pin_bytes(pin)
    source_audit_raw = pin_bytes(selection['source_audit'])
    source_audit = strict_json(source_audit_raw)
    check(source_audit['complete'] is True, 'Completed separate full-source audit required')
    for name, pin in doc['source_pins'].items():
        H.relative_name(name); H.bound(ROOT / name, pin['sha256'], pin['bytes'])
    check(set(doc['source_pins']) == {'scripts/prepare_codeformer_filter_train_v2.py',
          'research/codeformer_filter_pairs_v1/filters.py', 'research/codeformer_filter_train_v2/README.md'},
          'Exact producer-source namespace differs')
    check(np.all(H.decode(cache['observed-support.png'], (256, 256), 'L') == 255) and
          not H.decode(cache['corruption_footprints/clear.png'], (256, 256), 'L').any(),
          'Observed support/clear corruption mask differs')
    case_map, proof_map = {r['id']: r for r in doc['cases']}, {r['reference_id']: r for r in doc['references']}
    check(len(case_map) == 512 and len(proof_map) == 64, 'Unique case/reference census required')
    check(Counter(r['profile'] for r in doc['cases']) == {p['id']: 64 for p in PROFILES},
          'Exactly64 inputs for every profile required')
    catalog_map = {r['id']: r for r in refs}
    expected_names = {'cohort_selection.json', 'approved_profiles.json', 'input_only_review_criteria.json', 'README.md',
        'observed-support.png', 'corruption_footprints/clear.png', 'input_only_visual_manifest.json'}
    audits, input_digests, rectangles = [], {}, set()
    for selected in selection['cohort']:
        rid, row = selected['reference_id'], catalog_map[selected['reference_id']]
        check(re.fullmatch('tr_ffhq_[0-9]{5}', rid) and selected['split'] == 'TRAIN' and
              selected['historical_role'] == row['historical_role'] == 'train' and
              row['decision'] == 'accept_clean_restoration' and selected['reference_manifest_row'] == row and
              selected['attribution'] == row['attribution'], 'Exact historical role/catalog/attribution join')
        sp, tp = selected['source1024'], selected['target256']
        check(sp['path'] == row['original_source_path'] and sp['sha256'] == row['source_sha256'] and
              tp['path'] == row['original_existing_target_path'] and tp['sha256'] == row['existing_target_sha256'],
              'Original source/target binding differs')
        source_bytes, retained_target = pin_bytes(sp), pin_bytes(tp)
        source = H.decode(source_bytes, (1024, 1024), 'RGB')
        metadata_pixels(source, sp)
        with Image.open(io.BytesIO(source_bytes)) as image:
            clear = np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR)).copy()
            clean256 = np.asarray(image.resize((256, 256), Image.Resampling.LANCZOS)).copy()
            clean512 = np.asarray(image.resize((512, 512), Image.Resampling.LANCZOS)).copy()
        check(np.array_equal(clean256, H.decode(retained_target, (256, 256), 'RGB')),
              'Direct1024 clean256 target pixels differ')
        for pixels, meta in ((clean256, tp), (clean512, selected['clean512']), (clear, selected['clear256'])):
            metadata_pixels(pixels, meta)
        check(H.sha(clean512.tobytes()) == row['proposed_512_rgb_sha256'], 'Historical512 fingerprint differs')
        rectangle = rectangle_for_source(sp['sha256'])
        check(selected['local_rectangle_xyxy_half_open'] == rectangle, 'Source-hash rectangle differs')
        rectangles.add(tuple(rectangle))
        proof = proof_map[rid]
        targets = proof['targets']
        expected_names.update({'references/' + rid + '/clean512.png', 'references/' + rid + '/clean256.png',
                               'corruption_footprints/' + rid + '.png'})
        check(proof['split'] == 'TRAIN' and proof['historical_role'] == 'train' and proof['source1024'] == sp and
              proof['retained_target256_bytes_exact'] is True and proof['attribution'] == selected['attribution'] and
              proof['local_rectangle_xyxy_half_open'] == rectangle and
              proof['target256_pixels'] == H.pixel_pin(clean256) and
              proof['clean512_pixels'] == H.pixel_pin(clean512) and proof['clear256_pixels'] == H.pixel_pin(clear),
              'Saved reference proof differs')
        check(get(targets['clean256']) == retained_target and
              np.array_equal(H.decode(get(targets['clean512']), (512, 512), 'RGB'), clean512),
              'Saved exact clean256 bytes/direct512 pixels differ')
        for profile in PROFILES:
            case_id = rid + '__' + profile['id']
            expected_names.update({'inputs/' + case_id + '.png', 'pairs/' + case_id + '.json'})
            ci = case_map[case_id]
            rec = strict_json(get(ci['pair_record']))
            check(ci['reference_id'] == rid and ci['split'] == 'TRAIN' and ci['profile'] == profile['id'] and
                  ci['input_review'] == 'PENDING' and rec['id'] == case_id and rec['reference_id'] == rid and
                  rec['split'] == 'TRAIN' and rec['historical_role'] == 'train' and rec['profile'] == profile and
                  rec['format'] == 'codeformer-filter-train-v2-pair' and
                  rec['evidence'] == 'paired_synthetic_filter_TRAIN_admission' and rec['source1024'] == sp and
                  rec['attribution'] == selected['attribution'], 'Case profile/source/role binding differs')
            expected, footprint = reconstruct(clear, profile, rectangle)
            pixels = H.decode(get(rec['input']), (256, 256), 'RGB')
            mismatch = int(np.count_nonzero(pixels != expected))
            check(mismatch == 0, 'Independent exact input pixels differ: ' + case_id)
            check(ci['input'] == rec['input'] and rec['input_pixels'] == H.pixel_pin(pixels) and
                  rec['targets'] == targets and rec['target256_pixels'] == H.pixel_pin(clean256) and
                  rec['target512_pixels'] == H.pixel_pin(clean512), 'Input/target fingerprints differ')
            local = profile['operator'].startswith('local_')
            expected_foot_name = ('corruption_footprints/' + rid + '.png') if local else (
                'corruption_footprints/clear.png' if profile['id'] == 'clear' else 'observed-support.png')
            check(rec['observed_support'] == files['observed-support.png'] and
                  rec['corruption_footprint'] == files[expected_foot_name] and
                  np.array_equal(H.decode(get(rec['corruption_footprint']), (256, 256), 'L'), footprint) and
                  rec['corruption_footprint_pixels_sha256'] == H.sha(footprint.tobytes()) and
                  rec['local_rectangle_xyxy_half_open'] == (rectangle if local else None) and
                  rec['local_rectangle_rule'] == (RECTANGLE_RULE if local else None) and
                  rec['corruption_footprint_is_not_completion_or_removal_mask'] is True,
                  'Corruption footprint pixels/rule differs')
            check(np.array_equal(pixels[footprint == 0], clear[footprint == 0]), 'Outside footprint pixels changed')
            expected_admission = {'optimization_allowed_by_historical_split': True, 'optimization_permitted_now': False,
                'restoration_training_eligibility': 'PENDING_INPUT_ONLY_REVIEW_AND_EXPLICIT_PILOT_PROTOCOL',
                'strong_control_excluded_initially': profile['id'] == 'repeated_rows_4_blend0p75'}
            check(all(ci[k] == v and rec[k] == v for k, v in expected_admission.items()) and
                  rec['input_only_review'] == {'status': 'PENDING', 'decision': None, 'reviewer': None,
                      'input_only': True, 'model_output_used': False, 'criteria_sha256': CRITERIA_SHA} and
                  rec['automatic_usable_or_insufficient_label'] is None and rec['RNG_calls'] == 0 and
                  rec['models_or_output_based_parameters'] is False and rec['actual_training_performed'] is False and
                  rec['native_CCTV_DEV_wild_FINAL_used'] is False, 'No initial admission/training/output-selected labels')
            check(rec['geometry'] == {'crop': None, 'warp': None, 'alignment': None,
                  'target512': 'direct1024 LANCZOS; never256 upscale'}, 'Geometry policy differs')
            input_digests[case_id] = H.sha(pixels.tobytes())
            audits.append({'id': case_id, 'profile': profile['id'], 'independent_channel_mismatches': mismatch,
                'input_RGB8_sha256': input_digests[case_id], 'local_rectangle': rectangle if local else None,
                'outside_corruption_footprint_exact': True})
    visual = strict_json(cache['input_only_visual_manifest.json'])
    check(visual['complete'] is True and visual['input_cases'] == visual['exact256_cells_readback'] == 512 and
          len(visual['cells']) == 512 and len(visual['sheets']) == 32 and
          visual['prospective_profile_order'] == list(REVIEW_ORDER) and
          visual['root_input_only_review_status'] == 'PENDING' and visual['clean_reference_or_model_panels'] == 0 and
          visual['clear_profile_withheld_until_final_four_pages'] is True and
          visual['criteria_sha256'] == CRITERIA_SHA and
          visual['lock_earlier_input_decisions_before_clear_or_clean_reference_view'] is True and
          visual['blinded_to_other_degraded_versions'] is False and visual['no_automatic_input_quality_labels'] is True and
          doc['input_only_sheets'] == visual['sheets'], 'Exact input-only review census/order differs')
    cells_seen = set()
    for i, sheet in enumerate(visual['sheets']):
        profile, page = REVIEW_ORDER[i // 4], i % 4 + 1
        name = 'input_only_sheets/%02d_%s_%02d.png' % (i + 1, profile, page)
        expected_names.add(name)
        check(sheet['path'] == name and sheet['profile'] == profile and sheet['page'] == page and
              sheet['review_index'] == i + 1 and sheet['cases'] == 16 and
              sheet['width'] == 1024 and sheet['height'] == 1184, 'Exact sheet sequence/geometry differs')
        pixels = H.decode(get({k: sheet[k] for k in ('path', 'sha256', 'bytes')}), (1024, 1184), 'RGB')
        cells = [c for c in visual['cells'] if c['sheet'] == name]
        check(len(cells) == 16 and [c['reference_id'] for c in cells] == ids[(page - 1) * 16:page * 16],
              'Sixteen ordered source cells/sheet required')
        for cell_index, cell in enumerate(cells):
            case_id = cell['reference_id'] + '__' + profile
            x, y = cell_index % 4 * 256, 70 + cell_index // 4 * 286
            check(case_id not in cells_seen and cell['profile'] == profile and cell['split'] == 'TRAIN' and
                  cell['x'] == x and cell['y'] == y and cell['width'] == cell['height'] == 256,
                  'Exact unique256 cell geometry/source differs')
            cells_seen.add(case_id)
            check(H.sha(pixels[y:y + 256, x:x + 256].tobytes()) == cell['RGB8_sha256'] == input_digests[case_id],
                  'Sheet panel differs from actual input pixels')
    check(cells_seen == set(case_map) and expected_names == set(files), 'Exact expected payload namespace differs')
    check(doc['all_input_reviews_pending'] is True and doc['optimization_permitted_now'] is False and
          doc['TRAIN_rows_initially_strong_control_excluded'] == 64 and
          doc['all512_need_input_review_and_explicit_pilot_admission'] is True and
          doc['new_neural_forward_calls'] == doc['optimizer_updates'] == 0 and
          doc['actual_training_performed'] is False and doc['user_wild_native_CCTV_DEV_reserved_FINAL_used'] is False and
          doc['input_quality_threshold_calibrated'] is False and
          doc['historical_membership_or_quality_gates_changed'] is False,
          'Preparation/admission/research boundaries differ')
    # Recheck the externally frozen contracts and producer sources after pixels.
    H.bound(folder / 'manifest.json', manifest_sha)
    for name, pin in doc['source_pins'].items():
        H.bound(ROOT / name, pin['sha256'], pin['bytes'])
    pin_bytes(doc['input_only_criteria'])
    for selected in selection['cohort']:
        pin_bytes(selected['source1024']); pin_bytes(selected['target256'])
    no_producer_or_neural()
    receipt = {'format': 'codeformer-filter-train-v2-independent-saved-pixel-audit-v1',
        'complete': True, 'pass': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': H.sha(raw), 'manifest_bytes': len(raw),
        'auditor_sha256': H.sha(Path(__file__).read_bytes()), 'independent_helper_sha256': HELPER_SHA,
        'selection_sha256': SELECTION_SHA, 'approved_profiles_sha256': PROFILES_SHA,
        'input_only_criteria_sha256': CRITERIA_SHA,
        'pairs': 512, 'TRAIN_pairs': 512, 'DEV_pairs': 0, 'reference_photographs': 64, 'profiles': 8,
        'all512_exact_independent_filter_pixels': True, 'channel_mismatches': 0,
        'all128_direct1024_clean_reference_pixels': True, 'all512_RGB8_float32_fingerprints': True,
        'all512_exact_footprints_and_outside_pixels': True, 'all512_exact256_sheet_cells': True,
        'sheets': 32, 'unique_rectangles': len(rectangles), 'rectangle_size': [128, 128],
        'rectangle_sizes_varied': False, 'selected_ids_independently_reconstructed': True,
        'full_catalog_source_byte_split_overlap': False, 'source_person_identity_disjointness_proven': False,
        'cohort_attribution_terms_exact': True, 'separate_full391_source_audit_sha256': H.sha(source_audit_raw),
        'output_files': len(files) + 1, 'output_payload_bytes': sum(p['bytes'] for p in files.values()) + len(raw),
        'closed_namespace_hashes': True, 'arithmetic_tolerance_for_saved_pixels': 0,
        'producer_SciPy_neural_imported': False, 'new_image_writes': 0,
        'new_neural_gradient_optimizer_VM_browser_calls': 0, 'returned_code_executed': False,
        'optimization_admitted_by_this_receipt': False, 'operator_review_separate': True,
        'automatic_quality_threshold_validated': False, 'model_qualified': False, 'goal_complete': False,
        'versions': {'Python': sys.version, 'NumPy': np.__version__, 'Pillow': PILLOW_VERSION},
        'seconds': time.monotonic() - started, 'cases': audits}
    out.mkdir()
    with (out / 'saved_pixel_audit_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    return {'complete': True, 'pass': True, 'pairs': 512, 'channel_mismatches': 0,
            'receipt': str(out / 'saved_pixel_audit_v1.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.manifest_sha, args.out)), flush=True)
