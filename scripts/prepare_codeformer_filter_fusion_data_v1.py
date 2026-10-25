"""Verify existing TRAIN dark pixels and prepare input-only sheets; no models.

This writes prospective metadata/contact sheets only. It never synthesizes new
corruptions, modifies historical assets, admits training, or launches a VM.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_filter_fusion_training_v1 import data as d

HOLDS = 'outputs/codeformer_filter_train_v2_preparation/candidate_holds_v1.json'
DATA = 'outputs/codeformer_filter_train_v2_data'
MIXED = 'outputs/cctv_dgp_mixed_vm_v9_r2'
CRITERIA = 'outputs/codeformer_filter_train_v2_preparation/input_only_review_criteria_v1.json'
PREP = 'outputs/codeformer_filter_fusion_training_v1_preparation'
PINS = {
    HOLDS: '5c59dbf7826e976eb4e1605183b5a3255eaa518745e1ee085b2ba9062cbde24a',
    DATA + '/manifest.json': '16c804f3f6492332c02ed7c33ab1e2448d6ae84e4e80b0eede16cdd25068e683',
    'outputs/cctv_codeformer_adaptation_v2_vm/references.json':
        '55b892ee5d55a123ea6cb55f84f322824faaec956f59168e443ff94b93337cde',
    CRITERIA: '9b66c4e0ab4d460a9790b225e470a121cef379f254f4e6cf52d94d58c99045d6',
}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def pin(path):
    path = Path(path)
    info = path.lstat()
    d.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
              not path.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400,
              'Singly linked regular source required')
    record = {'path': path.relative_to(ROOT).as_posix(), 'bytes': info.st_size,
              'sha256': sha(path.read_bytes())}
    d.verified_bytes(ROOT, record)
    return record


def read(name, expected=None):
    p = pin(ROOT / name)
    d.require(expected is None or p['sha256'] == expected, 'Historical metadata pin differs: ' + name)
    return json.loads(d.verified_bytes(ROOT, p)), p


def pixel_record(pixels):
    import numpy as np
    return {'RGB8_sha256': sha(pixels.tobytes()),
            'float32_sha256': sha(np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255)).tobytes()),
            'shape': list(pixels.shape), 'dtype': 'uint8'}


def decode(record, size):
    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(d.verified_bytes(ROOT, record))) as image:
        d.require(image.mode == 'RGB' and image.size == (size, size), 'Exact existing RGB geometry required')
        image.load()
        return np.asarray(image).copy()


def collect():
    import numpy as np
    from PIL import Image
    start = time.monotonic()
    checked = {}
    def clock():
        d.require(time.monotonic() - start < 300, 'Data verification checkpoint cap300s exceeded')
    holds, checked[HOLDS] = read(HOLDS, PINS[HOLDS])
    candidates = d.validate_holds(holds)
    sources = {r['reference_id'] for r in candidates}
    manifest, checked[DATA + '/manifest.json'] = read(DATA + '/manifest.json', PINS[DATA + '/manifest.json'])
    refs_name = 'outputs/cctv_codeformer_adaptation_v2_vm/references.json'
    catalog, checked[refs_name] = read(refs_name, PINS[refs_name])
    criteria, checked[CRITERIA] = read(CRITERIA, PINS[CRITERIA])
    mixed_name = MIXED + '/cohort_assets_manifest_v9.json'
    mixed, checked[mixed_name] = read(mixed_name)
    producer_name = 'scripts/prepare_cctv_dgp_mixed_assets_v9_r2.py'
    camera_name = MIXED + '/scripts/cctv_camera_stress.py'
    for name in (producer_name, camera_name):
        checked[name] = pin(ROOT / name)
    # The exact historical producer and camera are retained as source evidence;
    # no dynamic import, camera call or new degradation is executed here.
    catalog = {r['id']: r for r in catalog['references']}
    refs = {r['reference_id']: r for r in manifest['references']}
    oldrefs = {r['id']: r for r in mixed['references']}
    dark = {r['reference_id']: r for r in mixed['training_cases'] if
            r['profile'] == 'lowlight_lr32' and r['reference_id'] in sources}
    d.require(set(dark) == sources and len(sources) == 57, 'Missing same-reference TRAIN dark join')
    pixel_buffers = {}
    dark_rows = []
    recipe = {'id': 'lowlight_lr32', 'sigma': .8, 'motion': 0, 'long_edge': 32,
              'exposure': .35, 'shot': .0008, 'read': .002, 'jpeg': 40}
    order = sorted(sources, key=lambda rid: (d._order('dark-review', rid), rid))
    for rid in order:
        clock()
        row, ref, cat, old = dark[rid], refs[rid], catalog[rid], oldrefs[rid]
        d.require(cat['historical_role'] == ref['historical_role'] == old['role'] == 'train'
                  and cat['decision'] == 'accept_clean_restoration' and
                  cat['suitable_as_hq_train_candidate'] is True and
                  old['evaluation_source_kind'] == 'HQ_FFHQ_counterpart' and
                  old['hq_source_sha256'] == ref['source1024']['sha256'] == cat['source_sha256'],
                  'Canonical TRAIN/HQ source lineage differs')
        d.require(row['camera'] == recipe and old['bounds'] == [0, 0, 256, 256],
                  'Exact retained lowlight recipe/full support required')
        d.require(ref['source1024']['path'] == cat['original_source_path'], 'Physical HQ source path differs')
        source_pin = {k: ref['source1024'][k] for k in ('path', 'bytes', 'sha256')}
        source = decode(source_pin, 1024)
        target_pin = pin(ROOT / MIXED / old['target'])
        d.require(target_pin['sha256'] == mixed['assets_sha256'][old['target']], 'Canonical old target byte hash differs')
        target = decode(target_pin, 256)
        direct = np.asarray(Image.fromarray(source).resize((256, 256), Image.Resampling.LANCZOS)).copy()
        d.require(np.array_equal(target, direct) and sha(target.tobytes()) ==
                  mixed['canonical_target_rgb_sha256'][rid] == ref['target256_pixels']['RGB8_sha256'],
                  'Historical camera target is not current direct-HQ canonical256')
        input_pin = pin(ROOT / MIXED / row['input'])
        d.require(input_pin['sha256'] == mixed['assets_sha256'][row['input']], 'Existing dark bytes differ')
        pixels = decode(input_pin, 256)
        pixel_buffers[row['id']] = pixels
        dark_rows.append({'id': row['id'], 'reference_id': rid, 'historical_role': 'train',
                         'split': 'TRAIN', 'role': 'dark', 'profile': 'lowlight_lr32',
                         'input': input_pin, 'input_pixels': pixel_record(pixels),
                         'clean_source1024': source_pin, 'canonical_camera_target256': target_pin,
                         'canonical_camera_target256_pixels': pixel_record(target),
                         'attribution': ref['attribution'], 'retained_camera_case': row,
                         'historical_mixed_reference': old,
                         'historical_thumbnail_fields_are_legacy_not_canonical_target': True,
                         'source_pretraining_or_person_overlap_unknown': True,
                         'input_only_review_decision': 'PENDING', 'optimization_permitted_now': False,
                         'no_new_corruption_generated': True, 'native_DEV_FINAL_used': False})
    # Selected filter/clear records retain exact prior pixels and metadata pins.
    allcases = {r['id']: r for r in manifest['cases']}
    case_by_id = {}
    for candidate in candidates:
        row = allcases[candidate['id']]
        record, record_pin = read(DATA + '/' + row['pair_record']['path'], row['pair_record']['sha256'])
        d.require(record['reference_id'] == candidate['reference_id'] and record['split'] == 'TRAIN'
                  and record['historical_role'] == 'train' and record['id'] == candidate['id']
                  and record['profile']['id'] == candidate['profile'], 'Filter candidate join differs')
        for field in ('input',):
            p = {**record[field], 'path': DATA + '/' + record[field]['path']}
            buf = decode(p, 256)
            d.require(pixel_record(buf)['RGB8_sha256'] == record['input_pixels']['RGB8_sha256'],
                      'Candidate input pixels differ')
        case_by_id[row['id']] = {'id': row['id'], 'reference_id': row['reference_id'],
            'split': 'TRAIN', 'profile': row['profile'], 'input': p,
            'input_pixels': record['input_pixels'], 'pair_record': record_pin,
            'target256': {**record['targets']['clean256'],
                          'path': DATA + '/' + record['targets']['clean256']['path']},
            'target256_pixels': record['target256_pixels'],
            'corruption_footprint': {**record['corruption_footprint'],
                                     'path': DATA + '/' + record['corruption_footprint']['path']},
            'corruption_footprint_pixels_sha256': record['corruption_footprint_pixels_sha256'],
            'local_rectangle_xyxy_half_open': record['local_rectangle_xyxy_half_open'],
            'attribution': record['attribution'], 'prior_hold_ledger_row': candidate,
            'optimization_permitted_now': False}
    # Closure check every metadata/input/source file already opened: no substitution.
    for p in checked.values():
        d.verified_bytes(ROOT, p)
    for row in dark_rows:
        for key in ('input', 'clean_source1024', 'canonical_camera_target256'):
            d.verified_bytes(ROOT, row[key])
    return {'holds': holds, 'checked': checked, 'dark_rows': dark_rows, 'pixels': pixel_buffers,
            'case_by_id': case_by_id, 'review_order': order, 'criteria': criteria,
            'seconds': time.monotonic() - start}


def write_json(path, document):
    payload = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8') + b'\n'
    with Path(path).open('xb') as handle:
        handle.write(payload)
    return pin(path)


def prepare():
    from PIL import Image, ImageDraw
    prepared = collect()  # Verify everything before creating the exclusive output namespace.
    out = ROOT / PREP
    sheetroot = out / 'data_dark_input_sheets_v1'
    d.require(not sheetroot.exists() and not (out / 'data_dark_coverage_v1.json').exists(),
              'Fresh namespace required; no overwrite or implicit retry')
    sheetroot.mkdir()
    pages = []
    rows = prepared['dark_rows']
    for offset in range(0, len(rows), 16):
        subset = rows[offset:offset + 16]
        sheet = Image.new('RGB', (1024, 50 + 286 * 4), '#e7e4de')
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 3), 'Existing TRAIN lowlight only | 256px exact panels | PENDING input-only review', fill='black')
        draw.text((4, 22), 'No target or model panels. Prior source/filter exposure prevents blinded FINAL claims.', fill='black')
        cells = []
        for cell, row in enumerate(subset):
            x, y = 256 * (cell % 4), 50 + 286 * (cell // 4)
            draw.text((x + 3, y), row['reference_id'], fill='black')
            draw.text((x + 3, y + 13), 'retained lowlight_lr32', fill='black')
            sheet.paste(Image.fromarray(prepared['pixels'][row['id']]), (x, y + 30))
            cells.append({'id': row['id'], 'reference_id': row['reference_id'],
                          'input': row['input'], 'RGB8_sha256': row['input_pixels']['RGB8_sha256'],
                          'panel_xyxy': [x, y + 30, x + 256, y + 286], 'decision': 'PENDING'})
        path = sheetroot / f'{len(pages) + 1:02d}.png'
        with path.open('xb') as handle:
            sheet.save(handle, format='PNG')
        pages.append({'page': len(pages) + 1, 'sheet': pin(path), 'cells': cells,
                      'nonempty_panels': len(subset), 'blank_cells': 16 - len(subset)})
    coverage = {'format': 'codeformer-filter-fusion-dark-coverage-v1', 'complete': True,
                'metadata_dependencies': prepared['checked'], 'selected_candidate_sources': 57,
                'same_reference_existing_dark_inputs': 57, 'missing_sources': [],
                'dark_profile': 'lowlight_lr32', 'new_corruptions_generated': 0,
                'new_neural_gradient_optimizer_calls': 0, 'source_target_alignment_verified': True,
                'prior_mixed_target_and_direct_HQ_LANCZOS256_pixels_identical': True,
                'criteria': prepared['checked'][CRITERIA], 'review_order': prepared['review_order'],
                'rows': rows, 'pages': pages, 'input_only_review_complete': False,
                'all57_labels_pending': True, 'optimization_permitted_now': False,
                'root_admission_still_required': True, 'seconds_data_only_verification': prepared['seconds'],
                'limits': ['Source IDs denote photographs, not proven unique people.',
                           'Historical synthetic camera is not calibrated native CCTV.',
                           'Copied saved camera pixels were checked; camera renderer was not rerun.',
                           'Prior source/filter exposure prevents blinded independent FINAL review.',
                           'Root must retain unreadable dark gaps without easier-source replacements.']}
    coverage_pin = write_json(out / 'data_dark_coverage_v1.json', coverage)
    schedule = d.build_schedule(prepared['holds'], rows)
    schedule_pin = write_json(out / 'data_schedule_provisional_v1.json', schedule)
    sampler = d.FiniteSampler(schedule)
    ledger = []
    for _ in range(d.UPDATES):
        preview = sampler.preview()
        ledger.append(preview)
        sampler.commit(preview)  # Metadata simulation only; no optimizer or model exists.
    ledger_doc = {'format': 'codeformer-filter-fusion-prospective-ledger-v1', 'schedule': schedule_pin,
                  'schedule_canonical_sha256': d.digest(schedule), 'states': ledger,
                  'final_prospective_state': sampler.state(), 'actual_updates': 0,
                  'simulated_metadata_commits_only': True, 'no_training_admission': True}
    ledger_pin = write_json(out / 'data_sampler_ledger_provisional_v1.json', ledger_doc)
    bundle = {'format': 'codeformer-filter-fusion-data-bundle-provisional-v1',
              'complete_metadata': True, 'candidate_holds': prepared['checked'][HOLDS],
              'original512_observations_and_holds_retained': True,
              'candidate_clear_filter_cases': prepared['case_by_id'], 'dark_cases': {r['id']: r for r in rows},
              'dark_coverage': coverage_pin, 'schedule': schedule_pin, 'sampler_ledger': ledger_pin,
              'optimization_permitted_now': False, 'explicit_admission': None,
              'historical391TRAIN_53DEV_full406DEV_old72_refused24_reservedFINAL_unchanged': True,
              'actual_training_performed': False, 'neural_calls': 0}
    bundle_pin = write_json(out / 'data_bundle_provisional_v1.json', bundle)
    print(json.dumps({'complete_data_only': True, 'coverage': coverage_pin, 'pages': len(pages),
                      'panels': 57, 'schedule': schedule_pin, 'bundle': bundle_pin,
                      'new_neural_calls': 0, 'training_admission': False}))


def linear_contract():
    """Freeze exact57 inputs/formula before rendering any new dark image."""
    coverage, covpin = read(PREP + '/data_dark_coverage_v1.json')
    bundle, bunpin = read(PREP + '/data_bundle_provisional_v1.json')
    holds, holdpin = read(HOLDS, PINS[HOLDS])
    selected = d.validate_holds(holds)
    clear = [bundle['candidate_clear_filter_cases'][r['id']] for r in selected if r['profile'] == 'clear']
    d.require(len(clear) == 57 and {r['reference_id'] for r in clear} ==
              {r['reference_id'] for r in coverage['rows']}, 'All57 samephoto anchors required')
    clear.sort(key=lambda row: (d._order('dark-review', row['reference_id']), row['reference_id']))
    for row in clear:
        d.rgb8(ROOT, row['input'], row['input_pixels']['RGB8_sha256'])
    doc = {'format': 'codeformer-filter-fusion-linear-dark-proxy-contract-v1', 'complete': True,
        'frozen_before_generation': True, 'root_formula_authorization': 'root_message_20261018_factor0p35',
        'seed': d.SEED, 'profile': d.DARK_LINEAR_PROFILE, 'exposure_linear': d.DARK_LINEAR_EXPOSURE,
        'formula': {'dtype': 'float64', 'normalize': 'RGB8.astype(float64)/255',
            'inverse': 's<=0.04045?s/12.92:((s+0.055)/1.055)**2.4',
            'linear_exposure': 'linear*0.35',
            'forward': 'l<=0.0031308?12.92*l:1.055*l**(1/2.4)-0.055',
            'quantization': 'floor(clip(s,0,1)*255+0.5).astype(uint8)',
            'no_floating_boundary_epsilon_or_decimal_tie_override': True},
        'no_interpolation_blur_noise_JPEG_crop_alignment': True,
        'same57clear_sources_all_included_no_visual_quality_choice': True,
        'old_dark57': covpin, 'old_provisional_bundle': bunpin, 'candidate_holds': holdpin,
        'criteria': coverage['criteria'], 'sources': clear,
        'source_pins': {name: pin(ROOT / name) for name in
                        ['research/codeformer_filter_fusion_training_v1/data.py',
                         'scripts/prepare_codeformer_filter_fusion_data_v1.py',
                         'tests/test_codeformer_filter_fusion_data_v1.py']},
        'caps': {'inputs': 57, 'PNGs': 57, 'sheets': 4, 'max_files': 64,
                 'max_bytes': 33554432, 'checkpoint_seconds': 180},
        'labels_initially': 'PENDING', 'optimization_permitted_now': False,
        'purpose': 'Brightness-only frozen-R2 behavior retention anchor; not paired restoration quality evidence',
        'native_camera_calibration_or_old_lowlight_equivalence': False,
        'old57stress_full406DEV_dark_controls_retained': True,
        'model_gradient_optimizer_calls': 0, 'real_CCTV_or_user_wild_or_FINAL_used': False}
    p = write_json(ROOT / PREP / 'data_dark_linear_contract_v1.json', doc)
    print(json.dumps(p))


def prepare_linear():
    """Render one fixed brightness-only proxy per57 clear source; pending review."""
    import numpy as np
    from PIL import Image, ImageDraw
    start = time.monotonic()
    contract, contractpin = read(PREP + '/data_dark_linear_contract_v1.json')
    approval, approvalpin = read(PREP + '/data_dark_linear_math_review_by_model_v1.json')
    d.require(approval.get('complete') is True and approval.get('source_blockers') == [] and
              approval.get('contract_sha256') == contractpin['sha256'], 'Independent exactcontract mathreview required')
    for p in contract['source_pins'].values():
        d.verified_bytes(ROOT, p)
    d.require(contract['exposure_linear'] == .35 and contract['profile'] == d.DARK_LINEAR_PROFILE
              and len(contract['sources']) == 57, 'Frozenbrightness recipe/source count differs')
    namespace = ROOT / PREP / 'data_dark_linear_v1'
    namespace.mkdir()  # Exclusive, no overwrite/resume/fallback.
    (namespace / 'inputs').mkdir()
    (namespace / 'sheets').mkdir()
    rows, pixels_by_id = [], {}
    for source in contract['sources']:
        d.require(time.monotonic() - start < 180, 'Linearproxy checkpoint elapsedcap180s exceeded')
        rgb = d.rgb8(ROOT, source['input'], source['input_pixels']['RGB8_sha256'])
        output = d.linear_dark_proxy(rgb)
        rid = source['reference_id']
        identifier = rid + '__' + d.DARK_LINEAR_PROFILE
        path = namespace / 'inputs' / (identifier + '.png')
        with path.open('xb') as handle:
            Image.fromarray(output).save(handle, format='PNG')
        pixels_by_id[identifier] = output
        rows.append({'id': identifier, 'reference_id': rid, 'historical_role': 'train', 'split': 'TRAIN',
            'role': 'dark', 'profile': d.DARK_LINEAR_PROFILE, 'input': pin(path),
            'input_pixels': pixel_record(output), 'source_clear_input': source['input'],
            'source_clear_pixels': source['input_pixels'], 'source_pair_record': source['pair_record'],
            'attribution': source['attribution'], 'recipe_contract': contractpin,
            'retention_target': 'frozen_R2_on_same_input_not_clean_identity_truth',
            'input_only_review_decision': 'PENDING', 'optimization_permitted_now': False,
            'realistic_camera_or_native_CCTV_claim': False})
    pages = []
    for offset in range(0, len(rows), 16):
        subset = rows[offset:offset + 16]
        sheet = Image.new('RGB', (1024, 1194), '#e7e4de')
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 3), 'TRAIN brightness-only retention proxy | linear exposure0.35 | exact256 input panels', fill='black')
        draw.text((4, 22), 'PENDING input-only review. No clean target or model panels; old camera stress remains retained.', fill='black')
        cells = []
        for cell, row in enumerate(subset):
            x, y = 256 * (cell % 4), 50 + 286 * (cell // 4)
            draw.text((x + 3, y), row['reference_id'], fill='black')
            draw.text((x + 3, y + 13), 'linear-sRGB exposure0.35', fill='black')
            sheet.paste(Image.fromarray(pixels_by_id[row['id']]), (x, y + 30))
            cells.append({'id': row['id'], 'reference_id': row['reference_id'], 'input': row['input'],
                          'RGB8_sha256': row['input_pixels']['RGB8_sha256'],
                          'panel_xyxy': [x, y + 30, x + 256, y + 286], 'decision': 'PENDING'})
        path = namespace / 'sheets' / f'{len(pages) + 1:02d}.png'
        with path.open('xb') as handle:
            sheet.save(handle, format='PNG')
        pages.append({'page': len(pages) + 1, 'sheet': pin(path), 'cells': cells,
                      'nonempty_panels': len(subset), 'blank_cells': 16 - len(subset)})
    doc = {'format': 'codeformer-filter-fusion-linear-dark-proxies-v1', 'complete': True,
        'contract': contractpin, 'independent_math_review': approvalpin, 'rows': rows, 'pages': pages,
        'all57_same_reference_clear_sources': True, 'newPNG_inputs': 57, 'input_only_sheets': 4,
        'all57_review_decisions_pending': True, 'optimization_permitted_now': False,
        'new_neural_gradient_optimizer_VM_calls': 0,
        'brightness_only_retention_anchor_not_paired_restoration_quality_or_native_evidence': True,
        'old57lowlightstress_and_full406DEV_unchanged': True,
        'seconds': time.monotonic() - start}
    files = {p.relative_to(namespace).as_posix(): pin(p) for p in namespace.rglob('*') if p.is_file()}
    d.require(len(files) == 61 and sum(p['bytes'] for p in files.values()) < 33554432,
              'Finite61image payload/max32MiB differs')
    doc['files'] = files
    manifestpin = write_json(namespace / 'manifest.json', doc)
    for p in [*contract['source_pins'].values(), *(r['input'] for r in rows),
              *(r['source_clear_input'] for r in rows), *(p['sheet'] for p in pages)]:
        d.verified_bytes(ROOT, p)
    print(json.dumps({'complete_data_only': True, 'manifest': manifestpin, 'inputs': 57,
                      'pages': 4, 'neural_calls': 0, 'optimization_permitted_now': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--prepare-dark', action='store_true')
    parser.add_argument('--freeze-linear-contract', action='store_true')
    parser.add_argument('--prepare-linear-dark', action='store_true')
    args = parser.parse_args()
    d.require(sum([args.verify, args.prepare_dark, args.freeze_linear_contract, args.prepare_linear_dark]) == 1,
              'Choose exactly one data-only action')
    if args.verify:
        result = collect()
        print(json.dumps({'complete': True, 'same_reference_dark_inputs': len(result['dark_rows']),
                          'candidate_clear_filter_cases': len(result['case_by_id']),
                          'new_images_written': 0, 'neural_calls': 0, 'seconds': result['seconds']}))
    elif args.prepare_dark:
        prepare()
    elif args.freeze_linear_contract:
        linear_contract()
    else:
        prepare_linear()


if __name__ == '__main__':
    main()
