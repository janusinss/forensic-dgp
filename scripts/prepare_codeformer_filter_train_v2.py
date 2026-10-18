"""Prepare 512 deterministic TRAIN filter variants; data only, no model or training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version
import scipy

ROOT = Path(__file__).resolve().parents[1]
FILTER_PATH = 'research/codeformer_filter_pairs_v1/filters.py'
FILTER_SHA = 'a806eaa0fe61f965e7a071fa6af084033af0ff55c84bb9a45e708baac1fa189c'
MAX_FILES, MAX_BYTES, MAX_SECONDS = 1300, 512 * 1024**2, 600
FAILURE_RESERVE_BYTES = 16 * 1024
PROFILE_IDS = ('clear', 'gaussian_2p4', 'anisotropic_h5_v0p7', 'anisotropic_h0p7_v5',
    'repeated_rows_4_blend0p75', 'global_mosaic_8', 'local_gaussian_3p0', 'local_mosaic_8')
INPUT_REVIEW_ORDER = ('global_mosaic_8', 'local_mosaic_8', 'local_gaussian_3p0',
    'repeated_rows_4_blend0p75', 'gaussian_2p4', 'anisotropic_h5_v0p7', 'anisotropic_h0p7_v5', 'clear')
SOURCE_FILES = ('scripts/prepare_codeformer_filter_train_v2.py', FILTER_PATH,
    'research/codeformer_filter_train_v2/README.md')
CRITERIA_PATH = 'outputs/codeformer_filter_train_v2_preparation/input_only_review_criteria_v1.json'
RECTANGLE_RULE = "SHA256 UTF8('codeformer-filter-train-v2/rectangle/'+source_sha256); x0=byte0%9*16,y0=byte1%9*16; half-open[x0,y0,x0+128,y0+128]"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8')


def read_json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    def finite_float(token):
        value = float(token)
        require(math.isfinite(value), 'Nonfinite JSON numeric token')
        return value
    require(isinstance(data, bytes), 'UTF8 JSON bytes required')
    return json.loads(data.decode('utf-8'), object_pairs_hook=pairs,
        parse_float=finite_float,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def canonical_relative(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
        'Canonical relative path required')
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and
        all(part not in ('', '.', '..') for part in path.parts), 'Canonical relative path required')
    return path


def safe_parents(path):
    path = Path(path).absolute()
    require(path.resolve(strict=True).is_relative_to(ROOT), 'File must remain within workspace')
    for part in (path, *path.parents):
        if part == ROOT.parent:
            break
        info = part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
            'Symlink/reparse path forbidden')
    return path


def regular(path):
    path = safe_parents(path)
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Ordinary single-link file required')
    return path


def read_bound(path, fingerprint, size=None, cap=8 * 1024**2):
    require(isinstance(fingerprint, str) and re.fullmatch('[0-9a-f]{64}', fingerprint),
        'Canonical SHA256 required')
    path = regular(path); before = path.stat()
    require(before.st_size <= cap and (size is None or before.st_size == size), 'Bounded exact source size required')
    data = path.read_bytes(); after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino, before.st_nlink) ==
        (after.st_size, after.st_mtime_ns, after.st_ino, after.st_nlink), 'Source changed during read')
    require(digest(data) == fingerprint, 'Source byte fingerprint differs')
    return data


def from_pin(pin):
    relative = canonical_relative(pin['path'])
    path = ROOT.joinpath(*relative.parts)
    require(path.absolute() == Path(pin['absolute_path']).absolute(), 'Physical/logical source path differs')
    return read_bound(path, pin['sha256'], pin['bytes'])


def no_neural_framework():
    require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')),
        'No neural framework permitted in data-only producer')


no_neural_framework()
read_bound(ROOT / FILTER_PATH, FILTER_SHA)
_spec = importlib.util.spec_from_file_location('_frozen_filter_pairs_v1_cpu', ROOT / FILTER_PATH)
FILTERS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FILTERS)
PROFILES = tuple(json.loads(json.dumps(p)) for p in FILTERS.PROFILES if p['id'] in PROFILE_IDS)
require(tuple(p['id'] for p in PROFILES) == PROFILE_IDS, 'Frozen eight-profile subset differs')


def profiles_document():
    return {'format': 'codeformer-filter-train-v2-profiles', 'approved_before_generation_and_review': True,
        'input_size': [256, 256], 'profile_order': list(INPUT_REVIEW_ORDER),
        'profiles': json.loads(json.dumps(PROFILES)),
        'input_geometry': 'direct RGB1024 to Pillow BILINEAR256',
        'target512_geometry': 'direct RGB1024 to Pillow LANCZOS512',
        'target256_geometry': 'retained exact PNG; pixels must equal direct RGB1024 to Pillow LANCZOS256',
        'Gaussian': {'arithmetic': 'float64', 'implementation': 'scipy.ndimage.gaussian_filter',
            'sigma_order': '[sigma_y,sigma_x,0]', 'edge_mode': 'reflect', 'truncate': 3.0},
        'mosaic': 'RGB255 float64 block means, round half up, constant block replication; cell8 divides128local square',
        'repeated_rows': 'original*(1-0.75)+first row of each4-row block*0.75, round half up',
        'rounding': 'floor(clip(float64RGB255,0,255)+0.5) to uint8',
        'local_rectangle_rule': RECTANGLE_RULE,
        'local_edges': 'hard replacement; Gaussian on full canvas then replace rectangle; mosaic only within128square; exact original outside',
        'corruption_footprint_is_not_completion_or_removal_mask': True,
        'random_number_generator_calls': 0, 'brightness_contrast_or_colour_operation': False,
        'clean_target_or_model_output_parameter_selection': False, 'all_cases_pending_input_only_review': True,
        'strong_profile_tag_does_not_override_actual_input_review': True,
        'initially_excluded_strong_profiles': ['repeated_rows_4_blend0p75'],
        'prospective_rows_admission': 'Only an explicit new pilot protocol may admit individually readable row cases; no automatic change to old strong-control exclusions',
        'unchanged_24_prior_insufficient_controls_retained_separately': True, 'new_training_calls': 0}


def local_rectangle(source_sha):
    require(isinstance(source_sha, str) and re.fullmatch('[0-9a-f]{64}', source_sha), 'Canonical source SHA256 required')
    raw = hashlib.sha256(('codeformer-filter-train-v2/rectangle/' + source_sha).encode('utf-8')).digest()
    x, y = (raw[0] % 9) * 16, (raw[1] % 9) * 16
    return [x, y, x + 128, y + 128]


def validate_rectangle(rectangle):
    require(isinstance(rectangle, (list, tuple)) and len(rectangle) == 4 and
        all(type(v) is int for v in rectangle), 'Integer half-open rectangle required')
    x0, y0, x1, y1 = rectangle
    require(0 <= x0 <= 128 and 0 <= y0 <= 128 and x0 % 16 == y0 % 16 == 0 and
        x1 == x0 + 128 and y1 == y0 + 128, 'Exact128 square on16-pixel grid required')
    return x0, y0, x1, y1


def apply_profile(image, profile, rectangle):
    FILTERS.rgb256(image)
    require(profile in PROFILES, 'Exactly an approved eight-profile specification required')
    x0, y0, x1, y1 = validate_rectangle(rectangle)
    if not profile['operator'].startswith('local_'):
        return FILTERS.apply_profile(image, profile)
    original = image.copy(); output = image.copy(); footprint = np.zeros((256, 256), np.uint8)
    if profile['operator'] == 'local_gaussian':
        p = profile['parameters']; filtered = FILTERS.gaussian(original, p['sigma_x'], p['sigma_y'])
        output[y0:y1, x0:x1] = filtered[y0:y1, x0:x1]
    elif profile['operator'] == 'local_mosaic':
        output[y0:y1, x0:x1] = FILTERS.mosaic(original[y0:y1, x0:x1], profile['parameters']['cell'])
    else:
        raise ValueError('Unknown local operator')
    footprint[y0:y1, x0:x1] = 255
    require(np.array_equal(image, original) and np.array_equal(output[footprint == 0], image[footprint == 0]),
        'Local filter mutated caller or pixels outside footprint')
    return np.ascontiguousarray(output), footprint


def stratum(row):
    return row['attribution']['license_url'], row['official_category']


def selected_ids(references):
    train = [r for r in references if r['historical_role'] == 'train']
    require(len(train) == 391, '391 historical TRAIN photographs required')
    groups = defaultdict(list)
    for row in train:
        key = stratum(row)
        require(all(isinstance(v, str) and v for v in key), 'Explicit license/source-category stratum required')
        groups[key].append(row)
    require(len(groups) == 9 and all(groups.values()), 'Nine nonempty frozen TRAIN strata required')
    denominator = len(train) - len(groups)
    extra_seats = 64 - len(groups)
    allocations = {key: 1 + extra_seats * (len(rows) - 1) // denominator for key, rows in groups.items()}
    remaining = 64 - sum(allocations.values())
    remainder_order = sorted(groups, key=lambda key: (-(extra_seats * (len(groups[key]) - 1) % denominator), key))
    for key in remainder_order[:remaining]:
        allocations[key] += 1
    chosen = []
    for key, rows in groups.items():
        require(allocations[key] <= len(rows), 'Stratum allocation exceeds available photographs')
        ranked = sorted(rows, key=lambda row: (digest(('codeformer-filter-fusion-train-admission-v1|' +
            row['id'] + '|' + row['source_sha256']).encode('utf-8')), row['id']))
        chosen.extend(row['id'] for row in ranked[:allocations[key]])
    require(len(set(chosen)) == len(chosen) == 64, 'Deterministic64 distinct TRAIN references required')
    return sorted(chosen), allocations


def validate_selection(selection, catalog):
    require(selection.get('format') == 'codeformer-filter-train-v2-prospective-source-cohort-v1' and selection.get('complete') is True,
        'Completed prospective TRAIN64 cohort contract required')
    references = catalog['references']; refs = {r['id']: r for r in references}
    require(len(refs) == len(references) == 444 and Counter(r['historical_role'] for r in references) ==
        {'train': 391, 'validation': 53}, 'Unchanged391 TRAIN/53 DEV catalog required')
    hashes = {role: {r['source_sha256'] for r in references if r['historical_role'] == role}
        for role in ('train', 'validation')}
    require(len(hashes['train']) == 391 and len(hashes['validation']) == 53 and
        not hashes['train'] & hashes['validation'], 'Distinct historical TRAIN/DEV source bytes required')
    expected, allocations = selected_ids(references)
    cohort = selection['cohort']
    require([r['reference_id'] for r in cohort] == expected, 'Prospective nine-stratum64 membership/order differs')
    require(selection['catalog_counts'] == {'TRAIN': 391, 'DEV': 53, 'total': 444},
        'Recorded historical pool counts differ')
    expected_strata = [{'license_url': key[0], 'official_category': key[1],
        'population': sum(stratum(r) == key for r in references if r['historical_role'] == 'train'),
        'selected': value} for key, value in sorted(allocations.items())]
    require(selection['strata'] == expected_strata, 'Recorded source/license allocation differs')
    for row in cohort:
        rid = row['reference_id']; reference = refs[rid]
        require(re.fullmatch('tr_ffhq_[0-9]{5}', rid) and row['split'] == 'TRAIN' and
            row['historical_role'] == reference['historical_role'] == 'train' and
            row['reference_manifest_row'] == reference and reference['decision'] == 'accept_clean_restoration' and
            reference['use_for_optimization'] is False and row['use_for_optimization'] is False,
            'Historical source role/decision/exact catalog row differs')
        require(reference['native_size'] == [1024, 1024] and reference['existing_target_size'] == [256, 256] and
            row['source1024']['sha256'] == reference['source_sha256'] and
            row['target256']['sha256'] == reference['existing_target_sha256'] and
            row['source1024']['path'] == reference['original_source_path'] and
            row['target256']['path'] == reference['original_existing_target_path'] and
            row['attribution'] == reference['attribution'], 'Exact source/target/attribution binding differs')
        require(row['local_rectangle_xyxy_half_open'] == local_rectangle(reference['source_sha256']),
            'Prospective source-hash rectangle differs')
    return cohort, allocations


def rgb(data, size):
    with Image.open(io.BytesIO(data)) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (size, size),
            'Exact square RGB PNG required')
        return np.asarray(image).copy()


def image_bytes(pixels):
    require(pixels.dtype == np.uint8 and pixels.ndim in (2, 3), 'PNG uint8 pixels required')
    stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format='PNG'); data = stream.getvalue()
    with Image.open(io.BytesIO(data)) as image:
        require(np.array_equal(np.asarray(image), pixels), 'Encoded PNG pixel readback differs')
    return data


def pixel_record(pixels):
    return {'shape': list(pixels.shape), 'dtype': str(pixels.dtype),
        'RGB8_sha256': digest(np.ascontiguousarray(pixels).tobytes()),
        'float32_sha256': digest(np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255)).tobytes())}


def check_pixels(pixels, metadata):
    actual = pixel_record(pixels)
    require(actual['RGB8_sha256'] == metadata['rgb8']['pixel_sha256'], 'Bound RGB8 pixel fingerprint differs')
    if 'float32' in metadata:
        require(actual['float32_sha256'] == metadata['float32']['pixel_sha256'], 'Bound float32 pixel fingerprint differs')


def training_admission(profile):
    require(profile in PROFILE_IDS, 'Known profile required')
    return {'optimization_allowed_by_historical_split': True, 'optimization_permitted_now': False,
        'restoration_training_eligibility': 'PENDING_INPUT_ONLY_REVIEW_AND_EXPLICIT_PILOT_PROTOCOL',
        'strong_control_excluded_initially': profile == 'repeated_rows_4_blend0p75'}


def input_only_sheet(profile, images, page):
    require(profile in INPUT_REVIEW_ORDER and type(page) is int and 1 <= page <= 4 and len(images) == 16 and
        len({rid for rid, _ in images}) == 16 and all(re.fullmatch('tr_ffhq_[0-9]{5}', rid) for rid, _ in images),
        'Sixteen distinct ordered TRAIN inputs/page1..4 required')
    sheet = Image.new('RGB', (1024, 1184), 'white'); draw = ImageDraw.Draw(sheet)
    draw.text((4, 3), profile + ' | TRAIN INPUT ONLY | page' + str(page) + '/4 | pending review', fill='black')
    draw.text((4, 19), 'Exact256 cells. No target/model panels. Clear profile withheld until final four pages.', fill='black')
    cells = []
    for index, (rid, pixels) in enumerate(images):
        FILTERS.rgb256(pixels)
        x, y = index % 4 * 256, 70 + index // 4 * 286
        draw.text((x + 3, y - 27), rid + ' | TRAIN', fill='black'); sheet.paste(Image.fromarray(pixels), (x, y))
        cells.append({'reference_id': rid, 'split': 'TRAIN', 'profile': profile, 'x': x, 'y': y,
            'width': 256, 'height': 256, 'RGB8_sha256': digest(pixels.tobytes())})
    data = image_bytes(np.asarray(sheet))
    with Image.open(io.BytesIO(data)) as saved:
        pixels = np.asarray(saved)
        for cell in cells:
            require(digest(pixels[cell['y']:cell['y'] + 256, cell['x']:cell['x'] + 256].tobytes()) ==
                cell['RGB8_sha256'], 'Exact256 sheet cell readback differs')
    return data, cells


class Writer:
    def __init__(self, root, started=None):
        root = Path(root).absolute()
        require(root.parent.resolve(strict=True).is_relative_to(ROOT / 'outputs') and not root.exists(),
            'New exclusive output namespace under workspace outputs required; preserve partial/history')
        safe_parents(root.parent); root.mkdir()
        self.root, self.files, self.bytes = root, {}, 0
        self.started = time.monotonic() if started is None else started

    def clock(self):
        require(time.monotonic() - self.started < MAX_SECONDS, '600-second data preparation cap exceeded')

    def put(self, name, data, failure=False):
        relative = canonical_relative(name)
        require(isinstance(data, bytes), 'Exact output bytes required')
        if not failure:
            self.clock()
        limit_files = MAX_FILES if failure else MAX_FILES - 1
        limit_bytes = MAX_BYTES if failure else MAX_BYTES - FAILURE_RESERVE_BYTES
        require(name not in self.files and len(self.files) + 1 <= limit_files and
            self.bytes + len(data) <= limit_bytes, 'Finite unique output file/byte cap exceeded')
        path = self.root.joinpath(*relative.parts); path.parent.mkdir(parents=True, exist_ok=True)
        safe_parents(path.parent)
        with path.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        require(regular(path).read_bytes() == data, 'Exclusive output byte readback differs')
        pin = {'path': name, 'bytes': len(data), 'sha256': digest(data)}
        self.files[name] = pin; self.bytes += len(data)
        return dict(pin)

    def close_namespace(self):
        actual_files, actual_dirs = set(), set()
        for path in self.root.rglob('*'):
            safe_parents(path)
            relative = path.relative_to(self.root).as_posix()
            if path.is_dir():
                actual_dirs.add(relative)
            else:
                regular(path); actual_files.add(relative)
        expected_dirs = {parent.as_posix() for name in self.files for parent in PurePosixPath(name).parents
            if parent.as_posix() != '.'}
        require(actual_files == set(self.files) and actual_dirs == expected_dirs, 'Output namespace closure differs')
        for name, pin in self.files.items():
            self.clock(); read_bound(self.root / name, pin['sha256'], pin['bytes'], cap=MAX_BYTES)


def prepare(selection_path, selection_sha, profiles_path, profiles_sha, out):
    no_neural_framework(); started = time.monotonic()
    selected_bytes = read_bound(selection_path, selection_sha)
    configuration_bytes = read_bound(profiles_path, profiles_sha)
    selection, configuration = read_json(selected_bytes), read_json(configuration_bytes)
    require(configuration == profiles_document(), 'Approved eight-profile arithmetic/geometry contract differs')
    catalog_bytes = from_pin(selection['catalog'])
    cohort, allocations = validate_selection(selection, read_json(catalog_bytes))
    supporting = list(selection.get('supporting_audit_and_approval_manifests', []))
    if 'source_audit' in selection:
        supporting.append(selection['source_audit'])
    for pin in supporting:
        from_pin(pin)
    sources = {name: {'sha256': digest(regular(ROOT / name).read_bytes()),
        'bytes': (ROOT / name).stat().st_size} for name in SOURCE_FILES}
    require(sources[FILTER_PATH]['sha256'] == FILTER_SHA, 'Historical CPU filter source changed')
    criteria_path = regular(ROOT / CRITERIA_PATH)
    criteria_bytes = criteria_path.read_bytes(); criteria = read_json(criteria_bytes)
    criteria_pin = {'path': CRITERIA_PATH, 'bytes': len(criteria_bytes), 'sha256': digest(criteria_bytes)}
    read_bound(criteria_path, criteria_pin['sha256'], criteria_pin['bytes'])
    require(criteria['format'] == 'codeformer-filter-train-v2-input-only-criteria' and
        criteria['frozen_before_new_inputs_viewed'] is True and
        set(criteria['decisions']) == {'readable', 'borderline', 'insufficient', 'out_of_scope'} and
        criteria['optimization_permitted_by_data_generation'] is False and
        criteria['existing_quality_gate_or_scope_changed'] is False,
        'Prospective input-only criteria/admission boundaries differ')
    versions = {'Python': sys.version, 'NumPy': np.__version__, 'Pillow': pillow_version, 'SciPy': scipy.__version__}
    writer = Writer(out, started); stage = 'metadata'
    try:
        writer.put('cohort_selection.json', selected_bytes); writer.put('approved_profiles.json', configuration_bytes)
        writer.put('input_only_review_criteria.json', criteria_bytes)
        writer.put('README.md', (ROOT / SOURCE_FILES[-1]).read_bytes())
        full = writer.put('observed-support.png', image_bytes(np.full((256, 256), 255, np.uint8)))
        empty = writer.put('corruption_footprints/clear.png', image_bytes(np.zeros((256, 256), np.uint8)))
        rows, proofs, all_inputs = [], [], {profile: [] for profile in INPUT_REVIEW_ORDER}
        for selected in cohort:
            rid = selected['reference_id']; stage = 'reference/' + rid; writer.clock()
            source_bytes, target_bytes = from_pin(selected['source1024']), from_pin(selected['target256'])
            source, target = rgb(source_bytes, 1024), rgb(target_bytes, 256)
            check_pixels(source, selected['source1024']); check_pixels(target, selected['target256'])
            with Image.open(io.BytesIO(source_bytes)) as image:
                clean512 = np.asarray(image.resize((512, 512), Image.Resampling.LANCZOS)).copy()
                direct256 = np.asarray(image.resize((256, 256), Image.Resampling.LANCZOS)).copy()
                clear256 = np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR)).copy()
            require(np.array_equal(target, direct256), 'Retained target differs from direct1024 LANCZOS256')
            check_pixels(clean512, selected['clean512']); check_pixels(clear256, selected['clear256'])
            require(digest(clean512.tobytes()) == selected['reference_manifest_row']['proposed_512_rgb_sha256'],
                'Historical direct512 fingerprint differs')
            targets = {'clean512': writer.put('references/' + rid + '/clean512.png', image_bytes(clean512)),
                'clean256': writer.put('references/' + rid + '/clean256.png', target_bytes)}
            rectangle = selected['local_rectangle_xyxy_half_open']; x0, y0, x1, y1 = validate_rectangle(rectangle)
            local_mask = np.zeros((256, 256), np.uint8); local_mask[y0:y1, x0:x1] = 255
            local_pin = writer.put('corruption_footprints/' + rid + '.png', image_bytes(local_mask))
            proofs.append({'reference_id': rid, 'split': 'TRAIN', 'historical_role': 'train',
                'source1024': selected['source1024'], 'retained_target256_bytes_exact': True,
                'target256_pixels': pixel_record(target), 'clean512_pixels': pixel_record(clean512),
                'clear256_pixels': pixel_record(clear256), 'targets': targets,
                'attribution': selected['attribution'], 'local_rectangle_xyxy_half_open': rectangle})
            for profile in PROFILES:
                writer.clock(); pid = profile['id']; case_id = rid + '__' + pid; stage = 'case/' + case_id
                output, footprint = apply_profile(clear256, profile, rectangle)
                all_inputs[pid].append((rid, output))
                inp = writer.put('inputs/' + case_id + '.png', image_bytes(output))
                foot = local_pin if profile['operator'].startswith('local_') else (empty if pid == 'clear' else full)
                record = {'format': 'codeformer-filter-train-v2-pair', 'id': case_id, 'reference_id': rid,
                    'split': 'TRAIN', 'historical_role': 'train', 'evidence': 'paired_synthetic_filter_TRAIN_admission',
                    'source1024': selected['source1024'], 'attribution': selected['attribution'],
                    'input': inp, 'input_pixels': pixel_record(output), 'targets': targets,
                    'target256_pixels': pixel_record(target), 'target512_pixels': pixel_record(clean512),
                    'observed_support': full, 'corruption_footprint': foot,
                    'corruption_footprint_pixels_sha256': digest(footprint.tobytes()),
                    'corruption_footprint_is_not_completion_or_removal_mask': True,
                    'profile': profile, 'local_rectangle_xyxy_half_open': rectangle if
                        profile['operator'].startswith('local_') else None,
                    'local_rectangle_rule': RECTANGLE_RULE if profile['operator'].startswith('local_') else None,
                    'operation_order': ['verify direct RGB1024', 'BILINEAR direct1024 to RGB256',
                        profile['operator'], 'round uint8 RGB255; save exact PNG'],
                    'geometry': {'crop': None, 'warp': None, 'alignment': None,
                        'target512': 'direct1024 LANCZOS; never256 upscale'},
                    'input_only_review': {'status': 'PENDING', 'decision': None, 'reviewer': None,
                        'input_only': True, 'model_output_used': False,
                        'criteria_sha256': criteria_pin['sha256']}, **training_admission(pid),
                    'RNG_calls': 0, 'models_or_output_based_parameters': False,
                    'actual_training_performed': False, 'native_CCTV_DEV_wild_FINAL_used': False,
                    'automatic_usable_or_insufficient_label': None, 'versions': versions}
                pair = writer.put('pairs/' + case_id + '.json', json_bytes(record))
                rows.append({'id': case_id, 'reference_id': rid, 'split': 'TRAIN', 'profile': pid,
                    'pair_record': pair, 'input': inp, 'input_review': 'PENDING', **training_admission(pid)})
            print(json.dumps({'prepared_references': len(proofs), 'of': 64, 'paired_inputs': len(rows)}), flush=True)
        require(len(rows) == 512 and len(proofs) == 64 and
            Counter(r['profile'] for r in rows) == dict.fromkeys(PROFILE_IDS, 64), 'Exact64x8 census differs')
        sheets, cells = [], []; stage = 'input_only_sheets'
        for profile in INPUT_REVIEW_ORDER:
            for page in range(1, 5):
                images = all_inputs[profile][(page - 1) * 16:page * 16]
                data, page_cells = input_only_sheet(profile, images, page)
                index = len(sheets) + 1
                pin = writer.put('input_only_sheets/%02d_%s_%02d.png' % (index, profile, page), data)
                sheets.append(dict(pin, profile=profile, page=page, review_index=index, cases=16,
                    width=1024, height=1184))
                cells.extend(dict(cell, sheet=pin['path']) for cell in page_cells)
        require(len(sheets) == 32 and len(cells) == 512, 'All32 sheets/512 exact256 cells required')
        writer.put('input_only_visual_manifest.json', json_bytes({'complete': True, 'sheets': sheets, 'cells': cells,
            'input_cases': 512, 'exact256_cells_readback': 512, 'clean_reference_or_model_panels': 0,
            'root_input_only_review_status': 'PENDING', 'prospective_profile_order': list(INPUT_REVIEW_ORDER),
            'criteria_sha256': criteria_pin['sha256'],
            'clear_profile_withheld_until_final_four_pages': True,
            'lock_earlier_input_decisions_before_clear_or_clean_reference_view': True,
            'blinded_to_other_degraded_versions': False, 'no_automatic_input_quality_labels': True}))
        stage = 'final_source_closure'
        read_bound(selection_path, selection_sha); read_bound(profiles_path, profiles_sha)
        read_bound(criteria_path, criteria_pin['sha256'], criteria_pin['bytes'])
        from_pin(selection['catalog'])
        for pin in supporting:
            writer.clock(); from_pin(pin)
        for name, pin in sources.items():
            writer.clock(); read_bound(ROOT / name, pin['sha256'], pin['bytes'])
        for selected in cohort:
            writer.clock(); from_pin(selected['source1024']); from_pin(selected['target256'])
        no_neural_framework(); writer.close_namespace()
        stage = 'manifest'
        manifest = {'format': 'codeformer-filter-train-v2-data', 'complete': True,
            'selection_sha256': selection_sha, 'approved_profiles_sha256': profiles_sha, 'source_pins': sources,
            'input_only_criteria': criteria_pin,
            'historical_pool_counts': {'TRAIN': 391, 'DEV': 53}, 'references': proofs, 'cases': rows,
            'selected_strata': [{'license_url': key[0], 'official_category': key[1], 'count': value}
                for key, value in sorted(allocations.items())],
            'profiles': list(PROFILES), 'input_only_sheets': sheets, 'versions': versions,
            'files': dict(writer.files), 'manifest_excludes_its_own_recursive_hash': True,
            'pairs': 512, 'reference_photographs': 64, 'TRAIN_pairs': 512, 'DEV_pairs': 0,
            'all_input_reviews_pending': True, 'optimization_permitted_now': False,
            'TRAIN_rows_initially_strong_control_excluded': 64,
            'all512_need_input_review_and_explicit_pilot_admission': True,
            'source_terms_and_exact_photo_attribution_retained': True,
            'broader_TRAIN_sample_not_whole391_or_final_evaluation': True,
            'prior72_comparisons_and24_insufficient_controls_unchanged': True,
            'new_neural_forward_calls': 0, 'optimizer_updates': 0, 'actual_training_performed': False,
            'user_wild_native_CCTV_DEV_reserved_FINAL_used': False,
            'no_identity_ethnicity_or_Zamboanga_performance_claim': True,
            'input_quality_threshold_calibrated': False, 'historical_membership_or_quality_gates_changed': False,
            'finite_caps': {'files': MAX_FILES, 'payload_bytes': MAX_BYTES, 'seconds': MAX_SECONDS},
            'seconds': time.monotonic() - started}
        final_pin = writer.put('manifest.json', json_bytes(manifest)); writer.close_namespace(); writer.clock()
        return {'complete': True, 'pairs': 512, 'files': len(writer.files), 'bytes': writer.bytes,
            'manifest': final_pin, 'seconds': time.monotonic() - started,
            'all_input_reviews_pending': True, 'optimization_permitted_now': False, 'new_neural_or_training_calls': 0}
    except BaseException as error:
        failure = {'complete': False, 'stage': stage, 'exception': type(error).__name__,
            'message': str(error)[:3000], 'traceback': traceback.format_exc()[-8000:],
            'files_committed': len(writer.files), 'bytes_committed': writer.bytes,
            'seconds': time.monotonic() - started, 'partial_outputs_preserved': True,
            'optimizer_updates': 0, 'new_neural_or_training_calls': 0}
        writer.put('failure.json', json_bytes(failure), failure=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path); parser.add_argument('--selection-sha', required=True)
    parser.add_argument('--profiles', required=True, type=Path); parser.add_argument('--profiles-sha', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.selection, args.selection_sha, args.profiles, args.profiles_sha, args.out)), flush=True)
