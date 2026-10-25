"""Independent actual PNG-checked returned pixels, selection and quality audit.

Uses NumPy/PIL saved-file arithmetic only. It imports exact pinned pure metric
and quality helpers for a crosscheck after independently rebuilding metrics.
No Torch, model, autograd, optimizer, parameter, worker or VM call is made.
"""
import hashlib
import importlib
import importlib.abc
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = '7f4546d4d911bea7100f42870514f08a42e349ec524a2b9795f8f3c7f55dcf60'
PARENT_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
OUTPUT = 'outputs/u64_png_checked_step_v1'
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
VIEWS = ('raw', 'native_png')
FIELDS = ('mae', 'adjacent_rgb_error')
SCALAR_ARITHMETIC_BOUND = 5e-7
METRIC_ARITHMETIC_BOUND = 5e-12
SHAPES = {'raw.npy': (256, 256, 3), 'unclipped.npy': (1, 3, 256, 256),
          'native.npy': (1, 3, 512, 512), 'logits.npy': (1, 256, 1024), 'LQ.npy': (1, 256, 16, 16)}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def binding(raw):
    return {'bytes': len(raw), 'sha256': digest(raw)}


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, child in items:
            need(key not in value, 'Duplicate JSON key')
            value[key] = child
        return value
    def bad(value):
        raise ValueError('Nonfinite JSON: ' + value)
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    def finite(value, depth=0):
        need(depth <= 45, 'Bounded JSON depth')
        if type(value) is float:
            need(math.isfinite(value), 'Finite JSON scalar')
        elif type(value) in (dict, list):
            for child in value.values() if type(value) is dict else value:
                finite(child, depth + 1)
    finite(result)
    return result


def relative(name):
    need(type(name) is str and name and '\\' not in name and ':' not in name, 'Canonical relative name')
    p = PurePosixPath(name)
    need(not p.is_absolute() and p.as_posix() == name and not any(v in ('', '.', '..') for v in p.parts), 'Confined path')
    return name


class Reader:
    def __init__(self):
        self.consumed = {}
    def read(self, path, expected=None, maximum=8 * 1024**2):
        path = path.absolute()
        need(path.is_relative_to(ROOT), 'Read escaped workspace')
        for parent in (path.parent, *path.parent.parents):
            status = parent.lstat()
            need(stat.S_ISDIR(status.st_mode) and not stat.S_ISLNK(status.st_mode)
                 and not getattr(status, 'st_file_attributes', 0) & 1024, 'Ordinary directory ancestry')
        before = path.lstat()
        need(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode) and before.st_nlink == 1
             and not getattr(before, 'st_file_attributes', 0) & 1024 and 0 <= before.st_size <= maximum,
             'Bounded single-link regular file: ' + str(path))
        raw = path.read_bytes()
        after = path.stat()
        need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'File changed during read')
        actual = binding(raw)
        if expected is not None:
            need(actual == {key: expected[key] for key in ('bytes', 'sha256')}, 'Exact consumed pin differs: ' + str(path))
        name = path.relative_to(ROOT).as_posix()
        need(name not in self.consumed or self.consumed[name] == actual, 'Consumed file changed')
        self.consumed[name] = actual
        return raw
    def after(self):
        before = dict(self.consumed)
        for name, pin in before.items():
            self.read(ROOT / name, pin)
        need(before == self.consumed, 'Consumed namespace unchanged')
        return [{'path': name, **value} for name, value in sorted(before.items())]


def npy(raw, leaf):
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    need(version in ((1, 0), (2, 0)), 'NPY1/2 only')
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream, max_header_size=4096)
    need(shape == SHAPES[leaf] and dtype == np.dtype('float32') and not dtype.hasobject and not fortran,
         'Exact C-order finite FP32 array schema')
    need(stream.tell() + math.prod(shape) * 4 == len(raw), 'Exact NPY payload extent')
    array = np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(shape)
    need(np.isfinite(array).all(), 'Finite actual saved array')
    if leaf == 'raw.npy':
        need(array.min() >= 0 and array.max() <= 1, 'Delivered RAW [0,1]')
    return array


def png(raw):
    need(len(raw) <= 256 * 1024, 'PNG256KiB cap')
    with Image.open(io.BytesIO(raw)) as image:
        need(image.format == 'PNG' and image.mode == 'RGB' and image.size == (256, 256)
             and getattr(image, 'n_frames', 1) == 1, 'Single RGB256 PNG')
        return np.asarray(image).copy()


def normalized(value):
    return value.astype(np.float32) / np.float32(255)


def floor_png(raw):
    return (raw * np.float32(255)).astype(np.uint8)


def region_metrics(raw, target):
    if target is None:
        return None
    residual = raw.astype(np.float64) - target.astype(np.float64)
    dh = np.abs(residual[:, 1:] - residual[:, :-1])
    dv = np.abs(residual[1:] - residual[:-1])
    result = {'whole': {'mae': float(np.abs(residual).mean()),
                        'adjacent_rgb_error': float(.5 * (dh.mean() + dv.mean()))}}
    center = np.zeros((256, 256), dtype=bool)
    center[64:192, 64:192] = True
    for name, mask in (('central128', center), ('outside', ~center)):
        horizontal, vertical = mask[:, 1:] & mask[:, :-1], mask[1:] & mask[:-1]
        result[name] = {'mae': float(np.abs(residual[mask]).mean()),
                        'adjacent_rgb_error': float((dh[horizontal].sum() + dv[vertical].sum()) /
                                                   (3 * int(horizontal.sum() + vertical.sum())))}
    return result


def blocks(raw):
    return (raw.astype(np.float64) @ np.asarray((.2126, .7152, .0722), np.float64)).reshape(16, 16, 16, 16).mean(axis=(1, 3))


def metrics(raw, quantized, target, control=None, teacher=None):
    result = {'metrics': {}, 'regional_metrics': {}}
    for view, value in (('raw', raw), ('native_png', normalized(quantized))):
        regions = region_metrics(value, target)
        result['regional_metrics'][view] = regions
        result['metrics'][view] = dict.fromkeys(FIELDS, None) if regions is None else dict(regions['whole'])
        if control is not None:
            reference = control[0] if view == 'raw' else normalized(control[1])
            result.setdefault('U96_behavior_distance', {})[view] = float(np.abs(value.astype(np.float64) - reference.astype(np.float64)).mean())
            result.setdefault('U96_illumination_distance', {})[view] = float(np.abs(blocks(value) - blocks(reference)).mean())
        if teacher is not None:
            reference = teacher[0] if view == 'raw' else normalized(teacher[1])
            result.setdefault('behavior_distance', {})[view] = float(np.abs(value.astype(np.float64) - reference.astype(np.float64)).mean())
    if control is None:
        result['behavior_movement_reference_available'] = False
    return result


def gate_scalars(raw, target, role):
    residual = raw.astype(np.float64) - target.astype(np.float64)
    if role == ROLES[0]:
        residual = residual[64:192, 64:192]
    return {'gate_mae': float(np.abs(residual).mean()), 'gate_adjacent': float(.5 * (
        np.abs(residual[:, 1:] - residual[:, :-1]).mean() + np.abs(residual[1:] - residual[:-1]).mean()))}


def ordered(values):
    result = 0.
    for value in values:
        result = result + float(value)
    return result


def cohort_means(bank, monitors):
    result = {view: {} for view in VIEWS}
    for role in ROLES:
        ids = sorted(row['id'] for row in monitors if row['role'] == role)
        need(len(ids) == 8, 'Eight adaptive TRAIN monitors per role')
        region = 'central128' if role == ROLES[0] else 'whole'
        for view in VIEWS:
            result[view][role] = {field: ordered(bank[cid]['regional_metrics'][view][region][field] for cid in ids) / 8 for field in FIELDS}
    return result


def added_gate(now, original, prior, terminal):
    checks, strict = [], []
    for view in VIEWS:
        for role in ROLES:
            for field in FIELDS:
                value = now[view][role][field]
                for label, baseline in (('original_u64', original), ('prior_state', prior)):
                    reference = baseline[view][role][field]
                    checks.append({'view': view, 'role': role, 'field': field, 'reference': label,
                                   'baseline': reference, 'current': value, 'delta': value - reference, 'pass': value <= reference})
        if terminal:
            for role in ('clear', ROLES[-1]):
                strict.append(now[view][role]['mae'] < original[view][role]['mae'])
    return checks, all(row['pass'] for row in checks) and all(strict)


def individual_changes(current, original, prior, specs):
    changes = []
    for cid in sorted(current):
        reference = specs[cid]
        region = 'central128' if reference['role'] == ROLES[0] else 'whole'
        for view in VIEWS:
            for field in FIELDS:
                values = [bank[cid]['regional_metrics'][view][region][field] if reference.get('target') is not None
                          else bank[cid]['metrics'][view][field] for bank in (current, original, prior)]
                now, start, preceding = values
                if now is None:
                    need(start is preceding is None, 'Null photographic targets remain unavailable')
                    changes.append({'id': cid, 'role': reference['role'], 'view': view, 'field': field,
                                    'region': region, 'available': False, 'reason': 'No aligned photographic target'})
                else:
                    changes.append({'id': cid, 'role': reference['role'], 'view': view, 'field': field,
                                    'region': region, 'available': True, 'current': now, 'original_u64': start,
                                    'prior_state': preceding, 'delta_original': now - start, 'delta_prior': now - preceding,
                                    'regresses_original': now > start, 'regresses_prior': now > preceding})
    return changes


class Checks:
    def __init__(self):
        self.rows = []
    def check(self, name, ok, detail=None):
        self.rows.append({'name': name, 'pass': bool(ok), 'detail': detail})
    def same(self, name, actual, expected, bound=METRIC_ARITHMETIC_BOUND):
        leaves, maximum, failures = 0, 0., []
        def walk(a, b, path):
            nonlocal leaves, maximum
            if type(b) is dict:
                if type(a) is not dict or set(a) != set(b):
                    failures.append(path + ':keys'); return
                for key in b:
                    walk(a[key], b[key], path + '/' + key)
            elif type(b) is list:
                if type(a) is not list or len(a) != len(b):
                    failures.append(path + ':length'); return
                for i, item in enumerate(b):
                    walk(a[i], item, path + '/' + str(i))
            elif type(b) in (int, float) and type(b) is not bool:
                leaves += 1
                if type(a) not in (int, float) or type(a) is bool:
                    failures.append(path + ':number'); return
                delta = abs(float(a) - float(b))
                maximum = max(maximum, delta)
                if delta > bound:
                    failures.append(path + ':numeric')
            elif type(a) != type(b) or a != b:
                failures.append(path + ':value')
        walk(actual, expected, '')
        self.check(name, not failures, {'numeric_leaves': leaves, 'max_absolute_difference': maximum,
                                      'arithmetic_bound_only': bound, 'failures': failures[:12]})
        return maximum


class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        need(fullname.split('.')[0] not in {'torch', 'torchvision', 'tensorflow', 'jax', 'scipy'}, 'Framework import forbidden')
        return None


def fixtures():
    checks = []
    def check(name, okay):
        need(okay, 'Independent fixture failed: ' + name)
        checks.append({'name': name, 'pass': True})
    z = np.zeros((256, 256, 3), np.float32)
    half = np.full_like(z, .5)
    m = region_metrics(half, z)
    check('constant_residual_half_MAE_zero_adjacent', all(v == {'mae': .5, 'adjacent_rgb_error': 0.} for v in m.values()))
    roi = z.copy(); roi[64:192, 64:192] = 1
    m = region_metrics(roi, z)
    check('half_open_region_excludes_cross_seam_edges', m['central128']['adjacent_rgb_error'] == m['outside']['adjacent_rgb_error'] == 0 and m['whole']['adjacent_rgb_error'] > 0)
    check('floor_not_round', floor_png(np.asarray([.5], np.float32))[0] == 127)
    value = {view: {role: dict.fromkeys(FIELDS, .5) for role in ROLES} for view in VIEWS}
    changed = json.loads(json.dumps(value)); changed['native_png'][ROLES[0]]['adjacent_rgb_error'] += 1e-12
    rows, passed = added_gate(changed, value, value, False)
    check('zero_quality_tolerance_small_increase_fails', not passed and sum(not v['pass'] for v in rows) == 2)
    return checks


def optimization_means(bank, optimization):
    need(set(bank) == {row['id'] for row in optimization}, 'Exact complete64 optimization bank')
    result = {view: {} for view in VIEWS}
    for role in ROLES:
        ids = sorted(row['id'] for row in optimization if row['role'] == role)
        need(len(ids) == 16, 'Exactly16 optimization photographs per role')
        region = 'central128' if role == ROLES[0] else 'whole'
        for view in VIEWS:
            result[view][role] = {field: math.fsum(
                bank[cid]['regional_metrics'][view][region][field] for cid in ids) / 16.
                for field in FIELDS}
    return result


def trial_checks(now, baseline, index, cap):
    rows, strict = [], []
    for view in VIEWS:
        for role in ROLES:
            for field in FIELDS:
                value, original = now[view][role][field], baseline[view][role][field]
                rows.append({'view': view, 'role': role, 'field': field, 'original': original,
                    'current': value, 'delta': value - original, 'pass': value <= original})
    for role in ROLES[:2]:
        value, original = now['native_png'][role]['adjacent_rgb_error'], baseline['native_png'][role]['adjacent_rgb_error']
        strict.append({'view': 'native_png', 'role': role, 'field': 'adjacent_rgb_error',
            'original': original, 'current': value, 'delta': value - original, 'pass': value < original})
    return {'candidate_index': index, 'maximum_absolute_coordinate_bound': cap,
        'eligible': all(row['pass'] for row in rows + strict), 'all16_nonincrease_checks': rows,
        'both_blur_PNG_strict_checks': strict, 'means': now}


def optimization_changes(current, original, optimization):
    rows = []
    refs = {row['id']: row for row in optimization}
    for cid in sorted(current):
        role = refs[cid]['role']; region = 'central128' if role == ROLES[0] else 'whole'
        for view in VIEWS:
            for field in FIELDS:
                value = current[cid]['regional_metrics'][view][region][field]
                start = original[cid]['regional_metrics'][view][region][field]
                rows.append({'id': cid, 'role': role, 'region': region, 'view': view, 'field': field,
                    'original_u64': start, 'current': value, 'delta_original': value - start,
                    'regresses_original': value > start})
    return rows


def regression_summary(changes):
    available = [row for row in changes if row.get('available', True)]
    increased = [row for row in available if row['regresses_original']]
    return {'rows': len(changes), 'available_rows': len(available), 'increased_rows': len(increased),
        'increased_case_ids': len({row['id'] for row in increased}),
        'by_role_view_field': [{'role': role, 'view': view, 'field': field,
            'increased': sum(row['role'] == role and row['view'] == view and row['field'] == field for row in increased)}
            for role in ROLES for view in VIEWS for field in FIELDS]}


def run():
    sys.meta_path.insert(0, BlockFrameworks())
    r, c = Reader(), Checks()
    test_rows = fixtures()
    baseline_fixture = {view: {role: dict.fromkeys(FIELDS, .5) for role in ROLES} for view in VIEWS}
    changed_fixture = strict_json(json.dumps(baseline_fixture))
    for role in ROLES[:2]:
        changed_fixture['native_png'][role]['adjacent_rgb_error'] -= .01
    need(trial_checks(changed_fixture, baseline_fixture, 0, 2e-5)['eligible'], 'Fixture eligible actual endpoint arithmetic')
    changed_fixture['raw']['clear']['mae'] += 1e-12
    need(not trial_checks(changed_fixture, baseline_fixture, 0, 2e-5)['eligible'], 'Fixture exact zero quality tolerance')
    test_rows.append({'name': 'all16_nonincrease_and_two_strict_blur_zero_tolerance', 'pass': True})
    r.read(Path(__file__).resolve())
    audit_root = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_return_audit'
    inventory = strict_json(r.read(audit_root / 'inventory_v1.json'))
    need(inventory['complete'] is True and inventory['protocol_sha256'] == PROTOCOL_SHA, 'Complete bound return inventory')
    evidence = Path(inventory['evidence_root']).resolve(strict=True)
    need(evidence == (audit_root / 'extracted').resolve(strict=True), 'Exact bounded extraction root')
    stage = ROOT / 'outputs/cctv_codeformer_u64_png_checked_step_v1_vm'
    parent = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
    protocol_raw = r.read(stage / 'protocol.json'); need(digest(protocol_raw) == PROTOCOL_SHA, 'Exact sealed PNG-checked protocol')
    p = strict_json(protocol_raw)
    parent_raw = r.read(parent / 'protocol.json'); need(digest(parent_raw) == PARENT_SHA, 'Exact immutable parent protocol')
    oldp = strict_json(parent_raw)
    standard_raw = r.read(parent / oldp['parent_protocol']['path'], oldp['parent_protocol'])
    source_names = ('research/__init__.py', 'research/codeformer_u64_cleananchors_continuation_v1/__init__.py',
        'research/codeformer_u64_cleananchors_continuation_v1/metrics.py',
        'research/codeformer_u64_cleananchors_continuation_v1/contract.py',
        'research/codeformer_u96_spatial_feasibility_v1/__init__.py',
        'research/codeformer_u96_spatial_feasibility_v1/metrics.py')
    for name in source_names:
        r.read(parent / name, oldp['files'][name])
    own_sources = ('research/codeformer_u64_png_checked_step_v1/selection.py',
        'research/codeformer_u64_png_checked_step_v1/quality.py',
        'research/codeformer_u64_png_checked_step_v1/worker.py',
        'research/codeformer_u64_gate_aligned_pilot_v1/__init__.py',
        'research/codeformer_u64_gate_aligned_pilot_v1/quality.py')
    for name in own_sources:
        r.read(stage / name, p['files'][name])
    sys.path.insert(0, str(parent))
    imported = importlib.import_module('research.codeformer_u64_cleananchors_continuation_v1.metrics')
    for name in source_names:
        module = sys.modules.get(name.removesuffix('/__init__.py').removesuffix('.py').replace('/', '.'))
        need(module is not None and Path(module.__file__).resolve() == parent / name, 'Exact pinned pure helper origin')
    import research
    research.__path__.append(str(stage / 'research'))
    def pure_module(name, relative):
        spec = importlib.util.spec_from_file_location(name, stage / relative)
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
        return value
    selector = pure_module('sealed_checked_selector_saved_pixel_audit', own_sources[0])
    quality = pure_module('sealed_checked_quality_saved_pixel_audit', own_sources[1])
    index = inventory['extracted_files']
    def returned(name):
        relative(name)
        need(index[name] == inventory['files'][name], 'Extracted/export-index join')
        return r.read(evidence / name, index[name])
    def source(row, leaf):
        name = relative(row['path'])
        need(oldp['files'][name] == {key: row[key] for key in ('bytes', 'sha256')}, 'Exact parent static binding')
        data = r.read(parent / name, row)
        return png(data) if leaf == 'png' else npy(data, 'raw.npy')
    pair_cache, png_cache = {}, {}
    def rgb(row):
        if row['path'] not in png_cache:
            png_cache[row['path']] = source(row, 'png')
        return png_cache[row['path']]
    def pair(row):
        key = row['raw']['path']
        if key not in pair_cache:
            pair_cache[key] = source(row['raw'], 'raw'), rgb(row['png'])
        return pair_cache[key]
    selection = p['selection']
    specs = {row['id']: row for row in selection['optimization'] + selection['monitors'] + selection['guards']}
    controls = {row['id']: row for row in oldp['controls']}
    parity = {row['id']: row for row in oldp['u64_parity']}
    quality_ids = {row['id'] for row in selection['monitors'] + selection['guards']}
    opt_ids = {row['id'] for row in selection['optimization']}
    need(len(quality_ids) == len(controls) == 56 and len(opt_ids) == 64 and not quality_ids & opt_ids,
         'Exact disjoint56 quality and64 optimization cases')
    audit_bytes = r.read(parent / oldp['historical_pixel_audit']['path'], oldp['historical_pixel_audit'])
    frozen = {}
    for cid, reference in controls.items():
        row = specs[cid]; control = pair(reference)
        c.check('frozen_U96_floor_' + cid, np.array_equal(floor_png(control[0]), control[1]))
        target = None if row.get('target') is None else normalized(rgb(row['target']))
        teacher = None if row.get('behavior_teacher') is None else pair(row['behavior_teacher'])
        measured = metrics(*control, target, control, teacher)
        options = {} if teacher is None else {'teacher_raw': teacher[0], 'teacher_png': teacher[1]}
        helper_value = imported.record_metrics(*control, target, row, np=np, frozen_U96_raw=control[0], frozen_U96_png=control[1], **options)
        c.same('independent_vs_pinned_frozen_metric_' + cid, measured, helper_value)
        frozen[cid] = {**{name: row[name] for name in ('id', 'reference_id', 'role', 'split')}, **measured}
    c.same('all56_frozen_U96_saved_metrics', frozen,
           strict_json(returned(OUTPUT + '/frozen_U96_control_metrics.json'))['records'])
    records, pixel_rows, first = {}, [], {}
    initial_quality, initial_opt, selected_quality = {}, {}, {}
    trial_banks = [dict() for _ in range(4)]
    available = {name for name in index if re.fullmatch(OUTPUT + r'/calls/\d{4}/record.json', name)}
    application = 0; head_by_application = {0: '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'}
    stock_saved, parity_count, repeat_count = 0, 0, 0
    for call in p['plan']:
        if call['kind'] == 'parameter_application':
            application += 1
            commit_name = OUTPUT + '/applications/a' + str(application).zfill(4) + '/commit.json'
            if commit_name in index:
                commit = strict_json(returned(commit_name))
                c.check('application_call_metadata_' + str(application), commit['complete'] is True and commit['call'] == call)
                head_by_application[application] = commit['actual_after']
            continue
        name = OUTPUT + '/calls/' + str(call['slot']).zfill(4) + '/record.json'
        if name not in available:
            continue
        saved_bytes = returned(name); saved = strict_json(saved_bytes)
        row, cid, slot = specs[call['id']], call['id'], call['slot']
        c.check('record_exact_plan_source_state_join_' + str(slot), saved['complete'] is True
            and saved['call'] == call and call['gradient_queries'] == []
            and all(saved[key] == row.get(key) for key in ('id', 'reference_id', 'role', 'split', 'input', 'target'))
            and saved['parameter_application_index'] == application
            and saved['head_before_sha256'] == head_by_application.get(application))
        observed = rgb(row['input'])
        c.check('actual_input_RGB8_hash_' + str(slot), digest(observed.tobytes()) == row['input_RGB8_sha256'])
        target_u8 = None if row.get('target') is None else rgb(row['target'])
        if target_u8 is not None:
            c.check('actual_target_RGB8_hash_' + str(slot), digest(target_u8.tobytes()) == row['target_RGB8_sha256'])
        target = None if target_u8 is None else normalized(target_u8)
        arrays = {}
        for field, leaf in (('raw', 'raw.npy'), ('png', 'result.png')):
            file_pin = saved['outputs'][field]
            need(file_pin['path'] == name.rsplit('/', 1)[0] + '/' + leaf, 'Exact image slot/path')
            data = returned(file_pin['path'])
            c.same('actual_image_file_pin_' + str(slot) + '_' + field, binding(data),
                   {key: file_pin[key] for key in ('bytes', 'sha256')}, 0)
            arrays[field] = png(data) if field == 'png' else npy(data, leaf)
        c.check('RAW_floor_PNG_exact_' + str(slot), np.array_equal(floor_png(arrays['raw']), arrays['png']))
        control = None if cid not in controls else pair(controls[cid])
        teacher = None if row.get('behavior_teacher') is None else pair(row['behavior_teacher'])
        measured = metrics(arrays['raw'], arrays['png'], target, control, teacher)
        c.same('all_independent_saved_metrics_' + str(slot), measured, {key: saved[key] for key in measured})
        options = {} if teacher is None else {'teacher_raw': teacher[0], 'teacher_png': teacher[1]}
        helper_value = (imported.record_metrics(arrays['raw'], arrays['png'], target, row, np=np,
            frozen_U96_raw=control[0], frozen_U96_png=control[1], **options) if control is not None else
            imported.record_training_metrics(arrays['raw'], arrays['png'], target, row, np=np))
        c.same('independent_vs_pinned_actual_metric_' + str(slot), measured, helper_value)
        for field in ('native', 'logits', 'LQ'):
            stock = saved['stock'][field]
            c.check('stock_schema_declaration_' + str(slot) + field,
                stock['shape'] == list(SHAPES[field + '.npy']) and stock['dtype'] == 'float32'
                and re.fullmatch('[0-9a-f]{64}', stock['bytes_sha256']) is not None)
            if 'file' in stock:
                file_pin = stock['file']; data = returned(file_pin['path']); value = npy(data, field + '.npy')
                c.same('stock_saved_file_pin_' + str(slot) + field, binding(data), {key: file_pin[key] for key in ('bytes', 'sha256')}, 0)
                c.check('stock_actual_value_hash_' + str(slot) + field, digest(value.tobytes()) == stock['bytes_sha256'])
                arrays[field] = value; stock_saved += 1
        if 'native' in arrays:
            native = (arrays['native'] + np.float32(1)) / np.float32(2)
            def resize(value):
                return (value[:, :, ::2, ::2] + value[:, :, ::2, 1::2]
                        + value[:, :, 1::2, ::2] + value[:, :, 1::2, 1::2]) * np.float32(.25)
            delivered = resize(np.clip(native, 0, 1))[0].transpose(1, 2, 0)
            unpin = saved['outputs']['unclipped']; unbytes = returned(unpin['path']); arrays['unclipped'] = npy(unbytes, 'unclipped.npy')
            c.same('unclipped_saved_file_pin_' + str(slot), binding(unbytes), {key: unpin[key] for key in ('bytes', 'sha256')}, 0)
            c.check('native_clamp_before_bilinear256_' + str(slot), float(np.abs(delivered - arrays['raw']).max()) <= np.finfo(np.float32).eps)
            c.check('native_unclipped_bilinear256_' + str(slot), float(np.abs(resize(native) - arrays['unclipped']).max()) <= np.finfo(np.float32).eps)
        recomputed = {**saved, **measured, 'receipt': {'path': name, **index[name]}}
        phase = call['phase']
        if phase == 'initial_u64_quality_bank':
            initial_quality[cid] = recomputed
            if cid in parity:
                witness = pair(parity[cid]); parity_count += 1
                c.check('historical_u64_exact_RAW_PNG_' + cid,
                    np.array_equal(arrays['raw'], witness[0]) and np.array_equal(arrays['png'], witness[1]))
                first[cid] = {key: value.copy() for key, value in arrays.items()}
        elif phase == 'deterministic_same_u64_repeat':
            repeat_count += 1
            c.check('full_six_saved_output_repeat_' + cid,
                set(arrays) == set(first[cid]) and all(np.array_equal(value, first[cid][key]) for key, value in arrays.items()))
        elif phase == 'initial_optimization_bank':
            initial_opt[cid] = recomputed
        elif phase == 'trial_optimization_bank':
            trial_banks[call['trial_index']][cid] = recomputed
        else:
            need(phase == 'selected_quality_bank', 'Known returned phase only')
            selected_quality[cid] = recomputed
        records[slot] = recomputed
        pixel_rows.append({'slot': slot, 'parameter_application_index': application, 'id': cid,
            'role': row['role'], 'phase': phase, 'trial_index': call['trial_index'],
            'RAW_floor_PNG_exact': True, 'outputs': saved['outputs'], 'metrics': measured,
            'RAW_value_sha256': digest(arrays['raw'].tobytes()), 'PNG_RGB8_sha256': digest(arrays['png'].tobytes())})
    c.check('all_actual_saved_records_consumed', len(records) == len(available) and len(records) <= 436)
    c.check('all4_historical_and4_repeat_witnesses', parity_count == repeat_count == 4 and stock_saved == 24)
    c.check('exact_initial56_and_original64', set(initial_quality) == quality_ids and set(initial_opt) == opt_ids)
    c.check('all_four_complete64_fixed_trial_banks', all(set(bank) == opt_ids for bank in trial_banks))
    initial_saved = strict_json(returned(OUTPUT + '/banks/initial_quality.json'))
    c.same('initial56_actual_output_records', initial_quality, initial_saved['records'])
    initial_gate = imported.evaluate_bank(initial_quality, initial_quality, frozen, standard_raw,
        phase='initializer', saved_audit_bytes=audit_bytes)
    c.same('initial56_unchanged_inherited_gate', initial_gate, initial_saved['gate'])
    c.check('exact_original_initial_quality_pass', initial_gate['pass'])
    def selector_bank(bank):
        result = {}
        for row in selection['optimization']:
            cid = row['id']; region = 'central128' if row['role'] == ROLES[0] else 'whole'
            result[cid] = {**{key: row[key] for key in selector.PROVENANCE if key != 'metric_region'},
                'metric_region': region, 'metrics': {view: {
                    field: bank[cid]['regional_metrics'][view][region][field] for field in FIELDS} for view in VIEWS}}
        return result
    opt_saved = strict_json(returned(OUTPUT + '/banks/initial_optimization.json'))
    c.same('initial64_actual_records', initial_opt, opt_saved['records'])
    c.same('initial64_selector_actual_provenance_metrics', selector_bank(initial_opt), opt_saved['selector_bank'])
    baseline = optimization_means(initial_opt, selection['optimization'])
    trial_results, individual, trial_summaries = [], {}, []
    caps = (2e-5, 1.5e-5, 1e-5, 5e-6)
    for number, (cap, bank) in enumerate(zip(caps, trial_banks)):
        saved = strict_json(returned(OUTPUT + '/banks/trial_' + str(number) + '.json'))
        c.same('trial_actual64_records_' + str(number), bank, saved['records'])
        c.same('trial_selector64_provenance_metrics_' + str(number), selector_bank(bank), saved['selector_bank'])
        c.check('fixed_trial_cap_' + str(number), saved['fixed_candidate_bound'] == cap)
        means = optimization_means(bank, selection['optimization'])
        trial_results.append(trial_checks(means, baseline, number, cap))
        changes = optimization_changes(bank, initial_opt, selection['optimization'])
        individual[str(number)] = changes
        c.same('trial_independent256_individual_changes_' + str(number), changes, saved['individual_changes'])
        c.check('trial_individual256_rows_' + str(number), len(changes) == 256)
        trial_summaries.append({'candidate_index': number, 'cap': cap, 'eligible': trial_results[-1]['eligible'],
            'failed_nonincrease': [row for row in trial_results[-1]['all16_nonincrease_checks'] if not row['pass']],
            'failed_strict_blur': [row for row in trial_results[-1]['both_blur_PNG_strict_checks'] if not row['pass']],
            'individual_regressions': regression_summary(changes)})
    chosen = next((row['candidate_index'] for row in trial_results if row['eligible']), None)
    actual_selection = strict_json(returned(OUTPUT + '/selection.json'))
    c.same('independent_all_four_eligibility_results', trial_results, actual_selection['trial_results'])
    c.same('independent_original64_endpoint_means', baseline, actual_selection['original_means'])
    c.check('first_eligible_fixed_order_no_monitor_leak', actual_selection['complete'] is True
        and actual_selection['selected_index'] == chosen and actual_selection['candidate_registry'] == list(caps)
        and actual_selection['no_eligible_candidate'] == (chosen is None)
        and actual_selection['all_four_trials_evaluated'] is True
        and actual_selection['all_four_original_restores_completed'] is True
        and actual_selection['only_optimization_records_supplied'] is True
        and actual_selection['monitor_records_supplied'] == 0 and actual_selection['quality_tolerance'] == 0
        and actual_selection['post_monitor_fallback_authorized'] is False)
    metadata = [{**row, 'metric_region': 'central128' if row['role'] == ROLES[0] else 'whole'} for row in selection['optimization']]
    helper_selection = selector.select_candidate(selector_bank(initial_opt), [
        {'maximum_absolute_coordinate_bound': cap, 'bank': selector_bank(trial_banks[i])}
        for i, cap in enumerate(caps)], metadata)
    c.same('pinned_selector_crosscheck_after_independent_means', helper_selection,
           {key: actual_selection[key] for key in helper_selection})
    quality_result = None; final_quality = {'attempted': bool(selected_quality), 'pass': False}
    if chosen is None:
        c.check('noeligible_has_no_commit_or_fake_final_bank', not selected_quality
            and OUTPUT + '/banks/selected_quality.json' not in index and 9 not in head_by_application
            and len(records) == 380)
    else:
        c.check('selected_candidate_has_exact_final56', set(selected_quality) == quality_ids and len(records) == 436)
        saved = strict_json(returned(OUTPUT + '/banks/selected_quality.json'))
        c.same('selected56_actual_output_records', selected_quality, saved['records'])
        inherited = imported.evaluate_bank(selected_quality, initial_quality, frozen, standard_raw,
            phase='u4', saved_audit_bytes=audit_bytes)
        c.same('selected56_all_inherited_original_guards', inherited, saved['inherited'])
        means = cohort_means(selected_quality, selection['monitors'])
        original_means = cohort_means(initial_quality, selection['monitors'])
        c.same('independent_selected_monitor16_means', means, saved['current_means'])
        c.same('independent_original_monitor16_means', original_means, saved['original_means'])
        own_checks, own_pass = added_gate(means, original_means, original_means, True)
        c.same('independent32_original_prior_nonincrease_checks', own_checks, saved['gate']['role_mean_nonincrease_checks'])
        strict4 = [{'view': view, 'role': role, 'field': 'mae',
            'baseline_original_u64': original_means[view][role]['mae'], 'current': means[view][role]['mae'],
            'delta': means[view][role]['mae'] - original_means[view][role]['mae'],
            'pass': means[view][role]['mae'] < original_means[view][role]['mae']}
            for view in VIEWS for role in ('clear', ROLES[-1])]
        c.same('independent4_strict_clear_dark_MAE', strict4, saved['gate']['terminal_clear_lowlight_strict_MAE_checks'])
        quality_result = quality.evaluate_quality(means, original_means, original_means, True, inherited)
        c.same('pinned_terminal_quality_wrapper_after_independent_math', quality_result, saved['gate'])
        c.check('independent_complete_quality_pass_boolean',
            quality_result['pass'] == (inherited['pass'] and inherited['original_terminal']['pass'] and own_pass))
        changes = individual_changes(selected_quality, initial_quality, initial_quality, specs)
        individual['selected_quality'] = changes
        c.same('independent224_monitor_guard_change_rows', changes, saved['individual_changes'])
        c.check('individual224_quality_rows', len(changes) == 224)
        final_quality = {'attempted': True, 'pass': quality_result['pass'], 'inherited_pass': inherited['pass'],
            'original_terminal_pass': inherited['original_terminal']['pass'], 'strict4': strict4,
            'failed_role_mean_checks': [row for row in own_checks if not row['pass']],
            'current_means': means, 'original_means': original_means,
            'individual_regressions': regression_summary(changes),
            'monitor_fallback_authorized': False, 'no_other_trial_monitor_quality_inferred': True}
    status = strict_json(returned(OUTPUT + '/status.json'))
    failure = strict_json(returned(OUTPUT + '/failure.json')) if OUTPUT + '/failure.json' in index else None
    success = strict_json(returned(OUTPUT + '/result.json')) if OUTPUT + '/result.json' in index else None
    expected_applications = 8 if chosen is None else 9
    c.check('saved_actual_counts_zero_new_queries', status['protocol_sha256'] == PROTOCOL_SHA
        and status['counters']['full_forward_attempts'] == status['counters']['full_forward_returns'] == len(records)
        and status['counters']['gradient_query_attempts'] == status['counters']['gradient_query_returns'] == 0
        and status['counters']['parameter_application_attempts'] == status['counters']['parameter_applications'] == expected_applications
        and status['counters']['trial_parameter_applications'] == status['counters']['original_restoration_applications'] == 4
        and status['counters']['selected_parameter_commits'] == int(chosen is not None)
        and status['operation_outcome_unknown'] is False and status['unknown_reservations'] == [])
    expected_quality_pass = chosen is not None and final_quality['pass']
    if expected_quality_pass:
        c.check('finite_success_matches_actual_quality', success is not None and success['complete'] is True and failure is None)
    else:
        c.check('authoritative_quality_stop_no_success', success is None and failure is not None
            and failure['quality_stop'] is True and failure['error_type'] == 'QualityStop'
            and failure['active_application'] == expected_applications and failure['operation_outcome_unknown'] is False
            and failure['counters'] == status['counters'])
    c.check('no_neural_framework_loaded', not any(name.split('.')[0] in {'torch', 'torchvision', 'tensorflow', 'jax', 'scipy'} for name in sys.modules))
    consumed = r.after()
    failed = [row for row in c.rows if not row['pass']]
    result = {'complete': True, 'saved_pixel_metric_selection_consistency_pass': not failed,
        'independent_saved_pixel_integrity_pass': not failed, 'protocol_sha256': PROTOCOL_SHA,
        'counts': {'actual_saved_forwards': len(records), 'exact_RAW_floor_PNG_pairs': len(pixel_rows),
            'historical_u64_RAW_PNG_parities': parity_count, 'full_six_output_repeats': repeat_count,
            'retained_stock_triples': stock_saved // 3, 'original_optimization_cases': len(initial_opt),
            'trial_optimization_cases': [len(bank) for bank in trial_banks],
            'original_quality_cases': len(initial_quality), 'selected_quality_cases': len(selected_quality),
            'trial_individual_change_rows': 4 * 256, 'selected_quality_individual_change_rows': len(individual.get('selected_quality', [])),
            'new_gradient_queries': 0, 'parameter_applications_declared_and_metadata_bound': expected_applications,
            'consumed_files_before_after': len(consumed)},
        'checks': len(c.rows), 'failed_checks': failed, 'quality_tolerance': 0,
        'metric_arithmetic_readback_bound_only': METRIC_ARITHMETIC_BOUND,
        'actual_trial_eligibility': trial_summaries, 'optimization_selected_candidate_index': chosen,
        'optimization_selected_cap': None if chosen is None else caps[chosen],
        'all_four_complete_before_first_eligible_selection': True, 'monitor_selection_or_fallback_authorized': False,
        'selected_quality': final_quality, 'pilot_quality_gates_pass': expected_quality_pass,
        'authoritative_quality_stop': not expected_quality_pass,
        'noeligible_original_restored_stop': chosen is None,
        'finite_gain_or_success_for_other_candidates_inferred': False,
        'terminal_clear_dark_strict_gains_sufficient_for_full_quality': False,
        'all_consumed_hashes_unchanged_after': True, 'array_fixtures_pass': True,
        'independent_metric_math_crosschecked_to_pinned_pure_helpers': True,
        'inherited_policy_reused_exact_pinned_source_after_independent_array_math': True,
        'unretained_native_arrays_for_other_calls_not_reconstructed': True,
        'original_u64_retained': True, 'app_incumbent_unchanged': True, 'model_qualified': False,
        'no_Torch_model_autograd_optimizer_parameter_application_VM_browser_calls': True,
        'scope': 'All actual saved output bytes, exact floor quantization, independent NumPy photographic/behavior metrics, original56 plus selected56 quality if attempted, all original64/four64 optimization banks and zero-tolerance fixed selection. Complete575/head-state correctness and journal execution are separate audits; no neural forward, new gradient, parameter application or useful-restoration/native/DEV/FINAL claim is made.'}
    out = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_return_pixels'
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {'receipt_v1.json': result, 'pixel_rows_v1.json': {'complete': True, 'rows': pixel_rows},
        'selection_readback_v1.json': {'complete': True, 'original_means': baseline, 'trial_results': trial_results,
            'selected_index': chosen, 'all64_actual_selector_inputs_bound': True, 'quality_tolerance': 0},
        'quality_readback_v1.json': {'complete': True, 'initial_gate': initial_gate,
            'selected_gate': quality_result, 'selected_quality': final_quality},
        'individual_changes_v1.json': {'complete': True, 'banks': individual},
        'checks_v1.json': {'complete': True, 'rows': c.rows},
        'source_array_tests_v1.json': {'complete': True, 'all_tests_pass': True, 'fixtures': test_rows},
        'consumed_files_v1.json': {'complete': True, 'all_before_after_exact': True, 'files': consumed}}
    for name, value in artifacts.items():
        with (out / name).open('x', encoding='utf8', newline='\n') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'saved_pixel_metric_selection_consistency_pass': not failed,
        'checks': len(c.rows), 'failed_checks': failed, 'counts': result['counts'],
        'eligible_indices': [row['candidate_index'] for row in trial_results if row['eligible']],
        'selected_index': chosen, 'pilot_quality_gates_pass': expected_quality_pass,
        'selected_quality_failed_checks': final_quality.get('failed_role_mean_checks', [])}, indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(run())


