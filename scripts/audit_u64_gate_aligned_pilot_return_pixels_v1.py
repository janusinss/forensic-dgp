"""Independent returned RAW/PNG, photographic scalars and quality-bank audit.

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
PROTOCOL_SHA = '35a5cfd877a97355622c6c69d9792fd9238ccf312335dd679081a8636ba23b48'
PARENT_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
OUTPUT = 'outputs/u64_gate_aligned_pilot_v1'
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
        need(fullname.split('.')[0] not in {'torch', 'torchvision', 'tensorflow', 'jax'}, 'Framework import forbidden')
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


def run():
    sys.meta_path.insert(0, BlockFrameworks())
    r, c = Reader(), Checks()
    test_rows = fixtures()
    r.read(Path(__file__).resolve())
    inventory = strict_json(r.read(ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit/inventory_v1.json'))
    need(inventory['complete'] is True and inventory['protocol_sha256'] == PROTOCOL_SHA, 'Complete bound return inventory')
    evidence = Path(inventory['evidence_root']).resolve(strict=True)
    stage = ROOT / 'outputs/cctv_codeformer_u64_gate_aligned_pilot_v1_vm'
    parent = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
    protocol_raw = r.read(stage / 'protocol.json'); need(digest(protocol_raw) == PROTOCOL_SHA, 'Exact reviewed pilot protocol')
    p = strict_json(protocol_raw)
    parent_raw = r.read(parent / 'protocol.json'); need(digest(parent_raw) == PARENT_SHA, 'Exact parent protocol')
    oldp = strict_json(parent_raw)
    standard_raw = r.read(parent / oldp['parent_protocol']['path'], oldp['parent_protocol'])
    source_names = ('research/__init__.py', 'research/codeformer_u64_cleananchors_continuation_v1/__init__.py',
                    'research/codeformer_u64_cleananchors_continuation_v1/metrics.py',
                    'research/codeformer_u64_cleananchors_continuation_v1/contract.py',
                    'research/codeformer_u96_spatial_feasibility_v1/__init__.py',
                    'research/codeformer_u96_spatial_feasibility_v1/metrics.py')
    for name in source_names:
        r.read(parent / name, oldp['files'][name])
    qname = 'research/codeformer_u64_gate_aligned_pilot_v1/quality.py'
    r.read(stage / qname, p['files'][qname])
    sys.path.insert(0, str(parent))
    imported = importlib.import_module('research.codeformer_u64_cleananchors_continuation_v1.metrics')
    for name in source_names:
        module = sys.modules.get(name.removesuffix('/__init__.py').removesuffix('.py').replace('/', '.'))
        need(module is not None and Path(module.__file__).resolve() == parent / name, 'Pinned pure source import origin')
    spec = importlib.util.spec_from_file_location('sealed_pilot_quality_saved_pixel_audit', stage / qname)
    quality = importlib.util.module_from_spec(spec); spec.loader.exec_module(quality)
    index = inventory['extracted_files']
    def returned(name):
        relative(name)
        need(index[name] == inventory['files'][name], 'Extracted/export-index join')
        return r.read(evidence / name, index[name])
    def source(row, leaf):
        name = relative(row['path'])
        need(oldp['files'][name] == {key: row[key] for key in ('bytes', 'sha256')}, 'Parent static binding')
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
    need(len(quality_ids) == len(controls) == 56, 'Exact56 quality bank')
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
    frozen_saved = strict_json(returned(OUTPUT + '/frozen_U96_control_metrics.json'))
    c.same('all56_frozen_U96_saved_metrics', frozen, frozen_saved['records'])
    records, pixel_rows, scalar_rows, first, bank_records = {}, [], [], {}, {0: {}, 1: {}, 2: {}}
    available_record_names = {name for name in index if re.fullmatch(OUTPUT + r'/calls/\d{4}/record.json', name)}
    for call in p['plan']:
        if call['kind'] != 'forward':
            continue
        name = OUTPUT + '/calls/' + str(call['slot']).zfill(4) + '/record.json'
        if name not in available_record_names:
            continue
        saved_bytes = returned(name); saved = strict_json(saved_bytes)
        row, cid, slot = specs[call['id']], call['id'], call['slot']
        need(call['new_update'] in bank_records, 'Bounded returned prefix updates0/1/2 only')
        c.check('record_exact_plan_join_' + str(slot), saved['complete'] is True and saved['call'] == call
                and all(saved[key] == row[key] for key in ('id', 'reference_id', 'role', 'split', 'input', 'target')))
        observed = rgb(row['input'])
        c.check('actual_input_RGB8_hash_' + str(slot), digest(observed.tobytes()) == row['input_RGB8_sha256'])
        target_u8 = None if row.get('target') is None else rgb(row['target'])
        if target_u8 is not None:
            c.check('actual_target_RGB8_hash_' + str(slot), digest(target_u8.tobytes()) == row['target_RGB8_sha256'])
        target = None if target_u8 is None else normalized(target_u8)
        arrays = {}
        for field, leaf in (('raw', 'raw.npy'), ('png', 'result.png')):
            pin = saved['outputs'][field]
            need(pin['path'] == name.rsplit('/', 1)[0] + '/' + leaf, 'Exact image output slot/path')
            data = returned(pin['path']); c.same('actual_image_file_pin_' + str(slot) + '_' + field, binding(data), {key: pin[key] for key in ('bytes', 'sha256')}, 0)
            arrays[field] = png(data) if field == 'png' else npy(data, leaf)
        c.check('RAW_floor_PNG_exact_' + str(slot), np.array_equal(floor_png(arrays['raw']), arrays['png']))
        control = None if cid not in controls else pair(controls[cid])
        teacher = None if row.get('behavior_teacher') is None else pair(row['behavior_teacher'])
        measured = metrics(arrays['raw'], arrays['png'], target, control, teacher)
        c.same('all_independent_saved_metrics_' + str(slot), measured, {key: saved[key] for key in measured})
        options = {} if teacher is None else {'teacher_raw': teacher[0], 'teacher_png': teacher[1]}
        if control is not None:
            helper_value = imported.record_metrics(arrays['raw'], arrays['png'], target, row, np=np,
                frozen_U96_raw=control[0], frozen_U96_png=control[1], **options)
        else:
            helper_value = imported.record_training_metrics(arrays['raw'], arrays['png'], target, row, np=np)
        c.same('independent_vs_pinned_pixel_metrics_' + str(slot), measured, helper_value)
        stock = saved['stock']
        for field in ('native', 'logits', 'LQ'):
            c.check('declared_stock_shape_' + str(slot) + field, stock[field]['shape'] == list(SHAPES[field + '.npy']) and stock[field]['dtype'] == 'float32')
            if 'file' in stock[field]:
                pin = stock[field]['file']; data = returned(pin['path']); a = npy(data, field + '.npy')
                c.same('retained_stock_file_pin_' + str(slot) + field, binding(data), {key: pin[key] for key in ('bytes', 'sha256')}, 0)
                c.check('retained_stock_value_hash_' + str(slot) + field, digest(a.tobytes()) == stock[field]['bytes_sha256'])
                arrays[field] = a
        if 'native' in arrays:
            native = (arrays['native'] + np.float32(1)) / np.float32(2)
            def resize(a):
                return (a[:, :, ::2, ::2] + a[:, :, ::2, 1::2] + a[:, :, 1::2, ::2] + a[:, :, 1::2, 1::2]) * np.float32(.25)
            delivered = resize(np.clip(native, 0, 1))[0].transpose(1, 2, 0)
            unpin = saved['outputs']['unclipped']; unbytes = returned(unpin['path']); arrays['unclipped'] = npy(unbytes, 'unclipped.npy')
            c.same('witness_unclipped_file_pin_' + str(slot), binding(unbytes), {key: unpin[key] for key in ('bytes', 'sha256')}, 0)
            c.check('retained_native_clamp_resize_' + str(slot), float(np.abs(delivered - arrays['raw']).max()) <= np.finfo(np.float32).eps)
            c.check('retained_native_unclipped_resize_' + str(slot), float(np.abs(resize(native) - arrays['unclipped']).max()) <= np.finfo(np.float32).eps)
        if call['phase'] == 'initial_u64_quality_bank':
            bank_records[0][cid] = {**saved, **measured}
            if cid in parity:
                witness = pair(parity[cid])
                c.check('four_historical_u64_RAW_PNG_exact_' + cid, np.array_equal(arrays['raw'], witness[0]) and np.array_equal(arrays['png'], witness[1]))
                first[cid] = {key: value.copy() for key, value in arrays.items()}
        elif call['phase'] == 'deterministic_same_u64_repeat':
            c.check('six_output_full_repeat_' + cid, set(arrays) == set(first[cid]) and all(np.array_equal(value, first[cid][key]) for key, value in arrays.items()))
        elif call['phase'] == 'current_quality_bank':
            bank_records[call['new_update']][cid] = {**saved, **measured}
        else:
            need(call['phase'] == 'paired_TRAIN_gate_gradients' and target is not None, 'Exact TRAIN query phase')
            scalar = strict_json(returned(name.rsplit('/', 1)[0] + '/gradients.json'))
            c.check('gradient_scalar_record_join_' + str(slot), scalar['call'] == call and scalar['output_record'] == {'path': name, **binding(saved_bytes)}
                    and scalar['head_before_sha256'] == saved['head_before_sha256'] and set(scalar['gradients']) == {'gate_mae', 'gate_adjacent'})
            expected = gate_scalars(arrays['raw'], target, row['role'])
            for query in ('gate_mae', 'gate_adjacent'):
                receipt = scalar['gradients'][query]
                standalone = strict_json(returned(name.rsplit('/', 1)[0] + '/gradient_' + query + '_receipt.json'))
                c.same('standalone_query_receipt_' + str(slot) + query, receipt, standalone, 0)
                error = c.same('actual_RAW_query_scalar_' + str(slot) + query, expected[query], receipt['scalar'], SCALAR_ARITHMETIC_BOUND)
                scalar_rows.append({'slot': slot, 'update': call['new_update'], 'id': cid, 'role': row['role'],
                                    'query': query, 'metric_region': call['metric_region'], 'reported_scalar': receipt['scalar'],
                                    'independent_NumPy_float64_scalar': expected[query], 'absolute_difference': error})
        recomputed = {**saved, **measured}; records[slot] = recomputed
        pixel_rows.append({'slot': slot, 'update': call['new_update'], 'id': cid, 'role': row['role'], 'phase': call['phase'],
                           'RAW_floor_PNG_exact': True, 'metrics': measured,
                           'RAW_value_sha256': digest(arrays['raw'].tobytes()), 'PNG_RGB8_sha256': digest(arrays['png'].tobytes())})
    c.check('actual_300_forward_prefix_no_extra_record', len(records) == len(available_record_names) == 300 and max(records) == 302)
    c.check('actual256_scalar_receipts', len(scalar_rows) == 256)
    c.check('exact56_quality_members_each_bank', all(set(bank) == quality_ids for bank in bank_records.values()))
    for update, bank in bank_records.items():
        for cid, record in bank.items():
            record['receipt'] = {'path': OUTPUT + '/calls/' + str(record['call']['slot']).zfill(4) + '/record.json', **index[OUTPUT + '/calls/' + str(record['call']['slot']).zfill(4) + '/record.json']}
    recomputed_banks = []
    original = bank_records[0]; prior = original
    first_failed = None
    for update in (0, 1, 2):
        bank = bank_records[update]
        saved = strict_json(returned(OUTPUT + '/banks/u' + str(update).zfill(4) + '/bank.json'))
        c.same('bank_record_actual_output_join_u' + str(update), bank, saved['records'])
        inherited = imported.evaluate_bank(bank, original, frozen, standard_raw,
            phase='initializer' if update == 0 else 'u4', saved_audit_bytes=audit_bytes)
        if update == 0:
            c.same('initial_original_inherited_gate', inherited, saved['gate'])
            final_gate = inherited
            means = cohort_means(bank, selection['monitors'])
            changes, own_checks = [], []
        else:
            means = cohort_means(bank, selection['monitors'])
            original_means = cohort_means(original, selection['monitors'])
            prior_means = cohort_means(prior, selection['monitors'])
            c.same('independent_current_means_u' + str(update), means, saved['current_means'])
            c.same('independent_original_means_u' + str(update), original_means, saved['original_means'])
            c.same('independent_prior_means_u' + str(update), prior_means, saved['prior_means'])
            c.same('original_inherited_case_gates_u' + str(update), inherited, saved['inherited'])
            own_checks, own_pass = added_gate(means, original_means, prior_means, False)
            c.same('independent32_zero_tolerance_checks_u' + str(update), own_checks, saved['gate']['role_mean_nonincrease_checks'])
            final_gate = quality.evaluate_quality(means, original_means, prior_means, False, inherited)
            c.same('pinned_added_gate_helper_u' + str(update), final_gate, saved['gate'])
            c.check('independent_pass_matches_added_helper_u' + str(update), own_pass == final_gate['additional_role_mean_checks_pass'])
            changes = individual_changes(bank, original, prior, specs)
            c.same('independent224_individual_change_rows_u' + str(update), changes, saved['individual_changes'])
            c.check('individual224_rows_u' + str(update), len(changes) == 224)
            failures = [row for row in own_checks if not row['pass']]
            if failures and first_failed is None:
                first_failed = {'update': update, 'failed_checks': failures, 'one_distinct_failed_endpoint': len({(row['view'], row['role'], row['field']) for row in failures}) == 1}
        recomputed_banks.append({'update': update, 'pass': final_gate['pass'], 'inherited_pass': inherited['pass'],
                                 'means': means, 'added_role_mean_checks': own_checks, 'individual_changes': changes,
                                 'original_terminal_table': inherited['original_terminal'], 'terminal4_attempted': False})
        if final_gate['pass']:
            prior = bank
    failure = strict_json(returned(OUTPUT + '/failure.json'))
    c.check('exact_saved_quality_stop_u2', failure['quality_stop'] is True and failure['error_type'] == 'QualityStop'
            and failure['active_update'] == 2 and failure['active_slot'] == 302
            and failure['counters']['full_forward_returns'] == 300 and failure['counters']['gradient_query_returns'] == 256
            and failure['counters']['parameter_applications'] == 2 and failure['operation_outcome_unknown'] is False)
    c.check('terminal4_and_later_banks_not_attempted', all(OUTPUT + '/banks/u' + str(update).zfill(4) + '/bank.json' not in index for update in (3, 4)))
    c.check('first_failed_endpoint_nativePNG_local_adjacent', first_failed is not None and first_failed['update'] == 2
            and len(first_failed['failed_checks']) == 2 and all(row['view'] == 'native_png' and row['role'] == ROLES[0]
                and row['field'] == 'adjacent_rgb_error' for row in first_failed['failed_checks']))
    c.check('no_framework_loaded', not any(name.split('.')[0] in {'torch', 'torchvision', 'tensorflow', 'jax'} for name in sys.modules))
    consumed = r.after()
    failed = [row for row in c.rows if not row['pass']]
    result = {'complete': True, 'audit_pass': not failed, 'independent_saved_pixel_integrity_pass': not failed,
              'format': 'gate-aligned-pilot-return-independent-saved-pixels-v1', 'protocol_sha256': PROTOCOL_SHA,
              'counts': {'actual_saved_forwards': len(records), 'RAW_floor_PNG_pairs': len(pixel_rows),
                         'actual_scalar_receipts': len(scalar_rows), 'quality_banks': 3, 'quality_cases_per_bank': 56,
                         'retained_stock_triples': 8, 'historical_RAW_PNG_parities': 4, 'full_six_output_repeats': 4,
                         'individual_change_rows_per_update': 224, 'consumed_files_before_after': len(consumed)},
              'checks': len(c.rows), 'failed_audit_checks': failed,
              'bank_gate_results': [{'update': row['update'], 'pass': row['pass'], 'inherited_pass': row['inherited_pass']} for row in recomputed_banks],
              'first_failed_endpoint': first_failed, 'authoritative_quality_stop': True, 'pilot_quality_gates_pass': False,
              'quality_tolerance': 0, 'metric_arithmetic_readback_bound_only': METRIC_ARITHMETIC_BOUND,
              'query_FP32_arithmetic_readback_bound_only': SCALAR_ARITHMETIC_BOUND,
              'maximum_query_scalar_absolute_difference': max(row['absolute_difference'] for row in scalar_rows),
              'later_update3_or_terminal4_attempted': False, 'terminal4_required_gains_demonstrated': False,
              'earlier_u1_state_selected_as_success': False, 'finite_gate_failure_repaired': False,
              'model_qualified': False, 'app_incumbent_unchanged': True, 'original_u64_retained': True,
              'all_consumed_hashes_unchanged_after': True, 'array_fixtures_pass': True,
              'metric_math_independent_and_crosschecked_to_pinned_pure_helpers': True,
              'unretained_native_arrays_for_other292_calls_not_reconstructed': True,
              'no_Torch_model_autograd_optimizer_parameter_application_VM_browser_calls': True,
              'scope': 'Actual saved RAW/PNG and photographic scalar/metric readback plus unchanged pure inherited gate evaluation. This audit does not replay a neural forward/autograd or establish useful restoration, identities, native CCTV, DEV or FINAL evidence.'}
    out = ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_return_pixels'; out.mkdir(parents=True, exist_ok=True)
    artifacts = {'audit_v1.json': result, 'pixel_rows_v1.json': {'complete': True, 'rows': pixel_rows},
                 'scalar_rows_v1.json': {'complete': True, 'rows': scalar_rows},
                 'recomputed_banks_v1.json': {'complete': True, 'banks': recomputed_banks},
                 'checks_v1.json': {'complete': True, 'rows': c.rows},
                 'source_array_tests_v1.json': {'complete': True, 'all_tests_pass': True, 'fixtures': test_rows,
                                              'audit_source': {'path': Path(__file__).relative_to(ROOT).as_posix(), **r.consumed[Path(__file__).relative_to(ROOT).as_posix()]}},
                 'consumed_files_v1.json': {'complete': True, 'all_before_after_exact': True, 'files': consumed}}
    for name, value in artifacts.items():
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(run())
