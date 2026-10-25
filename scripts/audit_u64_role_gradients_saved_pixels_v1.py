"""Finite independent NumPy saved-image/scalar audit; no neural execution.

Reads the already inventoried return, never extracts an archive. The FP32
arithmetic bounds below are numerical readback bounds, never quality tolerance.
The photographic objective is independently expressed here from sealed source;
no research, Torch, model, optimizer, or gradient module is imported/executed.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys

import numpy as np
from PIL import Image

PROTOCOL_SHA = '07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3'
PARENT_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
OBJECTIVE_SHA = 'cffafed3feae10ce91890cfd68ee5de96c8db8edad00041420615cd071f2b7ad'
WORKER_SHA = 'e425df3b2f6c0a44e4438efae620d0af9c81e82877cf093d4c27647c42abab52'
ARITHMETIC_BOUND = 5e-7
RESIZE_BOUND = float(np.finfo(np.float32).eps)
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
QUERIES = ('objective_div4', 'gate_mae', 'gate_adjacent')
SHAPES = {'raw.npy': (256, 256, 3), 'unclipped.npy': (1, 3, 256, 256),
          'native.npy': (1, 3, 512, 512), 'logits.npy': (1, 256, 1024),
          'LQ.npy': (1, 256, 16, 16)}
OUTPUT = 'outputs/u64_role_gradients_v1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def binding(raw):
    return {'bytes': len(raw), 'sha256': digest(raw)}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    def finite(item, depth=0):
        need(depth <= 40, 'Bounded JSON depth')
        if type(item) is float:
            need(math.isfinite(item), 'Finite JSON scalar')
        elif type(item) in (dict, list):
            for child in item.values() if type(item) is dict else item:
                finite(child, depth + 1)
    finite(value)
    return value


def relative(name):
    p = PurePosixPath(name)
    need(type(name) is str and name and '\\' not in name and ':' not in name
         and not p.is_absolute() and p.as_posix() == name
         and not any(part in ('', '.', '..') for part in p.parts), 'Canonical relative path')
    return name


class Reader:
    def __init__(self, workspace):
        self.workspace = workspace.resolve(strict=True)
        self.consumed = {}

    def read(self, path, expected=None, maximum=8 * 1024**2):
        path = path.absolute()
        need(path.is_relative_to(self.workspace), 'Read escaped workspace')
        for parent in (path.parent, *path.parent.parents):
            s = parent.lstat()
            need(stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode)
                 and not getattr(s, 'st_file_attributes', 0) & 1024, 'Ordinary directory ancestry')
        s = path.lstat()
        need(stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode) and s.st_nlink == 1
             and not getattr(s, 'st_file_attributes', 0) & 1024 and 0 <= s.st_size <= maximum,
             'Bounded single-link regular file: ' + str(path))
        raw = path.read_bytes()
        after = path.stat()
        need((s.st_size, s.st_mtime_ns, s.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino),
             'File changed while reading')
        actual = binding(raw)
        if expected is not None:
            need(actual == {k: expected[k] for k in ('bytes', 'sha256')}, 'Pinned bytes differ: ' + str(path))
        name = path.relative_to(self.workspace).as_posix()
        need(name not in self.consumed or self.consumed[name] == actual, 'Consumed file changed')
        self.consumed[name] = actual
        return raw

    def verify_after(self):
        before = dict(self.consumed)
        for name, expected in before.items():
            self.read(self.workspace / name, expected)
        need(self.consumed == before, 'Consumed namespace unchanged')
        return [{'path': k, **v} for k, v in sorted(before.items())]


def npy(raw, leaf):
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    need(version in ((1, 0), (2, 0)), 'Bounded NPY1/2 only')
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream, max_header_size=4096)
    need(leaf in SHAPES and shape == SHAPES[leaf] and dtype == np.dtype('float32')
         and not dtype.hasobject and not fortran, 'Exact FP32 C-order array shape')
    need(stream.tell() + math.prod(shape) * 4 == len(raw), 'Exact bounded NPY extent')
    result = np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(shape)
    need(np.isfinite(result).all(), 'Finite saved image/stock array')
    if leaf == 'raw.npy':
        need(result.min() >= 0 and result.max() <= 1, 'Delivered RAW in [0,1]')
    return result


def png(raw):
    need(len(raw) <= 256 * 1024, 'PNG256KiB cap')
    with Image.open(io.BytesIO(raw)) as im:
        need(im.format == 'PNG' and im.mode == 'RGB' and im.size == (256, 256)
             and getattr(im, 'n_frames', 1) == 1, 'Single RGB256 PNG')
        return np.asarray(im).copy()


def normalize_u8(value):
    need(value.dtype == np.uint8, 'RGB8 normalization input')
    return value.astype(np.float32) / np.float32(255)


def floor_png(value):
    need(value.dtype == np.float32 and np.isfinite(value).all()
         and value.min() >= 0 and value.max() <= 1, 'Finite normalized FP32 PNG input')
    return (value * np.float32(255)).astype(np.uint8)


def downsample512(value):
    """Half-pixel bilinear512->256: each output averages its exact2x2 neighbors."""
    need(value.shape == (1, 3, 512, 512) and value.dtype == np.float32, 'Native512 FP32')
    return (value[:, :, 0::2, 0::2] + value[:, :, 0::2, 1::2]
            + value[:, :, 1::2, 0::2] + value[:, :, 1::2, 1::2]) * np.float32(.25)


def delivered_from_native(value):
    normalized = (value + np.float32(1)) / np.float32(2)
    unclipped = downsample512(normalized)
    delivered = downsample512(np.clip(normalized, 0, 1))
    return delivered[0].transpose(1, 2, 0), unclipped


def objective(prediction, target, role, precision=np.float64):
    """Photographic target formula; outside edges require both endpoints outside."""
    need(role in ROLES and prediction.shape == target.shape == (256, 256, 3), 'Paired photographic role')
    residual = prediction.astype(precision) - target.astype(precision)
    whole = np.ones((256, 256), dtype=bool)
    inside = np.zeros_like(whole)
    inside[64:192, 64:192] = True
    def penalty(a, smooth=False):
        a = np.abs(a)
        if smooth:
            beta = precision(.01)
            a = np.where(a < beta, precision(.5) * a * a / beta, a - precision(.5) * beta)
        return a.mean(dtype=precision)
    def region(mask, smooth=False):
        hm, vm = mask[:, 1:] & mask[:, :-1], mask[1:] & mask[:-1]
        edges = np.concatenate(((residual[:, 1:] - residual[:, :-1])[hm].reshape(-1),
                                (residual[1:] - residual[:-1])[vm].reshape(-1)))
        return penalty(residual[mask], smooth), penalty(edges, smooth)
    if role == ROLES[0]:
        di, ei = region(inside)
        do, eo = region(~inside)
        terms = dict(inside_image_L1=float(di), outside_image_L1=float(do),
                     inside_adjacent_L1=float(ei), outside_adjacent_L1=float(eo))
        result = precision(.5) * (di + do) + precision(.125) * (ei + eo)
    elif role == 'clear':
        di, ei = region(whole)
        si, se = region(whole, True)
        terms = dict(photographic_image_anchor_L1=float(di), photographic_adjacent_anchor_L1=float(ei),
                     photographic_image_anchor_SmoothL1=float(si), photographic_adjacent_anchor_SmoothL1=float(se))
        result = precision(2) * (si + precision(.25) * se)
    else:
        di, ei = region(whole)
        terms = dict(target_image_L1=float(di), target_adjacent_L1=float(ei))
        result = di + precision(.25) * ei
        if role == ROLES[-1]:
            weights = np.asarray((.2126, .7152, .0722), dtype=precision)
            def blocks(a):
                return (a.astype(precision) * weights).sum(axis=2, dtype=precision).reshape(16, 16, 16, 16).mean(axis=(1, 3), dtype=precision)
            delta = blocks(prediction) - blocks(target)
            anchor = penalty(delta, True)
            terms.update(photographic_block16_luminance_anchor_L1=float(penalty(delta)),
                         photographic_block16_luminance_anchor_SmoothL1=float(anchor))
            result += precision(2) * anchor
    return float(result), terms


def gate_values(prediction, target, role, precision=np.float64):
    residual = prediction.astype(precision) - target.astype(precision)
    if role == ROLES[0]:
        residual = residual[64:192, 64:192]
    return {'gate_mae': float(np.abs(residual).mean(dtype=precision)),
            'gate_adjacent': float(precision(.5) * (np.abs(residual[:, 1:] - residual[:, :-1]).mean(dtype=precision)
                                                   + np.abs(residual[1:] - residual[:-1]).mean(dtype=precision)))}


def fixtures():
    rows = []
    def check(name, result):
        need(bool(result), 'Array fixture failed: ' + name)
        rows.append({'name': name, 'pass': True})
    u8 = np.asarray([0, 1, 127, 128, 254, 255], dtype=np.uint8)
    normalized = normalize_u8(u8)
    check('RGB8_normalization_FP32_endpoints_and_floor_identity', normalized.dtype == np.float32
          and normalized[0] == 0 and normalized[-1] == 1 and np.array_equal(floor_png(normalized), u8))
    check('floor_quantization_distinguishes_rounding', floor_png(np.asarray([.5], np.float32))[0] == 127)
    x = np.zeros((1, 3, 512, 512), np.float32)
    x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 1, 0], x[:, :, 1, 1] = 1, 2, 3, 6
    check('half_pixel_bilinear_2x2_neighbors', np.all(downsample512(x)[:, :, 0, 0] == 3))
    x.fill(-1)
    x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 1, 0], x[:, :, 1, 1] = -3, -1, 1, 3
    delivered, unclipped = delivered_from_native(x)
    check('native_clamp_before_resize_not_after_resize', np.all(delivered[0, 0] == .5)
          and np.all(unclipped[:, :, 0, 0] == .5))
    x[:, :, 0, 0] = -7
    delivered, unclipped = delivered_from_native(x)
    check('clamp_order_adversarial_difference', np.all(delivered[0, 0] == .5)
          and np.all(np.clip(unclipped[:, :, 0, 0], 0, 1) == 0))
    prediction = np.full((256, 256, 3), .5, np.float32)
    target = np.full_like(prediction, .25)
    for role, wanted in zip(ROLES, (.25, .25, .49, .74)):
        actual, _ = objective(prediction, target, role)
        check('photographic_constant_' + role, abs(actual - wanted) < 1e-14)
    prediction.fill(0)
    target.fill(0)
    prediction[:64] = 1
    check('local_gate_central128_excludes_outside', gate_values(prediction, target, ROLES[0])
          == {'gate_mae': 0., 'gate_adjacent': 0.})
    check('whole_gate_includes_outside', gate_values(prediction, target, ROLES[1])['gate_mae'] == .25)
    prediction.fill(0)
    prediction[64:192, 64:192] = 1
    _, terms = objective(prediction, target, ROLES[0])
    check('objective_excludes_cross_boundary_edges', terms['inside_adjacent_L1'] == 0.
          and terms['outside_adjacent_L1'] == 0.)
    return rows


class Checks:
    def __init__(self):
        self.rows = []
    def check(self, name, ok, slot=None, detail=None):
        self.rows.append({'name': name, 'pass': bool(ok), 'slot': slot, 'detail': detail})
    def close(self, name, a, b, slot, bound=ARITHMETIC_BOUND):
        delta = abs(float(a) - float(b))
        self.check(name, delta <= bound, slot, {'absolute_difference': delta, 'arithmetic_bound_only': bound})
        return delta


def audit(workspace, out):
    r = Reader(workspace)
    test_rows = fixtures()
    self_source = Path(__file__).resolve()
    r.read(self_source)
    index_path = workspace / 'outputs/codeformer_u64_role_gradients_v1_return_audit/inventory_v1.json'
    inventory = strict_json(r.read(index_path, maximum=4 * 1024**2))
    need(inventory['complete'] is True and inventory['protocol_sha256'] == PROTOCOL_SHA, 'Complete bound inventory')
    evidence = Path(inventory['evidence_root']).resolve(strict=True)
    own_stage = workspace / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm'
    parent_stage = workspace / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
    protocol_raw = r.read(own_stage / 'protocol.json')
    need(digest(protocol_raw) == PROTOCOL_SHA, 'Exact sealed diagnostic protocol')
    p = strict_json(protocol_raw)
    parent_raw = r.read(parent_stage / 'protocol.json')
    need(digest(parent_raw) == PARENT_SHA and binding(parent_raw)
         == {k: p['parent_protocol'][k] for k in ('bytes', 'sha256')},
         'Exact sealed parent protocol')
    parent = strict_json(parent_raw)
    need(len(parent['files']) == 469, 'Exact retained parent static schema')
    worker_name = 'research/codeformer_u64_role_gradients_v1/worker.py'
    worker_raw = r.read(own_stage / worker_name, p['files'][worker_name])
    need(digest(worker_raw) == WORKER_SHA, 'Exact reviewed worker source')
    objective_name = 'research/codeformer_u64_cleananchors_continuation_v1/objective.py'
    objective_raw = r.read(parent_stage / objective_name, parent['files'][objective_name])
    need(digest(objective_raw) == OBJECTIVE_SHA, 'Exact photographic objective source')
    need('torch' not in sys.modules, 'No Torch import')
    index = inventory['extracted_files']
    def returned(name):
        relative(name)
        need(name in index and index[name] == inventory['files'][name], 'Extracted/inventoried join')
        return r.read(evidence / name, index[name])
    cache = {}
    def photographic(row):
        name = relative(row['path'])
        need(parent['files'][name] == {k: row[k] for k in ('bytes', 'sha256')}, 'Target/input parent binding')
        if name not in cache:
            cache[name] = png(r.read(parent_stage / name, row))
        return cache[name]
    specs = {v['id']: v for cohort in ('optimization', 'monitors', 'guards') for v in p['selection'][cohort]}
    historical = {v['id']: v for v in parent['u64_parity']}
    need(len(p['plan']) == 72 and len(p['selection']['optimization']) == 64, 'Exact finite72/64 selection')
    c = Checks()
    pixel_rows, scalar_rows, parity_rows, repeat_rows, first = [], [], [], [], {}
    for call in p['plan']:
        slot, cid = call['slot'], call['id']
        spec = specs[cid]
        stem = OUTPUT + '/calls/' + str(slot).zfill(4)
        record_name = stem + '/record.json'
        record_raw = returned(record_name)
        record = strict_json(record_raw)
        c.check('record_plan_selection_join', record['complete'] is True and record['call'] == call
                and all(record[k] == spec[k] for k in ('id', 'reference_id', 'role', 'split', 'input', 'target')),
                slot)
        c.check('no_quality_or_PNG_derivative_claim', record['PNG_gradient_claimed'] is False
                and record['model_qualified'] is False, slot)
        observed = photographic(spec['input'])
        target_u8 = photographic(spec['target'])
        c.check('actual_photographic_input_RGB8_hash', digest(observed.tobytes()) == spec['input_RGB8_sha256'], slot)
        c.check('actual_photographic_target_RGB8_hash', digest(target_u8.tobytes()) == spec['target_RGB8_sha256'], slot)
        arrays = {}
        for field, leaf in (('raw', 'raw.npy'), ('unclipped', 'unclipped.npy'), ('png', 'result.png')):
            pin = record['outputs'][field]
            c.check('output_exact_path_' + field, pin['path'] == stem + '/' + leaf, slot)
            raw_bytes = returned(pin['path'])
            c.check('output_record_binding_' + field, binding(raw_bytes) == {k: pin[k] for k in ('bytes', 'sha256')}, slot)
            arrays[field] = png(raw_bytes) if field == 'png' else npy(raw_bytes, leaf)
        for field in ('native', 'logits', 'LQ'):
            stock = record['stock'][field]
            pin = stock['file']
            c.check('stock_exact_path_' + field, pin['path'] == stem + '/' + field + '.npy', slot)
            raw_bytes = returned(pin['path'])
            arrays[field] = npy(raw_bytes, field + '.npy')
            c.check('stock_file_binding_' + field, binding(raw_bytes) == {k: pin[k] for k in ('bytes', 'sha256')}, slot)
            c.check('stock_value_metadata_' + field, {k: stock[k] for k in ('shape', 'dtype', 'bytes_sha256')}
                    == {'shape': list(arrays[field].shape), 'dtype': 'float32', 'bytes_sha256': digest(arrays[field].tobytes())}, slot)
        c.check('RAW_floor_PNG_exact', np.array_equal(floor_png(arrays['raw']), arrays['png']), slot)
        delivered, unclipped = delivered_from_native(arrays['native'])
        raw_error = float(np.abs(delivered - arrays['raw']).max())
        un_error = float(np.abs(unclipped - arrays['unclipped']).max())
        c.check('native_clamp_before_bilinear256', raw_error <= RESIZE_BOUND, slot,
                {'maximum_absolute_difference': raw_error, 'FP32_arithmetic_bound_only': RESIZE_BOUND})
        c.check('unclipped_bilinear256', un_error <= RESIZE_BOUND, slot,
                {'maximum_absolute_difference': un_error, 'FP32_arithmetic_bound_only': RESIZE_BOUND})
        clipping = {'native_below_minus1': int((arrays['native'] < -1).sum()),
                    'native_above1': int((arrays['native'] > 1).sum()),
                    'unclipped_below0': int((arrays['unclipped'] < 0).sum()),
                    'unclipped_above1': int((arrays['unclipped'] > 1).sum())}
        c.check('clipping_count_exact', clipping == record['clipping'], slot)
        pixel_rows.append({'slot': slot, 'id': cid, 'role': spec['role'], 'phase': call['phase'],
                           'RAW_floor_PNG_exact': bool(np.array_equal(floor_png(arrays['raw']), arrays['png'])),
                           'native_resize_max_abs_difference': raw_error,
                           'unclipped_resize_max_abs_difference': un_error, 'clipping': clipping,
                           'RAW_RGB_sha256': digest(arrays['raw'].tobytes()),
                           'PNG_RGB8_sha256': digest(arrays['png'].tobytes())})
        if call['phase'] == 'historical_u64_parity':
            witness = historical[cid]
            old_raw = npy(r.read(parent_stage / witness['raw']['path'], witness['raw']), 'raw.npy')
            old_png = photographic(witness['png'])
            raw_exact, png_exact = np.array_equal(old_raw, arrays['raw']), np.array_equal(old_png, arrays['png'])
            c.check('historical_u64_RAW_PNG_exact', raw_exact and png_exact, slot)
            parity_rows.append({'id': cid, 'slot': slot, 'RAW_exact': bool(raw_exact), 'PNG_pixels_exact': bool(png_exact)})
            first[cid] = {k: v.copy() for k, v in arrays.items()}
        elif call['phase'] == 'deterministic_same_u64_repeat':
            matches = {k: bool(np.array_equal(v, first[cid][k])) for k, v in arrays.items()}
            c.check('repeat_full_RAW_PNG_unclipped_native_logits_LQ', all(matches.values()), slot)
            repeat_rows.append({'id': cid, 'slot': slot, 'actual_saved_array_equalities': matches})
        else:
            receipt = strict_json(returned(stem + '/gradients.json'))
            c.check('gradient_scalar_receipt_join', receipt['complete'] is True and receipt['call'] == call
                    and receipt['output_record'] == {'path': record_name, **binding(record_raw)}
                    and receipt['metric_region'] == call['metric_region']
                    and set(receipt['gradients']) == set(QUERIES), slot)
            c.check('paired_TRAIN_queries_only', spec['split'] == 'TRAIN' and spec['cohort'] == 'optimize'
                    and spec['optimization_permitted'] is True and call['gradient_queries'] == list(QUERIES), slot)
            c.check('receipt_RAW_only_once_division', receipt['objective_divided_exactly_once'] is True
                    and receipt['RAW_differentiable_gate_metrics_only'] is True
                    and receipt['PNG_gradient_claimed'] is False and receipt['model_qualified'] is False, slot)
            target = normalize_u8(target_u8)
            obj64, terms64 = objective(arrays['raw'], target, spec['role'], np.float64)
            obj32, terms32 = objective(arrays['raw'], target, spec['role'], np.float32)
            gates64 = gate_values(arrays['raw'], target, spec['role'], np.float64)
            gates32 = gate_values(arrays['raw'], target, spec['role'], np.float32)
            expected64 = {'objective_div4': obj64 / 4, **gates64}
            expected32 = {'objective_div4': float(np.float32(obj32) / np.float32(4)), **gates32}
            c.close('objective_saved_scalar', obj64, receipt['objective'], slot)
            c.check('objective_terms_exact_schema', set(terms64) == set(receipt['objective_terms']), slot)
            for name, value in terms64.items():
                c.close('objective_term_' + name, value, receipt['objective_terms'][name], slot)
            deltas = {}
            reported = {}
            for query in QUERIES:
                reported[query] = receipt['gradients'][query]['scalar']
                deltas[query] = c.close('actual_query_scalar_' + query, expected64[query], reported[query], slot)
            c.close('objective_divided_once_exact_FP32', float(np.float32(receipt['objective']) / np.float32(4)),
                    reported['objective_div4'], slot, 0)
            scalar_rows.append({'slot': slot, 'id': cid, 'role': spec['role'], 'metric_region': call['metric_region'],
                                'actual_photographic_target': spec['target'], 'reported_scalars': reported,
                                'independent_NumPy_float64_scalars': expected64, 'independent_NumPy_float32_scalars': expected32,
                                'absolute_float64_query_differences': deltas,
                                'reported_objective': receipt['objective'], 'NumPy_float64_objective': obj64,
                                'NumPy_float32_objective': obj32, 'reported_objective_terms': receipt['objective_terms'],
                                'NumPy_float64_terms': terms64, 'NumPy_float32_terms': terms32})
    c.check('actual_counts72_images64_objectives192_query_scalars', len(pixel_rows) == 72 and len(scalar_rows) == 64
            and len(parity_rows) == 4 and len(repeat_rows) == 4)
    c.check('torch_not_imported', 'torch' not in sys.modules)
    consumed = r.verify_after()
    failed = [v for v in c.rows if not v['pass']]
    summary = {'complete': True, 'all_checks_pass': not failed, 'format': 'u64-role-gradients-saved-pixels-v1',
               'protocol_sha256': PROTOCOL_SHA, 'counts': {'saved_image_outputs': len(pixel_rows),
               'stock_arrays': 3 * len(pixel_rows), 'objective_receipts': len(scalar_rows),
               'query_scalars': 3 * len(scalar_rows), 'historical_RAW_PNG_parities': len(parity_rows),
               'full_saved_repeats': len(repeat_rows), 'consumed_files_hashed_before_after': len(consumed)},
               'checks': len(c.rows), 'failed_checks': failed, 'array_fixture_tests_pass': True,
               'arithmetic_scalar_readback_bound_only': ARITHMETIC_BOUND,
               'native_resize_FP32_arithmetic_bound_only': RESIZE_BOUND, 'quality_threshold_epsilon': 0,
               'maximum_query_scalar_readback_difference': max(d for row in scalar_rows for d in row['absolute_float64_query_differences'].values()),
               'maximum_native_resize_absolute_difference': max(v['native_resize_max_abs_difference'] for v in pixel_rows),
               'maximum_unclipped_resize_absolute_difference': max(v['unclipped_resize_max_abs_difference'] for v in pixel_rows),
               'historical_u64_parity': parity_rows, 'full_saved_repeats': repeat_rows,
               'all_consumed_files_unchanged_after': True,
               'scope': 'Independent saved-pixel and photographic scalar arithmetic only; saved source documents connected graph construction, but CPU scalar agreement does not independently establish autograd validity.',
               'source_graph_evidence_only': {'worker_sha256': WORKER_SHA, 'photographic_objective_sha256': OBJECTIVE_SHA},
               'neural_gradient_optimizer_training_VM_calls': 0, 'gradient_vector_geometry_computed': False,
               'finite_step_or_quality_prediction': False, 'model_qualified': False,
               'original_u64_retained': True, 'app_incumbent_unchanged': True}
    out.mkdir(parents=True, exist_ok=True)
    artifacts = {'audit_v1.json': summary, 'pixel_rows_v1.json': {'complete': True, 'rows': pixel_rows},
                 'photographic_scalar_rows_v1.json': {'complete': True, 'rows': scalar_rows},
                 'checks_v1.json': {'complete': True, 'checks': c.rows},
                 'source_and_array_tests_v1.json': {'complete': True, 'all_tests_pass': True, 'fixtures': test_rows,
                    'audit_source': {'path': self_source.relative_to(workspace).as_posix(), **r.consumed[self_source.relative_to(workspace).as_posix()]},
                    'no_Torch_import': True, 'no_model_or_gradient_run': True},
                 'consumed_files_v1.json': {'complete': True, 'all_unchanged_after': True, 'files': consumed}}
    for name, value in artifacts.items():
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', default='.')
    parser.add_argument('--output', default='outputs/codeformer_u64_role_gradients_v1_return_pixels')
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve(strict=True)
    summary = audit(workspace, workspace / relative(args.output))
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary['all_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
