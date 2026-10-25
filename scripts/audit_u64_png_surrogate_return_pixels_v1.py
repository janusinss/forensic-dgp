"""Independent finite saved-array readback of the zero-update PNG measurement.

No model, Torch, autograd, optimizer, parameter application or VM is called.
The identity backward is documented by sealed source only; saved forward and
scalar agreement cannot independently establish actual autograd correctness.
"""
import hashlib
import importlib.abc
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = '0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c'
PARENT_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
OUTPUT = 'outputs/u64_png_surrogate_measurement_v1'
QUERIES = ('raw_mae', 'raw_adjacent', 'png_surrogate_mae', 'png_surrogate_adjacent')
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
ARITHMETIC_BOUND = 5e-7
RESIZE_BOUND = float(np.finfo(np.float32).eps)
SHAPES = {'raw.npy': (256, 256, 3), 'unclipped.npy': (1, 3, 256, 256),
          'native.npy': (1, 3, 512, 512), 'logits.npy': (1, 256, 1024),
          'LQ.npy': (1, 256, 16, 16), 'png_surrogate_forward.npy': (256, 256, 3)}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def pin(value):
    return {'bytes': len(value), 'sha256': digest(value)}


def relative(name):
    need(type(name) is str and name and '\\' not in name and ':' not in name, 'Canonical path')
    value = PurePosixPath(name)
    need(not value.is_absolute() and value.as_posix() == name and not any(v in ('', '.', '..') for v in value.parts), 'Confined path')
    return name


def json_value(raw):
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
        need(depth <= 40, 'Bounded JSON depth')
        if type(value) is float:
            need(math.isfinite(value), 'Finite JSON scalar')
        elif type(value) in (dict, list):
            for child in value.values() if type(value) is dict else value:
                finite(child, depth + 1)
    finite(result)
    return result


class Reader:
    def __init__(self):
        self.consumed = {}
    def read(self, path, expected=None, maximum=8 * 1024**2):
        path = path.absolute()
        need(path.is_relative_to(ROOT), 'Workspace-bound read')
        for parent in (path.parent, *path.parent.parents):
            value = parent.lstat()
            need(stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode)
                 and not getattr(value, 'st_file_attributes', 0) & 1024, 'Ordinary ancestry')
        before = path.lstat()
        need(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode)
             and before.st_nlink == 1 and not getattr(before, 'st_file_attributes', 0) & 1024
             and 0 <= before.st_size <= maximum, 'Bounded single-link regular file')
        data = path.read_bytes()
        after = path.stat()
        need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'File changed during read')
        actual = pin(data)
        if expected is not None:
            need(actual == {key: expected[key] for key in ('bytes', 'sha256')}, 'Pinned bytes differ: ' + str(path))
        name = path.relative_to(ROOT).as_posix()
        need(name not in self.consumed or self.consumed[name] == actual, 'Consumed file changed')
        self.consumed[name] = actual
        return data
    def after(self):
        before = dict(self.consumed)
        for name, expected in before.items():
            self.read(ROOT / name, expected)
        need(before == self.consumed, 'Consumed namespace stable')
        return [{'path': name, **value} for name, value in sorted(before.items())]


def npy(data, leaf):
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    need(version in ((1, 0), (2, 0)), 'NPY1/2 only')
    read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = read_header(stream, max_header_size=4096)
    need(shape == SHAPES[leaf] and dtype == np.dtype('float32') and not dtype.hasobject and not fortran, 'Exact C-order FP32 schema')
    need(stream.tell() + math.prod(shape) * 4 == len(data), 'Exact NPY extent')
    value = np.frombuffer(data, dtype=dtype, offset=stream.tell()).reshape(shape)
    need(np.isfinite(value).all(), 'Finite saved array')
    if leaf in ('raw.npy', 'png_surrogate_forward.npy'):
        need(value.min() >= 0 and value.max() <= 1, 'Normalized [0,1] saved pixels')
    return value


def png(data):
    need(len(data) <= 256 * 1024, 'PNG256KiB cap')
    with Image.open(io.BytesIO(data)) as image:
        need(image.format == 'PNG' and image.mode == 'RGB' and image.size == (256, 256)
             and getattr(image, 'n_frames', 1) == 1, 'Single RGB256 PNG')
        return np.asarray(image).copy()


def normalized(value):
    return value.astype(np.float32) / np.float32(255)


def floor_png(value):
    return (value * np.float32(255)).astype(np.uint8)


def scalar_metrics(observed, target, role, precision=np.float64):
    residual = observed.astype(precision) - target.astype(precision)
    if role == ROLES[0]:
        residual = residual[64:192, 64:192]
    return {'mae': float(np.abs(residual).mean(dtype=precision)),
            'adjacent': float(precision(.5) * (np.abs(residual[:, 1:] - residual[:, :-1]).mean(dtype=precision)
                                              + np.abs(residual[1:] - residual[:-1]).mean(dtype=precision)))}


def resize512(value):
    return (value[:, :, ::2, ::2] + value[:, :, ::2, 1::2]
            + value[:, :, 1::2, ::2] + value[:, :, 1::2, 1::2]) * np.float32(.25)


class Checks:
    def __init__(self):
        self.rows = []
    def check(self, name, okay, detail=None):
        self.rows.append({'name': name, 'pass': bool(okay), 'detail': detail})
    def scalar(self, name, actual, reported):
        error = abs(actual - reported)
        self.check(name, error <= ARITHMETIC_BOUND, {'absolute_difference': error, 'arithmetic_bound_only': ARITHMETIC_BOUND})
        return error


def fixtures():
    rows = []
    def check(name, value):
        need(value, 'Fixture failed: ' + name)
        rows.append({'name': name, 'pass': True})
    z = np.zeros((256, 256, 3), np.float32)
    half = np.full_like(z, .5)
    check('constant_half_mae_zero_edge', scalar_metrics(half, z, ROLES[1]) == {'mae': .5, 'adjacent': 0.})
    half[:64] = 1
    check('local_central128_region_excludes_outside', scalar_metrics(half, z, ROLES[0]) == {'mae': .5, 'adjacent': 0.})
    check('floor_quantizer_distinguishes_rounding', floor_png(np.asarray([.5], np.float32))[0] == 127)
    codes = np.asarray([0, 1, 85, 127, 128, 254, 255], np.uint8)
    check('decoded_png_exact_CPU_FP32', normalized(codes).dtype == np.float32 and np.array_equal(floor_png(normalized(codes)), codes))
    native = np.full((1, 3, 512, 512), -1, np.float32)
    native[:, :, 0, 0] = -7; native[:, :, 0, 1] = -1; native[:, :, 1, 0] = 1; native[:, :, 1, 1] = 3
    values = (native + np.float32(1)) / np.float32(2)
    check('native_clamp_before_resize_order', np.all(resize512(np.clip(values, 0, 1))[:, :, 0, 0] == .5)
          and np.all(np.clip(resize512(values), 0, 1)[:, :, 0, 0] == 0))
    return rows


class BlockFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        need(fullname.split('.')[0] not in {'torch', 'torchvision', 'tensorflow', 'jax'}, 'Neural framework import prohibited')
        return None


def run():
    sys.meta_path.insert(0, BlockFramework())
    r, c = Reader(), Checks()
    test_rows = fixtures()
    r.read(Path(__file__).resolve())
    inventory = json_value(r.read(ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit/inventory_v1.json'))
    need(inventory['complete'] and inventory['protocol_sha256'] == PROTOCOL_SHA, 'Complete exact measurement inventory')
    evidence = Path(inventory['evidence_root']).resolve(strict=True)
    stage = ROOT / 'outputs/cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
    parent = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
    raw_protocol = r.read(stage / 'protocol.json'); need(digest(raw_protocol) == PROTOCOL_SHA, 'Exact measurement protocol')
    p = json_value(raw_protocol)
    raw_parent = r.read(parent / 'protocol.json'); need(digest(raw_parent) == PARENT_SHA, 'Exact parent protocol')
    old = json_value(raw_parent)
    worker_name = 'research/codeformer_u64_png_surrogate_measurement_v1/worker.py'
    surrogate_name = 'research/codeformer_u64_png_surrogate_measurement_v1/surrogate.py'
    r.read(stage / worker_name, p['files'][worker_name]); r.read(stage / surrogate_name, p['files'][surrogate_name])
    index = inventory['extracted_files']
    def returned(name):
        relative(name)
        need(index[name] == inventory['files'][name], 'Exact extracted/export file join')
        return r.read(evidence / name, index[name])
    cache = {}
    def source(row, leaf):
        name = relative(row['path'])
        need(old['files'][name] == {key: row[key] for key in ('bytes', 'sha256')}, 'Immutable parent source binding')
        if name not in cache:
            data = r.read(parent / name, row)
            cache[name] = png(data) if leaf == 'png' else npy(data, 'raw.npy')
        return cache[name]
    specs = {row['id']: row for group in ('optimization', 'monitors', 'guards') for row in p['selection'][group]}
    historical = {row['id']: row for row in old['u64_parity']}
    pixel_rows, scalar_rows, parity_rows, repeats, first = [], [], [], [], {}
    forward_count, stock_count = 0, 0
    exact_surr = 0
    max_raw_resize = max_unclipped_resize = 0.
    need(len(p['plan']) == 72, 'Exact finite72 plan')
    for call in p['plan']:
        cid, slot = call['id'], call['slot']
        row = specs[cid]
        stem = OUTPUT + '/calls/' + str(slot).zfill(4)
        record_data = returned(stem + '/record.json'); record = json_value(record_data)
        c.check('exact_record_plan_case_' + str(slot), record['complete'] is True and record['call'] == call
                and all(record[key] == row[key] for key in ('id', 'reference_id', 'role', 'split', 'input', 'target')))
        observed = source(row['input'], 'png'); target_rgb = source(row['target'], 'png')
        c.check('actual_input_RGB8_' + str(slot), digest(observed.tobytes()) == row['input_RGB8_sha256'])
        c.check('actual_photographic_target_RGB8_' + str(slot), digest(target_rgb.tobytes()) == row['target_RGB8_sha256'])
        target = normalized(target_rgb)
        arrays = {}
        for field, leaf in (('raw', 'raw.npy'), ('png', 'result.png')):
            output_pin = record['outputs'][field]
            need(output_pin['path'] == stem + '/' + leaf, 'Exact call image path')
            data = returned(output_pin['path'])
            c.check('image_record_file_pin_' + str(slot) + field, pin(data) == {key: output_pin[key] for key in ('bytes', 'sha256')})
            arrays[field] = png(data) if field == 'png' else npy(data, leaf)
        c.check('RAW_floor_PNG_exact_' + str(slot), np.array_equal(floor_png(arrays['raw']), arrays['png']))
        retained = call['phase'] in ('historical_u64_parity', 'deterministic_same_u64_repeat')
        c.check('only8retained_stock_triples_' + str(slot), all(('file' in record['stock'][key]) == retained for key in ('native', 'logits', 'LQ')))
        for field in ('native', 'logits', 'LQ'):
            metadata = record['stock'][field]
            c.check('stock_declared_schema_' + str(slot) + field, metadata['dtype'] == 'float32' and metadata['shape'] == list(SHAPES[field + '.npy']))
            if retained:
                output_pin = metadata['file']; need(output_pin['path'] == stem + '/' + field + '.npy', 'Exact stock call path')
                data = returned(output_pin['path']); arrays[field] = npy(data, field + '.npy'); stock_count += 1
                c.check('stock_file_pin_' + str(slot) + field, pin(data) == {key: output_pin[key] for key in ('bytes', 'sha256')})
                c.check('stock_actual_value_hash_' + str(slot) + field, digest(arrays[field].tobytes()) == metadata['bytes_sha256'])
        if retained:
            output_pin = record['outputs']['unclipped']; need(output_pin['path'] == stem + '/unclipped.npy', 'Exact unclipped call path')
            data = returned(output_pin['path']); arrays['unclipped'] = npy(data, 'unclipped.npy')
            c.check('unclipped_file_pin_' + str(slot), pin(data) == {key: output_pin[key] for key in ('bytes', 'sha256')})
            native = (arrays['native'] + np.float32(1)) / np.float32(2)
            delivered = resize512(np.clip(native, 0, 1))[0].transpose(1, 2, 0)
            raw_error = float(np.abs(delivered - arrays['raw']).max())
            un_error = float(np.abs(resize512(native) - arrays['unclipped']).max())
            max_raw_resize, max_unclipped_resize = max(max_raw_resize, raw_error), max(max_unclipped_resize, un_error)
            c.check('native_clamp_before_resize_' + str(slot), raw_error <= RESIZE_BOUND, {'max_abs_difference': raw_error})
            c.check('unclipped_resize_' + str(slot), un_error <= RESIZE_BOUND, {'max_abs_difference': un_error})
            clipping = {'native_below_minus1': int((arrays['native'] < -1).sum()), 'native_above1': int((arrays['native'] > 1).sum()),
                        'unclipped_below0': int((arrays['unclipped'] < 0).sum()), 'unclipped_above1': int((arrays['unclipped'] > 1).sum())}
            c.check('actual_witness_clipping_counts_' + str(slot), clipping == record['clipping'])
        if call['phase'] == 'historical_u64_parity':
            witness = historical[cid]
            raw_before, png_before = source(witness['raw'], 'raw'), source(witness['png'], 'png')
            equal = np.array_equal(raw_before, arrays['raw']) and np.array_equal(png_before, arrays['png'])
            c.check('historical_u64_RAW_PNG_exact_' + cid, equal)
            parity_rows.append({'id': cid, 'slot': slot, 'RAW_and_PNG_exact': bool(equal)})
            first[cid] = {key: value.copy() for key, value in arrays.items()}
        elif call['phase'] == 'deterministic_same_u64_repeat':
            equal = {key: bool(np.array_equal(value, first[cid][key])) for key, value in arrays.items()}
            c.check('full_six_output_repeat_' + cid, set(equal) == set(first[cid]) and all(equal.values()))
            repeats.append({'id': cid, 'slot': slot, 'actual_saved_equalities': equal})
        else:
            need(call['phase'] == 'paired_TRAIN_gradient_diagnostic' and tuple(call['gradient_queries']) == QUERIES,
                 'Exact four ordered diagnostic queries')
            receipt_data = returned(stem + '/gradients.json'); receipt = json_value(receipt_data)
            c.check('gradient_receipt_call_record_' + str(slot), receipt['complete'] is True and receipt['call'] == call
                    and receipt['output_record'] == {'path': stem + '/record.json', **pin(record_data)}
                    and receipt['metric_region'] == call['metric_region'] and set(receipt['gradients']) == set(QUERIES))
            c.check('biased_surrogate_scope_' + str(slot), receipt['PNG_forward_actual_decoded_FP32_exact'] is True
                    and receipt['RAW_differentiable_gate_metrics'] is True and receipt['PNG_biased_surrogate_gradient_only'] is True
                    and receipt['true_PNG_derivative_claimed'] is False and receipt['model_qualified'] is False
                    and receipt['backward_rule'] == 'Identity to deliveredRAW; none to detached PNG reference')
            need(receipt['PNG_forward_reference']['path'] == stem + '/png_surrogate_forward.npy', 'Exact surrogate call path')
            forward_data = returned(receipt['PNG_forward_reference']['path']); forward = npy(forward_data, 'png_surrogate_forward.npy')
            c.check('surrogate_forward_file_pin_' + str(slot), pin(forward_data) == {key: receipt['PNG_forward_reference'][key] for key in ('bytes', 'sha256')})
            decoded = normalized(arrays['png'])
            equal = forward.tobytes() == decoded.tobytes()
            c.check('actual_decoded_PNG_FP32_surrogate_forward_exact_' + str(slot), equal)
            c.check('surrogate_actual_PNG_file_join_' + str(slot), receipt['PNG_RGB8_file'] == record['outputs']['png'])
            exact_surr += int(equal); forward_count += 1
            scalars64, scalars32 = {}, {}
            for domain, pixels in (('raw', arrays['raw']), ('png_surrogate', forward)):
                for field, value in scalar_metrics(pixels, target, row['role']).items():
                    scalars64[domain + '_' + field] = value
                for field, value in scalar_metrics(pixels, target, row['role'], np.float32).items():
                    scalars32[domain + '_' + field] = value
            for query in QUERIES:
                query_row = receipt['gradients'][query]
                c.check('query_labels_' + str(slot) + query, query_row['head_state_canonical_sha256'] == p['diagnostic']['head36_canonical_sha256']
                        and query_row['surrogate_backward_identity'] == query.startswith('png_surrogate_')
                        and query_row['true_delivered_PNG_derivative'] is False)
                error = c.scalar('actual_saved_query_scalar_' + str(slot) + query, scalars64[query], query_row['scalar'])
                scalar_rows.append({'slot': slot, 'id': cid, 'role': row['role'], 'query': query,
                                    'metric_region': call['metric_region'], 'actual_photographic_target': row['target'],
                                    'reported_Torch_FP32_scalar': query_row['scalar'], 'NumPy_FP64_scalar': scalars64[query],
                                    'NumPy_FP32_scalar': scalars32[query], 'absolute_FP64_difference': error,
                                    'gradient_file_binding_only_not_vector_geometry': query_row['file']})
        raw_metrics = scalar_metrics(arrays['raw'], target, row['role'])
        png_metrics = scalar_metrics(normalized(arrays['png']), target, row['role'])
        pixel_rows.append({'slot': slot, 'id': cid, 'role': row['role'], 'phase': call['phase'],
                           'RAW_floor_PNG_exact': bool(np.array_equal(floor_png(arrays['raw']), arrays['png'])),
                           'raw_photographic': raw_metrics, 'decoded_PNG_photographic': png_metrics,
                           'RAW_value_sha256': digest(arrays['raw'].tobytes()), 'PNG_RGB8_sha256': digest(arrays['png'].tobytes())})
    c.check('finite_exact72_64_256_8census', len(pixel_rows) == 72 and len(scalar_rows) == 256 and exact_surr == forward_count == 64
            and stock_count == 24 and len(parity_rows) == len(repeats) == 4)
    c.check('no_neural_framework_loaded', not any(name.split('.')[0] in {'torch', 'torchvision', 'tensorflow', 'jax'} for name in sys.modules))
    consumed = r.after()
    failures = [row for row in c.rows if not row['pass']]
    result = {'complete': True, 'saved_pixel_scalar_consistency_pass': not failures,
              'format': 'u64-png-surrogate-independent-saved-pixels-v1', 'protocol_sha256': PROTOCOL_SHA,
              'counts': {'RAW_floor_PNG_pairs': len(pixel_rows), 'actual_PNG_decoded_FP32_surrogate_forward_references': exact_surr,
                         'photographic_query_scalars': len(scalar_rows), 'historical_u64_RAW_PNG_parities': len(parity_rows),
                         'full_six_output_repeats': len(repeats), 'retained_native_logits_LQ_triples': stock_count // 3,
                         'consumed_files_hashed_before_after': len(consumed)},
              'checks': len(c.rows), 'failed_checks': failures, 'historical_parities': parity_rows, 'same_state_repeats': repeats,
              'maximum_query_scalar_FP64_difference': max(row['absolute_FP64_difference'] for row in scalar_rows),
              'scalar_arithmetic_readback_bound_only': ARITHMETIC_BOUND, 'quality_tolerance_added': 0,
              'maximum_native_clamp_resize_difference': max_raw_resize, 'maximum_unclipped_resize_difference': max_unclipped_resize,
              'resize_FP32_arithmetic_bound_only': RESIZE_BOUND, 'all_consumed_files_unchanged_after': True,
              'actual_identity_backward_independently_replayed': False,
              'biased_identity_backward_saved_source_evidence_only': {'worker': {'path': worker_name, **p['files'][worker_name]},
                                                                    'surrogate': {'path': surrogate_name, **p['files'][surrogate_name]}},
              'true_PNG_derivative_claimed': False, 'image_improvement_demonstrated_by_this_readback': False,
              'zero_update_measurement_not_quality_success': True, 'new_quality_gate_evaluation_or_threshold_change': False,
              'model_qualified': False, 'app_incumbent_unchanged': True, 'original_u64_retained': True,
              'neural_gradient_optimizer_parameter_application_training_VM_browser_calls': 0,
              'scope': 'Actual saved pixel/scalar and exact PNG-forward consistency only. Zero changed weights do not establish improvement; no finite-step, PNG-quality, identity, native CCTV, DEV or FINAL qualification follows.'}
    out = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_pixels'; out.mkdir(parents=True, exist_ok=True)
    artifacts = {'receipt_v1.json': result, 'pixel_rows_v1.json': {'complete': True, 'rows': pixel_rows},
                 'scalar_rows_v1.json': {'complete': True, 'rows': scalar_rows},
                 'checks_v1.json': {'complete': True, 'rows': c.rows},
                 'consumed_files_v1.json': {'complete': True, 'all_before_after_exact': True, 'files': consumed},
                 'source_array_tests_v1.json': {'complete': True, 'all_tests_pass': True, 'fixtures': test_rows,
                    'audit_source': {'path': Path(__file__).resolve().relative_to(ROOT).as_posix(),
                                     **r.consumed[Path(__file__).resolve().relative_to(ROOT).as_posix()]}}}
    for name, value in artifacts.items():
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(run())
