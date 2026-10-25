"""Saved-only auditor for a real returned complete-U96 zero-optimizer preflight.

NumPy/PIL/hash/JSON only. No Torch import, unpickling, model construction,
network, browser, VM or training action exists. A return archive is never
extracted. --help and --self-check do not consume or manufacture VM evidence.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'cctv_codeformer_u96_postfusion_preflight_v1_r1_vm'
FORMAT = 'complete-U96-postfusion-zero-optimizer-preflight-v1-r1'
O = 'outputs/u96_postfusion_preflight_v1_r1'
SITES = ('32', '64', '128', '256')
CHANNELS = {'32': 256, '64': 256, '128': 128, '256': 128}
REFS = ('tr_ffhq_00084', 'tr_ffhq_00178', 'tr_ffhq_00323', 'tr_ffhq_00616')
R2_STATE = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_STATE = 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
MAX_ARCHIVE = 160 * 1024**2
MAX_EXPANDED = 192 * 1024**2
MAX_MEMBER = 16 * 1024**2
MAX_MEMBERS = 500
LIMITS = {'full_forward_attempts': 20, 'full_forward_returns': 20, 'reference_forwards': 8,
          'candidate_parity_forwards': 8, 'candidate_gradient_forwards': 4, 'gradient_queries': 4,
          'optimizer_constructions': 0, 'optimizer_updates': 0, 'internal_seconds': 1200,
          'external_seconds': 1260, 'per_phase_seconds': 180, 'initial_free_disk_bytes': 6 * 1024**3,
          'continuing_disk_reserve': 4 * 1024**3, 'maximum_RSS_bytes': 10 * 1024**3,
          'maximum_CUDA_reserved_bytes': 18 * 1024**3, 'maximum_output_bytes': 128 * 1024**2}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative(name):
    need(type(name) is str and name and '\\' not in name and ':' not in name,
         'Canonical relative POSIX path required')
    value = PurePosixPath(name)
    need(not value.is_absolute() and value.as_posix() == name and name != '.' and
         all(x not in ('', '.', '..') for x in value.parts), 'Archive path escape refused')
    return name


def read_local(path, cap):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        info = part.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Linked/reparse evidence refused')
    before = path.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_size <= cap, 'Bounded ordinary evidence required')
    data = path.read_bytes(); after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'Evidence changed while reading')
    return data


def local_pin(path, cap=MAX_ARCHIVE):
    path = Path(path).absolute(); data = read_local(path, cap)
    return {'absolute_path': str(path), 'bytes': len(data), 'sha256': sha(data)}


def strict_json(data):
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'Duplicate JSON key: ' + k); result[k] = v
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    def finite(item):
        if type(item) is float:
            need(math.isfinite(item), 'Finite JSON number required')
        elif type(item) is dict:
            for child in item.values():
                finite(child)
        elif type(item) is list:
            for child in item:
                finite(child)
    finite(value)
    return value


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


class CountingReader:
    def __init__(self, stream):
        self.stream = stream; self.total = 0

    def read(self, size=-1):
        remaining = MAX_EXPANDED - self.total
        need(remaining >= 0, 'Bounded decompression cap reached')
        size = min(size, remaining + 1) if size >= 0 else remaining + 1
        value = self.stream.read(size); self.total += len(value)
        need(self.total <= MAX_EXPANDED, 'Bounded decompression cap exceeded')
        return value


class ReturnArchive:
    def __init__(self, path, expected_sha):
        compressed = read_local(path, MAX_ARCHIVE)
        need(sha(compressed) == expected_sha, 'Returned archive SHA differs')
        self.binding = local_pin(path); self.files = {}; self.total = 0
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode='rb') as expanded:
            reader = CountingReader(expanded)
            with tarfile.open(fileobj=reader, mode='r|') as tar:
                for member in tar:
                    need(len(self.files) < MAX_MEMBERS and member.isfile() and not member.issparse(),
                         'Only a bounded list of ordinary TAR files is permitted')
                    need(0 <= member.size <= MAX_MEMBER, 'Bounded TAR member required')
                    name = relative(member.name)
                    need(name.startswith(STAGE + '/'), 'Wrong returned stage root')
                    name = relative(name[len(STAGE) + 1:])
                    need(name not in self.files, 'Duplicate TAR path')
                    self.total += member.size
                    need(self.total <= MAX_ARCHIVE, 'Returned payload cap exceeded')
                    stream = tar.extractfile(member)
                    need(stream is not None, 'Missing TAR member stream')
                    value = stream.read(member.size + 1)
                    need(len(value) == member.size, 'TAR member length differs')
                    self.files[name] = value
            # Reach the actual gzip EOF, validating its trailer/CRC as well as
            # the cumulative inflation limit; tar's end blocks alone are not EOF.
            while reader.read(1024**2):
                pass
            self.expanded_stream_bytes = reader.total
        need(self.files, 'Empty return archive')

    def pin(self, name):
        name = relative(name); need(name in self.files, 'Missing returned file: ' + name)
        value = self.files[name]
        return {'path': name, 'bytes': len(value), 'sha256': sha(value)}

    def bound(self, binding):
        need(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'}, 'Exact returned pin required')
        need(self.pin(binding['path']) == binding, 'Returned file binding differs: ' + binding['path'])
        return self.files[binding['path']]

    def doc(self, name):
        self.pin(name)
        return strict_json(self.files[name])

    def bound_doc(self, binding):
        return strict_json(self.bound(binding))


def array_bytes(data, shape, allow_nonfinite=False):
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    need(version in ((1, 0), (2, 0)), 'Known bounded NPY header version required')
    header = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
              else np.lib.format.read_array_header_2_0(stream))
    declared_shape, fortran, dtype = header
    need(declared_shape == tuple(shape) and not fortran and dtype == np.dtype('float32') and
         len(data) - stream.tell() == math.prod(shape) * 4, 'Exact bounded NPY shape/dtype/payload required before allocation')
    stream.seek(0)
    value = np.load(stream, allow_pickle=False)
    need(isinstance(value, np.ndarray) and stream.tell() == len(data), 'Exactly one NPY array with no trailing bytes required')
    need(value.dtype == np.float32 and value.shape == tuple(shape) and value.flags.c_contiguous,
         'Exact C-contiguous FP32 tensor shape required')
    need(allow_nonfinite or np.isfinite(value).all(), 'Nonfinite saved array')
    return value


def rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        need(image.mode == 'RGB' and image.size == (256, 256) and getattr(image, 'n_frames', 1) == 1,
             'One RGB256 PNG required')
        return np.asarray(image).copy()


def head_shapes():
    result = {}
    for site in SITES:
        c = CHANNELS[site]
        entries = {'encoder.weight': (32, c, 1, 1), 'encoder.bias': (32,),
                   'decoder.weight': (32, c, 1, 1), 'fused.weight': (32, c, 1, 1),
                   'coarse.weight': (32, 256, 1, 1), 'spatial.weight': (32, 32, 3, 3),
                   'spatial.bias': (32,), 'output.weight': (c, 32, 1, 1), 'output.bias': (c,)}
        result.update({'heads.' + site + '.' + k: v for k, v in entries.items()})
    need(len(result) == 36 and sum(math.prod(s) for s in result.values()) == 168960,
         'Independent head census differs')
    return result


def canonical_heads(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        header = json.dumps([name, 'torch.float32', list(value.shape)], separators=(',', ':')).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header); digest.update(value.tobytes())
    return digest.hexdigest()


def head_npz(view, declaration):
    need(set(declaration) == {'file', 'arrays', 'canonical_sha256'}, 'Complete head-NPZ declaration required')
    data = view.bound(declaration['file']); schema = head_shapes()
    # Inspect ZIP sizes/names before NumPy can allocate any member; no pickle is interpreted.
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        need(len(zipped.namelist()) == 36 and set(zipped.namelist()) == {k + '.npy' for k in schema},
             'Exactly36 named head NPY members required')
        need(sum(i.file_size for i in zipped.infolist()) < 1024**2 and
             all(i.file_size < 256 * 1024 for i in zipped.infolist()), 'Bounded head NPZ required before decompression')
        need(zipped.testzip() is None, 'CRC-valid head NPZ required')
        arrays = {k: array_bytes(zipped.read(k + '.npy'), s) for k, s in schema.items()}
    descriptors = {r['name']: r for r in declaration['arrays']}
    need(len(descriptors) == len(declaration['arrays']) == 36, 'Exactly36 unique head descriptors required')
    for name, value in arrays.items():
        row = descriptors[name]
        need(row['shape'] == list(value.shape) and row['dtype'] == 'float32' and row['bytes_sha256'] == sha(value.tobytes()),
             'Head array descriptor differs: ' + name)
        if '.output.' in name:
            need(not np.count_nonzero(value), 'All final head projections must remain untrained zero')
    need(canonical_heads(arrays) == declaration['canonical_sha256'], 'Head canonical tensor digest differs')
    return arrays


def record(value):
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': int(value.nbytes),
            'RGB_or_tensor_bytes_sha256': sha(value.tobytes()), 'minimum': float(value.min()),
            'maximum': float(value.max()), 'negative_zero_count': int(np.count_nonzero((value == 0) & np.signbit(value)))}


def compare(left, right):
    return {'exact_value_equal': bool(np.array_equal(left, right)), 'exact_bits_equal': left.tobytes() == right.tobytes(),
            'maximum_absolute_error': float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))),
            'unequal_values': int(np.count_nonzero(left != right)),
            'signed_zero_mismatches': int(np.count_nonzero((left == 0) & (right == 0) & (np.signbit(left) != np.signbit(right)))),
            'reference': record(left), 'candidate': record(right)}


def close(actual, expected, label, atol=2e-6, rtol=2e-6):
    need(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
         math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol), 'Numeric receipt differs: ' + label)


def delivery_from_native(native):
    # PyTorch align_corners=False512->256 uses four half-pixel samples per output.
    # FP32 CUDA summation ordering can differ from NumPy; PNG is instead recomputed
    # exactly from the actual saved RAW, and tensor parity is always exact.
    scaled = (native[0] + np.float32(1)) / np.float32(2)
    def down(x):
        return ((x[:, 0::2, 0::2] + x[:, 0::2, 1::2] + x[:, 1::2, 0::2] + x[:, 1::2, 1::2]) * np.float32(.25)).transpose(1, 2, 0).copy()
    return down(np.clip(scaled, 0, 1)), down(scaled)


def objective(raw, target):
    residual = raw.astype(np.float64) - target.astype(np.float64)
    inside = np.zeros((256, 256), bool); inside[64:192, 64:192] = True
    terms = {}; counts = {}
    for name, region in (('central128', inside), ('outside', ~inside)):
        terms[name + '_image_L1'] = float(np.abs(residual[region]).mean())
        h = region[:, :-1] & region[:, 1:]; v = region[:-1, :] & region[1:, :]
        numerator = np.abs(np.diff(residual, axis=1))[h].sum() + np.abs(np.diff(residual, axis=0))[v].sum()
        denominator = int(h.sum() + v.sum()) * 3
        terms[name + '_adjacent_residual_L1'] = float(numerator / denominator)
        counts[name] = {'pixels': int(region.sum()), 'horizontal_pairs': int(h.sum()), 'vertical_pairs': int(v.sum())}
    image = .5 * (terms['central128_image_L1'] + terms['outside_image_L1'])
    adjacent = .5 * (terms['central128_adjacent_residual_L1'] + terms['outside_adjacent_residual_L1'])
    terms['equal_mean_image_L1'] = image; terms['equal_mean_adjacent_residual_L1'] = adjacent
    return image + .25 * adjacent, terms, counts


def image_metrics(raw, target):
    diff = raw.astype(np.float64) - target.astype(np.float64)
    bias = diff.mean(axis=(0, 1)); luma = diff @ np.array([.2126, .7152, .0722])
    return {'MAE': float(np.abs(diff).mean()), 'MSE': float(np.square(diff).mean()),
            'signed_luminance_bias': float(luma.mean()), 'absolute_luminance_bias': abs(float(luma.mean())),
            'mean_centered_MAE': float(np.abs(diff - bias).mean()),
            'adjacent_residual_MAE': float((np.abs(np.diff(diff, axis=0)).sum() + np.abs(np.diff(diff, axis=1)).sum()) /
                                           (3 * (256 * 255 * 2)))}


def protocol_and_data(view, protocol_sha):
    p = view.doc('protocol.json')
    need(view.pin('protocol.json')['sha256'] == protocol_sha, 'Exact sealed protocol SHA required')
    need(p['format'] == FORMAT and p['stage_name'] == STAGE and p['limits'] == LIMITS, 'Known finite preflight protocol required')
    need(p['manual_launch_only'] and not p['automatic_retry_resume'] and
         p['optimizer_constructions'] == p['optimizer_updates'] == 0, 'Zero-optimizer manual contract required')
    omitted = []
    for name, binding in p['files'].items():
        relative(name)
        if name.startswith('weights/'):
            need(name not in view.files, 'Large frozen weights should be omitted from return'); omitted.append(name)
        else:
            actual = view.pin(name)
            need({k: actual[k] for k in ('bytes', 'sha256')} == binding, 'Static source/evidence changed: ' + name)
    need(set(omitted) == {p['model']['r2']['path'], p['model']['u96']['path']}, 'Only two frozen initializer weights may be omitted')
    for section in ('prototype', 'data'):
        for binding in p[section].values():
            view.bound(binding)
    witnesses = p['witnesses']
    need([(r['reference_id'], r['role'], r['id']) for r in witnesses] ==
         [(ref, role, ref + '__' + role) for ref in REFS for role in ('clear', 'local_gaussian_1p2')],
         'Fixed four-TRAIN-pair witness order required')
    need(sum(r['gradient_witness'] is True for r in witnesses) == 4 and
         all(r['gradient_witness'] == (r['role'] == 'local_gaussian_1p2') for r in witnesses), 'Four local gradient witnesses required')
    candidate = view.bound_doc(p['data']['provenance']); admission = view.bound_doc(p['data']['admission'])
    reviewer = view.bound_doc(p['data']['review']); old_audit = view.bound_doc(p['data']['audit'])
    need(candidate['optimization_permitted'] is admission['optimization_permitted'] is False and old_audit['independent_preparation_audit_pass'],
         'Non-optimizing independently reviewed witnesses required')
    for original_pin, staged_pin in ((admission['candidate_manifest'], p['data']['provenance']),
                                     (admission['reviewer_record'], p['data']['review']),
                                     (old_audit['candidate_manifest'], p['data']['provenance']),
                                     (old_audit['input_admission'], p['data']['admission'])):
        need(all(original_pin[k] == staged_pin[k] for k in ('bytes', 'sha256')), 'Original-to-portable evidence pin join differs')
    need(admission['counts'] == {'readable': 8} and admission['readable_reference_ids'] == list(REFS), 'Prospective eight-readable input decision required')
    need(reviewer['clean_references_or_model_outputs_seen'] is False and reviewer['sheets_viewed'] == 2, 'Input-only admission scope differs')
    verdict = {r['anonymous_id']: r for r in reviewer['decisions']}
    need(len(verdict) == len(reviewer['decisions']) == len(admission['cases']) == 8, 'Eight unique prospective input verdicts required')
    candidate_cases = {r['id']: r for r in candidate['cases']}
    need(len(candidate_cases) == 12 and candidate['source_count'] == 4 and candidate['input_count'] == 12, 'Preserved12input TRAIN preparation required')
    for row in admission['cases']:
        need(row['decision'] == verdict[row['anonymous_id']]['decision'] == 'readable' and
             row['reason'] == verdict[row['anonymous_id']]['reason'], 'Input verdict join differs')
        need(row['original_input'] == candidate_cases[row['id']]['input'], 'Anonymous input original pin differs')
    values = {}
    for row in witnesses:
        need(row['reference_id'] in REFS and row['id'] in candidate_cases, 'TRAIN witness absent from preparation')
        original = candidate_cases[row['id']]
        need(original['split'] == 'TRAIN' and not original['optimization_permitted'], 'TRAIN boundary changed')
        inp = view.bound(row['input']); target = view.bound(row['target'])
        need(sha(inp) == original['input']['sha256'] and sha(target) == original['target']['sha256'], 'Portable original/staged asset join differs')
        pixels, clean = rgb(inp), rgb(target)
        need(sha(pixels.tobytes()) == row['input_RGB8_sha256'] and sha(clean.tobytes()) == row['target_RGB8_sha256'], 'Witness RGB hash differs')
        if row['role'] == 'clear':
            need(inp == target and row['input'] == row['target'], 'Exact clear control required')
        else:
            outside = np.ones((256, 256), bool); outside[64:192, 64:192] = False
            need(np.array_equal(pixels[outside], clean[outside]), 'Local corruption alters outside pixels')
        values[row['id']] = (pixels, clean.astype(np.float32) / np.float32(255))
    return p, values


def expected_namespace(p):
    expected = {name for name in p['files'] if not name.startswith('weights/')} | {'protocol.json', 'run.log', 'supervisor.json'}
    expected |= {O + '/' + name for name in ('started.json', 'preflight.json', 'events.jsonl', 'resources.jsonl',
                  'initial_heads.pt', 'final_heads.pt', 'initial_heads.npz', 'final_heads.npz', 'state_invariants.json', 'result.json')}
    for witness in p['witnesses']:
        folder = O + '/parity/' + witness['id'] + '/'
        expected.add(folder + 'parity.json')
        for arm in ('reference', 'candidate'):
            expected |= {folder + arm + '-' + label + '.npy' for label in ('native512', 'logits', 'LQ', 'raw', 'unclipped')}
            expected.add(folder + arm + '.png')
        if witness['gradient_witness']:
            folder = O + '/gradient/' + witness['id'] + '/'
            expected |= {folder + name for name in ('candidate-raw.npy', 'candidate.png', 'gradient.json')}
            expected |= {folder + 'gradients/' + name + '.npy' for name in head_shapes()}
    return expected


def full_namespace(view, p, result):
    expected = expected_namespace(p)
    need(set(view.files) == expected, 'Full returned output namespace differs; missing=' + str(sorted(expected - set(view.files))) +
         '; extra=' + str(sorted(set(view.files) - expected)))
    need(len(result['parity_receipts']) == 8 and len(result['gradient_receipts']) == 4, 'Eight parity/four gradient receipts required')


def audit_parity(view, p, result, values):
    by_id = {}; metrics = []; delivery_max_error = 0.; pairs_count = 0
    for witness, pin in zip(p['witnesses'], result['parity_receipts'], strict=True):
        receipt = view.bound_doc(pin); id_ = witness['id']
        need(pin['path'] == O + '/parity/' + id_ + '/parity.json' and receipt['complete'] and receipt['id'] == id_, 'Parity receipt order/path differs')
        need(receipt['input'] == witness['input'] and receipt['target'] == witness['target'] and receipt['role'] == witness['role'], 'Parity input provenance differs')
        need(receipt['RNG_before'] == receipt['RNG_after'] and receipt['RNG_exact'], 'Parity RNG receipt changed')
        need(not receipt['model_qualified'] and receipt['same_device_reference_not_CPU_L4_parity'], 'Parity claim exceeds scope')
        arrays = {}
        for arm in ('reference', 'candidate'):
            saved = receipt['saved'][arm]
            need(set(saved) == {'native512', 'logits', 'LQ', 'raw', 'unclipped', 'PNG'}, 'Exact saved stock/delivery keys required')
            arrays[arm] = {name: array_bytes(view.bound(saved[name]), shape) for name, shape in (
                ('native512', (1, 3, 512, 512)), ('logits', (1, 256, 1024)), ('LQ', (1, 256, 16, 16)),
                ('raw', (256, 256, 3)), ('unclipped', (256, 256, 3)))}
            raw = arrays[arm]['raw']; need(raw.min() >= 0 and raw.max() <= 1, 'Clipped delivery must be normalized')
            png = rgb(view.bound(saved['PNG']))
            need(np.array_equal(png, np.floor(raw * np.float32(255)).astype(np.uint8)), 'Exact floorRGB8 delivery differs')
            calculated = delivery_from_native(arrays[arm]['native512'])
            for name, expected in zip(('raw', 'unclipped'), calculated, strict=True):
                error = float(np.max(np.abs(expected.astype(np.float64) - arrays[arm][name].astype(np.float64))))
                need(np.allclose(expected, arrays[arm][name], atol=3e-7, rtol=3e-7),
                     'Native-to-delivery independent FP32 reduction check differs')
                delivery_max_error = max(delivery_max_error, error)
            metrics.append({'id': id_, 'role': witness['role'], 'arm': arm, 'format': 'RAW', **image_metrics(raw, values[id_][1])})
            metrics.append({'id': id_, 'role': witness['role'], 'arm': arm, 'format': 'PNG',
                            **image_metrics(png.astype(np.float32) / np.float32(255), values[id_][1])})
        for name in ('native512', 'logits', 'LQ', 'raw', 'unclipped'):
            left, right = arrays['reference'][name], arrays['candidate'][name]
            if name in ('raw', 'unclipped'):
                left, right = left.transpose(2, 0, 1)[None].copy(), right.transpose(2, 0, 1)[None].copy()
            actual = compare(left, right)
            need(receipt['comparisons'][name] == actual and actual['exact_value_equal'], 'Independent exact parity differs: ' + id_ + '/' + name)
            pairs_count += 1
        need(receipt['PNG_bytes_exact'] and view.bound(receipt['saved']['reference']['PNG']) == view.bound(receipt['saved']['candidate']['PNG']), 'Parity PNG bytes differ')
        by_id[id_] = {'receipt': receipt, 'arrays': arrays}
    return by_id, metrics, pairs_count, delivery_max_error


def audit_gradients(view, p, result, values, parity):
    records = []; gradient_arrays = 0; norm_max_error = 0.; objective_max_error = 0.
    witnesses = [r for r in p['witnesses'] if r['gradient_witness']]
    schema = head_shapes()
    for witness, pin in zip(witnesses, result['gradient_receipts'], strict=True):
        r = view.bound_doc(pin); id_ = witness['id']
        need(pin['path'] == O + '/gradient/' + id_ + '/gradient.json' and r['complete'] and r['id'] == id_ and r['role'] == 'local_gaussian_1p2', 'Gradient receipt order/path differs')
        raw = array_bytes(view.bound(r['saved']['raw']), (256, 256, 3)); png = rgb(view.bound(r['saved']['PNG']))
        need(np.array_equal(png, np.floor(raw * np.float32(255)).astype(np.uint8)), 'Gradient PNG delivery differs')
        need(np.array_equal(raw, parity[id_]['arrays']['candidate']['raw']), 'Zero-update gradient forward output differs from parity output')
        need(view.bound(r['saved']['PNG']) == view.bound(parity[id_]['receipt']['saved']['candidate']['PNG']),
             'Zero-update grad-enabled PNG bytes differ from parity PNG')
        need(r['all_joint_gradients_finite_gate_pass'] and r['output_projection_gradient_gate_pass'] and
             r['gradient_arrays_saved'] == 36 and r['connected_clamp_resize_delivery'], 'Saved gradient gates/array census differ')
        objective_, terms, counts = objective(raw, values[id_][1])
        close(r['objective'], objective_, id_ + '/objective', atol=2e-7)
        objective_max_error = max(objective_max_error, abs(r['objective'] - objective_))
        need(set(r['terms']) == set(terms), 'Objective term schema differs')
        for name, value in terms.items():
            close(r['terms'][name], value, id_ + '/' + name, atol=2e-7)
        gradients = {row['name']: row for row in r['gradients']}
        need(len(gradients) == len(r['gradients']) == 36 and set(gradients) == set(schema), 'Exactly36 named head gradients required')
        norm_by_name = {}
        for name, shape in schema.items():
            row = gradients[name]; array = array_bytes(view.bound(row['array']), shape)
            need(row['array']['path'] == O + '/gradient/' + id_ + '/gradients/' + name + '.npy', 'Gradient array path differs')
            need(row['shape'] == list(shape) and row['array_shape'] == list(shape) and row['array_dtype'] == 'float32', 'Gradient array shape declaration differs')
            norm = float(np.linalg.norm(array.astype(np.float64).ravel()))
            max_ = float(np.max(np.abs(array.astype(np.float64)))); nonzero = int(np.count_nonzero(array))
            close(row['L2_norm'], norm, name + '/L2', atol=1e-10, rtol=2e-10)
            close(row['maximum_absolute'], max_, name + '/max', atol=1e-12, rtol=1e-12)
            need(row['nonzero_count'] == nonzero and row['finite'], 'Gradient finite/nonzero receipt differs')
            if 'nonfinite_count' in row:
                need(row['nonfinite_count'] == 0, 'Nonfinite gradient count differs')
            norm_max_error = max(norm_max_error, abs(row['L2_norm'] - norm)); norm_by_name[name] = norm; gradient_arrays += 1
        for site in SITES:
            projection = math.sqrt(sum(norm_by_name['heads.' + site + '.' + kind]**2 for kind in ('output.weight', 'output.bias')))
            need(projection > 0, 'A final output projection lacks a nonzero gradient')
            close(r['output_projection_L2_by_site'][site], projection, site + '/projection', atol=1e-10, rtol=2e-10)
        native = parity[id_]['arrays']['candidate']['native512']; clipping = r['clipping']
        need(clipping['native_channels_below_minus1'] == int((native < -1).sum()) and
             clipping['native_channels_above1'] == int((native > 1).sum()) and clipping['native_channel_count'] == native.size,
             'Clipping count differs from zero-state native output')
        # NCHW records precede HWC persistence; match the actual value ordering.
        for name, data in (('unclipped256', parity[id_]['arrays']['candidate']['unclipped']), ('clipped_RAW256', raw)):
            need(clipping[name] == record(data.transpose(2, 0, 1)[None].copy()), 'Clipping tensor record differs')
        need(r['RNG_before'] == r['RNG_after'] and r['RNG_exact'] and r['sites'] == list(SITES), 'Gradient RNG/site receipt differs')
        need(r['frozen_grad_fields_None'] == 539 and r['all_parameter_grad_fields_None'] == 575 and
             r['optimizer_updates'] == r['weight_perturbations'] == 0 and not r['model_qualified'], 'Gradient state/quality declaration differs')
        scope = r['gradient_determinism']
        need(scope['id'] == id_ and scope['before'] == scope['after'] == {'enabled': True, 'warn_only': False} and
             scope['during'] == {'enabled': True, 'warn_only': True} and scope['restored_exact'] and
             not scope['deterministic_gradient_replay_claim'] and scope['warn_only_applies_to_all_operations_inside_this_query_scope'],
             'Narrow gradient-only warn scope or strict-mode restoration differs')
        unexpected = [w for w in scope['warnings'] if
                      ('deterministic' in w['message'].lower() or 'nondeterminism' in w['message'].lower()) and
                      re.match(r'^upsample_bilinear2d_backward_out_cuda\b', w['message']) is None]
        need(scope['unexpected_nondeterminism_warnings'] == unexpected == [], 'Unexpected nondeterministic gradient warning')
        need(result['gradient_determinism_scopes'][len(records)] == scope, 'Gradient scope receipt differs from terminal result')
        records.append({'id': id_, 'objective': objective_, 'terms': terms, 'independent_region_counts': counts,
                        'arrays': 36, 'projection_gradients_nonzero_all4': True,
                        'gradient_values_recomputed_without_autograd_reexecution': True})
    return records, gradient_arrays, norm_max_error, objective_max_error


def audit_states(view, p, result):
    state = view.bound_doc(result['state_invariants']); preflight = view.doc(O + '/preflight.json')
    started = view.doc(O + '/started.json')
    need(started['protocol_sha256'] == result['protocol_sha256'] and started['stage'] == STAGE and
         started['status'] == 'STARTED_NOT_TRAINED' and started['host']['system'] == 'Linux' and
         started['host']['tmux'] and started['host']['initial_free_disk_bytes'] >= LIMITS['initial_free_disk_bytes'] and
         started['training_optimizer_updates'] == 0 and not started['model_qualified'], 'Manual Linux VM start receipt differs')
    need(preflight['complete'] and preflight['protocol_sha256'] == result['protocol_sha256'] and
         preflight['status'] == 'PRE_FORWARD_INTEGRITY_ONLY' and preflight['GPU'] == 'NVIDIA L4' and
         preflight['limits'] == LIMITS and preflight['optimizer_constructions'] == preflight['optimizer_updates'] == 0 and
         not preflight['model_qualified'] and not preflight['actual_training_permitted'], 'Pre-forward integrity receipt differs')
    need(state['complete'] and state['unchanged'] and state['before'] == state['after'] == preflight['initial_states'], 'Frozen state before/after claims differ')
    for arm, count in (('reference', 539), ('candidate', 575)):
        row = state['before'][arm]
        need(row['R2_canonical515_sha256'] == R2_STATE and row['U96_canonical24_sha256'] == U96_STATE and
             row['frozen_tensors'] == row['frozen_grad_fields_None'] == 539 and row['full_state_tensors'] == count,
             'Exact complete-U96 census/state claim differs')
    need(state['all539_incumbent_tensors_frozen_and_exact'] and state['all36_untrained_heads_exact'] and
         state['all575_parameter_grad_fields_None'] and state['static_sources_models_data_exact'], 'Required unchanged-state declaration missing')
    need(state['run_RNG_before'] == state['run_RNG_after'] and state['factory_RNG']['exact'] and
         state['factory_RNG']['before'] == state['factory_RNG']['after'], 'Caller RNG preservation declaration differs')
    need(state['gradient_determinism_scopes'] == result['gradient_determinism_scopes'] and
         result['strict_forward_parity_preserved'] and not result['CUDA_bilinear_gradient_determinism_claim'],
         'Forward/gradient determinism claims differ')
    for label in ('initial_heads', 'final_heads'):
        data = view.bound(state[label]); need(data.startswith(b'PK'), 'Retained head .pt must be the expected ZIP serializer')
    need(preflight['initial_heads'] == state['initial_heads'] and preflight['initial_heads_NPZ'] == state['initial_heads_NPZ'],
         'Pre-forward and final portable head-state bindings differ')
    before = head_npz(view, state['initial_heads_NPZ']); after = head_npz(view, state['final_heads_NPZ'])
    need(all(before[k].tobytes() == after[k].tobytes() for k in before), 'Any untrained head change is forbidden')
    digest = canonical_heads(before)
    need(state['before']['candidate']['heads_canonical_sha256'] == digest and
         state['before']['candidate']['head_tensors'] == 36 and state['before']['candidate']['head_parameters'] == 168960,
         'Head tensor state census/digest claim differs')
    return {'head_arrays_independently_verified': 72, 'head_tensor_values_unchanged': True,
            'head_canonical_sha256': digest, 'all_output_projections_zero': True,
            'PT_files_hash_bound_but_not_unpickled_or_tensor_parsed': True,
            'NPZ_head_tensors_associated_to_PT_by_worker_state_receipt_not_independent_PT_decode': True,
            'frozen539_tensor_digest_and_grad_None_are_source_bound_runtime_receipts_not_returned_tensor_rehash': True}


def json_lines(view, name):
    return [strict_json(line) for line in view.files[name].splitlines() if line]


def audit_runtime(view, p, result):
    events = json_lines(view, O + '/events.jsonl'); resources = json_lines(view, O + '/resources.jsonl')
    need(events and resources, 'Durable operation/resource ledgers required')
    counter = result['counters']
    exact = {'full_forward_attempts': 20, 'full_forward_returns': 20, 'reference_forwards': 8,
             'candidate_parity_forwards': 8, 'candidate_gradient_forwards': 4, 'gradient_queries': 4,
             'gradient_returns': 4, 'optimizer_constructions': 0, 'optimizer_updates': 0,
             'frozen_U96_adapter_site_reservations': 80, 'frozen_U96_adapter_site_returns': 80,
             'new_head_site_reservations': 48, 'new_head_site_returns': 48}
    need(counter == exact, 'Exact successful operation reservations/returns differ')
    kinds = Counter(r['kind'] for r in events)
    need(kinds['forward_reserved'] == kinds['forward_returned'] == 20 and
         kinds['gradient_reserved'] == kinds['gradient_returned'] == kinds['gradient_determinism_scope'] == 4,
         'Durable operation event count differs')
    scope_events = [r for r in events if r['kind'] == 'gradient_determinism_scope']
    need(all(all(event[k] == v for k, v in scope.items()) for event, scope in
             zip(scope_events, result['gradient_determinism_scopes'], strict=True)), 'Durable gradient warn-scope records differ')
    pending = None; previous = {k: 0 for k in exact}
    for event in events:
        current = event['counters']
        need(set(current) == set(exact) and all(type(v) is int and previous[k] <= v <= exact[k] for k, v in current.items()),
             'Counters must be monotone finite integer reservations')
        previous = current
        if event['kind'] == 'forward_reserved':
            need(pending is None, 'Nested full-forward reservation'); pending = event['forward_kind']
        elif event['kind'] == 'forward_returned':
            need(pending == event['forward_kind'] and event['sites'] == list(SITES), 'Returned forward lacks one matching reservation')
            pending = None
        elif event['kind'] == 'phase_after':
            need(0 <= event['phase_seconds'] <= LIMITS['per_phase_seconds'], 'Finite phase duration exceeds180s')
    need(pending is None and previous == exact, 'Unreturned final operation')
    need(all(resources[i]['seconds'] <= resources[i + 1]['seconds'] for i in range(len(resources) - 1)), 'Resource times must be monotone')
    for r in resources:
        need(0 <= r['seconds'] < LIMITS['internal_seconds'] and r['free_disk_bytes'] >= LIMITS['continuing_disk_reserve'] and
             r['CUDA_reserved_bytes'] <= LIMITS['maximum_CUDA_reserved_bytes'] and
             r['CUDA_peak_reserved_bytes'] <= LIMITS['maximum_CUDA_reserved_bytes'] and
             r['output_bytes'] <= LIMITS['maximum_output_bytes'], 'Resource receipt violates sealed limits')
        need(r['current_RSS_bytes'] <= LIMITS['maximum_RSS_bytes'] and r['peak_RSS_bytes'] <= LIMITS['maximum_RSS_bytes'],
             'Current/peak RSS receipt violates sealed limits')
    need(result['elapsed_seconds'] < LIMITS['internal_seconds'], 'Internal duration exceeds limit')
    supervisor = view.doc('supervisor.json')
    need(supervisor['complete'] and supervisor['worker_exit'] == 0 and supervisor['timeout'] is False and
         supervisor['seconds'] <= LIMITS['external_seconds'] + 20 and supervisor['protocol_sha256'] == result['protocol_sha256'],
         'Successful finite subprocess supervisor required')
    need(sum(len(v) for k, v in view.files.items() if k.startswith(O + '/')) <= LIMITS['maximum_output_bytes'],
         'Actual owned output bytes exceed128MiB')
    return {'events': len(events), 'resource_samples': len(resources),
            'full_forwards_reserved_and_returned': 20, 'reference_candidate_parity_gradient': [8, 8, 4],
            'joint_gradient_queries_and_returns': 4, 'frozen_site_reservations_and_verified_returns': 80,
            'head_site_reservations_and_verified_returns': 48,
            'maximum_recorded_current_RSS_bytes': max(r['current_RSS_bytes'] for r in resources),
            'maximum_recorded_peak_RSS_bytes': max(r['peak_RSS_bytes'] for r in resources),
            'maximum_recorded_CUDA_peak_reserved_bytes': max(r['CUDA_peak_reserved_bytes'] for r in resources),
            'minimum_recorded_free_disk_bytes': min(r['free_disk_bytes'] for r in resources),
            'runtime_telemetry_is_saved_receipt_not_independent_VM_observation': True}


def partial_failure_readback(view, p, failure, supervisor):
    """Preserve a failed run honestly, checking existing arrays without a pass claim."""
    allowed = expected_namespace(p) | {O + '/failure.json'}
    need(set(view.files).issubset(allowed), 'Unexpected file in failed returned namespace')
    arrays = []; incomplete = []; schemas = head_shapes()
    for name, data in sorted(view.files.items()):
        if not name.startswith(O + '/') or not name.endswith('.npy'):
            continue
        gradient = '/gradients/' in name
        if gradient:
            tensor_name = name.rsplit('/', 1)[1][:-4]
            need(tensor_name in schemas, 'Unknown failed-gradient tensor name')
            shape = schemas[tensor_name]
        else:
            suffix = name.rsplit('/', 1)[1]
            shape = ((1, 3, 512, 512) if suffix.endswith('-native512.npy') else
                     (1, 256, 1024) if suffix.endswith('-logits.npy') else
                     (1, 256, 16, 16) if suffix.endswith('-LQ.npy') else (256, 256, 3))
        try:
            array = array_bytes(data, shape, allow_nonfinite=gradient)
            arrays.append({'file': view.pin(name), 'shape': list(shape), 'finite': bool(np.isfinite(array).all()),
                           'nonfinite_count': int((~np.isfinite(array)).sum()),
                           'nonfinite_allowed_only_as_failed_gradient_evidence': gradient})
        except (ValueError, EOFError) as error:
            incomplete.append({'file': view.pin(name), 'error': str(error), 'partial_file_not_reinterpreted_as_valid_array': True})
    if failure is not None:
        c = failure['counters']
        need(c['optimizer_constructions'] == c['optimizer_updates'] == 0 and
             0 <= c['full_forward_returns'] <= c['full_forward_attempts'] <= 20 and
             0 <= c['gradient_returns'] <= c['gradient_queries'] <= 4, 'Failed-call reservation/return counters exceed contract')
        need(c['frozen_U96_adapter_site_reservations'] == 4 * c['full_forward_attempts'] and
             c['new_head_site_reservations'] == 4 * (c['candidate_parity_forwards'] + c['candidate_gradient_forwards']),
             'Failed site reservations differ from full-call reservations')
    return {'owned_array_files_read': len(arrays), 'saved_array_readback': arrays,
            'partial_or_invalid_NPY_files_retained': incomplete,
            'failed_phase': None if failure is None else failure['phase'],
            'restored_outer_phase': None if failure is None else failure.get('restored_outer_phase'),
            'supervisor_worker_exit': None if supervisor is None else supervisor['worker_exit'],
            'site_reservations_are_not_actual_failed_partial_site_execution_counts': True,
            'no_failed_run_is_qualified_from_export_completion': True}


def visuals(view, p, parity, outdir):
    directory = outdir / 'visuals'; directory.mkdir()
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    sheets = []
    for start in (0, 4):
        batch = p['witnesses'][start:start + 4]
        sheet = Image.new('RGB', (5 * 264 + 8, 4 * 288 + 88), (245, 244, 238)); draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), 'Zero-optimizer parity witnesses | exact256 | no trained improvement established', font=font, fill='black')
        for col, title in enumerate(('Input', 'Clean target', 'Independent U96 reference', 'New zero-head candidate', 'Gradient forward (local only)')):
            draw.text((8 + col * 264, 36), title, font=font, fill='black')
        cells_count = 0
        for i, witness in enumerate(batch):
            id_ = witness['id']; y = 66 + i * 288
            draw.text((8, y), id_, font=font, fill='black')
            saved = parity[id_]['receipt']['saved']
            cells = [rgb(view.bound(witness['input'])), rgb(view.bound(witness['target'])),
                     rgb(view.bound(saved['reference']['PNG'])), rgb(view.bound(saved['candidate']['PNG']))]
            cells_count += 2
            if witness['gradient_witness']:
                cells.append(rgb(view.files[O + '/gradient/' + id_ + '/candidate.png'])); cells_count += 1
            for col, cell in enumerate(cells):
                sheet.paste(Image.fromarray(cell), (8 + col * 264, y + 22))
        path = directory / (f'parity_witnesses_{start // 4 + 1:02d}.png'); sheet.save(path)
        sheets.append({'file': local_pin(path), 'witnesses': [r['id'] for r in batch], 'generated_PNG_cells': cells_count})
    need(sum(s['generated_PNG_cells'] for s in sheets) == 20, 'All20 saved generated PNG cells required')
    write(directory / 'manifest.json', {'complete': True, 'review_complete': False, 'sheets': sheets,
                                      'generated_PNG_cells': 20, 'quality_or_new_training_qualification': False})
    return local_pin(directory / 'manifest.json')


def self_check():
    schema = head_shapes()
    need(len(schema) == 36 and len(REFS) == 4, 'Static schema check')
    inside = np.zeros((256, 256), bool); inside[64:192, 64:192] = True
    counts = {name: {'pixels': int(r.sum()), 'adjacent_pairs': int((r[:, :-1] & r[:, 1:]).sum() + (r[:-1, :] & r[1:, :]).sum())}
              for name, r in (('central128', inside), ('outside', ~inside))}
    need(counts['central128'] == {'pixels': 16384, 'adjacent_pairs': 32512}, 'Fixed central region count')
    need('torch' not in sys.modules, 'Self-check must remain Torch-free')
    print(json.dumps({'static_self_check_pass': True, 'head_tensors': 36, 'head_parameters': 168960,
                      'region_counts': counts, 'VM_returns_consumed_or_fabricated': 0,
                      'model_HTTP_VM_gradient_optimizer_training_calls': 0}))


def main(args):
    outdir = Path(args.out).absolute()
    need(outdir.resolve().is_relative_to(ROOT) and not outdir.exists() and not outdir.is_symlink(),
         'A new exclusive audit output directory within the workspace is required')
    view = ReturnArchive(args.archive, args.archive_sha)
    export_data = read_local(args.export_json, 1024**2); export = strict_json(export_data)
    need(export['complete'] and export['archive_sha256'] == args.archive_sha and export['bytes'] == view.binding['bytes'] and
         export['protocol_sha256'] == args.protocol_sha and export['training_success_not_implied'] and
         export['optimizer_updates'] == 0 and not export['model_qualified'], 'Bound matching export receipt required')
    need(export['payload_files'] == len(view.files) and export['payload_bytes'] == view.total, 'Export payload count/bytes differ')
    p, values = protocol_and_data(view, args.protocol_sha)
    need(export['result_present'] == (O + '/result.json' in view.files) and export['failure_present'] == (O + '/failure.json' in view.files),
         'Export success/failure presence receipt differs')
    if O + '/result.json' not in view.files or O + '/failure.json' in view.files:
        failure = view.doc(O + '/failure.json') if O + '/failure.json' in view.files else None
        supervisor = view.doc('supervisor.json') if 'supervisor.json' in view.files else None
        if failure is not None:
            need(failure['protocol_sha256'] == args.protocol_sha and not failure['complete'] and not failure['model_qualified'] and
                 failure['training_optimizer_updates'] == 0, 'Failed return receipt differs')
        partial = partial_failure_readback(view, p, failure, supervisor)
        outdir.mkdir(parents=True)
        write(outdir / 'returned_failure_readback_v1.json', {'complete': True, 'status': 'FAILED_OR_INCOMPLETE_RETURN_PRESERVED',
              'archive': view.binding, 'export_receipt': local_pin(args.export_json, 1024**2), 'protocol_sha256': args.protocol_sha,
              'static_nonweight_files_hash_verified': True, 'failure': failure, 'supervisor': supervisor,
              'partial_saved_array_and_counter_readback': partial,
              'partial_return_file_pins': [view.pin(k) for k in sorted(view.files)],
              'full_runtime_artifact_audit_pass': False, 'do_not_rerun_or_add_epochs_automatically': True,
              'model_qualified': False, 'auditor_model_HTTP_VM_training_calls': 0})
        print(json.dumps({'failed_or_incomplete_return': True, 'full_runtime_artifact_audit_pass': False,
                          'model_qualified': False})); return
    result = view.doc(O + '/result.json')
    need(result['complete'] and result['diagnostic_pass'] and result['protocol_sha256'] == args.protocol_sha and
         not result['model_qualified'] and not result['useful_restoration_proven'] and not result['automatic_training_authorized'] and
         result['actual_optimizer_steps_or_weight_changes'] == 0 and result['DEV_FINAL_native_evidence'] is False,
         'Successful zero-optimizer result must retain all claim limits')
    full_namespace(view, p, result)
    parity, metrics, pairs, max_delivery_error = audit_parity(view, p, result, values)
    gradient, arrays, max_norm_error, max_objective_error = audit_gradients(view, p, result, values, parity)
    states = audit_states(view, p, result); runtime = audit_runtime(view, p, result)
    outdir.mkdir(parents=True)
    visual_manifest = visuals(view, p, parity, outdir)
    with (outdir / 'paired_witness_metrics_v1.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    summary = {'complete': True, 'saved_return_artifact_audit_pass': True,
               'auditor': local_pin(Path(__file__).resolve(), 1024**2), 'archive': view.binding,
               'export_receipt': local_pin(args.export_json, 1024**2), 'protocol': view.pin('protocol.json'),
               'result': view.pin(O + '/result.json'), 'source_nonweight_files_verified': len(p['files']) - 2,
               'returned_file_count': len(view.files), 'returned_payload_bytes': view.total,
               'parity_cases': 8, 'exact_compared_tensor_pairs': pairs, 'saved_generated_PNGs': 20,
               'independent_native_to_delivery_max_FP32_rounding_error': max_delivery_error,
               'PNG_floor_conversion_recomputed_exact': True, 'gradient_witnesses': gradient,
               'full_gradient_arrays_recomputed': arrays, 'maximum_gradient_norm_reduction_difference': max_norm_error,
               'maximum_objective_FP32_vs_float64_reduction_difference': max_objective_error,
               'states': states, 'runtime': runtime, 'metrics': local_pin(outdir / 'paired_witness_metrics_v1.csv'),
               'visual_manifest': visual_manifest, 'root_visual_review_pending': True,
               'autograd_not_reexecuted_gradient_values_are_returned_evidence': True,
               'PT_serialization_not_unpickled': True, 'no_independent_frozen_weight_payload_in_return': True,
               'same_device_reference_not_CPU_L4_parity_or_independent_architecture_proof': True,
               'zero_optimizer_preflight_not_model_improvement_or_training_qualification': True,
               'native_FINAL_or_automatic_preparation_qualified': False, 'model_qualified': False,
               'auditor_neural_gradient_optimizer_HTTP_browser_VM_training_calls': 0}
    write(outdir / 'saved_return_audit_v1.json', summary)
    need('torch' not in sys.modules, 'Saved-only auditor must remain Torch-free')
    print(json.dumps({'saved_return_artifact_audit_pass': True, 'parity_cases': 8, 'gradient_arrays': arrays,
                      'generated_PNGs': 20, 'visual_sheets': 2, 'model_qualified': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', help='Inert schema/arithmetic checks only; consumes no VM return')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--archive-sha')
    parser.add_argument('--protocol-sha')
    parser.add_argument('--export-json', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.self_check:
        need(not any((args.archive, args.archive_sha, args.protocol_sha, args.export_json, args.out)),
             '--self-check cannot consume return arguments')
        self_check()
    else:
        need(all((args.archive, args.archive_sha, args.protocol_sha, args.export_json, args.out)),
             'A real archive, both expected SHAs, matching export receipt and new output directory are required')
        for digest in (args.archive_sha, args.protocol_sha):
            need(len(digest) == 64 and all(x in '0123456789abcdef' for x in digest), 'Exact lowercase SHA256 required')
        main(args)
