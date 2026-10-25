"""Saved u64 continuation values only; import is stdlib-only and inert.

No model/optimizer is constructed and no forward/backward/step is available.
The explicit --run path needs a peer source/input admission, then runs in a
separate CPU-only, RSS/time-bounded process. Torch only decodes bounded,
preflighted ZIP checkpoints with weights_only=True/map_location='cpu'.
"""
import argparse
import ast
import hashlib
import io
import json
import math
import ntpath
import os
from pathlib import Path, PurePosixPath
import pickletools
import re
import stat
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEST = 'tests/test_u64_cleananchors_return_state_v1.py'
PROTOCOL_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
PARENT_SHA = 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'
HEAD_SHA = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
R2_SHA = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_SHA = 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
OUTPUT = 'outputs/u64_cleananchors_continuation_v1'
MAX_FULL = 384 * 1024**2
MAX_SMALL = 8 * 1024**2
MAX_SECONDS = 240
MAX_RSS = 2 * 1024**3
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
COUNTERS = ('full_forward_attempts', 'full_forward_returns', 'backward_attempts',
            'backward_returns', 'optimizer_constructions', 'optimizer_construction_returns',
            'optimizer_update_attempts', 'optimizer_updates', 'initial_quality_bank_forwards',
            'deterministic_repeat_forwards', 'candidate_snapshot_forwards_after_initializer',
            'candidate_training_forwards')
TOLERANCES = {'first_moment': {'atol': 2e-9, 'rtol': 2e-5},
              'second_moment': {'atol': 2e-12, 'rtol': 2e-5},
              'parameter_update': {'atol': 6e-8, 'rtol': 3e-6}}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    def walk(item, depth=0):
        need(depth <= 30, 'Bounded JSON depth')
        if type(item) is float:
            need(math.isfinite(item), 'Finite JSON exponent')
        elif type(item) in (dict, list):
            for child in item.values() if type(item) is dict else item:
                walk(child, depth+1)
    walk(value)
    return value


def relative(value):
    need(type(value) is str and value and '\\' not in value and ':' not in value,
         'Canonical relative POSIX path')
    p = PurePosixPath(value)
    need(not p.is_absolute() and p.as_posix() == value and value != '.'
         and not any(x in ('', '.', '..') for x in p.parts), 'Relative scope escape')
    return value


def ordinary(path, maximum=MAX_FULL, directory=False):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        s = item.lstat()
        is_dir = item != path or directory
        need((stat.S_ISDIR(s.st_mode) if is_dir else stat.S_ISREG(s.st_mode))
             and not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024,
             'Ordinary unlinked file/directory ancestry')
        if not is_dir:
            need(s.st_nlink == 1 and 0 < s.st_size <= maximum, 'Bounded single-link input')
    return path.resolve(strict=True)


def pin(path, maximum=MAX_FULL):
    path = ordinary(path, maximum)
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns)
         == (after.st_ino, after.st_size, after.st_mtime_ns), 'Input changed during hashing')
    return {'bytes': after.st_size, 'sha256': h.hexdigest()}


def named_pin(path):
    return {'path': Path(path).resolve().relative_to(ROOT).as_posix(), **pin(path)}


def head_shapes():
    result = {}
    for site, c in (('32', 256), ('64', 256), ('128', 128), ('256', 128)):
        for name, shape in {'encoder.weight': (32, c, 1, 1), 'encoder.bias': (32,),
                            'decoder.weight': (32, c, 1, 1), 'fused.weight': (32, c, 1, 1),
                            'coarse.weight': (32, 256, 1, 1), 'spatial.weight': (32, 32, 3, 3),
                            'spatial.bias': (32,), 'output.weight': (c, 32, 1, 1),
                            'output.bias': (c,)}.items():
            result['heads.'+site+'.'+name] = shape
    need(len(result) == 36 and sum(math.prod(v) for v in result.values()) == 168960,
         'Independent36/168960 schema')
    return result


def header(name, shape):
    raw = json.dumps([name, 'torch.float32', list(shape)], separators=(',', ':')).encode()
    return len(raw).to_bytes(8, 'big') + raw


def canonical(arrays, np):
    h = hashlib.sha256()
    for name, a in sorted(arrays.items()):
        need(type(name) is str and isinstance(a, np.ndarray) and a.dtype == np.float32
             and np.isfinite(a).all(), 'Finite named FP32 values')
        h.update(header(name, a.shape)); h.update(a.tobytes(order='C'))
    return h.hexdigest()


def read_npz(path, shapes, np, *, maximum=MAX_SMALL, retain=None, tick=lambda: None):
    """Validate NPY headers/lengths before allocation; stream one array at a time."""
    path = ordinary(path, maximum)
    before = pin(path, maximum)
    need(type(shapes) is dict and 0 < len(shapes) <= 575, 'Bounded exact state namespace')
    for name, shape in shapes.items():
        need(re.fullmatch(r'[A-Za-z0-9_.]{1,200}', name) and '..' not in name
             and type(shape) in (tuple, list) and 0 < len(shape) <= 6
             and all(type(x) is int and 0 < x <= 65536 for x in shape), 'Bounded array schema')
    need(sum(math.prod(s)*4 for s in shapes.values()) <= maximum, 'Declared payload cap')
    digests = {k: hashlib.sha256() for k in ('all', 'base', 'adapters', 'heads', 'frozen')}
    result, rows, counts = {}, [], dict.fromkeys(digests, 0)
    with path.open('rb') as f:
        need(f.read(4) == b'PK\x03\x04', 'NPZ starts at ordinary ZIP header')
        f.seek(-22, 2)
        need(f.read(4) == b'PK\x05\x06' and f.read(18)[-2:] == b'\0\0', 'Exact uncommented NPZ end')
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        need(len(infos) == len(shapes) and len({i.filename for i in infos}) == len(infos)
             and {i.filename for i in infos} == {n+'.npy' for n in shapes}
             and not z.comment and sum(i.file_size for i in infos) <= maximum,
             'Exact unique bounded NPZ members')
        for name in sorted(shapes):
            tick(); info = z.getinfo(name+'.npy')
            need(info.compress_type == zipfile.ZIP_STORED and not info.flag_bits & 1
                 and not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16),
                 'Uncompressed unencrypted regular NPZ member')
            with z.open(info) as f:
                version = np.lib.format.read_magic(f)
                need(version in ((1, 0), (2, 0)), 'Supported NPY version')
                read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                               else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = read_header(f, max_header_size=4096)
                size = math.prod(shapes[name])*4
                need(tuple(shape) == tuple(shapes[name]) and dtype.str == '<f4'
                     and not dtype.hasobject and fortran is False
                     and info.file_size == f.tell()+size, 'Exact little-endian C-order FP32 NPY schema')
                raw = f.read(size+1)
                need(len(raw) == size, 'Complete NPY payload and CRC')
            a = np.frombuffer(raw, dtype=dtype).reshape(shape)
            need(np.isfinite(a).all(), 'Finite saved state/gradient values')
            row = {'name': name, 'shape': list(shape), 'dtype': 'float32',
                   'bytes': size, 'bytes_sha256': hashlib.sha256(raw).hexdigest()}
            rows.append(row)
            groups = [('all', name)]
            if name.startswith('base.'):
                groups += [('base', name[5:]), ('frozen', name)]
            elif name.startswith('adapters.'):
                groups += [('adapters', name), ('frozen', name)]
            elif name.startswith('heads.'):
                groups += [('heads', name)]
            for group, canonical_name in groups:
                digests[group].update(header(canonical_name, shape)); digests[group].update(raw)
                counts[group] += 1
            if retain is None or name in retain:
                result[name] = a.copy(order='C')
    need(pin(path, maximum) == before, 'NPZ changed during decode')
    return {'arrays': result, 'rows': rows, 'counts': counts,
            'digests': {k: v.hexdigest() for k, v in digests.items()}}


def expected_counts(update):
    need(type(update) is int and 0 <= update <= 16, 'New update0..16')
    snapshots = 56 * sum(s < update for s in (4, 8, 16))
    forward = 60 + 4*update + snapshots
    return dict(zip(COUNTERS, (forward, forward, 4*update, 4*update, 1, 1,
                              update, update, 56, 4, snapshots, 4*update)))


def learning_rate(update):
    need(type(update) is int and 0 <= update <= 16, 'Finite update cursor')
    return .0001 * min(1., update/4.) if update else .0001


def validate_rng(rng):
    need(type(rng) is dict and set(rng) == {'Python', 'NumPy', 'CPU_hex', 'CUDA_hex'}, 'Exact RNG fields')
    text = rng['Python']
    need(type(text) is str and len(text) <= 30000, 'Bounded Python RNG repr')
    value = ast.literal_eval(text)
    need(type(value) is tuple and len(value) == 3 and value[0] == 3
         and type(value[1]) is tuple and len(value[1]) == 625
         and all(type(x) is int and 0 <= x < 2**32 for x in value[1][:-1])
         and type(value[1][-1]) is int and 0 <= value[1][-1] <= 624
         and (value[2] is None or type(value[2]) is float and math.isfinite(value[2])), 'Python RNG schema')
    ns = rng['NumPy']
    need(type(ns) is dict and set(ns) == {'algorithm', 'keys_hex', 'position', 'has_gauss', 'cached_gaussian'}
         and ns['algorithm'] == 'MT19937' and type(ns['keys_hex']) is str
         and len(ns['keys_hex']) == 624*8 and re.fullmatch('[0-9a-f]+', ns['keys_hex'])
         and type(ns['position']) is int and 0 <= ns['position'] <= 624
         and type(ns['has_gauss']) is int and ns['has_gauss'] in (0, 1)
         and type(ns['cached_gaussian']) in (int, float) and math.isfinite(ns['cached_gaussian']), 'NumPy RNG schema')
    def hex_state(s):
        return type(s) is str and 0 < len(s) <= 131072 and len(s) % 2 == 0 and re.fullmatch('[0-9a-f]+', s)
    need(hex_state(rng['CPU_hex']) and type(rng['CUDA_hex']) is list
         and len(rng['CUDA_hex']) == 1 and hex_state(rng['CUDA_hex'][0]), 'CPU/one-CUDA RNG bytes')
    return hashlib.sha256(json.dumps(rng, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_optimizer(opt, update, names, shapes, np):
    need(type(opt) is dict and set(opt) == {'state', 'param_groups'}
         and type(opt['param_groups']) is list and len(opt['param_groups']) == 1, 'One saved Adam group')
    g = opt['param_groups'][0]
    policy = {'lr': learning_rate(update), 'betas': (.9, .99), 'eps': 1e-8, 'weight_decay': 0.,
              'amsgrad': False, 'foreach': False, 'fused': False, 'capturable': False,
              'differentiable': False, 'maximize': False}
    need(type(g) is dict and set(g) in (set(policy)|{'params'}, set(policy)|{'params', 'decoupled_weight_decay'})
         and g['params'] == list(range(len(names))) and g.get('decoupled_weight_decay', False) is False,
         'Exact Adam group fields and stable parameter IDs')
    for key, expected in policy.items():
        actual = tuple(g[key]) if key == 'betas' else g[key]
        need(actual == expected and (type(actual) is bool if type(expected) is bool else True), 'Saved Adam policy '+key)
    need(type(opt['state']) is dict and set(opt['state']) == (set() if update == 0 else set(range(len(names)))),
         'Fresh empty moments at0; complete moment census after commits')
    moments = {}
    for i, name in enumerate(names):
        if update == 0:
            moments[name] = (np.zeros(shapes[name], np.float32), np.zeros(shapes[name], np.float32))
            continue
        row = opt['state'][i]
        need(type(row) is dict and set(row) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Exact Adam state fields')
        step = row['step']
        need(isinstance(step, np.ndarray) and step.shape == () and step.dtype == np.float32
             and float(step) == update, 'Exact CPU scalar FP32 Adam step')
        m, v = row['exp_avg'], row['exp_avg_sq']
        need(all(isinstance(a, np.ndarray) and a.dtype == np.float32 and a.shape == tuple(shapes[name])
                 and np.isfinite(a).all() for a in (m, v)) and (v >= 0).all(), 'Finite exact moment shapes/nonnegative v')
        moments[name] = (m, v)
    return moments


def adam_predict(parameter, gradient, m0, v0, update, lr, np, *, actual_m=None, actual_v=None):
    """Independent saved-array arithmetic; never applies parameters to a model."""
    m = (m0 + (gradient-m0)*np.float32(.1)).astype(np.float32)
    v = (v0*np.float32(.99) + (gradient*gradient)*np.float32(.01)).astype(np.float32)
    cm, cv = (m if actual_m is None else actual_m), (v if actual_v is None else actual_v)
    denominator = (np.sqrt(cv, dtype=np.float32)/np.float32(math.sqrt(1-.99**update)) + np.float32(1e-8)).astype(np.float32)
    p = (parameter - np.float32(lr/(1-.9**update))*(cm/denominator)).astype(np.float32)
    return m, v, p


def compare(actual, expected, np, kind):
    need(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(), 'Finite matching comparison')
    error = np.abs(actual.astype(np.float64)-expected.astype(np.float64)); t = TOLERANCES[kind]
    need(not np.any(error > t['atol'] + t['rtol']*np.abs(actual.astype(np.float64))), 'Saved Adam recurrence mismatch: '+kind)
    return float(error.max(initial=0))


def check_clipping(pre, post, norm, coefficient, np):
    need(set(pre) == set(post) and type(norm) in (float, int) and math.isfinite(norm) and norm > 0,
         'Complete finite nonzero preclip norm')
    measured = math.sqrt(sum(float(np.square(a.astype(np.float64)).sum()) for a in pre.values()))
    need(abs(measured-norm) <= 2e-7 + measured*2e-5, 'Recorded preclip norm matches saved gradients')
    want = min(1., 1./(norm+1e-6))
    need(coefficient == want, 'Exact clip coefficient receipt')
    worst = 0.
    for name in pre:
        prediction = (pre[name]*np.float32(want)).astype(np.float32)
        need(prediction.shape == post[name].shape and np.allclose(post[name], prediction, atol=2e-9, rtol=2e-6), 'Actual saved pre/post clip values')
        worst = max(worst, float(np.max(np.abs(post[name].astype(np.float64)-prediction.astype(np.float64)))))
    after = math.sqrt(sum(float(np.square(a.astype(np.float64)).sum()) for a in post.values()))
    need(after <= 1.00001 and any(np.count_nonzero(a) for a in pre.values()), 'Nonzero bounded clipped gradient')
    return {'preclip_L2_float64': measured, 'postclip_L2_float64': after, 'maximum_clip_error': worst}


def objective_scalar(role, terms):
    need(role in ROLES and type(terms) is dict and all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in terms.values()), 'Finite saved objective terms')
    if role == ROLES[0]:
        return .5*(terms['inside_image_L1']+terms['outside_image_L1']) + .125*(terms['inside_adjacent_L1']+terms['outside_adjacent_L1'])
    if role == 'clear':
        return 2*(terms['photographic_image_anchor_SmoothL1'] + .25*terms['photographic_adjacent_anchor_SmoothL1'])
    value = terms['target_image_L1'] + .25*terms['target_adjacent_L1']
    return value + (2*terms['photographic_block16_luminance_anchor_SmoothL1'] if role == ROLES[3] else 0)


def pt_preflight(path, maximum=4*1024**2):
    path = ordinary(path, maximum)
    with zipfile.ZipFile(path) as z:
        members = z.infolist(); names = [x.filename for x in members]
        need(0 < len(members) <= 160 and len(set(names)) == len(names) and not z.comment
             and sum(x.file_size for x in members) <= maximum, 'Bounded unique Torch ZIP')
        roots = set()
        for item in members:
            relative(item.filename); p = PurePosixPath(item.filename); roots.add(p.parts[0])
            need(len(p.parts) >= 2 and item.compress_type == zipfile.ZIP_STORED and not item.flag_bits & 1
                 and not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16), 'Regular bounded CPU ZIP member')
        need(len(roots) == 1, 'One Torch ZIP root')
        name = next(iter(roots))+'/data.pkl'
        need(name in names and z.getinfo(name).file_size <= 256*1024, 'Bounded checkpoint pickle metadata')
        metadata = z.read(name); count = 0
        allowed = {'collections OrderedDict', 'torch._utils _rebuild_tensor_v2', 'torch._utils _rebuild_tensor', 'torch FloatStorage'}
        for op, arg, position in pickletools.genops(metadata):
            count += 1
            need(count <= 30000, 'Bounded checkpoint pickle operations')
            if op.name == 'GLOBAL':
                need(arg in allowed, 'Unexpected checkpoint GLOBAL')
            need(op.name not in ('STACK_GLOBAL', 'EXT1', 'EXT2', 'EXT4', 'INST', 'OBJ', 'NEWOBJ_EX', 'PERSID'), 'Unsupported checkpoint pickle opcode')
            if type(arg) is int:
                need(-1 <= arg <= maximum, 'Bounded checkpoint integer/storage declaration')
            if type(arg) is float:
                need(math.isfinite(arg), 'Finite checkpoint scalar')
            if type(arg) in (str, bytes):
                need(len(arg) <= 131072, 'Bounded checkpoint primitive string')
        need(metadata[-1:] == b'.', 'Complete pickle STOP without trailing metadata')


def decode_cpu(path, torch, np, maximum=4*1024**2):
    pt_preflight(path, maximum)
    before = pin(path, maximum)
    value = torch.load(path, map_location='cpu', weights_only=True)
    nodes = [0]
    def convert(x, depth=0):
        nodes[0] += 1
        need(depth <= 16 and nodes[0] <= 3000, 'Bounded primitive checkpoint tree')
        if isinstance(x, torch.Tensor):
            need(x.device.type == 'cpu' and not x.requires_grad and x.grad is None
                 and x.dtype == torch.float32 and x.numel()*4 <= maximum, 'Detached bounded FP32 CPU checkpoint tensor')
            a = x.detach().contiguous().numpy().copy()
            need(np.isfinite(a).all(), 'Finite CPU checkpoint tensor')
            return a
        if type(x) is dict:
            need(all(type(k) in (str, int) for k in x), 'Primitive checkpoint keys')
            return {k: convert(v, depth+1) for k, v in x.items()}
        if type(x) in (list, tuple):
            return type(x)(convert(v, depth+1) for v in x)
        need(x is None or type(x) in (str, int, float, bool), 'Primitive CPU checkpoint leaves')
        if type(x) is float:
            need(math.isfinite(x), 'Finite CPU checkpoint scalar')
        return x
    answer = convert(value)
    need(pin(path, maximum) == before, 'Checkpoint changed during CPU decode')
    return answer


class Evidence:
    def __init__(self, manifest_path):
        self.manifest_path = ordinary(manifest_path, 16*1024**2)
        self.manifest_pin = named_pin(self.manifest_path)
        self.manifest = strict_json(self.manifest_path.read_bytes())
        m = self.manifest
        need(m['complete'] is True and type(m['files']) is dict and type(m['extracted_files']) is dict,
             'Complete inventory and explicit extracted-file subset')
        self.root = ordinary(ROOT/relative(m['evidence_root']), directory=True)
        need(self.root.is_relative_to(ROOT/'outputs'), 'Extracted evidence stays within workspace outputs')
        self.files = m['extracted_files']; need(len(self.files) <= 6000, 'Bounded extracted namespace')
        for name, binding in self.files.items():
            relative(name)
            need(type(binding) is dict and set(binding) == {'bytes', 'sha256'} and binding == m['files'][name]
                 and type(binding['bytes']) is int and 0 <= binding['bytes'] <= MAX_FULL
                 and re.fullmatch('[0-9a-f]{64}', binding['sha256']), 'Exact extracted/archive inventory join')
        self.used = {}
        self.began = time.monotonic()

    def tick(self):
        need(time.monotonic()-self.began < MAX_SECONDS, 'Finite saved-state audit time')

    def file(self, name, binding=None, maximum=MAX_FULL):
        self.tick(); relative(name); need(name in self.files, 'Required evidence not extracted: '+name)
        if binding is not None:
            need(binding == {'path': name, **self.files[name]}, 'Returned file binding differs from inventory')
        path = ordinary(self.root/name, maximum)
        need(pin(path, maximum) == self.files[name], 'Extracted evidence bytes differ: '+name)
        self.used[name] = self.files[name]
        return path

    def bound(self, binding, maximum=MAX_SMALL):
        need(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'}, 'Exact returned file pin')
        return self.file(binding['path'], binding, maximum)

    def json(self, name):
        return strict_json(self.file(name, maximum=MAX_SMALL).read_bytes())


def independent_plan(p, parent):
    order = parent['learning']['optimization_reference_order']; monitor = parent['learning']['monitor_reference_order']
    need(len(order) == 16 and len(set(order)) == 16 and len(monitor) == 8 and len(set(monitor)) == 8
         and not set(order) & set(monitor) and p['cases'] == parent['cases'] and p['guards'] == parent['guards'], 'Exact retained photograph partitions')
    by = {(r['reference_id'], r['role']): r for r in p['cases']}
    bank = [by[(ref, role)] for ref in monitor for role in ROLES] + p['guards']
    need(len(bank) == 56 and len(by) == 96, 'Exact finite TRAIN bank')
    plan = []
    def append(row, phase, update, backward=False):
        plan.append({'slot': len(plan)+1, 'id': row['id'], 'phase': phase, 'new_update': update,
                     'backward': backward, 'optimizer_application_after_this_forward': backward and row['role'] == ROLES[-1]})
    for row in bank:
        append(row, 'initial_u64_quality_bank', 0)
    for row in bank[:4]:
        append(row, 'deterministic_same_u64_repeat', 0)
    for u in range(1, 17):
        for role, offset in zip(ROLES, (0, 4, 8, 12)):
            append(by[(order[(u-1+offset) % 16], role)], 'paired_TRAIN_optimization', u, True)
        if u in (4, 8, 16):
            for row in bank:
                append(row, 'new_quality_snapshot', u)
    need(len(plan) == 292 and sum(x['backward'] for x in plan) == 64, 'Independent292/64 plan')
    return plan, {r['id']: r for r in p['cases']+p['guards']}


def operation_coverage(counts, pending, last, unknown):
    need(type(counts) is dict and set(counts) == set(COUNTERS)
         and all(type(v) is int and v >= 0 for v in counts.values()),
         'Actual durable counters; absent/corrupt is never zero')
    need(type(pending) is list and type(unknown) is bool and unknown == bool(pending),
         'Explicit consistent pending outcome')
    for attempt, returned, cap in (('full_forward_attempts', 'full_forward_returns', 292),
                                   ('backward_attempts', 'backward_returns', 64),
                                   ('optimizer_constructions', 'optimizer_construction_returns', 1),
                                   ('optimizer_update_attempts', 'optimizer_updates', 16)):
        need(counts[returned] <= counts[attempt] <= cap, 'Finite attempted/returned counters')
    need(sum(counts[k] for k in COUNTERS[8:]) == counts['full_forward_attempts'], 'Forward categories account for attempts')
    need(all(type(x) is dict and x.get('kind') in ('forward', 'backward', 'optimizer_construction', 'optimizer_update') for x in pending), 'Known pending operation kinds')
    update_unknown = counts['optimizer_update_attempts'] != counts['optimizer_updates'] or any(x['kind'] == 'optimizer_update' for x in pending)
    covered = not update_unknown and counts['optimizer_updates'] == last
    attempted_through = max(counts['optimizer_update_attempts'], math.ceil(counts['candidate_training_forwards']/4))
    return {'covered': covered, 'optimizer_unknown': update_unknown,
            'later_unattempted': list(range(attempted_through+1, 17)) if not pending else None,
            'partially_attempted_update': attempted_through if attempted_through > last else None}


def active_cursor(status, failure, last):
    active = status.get('active_update')
    need(type(active) is int and last <= active <= 16, 'Actual finite active-update cursor')
    if failure is not None:
        need(type(failure.get('active_update')) is int and failure['active_update'] == active,
             'Failure/status active-update cursors agree')
    return active


def closing_relationship(closing, last, active, coverage, pending):
    """Known committed values must match; an unverified later state stays explicit."""
    required = coverage['covered'] and not pending
    for row in closing:
        need(type(row.get('update')) is int and row['update'] == active,
             'Closing full575 cursor binds actual status/failure active update')
        if required:
            need(row.get('heads_match_last_verified_checkpoint') is True,
                 'Known committed-stop closing575 heads must equal verified checkpoint'+str(last))
    exception = None
    if not required:
        exception = ('Pending operation leaves current-state relationship unvalidated'
                     if pending else 'Later optimizer outcome is not covered by the verified checkpoint prefix')
    return {'head_equality_required': required,
            'closing_head_trajectory_verified': bool(closing) and required,
            'unvalidated_relationship_exception': exception}


def partial_cursor(partial, status, failure, last, coverage, pending):
    active = active_cursor(status, failure, last)
    need(type(partial.get('new_update')) is int and partial['new_update'] == active,
         'Partial checkpoint new_update must equal actual status/failure active_update')
    # Eligibility comes from independently bound outcome/cursor evidence. Changing
    # the partial payload cursor can never turn a required comparison off.
    compare_committed = coverage['covered'] and not pending and active == last
    return {'active_update': active, 'committed_Adam_RNG_equality_required': compare_committed,
            'unvalidated_relationship_exception': None if compare_committed else
            'Actual partial/later or unknown operation state is retained outside the verified committed Adam/RNG prefix'}


def full_state(e, witness_path, np):
    witness = e.json(witness_path); rows = witness['arrays']
    need(witness['complete'] is True and witness['tensor_count'] == 575 and witness['model_qualified'] is False
         and type(rows) is list and len(rows) == 575 and len({r['name'] for r in rows}) == 575, 'Complete unique575 witness')
    shapes = {r['name']: tuple(r['shape']) for r in rows}
    need({k: v for k, v in shapes.items() if k.startswith('heads.')} == head_shapes(), 'Independent full-state head schema')
    state = read_npz(e.bound(witness['file'], MAX_FULL), shapes, np, maximum=MAX_FULL,
                     retain=set(head_shapes()), tick=e.tick)
    need(state['rows'] == sorted(rows, key=lambda r: r['name']) and state['digests']['all'] == witness['canonical_sha256']
         and state['counts'] == {'all': 575, 'base': 515, 'adapters': 24, 'heads': 36, 'frozen': 539}
         and state['digests']['base'] == R2_SHA and state['digests']['adapters'] == U96_SHA
         and state['digests']['frozen'] == witness['frozen539_canonical_sha256'], 'Exact actual575 values and retained539 initializer lineage')
    return witness, state


def validate_resume(resume, update, frozen_sha, np, shapes=None):
    shapes = head_shapes() if shapes is None else shapes; names = sorted(shapes)
    fields = {'format', 'protocol_sha256', 'update', 'automatic_resume_permitted', 'optimizer',
              'optimizer_parameter_names', 'RNG', 'sampler_next_new_update', 'initial_head_canonical_sha256',
              'frozen539_canonical_sha256', 'counters', 'model_qualified'}
    need(type(resume) is dict and set(resume) == fields
         and resume['format'] == 'u64-cleananchors-closed-finite-state-v1'
         and resume['protocol_sha256'] == PROTOCOL_SHA and resume['update'] == update
         and resume['automatic_resume_permitted'] is False and resume['model_qualified'] is False
         and resume['optimizer_parameter_names'] == names
         and resume['sampler_next_new_update'] == (update+1 if update < 16 else None)
         and resume['initial_head_canonical_sha256'] == HEAD_SHA
         and resume['frozen539_canonical_sha256'] == frozen_sha
         and resume['counters'] == expected_counts(update), 'Exact finite checkpoint lineage/cursor/counters')
    return validate_optimizer(resume['optimizer'], update, names, shapes, np), validate_rng(resume['RNG'])


def audit_saved(e, np, decode):
    """Read extracted/pinned values; caller supplies only the CPU decode adapter."""
    m = e.manifest; protocol_path = ordinary(ROOT/relative(m['protocol']['path']))
    need(named_pin(protocol_path) == m['protocol'] and m['protocol']['sha256'] == PROTOCOL_SHA, 'Exact sealed continuation protocol')
    p = strict_json(protocol_path.read_bytes()); parent_path = protocol_path.parent/p['parent_protocol']['path']
    need(pin(parent_path) == {k: p['parent_protocol'][k] for k in ('bytes', 'sha256')}
         and p['parent_protocol']['sha256'] == PARENT_SHA, 'Pinned retained parent protocol')
    plan, rows = independent_plan(p, strict_json(parent_path.read_bytes()))
    saved_plan = e.json(OUTPUT+'/operation_plan.json')
    need(saved_plan['complete'] is True and saved_plan['calls'] == plan, 'Saved sampler/call-plan matches independent construction')
    status = e.json(OUTPUT+'/status.json')
    failure = e.json(OUTPUT+'/failure.json') if OUTPUT+'/failure.json' in e.files else None
    result = e.json(OUTPUT+'/result.json') if OUTPUT+'/result.json' in e.files else None
    for owner in (status, failure, result):
        if owner is not None:
            need(owner['protocol_sha256'] == PROTOCOL_SHA and owner['model_qualified'] is False, 'Returned outcome protocol/nonqualification')
    initial_w, initial = full_state(e, OUTPUT+'/states/initial_u64/state.json', np)
    frozen_sha = initial['digests']['frozen']
    need(initial['digests']['heads'] == HEAD_SHA and initial_w['update'] == 0, 'Exact retained trained-u64 initial36; zero/reset state refused')
    checkpoints = sorted(int(re.fullmatch(re.escape(OUTPUT)+r'/checkpoints/u(\d{4})/checkpoint.json', n)[1])
                         for n in e.files if re.fullmatch(re.escape(OUTPUT)+r'/checkpoints/u(\d{4})/checkpoint.json', n))
    need(checkpoints and checkpoints == list(range(checkpoints[-1]+1)) and checkpoints[-1] <= 16, 'Contiguous complete checkpoint prefix0..atmost16')
    shapes = head_shapes(); previous_heads = initial['arrays']; previous_moments = None; rng_hash = None
    verified, worst = [], dict.fromkeys(TOLERANCES, 0.)
    for u in checkpoints:
        e.tick(); folder = OUTPUT+'/checkpoints/u'+str(u).zfill(4); cp = e.json(folder+'/checkpoint.json')
        need(cp['complete'] is True and cp['update'] == u and cp['model_qualified'] is False
             and cp['frozen539_canonical_sha256'] == frozen_sha and cp['counters'] == expected_counts(u), 'Checkpoint complete/cursor/frozen/counters')
        state = read_npz(e.bound(cp['heads']), shapes, np, tick=e.tick); heads = state['arrays']
        need(state['digests']['all'] == cp['head_canonical_sha256'], 'Committed head canonical values')
        resume = decode(e.bound(cp['resume']), 4*1024**2)
        moments, current_rng = validate_resume(resume, u, frozen_sha, np)
        need(rng_hash in (None, current_rng), 'RNG unchanged throughout fixed saved trajectory'); rng_hash = current_rng
        if u == 0:
            need(cp['head_canonical_sha256'] == HEAD_SHA and all(np.array_equal(heads[n], previous_heads[n]) for n in shapes), 'Checkpoint0 exactly matches full trained-u64 initializer')
        else:
            commit = e.json(OUTPUT+'/updates/u'+str(u).zfill(4)+'/commit.json')
            need(commit['complete'] is True and commit['new_update'] == u and commit['LR'] == learning_rate(u)
                 and commit['head_before_sha256'] == canonical(previous_heads, np)
                 and commit['head_after_sha256'] == canonical(heads, np) and commit['frozen539_exact'] is True
                 and commit['counters'] == expected_counts(u), 'Commit links exact prior/current heads and counters')
            pre = read_npz(e.bound(commit['preclip_gradients']), shapes, np, tick=e.tick)['arrays']
            post = read_npz(e.bound(commit['postclip_gradients']), shapes, np, tick=e.tick)['arrays']
            clip = check_clipping(pre, post, commit['preclip_norm'], commit['clip_coefficient'], np)
            calls = [c for c in plan if c['new_update'] == u and c['backward']]
            need(len(commit['microbatches']) == len(calls) == 4, 'Four saved objective/backward associations')
            for binding, call in zip(commit['microbatches'], calls):
                scalar = strict_json(e.bound(binding).read_bytes()); row = rows[call['id']]
                need(row['split'] == 'TRAIN' and row['cohort'] == 'optimize' and row['optimization_permitted'] is True
                     and scalar['complete'] is True and scalar['new_update'] == u and scalar['call_slot'] == call['slot']
                     and scalar['id'] == call['id'] and scalar['role'] == row['role'] and scalar['LR'] == learning_rate(u)
                     and scalar['photographic_target'] == row['target']
                     and scalar['head_before_commit_sha256'] == canonical(previous_heads, np)
                     and scalar['divided_objective'] == scalar['objective']/4, 'Exact TRAIN target/sampler/head/scalar association')
                predicted = objective_scalar(row['role'], scalar['terms'])
                need(abs(predicted-scalar['objective']) <= 2e-7 + 3e-6*abs(predicted), 'Saved objective term arithmetic')
                record = strict_json(e.bound(scalar['output_record']).read_bytes())
                need(record['call'] == call and record['target'] == row['target'] and record['input'] == row['input'], 'Actual returned record belongs to objective call/target')
            errors = dict.fromkeys(TOLERANCES, 0.)
            for name in sorted(shapes):
                m0, v0 = previous_moments[name]; cm, cv = moments[name]
                predicted = adam_predict(previous_heads[name], post[name], m0, v0, u, learning_rate(u), np, actual_m=cm, actual_v=cv)
                for kind, actual, estimate in zip(TOLERANCES, (cm, cv, heads[name]), predicted):
                    errors[kind] = max(errors[kind], compare(actual, estimate, np, kind)); worst[kind] = max(worst[kind], errors[kind])
            changed = sum(not np.array_equal(previous_heads[n], heads[n]) for n in shapes)
            need(changed > 0, 'Each committed Adam update changes actual head values')
            verified.append({'update': u, 'LR': learning_rate(u), 'changed_head_tensors': changed,
                             'clipping': clip, 'recurrence_max_abs': errors, 'head_canonical_sha256': canonical(heads, np)})
        previous_heads, previous_moments = heads, moments
    last = checkpoints[-1]; state_names = sorted(n for n in e.files if n.startswith(OUTPUT+'/states/') and n.endswith('/state.json'))
    need(1 <= len(state_names) <= 2, 'At most initial and one closing full575 witness')
    closing = []
    for name in state_names:
        if name == OUTPUT+'/states/initial_u64/state.json':
            continue
        w, state = full_state(e, name, np)
        need(state['digests']['frozen'] == frozen_sha, 'Every actual closing frozen539 value remains exact')
        closing.append({'path': name, 'update': w['update'], 'canonical_sha256': state['digests']['all'],
                        'heads_match_last_verified_checkpoint': all(np.array_equal(state['arrays'][n], previous_heads[n]) for n in shapes)})
    for owner in (status, failure, result):
        if owner is not None and owner.get('last_checkpoint') is not None:
            e.bound(owner['last_checkpoint'])
            need(owner['last_checkpoint']['path'] == OUTPUT+'/checkpoints/u'+str(last).zfill(4)+'/checkpoint.json', 'Outcome last checkpoint binding')
        if owner is not None and owner.get('last_complete_state') is not None:
            e.bound(owner['last_complete_state'])
    counts = status.get('counters'); pending = status.get('unknown_reservations')
    coverage = operation_coverage(counts, pending, last, status.get('operation_outcome_unknown'))
    update_unknown, covered = coverage['optimizer_unknown'], coverage['covered']
    if failure is not None:
        need(failure['counters'] == counts and failure['unknown_reservations'] == pending
             and failure['operation_outcome_unknown'] == status['operation_outcome_unknown'], 'Failure/status durable outcome agreement')
    active = active_cursor(status, failure, last)
    closing_check = closing_relationship(closing, last, active, coverage, pending)
    if result is not None:
        need(result['complete'] is True and last == 16 and len(closing) == 1 and closing[0]['heads_match_last_verified_checkpoint']
             and counts['full_forward_returns'] == 292 and counts['backward_returns'] == 64 and covered, 'Normal completion needs full finite trajectory and complete closing values')
    partial_check = None
    partial_path = OUTPUT+'/failure_values/partial_optimizer.pt'
    if partial_path in e.files:
        partial = decode(e.file(partial_path, maximum=MAX_SMALL), MAX_SMALL)
        need(type(partial) is dict and set(partial) == {'optimizer', 'heads', 'actual_gradients', 'RNG', 'new_update', 'operation_ledger', 'accepted_as_valid'}
             and partial['accepted_as_valid'] is False, 'Partial failure evidence retains original unvalidated status')
        cursor_check = partial_cursor(partial, status, failure, last, coverage, pending)
        need(type(partial['heads']) is dict and set(partial['heads']) == set(shapes)
             and all(isinstance(partial['heads'][n], np.ndarray) and partial['heads'][n].dtype == np.float32
                     and partial['heads'][n].shape == shapes[n] and np.isfinite(partial['heads'][n]).all() for n in shapes), 'Exact actual partial36 head arrays')
        need(partial['operation_ledger']['counters'] == counts
             and partial['operation_ledger']['unknown_reservations'] == pending
             and partial['operation_ledger']['operation_outcome_unknown'] == status['operation_outcome_unknown'], 'Partial failure ledger matches actual durable outcome')
        matches = all(np.array_equal(partial['heads'][n], previous_heads[n]) for n in shapes)
        if cursor_check['committed_Adam_RNG_equality_required']:
            need(matches, 'Known committed-stop partial heads match last checkpoint')
        if cursor_check['committed_Adam_RNG_equality_required']:
            pm = validate_optimizer(partial['optimizer'], last, sorted(shapes), shapes, np)
            need(all(np.array_equal(pm[n][i], previous_moments[n][i]) for n in shapes for i in (0, 1))
                 and validate_rng(partial['RNG']) == rng_hash, 'Known-stop partial Adam/RNG matches final committed checkpoint')
        partial_check = {'original_accepted_as_valid': False, 'heads_match_last_checkpoint': matches,
                         'all_saved_gradient_fields_None': set(partial['actual_gradients']) == set(shapes) and all(v is None for v in partial['actual_gradients'].values()),
                         'new_update': partial['new_update'], 'cursor_relationship': cursor_check}
    need(named_pin(e.manifest_path) == e.manifest_pin and named_pin(protocol_path) == m['protocol'], 'Inventory/protocol unchanged')
    return {'complete': True, 'saved_committed_prefix_audit_pass': True,
            'durable_optimizer_trajectory_fully_covered': covered, 'verified_checkpoints': checkpoints,
            'verified_new_updates': verified, 'later_updates_unattempted': coverage['later_unattempted'],
            'partially_attempted_update': coverage['partially_attempted_update'],
            'actual_status_counters': counts, 'actual_unknown_reservations': pending,
            'operation_outcome_unknown': status['operation_outcome_unknown'], 'optimizer_outcome_unknown': update_unknown,
            'quality_stop_recorded': bool((failure or {}).get('quality_stop') or status.get('quality_stop')),
            'full575_state_witnesses': state_names, 'initial_trained_u64_heads_exact': True,
            'frozen539_retained_R2_U96_values_exact': True, 'closing_full_states': closing,
            'closing_full_state_relationship': closing_check,
            'fresh_Adam_at0': True, 'RNG_saved_trajectory_unchanged': True,
            'partial_failure_checkpoint': partial_check, 'Adam_FP32_tolerances': TOLERANCES,
            'maximum_recurrence_absolute_error': worst, 'inventory': e.manifest_pin,
            'protocol': m['protocol'], 'archive': m['archive'], 'used_files': e.used,
            'objective_scope': 'Saved scalar/formula/target/record associations only; pixel arithmetic and autograd not recomputed.',
            'CPU_weights_only_decodes': len(checkpoints)+(partial_check is not None),
            'model_optimizer_constructions': 0, 'forward_backward_optimizer_steps': 0, 'VM_calls': 0,
            'model_qualified': False, 'quality_stop_is_not_repaired_by_state_integrity': True,
            'seconds': time.monotonic()-e.began}


def admission(review_path, manifest_path):
    review_path = ordinary(review_path, MAX_SMALL); review = strict_json(review_path.read_bytes())
    need(review['complete'] is True and review['source_review_pass'] is True
         and review['allow_CPU_weights_only_saved_checkpoint_decode'] is True,
         'Explicit peer source-reviewed CPU-deserialization admission required')
    need(review['auditor'] == named_pin(__file__) and review['tests'] == named_pin(ROOT/TEST)
         and review['manifest'] == named_pin(manifest_path), 'Admission binds exact current auditor/tests/input inventory')
    return named_pin(review_path)


def rss_bytes(pid):
    if os.name != 'nt':
        return int(Path('/proc', str(pid), 'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True); psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    handle = kernel.OpenProcess(0x410, False, pid)
    need(handle, 'Open CPU audit process for RSS guard')
    try:
        value = Counters(); value.cb = ctypes.sizeof(value)
        need(psapi.GetProcessMemoryInfo(handle, ctypes.byref(value), value.cb), 'Read CPU audit process RSS')
        return int(value.WorkingSetSize)
    finally:
        kernel.CloseHandle(handle)


def path_key(value, windows):
    need(type(value) is str and value, 'Explicit process-context path')
    return ntpath.normcase(ntpath.normpath(value)) if windows else os.path.normpath(value)


def running_python_image():
    """Actual interpreter image, independent of Windows venv launcher spelling."""
    if os.name != 'nt':
        return str(Path(sys.executable).resolve())
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetModuleFileNameW.argtypes = (wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD)
    kernel.GetModuleFileNameW.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    size = kernel.GetModuleFileNameW(None, buffer, len(buffer))
    need(0 < size < len(buffer), 'Read exact running Windows Python image')
    return str(ordinary(buffer.value))


def child_launch_plan(*, windows, executable, base_executable, image, prefix,
                      base_prefix, site_paths, parent_pid, environ):
    """Pure process plan: bypass one Windows launcher; retain direct ancestry."""
    need(type(parent_pid) is int and parent_pid > 0 and type(site_paths) is list
         and site_paths and all(type(x) is str and x for x in site_paths), 'Finite actual parent/venv site paths')
    environment = dict(environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       FORENSIC_U64_STATE_AUDIT_CHILD=str(parent_pid),
                       FORENSIC_U64_STATE_AUDIT_EXECUTABLE=executable,
                       FORENSIC_U64_STATE_AUDIT_PREFIX=prefix,
                       FORENSIC_U64_STATE_AUDIT_BASE_PREFIX=base_prefix,
                       FORENSIC_U64_STATE_AUDIT_IMAGE=image,
                       FORENSIC_U64_STATE_AUDIT_SITES=json.dumps(site_paths))
    if windows:
        need(path_key(image, True) == path_key(base_executable, True),
             'Actual running image must equal this venv base interpreter')
        need(path_key(executable, True) == path_key(ntpath.join(prefix, 'Scripts', 'python.exe'), True)
             and path_key(prefix, True) != path_key(base_prefix, True), 'Exact Windows virtualenv launcher context')
        # CPython's ordinary venv launcher supplies this variable to the same
        # base interpreter. Launch that verified binary directly so Popen owns
        # the actual interpreter PID while its normal getpath retains the venv.
        environment['__PYVENV_LAUNCHER__'] = executable
        binary = image
    else:
        environment.pop('__PYVENV_LAUNCHER__', None)
        binary = executable
    return binary, environment


def verify_child_context(*, windows, environ, parent_pid, executable, prefix,
                         base_prefix, image, site_paths):
    expected = environ.get('FORENSIC_U64_STATE_AUDIT_CHILD', '')
    need(type(expected) is str and expected.isascii() and expected.isdigit()
         and int(expected) > 0 and int(expected) == parent_pid
         and environ.get('CUDA_VISIBLE_DEVICES') == '',
         'Only admitted direct-parent CPU audit child may deserialize')
    actual = {'EXECUTABLE': executable, 'PREFIX': prefix, 'BASE_PREFIX': base_prefix, 'IMAGE': image}
    for field, value in actual.items():
        need(path_key(environ.get('FORENSIC_U64_STATE_AUDIT_'+field, ''), windows) == path_key(value, windows),
             'CPU child must preserve exact interpreter/venv context: '+field)
    encoded = environ.get('FORENSIC_U64_STATE_AUDIT_SITES', '')
    need(type(encoded) is str and len(encoded) <= 32768, 'Bounded expected venv paths')
    expected_sites = strict_json(encoded.encode('utf-8'))
    need(type(expected_sites) is list and 0 < len(expected_sites) <= 16
         and all(path_key(p, windows) in {path_key(s, windows) for s in site_paths} for p in expected_sites),
         'CPU child must retain exact venv site-packages imports')
    return {'direct_parent_verified': True, 'actual_parent_pid': parent_pid,
            'actual_interpreter_image': image, 'venv_executable': executable,
            'venv_prefix': prefix, 'base_prefix': base_prefix,
            'retained_site_packages': expected_sites,
            'Windows_launcher_bypassed': windows, 'CUDA_VISIBLE_DEVICES': ''}


def current_launch_plan():
    prefix = str(Path(sys.prefix).resolve())
    sites = [str(Path(p).resolve()) for p in sys.path if p
             and Path(p).name == 'site-packages' and Path(p).resolve().is_relative_to(Path(prefix))]
    return child_launch_plan(windows=os.name == 'nt', executable=str(Path(sys.executable).absolute()),
                             base_executable=str(Path(sys._base_executable).resolve()),
                             image=running_python_image(), prefix=prefix,
                             base_prefix=str(Path(sys.base_prefix).resolve()), site_paths=sites,
                             parent_pid=os.getpid(), environ=os.environ)


def current_child_context():
    return verify_child_context(windows=os.name == 'nt', environ=os.environ, parent_pid=os.getppid(),
                                executable=str(Path(sys.executable).absolute()), prefix=str(Path(sys.prefix).resolve()),
                                base_prefix=str(Path(sys.base_prefix).resolve()), image=running_python_image(),
                                site_paths=[str(Path(p).resolve()) for p in sys.path if p])


def run_bounded(args):
    review_pin = admission(args.source_review, args.manifest)
    need(not args.output.exists(), 'Audit output is exclusive; original results stay untouched')
    binary, environment = current_launch_plan()
    command = [binary, '-B', str(Path(__file__).resolve()), '--_cpu-child', '--manifest', str(args.manifest),
               '--source-review', str(args.source_review)]
    start = time.monotonic(); peak = 0; child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            try:
                stdout, stderr = child.communicate(timeout=.25)
                break
            except subprocess.TimeoutExpired:
                need(time.monotonic()-start < MAX_SECONDS+30, 'Absolute saved-audit270s subprocess limit')
                if child.poll() is None:
                    peak = max(peak, rss_bytes(child.pid)); need(peak <= MAX_RSS, 'Saved-audit2GiB process RSS limit')
        need(child.returncode == 0 and not stderr and len(stdout) <= 4*1024**2, 'CPU audit child failed: '+stderr.decode('utf-8', 'replace')[-6000:])
        result = strict_json(stdout); result.update(source_review=review_pin, auditor=named_pin(__file__), tests=named_pin(ROOT/TEST),
                                                   separate_CPU_process=True, peak_child_RSS_bytes=peak, CUDA_used=False)
        with args.output.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
        return {'complete': True, 'output': str(args.output), 'sha256': pin(args.output)['sha256'], 'model_qualified': False}
    finally:
        if child.poll() is None:
            child.kill(); child.communicate(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--run', action='store_true'); modes.add_argument('--_cpu-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--manifest', type=Path, required=True); parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.run:
        need(args.output is not None, 'Explicit exclusive audit output required')
        print(json.dumps(run_bounded(args))); return
    context = current_child_context()
    admission(args.source_review, args.manifest)
    evidence = Evidence(args.manifest)
    import numpy as np
    import torch
    need(not torch.cuda.is_initialized(), 'No CUDA initialization in saved-value audit')
    result = audit_saved(evidence, np, lambda path, maximum: decode_cpu(path, torch, np, maximum))
    result['CPU_child_context'] = context
    need(not torch.cuda.is_initialized(), 'No CUDA use in saved-value audit')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
