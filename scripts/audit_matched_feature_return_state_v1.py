"""Saved matched-run states only: bounded NPZ and CPU weights-only decoding.

No model, optimizer, gradient query, neural forward or parameter application is
implemented. This audit cannot convert a retained quality stop into success.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickletools
import stat
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_matched_feature_learning_v1_return_audit_v1'
STAGE = 'cctv_codeformer_matched_feature_learning_v1_vm'
RUN = 'outputs/matched_feature_learning_v1'
SEALED = ROOT / 'outputs' / STAGE
PROTOCOL_SHA = 'afd59b90951138f94f0fcf0dea6ef308f8860042ae3c4f25b43c279d13d0fbd8'
ARCHIVE_SHA = 'de7639af214d80be78ca73b24a3942afbf41223dae94591ce0e14076c079b860'
FULL_SHA = '47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a'
R2_SHA = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_SHA = 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
HEAD_SHA = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
HELPER = ROOT / 'scripts/audit_u64_cleananchors_return_state_v1.py'
HELPER_PIN = (52757, 'e62f7d78852b9f98ee4d08a24ddefe40a599109e95aa5c645a53f0d409fee4b1')
TEST = ROOT / 'tests/test_matched_feature_return_state_v1.py'
MAX_FULL = 384 * 1024**2
MAX_SMALL = 8 * 1024**2
MAX_SECONDS = 600
MAX_RSS = 2 * 1024**3
DIAGNOSTIC_NORM_MAX_ULPS = 8
TOLERANCES = {'first_moment': {'atol': 2e-9, 'rtol': 2e-5},
              'second_moment': {'atol': 2e-12, 'rtol': 2e-5},
              'parameter_update': {'atol': 6e-8, 'rtol': 3e-6}}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(raw):
    def pairs(rows):
        out = {}
        for key, value in rows:
            need(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    def reject(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    def finite(value, depth=0):
        need(depth <= 40, 'Bounded JSON depth')
        if type(value) is float:
            need(math.isfinite(value), 'Finite JSON scalar')
        elif type(value) in (dict, list):
            for child in value.values() if type(value) is dict else value:
                finite(child, depth + 1)
    finite(result)
    return result


def ordinary(path, maximum=MAX_FULL):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        s = item.lstat()
        need((stat.S_ISREG(s.st_mode) if item == path else stat.S_ISDIR(s.st_mode))
             and not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024,
             'Ordinary unlinked evidence required')
        if item == path:
            need(s.st_nlink == 1 and 0 < s.st_size <= maximum, 'Bounded single-link evidence required')
    return path.resolve(strict=True)


def pin(path, maximum=MAX_FULL):
    path = ordinary(path, maximum)
    before = path.stat(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Evidence changed during hashing')
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def helper():
    need(tuple(pin(HELPER)[key] for key in ('bytes', 'sha256')) == HELPER_PIN,
         'Exact reviewed inert saved-array helper required')
    spec = importlib.util.spec_from_file_location('matched_saved_state_legacy_helper', HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def shapes(arm):
    need(type(arm) is str and arm in ('A', 'B'), 'Explicit matched arm')
    result = {}
    for site, channels in (('32', 256), ('64', 256), ('128', 128), ('256', 128)):
        table = {'encoder.weight': (32, channels, 1, 1), 'encoder.bias': (32,),
                 'decoder.weight': (32, channels, 1, 1), 'fused.weight': (32, channels, 1, 1),
                 'coarse.weight': (32, 256, 1, 1), 'spatial.weight': (32, 32, 3, 3),
                 'spatial.bias': (32,), 'output.weight': (channels, 32, 1, 1), 'output.bias': (channels,)}
        result.update({'heads.' + site + '.' + name: value for name, value in table.items()})
        if arm == 'B':
            table = {'down.weight': (32, channels, 1, 1), 'down.bias': (32,),
                     'depthwise.weight': (32, 1, 3, 3), 'depthwise.bias': (32,),
                     'up.weight': (channels, 32, 1, 1), 'up.bias': (channels,)}
            result.update({'adapters.' + site + '.' + name: value for name, value in table.items()})
    need(len(result) == (36 if arm == 'A' else 60) and sum(math.prod(s) for s in result.values()) ==
         (168960 if arm == 'A' else 220288), 'Exact independent trainable census')
    return result


def learning_rate(update):
    need(type(update) is int and 0 <= update <= 48, 'Finite exact update')
    return 1e-5 * min(update / 4, 1.0) if update else 1e-5


class Global:
    def __init__(self, name): self.name = name


class Storage:
    def __init__(self, key, count): self.key, self.count = key, count


class Tensor:
    def __init__(self, shape): self.shape = shape


def preflight_pickle(raw, storage_bytes, maximum=MAX_SMALL):
    """Passively evaluate a restricted pickle syntax; execute no reducer.

    Only FloatStorage persistent IDs, bounded tensor rebuild descriptions and
    OrderedDict construction are admitted. Every tensor view fits its verified
    storage file before Torch can deserialize anything.
    """
    need(type(raw) is bytes and 0 < len(raw) <= 512 * 1024, 'Bounded pickle metadata')
    allowed = {'collections OrderedDict', 'torch._utils _rebuild_tensor_v2',
               'torch._utils _rebuild_tensor', 'torch FloatStorage'}
    stack, memo, marker = [], {}, object()
    storages, tensors, count, stop = {}, 0, 0, False
    def pop_mark():
        need(marker in stack, 'Missing pickle mark')
        pos = len(stack) - 1 - stack[::-1].index(marker)
        values = stack[pos + 1:]; del stack[pos:]
        return values
    def mapping_pairs(target, values):
        need(type(target) is dict and len(values) % 2 == 0, 'Primitive mapping updates required')
        for i in range(0, len(values), 2):
            key, value = values[i:i + 2]
            need(type(key) in (str, int) and key not in target, 'Unique primitive checkpoint map key')
            target[key] = value
    for op, arg, position in pickletools.genops(raw):
        count += 1; need(count <= 100000 and len(stack) <= 20000, 'Bounded pickle operations/stack')
        name = op.name
        if name == 'PROTO': need(arg == 2, 'Exact Torch protocol2 only')
        elif name == 'MARK': stack.append(marker)
        elif name in ('NONE', 'NEWTRUE', 'NEWFALSE'): stack.append({'NONE': None, 'NEWTRUE': True, 'NEWFALSE': False}[name])
        elif name in ('BININT', 'BININT1', 'BININT2', 'LONG1', 'LONG4', 'INT', 'LONG'):
            need(type(arg) is int and -1 <= arg <= 2**63 - 1, 'Bounded primitive integer')
            stack.append(arg)
        elif name in ('BINFLOAT', 'FLOAT'):
            need(math.isfinite(arg), 'Finite primitive float'); stack.append(arg)
        elif name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE', 'BINSTRING', 'SHORT_BINSTRING'):
            need(type(arg) in (str, bytes) and len(arg) <= 131072, 'Bounded primitive string'); stack.append(arg)
        elif name == 'EMPTY_DICT': stack.append({})
        elif name == 'EMPTY_LIST': stack.append([])
        elif name == 'EMPTY_TUPLE': stack.append(())
        elif name == 'TUPLE': stack.append(tuple(pop_mark()))
        elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
            n = int(name[-1]); need(len(stack) >= n, 'Complete tuple values')
            values = tuple(stack[-n:]); del stack[-n:]; stack.append(values)
        elif name == 'LIST': stack.append(pop_mark())
        elif name == 'DICT':
            values = pop_mark(); result = {}; mapping_pairs(result, values); stack.append(result)
        elif name == 'APPEND':
            value = stack.pop(); need(type(stack[-1]) is list, 'Primitive list append'); stack[-1].append(value)
        elif name == 'APPENDS':
            values = pop_mark(); need(type(stack[-1]) is list, 'Primitive list extends'); stack[-1].extend(values)
        elif name == 'SETITEM':
            value, key = stack.pop(), stack.pop(); mapping_pairs(stack[-1], [key, value])
        elif name == 'SETITEMS':
            values = pop_mark(); mapping_pairs(stack[-1], values)
        elif name in ('BINPUT', 'LONG_BINPUT', 'PUT'):
            index = int(arg); need(0 <= index <= 100000 and stack, 'Bounded memo store'); memo[index] = stack[-1]
        elif name in ('BINGET', 'LONG_BINGET', 'GET'):
            index = int(arg); need(index in memo, 'Known memo reference'); stack.append(memo[index])
        elif name == 'GLOBAL':
            need(arg in allowed, 'Unexpected checkpoint GLOBAL: ' + str(arg)); stack.append(Global(arg))
        elif name == 'BINPERSID':
            value = stack.pop()
            need(type(value) is tuple and len(value) == 5 and value[0] == 'storage' and
                 isinstance(value[1], Global) and value[1].name == 'torch FloatStorage' and
                 type(value[2]) is str and re_full_digits(value[2]) and value[3] == 'cpu' and
                 type(value[4]) is int and 0 < value[4] <= maximum // 4,
                 'Bounded explicit CPU FloatStorage persistent ID')
            key, n = value[2], value[4]
            need(storage_bytes.get(key) == n * 4 and storages.get(key, n) == n,
                 'Persistent storage size differs from verified ZIP bytes')
            storages[key] = n; stack.append(Storage(key, n))
        elif name == 'REDUCE':
            args, function = stack.pop(), stack.pop()
            need(isinstance(function, Global) and type(args) is tuple, 'Only named whitelisted reducer descriptions')
            if function.name == 'collections OrderedDict':
                need(args == (), 'Empty OrderedDict construction only'); stack.append({})
            else:
                need(function.name in ('torch._utils _rebuild_tensor_v2', 'torch._utils _rebuild_tensor')
                     and len(args) == (6 if function.name.endswith('_v2') else 4), 'Exact tensor reducer schema')
                storage, offset, shape, stride = args[:4]
                need(isinstance(storage, Storage) and type(offset) is int and offset >= 0 and
                     type(shape) is tuple and type(stride) is tuple and len(shape) == len(stride) <= 6 and
                     all(type(n) is int and 0 < n <= 65536 for n in shape) and
                     all(type(n) is int and n >= 0 for n in stride) and math.prod(shape) <= maximum // 4 and
                     offset + sum((n - 1) * s for n, s in zip(shape, stride)) < storage.count,
                     'Tensor view must fit bounded verified FloatStorage')
                if len(args) == 6:
                    need(args[4] is False and type(args[5]) is dict and not args[5], 'Detached tensor without hooks')
                tensors += 1; stack.append(Tensor(shape))
        elif name == 'STOP':
            need(position == len(raw) - 1 and len(stack) == 1 and type(stack[0]) is dict, 'Complete primitive dictionary pickle STOP')
            stop = True
        else:
            raise ValueError('Unsupported checkpoint pickle opcode: ' + name)
    need(stop and set(storages) == set(storage_bytes) and tensors <= 180,
         'Complete exact declared storage scope and at most180 moment/step tensors')
    return {'pickle_operations': count, 'storage_count': len(storages), 'tensor_count': tensors,
            'passive_only': True, 'reducer_execution': False}


def re_full_digits(value):
    return value.isascii() and value.isdigit() and str(int(value)) == value


def pt_preflight(path, maximum=MAX_SMALL):
    path = ordinary(path, maximum); before = pin(path, maximum)
    with path.open('rb') as stream:
        need(stream.read(4) == b'PK\x03\x04', 'Torch ZIP must start at byte0')
        stream.seek(-22, 2); tail = stream.read()
        need(tail[:4] == b'PK\x05\x06' and tail[-2:] == b'\0\0', 'Exact uncommented Torch ZIP ending')
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist(); names = [i.filename for i in infos]
        need(0 < len(infos) <= 260 and len(set(names)) == len(names) and not archive.comment and
             sum(i.file_size for i in infos) <= maximum, 'Bounded unique Torch ZIP members')
        roots, storages = set(), {}
        for item in infos:
            p = PurePosixPath(item.filename)
            need(p.as_posix() == item.filename and not p.is_absolute() and len(p.parts) >= 2 and
                 all(v not in ('', '.', '..') for v in p.parts) and '\\' not in item.filename and ':' not in item.filename and
                 item.compress_type == zipfile.ZIP_STORED and not item.flag_bits & 1 and
                 not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16), 'Ordinary exact Torch ZIP namespace')
            roots.add(p.parts[0])
        need(len(roots) == 1, 'One Torch ZIP root')
        prefix = next(iter(roots)) + '/'
        for item in infos:
            relative = item.filename[len(prefix):]
            if relative.startswith('data/'):
                key = relative[5:]; need(re_full_digits(key) and 0 < item.file_size <= maximum and
                                       item.file_size % 4 == 0, 'Bounded float storage ZIP member')
                storages[key] = item.file_size
            else:
                need(relative in ('data.pkl', 'byteorder', 'version', '.data/serialization_id',
                                  '.format_version', '.storage_alignment'), 'Unexpected Torch ZIP metadata')
                need(item.file_size <= (512 * 1024 if relative == 'data.pkl' else 4096), 'Bounded ZIP metadata')
            # CRC-check every bounded member before any unpickler runs.
            with archive.open(item) as stream:
                total = 0
                for block in iter(lambda: stream.read(1024**2), b''): total += len(block)
                need(total == item.file_size, 'Complete Torch ZIP payload/CRC')
        need(prefix + 'data.pkl' in names and prefix + 'byteorder' in names and
             archive.read(prefix + 'byteorder') == b'little', 'Explicit little-endian Torch archive')
        result = preflight_pickle(archive.read(prefix + 'data.pkl'), storages, maximum)
    need(pin(path, maximum) == before, 'Checkpoint changed during preflight')
    return {**result, 'file': before, 'CPU_only_and_weights_only_required': True}


def decode_cpu(path, torch, np):
    receipt = pt_preflight(path); before = receipt['file']
    value = torch.load(path, map_location='cpu', weights_only=True)
    nodes = [0]
    def convert(item, depth=0):
        nodes[0] += 1
        need(nodes[0] <= 12000 and depth <= 24, 'Bounded primitive CPU checkpoint tree')
        if isinstance(item, torch.Tensor):
            need(item.device.type == 'cpu' and item.dtype == torch.float32 and not item.requires_grad
                 and item.grad is None and item.numel() * 4 <= MAX_SMALL, 'Detached bounded FP32 CPU checkpoint tensor')
            array = item.detach().contiguous().numpy().copy()
            need(np.isfinite(array).all(), 'Finite CPU checkpoint values'); return array
        if type(item) is dict:
            need(all(type(key) in (str, int) for key in item), 'Primitive checkpoint keys')
            return {key: convert(v, depth + 1) for key, v in item.items()}
        if type(item) in (list, tuple): return type(item)(convert(v, depth + 1) for v in item)
        need(item is None or type(item) in (str, int, float, bool), 'Primitive checkpoint leaf')
        if type(item) is float: need(math.isfinite(item), 'Finite CPU scalar')
        return item
    value = convert(value)
    need(pin(path, MAX_SMALL) == before, 'Checkpoint changed during CPU decoding')
    return value, receipt


def validate_optimizer(opt, update, names, schema, np):
    need(type(update) is int and 0 <= update <= 8, 'Observed update0..8 required')
    need(type(opt) is dict and set(opt) == {'state', 'param_groups'} and
         type(opt['param_groups']) is list and len(opt['param_groups']) == 1, 'One explicit Adam group')
    group = opt['param_groups'][0]
    policy = {'lr': learning_rate(update), 'betas': (.9, .99), 'eps': 1e-8, 'weight_decay': 0.,
              'amsgrad': False, 'foreach': False, 'fused': False, 'capturable': False,
              'differentiable': False, 'maximize': False}
    need(type(group) is dict and set(group) in (set(policy) | {'params'}, set(policy) | {'params', 'decoupled_weight_decay'})
         and group['params'] == list(range(len(names))) and group.get('decoupled_weight_decay', False) is False,
         'Exact stable Adam parameter order/options')
    for key, wanted in policy.items():
        got = tuple(group[key]) if key == 'betas' else group[key]
        need(got == wanted and (type(got) is bool if type(wanted) is bool else type(got) is not bool), 'Exact saved Adam option: ' + key)
    need(type(opt['state']) is dict and set(opt['state']) == (set() if update == 0 else set(range(len(names)))),
         'Fresh moments at0 or complete saved moment census')
    out = {}
    for index, name in enumerate(names):
        if update == 0:
            out[name] = (np.zeros(schema[name], np.float32), np.zeros(schema[name], np.float32)); continue
        row = opt['state'][index]
        need(type(row) is dict and set(row) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Exact Adam moment record')
        step, first, second = row['step'], row['exp_avg'], row['exp_avg_sq']
        need(isinstance(step, np.ndarray) and step.dtype == np.float32 and step.shape == () and float(step) == update,
             'Exact scalar FP32 Adam step')
        need(all(isinstance(a, np.ndarray) and a.dtype == np.float32 and a.shape == schema[name] and
                 np.isfinite(a).all() for a in (first, second)) and (second >= 0).all(), 'Finite exact Adam moment values')
        out[name] = (first, second)
    return out


def independent_displacement(before, after, initial, np):
    need(set(before) == set(after) == set(initial) and bool(before), 'Same bound trainable coordinates')
    step = max(float(np.abs(after[n].astype(np.float64) - before[n].astype(np.float64)).max(initial=0)) for n in before)
    cumulative = max(float(np.abs(after[n].astype(np.float64) - initial[n].astype(np.float64)).max(initial=0)) for n in before)
    return {'actual_step_Linf': step, 'actual_cumulative_Linf': cumulative,
            'pass': 0 < step <= 1e-4 and 0 < cumulative <= 1e-3}


def diagnostic_norm_check(actual, recorded):
    """Compare saved float64 diagnostic norms within eight representational ULPs.

    This is never a quality-gate epsilon. Exact zero remains exact, and actual
    finite/positive gradient admission is checked separately without tolerance.
    The observed cross-platform discrepancy is one ULP at most in this return.
    """
    need(type(actual) is dict and type(recorded) is dict and bool(actual) and
         set(actual) == set(recorded), 'Exact per-site diagnostic norm scope')
    comparisons = {}
    for name, value in actual.items():
        reference = recorded[name]
        need(type(value) in (int, float) and type(reference) in (int, float) and
             math.isfinite(value) and math.isfinite(reference) and value >= 0 and reference >= 0,
             'Finite nonnegative diagnostic norms')
        if value == 0 or reference == 0:
            passed, ulps = value == reference == 0, 0. if value == reference else None
        else:
            ulps = abs(value - reference) / max(math.ulp(value), math.ulp(reference))
            passed = ulps <= DIAGNOSTIC_NORM_MAX_ULPS
        comparisons[name] = {'actual': value, 'recorded': reference,
                             'absolute_error': abs(value - reference), 'ULP_distance': ulps, 'pass': passed}
    return {'pass': all(row['pass'] for row in comparisons.values()), 'sites': comparisons,
            'maximum_allowed_ULPs': DIAGNOSTIC_NORM_MAX_ULPS,
            'representation_only_not_quality_epsilon': True, 'exact_zero_required': True}


def graph_witness(value, arm):
    need(type(value) is dict and value.get('status') == 'WITNESSED' and value.get('complete') is True and
         value.get('grad_enabled') is True and value.get('arm') == arm and
         set(value.get('sites', {})) == {'32', '64', '128', '256'}, 'Complete four-site actual graph receipt')
    for row in value['sites'].values():
        need(set(row) == {'stock_SFT', 'head_encoder'}, 'Both existing adapter branches')
        for branch in row.values():
            need(branch.get('pass') is True and branch.get('expected_present') is (arm == 'B') and
                 branch.get('current_adapter_leaves_reached') == (6 if arm == 'B' else 0) and
                 branch.get('autograd_queries_or_backward_calls') == 0 and
                 branch.get('nonzero_or_finite_branch_gradient_proven') is False,
                 'Saved structural graph receipt has exact limits, not numerical branch proof')


class Evidence:
    def __init__(self):
        self.began = time.monotonic(); self.used = {}; self.H = helper()
        self.transport = OUT / 'transport.json'; self.index_path = OUT / 'member_index.json'
        self.transport_pin = pin(self.transport); self.index_pin = pin(self.index_path)
        self.meta = strict_json(self.transport.read_bytes()); self.index = strict_json(self.index_path.read_bytes())
        need(self.meta.get('complete') is True and self.meta.get('archive', {}).get('sha256') == ARCHIVE_SHA and
             self.meta.get('archive', {}).get('bytes') == 2360260229 and
             self.meta.get('protocol', {}).get('sha256') == PROTOCOL_SHA and
             {key: self.meta['member_index'][key] for key in ('bytes', 'sha256')} == self.index_pin and
             self.index.get('archive_sha256') == ARCHIVE_SHA,
             'Complete hash-bound transport/index admission before saved-value decoding')
        self.root = OUT / 'retained' / STAGE
        need(self.root.is_dir() and self.root.resolve() == self.root.absolute() and
             self.meta.get('retained_root') == self.root.relative_to(ROOT).as_posix(), 'Exact retained data-only namespace')
        need(self.index.get('complete') is True, 'Complete archive member index')
        self.files = self.index['members']
        need(type(self.files) is dict and len(self.files) <= 15000, 'Bounded authenticated member file index')
    def tick(self): need(time.monotonic() - self.began < MAX_SECONDS, 'Finite600s saved-state audit')
    def file(self, name, maximum=MAX_FULL):
        self.tick()
        need(type(name) is str and name in self.files and '\\' not in name and ':' not in name and
             PurePosixPath(name).as_posix() == name and '..' not in PurePosixPath(name).parts,
             'Exact transport-indexed retained member')
        path = self.root / name; actual = pin(path, maximum)
        need(actual == self.files[name], 'Retained member bytes differ from transport hash: ' + name)
        self.used[name] = actual; return path
    def bound(self, value, maximum=MAX_SMALL):
        need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'Exact saved file binding')
        need(value['path'] in self.files and self.files[value['path']] == {k: value[k] for k in ('bytes', 'sha256')},
             'Saved binding differs from actual archive index')
        return self.file(value['path'], maximum)
    def json(self, name): return strict_json(self.file(name, MAX_SMALL).read_bytes())


def full_state(e, name, np, *, arm=None, initial=False):
    witness = e.json(name); rows = witness['arrays']
    need(witness['complete'] is True and witness['model_qualified'] is False and
         len(rows) == 575 and len({r['name'] for r in rows}) == 575, 'Complete unique575 witness')
    schema = {r['name']: tuple(r['shape']) for r in rows}
    need({k: v for k, v in schema.items() if k.startswith(('heads.', 'adapters.'))} == shapes('B'),
         'Independent original36 head/24 adapter shape schema')
    saved = e.H.read_npz(e.bound(witness['file'], MAX_FULL), schema, np, maximum=MAX_FULL,
                         retain=set(shapes('B')), tick=e.tick)
    need(saved['rows'] == sorted(rows, key=lambda r: r['name']) and
         saved['digests']['all'] == witness['canonical_sha256'] and
         saved['counts'] == {'all': 575, 'base': 515, 'adapters': 24, 'heads': 36, 'frozen': 539} and
         saved['digests']['base'] == R2_SHA, 'All actual575 arrays and frozen515 R2 exact')
    if initial:
        need(witness['update'] == 0 and saved['digests']['all'] == FULL_SHA and
             saved['digests']['heads'] == HEAD_SHA and saved['digests']['adapters'] == U96_SHA,
             'Exact complete original trained-u64 initializer')
    else:
        need(witness['arm'] == arm and witness['update'] == 8, 'Exact retained per-arm stop state8')
        if arm == 'A': need(saved['digests']['adapters'] == U96_SHA, 'ArmA frozen24 U96 exact')
    return witness, saved


def audit_saved(e, np, decode):
    protocol_path = SEALED / 'protocol.json'; protocol_pin = pin(protocol_path)
    need(protocol_pin['sha256'] == PROTOCOL_SHA, 'Exact locally sealed protocol')
    protocol = strict_json(protocol_path.read_bytes())
    status = e.json(RUN + '/status.json'); result = e.json(RUN + '/result.json')
    need(result['complete'] is True and result['worker_exit'] == 2 and result['quality_stop'] is True and
         result['model_qualified'] is False and status['protocol_sha256'] == result['protocol_sha256'] == PROTOCOL_SHA,
         'Actual complete recorded quality stop, never promotion')
    actual_execution = result['execution']
    need(actual_execution['status'] == 'FINISHED' and actual_execution['pending'] is None and
         actual_execution['committed_updates'] == {'A': 8, 'B': 8} and
         status['counters'] == actual_execution['counters'] and
         actual_execution['counters']['optimizer_updates'] == 16,
         'Exact known committed A8/B8 stop trajectory')
    initial_w, original = full_state(e, RUN + '/states/original_u64/state.json', np, initial=True)
    rng_hash = None; arm_results = {}; CPU_decodes = 0
    worst = dict.fromkeys(TOLERANCES, 0.)
    plan = protocol['operation_plan']; by_slot = {row['slot']: row for row in plan}
    for arm in ('A', 'B'):
        schema = shapes(arm); names = sorted(schema)
        previous = {n: original['arrays'][n] for n in names}; initial_values = previous
        previous_moments = None; verified = []
        for update in range(9):
            e.tick(); folder = RUN + '/arms/' + arm + '/checkpoints/u' + str(update).zfill(4)
            cp = e.json(folder + '/checkpoint.json')
            need(cp['complete'] is True and cp['arm'] == arm and cp['update'] == update and cp['model_qualified'] is False,
                 'Complete exact per-arm contiguous checkpoint0..8')
            state = e.H.read_npz(e.bound(cp['trainable']), schema, np, tick=e.tick)
            values = state['arrays']
            need(state['digests']['all'] == cp['trainable_canonical_sha256'], 'Checkpoint trainable canonical bytes')
            closed, preflight = decode(e.bound(cp['closed_state'])); CPU_decodes += 1
            fields = {'format', 'protocol_sha256', 'arm', 'update', 'automatic_resume_permitted',
                      'optimizer', 'optimizer_parameter_names', 'RNG', 'sampler_next_update',
                      'sampler_next_plan_call', 'initial_trainable_sha256', 'frozen_invariants',
                      'execution', 'model_qualified'}
            need(type(closed) is dict and set(closed) == fields and
                 closed['format'] == 'closed-matched-feature-state-v1' and closed['protocol_sha256'] == PROTOCOL_SHA and
                 closed['arm'] == arm and type(closed['update']) is int and closed['update'] == update and
                 closed['automatic_resume_permitted'] is False and closed['model_qualified'] is False and
                 closed['optimizer_parameter_names'] == names and closed['sampler_next_update'] == update + 1 and
                 closed['initial_trainable_sha256'] == e.H.canonical(initial_values, np) and
                 closed['frozen_invariants'] == cp['frozen_invariants'], 'Exact saved Adam/RNG/sampler lineage')
            invariant = closed['frozen_invariants']
            need(invariant['pass'] is True and invariant['arm'] == arm and
                 invariant['trainable_tensors'] == len(schema) and
                 invariant['trainable_parameters'] == sum(math.prod(v) for v in schema.values()) and
                 invariant['digests']['R2'] == R2_SHA and
                 (arm == 'B' or invariant['digests']['U96'] == U96_SHA), 'Declared arm/frozen value census')
            execution = closed['execution']
            need(execution['pending'] is None and execution['committed_updates'][arm] == update and
                 execution['counters']['optimizer_updates'] == (update if arm == 'A' else 8 + update) and
                 execution['per_arm'][arm]['optimizer_updates'] == update and
                 closed['sampler_next_plan_call'] == execution['next_forward'], 'Actual checkpoint execution/sampler commit binding')
            expected_next = next((row for row in plan if row['arm'] == arm and row['new_update'] > update), None)
            if update in (0,):
                expected_next = next(row for row in plan if row['arm'] == arm and row['backward'])
            if update == 8:
                expected_next = next(row for row in plan if row['arm'] == arm and row['phase'] == 'quality_snapshot' and row['new_update'] == 8)
            need(closed['sampler_next_plan_call'] == expected_next, 'Checkpoint next call equals exact prospective schedule')
            moments = validate_optimizer(closed['optimizer'], update, names, schema, np)
            current_rng = e.H.validate_rng(closed['RNG'])
            need(rng_hash in (None, current_rng), 'Saved seeded RNG unchanged throughout both arms'); rng_hash = current_rng
            if update == 0:
                need(all(values[n].tobytes() == previous[n].tobytes() for n in names), 'Checkpoint0 exact original trainables')
            else:
                commit = e.json(RUN + '/arms/' + arm + '/updates/u' + str(update).zfill(4) + '/commit.json')
                need(commit['complete'] is True and commit['arm'] == arm and commit['update'] == update and
                     commit['LR'] == learning_rate(update) and commit['projection_or_rollback_performed'] is False and
                     commit['before_canonical_sha256'] == e.H.canonical(previous, np) and
                     commit['after_canonical_sha256'] == state['digests']['all'], 'Exact actual before/after commit bindings')
                after = e.H.read_npz(e.bound(commit['actual_after_step']), schema, np, tick=e.tick)['arrays']
                need(all(after[n].tobytes() == values[n].tobytes() for n in names), 'Actual after-step state matches durable checkpoint')
                displacement = independent_displacement(previous, values, initial_values, np)
                need(displacement == commit['actual_displacement'] and displacement['pass'] is True,
                     'Actual coordinate step/cumulative bounds are retained without projection')
                pre = e.H.read_npz(e.bound(commit['preclip_gradients']), schema, np, tick=e.tick)['arrays']
                post = e.H.read_npz(e.bound(commit['postclip_gradients']), schema, np, tick=e.tick)['arrays']
                norm = commit['preclip_norm']; measured = math.sqrt(sum(float(np.square(v.astype(np.float64)).sum()) for v in pre.values()))
                need(type(norm) in (int, float) and math.isfinite(norm) and norm > 0 and
                     abs(measured - norm) <= 2e-7 + 2e-5 * measured, 'Recorded gradient norm from actual saved arrays')
                coefficient = min(1., 1. / (norm + 1e-6))
                need(all(np.allclose(post[n], (pre[n] * np.float32(coefficient)).astype(np.float32),
                                    atol=2e-9, rtol=2e-6) for n in names), 'Actual saved clipping arithmetic')
                per_site = {}
                for family in (('heads',) if arm == 'A' else ('heads', 'adapters')):
                    for site in ('32', '64', '128', '256'):
                        arrays = [v for n, v in pre.items() if n.startswith(family + '.' + site + '.')]
                        need(len(arrays) == (9 if family == 'heads' else 6), 'Exact gradient tensor site census')
                        per_site[family + '/' + site] = math.sqrt(sum(float(np.square(v.astype(np.float64)).sum()) for v in arrays))
                norm_comparison = diagnostic_norm_check(per_site, commit['per_site_preclip_norms'])
                need(norm_comparison['pass'] and
                     (all(v > 0 for v in per_site.values()) if update == 1 else any(v > 0 for v in per_site.values())),
                     'Actual finite positive first-update trainable-site gradients')
                errors = dict.fromkeys(TOLERANCES, 0.)
                for name in names:
                    m0, v0 = previous_moments[name]; actual_m, actual_v = moments[name]
                    prediction = e.H.adam_predict(previous[name], post[name], m0, v0, update, learning_rate(update), np,
                                                  actual_m=actual_m, actual_v=actual_v)
                    for kind, actual, expected in zip(TOLERANCES, (actual_m, actual_v, values[name]), prediction):
                        error = e.H.compare(actual, expected, np, kind)
                        errors[kind] = max(errors[kind], error); worst[kind] = max(worst[kind], error)
                calls = [r for r in plan if r['arm'] == arm and r['backward'] and r['new_update'] == update]
                need(len(calls) == len(commit['microbatches']) == 5, 'Exactly five fixed scheduled role losses')
                for call, binding in zip(calls, commit['microbatches']):
                    objective = strict_json(e.bound(binding).read_bytes())
                    need(objective['complete'] is True and objective['call'] == call and objective['arm'] == arm and
                         objective['update'] == update and objective['divisor_applied_once'] == 5 and
                         objective['quality_tolerance_added'] is False and objective['LR'] == learning_rate(update),
                         'Exact graph/objective five-role scheduled association')
                    graph_witness(objective['graph_witness'], arm)
                    if update == 1:
                        evidence = e.json(RUN + '/calls/' + str(call['slot']).zfill(4) + '/first_update_role_gradient_norms.json')
                        need(evidence['complete'] is True and evidence['arm'] == arm and evidence['role'] == call['id'].split('__', 1)[1] and
                             evidence['extra_autograd_or_backward_queries'] == 0 and
                             all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in evidence['site_norms'].values()),
                             'Five same-backward role diagnostic receipts, no extra derivative query')
                changed = sum(values[n].tobytes() != previous[n].tobytes() for n in names)
                need(changed > 0, 'Actual committed update must change trainable values')
                verified.append({'update': update, 'LR': learning_rate(update), 'changed_trainable_tensors': changed,
                                 'actual_displacement': displacement, 'preclip_L2_float64': measured,
                                 'per_site_preclip_norms': per_site, 'diagnostic_norm_scalar_comparison': norm_comparison,
                                 'maximum_recurrence_absolute_error': errors})
            previous, previous_moments = values, moments
        final_name = RUN + '/states/arm_' + arm + '_quality_failed_u0008/state.json'
        final_w, final = full_state(e, final_name, np, arm=arm)
        need(all(final['arrays'][n].tobytes() == previous[n].tobytes() for n in names), 'Full stop state matches final committed checkpoint8')
        need(status['last_checkpoint'][arm]['path'] == RUN + '/arms/' + arm + '/checkpoints/u0008/checkpoint.json',
             'Actual outcome last checkpoint points to verified8')
        e.bound(status['last_checkpoint'][arm]); e.bound(status['last_complete_state'][arm])
        arm_results[arm] = {'verified_checkpoints': list(range(9)), 'verified_updates': verified,
            'actual_trainable_tensors': len(names), 'actual_trainable_coordinates': sum(math.prod(v) for v in schema.values()),
            'full575_stop_canonical_sha256': final['digests']['all'], 'frozen515_R2_canonical_sha256': final['digests']['base'],
            'adapter_canonical_sha256': final['digests']['adapters'],
            'trainable_tensors_changed_from_original': sum(final['arrays'][n].tobytes() != initial_values[n].tobytes() for n in names),
            'same_committed_final575_and_checkpoint': True, 'quality_stopped_at': 8, 'later_updates_unattempted': list(range(9, 49))}
    need(pin(e.transport) == e.transport_pin and pin(e.index_path) == e.index_pin and pin(protocol_path) == protocol_pin,
         'Transport/index/protocol immutable through saved-state audit')
    for name, value in e.used.items(): need(pin(e.root / name) == value, 'Used state evidence changed at final readback')
    return {'complete': True, 'status': 'SAVED_COMMITTED_STATE_INTEGRITY_VERIFIED_QUALITY_STOPS_RETAINED',
        'protocol': protocol_pin, 'transport': e.transport_pin, 'index': e.index_pin,
        'arms': arm_results, 'complete_original575_initializer_exact': True,
        'frozen515_R2_exact_in_both_full_stop_states': True, 'Arm_A_frozen24_U96_exact': True,
        'maximum_recurrence_absolute_error': worst, 'Adam_FP32_arithmetic_tolerances_not_quality_epsilon': TOLERANCES,
        'diagnostic_norm_max_ULPs_not_quality_epsilon': DIAGNOSTIC_NORM_MAX_ULPS,
        'RNG_trajectory_unchanged_both_arms': True, 'CPU_weights_only_decodes': CPU_decodes,
        'used_files': e.used, 'seconds': time.monotonic() - e.began,
        'neural_model_optimizer_constructions': 0, 'forward_backward_optimizer_steps': 0, 'VM_calls': 0,
        'structural_graph_receipts_not_fresh_gradient_reproduction': True,
        'quality_stop_is_not_repaired_by_integrity': True, 'model_qualified': False}


def admission(review):
    review = ordinary(review, MAX_SMALL); value = strict_json(review.read_bytes())
    need(value.get('complete') is True and value.get('source_review_pass') is True and
         value.get('CPU_decode_permitted') is True and value.get('model_or_optimizer_construction_permitted') is False,
         'Peer source/input admission required before CPU decoding')
    sources = value['sources']
    for path in (Path(__file__), TEST, HELPER):
        name = path.resolve().relative_to(ROOT).as_posix()
        need(sources.get(name) == pin(path), 'Exact current reviewed saved-state auditor/test/helper bytes')
    need(value['transport'] == pin(OUT / 'transport.json') and value['index'] == pin(OUT / 'member_index.json'),
         'Peer admission binds completed exact transport/index')
    return pin(review)


def run_bounded(review):
    admitted = admission(review); output = OUT / 'state.json'
    need(not output.exists(), 'Existing state audit output immutable')
    H = helper(); binary, environment = H.current_launch_plan()
    command = [binary, '-B', str(Path(__file__).resolve()), '--_cpu-child', '--source-review', str(review)]
    began = time.monotonic(); peak = 0
    child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            try:
                stdout, stderr = child.communicate(timeout=.25); break
            except subprocess.TimeoutExpired:
                need(time.monotonic() - began < MAX_SECONDS + 30, 'Absolute630s CPU saved-state audit stop')
                if child.poll() is None:
                    peak = max(peak, H.rss_bytes(child.pid)); need(peak <= MAX_RSS, 'Saved-state audit2GiB RSS cap')
        need(child.returncode == 0 and not stderr and len(stdout) <= 8 * 1024**2,
             'CPU saved-state audit failed: ' + stderr.decode('utf-8', 'replace')[-8000:])
        result = strict_json(stdout)
        result.update(auditor=pin(Path(__file__)), tests=pin(TEST), source_review=admitted,
                      separate_CPU_process=True, peak_child_RSS_bytes=peak, CUDA_used=False)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        return {'complete': True, 'output': str(output), 'sha256': pin(output)['sha256'], 'model_qualified': False}
    finally:
        if child.poll() is None:
            child.kill(); child.communicate(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--run', action='store_true'); modes.add_argument('--_cpu-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--source-review', type=Path, required=True)
    args = parser.parse_args()
    if args.run:
        print(json.dumps(run_bounded(args.source_review))); return
    H = helper(); context = H.current_child_context(); admission(args.source_review)
    evidence = Evidence()
    # No returned source, model class or optimizer class is imported.
    import numpy as np
    import torch
    need(not torch.cuda.is_initialized(), 'No CUDA in saved-value audit')
    result = audit_saved(evidence, np, lambda path: decode_cpu(path, torch, np))
    result['CPU_child_context'] = context
    need(not torch.cuda.is_initialized(), 'No CUDA initialized by weights-only CPU decoding')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
