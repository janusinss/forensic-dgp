"""Bounded independent saved-pixel/objective/gate readback, without model calls.

Consumes the transport review's hash-bound extracted files, never imports or
executes returned sources, and never opens or extracts the results archive.
NumPy objective arithmetic is a saved-array readback, not CUDA/Torch replay.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'cctv_codeformer_u64_cleananchors_continuation_v1_vm'
RUN = 'outputs/u64_cleananchors_continuation_v1'
PROTOCOL_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
PARENT_SHA = 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'
HISTORY_SHA = '8b73cbb901452faff8b3679bcf9d077d4d8c98a90e461c18af260565fd04c18d'
VISUAL_SHA = '5a2c69d93e5c724144fcff07e136d2d281e81234c274d2110841a1df0904ca22'
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
VIEWS = ('raw', 'native_png')
FIELDS = ('mae', 'adjacent_rgb_error')
METRIC_READBACK_BOUND = 1e-15
OBJECTIVE_READBACK_BOUND = 5e-7
INVENTORY_SHA = '931ed0b38ccd9cedeb20e3ad28f189ced976b81df3f8652ed91a154713018c0a'
INVENTORY_BYTES = 455210
SOURCE_PINS = {
    'research/codeformer_u64_cleananchors_continuation_v1/worker.py': (48859, '6b84fb697da9324de10c8dc500f2f1259a9df6dd1770e6bcf89d3e681dee6e80'),
    'research/codeformer_u64_cleananchors_continuation_v1/metrics.py': (11967, '030649b25df131750460ae51767bdc0254d0017bcf7e9597bc169b7d4c8f386e'),
    'research/codeformer_u64_cleananchors_continuation_v1/objective.py': (8205, 'cffafed3feae10ce91890cfd68ee5de96c8db8edad00041420615cd071f2b7ad'),
    'research/codeformer_u64_cleananchors_continuation_v1/contract.py': (10844, '99e1bbef911750c5ef0622d350e203bcafed58559b3c2a2c500744ef76102bb0'),
    'scripts/audit_u96_decoder_readout_pixels_v1.py': (31186, 'c1b9bb3a7d30d4a7008428e7680a2d55ce472694e85f16d9f17485c82c9084ae'),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    def finite(item):
        if type(item) is float:
            need(math.isfinite(item), 'Finite JSON scalar required')
        elif isinstance(item, (list, dict)):
            for child in (item.values() if isinstance(item, dict) else item):
                finite(child)
    finite(value)
    return value


def relative(value):
    need(type(value) is str and value and '\\' not in value and ':' not in value,
         'Canonical relative POSIX path required')
    p = PurePosixPath(value)
    need(not p.is_absolute() and p.as_posix() == value and
         all(part not in ('', '.', '..') for part in value.split('/')),
         'Confined relative path required')
    return p


def read_regular(base, name, maximum):
    p = Path(base).joinpath(*relative(name).parts)
    need(p.resolve(strict=True).is_relative_to(Path(base).resolve(strict=True)), 'Confined file required')
    for ancestor in (p, *p.parents):
        info = ancestor.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'No symbolic/reparse input ancestry')
        if ancestor == Path(base):
            break
    before = p.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= maximum,
         'Bounded single-link ordinary file required: ' + name)
    raw = p.read_bytes()
    after = p.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'Input changed during read')
    return raw


def binding(raw):
    return {'bytes': len(raw), 'sha256': sha(raw)}


def npy(raw, name):
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    need(version in ((1, 0), (2, 0)), 'Only bounded NPY1/2 accepted')
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream, max_header_size=4096)
    shapes = {'raw.npy': (256, 256, 3), 'unclipped.npy': (1, 3, 256, 256),
              'native.npy': (1, 3, 512, 512), 'logits.npy': (1, 256, 1024),
              'LQ.npy': (1, 256, 16, 16)}
    need(PurePosixPath(name).name in shapes, 'Explicit array role required')
    need(shape == shapes[PurePosixPath(name).name] and dtype == np.dtype('float32')
         and not dtype.hasobject and not fortran, 'Exact finite C-order FP32 shape required')
    need(stream.tell() + math.prod(shape) * 4 == len(raw), 'Exact NPY extent required')
    result = np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(shape)
    need(np.isfinite(result).all(), 'Finite saved array required')
    if PurePosixPath(name).name == 'raw.npy':
        need(result.min() >= 0 and result.max() <= 1, 'Delivered RAW in [0,1] required')
    return result


def static_raw(raw, row):
    """A sealed static teacher/control can have a content-addressed basename."""
    need(type(row) is dict and set(row) == {'path','bytes','sha256'}, 'Exact static RAW binding required')
    relative(row['path'])
    need(binding(raw) == {key:row[key] for key in ('bytes','sha256')}, 'Pinned static RAW bytes required')
    return npy(raw, 'raw.npy')


def png(raw):
    need(len(raw) <= 256 * 1024, '256KiB RGB PNG cap')
    with Image.open(io.BytesIO(raw)) as image:
        need(image.format == 'PNG' and image.mode == 'RGB' and image.size == (256, 256)
             and getattr(image, 'n_frames', 1) == 1, 'Single RGB256 PNG required')
        return np.asarray(image).copy()


def floor_png(raw):
    need(raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all()
         and raw.min() >= 0 and raw.max() <= 1, 'Delivered FP32 RAW required')
    return (raw * np.float32(255)).astype(np.uint8)


def masks():
    middle = np.zeros((256, 256), dtype=bool)
    middle[64:192, 64:192] = True
    return {'whole': np.ones_like(middle), 'central128': middle, 'outside': ~middle}


MASKS = masks()


def regional(raw, target, mask):
    residual = raw.astype(np.float64) - target.astype(np.float64)
    dh, dv = np.abs(residual[:, 1:] - residual[:, :-1]), np.abs(residual[1:] - residual[:-1])
    hm, vm = mask[:, 1:] & mask[:, :-1], mask[1:] & mask[:-1]
    edge = (.5 * (dh.mean() + dv.mean()) if mask.all() else
            (dh[hm].sum() + dv[vm].sum()) / (3 * int(hm.sum() + vm.sum())))
    return {'mae': float(np.abs(residual[mask]).mean()), 'adjacent_rgb_error': float(edge)}


def blocks(raw):
    return (raw.astype(np.float64) @ np.asarray((.2126, .7152, .0722), dtype=np.float64)
            ).reshape(16, 16, 16, 16).mean(axis=(1, 3))


def pixel_metrics(raw, quantized, target, *, control=None, teacher=None):
    result = {'metrics': {}, 'regional_metrics': {}}
    for view, value in (('raw', raw), ('native_png', quantized.astype(np.float32) / np.float32(255))):
        regions = None if target is None else {key: regional(value, target, mask) for key, mask in MASKS.items()}
        result['regional_metrics'][view] = regions
        result['metrics'][view] = dict.fromkeys(FIELDS, None) if regions is None else regions['whole']
        if control is not None:
            reference = control[0] if view == 'raw' else control[1].astype(np.float32) / np.float32(255)
            result.setdefault('U96_behavior_distance', {})[view] = float(np.abs(value.astype(np.float64) - reference.astype(np.float64)).mean())
            result.setdefault('U96_illumination_distance', {})[view] = float(np.abs(blocks(value) - blocks(reference)).mean())
        if teacher is not None:
            reference = teacher[0] if view == 'raw' else teacher[1].astype(np.float32) / np.float32(255)
            result.setdefault('behavior_distance', {})[view] = float(np.abs(value.astype(np.float64) - reference.astype(np.float64)).mean())
    if control is None:
        result['behavior_movement_reference_available'] = False
    return result


def objective(raw, target, role, precision=np.float64):
    """Independent CPU arithmetic; neither precision replays CUDA reduction order."""
    need(role in ROLES and raw.shape == target.shape == (256, 256, 3), 'Paired photographic role required')
    need(np.isfinite(raw).all() and np.isfinite(target).all() and raw.min() >= 0
         and raw.max() <= 1 and target.min() >= 0 and target.max() <= 1, 'Finite [0,1] objective arrays required')
    residual = raw.astype(precision) - target.astype(precision)
    def penalty(a, smooth=False):
        a = np.abs(a)
        if smooth:
            beta = precision(.01)
            a = np.where(a < beta, precision(.5) * a * a / beta, a - precision(.5) * beta)
        return a.mean(dtype=precision)
    def terms(mask, smooth=False):
        h, v = mask[:, 1:] & mask[:, :-1], mask[1:] & mask[:-1]
        edges = np.concatenate(((residual[:, 1:] - residual[:, :-1])[h].reshape(-1),
                                (residual[1:] - residual[:-1])[v].reshape(-1)))
        return penalty(residual[mask], smooth), penalty(edges, smooth)
    if role == ROLES[0]:
        di, ei = terms(MASKS['central128']); do, eo = terms(MASKS['outside'])
        values = dict(inside_image_L1=float(di), outside_image_L1=float(do),
                      inside_adjacent_L1=float(ei), outside_adjacent_L1=float(eo))
        total = precision(.5) * (di + do) + precision(.125) * (ei + eo)
    elif role == 'clear':
        di, ei = terms(MASKS['whole']); si, se = terms(MASKS['whole'], True)
        values = dict(photographic_image_anchor_L1=float(di), photographic_adjacent_anchor_L1=float(ei),
                      photographic_image_anchor_SmoothL1=float(si), photographic_adjacent_anchor_SmoothL1=float(se))
        total = precision(2) * (si + precision(.25) * se)
    else:
        di, ei = terms(MASKS['whole']); total = di + precision(.25) * ei
        values = dict(target_image_L1=float(di), target_adjacent_L1=float(ei))
        if role == ROLES[-1]:
            weights = np.asarray((.2126, .7152, .0722), dtype=precision)
            def pooled(value):
                return (value.astype(precision) * weights).sum(axis=2, dtype=precision).reshape(16, 16, 16, 16).mean(axis=(1, 3), dtype=precision)
            difference = pooled(raw) - pooled(target); anchor = penalty(difference, True)
            values.update(photographic_block16_luminance_anchor_L1=float(penalty(difference)),
                          photographic_block16_luminance_anchor_SmoothL1=float(anchor))
            total += precision(2) * anchor
    return float(total), values


def ordered(values):
    total = 0.0
    for value in values:
        total += float(value)
    return total


def means(bank, monitors):
    result = {v: {} for v in VIEWS}
    for role in ROLES:
        ids = sorted(r['id'] for r in monitors if r['role'] == role)
        need(len(ids) == 8, 'Eight monitors per role required')
        region = 'central128' if role == ROLES[0] else 'whole'
        for view in VIEWS:
            result[view][role] = {field: ordered(bank[k]['regional_metrics'][view][region][field] for k in ids) / 8 for field in FIELDS}
    return result


def gates(bank, initial, frozen, parent, update):
    """Independent exact-threshold gates; original references never rebase to u64."""
    need(update in (0, 4, 8, 16), 'Explicit planned bank update required')
    monitors = [r for r in parent['cases'] if r['cohort'] == 'monitor']; guards = parent['guards']
    ids = {r['id'] for r in monitors + guards}
    need(len(monitors) == 32 and len(guards) == 24 and len(ids) == 56
         and all(set(b) == ids for b in (bank, initial, frozen)), 'All exact56 bank cases required')
    legacy, ordinary, terminal, extra = [], [], [], {}
    for role in ('clear', 'dark', 'filter'):
        rows = sorted((r for r in guards if r['role'] == role), key=lambda r:r['id'])
        need(len(rows) == 8, 'Eight legacy rows per role required')
        for view in VIEWS:
            for field in (FIELDS if role == 'filter' else ('behavior_distance',)):
                def read(row):
                    return row[field][view] if field == 'behavior_distance' else row['metrics'][view][field]
                before = [read(r) for r in rows]; now = [read(bank[r['id']]) for r in rows]
                values = [b-a for a, b in zip(before, now)] if role == 'filter' else now
                limits = [.03] * 8 if role == 'filter' else [max(.03, v) for v in before]
                limit = .01 if role == 'filter' else max(.01, ordered(before) / 8)
                bad = [r['id'] for r, value, bound in zip(rows, values, limits) if value > bound]
                mean = ordered(values) / 8
                legacy.append(dict(role=role, view=view, field=field, mean=mean, mean_limit=limit,
                                   measurements=dict(zip((r['id'] for r in rows), values)), failed_ids=bad,
                                   **{'pass': not bad and mean <= limit}))
    for role in ROLES:
        selected = sorted(r['id'] for r in monitors if r['role'] == role)
        need(len(selected) == 8, 'Eight monitor rows per role required')
        for view in VIEWS:
            for region in (tuple(MASKS) if role == ROLES[0] else ('whole',)):
                for field in FIELDS:
                    delta = [bank[k]['regional_metrics'][view][region][field] - frozen[k]['regional_metrics'][view][region][field] for k in selected]
                    average = ordered(delta) / 8
                    ordinary.append(dict(role=role, view=view, field=region+'/'+field, mean=average, worst=max(delta),
                                         **{'pass': average <= .01 and max(delta) <= .03}))
            if role in ('clear', ROLES[-1]):
                field = 'U96_behavior_distance' if role == 'clear' else 'U96_illumination_distance'
                values = [bank[k][field][view] for k in selected]
                ordinary.append(dict(role=role, view=view, field=field, mean=ordered(values)/8, worst=max(values),
                                     **{'pass': ordered(values)/8 <= .01 and max(values) <= .03}))
    start, now, base = (means(b, monitors) for b in (initial, bank, frozen))
    for view in VIEWS:
        for role in ROLES[:2]:
            a, b, c = (x[view][role]['adjacent_rgb_error'] for x in (base, start, now))
            need(a > 0, 'Positive original U96 denominator required')
            gain = (a-c)/a
            terminal.append(dict(role=role, view=view, field='one_percent_blur', initial_mean=a,
                                 current_mean=c, gain_fraction=gain, **{'pass': gain >= .01}))
            key = view+'/'+role
            if update == 0:
                extra[key+'/retained_original_one_percent'] = (a-b)/a >= .01
            if update == 16:
                extra[key+'/retained_original_one_percent'] = gain >= .01
                extra[key+'/retain_u64_structure'] = c <= b
                extra[key+'/blur_target_MAE_nonincrease'] = now[view][role]['mae'] <= start[view][role]['mae']
        for field in FIELDS:
            a, b = base[view][ROLES[-1]][field], now[view][ROLES[-1]][field]
            terminal.append(dict(role=ROLES[-1], view=view, field='dark_'+field,
                                 initial_mean=a, current_mean=b, **{'pass': b <= a}))
        if update in (8, 16):
            extra[view+'/clear_target_MAE_improves'] = now[view]['clear']['mae'] < start[view]['clear']['mae']
            a, b = start[view][ROLES[-1]]['mae'], now[view][ROLES[-1]]['mae']
            extra[view+'/dark_target_MAE'] = b < a if update == 16 else b <= a
    lp, mp, tp, ep = (all(r['pass'] for r in legacy), all(r['pass'] for r in ordinary),
                      all(r['pass'] for r in terminal), all(extra.values()))
    return {'update': update, 'phase': 'initializer' if update == 0 else 'u'+str(update),
            'pass': lp and mp and ep and (update != 16 or tp), 'legacy_pass': lp, 'ordinary_monitor_pass': mp,
            'original_terminal_pass': tp, 'original_terminal_binding': update == 16,
            'additional_retention_pass': ep, 'legacy': legacy, 'ordinary_monitor': ordinary,
            'original_terminal': terminal, 'additional_retention': extra,
            'cohort_means': {'initial_u64': start, 'current': now, 'frozen_U96': base},
            'quality_threshold_epsilon': 0, 'model_qualified': False}


def operation_plan(parent):
    learning = parent['learning']; train = {(r['reference_id'], r['role']): r for r in parent['cases'] if r['cohort'] == 'optimize'}
    by = {(r['reference_id'], r['role']): r for r in parent['cases']}
    monitors = [by[(ref, role)] for ref in learning['monitor_reference_order'] for role in ROLES]
    bank = monitors + parent['guards']; plan = []
    def add(row, phase, update, backward=False):
        plan.append(dict(slot=len(plan)+1, id=row['id'], phase=phase, new_update=update, backward=backward,
                         optimizer_application_after_this_forward=backward and row['role'] == ROLES[-1]))
    for row in bank: add(row, 'initial_u64_quality_bank', 0)
    for row in monitors[:4]: add(row, 'deterministic_same_u64_repeat', 0)
    order = learning['optimization_reference_order']
    for update in range(1, 17):
        for role, offset in zip(ROLES, (0, 4, 8, 12)):
            add(train[(order[(update-1+offset) % 16], role)], 'paired_TRAIN_optimization', update, True)
        if update in (4, 8, 16):
            for row in bank: add(row, 'new_quality_snapshot', update)
    need(len(plan) == 292 and len({r['id'] for r in parent['cases']+parent['guards']}) == 120,
         'Fixed292-call120-case contract required')
    return plan


class Comparison:
    def __init__(self):
        self.checks = Counter(); self.failures = []; self.differences = []

    def check(self, okay, label, detail=None):
        self.checks[label] += 1
        if not okay:
            self.failures.append({'check': label, 'detail': detail})

    def same(self, actual, saved, label, epsilon=METRIC_READBACK_BOUND):
        if isinstance(actual, dict):
            self.check(isinstance(saved, dict) and set(actual) == set(saved), label+'/keys')
            for key, value in actual.items():
                if isinstance(saved, dict) and key in saved: self.same(value, saved[key], label+'/'+key, epsilon)
        elif isinstance(actual, list):
            self.check(isinstance(saved, list) and len(actual) == len(saved), label+'/length')
            if isinstance(saved, list):
                for i, (a, b) in enumerate(zip(actual, saved)): self.same(a, b, label+'/'+str(i), epsilon)
        elif type(actual) is float and type(saved) in (int, float):
            delta = abs(actual-saved)
            if delta: self.differences.append(dict(field=label, actual=actual, saved=saved, absolute_difference=delta))
            self.check(delta <= epsilon, label, {'actual': actual, 'saved': saved, 'difference': delta})
        else:
            self.check(type(actual) == type(saved) and actual == saved, label, {'actual': actual, 'saved': saved})


def compare_gate(recomputed, saved, compare):
    for actual, field in ((recomputed['pass'], 'pass'), (recomputed['phase'], 'phase')):
        compare.same(actual, saved[field], 'gate/'+field)
    compare.same(recomputed['legacy_pass'], saved['legacy']['pass'], 'gate/legacy_pass')
    compare.same(recomputed['ordinary_monitor_pass'], saved['ordinary_monitor']['pass'], 'gate/ordinary_monitor_pass')
    compare.same(recomputed['original_terminal_pass'], saved['original_terminal']['pass'], 'gate/original_terminal_pass')
    compare.same(recomputed['original_terminal_binding'], saved['original_terminal']['binding'], 'gate/original_terminal_binding')
    compare.same(recomputed['additional_retention_pass'], saved['additional_retention']['pass'], 'gate/additional_retention_pass')
    compare.same(recomputed['additional_retention'], saved['additional_retention']['checks'], 'gate/additional_checks')
    compare.same(recomputed['cohort_means'], saved['cohort_means'], 'gate/means')
    for row in recomputed['legacy']:
        old = saved['legacy']['groups'][row['role']]['views'][row['view']][row['field']]
        for key in ('pass', 'mean', 'mean_limit', 'measurements', 'failed_ids'):
            compare.same(row[key], old[key], 'gate/legacy/'+key)
    for row in recomputed['ordinary_monitor']:
        old = saved['ordinary_monitor']['groups'][row['role']]['views'][row['view']][row['field']]
        compare.same(row['pass'], old['pass'], 'gate/ordinary/pass')
        values = old.get('deltas', old.get('values'))
        compare.same(row['mean'], ordered(values.values())/8, 'gate/ordinary/mean')
        compare.same(row['worst'], max(values.values()), 'gate/ordinary/worst')
    for row in recomputed['original_terminal']:
        old = saved['original_terminal']['groups'][row['role']]['views'][row['view']][row['field']]
        for key in ('pass', 'initial_mean', 'current_mean', 'gain_fraction'):
            if key in row: compare.same(row[key], old[key], 'gate/terminal/'+key)


def audit(inventory_path, stage, output):
    """Only explicit invocation reads hash-bound extraction; no archive access."""
    inventory_path, stage, output = map(lambda p: Path(p).resolve(), (inventory_path, stage, output))
    need(output.is_relative_to(ROOT/'outputs') and not output.exists(), 'Fresh workspace output directory required')
    inventory_bytes = read_regular(inventory_path.parent, inventory_path.name, 16*1024**2)
    need(len(inventory_bytes) == INVENTORY_BYTES and sha(inventory_bytes) == INVENTORY_SHA,
         'Exact independently supplied transport inventory required')
    inventory = strict_json(inventory_bytes)
    need(inventory['complete'] is True and len(inventory['files']) <= 8000, 'Complete bounded transport inventory required')
    evidence = ROOT.joinpath(*relative(inventory['evidence_root']).parts).resolve(strict=True)
    need(evidence.is_relative_to(ROOT/'outputs'), 'Workspace extracted evidence required')
    index, extracted = inventory['files'], inventory['extracted_files']
    c = Comparison(); consumed = {}
    for name, (size, digest) in SOURCE_PINS.items():
        actual = read_regular(ROOT, name, 1024**2)
        need(len(actual) == size and sha(actual) == digest, 'Exact local reviewed formula source required')
    def read(name, maximum=8*1024**2):
        need(name in index and name in extracted, 'Inventoried extracted input required: '+name)
        need(index[name] == extracted[name], 'Extraction and transport binding disagree')
        value = read_regular(evidence, name, maximum)
        need(binding(value) == index[name], 'Extracted bytes differ from inventory: '+name)
        consumed[name] = index[name]
        return value
    protocol_bytes = read_regular(stage, 'protocol.json', 1024**2)
    need(sha(protocol_bytes) == PROTOCOL_SHA, 'Exact sealed continuation protocol required')
    p = strict_json(protocol_bytes)
    need(inventory['protocol']['sha256'] == PROTOCOL_SHA, 'Transport protocol mismatch')
    cache = {}
    def source(row, image=False):
        name = row['path']; relative(name)
        need({k:row[k] for k in ('bytes','sha256')} == p['files'][name] == index[name], 'Exact returned static source binding required')
        if name not in cache:
            raw = read_regular(stage, name, 8*1024**2)
            need(binding(raw) == p['files'][name], 'Exact sealed static source bytes required')
            cache[name] = png(raw) if image else static_raw(raw, row)
            need(sum(a.nbytes for a in cache.values()) <= 256*1024**2, '256MiB source cache cap')
        return cache[name]
    def static_json(row, expected):
        raw = read_regular(stage, row['path'], 8*1024**2)
        need(binding(raw) == {k:row[k] for k in ('bytes','sha256')} == index[row['path']]
             and sha(raw) == expected, 'Exact static JSON contract required')
        return strict_json(raw)
    parent = static_json(p['parent_protocol'], PARENT_SHA)
    history = static_json(p['historical_pixel_audit'], HISTORY_SHA)
    visual = static_json(p['visual_plan'], VISUAL_SHA)
    plan = operation_plan(parent); all_rows = {r['id']:r for r in parent['cases']+parent['guards']}
    c.check(p['cases'] == parent['cases'] and p['guards'] == parent['guards'], 'immutable_parent_cases')
    controls = {r['id']:r for r in p['controls']}; parity = {r['id']:r for r in p['u64_parity']}
    need(len(controls) == 56 and len(parity) == 4, 'Exact56 controls/four witnesses required')
    def pair(row): return source(row['raw']), source(row['png'], True)
    frozen = {}; historical_u64 = {}
    def payload(row):
        return {key: row[key] for key in ('metrics','regional_metrics','U96_behavior_distance','U96_illumination_distance','behavior_distance') if key in row}
    historical_frozen = {}
    for old in history['all_saved_pixel_metrics']:
        if old['id'] in controls and old['kind'] == 'reference' and old['update'] == 0:
            need(old['id'] not in historical_frozen, 'Unique historical U96 record')
            historical_frozen[old['id']] = old
        if old['id'] in controls and old['kind'] == 'snapshot' and old['update'] == 64:
            need(old['id'] not in historical_u64, 'Unique historical terminal record')
            historical_u64[old['id']] = payload(old)
    need(set(historical_u64) == set(controls), 'All56 historical u64 rows required')
    need(set(historical_frozen) == set(controls), 'All56 historical U96 rows required')
    for cid, reference in controls.items():
        spec = all_rows[cid]; raw, quantized = pair(reference)
        c.check(np.array_equal(floor_png(raw), quantized), 'cached_U96_RAW_floor_PNG', cid)
        target = None if spec.get('target') is None else source(spec['target'], True).astype(np.float32)/np.float32(255)
        teacher = None if not spec.get('behavior_teacher') else pair(spec['behavior_teacher'])
        frozen[cid] = pixel_metrics(raw, quantized, target, control=(raw, quantized), teacher=teacher)
        for field, oldfield in (('raw','raw_file'),('png','png_file')):
            c.same({k:reference[field][k] for k in ('bytes','sha256')},
                   {k:historical_frozen[cid][oldfield][k] for k in ('bytes','sha256')}, 'cached_control_historical_file_join',0)
        expected = payload(historical_frozen[cid])
        expected.update(U96_behavior_distance=dict.fromkeys(VIEWS,0.0), U96_illumination_distance=dict.fromkeys(VIEWS,0.0))
        c.same(frozen[cid],expected,'cached_control_historical_metrics')
    records, banks, objectives, initial_records = [], {u:{} for u in (0,4,8,16)}, [], {}
    available_slots = []; missing_slots = []; incomplete_slots = []; parity_rows = []; repeats = []; partial_pixels = []; missing_objectives = []
    for call in plan:
        slot = call['slot']; stem = RUN+'/calls/'+str(slot).zfill(4); record_name = stem+'/record.json'
        if record_name not in index:
            partial = [name for name in index if name.startswith(stem+'/')]
            (incomplete_slots if partial else missing_slots).append(slot)
            decoded = {}
            for name in partial:
                leaf=PurePosixPath(name).name
                if leaf in ('raw.npy','unclipped.npy','native.npy','logits.npy','LQ.npy','result.png'):
                    data=read(name)
                    decoded[leaf]=png(data) if leaf=='result.png' else npy(data,leaf)
            if 'raw.npy' in decoded and 'result.png' in decoded:
                c.check(np.array_equal(floor_png(decoded['raw.npy']),decoded['result.png']), 'partial_RAW_floor_PNG_exact',slot)
            if decoded:
                partial_row={'slot':slot,'id':call['id'],'phase':call['phase'],
                    'decoded_available_files':sorted(decoded),'record_missing':True,'bank_membership_not_inferred':True}
                if 'raw.npy' in decoded and 'result.png' in decoded:
                    spec=all_rows[call['id']]
                    target=None if spec.get('target') is None else source(spec['target'],True).astype(np.float32)/np.float32(255)
                    control=pair(controls[call['id']]) if call['id'] in controls else None
                    teacher=pair(spec['behavior_teacher']) if spec.get('behavior_teacher') else None
                    partial_row['recomputed_available_pixel_metrics']=pixel_metrics(decoded['raw.npy'],decoded['result.png'],target,control=control,teacher=teacher)
                partial_pixels.append(partial_row)
            continue
        record_bytes = read(record_name, 128*1024); record = strict_json(record_bytes); spec = all_rows[call['id']]
        c.same(record['call'], call, 'record_exact_call_plan', 0)
        for field in ('id','reference_id','role','split','input','target'):
            c.same(record.get(field), spec.get(field), 'record_'+field, 0)
        c.same(record['update'], call['new_update'], 'record_update', 0)
        def output_file(row, expected_name):
            need(row['path'] == stem+'/'+expected_name, 'Exact call-owned output path required')
            value = read(row['path']);c.same(binding(value), {k:row[k] for k in ('bytes','sha256')}, 'output_binding', 0)
            return value
        raw = npy(output_file(record['outputs']['raw'], 'raw.npy'), 'raw.npy')
        q = png(output_file(record['outputs']['png'], 'result.png'))
        un = npy(output_file(record['outputs']['unclipped'], 'unclipped.npy'), 'unclipped.npy')
        c.check(np.array_equal(floor_png(raw), q), 'RAW_floor_PNG_exact', slot)
        target = None if spec.get('target') is None else source(spec['target'], True).astype(np.float32)/np.float32(255)
        control = pair(controls[call['id']]) if call['id'] in controls else None
        teacher = pair(spec['behavior_teacher']) if spec.get('behavior_teacher') else None
        measured = pixel_metrics(raw, q, target, control=control, teacher=teacher)
        c.same(measured, {key:record.get(key) for key in measured}, 'saved_metrics')
        if control is None:
            c.check(not any(key in record for key in ('U96_behavior_distance','U96_illumination_distance','behavior_distance')),
                    'training_missing_reference_not_fabricated', slot)
        if call['phase'] == 'initial_u64_quality_bank':
            banks[0][call['id']] = measured; initial_records[call['id']] = record
            c.same(payload(record), historical_u64[call['id']], 'exact_initial_saved_historical_metrics', 0)
            if call['id'] in parity:
                old_raw, old_png = pair(parity[call['id']])
                exact_raw, exact_png = np.array_equal(raw, old_raw), np.array_equal(q, old_png)
                c.check(exact_raw and exact_png, 'four_historical_u64_RAW_PNG_parities', call['id'])
                parity_rows.append(dict(id=call['id'], slot=slot, raw_exact=exact_raw, PNG_pixels_exact=exact_png))
        elif call['phase'] == 'new_quality_snapshot':
            banks[call['new_update']][call['id']] = measured
        elif call['phase'] == 'deterministic_same_u64_repeat':
            prior = initial_records[call['id']]
            for field in ('raw','unclipped'):
                c.same({k:record['outputs'][field][k] for k in ('bytes','sha256')},
                       {k:prior['outputs'][field][k] for k in ('bytes','sha256')}, 'repeat_'+field+'_bytes', 0)
            c.check(np.array_equal(q, png(read(prior['outputs']['png']['path']))), 'repeat_PNG_pixels', slot)
            repeats.append(slot)
        for name in ('native','logits','LQ'):
            stock = record['stock'][name]
            if 'file' in stock:
                data = output_file(stock['file'], name+'.npy'); a = npy(data, name+'.npy')
                c.same({key:stock[key] for key in ('shape','dtype','bytes_sha256')},
                       {'shape':list(a.shape),'dtype':str(a.dtype),'bytes_sha256':sha(a.tobytes())}, 'actual_stock_metadata', 0)
                if name == 'native':
                    z = (a+np.float32(1))/np.float32(2); clipped = np.clip(z,0,1)
                    def resize(x): return ((x[:,:,0::2,0::2]+x[:,:,0::2,1::2]+x[:,:,1::2,0::2]+x[:,:,1::2,1::2])*np.float32(.25))
                    c.check(float(np.abs(resize(clipped)[0].transpose(1,2,0)-raw).max()) <= np.finfo(np.float32).eps,
                            'native_clamp_before_resize_FP32_readback', slot)
                    c.check(float(np.abs(resize(z)-un).max()) <= np.finfo(np.float32).eps,
                            'unclipped_resize_FP32_readback', slot)
            if call['phase'] == 'deterministic_same_u64_repeat':
                prior = initial_records[call['id']]['stock'][name]
                c.same({k:v for k,v in stock.items() if k != 'file'}, {k:v for k,v in prior.items() if k != 'file'}, 'repeat_stock_'+name, 0)
            if name in ('logits','LQ') and call['id'] in initial_records:
                prior=initial_records[call['id']]['stock'][name]
                c.same({k:v for k,v in stock.items() if k != 'file'}, {k:v for k,v in prior.items() if k != 'file'}, 'frozen_upstream_'+name, 0)
        if call['backward']:
            objective_name = stem+'/objective.json'
            if objective_name in index:
                saved = strict_json(read(objective_name, 128*1024))
                need(spec['cohort'] == 'optimize' and spec['optimization_permitted'] is True
                     and spec['split'] == 'TRAIN' and target is not None, 'Paired TRAIN objective only')
                for key, value in [('id',call['id']),('role',spec['role']),('call_slot',slot),('new_update',call['new_update']),('photographic_target',spec['target'])]:
                    c.same(value, saved[key], 'objective_'+key, 0)
                c.same(saved['output_record'], {'path':record_name, **binding(record_bytes)}, 'objective_record_join', 0)
                c.same(saved['LR'], .0001*min(1.,call['new_update']/4), 'objective_LR', 0)
                v64,t64 = objective(raw,target,spec['role'],np.float64);v32,t32 = objective(raw,target,spec['role'],np.float32)
                c.same(v64,saved['objective'],'objective_arithmetic_bound_only',OBJECTIVE_READBACK_BOUND)
                c.same(t64,saved['terms'],'objective_terms_arithmetic_bound_only',OBJECTIVE_READBACK_BOUND)
                c.same(float(np.float32(saved['objective'])/np.float32(4)),saved['divided_objective'],'objective_divided_four',0)
                objectives.append(dict(slot=slot, update=call['new_update'], id=call['id'], role=spec['role'],
                    reported_Torch_FP32=saved['objective'], CPU_float64=v64, CPU_float32=v32,
                    CPU_float64_terms=t64, CPU_float32_terms=t32, reported_terms=saved['terms'],
                    absolute_float64_difference=abs(v64-saved['objective']), arithmetic_readback_bound=OBJECTIVE_READBACK_BOUND,
                    CPU_float32_is_not_CUDA_reduction_replay=True, quality_tolerance_added=False))
            else:
                missing_objectives.append(slot)
        records.append({'record_path':record_name,'call':call,**measured});available_slots.append(slot)
    gate_rows = []
    for update in (0,4,8,16):
        name = RUN+'/banks/u'+str(update).zfill(4)+'/bank.json'
        if len(banks[update]) == 56 and len(banks[0]) == 56:
            recomputed = gates(banks[update],banks[0],frozen,parent,update)
            if name in index:
                stored = strict_json(read(name,8*1024**2))
                c.check(stored['complete'] is True,'saved_bank_complete',update)
                c.check(set(stored['records']) == set(banks[update]),'saved_bank_case_membership',update)
                for cid, measured in banks[update].items():
                    c.same(measured,{k:stored['records'][cid].get(k) for k in measured},'bank_saved_metrics')
                compare_gate(recomputed,stored['gate'],c)
            else: c.check(False,'complete56_outputs_missing_bank_receipt',update)
            gate_rows.append(recomputed)
        else:
            c.check(name not in index,'incomplete_bank_not_reported_complete',update)
    coverage=[]
    for sheet in visual['sheets']:
        slots=sheet['generated_call_slots']
        coverage.append({'sheet':sheet['id'],'planned_slots':slots,'available_slots':[s for s in slots if s in available_slots],
                         'missing_slots':[s for s in slots if s not in available_slots]})
    visual_slots=[s for sheet in visual['sheets'] for s in sheet['generated_call_slots']]
    c.check(len(visual['sheets'])==31 and len(visual_slots)==292 and set(visual_slots)==set(range(1,293)), 'exact31_sheet292_slot_plan')
    actual_record_names={name for name in index if name.startswith(RUN+'/calls/') and name.endswith('/record.json')}
    c.check(actual_record_names=={r['record_path'] for r in records},'every_available_call_record_audited')
    pixel_names={name for name in index if name.startswith(RUN+'/calls/') and PurePosixPath(name).name in
                 ('raw.npy','unclipped.npy','native.npy','logits.npy','LQ.npy','result.png')}
    c.check(pixel_names <= set(consumed),'every_available_complete_or_partial_pixel_file_decoded')
    need(read_regular(inventory_path.parent,inventory_path.name,16*1024**2)==inventory_bytes,'Inventory changed during readback')
    report={'complete':True,'audit_type':'INDEPENDENT_SAVED_U64_CONTINUATION_PIXEL_OBJECTIVE_GATE_READBACK',
        'UTC':datetime.now(timezone.utc).isoformat(),'inventory':{'path':inventory_path.relative_to(ROOT).as_posix(),**binding(inventory_bytes)},
        'archive':inventory['archive'],'protocol':inventory['protocol'],'checks':sum(c.checks.values()),
        'saved_pixel_numeric_integrity_pass':not c.failures,'integrity_failures':c.failures,
        'counts':{'available_saved_calls':len(records),'training_objectives':len(objectives),
                  'historical_parity_records':len(parity_rows),'deterministic_repeats':len(repeats)},
        'available_call_slots':available_slots,'missing_call_slots':missing_slots,'partial_output_slots_without_record':incomplete_slots,
        'partial_outputs_decoded_without_inferred_record_or_bank':partial_pixels,
        'saved_training_output_without_objective_receipt':missing_objectives,
        'missing_calls_not_inferred_or_replayed':True,'snapshot_bank_cases':{str(u):len(b) for u,b in banks.items()},
        'complete_bank_gates':gate_rows,'historical_u64_parity':parity_rows,'same_candidate_repeat_slots':repeats,
        'metric_scalar_arithmetic_readback_bound':METRIC_READBACK_BOUND,'objective_scalar_arithmetic_readback_bound':OBJECTIVE_READBACK_BOUND,
        'quality_threshold_epsilon':0,'CPU_objective_arithmetic_not_connected_CUDA_replay':True,
        'consumed_runtime_files':consumed,'all_available_outputs_audited':True,'planned_visual_coverage':coverage,
        'actual_visual_sheets_reviewed':0,'pilot_complete':len(records)==292 and len(gate_rows)==4,
        'terminal_u16_quality_gate_pass':next((g['pass'] for g in gate_rows if g['update']==16),None),
        'inherited_broader_failures_cleared':False,'model_qualified':False,'Torch_neural_gradient_optimizer_training_VM_calls':0,
        'archive_streamed_or_sources_executed':False}
    output.mkdir(parents=True)
    def write(name,value):
        dest=output/name
        with dest.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        return {'path':dest.relative_to(ROOT).as_posix(),**binding(dest.read_bytes())}
    report['artifacts']={'metrics':write('recomputed_saved_pixel_metrics_v1.json',{'complete':True,'records':records}),
                         'objectives':write('recomputed_photographic_objectives_v1.json',{'complete':True,'rows':objectives}),
                         'differences':write('scalar_arithmetic_differences_v1.json',{'complete':True,'rows':c.differences})}
    result=write('saved_pixel_audit_v1.json',report)
    print(json.dumps({'audit':result,'saved_pixel_numeric_integrity_pass':not c.failures,'available_calls':len(records)}),flush=True)
    return 0 if not c.failures else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--stage',type=Path,default=ROOT/'outputs'/STAGE)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/codeformer_u64_cleananchors_continuation_v1_return_pixels')
    args=parser.parse_args()
    return audit(args.inventory,args.stage,args.output)


if __name__ == '__main__':
    raise SystemExit(main())
