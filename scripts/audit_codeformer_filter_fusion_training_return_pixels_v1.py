"""Independent saved-pixel audit of filter-fusion V1; no model execution.

Only a closed, hash-bound import is read. RAW float32 arrays use allow_pickle=False.
All478 comparison slots and24 repeated TRAIN guards are audited. Native CCTV stays
unpaired; mixed-role filter probes stay TRAIN/DEV separated. Exact256 contact-sheet
panels have no display enhancement or resampling. This does not qualify a model.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version
from skimage.metrics import structural_similarity
import skimage

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = '3d6feaf30d4477fe84bd51c57d560cb10d7b4cd6509db627fc61e03accb35d16'
PROTOCOL_SHA = 'c1f7d8386856bd95ab3af3a8623f33fcecddd4b7e08f0696ea7694a0542c1949'
RUN = 'outputs/filter_fusion_training_v1'
STAGE = ROOT / 'outputs/cctv_codeformer_filter_fusion_training_v1_vm'
BANKS = ('original', 'parent', 'r2', 'snapshot0', 'snapshot50')
VIEWS = ('raw', 'native_png')
FIELDS = ('mae', 'mse', 'psnr', 'ssim', 'luminance_bias', 'signed_luminance_bias',
          'coarse_rgb_mae', 'adjacent_rgb_error')
LUMA = np.asarray([.2126, .7152, .0722], np.float64)
DEADLINE_SECONDS = 1200


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def checked(root, name):
    require(isinstance(name, str), 'String evidence path required')
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.as_posix() == name and
            all(p not in ('', '.', '..') for p in relative.parts) and '\\' not in name and ':' not in name,
            'Canonical relative evidence path required')
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*relative.parts)
    require(path.resolve(strict=True).is_relative_to(root), 'Evidence escapes import')
    for item in (path, *path.parents):
        if item == root.parent:
            break
        info = item.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                'Linked/reparse evidence forbidden')
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Single-link regular evidence required')
    return path


def pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def floats(array):
    return np.ascontiguousarray(array.astype(np.float32) / np.float32(255))


def rgb8(path):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (256, 256) and image.format == 'PNG',
                'Exact RGB256 PNG required')
        image.load()
        return np.asarray(image).copy()


def support8(path):
    with Image.open(path) as image:
        require(image.mode == 'L' and image.size == (256, 256), 'Exact L256 support required')
        image.load()
        array = np.asarray(image).copy()
    require(np.isin(array, (0, 255)).all() and array.any(), 'Binary nonempty support required')
    return array


def raw256(path):
    require(Path(path).stat().st_size <= 256 * 256 * 3 * 4 + 256, 'Bounded RGB256 NPY required')
    with Path(path).open('rb') as stream:
        array = np.load(stream, allow_pickle=False)
        require(not stream.read(1), 'Trailing NPY data forbidden')
    require(array.shape == (256, 256, 3) and array.dtype == np.float32 and array.flags.c_contiguous and
            np.isfinite(array).all() and 0 <= array.min() <= array.max() <= 1,
            'Finite contiguous bounded float32 RGB256 RAW required')
    return array


def independent_metrics(raw, png, truth, support):
    if truth is None:
        return {view: dict.fromkeys(FIELDS) for view in VIEWS}
    answer = {}
    target = truth.astype(np.float64)
    for view, image in (('raw', raw), ('native_png', floats(png))):
        image = image.astype(np.float64)
        residual = image - target
        errors = residual[support]
        mse = float(np.square(errors).mean())
        signed = float((errors @ LUMA).mean())
        row = dict(mae=float(np.abs(errors).mean()), mse=mse,
            psnr=None if mse == 0 else -10 * math.log10(mse), ssim=None,
            luminance_bias=abs(signed), signed_luminance_bias=signed,
            coarse_rgb_mae=None, adjacent_rgb_error=None)
        if support.all():
            predicted_blocks = image.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3), dtype=np.float64)
            target_blocks = target.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3), dtype=np.float64)
            row['coarse_rgb_mae'] = float(np.abs(predicted_blocks - target_blocks).mean())
            row['ssim'] = float(structural_similarity(image, target, data_range=1., channel_axis=2))
            row['adjacent_rgb_error'] = float(.5 * (np.abs(residual[:, 1:] - residual[:, :-1]).mean() +
                np.abs(residual[1:] - residual[:-1]).mean()))
        require(all(value is None or math.isfinite(value) for value in row.values()), 'Finite metric required')
        answer[view] = row
    return answer


def compare(actual, expected, prefix=''):
    problems = []
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            problems.append(prefix + ':keys')
        for key in set(actual) & set(expected):
            problems.extend(compare(actual[key], expected[key], prefix + '/' + str(key)))
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            problems.append(prefix + ':length')
        for index, (left, right) in enumerate(zip(actual, expected)):
            problems.extend(compare(left, right, prefix + '/' + str(index)))
    elif type(actual) in (int, float) and type(expected) in (int, float):
        if abs(float(actual) - float(expected)) > 1e-10:
            problems.append(prefix + ':numeric')
    elif actual != expected:
        problems.append(prefix + ':value')
    return problems


class Evidence:
    def __init__(self, receipt):
        self.receipt_path = Path(receipt).resolve(strict=True)
        r = read(self.receipt_path)
        require(r.get('complete') is True and r['archive_sha256'] == ARCHIVE_SHA and
                r['protocol_sha256'] == PROTOCOL_SHA, 'Exact completed import required')
        self.root = Path(r['import_root']).resolve(strict=True)
        require(self.root.is_relative_to(ROOT), 'Imported evidence must stay in workspace')
        self.hashes, self.sizes = r['files_sha256'], r['files_bytes']
        require(set(self.hashes) == set(self.sizes), 'Import size/hash namespace differs')
        self.checked = {}
        self.start = time.monotonic()
        self.p = self.json('protocol.json')
        require(self.hashes['protocol.json'] == PROTOCOL_SHA and sha(STAGE / 'protocol.json') == PROTOCOL_SHA,
                'Original sealed protocol differs')
        manifest = self.json(RUN + '/export_manifest.json')
        require(manifest['protocol_sha256'] == PROTOCOL_SHA and
                set(manifest['files']) | {RUN + '/export_manifest.json'} == set(self.hashes),
                'Export manifest namespace differs from closed import')
        for name, value in manifest['files'].items():
            require(value == {'bytes': self.sizes[name], 'sha256': self.hashes[name]},
                    'Export/import file binding differs: ' + name)
        for name, digest in self.p['files_sha256'].items():
            require(self.hashes[name] == digest and self.sizes[name] == self.p['files_bytes'][name],
                    'Sealed static packet binding differs: ' + name)

    def deadline(self):
        require(time.monotonic() - self.start <= DEADLINE_SECONDS, 'Independent pixel audit1200s bound exceeded')

    def file(self, name, pin=None):
        self.deadline()
        require(name in self.hashes, 'Unbound evidence file: ' + name)
        if pin is not None:
            require(pin == {'path': name, 'bytes': self.sizes[name], 'sha256': self.hashes[name]},
                    'Caller pin differs from manifest: ' + name)
        path = checked(self.root, name)
        if name not in self.checked:
            before = path.stat()
            require(before.st_size == self.sizes[name] and sha(path) == self.hashes[name], 'Evidence bytes differ: ' + name)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                    (after.st_size, after.st_mtime_ns, after.st_ino), 'Evidence changed during hash')
            self.checked[name] = {'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns, 'inode': before.st_ino}
        else:
            value = path.stat(); old = self.checked[name]
            require((value.st_size, value.st_mtime_ns, value.st_ino) ==
                    (old['bytes'], old['mtime_ns'], old['inode']), 'Evidence changed after hash')
        return path

    def json(self, name):
        return read(self.file(name))


def spec_from_row(e, row):
    inp = rgb8(e.file(row['input']['path'], row['input']))
    ip = row.get('input_pixels', {})
    require(pixel_sha(inp) == ip.get('RGB8_sha256', row.get('input_RGB8_sha256')), 'Input RGB8 pixels differ')
    if ip.get('float32_sha256'):
        require(pixel_sha(floats(inp)) == ip['float32_sha256'], 'Input float32 pixels differ')
    pin = row.get('target256', row.get('target'))
    target8 = rgb8(e.file(pin['path'], pin)) if pin is not None else None
    if target8 is not None:
        tp = row.get('target256_pixels', row.get('target_pixels', {}))
        require(pixel_sha(target8) == tp.get('RGB8_sha256', row.get('target_RGB8_sha256')), 'Target RGB8 pixels differ')
        if tp.get('float32_sha256'):
            require(pixel_sha(floats(target8)) == tp['float32_sha256'], 'Target float32 pixels differ')
    sp = row.get('support')
    observed8 = support8(e.file(sp['path'], sp)) if sp else np.full((256, 256), 255, np.uint8)
    if row.get('support_pixels') is not None:
        require(pixel_sha(observed8) == row['support_pixels']['RGB8_or_L8_sha256'] and
                int((observed8 > 0).sum()) == row['support_pixels']['observed_pixels'], 'Support pixel binding differs')
    return {'row': row, 'input': inp, 'target8': target8,
            'target': floats(target8) if target8 is not None else None, 'support': observed8 > 0}


def load_specs(e):
    bundle = e.json('evaluation_bundle.json')
    rows = bundle['cases']
    require(bundle['complete'] is True and len(rows) == len({r['id'] for r in rows}) == 478,
            'Exact478 comparison slots required')
    require(Counter(r['cohort'] for r in rows) == {'historical_DEV406': 406, 'retained_filter72_mixed_roles': 72},
            'Historical/mixed-role census differs')
    old = [r for r in rows if r['cohort'] == 'historical_DEV406']
    new = [r for r in rows if r['cohort'] == 'retained_filter72_mixed_roles']
    require(Counter(r['evidence'] for r in old) == {'paired_synthetic': 358, 'native_unpaired': 48} and
            all(r['split'] == 'DEV' for r in old) and Counter(r['split'] for r in new) == {'TRAIN': 36, 'DEV': 36},
            'Paired/native/split census differs')
    require(all(r['reserved_FINAL'] is False for r in rows), 'Reserved FINAL evidence used')
    specs = {r['id']: spec_from_row(e, r) for r in rows}
    require(all((s['target'] is None) == (s['row']['evidence'] == 'native_unpaired') for s in specs.values()),
            'Native clean-reference policy differs')
    refused = bundle['refused_controls']
    require(len(refused) == len({r['id'] for r in refused}) == 24 and
            Counter(r['split'] for r in refused) == {'TRAIN': 12, 'DEV': 12}, 'Exact24 refusal scope required')
    for row in refused:
        require(row['restored_output'] is None and row['new_model_forward_permitted'] is False and
                row['input_review']['decision'] == 'insufficient', 'Input-only refusal policy differs')
        spec_from_row(e, row)
        require(not any('/' + row['id'] + suffix in name for name in e.hashes for suffix in ('.png', '.npy', '.json')),
                'Refused control acquired a generated output')
    return specs, refused


def bank_prefix(bank, rid):
    return RUN + (('/baseline/' + bank + '/') if bank in ('original', 'parent', 'r2') else
                  ('/snapshot_%04d/' % (0 if bank == 'snapshot0' else 50))) + rid


def audited_output(e, prefix, spec):
    document = e.json(prefix + '.json')
    row = spec['row']
    require(document['retained_case_metadata'] == row and document['id'] == row['id'] and
            document['input'] == row['input'] and document['target'] == row.get('target256', row.get('target')) and
            document['support'] == row.get('support'), 'Saved output full source/target metadata differs')
    for key in ('source', 'profile', 'split', 'cohort'):
        require(document[key] == row.get(key), 'Output field differs: ' + key)
    expected_evidence = row.get('evidence', 'paired_synthetic_TRAIN')
    if spec['target'] is None and row.get('profile') == 'linear_srgb_exposure0p35':
        expected_evidence = 'TRAIN_brightness_behavior_retention_no_clean_scores'
    require(document['evidence'] == expected_evidence, 'Native/TRAIN/paired evidence label differs')
    rawpin, pngpin = document['outputs']['raw'], document['outputs']['png']
    require(rawpin['path'] == prefix + '.npy' and pngpin['path'] == prefix + '.png', 'Output path join differs')
    raw = raw256(e.file(rawpin['path'], rawpin)); png = rgb8(e.file(pngpin['path'], pngpin))
    support = spec['support']
    wanted = np.floor(raw * np.float32(255)).astype(np.uint8)
    wanted[~support] = spec['input'][~support]
    require(np.array_equal(wanted, png) and np.array_equal(png[~support], spec['input'][~support]) and
            np.array_equal(raw[~support], floats(spec['input'])[~support]),
            'RAW-to-PNG conversion or protected observed-canvas pixels differ')
    measured = independent_metrics(raw, png, spec['target'], support)
    mismatches = compare(measured, document['metrics'], prefix + '/metrics')
    entry = {k: document[k] for k in ('id', 'source', 'profile', 'split', 'cohort', 'evidence')}
    entry['metrics'] = measured
    desc = {}
    for view, image in (('raw', raw), ('native_png', floats(png))):
        values = image.astype(np.float64)
        y = values @ LUMA
        desc[view] = {'observed_mean_luminance': float(y[support].mean()),
            'observed_luminance_std': float(y[support].std()),
            'input_relative_MAE_not_quality': float(np.abs(values[support] - floats(spec['input'])[support]).mean())}
    entry['unpaired_descriptors_not_quality_or_identity'] = desc
    check = {'prefix': prefix, 'RAW_PNG_floor_exact': True, 'protected_RAW_PNG_exact': True,
             'RAW_pixel_sha256': pixel_sha(raw), 'PNG_RGB8_sha256': pixel_sha(png)}
    return entry, raw, png, document, check, mismatches


def preservation(rows, baseline):
    report = {'pass': True, 'quality_gate_thresholds_changed': False, 'groups': {}}
    for label, ids, fields in (
        ('clear', sorted(k for k, r in rows.items() if r['evidence'] == 'paired_synthetic' and r['profile'] == 'clear'), ('mae',)),
        ('normal_blur', sorted(k for k, r in rows.items() if r['source'] == 'FFHQ validation' and r['profile'] == 'blur'),
         ('luminance_bias', 'coarse_rgb_mae'))):
        require(len(ids) == (61 if label == 'clear' else 53), 'Original61clear/53blur membership required')
        group = {'cases': ids, 'views': {}, 'pass': True}
        for view in VIEWS:
            group['views'][view] = {}
            for field in fields:
                deltas = {k: rows[k]['metrics'][view][field] - baseline[k]['metrics'][view][field] for k in ids}
                mean, worst = sum(deltas.values()) / len(ids), max(deltas.values())
                passed = mean <= .01 and worst <= .03
                group['views'][view][field] = {'deltas': deltas, 'mean_delta': mean, 'worst_delta': worst,
                    'mean_limit': .01, 'worst_limit': .03, 'cases_over_0_03': sum(v > .03 for v in deltas.values()), 'pass': passed}
                group['pass'] &= passed
        report['groups'][label] = group
        report['pass'] &= group['pass']
    return report


def dark_retention(rows, baseline):
    report = {'pass': True, 'profiles': {}}
    for profile in ('lowlight', 'compound'):
        ids = sorted(k for k, r in rows.items() if r['source'] == 'FFHQ validation' and r['profile'] == profile)
        require(len(ids) == 53, 'Exact53 lowlight/compound cases required')
        means = {label: {field: sum(bank[k]['metrics']['native_png'][field] for k in ids) / 53
                 for field in ('mae', 'ssim')} for label, bank in (('candidate', rows), ('control', baseline))}
        passed = means['candidate']['mae'] <= means['control']['mae'] and means['candidate']['ssim'] >= means['control']['ssim']
        report['profiles'][profile] = {**means, 'cases': ids, 'pass': passed}
        report['pass'] &= passed
    return report


def interim(now, original, initial):
    current, start = preservation(now, original), preservation(initial, original)
    result = {'pass': True, 'qualification': current, 'additional_stop_only': True,
              'arithmetic': 'Python float64; exact <=; no added tolerance', 'groups': {}}
    for label, group in current['groups'].items():
        result['groups'][label] = {}
        for view, fields in group['views'].items():
            result['groups'][label][view] = {}
            for field, row in fields.items():
                before = start['groups'][label]['views'][view][field]
                limits = {k: max(.03, before['deltas'][k]) for k in row['deltas']}
                failed = [k for k in row['deltas'] if row['deltas'][k] > limits[k]]
                mean_limit = max(.01, before['mean_delta'])
                passed = not failed and row['mean_delta'] <= mean_limit
                result['groups'][label][view][field] = {'pass': passed, 'case_limits': limits,
                    'failed_case_ids': failed, 'mean_limit': mean_limit, 'mean_delta': row['mean_delta']}
                result['pass'] &= passed
    return result


def filter_summary(rows):
    filtered = [r for r in rows.values() if r['cohort'] == 'retained_filter72_mixed_roles']
    require(Counter(r['split'] for r in filtered) == {'TRAIN': 36, 'DEV': 36}, 'Filter roles differ')
    answer = {'comparisons': 72, 'role_counts': {'TRAIN': 36, 'DEV': 36}, 'groups': {},
              'acceptance_established': False, 'independent_identity_count': None}
    for split in ('TRAIN', 'DEV'):
        for profile in sorted({r['profile'] for r in filtered}):
            cases = [r for r in filtered if r['split'] == split and r['profile'] == profile]
            if cases:
                values = {view: {field: sum(r['metrics'][view][field] for r in cases) / len(cases)
                    for field in ('mae', 'ssim', 'adjacent_rgb_error')} for view in VIEWS}
                answer['groups'][split + '/' + profile] = {'cases': [r['id'] for r in cases], 'metrics': values}
    return answer


def guard_interim(current, initial):
    require(set(current) == set(initial) and len(current) == 24, 'Exact24 TRAIN guards required')
    require(Counter(r['role'] for r in current.values()) == {'clear': 8, 'dark': 8, 'filter': 8}, 'TRAIN guard roles differ')
    result = {'pass': True, 'additional_TRAIN_stop_only': True, 'model_qualified': False, 'groups': {},
              'arithmetic': 'Python float64; exact <=; no added tolerance'}
    for role in ('clear', 'dark', 'filter'):
        ids = sorted(k for k, r in current.items() if r['role'] == role)
        group = {'pass': True, 'cases': ids, 'views': {}}
        for view in VIEWS:
            group['views'][view] = {}
            for field in (('mae', 'adjacent_rgb_error') if role == 'filter' else ('behavior_distance',)):
                def value(rows, key):
                    row = rows[key]
                    require(row['split'] == 'TRAIN', 'Repeated guard cannot become holdout')
                    return row['behavior_distance'][view] if field == 'behavior_distance' else row['metrics'][view][field]
                before = {k: value(initial, k) for k in ids}; now = {k: value(current, k) for k in ids}
                if role == 'filter':
                    deltas = {k: now[k] - before[k] for k in ids}; limits = dict.fromkeys(ids, .03)
                    mean, mean_limit = sum(deltas.values()) / len(ids), .01
                else:
                    deltas = now; limits = {k: max(.03, before[k]) for k in ids}
                    mean, mean_limit = sum(now.values()) / len(ids), max(.01, sum(before.values()) / len(ids))
                bad = [k for k in ids if deltas[k] > limits[k]]
                passed = not bad and mean <= mean_limit
                group['views'][view][field] = {'pass': passed, 'current': now, 'initial': before, 'case_limits': limits,
                    'failed_case_ids': bad, 'mean_measurement': mean, 'mean_limit': mean_limit,
                    'measurement': 'increase from initial' if role == 'filter' else 'distance from frozen teacher'}
                group['pass'] &= passed
        result['groups'][role] = group; result['pass'] &= group['pass']
    return result


def summarize(rows):
    summary, deltas = [], []
    keys = ('source', 'evidence', 'profile', 'cohort', 'split')
    for bank, values in rows.items():
        groups = defaultdict(list)
        for rid, row in values.items():
            groups[tuple(row[k] for k in keys)].append(rid)
        for group, ids in sorted(groups.items()):
            for view in VIEWS:
                result = {'bank': bank, **dict(zip(keys, group)), 'view': view, 'cases': len(ids)}
                for field in FIELDS:
                    vals = [values[k]['metrics'][view][field] for k in ids if values[k]['metrics'][view][field] is not None]
                    result['mean_' + field] = sum(vals) / len(vals) if vals else None
                summary.append(result)
                if bank == 'snapshot50':
                    for baseline in ('original', 'parent', 'r2'):
                        delta = {'comparison': 'snapshot50_minus_' + baseline, **dict(zip(keys, group)),
                                 'view': view, 'cases': len(ids)}
                        for field in FIELDS:
                            vals = [values[k]['metrics'][view][field] - rows[baseline][k]['metrics'][view][field]
                                for k in ids if values[k]['metrics'][view][field] is not None]
                            delta['mean_delta_' + field] = sum(vals) / len(vals) if vals else None
                            delta[field + '_lower'] = sum(v < 0 for v in vals)
                            delta[field + '_higher'] = sum(v > 0 for v in vals)
                            delta[field + '_equal'] = sum(v == 0 for v in vals)
                        deltas.append(delta)
    return summary, deltas


def render(out, e, specs, refused, pngs, gates):
    folder = out / 'visual_sheets'; folder.mkdir()
    failed = set()
    for group in gates['preservation']['groups'].values():
        for fields in group['views'].values():
            for row in fields.values():
                failed.update(k for k, v in row['deltas'].items() if v > .03)
    for group in gates['interim']['groups'].values():
        for fields in group.values():
            for row in fields.values():
                failed.update(row['failed_case_ids'])
    groups = []
    for split in ('TRAIN', 'DEV'):
        groups.append(('FILTER_' + split, sorted(k for k,s in specs.items() if
            s['row']['cohort'] == 'retained_filter72_mixed_roles' and s['row']['split'] == split)))
    for source in sorted({s['row']['source'] for s in specs.values() if s['row']['evidence'] == 'native_unpaired'}):
        groups.append(('NATIVE_' + source, sorted(k for k,s in specs.items() if
            s['row']['evidence'] == 'native_unpaired' and s['row']['source'] == source)))
    groups.append(('FAILED_CLEAR_BLUR', sorted(failed)))
    seen = {k for _, ids in groups for k in ids}
    remaining = sorted(set(specs) - seen, key=lambda k: (specs[k]['row']['source'], specs[k]['row']['profile'], k))
    groups.append(('REMAINING_DEV', remaining))
    allids = [k for _,ids in groups for k in ids]
    require(len(allids) == len(set(allids)) == 478 and set(allids) == set(specs), 'Every478 visual case exactly once required')
    ordered = [(group, rid) for group, ids in groups for rid in ids]
    sheets, cells = [], []
    for start in range(0, len(ordered), 8):
            page = ordered[start:start+8]
            group = ' / '.join(dict.fromkeys(g for g, _ in page))
            e.deadline()
            subset = [rid for _, rid in page]
            name = 'sheet_%02d.png' % (len(sheets) + 1)
            width, height = 1280, 64 + 284 * len(subset)
            canvas = Image.new('RGB', (width, height), (250,248,243)); draw = ImageDraw.Draw(canvas)
            draw.text((4, 4), 'Filter-fusion V1 u50 | ' + group + ' | exact256 panels, no display processing | not qualified', fill='black')
            for col, title in enumerate(('Observed input', 'Paired target / native blank', 'Untouched CodeFormer', 'Retained trained R2', 'New filter-fusion update50')):
                draw.text((col*256+4, 30), title, fill='black')
            for index, rid in enumerate(subset):
                spec = specs[rid]; row = spec['row']; y = 64 + index*284
                draw.text((4,y), rid + ' | ' + row['source'] + ' | ' + row['profile'] + ' | ' + row['split'], fill='black')
                for col,array in enumerate((spec['input'], spec['target8'], pngs['original',rid], pngs['r2',rid], pngs['snapshot50',rid])):
                    x, top = col*256, y+24
                    if array is None:
                        draw.rectangle((x,top,x+255,top+255), fill=(225,224,218))
                        draw.text((x+8,top+100), 'Native CCTV: UNPAIRED\nNo aligned clean target', fill='black')
                    else:
                        canvas.paste(Image.fromarray(array), (x,top))
                        cells.append({'sheet':name,'id':rid,'column':col,'x':x,'y':top,
                            'width':256,'height':256,'RGB8_sha256':pixel_sha(array)})
            path = folder / name; canvas.save(path)
            with Image.open(path) as image:
                saved = np.asarray(image)
                for cell in (c for c in cells if c['sheet'] == name):
                    require(pixel_sha(saved[cell['y']:cell['y']+256,cell['x']:cell['x']+256]) == cell['RGB8_sha256'],
                            'Saved exact contact-sheet pixels differ')
            sheets.append({'path':path.relative_to(out).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size,
                'group':group,'case_ids':subset,'case_count':len(subset),'width':width,'height':height})
    refusal_sheets = []
    for start in range(0,len(refused),8):
        subset = refused[start:start+8]
        name = 'refusals_%02d.png' % (len(refusal_sheets)+1)
        canvas = Image.new('RGB',(768,64+284*len(subset)),(250,248,243)); draw=ImageDraw.Draw(canvas)
        draw.text((4,4),'Input-only insufficient controls | no model forwards or generated result',fill='black')
        for index,row in enumerate(subset):
            y=64+index*284
            draw.text((4,y),row['id']+' | '+row['split'],fill='black')
            array=rgb8(e.file(row['input']['path'],row['input']))
            canvas.paste(Image.fromarray(array),(0,y+24))
            draw.text((268,y+60),'REQUEST CLEARER INPUT\n\n'+row['input_review']['reason'],fill='black')
            cells.append({'sheet':name,'id':row['id'],'column':0,'x':0,'y':y+24,'width':256,'height':256,'RGB8_sha256':pixel_sha(array)})
        path=folder/name; canvas.save(path)
        with Image.open(path) as image:
            saved=np.asarray(image)
            for cell in (c for c in cells if c['sheet']==name):
                require(pixel_sha(saved[cell['y']:cell['y']+256,cell['x']:cell['x']+256])==cell['RGB8_sha256'],'Refusal evidence panel differs')
        refusal_sheets.append({'path':path.relative_to(out).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size,
            'case_ids':[r['id'] for r in subset],'case_count':len(subset),'input_only':True,'width':768,'height':canvas.height})
    require(len(sheets) == 60 and len(refusal_sheets) == 3,
            'Complete60 comparison/3 input-only refusal visual sheets required')
    write(out/'sheet_cells.json',cells)
    visual={'complete':True,'sheets':sheets,'refusal_sheets':refusal_sheets,'comparison_slots':478,
        'native_unpaired_slots':48,'mixed_filter_roles':{'TRAIN':36,'DEV':36},'refused_input_only_controls':24,
        'all478_once':True,'comparison_sheets':60,'input_only_refusal_sheets':3,'total_sheets':63,
        'page_order':'filter TRAIN36, filter DEV36, native48 bysource, failed clear/blur, remaining DEV',
        'page_boundary_may_contain_explicitly_labeled_different_roles_no_metric_pooling':True,
        'root_must_view_every_sheet':True,'root_personal_visual_review_performed':False,
        'image_processing':'none; exact256 panels, no resampling, sharpening or enhancement','model_qualified':False}
    write(out/'visual_manifest.json',visual)
    with (folder/'INDEX.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('# Filter-fusion V1 exact-pixel comparison\n\nAll478 slots appear once. Five columns: input, paired target (native blank), untouched CodeFormer, R2, new update50. Filter TRAIN and DEV remain separate. Refusals contain inputs only. Every photograph panel is exact256 pixels with no display processing. Root review is pending; no model qualification follows from sheet creation.\n\n')
        stream.write('\n'.join('- ['+Path(r['path']).name+']('+Path(r['path']).name+')' for r in sheets+refusal_sheets)+'\n')
    return visual


def audit(receipt,out):
    require(not any(name in sys.modules for name in ('torch','tensorflow','jax')), 'Neural frameworks forbidden')
    out = Path(out).resolve(strict=False)
    require(out.is_relative_to((ROOT/'outputs').resolve()), 'Audit output must stay under workspace outputs')
    require(not out.is_symlink(), 'Linked audit destination forbidden')
    e=Evidence(receipt); started=time.monotonic()
    require(not out.exists(),'Exclusive new audit output required; retain prior/partial runs')
    out.mkdir(parents=True)
    specs,refused=load_specs(e)
    rows={bank:{} for bank in BANKS}; pngs={}; checks=[]; mismatches=[]; parity=[]
    for index,(rid,spec) in enumerate(specs.items(),1):
        rawbanks={}; docs={}
        for bank in BANKS:
            entry,raw,png,doc,check,bad=audited_output(e,bank_prefix(bank,rid),spec)
            rows[bank][rid]=entry; rawbanks[bank]=raw; pngs[bank,rid]=png; docs[bank]=doc
            checks.append(check); mismatches+=bad
        require(np.array_equal(rawbanks['snapshot0'],rawbanks['r2']) and
                np.array_equal(pngs['snapshot0',rid],pngs['r2',rid]), 'Snapshot0/R2 exact parity failed: '+rid)
        for key in ('raw', 'png'):
            initial_pin, r2_pin = docs['snapshot0']['outputs'][key], docs['r2']['outputs'][key]
            require(all(initial_pin[field] == r2_pin[field] for field in ('bytes', 'sha256')),
                    'Initial/R2 full physical RAW/PNG file parity differs: ' + rid)
        require(docs['snapshot0']['upstream']==docs['r2']['upstream']==docs['snapshot50']['upstream'],
                'Frozen LQ/logits/token readout hashes changed: '+rid)
        parity.append(rid)
        if index%40==0 or index==478:
            print(json.dumps({'independent_comparison_slots':index,'of':478,'banks':5,'seconds':time.monotonic()-started}),flush=True)
    old={bank:{k:r for k,r in bankrows.items() if r['cohort']=='historical_DEV406'} for bank,bankrows in rows.items()}
    gates={}
    gate_mismatches=[]
    for update,bank in ((0,'snapshot0'),(50,'snapshot50')):
        value={'preservation':preservation(old[bank],old['original']),
            'parent_dark_retention':dark_retention(old[bank],old['parent']),
            'R2_dark_retention':dark_retention(old[bank],old['r2']),
            'interim':interim(old[bank],old['original'],old['snapshot0']),
            'filters':filter_summary(rows[bank])}
        if update==0:
            value['initial_R2_full478_parity']={'pass':True,'cases':478,'RAW_and_PNG_files_exact':True,
                'upstream_readouts_also_checked_for_every_case':True,'current_runtime_only':True,'numerical_tolerance':0.}
        gates[str(update)]=value
        gate_mismatches+=compare(value,e.json(RUN+'/snapshot_%04d/gates.json'%update),'snapshot%d'%update)
    alltrain=e.json('data_bundle.json')['case_by_id']; guardbanks={}; guardmismatches=[]
    teacher_manifest=e.json(RUN+'/teacher/manifest.json')
    require(teacher_manifest['complete'] is True and len(teacher_manifest['targets'])==110,
            'All110 saved TRAIN role teacher bindings required')
    teacher_map={(r['role'],r['id']):r for r in teacher_manifest['targets']}
    require(len(teacher_map)==110, 'Duplicate role teacher binding')
    for update in (0,20):
        bank={}; saved=e.json(RUN+'/guard_%04d/bank.json'%update)
        require(saved['complete'] is True and len(saved['cases'])==24,'Guard bank incomplete')
        savedmap={r['id']:r for r in saved['cases']}
        require(set(savedmap)=={r['id'] for r in e.p['guard_cases']},'Guard ID scope differs')
        for guard in e.p['guard_cases']:
            rid,role=guard['id'],guard['role']; spec=spec_from_row(e,alltrain[rid])
            entry,raw,png,doc,check,bad=audited_output(e,RUN+'/guard_%04d/'%update+role+'/'+rid,spec)
            checks.append(check); mismatches+=bad
            entry['role']=entry['guard_role']=role
            if role in ('clear','dark'):
                teacher=savedmap[rid]['behavior_teacher']; rawpin,pngpin=teacher['RAW'],teacher['PNG']
                stem=RUN+'/teacher/'+role+'/'+rid
                require(teacher['kind']==('original' if role=='clear' else 'r2') and
                        rawpin['path']==stem+'_RAW.npy' and pngpin['path']==stem+'_PNG.png' and
                        teacher['U']['path']==stem+'_U.npy' and
                        all(teacher[key]==teacher_map[role,rid][key] for key in ('RAW','PNG','U')),
                        'Behavior teacher role/ID/manifest association differs')
                teacher_raw=raw256(e.file(rawpin['path'],rawpin)); teacher_png=rgb8(e.file(pngpin['path'],pngpin))
                entry['behavior_distance']={'raw':float(np.abs(raw.astype(np.float64)-teacher_raw.astype(np.float64)).mean()),
                    'native_png':float(np.abs(floats(png).astype(np.float64)-floats(teacher_png).astype(np.float64)).mean())}
                guardmismatches+=compare(entry['behavior_distance'],savedmap[rid]['behavior_distance'],rid+'/teacher')
            guardmismatches+=compare(doc['metrics'],savedmap[rid]['metrics'],rid+'/saved_bank')
            bank[rid]=entry
        guardbanks[str(update)]=bank
        gate=guard_interim(bank,guardbanks['0'])
        gate_mismatches+=compare(gate,e.json(RUN+'/guard_%04d/gates.json'%update),'guard%d'%update)
    visual=render(out,e,specs,refused,pngs,gates['50'])
    summary,deltas=summarize(rows)
    write(out/'metrics.json',{'complete':True,'cases':rows,'groups':summary})
    write(out/'guards.json',{'complete':True,'snapshots':gates,'TRAIN_guards':guardbanks})
    write(out/'source_profile_comparisons.json',{'complete':True,'comparisons':deltas,
        'native_clean_metrics_null':True,'mixed_filter_TRAIN_DEV_separate':True,'FINAL_used':False})
    write(out/'conversion_and_parity_checks.json',{'complete':True,'checks':checks,'snapshot0_exact_R2_ids':parity,
        'metric_mismatches':mismatches,'guard_teacher_mismatches':guardmismatches,'gate_mismatches':gate_mismatches,
        'scalar_comparison_atol':1e-10,'array_equality_tolerance':0.})
    failure=e.json(RUN+'/failure.json')
    for name,oldinfo in e.checked.items():
        info=checked(e.root,name).stat()
        require((info.st_size,info.st_mtime_ns,info.st_ino)==(oldinfo['bytes'],oldinfo['mtime_ns'],oldinfo['inode']),
                'Read evidence changed during completed audit: '+name)
    require(not any(name in sys.modules for name in ('torch','tensorflow','jax')),'Neural framework imported')
    outputnames=('metrics.json','guards.json','source_profile_comparisons.json','conversion_and_parity_checks.json',
                 'sheet_cells.json','visual_manifest.json')
    result={'complete':True,'audit_pass':not(mismatches or guardmismatches or gate_mismatches),
        'audit_source_sha256':sha(__file__),'archive_sha256':ARCHIVE_SHA,'protocol_sha256':PROTOCOL_SHA,
        'import_receipt_sha256':sha(e.receipt_path),'returned_root':str(e.root),
        'comparison_slots':478,'historical_DEV':406,'paired_synthetic_historical_DEV':358,'native_unpaired':48,
        'mixed_filter_TRAIN':36,'mixed_filter_DEV':36,'refused_input_only_controls':24,
        'comparison_RAW_PNG_pairs_checked':2390,'TRAIN_guard_RAW_PNG_pairs_checked':48,
        'exact_initial_R2_parity_slots':478,'saved_metric_mismatches':mismatches,
        'saved_guard_teacher_mismatches':guardmismatches,'saved_gate_mismatches':gate_mismatches,
        'preservation_pass':gates['50']['preservation']['pass'],
        'parent_dark_retention_pass':gates['50']['parent_dark_retention']['pass'],
        'R2_dark_retention_pass':gates['50']['R2_dark_retention']['pass'],
        'additional_interim_stop_pass':gates['50']['interim']['pass'],
        'completed_updates_reported':failure['completed_training_updates'],'recorded_failure':failure['error'],
        'verified_read_bindings':{name:{'bytes':e.sizes[name],'sha256':e.hashes[name]} for name in sorted(e.checked)},
        'outputs':{name:{'bytes':(out/name).stat().st_size,'sha256':sha(out/name)} for name in outputnames},
        'visual_sheets':visual['sheets'],'refusal_sheets':visual['refusal_sheets'],
        'root_personal_visual_review_pending':True,'user_acceptance_established':False,
        'native_sources_not_ethnicity_or_Zamboanga_claim':True,'FINAL_used':False,
        'actual_local_neural_gradient_optimizer_VM_calls':0,'app_changed':False,'model_qualified':False,
        'runtime':{'Python':sys.version,'NumPy':np.__version__,'Pillow':pillow_version,'scikit_image':skimage.__version__},
        'seconds':time.monotonic()-started,'deadline_seconds':DEADLINE_SECONDS,
        'created_at_UTC':datetime.now(timezone.utc).isoformat()}
    write(out/'audit.json',result)
    lines=['# Filter-fusion V1 independent saved-pixel audit','','The actual VM run stopped after50 updates. Audit integrity and model acceptance are separate. All478 slots,5 banks and RAW/PNG views were recomputed from saved pixels; no returned Python or model was executed. Native48 observations have no clean target. Filter TRAIN36 and DEV36 remain separate.','','| Requirement | Pass |','|---|---|']
    for field in ('preservation_pass','parent_dark_retention_pass','R2_dark_retention_pass','additional_interim_stop_pass'):
        lines.append('|'+field+'|'+str(result[field])+'|')
    lines+=['','Root must inspect every exact256 visual sheet. User acceptance, independent final evaluation and app promotion remain unproven. Saved metric changes do not prove identity recovery or useful native CCTV restoration.','']
    with (out/'AUDIT.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(lines))
    require(result['audit_pass'],'Saved metrics, teacher distances or gates differ from independent recomputation')
    return {k:result[k] for k in ('complete','audit_pass','preservation_pass','parent_dark_retention_pass',
        'R2_dark_retention_pass','additional_interim_stop_pass','seconds')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-receipt',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.import_receipt,args.out)),flush=True)
