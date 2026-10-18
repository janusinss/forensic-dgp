"""Independent CPU saved-pixel/response audit; no model or head-function replay.

Only NumPy/Pillow arithmetic, bound JSON/NPY/PNG files and exact256 contact
sheets are used. Photographic source IDs are not proven person identities.
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

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_fixed_state_response_v1_preparation'
REVIEW = ROOT / 'outputs/codeformer_fixed_state_response_v1_return_review'
RUN = 'outputs/fixed_state_response_v1'
PROTOCOL = '3ea4f1fe9ce90f3d29d29d6ace108eae7a809e6665cb770811f36773b888c089'
ARCHIVE = 'e60258356e148ae426dcc8863794ad62377559f858a2b5dbf75c3e0eee0d50ad'
MANIFEST = '725fbda90d5b61ad828d059fa88487358aaf4abb210f3f0f02f1717d4e77a422'
W = np.asarray([.2126, .7152, .0722], np.float64)
MEAN_BOUND = 2e-7  # CPU float64 readback versus recorded CUDA FP32 reduction.
COUNTS = {'CodeFormer_new_original': 64, 'CodeFormer_new_full_coarse': 64,
          'controller_fitted_initial': 200, 'controller_fitted_final': 200,
          'controller_heldout_initial': 64, 'controller_heldout_final': 64,
          'gradient_queries': 0, 'backward_calls': 0, 'optimizer_constructions': 0,
          'optimizer_steps': 0, 'committed_training_updates': 0}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    def unique(items):
        out = {}
        for k, v in items:
            require(k not in out, 'Duplicate JSON key')
            out[k] = v
        return out
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def checked(root, name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and p.as_posix() == name and not p.is_absolute()
        and all(x not in ('', '.', '..') for x in p.parts) and '\\' not in name and ':' not in name,
        'Unsafe relative evidence path')
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*p.parts)
    require(path.resolve(strict=True).is_relative_to(root), 'Evidence escapes root')
    for item in (path, *path.parents):
        if item == root.parent:
            break
        s = item.lstat()
        require(not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024,
                'Linked/reparse evidence')
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Unlinked regular evidence required')
    return path


def rgb8(path):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (256, 256), 'Exact RGB256 required')
        return np.asarray(image).copy()


def raw256(path):
    with Path(path).open('rb') as stream:
        a = np.load(stream, allow_pickle=False)
        require(stream.read(1) == b'', 'Trailing NPY bytes')
    require(a.shape == (256, 256, 3) and a.dtype == np.float32 and a.flags.c_contiguous
        and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1, 'Bounded FP32 RAW256 required')
    return a


def floats(a):
    return np.ascontiguousarray(a.astype(np.float32) / np.float32(255))


def png_pixels(raw, input8, mask):
    value = np.floor(raw * np.float32(255)).astype(np.uint8)
    value[~mask] = input8[~mask]
    return value


def compose(base, delta, input8, mask):
    value = np.clip(np.add(base, np.float32(delta), dtype=np.float32), np.float32(0), np.float32(1))
    value[~mask] = floats(input8)[~mask]
    return value


def feature_vector(input8, mask):
    """Independent cell loops, float64 support means then one FP32 cast."""
    a = floats(input8).astype(np.float64)
    out = np.zeros((4, 8, 8), np.float64)
    for iy in range(8):
        for ix in range(8):
            support = mask[iy*32:(iy+1)*32, ix*32:(ix+1)*32]
            out[3, iy, ix] = int(support.sum()) / 1024
            if support.any():
                out[:3, iy, ix] = a[iy*32:(iy+1)*32, ix*32:(ix+1)*32][support].mean(axis=0)
    return np.ascontiguousarray(out.reshape(256).astype(np.float32))


def mean_y(value, mask):
    return float((value.astype(np.float64) @ W)[mask].mean(dtype=np.float64))


def exact_json(a, b):
    return json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        b, sort_keys=True, separators=(',', ':'), allow_nan=False)


class Evidence:
    def __init__(self, receipt):
        self.receipt_path = Path(receipt)
        r = self.receipt = read(receipt)
        require(r.get('complete') is True and r['archive_sha256'] == ARCHIVE
            and r['protocol_sha256'] == PROTOCOL, 'Exact complete root import required')
        self.root = Path(r['import_root']).resolve(strict=True)
        self.hashes, self.sizes = r['files_sha256'], r['files_bytes']
        require(set(self.hashes) == set(self.sizes), 'Import namespaces differ')
        self.checked, self.deps = set(), {}
        self.p = self.json('protocol.json')
        require(self.hashes['protocol.json'] == PROTOCOL, 'Imported protocol bytes differ')
        self.stage = read(PREP / 'stage_v1.json')
        require(self.stage['complete'] is True and self.stage['protocol_sha256'] == PROTOCOL
            and self.stage['dependencies'] == self.p['dependencies'], 'Stage resolver/protocol differs')
        mname = RUN + '/export_manifest.json'
        m = self.json(mname)
        require(self.hashes[mname] == MANIFEST and m['protocol_sha256'] == PROTOCOL
            and set(m['files_sha256']) | {mname} == set(self.hashes)
            and all(self.hashes[n] == h for n, h in m['files_sha256'].items())
            and all(self.sizes[n] == b for n, b in m['files_bytes'].items()), 'Export/import closure differs')
        for n, h in self.p['files_sha256'].items():
            require(self.hashes[n] == h and self.sizes[n] == self.p['files_bytes'][n], 'Static packet differs')

    def file(self, name):
        require(name in self.hashes, 'Unbound returned file: ' + name)
        p = checked(self.root, name)
        if name not in self.checked:
            require(p.stat().st_size == self.sizes[name] and sha(p) == self.hashes[name], 'Returned hash differs: ' + name)
            self.checked.add(name)
        return p

    def json(self, name):
        return read(self.file(name))

    def dep(self, alias, name):
        pin = self.p['dependencies'][alias][name]
        p = Path(self.stage['local_dependency_resolution'][alias][name])
        p = checked(ROOT, p.relative_to(ROOT).as_posix())
        key = alias + ':' + name
        if key not in self.deps:
            require(p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], 'Dependency hash differs: ' + key)
            self.deps[key] = dict(pin, local_path=str(p))
        return p

    def binding(self, value, expected):
        require(value == {'path': expected, 'bytes': self.sizes[expected], 'sha256': self.hashes[expected]},
                'Record artifact binding differs: ' + expected)


def load_specs(e):
    p = e.p
    fitted = e.json(p['fitted_index_file'])['cases']
    held = e.json(p['new_pairs_file'])['cases']
    parity = read(e.dep('adain', 'training_parity_v1.json'))['cases']
    refs = {r['id']: r for r in read(e.dep('adain', 'references.json'))['references']}
    exclusions = e.json(p['exclusions_file'])
    selection = e.json(p['selection_file'])['selected']
    require(len(fitted) == len(parity) == 200 and len(held) == 64, 'Exact264 input census required')
    require([(r['update'], r['slot']) for r in parity] == [(u, s) for u in range(1, 51) for s in range(1, 5)],
            'Fitted chronological census differs')
    fitted_ids, fitted_hashes = {r['reference_id'] for r in fitted}, {r['source_sha256'] for r in fitted}
    require(len(fitted_ids) == len(fitted_hashes) == 100, 'Exact100 fitted photographic references required')
    candidates = sorted((r for r in refs.values() if r['historical_role'] == 'train'
        and r['id'] not in fitted_ids | set(exclusions['reference_ids'])
        and r['source_sha256'] not in fitted_hashes | set(exclusions['source_sha256'])),
        key=lambda r: (r['source_sha256'], r['id']))[:32]
    require([r['reference_id'] for r in selection] == [r['id'] for r in candidates]
        and [r['id'] for r in held] == ['heldout_' + r['id'] + '_' + role for r in candidates for role in ('normal', 'dark')],
        'Canonical heldout source selection/order differs')
    specs = []
    for row, source in zip(fitted, parity):
        rid, role = row['id'], 'normal' if source['slot'] in (1, 3) else 'dark'
        ref = refs[source['reference_id']]
        pair = read(e.dep('adain', source['pair_record_path']))
        cache = read(e.dep('tone_training', row['cache_path']))
        require(rid == 'train_u%04d_slot%d' % (source['update'], source['slot']) and row['role'] == role
            and row['reference_id'] == source['reference_id'] == pair['reference_id']
            and row['source_sha256'] == source['source_sha256'] == ref['source_sha256'] == pair['source_sha256']
            and ref['historical_role'] == pair['role'] == 'train' and pair['member'] == role
            and cache['id'] == rid and cache['role'] == role
            and cache['update'] == source['update'] and cache['slot'] == source['slot']
            and cache['target_source'] == ('original' if role == 'normal' else 'full_coarse_base')
            and all(cache['sourcebindings'][k] == source[k] for k in cache['sourcebindings']), 'Fitted source/cache/role join differs')
        inp = rgb8(e.dep('tone_training', row['input_path']))
        require(array_sha(inp) == source['input256_rgb8_sha256'] and array_sha(floats(inp)) == source['input256_float32_sha256'],
                'Fitted input pixel hash differs')
        for key, name in [('base_raw', row['base_raw_path']), ('original_raw', row['original_raw_path'])]:
            pin = e.p['dependencies']['tone_training'][name]
            require(cache[key] == {'path': name, **pin}, 'Fitted parent array cache binding differs')
        base, original = [raw256(e.dep('tone_training', row[k])) for k in ('base_raw_path', 'original_raw_path')]
        mask = np.ones((256, 256), bool)
        feature = feature_vector(inp, mask)
        require(np.array_equal(feature, np.asarray(cache['features'], np.float32)), 'Cached input features differ')
        target = cache['target_mean_y']
        require(abs(target - mean_y(original if role == 'normal' else base, mask)) <= MEAN_BOUND, 'Fitted target is not declared parent mean')
        specs.append(dict(row, input=inp, observed=mask, features=feature, base=base, original=original,
            target=target, sourcebindings=source, pair_id=pair['pair_id'], profile=pair['profile'],
            target_policy='exact_retained_fitted_target_mean'))
    for row in held:
        pair = e.json(row['pair_record_path'])
        ref = refs[row['reference_id']]
        inp = rgb8(e.file(row['input_path']))
        require(e.hashes[row['input_path']] == row['input_sha256'] and e.hashes[row['pair_record_path']] == row['pair_record_sha256']
            and pair['reference_id'] == row['reference_id'] and pair['source_sha256'] == row['source_sha256'] == ref['source_sha256']
            and pair['role'] == ref['historical_role'] == 'train' and pair['member'] == row['role']
            and pair['update'] == row['camera_update'] and pair['pair_id'] == row['pair_id']
            and pair['source_path'] == ref['source_path']
            and array_sha(inp) == row['input256_rgb8_sha256'] == pair['input256_rgb8_sha256']
            and array_sha(floats(inp)) == row['input256_float32_sha256'] == pair['input256_float32_sha256'], 'Heldout input/source join differs')
        prefix = RUN + '/new/' + row['id'] + '/'
        original, base = raw256(e.file(prefix + 'original_raw.npy')), raw256(e.file(prefix + 'base_raw.npy'))
        mask = np.ones((256, 256), bool)
        specs.append(dict(row, input=inp, observed=mask, features=feature_vector(inp, mask), base=base,
            original=original, target=None, sourcebindings=pair, profile=pair['profile'],
            target_policy='fresh_frozen_original_mean' if row['role'] == 'normal' else 'fresh_frozen_base_mean'))
    require(Counter((r['cohort'], r['role']) for r in specs) ==
        {('fitted', 'normal'): 100, ('fitted', 'dark'): 100, ('heldout', 'normal'): 32, ('heldout', 'dark'): 32}, 'Four cohort census differs')
    require(len({r['id'] for r in specs}) == 264, 'Duplicate response input')
    return specs


def stats(values):
    values = [float(v) for v in values]
    return {'minimum': min(values), 'mean': math.fsum(values) / len(values), 'maximum': max(values)}


def recorded_summary(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['cohort'], r['role'], r['head']].append(r)
    summary = []
    for (cohort, role, head), rows in sorted(groups.items()):
        summary.append({'cohort': cohort, 'role': role, 'head': head, 'cases': len(rows),
            **{key: stats(r[key] for r in rows) for key in ('delta', 'loss', 'signed_residual')},
            'negative_offsets': sum(r['delta'] < 0 for r in rows), 'positive_offsets': sum(r['delta'] > 0 for r in rows),
            'zero_offsets': sum(r['delta'] == 0 for r in rows)})
    pairs = []
    for rid in sorted({r['id'] for r in records}):
        a, b = [next(r for r in records if r['id'] == rid and r['head'] == head) for head in ('initial', 'final')]
        pairs.append({'id': rid, 'reference_id': a['reference_id'], 'cohort': a['cohort'], 'role': a['role'],
            'initial_loss': a['loss'], 'final_loss': b['loss'], 'loss_change': b['loss'] - a['loss'],
            'initial_signed_residual': a['signed_residual'], 'final_signed_residual': b['signed_residual'],
            'final_delta': b['delta'], 'target_mean_y': a['target_mean_y'],
            'desired_mean_change': a['target_mean_y'] - a['mean_y']})
    return summary, pairs


def render(out, specs, pngs):
    folder = out / 'visual_sheets'; folder.mkdir()
    cells, sheets = [], []
    columns = ('input', 'untouched original / derived PNG', 'full-coarse base / derived PNG', 'initial head / saved PNG', 'final head50 / saved PNG')
    for cohort, role in [('fitted', 'normal'), ('fitted', 'dark'), ('heldout', 'normal'), ('heldout', 'dark')]:
        rows = [r for r in specs if (r['cohort'], r['role']) == (cohort, role)]
        for start in range(0, len(rows), 12):
            subset = rows[start:start+12]
            name = cohort + '_' + role + '_%02d.png' % (start//12 + 1)
            image = Image.new('RGB', (1280, 55 + 280 * len(subset)), 'white')
            draw = ImageDraw.Draw(image)
            draw.text((4, 4), cohort + ' ' + role + ' TRAIN | exact256 panels | no clean target, enhancement or resize', fill='black')
            for col, title in enumerate(columns):
                draw.text((col*256 + 4, 25), title, fill='black')
            for num, row in enumerate(subset):
                rid, mask = row['id'], row['observed']
                values = [row['input'], png_pixels(row['original'], row['input'], mask),
                    png_pixels(row['base'], row['input'], mask), pngs['initial', rid], pngs['final', rid]]
                draw.text((4, 55 + 280*num), rid + ' | ' + row['reference_id'] + ' | ' + row['profile'], fill='black')
                for col, value in enumerate(values):
                    x, y = col*256, 79+280*num
                    image.paste(Image.fromarray(value), (x, y))
                    cells.append({'sheet': name, 'id': rid, 'column': col, 'x': x, 'y': y,
                        'RGB8_SHA256': array_sha(value), 'width': 256, 'height': 256,
                        'source': 'derived floor(RAW*255)' if col in (1, 2) else 'retained exact RGB8'})
            path = folder / name; image.save(path)
            check = np.asarray(Image.open(path))
            for cell in (c for c in cells if c['sheet'] == name):
                require(array_sha(check[cell['y']:cell['y']+256, cell['x']:cell['x']+256]) == cell['RGB8_SHA256'], 'Saved sheet panel differs')
            sheets.append({'path': path.relative_to(out).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size,
                'cohort': cohort, 'role': role, 'case_ids': [r['id'] for r in subset], 'case_count': len(subset)})
    require(len(sheets) == 24 and len(cells) == 1320 and len({i for s in sheets for i in s['case_ids']}) == 264,
            'All24 exact grouped sheets required')
    write(out / 'sheet_cells.json', cells)
    write(out / 'visual_manifest.json', {'complete': True, 'sheets': sheets, 'planned_sheets': 24,
        'all264_unique_inputs_once': True, 'fitted_inputs': 200, 'heldout_inputs': 64,
        'exact256_cells_readback': 1320, 'columns': columns, 'no_clean_target_panels': True,
        'altered_returned_evidence_images': 0, 'personal_visual_review_performed': False,
        'root_must_view_every_sheet': True, 'source_reference_is_not_person_identity': True})
    with (folder / 'INDEX.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('# Fixed-state TRAIN response sheets\n\n24 grouped sheets cover all264 inputs exactly once: fitted normal100, fitted dark100, heldout normal32, heldout dark32. Columns: input, untouched original, full-coarse base, initial head, final head50. All panels retain exact256 pixels. Original/base panels are the declared floor(RAW*255) delivery, not new neural outputs. No clean target or independent DEV/FINAL/native evaluation is shown.\n\n')
        stream.write('\n'.join('- [' + Path(s['path']).name + '](' + Path(s['path']).name + ')' for s in sheets) + '\n')
    return sheets


def audit(receipt, out):
    require(not any(k in sys.modules for k in ('torch', 'tensorflow', 'jax')), 'Framework import forbidden')
    require(not out.exists(), 'Preserve prior audit version')
    start = time.monotonic(); e = Evidence(receipt); specs = load_specs(e)
    out.mkdir(parents=True)
    records, results, pngs = [], [], {}
    max_errors = defaultdict(float)
    for num, row in enumerate(specs, 1):
        rid, mask = row['id'], row['observed']
        base_mean, original_mean = mean_y(row['base'], mask), mean_y(row['original'], mask)
        record_pair = []
        for head in ('initial', 'final'):
            prefix = RUN + '/' + head + '/' + rid + '/'
            record = e.json(prefix + 'record.json')
            require(all(record[k] == row[k] for k in ('id', 'reference_id', 'source_sha256', 'cohort', 'role', 'pair_id'))
                and record['head'] == head and record['sourcebindings'] == row['sourcebindings']
                and record['target_policy'] == row['target_policy'] and record['support_is_not_covering_mask'] is True
                and record['clean_pixels_used_for_target'] is False and record['independent_FINAL_or_native_evaluation'] is False
                and record['fitted_parent_arrays_reused_without_new_CodeFormer_calls'] == (row['cohort'] == 'fitted'),
                'Saved response source/role/target scope differs')
            e.binding(record['RAW'], prefix + 'raw.npy'); e.binding(record['PNG'], prefix + 'native.png')
            if row['cohort'] == 'heldout':
                for key, name in [('original_raw', 'original_raw.npy'), ('base_raw', 'base_raw.npy')]:
                    e.binding(record['new_parent_RAW'][key], RUN + '/new/' + rid + '/' + name)
            else:
                require(record['new_parent_RAW'] is None, 'Fitted arrays represented as fresh forwards')
            require(np.array_equal(np.asarray(record['features'], np.float32), row['features']), 'Recorded features differ')
            raw, png = raw256(e.file(prefix + 'raw.npy')), rgb8(e.file(prefix + 'native.png'))
            delta = record['delta']; target = record['target_mean_y']
            require(type(delta) in (int, float) and math.isfinite(delta) and -.25 <= delta <= .25, 'Bounded delta required')
            require(0 <= target <= 1 and math.isfinite(target), 'Finite behavioral scalar target required')
            if row['target'] is not None:
                require(target == row['target'], 'Retained fitted target scalar changed')
            else:
                require(abs(target - (original_mean if row['role'] == 'normal' else base_mean)) <= MEAN_BOUND,
                        'Fresh heldout target is not declared parent mean')
            reconstructed = compose(row['base'], delta, row['input'], mask)
            require(np.array_equal(raw, reconstructed), 'Actual clipped FP32 scalar composition differs: ' + rid)
            require(np.array_equal(png, png_pixels(raw, row['input'], mask)), 'RAW-to-delivered-PNG differs: ' + rid)
            require(np.array_equal(raw[~mask], floats(row['input'])[~mask])
                and np.array_equal(png[~mask], row['input'][~mask]), 'Protected support pixels changed')
            if head == 'initial':
                require(delta == 0 and np.array_equal(raw[mask], row['base'][mask]), 'Actual zero-head/base parity differs')
            actual_mean = mean_y(raw, mask)
            for name, expected in [('mean_y', actual_mean), ('original_mean_y', original_mean), ('base_mean_y', base_mean)]:
                err = abs(record[name] - expected); max_errors[name] = max(max_errors[name], err)
                require(err <= MEAN_BOUND, 'CPU float64/recorded CUDA FP32 mean outside bound: ' + name)
            scalar_error = np.float32(record['mean_y']) - np.float32(target)
            expected_loss = float(np.float32(scalar_error * scalar_error))
            require(record['loss'] == expected_loss and record['signed_residual'] == record['mean_y'] - target,
                    'Recorded FP32 loss/residual does not follow actual mean/target')
            unclipped = row['base'][mask] + np.float32(delta)
            result = {'id': rid, 'reference_id': row['reference_id'], 'cohort': row['cohort'], 'role': row['role'],
                'profile': row['profile'], 'head': head, 'delta': delta, 'target_mean_y': target,
                'CPU_float64_mean_y': actual_mean, 'CPU_float64_signed_residual': actual_mean - target,
                'CPU_float64_behavioral_loss': (actual_mean - target)**2,
                'recorded_CUDA_FP32_mean_y': record['mean_y'], 'recorded_CUDA_FP32_loss': record['loss'],
                'clipped_scalar_fraction': float(((unclipped < 0) | (unclipped > 1)).mean()),
                'RAW_composition_exact': True, 'RAW_PNG_exact': True, 'input_features_exact': True,
                'observed_pixels': int(mask.sum()), 'protected_pixels': int((~mask).sum()),
                'protected_check_has_nonempty_support': bool((~mask).any()),
                'zero_head_parity_exact': True if head == 'initial' else None,
                'clean_target_metrics': None, 'native_CCTV_metrics': None}
            records.append(record); results.append(result); record_pair.append(result); pngs[head, rid] = png
        require(record_pair[0]['target_mean_y'] == record_pair[1]['target_mean_y'], 'Head states use different target')
        if num % 50 == 0 or num == 264:
            print(json.dumps({'pixel_inputs_verified': num, 'of': 264}), flush=True)
    for head in ('initial', 'final'):
        prefix = RUN + '/' + head + '/'
        expected = {prefix + r['id'] + '/' + name for r in specs for name in ('raw.npy', 'native.png', 'record.json')}
        require({n for n in e.hashes if n.startswith(prefix)} == expected, 'Exact528 bank output census differs')
    require({n for n in e.hashes if n.startswith(RUN + '/new/')} ==
        {RUN + '/new/' + r['id'] + '/' + name for r in specs if r['cohort'] == 'heldout' for name in ('base_raw.npy', 'original_raw.npy')},
        'Exact128 new parent RAW census differs')
    summary = e.json(RUN + '/response_summary.json'); group_saved, pairs_saved = recorded_summary(records)
    require(summary['complete'] is True and exact_json(summary['groups'], group_saved)
        and exact_json(summary['paired_fixed_state_comparisons'], pairs_saved)
        and summary['new_acceptance_threshold_or_selection'] is False and summary['model_qualified'] is False,
        'Worker response summary does not reproduce from independently validated records')
    terminal = e.json(RUN + '/run_results.json')
    require(terminal['measurement_complete'] is True and terminal['counts'] == COUNTS
        and terminal['finished_neural_calls'] == 656 and terminal['new_checkpoints_written'] is False
        and all(terminal[k] == 0 for k in ('head_updates_this_run', 'restorer_updates_this_run', 'gradient_queries',
            'backward_calls', 'optimizer_constructions', 'optimizer_steps', 'committed_training_updates')),
        'Recorded complete zero-update operation census differs')
    require(e.json(RUN + '/initial_state.json') == e.json(RUN + '/final_state.json'), 'Recorded frozen state before/after differs')
    groups = []; by_id = {r['id']: {h: next(v for v in results if v['id'] == r['id'] and v['head'] == h)
                                  for h in ('initial', 'final')} for r in specs}
    for cohort, role in [('fitted', 'normal'), ('fitted', 'dark'), ('heldout', 'normal'), ('heldout', 'dark')]:
        subset = [r for r in specs if (r['cohort'], r['role']) == (cohort, role)]
        pairs = [by_id[r['id']] for r in subset]
        changes = [v['final']['CPU_float64_behavioral_loss'] - v['initial']['CPU_float64_behavioral_loss'] for v in pairs]
        groups.append({'cohort': cohort, 'role': role, 'cases': len(subset),
            'final_delta': stats(v['final']['delta'] for v in pairs),
            'negative_final_offsets': sum(v['final']['delta'] < 0 for v in pairs),
            'positive_final_offsets': sum(v['final']['delta'] > 0 for v in pairs),
            'zero_final_offsets': sum(v['final']['delta'] == 0 for v in pairs),
            'initial_CPU_behavioral_loss': stats(v['initial']['CPU_float64_behavioral_loss'] for v in pairs),
            'final_CPU_behavioral_loss': stats(v['final']['CPU_float64_behavioral_loss'] for v in pairs),
            'CPU_behavioral_loss_change': stats(changes), 'strictly_improved': sum(v < 0 for v in changes),
            'strictly_worse': sum(v > 0 for v in changes), 'exactly_equal': sum(v == 0 for v in changes),
            'initial_CPU_signed_residual': stats(v['initial']['CPU_float64_signed_residual'] for v in pairs),
            'final_CPU_signed_residual': stats(v['final']['CPU_float64_signed_residual'] for v in pairs),
            'normal_positive_to_negative_overshoot': sum(v['initial']['CPU_float64_signed_residual'] > 0
                and v['final']['CPU_float64_signed_residual'] < 0 for v in pairs) if role == 'normal' else None})
    references = []
    for cohort, rid in sorted({(r['cohort'], r['reference_id']) for r in specs}):
        subset = {r['role']: r for r in specs if (r['cohort'], r['reference_id']) == (cohort, rid)}
        require(set(subset) == {'normal', 'dark'}, 'Per-reference normal/dark pairing differs')
        references.append({'cohort': cohort, 'reference_id': rid, 'source_sha256': subset['normal']['source_sha256'],
            'normal': by_id[subset['normal']['id']], 'dark': by_id[subset['dark']['id']],
            'normal_dark_feature_L2': float(np.linalg.norm(subset['normal']['features'].astype(np.float64) - subset['dark']['features'])),
            'reference_id_is_not_proven_person_identity': True})
    write(out / 'input_audit.json', {'complete': True, 'cases': [{k: r[k] for k in
        ('id', 'reference_id', 'source_sha256', 'cohort', 'role', 'profile', 'pair_id', 'target_policy')} for r in specs],
        'all264_unique_inputs': True, 'fitted_sources': 100, 'heldout_sources': 32,
        'exact_source_hashes_disjoint': True, 'all_observed_masks_full_true': True,
        'protected_pixel_count': 0, 'support_is_not_covering_mask': True, 'DEV_FINAL_native_pixels_opened': 0})
    write(out / 'case_results.json', {'complete': True, 'responses': results, 'cases': by_id})
    write(out / 'reference_pairs.json', {'complete': True, 'references': references, 'source_not_person_identity': True})
    write(out / 'response_groups.json', {'complete': True, 'groups': groups, 'quality_acceptance_thresholds': None})
    print(json.dumps({'verified_response_groups': groups}), flush=True)
    sheets = render(out, specs, pngs)
    report = {'format': 'codeformer-fixed-state-response-independent-pixels-v1', 'complete': True, 'audit_pass': True,
        'UTC': datetime.now(timezone.utc).isoformat(), 'seconds': time.monotonic()-start,
        'protocol_sha256': PROTOCOL, 'archive_sha256': ARCHIVE, 'manifest_sha256': MANIFEST,
        'import_receipt_sha256': sha(receipt), 'stage_receipt_sha256': sha(PREP/'stage_v1.json'),
        'audit_script_sha256': sha(__file__), 'current_source_sha256': {n: sha(ROOT/n) for n in
            ('research/codeformer_fixed_state_response_v1/model.py', 'research/codeformer_fixed_state_response_v1/data.py',
             'research/codeformer_fixed_state_response_v1/worker.py', 'research/codeformer_fixed_state_response_v1/design.json')},
        'counts': {'unique_inputs': 264, 'fitted': 200, 'heldout': 64, 'responses_RAW_PNG': 528,
            'new_parent_RAW': 128, 'reused_fitted_parent_RAW': 400, 'zero_head_parity_cases': 264,
            'verified_features': 528, 'planned_sheets': 24, 'exact256_panels': 1320},
        'groups': groups, 'CPU_float64_vs_recorded_CUDA_FP32_mean_error_max': dict(max_errors),
        'CPU_mean_readback_bound': MEAN_BOUND, 'loss_from_recorded_FP32_mean_target_exact': True,
        'recorded_summary_exact': True, 'RAW_composition_all_exact': True, 'RAW_PNG_all_exact': True,
        'frozen_state_before_after_recorded_exact': True, 'recorded_counts': terminal['counts'],
        'all_masks_full_observed_support': True, 'nonempty_protected_pixel_cases': 0,
        'protected_check_vacuous_for_this_full_support_cohort': True,
        'support_is_not_covering_mask': True, 'source_reference_is_not_person_identity': True,
        'checked_returned_file_sha256': {n: e.hashes[n] for n in sorted(e.checked)}, 'checked_dependencies': e.deps,
        'visual_manifest_sha256': sha(out/'visual_manifest.json'), 'sheet_cells_sha256': sha(out/'sheet_cells.json'),
        'sheets': sheets, 'visual_review_complete': False, 'root_must_view_all24_sheets': True,
        'environment': {'Python': sys.version, 'NumPy': np.__version__, 'Pillow': pillow_version},
        'local_neural_calls': 0, 'local_autograd_queries': 0, 'local_backward_calls': 0, 'local_optimizer_steps': 0,
        'model_qualified': False, 'app_promoted': False, 'new_training_or_model': False,
        'limitations': ['Saved-delta composition is arithmetic readback, not a head forward or independent parameter-to-delta replay.',
            'Actual zero gradient/backward/optimizer counts and live state immutability remain source/record-bound; no gradient arrays exist.',
            'Behavioral scalar losses compare normal to original mean and dark to base mean; no clean-target restoration metric is inferred.',
            'Fitted and heldout are photographic TRAIN-source roles, not independent FINAL, identity separation or native CCTV evidence.',
            'Full support makes protected-pixel checks vacuous; this does not qualify covering/completion behavior.',
            'Float64 CPU means check recorded CUDA FP32 reductions within explicit bound; identical CUDA reduction execution is not claimed.']}
    require(not any(k in sys.modules for k in ('torch', 'tensorflow', 'jax')), 'Framework imported during audit')
    write(out/'audit.json', report)
    with (out/'audit.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('# Independent fixed-state saved-pixel audit\n\nAll528 RAW/PNG responses,128 fresh parent arrays and400 reused fitted parent arrays passed independent input/source/features/composition/scalar checks. All264 initial-head outputs exactly equal their base on observed support. All masks are full; no covering/protected-pixel capability follows.\n\n')
        for group in groups:
            stream.write('- %s %s: %d inputs; final delta mean %.9g; negative %d; behavioral loss mean %.9g to %.9g; worse %d.\n' %
                (group['cohort'], group['role'], group['cases'], group['final_delta']['mean'], group['negative_final_offsets'],
                 group['initial_CPU_behavioral_loss']['mean'], group['final_CPU_behavioral_loss']['mean'], group['strictly_worse']))
        stream.write('\n[All24 exact256 comparison sheets](visual_sheets/INDEX.md) require root visual review. This measurement created no trained model and establishes no clean-target restoration quality. No neural framework, model/head forward, autograd, optimizer or VM execution occurred locally.\n')
    print(json.dumps({'complete': True, 'audit': str(out/'audit.json'), 'sha256': sha(out/'audit.json'),
        'all24_sheets_ready': str(out/'visual_sheets/INDEX.md')}), flush=True)
    return report


def self_test():
    import unittest
    class Fixtures(unittest.TestCase):
        def test_features_support(self):
            inp = np.full((256,256,3), 64, np.uint8); mask = np.ones((256,256), bool)
            f = feature_vector(inp, mask).reshape(4,8,8)
            self.assertTrue(np.all(f[:3] == np.float32(64)/np.float32(255)))
            self.assertTrue(np.all(f[3] == 1))
            mask[:32,:32] = False; f = feature_vector(inp, mask).reshape(4,8,8)
            self.assertTrue(np.all(f[:,0,0] == 0))
        def test_composition_clamp_protection(self):
            inp = np.full((256,256,3), 99, np.uint8); mask = np.ones((256,256), bool); mask[0,0] = False
            base = np.full((256,256,3), .02, np.float32)
            value = compose(base, -.1, inp, mask)
            self.assertTrue(np.all(value[mask] == 0)); self.assertTrue(np.array_equal(value[0,0], floats(inp)[0,0]))
            self.assertTrue(np.array_equal(png_pixels(value, inp, mask)[0,0], inp[0,0]))
        def test_zero_head_and_mean(self):
            inp = np.full((256,256,3), 255, np.uint8); mask = np.ones((256,256), bool)
            base = np.full((256,256,3), .25, np.float32)
            self.assertTrue(np.array_equal(compose(base, 0, inp, mask), base))
            self.assertAlmostEqual(mean_y(base, mask), .25, places=15)
        def test_recorded_loss_cast(self):
            a, b = .4832971394062042, .50000001
            loss = float(np.float32((np.float32(a)-np.float32(b))**2))
            self.assertTrue(math.isfinite(loss) and loss > 0)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Fixtures)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), 'CPU arithmetic fixture failure')
    require(not any(n in sys.modules for n in ('torch', 'tensorflow', 'jax')), 'Framework imported by fixtures')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-receipt', type=Path, default=REVIEW/'return_import_v1.json')
    parser.add_argument('--out', type=Path, default=REVIEW/'pixels_v1')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        audit(args.import_receipt, args.out)


if __name__ == '__main__':
    main()
