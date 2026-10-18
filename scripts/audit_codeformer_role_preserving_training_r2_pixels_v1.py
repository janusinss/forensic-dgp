"""Independent CPU saved-pixel audit of the finite R2 restoration pilot.

Read only a strict hash-closed import. No returned Python, checkpoint, neural
model, training or gradient executes. Contact-sheet cells retain exact pixels.
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
from scipy.ndimage import binary_erosion
import scipy
import skimage
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = 'd47c88c8889b9bda66eeba03f498c7ab81173e4a75ffa2572a75049e73084689'
PROTOCOL_SHA = '82fcd34cada1eed238ef65a341c22580d12146e3ee30c2ff3094312d3ba1c85a'
RUN = 'outputs/role_preserving_training_v1_r2'
PREP = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_preparation'
BANKS = ('original0', 'parent50', 'candidate0', 'candidate50')
VIEWS = ('raw', 'native_png')
WORKER_METRICS = ('mae', 'mse', 'psnr', 'ssim', 'luminance_bias', 'coarse_rgb_mae')
W = np.asarray([.2126, .7152, .0722], np.float64)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def checked(root, name):
    relative = PurePosixPath(name)
    require(isinstance(name, str) and not relative.is_absolute() and relative.as_posix() == name and
        all(part not in ('', '.', '..') for part in relative.parts) and '\\' not in name and ':' not in name,
        'Noncanonical evidence path')
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*relative.parts)
    require(path.resolve(strict=True).is_relative_to(root), 'Evidence path escapes root')
    for item in (path, *path.parents):
        if item == root.parent:
            break
        info = item.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
            'Linked/reparse evidence forbidden')
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Unlinked regular evidence required')
    return path


def pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rgb8(path):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (256, 256), 'Exact RGB256 PNG required')
        return np.asarray(image).copy()


def observed(path):
    with Image.open(path) as image:
        require(image.size == (256, 256), 'Observed support geometry differs')
        array = np.asarray(image.convert('L')).copy()
    require(np.isin(array, (0, 255)).all() and array.any(), 'Binary nonempty observed support required')
    return array > 0


def floats(array):
    return np.ascontiguousarray(array.astype(np.float32) / np.float32(255))


def raw256(path):
    require(Path(path).stat().st_size <= 3 * 256 * 256 * 4 + 256, 'Bounded RGB256 NPY required')
    with Path(path).open('rb') as stream:
        array = np.load(stream, allow_pickle=False)
        require(not stream.read(1), 'Trailing NPY data')
    require(array.shape == (256, 256, 3) and array.dtype == np.float32 and array.flags.c_contiguous and
        np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1, 'Finite bounded contiguous RAW256 required')
    return array


def expected_png(raw, input8, mask):
    png = np.floor(raw * np.float32(255)).astype(np.uint8)
    png[~mask] = input8[~mask]
    return png


def metrics(image, target, mask, native):
    if native:
        require(target is None, 'Native aligned clean target is forbidden')
        return dict.fromkeys((*WORKER_METRICS, 'signed_luminance_bias', 'gradient_vector_mae', 'masked_paired_SSIM'))
    require(target is not None and target.shape == image.shape, 'Paired target geometry required')
    predicted, truth = image.astype(np.float64), target.astype(np.float64)
    diff = predicted[mask] - truth[mask]
    mse = float(np.square(diff).mean())
    result = {'mae': float(np.abs(diff).mean()), 'mse': mse,
        'psnr': None if mse == 0 else -10 * math.log10(mse), 'ssim': None,
        'luminance_bias': float(abs((diff @ W).mean())), 'signed_luminance_bias': float((diff @ W).mean()),
        'coarse_rgb_mae': None, 'gradient_vector_mae': None, 'masked_paired_SSIM': None}
    score, field = structural_similarity(predicted, truth, data_range=1., channel_axis=2, win_size=7, full=True)
    valid7 = binary_erosion(mask, structure=np.ones((7, 7), bool), border_value=0)
    if valid7.any():
        result['masked_paired_SSIM'] = float(field[valid7].mean(dtype=np.float64))
    if mask.all():
        result['ssim'] = float(score)
        pool = lambda a: a.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3), dtype=np.float64)
        result['coarse_rgb_mae'] = float(np.abs(pool(predicted) - pool(truth)).mean())
    valid3 = binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)[1:-1, 1:-1]
    if valid3.any():
        py, ty = predicted @ W, truth @ W
        px, pygrad = (py[1:-1, 2:] - py[1:-1, :-2]) / 2, (py[2:, 1:-1] - py[:-2, 1:-1]) / 2
        tx, tygrad = (ty[1:-1, 2:] - ty[1:-1, :-2]) / 2, (ty[2:, 1:-1] - ty[:-2, 1:-1]) / 2
        result['gradient_vector_mae'] = float((np.abs(px - tx)[valid3].mean() + np.abs(pygrad - tygrad)[valid3].mean()) / 2)
    return result


def descriptors(image, input8, mask):
    values = image.astype(np.float64)
    y = values @ W
    return {'observed_mean_luminance': float(y[mask].mean()),
        'observed_luminance_std': float(y[mask].std()),
        'mean_R_minus_G': float((values[..., 0] - values[..., 1])[mask].mean()),
        'mean_B_minus_G': float((values[..., 2] - values[..., 1])[mask].mean()),
        'MAE_from_input_not_clean': float(np.abs(values[mask] - floats(input8)[mask]).mean()),
        'output_change_is_not_identity_or_quality_proof': True}


def near(a, b, atol=1e-10):
    return a is None and b is None or a is not None and b is not None and abs(a - b) <= atol


def compare_tree(actual, expected, prefix=''):
    errors = []
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            errors.append(prefix + ':keys')
        for key in set(actual) & set(expected):
            errors += compare_tree(actual[key], expected[key], prefix + '/' + key)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            errors.append(prefix + ':length')
        for index, (a, b) in enumerate(zip(actual, expected)):
            errors += compare_tree(a, b, prefix + '/' + str(index))
    elif type(actual) in (int, float) and type(expected) in (int, float):
        if not near(float(actual), float(expected)):
            errors.append(prefix + ':numeric')
    elif actual != expected:
        errors.append(prefix + ':value')
    return errors


class Evidence:
    def __init__(self, receipt_path):
        self.receipt_path = Path(receipt_path)
        self.receipt = read(self.receipt_path)
        r = self.receipt
        require(r.get('complete') is True and r['archive_sha256'] == ARCHIVE_SHA and
            r['protocol_sha256'] == PROTOCOL_SHA, 'Completed strict exact R2 import required')
        self.root = Path(r['import_root']).resolve(strict=True)
        self.hashes, self.sizes = r['files_sha256'], r['files_bytes']
        require(set(self.hashes) == set(self.sizes), 'Import hash/size namespace differs')
        self.checked, self.dep_checked = set(), {}
        self.p = self.json('protocol.json')
        require(self.hashes['protocol.json'] == PROTOCOL_SHA, 'Imported protocol changed')
        self.stage = read(PREP / 'stage_v1.json')
        require(self.stage['complete'] is True and self.stage['protocol_sha256'] == PROTOCOL_SHA,
            'Actual local preparation resolver differs')
        self.pins = {}
        for key in ('dependencies', 'retained_output_dependencies', 'extra_returned_dependencies'):
            require(self.stage[key] == self.p[key], 'Dependency map changed: ' + key)
            self.pins.update(self.p[key])
        manifest_name = RUN + '/export_manifest.json'
        manifest = self.json(manifest_name)
        require(manifest['protocol_sha256'] == PROTOCOL_SHA and set(manifest['files_sha256']) | {manifest_name} == set(self.hashes)
            and all(self.hashes[n] == h for n, h in manifest['files_sha256'].items())
            and all(self.sizes[n] == size for n, size in manifest['files_bytes'].items()), 'Return manifest/import closure differs')

    def file(self, name):
        require(name in self.hashes, 'Unbound returned file: ' + name)
        path = checked(self.root, name)
        if name not in self.checked:
            before = path.stat()
            require(before.st_size == self.sizes[name] and sha(path) == self.hashes[name], 'Returned bytes changed: ' + name)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino), 'Returned file changed while hashing')
            self.checked.add(name)
        return path

    def json(self, name):
        return read(self.file(name))

    def dep(self, name):
        require(name in self.pins, 'Unbound parent file: ' + name)
        path = Path(self.stage['local_dependency_resolution'][name])
        require(path.resolve(strict=True).is_relative_to(ROOT), 'Dependency outside workspace')
        checked(ROOT, path.relative_to(ROOT).as_posix())
        if name not in self.dep_checked:
            before = path.stat(); pin = self.pins[name]
            require(before.st_size == pin['bytes'] and sha(path) == pin['sha256'], 'Parent bytes changed: ' + name)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino), 'Parent changed while hashing')
            self.dep_checked[name] = dict(pin, local_path=str(path))
        return path


def load_specs(e):
    bundle = e.json('data_bundle.json')
    refs = {row['id']: row for row in read(e.dep('references.json'))['references']}
    require(Counter(row['historical_role'] for row in refs.values()) == {'train': 391, 'validation': 53}, 'Reference role census differs')
    train_hashes = {r['source_sha256'] for r in refs.values() if r['historical_role'] == 'train'}
    require(not train_hashes & {r['source_sha256'] for r in refs.values() if r['historical_role'] == 'validation'}, 'TRAIN/validation photograph overlap')
    specs, joins = {}, []
    for row in bundle['development']:
        rid = row['id']; require(rid not in specs and not rid.lower().startswith('final'), 'Duplicate or reserved FINAL input')
        inp = rgb8(e.dep(row['input_path']))
        mask = observed(e.dep(row['observed_path'])) if row['observed_path'] else np.ones((256, 256), bool)
        native = row['evidence'] == 'native_unpaired'
        target8 = rgb8(e.dep(row['target_path'])) if row['target_path'] else None
        require(native == (target8 is None), 'Native/paired target policy differs')
        if row['source'] == 'FFHQ validation':
            origin = row['sourcebindings']['validation_saved']; pair = row['sourcebindings']['pair_record']
            require(read(e.dep(origin['pair_record_path'])) == pair, 'Saved pair source record differs')
            ref = refs[pair['reference_id']]
            require(ref['historical_role'] == pair['role'] == 'validation' and ref['source_sha256'] == pair['source_sha256']
                and row['target_path'] == ref['existing_target_path'] and row['target_path'] == origin['target_path']
                and pixel_sha(inp) == pair['input256_rgb8_sha256'] and pixel_sha(floats(inp)) == pair['input256_float32_sha256'], 'FFHQ observed pixel/reference join differs')
        else:
            source = row['sourcebindings']
            require(source['prepared_path'] == row['input_path'] and source['observed_path'] == row['observed_path']
                and source.get('target_path') == row['target_path'], 'Comparison geometry/source join differs')
        specs[rid] = {'id': rid, 'source': row['source'], 'profile': row['profile'], 'evidence': row['evidence'],
            'sourcebindings': row['sourcebindings'], 'input': inp, 'observed': mask, 'target8': target8,
            'target': None if target8 is None else floats(target8), 'row': row}
        joins.append({'id': rid, 'input_file_SHA256': e.pins[row['input_path']]['sha256'], 'input_RGB8_SHA256': pixel_sha(inp),
            'observed_file_SHA256': e.pins[row['observed_path']]['sha256'] if row['observed_path'] else None,
            'target_file_SHA256': e.pins[row['target_path']]['sha256'] if row['target_path'] else None,
            'native_clean_target_null': native, 'source': row['source'], 'evidence': row['evidence']})
    require(len(specs) == 406 and Counter(r['evidence'] for r in specs.values()) == {'paired_synthetic': 358, 'native_unpaired': 48}, 'Full406 evidence census differs')
    return specs, joins


def preservation(rows, baseline):
    reports = {}
    for label, cases, names in [('clear', [k for k, r in rows.items() if r['evidence'] == 'paired_synthetic' and r['profile'] == 'clear'], ['mae']),
        ('normal_blur', [k for k, r in rows.items() if r['source'] == 'FFHQ validation' and r['profile'] == 'blur'], ['luminance_bias', 'coarse_rgb_mae'])]:
        require(len(cases) == (61 if label == 'clear' else 53), 'Unchanged guard membership required')
        report = {'pass': True, 'cases': len(cases), 'mean_limit': .01, 'worst_limit': .03, 'views': {}}
        for view in VIEWS:
            report['views'][view] = {}
            for metric in names:
                deltas = {rid: rows[rid]['metrics'][view][metric] - baseline[rid]['metrics'][view][metric] for rid in sorted(cases)}
                avg, worst = sum(deltas.values()) / len(deltas), max(deltas.values())
                passed = avg <= .01 and worst <= .03
                report['views'][view][metric] = {'mean_delta': avg, 'worst_delta': worst, 'deltas': deltas,
                    'cases_over_0_03': sum(value > .03 for value in deltas.values()), 'pass': passed}
                report['pass'] = report['pass'] and passed
        reports[label] = report
    reports['pass'] = reports['clear']['pass'] and reports['normal_blur']['pass']
    return reports


def dark_retention(rows, baseline):
    result = {'pass': True, 'profiles': {}, 'prospective_additional_gate': True}
    for profile in ('lowlight', 'compound'):
        ids = sorted(rid for rid, row in rows.items() if row['source'] == 'FFHQ validation' and row['profile'] == profile)
        require(len(ids) == 53, 'Full53 dark cases required')
        means = {name: {metric: sum(bank[rid]['metrics']['native_png'][metric] for rid in ids) / len(ids)
            for metric in ('mae', 'ssim')} for name, bank in [('candidate', rows), ('base', baseline)]}
        passed = means['candidate']['mae'] <= means['base']['mae'] and means['candidate']['ssim'] >= means['base']['ssim']
        result['profiles'][profile] = dict(means, cases=ids, pass_=passed)
        result['profiles'][profile]['pass'] = result['profiles'][profile].pop('pass_')
        result['pass'] = result['pass'] and passed
    return result


def render(out, specs, pngs):
    folder = out / 'visual_sheets'; folder.mkdir()
    ids = sorted(specs); sheets, cells = [], []
    for number, begin in enumerate(range(0, len(ids), 12), 1):
        subset = ids[begin:begin + 12]
        name = 'sheet_%02d.png' % number
        canvas = Image.new('RGB', (1280, 55 + 280 * len(subset)), 'white'); draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), 'R2 all406 DEV | exact256 panels; no enhancement/resampling | native unpaired; candidates unqualified', fill='black')
        for col, title in enumerate(('observed input', 'clean target / native blank', 'untouched CodeFormer', 'full-coarse parent50', 'new trained R2 update50')):
            draw.text((col * 256 + 4, 25), title, fill='black')
        for row_number, rid in enumerate(subset):
            spec = specs[rid]
            draw.text((4, 55 + row_number * 280), rid + ' | ' + spec['source'] + ' | ' + spec['profile'], fill='black')
            for col, array in enumerate((spec['input'], spec['target8'], pngs['original0', rid], pngs['parent50', rid], pngs['candidate50', rid])):
                y = 79 + row_number * 280
                if array is None:
                    draw.rectangle((col * 256, y, (col + 1) * 256 - 1, y + 255), fill=(235, 235, 235))
                    draw.text((col * 256 + 8, y + 120), 'Native CCTV: no clean target', fill='black')
                    continue
                require(array.shape == (256, 256, 3) and array.dtype == np.uint8, 'Exact256 evidence panel required')
                canvas.paste(Image.fromarray(array), (col * 256, y))
                cells.append({'sheet': name, 'id': rid, 'column': col, 'x': col * 256, 'y': y, 'width': 256, 'height': 256, 'RGB8_SHA256': pixel_sha(array)})
        path = folder / name; canvas.save(path)
        with Image.open(path) as image:
            saved = np.asarray(image)
            for cell in [r for r in cells if r['sheet'] == name]:
                require(pixel_sha(saved[cell['y']:cell['y'] + 256, cell['x']:cell['x'] + 256]) == cell['RGB8_SHA256'], 'Saved contact-sheet cell changed')
        sheets.append({'path': path.relative_to(out).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'case_ids': subset, 'case_count': len(subset), 'width': 1280, 'height': 55 + 280 * len(subset)})
    require(len(sheets) == 34 and len(cells) == 1982, 'Full406/34sheet/1982exactcell census required')
    write(out / 'sheet_cells.json', cells)
    write(out / 'visual_manifest.json', {'complete': True, 'sheets': sheets, 'DEV_cases': 406,
        'sheets_count': 34, 'all406_once': True, 'exact256_cells_readback': len(cells), 'root_must_view_every_sheet': True,
        'personal_visual_review_performed': False, 'image_processing': 'none; no resizing, sharpening or display enhancement',
        'native_aligned_clean_targets': 0, 'model_qualified': False})
    with (folder / 'INDEX.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('# R2 exact-pixel development comparison\n\n34 sheets cover all406 cases once. Columns: observed input, clean target (blank for native), untouched CodeFormer, full-coarse parent50, new trained R2 update50. Every evidence panel is256x256, with no enhancement/resampling. Native sources remain unpaired. Root visual review is pending.\n\n')
        stream.write('\n'.join('- [' + Path(row['path']).name + '](' + Path(row['path']).name + ')' for row in sheets) + '\n')
    return sheets


def audit(receipt, out):
    require(not any(n in sys.modules for n in ('torch', 'tensorflow', 'jax')), 'No neural framework may be imported')
    e = Evidence(receipt)
    require(not out.exists(), 'Preserve old/partial pixel audit; choose a new output version')
    out.mkdir(parents=True); started = time.monotonic()
    specs, joins = load_specs(e)
    require(e.p['evaluation']['mean_regression_limit'] == .01 and e.p['evaluation']['worst_case_regression_limit'] == .03
        and e.p['quality_gates_changed'] is False, 'Restoration quality gates changed')
    write(out / 'input_audit.json', {'complete': True, 'cases': joins, 'DEV': 406, 'paired_synthetic': 358, 'native_unpaired': 48,
        'source_identifiers_not_proven_person_ids': True, 'FINAL_used': False,
        'observed_mask_policy': 'Observed image/letterbox support, not a covering/removal/completion mask'})
    pngs = {}; arrays_checked = []; metric_mismatches = []; rows = {bank: {} for bank in BANKS}
    documents = {at: e.json(RUN + '/snapshot_' + at + '/metrics.json') for at in ('0000', '0050')}
    for at, doc in documents.items():
        require(doc['complete'] is True and set(doc['cases']) == set(specs), 'Exact406 returned metric bank missing')
    for rid in sorted(specs):
        spec = specs[rid]; source = spec['row']
        pngs['original0', rid] = rgb8(e.dep(source['original_png_path']))
        pngs['parent50', rid] = rgb8(e.dep(source['parent_png_path']))
        for bank, at in [('candidate0', '0000'), ('candidate50', '0050')]:
            pngs[bank, rid] = rgb8(e.file(RUN + '/snapshot_' + at + '/' + rid + '/native.png'))
    sheets = render(out, specs, pngs)
    print(json.dumps({'all34_sheets_ready': str(out / 'visual_sheets/INDEX.md'), 'audit_still_running': True}), flush=True)
    for number, rid in enumerate(sorted(specs), 1):
        spec = specs[rid]; source = spec['row']; mask = spec['observed']; native = spec['evidence'] == 'native_unpaired'
        raws = {'original0': raw256(e.dep(source['original_raw_path'])), 'parent50': raw256(e.dep(source['parent_raw_path']))}
        for bank, at in [('candidate0', '0000'), ('candidate50', '0050')]:
            prefix = RUN + '/snapshot_' + at + '/' + rid + '/'
            evidence = e.json(prefix + 'evidence.json')
            require(evidence == documents[at]['cases'][rid] and evidence['sourcebindings'] == spec['sourcebindings']
                and all(evidence[k] == spec[k] for k in ('id', 'source', 'profile', 'evidence'))
                and evidence['observed_pixels'] == int(mask.sum()) and evidence['checkpoint_updates'] == int(at), 'Returned evidence/source/metric join differs')
            for key, filename in [('raw', 'raw.npy'), ('png', 'native.png')]:
                name = prefix + filename
                require(evidence[key] == {'path': name, 'sha256': e.hashes[name], 'bytes': e.sizes[name]}, 'Saved output evidence pin differs')
            raws[bank] = raw256(e.file(prefix + 'raw.npy'))
        parity = np.array_equal(raws['candidate0'], raws['parent50']) and np.array_equal(pngs['candidate0', rid], pngs['parent50', rid])
        require(parity, 'Snapshot0 must reproduce full-coarse parent RAW/PNG exactly: ' + rid)
        for bank in BANKS:
            raw, png = raws[bank], pngs[bank, rid]
            conversion = np.array_equal(png, expected_png(raw, spec['input'], mask))
            padding = np.array_equal(png[~mask], spec['input'][~mask])
            require(conversion and padding, 'RAW-to-PNG/protected observed-support pixels differ: ' + bank + '/' + rid)
            arrays_checked.append({'bank': bank, 'id': rid, 'RAW_PNG_exact': True, 'PNG_padding_exact': True,
                'RAW_SHA256': pixel_sha(raw), 'PNG_RGB8_SHA256': pixel_sha(png),
                'candidate0_exact_parent': True if bank == 'candidate0' else None})
            if bank == 'candidate0':
                measured = rows['parent50'][rid]['metrics']
                desc = rows['parent50'][rid]['descriptors']
            else:
                measured = {view: metrics(image, spec['target'], mask, native)
                    for view, image in [('raw', raw), ('native_png', floats(png))]}
                desc = {view: descriptors(image, spec['input'], mask)
                    for view, image in [('raw', raw), ('native_png', floats(png))]}
            rows[bank][rid] = {k: spec[k] for k in ('id', 'source', 'profile', 'evidence')}
            rows[bank][rid].update(metrics=measured, descriptors=desc)
            if bank in ('candidate0', 'candidate50'):
                saved = documents['0000' if bank == 'candidate0' else '0050']['cases'][rid]['metrics']
                for view in VIEWS:
                    for metric in WORKER_METRICS:
                        if not near(measured[view][metric], saved[view][metric]):
                            metric_mismatches.append(bank + '/' + rid + '/' + view + '/' + metric)
        if number % 50 == 0 or number == 406:
            print(json.dumps({'independent_DEV_cases': number, 'of': 406, 'seconds': time.monotonic() - started}), flush=True)
    guards = preservation(rows['candidate50'], rows['original0'])
    initial_guards = preservation(rows['candidate0'], rows['original0'])
    dark = dark_retention(rows['candidate50'], rows['parent50'])
    gate_mismatches = compare_tree(guards, e.json(RUN + '/preservation_guards.json'), 'normal')
    gate_mismatches += compare_tree(dark, e.json(RUN + '/dark_retention.json'), 'dark')
    gate_mismatches += compare_tree(dict(initial_guards, initial_known_failed_parent_recorded_without_abort=True),
        e.json(RUN + '/snapshot_0000/preservation_guards.json'), 'initial')
    summaries = []
    for bank in BANKS:
        groups = defaultdict(list)
        for row in rows[bank].values():
            for view in VIEWS:
                groups[row['source'], row['evidence'], row['profile'], view].append(row['metrics'][view])
        for key, cases in sorted(groups.items()):
            result = dict(bank=bank, **dict(zip(('source', 'evidence', 'profile', 'view'), key)), cases=len(cases))
            for metric in cases[0]:
                values = [case[metric] for case in cases if case[metric] is not None]
                result['mean_' + metric] = sum(values) / len(values) if values else None
            summaries.append(result)
    deltas = []
    for baseline in ('original0', 'parent50'):
        grouped = defaultdict(list)
        for rid, spec in specs.items():
            for view in VIEWS:
                grouped[spec['source'], spec['evidence'], spec['profile'], view].append(rid)
        for key, ids in sorted(grouped.items()):
            result = dict(comparison='candidate50_minus_' + baseline,
                **dict(zip(('source', 'evidence', 'profile', 'view'), key)), cases=len(ids))
            view = key[-1]
            for metric in rows['candidate50'][ids[0]]['metrics'][view]:
                values = [(rid, rows['candidate50'][rid]['metrics'][view][metric], rows[baseline][rid]['metrics'][view][metric]) for rid in ids]
                valid = [(rid, a - b) for rid, a, b in values if a is not None and b is not None]
                result['mean_delta_' + metric] = sum(v for _, v in valid) / len(valid) if valid else None
                if metric in ('mae', 'ssim', 'luminance_bias', 'coarse_rgb_mae'):
                    result[metric + '_lower'] = sum(v < 0 for _, v in valid)
                    result[metric + '_higher'] = sum(v > 0 for _, v in valid)
                    result[metric + '_equal'] = sum(v == 0 for _, v in valid)
            deltas.append(result)
    write(out / 'metrics.json', {'complete': True, 'cases': rows, 'groups': summaries,
        'candidate0_metrics_reused_only_after_bit_exact_parent_RAW_PNG_equality': True})
    write(out / 'conversion_and_parity_checks.json', {'complete': True, 'checks': arrays_checked,
        'saved_metric_mismatches': metric_mismatches, 'saved_gate_mismatches': gate_mismatches,
        'scalar_comparison_atol': 1e-10, 'image_array_equality_atol': 0.0})
    write(out / 'guards.json', {'complete': True, 'initial_preservation': initial_guards,
        'candidate50_preservation': guards, 'dark_retention': dark})
    write(out / 'source_profile_comparisons.json', {'complete': True, 'comparisons': deltas,
        'native_clean_reference_metrics_null': True, 'paired_sources_reported_separately': True})
    runtime = e.json(RUN + '/runtime.json')
    failure = e.json(RUN + '/failure.json')
    outputs = ['input_audit.json', 'metrics.json', 'conversion_and_parity_checks.json', 'guards.json',
        'source_profile_comparisons.json', 'sheet_cells.json', 'visual_manifest.json']
    result = {'complete': True, 'audit_pass': not metric_mismatches and not gate_mismatches,
        'archive_sha256': ARCHIVE_SHA, 'protocol_sha256': PROTOCOL_SHA, 'import_receipt_sha256': sha(e.receipt_path),
        'audit_source_sha256': sha(__file__), 'outputs_sha256': {name: sha(out / name) for name in outputs},
        'DEV_cases': 406, 'paired_synthetic': 358, 'native_unpaired': 48, 'new_RAW_PNG_pairs_checked': 812,
        'retained_original_parent_RAW_PNG_pairs_checked': 812, 'total_RAW_PNG_pairs_checked': 1624,
        'exact_snapshot0_parent_cases': 406, 'new_candidate50_cases': 406,
        'full_worker_scalar_metrics_reproduced': not metric_mismatches,
        'full_worker_normal_and_dark_gates_reproduced': not gate_mismatches,
        'saved_metric_mismatches': metric_mismatches, 'saved_gate_mismatches': gate_mismatches,
        'preservation_pass': guards['pass'], 'dark_retention_pass': dark['pass'],
        'worker_quality_requirements_pass': guards['pass'] and dark['pass'],
        'worker_recorded_completed_training_updates': failure['completed_training_updates'],
        'worker_recorded_stop': failure['reason'],
        'source_profile_summary': summaries, 'source_profile_comparisons': deltas,
        'CPU_runtime': {'Python': sys.version, 'NumPy': np.__version__, 'Pillow': pillow_version,
            'SciPy': scipy.__version__, 'scikit_image': skimage.__version__},
        'VM_runtime': {k: runtime[k] for k in ('python', 'numpy', 'Pillow', 'torch', 'device')},
        'returned_files_freshly_hashed': len(e.checked), 'parent_files_freshly_hashed': len(e.dep_checked),
        'returned_bindings_checked': sorted(e.checked), 'parent_bindings_checked': e.dep_checked,
        'visual_manifest_sha256': sha(out / 'visual_manifest.json'), 'visual_sheets': sheets,
        'root_personal_visual_review_pending': True, 'user_acceptance_established': False,
        'FINAL_used': False, 'native_sources_not_ethnicity_or_Zamboanga_claim': True,
        'observed_support_is_not_completion_mask': True, 'local_neural_gradient_optimizer_VM_calls': 0,
        'actual_new_local_training_updates': 0, 'app_changed': False, 'model_qualified': False,
        'seconds': time.monotonic() - started}
    write(out / 'audit.json', result)
    lines = ['# R2 independent saved-pixel audit', '',
        'The VM committed50 Adam updates and retained the trained state. It stopped on restoration preservation after evaluation. The audit can pass while the model remains unqualified.', '',
        'All812 new RAW/PNG pairs and812 retained original/parent pairs are checked. Snapshot0 reproduces the full-coarse parent exactly in all406 cases. Native48 cases have no aligned clean target; their clean-reference metrics remain null.', '',
        '| Control | View | Metric | Mean delta | Worst delta | Cases >0.03 | Pass |',
        '|---|---|---|---:|---:|---:|---|']
    for control in ('clear', 'normal_blur'):
        for view, values in guards[control]['views'].items():
            for metric, row in values.items():
                lines.append('|%s|%s|%s|%.9f|%.9f|%d|%s|' % (control, view, metric, row['mean_delta'], row['worst_delta'], row['cases_over_0_03'], row['pass']))
    lines += ['', 'Mean0.01 and worst-case0.03 preservation limits are unchanged. Lower MAE/luminance bias/coarse RGB error is better; higher SSIM is better.', '',
        '| Dark profile | Parent PNG MAE | New PNG MAE | Parent SSIM | New SSIM | Retention pass |', '|---|---:|---:|---:|---:|---|']
    for profile, row in dark['profiles'].items():
        lines.append('|%s|%.9f|%.9f|%.9f|%.9f|%s|' % (profile, row['base']['mae'], row['candidate']['mae'], row['base']['ssim'], row['candidate']['ssim'], row['pass']))
    lines += ['', 'All34 exact-pixel sheets are prepared for root review. Audit completion, training completion, paired metric improvements and user-visible or native-CCTV usefulness are separate claims. This audit does not establish user acceptance, final evaluation, identity recovery or app promotion.', '']
    with (out / 'AUDIT.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    require(not metric_mismatches and not gate_mismatches, 'Saved worker metrics or gate evidence differs from independent pixel recomputation')
    require(not any(n in sys.modules for n in ('torch', 'tensorflow', 'jax')), 'Neural framework imported during pixel audit')
    return {'complete': True, 'audit_pass': True, 'preservation_pass': guards['pass'], 'dark_retention_pass': dark['pass'],
        'audit_sha256': sha(out / 'audit.json'), 'sheets': 34, 'seconds': time.monotonic() - started}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-receipt', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.import_receipt, args.out)), flush=True)
