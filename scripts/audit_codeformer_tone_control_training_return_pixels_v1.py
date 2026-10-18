"""CPU-only independent saved-pixel audit; never executes returned Python or Torch.

All panels preserve exact256 returned/source RGB8 values. Clean metrics for
native unpaired cases remain null; descriptors are not identity/quality proof.
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
PREP = ROOT / 'outputs/codeformer_tone_control_training_v1_preparation'
REVIEW = ROOT / 'outputs/codeformer_tone_control_training_v1_return_review'
ARCHIVE_SHA = '6a4386f84ebc27d6201ad188bfb19dc36bdb0a6fbab3c6f77c0662ecd4e547bf'
RUN = 'outputs/tone_control_training_v1'
OLD_RUN = 'outputs/adain_gradient_isolation_v1'
BANKS = ('original0', 'head0000', 'head0050')
VIEWS = ('raw', 'native_png')
W = np.asarray([.2126, .7152, .0722], np.float64)
PROFILES = ('clear', 'light', 'blur', 'jpeg', 'lowlight', 'compound')
WORKER_METRICS = ('mae', 'mse', 'psnr', 'ssim', 'luminance_bias', 'coarse_rgb_mae')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def pixels_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    def unique(rows):
        d = {}
        for k, v in rows:
            require(k not in d, 'Duplicate JSON key')
            d[k] = v
        return d
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def checked(root, name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and p.as_posix() == name and not p.is_absolute() and
            all(s not in ('', '.', '..') for s in p.parts) and '\\' not in name and ':' not in name,
            'Unsafe evidence path')
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*p.parts)
    require(path.resolve(strict=True).is_relative_to(root), 'Evidence escapes root')
    for item in (path, *path.parents):
        if item == root.parent:
            break
        st = item.lstat()
        require(not stat.S_ISLNK(st.st_mode) and not getattr(st, 'st_file_attributes', 0) & 1024,
                'Linked/reparse evidence path')
    require(stat.S_ISREG(path.lstat().st_mode), 'Regular evidence file required')
    return path


def rgb8(path):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (256, 256), 'Exact RGB256 PNG required')
        return np.asarray(image).copy()


def mask256(path):
    with Image.open(path) as image:
        require(image.size == (256, 256), 'Mask geometry differs')
        a = np.asarray(image.convert('L')).copy()
    require(np.isin(a, (0, 255)).all() and a.any(), 'Binary nonempty mask required')
    return a > 0


def floats(a):
    return np.ascontiguousarray(a.astype(np.float32) / np.float32(255))


def raw256(path):
    with Path(path).open('rb') as stream:
        a = np.load(stream, allow_pickle=False)
        require(stream.read(1) == b'', 'Trailing NPY bytes')
    require(a.shape == (256, 256, 3) and a.dtype == np.float32 and a.flags.c_contiguous and
            np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1, 'Bounded contiguous RAW256 required')
    return a


def expected_png(raw, input8, mask):
    a = np.floor(raw * np.float32(255)).astype(np.uint8)
    a[~mask] = input8[~mask]
    return a


def independent_features(input8, mask):
    a = floats(input8).astype(np.float64)
    out = np.zeros((4, 8, 8), np.float64)
    for iy in range(8):
        for ix in range(8):
            support = mask[iy*32:(iy+1)*32, ix*32:(ix+1)*32]
            out[3, iy, ix] = support.sum() / 1024
            if support.any():
                out[:3, iy, ix] = a[iy*32:(iy+1)*32, ix*32:(ix+1)*32][support].mean(axis=0)
    return out.reshape(256).astype(np.float32)


def descriptors(image, input8, mask):
    a = image.astype(np.float64)
    y = a @ W
    valid = binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)
    gx = (y[1:-1, 2:] - y[1:-1, :-2]) / 2
    gy = (y[2:, 1:-1] - y[:-2, 1:-1]) / 2
    edge = np.hypot(gx, gy)[valid[1:-1, 1:-1]]
    return {'observed_mean_luminance': float(y[mask].mean()),
        'observed_luminance_std': float(y[mask].std()),
        'mean_R_minus_G': float((a[..., 0] - a[..., 1])[mask].mean()),
        'mean_B_minus_G': float((a[..., 2] - a[..., 1])[mask].mean()),
        'central_gradient_magnitude_mean': float(edge.mean()) if edge.size else None,
        'saturated_scalar_fraction': float(((a[mask] == 0) | (a[mask] == 1)).mean()),
        'MAE_from_input_not_clean_target': float(np.abs(a[mask] - floats(input8)[mask]).mean()),
        'descriptors_are_quality_or_identity_proof': False}


def metrics(image, target, mask, native):
    if native:
        require(target is None, 'Invented native clean target')
        return dict.fromkeys((*WORKER_METRICS, 'masked_paired_SSIM', 'signed_luminance_bias'))
    require(target is not None and target.shape == image.shape, 'Paired target missing')
    a, b = image.astype(np.float64), target.astype(np.float64)
    diff = a[mask] - b[mask]
    mse = float((diff * diff).mean())
    r = {'mae': float(np.abs(diff).mean()), 'mse': mse,
         'psnr': None if mse == 0 else -10 * math.log10(mse), 'ssim': None,
         'luminance_bias': float(abs((diff @ W).mean())),
         'signed_luminance_bias': float((diff @ W).mean()), 'coarse_rgb_mae': None,
         'masked_paired_SSIM': None}
    valid = binary_erosion(mask, structure=np.ones((7, 7), bool), border_value=0)
    if valid.any():
        score, field = structural_similarity(a, b, data_range=1., channel_axis=2,
                                             win_size=7, full=True)
        r['masked_paired_SSIM'] = float(field[valid].mean(dtype=np.float64))
        if mask.all():
            r['ssim'] = float(score)
    if mask.all():
        pool = lambda x: x.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3), dtype=np.float64)
        r['coarse_rgb_mae'] = float(np.abs(pool(a) - pool(b)).mean())
    return r


class Evidence:
    def __init__(self, import_receipt):
        self.receipt_path = Path(import_receipt)
        self.receipt = read(self.receipt_path)
        r = self.receipt
        require(r.get('complete') is True and r['archive_sha256'] == ARCHIVE_SHA,
                'Hash-closed exact root import required before pixel audit')
        self.root = Path(r['import_root']).resolve(strict=True)
        self.hashes, self.sizes = r['files_sha256'], r['files_bytes']
        require(set(self.hashes) == set(self.sizes), 'Import binding namespaces differ')
        self.checked = set()
        self.dep_checked = {}
        self.p = self.json('protocol.json')
        require(self.hashes['protocol.json'] == r['protocol_sha256'], 'Protocol/import binding differs')
        self.stage = read(PREP / 'stage_v1.json')
        require(self.stage['complete'] is True and self.stage['protocol_sha256'] == r['protocol_sha256'] and
                self.stage['dependencies'] == self.p['dependencies'] and
                self.stage['retained_output_dependencies'] == self.p['retained_output_dependencies'],
                'Local overlay resolver differs from actual returned protocol')
        manifest_name = RUN + '/export_manifest.json'
        self.manifest = self.json(manifest_name)
        require(self.manifest['protocol_sha256'] == r['protocol_sha256'] and
                set(self.manifest['files_sha256']) | {manifest_name} == set(self.hashes) and
                all(self.hashes[n] == h for n, h in self.manifest['files_sha256'].items()) and
                all(self.sizes[n] == b for n, b in self.manifest['files_bytes'].items()),
                'Exact import/export manifest closure differs')
        self.pins = dict(self.p['dependencies'], **self.p['retained_output_dependencies'])

    def file(self, name):
        require(name in self.hashes, 'Unbound returned evidence: ' + name)
        path = checked(self.root, name)
        if name not in self.checked:
            require(path.stat().st_size == self.sizes[name] and sha(path) == self.hashes[name],
                    'Returned bytes differ: ' + name)
            self.checked.add(name)
        return path

    def json(self, name):
        return read(self.file(name))

    def dep(self, name, digest=None):
        require(name in self.pins and (digest is None or digest == self.pins[name]['sha256']),
                'Unbound dependency')
        path = Path(self.stage['local_dependency_resolution'][name])
        require(path.resolve(strict=True).is_relative_to(ROOT), 'Dependency escapes workspace')
        checked(ROOT, path.relative_to(ROOT).as_posix())
        if name not in self.dep_checked:
            pin = self.pins[name]
            require(path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
                    'Dependency bytes differ: ' + name)
            self.dep_checked[name] = dict(pin, local_path=str(path))
        return path


def case_specs(e):
    p = e.p
    refs = read(e.dep(p['data_manifest']))['references']
    require(Counter(r['historical_role'] for r in refs) == {'train': 391, 'validation': 53}, 'Split census differs')
    train_hashes = {r['source_sha256'] for r in refs if r['historical_role'] == 'train'}
    valid = {r['id']: r for r in refs if r['historical_role'] == 'validation'}
    require(not train_hashes & {r['source_sha256'] for r in valid.values()}, 'TRAIN/DEV source overlap')
    saved = read(e.dep(p['validation_saved']['manifest'], p['validation_saved']['sha256']))
    old = read(e.dep(saved['retained_original_manifest_path'], saved['retained_original_manifest_sha256']))
    require(saved['case_count'] == len(saved['items']) == 318 and not saved['locally_regenerated_development_pixels'],
            '318 exact saved DEV cases required')
    require({r['id'] for r in saved['items']} == {'validation_' + rid + '_' + k for rid in valid for k in PROFILES},
            'Exact DEV profile/identity namespace differs')
    specs, joins = {}, []
    for r in saved['items']:
        pair = read(e.dep(r['pair_record_path'], r['pair_record_sha256']))
        ref = valid[pair['reference_id']]
        require(pair['role'] == 'validation' and pair['source_path'] == ref['source_path'] and
                pair['source_sha256'] == ref['source_sha256'] and
                pair['reference_manifest_sha256'] == p['dependencies'][p['data_manifest']]['sha256'] and
                pair['clean512_rgb8_sha256'] == ref['proposed_512_rgb_sha256'] and
                r['id'] == 'validation_' + ref['id'] + '_' + pair['profile'] and
                r['target_path'] == ref['existing_target_path'] and r['target_sha256'] == ref['existing_target_sha256'] and
                old['cases'][r['id']]['sha256'] == r['input_sha256'], 'DEV source/target/record joins differ')
        e.dep(ref['source_path'], ref['source_sha256'])
        inp = rgb8(e.dep(r['input_path'], r['input_sha256']))
        require(pixels_sha(inp) == pair['input256_rgb8_sha256'] and
                pixels_sha(floats(inp)) == pair['input256_float32_sha256'], 'DEV retained input fingerprints differ')
        target = rgb8(e.dep(r['target_path'], r['target_sha256']))
        specs[r['id']] = {'id': r['id'], 'source': 'FFHQ validation', 'profile': pair['profile'],
            'evidence': 'paired_synthetic', 'input': inp, 'target8': target, 'target': floats(target),
            'observed': np.ones((256, 256), bool), 'sourcebindings': r}
        joins.append({'id': r['id'], 'input_SHA256': r['input_sha256'], 'target_SHA256': r['target_sha256'],
                      'source_SHA256': ref['source_sha256'], 'input_float32_exact': True})
    compare = read(e.dep(p['comparison_cases']))['cases']
    require(len(compare) == 88 and Counter(r['evidence'] for r in compare) ==
            {'paired_synthetic': 40, 'native_unpaired': 48}, 'Comparison role census differs')
    for r in compare:
        native = r['evidence'] == 'native_unpaired'
        require(not native or not r.get('target_path') and not r.get('target_sha256'), 'Invented native target')
        inp = rgb8(e.dep(r['prepared_path'], r['prepared_sha256']))
        mask = mask256(e.dep(r['observed_path'], r['observed_sha256']))
        target = None if native else rgb8(e.dep(r['target_path'], r['target_sha256']))
        rid = 'comparison_' + r['id']
        require(rid not in specs and not r['id'].lower().startswith('final'), 'Duplicate/FINAL comparison')
        specs[rid] = {'id': rid, 'source': r['source'], 'profile': r['profile'], 'evidence': r['evidence'],
            'input': inp, 'target8': target, 'target': None if native else floats(target),
            'observed': mask, 'sourcebindings': r}
        joins.append({'id': rid, 'input_SHA256': r['prepared_sha256'], 'mask_SHA256': r['observed_sha256'],
                      'target_SHA256': r.get('target_sha256'), 'native_target_null': native})
    require(len(specs) == 406 and sum(r['profile'] == 'clear' and r['evidence'] == 'paired_synthetic'
        for r in specs.values()) == 61, 'Full406/clear61 namespace differs')
    return specs, joins


def near(a, b, atol=1e-10):
    return a is None and b is None or a is not None and b is not None and abs(a - b) <= atol


def compare_tree(a, b, prefix='', errors=None):
    errors = [] if errors is None else errors
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            errors.append(prefix + ': keys differ')
        for k in set(a) & set(b):
            compare_tree(a[k], b[k], prefix + '/' + str(k), errors)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): errors.append(prefix + ': length differs')
        for i, (x, y) in enumerate(zip(a, b)): compare_tree(x, y, prefix + '/' + str(i), errors)
    elif type(a) in (int, float) and type(b) in (int, float):
        if not near(float(a), float(b)): errors.append(prefix + ': numeric differs')
    elif a != b:
        errors.append(prefix + ': value differs')
    return errors


def preservation(rows, baseline):
    reports = {}
    for label, ids, names in [
        ('clear', [k for k, r in rows.items() if r['evidence'] == 'paired_synthetic' and r['profile'] == 'clear'], ['mae']),
        ('normal_blur', [k for k, r in rows.items() if r['source'] == 'FFHQ validation' and r['profile'] == 'blur'],
         ['luminance_bias', 'coarse_rgb_mae'])]:
        require(len(ids) == (61 if label == 'clear' else 53), 'Guard identity census differs')
        report = {'pass': True, 'cases': len(ids), 'mean_limit': .01, 'worst_limit': .03, 'views': {}}
        for view in VIEWS:
            report['views'][view] = {}
            for name in names:
                delta = {rid: rows[rid]['metrics'][view][name] - baseline[rid]['metrics'][view][name] for rid in sorted(ids)}
                avg, worst = sum(delta.values()) / len(ids), max(delta.values())
                passed = avg <= .01 and worst <= .03
                report['views'][view][name] = {'mean_delta': avg, 'worst_delta': worst, 'deltas': delta,
                    'cases_over_0_03': sum(v > .03 for v in delta.values()), 'pass': passed}
                report['pass'] &= passed
        reports[label] = report
    reports['pass'] = all(reports[k]['pass'] for k in ('clear', 'normal_blur'))
    return reports


def dark_retention(rows, baseline):
    report = {'pass': True, 'profiles': {}, 'prospective_additional_gate': True}
    for profile in ('lowlight', 'compound'):
        ids = sorted(k for k, r in rows.items() if r['source'] == 'FFHQ validation' and r['profile'] == profile)
        require(len(ids) == 53, 'Dark53 identity census differs')
        means = {name: {m: sum(bank[k]['metrics']['native_png'][m] for k in ids) / 53
                         for m in ('mae', 'ssim')} for name, bank in [('candidate', rows), ('base', baseline)]}
        passed = means['candidate']['mae'] <= means['base']['mae'] and means['candidate']['ssim'] >= means['base']['ssim']
        report['profiles'][profile] = dict(means, cases=ids, pass_=passed)
        report['profiles'][profile]['pass'] = report['profiles'][profile].pop('pass_')
        report['pass'] &= passed
    return report


def render(out, specs, pngs, stress):
    folder = out / 'visual_sheets'; folder.mkdir()
    cells, sheets = [], []
    ids = sorted(specs)
    groups = [(f'sheet_{i//12+1:02d}', ids[i:i+12], False) for i in range(0, len(ids), 12)]
    groups.append(('known_TRAIN_stress', list(stress), True))
    for name, subset, is_stress in groups:
        canvas = Image.new('RGB', (1280, 55 + len(subset) * 280), 'white'); draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), ('Known TRAIN stress; not generalization' if is_stress else 'All406 DEV saved pixels') +
            ' | exact256 panels; no resize/enhancement', fill='black')
        for col, title in enumerate(('input', 'clean target / native blank', 'untouched original', 'frozen full-coarse base', 'trained head50')):
            draw.text((col * 256 + 4, 25), title, fill='black')
        for rownum, rid in enumerate(subset):
            spec = stress[rid] if is_stress else specs[rid]
            values = [spec['input'], spec['target8'], *[pngs[(bank, rid)] for bank in BANKS]]
            draw.text((4, 55 + rownum * 280), rid + ' | ' + spec['evidence'] + ' | ' + spec['profile'], fill='black')
            for col, value in enumerate(values):
                y = 79 + rownum * 280
                if value is None:
                    draw.rectangle((col * 256, y, (col + 1) * 256 - 1, y + 255), fill=(235, 235, 235))
                    draw.text((col * 256 + 10, y + 120), 'No native clean target', fill='black')
                    continue
                require(value.shape == (256, 256, 3) and value.dtype == np.uint8, 'Exact256 sheet source required')
                canvas.paste(Image.fromarray(value), (col * 256, y))
                cells.append({'sheet': name + '.png', 'case': rid, 'column': col, 'x': col * 256, 'y': y,
                              'RGB8_SHA256': pixels_sha(value), 'width': 256, 'height': 256})
        path = folder / (name + '.png'); canvas.save(path)
        sheets.append({'path': path.relative_to(out).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size,
                       'case_ids': subset, 'case_count': len(subset), 'known_TRAIN': is_stress})
        with Image.open(path) as image:
            a = np.asarray(image)
            for cell in [c for c in cells if c['sheet'] == path.name]:
                crop = a[cell['y']:cell['y']+256, cell['x']:cell['x']+256]
                require(pixels_sha(crop) == cell['RGB8_SHA256'], 'Saved sheet cell changed')
    write(out / 'sheet_cells.json', cells)
    write(out / 'visual_manifest.json', {'complete': True, 'sheets': sheets, 'DEV_sheets': 34, 'TRAIN_stress_sheets': 1,
        'all406_DEV_rows_once': True, 'exact256_cells_readback': len(cells), 'altered_evidence_images': 0,
        'personal_visual_review_performed': False, 'root_must_view_every_sheet': True})
    with (folder / 'INDEX.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('# Saved-pixel comparison sheets\n\n34 DEV sheets cover all 406 cases once. One additional sheet covers the two known TRAIN stress cases. Columns: input, clean target (blank for native), untouched original, frozen full-coarse base, trained head50. Every evidence panel remains exactly 256 by 256 pixels.\n\n')
        stream.write('\n'.join('- [' + Path(s['path']).name + '](' + Path(s['path']).name + ')' for s in sheets) + '\n')
    return sheets


def audit(import_receipt, out):
    require(not any(k in sys.modules for k in ('torch', 'tensorflow', 'jax')), 'Neural framework already imported')
    e = Evidence(import_receipt)
    runtime = e.json(RUN + '/runtime.json')
    require(not out.exists(), 'Preserve prior pixel audit; choose a new version')
    out.mkdir(parents=True)
    start = time.monotonic(); specs, joins = case_specs(e)
    write(out / 'input_audit.json', {'complete': True, 'cases': joins, 'DEV': 406, 'paired': 358, 'native': 48,
        'observed_mask_policy': 'Observed image/letterbox support from prepared geometry; not a covering or completion mask.'})
    require(e.p['design']['evaluation']['clear']['mean_stop'] == .01 and
            e.p['design']['evaluation']['clear']['worst_per_case_stop'] == .03 and
            e.p['design']['evaluation']['normal_blur']['mean_stop'] == .01 and
            e.p['design']['evaluation']['normal_blur']['worst_per_case_stop'] == .03, 'Quality gate changed')
    all_rows, pngs, checks, saved_mismatches = {}, {}, [], []
    # Render hash-bound returned PNGs first so root can inspect all rows while
    # the independent CPU RAW/SSIM calculations continue. Rendering is not a pass.
    for bank in BANKS:
        for rid in sorted(specs):
            pngs[bank, rid] = rgb8(e.file(RUN + '/' + bank + '/' + rid + '/native.png'))
    stress_specs = {}
    parity = read(e.dep(e.p['training_parity']['path'], e.p['training_parity']['sha256']))
    saved_stress = e.json(RUN + '/known_overlap_stress.json')
    require(saved_stress['complete'] is True and [r['id'] for r in saved_stress['cases']] ==
            e.p['design']['evaluation']['known_overlap_stress']['cases'], 'Exact2known TRAIN stress IDs/order required')
    for saved in saved_stress['cases']:
        rid = saved['id']; u, slot = int(rid[7:11]), int(rid[-1])
        entry = next(r for r in parity['cases'] if r['update'] == u and r['slot'] == slot)
        inp = rgb8(e.file(RUN + '/TRAIN/' + rid + '/input.png'))
        require(np.array_equal(inp, rgb8(e.dep(entry['input_path'], entry['input_sha256']))) and
                pixels_sha(inp) == entry['input256_rgb8_sha256'], 'Stress TRAIN input differs')
        with Image.open(e.dep(entry['clean_source_path'], entry['source_sha256'])) as image:
            require(image.mode == 'RGB' and image.size == (1024, 1024), 'Approved stress source differs')
            target8 = np.asarray(image.resize((256, 256), Image.Resampling.LANCZOS)).copy()
        mask = np.ones((256, 256), bool)
        for bank, filename in [('original0', 'original_raw.npy'), ('head0000', 'base_raw.npy')]:
            a = raw256(e.file(RUN + '/TRAIN/' + rid + '/' + filename))
            pngs[bank, rid] = expected_png(a, inp, mask)
        pngs['head0050', rid] = rgb8(e.file(RUN + '/known_stress/' + rid + '/native.png'))
        stress_specs[rid] = {'input': inp, 'target8': target8, 'evidence': 'known TRAIN; paired',
            'profile': entry['profile'], 'source_entry': entry,
            'target_CPU_Pillow': pillow_version, 'target_pixels_SHA256': pixels_sha(target8)}
    sheets = render(out, specs, pngs, stress_specs)
    print(json.dumps({'all35_sheets_ready': str(out / 'visual_sheets/INDEX.md'),
                      'audit_still_running': True}), flush=True)
    base0 = {}
    for bank in BANKS:
        document = e.json(RUN + '/' + bank + '_metrics.json')
        require(document['complete'] is True and set(document['cases']) == set(specs), 'Complete406 bank required')
        bank_prefix = RUN + '/' + bank + '/'
        expected = {bank_prefix + rid + '/' + n for rid in specs for n in
            (('raw.npy', 'native.png', 'evidence.json') if bank == 'original0' else
             ('raw.npy', 'native.png', 'base_raw.npy', 'evidence.json'))}
        require({n for n in e.hashes if n.startswith(bank_prefix)} == expected, 'Exact bank member census differs')
        rows = {}
        for number, rid in enumerate(sorted(specs), 1):
            spec = specs[rid]; prefix = bank_prefix + rid + '/'
            evidence = e.json(prefix + 'evidence.json')
            require(evidence == document['cases'][rid] and all(evidence[k] == spec[k] for k in
                ('id', 'source', 'profile', 'evidence')) and evidence['sourcebindings'] == spec['sourcebindings'] and
                evidence['observed_pixels'] == int(spec['observed'].sum()), 'Evidence/input/source joins differ')
            for key, filename in [('raw', 'raw.npy'), ('png', 'native.png')]:
                binding = evidence[key]; name = prefix + filename
                require(binding == {'path': name, 'sha256': e.hashes[name], 'bytes': e.sizes[name]}, 'Artifact record binding differs')
            raw, png = raw256(e.file(prefix + 'raw.npy')), rgb8(e.file(prefix + 'native.png'))
            mask = spec['observed']; delta = evidence['delta']
            check = {'bank': bank, 'id': rid, 'RAW_PNG_exact': np.array_equal(png, expected_png(raw, spec['input'], mask)),
                'PNG_protected_exact': np.array_equal(png[~mask], spec['input'][~mask]),
                'input_features_exact': np.array_equal(np.asarray(evidence['features'], np.float32), independent_features(spec['input'], mask))}
            if bank != 'original0':
                base = raw256(e.file(prefix + 'base_raw.npy'))
                require(type(delta) in (int, float) and np.isfinite(delta) and -.25 <= delta <= .25, 'Finite bounded scalar delta required')
                reconstructed = np.clip(base + np.float32(delta), np.float32(0), np.float32(1))
                reconstructed[~mask] = floats(spec['input'])[~mask]
                check.update(RAW_composite_exact=np.array_equal(raw, reconstructed),
                    RAW_protected_exact=np.array_equal(raw[~mask], floats(spec['input'])[~mask]),
                    observed_clipping_fraction=float(((base[mask] + np.float32(delta) < 0) | (base[mask] + np.float32(delta) > 1)).mean()))
                if bank == 'head0000':
                    base0[rid] = base
                    oldraw = raw256(e.dep(OLD_RUN + '/snapshot_0050/' + rid + '/raw.npy'))
                    oldpng = rgb8(e.dep(OLD_RUN + '/snapshot_0050/' + rid + '/native.png'))
                    check.update(zero_delta_exact=delta == 0, base_historical_RAW_exact=np.array_equal(base, oldraw),
                                 zero_head_historical_PNG_exact=np.array_equal(png, oldpng))
                else:
                    check['base_repeat_RAW_exact'] = np.array_equal(base, base0[rid])
                if not near(check['observed_clipping_fraction'], evidence['clipped_scalar_fraction_observed']):
                    saved_mismatches.append(bank + '/' + rid + '/clipping_fraction')
            else:
                oldraw = raw256(e.dep(OLD_RUN + '/snapshot_0000/' + rid + '/raw.npy'))
                oldpng = rgb8(e.dep(OLD_RUN + '/snapshot_0000/' + rid + '/native.png'))
                check.update(original_historical_RAW_exact=np.array_equal(raw, oldraw),
                             original_historical_PNG_exact=np.array_equal(png, oldpng))
            independent_mean = float((raw.astype(np.float64) @ W)[mask].mean())
            if not near(independent_mean, evidence['mean_y'], atol=2e-7):
                saved_mismatches.append(bank + '/' + rid + '/mean_y_FP32_vs_CPU')
            row = {k: spec[k] for k in ('id', 'source', 'profile', 'evidence')}
            row.update(delta=delta, observed_pixels=int(mask.sum()),
                       mask_kind='observed geometry/letterbox support; not covering mask',
                       independent_RAW_mean_luminance=independent_mean)
            row['metrics'], row['descriptors'] = {}, {}
            for view, image in [('raw', raw), ('native_png', floats(png))]:
                result = metrics(image, spec['target'], mask, spec['evidence'] == 'native_unpaired')
                row['metrics'][view] = result
                row['descriptors'][view] = descriptors(image, spec['input'], mask)
                for name in WORKER_METRICS:
                    if not near(result[name], evidence['metrics'][view][name]):
                        saved_mismatches.append(bank + '/' + rid + '/' + view + '/' + name)
            checks.append(check); rows[rid] = row; pngs[bank, rid] = png
            if number % 50 == 0 or number == 406:
                print(json.dumps({'bank': bank, 'cases': number, 'seconds': time.monotonic() - start}), flush=True)
        all_rows[bank] = rows
        write(out / (bank + '_metrics.json'), {'complete': True, 'cases': rows})
    zero = preservation(all_rows['head0000'], all_rows['original0'])
    gate = preservation(all_rows['head0050'], all_rows['original0'])
    dark = dark_retention(all_rows['head0050'], all_rows['head0000'])
    gate_mismatch = []
    for name, actual in [('zero_head_guards.json', zero), ('preservation_guards.json', gate), ('dark_retention.json', dark)]:
        gate_mismatch += compare_tree(actual, e.json(RUN + '/' + name), name)
    stress_checks = []
    for saved in saved_stress['cases']:
        rid = saved['id']; u, slot = int(rid[7:11]), int(rid[-1])
        entry = next(r for r in parity['cases'] if r['update'] == u and r['slot'] == slot)
        role = 'normal' if slot in (1, 3) else 'dark'
        require(saved['role'] == role and type(saved['delta']) in (int, float) and -.25 <= saved['delta'] <= .25,
                'Known TRAIN stress role/delta differs')
        cache = e.json(RUN + '/TRAIN/' + rid + '/cache.json')
        require(cache['id'] == rid and cache['role'] == role and cache['update'] == u and cache['slot'] == slot,
                'Known stress cache identity differs')
        inp = rgb8(e.file(RUN + '/TRAIN/' + rid + '/input.png'))
        require(np.array_equal(inp, rgb8(e.dep(entry['input_path'], entry['input_sha256']))), 'Stress TRAIN input differs')
        require(pixels_sha(inp) == entry['input256_rgb8_sha256'], 'Stress input pixel digest differs')
        target8 = stress_specs[rid]['target8']
        mask = np.ones((256, 256), bool)
        raw = raw256(e.file(RUN + '/known_stress/' + rid + '/raw.npy'))
        png = rgb8(e.file(RUN + '/known_stress/' + rid + '/native.png'))
        base = raw256(e.file(RUN + '/TRAIN/' + rid + '/base_raw.npy'))
        original = raw256(e.file(RUN + '/TRAIN/' + rid + '/original_raw.npy'))
        before, after = {}, {}
        for view, a, b in [('raw', base, raw), ('native_png', floats(expected_png(base, inp, mask)), floats(png))]:
            before[view] = metrics(a, floats(target8), mask, False)
            after[view] = metrics(b, floats(target8), mask, False)
        policy_pass = after['native_png']['luminance_bias'] < before['native_png']['luminance_bias'] if saved['role'] == 'normal' else (
            after['native_png']['mae'] <= before['native_png']['mae'] and after['native_png']['ssim'] >= before['native_png']['ssim'])
        result = {'id': rid, 'role': saved['role'], 'before': before, 'after': after, 'pass': policy_pass,
            'delta': saved['delta'], 'target_CPU_Pillow': pillow_version,
            'target_RGB8_SHA256': pixels_sha(target8),
            'source_SHA256': entry['source_sha256'],
            'RAW_PNG_exact': np.array_equal(png, expected_png(raw, inp, mask)),
            'composite_exact': np.array_equal(raw, np.clip(base + np.float32(saved['delta']), 0, 1)),
            'independent_generalization': False}
        for section in ('before', 'after'):
            for view in VIEWS:
                for key in WORKER_METRICS:
                    if not near(result[section][view][key], saved[section][view][key]):
                        saved_mismatches.append('stress/' + rid + '/' + section + '/' + view + '/' + key)
        if policy_pass != saved['pass']: saved_mismatches.append('stress/' + rid + '/pass')
        stress_checks.append(result)
        pngs['original0', rid] = expected_png(original, inp, mask)
        pngs['head0000', rid] = expected_png(base, inp, mask)
        pngs['head0050', rid] = png
    write(out / 'known_TRAIN_stress.json', {'complete': True, 'cases': stress_checks, 'pass': all(r['pass'] for r in stress_checks)})
    write(out / 'conversion_and_parity_checks.json', {'complete': True, 'checks': checks,
        'saved_scalar_mismatches': saved_mismatches, 'gate_record_mismatches': gate_mismatch,
        'luminance_FP32_record_comparison_atol': 2e-7, 'metric_comparison_atol': 1e-10})
    write(out / 'guards.json', {'complete': True, 'zero_head': zero, 'trained_head': gate, 'dark_retention': dark})
    grouped = defaultdict(list)
    for bank, cases in all_rows.items():
        for row in cases.values():
            for view in VIEWS:
                grouped[(bank, row['source'], row['evidence'], row['profile'], view)].append(row['metrics'][view])
    summaries = []
    for key, rows in sorted(grouped.items()):
        result = dict(zip(('bank', 'source', 'evidence', 'profile', 'view'), key), cases=len(rows))
        for name in rows[0]:
            values = [r[name] for r in rows if r[name] is not None]
            result['mean_' + name] = sum(values) / len(values) if values else None
        summaries.append(result)
    write(out / 'source_profile_summaries.json', summaries)
    delta_groups = defaultdict(list)
    for bank, cases in all_rows.items():
        if bank == 'original0': continue
        for rid, row in cases.items():
            delta_groups[(bank, row['source'], row['evidence'], row['profile'])].append((rid, row['delta']))
    delta_summary = []
    for key, values in sorted(delta_groups.items()):
        ds = [v for _, v in values]
        delta_summary.append(dict(zip(('bank', 'source', 'evidence', 'profile'), key),
            cases=len(ds), minimum=min(ds), mean=sum(ds)/len(ds), maximum=max(ds),
            negative_count=sum(v < 0 for v in ds), positive_count=sum(v > 0 for v in ds),
            exact_zero_count=sum(v == 0 for v in ds), deltas=dict(values),
            input_only_controller_NO_role_at_inference=True))
    write(out / 'delta_distributions.json', {'complete': True, 'groups': delta_summary,
        'mask_policy': 'Observed support/letterbox geometry, not a covering mask',
        'negative_shift_is_not_itself_a_quality_or_identity_score': True})
    parity_failures = [{k: v for k, v in r.items() if type(v) is bool and not v} | {'bank': r['bank'], 'id': r['id']}
                       for r in checks if any(type(v) is bool and not v for v in r.values())]
    result = {'complete': True, 'audit_pass': not parity_failures and not saved_mismatches and not gate_mismatch,
        'archive_sha256': ARCHIVE_SHA, 'protocol_sha256': e.receipt['protocol_sha256'],
        'import_receipt_sha256': sha(e.receipt_path), 'audit_source_sha256': sha(__file__),
        'CPU_runtime': {'Python': sys.version, 'NumPy': np.__version__, 'Pillow': pillow_version,
                        'SciPy': scipy.__version__, 'scikit-image': skimage.__version__},
        'VM_runtime': {k: runtime[k] for k in ('python', 'numpy', 'Pillow', 'torch', 'device')},
        'DEV_cases': 406, 'banks': 3, 'RAW_PNG_pairs': 1218, 'base_RAW_arrays': 812,
        'known_TRAIN_stress_pairs': 2, 'clear_cases': 61, 'normal_blur_cases': 53,
        'original_historical_parity_cases': 406, 'zero_head_historical_parity_cases': 406,
        'base_fresh_repeat_comparisons': 406, 'parity_failures': parity_failures,
        'saved_scalar_mismatches': saved_mismatches, 'gate_record_mismatches': gate_mismatch,
        'preservation_pass': gate['pass'], 'dark_retention_pass': dark['pass'],
        'known_stress_pass': all(r['pass'] for r in stress_checks),
        'head_delta_groups': [{k:v for k,v in row.items() if k != 'deltas'} for row in delta_summary],
        'worker_overall_quality_requirements_pass': gate['pass'] and dark['pass'] and all(r['pass'] for r in stress_checks),
        'visual_sheets': sheets, 'root_personal_visual_review_pending': True,
        'returned_files_freshly_hashed': len(e.checked), 'dependency_files_freshly_hashed': len(e.dep_checked),
        'seconds': time.monotonic() - start, 'Torch_NN_autograd_optimizer_VM_calls': 0,
        'evidence_images_altered': 0, 'model_qualified': False,
        'limitations': ['Recorded controller deltas are composition inputs; no checkpoint head inference or live gradients replayed.',
            'CPU library versions differ from VM; all saved metric scalars are independently checked at1e-10, not assumed portable. Two known TRAIN clean targets use the bound1024source and declared direct256PillowLANCZOS policy, with CPUtarget hashes retained.',
            'Saved base repeats and historical parity do not establish native clean-reference quality or identity.',
            'The known stress examples were in TRAIN and do not establish independent generalization.',
            'All planned exact256 sheets still require root personal inspection and user acceptance.']}
    write(out / 'dependency_bindings.json', e.dep_checked)
    write(out / 'audit.json', result)
    with (out / 'SCIENTIFIC_READBACK.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('# Independent saved-pixel readback\n\n' +
            f"All 1,218 DEV RAW/PNG pairs and 812 base arrays were decoded and hashed. Independent audit pass: {result['audit_pass']}. " +
            f"Preservation gates: {gate['pass']}; dark retention: {dark['pass']}; known TRAIN stress: {result['known_stress_pass']}.\n\n" +
            'Quality failures remain failures even when export, state serialization, or this audit completes. ' +
            'Native unpaired clean metrics are null. This audit executes no head or restorer, and qualifies no model.\n\n' +
            '34 DEV sheets and one known TRAIN stress sheet retain exact 256-pixel panels. Root must inspect every sheet.\n')
    print(json.dumps({k: result[k] for k in ('complete', 'audit_pass', 'preservation_pass', 'dark_retention_pass', 'known_stress_pass', 'seconds')}), flush=True)
    return result


def self_test():
    a = np.full((256, 256, 3), .5, np.float32); inp = np.full(a.shape, 217, np.uint8)
    mask = np.ones((256, 256), bool); mask[:4] = False
    png = expected_png(a, inp, mask)
    require(np.all(png[mask] == 127) and np.array_equal(png[~mask], inp[~mask]), 'Floor/protected fixture failed')
    f = independent_features(inp, mask)
    require(f.shape == (256,) and f.dtype == np.float32 and f[192] == .875, 'Independent feature support fixture failed')
    x = metrics(a, np.full_like(a, .4), np.ones((256, 256), bool), False)
    require(abs(x['mae'] - .1) < 1e-7 and abs(x['coarse_rgb_mae'] - .1) < 1e-7, 'Metric fixture failed')
    require(all(v is None for v in metrics(a, None, mask, True).values()), 'Native clean-null fixture failed')
    require(not any(k in sys.modules for k in ('torch', 'tensorflow', 'jax')), 'Framework imported')
    print(json.dumps({'complete': True, 'CPU_fixtures': 4, 'Torch_NN_autograd_optimizer_VM_calls': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-receipt', type=Path, default=REVIEW / 'return_import_v1.json')
    parser.add_argument('--out', type=Path, default=REVIEW / 'pixels_v1')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        audit(args.import_receipt, args.out)


if __name__ == '__main__':
    main()
