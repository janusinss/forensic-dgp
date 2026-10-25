"""Independent saved-pixel audit of the sealed skip-feature adapter return.

No returned Python, checkpoint or neural model is imported or executed. Saved
RAW/PNG values, finite feature caches and metadata arithmetic are audited. The
current stop is preserved. Native/unpaired and filter TRAIN/DEV stay separate.
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
ARCHIVE_SHA = 'f445f5ffe3a97215046cb8e265eece2dd2d36961ac0b607948a52a29a5809c76'
PROTOCOL_SHA = '0d358ab29e0344c7243b6312cb1d04ca6cc0e0a1de270bca7e6b4040dd566c90'
RUN = 'outputs/skip_feature_adapter_v1'
STAGE = ROOT / 'outputs/cctv_codeformer_skip_feature_adapter_v1_vm'
IMPORT_ROOT = ROOT / 'outputs/codeformer_skip_feature_adapter_v1_return'
VIEWS = ('raw', 'native_png')
FIELDS = ('mae', 'mse', 'psnr', 'ssim', 'luminance_bias', 'signed_luminance_bias',
          'coarse_rgb_mae', 'adjacent_rgb_error')
LUMA = np.asarray([.2126, .7152, .0722], np.float64)
DEADLINE_SECONDS = 1500
SOURCE_COPY = {'path':'scripts/audit_codeformer_filter_fusion_training_return_pixels_v1.py',
 'sha256':'c7925a3117cfa9132f219bfab45b0cfa732515d4038da42e6c319aa6e6c47e2c',
 'policy':'copied named pure local functions; source not imported at runtime'}
SITES = {'32':256,'64':256,'128':128,'256':128}

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


class Evidence:
    """Only imported files bound by a completed independent physical receipt."""
    def __init__(self,receipt):
        self.receipt_path = Path(receipt).resolve(strict=True)
        require(self.receipt_path.is_relative_to(ROOT/'outputs'),'Local completed import receipt required')
        value = read(self.receipt_path)
        require(value.get('complete') is True and value.get('transport_pass') is True and
                value['archive']['sha256'] == ARCHIVE_SHA and value['protocol_sha256'] == PROTOCOL_SHA and
                value['returned_code_executed'] is False and value['checkpoint_or_array_deserialized'] is False,
                'Exact completed inert import required')
        self.root = Path(value['import_root']).resolve(strict=True)
        require(self.root == IMPORT_ROOT.resolve(strict=True),'Exact new returned root required')
        bindings = value['files']
        require(type(bindings) is dict and len(bindings) <= 20000,'Finite imported binding map required')
        self.hashes = {name:pin['sha256'] for name,pin in bindings.items()}
        self.sizes = {name:pin['bytes'] for name,pin in bindings.items()}
        self.checked = {}; self.start = time.monotonic(); self.import_receipt = value
        self.p = self.json('protocol.json')
        require(self.hashes['protocol.json'] == PROTOCOL_SHA and sha(STAGE/'protocol.json') == PROTOCOL_SHA,
                'Actual returned and original sealed protocol differ')
        manifest = self.json(RUN+'/export_manifest.json')
        require(manifest['protocol_sha256'] == PROTOCOL_SHA and
                set(manifest['files'])|{RUN+'/export_manifest.json'} == set(self.hashes),
                'Export/import full namespace differs')
        for name,pin in manifest['files'].items():
            require(pin == {'bytes':self.sizes[name],'sha256':self.hashes[name]},'Export binding differs: '+name)
        for name,pin in self.p['files'].items():
            require(pin == {'bytes':self.sizes[name],'sha256':self.hashes[name]},'Sealed static binding differs: '+name)

    def deadline(self):
        require(time.monotonic()-self.start <= DEADLINE_SECONDS,'Independent saved-pixel1500s bound exceeded')

    def file(self,name,pin=None):
        self.deadline(); require(name in self.hashes,'Unbound evidence file: '+name)
        if pin is not None:
            require(pin == {'path':name,'bytes':self.sizes[name],'sha256':self.hashes[name]},'Exact caller binding differs: '+name)
        path = checked(self.root,name); before = path.stat()
        if name not in self.checked:
            require(before.st_size == self.sizes[name] and sha(path) == self.hashes[name],'Physical evidence bytes differ: '+name)
            after = path.stat()
            require((before.st_size,before.st_mtime_ns,before.st_ino) ==
                    (after.st_size,after.st_mtime_ns,after.st_ino),'Evidence changed while hashing')
            self.checked[name] = {'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'inode':after.st_ino}
        else:
            old = self.checked[name]
            require((before.st_size,before.st_mtime_ns,before.st_ino) ==
                    (old['bytes'],old['mtime_ns'],old['inode']),'Previously hashed evidence changed')
        return path

    def json(self,name):
        return read(self.file(name))


def finite_array(e,pin,shape,*,bounded=False):
    path = e.file(pin['path'],pin)
    require(path.stat().st_size <= 40*1024**2,'Bounded finite array required')
    with path.open('rb') as source:
        array = np.load(source,allow_pickle=False)
        require(not source.read(1),'Trailing or concatenated NPY data forbidden')
    require(array.shape == shape and array.dtype == np.float32 and array.flags.c_contiguous and
            np.isfinite(array).all(),'Exact finite contiguous float32 array required')
    if bounded:
        require(0 <= array.min() <= array.max() <= 1,'Delivered array outside [0,1]')
    return array


def bank_prefix(bank,rid):
    folder = '/baseline/'+bank+'/' if bank in ('original','parent','r2') else '/snapshot_%04d/'%int(bank[8:])
    return RUN+folder+rid


def discovered_banks(e,specs):
    banks = ['original','parent','r2']; coverage = {}
    for bank in banks+['snapshot0','snapshot55','snapshot110']:
        present,missing,partial = [],[],[]
        for rid in specs:
            names = [bank_prefix(bank,rid)+suffix for suffix in ('.npy','.png','.json')]
            flags = [name in e.hashes for name in names]
            if all(flags):
                present.append(rid)
            elif any(flags):
                partial.append({'id':rid,'present':[name for name,flag in zip(names,flags) if flag],
                    'missing':[name for name,flag in zip(names,flags) if not flag]})
            else:
                missing.append(rid)
        coverage[bank] = {'complete_triplets':present,'absent_ids':missing,'partial_triplets':partial,
            'all478_complete':len(present)==478}
        if bank.startswith('snapshot') and present:
            banks.append(bank)
    require(all(coverage[bank]['all478_complete'] for bank in ('original','parent','r2')),
            'Three fresh full478 baseline controls are required')
    require(coverage['snapshot0']['all478_complete'],'Full478 initial parity controls required')
    return banks,coverage


def audit_teachers(e,cases):
    manifest = e.json(RUN+'/teacher/manifest.json')
    require(manifest['complete'] is True and manifest['unique_feature_teacher_forwards']==55 and
            manifest['unique_behavior_teacher_forwards']==110,'Exact55feature/110behavior teacher census required')
    features = manifest['features']; refs = {row['reference_id'] for row in cases.values()}
    require(set(features)==refs and len(refs)==55,'Every55 teacher reference required')
    report = {'complete':True,'feature_sources':{},'behavior':[],'RMS_scalar_atol':1e-10,
        'neural_teacher_correctness_independently_replayed':False,'no_new_forward_calls':True}
    for rid,entry in sorted(features.items()):
        clear = next(row for row in cases.values() if row['reference_id']==rid and row['profile']=='clear')
        require(entry['clear_input']==clear['input'] and set(entry['features'])==set(SITES) and
                set(entry['scales'])==set(SITES),'Clean feature/source/site joins differ')
        scale_doc = e.json(RUN+'/teacher/features/'+rid+'/scales.json')
        require(scale_doc['reference_id']==rid and scale_doc['input']==clear['input'],'Scale/source joins differ')
        scales,site_details = {},{}
        for site,channels in SITES.items():
            pin = entry['features'][site]
            require(pin['path']==RUN+'/teacher/features/'+rid+'/encoder'+site+'.npy','Exact feature file path required')
            array = finite_array(e,pin,(1,channels,int(site),int(site)))
            rms = max(1e-6,float(np.sqrt(np.square(array.astype(np.float64)).mean())))
            require(abs(rms-entry['scales'][site])<=1e-10 and abs(rms-scale_doc['scales'][site])<=1e-10,
                    'Fixed feature RMS differs from actual finite array')
            scales[site] = rms
            site_details[site] = {'shape':list(array.shape),'float32_pixel_sha256':pixel_sha(array),
                'RMS':rms,'minimum':float(array.min()),'maximum':float(array.max()),'file':pin}
            del array
        report['feature_sources'][rid] = {'scales':entry['scales'],'recomputed_scales':scales,
            'sites':site_details,'clear_input':clear['input']}
        e.deadline()
    behavior = manifest['behavior']
    require(len(behavior)==110 and len({(row['role'],row['id']) for row in behavior})==110,
            'Distinct110 role behavior teacher bindings required')
    behavior_map = {}
    for row in behavior:
        role,rid = row['role'],row['id']; source = cases[rid]
        require(role in ('clear','dark') and row['kind']==('original' if role=='clear' else 'r2') and
                source['profile']==('clear' if role=='clear' else 'linear_srgb_exposure0p35'),
                'Behavior source/role/frozen-model join differs')
        stem = RUN+'/teacher/'+role+'/'+rid
        require(all(row[key]['path']==stem+suffix for key,suffix in
            (('U','_U.npy'),('RAW','_RAW.npy'),('PNG','_PNG.png'))),'Behavior artifact paths differ')
        u = finite_array(e,row['U'],(256,256,3)); raw = finite_array(e,row['RAW'],(256,256,3),bounded=True)
        png = rgb8(e.file(row['PNG']['path'],row['PNG']))
        require(np.array_equal(np.floor(raw*np.float32(255)).astype(np.uint8),png),'Behavior RAW/PNG floor mismatch')
        behavior_map[role,rid] = row
        report['behavior'].append({'role':role,'id':rid,'kind':row['kind'],'U':row['U'],'RAW':row['RAW'],'PNG':row['PNG'],
            'U_pixel_sha256':pixel_sha(u),'RAW_pixel_sha256':pixel_sha(raw),'PNG_RGB8_sha256':pixel_sha(png),
            'RAW_PNG_floor_exact':True,'U_to_RAW_independently_verified':False,
            'U_RAW_contract':'Source uses native512 clamp before RAW256 resize, while U256 resizes unclipped native512. They cannot be compared by clipping U256; native512 arrays are not retained.'})
        del u,raw,png
    require(Counter(row['role'] for row in behavior)=={'clear':55,'dark':55},'Complete clear/dark role teacher counts required')
    return behavior_map,report


def audited_guards(e,cases,behavior_map,checks,mismatches):
    banks,gates,gate_mismatches,coverage = {},{},{},{}
    for update in (0,20,55,110):
        stem = RUN+'/guard_%04d/'%update
        if stem+'bank.json' not in e.hashes:
            coverage[str(update)] = {'bank_present':False,'present_files':sorted(name for name in e.hashes if name.startswith(stem))}
            continue
        saved = e.json(stem+'bank.json'); savedmap = {row['id']:row for row in saved['cases']}
        require(saved['complete'] is True and len(savedmap)==24 and
                set(savedmap)=={row['id'] for row in e.p['guard_cases']},'Full24 repeated TRAIN guard bank required')
        bank = {}
        for guard in e.p['guard_cases']:
            rid,role = guard['id'],guard['role']; spec = spec_from_row(e,cases[rid])
            entry,raw,png,document,conversion,bad = audited_output(e,stem+role+'/'+rid,spec)
            checks.append(conversion); mismatches.extend(bad); entry['role']=entry['guard_role']=role
            if role in ('clear','dark'):
                teacher = savedmap[rid]['behavior_teacher']; expected = behavior_map[role,rid]
                require(all(teacher[key]==expected[key] for key in ('kind','U','RAW','PNG')),'Guard behavior teacher manifest mismatch')
                target_raw = raw256(e.file(teacher['RAW']['path'],teacher['RAW']))
                target_png = rgb8(e.file(teacher['PNG']['path'],teacher['PNG']))
                entry['behavior_distance'] = {'raw':float(np.abs(raw.astype(np.float64)-target_raw.astype(np.float64)).mean()),
                    'native_png':float(np.abs(floats(png).astype(np.float64)-floats(target_png).astype(np.float64)).mean())}
                mismatches.extend(compare(entry['behavior_distance'],savedmap[rid]['behavior_distance'],rid+'/behavior_distance'))
            mismatches.extend(compare(document['metrics'],savedmap[rid]['metrics'],rid+'/guard_bank'))
            bank[rid] = entry
        banks[str(update)] = bank
        require('0' in banks,'Initial guard controls must precede later guards')
        gate = guard_interim(bank,banks['0']); gates[str(update)] = gate
        gate_mismatches[str(update)] = compare(gate,e.json(stem+'gates.json'),'guard'+str(update))
        coverage[str(update)] = {'bank_present':True,'cases':24}
    return banks,gates,coverage,gate_mismatches


def audited_training_scalars(e,cases,teacher_report,behavior_map):
    """Check schedule, arithmetic and pins; absent training activations stay absent."""
    schedule = e.json(e.p['data']['schedule_path']); reports = []; errors = []
    completed = e.import_receipt['independently_recomputed_status']['completed_training_updates']
    require(type(completed) is int and 0 <= completed <=110,'Authoritative finite committed update count required')
    for item in schedule['updates']:
        update = item['update']; slot_rows = []
        for slot in item['slots']:
            name = RUN+'/training/u%04d_s%d.json'%(update,slot['slot'])
            if name not in e.hashes:
                continue
            row = e.json(name); source = cases[slot['id']]
            require(all(row[key]==slot[key] for key in ('slot','id','reference_id','role')) and row['update']==update and
                    row['input']==source['input'],'TRAIN schedule/observation/source join differs')
            expected_lr = 1e-4*min(1.,update/10.)
            require(row['LR']==expected_lr,'Prospective LR/warmup schedule differs')
            terms = row['terms']
            require(type(terms) is dict and all(type(value) in (int,float) and math.isfinite(value) and value>=0
                    for value in terms.values()),'Finite nonnegative captured loss components required')
            if slot['role']=='filter':
                require(set(terms)=={'normalized_feature_l1','spatial_rgb_l1','adjacent_residual_l1'} and
                        row['feature_scales']==teacher_report['feature_sources'][source['reference_id']]['scales'] and
                        row['photo_target']==source['target256'],'TRAIN filter captured feature/target binding differs')
                require(row['clean_feature_files']==e.json(RUN+'/teacher/manifest.json')['features'][source['reference_id']]['features'],
                        'Captured training clean-feature binding differs')
                expected = .1*terms['normalized_feature_l1']+terms['spatial_rgb_l1']+.25*terms['adjacent_residual_l1']
            else:
                require(set(terms)=={'teacher_rgb_l1'} and row['photo_target'] is None and row['feature_scales'] is None and
                        row['behavior_teacher']==behavior_map[slot['role'],slot['id']]['U'],'TRAIN role teacher association differs')
                expected = 2.*terms['teacher_rgb_l1']
            tolerance = 1e-6*max(1.,abs(expected))
            if abs(row['scalar']-expected)>tolerance or abs(row['divided_scalar']-row['scalar']/4.)>tolerance:
                errors.append(name+'/captured_FP32_component_arithmetic')
            slot_rows.append({'slot':row['slot'],'id':row['id'],'role':row['role'],'scalar':row['scalar'],
                'divided_scalar':row['divided_scalar'],'terms':terms,'LR':row['LR']})
        commit_name = RUN+'/training/u%04d_commit.json'%update
        committed = commit_name in e.hashes
        if committed:
            commit = e.json(commit_name)
            require(len(slot_rows)==4 and commit['complete'] is True and commit['update']==update and
                    commit['slot_scalars']==[row['scalar'] for row in slot_rows] and
                    commit['all515_base_values_unchanged'] is True,'Captured finite commit/slot accounting differs')
            require(commit['sampler']['completed_updates']==update and commit['sampler']['completed_microbatches']==update*4,
                    'Captured sampler commit differs')
        require(committed==(update<=completed),'Physical TRAIN commit census differs from terminal status')
        if slot_rows or committed:
            reports.append({'update':update,'committed':committed,'slots':slot_rows})
    return {'complete':True,'committed_updates':completed,'updates':reports,'arithmetic_mismatches':errors,
        'captured_component_arithmetic_relative_tolerance':1e-6,
        'limitation':'Scalar component arithmetic and source joins only. Full native training predictions, adapted activations and autograd are not saved and are not independently replayed.',
        'training_loss_or_gradient_recomputed_from_model':False,'neural_calls':0}


def audited_witness(e,cases,behavior_map):
    document = e.json(RUN+'/witness/receipt.json')
    require(document['complete'] is True and document['candidate_forwards']==3 and
            document['gradient_queries']==4 and document['optimizer_updates']==0 and
            set(document['roles'])=={'filter','clear','dark'},'Exact zero-update3-forward/4-query witness receipt required')
    report = {}; mismatches = []
    for role,row in document['roles'].items():
        rid = e.p['witness_cases'][role]
        require(row['id']==rid,'Witness source association differs')
        upin = {'path':RUN+'/witness/'+role+'_U.npy'}; upin.update(bytes=e.sizes[upin['path']],sha256=e.hashes[upin['path']])
        rawpin = {'path':RUN+'/witness/'+role+'_RAW.npy'}; rawpin.update(bytes=e.sizes[rawpin['path']],sha256=e.hashes[rawpin['path']])
        u = finite_array(e,upin,(256,256,3)); raw = finite_array(e,rawpin,(256,256,3),bounded=True)
        # Delivered RAW clamps native512 before resize. U resizes the unclipped
        # native output; clamp(U256) is not an independent RAW correctness test.
        if role=='filter':
            spec = spec_from_row(e,cases[rid]); residual = u.astype(np.float64)-spec['target'].astype(np.float64)
            image_terms = {'spatial_rgb_l1':float(np.abs(residual).mean()),'adjacent_residual_l1':float(.5*(
                np.abs(residual[:,1:]-residual[:,:-1]).mean()+np.abs(residual[1:]-residual[:-1]).mean()))}
            for name,value in image_terms.items():
                if abs(value-row['terms'][name])>1e-6:
                    mismatches.append(role+'/'+name)
            scalar = (image_terms['spatial_rgb_l1']+.25*image_terms['adjacent_residual_l1'])/4
            if abs(scalar-row['image_only_divided_scalar'])>1e-6:
                mismatches.append(role+'/image_only_divided_scalar')
            report[role] = {'id':rid,'recomputed_image_terms':image_terms,
                'image_only_divided_scalar':scalar,'normalized_feature_l1_captured_not_independently_replayed':row['terms']['normalized_feature_l1']}
        else:
            teacher = behavior_map[role,rid]; target = finite_array(e,teacher['U'],(256,256,3))
            l1 = float(np.abs(u.astype(np.float64)-target.astype(np.float64)).mean())
            if abs(l1-row['terms']['teacher_rgb_l1'])>1e-6 or abs(2*l1-row['scalar'])>1e-6:
                mismatches.append(role+'/behavior_loss')
            report[role] = {'id':rid,'teacher_kind':teacher['kind'],'recomputed_teacher_rgb_l1':l1}
        report[role].update(U_pixel_sha256=pixel_sha(u),RAW_pixel_sha256=pixel_sha(raw),
            U_to_RAW_independently_verified=False)
    return {'complete':True,'roles':report,'mismatches':mismatches,'image_scalar_atol':1e-6,
        'gradient_claims_captured_only_not_independent_gradient_replay':True,
        'no_new_forward_gradient_or_optimizer_calls':True}


def summaries(rows):
    result,deltas = [],[]; keys = ('source','evidence','profile','cohort','split')
    for bank,values in rows.items():
        groups = defaultdict(list)
        for rid,row in values.items():
            groups[tuple(row[key] for key in keys)].append(rid)
        for group,ids in sorted(groups.items()):
            for view in VIEWS:
                entry = {'bank':bank,**dict(zip(keys,group)),'view':view,'case_ids':ids,'cases':len(ids)}
                for field in FIELDS:
                    array = [values[rid]['metrics'][view][field] for rid in ids if values[rid]['metrics'][view][field] is not None]
                    entry['mean_'+field] = sum(array)/len(array) if array else None
                result.append(entry)
                if bank.startswith('snapshot') and bank!='snapshot0':
                    for control in ('original','parent','r2'):
                        shared = [rid for rid in ids if rid in rows[control]]
                        delta = {'comparison':bank+'_minus_'+control,**dict(zip(keys,group)),
                            'view':view,'case_ids':shared,'cases':len(shared)}
                        for field in FIELDS:
                            array = [values[rid]['metrics'][view][field]-rows[control][rid]['metrics'][view][field]
                                for rid in shared if values[rid]['metrics'][view][field] is not None]
                            delta['mean_delta_'+field] = sum(array)/len(array) if array else None
                            delta[field+'_lower'] = sum(value<0 for value in array)
                            delta[field+'_higher'] = sum(value>0 for value in array)
                            delta[field+'_equal'] = sum(value==0 for value in array)
                        deltas.append(delta)
    return result,deltas


def render_sheets(out,e,specs,refused,pngs,rows,aux_specs,aux_pngs,guardbanks,behavior_map,cases):
    """Every intended comparison, auxiliary, saved guard and refusal; eight rows max."""
    folder = out/'visual_sheets'; folder.mkdir()
    sheets,cells = [],[]
    snapshots = sorted((bank for bank in rows if bank.startswith('snapshot')),key=lambda name:int(name[8:]))
    newest = snapshots[-1]
    def panel_source(spec,bank):
        if bank=='input':
            return spec['input'],spec['row']['input']
        if bank=='target':
            return spec['target8'],spec['row'].get('target256',spec['row'].get('target'))
        rid = spec['row']['id']; return pngs.get((bank,rid)),{'path':bank_prefix(bank,rid)+'.png'}
    def page(kind,label,page_rows,column_titles,provider):
        e.deadline(); name = kind+'_%03d.png'%(1+sum(row['kind']==kind for row in sheets))
        width,height = 256*len(column_titles),64+284*len(page_rows)
        canvas = Image.new('RGB',(width,height),(250,248,243)); draw = ImageDraw.Draw(canvas)
        draw.text((4,4),label+' | exact256 saved pixels | not qualified | no enhancement/resampling',fill='black')
        for index,title in enumerate(column_titles):
            draw.text((index*256+4,31),title,fill='black')
        page_cells = []
        for index,(rid,spec) in enumerate(page_rows):
            y = 64+index*284; row = spec['row']
            text = rid+' | '+str(row.get('source'))+' | '+str(row.get('profile'))+' | '+str(row.get('split'))
            if row.get('input_only_decision'):
                text += ' | '+row['input_only_decision']
            draw.text((4,y),text,fill='black')
            for col,(array,pin,note) in enumerate(provider(rid,spec)):
                x,top = col*256,y+24
                if array is None:
                    draw.rectangle((x,top,x+255,top+255),fill=(225,224,218))
                    draw.text((x+6,top+76),note or 'No saved output\nnot generated by auditor',fill='black')
                else:
                    require(array.shape==(256,256,3) and array.dtype==np.uint8,'Exact saved RGB8 sheet panel required')
                    canvas.paste(Image.fromarray(array),(x,top))
                    cell = {'sheet':name,'kind':kind,'id':rid,'column':col,'x':x,'y':top,'width':256,'height':256,
                        'RGB8_sha256':pixel_sha(array),'source':pin}
                    page_cells.append(cell); cells.append(cell)
        path = folder/name; canvas.save(path)
        with Image.open(path) as image:
            saved = np.asarray(image)
            for cell in page_cells:
                require(pixel_sha(saved[cell['y']:cell['y']+256,cell['x']:cell['x']+256])==cell['RGB8_sha256'],
                        'Saved exact sheet pixel panel differs')
        sheets.append({'path':path.relative_to(out).as_posix(),'kind':kind,'label':label,'case_ids':[rid for rid,_ in page_rows],
            'case_count':len(page_rows),'width':width,'height':height,'bytes':path.stat().st_size,'sha256':sha(path)})
    groups = []
    for split in ('TRAIN','DEV'):
        groups.append(('FILTER_'+split,sorted(rid for rid,spec in specs.items() if
            spec['row']['cohort']=='retained_filter72_mixed_roles' and spec['row']['split']==split)))
    for source in sorted({spec['row']['source'] for spec in specs.values() if spec['row']['evidence']=='native_unpaired'}):
        groups.append(('NATIVE_UNPAIRED_'+source,sorted(rid for rid,spec in specs.items() if
            spec['row']['evidence']=='native_unpaired' and spec['row']['source']==source)))
    already = {rid for _,ids in groups for rid in ids}
    groups.append(('PAIRED_HISTORICAL_DEV',sorted(set(specs)-already,key=lambda rid:(specs[rid]['row']['source'],specs[rid]['row']['profile'],rid))))
    allids = [rid for _,ids in groups for rid in ids]
    require(len(allids)==len(set(allids))==478 and set(allids)==set(specs),'Every478 visual case exactly once required')
    def comparison(rid,spec):
        panels = []
        for bank in ('input','target','original','r2',newest):
            array,pin = panel_source(spec,bank)
            note = 'Native CCTV: UNPAIRED\nNo aligned clean reference' if bank=='target' and spec['row']['evidence']=='native_unpaired' else ''
            panels.append((array,pin,note))
        return panels
    ordered = [(label,rid) for label,ids in groups for rid in ids]
    for start in range(0,len(ordered),8):
        subset = ordered[start:start+8]; label = ' / '.join(dict.fromkeys(label for label,_ in subset))
        page('comparison',label,[(rid,specs[rid]) for _,rid in subset],
            ('Observed input','Paired target / native note','Untouched CodeFormer','Retained R2','New skip adapter '+newest),comparison)
    aux_alias = sorted({alias for alias,_ in aux_pngs},key=lambda value:-1 if value=='r2' else int(value.split('_')[-1]))
    if aux_alias:
        latest = aux_alias[-1]
        def auxiliary(rid,spec):
            return [(spec['input'],spec['row']['input'],''),(spec['target8'],spec['row'].get('target256'),''),
                (aux_pngs.get(('r2',rid)),{'path':RUN+'/auxiliary/r2/'+rid+'.png'},''),
                (aux_pngs.get(('candidate_0000',rid)),{'path':RUN+'/auxiliary/candidate_0000/'+rid+'.png'},''),
                (aux_pngs.get((latest,rid)),{'path':RUN+'/auxiliary/'+latest+'/'+rid+'.png'},'')]
        auxids = sorted(aux_specs,key=lambda rid:(aux_specs[rid]['row']['reference_id'],aux_specs[rid]['row']['profile']))
        for start in range(0,len(auxids),8):
            page('auxiliary','AUXILIARY repeated TRAIN diagnosis; readable/borderline remain separate',
                [(rid,aux_specs[rid]) for rid in auxids[start:start+8]],('Observed input','Paired photo','Retained R2','Initial identity adapter','New adapter '+latest),auxiliary)
    if guardbanks:
        last_guard = max(int(update) for update in guardbanks)
        def guard(rid,spec):
            role = next(row['role'] for row in e.p['guard_cases'] if row['id']==rid)
            teacher = behavior_map.get((role,rid))
            teacher_png = rgb8(e.file(teacher['PNG']['path'],teacher['PNG'])) if teacher else spec['target8']
            init = RUN+'/guard_0000/'+role+'/'+rid+'.png'; current = RUN+'/guard_%04d/'%last_guard+role+'/'+rid+'.png'
            return [(spec['input'],spec['row']['input'],''),(spec['target8'],spec['row'].get('target256'),'Behavior-only anchor\nNo photographic clean score'),
                (teacher_png,teacher['PNG'] if teacher else spec['row'].get('target256'),''),
                (rgb8(e.file(init)),{'path':init},''),(rgb8(e.file(current)),{'path':current},'')]
        guardids = sorted(row['id'] for row in e.p['guard_cases'])
        for start in range(0,len(guardids),8):
            page('guard','Repeated TRAIN guards; saved stage'+str(last_guard),
                [(rid,spec_from_row(e,cases[rid])) for rid in guardids[start:start+8]],
                ('Observed input','Photo target when present','Frozen teacher / paired photo','Initial adapter guard','Latest saved TRAIN guard'),guard)
    def refusal(rid,spec):
        return [(spec['input'],spec['row']['input'],''),(None,None,'REQUEST CLEARER INPUT\n'+spec['row']['input_review']['reason'])]
    for start in range(0,len(refused),8):
        page('refusal','Input-only insufficient controls: no generated output',
            [(row['id'],spec_from_row(e,row)) for row in refused[start:start+8]],('Observed input','Input-only decision'),refusal)
    manifest = {'complete':True,'sheets':sheets,'total_sheets':len(sheets),'sheet_kind_counts':dict(Counter(row['kind'] for row in sheets)),
        'comparison_slots':478,'native_unpaired_slots':48,'mixed_filter_roles':{'TRAIN':36,'DEV':36},
        'refused_input_only_controls':24,'auxiliary_repeated_TRAIN_cases':18,'all478_once':True,
        'comparison_newest_saved_snapshot':newest,'max_rows_per_sheet':8,
        'root_must_view_every_sheet':True,'root_personal_visual_review_performed':False,
        'all_panels_exact256_saved_RGB8':True,'image_processing':'none; no resampling/sharpening/enhancement',
        'missing_or_absent_stages_not_synthesized':True,'model_qualified':False}
    write(out/'sheet_cells.json',cells); write(out/'visual_manifest.json',manifest)
    with (folder/'INDEX.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('# Skip adapter exact saved-pixel review\n\nAll478 comparison slots,18 auxiliary TRAIN probes, saved TRAIN guards and24 input-only refusals. Eight rows maximum; each photo panel is exact256 pixels. Native CCTV stays unpaired. Scores do not qualify identity recovery. Every sheet requires root visual review.\n\n')
        stream.write('\n'.join('- ['+Path(row['path']).name+']('+Path(row['path']).name+') '+row['label'] for row in sheets)+'\n')
    return manifest


def audit(receipt,out):
    require(not any(name in sys.modules for name in ('torch','tensorflow','jax')),'No neural frameworks allowed')
    out = Path(out).resolve(strict=False)
    require(out.is_relative_to((ROOT/'outputs').resolve()) and not out.is_symlink(),'New local audit output required')
    e = Evidence(receipt); started = time.monotonic()
    require(not out.exists(),'Preserve previous or partial audits; exclusive new destination required')
    out.mkdir(parents=True)
    specs,refused = load_specs(e); banks,coverage = discovered_banks(e,specs)
    rows = {bank:{} for bank in banks}; pngs = {}; checks = []; mismatches = []; parity = []; gate_mismatches = []
    for index,(rid,spec) in enumerate(specs.items(),1):
        rawbanks,documents = {},{}
        for bank in banks:
            if rid not in coverage[bank]['complete_triplets']:
                continue
            entry,raw,png,document,conversion,bad = audited_output(e,bank_prefix(bank,rid),spec)
            rows[bank][rid] = entry; pngs[bank,rid] = png; rawbanks[bank] = raw; documents[bank] = document
            checks.append(conversion); mismatches.extend(bad)
        require(np.array_equal(rawbanks['snapshot0'],rawbanks['r2']) and
                np.array_equal(pngs['snapshot0',rid],pngs['r2',rid]),'Initial candidate/frozen R2 exact saved-pixel parity failed')
        require(all(documents['snapshot0']['outputs'][view][field]==documents['r2']['outputs'][view][field]
            for view in ('raw','png') for field in ('bytes','sha256')),'Initial candidate/R2 physical file parity failed')
        require(all(document['upstream']==documents['r2']['upstream'] for bank,document in documents.items() if bank.startswith('snapshot')),
                'Frozen saved LQ/logit/token/postAdaIN readouts changed')
        parity.append(rid)
        if index%50==0 or index==478:
            print(json.dumps({'independent_saved_comparison_slots':index,'of':478,'banks':banks,'seconds':time.monotonic()-started}),flush=True)
    historical = {bank:{rid:row for rid,row in values.items() if row['cohort']=='historical_DEV406'} for bank,values in rows.items()}
    gates = {}
    for bank in banks:
        if not bank.startswith('snapshot') or not coverage[bank]['all478_complete']:
            continue
        update = int(bank[8:])
        value = {'preservation':preservation(historical[bank],historical['original']),
            'parent_dark_retention':dark_retention(historical[bank],historical['parent']),
            'R2_dark_retention':dark_retention(historical[bank],historical['r2']),
            'interim':interim(historical[bank],historical['original'],historical['snapshot0']),
            'filters':filter_summary(rows[bank]),'inherited_initial_failures_not_qualified':update==0}
        if update==0:
            value['initial_R2_full478_parity'] = {'pass':True,'cases':478,'RAW_and_PNG_files_exact':True,
                'upstream_readouts_also_checked_for_every_case':True,'current_runtime_only':True,'numerical_tolerance':0.}
        gates[str(update)] = value
        saved_name = RUN+'/snapshot_%04d/gates.json'%update
        if saved_name in e.hashes:
            gate_mismatches.extend(compare(value,e.json(saved_name),'snapshot'+str(update)))
    bundle = e.json(e.p['data']['bundle_path']); cases = bundle['case_by_id']
    require(len(cases)==165 and all(row['split']=='TRAIN' for row in cases.values()),'Exact165 active TRAIN role inputs required')
    for row in cases.values():
        spec_from_row(e,row)
    behavior_map,teacher_report = audit_teachers(e,cases)
    guardbanks,guardgates,guardcoverage,guard_gate_errors = audited_guards(e,cases,behavior_map,checks,mismatches)
    gate_mismatches.extend(error for errors in guard_gate_errors.values() for error in errors)
    auxiliary = e.json(e.p['auxiliary_path'])['cases']
    require(len(auxiliary)==18 and Counter(row['input_only_decision'] for row in auxiliary)=={'readable':16,'borderline':2} and
            all(row['split']=='TRAIN' and row['optimization_permitted_now'] is False for row in auxiliary),
            'Fixed18 repeated TRAIN auxiliary diagnostic labels required')
    aux_specs = {row['id']:spec_from_row(e,row) for row in auxiliary}; auxrows,auxpngs,auxcoverage = {},{},{}
    for alias in ('r2','candidate_0000','candidate_0055','candidate_0110'):
        stem = RUN+'/auxiliary/'+alias+'/'; present = []
        for rid,spec in aux_specs.items():
            names = [stem+rid+suffix for suffix in ('.npy','.png','.json')]
            if not all(name in e.hashes for name in names):
                continue
            entry,raw,png,document,conversion,bad = audited_output(e,stem+rid,spec)
            entry['input_only_decision'] = spec['row']['input_only_decision']
            auxrows.setdefault(alias,{})[rid] = entry; auxpngs[alias,rid] = png; present.append(rid)
            checks.append(conversion); mismatches.extend(bad)
            if alias=='candidate_0000':
                require(np.array_equal(png,auxpngs['r2',rid]),'Initial auxiliary PNG/R2 parity failed')
                require(np.array_equal(raw,raw256(e.file(RUN+'/auxiliary/r2/'+rid+'.npy'))),'Initial auxiliary RAW/R2 parity failed')
            if stem+'bank.json' in e.hashes:
                saved = e.json(stem+'bank.json'); savedmap = {row['id']:row for row in saved['cases']}
                require(saved['complete'] is True and set(savedmap)==set(aux_specs),'Full18 saved auxiliary bank required')
                mismatches.extend(compare(document,savedmap[rid],alias+'/'+rid+'/auxbank'))
        auxcoverage[alias] = {'present_complete_triplets':present,'absent_ids':sorted(set(aux_specs)-set(present)),
            'bank_present':stem+'bank.json' in e.hashes}
    training = audited_training_scalars(e,cases,teacher_report,behavior_map); witness = audited_witness(e,cases,behavior_map)
    visual = render_sheets(out,e,specs,refused,pngs,rows,aux_specs,auxpngs,guardbanks,behavior_map,cases)
    summary,deltas = summaries(rows)
    outputs = {'metrics.json':{'complete':True,'banks':rows,'source_profile_groups':summary},
        'source_profile_comparisons.json':{'complete':True,'comparisons':deltas,'native_clean_metrics_null':True,'FINAL_used':False,
            'filter_TRAIN_DEV_separate':True},
        'gates.json':{'complete':True,'snapshots':gates,'TRAIN_guards':guardgates,'comparison_coverage':coverage,'guard_coverage':guardcoverage},
        'teachers.json':teacher_report,'training_scalar_evidence.json':training,'witness_evidence.json':witness,
        'auxiliary_metrics.json':{'complete':True,'banks':auxrows,'coverage':auxcoverage,'repeated_TRAIN_only':True,
            'borderline_labels_unchanged':True,'independent_holdout_claim':False},
        'conversion_and_parity_checks.json':{'complete':True,'checks':checks,'snapshot0_exact_R2_ids':parity,
            'metric_mismatches':mismatches,'gate_mismatches':gate_mismatches,'scalar_metric_atol':1e-10,'array_equality_tolerance':0.}}
    for name,value in outputs.items():
        write(out/name,value)
    status = e.import_receipt['independently_recomputed_status']; newest = str(max(int(update) for update in gates))
    terminal_name = status['worker_terminal_evidence']; terminal = e.json(terminal_name) if terminal_name else {}
    errors = bool(mismatches or gate_mismatches or training['arithmetic_mismatches'] or witness['mismatches'])
    for name,old in e.checked.items():
        current = checked(e.root,name).stat()
        require((current.st_size,current.st_mtime_ns,current.st_ino)==(old['bytes'],old['mtime_ns'],old['inode']),
                'Bound evidence changed during independent completed audit')
    require(not any(name in sys.modules for name in ('torch','tensorflow','jax')),'Neural framework imported')
    result = {'complete':True,'audit_pass':not errors,'audit_source_sha256':sha(__file__),'copied_local_pure_source':SOURCE_COPY,
        'archive_sha256':ARCHIVE_SHA,'protocol_sha256':PROTOCOL_SHA,'import_receipt_sha256':sha(e.receipt_path),
        'returned_root':str(e.root),'comparison_slots':478,'historical_DEV':406,'paired_synthetic_historical_DEV':358,
        'native_unpaired':48,'mixed_filter_TRAIN':36,'mixed_filter_DEV':36,'refused_input_only_controls':24,
        'comparison_RAW_PNG_pairs_checked':sum(len(values) for values in rows.values()),
        'TRAIN_guard_RAW_PNG_pairs_checked':sum(len(values) for values in guardbanks.values()),
        'auxiliary_RAW_PNG_pairs_checked':sum(len(values) for values in auxrows.values()),
        'feature_arrays_checked':220,'behavior_U_RAW_PNG_teacher_triples_checked':110,'witness_U_RAW_pairs_checked':3,
        'exact_initial_R2_parity_slots':len(parity),'actual_snapshot_updates':sorted(int(update) for update in gates),
        'latest_completed_saved_snapshot':int(newest),'preservation_pass':gates[newest]['preservation']['pass'],
        'parent_dark_retention_pass':gates[newest]['parent_dark_retention']['pass'],
        'R2_dark_retention_pass':gates[newest]['R2_dark_retention']['pass'],
        'additional_interim_stop_pass':gates[newest]['interim']['pass'],
        'saved_metric_mismatches':mismatches,'saved_gate_mismatches':gate_mismatches,
        'training_scalar_arithmetic_mismatches':training['arithmetic_mismatches'],'witness_scalar_mismatches':witness['mismatches'],
        'authoritative_training_status':status,'recorded_failure':terminal.get('error'),
        'verified_read_bindings':{name:{'bytes':e.sizes[name],'sha256':e.hashes[name]} for name in sorted(e.checked)},
        'outputs':{name:{'bytes':(out/name).stat().st_size,'sha256':sha(out/name)} for name in list(outputs)+['sheet_cells.json','visual_manifest.json']},
        'visual_manifest':{'path':'visual_manifest.json','sheets':visual['total_sheets'],'kind_counts':visual['sheet_kind_counts']},
        'root_personal_visual_review_pending':True,'user_acceptance_established':False,'model_qualified':False,'app_changed':False,
        'native_sources_not_ethnicity_or_Zamboanga_claim':True,'FINAL_used':False,'actual_local_neural_gradient_optimizer_VM_calls':0,
        'runtime':{'Python':sys.version,'NumPy':np.__version__,'Pillow':pillow_version,'scikit_image':skimage.__version__},
        'seconds':time.monotonic()-started,'deadline_seconds':DEADLINE_SECONDS,'UTC':datetime.now(timezone.utc).isoformat()}
    write(out/'audit.json',result)
    with (out/'AUDIT.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('# Independent skip-adapter saved-pixel audit\n\nAudit integrity and model acceptance remain separate. The run stopped at'+str(status['completed_training_updates'])+' updates. Full actual banks and captured teacher/feature/scalar records were audited without any model or returned Python execution. Native CCTV stays unpaired; repeated TRAIN auxiliary cases do not become holdout evidence.\n\n')
        stream.write('| Requirement | Pass |\n|---|---|\n')
        for field in ('preservation_pass','parent_dark_retention_pass','R2_dark_retention_pass','additional_interim_stop_pass'):
            stream.write('|'+field+'|'+str(result[field])+'|\n')
        stream.write('\nEvery planned sheet requires root visual review. No app or model promotion follows from this integrity audit. Captured gradients and feature-loss components were not independently replayed.\n')
    require(result['audit_pass'],'Saved metric/gate/captured scalar arithmetic disagrees with independent audit')
    return {key:result[key] for key in ('complete','audit_pass','comparison_RAW_PNG_pairs_checked','preservation_pass',
        'parent_dark_retention_pass','R2_dark_retention_pass','additional_interim_stop_pass','seconds')}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-receipt',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.import_receipt,args.out)),flush=True)

