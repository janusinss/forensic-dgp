"""Independently verify saved 96-case synthetic pixels; no producer or model import."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys
import time

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
SELECT_SHA = 'f1e0fa95dd2140a70f5ac091ecf83d09f51ac80e6bc1eabd8acb9223594ced9e'
IDS = {
    'TRAIN': ('tr_ffhq_00084', 'tr_ffhq_00178', 'tr_ffhq_00323', 'tr_ffhq_00616'),
    'DEV': ('va_ffhq_00383', 'va_ffhq_01093', 'va_ffhq_08025', 'va_ffhq_09056'),
}
RECT = (64, 64, 192, 192)
# Independent literal transcription of root's prospective contract, not producer dispatch.
PROFILES = (
    ('clear', 'identity', {}, False),
    ('gaussian_2p4', 'gaussian', {'sigma_x': 2.4, 'sigma_y': 2.4}, False),
    ('gaussian_6p0', 'gaussian', {'sigma_x': 6.0, 'sigma_y': 6.0}, True),
    ('anisotropic_h5_v0p7', 'gaussian', {'sigma_x': 5.0, 'sigma_y': .7}, False),
    ('anisotropic_h0p7_v5', 'gaussian', {'sigma_x': .7, 'sigma_y': 5.0}, False),
    ('repeated_rows_4_blend0p75', 'repeated_rows', {'block_rows': 4, 'repeat_weight': .75}, True),
    ('global_mosaic_8', 'mosaic', {'cell': 8}, False),
    ('global_mosaic_16', 'mosaic', {'cell': 16}, True),
    ('local_gaussian_3p0', 'local_gaussian', {'sigma_x': 3.0, 'sigma_y': 3.0}, False),
    ('local_gaussian_10p0', 'local_gaussian', {'sigma_x': 10.0, 'sigma_y': 10.0}, True),
    ('local_mosaic_8', 'local_mosaic', {'cell': 8}, False),
    ('local_mosaic_16', 'local_mosaic', {'cell': 16}, True),
)
PROFILE_ROWS = tuple(dict(id=n, operator=o, parameters=p, strong_control=s)
                     for n, o, p, s in PROFILES)
REVIEW_ORDER = ('global_mosaic_16', 'local_mosaic_16', 'local_gaussian_10p0', 'gaussian_6p0',
    'repeated_rows_4_blend0p75', 'gaussian_2p4', 'anisotropic_h5_v0p7', 'anisotropic_h0p7_v5',
    'global_mosaic_8', 'local_gaussian_3p0', 'local_mosaic_8', 'clear')


def check(value, reason):
    if not value:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def unique(items):
        result = {}
        for k, v in items:
            check(k not in result, 'Duplicate JSON key')
            result[k] = v
        return result
    return json.loads(data.decode('utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def safe(path):
    path = Path(path).absolute()
    check(path.resolve(strict=True).is_relative_to(ROOT), 'Evidence outside workspace')
    for part in (path, *path.parents):
        if part == ROOT.parent:
            break
        s = part.lstat()
        check(not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024,
              'Linked/reparse evidence')
    s = path.stat()
    check(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Ordinary single-link evidence required')
    return path


def bound(path, expected_sha=None, size=None, cap=8 * 1024**2):
    path = safe(path)
    before = path.stat()
    check(before.st_size <= cap and (size is None or before.st_size == size), 'Bounded exact bytes differ')
    data = path.read_bytes()
    after = path.stat()
    check((before.st_ino, before.st_mtime_ns, before.st_size) ==
          (after.st_ino, after.st_mtime_ns, after.st_size), 'File changed across read')
    check(expected_sha is None or sha(data) == expected_sha, 'SHA fingerprint differs')
    return data


def relative_name(value):
    check(isinstance(value, str) and value and ':' not in value and '\\' not in value,
          'Canonical relative name required')
    p = PurePosixPath(value)
    check(not p.is_absolute() and p.as_posix() == value and
          all(x not in ('', '.', '..') for x in p.parts), 'Invalid relative name')
    return value


def decode(data, shape, mode):
    with Image.open(io.BytesIO(data)) as image:
        check(image.format == 'PNG' and image.mode == mode and image.size == shape,
              'Exact PNG geometry/mode required')
        check(getattr(image, 'n_frames', 1) == 1, 'Single frame required')
        return np.asarray(image).copy()


def half_up(value):
    check(np.isfinite(value).all(), 'Finite arithmetic required')
    return np.floor(np.clip(value, 0., 255.) + .5).astype(np.uint8)


def gaussian_kernel(sigma):
    check(type(sigma) in (int, float) and math.isfinite(sigma) and sigma >= 0, 'Invalid sigma')
    if sigma == 0:
        return np.ones(1, np.float64)
    radius = int(3. * sigma + .5)
    # Explicit scalar mathematical kernel: no scipy filter or producer helper.
    kernel = np.asarray([math.exp(-.5 * (d / sigma)**2)
                         for d in range(-radius, radius + 1)], dtype=np.float64)
    return kernel / kernel.sum()


def correlate_axis(value, kernel, axis):
    radius = len(kernel) // 2
    if radius == 0:
        return value.copy()
    pads = [(0, 0)] * value.ndim
    pads[axis] = (radius, radius)
    # ndimage mode=reflect is half-sample symmetric, equivalent to symmetric pad.
    padded = np.pad(value, pads, mode='symmetric')
    answer = np.zeros_like(value, dtype=np.float64)
    for offset, weight in enumerate(kernel):
        slices = [slice(None)] * value.ndim
        slices[axis] = slice(offset, offset + value.shape[axis])
        answer += padded[tuple(slices)] * weight
    return answer


def gaussian(value, sx, sy):
    result = correlate_axis(value.astype(np.float64), gaussian_kernel(sy), 0)
    result = correlate_axis(result, gaussian_kernel(sx), 1)
    return half_up(result)


def mosaic(value, cell):
    h, w, _ = value.shape
    check(type(cell) is int and cell > 0 and h % cell == w % cell == 0, 'Invalid cell grid')
    result = np.empty_like(value)
    # Independent block loop, not producer reshape/reduce/repeat implementation.
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            mean = value[y:y+cell, x:x+cell].astype(np.float64).sum(axis=(0, 1)) / (cell * cell)
            result[y:y+cell, x:x+cell] = half_up(mean)
    return result


def reconstruct(clear, profile):
    check(clear.dtype == np.uint8 and clear.shape == (256, 256, 3), 'RGB256 input required')
    _, operation, parameters, _ = profile
    result = clear.copy()
    footprint = np.zeros((256, 256), np.uint8)
    if operation == 'identity':
        return result, footprint
    if operation in ('gaussian', 'local_gaussian'):
        result = gaussian(clear, parameters['sigma_x'], parameters['sigma_y'])
    elif operation == 'repeated_rows':
        for y in range(256):
            source_y = y - y % parameters['block_rows']
            result[y] = half_up(clear[y].astype(np.float64) * (1 - parameters['repeat_weight']) +
                                clear[source_y].astype(np.float64) * parameters['repeat_weight'])
    elif operation == 'mosaic':
        result = mosaic(clear, parameters['cell'])
    elif operation == 'local_mosaic':
        x0, y0, x1, y1 = RECT
        result[y0:y1, x0:x1] = mosaic(clear[y0:y1, x0:x1], parameters['cell'])
    else:
        raise ValueError('Undeclared operator')
    if operation.startswith('local_'):
        x0, y0, x1, y1 = RECT
        footprint[y0:y1, x0:x1] = 255
        result[footprint == 0] = clear[footprint == 0]
    else:
        footprint[:] = 255
    return result, footprint


def pixel_pin(pixels):
    return {'shape': list(pixels.shape), 'dtype': str(pixels.dtype),
            'RGB8_sha256': sha(pixels.tobytes()),
            'float32_sha256': sha((pixels.astype(np.float32) / np.float32(255)).tobytes())}


def audit(folder, manifest_sha, out):
    check(not any(x in sys.modules for x in ('torch', 'scipy', 'tensorflow', 'jax')),
          'Independent audit may import no neural/producer/Scipy modules')
    folder = Path(folder).resolve(strict=True)
    check(folder.is_relative_to(ROOT / 'outputs') and folder.is_dir(), 'Saved output folder required')
    out = Path(out).absolute()
    check(out.parent.resolve(strict=True).is_relative_to(ROOT / 'outputs') and not out.exists(),
          'New exclusive audit directory required')
    started = time.monotonic()
    raw = bound(folder / 'manifest.json', manifest_sha)
    doc = strict_json(raw)
    check(doc['complete'] is True and doc['format'] == 'codeformer-fixed-filter-pair-audit-v1',
          'Completed expected manifest required')
    check(doc['pairs'] == 96 and doc['TRAIN_pairs'] == doc['DEV_pairs'] == 48 and
          doc['reference_photographs'] == 8 and doc['profiles'] == list(PROFILE_ROWS), 'Exact fixed census/profile contract')
    files = doc['files']
    check(0 < len(files) < 256 and sum(p['bytes'] for p in files.values()) < 64*1024**2,
          'Finite namespace caps')
    cache = {}
    for name, pin in files.items():
        relative_name(name)
        check(set(pin) == {'path', 'sha256', 'bytes'} and pin['path'] == name, 'Exact file pin schema')
        cache[name] = bound(folder / name, pin['sha256'], pin['bytes'])
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    check(actual == set(files) | {'manifest.json'}, 'Closed output file namespace differs')
    for p in folder.rglob('*'):
        if p.is_dir():
            check(not p.is_symlink() and not getattr(p.lstat(), 'st_file_attributes', 0) & 1024,
                  'Linked/reparse output directory')
    selection = strict_json(cache['cohort_selection.json'])
    check(doc['selection_sha256'] == SELECT_SHA == sha(cache['cohort_selection.json']), 'Locked selection SHA')
    conf = strict_json(cache['approved_profiles.json'])
    check(doc['approved_profiles_sha256'] == sha(cache['approved_profiles.json']) and
          conf['profiles'] == list(PROFILE_ROWS) and conf['local_rectangle_xyxy_half_open'] == list(RECT),
          'Approved profile document differs')
    check(conf['Gaussian'] == {'arithmetic': 'float64', 'implementation': 'scipy.ndimage.gaussian_filter',
          'sigma_order': '[sigma_y,sigma_x,0]', 'edge_mode': 'reflect', 'truncate': 3.0}, 'Gaussian arithmetic policy')
    for name, pin in doc['source_pins'].items():
        relative_name(name)
        bound(ROOT / name, pin['sha256'], pin['bytes'])
    check(set(doc['source_pins']) == {'scripts/prepare_codeformer_filter_pairs_v1.py',
          'research/codeformer_filter_pairs_v1/__init__.py', 'research/codeformer_filter_pairs_v1/filters.py',
          'research/codeformer_filter_pairs_v1/README.md'}, 'Exact producer-source namespace')
    catalog_pin = selection['catalog']
    catalog = strict_json(bound(ROOT / catalog_pin['path'], catalog_pin['sha256'], catalog_pin['bytes']))
    refs = catalog['references']
    check(len(refs) == 444 and Counter(r['historical_role'] for r in refs) == {'train':391,'validation':53},
          'Historical catalog census')
    check(not {r['source_sha256'] for r in refs if r['historical_role']=='train'} &
          {r['source_sha256'] for r in refs if r['historical_role']=='validation'}, 'TRAIN/DEV source hash overlap')
    check([r['reference_id'] for r in selection['cohort']] == list(IDS['TRAIN'] + IDS['DEV']), 'Eight prospective IDs')
    for p in selection['supporting_audit_and_approval_manifests']:
        bound(ROOT / p['path'], p['sha256'], p['bytes'])
    support = decode(cache['observed-support.png'], (256,256), 'L')
    check(np.all(support == 255), 'All-pixel observed support differs')

    def get(pin):
        check(pin == files[pin['path']], 'Record file pin differs from manifest')
        return cache[pin['path']]

    case_index = {c['id']:c for c in doc['cases']}
    check(len(case_index) == 96, 'Unique96 cases required')
    reference_index = {r['reference_id']:r for r in doc['references']}
    check(len(reference_index) == 8, 'Eight reference proofs required')
    rows, expected_names = [], {'cohort_selection.json','approved_profiles.json','README.md','observed-support.png',
                                    'input_only_visual_manifest.json'}
    for profile in PROFILES:
        expected_names.add('corruption_footprints/' + profile[0] + '.png')
    expected_names.update('input_only_sheets/%02d_%s.png' % (i, name)
                          for i, name in enumerate(REVIEW_ORDER, 1))
    clear_by_id = {}
    for selected in selection['cohort']:
        rid, split = selected['reference_id'], selected['split']
        check(rid in IDS[split] and selected['historical_role'] == ('train' if split=='TRAIN' else 'validation'),
              'Historical split changed')
        catalog_row = next(r for r in refs if r['id'] == rid)
        check(selected['reference_manifest_row'] == catalog_row and selected['attribution'] == catalog_row['attribution'],
              'Complete original catalog/attribution join')
        sp, tp = selected['source1024'], selected['target256']
        source_data = bound(ROOT / sp['path'], sp['sha256'], sp['bytes'])
        retained_target_data = bound(ROOT / tp['path'], tp['sha256'], tp['bytes'])
        source = decode(source_data, (1024,1024), 'RGB')
        check(sha(source.tobytes()) == sp['rgb8']['pixel_sha256'], 'Source RGB pixel fingerprint')
        with Image.open(io.BytesIO(source_data)) as image:
            clear = np.asarray(image.resize((256,256), Image.Resampling.BILINEAR)).copy()
            clean256 = np.asarray(image.resize((256,256), Image.Resampling.LANCZOS)).copy()
            clean512 = np.asarray(image.resize((512,512), Image.Resampling.LANCZOS)).copy()
        check(np.array_equal(decode(retained_target_data,(256,256),'RGB'),clean256), 'Direct target256 pixels')
        for name, pixels in [('clear256',clear),('target256',clean256),('clean512',clean512)]:
            metadata = selected[name]
            check(sha(pixels.tobytes()) == metadata['rgb8']['pixel_sha256'] and
                  sha((pixels.astype(np.float32)/np.float32(255)).tobytes()) == metadata['float32']['pixel_sha256'],
                  'Direct256/512 source derivation digest')
        expected_names.update({'references/'+rid+'/clean512.png','references/'+rid+'/clean256.png'})
        proof = reference_index[rid]
        check(proof['split']==split and proof['source1024']==sp and proof['attribution']==selected['attribution'] and
              proof['target256_retained_bytes_exact'] is True and
              proof['clean512_direct1024_proof']==pixel_pin(clean512) and
              proof['clean256_direct1024_proof']==pixel_pin(clean256) and
              proof['clear256_direct1024_proof']==pixel_pin(clear), 'Reference proof schema join')
        target_pins = proof['generated_reference_files']
        check(get(target_pins['clean256'])==retained_target_data and
              np.array_equal(decode(get(target_pins['clean512']),(512,512),'RGB'),clean512), 'Saved clean reference exact')
        clear_by_id[rid] = clear
        for profile, profile_row in zip(PROFILES,PROFILE_ROWS):
            name = rid+'__'+profile[0]
            check(name in case_index, 'Missing fixed case')
            ci = case_index[name]
            expected_names.update({'inputs/'+name+'.png','pairs/'+name+'.json'})
            rec = strict_json(get(ci['pair_record']))
            check(ci['reference_id']==rid and ci['split']==split and ci['profile']==profile[0] and
                  ci['input_review']=='PENDING', 'Index/source/role binding')
            check(rec['format']=='codeformer-filter-synthetic-pair-v1' and rec['id']==name and
                  rec['reference_id']==rid and rec['split']==split and rec['historical_role']==selected['historical_role'] and
                  rec['source1024']==sp and rec['attribution']==selected['attribution'] and
                  rec['profile']==profile_row and rec['evidence']=='paired_synthetic_fixed_filter_stress',
                  'Pair source/profile/terms binding')
            check(rec['observed_input']==ci['input'] and rec['targets']==target_pins and
                  rec['target512_pixels']==pixel_pin(clean512) and rec['target256_pixels']==pixel_pin(clean256) and
                  rec['observed_support']==files['observed-support.png'], 'Pair target/support binding')
            predicted, footprint = reconstruct(clear,profile)
            actual_pixels = decode(get(rec['observed_input']),(256,256),'RGB')
            mismatch = int(np.count_nonzero(actual_pixels != predicted))
            check(mismatch==0, 'Independent filter reconstruction differs: '+name+' channels='+str(mismatch))
            check(rec['input256_pixels']==pixel_pin(actual_pixels), 'Input RGB8/float32 fingerprints')
            check(rec['corruption_footprint']==files['corruption_footprints/'+profile[0]+'.png'] and
                  np.array_equal(decode(get(rec['corruption_footprint']),(256,256),'L'),footprint) and
                  rec['corruption_footprint_pixels_sha256']==sha(footprint.tobytes()), 'Footprint pixels/fingerprints')
            check(rec['corruption_footprint_is_not_completion_or_removal_mask'] is True and
                  rec['local_rectangle_xyxy_half_open']==(list(RECT) if profile[1].startswith('local_') else None),
                  'Corruption footprint is not completion mask')
            expected_seed = int.from_bytes(hashlib.sha256(('codeformer-filter-pairs-v1/'+rid+'/'+profile[0]).encode('ascii')).digest()[:4],'big')
            check(rec['seed']==expected_seed and rec['RNG_calls']==0, 'Seed metadata/no RNG policy')
            check(rec['input_only_review']=={'status':'PENDING','decision':None,'reviewer':None,'input_only':True,'model_output_used':False}
                  and rec['automatic_usable_or_insufficient_label'] is None, 'No automatic input label permitted')
            eligibility = ('EXCLUDED_DEV' if split == 'DEV' else
                           'EXCLUDED_INITIALLY_STRONG_CONTROL' if profile[3] else 'PENDING_INPUT_ONLY_REVIEW')
            check(rec['restoration_training_eligibility']==ci['restoration_training_eligibility']==eligibility and
                  rec['optimization_allowed_by_split']==ci['optimization_allowed_by_split']==(split=='TRAIN') and
                  rec['optimization_permitted_now'] is ci['optimization_permitted_now'] is False and
                  rec['actual_training_performed'] is False and rec['models_or_output_based_parameters'] is False and
                  rec['user_wild_images_used'] is False and rec['reserved_FINAL_used'] is False,
                  'Initial preparation only; no optimization admission')
            check(rec['geometry']=={'crop':None,'warp':None,'face_alignment':None,'observed_support':'all256pixels',
                  'targets':'clean512 and clean256 direct1024 LANCZOS, never input256 upsampling'}, 'No geometry transformation')
            rows.append({'id':name,'split':split,'profile':profile[0],'input256_sha256':sha(actual_pixels.tobytes()),
                         'independent_pixel_mismatches':mismatch,'local_outside_unchanged':bool(np.array_equal(actual_pixels[footprint==0],clear[footprint==0]))})
    visual = strict_json(cache['input_only_visual_manifest.json'])
    check(visual['complete'] and visual['input_cases']==visual['exact256_cells_readback']==96 and
          len(visual['cells'])==96 and len(visual['sheets'])==12 and
          visual['root_input_only_review_status']=='PENDING' and visual['no_automatic_input_quality_labels'] is True,
          'Input-only sheet manifest census')
    check(visual['prospective_review_order']==list(REVIEW_ORDER) and
          visual['clear_profile_withheld_until_last'] is True and
          visual['blinded_to_other_degraded_versions'] is False and
          visual['clean_reference_or_model_output_panels']==0, 'Input-only review ordering/cue limits')
    check(doc['input_only_sheets']==visual['sheets'], 'Top-level input-only sheet pins differ')
    cells_seen=set()
    for review_index,sheet in enumerate(visual['sheets'],1):
        pixels=decode(get({k:sheet[k] for k in ('path','sha256','bytes')}),(1024,612),'RGB')
        cells=[c for c in visual['cells'] if c['sheet']==sheet['path']]
        check(len(cells)==8 and sheet['cases']==8 and sheet['profile']==REVIEW_ORDER[review_index-1] and
              sheet['review_index']==review_index and sheet['width']==1024 and sheet['height']==612 and
              [c['reference_id'] for c in cells]==list(IDS['TRAIN']+IDS['DEV']), 'Sheet eight prospective cells')
        for cell in cells:
            key=cell['reference_id']+'__'+cell['profile'];check(key not in cells_seen,'Duplicate visual cell');cells_seen.add(key)
            check(cell['profile']==sheet['profile'] and cell['reference_id'] in clear_by_id and
                  cell['split']==case_index[key]['split'] and cell['width']==cell['height']==256,'Exact256 cell metadata')
            tile=pixels[cell['y']:cell['y']+256,cell['x']:cell['x']+256]
            check(tile.shape==(256,256,3) and sha(tile.tobytes())==cell['RGB8_sha256']==
                  next(r['input256_sha256'] for r in rows if r['id']==key),'Sheet cell exact input pixels')
    check(cells_seen==set(case_index) and set(files)==expected_names,'Exact fixed output namespace')
    check(doc['all_input_reviews_pending'] is True and doc['strong_control_cases_excluded_initially_from_training']==40 and
          doc['TRAIN_strong_control_cases_excluded_initially_from_training']==20 and
          doc['DEV_cases_excluded_from_optimization_regardless_of_input_review']==48 and
          doc['TRAIN_nonstrong_cases_pending_input_review_before_any_training_admission']==28 and
          doc['new_model_or_inference_or_training_calls']==0 and doc['input_quality_threshold_calibrated'] is False and
          doc['user_images_native_CCTV_FINAL_used'] is False and doc['historical_membership_or_quality_gates_changed'] is False,
          'Preparation scope/claims differ')
    check(not any(x in sys.modules for x in ('torch','scipy','tensorflow','jax')),'No producer/neural import')
    receipt={'format':'codeformer-filter-pairs-independent-saved-pixel-audit-v1','complete':True,'pass':True,
        'UTC':datetime.now(timezone.utc).isoformat(),'manifest_sha256':sha(raw),'manifest_bytes':len(raw),
        'manifest_path':str(folder/'manifest.json'),'auditor_sha256':sha(Path(__file__).read_bytes()),
        'pairs':96,'TRAIN_pairs':48,'DEV_pairs':48,'reference_photographs':8,'profiles':12,
        'all96_independent_exact_input_pixels':True,'all96_RGB8_and_float32_fingerprints':True,
        'all16_direct1024_clean256_clean512_reference_pixels':True,'all96_footprints_and_support_exact':True,
        'all96_exact_sheet_cells':True,'output_files':len(files)+1,'output_payload_bytes':sum(p['bytes'] for p in files.values())+len(raw),
        'closed_namespace_hashes':True,'source_cohort_catalog_attribution_exact':True,
        'independent_Gaussian':'math.exp normalized kernels radius=int(3*sigma+.5); NumPy half-sample symmetric pad; explicit y/x float64 correlation; half-up quantization; no Scipy/producer import',
        'arithmetic_tolerance_for_saved_pixels':0,'cases':rows,'operator_input_review':'All96 PENDING at generation; root review is a separate later artifact.',
        'optimization_admission':'None. DEV never for optimization;40strong controls initially excluded; remaining cases await separate review/design.',
        'no_automatic_refusal_threshold':True,'new_NN_autograd_optimizer_VM_calls':0,'returned_code_executed':False,
        'producer_imported':False,'new_images_written':0,'model_qualified':False,'goal_complete':False,
        'versions':{'Python':sys.version,'NumPy':np.__version__,'Pillow':PILLOW_VERSION},'seconds':time.monotonic()-started}
    out.mkdir()
    with (out/'saved_pixel_audit_v1.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    return {'complete':True,'pass':True,'pairs':96,'receipt':str(out/'saved_pixel_audit_v1.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest-sha',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.root,args.manifest_sha,args.out)),flush=True)
