"""Prospective independent geometry, canonical-pixel and OpenCV-filter review."""
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2'
NATIVE = ROOT/'outputs/cctv_chokepoint_native_development_v1'
OUT = ROOT/'outputs/cctv_dgp_full_training_coverage_v1'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pixels(path,mode='RGB'):
    with Image.open(path) as image:
        assert image.size == (256,256)
        return np.asarray(image.convert(mode)).copy()


def high(rgb):
    x = np.linspace(-6,6,13,dtype=np.float64)
    kernel = np.exp(-x*x/8); kernel /= kernel.sum()
    luma = np.sum(rgb.astype(np.float64)/255*np.array([.299,.587,.114]),axis=2)
    return luma-cv2.sepFilter2D(luma,cv2.CV_64F,kernel,kernel,borderType=cv2.BORDER_REFLECT)


def dist(values):
    values = [v for v in values if v is not None]
    return None if not values else {'count':len(values),'minimum':min(values),'q25':float(np.percentile(values,25)),
        'median':float(np.median(values)),'q75':float(np.percentile(values,75)),'maximum':max(values)}


def main():
    start = time.monotonic(); plan = read(OUT/'plan.json'); result = read(OUT/'results.json')
    assert not (OUT/'independent_audit.json').exists()
    assert result['complete'] and result['plan_sha256'] == sha(OUT/'plan.json')
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    p = read(BASE/'mixed_protocol_v9.json'); schedule = read(ROOT/'outputs/cctv_dgp_profile_batches_vm_v31/protocol.json')
    refs = {r['id']:r for r in p['references'] if r['role']=='train'}
    targets = {r['id']:r for r in result['TRAIN_targets']}; cases = {c['id']:c for c in schedule['case_rows']}
    assert set(targets) == set(refs) and [r['id'] for r in result['TRAIN_rows']] == plan['TRAIN_case_ids']
    assert len(cases) == len(result['TRAIN_rows']) == 3905 and len(targets) == 781
    cache = {}; worst = 0.; checked = 0
    for rid,ref in refs.items():
        target = pixels(BASE/ref['target']); observed = pixels(BASE/ref['observed'],'L') > 0
        assert hashlib.sha256(target.tobytes()).hexdigest() == targets[rid]['target_rgb_sha256'] == p['canonical_target_rgb_sha256'][rid]
        assert targets[rid]['canonical_RGB_hash_verified']
        assert targets[rid]['legacy_metadata_RGB_hash_matches'] == (targets[rid]['target_rgb_sha256']==ref['target_rgb_sha256'])
        first = next(c for c in schedule['case_rows'] if c['source_person_or_reference']==rid)
        yy,xx = np.mgrid[:256,:256]; near = np.zeros((256,256),bool)
        for x,y in np.floor(first['landmarks5_canvas_xy']).astype(int): near |= (xx>=x-12)&(xx<x+12)&(yy>=y-12)&(yy<y+12)
        valid = cv2.erode(observed.astype(np.uint8),np.ones((13,13),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        feature = near&valid; assert feature.any()
        hf = high(target); energy = float(np.square(hf)[feature].mean())
        worst = max(worst,abs(energy-targets[rid]['landmark_HF_energy']),
            abs(float(np.square(hf)[valid].mean())-targets[rid]['observed_interior_HF_energy']))
        cache[rid] = (target,observed,feature,hf)
    for row in result['TRAIN_rows']:
        assert time.monotonic()-start < plan['audit_seconds']
        c = cases[row['id']]; target,observed,feature,th = cache[c['source_person_or_reference']]
        image = pixels(BASE/c['input']); ih = high(image)
        assert row['role'] == c['role'] == 'train' and row['source'] == c['source'] and row['profile'] == c['profile']
        dx,dy = np.asarray(c['landmarks5_canvas_xy'],np.float64)[1]-np.asarray(c['landmarks5_canvas_xy'],np.float64)[0]
        size = c['proxy_details']['low_resolution_size'] or c['proxy_details']['working_size']
        w,h = c['proxy_details']['working_size']
        spacing = float(np.sqrt((dx*size[0]/w)**2+(dy*size[1]/h)**2))
        assert abs(spacing-row['geometric_eye_spacing_pixels']) <= 1e-12 and row['sampling_grid_size'] == size
        worst = max(worst,abs(float(np.square(ih)[feature].mean())-row['input_landmark_HF_energy']),
            abs(float(np.square(ih-th)[feature].mean())-row['input_landmark_HF_MSE_to_target']))
        error = image.astype(np.float32)/255-target.astype(np.float32)/255
        assert float(np.square(error[observed]).astype(np.float64).mean()) == row['input_MSE_to_target']
        if c['profile'] == 'clear': assert row['clear_input_exact_target'] == bool(np.array_equal(image,target))
        s = row['quality']; assert s['blur_threshold'] == 24 and s['noise_threshold'] == 8
        assert s['suggest_restoration'] == bool(s['blur_variance'] is not None and (s['blur_variance']<24 or s['noise_sigma_255']>=8))
        checked += 1
    native = {c['id']:c for c in read(NATIVE/'frozen_subset.json')['cases'] if c['role']=='development'}
    assert [r['id'] for r in result['native_input_rows']] == plan['native_development_case_ids']
    for row in result['native_input_rows']:
        c = native[row['id']]; eyes = np.asarray(c['eyes'],np.float64)
        assert abs(float(np.linalg.norm(eyes[1]-eyes[0]))-row['geometric_eye_spacing_pixels']) <= 1e-12
        assert row['paired_target'] is None and row['input_MSE_to_target'] is row['input_landmark_HF_MSE_to_target'] is None
    for source_profile,group in result['TRAIN_source_profile_groups'].items():
        chosen = [r for r in result['TRAIN_rows'] if r['source']+'/'+r['profile']==source_profile]
        assert group['cases']==len(chosen) and group['Auto_restore_suggestions']==sum(r['quality']['suggest_restoration'] for r in chosen)
        for name in ['geometric_eye_spacing_pixels','input_landmark_HF_energy','input_landmark_HF_MSE_to_target','input_MSE_to_target']:
            assert group[name] == dist([r[name] for r in chosen])
        for name in ['blur_variance','noise_sigma_255']: assert group[name]==dist([r['quality'][name] for r in chosen])
    assert worst <= plan['filter_recomputation_arithmetic_atol'] and checked == 3905
    assert result['canonical_target_hashes_verified']==781 and result['validation_or_reserved_final_pixels_decoded']==0
    for name in ['model_forwards','gradient_queries','optimizer_updates','epochs']: assert result[name] == 0
    for name in ['threshold_fitting','app_changed','model_qualification','goal_complete']: assert result[name] is False
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    receipt = {'complete':True,'plan_sha256':sha(OUT/'plan.json'),'results_sha256':sha(OUT/'results.json'),
        'checker_sha256':sha(Path(__file__)),'source_bindings_verified':len(plan['source_bindings']),
        'canonical_TRAIN_targets_verified':781,'all_TRAIN_input_geometry_and_pixel_metrics_verified':3905,
        'native_unpaired_inputs_and_geometry_verified':24,'maximum_independent_filter_arithmetic_error':worst,
        'independent_quality_filter_algorithm_reproduction':False,'quality_signal_sufficiency_proven':False,
        'model_forwards':0,'gradient_queries':0,'optimizer_updates':0,'validation_or_reserved_pixels_decoded':0,
        'model_qualification':False,'goal_complete':False,'seconds':time.monotonic()-start}
    with (OUT/'independent_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print({'complete':True,'TRAIN_inputs':checked,'targets':781,'maximum_filter_error':worst,'seconds':time.monotonic()-start},flush=True)


if __name__ == '__main__': main()
