"""Frozen full TRAIN input/target review. No neural imports or training."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2'
SCHEDULE = ROOT/'outputs/cctv_dgp_profile_batches_vm_v31/protocol.json'
NATIVE = ROOT/'outputs/cctv_chokepoint_native_development_v1'
OUT = ROOT/'outputs/cctv_dgp_full_training_coverage_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): digest.update(block)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def pixels(path,mode='RGB'):
    with Image.open(path) as image:
        assert image.size == (256,256)
        return np.asarray(image.convert(mode)).copy()


def pure_quality():
    tree = ast.parse((ROOT/'face_workflow.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'quality_signals']
    assert len(nodes) == 1
    module = ast.Module(body=nodes,type_ignores=[]); ast.fix_missing_locations(module)
    namespace = {'cv2':cv2,'np':np}
    exec(compile(module,'pinned-quality-signals','exec'),namespace)
    return namespace['quality_signals']


def supports(mask,landmarks):
    valid = cv2.erode(mask.astype(np.uint8),np.ones((13,13),np.uint8),
        borderType=cv2.BORDER_CONSTANT,borderValue=0) > 0
    patches = np.zeros((256,256),bool)
    for x,y in np.floor(np.asarray(landmarks,np.float64)).astype(int):
        patches[max(0,y-12):min(256,y+12),max(0,x-12):min(256,x+12)] = True
    patches &= valid
    assert patches.any() and valid.any()
    return valid,patches


def high(rgb):
    positions = np.arange(-6,7,dtype=np.float64)
    kernel = np.exp(-.5*(positions/2)**2); kernel /= kernel.sum()
    luma = (rgb.astype(np.float64)/255*np.array([.299,.587,.114])).sum(2)
    return luma-convolve1d(convolve1d(luma,kernel,axis=0,mode='reflect'),kernel,axis=1,mode='reflect')


def distribution(values):
    values = [v for v in values if v is not None]
    return None if not values else {'count':len(values),'minimum':min(values),
        'q25':float(np.percentile(values,25)),'median':float(np.median(values)),
        'q75':float(np.percentile(values,75)),'maximum':max(values)}


def grouped(rows,by):
    groups = {}
    for key in sorted({'/'.join(row[k] for k in by) for row in rows}):
        selected = [r for r in rows if '/'.join(r[k] for k in by) == key]
        groups[key] = {'cases':len(selected),'Auto_restore_suggestions':sum(r['quality']['suggest_restoration'] for r in selected),
            **{name:distribution([r[name] for r in selected]) for name in ['geometric_eye_spacing_pixels',
                'input_landmark_HF_energy','input_landmark_HF_MSE_to_target','input_MSE_to_target']},
            **{name:distribution([r['quality'][name] for r in selected]) for name in ['blur_variance','noise_sigma_255']}}
    return groups


def prepare():
    assert not OUT.exists(), 'Preserve every prior review or failure'
    p,q = read(BASE/'mixed_protocol_v9.json'),read(SCHEDULE)
    assert sha(BASE/'mixed_protocol_v9.json') == q['mixed_data_protocol_sha256']
    cases = q['case_rows']; refs = [r for r in p['references'] if r['role'] == 'train']
    assert len(cases) == 3905 and len(refs) == 781 and all(c['role'] == 'train' for c in cases)
    assert {c['source_person_or_reference'] for c in cases} == {r['id'] for r in refs}
    assert sum(r.get('evaluation_source_kind') == 'HQ_FFHQ_counterpart' for r in refs) == 391
    assert len(q['mixed_TRAIN_assets_sha256']) == 5467
    native = read(NATIVE/'frozen_subset.json'); reviewed = read(NATIVE/'input_review.json')
    nc = [c for c in native['cases'] if c['role'] == 'development']
    assert len(nc) == reviewed['usable_cases'] == 24 and reviewed['reviewed_before_model_outputs']
    assert reviewed['subset_sha256'] == sha(NATIVE/'frozen_subset.json')
    bindings = {}
    def bind(path,digest=None):
        actual = sha(path)
        if digest is not None: assert actual == digest,path
        bindings[Path(path).relative_to(ROOT).as_posix()] = actual
    for name,digest in q['mixed_TRAIN_assets_sha256'].items(): bind(BASE/name,digest)
    for c in nc:
        for key in ['input','observed']: bind(NATIVE/c[key],c[key+'_sha256'])
    for path in [Path(__file__),ROOT/'scripts/audit_cctv_dgp_full_training_coverage_v1.py',ROOT/'face_workflow.py',
        BASE/'mixed_protocol_v9.json',SCHEDULE,NATIVE/'frozen_subset.json',NATIVE/'input_review.json',
        ROOT/'outputs/cctv_dgp_normfix_return_v2/protocol.json',ROOT/'scripts/run_cctv_dgp_pilot_vm.py',
        ROOT/'cctv_dgp_pilot.py',ROOT/'cctv_dgp_frozen_norm.py',
        ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm/protocol.json',
        ROOT/'outputs/cctv_dgp_post_finite_guard_parameter_inventory_v1/inventory.json']:
        bind(path)
    OUT.mkdir()
    write(OUT/'plan.json',{'complete':True,'prepared_before_measurements':True,'source_bindings':bindings,
        'TRAIN_case_ids':[c['id'] for c in cases],'TRAIN_reference_ids':[r['id'] for r in refs],
        'native_development_case_ids':[c['id'] for c in nc],
        'historical_reference_hashes':p['legacy_metadata_note'],
        'canonical_target_hash_policy':'mixed_protocol_v9.canonical_target_rgb_sha256; not legacy thumbnail metadata',
        'existing_Auto_thresholds':{'blur':24,'noise':8},'quality_function':'Exact current pure quality_signals AST',
        'HF_energy':'Float64 RGB/255 luma high pass; retained sigma2,13tap separable reflect kernel',
        'filter_recomputation_arithmetic_atol':1e-12,'worker_seconds':420,'audit_seconds':420,
        'previous_completed_turn_classification':'progress: verified return, conversion and input coverage evidence',
        'previous_interrupted_turn_classification':'progress: parameter inventory and verified historical learning settings',
        'model_forwards':0,'gradient_queries':0,'optimizer_updates':0,'threshold_fitting':False,
        'validation_or_reserved_final_pixels_decoded':0,'native_target_or_paired_metric':None,
        'prepared_UTC':datetime.now(timezone.utc).isoformat()})
    print({'prepared':True,'source_bindings':len(bindings),'TRAIN_cases':3905,'TRAIN_references':781},flush=True)


def run():
    start = time.monotonic(); plan = read(OUT/'plan.json')
    assert not (OUT/'results.json').exists() and not (OUT/'failure.json').exists()
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    p,q = read(BASE/'mixed_protocol_v9.json'),read(SCHEDULE); quality = pure_quality()
    refs = {r['id']:r for r in p['references'] if r['role'] == 'train'}
    by_ref = {r['id']:[c for c in q['case_rows'] if c['source_person_or_reference'] == r['id']] for r in refs.values()}
    rows,target_rows = [],[]
    for number,(rid,ref) in enumerate(refs.items(),1):
        assert time.monotonic()-start < plan['worker_seconds']
        target = pixels(BASE/ref['target']); mask = pixels(BASE/ref['observed'],'L') > 0
        digest = hashlib.sha256(target.tobytes()).hexdigest()
        assert digest == p['canonical_target_rgb_sha256'][rid],rid
        group = by_ref[rid]
        assert len(group) == 5 and len({c['profile'] for c in group}) == 5
        assert all(c['source'] == ref['source'] and c['target'] == ref['target'] and c['observed'] == ref['observed'] for c in group)
        valid,feature = supports(mask,group[0]['landmarks5_canvas_xy']); th = high(target)
        target_rows.append({'id':rid,'source':ref['source'],'role':'train',
            'target_rgb_sha256':digest,'canonical_RGB_hash_verified':True,
            'legacy_metadata_RGB_hash_matches':digest == ref['target_rgb_sha256'],
            'target_kind':ref.get('evaluation_source_kind') or 'original_photographic_replay',
            'historical_native_size':ref['native_size'],'HQ_source_sha256':ref.get('hq_source_sha256'),
            'landmark_HF_energy':float(np.square(th)[feature].mean()),
            'observed_interior_HF_energy':float(np.square(th)[valid].mean()),
            'quality':quality(target,(~mask).astype(np.uint8))})
        for c in group:
            image = pixels(BASE/c['input']); ih = high(image)
            eyes = np.asarray(c['landmarks5_canvas_xy'],np.float64)[:2]
            size = c['proxy_details']['low_resolution_size'] or c['proxy_details']['working_size']
            scale = np.asarray(size,np.float64)/np.asarray(c['proxy_details']['working_size'],np.float64)
            delta = image.astype(np.float32)/np.float32(255)-target.astype(np.float32)/np.float32(255)
            rows.append({'id':c['id'],'reference':rid,'role':'train','source':c['source'],'profile':c['profile'],
                'sampling_grid_size':size,'geometric_eye_spacing_pixels':float(np.linalg.norm((eyes[1]-eyes[0])*scale)),
                'quality':quality(image,(~mask).astype(np.uint8)),
                'clear_input_exact_target':bool(np.array_equal(image,target)) if c['profile'] == 'clear' else None,
                'input_landmark_HF_energy':float(np.square(ih)[feature].mean()),
                'input_landmark_HF_MSE_to_target':float(np.square(ih-th)[feature].mean()),
                'input_MSE_to_target':float(np.square(delta[mask]).astype(np.float64).mean())})
        if number % 100 == 0: print({'references':number,'of':781,'inputs':len(rows)},flush=True)
    by_id = {r['id']:r for r in rows}; rows = [by_id[cid] for cid in plan['TRAIN_case_ids']]
    native_rows = []
    for c in read(NATIVE/'frozen_subset.json')['cases']:
        if c['role'] != 'development': continue
        image = pixels(NATIVE/c['input']); mask = pixels(NATIVE/c['observed'],'L') > 0
        valid,_ = supports(mask,[[128,128]]); energy = high(image)
        native_rows.append({'id':c['id'],'role':'development','source':'ChokePoint/P1E_S1_C1','profile':'native',
            'native_size':[c['native_width'],c['native_height']],
            'geometric_eye_spacing_pixels':c['inter_eye_distance_native'],
            'quality':quality(image,(~mask).astype(np.uint8)),
            'input_landmark_HF_energy':None,'input_landmark_HF_MSE_to_target':None,'input_MSE_to_target':None,
            'input_observed_interior_HF_energy':float(np.square(energy)[valid].mean()),'paired_target':None})
    assert len(rows) == 3905 and len(target_rows) == 781 and len(native_rows) == 24
    optimizer = read(ROOT/'outputs/cctv_dgp_normfix_return_v2/protocol.json')['optimizer']
    assert optimizer['backbone_lr'] == 2e-6 and optimizer['head_lr'] == 1e-5 and optimizer['normalization_running_statistics_frozen']
    result = {'complete':True,'plan_sha256':sha(OUT/'plan.json'),'TRAIN_rows':rows,'TRAIN_targets':target_rows,
        'native_input_rows':native_rows,'TRAIN_source_profile_groups':grouped(rows,['source','profile']),
        'native_input_groups':grouped(native_rows,['source','profile']),
        'target_source_summaries':{source:{'references':sum(r['source']==source for r in target_rows),
            'legacy_hash_mismatches':sum(r['source']==source and not r['legacy_metadata_RGB_hash_matches'] for r in target_rows),
            'target_landmark_HF_energy':distribution([r['landmark_HF_energy'] for r in target_rows if r['source']==source]),
            'target_blur_variance':distribution([r['quality']['blur_variance'] for r in target_rows if r['source']==source]),
            'native_min_edge':distribution([min(r['historical_native_size']) for r in target_rows if r['source']==source])}
            for source in sorted({r['source'] for r in target_rows})},
        'retained_identity_v2_optimizer':optimizer,'canonical_target_hashes_verified':781,
        'model_forwards':0,'gradient_queries':0,'optimizer_updates':0,'epochs':0,
        'validation_or_reserved_final_pixels_decoded':0,'threshold_fitting':False,'native_is_unpaired':True,
        'quality_heuristics_establish_face_sufficiency':False,'source_or_ethnicity_inferred':False,
        'app_changed':False,'model_qualification':False,'goal_complete':False,'seconds':time.monotonic()-start}
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    assert result['seconds'] < plan['worker_seconds']; write(OUT/'results.json',result)
    print({'complete':True,'TRAIN_cases':3905,'TRAIN_targets':781,'native_inputs':24,'seconds':time.monotonic()-start},flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); args = parser.parse_args()
    if args.prepare: prepare()
    else:
        try: run()
        except Exception:
            import traceback
            if OUT.exists() and not (OUT/'failure.json').exists():
                write(OUT/'failure.json',{'complete':False,'traceback':traceback.format_exc(),
                    'model_forwards':0,'gradient_queries':0,'optimizer_updates':0,'goal_complete':False})
            raise


if __name__ == '__main__': main()
