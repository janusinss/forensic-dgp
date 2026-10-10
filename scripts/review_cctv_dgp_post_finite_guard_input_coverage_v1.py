"""Input-only coverage review. No model, target pixels, fitting or final images."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm'
NATIVE = ROOT/'outputs/cctv_chokepoint_native_development_v1'
OLD = ROOT/'outputs/cctv_dgp_v38_quarter_native_development_v1'
OUT = ROOT/'outputs/cctv_dgp_post_finite_guard_input_coverage_v1'
PIN = 'efdd62759213136114a56f0aaa256278753cac8bac4bc6c6a52f1f8b7550598f'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def pixels(path,mode='RGB'):
    with Image.open(path) as image:
        assert image.size == (256,256)
        return np.asarray(image.convert(mode)).copy()


def quality_function():
    # Execute only the unchanged pure filter function; do not import model loaders.
    source = ast.parse((ROOT/'face_workflow.py').read_text(encoding='utf-8'))
    selected = [n for n in source.body if isinstance(n,ast.FunctionDef) and n.name == 'quality_signals']
    assert len(selected) == 1
    code = ast.Module(body=selected,type_ignores=[]); ast.fix_missing_locations(code)
    namespace = {'np':np,'cv2':cv2}
    exec(compile(code,'frozen-quality-signals','exec'),namespace)
    return namespace['quality_signals']


def summary(rows):
    result = {}
    for key in sorted({r['cohort']+'/'+r['source']+'/'+r['profile'] for r in rows}):
        selected = [r for r in rows if r['cohort']+'/'+r['source']+'/'+r['profile'] == key]
        stats = {}
        for name in ['blur_variance','noise_sigma_255']:
            values = [r['quality'][name] for r in selected if r['quality'][name] is not None]
            stats[name] = None if not values else {'minimum':min(values),'median':float(np.median(values)),'maximum':max(values)}
        eyes = [r['geometric_eye_spacing_pixels'] for r in selected]
        result[key] = {'cases':len(selected),'Auto_restore_suggestions':sum(r['quality']['suggest_restoration'] for r in selected),
            'geometric_eye_spacing_pixels':{'minimum':min(eyes),'median':float(np.median(eyes)),'maximum':max(eyes)},**stats}
    return result


def prepare():
    assert not OUT.exists(); assert sha(BUNDLE/'protocol.json') == PIN
    p = read(BUNDLE/'protocol.json'); old = read(OLD/'plan.json'); subset = read(NATIVE/'frozen_subset.json')
    cases = [c for c in subset['cases'] if c['role'] == 'development']
    assert cases == old['cases'] and len(cases) == 24
    review = read(NATIVE/'input_review.json')
    assert review['usable_cases'] == 24 and review['reviewed_before_model_outputs']
    assert review['subset_sha256'] == sha(NATIVE/'frozen_subset.json')
    bindings = {}
    def bind(path,digest=None):
        value = sha(path)
        if digest is not None: assert value == digest,path
        bindings[Path(path).relative_to(ROOT).as_posix()] = value
    for path in [Path(__file__),ROOT/'face_workflow.py',ROOT/'cctv_input_quality.py',BUNDLE/'protocol.json',
                 OLD/'plan.json',NATIVE/'frozen_subset.json',NATIVE/'input_review.json']:
        bind(path)
    for c in p['cases']:
        for name in ['input','observed']: bind(BUNDLE/c[name],p['assets_sha256'][c[name]])
    for c in cases:
        for name in ['input','observed']: bind(NATIVE/c[name],c[name+'_sha256'])
    assert sha(ROOT/'cctv_input_quality.py') == old['sources_sha256']['cctv_input_quality.py']
    OUT.mkdir()
    write(OUT/'plan.json',{'complete':True,'prepared_before_new_measurements':True,
        'source_bindings':bindings,'parent_protocol_sha256':PIN,'TRAIN_cases':[c['id'] for c in p['cases']],
        'native_cases':[c['id'] for c in cases],'existing_Auto_thresholds':{'blur':24,'noise':8},
        'TRAIN_geometry':'Saved paired-reference five-point coordinates scaled to the synthetic sampling grid; descriptive only, never an inference input',
        'native_geometry':'Published native eye coordinates; file dimensions do not establish usable detail',
        'quality_function':'Exact AST of existing face_workflow.quality_signals; only input pixels and ignored padding',
        'historical_transitive_face_workflow_binding_absent':True,'geometric_recompute_absolute_tolerance':1e-12,
        'new_threshold_fitting':False,'input_usable_labels_changed':False,'target_pixels_decoded':0,
        'reserved_final_pixels_decoded':0,'model_forwards':0,'gradient_queries':0,'parameter_updates':0,
        'maximum_seconds':120,'prepared_UTC':datetime.now(timezone.utc).isoformat()})


def run():
    started = time.monotonic(); plan = read(OUT/'plan.json')
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    quality = quality_function(); p = read(BUNDLE/'protocol.json'); old = read(OLD/'plan.json'); rows = []
    by_id = {cid:co['name'] for co in p['cohorts'] for cid in co['case_ids']}
    for case in p['cases']:
        image = pixels(BUNDLE/case['input']); observed = pixels(BUNDLE/case['observed'],'L') > 0
        signals = quality(image,(~observed).astype(np.uint8))
        eyes = np.asarray(case['landmarks5_canvas_xy'],np.float64)[:2]
        size = case['proxy_details']['low_resolution_size'] or case['proxy_details']['working_size']
        scale = np.asarray(size,np.float64)/np.asarray(case['proxy_details']['working_size'],np.float64)
        distance = float(np.linalg.norm((eyes[1]-eyes[0])*scale))
        rows.append({'id':case['id'],'cohort':by_id[case['id']],'source':case['source'],'profile':case['profile'],
            'sampling_grid_size':size,'geometric_eye_spacing_pixels':distance,'quality':signals})
    for case in old['cases']:
        image = pixels(NATIVE/case['input']); observed = pixels(NATIVE/case['observed'],'L') > 0
        signals = quality(image,(~observed).astype(np.uint8))
        eyes = np.asarray(case['eyes'],np.float64)
        distance = float(np.linalg.norm(eyes[1]-eyes[0])); assert abs(distance-case['inter_eye_distance_native']) <= 1e-12
        rows.append({'id':case['id'],'cohort':'native_development','source':old['source'],'profile':'native',
            'sampling_grid_size':[case['native_width'],case['native_height']],
            'geometric_eye_spacing_pixels':distance,'quality':signals})
    assert len(rows) == 124 and [r['id'] for r in rows[:100]] == plan['TRAIN_cases']
    assert [r['id'] for r in rows[100:]] == plan['native_cases']
    for row in rows:
        s = row['quality']
        assert s['blur_threshold'] == 24 and s['noise_threshold'] == 8
        assert s['suggest_restoration'] == bool(s['blur_variance'] is not None and (s['blur_variance'] < 24 or s['noise_sigma_255'] >= 8))
    result = {'complete':True,'plan_sha256':sha(OUT/'plan.json'),'rows':rows,'source_cohort_profile_groups':summary(rows),
        'source_labels_not_ethnicity':True,'native_unpaired':True,'native_PSNR_SSIM_identity_accuracy':None,
        'model_forwards':0,'gradient_queries':0,'parameter_updates':0,'target_pixels_decoded':0,
        'reserved_final_pixels_decoded':0,'thresholds_fitted':False,'app_changed':False,
        'model_qualification':False,'goal_complete':False,'seconds':time.monotonic()-started}
    assert result['seconds'] < plan['maximum_seconds']; write(OUT/'results.json',result)
    # Re-read the saved records and independently verify metadata and summary arithmetic.
    saved = read(OUT/'results.json'); assert saved == result
    assert saved['source_cohort_profile_groups'] == summary(saved['rows'])
    for case,row in zip(p['cases'],saved['rows'][:100]):
        delta = np.asarray(case['landmarks5_canvas_xy'],np.float64)[1]-np.asarray(case['landmarks5_canvas_xy'],np.float64)[0]
        expected = float(np.sqrt(np.sum((delta*np.array(row['sampling_grid_size'])/np.array(case['proxy_details']['working_size']))**2)))
        assert abs(expected-row['geometric_eye_spacing_pixels']) <= 1e-12
    for case,row in zip(old['cases'],saved['rows'][100:]): assert abs(case['inter_eye_distance_native']-row['geometric_eye_spacing_pixels']) <= 1e-12
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    write(OUT/'saved_arithmetic_audit.json',{'complete':True,'plan_sha256':sha(OUT/'plan.json'),
        'results_sha256':sha(OUT/'results.json'),'source_bindings_verified':len(plan['source_bindings']),
        'input_records_verified':124,'all_sampling_geometry_and_summaries_verified':True,
        'independent_quality_algorithm_reproduction':False,'quality_heuristic_validity_proven':False,
        'existing_24_native_usable_labels_preserved':True,'target_pixels_decoded':0,
        'reserved_final_pixels_decoded':0,'model_forwards':0,'gradient_queries':0,'parameter_updates':0,
        'model_qualification':False,'goal_complete':False})
    print({'complete':True,'TRAIN_inputs':100,'native_inputs':24,'groups':len(saved['source_cohort_profile_groups']),
        'model_forwards':0,'seconds':time.monotonic()-started},flush=True)


if __name__ == '__main__': prepare(); run()
