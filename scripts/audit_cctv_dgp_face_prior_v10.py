"""Independently reconstruct saved V10 pixels/metrics; no neural forwards."""
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import cctv_dgp_face_prior_v10 as v


def close(a, b):
    if isinstance(b, dict):
        v.require(set(a) == set(b), 'Summary keys differ')
        for key in b: close(a[key], b[key])
    elif isinstance(b, (float, int)) and not isinstance(b, bool):
        v.require(math.isfinite(a) and math.isclose(a, b, abs_tol=3e-7, rel_tol=3e-6), 'Arithmetic differs')
    else: v.require(a == b, 'Saved field differs')


def pixels(actual, target, support):
    a, t = actual.astype(np.float32)/255, target.astype(np.float32)/255
    error = a-t; selected = error[support]
    mse = float(np.square(selected).astype(np.float64).mean())
    _, scores = structural_similarity(t, a, data_range=1, channel_axis=-1, win_size=7, full=True)
    interior = cv2.erode(support.astype(np.uint8), np.ones((7,7),np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return {'MSE': mse, 'PSNR': -10*math.log10(mse) if mse else None, 'perfect_match': mse == 0,
            'SSIM': float(scores[interior].astype(np.float64).mean()),
            'MAE': float(np.abs(selected).astype(np.float64).mean())}


def summarize(rows):
    groups = {'degraded': [r for r in rows if r['profile']!='clear'],
              'clear': [r for r in rows if r['profile']=='clear']}
    for source in sorted({r['source'] for r in rows}):
        for profile in sorted({r['profile'] for r in rows}):
            groups[source+'/'+profile] = [r for r in rows if r['source']==source and r['profile']==profile]
        groups[source+'/degraded'] = [r for r in rows if r['source']==source and r['profile']!='clear']
    result={}
    for group, items in groups.items():
        v.require(bool(items), 'Empty aggregate group')
        avg=lambda key: math.fsum(row[key] for row in items)/len(items)
        mse=avg('MSE')
        result[group]={'cases':len(items),'identity_pairs':len(items),'MSE':mse,
            'PSNR':-10*math.log10(mse) if mse else None,
            'perfect_matches':sum(r['perfect_match'] for r in items),
            **{key:avg(key) for key in ['SSIM','MAE','ArcFace_observed_fixed']}}
    return result


def audit():
    started=time.monotonic(); p=v.verify(); out=v.OUT/'run'
    v.require(not (out/'independent_audit.json').exists(), 'Preserve completed audit')
    e=v.read(out/'execution.json'); r=v.read(out/'results.json')
    v.require(r['complete'] and r['plan_sha256']==e['plan_sha256']==v.sha(v.OUT/'frozen_plan.json'), 'Plan/execution differs')
    v.require(e['source_sha256']==p['assets_sha256']['scripts/compare_cctv_dgp_face_prior_v10.py'], 'Executed source differs')
    v.require(r['model_forwards']=={'dgp':24,'codeformer':124,'recognizer':260}, 'Forward count differs')
    v.require(r['model_state_after']==e['model_state_before'] and r['model_states_unchanged'], 'Reported model state changed')
    v.require(0<r['seconds_after_loading']<=1200 and not e['optimizer_constructed'], 'Budget/training differs')
    v.require(all(not doc[key] for doc in [r,e] for key in ['optimizer_updates','backward_calls','native_reserved_used']), 'Training/reserved scope differs')
    v.require(not r['training'] and not r['production_promoted'] and not r['checkpoint_selected'], 'Forbidden promotion')
    timing=v.read(out/'timing_at5.json')
    v.require(0<timing['projected_seconds']<=timing['cap_seconds']==1200, 'Timing cap differs')
    for name, pin in r['artifacts_sha256'].items():
        v.require(v.sha(v.safe(out,name))==pin, 'Changed output artifact: '+name)
    def array(name):
        a=np.load(v.safe(out,name),allow_pickle=False)
        v.require(a.dtype==np.float32 and np.isfinite(a).all(), 'Bad saved float dtype/range')
        return a
    def embedding(name):
        a=array(name);v.require(a.shape==(512,) and np.isclose(np.linalg.norm(a),1,atol=1e-5), 'Bad embedding')
        return a
    def expected_png(raw, source, support):
        v.require(raw.shape==(256,256,3) and 0<=raw.min()<=raw.max()<=1,'Bad raw image')
        result=np.floor(raw*255).astype(np.uint8);result[~support]=source[~support]
        return result
    v.require(len(r['paired_rows'])==50 and [x['id'] for x in r['paired_rows']]==[x['id'] for x in p['paired_cases']], 'Missing/reordered paired cases')
    pngs=raws=cosines=cached_copies=0
    for case,saved in zip(p['paired_cases'],r['paired_rows']):
        v.require(set(saved['arms'])==set(p['paired_arms']),'Wrong paired arms')
        target=v.rgb(v.ROOT/case['files']['target']);source=v.rgb(v.ROOT/case['files']['input'])
        support=np.asarray(Image.open(v.ROOT/case['files']['observed']))>0
        reference=embedding('targets/'+case['reference_id']+'.npy')
        for arm,row in saved['arms'].items():
            v.require(all(row[k]==case[k] for k in ['id','reference_id','source','profile']),'Wrong row role')
            image=v.rgb(out/row['prediction']); np.testing.assert_array_equal(image[~support],source[~support])
            for key,value in pixels(image,target,support).items():close(row[key],value)
            vec=embedding(row['embedding']); close(row['ArcFace_observed_fixed'],float(np.clip(vec@reference,-1,1)))
            if arm in p['paired_arms'][:3]:
                np.testing.assert_array_equal(image,v.rgb(v.ROOT/case['files'][arm]));cached_copies+=1
            else:
                np.testing.assert_array_equal(image,expected_png(array(row['raw_float']),source,support));raws+=1
            pngs+=1;cosines+=1
    for arm in p['paired_arms']:close(r['paired_summary'][arm],summarize([row['arms'][arm] for row in r['paired_rows']]))
    v.require(len(r['native_rows'])==24 and [x['id'] for x in r['native_rows']]==[x['id'] for x in p['native_cases']], 'Wrong native cohort')
    for case,row in zip(p['native_cases'],r['native_rows']):
        v.require(case['role']=='development' and row['input_review']==case['input_review'],'Native role/review differs')
        v.require(all(row[k] is None for k in ['PSNR','SSIM','identity_accuracy']),'Unpaired truth claim')
        with Image.open(v.ROOT/case['files']['source']) as native: native=np.asarray(native.convert('RGB')).copy()
        h,w=native.shape[:2];side=max(h,w);top,left=(side-h)//2,(side-w)//2
        canvas=np.full((side,side,3),128,np.uint8);canvas[top:top+h,left:left+w]=native
        support=np.zeros((side,side),np.uint8);support[top:top+h,left:left+w]=255
        source=np.asarray(Image.fromarray(canvas).resize((256,256),Image.Resampling.BILINEAR))
        support=np.asarray(Image.fromarray(support).resize((256,256),Image.Resampling.NEAREST))>0
        np.testing.assert_array_equal(source,v.rgb(v.ROOT/case['files']['input']))
        v.require(set(row['images'])==set(p['native_arms']), 'Wrong native arms')
        for arm,name in row['images'].items():
            image=v.rgb(out/name)
            if arm=='input': expected=source;cached_copies+=1
            elif arm.startswith('cached_'):
                expected=expected_png(np.load(v.ROOT/case['files'][arm],allow_pickle=False),source,support);cached_copies+=1
            else:
                expected=expected_png(array(row['raw_float'][arm]),source,support);raws+=1
            np.testing.assert_array_equal(image,expected);pngs+=1
    paired={r['id']:r for r in r['paired_rows']};native={r['id']:r for r in r['native_rows']}
    core=[r['id'] for r in r['native_rows'] if r['input_review']['pose_review']=='frontal_or_mild_approximate'
          and r['input_review']['input_structure_review']=='coarse']
    v.require(core==r['native_core_ids'] and len(core)==6,'Native core selection differs')
    groups=[]
    for profile in v.PROFILES:groups.append(('paired_'+profile+'_10_rows.png', [ref['id']+'_'+profile for ref in p['references']],p['paired_arms'],True))
    for i in range(4):groups.append(('native_gallery_'+str(i+1)+'.png',[x['id'] for x in r['native_rows'][i*6:(i+1)*6]],p['native_arms'],False))
    groups.append(('native_preview_10_rows.png',core+[x['id'] for x in r['native_rows'] if x['id'] not in core][:4],p['native_arms'],False))
    v.require([g[0] for g in groups]==r['grids'],'Missing gallery')
    cells=0
    for name,ids,arms,is_paired in groups:
        with Image.open(out/name) as grid:sheet=np.asarray(grid)
        v.require(sheet.shape==((24+len(ids)*288),(len(arms)+int(is_paired))*260,3),'Grid geometry differs')
        for i,cid in enumerate(ids):
            row=(paired if is_paired else native)[cid];y=24+i*288+28
            for j,arm in enumerate(arms):
                image=row['arms'][arm]['prediction'] if is_paired else row['images'][arm]
                np.testing.assert_array_equal(sheet[y:y+256,j*260+2:j*260+258],v.rgb(out/image));cells+=1
            if is_paired:
                case=next(c for c in p['paired_cases'] if c['id']==cid);j=len(arms)
                np.testing.assert_array_equal(sheet[y:y+256,j*260+2:j*260+258],v.rgb(v.ROOT/case['files']['target']));cells+=1
    receipt={'complete':True,'plan_sha256':r['plan_sha256'],'results_sha256':v.sha(out/'results.json'),
        'auditor_sha256':v.sha(Path(__file__)),'png_metrics_or_compositions_rebuilt':pngs,'new_raw_float_images_checked':raws,
        'embedding_cosines_rebuilt':cosines,'inherited_image_copies_or_composites_checked':cached_copies,
        'grid_cells_rebuilt':cells,'grids_checked':len(groups),'paired_reference_count':10,'paired_cases':50,'native_cases':24,
        'summary_groups_checked':70,'seconds':time.monotonic()-started,'local_model_forwards':0,'local_backward_calls':0,
        'local_optimizer_updates':0,'native_reserved_used':False,'production_promoted':False,'independent_final_review_pending':True,
        'limitation':'Verifies pinned source/input/output bytes, reported state/counts and independent pixel/embedding arithmetic. Does not replay neural inference, verify native hidden identity or replace original-cell/independent human review.'}
    v.write(out/'independent_audit.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':audit()
