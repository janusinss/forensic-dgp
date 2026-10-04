"""Audit V10 recovered artifacts, retaining the missing terminal-state proof."""
import ast
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import cctv_dgp_face_prior_v10 as v
from recover_cctv_dgp_face_prior_v10 import OUT,verify
from audit_cctv_dgp_face_prior_v10 import close


def audit():
    started=time.monotonic();plan=verify();p=v.read(v.OUT/'frozen_plan.json');out=OUT/'review'
    v.require(not (out/'independent_audit.json').exists(),'Preserve completed artifact audit')
    r=v.read(out/'results.json')
    v.require(r['complete'] and r['scope']=='artifact_postprocessing_only','Wrong recovered scope')
    v.require(r['recovery_plan_sha256']==v.sha(OUT/'frozen_recovery_plan.json') and r['parent_plan_sha256']==plan['parent_plan_sha256'],'Recovery binding differs')
    v.require(all(r[k] is False for k in ['original_neural_execution_complete','post_forward_model_state_available',
        'original_forward_counts_captured','native_reserved_used','production_promoted','checkpoint_selected']), 'Missing neural proof concealed or promotion claimed')
    v.require(r['local_model_forwards']==r['local_backward_calls']==r['local_optimizer_updates']==0 and 0<r['seconds']<=120,'No-forward postprocessing/cap differs')
    for name,pin in r['artifacts_sha256'].items():v.require(v.sha(v.safe(out,name))==pin,'Recovered artifact differs')
    for name,pin in plan['inherited_sha256'].items():v.require(v.sha(out/'inherited'/name)==pin,'Inherited artifact copy differs')
    # Execute only the separately pinned V9 numeric functions, not its NN imports.
    source=ROOT/'cctv_dgp_pilot.py';tree=ast.parse(source.read_text(encoding='utf-8'))
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ['exported_pixel_metrics','aggregate']]
    v.require(len(nodes)==2,'Metric source functions differ')
    ns={'np':np,'cv2':cv2,'structural_similarity':structural_similarity}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    def vector(name):
        vec=np.load(out/name,allow_pickle=False)
        v.require(vec.dtype==np.float32 and vec.shape==(512,) and np.isfinite(vec).all() and np.isclose(np.linalg.norm(vec),1,atol=1e-5),'Bad embedding vector')
        return vec
    def quantized(name,source,support):
        raw=np.load(out/name,allow_pickle=False)
        v.require(raw.dtype==np.float32 and raw.shape==(256,256,3) and np.isfinite(raw).all() and 0<=raw.min()<=raw.max()<=1,'Bad model float image')
        result=np.clip(raw*255,0,255).astype(np.uint8);result[~support]=source[~support];return result
    v.require([row['id'] for row in r['paired_rows']]==[case['id'] for case in p['paired_cases']],'Paired cohort differs')
    pngs=raws=cosines=cached=0
    for case,saved in zip(p['paired_cases'],r['paired_rows']):
        v.require(set(saved['arms'])==set(p['paired_arms']),'Paired arm missing')
        target=v.rgb(ROOT/case['files']['target']);source=v.rgb(ROOT/case['files']['input'])
        support=np.asarray(Image.open(ROOT/case['files']['observed']))>0
        reference=vector('inherited/targets/'+case['reference_id']+'.npy')
        for arm,row in saved['arms'].items():
            v.require(all(row[key]==case[key] for key in ['id','reference_id','source','profile']),'Reassigned paired role')
            image=v.rgb(out/row['prediction']);np.testing.assert_array_equal(image[~support],source[~support])
            for key,value in ns['exported_pixel_metrics'](image,target,support).items():close(row[key],value)
            close(row['ArcFace_observed_fixed'],float(np.clip(vector(row['embedding'])@reference,-1,1)))
            if arm in p['paired_arms'][:3]:np.testing.assert_array_equal(image,v.rgb(ROOT/case['files'][arm]));cached+=1
            else:np.testing.assert_array_equal(image,quantized(row['raw_float'],source,support));raws+=1
            pngs+=1;cosines+=1
    for arm in p['paired_arms']:close(r['paired_summary'][arm],ns['aggregate']([row['arms'][arm] for row in r['paired_rows']]))
    v.require([row['id'] for row in r['native_rows']]==[case['id'] for case in p['native_cases']],'Native cohort differs')
    for case,row in zip(p['native_cases'],r['native_rows']):
        v.require(case['role']=='development' and row['input_review']==case['input_review'],'Native role/review differs')
        v.require(all(row[key] is None for key in ['PSNR','SSIM','identity_accuracy']),'Unpaired truth claimed')
        with Image.open(ROOT/case['files']['source']) as im:source_native=np.asarray(im.convert('RGB')).copy()
        h,w=source_native.shape[:2];side=max(h,w);y,x=(side-h)//2,(side-w)//2
        canvas=np.full((side,side,3),128,np.uint8);canvas[y:y+h,x:x+w]=source_native
        mask=np.zeros((side,side),np.uint8);mask[y:y+h,x:x+w]=255
        source=np.asarray(Image.fromarray(canvas).resize((256,256),Image.Resampling.BILINEAR))
        support=np.asarray(Image.fromarray(mask).resize((256,256),Image.Resampling.NEAREST))>0
        np.testing.assert_array_equal(source,v.rgb(ROOT/case['files']['input']))
        for arm,name in row['images'].items():
            image=v.rgb(out/name)
            if arm=='input':expected=source;cached+=1
            elif arm.startswith('cached_'):
                raw=np.load(ROOT/case['files'][arm],allow_pickle=False)
                expected=(raw*255).astype(np.uint8);expected[~support]=source[~support];cached+=1
            else:expected=quantized(row['raw_float'][arm],source,support);raws+=1
            np.testing.assert_array_equal(image,expected);pngs+=1
    paired={row['id']:row for row in r['paired_rows']};native={row['id']:row for row in r['native_rows']}
    core=[row['id'] for row in r['native_rows'] if row['input_review']['pose_review']=='frontal_or_mild_approximate' and row['input_review']['input_structure_review']=='coarse']
    v.require(core==r['native_core_ids'] and len(core)==6,'Core selection differs')
    groups=[('paired_'+profile+'_10_rows.png',[ref['id']+'_'+profile for ref in p['references']],p['paired_arms'],True) for profile in v.PROFILES]
    groups.extend(('native_gallery_'+str(i+1)+'.png',[row['id'] for row in r['native_rows'][i*6:(i+1)*6]],p['native_arms'],False) for i in range(4))
    groups.append(('native_preview_10_rows.png',core+[row['id'] for row in r['native_rows'] if row['id'] not in core][:4],p['native_arms'],False))
    v.require(r['grids']==[g[0] for g in groups],'Grids missing')
    cells=0
    for name,ids,arms,paired_group in groups:
        with Image.open(out/name) as im:sheet=np.asarray(im).copy()
        v.require(sheet.shape==(24+len(ids)*288,260*(len(arms)+int(paired_group)),3),'Grid geometry differs')
        for i,cid in enumerate(ids):
            row=(paired if paired_group else native)[cid];y=52+i*288
            for j,arm in enumerate(arms):
                file=row['arms'][arm]['prediction'] if paired_group else row['images'][arm]
                np.testing.assert_array_equal(sheet[y:y+256,j*260+2:j*260+258],v.rgb(out/file));cells+=1
            if paired_group:
                case=next(c for c in p['paired_cases'] if c['id']==cid);j=len(arms)
                np.testing.assert_array_equal(sheet[y:y+256,j*260+2:j*260+258],v.rgb(ROOT/case['files']['target']));cells+=1
    v.require(pngs==370 and raws==148 and cosines==250 and cells==470,'Artifact counts differ')
    receipt={'complete':True,'scope':'recovered_artifact_integrity_and_arithmetic',
        'recovery_plan_sha256':r['recovery_plan_sha256'],'results_sha256':v.sha(out/'results.json'),'auditor_sha256':v.sha(Path(__file__)),
        'inherited_byte_copies_checked':len(plan['inherited_sha256']),'PNG_metrics_or_compositions_rebuilt':pngs,
        'raw_float_images_checked':raws,'embedding_cosines_rebuilt':cosines,'cached_image_copies_or_composites_checked':cached,
        'summary_groups_rebuilt':70,'grid_cells_rebuilt':cells,'grids_checked':10,'seconds':time.monotonic()-started,
        'model_forwards_in_audit':0,'local_backward_calls':0,'local_optimizer_updates':0,
        'native_reserved_used':False,'production_promoted':False,'original_neural_execution_complete':False,
        'post_forward_model_state_available':False,'original_forward_counts_captured':False,
        'independent_final_review_pending':True,'limitation':r['limitation']}
    v.write(out/'independent_audit.json',receipt);print(receipt)


if __name__=='__main__':audit()
