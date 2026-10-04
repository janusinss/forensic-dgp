"""Recover reviewable V10 artifacts after preview failure; never rerun models."""
import argparse
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import cctv_dgp_face_prior_v10 as v
from face_prior_grid_v10 import render_grid
# Pinned independent arithmetic from the original unused audit, no neural imports.
from audit_cctv_dgp_face_prior_v10 import pixels, summarize

OUT=ROOT/'outputs/cctv_dgp_face_prior_recovery_v10_r2'
PARENT=v.OUT/'run'


def prepare():
    v.require(not OUT.exists(),'Preserve recovery/partial artifacts')
    v.verify();v.require(not (PARENT/'results.json').exists(),'Recovery only for original incomplete run')
    failure=v.read(PARENT/'failure.json')
    v.require(failure['error_type']=='ValueError' and failure['error']=='cannot determine region size; use 4-item box',
              'Recovery requires the known preview-only error')
    names=['scripts/recover_cctv_dgp_face_prior_v10.py','scripts/audit_cctv_dgp_face_prior_recovery_v10.py',
           'face_prior_grid_v10.py','tests/test_face_prior_grid_v10.py',
           'outputs/cctv_dgp_face_prior_v10_grid_tests.json']
    assets={name:v.sha(ROOT/name) for name in names}
    inherited={path.relative_to(PARENT).as_posix():v.sha(path) for path in PARENT.rglob('*') if path.is_file()}
    v.require(len(inherited)==781,'Expected complete pre-grid artifact set781 files')
    plan={'format':'dgp-face-prior-artifact-recovery-v10-r2','date':'2026-10-04',
        'parent_plan_sha256':v.sha(v.OUT/'frozen_plan.json'),'parent_failure_sha256':v.sha(PARENT/'failure.json'),
        'assets_sha256':assets,'inherited_sha256':inherited,'frozen_before_postprocessing':True,
        'neural_forwards':0,'backward_calls':0,'optimizer_updates':0,'cap_seconds':120,
        'original_neural_execution_complete':False,'post_forward_model_state_available':False,
        'native_reserved_used':False,'production_promotion':False,
        'limitation':'Original model run finished saving case images/embeddings but failed before grids/results and final count/state receipt. This recovery cannot supply that missing model-state proof.'}
    OUT.mkdir();v.write(OUT/'frozen_recovery_plan.json',plan)
    (OUT/'frozen_recovery_plan.sha256').write_text(v.sha(OUT/'frozen_recovery_plan.json')+'\n',encoding='ascii')
    print({'recovery_prepared':True,'parent_artifacts':len(inherited),'plan_sha256':v.sha(OUT/'frozen_recovery_plan.json')})


def verify():
    v.require((OUT/'frozen_recovery_plan.sha256').read_text(encoding='ascii').strip()==v.sha(OUT/'frozen_recovery_plan.json'),'Recovery fingerprint differs')
    p=v.read(OUT/'frozen_recovery_plan.json');v.verify()
    v.require(p['parent_plan_sha256']==v.sha(v.OUT/'frozen_plan.json'),'Original comparison changed')
    for name,pin in p['assets_sha256'].items():v.require(v.sha(v.safe(ROOT,name))==pin,'Recovery source changed')
    for name,pin in p['inherited_sha256'].items():v.require(v.sha(v.safe(PARENT,name))==pin,'Original saved artifact changed')
    return p


def run():
    plan=verify();p=v.read(v.OUT/'frozen_plan.json');out=OUT/'review'
    v.require(not out.exists(),'Preserve prior/partial postprocessing; no overwrite')
    started=time.monotonic();out.mkdir();artifacts={}
    def clock():v.require(time.monotonic()-started<=120,'Recovery exceeded120 seconds')
    for name,pin in plan['inherited_sha256'].items():
        clock();dst=out/'inherited'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PARENT/name,dst);v.require(v.sha(dst)==pin,'Copy differs');artifacts['inherited/'+name]=pin
    paired=[];native=[]
    for case in p['paired_cases']:
        clock();support=np.asarray(Image.open(ROOT/case['files']['observed']))>0
        target=v.rgb(ROOT/case['files']['target']);rid=case['reference_id']
        reference=np.load(out/'inherited/targets'/f'{rid}.npy',allow_pickle=False)
        rows={}
        for arm in p['paired_arms']:
            prefix='inherited/paired/'+arm
            image=prefix+'/images/'+case['id']+'.png';vecname=prefix+'/embeddings/'+case['id']+'.npy'
            rgb=v.rgb(out/image);vec=np.load(out/vecname,allow_pickle=False)
            row={**{k:case[k] for k in ['id','reference_id','source','profile']},**pixels(rgb,target,support),
                'ArcFace_observed_fixed':float(np.clip(vec@reference,-1,1)), 'prediction':image,'embedding':vecname}
            if not arm.startswith('cached_') and arm!='input':row['raw_float']=prefix+'/raw/'+case['id']+'.npy'
            rows[arm]=row
        paired.append({'id':case['id'],'arms':rows})
    for case in p['native_cases']:
        clock();cid=case['id']
        native.append({'id':cid,'input_review':case['input_review'],
            'images':{arm:'inherited/native/'+cid+'_'+arm+'.png' for arm in p['native_arms']},
            'raw_float':{arm:'inherited/native/'+cid+'_'+arm+'.npy' for arm in ['v9_epoch20_diagnostic','codeformer_dgp_w1_diagnostic']},
            'PSNR':None,'SSIM':None,'identity_accuracy':None})
    paired_by={r['id']:r for r in paired};native_by={r['id']:r for r in native};grids=[]
    def grid(name,ids,arms,is_paired):
        clock();rows=[]
        for cid in ids:
            row=(paired_by if is_paired else native_by)[cid]
            files=[row['arms'][a]['prediction'] for a in arms] if is_paired else [row['images'][a] for a in arms]
            images=[v.rgb(out/n) for n in files]
            if is_paired:images.append(v.rgb(ROOT/next(c for c in p['paired_cases'] if c['id']==cid)['files']['target']))
            rows.append({'id':cid,'images':images})
        render_grid(arms+(['reference'] if is_paired else []),rows).save(out/name)
        artifacts[name]=v.sha(out/name);grids.append(name)
    for profile in v.PROFILES:grid('paired_'+profile+'_10_rows.png',[r['id']+'_'+profile for r in p['references']],p['paired_arms'],True)
    for i in range(4):grid('native_gallery_'+str(i+1)+'.png',[r['id'] for r in native[i*6:(i+1)*6]],p['native_arms'],False)
    core=[r['id'] for r in native if r['input_review']['pose_review']=='frontal_or_mild_approximate' and r['input_review']['input_structure_review']=='coarse']
    v.require(len(core)==6,'Core review differs');grid('native_preview_10_rows.png',core+[r['id'] for r in native if r['id'] not in core][:4],p['native_arms'],False)
    clock();verify()
    results={'complete':True,'scope':'artifact_postprocessing_only','recovery_plan_sha256':v.sha(OUT/'frozen_recovery_plan.json'),
        'parent_plan_sha256':plan['parent_plan_sha256'],'original_neural_execution_complete':False,
        'post_forward_model_state_available':False,'original_forward_counts_captured':False,
        'artifact_implied_original_counts':{'dgp':24,'codeformer':124,'recognizer':260},
        'local_model_forwards':0,'local_backward_calls':0,'local_optimizer_updates':0,
        'paired_rows':paired,'paired_summary':{arm:summarize([r['arms'][arm] for r in paired]) for arm in p['paired_arms']},
        'native_rows':native,'native_core_ids':core,'grids':grids,'artifacts_sha256':artifacts,
        'seconds':time.monotonic()-started,'native_reserved_used':False,'production_promoted':False,
        'checkpoint_selected':False,'independent_final_review_pending':True,
        'limitation':plan['limitation']}
    v.write(out/'results.json',results)
    print({'postprocessing_complete':True,'seconds':results['seconds'],'new_model_forwards':0,'original_neural_execution_complete':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.run:run()
    else:parser.error('Choose --prepare or --run')
