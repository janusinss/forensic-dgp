"""Verify native pilot comparison composition/lineage; zero model forwards."""
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cctv_dgp_pilot import read,write,sha,state_hash
from dgp_face_restoration import prepare_crop,png_rgb
from audit_cctv_dgp_pilot_results import safe_result_path,same

NATIVE=ROOT/'outputs/cctv_native_development_v2'
BASE=ROOT/'outputs/cctv_native_comparison_v1'
OUT=ROOT/'outputs/cctv_dgp_native_pilot_review_v1'


def require(ok,message):
    if not ok:
        raise ValueError(message)


def validate_selection(execution,returned):
    expected=[{'id':b['arm']['id'],'sha256':b['selection']['best_sha256'],'selected_epoch':b['selection']['selected_epoch']}
              for b in returned['branches'] if b['selection']['selected_epoch']>0]
    actual=[{k:c[k] for k in ('id','sha256','selected_epoch')} for c in execution['candidates']]
    require(actual==expected and len(actual)<=2,'Native candidate selection differs from audited VM guards')
    return expected


def audit(return_dir,out=OUT):
    plan=read(out/'frozen_plan.json');execution=read(out/'execution.json');result=read(out/'results.json')
    require(result['complete'] and sha(out/'frozen_plan.json')==result['plan_sha256']==execution['plan_sha256']
            and sha(out/'execution.json')==result['execution_sha256'],'Native plan/execution binding differs')
    for n,pin in plan['assets_sha256'].items():
        require(sha(ROOT/n)==pin,'Changed frozen review asset: '+n)
    returned=read(return_dir/'results.json');candidates=validate_selection(execution,returned)
    require(sha(return_dir/'results.json')==execution['return_results_sha256'],'Returned results changed')
    receipt=read(out/'return_audit_before_native.json')
    require(receipt['complete'] and receipt['returned_results_sha256']==sha(return_dir/'results.json')
            and receipt['protocol_sha256']=='b652914335e33375f8108ba427e4b5be99fd86e811f455b307ab72ae7cf152f6'
            and sha(out/'return_audit_before_native.json')==execution['return_audit_sha256'],'Actual return-audit binding differs')
    require(receipt['png_predictions_checked']==2750 and receipt['update_trace_checked']==452,'Return audit coverage differs')
    require(result['optimizer_updates']==result['backward_calls']==0 and not result['reserved_evaluation_used']
            and not result['application_default_change'] and result['visual_review_pending']
            and result['independent_final_review_pending'],'Native inference/review claim differs')
    require(result['PSNR'] is None and result['SSIM'] is None and result['identity_accuracy'] is None,'Invented paired/identity native metrics')
    require(result['status']==('native_review_required' if candidates else 'no_trained_candidate_passed_vm_guards'),'Native selection status differs')
    for n,pin in result['artifacts_sha256'].items():
        require(sha(safe_result_path(out,n))==pin,'Changed native output asset: '+n)
    case_map={c['id']:c for c in plan['cases']}
    require([r['id'] for r in result['rows']]==list(case_map) and len(case_map)==24,'Fixed native cases/order differ')
    expected_arms=['input','cached_phase3','cached_codeformer']+[c['id'] for c in candidates]
    require(result['arms']==expected_arms,'Native arms differ')
    require(result['candidate_forwards']=={c['id']:24 for c in candidates}
            and sum(result['seconds_after_each_model_load'].values())<=480,'Finite native budget record differs')
    for c in candidates:
        path=return_dir/c['id']/'best.pth';require(sha(path)==c['sha256'],'Candidate checkpoint lineage differs')
        state=torch.load(path,map_location='cpu',weights_only=True)
        fields=result['model_states'][c['id']]
        require(fields['before']==fields['after']==state_hash(state) and fields['unchanged']
                and fields['provenance']['weights_sha256']==c['sha256'],'Candidate model-state record differs')
    pngs=0;raws=0
    for row in result['rows']:
        caseid=row['id'];case=case_map[caseid]
        require(row['input_review']==plan['input_reviews'][caseid],'Input-only review labels differ')
        with Image.open(NATIVE/case['source_file']) as img:
            rgb=np.asarray(img.convert('RGB'))
        common,observed,_,geometry=prepare_crop(rgb)
        require(geometry==row['geometry'],'Observation geometry differs')
        require(set(row['images'])==set(expected_arms),'Image arm cohort differs')
        def match(name,expected):
            nonlocal pngs
            with Image.open(safe_result_path(out,name)) as img:
                actual=np.asarray(img.convert('RGB'))
            np.testing.assert_array_equal(actual,expected);pngs+=1
        match(row['images']['input'],common)
        for arm,name in [('cached_phase3','dgp'),('cached_codeformer','codeformer')]:
            raw=np.load(BASE/f'stages/{caseid}_{name}.npy',allow_pickle=False)
            composed=np.where(observed[...,None],raw,common.astype(np.float32)/255)
            match(row['images'][arm],png_rgb(composed))
        require(set(row['candidate_stages'])=={c['id'] for c in candidates},'Candidate stage cohort differs')
        for c in candidates:
            stage=np.load(safe_result_path(out,row['candidate_stages'][c['id']]),allow_pickle=False)
            raw,composed=stage['raw_rgb'],stage['observed_rgb']
            require(raw.shape==composed.shape==(256,256,3) and raw.dtype==composed.dtype==np.float32
                    and np.isfinite(raw).all() and 0<=raw.min()<=raw.max()<=1,'Invalid raw candidate output')
            np.testing.assert_array_equal(composed,np.where(observed[...,None],raw,common.astype(np.float32)/255))
            match(row['images'][c['id']],png_rgb(composed));raws+=1
            match('images/'+caseid+'_'+c['id']+'_raw.png',png_rgb(raw))
            expected=float(np.abs(raw[observed]-common[observed].astype(np.float32)/255).mean())
            same(row['candidate_input_change_mae_diagnostic_only'][c['id']],expected,'input change only',1e-8)
    core=[r['id'] for r in result['rows'] if r['input_review']['pose_review']=='frontal_or_mild_approximate' and r['input_review']['input_structure_review']=='coarse']
    require(core==result['core_coarse_frontal_ids'] and len(core)==6,'Input-selected core cohort differs')
    for name in result['galleries']:
        with Image.open(out/name) as img:
            rows=10 if name=='preview_10_rows.png' else 6
            require(img.size==(len(expected_arms)*172,rows*202+26),'Gallery dimensions differ')
    return {'complete':True,'native_results_sha256':sha(out/'results.json'),'native_plan_sha256':sha(out/'frozen_plan.json'),
            'candidate_count':len(candidates),'native_cases':24,'core_coarse_frontal_cases':6,
            'pngs_reconstructed':pngs,'raw_stages_checked':raws,'model_forwards':0,'optimizer_updates':0,
            'reserved_native_used':False,'native_visual_review_pending':True,'independent_final_review_pending':True,
            'application_default_selected':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--return-dir',type=Path,default=ROOT/'outputs/cctv_dgp_return_v1/outputs/cctv_dgp_pilot')
    args=parser.parse_args();report=audit(args.return_dir)
    if not (OUT/'independent_audit.json').exists():
        write(OUT/'independent_audit.json',report)
    print(report)
