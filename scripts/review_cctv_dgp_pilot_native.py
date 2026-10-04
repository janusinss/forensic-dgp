"""Freeze/run the native development comparison after an audited VM pilot return.

No clean references, PSNR/SSIM or recognizer identity claims for native CCTV.
Never touches reserved image bytes or trains. A saved gallery is not a pass.
"""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cctv_dgp_pilot import read,write,sha,state_hash
from dgp_face_restoration import prepare_crop,restore_crop,load_dgp_restorer,png_rgb

NATIVE=ROOT/'outputs/cctv_native_development_v2'
BASE=ROOT/'outputs/cctv_native_comparison_v1'
VM=ROOT/'outputs/cctv_dgp_vm_bundle_v1'
OUT=ROOT/'outputs/cctv_dgp_native_pilot_review_v1'
PINS={
 'outputs/cctv_native_development_v2/frozen_subset.json':'c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e',
 'outputs/cctv_native_development_v2/input_review.json':'41912033f5070547e120a6b384024fa929eeaf3a7ecca408c0c67c53d37d36e2',
 'outputs/cctv_native_comparison_v1/results.json':'3be73af7b0cdae85f54adc8173fd8f48b42511a604fcd60244e3b1d2565a4c34',
 'outputs/cctv_dgp_vm_bundle_v1/protocol.json':'b652914335e33375f8108ba427e4b5be99fd86e811f455b307ab72ae7cf152f6'}


def require(ok,message):
    if not ok:
        raise ValueError(message)


def prepare():
    require(not OUT.exists(),'Preserve prior/partial native comparison plan')
    for n,pin in PINS.items():
        require(sha(ROOT/n)==pin,'Parent binding differs: '+n)
    cases=[c for c in read(NATIVE/'frozen_subset.json')['cases'] if c['role']=='development']
    review=read(NATIVE/'input_review.json');reviews={r['id']:r for r in review['rows']}
    require(len(cases)==24 and set(reviews)=={c['id'] for c in cases} and not review['reserved_inputs_viewed'],'Input-only review cohort differs')
    assets=dict(PINS)
    names=['dgp_face_restoration.py','scripts/review_cctv_dgp_pilot_native.py',
           'scripts/audit_cctv_dgp_native_pilot_review.py','cctv_dgp_pilot.py',
           'scripts/audit_cctv_dgp_pilot_results.py','tests/test_dgp_face_restoration.py',
           'tests/test_cctv_native_candidate_selection.py']
    names.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'models').glob('*.py')))
    baseline=read(BASE/'results.json')
    for c in cases:
        require(sha(NATIVE/c['source_file'])==c['source_sha256'],'Native source bytes differ from frozen subset')
        names.append((NATIVE/c['source_file']).relative_to(ROOT).as_posix())
        cached=[f"stages/{c['id']}_{n}.npy" for n in ('dgp','codeformer')]+[f"images/{c['id']}_input.png"]
        for n in cached:
            require(sha(BASE/n)==baseline['artifacts_sha256'][n],'Audited cached baseline stage changed')
            names.append('outputs/cctv_native_comparison_v1/'+n)
    assets.update({n:sha(ROOT/n) for n in names})
    plan={'format':'dgp-native-pilot-review-plan-v1','date':'2026-10-03','assets_sha256':assets,'cases':cases,
          'input_reviews':reviews,'candidate_selection':'One diagnostic best.pth per frozen VM arm, only if selected_epoch>0; require independently audited return and preserved metrics/trace before native forwards',
          'input_policy':'Native RGB128 center-pad, PIL bilinear256; identical to cached baseline inputs; no target landmarks, prefilters or enhancement',
          'arms':['input','cached_phase3','cached_codeformer','camera_no_identity_best_if_qualified','camera_identity_best_if_qualified'],
          'outputs':'Raw model float/PNG and observation composite separately; galleries show observation composites identically for baselines/candidates',
          'budget':{'candidate_count':2,'cases_per_candidate':24,'restoration_forwards':48,'wall_seconds_after_loading':480,'cpu_threads':4,'optimizer_updates':0},
          'native_metrics':'Input change is descriptive only. No aligned clean truth, PSNR, SSIM or verified identity accuracy',
          'visual_criteria':['Retain input-selected six coarse frontal/mild cases as a separate core review; report all24 diagnostics',
                             'Preserve apparent eyes/nose/mouth, contour and visible appearance; softness is acceptable',
                             'Reject added generic anatomy, changed expression or strong artifacts',
                             'Seven manually input-reviewed insufficient cases should request a clearer crop; do not treat sharper hallucinations as recovered structure',
                             'Inspect full24-case gallery and ten-row preview; independent final review remains pending'],
          'production_promotion':False,'reserved_native_used':False,'frozen_before_return_native_outputs':True}
    OUT.mkdir();write(OUT/'frozen_plan.json',plan)
    print({'native_plan_prepared':True,'cases':24,'max_forwards':48,'plan_sha256':sha(OUT/'frozen_plan.json')})


def galleries(out,rows,arms):
    names=[];by_id={r['id']:r for r in rows}
    core=[r['id'] for r in rows if r['input_review']['pose_review']=='frontal_or_mild_approximate' and r['input_review']['input_structure_review']=='coarse']
    require(len(core)==6,'Frozen native core cohort differs')
    other=[r['id'] for r in rows if r['id'] not in core]
    groups=[('preview_10_rows.png',core+other[:4])]
    groups.extend((f'gallery_{i//6+1}.png',[r['id'] for r in rows[i:i+6]]) for i in range(0,24,6))
    for name,ids in groups:
        sheet=Image.new('RGB',(len(arms)*172,len(ids)*202+26),(238,238,238));draw=ImageDraw.Draw(sheet)
        for j,arm in enumerate(arms):
            draw.text((j*172+3,3),arm[:24],fill='black')
        for i,caseid in enumerate(ids):
            row=by_id[caseid]
            for j,arm in enumerate(arms):
                x,y=j*172,26+i*202
                draw.text((x+2,y+2),caseid,fill='black')
                draw.text((x+2,y+15),row['input_review']['input_structure_review'],fill='black')
                with Image.open(out/row['images'][arm]) as img:
                    sheet.paste(img.convert('RGB').resize((164,164),Image.Resampling.BILINEAR),(x,y+32))
        sheet.save(out/name);names.append(name)
    return names,core


def audited_candidates(return_dir):
    """Re-audit the actual return once before native outputs, not a green flag alone."""
    from audit_cctv_dgp_pilot_results import audit
    receipt=audit(VM,return_dir)
    result=read(return_dir/'results.json');candidates=[]
    for branch in result['branches']:
        selection=branch['selection'];epoch=selection['selected_epoch']
        if epoch==0:
            continue
        path=return_dir/branch['arm']['id']/'best.pth'
        require(sha(path)==selection['best_sha256'],'Qualified candidate fingerprint differs')
        candidates.append({'id':branch['arm']['id'],'path':path,'sha256':sha(path),'selected_epoch':epoch})
    require(len(candidates)<=2,'Finite candidate count exceeded')
    return candidates,receipt


def run(return_dir):
    require(not (OUT/'execution.json').exists(),'Preserve completed/partial native run; no automatic repeat')
    plan=read(OUT/'frozen_plan.json')
    for n,pin in plan['assets_sha256'].items():
        require(sha(ROOT/n)==pin,'Changed frozen native review asset: '+n)
    candidates,receipt=audited_candidates(return_dir)
    write(OUT/'return_audit_before_native.json',receipt)
    write(OUT/'execution.json',{'plan_sha256':sha(OUT/'frozen_plan.json'),'return_results_sha256':sha(return_dir/'results.json'),
          'return_audit_sha256':sha(OUT/'return_audit_before_native.json'),
          'candidates':[{**c,'path':c['path'].resolve().as_posix()} for c in candidates],
          'device':'cpu','optimizer_constructed':False,'candidate_selection_before_native_outputs':True,
          'reserved_native_used':False})
    (OUT/'images').mkdir();(OUT/'stages').mkdir()
    artifacts={};rows=[];sources={};supports={};counts={};states={};seconds={}
    torch.set_num_threads(4)
    def save_image(rgb,name):
        Image.fromarray(rgb).save(OUT/name);artifacts[name]=sha(OUT/name)
    for case in plan['cases']:
        caseid=case['id']
        with Image.open(NATIVE/case['source_file']) as image:
            rgb=np.asarray(image.convert('RGB'))
        source,observed,_,geometry=prepare_crop(rgb);sources[caseid]=rgb;supports[caseid]=observed
        cached=np.asarray(Image.open(BASE/f'images/{caseid}_input.png').convert('RGB'))
        np.testing.assert_array_equal(source,cached)
        images={'input':f'images/{caseid}_input.png'};save_image(source,images['input'])
        for arm,stage in [('cached_phase3','dgp'),('cached_codeformer','codeformer')]:
            raw=np.load(BASE/f'stages/{caseid}_{stage}.npy',allow_pickle=False)
            composite=np.where(observed[...,None],raw,source.astype(np.float32)/255)
            name=f'images/{caseid}_{arm}.png';save_image(png_rgb(composite),name);images[arm]=name
        rows.append({'id':caseid,'size_bin':case['size_bin'],'native_size':[case['native_width'],case['native_height']],
                     'input_review':plan['input_reviews'][caseid],'geometry':geometry,'images':images,'candidate_stages':{}})
    inference_seconds=0
    for c in candidates:
        model,provenance=load_dgp_restorer(c['path'],'cpu',c['sha256']);before=state_hash(model.net);count=0;start=time.monotonic()
        for row in rows:
            if inference_seconds+time.monotonic()-start>480:
                raise TimeoutError('Finite native candidate budget exceeded')
            result=restore_crop(model,sources[row['id']]);count+=1
            require(count<=24,'Finite per-candidate forward count exceeded')
            prefix=f"{row['id']}_{c['id']}";file='stages/'+prefix+'.npz'
            np.savez_compressed(OUT/file,raw_rgb=result['raw_rgb'],observed_rgb=result['observed_rgb'])
            artifacts[file]=sha(OUT/file);row['candidate_stages'][c['id']]=file
            name='images/'+prefix+'.png';save_image(png_rgb(result['observed_rgb']),name);row['images'][c['id']]=name
            raw_name='images/'+prefix+'_raw.png';save_image(png_rgb(result['raw_rgb']),raw_name)
            # Describe change only, never interpret it as restoration quality.
            mask=supports[row['id']]
            row.setdefault('candidate_input_change_mae_diagnostic_only',{})[c['id']]=float(np.abs(result['raw_rgb'][mask]-result['input'][mask].astype(np.float32)/255).mean())
        elapsed=time.monotonic()-start;inference_seconds+=elapsed
        require(count==24 and inference_seconds<=480 and state_hash(model.net)==before
                and not any(p.requires_grad or p.grad is not None for p in model.parameters()),'Frozen native inference invariant failed')
        counts[c['id']]=count;states[c['id']]={'before':before,'after':state_hash(model.net),'unchanged':True,'provenance':provenance};seconds[c['id']]=elapsed
        print({'candidate':c['id'],'native_forwards':24,'seconds':round(elapsed,2)},flush=True)
        del model
    arms=['input','cached_phase3','cached_codeformer']+[c['id'] for c in candidates]
    gallery_files,core=galleries(OUT,rows,arms)
    artifacts.update({n:sha(OUT/n) for n in gallery_files})
    write(OUT/'results.json',{'complete':True,'plan_sha256':sha(OUT/'frozen_plan.json'),
          'execution_sha256':sha(OUT/'execution.json'),'rows':rows,'arms':arms,'core_coarse_frontal_ids':core,
          'candidate_forwards':counts,'seconds_after_each_model_load':seconds,'model_states':states,
          'artifacts_sha256':artifacts,'galleries':gallery_files,'optimizer_updates':0,'backward_calls':0,
          'PSNR':None,'SSIM':None,'identity_accuracy':None,'reserved_evaluation_used':False,'application_default_change':False,
          'visual_review_pending':True,'independent_final_review_pending':True,
          'status':'native_review_required' if candidates else 'no_trained_candidate_passed_vm_guards'})
    print({'native_comparison_complete':True,'candidates':len(candidates),'forwards':sum(counts.values()),'visual_review_pending':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--return-dir',type=Path,default=ROOT/'outputs/cctv_dgp_return_v1/outputs/cctv_dgp_pilot')
    args=parser.parse_args();prepare() if args.stage=='prepare' else run(args.return_dir)
