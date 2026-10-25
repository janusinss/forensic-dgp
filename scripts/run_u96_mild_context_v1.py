"""Finite prospective matched-operator local inference; no training or retries."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, distance_transform_edt
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_u96_mild_context_v1'
PLAN=OUT/'comparison_plan_v1.json'
PRIOR=ROOT/'outputs/codeformer_u96_context_study_v1'
LIMITS={'wall_seconds':1200,'HTTP_seconds':150,'disk_reserve_bytes':4*1024**3,
        'RAM_reserve_bytes':384*1024**2,'output_bytes':1024**3,'maximum_new_forwards':40,
        'failure_telemetry_grace_seconds':5}
SIGMA=1.2
GUARD=6.
FEATHER=8.


def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p):
    p=Path(p).resolve();assert p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink()
    b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':sha(b)}
def bound(row):
    p=ROOT/row['path'];assert p.resolve().is_relative_to(ROOT)
    assert pin(p)==row,row['path'];return p.read_bytes()
def doc(p):return json.loads(Path(p).read_bytes())
def write(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def update(p,value):
    temp=Path(str(p)+'.writing')
    write(temp,value);temp.replace(p)
def png(p,a):
    with Path(p).open('xb') as f:Image.fromarray(a).save(f,format='PNG')
def rgb(b):
    with Image.open(io.BytesIO(b)) as im:
        assert im.mode=='RGB' and im.size==(256,256)
        return np.asarray(im).copy()
def load_helpers():
    paths=[('sealed_context_io','scripts/run_u96_context_study_v1.py','01b6505849a5a0b7aeb6d8e2876d69bb1c80cce8d3a0519b4e678c2f8b762427'),
           ('keepalive_transport','scripts/recover_u96_context_transport_v1_r1.py','35f02a60fac13faadec825b2793a4020c085c4c34ab92fd71afad9e74c4a42ef')]
    result=[]
    for name,relative,digest in paths:
        p=ROOT/relative;assert pin(p)['sha256']==digest
        spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module);result.append(module)
    return result


def freeze(admission_path):
    assert not PLAN.exists() and not (OUT/'preparation').exists()
    admission_path=Path(admission_path).resolve();admission=doc(admission_path)
    assert admission['complete'] and admission['frozen_before_study_model_outputs']
    assert admission['selected_sigma']==SIGMA and len(admission['selected_reference_ids'])>=4
    assert len(admission['selected_reference_ids'])<=8
    assert len(set(admission['selected_reference_ids']))==len(admission['selected_reference_ids'])
    candidate_path=OUT/'candidate_manifest_v1.json';candidate=doc(candidate_path)
    assert pin(candidate_path)==admission['candidate_manifest']
    assert set(admission['selected_reference_ids']).issubset(candidate['fixed_reference_order'])
    bound(admission['selection_contract']);bound(admission['reviewer_record']);bound(admission['anonymous_map'])
    oldplan=doc(PRIOR/'comparison_plan_v1.json');oldrun=doc(PRIOR/'recovery_r1/comparison.json')
    assert oldrun['complete']
    oldreviews={r['reference_id']:r for r in doc(PRIOR/'root_input_review_v1.json')['references']}
    iohelper,_=load_helpers()
    source_pins=[pin(Path(__file__)),pin(ROOT/'scripts/recover_u96_context_transport_v1_r1.py'),
                 pin(ROOT/'scripts/run_u96_context_study_v1.py')]
    for row in oldplan['source_pins']:
        bound(row)
        if row not in source_pins:source_pins.append(row)
    for row in [*oldplan['model_files'].values(),*oldplan['model_manifest_and_reviews']]:bound(row)
    case_lookup={(r['reference_id'],r['role']):r for r in candidate['cases']}
    assert len(case_lookup)==len(candidate['cases'])==40
    oldlookup={r['id']:r for r in oldrun['cases']}
    decisions={r['id']:r for r in admission['cases']}
    cases=[];sheets=[];protection_dir=OUT/'preparation';protection_dir.mkdir()
    for ref in admission['selected_reference_ids']:
        rect=oldreviews[ref]['protection_rectangle_inclusive']
        base=np.zeros((256,256),bool);left,top,right,bottom=rect;base[top:bottom+1,left:right+1]=True
        protected=distance_transform_edt(~base)<=GUARD
        footprint=np.zeros((256,256),bool);footprint[64:192,64:192]=True
        assert protected[footprint].all()
        t=np.clip(distance_transform_edt(~protected)/FEATHER,0,1)
        alpha=(t*t*(3-2*t)).astype(np.float32)
        radius=int(3*SIGMA+.5);assert radius==4 and not alpha[protected].any()
        # A square finite Gaussian footprint cannot intersect corruption at any positive-alpha center.
        ys,xs=np.nonzero(alpha>0)
        intersects=(xs+radius>=64)&(xs-radius<192)&(ys+radius>=64)&(ys-radius<192)
        assert not intersects.any()
        for role in ('clear','global_gaussian_1p2','local_gaussian_1p2'):
            c=case_lookup[(ref,role)];inp=rgb(bound(c['input']))
            if role!='clear':assert decisions[c['id']]['decision']=='readable'
            target=rgb(bound(c['target']))
            if role=='clear':assert np.array_equal(inp,target)
            folder=protection_dir/c['id'];folder.mkdir()
            softened=np.rint(gaussian_filter(inp.astype(np.float64),(SIGMA,SIGMA,0),mode='reflect',truncate=3)).clip(0,255).astype(np.uint8)
            derived=np.rint(inp.astype(np.float64)*(1-alpha[...,None].astype(np.float64))+softened.astype(np.float64)*alpha[...,None].astype(np.float64)).clip(0,255).astype(np.uint8)
            assert np.array_equal(derived[protected],inp[protected])
            png(folder/'derived.png',derived);png(folder/'protection.png',(protected*255).astype(np.uint8))
            with (folder/'alpha.npy').open('xb') as f:np.save(f,alpha,allow_pickle=False)
            colors=np.zeros_like(inp);colors[:]=(200,60,60);colors[(alpha>0)&(alpha<1)]=(245,180,45);colors[protected]=(45,185,90)
            overlay=np.rint(inp.astype(np.float64)*.65+colors.astype(np.float64)*.35).astype(np.uint8)
            png(folder/'overlay.png',overlay)
            _,support,_,geometry=iohelper.prepare_crop(inp)
            _,support2,_,derived_geometry=iohelper.prepare_crop(derived);assert support.all() and np.array_equal(support,support2)
            row={**c,'derived':pin(folder/'derived.png'),'protection':pin(folder/'protection.png'),
                 'alpha':pin(folder/'alpha.npy'),'overlay':pin(folder/'overlay.png'),
                 'protection_rectangle_inclusive':rect,'protected_pixels':int(protected.sum()),
                 'geometry':geometry,'derived_geometry':derived_geometry,
                 'input_review':'readable' if role!='clear' else 'historical clear control; retained exact original',
                 'appearance_flag':oldreviews[ref]['appearance_flag'],'optimization_permitted':False,
                 'forced_harm_control':role!='local_gaussian_1p2','reused_original':None}
            if role=='clear':
                prior=oldlookup[ref+'__clear']['outputs']['original:spatial_u']
                for item in prior.values():bound(item)
                archive=bound(prior['ZIP']);metadata=doc(ROOT/prior['metadata']['path'])
                iohelper.verify_payload('spatial_u',archive,metadata,bound(c['input']),geometry)
                row['reused_original']={'outputs':prior,'from_comparison':pin(PRIOR/'recovery_r1/comparison.json'),
                                        'exact_input_bytes':True,'same_frozen_settings':True}
            cases.append(row)
    # Review geometry and preparation visually before explicit plan confirmation.
    selected=[r for r in cases if r['role']=='local_gaussian_1p2']
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for start in range(0,len(selected),4):
        subset=selected[start:start+4];sheet=Image.new('RGB',(4*264+8,len(subset)*288+70),'#f4f2ed');d=ImageDraw.Draw(sheet)
        for col,label in enumerate(['Observed input','Protection overlay','Derived input','Binary protection']):d.text((8+264*col,8),label,font=font,fill='black')
        for index,row in enumerate(subset):
            y=46+index*288;d.text((8,y),row['reference_id'],font=font,fill='black')
            images=[rgb(bound(row['input'])),rgb(bound(row['overlay'])),rgb(bound(row['derived'])),np.repeat(np.asarray(Image.open(io.BytesIO(bound(row['protection']))))[...,None],3,axis=2)]
            for col,a in enumerate(images):sheet.paste(Image.fromarray(a),(8+264*col,y+22))
        path=protection_dir/f'review_{start//4+1:02d}.png';sheet.save(path);sheets.append(pin(path))
    total=len(cases)*2;new=total-len(selected);assert new<=LIMITS['maximum_new_forwards']
    plan={'complete':True,'status':'PREPARED_REQUIRES_ROOT_PROTECTION_REVIEW_BEFORE_INFERENCE','created_UTC':datetime.now(timezone.utc).isoformat(),
          'worker':pin(Path(__file__)),'candidate_manifest':pin(candidate_path),'input_admission':pin(admission_path),
          'source_pins':source_pins,'model_files':oldplan['model_files'],'model_manifest_and_reviews':oldplan['model_manifest_and_reviews'],
          'selection_contract':admission['selection_contract'],
          'prior_masks':pin(PRIOR/'root_input_review_v1.json'),'cases':cases,'selected_reference_ids':admission['selected_reference_ids'],
          'algorithm':{'sigma':SIGMA,'truncate':3.,'finite_spatial_radius':4,'mode':'reflect','dtype':'float64','quantization':'rint clip uint8',
                       'protection':'previous reviewed whole-face rectangle, Euclidean outward guard6','guard_pixels':GUARD,'outside_feather_pixels':FEATHER,
                       'alpha':'float32 smoothstep(distance_to_guarded_protection/8)',
                       'source_of_filter':'observed input only; never clean target','matches_declared_operator_not_exact_global_uniform_blur':True},
          'limits':LIMITS,'planned_new_forwards':new,'planned_reused_outputs':len(selected),'planned_total_outputs':total,
          'roles':['clear','global_gaussian_1p2','local_gaussian_1p2'],'primary_sigma':1.2,'sigma1p8_not_selected_for_this_run':True,
          'protection_sheets':sheets,'assisted_known_footprint_and_nominal_sigma_not_automatic_estimation':True,
          'clear_global_on_are_forced_harm_probes_not_normal_routing':True,
          'whole_face_protection_leaves_sharp_context_inside_guard':True,
          'earlier_milder_input_kernel_and_strength_changes_are_not_single_variable_causal_proof':True,
          'neural_training_optimizer_VM_calls_at_freeze':0,'native_or_FINAL':False,'model_qualified':False,
          'retry_or_resume':False}
    write(PLAN,plan)
    assert 'torch' not in sys.modules
    print(json.dumps({'plan':pin(PLAN),'photographs':len(selected),'new_forwards':new,'reused_outputs':len(selected),'total_outputs':total,'protection_review_sheets':sheets}))


def resources(start):
    assert time.monotonic()-start<LIMITS['wall_seconds']
    assert shutil.disk_usage(ROOT).free>=LIMITS['disk_reserve_bytes']
    assert psutil.virtual_memory().available>=LIMITS['RAM_reserve_bytes']
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<LIMITS['output_bytes']


def run(receipt,expected_sha):
    assert pin(PLAN)['sha256']==expected_sha
    plan=doc(PLAN);assert plan['worker']==pin(Path(__file__))
    confirmation=doc(OUT/'root_protection_review_v1.json')
    assert confirmation['complete'] and confirmation['plan']==pin(PLAN) and confirmation['confirmed_before_model_outputs']
    assert not (OUT/'comparison.json').exists() and not (OUT/'results').exists()
    for row in [plan['candidate_manifest'],plan['input_admission'],*plan['source_pins'],*plan['model_files'].values(),*plan['model_manifest_and_reviews']]:bound(row)
    iohelper,transport=load_helpers();receipt=Path(receipt).resolve();assert receipt.is_relative_to(ROOT/'scratch')
    server=doc(receipt);assert server['host']=='127.0.0.1' and server['port']!=61815 and psutil.pid_exists(server['pid'])
    start=time.monotonic();deadline=start+LIMITS['wall_seconds'];url=server['url']
    result={'complete':False,'plan':pin(PLAN),'worker':pin(Path(__file__)),'confirmation':pin(OUT/'root_protection_review_v1.json'),
            'server_receipt':pin(receipt),'cases':[],'new_forwards':0,'reused_outputs':0,'submitted_HTTP_calls':0,
            'training_optimizer_VM_calls':0,'model_qualified':False,'retry_or_resume':False}
    update(OUT/'comparison.json',result)
    try:
        before=transport.request(url+'/status',deadline=min(deadline,time.monotonic()+LIMITS['HTTP_seconds']));result['server_before']=before
        assert all(before[k]==0 for k in ['neural_forward_attempts','neural_forward_returns','neural_forward_successes','completed_comparisons'])
        assert before['models']['spatial_u']['available']
        for index,row in enumerate(plan['cases']):
            case={'id':row['id'],'reference_id':row['reference_id'],'role':row['role'],'complete':False,'outputs':{}}
            result['cases'].append(case)
            for state in ('original','prepared'):
                resources(start)
                if state=='original' and row['reused_original']:
                    case['outputs'][state]=row['reused_original']['outputs'];result['reused_outputs']+=1;continue
                upload=bound(row['input'] if state=='original' else row['derived'])
                geometry=row['geometry'] if state=='original' else row['derived_geometry']
                folder=OUT/'results'/row['id']/state;folder.mkdir(parents=True)
                boundary='FrozenMildContextStudyV1'
                prefix=''.join('--'+boundary+'\r\nContent-Disposition: form-data; name="'+k+'"\r\n\r\n'+v+'\r\n' for k,v in [('selection','spatial_u'),('input_review','usable')]).encode()
                body=prefix+('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="frozen.png"\r\nContent-Type: image/png\r\n\r\n').encode()+upload+('\r\n--'+boundary+'--\r\n').encode()
                result['active_call']={'id':row['id'],'state':state,'folder':folder.relative_to(ROOT).as_posix(),'uploaded_sha256':sha(upload)}
                result['submitted_HTTP_calls']+=1;update(OUT/'comparison.json',result)
                response=transport.request(url+'/compare',body=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary},folder=folder,deadline=min(deadline,time.monotonic()+LIMITS['HTTP_seconds']))
                assert set(response['outputs'])=={'spatial_u'}
                archive=iohelper.blob(response['bundle']);metadata=response['metadata']
                raw,pixels=iohelper.verify_payload('spatial_u',archive,metadata,upload,geometry)
                assert iohelper.blob(response['outputs']['spatial_u'])==pixels
                with (folder/'app-comparison.zip').open('xb') as f:f.write(archive)
                write(folder/'metadata.json',metadata)
                with (folder/'raw.npy').open('xb') as f:np.save(f,raw,allow_pickle=False)
                with (folder/'result.png').open('xb') as f:f.write(pixels)
                case['outputs'][state]={key:pin(folder/name) for key,name in [('PNG','result.png'),('RAW','raw.npy'),('ZIP','app-comparison.zip'),('metadata','metadata.json'),('body','response.body'),('headers','response-headers.json')]}
                result['new_forwards']+=1;result.pop('active_call');update(OUT/'comparison.json',result)
            case['complete']=True;result['seconds']=time.monotonic()-start;update(OUT/'comparison.json',result)
            print(json.dumps({'complete_cases':index+1,'of':len(plan['cases']),'id':row['id'],'new_forwards':result['new_forwards'],'seconds':round(result['seconds'],1)}),flush=True)
        after=transport.request(url+'/status',deadline=min(deadline,time.monotonic()+LIMITS['HTTP_seconds']));result['server_after']=after
        count=plan['planned_new_forwards'];assert result['new_forwards']==result['submitted_HTTP_calls']==count
        assert result['reused_outputs']==plan['planned_reused_outputs']
        assert all(after[k]-before[k]==count for k in ['neural_forward_attempts','neural_forward_returns','neural_forward_successes','completed_comparisons'])
        assert all(after['per_arm']['spatial_u'][k]-before['per_arm']['spatial_u'][k]==count for k in ['attempts','returns','successes'])
        for arm in ('original','r2','spatial_s'):assert after['per_arm'][arm]==before['per_arm'][arm]
        result['complete']=True
    except BaseException as error:
        result['failure']={'type':type(error).__name__,'message':str(error),'do_not_retry_or_restart_on_timeout':True}
        try:
            result['server_failure_status']=transport.request(url+'/status',deadline=time.monotonic()+LIMITS['failure_telemetry_grace_seconds'])
        except BaseException as telemetry_error:
            result['failure_telemetry_unavailable']={'type':type(telemetry_error).__name__,'message':str(telemetry_error),
                'submitted_or_verified_count_is_not_proof_of_actual_completion':True}
        raise
    finally:
        result['seconds']=time.monotonic()-start;update(OUT/'comparison.json',result)
        assert 'torch' not in sys.modules


if __name__=='__main__':
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--admission');p.add_argument('--server-receipt');p.add_argument('--plan-sha');a=p.parse_args()
    freeze(a.admission) if a.freeze else run(a.server_receipt,a.plan_sha)
