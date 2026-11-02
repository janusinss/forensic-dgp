"""Independent root replay of admitted saved PNGs and protected R1 pins.

No returned executable source, model, optimizer, neural work or VM is invoked.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/completion_codeformer_predictor_ce_v1_return'
RUN=BASE/'cctv_completion_codeformer_predictor_ce_v1_results'
STAGE=ROOT/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001/cctv_completion_codeformer_predictor_ce_v1_vm'

def need(ok,message):
    if not ok:raise ValueError(message)
def physical(path):
    p=Path(path).resolve()
    return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p
def sha(path):
    h=hashlib.sha256()
    with physical(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':physical(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(physical(path).read_text(encoding='utf-8-sig'))
def image(path,mode):
    with Image.open(physical(path)) as im:
        need(im.format=='PNG' and im.mode==mode and im.size==(256,256),'Exact image schema')
        return np.array(im)
def avg(rows):
    need(bool(rows),'Empty group')
    return {k:sum(r[k] for r in rows)/len(rows) for k in ('PNG_hole_L1','PNG_boundary_L1')}

def main():
    dest=BASE/'root_endpoint_and_preservation_closure.json'
    need(not dest.exists(),'Preserve previous closure')
    need(sha(BASE/'transport_audit.json')=='60bcc196d165e082010d67d1a62708792cf1f08dfa732921a58a1d65f13408c0','Transport pin')
    transport=read(BASE/'transport_audit.json');need(transport['complete'] is True,'Admitted transport')
    need(sha(RUN/'protocol.json')==sha(STAGE/'protocol.json')=='15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b','Protocol pin')
    for f in transport['files']:
        path=RUN/f['path']
        need(physical(path).stat().st_size==f['bytes'] and sha(path)==f['sha256'],'Returned payload drift '+f['path'])
    p=read(RUN/'protocol.json');result=read(RUN/'outputs/results.json')
    need(result['predictor_steps_completed_newphase']==32 and result['stop_reason'] is None,'Actual fixed32 endpoint required')
    rows=[];outside=0;PNGs=0;gray=[]
    for cycle in (0,8,16,32):
        for kind,key in (('monitor','monitor_cases'),('challenge','challenge_cases'),('public_development','development_cases')):
            if kind=='public_development' and cycle not in (0,32):continue
            summary=read(RUN/('outputs/u%04d'%cycle)/kind/'results.json');saved={r['id']:r for r in summary['records']}
            need(set(saved)=={r['id'] for r in p[key]},'Fixed case membership')
            for case in p[key]:
                ident=case['id'];source=image(STAGE/case['input256'],'RGB');mask=image(STAGE/case['mask'],'L')==255
                path=RUN/('outputs/u%04d'%cycle)/kind/(ident+'.png');PNG=image(path,'RGB')
                need(np.array_equal(PNG[~mask],source[~mask]),'Exact source outside mask '+ident)
                need(sha(path)==saved[ident]['PNG_sha256'],'Saved PNG digest differs')
                outside+=int((~mask).sum())*3;PNGs+=1
                if not mask.any():need(np.array_equal(PNG,source) and saved[ident]['empty'] is True,'Exact empty bypass');continue
                if kind=='public_development':continue
                target=image(STAGE/case['clean256'],'RGB')
                inverse=np.pad((~mask).astype(np.uint8),3,constant_values=0)
                boundary=mask & np.lib.stride_tricks.sliding_window_view(inverse,(7,7)).max(axis=(-1,-2)).astype(bool)
                delta=np.abs(PNG.astype(np.float64)-target)
                row={'id':ident,'cycle':cycle,'kind':kind,'mask_kind':case['mask_kind'],'condition':case['condition'],'style':case['style'],
                     'PNG_hole_L1':float(delta[mask].mean()/255),'PNG_boundary_L1':float(delta[boundary].mean()/255)}
                need(all(abs(row[k]-saved[ident][k])<=1e-14 for k in ('PNG_hole_L1','PNG_boundary_L1')),'PNG independent replay')
                if case['style']=='GRAY':
                    row['GRAY_PNG_mean_channel_spread']=float(np.ptp(PNG.astype(np.int16),axis=-1)[mask].mean())
                    need(abs(row['GRAY_PNG_mean_channel_spread']-saved[ident]['GRAY_PNG_mean_channel_spread'])<=1e-14,'PNG gray replay')
                    gray.append(row)
                rows.append(row)
    need(PNGs==256 and len(rows)==192,'All256 output scope')
    comparisons={}
    for cycle in (8,16,32):
        comparisons[str(cycle)]={}
        for kind in ('monitor','challenge'):
            b=[r for r in rows if r['cycle']==0 and r['kind']==kind];a=[r for r in rows if r['cycle']==cycle and r['kind']==kind]
            byid={r['id']:r for r in b};groups=[('macro',a,b)]
            groups += [('geometry/'+k,[r for r in a if r['mask_kind']==k],[r for r in b if r['mask_kind']==k]) for k in sorted({r['mask_kind'] for r in b})]
            if kind=='challenge':groups += [('condition/'+k,[r for r in a if r['condition']==k],[r for r in b if r['condition']==k]) for k in sorted({r['condition'] for r in b})]
            failures=[]
            for label,x,y in groups:
                xx=avg(x);yy=avg(y)
                for metric in xx:
                    if xx[metric]>yy[metric]+1e-12:failures.append({'group':label,'metric':metric,'baseline':yy[metric],'candidate':xx[metric],'delta':xx[metric]-yy[metric]})
            am,bm=avg(a),avg(b);gain=(bm['PNG_hole_L1']-am['PNG_hole_L1'])/bm['PNG_hole_L1']
            comparisons[str(cycle)][kind]={'PNG_hole_error_reduction_percent':100*gain,'baseline':bm,'candidate':am,'PNG_subgroup_failures':failures,
                'individual_holes_worse':[r['id'] for r in a if r['PNG_hole_L1']>byid[r['id']]['PNG_hole_L1']+1e-12],
                'individual_boundaries_worse':[r['id'] for r in a if r['PNG_boundary_L1']>byid[r['id']]['PNG_boundary_L1']+1e-12]}
            screen=result['numeric_screen']['clear_incremental_vs_R1' if kind=='monitor' else 'challenge_incremental_vs_R1'] if cycle==32 else read(RUN/('outputs/screen_u%04d.json'%cycle))[kind]
            need(abs(gain-screen['PNG_macro_hole_gain'])<=1e-12,'Saved macro gain differs')
    grayrows={cycle:[r for r in gray if r['cycle']==cycle] for cycle in (0,8,16,32)}
    need(all(len(v)==8 for v in grayrows.values()),'All8 gray challenge cases per snapshot')
    graycomparison={}
    for cycle in (8,16,32):
        def mean(part):return sum(r['GRAY_PNG_mean_channel_spread'] for r in part)/len(part)
        baseline=grayrows[0];candidate=grayrows[cycle]
        groups=[('macro',candidate,baseline)]+[('geometry/'+k,[r for r in candidate if r['mask_kind']==k],[r for r in baseline if r['mask_kind']==k]) for k in sorted({r['mask_kind'] for r in baseline})]
        graycomparison[str(cycle)]={label:{'baseline':mean(b),'candidate':mean(a),'delta':mean(a)-mean(b),'nonworse':mean(a)<=mean(b)+1e-12} for label,a,b in groups}
    a=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json');pins=[]
    for r in [*a['source_files'],a['candidate_manifest'],a['checkpoint']]:
        actual=pin(r['path']);need(actual['bytes']==r['bytes'] and actual['sha256']==r['sha256'],'R1 current app/model drift');pins.append(actual)
    need(len(pins)==22,'All22 R1 pins')
    report={'format':'root-predictor-ce-v1-all-saved-PNG-endpoint-preservation-closure-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),
            'transport':pin(BASE/'transport_audit.json'),'protocol':pin(RUN/'protocol.json'),'payload_files_rehashed':len(transport['files']),'PNG_outputs_replayed':PNGs,'paired_metric_observations':len(rows),'exact_outside_RGB_bytes':outside,
            'paired_PNG_records':rows,'separate_cohort_comparisons':comparisons,'GRAY_PNG_style':graycomparison,'R1_app_model_assets_exact':pins,
            'working_model':'R1 retained','candidate_adopted':False,'actual_training_local':False,'neural_calls':0,'VM_calls':0,'model_qualified':False,
            'scope':'Paired adaptive TRAIN hole/boundary evidence and32public previouslyexposed unpaired cases. Not restoration feature scores, hidden identity, human FINAL, native clean-reference evidence or full-family qualification.'}
    with physical(dest).open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'closure':pin(dest),'outputs':PNGs,'exact_outside_RGB_bytes':outside,'final32':comparisons['32'],'GRAY32':graycomparison['32']}))

if __name__=='__main__':main()
