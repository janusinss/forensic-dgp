"""Root geometry-stratified saved-code diagnosis; no network or model runtime.

These are geometric 16x16 token cells, not receptive fields or identity scores.
The clean-code labels were already retained for paired adaptive TRAIN evidence.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, zipfile, io
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/completion_codeformer_predictor_ce_v1_return'
RUN = BASE/'cctv_completion_codeformer_predictor_ce_v1_results'
STAGE = ROOT/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001/cctv_completion_codeformer_predictor_ce_v1_vm'
PROTOCOL_SHA = '15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b'

def need(ok, label):
    if not ok: raise ValueError(label)

def physical(path):
    p = Path(path).resolve()
    return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p

def sha(path):
    h = hashlib.sha256()
    with physical(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def pin(path):
    return {'path':str(Path(path).resolve()),'bytes':physical(path).stat().st_size,'sha256':sha(path)}

def read(path): return json.loads(physical(path).read_text(encoding='utf-8-sig'))

def array(path, name, shape, dtype):
    # Load only the desired bounded member, with no pickle/object support.
    with zipfile.ZipFile(physical(path)) as z:
        need(len(z.infolist())<=10 and len(set(z.namelist()))==len(z.namelist()), 'Bounded unique NPZ')
        info = z.getinfo(name+'.npy')
        need(info.file_size <= 1024+int(np.prod(shape))*np.dtype(dtype).itemsize, 'NPY size bound')
        value = np.load(io.BytesIO(z.read(info)), allow_pickle=False, max_header_size=512)
    need(value.shape==shape and value.dtype==np.dtype(dtype), 'Exact numeric schema '+name)
    need(np.isfinite(value).all(), 'Finite numeric array')
    return value

def image(path, mode):
    with Image.open(physical(path)) as im:
        need(im.format=='PNG' and im.mode==mode and im.size==(256,256), 'Exact PNG schema')
        return np.array(im)

def macro(rows):
    out = {}
    for region in ('all','touched','majority','fully_hidden','visible','crossing'):
        relevant = [r['regions'][region] for r in rows if r['regions'][region]['tokens']]
        out[region] = {
            'cases':len(relevant),
            'tokens':sum(r['tokens'] for r in relevant),
            'mean_case_agreement':float(np.mean([r['agreement'] for r in relevant])) if relevant else None,
            'pooled_token_agreement':sum(r['correct'] for r in relevant)/sum(r['tokens'] for r in relevant) if relevant else None,
            'mean_case_changed_fraction_from_R1':float(np.mean([r['changed_from_R1_fraction'] for r in relevant])) if relevant else None,
            'pooled_changed_tokens':sum(r['changed_from_R1_tokens'] for r in relevant),
        }
    return out

def main():
    need(not (BASE/'root_saved_code_geometry_diagnosis.json').exists(), 'Preserve prior diagnosis')
    transport = read(BASE/'transport_audit.json')
    need(transport.get('complete') is True, 'Wait for complete admitted transport')
    need(sha(RUN/'protocol.json')==PROTOCOL_SHA==sha(STAGE/'protocol.json'), 'Exact protocol')
    p = read(RUN/'protocol.json')
    result = read(RUN/'outputs/results.json')
    terminal = result['predictor_steps_completed_newphase']
    snapshots = [0]+[r['cycle'] for r in result['metric_snapshots']]
    need(snapshots==[0,8,16,32] and terminal==32, 'This diagnosis covers only the completed fixed32 endpoint')
    labels = {r['teacher_key']:r for r in p['cached_labels']}
    records = []
    metric_rows = {}
    for cycle in snapshots:
        for kind in ('monitor','challenge'):
            summary = read(RUN/('outputs/u%04d'%cycle)/kind/'results.json')
            metric_rows[(cycle,kind)] = {r['id']:r for r in summary['records']}
    for kind, casekey in (('monitor','monitor_cases'),('challenge','challenge_cases')):
        for case in p[casekey]:
            ident=case['id']; mask=image(STAGE/case['mask'],'L')==255
            fraction=mask.reshape(16,16,16,16).mean(axis=(1,3)).reshape(-1)
            regions={'all':np.ones(256,bool),'touched':fraction>0,'majority':fraction>=.5,'fully_hidden':fraction==1,'visible':fraction==0,'crossing':(fraction>0)&(fraction<1)}
            labelrow=labels[case['teacher_key']]
            labelpath=STAGE/labelrow['path']
            need(sha(labelpath)==labelrow['sha256'], 'Teacher label byte pin')
            clean=array(labelpath,'nearest_codes',(1,256),'int64').reshape(-1)
            initial=array(RUN/'outputs/u0000'/kind/(ident+'.npz'),'predicted_codes',(1,256),'int64').reshape(-1)
            previous=initial
            for cycle in snapshots:
                path=RUN/('outputs/u%04d'%cycle)/kind/(ident+'.npz')
                pred=array(path,'predicted_codes',(1,256),'int64').reshape(-1)
                need(np.all((pred>=0)&(pred<512)), 'Original512 code-ID range')
                regionrows={}
                for name,selected in regions.items():
                    n=int(selected.sum());correct=int(((pred==clean)&selected).sum());changed=int(((pred!=initial)&selected).sum())
                    regionrows[name]={'tokens':n,'correct':correct,'agreement':correct/n if n else None,'changed_from_R1_tokens':changed,'changed_from_R1_fraction':changed/n if n else None}
                saved=metric_rows[(cycle,kind)][ident]
                need(abs(regionrows['all']['agreement']-saved['same_codebook_clean_context_agreement_diagnostic'])<=1e-8, 'Saved whole-grid agreement differs')
                baseline=metric_rows[(0,kind)][ident]
                PNG = image(RUN/('outputs/u%04d'%cycle)/kind/(ident+'.png'),'RGB')
                basePNG=image(RUN/'outputs/u0000'/kind/(ident+'.png'),'RGB')
                need(np.array_equal(PNG[~mask],basePNG[~mask]), 'Outside PNG changed')
                row={'id':ident,'cohort':kind,'cycle':cycle,'mask_kind':case['mask_kind'],'style':case['style'],'condition':case['condition'],'regions':regionrows,
                     'changed_tokens_from_previous_snapshot':int((pred!=previous).sum()),'PNG_hole_L1':saved['PNG_hole_L1'],
                     'PNG_hole_error_reduction_percent_from_R1':100*(baseline['PNG_hole_L1']-saved['PNG_hole_L1'])/baseline['PNG_hole_L1'],
                     'PNG_mean_absolute_change_within_mask_from_R1':float(np.abs(PNG.astype(np.float64)-basePNG)[mask].mean()/255),
                     'predicted_codes_file':pin(path),'fixed_label_file':pin(labelpath)}
                records.append(row);previous=pred
    aggregates={}
    for kind in ('monitor','challenge'):
        aggregates[kind]={}
        for cycle in snapshots:
            group=[r for r in records if r['cohort']==kind and r['cycle']==cycle]
            aggregates[kind][str(cycle)]={'macro':macro(group),'geometry':{k:macro([r for r in group if r['mask_kind']==k]) for k in sorted({r['mask_kind'] for r in group})}}
    transitions=[]
    for kind in ('monitor','challenge'):
        finals=[r for r in records if r['cohort']==kind and r['cycle']==32]
        transition={'cohort':kind,'cases':len(finals),'code_touched_agreement_improved_and_PNG_worse':[], 'code_touched_agreement_worse_and_PNG_better':[]}
        byid={r['id']:r for r in records if r['cohort']==kind and r['cycle']==0}
        for r in finals:
            delta=r['regions']['touched']['agreement']-byid[r['id']]['regions']['touched']['agreement']
            if delta>0 and r['PNG_hole_error_reduction_percent_from_R1']<0:transition['code_touched_agreement_improved_and_PNG_worse'].append(r['id'])
            if delta<0 and r['PNG_hole_error_reduction_percent_from_R1']>0:transition['code_touched_agreement_worse_and_PNG_better'].append(r['id'])
        transitions.append(transition)
    report={'format':'root-predictor-ce-v1-saved-code-geometric-diagnosis-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'protocol':pin(RUN/'protocol.json'),'transport_audit':pin(BASE/'transport_audit.json'),'result':pin(RUN/'outputs/results.json'),
            'snapshots':snapshots,'paired_cases':48,'saved_code_observations':len(records),'aggregates':aggregates,'individual_cross_endpoint_disagreements':transitions,'records':records,
            'definitions':{'touched':'16x16 original256 image block contains any reviewed removal pixels','majority':'at least half its original256 pixels are masked','fully_hidden':'all original256 pixels in the block are masked','visible':'no reviewed removal pixel in its block','crossing':'both removal and visible pixels in its block'},
            'limits':['Geometric token cells are not the encoder receptive fields; convolutions and attention mix context.','Same-codebook clean labels were computed from paired adaptive TRAIN targets. They are unavailable hidden information at deployment.','Code agreement is a diagnostic, not face identity accuracy, calibrated confidence or PNG/anatomy quality.','Snapshots8/16 are safety diagnostics only, not candidate selection or recoverable checkpoints.','No causal attribution follows from a correlation, and distinct FIT examples cannot form a before/after learning curve.'],
            'local_neural_calls':0,'optimizer_updates':0,'VM_calls':0,'model_qualified':False,'R1_remains_selected':True}
    with physical(BASE/'root_saved_code_geometry_diagnosis.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'report':pin(BASE/'root_saved_code_geometry_diagnosis.json'),'cohort_macro':{k:{c:v['macro'] for c,v in d.items()} for k,d in aggregates.items()},'cross_endpoint_disagreements':transitions}))

if __name__=='__main__':main()
