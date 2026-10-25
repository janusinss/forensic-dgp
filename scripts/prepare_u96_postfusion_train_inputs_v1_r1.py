"""Four TRAIN-source preflight witnesses; no model or optimization entrypoint."""
from pathlib import Path
import hashlib
import io
import json
import random
import shutil
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_u96_postfusion_v1_preparation_r1'
PACKET=ROOT/'scratch/u96_postfusion_train_input_admission_v1_r1'
SOURCE=ROOT/'outputs/codeformer_original_spatial_correction_v1_preparation/data_audit_v1/root_clean_target_review_v1.json'
SOURCE_SHA='2e06c37a59269a19582b4d3066af456ae6b330dcf8fd3f73b54f07366e9b804f'

def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p):
    b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':sha(b)}
def bound(row):
    p=ROOT/row['path'];assert p.resolve().is_relative_to(ROOT) and not p.is_symlink()
    assert pin(p)=={k:row[k] for k in ['path','bytes','sha256']};return p.read_bytes()
def write(p,a):
    with p.open('x',encoding='utf-8') as f:json.dump(a,f,indent=2,allow_nan=False);f.write('\n')
def save_png(p,a):
    with p.open('xb') as f:Image.fromarray(a).save(f,format='PNG')

def main():
    start=time.monotonic();assert not OUT.exists() and not PACKET.exists()
    assert pin(SOURCE)['sha256']==SOURCE_SHA and shutil.disk_usage(ROOT).free>=4*1024**3
    source=json.loads(SOURCE.read_bytes());assert source['complete'] and source['clean_target_sources_admitted']==64
    refs=source['admitted_references'][:4];assert len({r['reference_id'] for r in refs})==4
    criteria_path=ROOT/'outputs/codeformer_filter_pairs_v1_preparation/input_only_review_criteria_v1.json'
    criteria=json.loads(criteria_path.read_bytes())
    OUT.mkdir();PACKET.mkdir();(OUT/'inputs').mkdir();(OUT/'targets').mkdir()
    sources=[];cases=[]
    for row in refs:
        assert row['historical_role']=='TRAIN' and row['reference_id'] in source['admitted_reference_ids']
        source_bytes=bound(row['physical_source']);target_bytes=bound(row['clean256_existing'])
        with Image.open(io.BytesIO(source_bytes)) as im:
            assert im.mode=='RGB' and im.size==(1024,1024)
            clean=np.asarray(im.resize((256,256),Image.Resampling.LANCZOS)).copy()
        with Image.open(io.BytesIO(target_bytes)) as im:assert im.mode=='RGB' and np.array_equal(clean,np.asarray(im))
        target=OUT/'targets'/(row['reference_id']+'.png');target.write_bytes(target_bytes)
        sources.append({'reference_id':row['reference_id'],'historical_role':'TRAIN','physical_source':row['physical_source'],
                        'clean_target':pin(target),'attribution':row['attribution'],'retained_clean_admission_review':pin(SOURCE),
                        'source_is_genuine1024_not_upsampled128':True,'new_corruption_admission':'PENDING',
                        'subject_and_pretraining_overlap':'unknown','optimization_permitted':False})
        global_=np.rint(gaussian_filter(clean.astype(np.float64),(1.2,1.2,0),mode='reflect',truncate=3.)).clip(0,255).astype(np.uint8)
        local=clean.copy();local[64:192,64:192]=global_[64:192,64:192]
        outside=np.ones((256,256),bool);outside[64:192,64:192]=False;assert np.array_equal(local[outside],clean[outside])
        for role,a in [('clear',clean),('global_gaussian_1p2',global_),('local_gaussian_1p2',local)]:
            id_=row['reference_id']+'__'+role;p=OUT/'inputs'/(id_+'.png')
            if role=='clear':p.write_bytes(target_bytes)
            else:save_png(p,a)
            cases.append({'id':id_,'reference_id':row['reference_id'],'role':role,'split':'TRAIN','input':pin(p),
                          'target':pin(target),'input_only_readability':'PENDING' if role!='clear' else 'historical clean source control',
                          'optimization_permitted':False,'paired_clean_target':True})
    degraded=[r for r in cases if r['role']!='clear'];random.Random(2026102106).shuffle(degraded);mapping=[]
    for i,row in enumerate(degraded,1):
        p=PACKET/f'input_{i:02d}.png';shutil.copyfile(ROOT/row['input']['path'],p)
        mapping.append({'anonymous_id':f'T{i:02d}','id':row['id'],'reference_id':row['reference_id'],'role':row['role'],
                        'original_input':row['input'],'anonymous_input':pin(p)})
    sheets=[]
    for start_ in (0,4):
        canvas=Image.new('RGB',(1088,296),'#eeeae3');draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(mapping[start_:start_+4]):
            draw.text((8+i*272,8),row['anonymous_id'],fill='black')
            with Image.open(ROOT/row['anonymous_input']['path']) as im:canvas.paste(im,(8+i*272,32))
        p=PACKET/f'inputs_{start_//4+1:02d}.png';canvas.save(p);sheets.append(pin(p))
    write(PACKET/'review_rules.json',{'purpose':'Prospective TRAIN preflight witness admission only; no optimization permission or quality claim.',
        'observation':criteria['observation'],'decisions':criteria['decisions'],
        'prohibited_decision_evidence':criteria['prohibited_decision_evidence'],
        'access_limit':'Read only this scratch directory; no target, model, output, mapping, memory or parent-history access.',
        'case_count':8,'sheet_count':2,'sheets':sheets,'clean_reference_or_model_outputs_in_packet':False})
    write(OUT/'anonymous_input_map_v1.json',{'complete':True,'cases':mapping,'rules':pin(PACKET/'review_rules.json'),'sheets':sheets})
    result={'complete':True,'status':'PREPARATION_ONLY_NOT_AN_ADMITTED_OR_SEALED_VM_PACKET','script':pin(Path(__file__).resolve()),
            'source_clean_review':pin(SOURCE),'criteria':pin(criteria_path),'selection':'First4 of final64 admitted clean TRAIN references in original preserved order; chosen before any new output.',
            'sources':sources,'cases':cases,'source_count':4,'input_count':12,'anonymous_degraded_count':8,
            'operator':{'function':'scipy.ndimage.gaussian_filter','dtype':'float64','sigma':[1.2,1.2,0],'mode':'reflect','truncate':3.,'finite_radius':4,'quantization':'rint clipRGB8','local_half_open_ROI':[64,64,192,192]},
            'mapping':pin(OUT/'anonymous_input_map_v1.json'),'rules':pin(PACKET/'review_rules.json'),'input_only_review_complete':False,
            'no_DEV_FINAL_native_or_wild_inputs':True,'terms':'Exact retained source terms snapshots; not current web verification.','optimization_permitted':False,
            'training_packet_sealed':False,'VM_preflight_run':False,'Torch_imported':False,'neural_gradient_optimizer_training_VM_calls':0}
    assert time.monotonic()-start<120 and sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<32*1024**2
    assert 'torch' not in sys.modules
    path=OUT/'candidate_manifest_v1.json';write(path,result)
    print(json.dumps({'complete':True,'manifest':pin(path),'sources':4,'inputs':12,'degraded_review_inputs':8,'neural_training_VM_calls':0}))

if __name__=='__main__':main()
