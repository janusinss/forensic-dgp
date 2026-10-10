"""One finite corrected-mask Off ablation, existing pretrained component only."""
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cctv_dgp_pilot import state_hash
from dgp_face_workflow_v3 import DGPFaceWorkflow

OUT=ROOT/'outputs/completion_input_footprints_comparison_v1_r1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,data):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    assert not (OUT/'execution.json').exists(),'No automatic repeat of a failed/completed inference'
    p=read(OUT/'protocol.json');audit=read(OUT/'independent_mask_audit.json')
    assert audit['complete'] and audit['protocol_sha256']==sha(OUT/'protocol.json')
    for n,h in {**p['sources_sha256'],**p['app_preservation_sha256']}.items(): assert sha(ROOT/n)==h,n
    assert p['max_forwards']=={'completion':28,'internal512':28,'DGP':0,'detector':0}
    started=time.monotonic();torch.manual_seed(p['seed']);np.random.seed(p['seed'])
    engine=DGPFaceWorkflow(device='cpu');engine._runtime();model=engine._generator()
    before=state_hash(model);assert engine.restorer is None and engine.detector is None
    counts={'completion':0,'internal512':0,'DGP':0,'detector':0};captured={}
    def capture(name):
        def hook(module,args,result):
            counts[name]+=1;assert counts[name]<=p['max_forwards'][name]
            value=result[0] if name=='internal512' else result
            captured[name]=value.detach().cpu().numpy().copy()
        return hook
    handles=[model.register_forward_hook(capture('completion')),model.net.register_forward_hook(capture('internal512'))]
    write(OUT/'execution.json',{'protocol_sha256':sha(OUT/'protocol.json'),'mask_audit_sha256':sha(OUT/'independent_mask_audit.json'),
        'state_before':before,'device':'cpu','optimizer_constructed':False,'backward_calls':0,'cap_seconds':p['cap_seconds']})
    for folder in ['images','stages','metadata','pages']:(OUT/folder).mkdir()
    rows=[]
    try:
        for c in p['cases']:
            assert time.monotonic()-started<p['cap_seconds'],'600-second mask ablation time cap'
            with Image.open(ROOT/c['input']) as im:source=np.array(im.convert('RGB'))
            mask=np.zeros((256,256),np.uint8) if c['rejected'] else np.array(Image.open(ROOT/c['masks']['removal']).convert('L'))//255
            captured.clear()
            try:result=engine.generate(source,mask,'off',c['input_review'],True)
            except ValueError as exc:
                assert c['rejected'],str(exc)
                rows.append({'id':c['id'],'rejected_before_neural':True,'message':str(exc),'input_review':c['input_review']})
                assert not captured
                continue
            assert not c['rejected'] and result['metadata']['restoration_applied'] is False
            assert engine.restorer is None and engine.detector is None
            np.testing.assert_array_equal(result['output'][mask==0],source[mask==0])
            if mask.any():
                assert set(captured)=={'completion','internal512'}
                assert captured['internal512'].shape==(1,3,512,512) and captured['completion'].shape==(1,3,256,256)
            else:
                assert not captured and np.array_equal(result['output'],source)
            path='images/'+c['id']+'.png';Image.fromarray(result['output']).save(OUT/path)
            np.savez_compressed(OUT/'stages'/ (c['id']+'.npz'),**captured)
            write(OUT/'metadata'/ (c['id']+'.json'),result['metadata'])
            old=np.array(Image.open(ROOT/c['prior_output']).convert('RGB'))
            common=(mask==0)&(np.array(Image.open(ROOT/c['reviewed']).convert('L'))==0)
            np.testing.assert_array_equal(old[common],result['output'][common])
            rows.append({'id':c['id'],'rejected_before_neural':False,'output':path,'mask_pixels':int(mask.sum()),
                'empty_bypass':not bool(mask.any()),'source_changed_pixels_outside_mask':0,
                'shared_visible_old_new_exact':True,'metadata':'metadata/'+c['id']+'.json','stages':'stages/'+c['id']+'.npz'})
            print(json.dumps({'case':c['id'],'forwards':counts['completion'],'seconds':round(time.monotonic()-started,2)}),flush=True)
        after=state_hash(model);assert before==after and counts==p['max_forwards']
        assert len(rows)==36 and sum(r['rejected_before_neural'] for r in rows)==4 and sum(r.get('empty_bypass',False) for r in rows)==4
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13);pages=[]
        for condition in ['original_photo','synthetic_degraded_photo']:
            eligible=[c for c in p['cases'] if c['condition']==condition and not c['rejected']]
            assert len(eligible)==16
            for start in range(0,16,4):
                page=Image.new('RGB',(1280,1214),'white');draw=ImageDraw.Draw(page);entries=[]
                for k,label in enumerate(['Input256','Old assisted overlay','Previous Off estimate','New assisted overlay','New Off estimate']):
                    draw.text((k*256+4,7),label,font=font,fill='black')
                for i,c in enumerate(eligible[start:start+4]):
                    source=np.array(Image.open(ROOT/c['input']).convert('RGB'))
                    oldmask=np.array(Image.open(ROOT/c['reviewed']).convert('L'))!=0
                    newmask=np.array(Image.open(ROOT/c['masks']['removal']).convert('L'))!=0
                    def overlay(mask):
                        x=source.astype(np.float64);x[mask]=.55*x[mask]+.45*np.array([16,185,129]);return np.floor(x+.5).astype(np.uint8)
                    old=np.array(Image.open(ROOT/c['prior_output']).convert('RGB'))
                    new=np.array(Image.open(OUT/'images'/(c['id']+'.png')).convert('RGB'))
                    y=30+i*296
                    for k,x in enumerate([source,overlay(oldmask),old,overlay(newmask),new]):page.paste(Image.fromarray(x),(k*256,y))
                    draw.text((4,y+260),c['id']+' | '+c['family'],font=font,fill='black');entries.append({'id':c['id'],'row':i})
                path='pages/'+condition+'_'+str(start//4+1)+'.png';page.save(OUT/path)
                pages.append({'path':path,'sha256':sha(OUT/path),'entries':entries,'source_and_output_cells_exact256_unresampled':True})
        seconds=time.monotonic()-started;assert seconds<p['cap_seconds']
        artifacts={q.relative_to(OUT).as_posix():sha(q) for folder in ['images','stages','metadata','pages'] for q in sorted((OUT/folder).rglob('*')) if q.is_file()}
        artifact_bytes=sum((OUT/name).stat().st_size for name in artifacts)
        assert artifact_bytes<=p['artifact_cap_bytes']
        write(OUT/'results.json',{'complete':True,'protocol_sha256':sha(OUT/'protocol.json'),'seconds':seconds,
            'cap_seconds':p['cap_seconds'],'requests':36,'rows':rows,'forwards':counts,'state_before':before,'state_after':after,
            'artifacts_sha256':artifacts,'artifact_bytes':artifact_bytes,'pages':pages,'optimizer_updates':0,'backward_calls':0,'new_checkpoint':False,
            'automatic_forwards':0,'automatic_quality_qualification':False,'assisted_quality_qualification':False,
            'app_changes':False,'native_CCTV_or_reserved_final_used':False,'hidden_metrics':None,
            'visual_review_pending':True,'independent_final_review':False,'goal_complete':False})
        print(json.dumps({'complete':True,'seconds':seconds,'forwards':counts['completion'],'eligible_outputs':32,'state_unchanged':True}),flush=True)
    except BaseException as exc:
        write(OUT/'failure.json',{'complete':False,'error':repr(exc),'traceback':traceback.format_exc(),
            'protocol_sha256':sha(OUT/'protocol.json'),'seconds':time.monotonic()-started,'rows_completed':len(rows),
            'forwards':counts,'optimizer_updates':0,'app_changes':False,'automatic_repeat_permitted':False})
        raise
    finally:
        for h in handles:h.remove()


if __name__=='__main__':main()
