"""Finite CPU inference on frozen reviewed support; saved raw stages and all failures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mat_mirror_completion_v1 import load_mat_mirror,sha
OUT=ROOT/'outputs/completion_mat_mirror_comparison_v1'


def pixels(path,mode='RGB'):
    with Image.open(path) as image:return np.asarray(image.convert(mode)).copy()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def state_sha(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol-sha',required=True);args=parser.parse_args()
    assert sha(OUT/'protocol.json')==args.protocol_sha
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert p['status']=='frozen_before_generation' and p['budgets']['requests']==32 and p['assisted_only']
    for name,digest in p['sources_sha256'].items():assert sha(ROOT/name)==digest,name
    assert not (OUT/'execution.json').exists() and not (OUT/'images').exists()
    torch.set_num_threads(p['threads']);torch.set_grad_enabled(False)
    start=time.monotonic();rows=[];model=None;before=None
    (OUT/'images').mkdir();(OUT/'stages').mkdir();(OUT/'preview').mkdir()
    write(OUT/'execution.json',{'complete':False,'protocol_sha256':args.protocol_sha,'torch':str(torch.__version__),
          'device':'cpu','threads':p['threads'],'autograd_enabled':False,'optimizer_constructed':False,'optimizer_updates':0})
    try:
        model,provenance=load_mat_mirror();before=state_sha(model)
        assert not model.net.training and not any(t.requires_grad for t in model.parameters())
        print(json.dumps({'model_loaded_strictly':True,'state':before,'weight_tensors':provenance['weight_tensors']}),flush=True)
        for number,case in enumerate(p['cases']):
            assert time.monotonic()-start<p['budgets']['total_seconds'],'Finite total inference cap1200s'
            began=time.monotonic();image=pixels(ROOT/case['input']);mask8=pixels(ROOT/case['reviewed'],'L')
            assert image.shape==(256,256,3) and mask8.shape==(256,256) and np.isin(mask8,[0,255]).all()
            removal=mask8==255
            x=torch.from_numpy(image.astype(np.float32)/255).permute(2,0,1)[None]
            mask=torch.from_numpy(removal.astype(np.float32))[None,None]
            result,raw=model(x,mask,return_raw=True)
            result256=result[0].permute(1,2,0).cpu().numpy().copy()
            delivered=np.rint(result256*255).clip(0,255).astype(np.uint8)
            assert np.array_equal(delivered[~removal],image[~removal]),'Visible source support changed'
            name=case['id'];output=OUT/'images'/(name+'.png');Image.fromarray(delivered).save(output)
            completed=OUT/'stages'/(name+'_completed256.npy');np.save(completed,result256,allow_pickle=False)
            raw_path=None
            if raw is not None:
                raw_path=OUT/'stages'/(name+'_raw512.npy')
                np.save(raw_path,raw[0].permute(1,2,0).cpu().numpy(),allow_pickle=False)
            elapsed=time.monotonic()-began
            assert elapsed<=p['budgets']['case_seconds'],'Finite case cap90s'
            rows.append({'id':name,'family':case['family'],'condition':case['condition'],'mode':'assisted',
                         'output':output.relative_to(OUT).as_posix(),'output_sha256':sha(output),
                         'completed256':completed.relative_to(OUT).as_posix(),'completed256_sha256':sha(completed),
                         'raw512':raw_path.relative_to(OUT).as_posix() if raw_path else None,
                         'raw512_sha256':sha(raw_path) if raw_path else None,'mask_pixels':int(removal.sum()),
                         'exact_outside_reviewed_mask':True,'empty_mask_bypass':not removal.any(),'seconds':elapsed,
                         'hidden_PSNR_SSIM':None,'exact_hidden_identity_claim':False})
            write(OUT/(f'progress_{number+1:02d}.json'),{'complete':False,'finished':number+1,'row':rows[-1]})
            print(json.dumps({'case':number+1,'of':32,'id':name,'seconds':round(elapsed,3),'forward_count':model.forwards}),flush=True)
        after=state_sha(model);assert before==after and model.forwards==28 and len(rows)==32
        assert sha(OUT/'protocol.json')==args.protocol_sha and time.monotonic()-start<=p['budgets']['total_seconds']
        for name,digest in p['app_preservation_sha256'].items():assert sha(ROOT/name)==digest
        for page in range(8):
            sheet=Image.new('RGB',(1024,4*280+28),'#16181c');draw=ImageDraw.Draw(sheet)
            for col,label in enumerate(['input256','reviewed removal','current CodeFormer Off','converted MAT512->256']):
                draw.text((col*256+4,5),label,fill='white')
            for slot,case in enumerate(p['cases'][page*4:(page+1)*4]):
                image=pixels(ROOT/case['input']);mask=pixels(ROOT/case['reviewed'],'L')==255
                overlay=image.astype(np.float32);overlay[mask]=.6*overlay[mask]+.4*np.array([255,120,20])
                images=[image,np.rint(overlay).astype(np.uint8),pixels(ROOT/case['output']),pixels(OUT/'images'/(case['id']+'.png'))]
                for col,array in enumerate(images):sheet.paste(Image.fromarray(array),(256*col,28+280*slot))
                draw.text((4,28+280*slot+258),case['id']+' / assisted',fill='white')
            sheet.save(OUT/'preview'/(f'page_{page+1:02d}.png'))
        assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<=p['budgets']['maximum_output_bytes']
        report={'complete':True,'protocol_sha256':args.protocol_sha,'requests':32,'pretrained_forwards':28,
                'empty_mask_bypasses':4,'all_seven_covering_families_included':True,'assisted_only':True,
                'rows':rows,'provenance':provenance,'generator_state_before':before,'generator_state_after':after,
                'state_unchanged':True,'app_unchanged':True,'optimizer_constructed':False,'optimizer_updates':0,
                'gradient_calls':0,'seconds':time.monotonic()-start,'automatic_quality_qualification':False,
                'assisted_quality_qualification':False,'independent_final_review':False,'goal_complete':False}
        write(OUT/'results.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['rows','provenance']},indent=2))
    except BaseException as exc:
        write(OUT/'failure.json',{'complete':False,'protocol_sha256':args.protocol_sha,'cause':repr(exc),
              'traceback':traceback.format_exc(),'finished_rows':rows,'model_forwards':model.forwards if model else 0,
              'optimizer_updates':0,'gradient_calls':0,'seconds':time.monotonic()-start,'goal_complete':False})
        raise


if __name__=='__main__':main()
