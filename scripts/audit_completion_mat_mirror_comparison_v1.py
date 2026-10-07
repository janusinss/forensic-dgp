"""Independent saved-stage/PNG/support audit and finite CPU replay, never hidden metrics."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mat_mirror_completion_v1 import load_mat_mirror,sha
OUT=ROOT/'outputs/completion_mat_mirror_comparison_v1'


def rgb(path,mode='RGB'):
    with Image.open(path) as image:return np.asarray(image.convert(mode)).copy()


def main():
    start=time.monotonic();p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));r=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    assert r['complete'] and r['protocol_sha256']==sha(OUT/'protocol.json')
    assert r['requests']==32 and r['pretrained_forwards']==28 and r['empty_mask_bypasses']==4
    assert r['generator_state_before']==r['generator_state_after'] and r['assisted_only'] and r['app_unchanged']
    assert r['optimizer_updates']==r['gradient_calls']==0 and not r['optimizer_constructed']
    for name,digest in p['sources_sha256'].items():assert sha(ROOT/name)==digest,name
    assert [row['id'] for row in r['rows']]==[case['id'] for case in p['cases']]
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    counts={'outside':0,'empty':0,'nonempty':0};measurements=[]
    for case,row in zip(p['cases'],r['rows']):
        assert time.monotonic()-start<180,'Saved-stage audit180s cap'
        source=rgb(ROOT/case['input']);mask=rgb(ROOT/case['reviewed'],'L')==255
        assert sha(OUT/row['output'])==row['output_sha256'] and sha(OUT/row['completed256'])==row['completed256_sha256']
        completed=np.load(OUT/row['completed256'],allow_pickle=False);delivered=rgb(OUT/row['output'])
        assert completed.shape==(256,256,3) and completed.dtype==np.float32 and np.isfinite(completed).all()
        assert completed.min()>=0 and completed.max()<=1
        assert np.array_equal(completed[~mask],(source.astype(np.float32)/255)[~mask])
        assert np.array_equal(delivered[~mask],source[~mask]);counts['outside']+=int((~mask).sum())*3
        assert np.array_equal(delivered,np.rint(completed*255).clip(0,255).astype(np.uint8))
        if mask.any():
            assert row['raw512'] and sha(OUT/row['raw512'])==row['raw512_sha256']
            raw=np.load(OUT/row['raw512'],allow_pickle=False);assert raw.shape==(512,512,3) and raw.dtype==np.float32 and np.isfinite(raw).all()
            estimate=(torch.from_numpy(raw).permute(2,0,1)[None]+1).mul(.5).clamp(0,1)
            estimate=F.interpolate(estimate,(256,256),mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy()
            assert np.array_equal(estimate[mask],completed[mask]);counts['nonempty']+=1
        else:
            assert row['raw512'] is None and row['empty_mask_bypass'] and np.array_equal(delivered,source);counts['empty']+=1
        assert row['mask_pixels']==int(mask.sum()) and row['mode']=='assisted' and row['hidden_PSNR_SSIM'] is None
        assert row['seconds']<=p['budgets']['case_seconds']
        measurements.append({'id':case['id'],'raw_display_composition_verified':True,'visible_source_bytes':int((~mask).sum())*3})
    assert counts['empty']==4 and counts['nonempty']==28
    assert all(sha(ROOT/name)==digest for name,digest in p['app_preservation_sha256'].items())
    # Independently replay a fixed hand/eyes case once; no selection by generated quality.
    target=next(c for c in p['cases'] if c['id']=='val_18_hand_eyes_native')
    model,prov=load_mat_mirror();input8=rgb(ROOT/target['input']);mask8=rgb(ROOT/target['reviewed'],'L')==255
    x=torch.from_numpy(input8.astype(np.float32)/255).permute(2,0,1)[None]
    mask=torch.from_numpy(mask8.astype(np.float32))[None,None]
    result,raw=model(x,mask,return_raw=True)
    cached=np.load(OUT/'stages'/(target['id']+'_raw512.npy'),allow_pickle=False)
    error=float(np.abs(raw[0].permute(1,2,0).numpy()-cached).max());assert error<=1e-6
    png=(result[0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
    assert np.array_equal(png,rgb(OUT/'images'/(target['id']+'.png')))
    report={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':sha(OUT/'protocol.json'),
            'results_sha256':sha(OUT/'results.json'),'all32_saved_outputs_verified':True,
            'all28_raw512_to256_compositions_verified':True,'exact_visible_source_bytes':counts['outside'],
            'all4_clear_controls_exact_bypasses':True,'independent_CPU_replay':{'id':target['id'],'forwards':1,'maximum_raw_error':error,'exact_PNG':True},
            'full_family_visual_review_pending':True,'assisted_only':True,'automatic_quality_qualification':False,
            'assisted_quality_qualification':False,'hidden_reference_metrics':None,'measurements':measurements,
            'app_unchanged':True,'gradient_calls':0,'optimizer_updates':0,'goal_complete':False,'seconds':time.monotonic()-start}
    with (OUT/'independent_saved_output_audit.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='measurements'},indent=2))


if __name__=='__main__':main()
