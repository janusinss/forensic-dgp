"""Independent tensor-to-PNG/support arithmetic audit; no model construction."""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/completion_input_footprints_comparison_v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rgb(path):
    with Image.open(path) as im:
        assert im.size==(256,256);return np.array(im.convert('RGB'))


def binary(path):
    with Image.open(path) as im:
        assert im.mode=='L' and im.size==(256,256);a=np.array(im)
    assert set(np.unique(a))<={0,255};return a!=0


def main():
    started=time.monotonic();torch.set_num_threads(4)
    p,r,outer=read(OUT/'protocol.json'),read(OUT/'results.json'),read(OUT/'external_receipt.json')
    assert r['complete'] and outer['complete'] and not outer['timeout'] and outer['worker_exit_code']==0
    assert r['protocol_sha256']==outer['protocol_sha256']==sha(OUT/'protocol.json')
    assert outer['log_sha256']==sha(OUT/'inference.log')
    assert 0<r['seconds']<=p['cap_seconds']==600 and 0<outer['external_seconds']<=p['external_timeout_seconds']==630
    assert r['state_before']==r['state_after'] and r['forwards']==p['max_forwards']
    assert r['requests']==36 and not r['app_changes'] and not r['new_checkpoint']
    for n,h in {**p['sources_sha256'],**p['app_preservation_sha256']}.items():assert sha(ROOT/n)==h,n
    for n,h in r['artifacts_sha256'].items():assert sha(OUT/n)==h,n
    assert sum((OUT/n).stat().st_size for n in r['artifacts_sha256'])==r['artifact_bytes']<=p['artifact_cap_bytes']
    mask_audit=read(OUT/'independent_mask_audit.json');assert mask_audit['complete'] and mask_audit['protocol_sha256']==sha(OUT/'protocol.json')
    checked=completed=bypassed=rejections=visible_bytes=protected_bytes=0;rows=[]
    mapping={c['id']:c for c in p['cases']}
    assert len(r['rows'])==len(mapping)==36
    for row in r['rows']:
        c=mapping[row['id']]
        assert c['rejected']==row['rejected_before_neural']
        if c['rejected']:
            assert row['input_review'] in ['out_of_scope','needs_clearer'];rejections+=1
            assert not (OUT/'stages'/(c['id']+'.npz')).exists()
            continue
        src,out=rgb(ROOT/c['input']),rgb(OUT/row['output']);mask=binary(ROOT/c['masks']['removal']);protected=binary(ROOT/c['masks']['protected'])
        old=rgb(ROOT/c['prior_output']);oldmask=binary(ROOT/c['reviewed'])
        meta=read(OUT/row['metadata']);assert meta['restoration_requested']=='off' and not meta['restoration_applied']
        assert meta['mask_source']=='assisted_reviewed' and meta['additional_mask_expansion']==0
        assert meta['native_mask_sha256']==hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
        assert meta['original_rgb_sha256']==hashlib.sha256(src.tobytes()).hexdigest()
        expected=src.copy()
        with np.load(OUT/row['stages'],allow_pickle=False) as stages:
            if mask.any():
                assert set(stages.files)=={'completion','internal512'}
                raw,net=stages['completion'],stages['internal512']
                assert raw.shape==(1,3,256,256) and net.shape==(1,3,512,512) and raw.dtype==net.dtype==np.float32
                assert np.isfinite(raw).all() and np.isfinite(net).all() and raw.min()>=0 and raw.max()<=1
                # Tensor arithmetic only: reconstruct the internal512 clamp and
                # bilinear resize. No neural model, parameters or gradients.
                with torch.inference_mode():
                    resized=F.interpolate(((torch.from_numpy(net)+1)/2).clamp(0,1),size=(256,256),mode='bilinear',align_corners=False).numpy()
                np.testing.assert_array_equal(raw[0].transpose(1,2,0)[mask],resized[0].transpose(1,2,0)[mask])
                canonical=(src.astype(np.float32)/np.float32(255)).transpose(2,0,1)
                np.testing.assert_array_equal(raw[0].transpose(1,2,0)[~mask],canonical.transpose(1,2,0)[~mask])
                expected[mask]=np.floor(raw[0].transpose(1,2,0)[mask]*np.float32(255)).astype(np.uint8)
                completed+=1
                assert meta['completion']['weights_sha256']=='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
                assert meta['completion']['input_policy']=='visible-normalized-resize-v1'
            else:
                assert not stages.files and meta['completion']=={'bypassed':'empty mask'};bypassed+=1
        colour=meta['display_processing']['colour_policy']
        if colour['applied']:
            converted=np.repeat(cv2.cvtColor(expected,cv2.COLOR_RGB2GRAY)[...,None],3,axis=-1)
            expected[mask]=converted[mask]
        np.testing.assert_array_equal(out,expected)
        np.testing.assert_array_equal(out[~mask],src[~mask])
        np.testing.assert_array_equal(out[protected],src[protected])
        np.testing.assert_array_equal(out[(~mask)&(~oldmask)],old[(~mask)&(~oldmask)])
        assert meta['output_rgb_sha256']==hashlib.sha256(out.tobytes()).hexdigest()
        assert meta['optimizer_updates']==meta['backward_calls']==0
        visible_bytes+=int((~mask).sum())*3;protected_bytes+=int(protected.sum())*3
        rows.append({'id':c['id'],'condition':c['condition'],'family':c['family'],
            'source_changed_pixels_outside_mask':0,'protected_changed_pixels':0,'shared_visible_old_new_changed_pixels':0,
            'internal512_to256_exact':bool(mask.any()),'empty_bypass':not bool(mask.any()),'mask_pixels':int(mask.sum()),
            'colour_display_applied':bool(colour['applied'])})
        checked+=1;assert time.monotonic()-started<180
    assert checked==32 and completed==28 and bypassed==rejections==4
    assert len(r['pages'])==8
    for page in r['pages']:
        assert page['sha256']==sha(OUT/page['path']) and len(page['entries'])==4
        with Image.open(OUT/page['path']) as im:assert im.size==(1280,1214)
    assert not any(name in sys.modules for name in ['dgp_face_workflow_v3','pretrained_completion','third_party.codeformer.codeformer_arch'])
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':sha(OUT/'protocol.json'),
        'results_sha256':sha(OUT/'results.json'),'external_receipt_sha256':sha(OUT/'external_receipt.json'),
        'mask_audit_sha256':sha(OUT/'independent_mask_audit.json'),'seconds':time.monotonic()-started,'cap_seconds':180,
        'exact_outputs':32,'internal512_to256_compositions':28,'exact_empty_bypasses':4,'input_exclusions_retained':4,
        'visible_source_bytes_exact':visible_bytes,'protected_source_bytes_exact':protected_bytes,
        'shared_visible_old_new_exact':True,'source_bindings':len(p['sources_sha256']),'artifact_bindings':len(r['artifacts_sha256']),
        'rows':rows,'model_forwards':0,'gradient_calls':0,'optimizer_updates':0,'app_changes':False,
        'new_hidden_PSNR_SSIM':None,'native_CCTV_or_reserved_final_used':False,
        'automatic_quality_qualification':False,'assisted_quality_qualification':False,'independent_final_review':False,'goal_complete':False}
    with (OUT/'independent_saved_output_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ['complete','seconds','exact_outputs','internal512_to256_compositions','visible_source_bytes_exact']}))


if __name__=='__main__':main()
