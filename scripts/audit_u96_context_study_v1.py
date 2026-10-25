"""Independent saved-pixel reconstruction and metrics; no app/model/HTTP calls."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import zipfile

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from scipy.ndimage import distance_transform_edt,gaussian_filter

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/codeformer_u96_context_study_v1'
OUT=BASE/'recovery_r1'
KEYS=('original:original','original:r2','original:spatial_u','prepared:spatial_u')
WEIGHTS={'original':'1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7',
    'r2':'d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896',
    'spatial_u':'958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'}

def sha(b):return hashlib.sha256(b).hexdigest()
def pin(p):
    b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':sha(b)}
def bound(row):
    p=ROOT/row['path'];assert p.resolve().is_relative_to(ROOT) and not p.is_symlink()
    b=p.read_bytes();assert len(b)==row['bytes'] and sha(b)==row['sha256'],row['path'];return b
def rgb(b):
    with Image.open(io.BytesIO(b)) as image:
        assert image.mode=='RGB' and image.size==(256,256)
        return np.asarray(image).copy()
def mask(b):
    with Image.open(io.BytesIO(b)) as image:
        assert image.mode=='L' and image.size==(256,256)
        a=np.asarray(image).copy();assert set(np.unique(a)).issubset({0,255});return a>0
def blob(b):return base64.b64decode(b.split(',',1)[1],validate=True)
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
    plan_path=BASE/'comparison_plan_v1.json';plan=json.loads(plan_path.read_bytes())
    run=json.loads((OUT/'comparison.json').read_bytes());assert run['complete']
    assert run['plan']==pin(plan_path)
    assert len(plan['cases'])==len(run['cases'])==24
    review=json.loads((BASE/'root_input_review_v1.json').read_bytes())
    by_ref={r['reference_id']:r for r in review['references']}
    expected={(ref,role) for ref in review['selected_reference_ids'] for role in ('clear','global_gaussian_2p4','local_gaussian_3p0')}
    assert run['new_forwards']==run['submitted_HTTP_calls']==96
    assert run['transport_recovery']['combined_completed_forward_count']==97
    assert run['transport_recovery']['recovery_plan']==pin(OUT/'recovery_plan_v1.json')
    for field in ('neural_forward_attempts','neural_forward_returns','neural_forward_successes','completed_comparisons'):
        assert run['server_before'][field]==0 and run['server_after'][field]==96
    for arm,count in [('original',24),('r2',24),('spatial_u',48),('spatial_s',0)]:
        assert all(run['server_after']['per_arm'][arm][key]==count for key in ('attempts','returns','successes'))
    for name,row in plan['model_files'].items():assert pin(ROOT/row['path'])==row
    for row in [plan['candidate_manifest'],plan['root_input_review'],*plan['source_pins'],*plan['model_manifest_and_reviews']]:bound(row)
    assert len({row['id'] for row in plan['cases']})==24
    assert len({row['id'] for row in run['cases']})==24
    assert {(row['reference_id'],row['role']) for row in plan['cases']}==expected
    helper=ROOT/'scripts/audit_codeformer_spatial_dev_local_v1_r2.py'
    assert pin(helper)['sha256']=='619a7863031ede9d1dcb78fa0bd116014e514cfc7ee1e5fb3d4c7155bc91fb2b'
    spec=importlib.util.spec_from_file_location('retained_metric_helper',helper)
    metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    rows=[];csvrows=[]
    for source,saved in zip(plan['cases'],run['cases'],strict=True):
        assert source['id']==saved['id'] and set(saved['outputs'])==set(KEYS)
        ref=source['reference_id'];decision=by_ref[ref];assert decision['selected']
        assert source['appearance_flag']==decision['appearance_flag']
        assert decision['input_readability'][source['role']]=='usable'
        inp=rgb(bound(source['input']));target=rgb(bound(source['target']));derived=rgb(bound(source['derived']))
        blur=lambda sigma:np.rint(gaussian_filter(target.astype(np.float64),(sigma,sigma,0),mode='reflect',truncate=3)).clip(0,255).astype(np.uint8)
        expected=target.copy()
        if source['role']=='global_gaussian_2p4':expected=blur(2.4)
        elif source['role']=='local_gaussian_3p0':expected[64:192,64:192]=blur(3.)[64:192,64:192]
        else:assert source['role']=='clear'
        assert np.array_equal(inp,expected)
        left,top,right,bottom=decision['protection_rectangle_inclusive']
        union=np.zeros((256,256),dtype=bool);union[top:bottom+1,left:right+1]=True
        protected=distance_transform_edt(~union)<=8
        assert np.array_equal(mask(bound(source['protection'])),protected)
        assert protected[64:192,64:192].all()
        distance=distance_transform_edt(~protected);t=np.clip(distance/16,0,1);alpha=(t*t*(3-2*t)).astype(np.float32)
        blurred=np.asarray(Image.fromarray(inp).filter(ImageFilter.GaussianBlur(radius=6.)))
        reconstructed=np.rint(inp.astype(np.float64)*(1-alpha[...,None].astype(np.float64))+blurred.astype(np.float64)*alpha[...,None].astype(np.float64)).clip(0,255).astype(np.uint8)
        assert np.array_equal(derived,reconstructed) and np.array_equal(derived[protected],inp[protected])
        assert not alpha[protected].any()
        pm=json.loads(bound(source['preprocessing_metadata']))
        assert pm['protected_native_RGB_values_exact'] and pm['protected_prepared_interior_RGB_exact']
        assert pm['source_sha256']==sha(bound(source['input']))
        assert pm['protection']=={'rectangle':decision['protection_rectangle_inclusive']}
        assert pm['Gaussian_native_radius']==6.
        assert pm['algorithm']['outward_guard_native_pixels']==8.
        assert pm['algorithm']['outside_feather_native_pixels']==16.
        assert pm['protected_native_pixels']==int(protected.sum())
        assert pm['protected_prepared_interior_pixels']==int(protected.sum())
        with zipfile.ZipFile(io.BytesIO(bound(source['preprocessing_bundle']))) as z:
            assert z.testzip() is None and json.loads(z.read('preprocessing.json'))==pm
            assert len(z.namelist())==len(set(z.namelist()))==10
            for name,item in pm['bundle_payloads'].items():
                data=z.read(name);assert len(data)==item['bytes'] and sha(data)==item['sha256']
            assert z.read('uploaded-original.bin')==bound(source['input'])
            for name in ('original-decoded-rgb.png','prepared-original-256.png'):
                assert np.array_equal(rgb(z.read(name)),inp)
            assert mask(z.read('observed-support-256.png')).all()
            assert np.array_equal(mask(z.read('protection-native.png')),protected)
            assert np.array_equal(mask(z.read('protected-interior-256.png')),protected)
            assert np.array_equal(rgb(z.read('prepared-derived-256.png')),derived)
            assert np.array_equal(np.load(io.BytesIO(z.read('blur-weight-float32.npy')),allow_pickle=False),alpha)
            assert np.array_equal(rgb(z.read('derived-native.png')),derived)
        outputs={};metrics={}
        for key in KEYS:
            kind,arm=key.split(':');pins=saved['outputs'][key]
            body_data=bound(pins['body']);body=json.loads(body_data)
            headers={k.lower():v for k,v in json.loads(bound(pins['headers'])).items()}
            assert int(headers['content-length'])==len(body_data)
            meta=json.loads(bound(pins['metadata']));assert body['metadata']==meta
            assert meta['selection']==arm and meta['inference_only'] and meta['model_qualified'] is False
            submitted=source['input'] if kind=='original' else source['derived']
            assert meta['uploaded_bytes_sha256']==sha(bound(submitted))
            image=rgb(bound(pins['PNG']));raw=np.load(io.BytesIO(bound(pins['RAW'])),allow_pickle=False)
            assert raw.shape==(256,256,3) and raw.dtype==np.float32 and np.isfinite(raw).all()
            assert raw.min()>=0 and raw.max()<=1
            assert np.array_equal(np.floor(raw*np.float32(255)).astype(np.uint8),image)
            assert blob(body['outputs'][arm])==bound(pins['PNG']) and set(body['outputs'])=={arm}
            declaration=meta['outputs'][arm];provenance=declaration['metadata']
            assert declaration['raw_rgb_sha256']==sha(raw.tobytes())
            assert declaration['PNG_RGB8_sha256']==sha(image.tobytes())
            assert provenance['arm']==arm and provenance['fidelity']==1. and provenance['adain']
            assert provenance['display_processing']=='none'
            assert provenance['weights_sha256']==WEIGHTS[arm]
            assert provenance['internal_resolution']==512 and provenance['output_resolution']==256
            if arm in ('original','r2'):
                expected_state='6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd' if arm=='original' else 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
                assert provenance['model_tensor_sha256']==expected_state
            if arm=='spatial_u':
                assert provenance['weights_sha256']=='958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'
                assert provenance['R2_weights_sha256']=='d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
                assert provenance['adapter_tensor_sha256']=='fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
                assert provenance['base_tensor_sha256']=='a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
            with zipfile.ZipFile(io.BytesIO(bound(pins['ZIP']))) as z:
                assert z.testzip() is None and len(z.namelist())==6 and len(set(z.namelist()))==6
                assert blob(body['bundle'])==bound(pins['ZIP'])
                assert z.read(arm+'.png')==bound(pins['PNG']) and z.read(arm+'-raw.npy')==bound(pins['RAW'])
                assert json.loads(z.read('processing.json'))==meta
                assert np.array_equal(rgb(z.read('resize-control.png')),inp if kind=='original' else derived)
                assert mask(z.read('observed-support.png')).all()
            outputs[key]=image;metrics[key]={}
            for format_,values in [('RAW',raw),('PNG',image.astype(np.float32)/np.float32(255))]:
                clean=target.astype(np.float32)/np.float32(255)
                scores={name:metric.metrics(values,clean,roi) for name,roi in [('whole_image',np.ones((256,256),bool)),('protected_region',protected),('surroundings',~protected)]}
                scores['central128_patch']=metric.metrics(values[64:192,64:192],clean[64:192,64:192],np.ones((128,128),bool))
                metrics[key][format_]=scores
                for region,score in scores.items():csvrows.append({'id':source['id'],'reference':ref,'role':source['role'],'appearance_flag':source['appearance_flag'],'condition':key,'format':format_,'region':region,**{m:score[m] for m in ['mae','psnr','ssim','adjacent_RGB_residual_gradient_mae','mean_centered_residual_mae','signed_luminance_bias']}})
        rows.append({'id':source['id'],'reference_id':ref,'role':source['role'],'appearance_flag':source['appearance_flag'],'protected_pixels':int(protected.sum()),'metrics':metrics,'source':source,'outputs':saved['outputs']})
    summaries=[]
    for role in ('clear','global_gaussian_2p4','local_gaussian_3p0'):
        selected=[r for r in rows if r['role']==role];assert len(selected)==8
        group={'role':role,'photographs':8,'arms':{},'preparation_vs_original_U96':{}}
        group['central128_patch_equals_known_local_corruption_footprint']=role=='local_gaussian_3p0'
        group['preparation_is_forced_negative_control']=role!='local_gaussian_3p0'
        for key in KEYS:
            group['arms'][key]={region:{m:float(np.mean([r['metrics'][key]['PNG'][region][m] for r in selected])) for m in ['mae','adjacent_RGB_residual_gradient_mae','mean_centered_residual_mae','signed_luminance_bias']} for region in ['whole_image','protected_region','surroundings','central128_patch']}
        for region in ['whole_image','protected_region','surroundings','central128_patch']:
            delta=[r['metrics']['prepared:spatial_u']['PNG'][region]['mae']-r['metrics']['original:spatial_u']['PNG'][region]['mae'] for r in selected]
            group['preparation_vs_original_U96'][region]={'better':sum(x< -1e-12 for x in delta),'equal':sum(abs(x)<=1e-12 for x in delta),'worse':sum(x>1e-12 for x in delta),'mean_MAE_difference':float(np.mean(delta))}
        summaries.append(group)
        for flag in (False,True):
            subset=[r for r in selected if r['appearance_flag']==flag]
            summaries.append({'role':role,'appearance_flag':flag,'photographs':len(subset),
                'central128_patch_PNG_MAE':{key:float(np.mean([r['metrics'][key]['PNG']['central128_patch']['mae'] for r in subset])) for key in KEYS},
                'subset_not_independent_replication':True})
    write(OUT/'saved_output_audit_v1.json',{'complete':True,'audit_pass':True,'auditor':pin(Path(__file__)),'comparison':pin(OUT/'comparison.json'),'plan':pin(plan_path),'retained_metrics':pin(helper),'summary':summaries,'cases':rows,'new_forwards':96,'native_or_FINAL':False,'paired_photo_count':8,'identity_pretraining_overlap_unknown':True,'automatic_detector_qualification':False,'model_qualified':False,'auditor_model_HTTP_training_VM_calls':0})
    with (OUT/'paired_region_metrics_v1.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    visuals=OUT/'visuals';visuals.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    coded=list(KEYS);random.Random(20261022).shuffle(coded)
    sheets=[]
    for role in ('clear','global_gaussian_2p4','local_gaussian_3p0'):
        selected=[r for r in rows if r['role']==role]
        for batch in range(0,8,4):
            subset=selected[batch:batch+4];sheet=Image.new('RGB',(8*264+8,4*288+110),(245,244,238));draw=ImageDraw.Draw(sheet)
            draw.text((8,8),f'U96 context study | {role} | exact256 cells | assisted masks, not automatic detector evidence',font=font,fill='black')
            for index,label in enumerate(['Input','Prepared input','Clean target','Protection','A','B','C','D']):draw.text((8+index*264,44),label,font=font,fill='black')
            for ri,row in enumerate(subset):
                y=82+ri*288;draw.text((8,y),row['reference_id']+(' | APPEARANCE FLAG' if row['appearance_flag'] else ''),font=font,fill='black')
                src=row['source'];cells=[rgb(bound(src['input'])),rgb(bound(src['derived'])),rgb(bound(src['target'])),np.repeat((mask(bound(src['protection']))*255).astype(np.uint8)[...,None],3,axis=2)]
                cells += [rgb(bound(row['outputs'][key]['PNG'])) for key in coded]
                for col,cell in enumerate(cells):sheet.paste(Image.fromarray(cell),(8+col*264,y+22))
            path=visuals/(f'{len(sheets)+1:02d}_{role}.png');sheet.save(path);sheets.append({'file':pin(path),'cases':[r['id'] for r in subset],'model_cells':16})
    write(visuals/'panel_key.json',dict(zip('ABCD',coded)))
    write(visuals/'manifest.json',{'complete':True,'review_complete':False,'sheets':sheets,'panel_key':pin(visuals/'panel_key.json')})
    assert 'torch' not in sys.modules
    print(json.dumps({'audit_pass':True,'cases':len(rows),'outputs':96,'visual_sheets':6,'summary':summaries}))

if __name__=='__main__':main()
