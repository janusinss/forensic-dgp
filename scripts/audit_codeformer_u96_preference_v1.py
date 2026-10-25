"""Saved-only readback of the selected U96 application's functional evidence."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_u96_preference_v1'
UI=ROOT/'scratch/codeformer-u96-preference-v1'

def sha(data):return hashlib.sha256(data).hexdigest()
def pin(p):
    b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':sha(b)}
def blob(value):return base64.b64decode(value.split(',',1)[1],validate=True)
def pixels(data):return np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
def archive(data):
    z=zipfile.ZipFile(io.BytesIO(data));assert z.testzip() is None
    assert len(z.namelist())==len(set(z.namelist()))
    return z

def verify_result(data):
    z=archive(data)
    assert set(z.namelist())=={'original-upload.png','resize-control.png','observed-support.png',
        'spatial_u.png','spatial_u-raw.npy','processing.json'}
    meta=json.loads(z.read('processing.json'));assert meta['selection']=='spatial_u'
    assert meta['preferred_development_model']=='spatial_u' and meta['model_qualified'] is False
    assert meta['inference_only'] and meta['no_completion_or_covering_removal']
    row=meta['outputs']['spatial_u'];m=row['metadata']
    assert m['weights_sha256']=='958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'
    assert m['R2_weights_sha256']=='d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
    assert m['source_training_updates']==96 and m['all515_R2_tensors_frozen']
    raw=np.load(io.BytesIO(z.read('spatial_u-raw.npy')),allow_pickle=False)
    png=pixels(z.read('spatial_u.png'));control=pixels(z.read('resize-control.png'))
    support=np.asarray(Image.open(io.BytesIO(z.read('observed-support.png'))))>0
    assert raw.shape==(256,256,3) and raw.dtype==np.float32 and np.isfinite(raw).all()
    assert np.array_equal(png[~support],control[~support])
    assert np.array_equal(raw[~support],(control.astype(np.float32)/np.float32(255))[~support])
    quantized=(np.clip(raw,0,1)*np.float32(255)).astype(np.uint8)
    quantized[~support]=control[~support]
    assert np.array_equal(png,quantized)
    assert sha(raw.tobytes())==row['raw_rgb_sha256'] and sha(png.tobytes())==row['PNG_RGB8_sha256']
    assert sha(z.read('spatial_u.png'))==row['PNG_file_sha256']
    return z,raw,meta

def main():
    before=json.loads((OUT/'before_v1.json').read_bytes())
    for name,expected in before['models_before'].items():assert pin(ROOT/name)==expected
    ui=json.loads((ROOT/'scratch/codeformer-u96-preference-ui-v1.json').read_bytes())
    assert ui['complete'] and not ui['errors'] and len(ui['layouts'])==9 and len(ui['downloads'])==5
    assert {row['width'] for row in ui['layouts']}=={375,768,1280}
    assert all(not row['overflow'] for row in ui['layouts'])
    assert ui['status']['neural_forward_successes']==3 and ui['status']['per_arm']['spatial_u']['successes']==3
    assert all(ui['status']['per_arm'][a]['successes']==0 for a in ('r2','spatial_s','original'))
    normal=json.loads((UI/'restoration-response.json').read_bytes())
    pair=json.loads((UI/'context-response.json').read_bytes())
    nz,nraw,nmeta=verify_result(blob(normal['bundle']))
    pz=archive(blob(pair['bundle']))
    assert set(pz.namelist())=={'original-comparison.zip','prepared-comparison.zip','preprocessing.zip','comparison.json'}
    assert json.loads(pz.read('comparison.json'))==pair['metadata']
    oz,oraw,ometa=verify_result(pz.read('original-comparison.zip'))
    dz,draw,dmeta=verify_result(pz.read('prepared-comparison.zip'))
    assert np.array_equal(nraw,oraw) and nz.read('spatial_u.png')==oz.read('spatial_u.png')
    assert blob(normal['outputs']['spatial_u'])==nz.read('spatial_u.png')
    assert blob(pair['original_result'])==oz.read('spatial_u.png')
    assert blob(pair['prepared_result'])==dz.read('spatial_u.png')
    prep=archive(pz.read('preprocessing.zip'));pm=json.loads(prep.read('preprocessing.json'))
    assert prep.read('uploaded-original.bin')==(ROOT/'test_images/wild5.png').read_bytes()
    for name,row in pm['bundle_payloads'].items():
        data=prep.read(name);assert len(data)==row['bytes'] and sha(data)==row['sha256']
    original=pixels(prep.read('original-decoded-rgb.png'));derived=pixels(prep.read('derived-native.png'))
    mask=np.asarray(Image.open(io.BytesIO(prep.read('protection-native.png'))))>0
    interior=np.asarray(Image.open(io.BytesIO(prep.read('protected-interior-256.png'))))>0
    assert np.array_equal(original[mask],derived[mask])
    assert np.array_equal(pixels(prep.read('prepared-original-256.png'))[interior],pixels(prep.read('prepared-derived-256.png'))[interior])
    assert pm['protected_native_pixels']==int(mask.sum()) and pm['protected_prepared_interior_pixels']==int(interior.sum())
    assert pixels(nz.read('original-upload.png')).shape==original.shape
    assert np.array_equal(pixels(nz.read('original-upload.png')),original)
    assert np.array_equal(pixels(dz.read('original-upload.png')),derived)
    expected_downloads={'download-spatial_u':blob(normal['outputs']['spatial_u']),
        'research-bundle':blob(normal['bundle']),'download-original-context':blob(pair['original_result']),
        'download-prepared-context':blob(pair['prepared_result']),'context-bundle':blob(pair['bundle'])}
    for row in ui['downloads']:assert (ROOT/row['path']).read_bytes()==expected_downloads[row['id']]
    evidence=[pin(ROOT/row['screenshot']) for row in ui['layouts']]
    evidence += [pin(ROOT/row['path']) for row in ui['downloads']]
    evidence += [pin(UI/(n+'.json')) for n in ('restoration-response','context-response','preview-response','confirmed-preview-response')]
    result={'complete':True,'audit_pass':True,'author_saved_only_audit_not_independent_study':True,
        'UI_receipt':pin(ROOT/'scratch/codeformer-u96-preference-ui-v1.json'),'evidence':evidence,
        'model_bindings_unchanged':before['models_before'],'preferred_development_model':'spatial_u',
        'actual_U96_forwards':3,'download_files_verified':5,'responsive_layouts_verified':9,
        'same_original_raw_PNG_parity_across_routes':True,'native_protected_pixels_exact':int(mask.sum()),
        'prepared_protected_interior_pixels_exact':int(interior.sum()),'RAW_PNG_support_audit_pass':True,
        'model_qualified':False,'main_app_default_changed':False,'training_optimizer_VM_calls':0}
    path=OUT/'saved_flow_audit_v1.json'
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'audit':pin(path),'pass':True,'U96_forwards':3,'models_unchanged':True}))

if __name__=='__main__':main()
