"""Exact saved256 comparison sheet and descriptive output changes; no inference."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_wild5_surroundings_blur_v1_r1'
PRIOR=ROOT/'outputs/codeformer_wild5_whole_blur_v1_r1'
ARMS=('original','r2','spatial_u','spatial_s')


def pin(p):
    b=p.read_bytes()
    return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def pixels(p):
    with Image.open(p) as im:
        assert im.mode=='RGB' and im.size==(256,256)
        return np.array(im)


def main():
    experiment=json.loads((OUT/'experiment.json').read_bytes());assert experiment['complete']
    sheet=Image.new('RGB',(1328,1004),'#e6e4df');draw=ImageDraw.Draw(sheet)
    draw.text((8,6),'wild5: original vs whole-image blur vs surroundings-only blur | exact saved256 pixels',fill='black')
    draw.text((8,24),'Surroundings-only input keeps protected native face RGB exactly. Blur matching is approximate. No clean reference.',fill='black')
    for col,label in enumerate(('Input / resize control','Untouched CodeFormer','Our R2','U96 experimental','S96 experimental')):
        draw.text((8+264*col,49),label,fill='black')
    variants=[('Original input',PRIOR/'original_input'),
        ('Whole-image blur (earlier test): nominal radius1.2 at256, FACE ALSO REBLURRED',PRIOR/'whole_gaussian_1p2'),
        ('Surroundings-only: nominal radius6 at256; protected face UNCHANGED',OUT/'surroundings')]
    rows=[]
    for row,(label,folder) in enumerate(variants):
        y=82+row*306;draw.text((8,y),label,fill='black')
        paths=[folder/'prepared-input.png']+[folder/(arm+'.png') for arm in ARMS]
        for col,p in enumerate(paths):
            array=pixels(p);sheet.paste(Image.fromarray(array),(8+264*col,y+24))
        rows.append({'label':label,'image_pins':[pin(p) for p in paths]})
    path=OUT/'wild5_three_way_comparison.png';assert not path.exists();sheet.save(path)
    with Image.open(OUT/'protected_face_interior256.png') as im:interior=np.asarray(im)>0
    a=pixels(PRIOR/'original_input/prepared-input.png');b=pixels(OUT/'surroundings/prepared-input.png')
    assert np.array_equal(a[interior],b[interior])
    delta={}
    for arm in ARMS:
        before=np.load(PRIOR/'original_input'/(arm+'-raw.npy'),allow_pickle=False).astype(np.float64)
        after=np.load(OUT/'surroundings'/(arm+'-raw.npy'),allow_pickle=False).astype(np.float64)
        delta[arm]={'mean_absolute_generated_face_RGB_change':float(np.abs(after-before)[interior].mean()),
                    'measurement_is_output_difference_not_accuracy_or_recovered_detail':True}
    result={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'sheet':pin(path),'rows':rows,
        'root_personal_visual_review_complete':False,'exact256_pasting_no_display_processing':True,
        'protected_prepared_interior_input_RGB_equal':True,'protected_prepared_interior_pixels':int(interior.sum()),
        'descriptive_changes_relative_original_input':delta,'model_qualified':False,
        'new_training_inference_VM_calls':0,'no_clean_reference_no_PSNR_SSIM_accuracy':True}
    (OUT/'comparison_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'sheet':str(path),'root_visual_review_pending':True}))


if __name__=='__main__':main()
