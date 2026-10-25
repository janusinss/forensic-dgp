"""Exact256 original/surroundings/whole stronger-blur comparison; no inference."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_wild5_whole_strong_blur_v1'
ARMS=('original','r2','spatial_u','spatial_s')


def pin(p):
    b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def main():
    x=json.loads((OUT/'experiment.json').read_bytes());assert x['complete']
    variants=[('Original wild5 input',ROOT/x['baseline_reuse']['original_folder']),
        ('Surroundings-only blur: nominal6 at256; protected face UNCHANGED',ROOT/x['baseline_reuse']['surroundings_folder']),
        ('Whole-image blur: SAME added nominal6 at256; face ALSO blurred again',OUT/'whole')]
    sheet=Image.new('RGB',(1328,1004),'#e6e4df');draw=ImageDraw.Draw(sheet)
    draw.text((8,6),'wild5: original vs surroundings-only vs WHOLE image at the same stronger blur setting',fill='black')
    draw.text((8,24),'Exact saved256 pixels. One unpaired stress image; sharper plausible features do not verify identity.',fill='black')
    for col,name in enumerate(('Input / resize control','Untouched CodeFormer','Our R2','U96 experimental','S96 experimental')):
        draw.text((8+264*col,49),name,fill='black')
    rows=[]
    for row,(label,folder) in enumerate(variants):
        y=82+306*row;draw.text((8,y),label,fill='black')
        paths=[folder/'prepared-input.png']+[folder/(a+'.png') for a in ARMS]
        for col,p in enumerate(paths):
            with Image.open(p) as im:
                assert im.mode=='RGB' and im.size==(256,256)
                sheet.paste(im,(8+264*col,y+24))
        rows.append({'label':label,'images':[pin(p) for p in paths]})
    path=OUT/'wild5_same_strength_three_way.png';assert not path.exists();sheet.save(path)
    result={'complete':True,'sheet':pin(path),'rows':rows,'candidate_outputs':12,'new_model_calls':4,
        'reused_outputs':8,'exact256_pasting':True,'root_personal_visual_review_complete':False,
        'added_kernel_matches_surroundings_transform':True,'uniform_final_blur_claimed':False,
        'no_clean_reference_no_PSNR_SSIM_accuracy':True,'model_qualified':False}
    (OUT/'comparison_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'sheet':str(path),'root_review_pending':True}))


if __name__=='__main__':main()
