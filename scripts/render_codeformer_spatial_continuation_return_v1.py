"""Exact saved256 PNG contact sheets; no inference, filtering or resampling."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
RETURN=ROOT/'outputs/cctv_codeformer_spatial_continuation_v1_return'
REVIEW=ROOT/'outputs/codeformer_spatial_continuation_return_review_v1'


def load(path):return json.loads(path.read_bytes())


def main():
    p=load(RETURN/'protocol.json');folder=REVIEW/'visuals';folder.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    columns=['Observed input','Clean paired target / note','R2 fixed reference','U: ordinary feature loss','S: centered feature loss']
    plans=[]
    for update in (30,36,48,72,96):
        banks={}
        for phase in ('guards','strong'):
            for arm in ('U','S'):
                path=RETURN/f'outputs/spatial_continuation_v1/{phase}/{arm}/u{update:04d}/bank.json'
                if not path.exists():continue
                bank=load(path)
                banks[(phase,arm)]={r['id']:r for r in bank['cases']}
        for group in ('filter','clear','dark','strong'):
            if group=='strong' and update not in (48,72,96):continue
            sources=p['strong'] if group=='strong' else [r for r in p['guards'] if r['role']==group]
            phase='strong' if group=='strong' else 'guards'
            sheet=Image.new('RGB',(5*266+10,90+len(sources)*290),'#eeece7');draw=ImageDraw.Draw(sheet)
            draw.text((10,7),f'Spatial continuation | {group} TRAIN | update {update} | exact saved256 pixels',font=font,fill='black')
            draw.text((10,30),'No resampling/enhancement. TRAIN screen only; no native CCTV / DEV / FINAL evidence.',font=small,fill='black')
            for col,label in enumerate(columns):draw.text((10+266*col,60),label,font=small,fill='black')
            rows=[]
            for i,source in enumerate(sources):
                y=90+i*290;draw.text((10,y),source['id'],font=small,fill='black')
                current=[banks[(phase,arm)][source['id']] for arm in ('U','S')]
                bindings=[source['input'],source['target'],source['outputs']['png'],*[r['outputs']['png'] for r in current]]
                for col,binding in enumerate(bindings):
                    xy=(10+266*col,y+22)
                    if binding is None:
                        draw.rectangle((*xy,xy[0]+255,xy[1]+255),fill='#d6d3cd')
                        draw.text((xy[0]+12,xy[1]+80),'No paired clean target\nDark preservation anchor\nCompare with R2 behavior',font=small,fill='black')
                    else:
                        path=RETURN/binding['path'];b=path.read_bytes()
                        assert len(b)==binding['bytes'] and hashlib.sha256(b).hexdigest()==binding['sha256']
                        with Image.open(path) as image:
                            assert image.mode=='RGB' and image.size==(256,256)
                            sheet.paste(image,xy)
                rows.append({'id':source['id'],'columns':bindings})
            name=f'{group}_u{update:04d}.png';path=folder/name;sheet.save(path)
            plans.append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                          'group':group,'update':update,'rows':rows,'image_size':list(sheet.size)})
    manifest={'complete':True,'sheets':plans,'unique_cases':28,'case_rows':132,'candidate_PNG_outputs':264,
              'raw_outputs_numerically_audited_separately':True,'resampled_or_enhanced':False,
              'root_visual_review_pending':True,'model_qualified':False}
    (REVIEW/'visual_manifest_v1.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='sheets'}))


if __name__=='__main__':main()
