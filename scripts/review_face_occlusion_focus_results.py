"""Fixed visual cohort and reflection-only recount of verified saved masks."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_face_occlusion_focus_results import ARCHIVE,EXTRACTION,AUDIT_OUTPUT,PREFIX,ARMS,EPOCHS
from scripts.audit_face_occlusion_results import require,read_json,verified_inputs
from scripts.audit_face_occlusion_glare_pixels import added_region_counts
from scripts.package_face_occlusion_vm import sha
from scripts.evaluate_coverage_results import binary
from detector_training import load_manifest,ReviewedMasks
from scripts.train_face_occlusion_focus_vm import PRIORITY_PATH,priority_masks


def fixed_preview(previous,rows):
    ids = previous.get('preview_validation_indices')
    require(isinstance(ids,list) and len(ids) == 10 and len(set(ids)) == 10 and
            all(type(i) is int and 0 <= i < len(rows) for i in ids), 'Previous preview cohort invalid')
    mannequin = [i for i,r in enumerate(rows) if Path(r['image']).name == 'new_covered_40.png']
    glare = [i for i,r in enumerate(rows) if r.get('glare_stratum') == 'strong_lens_reflection']
    require(len(mannequin) == len(glare) == 1 and all(i in ids for i in mannequin+glare),
            'Previous preview must include mannequin and reflection')
    return list(ids)


def main():
    out = ROOT/'outputs/face_occlusion_focus_review'; require(not out.exists(), 'Preserve existing review evidence')
    audit = read_json(AUDIT_OUTPUT/'results.json'); members = read_json(AUDIT_OUTPUT/'members.json')
    require(sha(ARCHIVE) == audit['archive_sha256'], 'Verified return archive changed')
    _,rows,data = verified_inputs(); returned = EXTRACTION/PREFIX
    prior_path = ROOT/'outputs/face_occlusion_continuation_validation/results.json'
    ids = fixed_preview(read_json(prior_path),rows)
    choices = [('source30','initial_masks')]+[(f'{a}/{e}',f'{a}/epoch_{e}_masks') for e in EPOCHS for a in ARMS]
    def prediction(folder,domain,i):
        path = returned/folder/domain/f'{i:04}.png'
        require(sha(path) == members[path.relative_to(EXTRACTION).as_posix()], 'Verified prediction mask changed')
        return binary(path)
    out.mkdir(); sheet = Image.new('RGB',(1024,148*len(ids)),'white')
    for line,i in enumerate(ids):
        x,m = data['real'][i]
        tiles = [Image.fromarray(np.rint(x.permute(1,2,0).numpy()*255).astype('uint8'))]
        masks = [m[0].numpy().astype(bool),prediction('baseline_masks','real',i)]
        masks.extend(prediction(folder,'real',i) for _,folder in choices)
        tiles.extend(Image.fromarray(mask.astype('uint8')*255).convert('RGB') for mask in masks)
        ImageDraw.Draw(sheet).text((2,line*148+2),
            f'Validation {i}: input | target | parent | source30 | control35 | focus35 | control40 | focus40',fill='black')
        for j,tile in enumerate(tiles): sheet.paste(tile.resize((128,128)),(j*128,line*148+20))
    sheet.save(out/'review_preview.png')
    registry = read_json(ROOT/PRIORITY_PATH); parent_path = ROOT/'dataset/detector_expanded_review_v2/manifest.json'
    v3_path = ROOT/'dataset/detector_glare_review_v3/manifest.json'
    require(sha(parent_path) == registry['v2_manifest_sha256'] and sha(v3_path) == registry['v3_manifest_sha256'],
            'Reviewed reflection lineage changed')
    old_rows = load_manifest(parent_path); old = {r['image_sha256']:r for r in old_rows}
    priorities = priority_masks(registry,data['training_real'].rows); records = []; visuals = []
    for domain,current_rows in [('training_real',data['training_real'].rows),('real',rows)]:
        for i,row in enumerate(current_rows):
            if row.get('glare_stratum') != 'strong_lens_reflection': continue
            original = old[row['image_sha256']]
            require(row['split'] == original['split'] and row['v2_mask_sha256'] == original['mask_sha256'],
                    'Reflection source/split lineage differs')
            x,m = data[domain][i]; target = m[0].numpy().astype(bool)
            parent = ReviewedMasks([original],256)[0][1][0].numpy().astype(bool); added = target & ~parent
            if domain == 'training_real': require(np.array_equal(added,priorities[i]), 'Registered training reflection differs')
            states = {}; visible_predictions = {}
            for label,folder in choices:
                p = prediction(folder,domain,i); states[label] = added_region_counts(p,target,parent)
                visible_predictions[label] = p & ~parent
            records.append({'domain':domain,'case':i,'image':row['image'],'image_sha256':row['image_sha256'],
                            'target_sha256':row['mask_sha256'],'v2_mask_sha256':original['mask_sha256'],'states':states})
            visuals.append((domain,i,x,added,visible_predictions))
    require([(r['domain'],r['case']) for r in records] == [('training_real',15),('training_real',46),('real',23)],
            'Reflection review cohort differs')
    summary = {}
    for label,_ in choices:
        summary[label] = {}
        for domain in ('training_real','real'):
            values = [r['states'][label] for r in records if r['domain'] == domain]
            pixels = sum(r['added_glare_pixels'] for r in values); recovered = sum(r['recovered_added_glare_pixels'] for r in values)
            summary[label][domain] = {'cases':len(values),'added_glare_pixels':pixels,
                                     'recovered_added_glare_pixels':recovered,'glare_recall':recovered/pixels}
    sheet = Image.new('RGB',(1024,148*len(visuals)),'white')
    for line,(domain,i,x,added,predictions) in enumerate(visuals):
        image = Image.fromarray(np.rint(x.permute(1,2,0).numpy()*255).astype('uint8')); yy,xx = np.where(added)
        box = (max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(256,int(xx.max())+13),min(256,int(yy.max())+13))
        tiles = [image,image.crop(box),Image.fromarray(added.astype('uint8')*255).convert('RGB').crop(box)]
        tiles.extend(Image.fromarray(predictions[label].astype('uint8')*255).convert('RGB').crop(box) for label,_ in choices)
        ImageDraw.Draw(sheet).text((2,line*148+2),
            f'{domain} {i}: input | zoom | V3 addition | source30 | control35 | focus35 | control40 | focus40',fill='black')
        for j,tile in enumerate(tiles): sheet.paste(tile.resize((128,128),Image.Resampling.NEAREST),(j*128,line*148+20))
    sheet.save(out/'reflection_preview.png')
    report = {'complete':True,'archive_sha256':audit['archive_sha256'],'script_sha256':sha(__file__),
              'audit_sha256':sha(AUDIT_OUTPUT/'results.json'),'previous_preview_sha256':sha(prior_path),
              'preview_validation_indices':ids,'priority_manifest_sha256':sha(ROOT/PRIORITY_PATH),
              'records':records,'reflection_summary':summary,'optimizer_updates_locally':0,'promoted':False,
              'preview_correction':'Original recount preview used a clear case22 instead of mannequin8; preserved. This review reuses the previous declared cohort.',
              'scope':'Accepted V3-minus-V2 reflection pixels only; no training/selection changes or test scoring',
              'limitation':'Two training reflection cases, one validation case; approximate assistant polygons without independent adjudication'}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'preview_indices':ids,'reflection_summary':summary,'optimizer_updates_locally':0},indent=2),flush=True)


if __name__ == '__main__': main()
