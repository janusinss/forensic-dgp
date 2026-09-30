"""Independent saved-mask recount; no fitting or automatic promotion."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from detector_replay import retention_passes
from scripts.train_coverage_vm import real_gate, sha, PROTOCOL_SHA, PARENT_SHA


def binary(path):
    a = np.array(Image.open(path))
    if a.ndim == 3 and np.all(a == a[:,:,:1]):
        a = a[:,:,0]
    if a.ndim != 2 or not np.isin(a,[0,255]).all():
        raise ValueError(f'Non-binary mask: {path}')
    return a.astype(bool)


def recount(predictions, targets):
    if len(predictions) != len(targets) or not targets:
        raise ValueError('Missing/extra masks or empty evaluation')
    tp=fp=fn=visible=empty=positive=negative=false_positive=0
    for p,t in zip(predictions,targets):
        if p.shape != t.shape:
            raise ValueError('Mask shape mismatch')
        tp += int(np.count_nonzero(p&t)); fp += int(np.count_nonzero(p&~t))
        fn += int(np.count_nonzero(~p&t)); visible += int(np.count_nonzero(~t))
        positive += int(t.any()); negative += int(not t.any())
        empty += int(t.any() and not p.any())
        false_positive += int(not t.any() and p.any())
    return dict(iou=tp/max(1,tp+fp+fn),missed_fraction=fn/max(1,tp+fn),
                visible_false_positive=fp/max(1,visible),empty_mask_cases=empty,
                covered_cases=positive,negative_false_positive_cases=false_positive,negative_cases=negative)


def main(args):
    root = Path(args.results)
    complete = json.loads((root/'complete.json').read_text())
    if complete.get('complete') is not True or complete['updates_per_arm'] != 210:
        raise ValueError('Both arms must be complete')
    run = json.loads((root/'run.json').read_text())
    if run['protocol_sha256'] != PROTOCOL_SHA or run['parent_sha256'] != PARENT_SHA or run['preflight']:
        raise ValueError('Unexpected run provenance')
    for name,key in [('initial.pth','initial_sha256'),('replay_pixels.pth','replay_sha256')]:
        if sha(root/name) != run[key]:
            raise ValueError('Returned tensor archive changed')
    inventory_path = Path('outputs/coverage_protocol_v1/vm_inventory.json')
    if sha(inventory_path) != run['inventory_sha256']:
        raise ValueError('VM inventory differs')
    for path,digest in json.loads(inventory_path.read_text()).items():
        if sha(path) != digest:
            raise ValueError(f'Local reference changed: {path}')
    real_root = Path('dataset/detector_glare_review_v3')
    real = [r for r in json.loads((real_root/'manifest.json').read_text())['records'] if r['split']=='validation']
    bench = Path('outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm')
    cases = json.loads((bench/'manifest.json').read_text())['cases']
    targets = {'real':[binary(real_root/r['mask']) for r in real],
               'synthetic':[binary(bench/'mask'/r['file']) for r in cases]}
    baseline = json.loads((root/'baseline.json').read_text())
    result = {'arms':{},'saved_masks_recounted':0,'promoted':False,
              'limitation':'Recount validates saved masks and logged selection only. Baseline inference and checkpoint-to-mask reproduction still require verification.'}
    final_masks = {}
    for arm in ('control','extended'):
        history = [json.loads(line) for line in (root/arm/'metrics.jsonl').read_text().splitlines() if line.strip()]
        if [r['epoch'] for r in history] != list(range(1,11)):
            raise ValueError('Missing/duplicated epochs')
        best = baseline['real']['iou']
        checked = []
        for row in history:
            if row['updates'] != row['epoch']*21:
                raise ValueError('Update count mismatch')
            scores = {}
            for domain,truth in targets.items():
                folder = root/arm/f"epoch_{row['epoch']}_masks"/domain
                expected = {f'{i:04}.png' for i in range(len(truth))}
                if {p.name for p in folder.glob('*.png')} != expected:
                    raise ValueError('Missing/extra saved mask files')
                predictions = [binary(folder/f'{i:04}.png') for i in range(len(truth))]
                score = recount(predictions,truth)
                for k,v in score.items():
                    if not math.isclose(v,row[domain][k],abs_tol=1e-12,rel_tol=0):
                        raise ValueError(f'Recount mismatch: {arm}/{row["epoch"]}/{domain}/{k}')
                scores[domain] = score
                result['saved_masks_recounted'] += len(truth)
                if row['epoch']==10 and domain=='real':
                    final_masks[arm] = predictions
            real_ok = real_gate(scores['real'],baseline['real'],best)
            synthetic_ok = retention_passes(scores['synthetic'],baseline['synthetic'])
            selected = real_ok and synthetic_ok
            if (row['real_gate'],row['synthetic_gate'],row['selected']) != (real_ok,synthetic_ok,selected):
                raise ValueError('Selection decision differs')
            if selected:
                best = scores['real']['iou']
            checked.append({'epoch':row['epoch'],'selected':selected,**scores})
        if (root/arm/'best_detector.pth').exists() != any(r['selected'] for r in checked):
            raise ValueError('Best checkpoint presence differs from selection')
        result['arms'][arm] = checked
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    sheet = Image.new('RGB',(512,10*148),'white')
    for i,r in enumerate(real[:10]):
        tiles = [Image.open(real_root/r['image']).convert('RGB'),
                 Image.fromarray(targets['real'][i].astype('uint8')*255).convert('RGB')]
        tiles += [Image.fromarray(final_masks[a][i].astype('uint8')*255).convert('RGB') for a in ('control','extended')]
        ImageDraw.Draw(sheet).text((2,i*148+2),f'{i}: input | target | control10 | extended10',fill='black')
        for j,tile in enumerate(tiles):
            sheet.paste(tile.resize((128,128)),(j*128,i*148+20))
    sheet.save(out/'preview.png')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print({'masks_recounted':result['saved_masks_recounted'],'promoted':False})


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True)
    parser.add_argument('--output',default='outputs/coverage_validation')
    main(parser.parse_args())
