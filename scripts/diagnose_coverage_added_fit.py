"""Inference-only diagnostic on the five added training examples."""
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks
from scripts.train_coverage_vm import sha, PARENT_SHA
from scripts.evaluate_coverage_results import recount


def main():
    torch.set_num_threads(4)
    manifest=Path('dataset/detector_training_extension_v2/manifest.json')
    assert sha(manifest)=='1f9787ba7bf184f641b298863ae90250b90706e74b6163517a22efe293fcbb2c'
    original=json.loads(Path('dataset/detector_glare_review_v3/manifest.json').read_text())
    original_images={r['image_sha256'] for r in original['records']}
    all_rows=load_manifest(manifest)
    train_rows=[r for r in all_rows if r['split']=='train']
    added=[r for r in train_rows if r['image_sha256'] not in original_images]
    assert len(added)==5
    data=ReviewedMasks(added,256)
    protocol=json.loads(Path('outputs/coverage_protocol_v1/protocol.json').read_text())
    exposures=Counter(i for epoch in protocol['schedules']['extended']['batches'] for batch in epoch for i in batch if i<len(train_rows))
    root=Path('outputs/downloaded_coverage/outputs/coverage_training_vm')
    audit=json.loads(Path('outputs/coverage_validation/checkpoint_audit.json').read_text())
    checkpoints={'parent':Path('outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'),
                 'control':root/'control/epoch_10.pth','extended':root/'extended/epoch_10.pth'}
    expected={'parent':PARENT_SHA,**{k:v['final_checkpoint_sha256'] for k,v in audit['arms'].items()}}
    out=Path('outputs/coverage_added_fit')
    out.mkdir(exist_ok=False)
    results={'scope':'Five training-only examples; no generalization or selection evidence',
             'optimizer_updates':0,'records':[],'checkpoints':expected}
    predictions={}
    for arm,path in checkpoints.items():
        assert sha(path)==expected[arm]
        model,_=load_completion(path,'cpu')
        model.requires_grad_(False).eval()
        (out/arm).mkdir()
        predictions[arm]=[]
        with torch.inference_mode():
            for i,(x,m) in enumerate(data):
                probability=model.detect(x[None]).sigmoid()[0,0].numpy()
                p=probability>=.5
                predictions[arm].append((p,probability))
                Image.fromarray(p.astype('uint8')*255).save(out/arm/f'{i}.png')
    sheet=Image.new('RGB',(640,5*156),'white')
    for i,r in enumerate(added):
        x,m=data[i];target=m[0].numpy().astype(bool)
        record={'image':r['image'],'stratum':r['occlusion_stratum'],
                'extended_training_exposures':exposures[train_rows.index(r)],'arms':{}}
        tiles=[Image.fromarray((x.permute(1,2,0).numpy()*255).round().astype('uint8')),
               Image.fromarray(target.astype('uint8')*255).convert('RGB')]
        for arm in checkpoints:
            p,prob=predictions[arm][i]
            record['arms'][arm]={'metrics':recount([p],[target]),
                'target_mean_probability':float(prob[target].mean()) if target.any() else None,
                'visible_mean_probability':float(prob[~target].mean()),'predicted_pixels':int(p.sum())}
            tiles.append(Image.fromarray(p.astype('uint8')*255).convert('RGB'))
        results['records'].append(record)
        ImageDraw.Draw(sheet).text((3,i*156+2),f"{r['occlusion_stratum']} | exposures {record['extended_training_exposures']}",fill='black')
        ImageDraw.Draw(sheet).text((3,i*156+15),'input | target | parent | control | extended',fill='black')
        for col,tile in enumerate(tiles):
            sheet.paste(tile.resize((128,128)),(col*128,i*156+28))
    sheet.save(out/'preview.png')
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print([{ 'stratum':r['stratum'],'exposures':r['extended_training_exposures'],
             'iou':{a:v['metrics']['iou'] for a,v in r['arms'].items()},
             'pixels':{a:v['predicted_pixels'] for a,v in r['arms'].items()}} for r in results['records']])


if __name__=='__main__':main()
