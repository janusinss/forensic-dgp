"""Training-only all-real fit stratification; no optimization or selection."""
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks
from scripts.evaluate_coverage_results import recount
from scripts.train_coverage_vm import sha, PARENT_SHA


def main():
    torch.set_num_threads(4)
    manifest=Path('dataset/detector_training_extension_v2/manifest.json')
    assert sha(manifest)=='1f9787ba7bf184f641b298863ae90250b90706e74b6163517a22efe293fcbb2c'
    rows=[r for r in load_manifest(manifest) if r['split']=='train']
    data=ReviewedMasks(rows,256)
    protocol=json.loads(Path('outputs/coverage_protocol_v1/protocol.json').read_text())
    exposure={a:Counter(i for epoch in protocol['schedules'][a]['batches'] for batch in epoch for i in batch if i<(68 if a=='control' else 73)) for a in ('control','extended')}
    audit=json.loads(Path('outputs/coverage_validation/checkpoint_audit.json').read_text())
    root=Path('outputs/downloaded_coverage/outputs/coverage_training_vm')
    paths={'parent':Path('outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'),
           'control':root/'control/epoch_10.pth','extended':root/'extended/epoch_10.pth'}
    expected={'parent':PARENT_SHA,**{a:v['final_checkpoint_sha256'] for a,v in audit['arms'].items()}}
    targets=[data[i][1][0].numpy().astype(bool) for i in range(len(data))]
    groups={}
    records=[]
    for i,r in enumerate(rows):
        if r['kind']=='uncovered':category='clear_controls'
        elif r.get('occlusion_stratum'):category=r['occlusion_stratum']
        elif r.get('glare_stratum') not in (None,'no_added_glare'):category='original_glare'
        else:category='original_masks'
        groups.setdefault(category,[]).append(i)
        records.append({'image':r['image'],'category':category,'exposure':{a:exposure[a][i] for a in exposure},'arms':{}})
    out=Path('outputs/coverage_full_fit');out.mkdir(exist_ok=False)
    result={'scope':'Training fit only; control did not train on the five additions','optimizer_updates':0,'arms':{},'records':records}
    for arm,path in paths.items():
        assert sha(path)==expected[arm]
        model,_=load_completion(path,'cpu');model.requires_grad_(False).eval()
        preds=[];(out/arm).mkdir()
        with torch.inference_mode():
            for i,(x,m) in enumerate(data):
                p=(model.detect(x[None]).sigmoid()[0,0]>=.5).numpy();preds.append(p)
                Image.fromarray(p.astype('uint8')*255).save(out/arm/f'{i:03}.png')
                metrics=recount([p],[targets[i]])
                # Undefined IoU on empty target/prediction is not a quality failure.
                if not targets[i].any() and not p.any():metrics['iou']=None
                records[i]['arms'][arm]=metrics
        result['arms'][arm]={}
        for category,indices in groups.items():
            metrics=recount([preds[i] for i in indices],[targets[i] for i in indices])
            result['arms'][arm][category]={'count':len(indices),'metrics':metrics,
                'per_image_iou_median':float(np.median([records[i]['arms'][arm]['iou'] for i in indices])) if all(targets[i].any() for i in indices) else None,
                'training_exposure_range':([min(exposure[arm][i] for i in indices),max(exposure[arm][i] for i in indices)] if arm in exposure else None)}
        print(arm,json.dumps(result['arms'][arm]),flush=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
