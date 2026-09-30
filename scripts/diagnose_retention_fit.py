"""Parent/final real training fit and checkpoint-mask reproduction. No fitting."""
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks, segmentation_loss
from scripts.evaluate_coverage_results import binary, recount
from scripts.train_coverage_vm import sha, PARENT_SHA


def main():
    torch.set_num_threads(4)
    root = Path('outputs/downloaded_retention/outputs/retention_training_vm')
    manifest = Path('dataset/detector_training_extension_v2/manifest.json')
    assert sha(manifest) == '1f9787ba7bf184f641b298863ae90250b90706e74b6163517a22efe293fcbb2c'
    rows = load_manifest(manifest)
    train = [r for r in rows if r['split']=='train']
    validation = [r for r in rows if r['split']=='validation']
    steps = [json.loads(line) for line in (root/'steps.jsonl').read_text().splitlines()]
    exposures = Counter(i for s in steps if s['accepted'] for i in s['indices'] if i < 73)
    paths = {'parent': Path('outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'), 'retained': root/'final.pth'}
    expected = {'parent': PARENT_SHA, 'retained': '6ecbd889ebbee97d2dad54defa04acdab131ba558b80e4c3669c1c8c50a3e727'}
    out = Path('outputs/retention_fit'); out.mkdir(exist_ok=False)
    result = {'optimizer_updates': 0, 'checkpoint_hashes': expected, 'arms': {}}
    groups = {'all': list(range(73)), 'covered': [i for i,r in enumerate(train) if r['kind']=='covered'],
              'clear': [i for i,r in enumerate(train) if r['kind']!='covered'],
              'accepted_batch_seen': [i for i in range(73) if exposures[i]],
              'no_accepted_batch': [i for i in range(73) if not exposures[i]]}
    for arm,path in paths.items():
        assert sha(path)==expected[arm]
        model,_ = load_completion(path,'cpu'); model.requires_grad_(False).eval()
        predictions=[];targets=[];losses=[];records=[]
        data=ReviewedMasks(train,256)
        with torch.inference_mode():
            for i,(x,m) in enumerate(data):
                logits=model.detect(x[None])
                p=(logits.sigmoid()[0,0]>=.5).numpy();t=m[0].numpy().astype(bool)
                loss=float(segmentation_loss(logits,m[None],.25,.1))
                predictions.append(p);targets.append(t);losses.append(loss)
                records.append({'image':train[i]['image'],'accepted_batch_exposures':exposures[i],
                                'loss':loss,**recount([p],[t])})
                if i%24==0:print(arm,i,len(data),flush=True)
        result['arms'][arm]={'records':records,'groups':{k:{'cases':len(ids),'loss_mean':sum(losses[i] for i in ids)/len(ids),
                   **recount([predictions[i] for i in ids],[targets[i] for i in ids])} for k,ids in groups.items() if ids}}
        if arm=='retained':
            matches=[]
            with torch.inference_mode():
                for i,(x,m) in enumerate(ReviewedMasks(validation,256)):
                    p=(model.detect(x[None]).sigmoid()[0,0]>=.5).numpy()
                    saved=binary(root/'final_masks/real'/f'{i:04}.png')
                    matches.append({'index':i,'different_pixels':int(np.count_nonzero(p!=saved))})
            result['validation_reproduction']={'cases':matches,'all_exact':all(r['different_pixels']==0 for r in matches)}
        (out/'partial.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({a:v['groups'] for a,v in result['arms'].items()},indent=2),flush=True)
    print(result['validation_reproduction'],flush=True)


if __name__=='__main__':main()
