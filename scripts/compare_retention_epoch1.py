"""Inference-only ordinary epoch1 comparator for the constrained pilot."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks, segmentation_loss
from scripts.evaluate_coverage_results import recount
from scripts.train_coverage_vm import sha, PARENT_SHA


def main():
    torch.set_num_threads(4)
    archive = Path('outputs/projection-results.tar.gz')
    assert sha(archive)=='c815e4aa5d825fa41ab9a21e07ea61adde9109275f4fa3ab5c95cc3b4cf97ceb'
    relative = 'outputs/projection_training_vm/ordinary/epoch_1.pth'
    path = Path('outputs/downloaded_projection')/relative
    with tarfile.open(archive) as tar, tar.extractfile(relative) as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==sha(path)
    root = path.parent.parent
    run = json.loads((root/'run.json').read_text())
    retained_run = json.loads(Path('outputs/downloaded_retention/outputs/retention_training_vm/run.json').read_text())
    for key in ('protocol_sha256','parent_sha256','inventory_sha256','replay_sha256','lr','weight_decay','clip','background_weight','consistency_weight'):
        assert run[key]==retained_run[key], key
    assert run['parent_sha256']==PARENT_SHA
    manifest=Path('dataset/detector_training_extension_v2/manifest.json')
    assert sha(manifest)=='1f9787ba7bf184f641b298863ae90250b90706e74b6163517a22efe293fcbb2c'
    rows=[r for r in load_manifest(manifest) if r['split']=='train']
    previous_path=Path('outputs/retention_fit/results.json')
    previous=json.loads(previous_path.read_text())
    assert [r['image'] for r in previous['arms']['parent']['records']]==[r['image'] for r in rows]
    model,_=load_completion(path,'cpu');model.requires_grad_(False).eval()
    predictions=[];targets=[];losses=[]
    with torch.inference_mode():
        for i,(x,m) in enumerate(ReviewedMasks(rows,256)):
            logits=model.detect(x[None])
            predictions.append((logits.sigmoid()[0,0]>=.5).numpy())
            targets.append(m[0].numpy().astype(bool))
            losses.append(float(segmentation_loss(logits,m[None],.25,.1)))
            if i%24==0:print('ordinary epoch1',i,73,flush=True)
    groups={'all':list(range(73)), 'covered':[i for i,r in enumerate(rows) if r['kind']=='covered'],
            'clear':[i for i,r in enumerate(rows) if r['kind']!='covered']}
    result={'optimizer_updates':0,'checkpoint_sha256':sha(path),'reused_fit_sha256':sha(previous_path),
            'ordinary':{k:{'cases':len(ids),'loss_mean':sum(losses[i] for i in ids)/len(ids),
                **recount([predictions[i] for i in ids],[targets[i] for i in ids])} for k,ids in groups.items()},
            'parent':previous['arms']['parent']['groups']['all'],
            'retained':previous['arms']['retained']['groups']['all'],
            'limitation':'21 ordinary updates vs18 retained updates with reduced rates; not matched accepted dose or compute'}
    out=Path('outputs/retention_epoch1_comparison');out.mkdir(exist_ok=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
