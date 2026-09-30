"""Inference-only replay training fit, grouped by fixed synthetic case metadata."""
import json
from pathlib import Path
import sys
import torch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from completion import KINDS
from completion_inference import load_completion
from scripts.train_coverage_vm import sha,PARENT_SHA
from scripts.evaluate_coverage_results import binary


def counts(p,t):
    return dict(tp=int((p&t).sum()),fp=int((p&~t).sum()),fn=int((~p&t).sum()),
                visible=int((~t).sum()),empty=int(t.any() and not p.any()),
                positive=int(t.any()),negative=int(not t.any()),false_positive=int(not t.any() and p.any()))


def aggregate(rows):
    c={k:sum(r[k] for r in rows) for k in rows[0]}
    return {'cases':len(rows),'iou':c['tp']/max(1,c['tp']+c['fp']+c['fn']),
            'missed_fraction':c['fn']/max(1,c['tp']+c['fn']),
            'visible_false_positive':c['fp']/max(1,c['visible']),
            'empty_mask_cases':c['empty'],'covered_cases':c['positive'],
            'negative_false_positive_cases':c['false_positive'],'negative_cases':c['negative']}


def main():
    torch.set_num_threads(4)
    prior=Path('outputs/downloaded_coverage/outputs/coverage_training_vm')
    root=Path('outputs/downloaded_projection/outputs/projection_training_vm')
    run=json.loads((root/'run.json').read_text())
    cache_path=prior/'replay_pixels.pth';assert sha(cache_path)==run['replay_sha256']
    cache=torch.load(cache_path,weights_only=True,map_location='cpu')
    protocol_path=Path('outputs/coverage_protocol_v1/protocol.json');assert sha(protocol_path)==run['protocol_sha256']
    protocol=json.loads(protocol_path.read_text());indices=sorted(cache)
    expected={i-73 for e in protocol['schedules']['extended']['batches'] for b in e for i in b if i>=73}
    assert set(indices)==expected and len(indices)==638
    audit=json.loads(Path('outputs/projection_validation/results.json').read_text())
    paths={'parent':Path('outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'),
           'ordinary':root/'ordinary/epoch_10.pth','projected':root/'projected/epoch_10.pth'}
    hashes={'parent':PARENT_SHA,**{a:r['final_checkpoint_sha256'] for a,r in audit['arms'].items()}}
    out=Path('outputs/replay_fit_comparison');out.mkdir(exist_ok=False)
    report={'optimizer_updates':0,'cache_sha256':sha(cache_path),'checkpoints':hashes,'arms':{}}
    for arm,path in paths.items():
        assert sha(path)==hashes[arm]
        model,_=load_completion(path,'cpu');model.requires_grad_(False).eval();records=[]
        with torch.inference_mode():
            for start in range(0,len(indices),8):
                batch=indices[start:start+8]
                x=torch.stack([cache[i][0].float()/255 for i in batch])
                preds=(model.detect(x).sigmoid()[:,0]>=.5).numpy()
                for i,p in zip(batch,preds):
                    t=cache[i][1][0].numpy().astype(bool)
                    records.append({'case':i,'kind':KINDS[i%5],'degraded':i%10>=5,
                        'source':protocol['sources'][i//10]['source'],'counts':counts(p,t)})
                if start%160==0:print(arm,start,len(indices),flush=True)
        groups={'all':records}
        groups.update({kind+'/'+str(degraded):[r for r in records if r['kind']==kind and r['degraded']==degraded]
                       for kind in KINDS for degraded in (False,True)})
        report['arms'][arm]={'records':records,'groups':{k:aggregate([r['counts'] for r in v]) for k,v in groups.items() if v}}
        (out/'partial.json').write_text(json.dumps(report,indent=2)+'\n')
    bench=Path('outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm')
    cases=json.loads((bench/'manifest.json').read_text())['cases']
    report['validation']={}
    for arm in ('ordinary','projected'):
        records=[]
        for i,r in enumerate(cases):
            p=binary(root/arm/'epoch_10_masks/synthetic'/f'{i:04}.png');t=binary(bench/'mask'/r['file'])
            records.append({'kind':r['kind'],'degraded':r['degraded'],'counts':counts(p,t)})
        groups={'all':records,**{kind+'/'+str(d):[r for r in records if r['kind']==kind and r['degraded']==d] for kind in KINDS for d in (False,True)}}
        report['validation'][arm]={k:aggregate([r['counts'] for r in v]) for k,v in groups.items() if v}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print({a:r['groups']['all'] for a,r in report['arms'].items()})


if __name__=='__main__':main()
