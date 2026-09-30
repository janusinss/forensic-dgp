"""Frozen-state, exposure-weighted loss audit; inference only."""
import json
from collections import Counter
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks, segmentation_loss
from scripts.train_coverage_vm import sha,PARENT_SHA


def terms(logits,target,teacher=None):
    p=logits.sigmoid()
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none').flatten(1).mean(1)
    dice=1-(2*(p*target).sum((1,2,3))+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1)
    penalty=torch.stack([F.softplus(x)[y==0].topk(max(1,__import__('math').ceil(int((y==0).sum())*.1))).values.mean()*.25 for x,y in zip(logits,target)])
    result={'bce':bce,'dice':dice,'weighted_visible':penalty,'supervised':bce+dice+penalty}
    if teacher is not None:
        q=teacher.sigmoid()
        result['teacher_kl']=(q*(F.logsigmoid(teacher)-F.logsigmoid(logits))+(1-q)*(F.logsigmoid(-teacher)-F.logsigmoid(-logits))).flatten(1).mean(1)
    if not torch.allclose(result['supervised'].mean(),segmentation_loss(logits,target,.25,.1),atol=1e-6,rtol=1e-5):
        raise ValueError('Loss decomposition mismatch')
    return result


def main():
    torch.set_num_threads(4)
    prior=Path('outputs/downloaded_coverage/outputs/coverage_training_vm')
    root=Path('outputs/downloaded_projection/outputs/projection_training_vm')
    run=json.loads((root/'run.json').read_text())
    protocol_path=Path('outputs/coverage_protocol_v1/protocol.json');assert sha(protocol_path)==run['protocol_sha256']
    protocol=json.loads(protocol_path.read_text())
    exposures=Counter(i for e in protocol['schedules']['extended']['batches'] for b in e for i in b)
    cache_path=prior/'replay_pixels.pth';assert sha(cache_path)==run['replay_sha256']
    cache=torch.load(cache_path,map_location='cpu',weights_only=True)
    real=ReviewedMasks([r for r in load_manifest('dataset/detector_training_extension_v2/manifest.json') if r['split']=='train'],256)
    indices=list(range(73))+[73+i for i in sorted(cache)]
    assert set(indices)==set(exposures) and sum(exposures.values())==1680
    audit=json.loads(Path('outputs/projection_validation/results.json').read_text())
    paths={'parent':Path('outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'),**{a:root/a/'epoch_10.pth' for a in ('ordinary','projected')}}
    expected={'parent':PARENT_SHA,**{a:v['final_checkpoint_sha256'] for a,v in audit['arms'].items()}}
    out=Path('outputs/replay_loss_audit');out.mkdir(exist_ok=False)
    teacher_logits={};report={'optimizer_updates':0,'checkpoint_hashes':expected,'arms':{},'scope':'Fixed checkpoints on frozen scheduled training inputs, not trajectory losses'}
    for arm,path in paths.items():
        assert sha(path)==expected[arm]
        model,_=load_completion(path,'cpu');model.requires_grad_(False).eval();records=[]
        with torch.inference_mode():
            # Separate real and replay to keep teacher handling explicit.
            for domain,ids in [('real',list(range(73))),('replay',[73+i for i in sorted(cache)])]:
                for start in range(0,len(ids),8):
                    batch=ids[start:start+8]
                    items=[real[i] if i<73 else (cache[i-73][0].float()/255,cache[i-73][1].float()) for i in batch]
                    x,m=(torch.stack([item[j] for item in items]) for j in (0,1))
                    logits=model.detect(x)
                    if domain=='replay' and arm=='parent':
                        teacher_logits.update({i:z.clone() for i,z in zip(batch,logits)})
                    reference=torch.stack([teacher_logits[i] for i in batch]) if domain=='replay' else None
                    values=terms(logits,m,reference)
                    for j,i in enumerate(batch):records.append({'index':i,'domain':domain,'exposures':exposures[i],**{k:float(v[j]) for k,v in values.items()}})
                    if start%160==0:print(arm,domain,start,len(ids),flush=True)
        summaries={}
        for domain in ('real','replay'):
            subset=[r for r in records if r['domain']==domain];weight=sum(r['exposures'] for r in subset)
            keys=['bce','dice','weighted_visible','supervised']+(['teacher_kl'] if domain=='replay' else [])
            summaries[domain]={'exposures':weight,**{k:sum(r[k]*r['exposures'] for r in subset)/weight for k in keys}}
        total=.5*summaries['real']['supervised']+.5*summaries['replay']['supervised']+summaries['replay']['teacher_kl']
        report['arms'][arm]={'records':records,'weighted':summaries,'mixed_objective':total}
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({a:{'weighted':r['weighted'],'mixed_objective':r['mixed_objective']} for a,r in report['arms'].items()},indent=2))


if __name__=='__main__':main()
