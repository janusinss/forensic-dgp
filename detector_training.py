"""Detector-only pilot on reviewed real-occlusion labels. Test split stays sealed."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from completion_inference import load_completion


def validate_records(rows):
    if not rows:raise ValueError('No reviewed data')
    owners={'group':{},'image_sha256':{}}
    for r in rows:
        if r.get('reviewed') is not True:raise ValueError('Every included label must be reviewed')
        if r.get('split') not in ('train','validation','test') or r.get('kind') not in ('covered','uncovered'):
            raise ValueError('Invalid split or kind')
        for key in owners:
            value=r.get(key)
            if not isinstance(value,str) or not value.strip():raise ValueError('Missing grouping or image hash')
            if value in owners[key] and owners[key][value]!=r['split']:
                raise ValueError('Cross-split group or duplicate leakage')
            owners[key][value]=r['split']
    for split in ('train','validation','test'):
        if {r['kind'] for r in rows if r['split']==split}!={'covered','uncovered'}:
            raise ValueError('Each split needs covered and uncovered controls')


def detector_optimizer(model,lr):
    model.generator.requires_grad_(False).eval()
    model.segmenter.requires_grad_(True).train()
    return torch.optim.AdamW(model.segmenter.parameters(),lr=lr,weight_decay=1e-4)


class BalancedMaskBatches:
    """Equal covered/uncovered counts; cycle shuffled minority rather than dropping majority."""
    def __init__(self,rows,batch_size,seed):
        self.groups=[[i for i,r in enumerate(rows) if r['kind']==kind] for kind in ('covered','uncovered')]
        if batch_size<2 or batch_size%2 or not all(self.groups):
            raise ValueError('Balanced batches require both kinds and an even batch size >=2')
        self.half=batch_size//2;self.seed=seed;self.epoch=0
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return math.ceil(max(map(len,self.groups))/self.half)
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);needed=len(self)*self.half;streams=[]
        for group in self.groups:
            stream=[]
            while len(stream)<needed:stream.extend(rng.permutation(group).tolist())
            streams.append(stream[:needed])
        for start in range(0,needed,self.half):
            batch=streams[0][start:start+self.half]+streams[1][start:start+self.half]
            yield rng.permutation(batch).tolist()


def visible_penalty(logits,mask,hard_fraction=.1):
    """Per-image negative BCE over highest-error visible pixels only."""
    if not 0<hard_fraction<=1:raise ValueError('hard_fraction must be in (0,1]')
    values=[]
    for logit,target in zip(logits,mask):
        errors=F.softplus(logit)[target==0]
        if errors.numel():values.append(errors.topk(max(1,math.ceil(errors.numel()*hard_fraction))).values.mean())
    return torch.stack(values).mean() if values else logits.sum()*0


def segmentation_loss(logits,mask,background_weight=0.,hard_fraction=.1):
    if not math.isfinite(background_weight) or background_weight<0:raise ValueError('Invalid background weight')
    p=logits.sigmoid()
    dice=1-((2*(p*mask).sum((1,2,3))+1)/(p.sum((1,2,3))+mask.sum((1,2,3))+1)).mean()
    loss=F.binary_cross_entropy_with_logits(logits,mask)+dice
    return loss+background_weight*visible_penalty(logits,mask,hard_fraction) if background_weight else loss


def load_manifest(path):
    path=Path(path).resolve();data=json.loads(path.read_text());rows=data['records']
    validate_records(rows)
    seen=set()
    for r in rows:
        for key in ('image','mask'):
            p=(path.parent/r[key]).resolve()
            if not p.is_relative_to(path.parent):raise ValueError('Data path escapes manifest directory')
            if hashlib.sha256(p.read_bytes()).hexdigest()!=r[key+'_sha256']:
                raise ValueError('Image or reviewed mask hash changed')
            r[key+'_path']=str(p)
        if r['image_sha256'] in seen:raise ValueError('Duplicate image; keep one reviewed record')
        seen.add(r['image_sha256'])
        with Image.open(r['image_path']) as im:shape=im.size
        with Image.open(r['mask_path']) as im:
            a=np.array(im)
            if im.size!=shape or a.ndim!=2 or not np.isin(a,[0,255]).all():
                raise ValueError('Use a matching binary grayscale PNG mask')
        if min(shape)<32 or shape[0]!=shape[1]:raise ValueError('Review square face crops >=32px')
        if bool(a.any())!=(r['kind']=='covered'):raise ValueError('Mask disagrees with covered/uncovered label')
    return rows


class ReviewedMasks(torch.utils.data.Dataset):
    def __init__(self,rows,size):self.rows,self.size=rows,size
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['image_path']) as im:a=np.array(im.convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR))
        with Image.open(r['mask_path']) as im:m=np.array(im.resize((self.size,self.size),Image.Resampling.NEAREST))
        return torch.from_numpy(a.copy()).permute(2,0,1).float()/255,torch.from_numpy(m.copy())[None].float()/255


@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval();tp=fp=fn=visible=0;empty=positive=negative=false_positive=0
    for x,m in loader:
        p=model.detect(x.to(device)).sigmoid()>=.5;t=m.to(device).bool()
        tp+=int((p&t).sum());fp+=int((p&~t).sum());fn+=int((~p&t).sum());visible+=int((~t).sum())
        has=t.flatten(1).any(1);pred=p.flatten(1).any(1)
        positive+=int(has.sum());empty+=int((has&~pred).sum())
        negative+=int((~has).sum());false_positive+=int((~has&pred).sum())
    return dict(iou=tp/max(1,tp+fp+fn),missed_fraction=fn/max(1,tp+fn),visible_false_positive=fp/max(1,visible),
                empty_mask_cases=empty,covered_cases=positive,negative_false_positive_cases=false_positive,negative_cases=negative)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output_dir',required=True);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch_size',type=int,default=4);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--validate_only',action='store_true')
    p.add_argument('--balanced_batches',action='store_true')
    p.add_argument('--background_weight',type=float,default=0.)
    p.add_argument('--hard_fraction',type=float,default=.1)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1 or not 0<args.lr<1: p.error('Invalid pilot parameters')
    if not math.isfinite(args.background_weight) or args.background_weight<0 or not 0<args.hard_fraction<=1:
        p.error('Invalid visible-background penalty parameters')
    if args.balanced_batches and (args.batch_size<2 or args.batch_size%2):p.error('Balanced batch size must be even and >=2')
    rows=load_manifest(args.manifest)
    if args.validate_only:
        print(json.dumps({s:sum(r['split']==s for r in rows) for s in ('train','validation','test')}));return
    out=Path(args.output_dir)
    if out.exists():raise ValueError('Choose a new output directory')
    torch.manual_seed(args.seed);torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,state=load_completion(args.checkpoint,device)
    original={k:v.detach().cpu().clone() for k,v in model.generator.state_dict().items()}
    datasets={s:ReviewedMasks([r for r in rows if r['split']==s],state['size']) for s in ('train','validation')}
    sampler=BalancedMaskBatches(datasets['train'].rows,args.batch_size,args.seed) if args.balanced_batches else None
    loaders={'validation':torch.utils.data.DataLoader(datasets['validation'],batch_size=args.batch_size)}
    loaders['train']=(torch.utils.data.DataLoader(datasets['train'],batch_sampler=sampler) if sampler is not None
                      else torch.utils.data.DataLoader(datasets['train'],batch_size=args.batch_size,shuffle=True))
    baseline=evaluate(model,loaders['validation'],device);best=baseline['iou']
    optimizer=detector_optimizer(model,args.lr);out.mkdir(parents=True)
    metadata={**vars(args),'device':device,'baseline':baseline,'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              'initial_checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),'test_split':'not evaluated',
              'training_policy':'balanced-hard-visible-v1' if args.balanced_batches and args.background_weight else 'configured',
              'steps_per_epoch':len(loaders['train'])}
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    for epoch in range(1,args.epochs+1):
        if sampler is not None:sampler.set_epoch(epoch-1)
        model.segmenter.train();total=0
        for x,m in loaders['train']:
            loss=segmentation_loss(model.detect(x.to(device)),m.to(device),args.background_weight,args.hard_fraction)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite detector loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.segmenter.parameters(),1.,error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())
        metrics=evaluate(model,loaders['validation'],device)
        assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in model.generator.state_dict().items()),'Generator changed'
        selected=(metrics['iou']>best and metrics['visible_false_positive']<=baseline['visible_false_positive']
                  and metrics['empty_mask_cases']<=baseline['empty_mask_cases']
                  and metrics['negative_false_positive_cases']<=baseline['negative_false_positive_cases'])
        export={**state,'model':model.state_dict(),'detector_finetune_epoch':epoch,'detector_only':True,'detector_training':metadata}
        torch.save(export,out/f'detector_epoch_{epoch}.pth')
        if selected:best=metrics['iou'];torch.save(export,out/'best_detector.pth')
        row={'epoch':epoch,'loss':total/len(loaders['train']),'selected':selected,**metrics}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
