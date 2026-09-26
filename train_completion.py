"""Train a separate gated face-completion and occlusion-detection baseline."""
import argparse
import json
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from completion import CompletionNet, FORMAT, blend_mask, compose, region_error
from completion_data import CompletionDataset, image_paths, signature
from completion_inference import load_restorer, visible_base
from phase5_utils import ModelEMA
from training_state import save_training_state, load_training_state


def qualifies(candidate,baseline,best):
    keys = ('hole_mae','visible_mae','mask_iou')
    if any(r.get(k) is None or not math.isfinite(r[k]) for r in (candidate,baseline,best) for k in keys):
        return False
    return (candidate['hole_mae'] < best['hole_mae'] and
            candidate['visible_mae'] <= baseline['visible_mae']+1e-6 and
            candidate['mask_iou'] >= baseline['mask_iou'])


def atomic_save(value,path):
    temporary = str(path)+'.tmp'
    torch.save(value,temporary)
    os.replace(temporary,path)


def export(model,args,epoch,path):
    atomic_save({'format':FORMAT,'width':args.width,'size':args.size,
                 'epoch':epoch,'dry_run':args.dry_run,'model':model.state_dict(),
                 'restorer_sha256':args.restorer_sha256},path)


def mean_region(pred,target,mask):
    values,valid = region_error(pred,target,mask)
    return values[valid].mean() if valid.any() else pred.sum()*0


@torch.no_grad()
def evaluate(model,loader,restorer,device,prefix,baseline=False):
    model.eval()
    rows = []
    previews = []
    for batch_index,batch in enumerate(tqdm(loader,desc='Completion validation',dynamic_ncols=True)):
        x,y,m = (batch[k].to(device) for k in ('input','target','mask'))
        base = visible_base(x,batch['degraded'],restorer)
        predicted = (model.detect(x).sigmoid()>=.5).float()
        if baseline:
            predicted = torch.zeros_like(m)
            oracle = output = base
        else:
            oracle = compose(base,model.complete(base,m),blend_mask(m))
            output = compose(base,model.complete(base,predicted),blend_mask(predicted))
        if not torch.isfinite(output).all():
            raise FloatingPointError('Non-finite validation output')
        hole,valid = region_error(output,y,m)
        visible,visible_valid = region_error(output,y,1-m)
        ideal,_ = region_error(oracle,y,m)
        union = ((predicted+m)>0).sum((1,2,3))
        intersection = (predicted*m).sum((1,2,3))
        iou = torch.where(union>0,intersection/union.clamp_min(1),torch.ones_like(union,dtype=torch.float))
        for i in range(len(x)):
            rows.append({'path':batch['path'][i],'kind':batch['kind'][i],
                         'source':batch['source'][i],'degraded':bool(batch['degraded'][i]),
                         'seed':int(batch['seed'][i]),'coverage':m[i].mean().item(),
                         'hole_mae':hole[i].item() if valid[i] else None,
                         'oracle_hole_mae':ideal[i].item() if valid[i] else None,
                         'visible_mae':visible[i].item() if visible_valid[i] else None,
                         'mask_iou':iou[i].item(),
                         'unsupported':bool(predicted[i].mean()>=.85)})
        for i in range(min(len(x),10-len(previews))):
            previews.append(torch.stack((x[i],m[i].expand(3,-1,-1),oracle[i],output[i],y[i])).cpu())
    def aggregate(items):
        result = {'samples':len(items),'unsupported':sum(r['unsupported'] for r in items)}
        for key in ('hole_mae','oracle_hole_mae','visible_mae','mask_iou'):
            values = [r[key] for r in items if r[key] is not None]
            result[key] = sum(values)/len(values) if values else None
        return result
    result = aggregate(rows)
    result['groups'] = {f'{kind}/{condition}':aggregate([r for r in rows if r['kind']==kind and r['degraded']==condition])
                        for kind,condition in sorted({(r['kind'],r['degraded']) for r in rows})}
    result['sources'] = {s:aggregate([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})}
    if previews:
        save_image(torch.cat(previews),str(prefix)+'.png',nrow=5)
    Path(str(prefix)+'.json').write_text(json.dumps({'summary':result,'rows':rows},indent=2),encoding='utf-8')
    return result


def train(args):
    if args.epochs<1 or args.batch_size<1 or args.num_workers<0 or args.lr<=0 or args.lambda_perceptual<0:
        raise ValueError('Invalid training configuration')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists() and not args.resume_state:
        raise ValueError('Output exists: resume it or choose a new output directory')
    paths = image_paths(args.data_dir)
    if args.split_file:
        split = json.loads(Path(args.split_file).read_text(encoding='utf-8'))
        a,b = split['train'],split['validation']
        resolved_a,resolved_b = ({str(Path(p).resolve()) for p in items} for items in (a,b))
        if (resolved_a & resolved_b or len(resolved_a)!=len(a) or len(resolved_b)!=len(b)
                or resolved_a|resolved_b != {str(Path(p).resolve()) for p in paths}):
            raise ValueError('Split must contain each dataset image exactly once without overlap')
    else:
        shuffled = list(paths)
        random.Random(args.seed).shuffle(shuffled)
        n = max(1,int(len(shuffled)*.05))
        a,b = shuffled[n:],shuffled[:n]
    if args.dry_run:
        a,b = a[:args.batch_size],b[:1]
    if not a or not b:
        raise ValueError('Need both training and validation images')
    split = {'train':a,'validation':b}
    print('Verifying dataset contents...',flush=True)
    args.data_sha256 = signature(a+b)
    import hashlib
    args.restorer_sha256 = hashlib.sha256(Path(args.restorer).read_bytes()).hexdigest()
    train_set = CompletionDataset(a,args.size,args.seed)
    val_set = CompletionDataset(b,args.size,args.seed,validation=True)
    options = dict(batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=device=='cuda')
    train_loader = DataLoader(train_set,shuffle=True,**options)
    val_loader = DataLoader(val_set,shuffle=False,**options)
    model = CompletionNet(args.width).to(device)
    restorer = load_restorer(args.restorer,device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    ema = ModelEMA(model,.99)
    perceptual = None
    if args.lambda_perceptual:
        from models.losses import MultiLayerVGGPerceptualLoss
        perceptual = MultiLayerVGGPerceptualLoss(device).eval()
    state = None
    if args.resume_state:
        if Path(args.resume_state).resolve() != (output/'last_state.pth').resolve():
            raise ValueError('Resume the state belonging to this output directory')
        saved = torch.load(args.resume_state,map_location='cpu',weights_only=True)
        if saved.get('extra',{}).get('format') != FORMAT:
            raise ValueError('Not a completion training state')
        for key,value in vars(args).items():
            if key not in {'resume_state','num_workers','split_file','restorer','data_dir'} and saved['config'].get(key)!=value:
                raise ValueError(f'Resume configuration mismatch: {key}')
        if json.loads((output/'split.json').read_text()) != split:
            raise ValueError('Resume split changed')
        state = load_training_state(args.resume_state,model,optimizer,scheduler,device)
        ema.model.load_state_dict(state['extra']['ema'],strict=True)
    (output/'split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    (output/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    print(f'{device}: {len(train_set)} training examples/epoch, {len(val_set)} fixed validation cases',flush=True)
    if state:
        baseline,best = state['extra']['baseline'],state['extra']['best']
        start = state['epoch']+1
    else:
        baseline = evaluate(model,val_loader,restorer,device,output/'baseline',baseline=True)
        best,start = baseline,1
        (output/'best_selection.json').write_text(json.dumps({'epoch':0,'reason':'Restoration-only baseline; no completion weights selected'}))
    for epoch in range(start,args.epochs+1):
        train_set.epoch = epoch
        torch.manual_seed(args.seed+epoch)
        model.train()
        total,samples = 0.,0
        for batch in tqdm(train_loader,desc=f'Completion epoch {epoch}/{args.epochs}',dynamic_ncols=True):
            x,y,m = (batch[k].to(device) for k in ('input','target','mask'))
            base = visible_base(x,batch['degraded'],restorer)
            generated = model.complete(base,m)
            logits = model.detect(x)
            oracle = compose(base,generated,blend_mask(m))
            hole = mean_region(generated,y,m)
            context = mean_region(generated,y,1-m)
            segmentation = F.binary_cross_entropy_with_logits(logits,m)
            probs = logits.sigmoid()
            dice = 1-((2*(probs*m).sum((1,2,3))+1)/(probs.sum((1,2,3))+m.sum((1,2,3))+1)).mean()
            loss = hole + .1*context + segmentation + dice
            if perceptual is not None:
                loss = loss + args.lambda_perceptual*perceptual(oracle,y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            ema.update(model)
            total += loss.item()*len(x)
            samples += len(x)
        scheduler.step()
        metrics = evaluate(ema.model,val_loader,restorer,device,output/f'epoch_{epoch}')
        accepted = qualifies(metrics,baseline,best) and metrics['unsupported']==0
        # A group regression must not hide behind an overall average.
        for group,reference in baseline['groups'].items():
            current = metrics['groups'][group]
            accepted = accepted and current['visible_mae'] <= reference['visible_mae']+1e-6
        for source,reference in baseline['sources'].items():
            current = metrics['sources'][source]
            accepted = accepted and current['visible_mae'] <= reference['visible_mae']+1e-6
        export(ema.model,args,epoch,output/f'epoch_{epoch}.pth')
        if accepted:
            best = metrics
            export(ema.model,args,epoch,output/'best.pth')
            (output/'best_selection.json').write_text(json.dumps({'epoch':epoch,'metrics':metrics,'reason':'Region-error and segmentation gates passed; visual/identity review still required'},indent=2))
        save_training_state(output/'last_state.pth',model,optimizer,scheduler,epoch,0.,vars(args),
                            extra={'format':FORMAT,'ema':ema.model.state_dict(),'baseline':baseline,'best':best})
        with (output/'metrics.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps({'epoch':epoch,'loss':total/samples,'accepted':accepted,**metrics})+'\n')
        print(f'Epoch {epoch} saved; selected={accepted}; predicted-mask hole MAE={metrics["hole_mae"]:.4f}',flush=True)
        if args.dry_run:
            print('SMOKE ONLY: one batch; not trained quality evidence.',flush=True)
            break


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir',default='dataset/thumbnails128x128,dataset/asian_faces')
    p.add_argument('--split_file',default='')
    p.add_argument('--restorer',default='checkpoints/dgp_zamboanga_final.pth')
    p.add_argument('--output_dir',default='outputs/completion_pilot')
    p.add_argument('--resume_state',default='')
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--num_workers',type=int,default=2)
    p.add_argument('--size',type=int,default=256)
    p.add_argument('--width',type=int,default=24)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--lambda_perceptual',type=float,default=.05)
    p.add_argument('--dry_run',action='store_true')
    return p


if __name__ == '__main__':
    train(parser().parse_args())
