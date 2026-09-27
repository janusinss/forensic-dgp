"""Mixed detector adaptation with strict real and synthetic retention gates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from completion_data import CompletionDataset
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks, evaluate, detector_optimizer, segmentation_loss


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_sources(rows, excluded, count, seed):
    unique={}
    for path,digest in rows:
        if digest not in excluded:unique.setdefault(digest,path)
    candidates=sorted((p,h) for h,p in unique.items())
    if len(candidates)<count:raise ValueError('Insufficient disjoint training sources; check original split')
    order=np.random.default_rng(seed).permutation(len(candidates))[:count]
    return [candidates[i] for i in order]


class MixedBatches:
    """Equal real-covered, real-clear, synthetic-covered, synthetic-clear exposure."""
    def __init__(self,groups,batch_size,steps,seed):
        if len(groups)!=4 or not all(groups) or batch_size<4 or batch_size%4 or steps<1:
            raise ValueError('Four nonempty groups, batch size divisible by four, positive steps required')
        self.groups,self.quarter,self.steps,self.seed,self.epoch=groups,batch_size//4,steps,seed,0
    def __len__(self):return self.steps
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);streams=[]
        for group in self.groups:
            stream=[]
            while len(stream)<self.steps*self.quarter:stream.extend(rng.permutation(group).tolist())
            streams.append(stream)
        for step in range(self.steps):
            start=step*self.quarter
            yield rng.permutation([i for stream in streams for i in stream[start:start+self.quarter]]).tolist()


def retention_passes(metrics,baseline):
    return (metrics['iou']>=baseline['iou'] and all(metrics[k]<=baseline[k] for k in
            ('missed_fraction','visible_false_positive','empty_mask_cases','negative_false_positive_cases')))


class SyntheticMasks(torch.utils.data.Dataset):
    def __init__(self,paths,size,seed):self.data=CompletionDataset(paths,size,seed,validation=True)
    def __len__(self):return len(self.data)
    def __getitem__(self,index):
        row=self.data[index]
        return row['input'],row['mask']


class BenchmarkMasks(torch.utils.data.Dataset):
    def __init__(self,root,size):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.cases=self.manifest['cases']
        if self.manifest['size']!=size or not self.cases:raise ValueError('Benchmark size mismatch or empty cases')
    def __len__(self):return len(self.cases)
    def __getitem__(self,index):
        name=self.cases[index]['file']
        if Path(name).name!=name:raise ValueError('Invalid benchmark filename')
        with Image.open(self.root/'input'/name) as im:x=np.array(im.convert('RGB'))
        with Image.open(self.root/'mask'/name) as im:m=np.array(im)
        if m.ndim==3:
            if not np.all(m==m[:,:,:1]):raise ValueError('Benchmark mask channels differ')
            m=m[:,:,0]
        if m.shape!=x.shape[:2] or not np.isin(m,[0,255]).all():raise ValueError('Invalid benchmark mask')
        return torch.from_numpy(x.copy()).permute(2,0,1).float()/255,torch.from_numpy(m.copy())[None].float()/255


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--split',required=True)
    p.add_argument('--benchmark',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--sources',type=int,default=200);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--steps',type=int,default=21);p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=1e-5)
    args=p.parse_args()
    if min(args.sources,args.epochs,args.steps)<1 or not 0<args.lr<1:p.error('Invalid training budget')
    out=Path(args.output_dir)
    if out.exists():raise ValueError('Choose a new output directory')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    rows=load_manifest(args.manifest)
    model,state=load_completion(args.checkpoint,device)
    benchmark=BenchmarkMasks(args.benchmark,state['size'])
    split=json.loads(Path(args.split).read_text())
    if not split.get('train') or not split.get('validation'):raise ValueError('Original train/validation split required')
    # Resolve only exact listed paths. Missing files fail rather than silently creating a new split.
    def hashes(paths):
        return [(str(Path(path.replace('\\','/'))),sha(Path(path.replace('\\','/')))) for path in paths]
    print('Auditing original split and replay source hashes...',flush=True)
    excluded={h for _,h in hashes(split['validation'])}
    excluded.update(h for _,h in hashes(split.get('test',[])))
    excluded.update(c['source_sha256'] for c in benchmark.cases)
    excluded.update(r[key] for r in rows for key in ('source_sha256','image_sha256') if r.get(key))
    sources=select_sources(hashes(split['train']),excluded,args.sources,args.seed)
    real=ReviewedMasks([r for r in rows if r['split']=='train'],state['size'])
    synthetic=SyntheticMasks([path for path,_ in sources],state['size'],args.seed)
    train=torch.utils.data.ConcatDataset([real,synthetic])
    groups=[[i for i,r in enumerate(real.rows) if r['kind']==kind] for kind in ('covered','uncovered')]
    groups.extend([[len(real)+i for i in range(len(synthetic)) if (i%5!=0)==covered] for covered in (True,False)])
    sampler=MixedBatches(groups,args.batch_size,args.steps,args.seed)
    loader=torch.utils.data.DataLoader(train,batch_sampler=sampler)
    validations={'real':ReviewedMasks([r for r in rows if r['split']=='validation'],state['size']), 'synthetic':benchmark}
    val_loaders={k:torch.utils.data.DataLoader(v,batch_size=args.batch_size) for k,v in validations.items()}
    baseline={k:evaluate(model,v,device) for k,v in val_loaders.items()}
    original={k:v.detach().cpu().clone() for k,v in model.generator.state_dict().items()}
    metadata={**vars(args),'device':device,'baseline':baseline,'sources':sources,
              'manifest_sha256':sha(args.manifest),'split_sha256':sha(args.split),
              'benchmark_manifest_sha256':sha(Path(args.benchmark)/'manifest.json'),
              'checkpoint_sha256':sha(args.checkpoint),'background_weight':.25,
              'selection':'real IoU improvement; no real FP/empty regression; strict synthetic retention',
              'test_split':'not evaluated by this run; earlier seven-image test already used diagnostically'}
    out.mkdir(parents=True);(out/'run.json').write_text(json.dumps(metadata,indent=2))
    optimizer=detector_optimizer(model,args.lr);best=baseline['real']['iou']
    for epoch in range(1,args.epochs+1):
        sampler.epoch=epoch-1;model.segmenter.train();total=0.
        for x,m in loader:
            loss=segmentation_loss(model.detect(x.to(device)),m.to(device),.25,.1)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.segmenter.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();total+=float(loss.detach())
        metrics={k:evaluate(model,v,device) for k,v in val_loaders.items()}
        real_ok=metrics['real']['iou']>best and all(metrics['real'][k]<=baseline['real'][k] for k in
                 ('visible_false_positive','empty_mask_cases','negative_false_positive_cases'))
        synthetic_ok=retention_passes(metrics['synthetic'],baseline['synthetic'])
        selected=real_ok and synthetic_ok
        assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in model.generator.state_dict().items()),'Generator changed'
        export={**state,'model':model.state_dict(),'detector_only':True,'detector_finetune_epoch':epoch,'detector_training':metadata}
        torch.save(export,out/f'detector_epoch_{epoch}.pth')
        if selected:best=metrics['real']['iou'];torch.save(export,out/'best_detector.pth')
        report={'epoch':epoch,'loss':total/args.steps,'selected':selected,'real_gate':real_ok,'synthetic_gate':synthetic_ok,**metrics}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        print(json.dumps(report),flush=True)
    print('Complete. Review both validation sets and images before promotion.',flush=True)


if __name__=='__main__':main()
