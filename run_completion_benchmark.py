"""Run frozen completion candidates on a fixed manifest; never read targets at inference."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from completion_benchmark import save_rgb, score
from completion_inference import load_completion, load_restorer, visible_base
from completion import compose
from pretrained_completion import load_codeformer


def run_cases(benchmark,output,predictor,provenance,device):
    benchmark,output=Path(benchmark),Path(output)
    if output.exists():
        raise ValueError('Prediction output already exists; choose a new directory')
    manifest=json.loads((benchmark/'manifest.json').read_text(encoding='utf-8'))
    names=[c['file'] for c in manifest['cases']]
    if not names or len(set(names))!=len(names) or any(Path(n).name!=n or '/' in n or '\\' in n for n in names):
        raise ValueError('Manifest requires unique plain file names')
    output.mkdir(parents=True)
    rows=[]
    def read(path):
        with Image.open(path) as im:
            a=np.array(im.convert('RGB')).copy()
        return torch.from_numpy(a).permute(2,0,1).float()[None].to(device)/255
    for i,case in enumerate(manifest['cases']):
        start=time.perf_counter()
        try:
            x=read(benchmark/'input'/case['file'])
            m=read(benchmark/'mask'/case['file'])[:,:1]
            with torch.inference_mode():
                y=predictor(x,m,bool(case['degraded']))
            if y.shape!=x.shape or not torch.isfinite(y).all():
                raise ValueError('Invalid prediction shape or values')
            save_rgb(y[0].cpu(),output/case['file'])
            error=None
        except (ValueError,RuntimeError,OSError) as exc:
            error=f'{type(exc).__name__}: {exc}'
        rows.append({'file':case['file'],'seconds':time.perf_counter()-start,'error':error})
        print(f'{i+1}/{len(names)} {case["file"]}: {error or "ok"}',flush=True)
    report={'provenance':provenance,'manifest_sha256':hashlib.sha256((benchmark/'manifest.json').read_bytes()).hexdigest(),
            'rows':rows,'failures':sum(r['error'] is not None for r in rows)}
    (output/'run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True)
    p.add_argument('--output_dir',required=True)
    p.add_argument('--backend',choices=['custom','codeformer'],required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--mask_mode',choices=['oracle','predicted'],default='oracle')
    p.add_argument('--detector',help='Trained custom checkpoint for predicted masks')
    p.add_argument('--restorer',help='Optional restoration for degraded cases only')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    if args.backend=='codeformer':
        model,provenance=load_codeformer(args.checkpoint,args.device)
    else:
        model,state=load_completion(args.checkpoint,args.device)
        provenance={'backend':'custom','epoch':state['epoch'],
                    'weights_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()}
    detector=None
    if args.mask_mode=='predicted':
        if not args.detector:
            p.error('--detector is required for predicted-mask comparisons')
        detector,_=load_completion(args.detector,args.device)
        provenance['detector_sha256']=hashlib.sha256(Path(args.detector).read_bytes()).hexdigest()
    restorer=load_restorer(args.restorer,args.device) if args.restorer else None
    provenance.update(mask_mode=args.mask_mode,compositing_policy='mask-only-v1',device=args.device,
                      restorer_sha256=hashlib.sha256(Path(args.restorer).read_bytes()).hexdigest() if args.restorer else None)
    def predictor(x,known,degraded):
        m=(detector.detect(x).sigmoid()>=.5).float() if detector is not None else known
        if (m.mean((1,2,3))>=.85).any():
            raise ValueError('Too little visible face remains')
        base=visible_base(x,[degraded and restorer is not None],restorer)
        if args.backend=='codeformer': return model(base,m)
        return compose(base,model.complete(base,m),m)
    run=run_cases(args.benchmark,args.output_dir,predictor,provenance,args.device)
    metrics=score(args.benchmark,args.output_dir)
    Path(args.output_dir,'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k!='rows'}))
    if run['failures'] or not metrics['complete']:
        raise SystemExit('Benchmark incomplete; inspect run.json')


if __name__=='__main__':
    main()
