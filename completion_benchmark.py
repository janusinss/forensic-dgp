"""Export fixed completion cases and score external predictions without dropping failures."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from completion_data import CompletionDataset, image_paths
from completion import region_error


def save_rgb(tensor,path):
    array = (tensor.permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def prepare(paths,output,size=512,seed=42):
    output = Path(output)
    if (output/'manifest.json').exists():
        raise ValueError('Benchmark already exists; choose a new directory')
    for folder in ('input','target','mask','codeformer_input'):
        (output/folder).mkdir(parents=True,exist_ok=True)
    data = CompletionDataset(paths,size,seed,validation=True)
    records = []
    for i in range(len(data)):
        item = data[i]
        name = f'{i:06d}.png'
        save_rgb(item['input'],output/'input'/name)
        save_rgb(item['target'],output/'target'/name)
        save_rgb(item['mask'].expand(3,-1,-1),output/'mask'/name)
        # CodeFormer convention: white holes. Explicit masks remain authoritative.
        white = item['input']*(1-item['mask'])+item['mask']
        save_rgb(white,output/'codeformer_input'/name)
        records.append({'file':name,'source':item['path'],'source_sha256':hashlib.sha256(Path(item['path']).read_bytes()).hexdigest(),
                        'kind':item['kind'],'degraded':item['degraded'],'seed':item['seed']})
    (output/'manifest.json').write_text(json.dumps({'size':size,'seed':seed,'cases':records},indent=2),encoding='utf-8')
    return records


def score(benchmark,predictions):
    benchmark,predictions = Path(benchmark),Path(predictions)
    manifest = json.loads((benchmark/'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for case in manifest['cases']:
        name = case['file']
        row = dict(case)
        try:
            def read(path):
                with Image.open(path) as image:
                    return torch.from_numpy(np.array(image.convert('RGB')).copy()).permute(2,0,1).float()[None]/255
            pred,target = read(predictions/name),read(benchmark/'target'/name)
            if pred.shape != target.shape:
                raise ValueError('Prediction dimensions differ; no silent resizing allowed')
            mask = read(benchmark/'mask'/name)[:,:1]
            for key,region in (('hole_mae',mask),('visible_mae',1-mask)):
                error,valid = region_error(pred,target,region)
                row[key] = error.item() if valid.item() else None
            row['error'] = None
        except (OSError,ValueError) as exc:
            row.update(error=type(exc).__name__,hole_mae=None,visible_mae=None)
        rows.append(row)
    report = {'expected':len(rows),'failures':sum(r['error'] is not None for r in rows),'rows':rows}
    for key in ('hole_mae','visible_mae'):
        values = [r[key] for r in rows if r[key] is not None]
        report[key] = sum(values)/len(values) if values else None
    report['complete'] = report['failures']==0
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    export = sub.add_parser('prepare')
    export.add_argument('--data_dir',required=True)
    export.add_argument('--split_file',required=True,help='Use validation paths only')
    export.add_argument('--output_dir',required=True)
    export.add_argument('--images',type=int,default=20)
    export.add_argument('--size',type=int,default=512)
    export.add_argument('--seed',type=int,default=42)
    evaluate = sub.add_parser('score')
    evaluate.add_argument('--benchmark',required=True)
    evaluate.add_argument('--predictions',required=True)
    evaluate.add_argument('--report',required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        available = {str(Path(p).resolve()) for p in image_paths(args.data_dir)}
        split = json.loads(Path(args.split_file).read_text(encoding='utf-8'))
        paths = sorted(split['validation'])
        train = {str(Path(p).resolve()) for p in split['train']}
        validation = {str(Path(p).resolve()) for p in paths}
        if not validation or len(validation)!=len(paths) or train&validation or not validation<=available or args.images<1:
            raise ValueError('Invalid validation membership or sample count')
        random.Random(args.seed).shuffle(paths)
        records = prepare(paths[:args.images],args.output_dir,args.size,args.seed)
        print(f'Exported {len(records)} cases. CodeFormer requires size 512 and aligned faces.')
    else:
        report = score(args.benchmark,args.predictions)
        Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
        if not report['complete']:
            raise SystemExit('Incomplete predictions: metrics cover successful cases only.')
