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


def balanced_sample(paths,roots,per_source,seed=42):
    """Sample validation paths equally by dataset root, never silently underfill."""
    if per_source<1 or not roots:
        raise ValueError('Positive per-source count and dataset roots required')
    groups={str(Path(root).resolve()):[] for root in roots}
    if len(groups)!=len(roots):
        raise ValueError('Dataset roots must be unique')
    for path in sorted(paths):
        matches=[root for root in groups if Path(path).resolve().is_relative_to(root)]
        if len(matches)!=1:
            raise ValueError('Each validation image must belong to exactly one dataset root')
        groups[matches[0]].append(path)
    selected=[]
    rng=random.Random(seed)
    for root,items in groups.items():
        if len(items)<per_source:
            raise ValueError(f'Insufficient validation images in {root}: {len(items)} < {per_source}')
        selected.extend(rng.sample(items,per_source))
    return selected


def prepare(paths,output,size=512,seed=42,source_roots=None):
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
        dataset_source=Path(item['path']).parent.name
        if source_roots:
            matches=[str(Path(root)) for root in source_roots if Path(item['path']).resolve().is_relative_to(Path(root).resolve())]
            if len(matches)!=1:
                raise ValueError('Ambiguous dataset source')
            dataset_source=matches[0]
        records.append({'file':name,'source':item['path'],'dataset_source':dataset_source,'source_sha256':hashlib.sha256(Path(item['path']).read_bytes()).hexdigest(),
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
    def aggregate(items):
        result={'expected':len(items),'failures':sum(r['error'] is not None for r in items)}
        for key in ('hole_mae','visible_mae'):
            values=[r[key] for r in items if r[key] is not None]
            result[key]=sum(values)/len(values) if values else None
        result['complete']=bool(items) and result['failures']==0
        return result
    report=aggregate(rows)
    report['rows']=rows
    def source(row): return row.get('dataset_source',Path(row['source']).parent.name)
    report['sources']={s:aggregate([r for r in rows if source(r)==s]) for s in sorted({source(r) for r in rows})}
    report['groups']={f'{kind}/{degraded}':aggregate([r for r in rows if r['kind']==kind and r['degraded']==degraded])
                      for kind,degraded in sorted({(r['kind'],r['degraded']) for r in rows})}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    export = sub.add_parser('prepare')
    export.add_argument('--data_dir',required=True)
    export.add_argument('--split_file',required=True,help='Use validation paths only')
    export.add_argument('--output_dir',required=True)
    export.add_argument('--images',type=int,default=20)
    export.add_argument('--images_per_source',type=int,help='Equal validation image count from each data_dir root; overrides --images')
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
        roots=[root.strip() for root in args.data_dir.split(',')]
        if args.images_per_source is not None:
            paths=balanced_sample(paths,roots,args.images_per_source,args.seed)
        else:
            random.Random(args.seed).shuffle(paths)
            paths=paths[:args.images]
        records = prepare(paths,args.output_dir,args.size,args.seed,source_roots=roots)
        print(f'Exported {len(records)} cases from {len(paths)} validation images. Use aligned face crops.')
    else:
        report = score(args.benchmark,args.predictions)
        Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
        if not report['complete']:
            raise SystemExit('Incomplete predictions: metrics cover successful cases only.')
