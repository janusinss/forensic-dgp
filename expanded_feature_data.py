"""Versioned training-only source expansion; original completion benchmark is untouched."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from anatomical_augmentation import anatomical_covering
from completion import KINDS, synthetic_covering

FORMAT='expanded-anatomical-training-v1'
SOURCES=('dataset/asian_faces','dataset/thumbnails128x128')


def local_path(root,name):
    name=str(name).replace('\\','/')
    p=PurePosixPath(name)
    if p.is_absolute() or PureWindowsPath(name).drive or '..' in p.parts:
        raise ValueError('Expected a workspace-relative path')
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Path leaves workspace')
    return path


class ExpandedCoveringDataset(Dataset):
    """Ten fixed variants per source, minus explicitly rejected anatomy variants.

    Landmark cache coordinates refer to the legacy square 256 resize. Keeping
    that transform fixed isolates source/covering changes; distortion remains a
    documented limitation. This dataset is for detector fitting, not a claim that
    provisional source photos are verified uncovered reconstruction ground truth.
    """
    def __init__(self,manifest,root='.'):
        self.manifest=copy.deepcopy(manifest);self.root=Path(root).resolve()
        if manifest.get('format')!=FORMAT or manifest.get('partition')!='train':
            raise ValueError('Expected expanded training manifest')
        split_path=local_path(self.root,manifest['split_path'])
        if hashlib.sha256(split_path.read_bytes()).hexdigest()!=manifest['split_sha256']:
            raise ValueError('Split hash mismatch')
        split=json.loads(split_path.read_text())
        normalize=lambda p:p.replace('\\','/')
        train=set(map(normalize,split['train']));held=set(map(normalize,split['validation']))
        held.update(map(normalize,split.get('test',[])))
        if train&held:raise ValueError('Training and held-out split overlap')
        self.sources=self.manifest['sources'];self.seed=int(manifest['seed'])
        excluded=set(manifest['excluded_sha256']);paths=set();hashes=set()
        if not self.sources:raise ValueError('No training sources')
        self.cases=[];self.rejections=[]
        for n,r in enumerate(self.sources):
            path=local_path(self.root,r['path']);name=normalize(r['path'])
            if r['source'] not in SOURCES or PurePosixPath(name).parent.as_posix()!=r['source']:
                raise ValueError('Unexpected source group')
            if name not in train or name in held:raise ValueError('Source outside training split')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=r['sha256'] or digest in excluded:raise ValueError('Source hash mismatch or held-out content')
            if name in paths or digest in hashes:raise ValueError('Duplicate training source')
            paths.add(name);hashes.add(digest)
            rgb=self._read(n)
            for variant in range(10):
                kind=KINDS[variant%5];event=None
                if kind in ('lower','eyes'):
                    _,_,event=anatomical_covering(rgb,r['landmarks'],kind,self._seed(r)+variant%5,boundary_policy='clip')
                    if event['status']!='generated':
                        self.rejections.append({'source_index':n,'path':name,'variant':variant,'event':event})
                        continue
                self.cases.append({'source_index':n,'variant':variant,'kind':kind,'degraded':variant>=5,'source':r['source']})

    def _seed(self,row):
        stable=int(hashlib.sha256(row['path'].replace('\\','/').encode()).hexdigest()[:8],16)
        return (self.seed+stable)%(2**32)

    def _read(self,n):
        row=self.sources[n];raw=local_path(self.root,row['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Source content changed')
        bgr=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if bgr is None:raise ValueError('Undecodable source')
        return cv2.cvtColor(cv2.resize(bgr,(256,256)),cv2.COLOR_BGR2RGB)

    def __len__(self):return len(self.cases)

    def __getitem__(self,index):
        case=self.cases[index];r=self.sources[case['source_index']]
        rgb=self._read(case['source_index']);seed=self._seed(r);kind=case['kind']
        if kind in ('lower','eyes'):
            covered,geometry,event=anatomical_covering(rgb,r['landmarks'],kind,seed+case['variant']%5,boundary_policy='clip')
            if event['status']!='generated':raise ValueError('Previously accepted geometry changed')
        else:
            covered,geometry=synthetic_covering(rgb,seed+case['variant']%5,kind)
            event={'version':'legacy-generic','status':'generated','kind':kind}
        mask=geometry.copy()
        if case['degraded']:
            # Identical camera recipe and RNG ordering to fixed CompletionDataset.
            rng=np.random.default_rng(seed)
            covered=cv2.GaussianBlur(covered,(7,7),float(rng.uniform(.5,1.5)))
            low=int(rng.integers(64,257))
            covered=cv2.resize(cv2.resize(covered,(low,low),interpolation=cv2.INTER_AREA),(256,256),interpolation=cv2.INTER_LINEAR)
            covered=np.clip(covered.astype(float)+rng.normal(0,rng.uniform(1,12),covered.shape),0,255).astype(np.uint8)
            ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(covered,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(35,96))])
            if not ok:raise ValueError('JPEG simulation failed')
            covered=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            mask=cv2.dilate(mask,np.ones((17,17),np.uint8))
        tensor=lambda a:torch.from_numpy(a.copy()).permute(2,0,1).float()/255
        return {'input':tensor(covered),'target':tensor(rgb),'mask':torch.from_numpy(mask[None].astype(np.float32)),
                'geometry':torch.from_numpy(geometry[None].astype(np.float32)),**case,
                'seed':seed,'path':r['path'],'augmentation':event}


def expanded_balanced_schedule(groups,seed,epochs=20,steps_per_epoch=80):
    """Two samples per group per batch; cycle without epoch-boundary starvation.

    Budget stays fixed across dataset sizes. Fail if any group cannot be covered
    at least once over the complete experiment. This schedules data, never fitting.
    """
    groups=[list(g) for g in groups];flat=[i for g in groups for i in g]
    if epochs<1 or steps_per_epoch<1 or len(groups)!=6 or not all(groups):
        raise ValueError('Expected six nonempty groups and a positive budget')
    if len(flat)!=len(set(flat)):raise ValueError('Duplicate or overlapping group indices')
    draws=epochs*steps_per_epoch*2
    if max(map(len,groups))>draws:raise ValueError('Budget cannot cover the largest group')
    rng=np.random.default_rng(seed);streams=[]
    for group in groups:
        stream=[]
        while len(stream)<draws:stream.extend(rng.permutation(group).tolist())
        streams.append(stream[:draws])
    for epoch in range(epochs):
        batches=[]
        for step in range(steps_per_epoch):
            start=(epoch*steps_per_epoch+step)*2
            batches.append([i for stream in streams for i in stream[start:start+2]])
        yield batches
