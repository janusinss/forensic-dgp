"""Reproducible paired coverings: geometry -> object -> camera degradation."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from completion import KINDS, synthetic_covering


def image_paths(roots):
    paths = []
    for root in roots.split(','):
        folder = Path(root.strip())
        items = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg'))
        if not items:
            raise ValueError(f'Missing or empty dataset: {folder}')
        paths.extend(str(p) for p in items)
    return sorted(set(paths))


def signature(paths):
    """Content signature detects modified files, not just changed membership."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(Path(path)).replace('\\','/').encode())
        digest.update(hashlib.sha256(Path(path).read_bytes()).digest())
    return digest.hexdigest()


class CompletionDataset(Dataset):
    def __init__(self,paths,size=256,seed=42,validation=False):
        if size < 32 or size % 8 or not paths:
            raise ValueError('Nonempty dataset and size >=32 divisible by eight required')
        self.paths = list(paths)
        self.size,self.seed,self.validation = size,seed,validation
        self.epoch = 0

    def __len__(self):
        return len(self.paths)*(10 if self.validation else 1)

    def __getitem__(self,index):
        source_index = index//10 if self.validation else index
        path = self.paths[source_index]
        stable = int(hashlib.sha256(Path(path).as_posix().encode()).hexdigest()[:8],16)
        seed = (self.seed+stable+(0 if self.validation else self.epoch*1000003)) % (2**32)
        rng = np.random.default_rng(seed)
        kind_index = index%5 if self.validation else int(rng.integers(5))
        degraded = index%10 >= 5 if self.validation else bool(rng.integers(2))
        rgb = cv2.imread(path)
        if rgb is None:
            raise ValueError(f'Cannot decode {path}')
        rgb = cv2.cvtColor(cv2.resize(rgb,(self.size,self.size)),cv2.COLOR_BGR2RGB)
        covered,geometry = synthetic_covering(rgb,seed+kind_index,KINDS[kind_index])
        mask = geometry.copy()
        if degraded:
            # Local RNG only: no worker/global RNG interference.
            covered = cv2.GaussianBlur(covered,(7,7),float(rng.uniform(.5,1.5)))
            low = int(rng.integers(max(8,self.size//4),self.size+1))
            covered = cv2.resize(cv2.resize(covered,(low,low),interpolation=cv2.INTER_AREA),
                                 (self.size,self.size),interpolation=cv2.INTER_LINEAR)
            covered = np.clip(covered.astype(float)+rng.normal(0,rng.uniform(1,12),covered.shape),0,255).astype(np.uint8)
            ok,encoded = cv2.imencode('.jpg',cv2.cvtColor(covered,cv2.COLOR_RGB2BGR),
                                     [cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(35,96))])
            if not ok:
                raise ValueError('JPEG simulation failed')
            covered = cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            radius = max(2,self.size//32)
            mask = cv2.dilate(mask,np.ones((2*radius+1,2*radius+1),np.uint8))
        def tensor(array):
            return torch.from_numpy(array.copy()).permute(2,0,1).float()/255
        return {'input':tensor(covered),'target':tensor(rgb),
                'mask':torch.from_numpy(mask[None].astype(np.float32)),
                'geometry':torch.from_numpy(geometry[None].astype(np.float32)),
                'degraded':degraded,'kind':KINDS[kind_index],'path':path,
                'seed':int(seed),'source':Path(path).parent.name}
