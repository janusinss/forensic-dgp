"""Append-only, disk-backed frozen feature batches with explicit completion state."""
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
import numpy as np


def digest(feature,target,record):
    h=hashlib.sha256(feature.tobytes());h.update(target.tobytes())
    h.update(json.dumps(record,sort_keys=True,separators=(',',':')).encode())
    return h.hexdigest()


class FeatureDiskCache:
    @classmethod
    def create(cls,path,count,feature_shape,target_shape,context):
        path=Path(path)
        if count<1 or any(n<1 for n in (*feature_shape,*target_shape)):raise ValueError('Invalid cache shape')
        needed=count*(math.prod(feature_shape)*4+math.prod(target_shape))
        parent=path.parent
        if shutil.disk_usage(parent).free<needed+64*1024**2:raise ValueError('Insufficient free disk for cache')
        path.mkdir(exist_ok=False)
        obj=cls();obj.path=path;obj.writable=True
        obj.state={'format':'frozen-feature-disk-v1','count':count,'feature_shape':list(feature_shape),
                   'target_shape':list(target_shape),'context':copy.deepcopy(context),'written':0,'complete':False,'rows':[]}
        obj._save()
        obj.features=np.memmap(path/'features.bin',dtype=np.float32,mode='w+',shape=(count,*feature_shape))
        obj.targets=np.memmap(path/'targets.bin',dtype=np.uint8,mode='w+',shape=(count,*target_shape))
        return obj

    @classmethod
    def open(cls,path,context):
        obj=cls();obj.path=Path(path);obj.writable=False
        obj.state=json.loads((obj.path/'state.json').read_text());s=obj.state
        if s.get('format')!='frozen-feature-disk-v1' or s.get('context')!=context:raise ValueError('Cache provenance mismatch')
        if not s.get('complete') or s['written']!=s['count'] or len(s['rows'])!=s['count']:raise ValueError('Incomplete cache; preserve and inspect')
        for name,shape,dtype in [('features',s['feature_shape'],np.float32),('targets',s['target_shape'],np.uint8)]:
            expected=s['count']*math.prod(shape)*np.dtype(dtype).itemsize
            if (obj.path/(name+'.bin')).stat().st_size!=expected:raise ValueError('Cache size mismatch')
        obj.features=np.memmap(obj.path/'features.bin',dtype=np.float32,mode='r',shape=(s['count'],*s['feature_shape']))
        obj.targets=np.memmap(obj.path/'targets.bin',dtype=np.uint8,mode='r',shape=(s['count'],*s['target_shape']))
        return obj

    def _save(self):
        tmp=self.path/'state.json.tmp';tmp.write_text(json.dumps(self.state,indent=2))
        tmp.replace(self.path/'state.json')

    def append(self,feature,target,record):
        if not self.writable or self.state['complete']:raise ValueError('Cache is not writable')
        i=self.state['written']
        if i>=self.state['count']:raise ValueError('Cache is full')
        f=np.asarray(feature);t=np.asarray(target)
        if list(f.shape)!=self.state['feature_shape'] or list(t.shape)!=self.state['target_shape']:raise ValueError('Unexpected cache row shape')
        if not np.isfinite(f).all() or not np.isin(t,[0,1]).all():raise ValueError('Nonfinite features or invalid mask')
        f=f.astype(np.float32);t=t.astype(np.uint8)
        if not np.isfinite(f).all():raise ValueError('Feature overflow')
        record=copy.deepcopy(record);row={'record':record,'sha256':digest(f,t,record)}
        self.features[i]=f;self.targets[i]=t
        self.features.flush();self.targets.flush()
        self.state['rows'].append(row);self.state['written']=i+1;self._save()

    def finish(self):
        if not self.writable or self.state['written']!=self.state['count']:raise ValueError('Incomplete cache')
        self.features.flush();self.targets.flush();self.state['complete']=True;self._save()

    def batch(self,indices):
        if not self.state['complete']:raise ValueError('Incomplete cache')
        indices=list(indices)
        if not indices or any(not isinstance(i,(int,np.integer)) or i<0 or i>=self.state['count'] for i in indices):raise ValueError('Invalid batch indices')
        f=np.array(self.features[indices],copy=True);t=np.array(self.targets[indices],copy=True)
        records=[]
        for n,i in enumerate(indices):
            row=self.state['rows'][i]
            if digest(f[n],t[n],row['record'])!=row['sha256']:raise ValueError('Cache row checksum mismatch')
            records.append(copy.deepcopy(row['record']))
        return f,t,records

    def close(self):
        for name in ('features','targets'):
            value=getattr(self,name,None)
            if value is not None:
                value._mmap.close();setattr(self,name,None)
