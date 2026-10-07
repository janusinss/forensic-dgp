"""Research-only MAT FFHQ512 conversion comparison; no application promotion."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'outputs/completion_mat_mirror_assets_v1'
WEIGHTS_SHA='eedb8504aef8a07feda7e89ef34e53344eaf3039cb1543615bf1092439ce3d98'
WEIGHTS_BYTES=125280246


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def no_duplicates(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate safetensors header key')
        result[key]=value
    return result


def inspect_header(stream,file_size):
    prefix=stream.read(8)
    if len(prefix)!=8:raise ValueError('Missing safetensors size prefix')
    length=struct.unpack('<Q',prefix)[0]
    if not 2<=length<=256*1024 or 8+length>file_size:raise ValueError('Header finite bound')
    raw=stream.read(length)
    if len(raw)!=length or not raw.startswith(b'{'):raise ValueError('Invalid safetensors JSON header')
    header=json.loads(raw,object_pairs_hook=no_duplicates)
    if not isinstance(header,dict) or not 1<=len(header)<=2000:raise ValueError('Invalid tensor table')
    rows=[];cursor=0
    for name,entry in header.items():
        if name=='__metadata__':
            if not isinstance(entry,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in entry.items()):
                raise ValueError('String-only safetensors metadata')
            continue
        if not isinstance(name,str) or not name or not isinstance(entry,dict) or set(entry)!={'dtype','shape','data_offsets'}:
            raise ValueError('Unexpected tensor entry')
        if entry['dtype']!='F16':raise ValueError('This pinned conversion must contain F16 only')
        shape=entry['shape'];offsets=entry['data_offsets']
        if not isinstance(shape,list) or len(shape)>5 or any(type(s)!=int or s<0 for s in shape):
            raise ValueError('Invalid tensor dimensions')
        if not isinstance(offsets,list) or len(offsets)!=2 or any(type(s)!=int for s in offsets):
            raise ValueError('Invalid offsets')
        count=math.prod(shape)
        if count>50000000 or offsets[0]<0 or offsets[1]<offsets[0] or offsets[1]-offsets[0]!=2*count:
            raise ValueError('Tensor size or offset mismatch')
        rows.append({'name':name,'shape':shape,'begin':offsets[0],'end':offsets[1],'elements':count})
    for row in sorted(rows,key=lambda r:(r['begin'],r['end'])):
        if row['begin']!=cursor:raise ValueError('Overlapping or unindexed tensor bytes')
        cursor=row['end']
    if cursor!=file_size-8-length:raise ValueError('Unindexed or truncated data buffer')
    return rows,8+length


def read_tensor_state(path):
    path=Path(path)
    if path.stat().st_size!=WEIGHTS_BYTES or sha(path)!=WEIGHTS_SHA:raise ValueError('Pinned MAT conversion fingerprint differs')
    with path.open('rb') as stream:
        rows,body_start=inspect_header(stream,path.stat().st_size)
        if len(rows)!=465:raise ValueError('Pinned generator must contain465 tensors')
        state={}
        for row in rows:
            stream.seek(body_start+row['begin'])
            raw=stream.read(row['end']-row['begin'])
            if len(raw)!=row['end']-row['begin']:raise ValueError('Truncated tensor')
            array=np.frombuffer(raw,dtype='<f2').reshape(row['shape'])
            if not np.isfinite(array).all():raise ValueError('Nonfinite pretrained tensor')
            state[row['name']]=torch.from_numpy(array.astype(np.float32,copy=True))
    return state,rows


class MATMirrorCompletion(torch.nn.Module):
    """One fixed estimate; binary reviewed removal support; source copy outside it."""
    def __init__(self,net,seed=240):
        super().__init__()
        self.net=net.eval().requires_grad_(False)
        self.seed=seed
        self.forwards=0
        self.z=torch.from_numpy(np.random.RandomState(seed).randn(1,512).astype(np.float32))
        self.label=torch.zeros((1,0),dtype=torch.float32)

    @torch.inference_mode()
    def forward(self,image,removal,return_raw=False):
        if image.shape!=(1,3,256,256) or removal.shape!=(1,1,256,256):raise ValueError('One prepared256 RGB face and removal mask required')
        if image.device.type!='cpu' or removal.device.type!='cpu':raise ValueError('This comparison uses CPU only')
        if not image.is_floating_point() or not torch.isfinite(image).all() or image.min()<0 or image.max()>1:raise ValueError('Finite RGB[0,1] required')
        if not torch.isfinite(removal).all() or not ((removal==0)|(removal==1)).all():raise ValueError('Explicit binary removal mask required')
        if not removal.any():return (image.clone(),None) if return_raw else image.clone()
        if removal.all():raise ValueError('No visible information; request a less-covered face')
        if self.net.training or any(p.requires_grad for p in self.net.parameters()):raise ValueError('Frozen evaluation model required')
        visible=1-removal
        support=F.interpolate(visible,(512,512),mode='bilinear',align_corners=False)
        rgb=F.interpolate(image*visible,(512,512),mode='bilinear',align_corners=False)/support.clamp_min(1e-8)
        keep=1-F.interpolate(removal,(512,512),mode='nearest')
        # Eliminate covered colors at source and after geometry conversion.
        conditioned=torch.where(keep.bool(),rgb.clamp(0,1)*2-1,torch.zeros_like(rgb))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            raw=self.net(conditioned,keep,self.z,self.label,truncation_psi=1,noise_mode='const')
        self.forwards+=1
        if raw.shape!=(1,3,512,512) or not torch.isfinite(raw).all():raise FloatingPointError('Invalid MAT raw output')
        estimate512=((raw+1)/2).clamp(0,1)
        estimate=F.interpolate(estimate512,(256,256),mode='bilinear',align_corners=False)
        result=torch.where(removal.bool(),estimate,image)
        return (result,raw) if return_raw else result


def load_mat_mirror():
    prepared=json.loads((ASSETS/'preparation.json').read_text(encoding='utf-8'))
    for name,digest in prepared['vendor_sha256'].items():
        if sha(ASSETS/name)!=digest:raise ValueError('Reviewed inference source changed')
    state,rows=read_tensor_state(ASSETS/'MAT_FFHQ_512_fp16.safetensors')
    package='_thesis_mat_mirror_reviewed_v1'
    spec=importlib.util.spec_from_file_location(package,ASSETS/'vendor/__init__.py',submodule_search_locations=[str(ASSETS/'vendor')])
    module=importlib.util.module_from_spec(spec);sys.modules[package]=module;spec.loader.exec_module(module)
    spec=importlib.util.spec_from_file_location(package+'.MAT',ASSETS/'vendor/MAT.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(240)
        generator=module.Generator(z_dim=512,c_dim=0,w_dim=512,img_resolution=512,img_channels=3).eval().requires_grad_(False)
        generator.load_state_dict(state,strict=True)
    if any(not torch.isfinite(t).all() for t in generator.state_dict().values()):raise ValueError('Nonfinite loaded state')
    return MATMirrorCompletion(generator),{'backend':'spacepxl-MAT-FFHQ512-FP16-conversion',
           'weight_sha256':WEIGHTS_SHA,'publisher_revision':prepared['publisher_revision'],
           'CPU_source_revision':prepared['CPU_source_revision'],'weight_tensors':len(rows),
           'weight_values':sum(r['elements'] for r in rows),'CPU_computation':'float32 of published half-precision values',
           'internal_resolution':512,'delivered_resolution':256,'noise_mode':'const','seed':240,'truncation':1,
           'author_original_parity_verified':False,'latent_policy':'same fixed NumPy RandomState240 first512-vector per case',
           'geometry':'visible-support-normalized bilinear RGB; nearest binary mask; bilinear512 estimate to256',
           'mask_policy':'unchanged reviewed assisted removal only; no extra margin',
           'quality_acceptance':False,'automatic_app_promotion':False}
