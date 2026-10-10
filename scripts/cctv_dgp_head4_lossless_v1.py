"""Bit-exact float32 storage. Byte planes/XOR change neither values nor display."""
from pathlib import Path
import sys
import zipfile
import numpy as np


def pack(path,raw,ids,embeddings,baseline=None):
    assert sys.byteorder=='little' and raw.dtype==np.float32 and raw.shape==(5,256,256,3)
    assert np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
    assert len(ids)==5 and len(set(ids))==5 and embeddings.dtype==np.float32 and embeddings.shape==(5,3,512)
    assert np.isfinite(embeddings).all() and not Path(path).exists()
    bits=np.ascontiguousarray(raw).view('<u4')
    if baseline is not None:
        assert baseline.dtype==np.float32 and baseline.shape==raw.shape
        bits=np.bitwise_xor(bits,np.ascontiguousarray(baseline).view('<u4'))
    planes=bits.view(np.uint8).reshape(-1,4).T.copy()
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,planes=planes,shape=np.array(raw.shape,dtype=np.int64),xor=np.array(baseline is not None),
            ids=np.array(ids),embeddings=embeddings)


def unpack(path,ids,baseline=None):
    assert sys.byteorder=='little'
    with zipfile.ZipFile(path) as z:
        members=z.infolist()
        assert {m.filename for m in members}=={'planes.npy','shape.npy','xor.npy','ids.npy','embeddings.npy'} and len(members)==5
        assert sum(m.file_size for m in members)<8*1024**2 and all(not m.is_dir() for m in members)
    with np.load(path,allow_pickle=False) as a:
        assert a['shape'].dtype==np.int64 and a['shape'].tolist()==[5,256,256,3]
        assert a['ids'].dtype.kind=='U' and a['ids'].tolist()==ids
        assert a['xor'].dtype==np.bool_ and a['xor'].shape==() and bool(a['xor'])==(baseline is not None)
        planes=a['planes'];assert planes.dtype==np.uint8 and planes.shape==(4,5*256*256*3)
        bits=planes.T.copy().reshape(5,256,256,3,4).reshape(5,256,256,12).view('<u4').reshape(5,256,256,3)
        if baseline is not None:
            assert baseline.dtype==np.float32 and baseline.shape==(5,256,256,3)
            bits=np.bitwise_xor(bits,np.ascontiguousarray(baseline).view('<u4'))
        raw=bits.view('<f4').copy();em=a['embeddings'].copy()
    assert raw.dtype==np.float32 and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
    assert em.dtype==np.float32 and em.shape==(5,3,512) and np.isfinite(em).all()
    return raw,em
