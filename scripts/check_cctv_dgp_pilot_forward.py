"""Bounded local forward-only compatibility check. GPU backward remains VM-only."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cctv_dgp_pilot import (verify_bundle,sha,write,read,state_hash,PilotDataset,FixedObservedIdentity,
                            PilotPerceptual,identity_crop,composite,base_loss)
from models import DGPSynthesizer


def main(root,save):
    import onnxruntime as ort
    if save.exists():
        raise ValueError('Preserve completed/partial forward check')
    p=verify_bundle(root);torch.set_num_threads(4)
    cases=[next(c for c in p['validation_cases'] if c['source']==s and c['profile']=='compound_lr24')
           for s in ('dataset/thumbnails128x128','dataset/asian_faces')]
    dataset=PilotDataset(root,p,cases)
    batch={k:torch.stack([dataset[i][k] for i in range(2)]) for k in ('low','target','mask','grid')}
    model=DGPSynthesizer().cpu().eval().requires_grad_(False)
    model.load_state_dict(torch.load(root/p['weights']['dgp'],map_location='cpu',weights_only=True),strict=True)
    identity=FixedObservedIdentity(root/p['weights']['arcface'],'cpu')
    perceptual=PilotPerceptual(root/p['weights']['vgg_trunk'],'cpu')
    before={'student':state_hash(model),'identity':state_hash(identity),'perceptual':state_hash(perceptual)}
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(root/p['weights']['arcface']),sess_options=options,providers=['CPUExecutionProvider'])
    start=time.monotonic()
    with torch.inference_mode():
        prediction=composite(model(batch['low']),batch['low'],batch['mask'])
        crop=identity_crop(batch['target'],batch['mask'],batch['grid'])*2-1
        expected=session.run(None,{session.get_inputs()[0].name:crop.numpy()})[0]
        actual=identity.encoder(crop).numpy()
        np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-4)
        embeddings=identity.embedding(prediction,batch['mask'],batch['grid'])
        loss,parts=base_loss(prediction,batch['target'],batch['mask'],perceptual)
    if not torch.isfinite(prediction).all() or not torch.isfinite(embeddings).all() or not torch.isfinite(loss):
        raise ValueError('Nonfinite forward check')
    after={'student':state_hash(model),'identity':state_hash(identity),'perceptual':state_hash(perceptual)}
    if before!=after or any(x.grad is not None for m in (model,identity,perceptual) for x in m.parameters()):
        raise ValueError('Forward-only check changed state or gradients')
    write(save,{'complete':True,'protocol_sha256':sha(root/'protocol.json'),'check_script_sha256':sha(Path(__file__)),
        'cases':[c['id'] for c in cases],'dgp_forward_batches':1,'dgp_forward_images':2,
        'converted_arcface_forward_batches':2,'onnx_forward_batches':1,'vgg_forward_batches':2,
        'encoder_conversion_max_error':float(np.abs(expected-actual).max()),'loss_finite':True,
        'seconds_after_loading':time.monotonic()-start,'states_unchanged':True,'state_sha256':before,
        'local_backward_calls':0,'optimizer_constructed':False,'optimizer_updates':0,
        'gpu_backward_and_vram_verified':False,'training_quality_established':False})
    print(read(save))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/cctv_dgp_vm_bundle_v1')
    parser.add_argument('--save',type=Path,default=ROOT/'outputs/cctv_dgp_pilot_protocol_v1/local_forward_check.json')
    args=parser.parse_args();main(args.root,args.save)
