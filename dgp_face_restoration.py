"""Strict, inference-only DGP256 adapter for the next reviewed application route.

Loading an explicitly fingerprinted checkpoint does not establish output quality
or select it as the application's default. No display enhancement or top-k scores.
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch

PHASE3_SHA256="b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c"
POLICY="dgp256-observed-canvas-v1"


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_crop(rgb,removal_mask=None):
    """Native neutral128 center-pad; PIL bilinear256, nearest masks, no warping."""
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
        raise ValueError('Expected nonempty uint8 RGB face pixels')
    h,w=rgb.shape[:2]
    if max(h,w)>4096 or h*w>16_000_000:
        raise ValueError('Face crop exceeds the image bounds')
    if removal_mask is not None and (not isinstance(removal_mask,np.ndarray) or removal_mask.shape!=(h,w) or not np.isin(removal_mask,(0,1)).all()):
        raise ValueError('Use a matching binary removal mask')
    side=max(h,w);top,left=(side-h)//2,(side-w)//2
    canvas=Image.new('RGB',(side,side),(128,128,128));canvas.paste(Image.fromarray(rgb),(left,top))
    support=Image.new('L',(side,side),0);support.paste(255,(left,top,left+w,top+h))
    removal=Image.new('L',(side,side),0)
    if removal_mask is not None:
        removal.paste(Image.fromarray(removal_mask.astype(np.uint8)*255),(left,top))
    target=np.asarray(canvas.resize((256,256),Image.Resampling.BILINEAR)).copy()
    observed=np.asarray(support.resize((256,256),Image.Resampling.NEAREST)).copy()>0
    mask=(np.asarray(removal.resize((256,256),Image.Resampling.NEAREST)).copy()>0)&observed
    yy,xx=np.nonzero(observed)
    geometry={'native_size':[w,h],'native_center_offset':[left,top],
              'observed_bounds_256':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
              'output_size':[256,256],'policy':POLICY,
              'input_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
              'canvas_rgb_sha256':hashlib.sha256(target.tobytes()).hexdigest(),
              'observed_sha256':hashlib.sha256(observed.tobytes()).hexdigest(),
              'removal_mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()}
    return target,observed,mask,geometry


def as_tensor(rgb,device='cpu'):
    return torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None].to(device)/255


class DGPFaceRestoration(torch.nn.Module):
    """No hidden resize or blending: exactly one frozen DGP forward at256."""
    def __init__(self,net):
        super().__init__();self.net=net.eval().requires_grad_(False);self.eval()

    def train(self,mode=True):
        return super().train(False)

    @torch.inference_mode()
    def forward(self,image):
        if not isinstance(image,torch.Tensor) or image.ndim!=4 or image.shape[1:]!=(3,256,256) or not len(image):
            raise ValueError('Prepare a nonempty N x3 x256 x256 RGB batch first')
        if image.dtype!=torch.float32 or not torch.isfinite(image).all() or image.min()<0 or image.max()>1:
            raise ValueError('Expected finite float32 RGB in[0,1]')
        self.net.eval()
        output=self.net(image)
        if output.shape!=image.shape or output.dtype!=torch.float32 or not torch.isfinite(output).all() or output.min()<0 or output.max()>1:
            raise FloatingPointError('Invalid DGP restoration output')
        return output


def load_dgp_restorer(path,device='cpu',expected_sha256=PHASE3_SHA256):
    """Strict fingerprint/schema/finite loading; a missing model never random-fills."""
    if not isinstance(expected_sha256,str) or len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('Supply the frozen lowercase SHA256 checkpoint fingerprint')
    path=Path(path);digest=sha(path)
    if digest!=expected_sha256:
        raise ValueError('DGP checkpoint fingerprint differs from the reviewed selection')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not state or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError('Expected a plain DGP tensor state, not a training/smoke container')
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Nonfinite DGP checkpoint tensors')
    from models import DGPSynthesizer
    net=DGPSynthesizer();net.load_state_dict(state,strict=True)
    if sha(path)!=digest:
        raise ValueError('DGP checkpoint changed while loading')
    model=DGPFaceRestoration(net).to(device).eval()
    provenance={'backend':'dgp-restoration','policy':POLICY,'weights_sha256':digest,
                'starting_checkpoint':'retained_phase3' if digest==PHASE3_SHA256 else 'explicitly_fingerprinted_candidate',
                'inference_only':True,'internal_resolution':256,'input_policy':'Prepared RGB float32[0,1]; DGP internally uses[-1,1]',
                'enhancement':'none','fidelity_control':None,'automatic_app_promotion':False,
                'quality_claim':'Checkpoint loading is not evidence of useful restoration or recovered identity'}
    return model,provenance


@torch.inference_mode()
def restore_crop(model,rgb,device='cpu'):
    """Return native DGP float and observed composite separately at256."""
    canvas,observed,_,geometry=prepare_crop(rgb)
    x=as_tensor(canvas,device);raw=model(x)[0].permute(1,2,0).cpu().numpy().copy()
    composite=np.where(observed[...,None],raw,canvas.astype(np.float32)/255)
    return {'input':canvas,'observed':observed,'raw_rgb':raw,'observed_rgb':composite,'geometry':geometry}


def png_rgb(float_rgb):
    if float_rgb.shape!=(256,256,3) or not np.isfinite(float_rgb).all() or float_rgb.min()<0 or float_rgb.max()>1:
        raise ValueError('Expected finite256 RGB output in[0,1]')
    return np.floor(float_rgb*255).astype(np.uint8)
