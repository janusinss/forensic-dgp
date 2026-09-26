"""Completion inference without target photos or target landmarks."""
from pathlib import Path
import torch
from completion import CompletionNet, FORMAT, blend_mask, compose, validate_mask


def load_completion(path,device='cpu',allow_smoke=False):
    state = torch.load(path,map_location=device,weights_only=True)
    if state.get('format') != FORMAT:
        raise ValueError('Not a versioned completion inference checkpoint')
    if state.get('dry_run',True) and not allow_smoke:
        raise ValueError('A smoke checkpoint is not a trained completion model')
    if state.get('epoch',0) < 1:
        raise ValueError('Checkpoint contains no completed training epoch')
    model = CompletionNet(state['width']).to(device).eval()
    model.load_state_dict(state['model'],strict=True)
    return model,state


def load_restorer(path,device):
    from models import DGPSynthesizer
    model = DGPSynthesizer().to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(Path(path),map_location=device,weights_only=True),strict=True)
    return model


@torch.no_grad()
def visible_base(image,degraded,restorer):
    if not torch.as_tensor(degraded).any():
        return image
    if restorer is None:
        raise ValueError('A restoration checkpoint is required to restore visible regions')
    restored = restorer(image)
    if restored.shape != image.shape:
        restored = torch.nn.functional.interpolate(restored,size=image.shape[-2:],mode='bilinear',align_corners=False)
    use = torch.as_tensor(degraded,device=image.device).reshape(-1,1,1,1)
    return torch.where(use,restored,image)


@torch.no_grad()
def predict(model,image,mask=None,restore_visible=False,restorer=None,threshold=.5,radius=0):
    if not 0 < threshold < 1:
        raise ValueError('Threshold must be between zero and one')
    if not torch.isfinite(image).all() or image.min()<0 or image.max()>1:
        raise ValueError('Expected finite RGB pixels in [0,1]')
    probability = model.detect(image).sigmoid()
    supplied = mask is not None
    mask = (probability>=threshold).float() if mask is None else mask
    validate_mask(image,mask)
    if (mask.mean((1,2,3)) >= .85).any():
        raise ValueError('Too little visible face remains; correct the region or use a clearer image')
    base = visible_base(image,[restore_visible]*len(image),restorer)
    alpha = blend_mask(mask,radius)
    generated = model.complete(base,mask)
    result = compose(base,generated,alpha)
    return {'output':result,'visible':base,'mask':mask,'alpha':alpha,
            'probability':probability,'mask_source':'supplied' if supplied else 'predicted'}
