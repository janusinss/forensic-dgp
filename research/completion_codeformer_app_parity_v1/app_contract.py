"""Exact active-index tensor preparation/composition from the admitted app.

Research repair proposal only. This helper does not construct/load/call a model,
change parameters or apply inference_mode. The caller controls gradient context.
Keep arithmetic, advanced indexing and layout aligned with the frozen app wrapper.
"""
import torch
from torch.nn import functional as F

APP_SOURCE_SHA256='29f2d03d692d762d1d9b39c9747e7becf30a14399651e8c419de2a5b39cae9ce'
POLICY='visible-normalized-resize-active-index-parity-v1'

def prepare_active(image,mask):
    """Return active selector, normalized active input512 and active mask512.

    Inputs must already pass the app's RGB/binary-mask/visibility validation.
    An empty selector returns None stages; no model needs to be called.
    Do not replace active indexing with algebraic equivalence or .contiguous().
    """
    active=mask.sum((1,2,3))>0
    if not active.any():return active,None,None
    visible=1-mask[active]
    support=F.interpolate(visible,size=(512,512),mode='bilinear',align_corners=False)
    x=F.interpolate(image[active]*visible,size=(512,512),mode='bilinear',align_corners=False)
    x=x/support.clamp_min(1e-8)
    m=F.interpolate(mask[active],size=(512,512),mode='nearest')
    x=(x*(1-m)+m)*2-1
    return active,x,m

def compose_active(image,mask,active,raw):
    """Reproduce app active-batch generated resize and indexed composition."""
    result=image.clone()
    if not active.any():return result
    if raw is None or raw.shape!=(int(active.sum().item()),3,512,512) or not torch.isfinite(raw).all():
        raise ValueError('Expected finite active RGB512 output')
    generated=((raw+1)/2).clamp(0,1)
    generated=F.interpolate(generated,size=image.shape[-2:],mode='bilinear',align_corners=False)
    result[active]=image[active]*(1-mask[active])+generated*mask[active]
    return result

def tensor_layout(value):
    """Small input/layout receipt; does not claim numerical model parity."""
    if value is None:return None
    return {'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype),
            'device':str(value.device),'contiguous_NCHW':value.is_contiguous(),
            'contiguous_channels_last':value.is_contiguous(memory_format=torch.channels_last),
            'finite':bool(torch.isfinite(value).all().item())}
