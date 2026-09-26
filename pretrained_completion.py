"""CodeFormer inpainting with explicit masks and exact visible-pixel preservation."""
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from completion import validate_mask, compose

CODEFORMER_REVISION = 'b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b'
WEIGHTS_URL = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth'
WEIGHTS_SHA256 = 'b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'


class CodeFormerCompletion(torch.nn.Module):
    """Accept aligned RGB crops in [0,1] and a binary N1HW covered-region mask.

    Generation uses the official 512px normalization, w=1 and adain=False.
    Only generated pixels are resized back; observed pixels retain input values.
    """
    def __init__(self, net):
        super().__init__()
        self.net = net.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, image, mask):
        validate_mask(image,mask)
        if image.shape[-1]!=image.shape[-2] or image.shape[-1]<32:
            raise ValueError('Use aligned square face crops of at least 32 pixels')
        if not torch.isfinite(image).all() or image.min()<0 or image.max()>1:
            raise ValueError('Expected finite RGB values in [0,1]')
        if not ((mask==0)|(mask==1)).all():
            raise ValueError('Use a binary explicit mask')
        if (mask.mean((1,2,3))>=.85).any():
            raise ValueError('Too little visible face remains')
        result=image.clone()
        active=mask.sum((1,2,3))>0
        if not active.any():
            return result
        # Normalize by visible support: neither the covering nor the white fill
        # may contaminate observed context while resizing along a mask edge.
        visible=1-mask[active]
        support=F.interpolate(visible,size=(512,512),mode='bilinear',align_corners=False)
        x=F.interpolate(image[active]*visible,size=(512,512),mode='bilinear',align_corners=False)
        x=x/support.clamp_min(1e-8)
        m=F.interpolate(mask[active],size=(512,512),mode='nearest')
        x=(x*(1-m)+m)*2-1
        generated=self.net(x,w=1,adain=False)[0]
        if generated.shape!=x.shape or not torch.isfinite(generated).all():
            raise FloatingPointError('Invalid pretrained model output')
        generated=((generated+1)/2).clamp(0,1)
        generated=F.interpolate(generated,size=image.shape[-2:],mode='bilinear',align_corners=False)
        result[active]=compose(image[active],generated,mask[active])
        return result


def load_codeformer(path,device='cpu'):
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=WEIGHTS_SHA256:
        raise ValueError('CodeFormer inpainting checkpoint SHA256 mismatch')
    from third_party.codeformer.codeformer_arch import CodeFormer
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=CodeFormer(dim_embd=512,codebook_size=512,n_head=8,n_layers=9,
                   connect_list=['32','64','128'])
    net.load_state_dict(state['params_ema'],strict=True)
    if not all(torch.isfinite(p).all() for p in net.parameters()):
        raise ValueError('Non-finite pretrained weights')
    model=CodeFormerCompletion(net).to(device).eval()
    provenance={'backend':'codeformer-inpainting','source_revision':CODEFORMER_REVISION,
                'weights_sha256':digest,
                'weights_url':WEIGHTS_URL,'compositing_policy':'mask-only-v1',
                'input_policy':'visible-normalized-resize-v1'}
    return model,provenance
