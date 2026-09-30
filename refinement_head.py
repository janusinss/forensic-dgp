"""Experimental frozen-parent residual refinement; not an application default."""
import torch
from torch import nn
from torch.nn import functional as F
from scripts.compare_pixel_heads_vm import PixelHead


class RefinementHead(nn.Module):
    def __init__(self, parent_state, *, use_rgb):
        super().__init__()
        self.use_rgb=bool(use_rgb)
        self.parent=PixelHead(3)
        self.parent.load_state_dict(parent_state)
        self.parent.eval().requires_grad_(False)
        self.semantic=nn.Sequential(nn.Conv2d(256,16,1),nn.GELU())
        self.refine=nn.Sequential(nn.Conv2d(20,32,3,padding=1),nn.GELU(),
                                  nn.Conv2d(32,16,3,padding=1),nn.GELU())
        self.output=nn.Conv2d(16,1,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def train(self,mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def forward(self,features,rgb):
        if features.ndim!=4 or features.shape[1:]!=(256,64,64):
            raise ValueError('Expected Bx256x64x64 cached features')
        if rgb.shape!=(features.shape[0],3,256,256) or not torch.isfinite(rgb).all() or torch.any((rgb<0)|(rgb>1)):
            raise ValueError('Expected finite Bx3x256x256 RGB in [0,1]')
        if not torch.isfinite(features).all():
            raise ValueError('Nonfinite features')
        features=features.detach().float()
        with torch.no_grad():base=self.parent(features)
        normalized=F.layer_norm(features.permute(0,2,3,1),(256,)).permute(0,3,1,2)
        semantic=F.interpolate(self.semantic(normalized),(128,128),mode='bilinear',align_corners=False)
        image=F.interpolate(rgb.detach().float(),(128,128),mode='bilinear',align_corners=False)*2-1
        if not self.use_rgb:image=torch.zeros_like(image)
        coarse=F.interpolate(base,(128,128),mode='bilinear',align_corners=False)
        residual=self.output(self.refine(torch.cat((semantic,coarse,image),dim=1)))
        return base+F.interpolate(residual,(256,256),mode='bilinear',align_corners=False)
