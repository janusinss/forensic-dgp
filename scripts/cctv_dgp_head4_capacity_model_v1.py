"""Isolated original DGP branch learning; exact-current-output initializer."""
import copy
from pathlib import Path
import platform
import sys
import torch
from torch import nn
from torch.nn import functional as F

NAME='cctv_dgp_head4_capacity_vm_v1'


def require_vm(root,value):
    assert sys.platform=='linux' and platform.node().split('.')[0]=='forensic-dgp-thesis','Manual existing VM only'
    assert Path(root).resolve()==(Path.home()/'forensic-dgp'/NAME).resolve()
    assert value.device.type=='cuda' and torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA L4'


class Head4CapacityDGP(nn.Module):
    def __init__(self,original):
        super().__init__();assert not original.training and all(not p.requires_grad for p in original.parameters())
        self.net=copy.deepcopy(original).eval().requires_grad_(False);self.vm_root=None
        for name,value in [('original_head4_0',original.head4.block0.weight),('original_head4_1',original.head4.block1.weight),
            ('original_fusion4',original.smooth[0].weight[:,:64]),('anchor_head4_0',original.head3.block0.weight),
            ('anchor_head4_1',original.head3.block1.weight),('anchor_fusion4',original.smooth[0].weight[:,64:128])]:
            self.register_buffer(name,value.detach().clone().contiguous())
        with torch.no_grad():
            self.net.head4.block0.weight.copy_(self.anchor_head4_0);self.net.head4.block1.weight.copy_(self.anchor_head4_1)
        # This independently owned slice avoids Adam/weight decay changing the
        # other 192 frozen fusion channels. Same live computation as the VM probe.
        self.live_fusion4=nn.Parameter(self.anchor_fusion4.clone(),requires_grad=False)
        assert sum(p.numel() for p in self.learning_parameters())==147456
        assert all(not m.training for m in self.net.modules())

    def train(self,mode=True):return super().train(False)
    def learning_parameters(self):return [self.net.head4.block0.weight,self.net.head4.block1.weight,self.live_fusion4]
    def learning_names(self):return ['net.head4.block0.weight','net.head4.block1.weight','live_fusion4']
    def enable_vm_learning(self,root):
        require_vm(root,self.live_fusion4)
        import urllib.request
        q=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',headers={'Metadata-Flavor':'Google'})
        with urllib.request.urlopen(q,timeout=3) as response:assert response.read().decode().rstrip().endswith('/g2-standard-4')
        self.vm_root=Path(root).resolve()
        for p in self.learning_parameters():p.requires_grad_(True)
        assert len([p for p in self.parameters() if p.requires_grad])==3

    @staticmethod
    def fixed_head(x,w0,w1):return F.relu(F.conv2d(F.relu(F.conv2d(x,w0,padding=1)),w1,padding=1))
    def forward(self,image):
        if torch.is_grad_enabled():require_vm(self.vm_root,self.live_fusion4)
        assert image.dtype==torch.float32 and image.shape[1:]==(3,256,256) and not image.requires_grad
        assert torch.isfinite(image).all() and ((image>=0)&(image<=1)).all()
        assert not self.training and all(not m.training for m in self.net.modules())
        x=image*2-1;map0,map1,map2,map3,map4=self.net.fpn(x)
        base4=F.interpolate(self.fixed_head(map4,self.original_head4_0,self.original_head4_1),scale_factor=8,mode='nearest')
        live4=F.interpolate(self.net.head4(map4),scale_factor=8,mode='nearest')
        anchor4=F.interpolate(self.fixed_head(map4,self.anchor_head4_0,self.anchor_head4_1),scale_factor=8,mode='nearest')
        map3=F.interpolate(self.net.head3(map3),scale_factor=4,mode='nearest')
        map2=F.interpolate(self.net.head2(map2),scale_factor=2,mode='nearest');map1=self.net.head1(map1)
        basew=torch.cat([self.original_fusion4,self.net.smooth[0].weight[:,64:]],dim=1)
        base=F.conv2d(torch.cat([base4,map3,map2,map1],dim=1),basew,self.net.smooth[0].bias,padding=1)
        extra=F.conv2d(live4,self.live_fusion4,padding=1)-F.conv2d(anchor4,self.anchor_fusion4,padding=1)
        y=self.net.smooth[2](self.net.smooth[1](base+extra))
        y=F.interpolate(y,scale_factor=2,mode='nearest');y=self.net.smooth2(y+map0)
        y=F.interpolate(y,scale_factor=2,mode='nearest')
        return ((torch.tanh(self.net.final(y))+x).clamp(-1,1)+1)/2
