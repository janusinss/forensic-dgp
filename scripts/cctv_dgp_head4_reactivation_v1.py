"""Isolated original-DGP repair; initial output anchored to the current model.

Reseed only the underflowed deepest head and its fusion slice from our adjacent
trained head3. No external prior, fitting, unanchored reset or display processing.
All learning entry points require the existing Linux L4 VM.
"""
import copy
from pathlib import Path
import platform
import sys
import torch
from torch import nn
from torch.nn import functional as F

NAME = 'cctv_dgp_head4_reactivation_vm_v1'


def require_vm(root, value):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Manual existing VM only'
    assert Path(root).resolve() == (Path.home()/'forensic-dgp'/NAME).resolve()
    assert value.device.type == 'cuda' and torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == 'NVIDIA L4'


class ReactivatedDGP(nn.Module):
    def __init__(self, original):
        super().__init__()
        assert not original.training and all(not p.requires_grad for p in original.parameters())
        self.net = copy.deepcopy(original).eval().requires_grad_(False)
        self.vm_root = None
        assert self.net.smooth[0].weight.shape == (64,256,3,3)
        assert self.net.head3.block0.weight.shape == self.net.head4.block0.weight.shape == (64,128,3,3)
        assert self.net.head3.block1.weight.shape == self.net.head4.block1.weight.shape == (64,64,3,3)
        # Frozen base keeps the original function; frozen seed anchor cancels the
        # live seed at initialization. Input derivatives through both remain real.
        for name,value in [
            ('original_head4_0',original.head4.block0.weight),
            ('original_head4_1',original.head4.block1.weight),
            ('original_fusion4',original.smooth[0].weight[:,:64]),
            ('anchor_head4_0',original.head3.block0.weight),
            ('anchor_head4_1',original.head3.block1.weight),
            ('anchor_fusion4',original.smooth[0].weight[:,64:128])]:
            self.register_buffer(name,value.detach().clone().contiguous())
        with torch.no_grad():
            self.net.head4.block0.weight.copy_(self.anchor_head4_0)
            self.net.head4.block1.weight.copy_(self.anchor_head4_1)
            self.net.smooth[0].weight[:,:64].copy_(self.anchor_fusion4)
        assert float(self.original_head4_0.abs().max()) < 1e-38
        assert float(self.original_head4_1.abs().max()) < 1e-38
        assert float(self.original_fusion4.abs().max()) < 1e-38
        assert all(float(getattr(self,n).norm()) > .1 for n in ['anchor_head4_0','anchor_head4_1','anchor_fusion4'])
        original_params = dict(original.named_parameters())
        for name,value in self.net.named_parameters():
            assert value.data_ptr() != original_params[name].data_ptr()
        # The FPN forward uses features0:16 through enc0..enc4. Its retained
        # classifier-tail features16:19 are registered but unreachable here.
        self.active_names = [name for name,_ in self.net.named_parameters()
            if not name.startswith(('fpn.features.16.','fpn.features.17.','fpn.features.18.'))]
        assert len(self.active_names) == 160
        assert sum(dict(self.net.named_parameters())[name].numel() for name in self.active_names) == 2106627

    def train(self, mode=True): return super().train(False)

    def enable_vm_gradients(self, root):
        require_vm(root,self.net.head4.block0.weight)
        import urllib.request
        request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
            headers={'Metadata-Flavor':'Google'})
        with urllib.request.urlopen(request,timeout=3) as response:
            assert response.read().decode().rstrip().endswith('/g2-standard-4')
        self.vm_root = Path(root).resolve()
        for name,value in self.net.named_parameters(): value.requires_grad_(name in self.active_names)

    @staticmethod
    def fixed_head(value,w0,w1):
        return F.relu(F.conv2d(F.relu(F.conv2d(value,w0,padding=1)),w1,padding=1))

    def forward(self, image):
        if torch.is_grad_enabled(): require_vm(self.vm_root,self.net.head4.block0.weight)
        assert image.dtype == torch.float32 and image.shape[1:] == (3,256,256)
        assert not image.requires_grad and torch.isfinite(image).all() and ((image>=0)&(image<=1)).all()
        assert not self.training and all(not m.training for m in self.net.modules())
        normalized = image*2.-1.
        map0,map1,map2,map3,map4 = self.net.fpn(normalized)
        base4 = self.fixed_head(map4,self.original_head4_0,self.original_head4_1)
        live4 = self.net.head4(map4)
        anchored4 = self.fixed_head(map4,self.anchor_head4_0,self.anchor_head4_1)
        base4 = F.interpolate(base4,scale_factor=8,mode='nearest')
        live4 = F.interpolate(live4,scale_factor=8,mode='nearest')
        anchored4 = F.interpolate(anchored4,scale_factor=8,mode='nearest')
        map3 = F.interpolate(self.net.head3(map3),scale_factor=4,mode='nearest')
        map2 = F.interpolate(self.net.head2(map2),scale_factor=2,mode='nearest')
        map1 = self.net.head1(map1)
        # This base convolution retains original deep weights and current other
        # fusion weights. The live deep slice gets normal scale and true gradients.
        base_weight = torch.cat([self.original_fusion4,self.net.smooth[0].weight[:,64:]],dim=1)
        base = F.conv2d(torch.cat([base4,map3,map2,map1],dim=1),base_weight,self.net.smooth[0].bias,padding=1)
        live = F.conv2d(live4,self.net.smooth[0].weight[:,:64].contiguous(),padding=1)
        anchored = F.conv2d(anchored4,self.anchor_fusion4,padding=1)
        smoothed = self.net.smooth[2](self.net.smooth[1](base+(live-anchored)))
        smoothed = F.interpolate(smoothed,scale_factor=2,mode='nearest')
        smoothed = self.net.smooth2(smoothed+map0)
        smoothed = F.interpolate(smoothed,scale_factor=2,mode='nearest')
        residual = (torch.tanh(self.net.final(smoothed))+normalized).clamp(-1,1)
        return (residual+1)/2
