"""Isolated current DGP; exact-output initializer, controlled decoder partitions."""
import copy
from pathlib import Path
import platform
import sys
import torch
from torch import nn
from torch.nn import functional as F

NAME = 'cctv_dgp_multiscale_calibration_vm_v1'
DEEP = ['net.head4.block0.weight', 'net.head4.block1.weight', 'live_fusion4']
FULL = DEEP + [f'net.head{i}.block{j}.weight' for i in [1, 2, 3] for j in [0, 1]] + [
    'live_fusion_fine', 'net.smooth.0.bias', 'net.smooth2.0.weight', 'net.smooth2.0.bias',
    'net.final.weight', 'net.final.bias']


def require_vm(root, value):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert Path(root).resolve() == (Path.home() / 'forensic-dgp' / NAME).resolve()
    assert value.device.type == 'cuda' and torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L4'


class MultiscaleCalibrationDGP(nn.Module):
    def __init__(self, original):
        super().__init__()
        assert not original.training and all(not p.requires_grad for p in original.parameters())
        self.net = copy.deepcopy(original).eval().requires_grad_(False)
        self.vm_root = None
        for name, value in [
            ('original_head4_0', original.head4.block0.weight), ('original_head4_1', original.head4.block1.weight),
            ('original_fusion4', original.smooth[0].weight[:, :64]), ('anchor_head4_0', original.head3.block0.weight),
            ('anchor_head4_1', original.head3.block1.weight), ('anchor_fusion4', original.smooth[0].weight[:, 64:128])]:
            self.register_buffer(name, value.detach().clone().contiguous())
        with torch.no_grad():
            self.net.head4.block0.weight.copy_(self.anchor_head4_0)
            self.net.head4.block1.weight.copy_(self.anchor_head4_1)
        self.live_fusion4 = nn.Parameter(self.anchor_fusion4.clone(), requires_grad=False)
        self.live_fusion_fine = nn.Parameter(original.smooth[0].weight[:, 64:].detach().clone().contiguous(), requires_grad=False)
        assert sum(p.numel() for p in self.selected_parameters('deep3')) == 147456
        assert sum(p.numel() for p in self.selected_parameters('decoder15')) == 609219

    def train(self, mode=True):
        return super().train(False)

    def selected_names(self, partition):
        assert partition in ['deep3', 'decoder15']
        return DEEP if partition == 'deep3' else FULL

    def selected_parameters(self, partition):
        values = dict(self.named_parameters())
        return [values[n] for n in self.selected_names(partition)]

    def enable_vm_learning(self, root, partition):
        require_vm(root, self.live_fusion4)
        import urllib.request
        request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
                                         headers={'Metadata-Flavor': 'Google'})
        with urllib.request.urlopen(request, timeout=3) as response:
            assert response.read().decode().rstrip().endswith('/g2-standard-4')
        self.vm_root = Path(root).resolve()
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.selected_parameters(partition):
            p.requires_grad_(True)
        assert [n for n, p in self.named_parameters() if p.requires_grad] == [n for n, _ in self.named_parameters() if n in self.selected_names(partition)]
        assert len([p for p in self.parameters() if p.requires_grad]) == len(self.selected_names(partition))

    @staticmethod
    def fixed_head(x, w0, w1):
        return F.relu(F.conv2d(F.relu(F.conv2d(x, w0, padding=1)), w1, padding=1))

    def forward(self, image):
        if torch.is_grad_enabled():
            require_vm(self.vm_root, self.live_fusion4)
        assert image.dtype == torch.float32 and image.shape[1:] == (3, 256, 256) and not image.requires_grad
        assert torch.isfinite(image).all() and ((image >= 0) & (image <= 1)).all()
        assert not self.training and all(not m.training for m in self.net.modules())
        x = image * 2 - 1
        map0, map1, map2, map3, map4 = self.net.fpn(x)
        base4 = F.interpolate(self.fixed_head(map4, self.original_head4_0, self.original_head4_1), scale_factor=8, mode='nearest')
        live4 = F.interpolate(self.net.head4(map4), scale_factor=8, mode='nearest')
        anchor4 = F.interpolate(self.fixed_head(map4, self.anchor_head4_0, self.anchor_head4_1), scale_factor=8, mode='nearest')
        map3 = F.interpolate(self.net.head3(map3), scale_factor=4, mode='nearest')
        map2 = F.interpolate(self.net.head2(map2), scale_factor=2, mode='nearest')
        map1 = self.net.head1(map1)
        basew = torch.cat([self.original_fusion4, self.live_fusion_fine], dim=1)
        base = F.conv2d(torch.cat([base4, map3, map2, map1], dim=1), basew, self.net.smooth[0].bias, padding=1)
        extra = F.conv2d(live4, self.live_fusion4, padding=1) - F.conv2d(anchor4, self.anchor_fusion4, padding=1)
        y = self.net.smooth[2](self.net.smooth[1](base + extra))
        y = F.interpolate(y, scale_factor=2, mode='nearest')
        y = self.net.smooth2(y + map0)
        y = F.interpolate(y, scale_factor=2, mode='nearest')
        return ((torch.tanh(self.net.final(y)) + x).clamp(-1, 1) + 1) / 2
