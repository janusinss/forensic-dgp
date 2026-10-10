"""Separate current-DGP copy. Derivatives and trial assignments are VM-only."""
import copy
from pathlib import Path
import platform
import sys
import torch
from torch import nn

NAME = 'cctv_dgp_original_loss_balance_v1_vm'


def require_vm(root, image):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert Path(root).resolve() == (Path.home()/'forensic-dgp'/NAME).resolve()
    assert image.device.type == 'cuda' and torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == 'NVIDIA L4'


class OriginalLossBalanceProbe(nn.Module):
    def __init__(self, original, layout):
        super().__init__()
        assert not original.training and all(not v.requires_grad for v in original.parameters())
        self.net = copy.deepcopy(original).eval().requires_grad_(False)
        self.layout = layout; self.vm_root = None
        unique = dict(self.net.named_parameters())
        self.selected = [(r['name'], unique[r['name']]) for r in layout]
        assert len({id(v) for _, v in self.selected}) == 158
        assert sum(v.numel() for _, v in self.selected) == 1996035
        for r, (_, v) in zip(layout, self.selected): assert list(v.shape) == r['shape']
        for a, b in zip(original.parameters(), self.net.parameters()):
            assert a is not b and a.data_ptr() != b.data_ptr() and torch.equal(a, b)
        for a, b in zip(original.buffers(), self.net.buffers()):
            assert a is not b and a.data_ptr() != b.data_ptr() and torch.equal(a, b)

    def train(self, mode=True): return super().train(False)

    def enable_finite_trials(self, root):
        require_vm(root, self.selected[0][1]); self.vm_root = Path(root).resolve()
        for _, v in self.selected: v.requires_grad_(False)

    def vector(self):
        return torch.cat([v.detach().reshape(-1).cpu() for _, v in self.selected]).numpy().copy()

    def assign_trial(self, vector):
        require_vm(self.vm_root, self.selected[0][1])
        assert vector.dtype.name == 'float32' and vector.shape == (1996035,)
        with torch.no_grad():
            for r, (_, v) in zip(self.layout, self.selected):
                v.copy_(torch.from_numpy(vector[r['start']:r['end']].reshape(r['shape'])).to(v.device))

    def forward(self, image, support, baseline):
        assert image.dtype == torch.float32 and image.shape[1:] == (3, 256, 256)
        assert not image.requires_grad and not baseline.requires_grad and not support.requires_grad
        assert torch.isfinite(image).all() and ((image >= 0) & (image <= 1)).all()
        assert baseline.shape == image.shape and baseline.dtype == image.dtype
        assert support.shape == (len(image), 1, 256, 256) and ((support == 0) | (support == 1)).all()
        assert (support.sum((1, 2, 3)) > 0).all()
        assert not self.net.training and all(not m.training for m in self.net.modules())
        if torch.is_grad_enabled(): require_vm(self.vm_root, image)
        raw = self.net(image)
        delta = raw-baseline
        mean = (delta*support).sum((2, 3), keepdim=True)/support.sum((2, 3), keepdim=True)
        result = torch.where(support.bool(), (baseline+delta-mean).clamp(0, 1), image)
        assert torch.isfinite(result).all()
        return result
