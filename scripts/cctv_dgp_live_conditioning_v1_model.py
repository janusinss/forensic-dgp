"""Untrained anchored feature path; construction/forwards require manual L4.

The frozen reference is a function of the SAME observed input, not a clean
target. Nonzero channel projections are a declared deterministic initializer.
At initialization live/reference outputs must cancel exactly on the VM. The
reference remains fixed if learning is later justified. This is not a display
filter, a pretrained restorer, or evidence of recovered facial identity.
"""
import copy
import os
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn

NAME = 'cctv_dgp_live_conditioning_v1_vm'


def require_manual(root):
    assert platform.system() == 'Linux' and platform.node().split('.')[0]=='forensic-dgp-thesis'
    assert Path(root).resolve()==(Path.home()/'forensic-dgp'/NAME).resolve()
    assert Path.cwd().resolve()==Path(root).resolve() and os.environ.get('TMUX')
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0)=='NVIDIA L4'


class AnchoredBankDGP(nn.Module):
    def __init__(self, live, readout_path, root):
        require_manual(root)
        super().__init__()
        assert not live.training and all(not v.requires_grad for v in live.parameters())
        self.root=Path(root).resolve()
        self.live=live
        with np.load(readout_path,allow_pickle=False) as archive:
            assert set(archive.files)=={f'f{size}_{term}' for size in [16,32,64,128,256] for term in ['weight','bias']}
            with torch.no_grad():
                for size,layer in self.live.fusions.items():
                    for term in ['weight','bias']:
                        value=archive[f'f{size}_{term}']
                        expected=getattr(layer,term)
                        assert value.dtype==np.float32 and value.shape==tuple(expected.shape) and np.isfinite(value).all()
                        expected.copy_(torch.from_numpy(value).to(expected.device))
        self.anchor_style_conditioner=copy.deepcopy(self.live.style_conditioner).eval().requires_grad_(False)
        self.anchor_spatial_conditioners=copy.deepcopy(self.live.spatial_conditioners).eval().requires_grad_(False)
        self.anchor_fusions=copy.deepcopy(self.live.fusions).eval().requires_grad_(False)
        self.branch_calls={'live':0,'anchor':0}
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    def learning_names(self):
        return ['live.'+name for name in self.live.learning_names()]

    def learning_parameters(self):
        values=dict(self.named_parameters())
        return [values[name] for name in self.learning_names()]

    def enable_vm_learning(self, parent):
        require_manual(self.root)
        self.live.enable_vm_learning(parent)
        assert len(self.learning_names())==37
        assert all(not value.requires_grad for name,value in self.named_parameters() if name not in self.learning_names())

    def anchor_features(self, maps):
        # Frozen copy of the old input/style/spatial conditioning, sharing only
        # the unchanged external generator. No reference-image inversion.
        _, map1, map2, map3, map4=maps
        prior=self.live.prior
        styles=self.live.mean_style[None,None]+self.anchor_style_conditioner(map4).reshape(-1,13,512)
        noises=[getattr(prior.noises,f'noise{i}') for i in range(13)]
        out=prior.constant_input(len(styles))
        out=prior.style_conv1(out,styles[:,0],noise=noises[0])
        conditioning={'8':map4,'16':map3,'32':map2}
        features={}
        for index in range(6):
            size=2**(index+3)
            out=prior.style_convs[2*index](out,styles[:,1+2*index],noise=noises[1+2*index])
            out=prior.style_convs[2*index+1](out,styles[:,2+2*index],noise=noises[2+2*index])
            if str(size) in conditioning:
                out=out+self.anchor_spatial_conditioners[str(size)](conditioning[str(size)])
            if size>=16: features[str(size)]=out
        assert set(features)=={'16','32','64','128','256'}
        return features

    def forward(self, image):
        require_manual(self.root)
        assert all(not module.training for module in self.modules())
        maps=self.live.current.observe(image)
        features=self.live.conditioned_features(maps)
        self.branch_calls['live']+=1
        with torch.no_grad():
            anchor=self.anchor_features(maps)
            reference={key:self.anchor_fusions[key](value) for key,value in anchor.items()}
        self.branch_calls['anchor']+=1
        extras={key:self.live.fusions[key](features[key])-reference[key] for key in features}
        return self.live.current.reconstruct(image,maps,extras)
