"""Reset code-only component for a broader diagnostic, not an adopted restorer."""
from pathlib import Path

import torch
from torch import nn

from dgp_face_code_conditioner_v11 import DGPFaceCodePrior, check_rgb
from dgp_direct_face_code_v14 import render_codes


class BroaderCodeConditioner(nn.Module):
    """Direct residual code classifier; no mean/std branch or rendering statistics."""

    def __init__(self):
        super().__init__()
        layers=[]
        for a,b in zip([6,32,64,128],[32,64,128,256]):
            layers.extend([nn.Conv2d(a,b,3,stride=2,padding=1),nn.SiLU()])
        self.image_features=nn.Sequential(*layers)
        self.fusion=nn.Sequential(nn.Conv2d(512,256,3,padding=1),nn.SiLU(),
                                  nn.Conv2d(256,256,3,padding=1),nn.SiLU())
        self.code_projection=nn.Conv2d(256,1024,1)
        nn.init.zeros_(self.code_projection.weight);nn.init.zeros_(self.code_projection.bias)
        self._training_root=None
        nn.Module.train(self,False);self.requires_grad_(False)

    def train(self,mode=True):
        if mode:raise RuntimeError('Use enable_vm_training(root); local training forbidden')
        nn.Module.train(self,False);self.requires_grad_(False);self._training_root=None
        return self

    def enable_vm_training(self,root):
        from cctv_dgp_targets_v6 import require_vm
        require_vm(root)
        self._training_root=Path(root).resolve()
        nn.Module.train(self,True);self.requires_grad_(True)
        return self

    def forward(self,image,dgp_image,original_features,base_logits):
        if self.training or any(p.requires_grad for p in self.parameters()):
            if self._training_root is None:raise RuntimeError('Training graph requires explicit VM authorization')
            from cctv_dgp_targets_v6 import require_vm
            require_vm(self._training_root)
        check_rgb(image);check_rgb(dgp_image);n=len(image)
        if (original_features.shape!=(n,256,16,16) or base_logits.shape!=(n,256,1024) or
                any(x.dtype!=torch.float32 or x.requires_grad or not torch.isfinite(x).all()
                    for x in [image,dgp_image,original_features,base_logits])):
            raise ValueError('Require detached finite float32 input/features/logits')
        x=self.image_features(torch.cat([image,dgp_image],1))
        x=self.fusion(torch.cat([original_features,x],1))
        return base_logits+self.code_projection(x).flatten(2).transpose(1,2)


class DGPBroaderCodePrior(nn.Module):
    """Experimental own classifier plus frozen DGP/declared prior; fresh-image path."""

    def __init__(self,dgp_net,prior_net):
        super().__init__();self.core=DGPFaceCodePrior(dgp_net,prior_net)
        self.conditioner=BroaderCodeConditioner();nn.Module.train(self,False)

    def train(self,mode=True):
        if mode:raise RuntimeError('Enable only the conditioner on the existing L4 VM')
        nn.Module.train(self,False);return self

    @torch.no_grad()
    def frozen_inputs(self,image):
        check_rgb(image);dgp=self.core.dgp(image).clamp(0,1)
        features,_=self.core.encode(image)
        return dgp,features,self.core.logits(features)

    @torch.inference_mode()
    def forward(self,image):
        dgp,features,base=self.frozen_inputs(image)
        logits=self.conditioner(image,dgp,features,base)
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite face codes')
        return render_codes(self.core.prior,logits.argmax(2))
