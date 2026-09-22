"""Differentiable ArcFace cosine loss with shared reference-landmark alignment.

The recognizer is frozen; gradients flow through its input into the restorer.
Preprocessing follows InsightFace RGB [-1, 1] and its 112px five-point template.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ARCFACE_TEMPLATE = torch.tensor([[38.2946, 51.6963], [73.5318, 51.5014],
                                [56.0252, 71.7366], [41.5493, 92.3655],
                                [70.7299, 92.2041]], dtype=torch.float32)


class ArcFaceIdentityLoss(nn.Module):
    def __init__(self, model_path=None, encoder=None, device='cpu'):
        super().__init__()
        if encoder is None:
            import onnx
            from onnx2torch import convert
            path = Path(model_path or '~/.insightface/models/buffalo_l/w600k_r50.onnx').expanduser()
            if not path.is_file():
                raise FileNotFoundError(f'ArcFace weights missing: {path}')
            graph = onnx.load(str(path))
            shape = graph.graph.input[0].type.tensor_type.shape.dim
            if [d.dim_value for d in shape[1:]] != [3, 112, 112]:
                raise ValueError('Expected RGB ArcFace input with shape N x 3 x 112 x 112')
            # Passing ModelProto avoids a temporary write beside the ONNX file.
            encoder = convert(graph)
        self.encoder = encoder.to(device).eval().requires_grad_(False)
        self.register_buffer('template', ARCFACE_TEMPLATE.clone().to(device))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def alignment_grid(self, landmarks, height, width):
        """Fit reference->ArcFace similarity transform, then inverse-sample pixels."""
        points = torch.stack([landmarks[:,36:42].mean(1), landmarks[:,42:48].mean(1),
                              landmarks[:,30], landmarks[:,48], landmarks[:,54]], dim=1).detach().float()
        valid = (torch.isfinite(points).all(dim=(1,2)) &
                 ((points[:,0]-points[:,1]).norm(dim=1) >= 8) &
                 (points[:,:,0].amin(1) >= 0) & (points[:,:,0].amax(1) < width) &
                 (points[:,:,1].amin(1) >= 0) & (points[:,:,1].amax(1) < height))
        p = points[valid]
        if not len(p):
            return points.new_empty((0,112,112,2)), valid
        x,y = p.unbind(-1)
        ones, zeros = torch.ones_like(x), torch.zeros_like(x)
        a = torch.stack([torch.stack([x,-y,ones,zeros],-1),
                         torch.stack([y,x,zeros,ones],-1)], dim=2).reshape(-1,10,4)
        target = self.template.to(p).reshape(1,10,1).expand(len(p),-1,-1)
        fit = torch.linalg.lstsq(a, target).solution.squeeze(-1)
        scale, rot, tx, ty = fit.unbind(-1)
        denom = (scale.square()+rot.square()).clamp_min(1e-8)
        yy,xx = torch.meshgrid(torch.arange(112,device=p.device,dtype=p.dtype),
                               torch.arange(112,device=p.device,dtype=p.dtype),indexing='ij')
        u,v = xx[None]-tx[:,None,None], yy[None]-ty[:,None,None]
        src_x = (scale[:,None,None]*u+rot[:,None,None]*v)/denom[:,None,None]
        src_y = (-rot[:,None,None]*u+scale[:,None,None]*v)/denom[:,None,None]
        grid = torch.stack([2*(src_x+.5)/width-1, 2*(src_y+.5)/height-1],dim=-1)
        return grid, valid

    def embeddings(self, images, grid, valid):
        crop = F.grid_sample(images[valid].float(), grid, mode='bilinear',
                             padding_mode='zeros', align_corners=False)
        return F.normalize(self.encoder(crop*2-1).float(), dim=1)

    def similarities(self, generated, target, landmarks):
        grid, valid = self.alignment_grid(landmarks, *generated.shape[-2:])
        if not valid.any():
            return generated.new_empty((0,)), valid
        gen = self.embeddings(generated, grid, valid)
        with torch.no_grad():
            ref = self.embeddings(target, grid, valid)
        return (gen*ref).sum(1).clamp(-1,1), valid

    def forward(self, generated, target, landmarks):
        similarity, valid = self.similarities(generated, target, landmarks)
        count = int(valid.sum().item())
        if not count:
            return generated.sum()*0, 0
        return (1-similarity).mean(), count
