"""Verify actual converted ArcFace outputs and restoration-input gradients."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import hashlib
import numpy as np
import onnxruntime as ort
import torch
from models.identity_loss import ArcFaceIdentityLoss, ARCFACE_TEMPLATE


def check(path, device):
    path = Path(path).expanduser()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    loss_fn = ArcFaceIdentityLoss(path,device=device)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    reference = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    inputs = torch.rand(2,3,112,112)*2-1
    expected = reference.run(None,{reference.get_inputs()[0].name:inputs.numpy()})[0]
    with torch.no_grad():
        actual = loss_fn.encoder(inputs.to(device)).cpu().numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-4)
    landmarks = torch.zeros(2,68,2,device=device)
    for section,point in [(slice(36,42),0),(slice(42,48),1),(30,2),(48,3),(54,4)]:
        landmarks[:,section] = ARCFACE_TEMPLATE[point].to(device)
    generated = torch.rand(2,3,112,112,device=device,requires_grad=True)
    target = torch.rand_like(generated)
    loss,count = loss_fn(generated,target,landmarks)
    loss.backward()
    assert count == 2 and torch.isfinite(generated.grad).all() and generated.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in loss_fn.encoder.parameters())
    result = {'device':device,'max_embedding_error':float(np.abs(actual-expected).max()),
              'gradient_mean':generated.grad.abs().mean().item(),'valid_pairs':count,
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='~/.insightface/models/buffalo_l/w600k_r50.onnx')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    check(args.model,args.device)
