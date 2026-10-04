"""Keep frozen InstanceNorm statistics bitwise stable on older PyTorch kernels.

The 2.9 kernel averages and writes back repeated statistics even in evaluation.
Pass disposable copies to that kernel while retaining its original output and
gradient path. Parameters, checkpoint keys and normalization semantics remain.
"""
from types import MethodType

from torch import nn
from torch.nn import functional as F


def _frozen_apply(module, value):
    if module.training or not module.track_running_stats:
        raise RuntimeError("CCTV pilot requires frozen evaluation InstanceNorm")
    if module.running_mean is None or module.running_var is None:
        raise RuntimeError("Frozen InstanceNorm statistics are missing")
    return F.instance_norm(
        value, module.running_mean.clone(), module.running_var.clone(),
        module.weight, module.bias, False,
        module.momentum if module.momentum is not None else 0.0, module.eps
    )


def install_frozen_instance_norm(model):
    """Install an evaluation-only adapter without registering additional state."""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.InstanceNorm2d):
            if not module.track_running_stats:
                raise RuntimeError("CCTV pilot requires stored InstanceNorm statistics")
            module._apply_instance_norm = MethodType(_frozen_apply, module)
            count += 1
    return count
