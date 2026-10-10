"""Direct correction training terms. Clean labels never enter model.forward."""
import torch
from torch.nn import functional as F
from cctv_dgp_residual_supervision_v1 import correction_terms, projected_teacher
from cctv_dgp_residual_epochs_v42_contract import LOSS_WEIGHTS as W


def reconstruction_terms(b, components, normalizers):
    teacher, _ = projected_teacher(b['base'], b['target'], b['mask'].bool(), b['clear'])
    values = correction_terms(components['centered_correction'], teacher,
                              b['mask'].bool(), b['feature'].bool())
    multiplier = torch.where(b['clear'], W['clear_correction_multiplier'],
                             W['degraded_correction_multiplier'])
    return {key: multiplier * W[key] * value / normalizers[key]
            for key, value in values.items()}


def objective_terms(b, components, identity, ssim, normalizers):
    terms = reconstruction_terms(b, components, normalizers)
    pred = components['result']; n = len(pred)
    vectors = identity.embedding(torch.cat((b['base'].detach(), pred), 0),
        torch.cat((b['mask'], b['mask']), 0), torch.cat((b['grid'], b['grid']), 0))
    before = (vectors[:n].detach() * b['truth'].detach()).sum(1)
    after = (vectors[n:] * b['truth'].detach()).sum(1)
    denominator = 3 * b['mask'].sum((1, 2, 3))
    mse = lambda image: ((image-b['target']).square()*b['mask']).sum((1, 2, 3))/denominator
    base_mse, current_mse = mse(b['base']).detach(), mse(pred)
    terms.update({
        'identity_absolute': W['identity_absolute'] * (~b['clear']) * (1-after),
        'identity_regression': W['identity_regression'] * F.relu(before-after),
        'pixel_regression': W['pixel_regression'] * F.relu((current_mse-base_mse)/base_mse.clamp_min(1e-5)),
        'SSIM_regression': W['SSIM_regression'] * F.relu(
            ssim(b['base'], b['target'], b['valid7']).detach()-ssim(pred, b['target'], b['valid7']))})
    assert all(value.shape == (n,) and bool(torch.isfinite(value).all()) for value in terms.values())
    return terms
