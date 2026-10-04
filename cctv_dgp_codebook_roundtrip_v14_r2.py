"""Check the actual float32 codebook invariant, not exact decoder identity."""
import torch
from third_party.codeformer.codeformer_arch import calc_mean_std


@torch.no_grad()
def check_codebook_statistics_roundtrip(features, mean, std):
    if features.dtype != torch.float32 or features.ndim != 4:
        raise ValueError('Expected float32 codebook features')
    cm, cs = calc_mean_std(features)
    if not torch.equal(mean, cm.flatten(1)) or not torch.equal(std, cs.flatten(1)):
        raise ValueError('Target statistics do not exactly match the codebook')
    normalized = (features - cm) / cs * std[:, :, None, None] + mean[:, :, None, None]
    difference = (normalized - features).abs().max().item()
    scale = max(1.0, features.abs().max().item())
    # Four float operations; eight float32 epsilons is a conservative roundtrip
    # bound at the latent value scale. It is not an image-quality tolerance.
    bound = 8 * torch.finfo(torch.float32).eps * scale
    if not torch.isfinite(normalized).all() or difference > bound:
        raise ValueError('Codebook normalization roundtrip exceeds floating-point bound')
    return normalized, {'exact_codebook_statistics': True, 'maximum_latent_difference': difference,
                        'latent_scale': scale, 'latent_float32_bound': bound}
