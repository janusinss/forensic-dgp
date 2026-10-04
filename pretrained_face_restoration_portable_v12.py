"""Python3.10-compatible loader; frozen historical CodeFormer adapter unchanged."""
from pathlib import Path

import torch

from dgp_face_restoration import sha
from pretrained_face_restoration import CodeFormerRestoration, REVISION, WEIGHTS_URL, WEIGHTS_SHA256


def load_face_restorer(path, device='cpu'):
    path = Path(path)
    digest = sha(path)  # Streaming SHA256 works on the existing VM's Python3.10.
    if digest != WEIGHTS_SHA256:
        raise ValueError('CodeFormer restoration checkpoint fingerprint differs')
    from third_party.codeformer.codeformer_arch import CodeFormer
    state = torch.load(path, map_location='cpu', weights_only=True)
    net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                     connect_list=['32', '64', '128', '256'])
    net.load_state_dict(state['params_ema'], strict=True)
    if not all(torch.isfinite(value).all() for value in net.state_dict().values()):
        raise ValueError('Non-finite restoration weights')
    return CodeFormerRestoration(net).to(device).eval(), {
        'backend': 'codeformer-restoration', 'source_revision': REVISION,
        'weights_url': WEIGHTS_URL, 'weights_sha256': digest,
        'checksum_scope': 'Observed acquisition fingerprint, not a publisher-supplied checksum',
        'codebook_size': 1024, 'input_policy': 'RGB bilinear512 normalized [-1,1]',
        'adain': True, 'output_policy': 'Bilinear back to input dimensions',
        'license': 'S-Lab License 1.0; retained in third_party/codeformer/LICENSE',
        'loader_policy': 'Explicit streaming SHA256 for Python3.10; original frozen architecture/adapter/weights unchanged'}
