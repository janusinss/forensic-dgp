"""Frozen local covering proposals; a proposal is never a reviewed removal mask.

The segmenter is extracted, without learning, from the project's retained epoch2
checkpoint. Its generator is not included. No Torch or model is loaded on import.
"""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import numpy as np

from .managed_paths import data_path

ROOT = Path(__file__).resolve().parents[1]
DETECTOR_PATH = data_path(ROOT, 'models', 'completion') / 'covering_detector_segmenter_v1.pth'
DETECTOR_BYTES = 2681549
DETECTOR_SHA256 = 'ccc22d524073368bc4f5d6b6d2136c90d0e2a117083e4c75fa98d42d7bfff6cf'
DETECTOR_STATE = '888250f7a54aa95dac5789943bb09a8c3763ca9e986940421ed9aa400986dd0f'
PARENT_SHA256 = 'c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93'
THRESHOLD = 0.5
PROPOSAL_MARGIN = 3


def need(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def provenance():
    return {
        'backend': 'project-epoch2-covering-segmenter',
        'weights_sha256': DETECTOR_SHA256, 'weights_bytes': DETECTOR_BYTES,
        'source_checkpoint_sha256': PARENT_SHA256,
        'logical_state_sha256': DETECTOR_STATE,
        'source_epoch': 2, 'source_width': 24, 'input_size': [256, 256],
        'tensor_count': 50, 'device': 'cpu', 'precision': 'float32',
        'threshold': THRESHOLD, 'proposal_margin_pixels_at_256': PROPOSAL_MARGIN,
        'proposal_margin_policy': 'Fixed 3-pixel square dilation, clipped to observed support; editable before confirmation.',
        'automatic_covering_quality_qualified': False,
        'assisted_quality_qualified': False,
        'mask_review_required': True, 'automatic_confirmation': False,
        'covering_type_classification': False,
        'training_optimizer_VM_calls': 0,
        'warning': 'Development proposal only. It can miss coverings or mark clear glasses, hair or visible features. Review, paint and erase before completion.',
    }


def build_segmenter(torch):
    """Exact GatedBlock/GatedUNet segmenter operations from completion.py.

    Only namespace/ownership changed. No completion generator is constructed.
    """
    nn = torch.nn
    F = torch.nn.functional

    class GatedBlock(nn.Module):
        def __init__(self, inputs, outputs, stride=1):
            super().__init__()
            self.features = nn.Conv2d(inputs,outputs,3,stride,1)
            self.gate = nn.Conv2d(inputs,outputs,3,stride,1)
            self.norm = nn.GroupNorm(4,outputs)

        def forward(self,x):
            return F.silu(self.norm(self.features(x)))*torch.sigmoid(self.gate(x))

    class GatedUNet(nn.Module):
        def __init__(self, inputs, outputs, width):
            super().__init__()
            self.enc1 = GatedBlock(inputs,width)
            self.enc2 = GatedBlock(width,width*2,2)
            self.enc3 = GatedBlock(width*2,width*4,2)
            self.middle = nn.Sequential(GatedBlock(width*4,width*4,2),GatedBlock(width*4,width*4))
            self.dec3 = GatedBlock(width*8,width*2)
            self.dec2 = GatedBlock(width*4,width)
            self.dec1 = GatedBlock(width*2,width)
            self.head = nn.Conv2d(width,outputs,1)

        def forward(self,x):
            a = self.enc1(x)
            b = self.enc2(a)
            c = self.enc3(b)
            def up(t,skip):
                return torch.cat((F.interpolate(t,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),1)
            z = self.dec3(up(self.middle(c),c))
            z = self.dec2(up(z,b))
            return self.head(self.dec1(up(z,a)))

    return GatedUNet(3, 1, 24)


def logical_state(params):
    h = hashlib.sha256()
    for name, tensor in sorted(params.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def load_frozen_detector(path=DETECTOR_PATH):
    """Load only the exact admitted CPU segmenter; no inference occurs here."""
    path = Path(os.path.abspath(path))
    model_root = data_path(ROOT, 'models', 'completion')
    need((path.is_relative_to(ROOT) or path.is_relative_to(model_root)) and path.is_file() and not path.is_symlink(),
         'A local ordinary detector asset is required. Run install.bat.')
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
         and not getattr(info, 'st_file_attributes', 0) & 1024,
         'An ordinary detector file is required.')
    for ancestor in path.parents:
        info = ancestor.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024, 'Linked detector directories are refused.')
        if ancestor == ROOT:
            break
    need(path.stat().st_size == DETECTOR_BYTES and file_sha(path) == DETECTOR_SHA256,
         'Covering detector size/SHA-256 differs. Existing files were preserved.')
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    need(type(state) is dict and set(state) == {'format', 'width', 'size', 'source_checkpoint_sha256', 'segmenter'},
         'Unexpected detector checkpoint schema.')
    need(state['format'] == 'covering-segmenter-portable-v1' and state['width'] == 24
         and state['size'] == 256 and state['source_checkpoint_sha256'] == PARENT_SHA256,
         'Detector source/configuration differs.')
    params = state['segmenter']
    need(type(params) is dict and len(params) == 50
         and all(type(k) is str and isinstance(v, torch.Tensor) and v.device.type == 'cpu'
                 and v.dtype == torch.float32 and torch.isfinite(v).all().item() for k, v in params.items()),
         'Expected 50 finite FP32 detector tensors.')
    need(logical_state(params) == DETECTOR_STATE, 'Detector logical state differs.')
    model = build_segmenter(torch)
    model.load_state_dict(params, strict=True)
    model = model.cpu().float().eval().requires_grad_(False)
    need(not model.training and not any(p.requires_grad for p in model.parameters()),
         'The detector must remain frozen for local inference.')
    need(path.stat().st_size == DETECTOR_BYTES and file_sha(path) == DETECTOR_SHA256,
         'Detector asset changed while loading.')
    return model


def proposal_from_probability(probability, observed):
    """Pure saved-probability processing; always leaves padding unselected.

    Return raw/proposed masks as uint8 0/255, plus a non-approval receipt. A 3px
    margin is offered for review; completion never adds this margin a second time.
    """
    need(isinstance(probability, np.ndarray) and probability.shape == (256, 256)
         and probability.dtype == np.float32 and np.isfinite(probability).all()
         and probability.min() >= 0 and probability.max() <= 1,
         'Expected finite FP32 256x256 detector probabilities in [0,1].')
    need(isinstance(observed, np.ndarray) and observed.shape == (256, 256)
         and observed.dtype == np.bool_ and observed.any(),
         'Expected nonempty boolean 256x256 observed support.')
    thresholded = probability >= np.float32(THRESHOLD)
    raw = thresholded & observed
    padded = np.pad(raw, PROPOSAL_MARGIN, mode='constant', constant_values=False)
    dilated = np.zeros_like(raw)
    for dy in range(2 * PROPOSAL_MARGIN + 1):
        for dx in range(2 * PROPOSAL_MARGIN + 1):
            dilated |= padded[dy:dy + 256, dx:dx + 256]
    proposed = dilated & observed
    metadata = {
        **provenance(), 'raw_mask_pixels': int(raw.sum()),
        'proposed_mask_pixels': int(proposed.sum()),
        'proposed_margin_pixels_added': int((proposed & ~raw).sum()),
        'threshold_positive_padding_pixels_discarded': int((thresholded & ~observed).sum()),
        'observed_pixels': int(observed.sum()), 'proposal_empty': not bool(proposed.any()),
        'empty_proposal_does_not_prove_uncovered': True,
        'padding_selected': False, 'mask_reviewed': False,
    }
    return (raw.astype(np.uint8) * 255, proposed.astype(np.uint8) * 255, metadata)
