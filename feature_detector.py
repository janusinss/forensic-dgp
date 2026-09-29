"""Experimental prompt-free head for frozen image features; not an app backend."""
import torch
from torch import nn
from torch.nn import functional as F


class FrozenFeatureHead(nn.Module):
    def __init__(self, channels=256, hidden=64):
        super().__init__()
        self.channels = channels
        self.head = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))

    def forward(self, features, output_size):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError('Expected NCHW features with configured channel count')
        # Detach is an explicit boundary: this diagnostic never tunes the encoder.
        x = features.detach().float().permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.channels,)).permute(0, 3, 1, 2)
        return F.interpolate(self.head(x), size=output_size, mode='bilinear', align_corners=False)
