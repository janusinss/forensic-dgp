"""Versioned compact completion baseline. Masks use 1 for generated regions.

This model needs training; it is not a pretrained face prior. Detection and
completion have independent networks so predicted masks can be corrected.
"""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'dgp-completion-v1'
KINDS = ('none', 'lower', 'eyes', 'object', 'irregular')


def validate_mask(image, mask):
    if (image.ndim != 4 or image.shape[1] != 3 or
            mask.shape != (image.shape[0],1,*image.shape[2:])):
        raise ValueError('Expected RGB NCHW and matching N1HW mask')
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError('Mask must contain finite values in [0,1]')


def compose(visible, generated, alpha):
    validate_mask(visible, alpha)
    if generated.shape != visible.shape:
        raise ValueError('Generated and visible images must have equal shape')
    return visible*(1-alpha) + generated*alpha


def blend_mask(mask, radius=0):
    """Preserve visible pixels by default; external feathering is opt-in."""
    if radius < 0:
        raise ValueError('Blend radius must be nonnegative')
    if radius == 0:
        return mask
    soft = F.avg_pool2d(mask,2*radius+1,stride=1,padding=radius)
    return torch.maximum(mask,soft)


def region_error(prediction, target, mask, squared=False):
    validate_mask(prediction,mask)
    error = (prediction-target).abs()
    if squared:
        error = error.square()
    area = mask.sum((1,2,3))*prediction.shape[1]
    return (error*mask).sum((1,2,3))/area.clamp_min(1), area > 0


def synthetic_covering(rgb, seed, kind='lower'):
    """Procedural aligned-face masks, not photorealistic hands or accessories."""
    if kind not in KINDS:
        raise ValueError(f'Unknown covering {kind}')
    rng = np.random.default_rng(seed)
    h,w = rgb.shape[:2]
    mask = np.zeros((h,w),np.uint8)
    if kind == 'lower':
        top = rng.uniform(.48,.60)
        points = np.array([[.17,top+.04],[.48,top],[.83,top+.04],
                           [.76,.82],[.5,.91],[.23,.82]])
        points += rng.uniform(-.025,.025,points.shape)
        cv2.fillPoly(mask,[np.rint(points*[w,h]).astype(np.int32)],1)
    elif kind == 'eyes':
        cv2.rectangle(mask,(int(.13*w),int(rng.uniform(.25,.33)*h)),
                      (int(.87*w),int(rng.uniform(.43,.49)*h)),1,-1)
    elif kind == 'object':
        x,y = rng.uniform(.15,.55,2)
        rw,rh = rng.uniform(.2,.4,2)
        cv2.rectangle(mask,(int(x*w),int(y*h)),
                      (int(min(.95,x+rw)*w),int(min(.95,y+rh)*h)),1,-1)
    elif kind == 'irregular':
        points = np.rint(rng.uniform(.15,.85,(5,2))*[w,h]).astype(np.int32)
        cv2.polylines(mask,[points],False,1,max(2,int(.08*min(h,w))))
    color = rng.uniform(25,230,(1,1,3))
    texture = np.clip(color+rng.normal(0,8,(h,w,3)),0,255).astype(np.uint8)
    covered = np.where(mask[...,None].astype(bool),texture,rgb).astype(np.uint8)
    return covered,mask


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


class CompletionNet(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError('Width must be a positive multiple of four')
        self.width = width
        self.segmenter = GatedUNet(3,1,width)
        self.generator = GatedUNet(4,3,width)

    def detect(self,image):
        return self.segmenter(image)

    def complete(self,visible,mask):
        validate_mask(visible,mask)
        # Erase object colors before generation; skip connections cannot copy them.
        return self.generator(torch.cat((visible*(1-mask),mask),1)).sigmoid()

    def forward(self,visible,mask):
        return self.complete(visible,mask)
