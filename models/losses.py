import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except Exception:
    pass

from torchvision.models import vgg19, VGG19_Weights
from .morphological_loss import MorphologicalFANLoss

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Smooth L1).
    Outperforms standard L1/L2 by remaining robust to edge outliers while
    providing continuous, differentiable gradients near zero.
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps ** 2

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps2))
        return loss

class ColorLoss(nn.Module):
    """
    Color Consistency Loss.
    Penalizes chromatic deviations on low-frequency average-pooled feature maps.
    Effectively eliminates color collapse and chromatic drift (green/yellow/blue tints).
    """
    def __init__(self, kernel_size=11):
        super(ColorLoss, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x, y):
        return F.l1_loss(self.pool(x), self.pool(y))

class SobelGradientLoss(nn.Module):
    """
    Directional Spatial Gradient Loss (Sobel Operators).
    Computes horizontal (gx) and vertical (gy) first-order spatial derivatives.
    Directly penalizes blurry edge transitions, forcing sharp pupil boundaries,
    eyelashes, and facial contours.
    """
    def __init__(self, device='cpu'):
        super(SobelGradientLoss, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer("sobel_x", sobel_x.to(device))
        self.register_buffer("sobel_y", sobel_y.to(device))

    def forward(self, x, y):
        sx = self.sobel_x.to(x.device)
        sy = self.sobel_y.to(x.device)
        gx_x = F.conv2d(x, sx, padding=1, groups=3)
        gy_x = F.conv2d(x, sy, padding=1, groups=3)
        gx_y = F.conv2d(y, sx, padding=1, groups=3)
        gy_y = F.conv2d(y, sy, padding=1, groups=3)
        return F.l1_loss(gx_x, gx_y) + F.l1_loss(gy_x, gy_y)

class FocalFFTLoss(nn.Module):
    """
    Fast Fourier Transform (FFT) Frequency Spectrum Loss.
    Measures discrepancy in the 2D frequency domain using real FFT (rfft2).
    Forces the generator to synthesize missing high-frequency spectrum.
    """
    def __init__(self):
        super(FocalFFTLoss, self).__init__()

    def forward(self, x, y):
        fft_x = torch.fft.rfft2(x, norm='ortho')
        fft_y = torch.fft.rfft2(y, norm='ortho')
        return F.l1_loss(torch.abs(fft_x), torch.abs(fft_y))

class FacialComponentAttentionLoss(nn.Module):
    """
    Landmark-Guided Facial Component Attention Loss.
    Constructs a smooth, continuous 2D Gaussian weight mask centered on
    biometric landmarks:
    - Ocular region (Left eye 36-41, Right eye 42-47): 2.5x weight
    - Perioral region (Mouth & lips 48-67): 2.0x weight
    - Ambient facial skin & background: 1.0x baseline
    """
    def __init__(self, eps=1e-3):
        super(FacialComponentAttentionLoss, self).__init__()
        self.eps2 = eps ** 2

    def build_attention_mask(self, landmarks_batch, H=256, W=256, device='cpu'):
        B = landmarks_batch.shape[0]
        masks = torch.ones((B, 1, H, W), device=device, dtype=torch.float32)
        y_coords = torch.arange(H, device=device).view(1, 1, H, 1).float()
        x_coords = torch.arange(W, device=device).view(1, 1, 1, W).float()

        for b in range(B):
            lm = landmarks_batch[b]
            if torch.sum(torch.abs(lm)) < 1e-4:
                continue

            # Ocular centers
            le_x, le_y = lm[36:42, 0].mean(), lm[36:42, 1].mean()
            re_x, re_y = lm[42:48, 0].mean(), lm[42:48, 1].mean()
            # Perioral center
            m_x, m_y = lm[48:68, 0].mean(), lm[48:68, 1].mean()

            sigma_eye = 22.0
            sigma_mouth = 28.0

            g_le = torch.exp(-((x_coords - le_x)**2 + (y_coords - le_y)**2) / (2 * sigma_eye**2))
            g_re = torch.exp(-((x_coords - re_x)**2 + (y_coords - re_y)**2) / (2 * sigma_eye**2))
            g_m = torch.exp(-((x_coords - m_x)**2 + (y_coords - m_y)**2) / (2 * sigma_mouth**2))

            # 2.5x ocular boost, 2.0x perioral boost
            masks[b, 0] = masks[b, 0] + (1.5 * g_le[0, 0]) + (1.5 * g_re[0, 0]) + (1.0 * g_m[0, 0])

        return masks

    def forward(self, reconstructed, hr_target, landmarks=None):
        diff = reconstructed - hr_target
        charb = torch.sqrt(diff * diff + self.eps2)

        if landmarks is not None and landmarks.shape[-1] == 2:
            mask = self.build_attention_mask(landmarks, H=reconstructed.shape[2], W=reconstructed.shape[3], device=reconstructed.device)
            return torch.mean(mask * charb)

        return torch.mean(charb)

class MultiLayerVGGPerceptualLoss(nn.Module):
    """
    Multi-Layer VGG19 Deep Perceptual Loss.
    Extracts features across multiple representational depths:
    - relu1_2: shallow edge/eyelash textures
    - relu2_2: fine iris/skin micro-texture
    - relu3_4: mid-level facial geometry
    - relu4_4: deep semantic biometric morphology
    """
    def __init__(self, device='cpu'):
        super(MultiLayerVGGPerceptualLoss, self).__init__()
        self.device = device
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False

        self.slice1 = vgg[:4]    # relu1_2
        self.slice2 = vgg[4:9]   # relu2_2
        self.slice3 = vgg[9:18]  # relu3_4
        self.slice4 = vgg[18:27] # relu4_4

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, x, y):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x - mean) / std
        y_norm = (y - mean) / std

        if next(self.slice1.parameters()).device != x.device:
            self.slice1 = self.slice1.to(x.device)
            self.slice2 = self.slice2.to(x.device)
            self.slice3 = self.slice3.to(x.device)
            self.slice4 = self.slice4.to(x.device)

        h1_x = self.slice1(x_norm)
        h2_x = self.slice2(h1_x)
        h3_x = self.slice3(h2_x)
        h4_x = self.slice4(h3_x)

        with torch.no_grad():
            h1_y = self.slice1(y_norm)
            h2_y = self.slice2(h1_y)
            h3_y = self.slice3(h2_y)
            h4_y = self.slice4(h3_y)

        loss = (0.1 * F.l1_loss(h1_x, h1_y) +
                0.2 * F.l1_loss(h2_x, h2_y) +
                1.0 * F.l1_loss(h3_x, h3_y) +
                1.0 * F.l1_loss(h4_x, h4_y))
        return loss

class OptimalFaceRestorationLoss(nn.Module):
    """
    Composite Multi-Component Loss for Optimal Face Reconstruction (Phase 2):
    L_total = L_comp + lambda_vgg * L_vgg_multi + lambda_color * L_color 
              + lambda_fan * L_fan + lambda_sobel * L_sobel + lambda_fft * L_fft
    """
    def __init__(self, device='cpu', lambda_vgg=0.15, lambda_color=0.05, lambda_fan=0.05, lambda_sobel=0.10, lambda_fft=0.05):
        super(OptimalFaceRestorationLoss, self).__init__()
        self.comp_loss = FacialComponentAttentionLoss()
        self.vgg_loss = MultiLayerVGGPerceptualLoss(device=device)
        self.color_loss = ColorLoss()
        self.fan_loss = MorphologicalFANLoss(device=str(device))
        self.sobel_loss = SobelGradientLoss(device=device)
        self.fft_loss = FocalFFTLoss()
        
        self.lambda_vgg = lambda_vgg
        self.lambda_color = lambda_color
        self.lambda_fan = lambda_fan
        self.lambda_sobel = lambda_sobel
        self.lambda_fft = lambda_fft

    def forward(self, reconstructed, hr_target, landmarks=None):
        l_comp = self.comp_loss(reconstructed, hr_target, landmarks=landmarks)
        l_vgg = self.vgg_loss(reconstructed, hr_target)
        l_color = self.color_loss(reconstructed, hr_target)
        l_fan = self.fan_loss(reconstructed, hr_target)
        l_sobel = self.sobel_loss(reconstructed, hr_target)
        l_fft = self.fft_loss(reconstructed, hr_target)

        total_loss = (l_comp + 
                      (self.lambda_vgg * l_vgg) + 
                      (self.lambda_color * l_color) + 
                      (self.lambda_fan * l_fan) +
                      (self.lambda_sobel * l_sobel) +
                      (self.lambda_fft * l_fft))

        return total_loss, {
            "Comp": l_comp.item(),
            "VGG": l_vgg.item(),
            "Color": l_color.item(),
            "FAN": l_fan.item(),
            "Sobel": l_sobel.item(),
            "FFT": l_fft.item(),
            "Total": total_loss.item()
        }

# Alias for backward compatibility
VGGPerceptualLoss = MultiLayerVGGPerceptualLoss
