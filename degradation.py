import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except Exception:
    pass

import cv2
import numpy as np
import torch
import math

def apply_optical_motion_blur(image, d, theta):
    """
    Simulates optical motion blur.
    :param image: Input image (numpy array, HxWxC or HxW)
    :param d: Displacement distance (pixels)
    :param theta: Trajectory angle (degrees)
    :return: Motion-blurred image
    """
    # Ensure d is an integer and > 0
    d = max(1, int(d))
    
    # Create the motion blur kernel
    kernel = np.zeros((d, d), dtype=np.float32)
    center = d // 2
    
    # Calculate the end point of the line based on the angle
    theta_rad = math.radians(theta)
    x = int(center * math.cos(theta_rad))
    y = int(center * math.sin(theta_rad))
    
    # Draw a line on the kernel to represent the motion path
    cv2.line(kernel, (center - x, center - y), (center + x, center + y), 1.0, 1)
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel)
    
    # Apply the 2D filter (convolution)
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def apply_atmospheric_scattering(image, t, A):
    """
    Simulates tropical rainfall, lens fogging, and humidity using the transmission model:
    I(x) = J(x)t(x) + A(1 - t(x))
    :param image: Input pristine image J(x) (numpy array, float32, range [0, 1])
    :param t: Transmission map (float or numpy array same size as image, range [0, 1]).
              Lower t means more scattering/fog.
    :param A: Atmospheric light (float or numpy array, typically between 0.5 and 1.0)
    :return: Scattered image I(x)
    """
    # Ensure image is float in [0, 1] for the math to work correctly
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
        
    degraded = image * t + A * (1 - t)
    
    # Clip and convert back if necessary
    degraded = np.clip(degraded, 0.0, 1.0)
    return degraded

def apply_spatial_downsampling(image, target_size=(24, 24)):
    """
    Bicubic interpolation reducing ground truth to sub-32x32 resolution.
    :param image: Input image
    :param target_size: Tuple (width, height) for the downsampled size
    :return: Downsampled image
    """
    downsampled = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return downsampled

def apply_thermal_noise(image, mean=0, std=10):
    """
    Injects zero-mean additive Gaussian noise N(0, sigma^2).
    :param image: Input image (numpy array, uint8 or float32)
    :param mean: Mean of the Gaussian noise
    :param std: Standard deviation of the Gaussian noise
    :return: Noisy image
    """
    is_uint8 = image.dtype == np.uint8
    if is_uint8:
        image = image.astype(np.float32)
        
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image + noise
    
    if is_uint8:
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    else:
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        
    return noisy_image

def apply_h264_quantization(image, qp=35):
    """
    Approximates aggressive H.264 quantization (Q_p >= 35).
    Since true H.264 encoding requires a video codec, we simulate it here
    by applying heavy JPEG compression, which uses a similar DCT block-based 
    quantization and chroma subsampling approach.
    :param image: Input image (numpy array, uint8)
    :param qp: Quantization parameter approximation. Higher QP = lower quality.
               (Typically H.264 QP ranges 0-51. We map it roughly to JPEG quality.)
    :return: Quantized image
    """
    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        
    # Map QP (approx 0-51) to JPEG quality (1-100). 
    # QP 35 is very lossy. A simple inverse linear mapping:
    # QP 0 -> Quality 100
    # QP 51 -> Quality 10
    quality = max(1, int(100 - (qp / 51.0) * 90))
    
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    if result:
        decimg = cv2.imdecode(encimg, 1)
        return decimg
    return image

def estimate_noise_sigma(image_gray):
    """
    Estimates Gaussian noise standard deviation sigma using the 
    Laplacian operator & Median Absolute Deviation (MAD).
    Reference: Immerkaer (1996) / Donoho (1994)
    """
    if len(image_gray.shape) == 3:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
        
    H, W = image_gray.shape
    mask = np.array([[ 1, -2,  1],
                     [-2,  4, -2],
                     [ 1, -2,  1]], dtype=np.float32)
    
    sigma = np.sum(np.abs(cv2.filter2D(image_gray.astype(np.float32), -1, mask)))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6.0 * max(1, W - 2) * max(1, H - 2))
    return float(sigma)

def adaptive_cctv_denoise(image_bgr):
    """
    Intelligently analyzes noise density in the degraded CCTV crop and applies
    appropriate edge-preserving filtering (Bilateral / Median) to prevent noise
    leakage into the generative residual reconstruction.
    """
    if image_bgr is None:
        return image_bgr
        
    # Work with uint8
    is_float = image_bgr.dtype in [np.float32, np.float64]
    if is_float:
        img_uint8 = (np.clip(image_bgr, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        img_uint8 = image_bgr.copy()
        
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect and eliminate Salt & Pepper / Impulse Noise
    median_filtered = cv2.medianBlur(gray, 3)
    diff = np.abs(gray.astype(np.int32) - median_filtered.astype(np.int32))
    impulse_ratio = np.mean(diff > 45)
    
    denoised = img_uint8
    if impulse_ratio > 0.008:
        denoised = cv2.medianBlur(denoised, 3)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
    # 2. Detect and eliminate Thermal Gaussian Sensor Noise
    sigma = estimate_noise_sigma(gray)
    if sigma > 6.0:
        # Edge-preserving Bilateral Filter
        denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=35, sigmaSpace=35)
        
    if is_float:
        return denoised.astype(np.float32) / 255.0
    return denoised

def detect_interlacing_energy(image_bgr):
    """
    Blindly detects analog CCTV comb-line interlacing artifacts (e.g. NTSC/PAL 480i/576i).
    Computes ratio of vertical-to-horizontal high-frequency differential energy.
    Returns (is_interlaced: bool, ratio: float).
    """
    if image_bgr is None or image_bgr.shape[0] < 8 or image_bgr.shape[1] < 8:
        return False, 1.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Odd/even line difference energy vs horizontal neighboring pixel difference
    diff_vert = np.abs(gray[1:-1, :] - 0.5 * (gray[:-2, :] + gray[2:, :]))
    diff_horiz = np.abs(gray[:, 1:-1] - 0.5 * (gray[:, :-2] + gray[:, 2:]))
    ratio = float(np.mean(diff_vert) / (np.mean(diff_horiz) + 1e-5))
    return (ratio > 2.0), ratio

def detect_and_deinterlace_cctv(image_bgr, threshold=2.0):
    """
    Blindly analyzes comb-line energy and applies vertical field de-interlacing
    only if interlaced video fields are detected. Otherwise leaves image unaltered.
    """
    is_interlaced, ratio = detect_interlacing_energy(image_bgr)
    if not is_interlaced:
        return image_bgr, False, ratio
        
    # Vertical line averaging to dissolve comb artifacts
    h, w = image_bgr.shape[:2]
    deint = image_bgr.copy().astype(np.float32)
    deint[1:-1:2, :] = 0.5 * (deint[:-2:2, :] + deint[2::2, :])
    deint = np.clip(deint, 0, 255).astype(np.uint8)
    return deint, True, ratio

def detect_and_smooth_mosaic(image_bgr):
    """
    Detects severe step-edge pixelation / mosaic censorship and applies
    localized anti-aliasing to melt block boundaries before generative synthesis.
    """
    if image_bgr is None:
        return image_bgr, False
        
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1))
    mag = np.sqrt(gx**2 + gy**2)
    
    # In mosaic pixelation, a large proportion of pixels have zero gradient inside blocks
    flat_ratio = float(np.mean(mag < 1.0))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    
    # Trigger if high flatness with localized sharp edges
    is_mosaic = (flat_ratio > 0.35 and lap_var < 50.0) or (flat_ratio > 0.40)
    if not is_mosaic:
        return image_bgr, False
        
    # Anti-mosaic: Adaptive downsample-upsample smoothing pass
    h, w = image_bgr.shape[:2]
    # Downsample by 4x using area averaging to dissolve block edges then bicubic upsample
    small = cv2.resize(image_bgr, (max(16, w // 4), max(16, h // 4)), interpolation=cv2.INTER_AREA)
    smoothed = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return smoothed, True

# Global singleton face alignment instance
_GLOBAL_FA = None

def get_face_aligner():
    global _GLOBAL_FA
    if _GLOBAL_FA is None:
        import face_alignment
        _GLOBAL_FA = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')
    return _GLOBAL_FA

def align_canonical_face(image_bgr, target_size=(256, 256)):
    """
    Universal 3-tier canonical face registration:
    - Tier 1: 5-point landmark extraction with FAN.
    - Tier 2: Anti-aliased fallback (runs on smoothed gradient if pixelated).
    - Tier 3: Aspect-ratio center crop fallback.
    Returns: (aligned_bgr, affine_M, registration_method)
    """
    h, w = image_bgr.shape[:2]
    
    # FFHQ canonical 5-point reference coordinates for 256x256
    dst_pts = np.array([
        [76.8, 97.3],    # Left eye
        [179.2, 97.3],   # Right eye
        [128.0, 143.4],  # Nose tip
        [92.2, 184.3],   # Left mouth corner
        [163.8, 184.3]   # Right mouth corner
    ], dtype=np.float32)

    try:
        fa = get_face_aligner()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(rgb)
        
        # Tier 2: If no face found, test with anti-aliasing pre-filter
        if preds is None or len(preds) == 0:
            smoothed_rgb = cv2.GaussianBlur(rgb, (9, 9), 3)
            preds = fa.get_landmarks(smoothed_rgb)
            
        if preds is not None and len(preds) > 0:
            pts = preds[0]
            left_eye = pts[36:42].mean(axis=0)
            right_eye = pts[42:48].mean(axis=0)
            nose = pts[30]
            left_mouth = pts[48]
            right_mouth = pts[54]
            
            # Geometric Sanity Gate:
            # 1. Eye distance must be >= 12 pixels
            eye_dist = np.linalg.norm(right_eye - left_eye)
            # 2. Eye tilt angle must be within +/- 45 degrees
            eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
            # 3. Mouth center must be below eye line
            mouth_y = 0.5 * (left_mouth[1] + right_mouth[1])
            eye_y = 0.5 * (left_eye[1] + right_eye[1])
            
            if eye_dist >= 12 and abs(eye_angle) <= 45.0 and mouth_y > eye_y:
                src_pts = np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)
                M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if M is not None:
                    aligned = cv2.warpAffine(image_bgr, M, target_size, borderMode=cv2.BORDER_REFLECT)
                    return aligned, M, "CANONICAL_5POINT"
    except Exception as e:
        pass
        
    # Tier 3: Morphological / Center-Crop Fallback
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    crop = image_bgr[top:top+min_dim, left:left+min_dim]
    aligned = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Compute affine matrix for fallback center-crop so inverse warping works
    scale = float(target_size[0]) / float(min_dim)
    M_fallback = np.array([
        [scale, 0.0, -left * scale],
        [0.0, scale, -top * scale]
    ], dtype=np.float32)
    return aligned, M_fallback, "FALLBACK_CENTER"

def paste_back_canonical_face(original_bgr, restored_crop_bgr, affine_M, mask_feather=15):
    """
    Seamlessly warps the restored canonical 256x256 crop back into the original
    surveillance image space using M^(-1) with soft boundary feathering.
    """
    if affine_M is None:
        return original_bgr
        
    h_orig, w_orig = original_bgr.shape[:2]
    h_crop, w_crop = restored_crop_bgr.shape[:2]
    
    # Invert affine transformation matrix
    M_inv = cv2.invertAffineTransform(affine_M)
    
    # Create elliptical feathering mask for seamless boundary transition
    mask = np.zeros((h_crop, w_crop), dtype=np.float32)
    center = (w_crop // 2, h_crop // 2)
    axes = (int(w_crop * 0.46), int(h_crop * 0.46))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    
    if mask_feather > 0:
        ksize = mask_feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), mask_feather / 2.0)
        
    mask = np.clip(mask, 0.0, 1.0)[:, :, np.newaxis]
    
    # Warp restored face and mask back to original resolution
    warped_rec = cv2.warpAffine(restored_crop_bgr, M_inv, (w_orig, h_orig), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    warped_mask = cv2.warpAffine(mask, M_inv, (w_orig, h_orig), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if len(warped_mask.shape) == 2:
        warped_mask = warped_mask[:, :, np.newaxis]
        
    # Alpha blend: original * (1 - mask) + restored * mask
    composite = (original_bgr.astype(np.float32) * (1.0 - warped_mask) + warped_rec.astype(np.float32) * warped_mask)
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    return composite

