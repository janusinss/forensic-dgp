"""Retained input-only restoration suggestion; no identity/usefulness score."""
import cv2
import numpy as np

def quality_signals(square_rgb, square_mask):
    """Exact retained face_workflow.quality_signals math, without its Torch import."""
    rgb = cv2.resize(square_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(square_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    support = cv2.erode(1-mask, np.ones((9, 9), np.uint8)).astype(bool)
    support[:4] = False
    support[-4:] = False
    support[:, :4] = False
    support[:, -4:] = False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    enough = int(support.sum()) >= 512
    if not enough:
        return {'blur_variance': None, 'noise_sigma_255': None, 'suggest_restoration': False,
                'reason': 'Too little visible area for the automatic quality signal; choose an override.',
                'visible_signal_pixels': int(support.sum()), 'blur_threshold': 24, 'noise_threshold': 8}
    smoothed = cv2.GaussianBlur(gray, (3, 3), .6)
    blur = float(cv2.Laplacian(smoothed, cv2.CV_32F)[support].var())
    low = cv2.GaussianBlur(gray, (7, 7), 1.5)
    gradient = np.hypot(cv2.Sobel(low, cv2.CV_32F, 1, 0), cv2.Sobel(low, cv2.CV_32F, 0, 1))
    flat = support & (gradient <= np.percentile(gradient[support], 35))
    response = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32))
    noise = float(np.median(np.abs(response[flat])) / (6 * .67448975))
    reasons = []
    if blur < 24:
        reasons.append('blur signal')
    if noise >= 8:
        reasons.append('noise signal')
    return {'blur_variance': blur, 'noise_sigma_255': noise, 'suggest_restoration': bool(reasons),
            'reason': ', '.join(reasons) if reasons else 'Visible detail does not trigger restoration.',
            'visible_signal_pixels': int(support.sum()), 'blur_threshold': 24, 'noise_threshold': 8}

