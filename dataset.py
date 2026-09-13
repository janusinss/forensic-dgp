import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import face_alignment

from degradation import (
    apply_optical_motion_blur,
    apply_atmospheric_scattering,
    apply_spatial_downsampling,
    apply_thermal_noise,
    apply_h264_quantization
)

class DegradedFacesDataset(Dataset):
    """
    PyTorch Dataset that dynamically generates heavily degraded low-resolution
    CCTV-like faces paired with high-resolution pristine targets and 68-point
    facial landmarks.
    """
    def __init__(self, root_dir, transform=None, target_size=(24, 24), hr_size=(256, 256), curriculum=True):
        """
        :param root_dir: Path to directory containing high-res face images (e.g., FFHQ).
        :param transform: Optional torchvision transforms.
        :param target_size: The degraded sub-32x32 target size (used if curriculum=False).
        :param hr_size: The standardized high-res size for the pristine image.
        :param curriculum: If True, dynamically samples across 24x24 to 256x256 scales.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.hr_size = hr_size
        self.curriculum = curriculum
        
        # Load all valid image file paths from one or more directories (supports comma-separated paths)
        valid_exts = {'.png', '.jpg', '.jpeg'}
        self.image_paths = []
        dirs = root_dir.split(",") if isinstance(root_dir, str) else list(root_dir)
        for d in dirs:
            d = d.strip()
            if os.path.exists(d):
                for root, _, fnames in os.walk(d):
                    for fname in fnames:
                        if os.path.splitext(fname)[1].lower() in valid_exts:
                            self.image_paths.append(os.path.join(root, fname))
                    
        # Initialize face-alignment network (FAN)
        # Using CPU by default for the dataloader to avoid GPU memory conflicts, 
        # but can be switched to 'cuda' if sufficient VRAM is available.
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

    def __len__(self):
        return len(self.image_paths)
        
    def _get_landmarks(self, image):
        """
        Extracts 68-point Cartesian coordinates.
        :param image: numpy array of the image (RGB format expected for face-alignment)
        :return: numpy array of shape (68, 2)
        """
        preds = self.fa.get_landmarks(image)
        if preds is None or len(preds) == 0:
            return np.zeros((68, 2), dtype=np.float32)
        coords = preds[0].astype(np.float32)
        return coords

    def apply_compound_degradation(self, image):
        """
        Applies Second-Order Compound CCTV Degradation Model (Real-ESRGAN / BSRGAN standard):
        HR -> Stage 1 (Optical Motion / Gaussian Blur + Curriculum Downsampling + Sensor Thermal Noise)
           -> Stage 2 (Atmospheric Fog + Secondary Sinc Blur + Interlacing Comb Injection + Aggressive H.264/JPEG Codec)
           -> Standardize to (256, 256)
        """
        # --- STAGE 1: Primary Optical Capture & Sensor Degradation ---
        # 1. Primary Blur: 60% Optical Motion Blur, 40% Gaussian Blur
        if np.random.rand() < 0.60:
            d = np.random.randint(2, 10)
            theta = np.random.uniform(0, 360)
            img_deg = apply_optical_motion_blur(image, d, theta)
        else:
            k_size = int(np.random.choice([3, 5, 7]))
            sigma = float(np.random.uniform(0.8, 3.0))
            img_deg = cv2.GaussianBlur(image, (k_size, k_size), sigma)
        
        # 2. Primary Downsampling (Multi-Scale Curriculum)
        if self.curriculum:
            rand_p = np.random.rand()
            if rand_p < 0.40:
                scale = np.random.randint(24, 33)   # Sub-32x32 severe benchmark (Mode B)
            elif rand_p < 0.70:
                scale = np.random.randint(48, 65)   # Intermediate distance CCTV
            else:
                scale = np.random.randint(128, 257) # Native resolution CCTV crops (Mode A)
            downsample_target = (scale, scale)
        else:
            downsample_target = self.target_size

        # Use randomized interpolation methods (Area, Bilinear, Bicubic, Nearest for mosaic simulation)
        interp = int(np.random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]))
        img_deg = cv2.resize(img_deg, downsample_target, interpolation=interp)
        
        # 3. Sensor Thermal Noise Injection (N(0, sigma^2))
        std = float(np.random.uniform(4, 22))
        img_deg = apply_thermal_noise(img_deg, mean=0, std=std)

        # --- STAGE 2: Environmental & Transmission / Codec Degradation ---
        # 4. Atmospheric Scattering Simulation (Tropical Zamboanga Weather - 35% probability)
        if np.random.rand() < 0.35:
            t = float(np.random.uniform(0.45, 0.90))
            A = float(np.random.uniform(0.60, 1.0))
            img_deg = apply_atmospheric_scattering(img_deg, t, A)
        
        if img_deg.dtype != np.uint8:
            img_deg = (np.clip(img_deg, 0.0, 1.0) * 255).astype(np.uint8)
            
        # 5. Interlacing Simulation (Analog CCTV comb line artifact - 15% probability)
        if np.random.rand() < 0.15 and img_deg.shape[0] > 16:
            shift = int(np.random.choice([-3, -2, 2, 3]))
            img_deg[1::2, :] = np.roll(img_deg[1::2, :], shift, axis=1)

        # 6. Secondary Blur & Sinc Ringing Simulation (30% probability)
        if np.random.rand() < 0.30:
            img_deg = cv2.GaussianBlur(img_deg, (3, 3), float(np.random.uniform(0.5, 1.5)))

        # 7. Aggressive H.264 / Lossy Codec Quantization (Q_p >= 32)
        qp = int(np.random.randint(32, 51))
        img_deg = apply_h264_quantization(img_deg, qp=qp)
        
        # 8. Standardize degraded image to (256, 256) via bicubic interpolation
        img_deg = cv2.resize(img_deg, self.hr_size, interpolation=cv2.INTER_CUBIC)
        
        return img_deg

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load high-res pristine image I_HR
        i_hr_bgr = cv2.imread(img_path)
        if i_hr_bgr is None:
            raise ValueError(f"Failed to load image at {img_path}")
            
        # Standardize size
        i_hr_bgr = cv2.resize(i_hr_bgr, self.hr_size)
        
        # Convert I_HR to RGB (since cv2 loads in BGR) for landmarks and output
        i_hr_rgb = cv2.cvtColor(i_hr_bgr, cv2.COLOR_BGR2RGB)
        
        # Get 68-point landmarks from I_HR
        landmarks = self._get_landmarks(i_hr_rgb)
        
        # Generate I_LR (Degraded) from BGR (for opencv operations)
        i_lr_bgr = self.apply_compound_degradation(i_hr_bgr)
        i_lr_rgb = cv2.cvtColor(i_lr_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to torch Tensors (C, H, W format)
        i_hr_tensor = torch.from_numpy(i_hr_rgb).permute(2, 0, 1).float() / 255.0
        i_lr_tensor = torch.from_numpy(i_lr_rgb).permute(2, 0, 1).float() / 255.0
        landmarks_tensor = torch.from_numpy(landmarks)
        
        if self.transform:
            i_hr_tensor = self.transform(i_hr_tensor)
            i_lr_tensor = self.transform(i_lr_tensor)
            
        return i_lr_tensor, i_hr_tensor, landmarks_tensor
