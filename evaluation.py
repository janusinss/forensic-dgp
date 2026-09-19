import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
from tqdm import tqdm

# We use InsightFace to extract ArcFace embeddings for the Cosine Similarity metric.
# Note: In a production environment, you might need to build insightface from source or use onnx.
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("WARNING: InsightFace not installed properly. ArcFace Cosine Similarity will be skipped.")

class Evaluator:
    def __init__(self, device='cpu', use_arcface=True):
        self.device = device
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
        self.face_app = None
        if INSIGHTFACE_AVAILABLE and use_arcface:
            # Initialize ArcFace model via InsightFace for 512-d embeddings
            self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(256, 256))

    def compute_metrics(self, reconstructed, hr_target):
        """
        Computes PSNR, SSIM, and ArcFace Cosine Similarity.
        
        :param reconstructed: Tensor of shape (B, 3, 256, 256), range [0, 1]
        :param hr_target: Tensor of shape (B, 3, 256, 256), range [0, 1]
        :return: dict of metric scores
        """
        # 1. PSNR & SSIM
        psnr_val = self.psnr_metric(reconstructed, hr_target).item()
        ssim_val = self.ssim_metric(reconstructed, hr_target).item()
        
        cosine_sim = None
        
        # 2. ArcFace Cosine Similarity (Identity Loss)
        if self.face_app is not None:
            batch_size = reconstructed.size(0)
            sim_scores = []
            
            for i in range(batch_size):
                # Convert tensors [0, 1] to numpy BGR [0, 255] for InsightFace
                rec_np = (reconstructed[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rec_bgr = rec_np[:, :, ::-1] # RGB to BGR
                
                hr_np = (hr_target[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                hr_bgr = hr_np[:, :, ::-1] # RGB to BGR
                
                # Extract embeddings
                rec_faces = self.face_app.get(rec_bgr)
                hr_faces = self.face_app.get(hr_bgr)
                
                if len(rec_faces) > 0 and len(hr_faces) > 0:
                    rec_emb = torch.tensor(rec_faces[0].embedding)
                    hr_emb = torch.tensor(hr_faces[0].embedding)
                    
                    # Compute Cosine Similarity
                    sim = F.cosine_similarity(rec_emb.unsqueeze(0), hr_emb.unsqueeze(0)).item()
                    sim_scores.append(sim)
                    
            if len(sim_scores) > 0:
                cosine_sim = sum(sim_scores) / len(sim_scores)
                
        return {
            "PSNR": round(psnr_val, 2),
            "SSIM": round(ssim_val, 4),
            "ArcFace_Sim": round(cosine_sim, 4) if cosine_sim is not None else None
        }


@torch.no_grad()
def validate_model(model, loader, evaluator, device, preview_path=None, max_batches=None):
    """Mean per-image metrics on a fixed held-out set, including detection coverage."""
    model.eval()
    totals = {'PSNR': 0.0, 'SSIM': 0.0, 'ArcFace_Sim': 0.0}
    counts = dict.fromkeys(totals, 0)
    samples = 0
    total_samples = len(loader.dataset)
    if max_batches is not None and loader.batch_size is not None:
        total_samples = min(total_samples, max_batches * loader.batch_size)
    print('Validation starting: restoration and per-image quality metrics.', flush=True)
    with tqdm(total=total_samples, desc='Validation', unit='image', dynamic_ncols=True) as progress:
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            for i in range(len(inputs)):
                metrics = evaluator.compute_metrics(outputs[i:i+1], targets[i:i+1])
                for name in totals:
                    if metrics[name] is not None:
                        totals[name] += metrics[name]
                        counts[name] += 1
                samples += 1
                progress.set_postfix(arcface_valid=counts['ArcFace_Sim'], refresh=False)
                progress.update(1)
            if batch_idx == 0 and preview_path:
                from torchvision.utils import save_image
                # Each row: degraded input, raw model output, reference target.
                rows = torch.stack((inputs[:4], outputs[:4], targets[:4]), dim=1).flatten(0, 1)
                save_image(rows, preview_path, nrow=3)
    if not samples:
        raise ValueError('Validation loader contains no samples')
    result = {name: totals[name] / counts[name] if counts[name] else None for name in totals}
    result.update(samples=samples, arcface_valid=counts['ArcFace_Sim'])
    return result
