import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except Exception:
    pass

import io
import cv2
import base64
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from models import DGPSynthesizer
from degradation import (
    adaptive_cctv_denoise,
    detect_and_deinterlace_cctv,
    detect_and_smooth_mosaic,
    align_canonical_face,
    paste_back_canonical_face,
    estimate_noise_sigma
)

app = FastAPI(title="Optimal Face Restoration Web UI")

# Mount static files (CSS, JS)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model globally (loads once on startup)
print("Loading Optimal Generative Face Restoration Synthesizer...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGPSynthesizer().to(device)

# Auto-detect best model weights (searches highest improved epoch first, then weights/, then root)
checkpoint_path = None
for ep in range(30, 0, -1):
    cand = f"checkpoints/dgp_improved_epoch_{ep}.pth"
    if os.path.exists(cand):
        checkpoint_path = cand
        break

if checkpoint_path is None:
    for cand in ["weights/mapped_deblurgan.pth", "checkpoints/mapped_deblurgan.pth", "mapped_deblurgan.pth"]:
        if os.path.exists(cand):
            checkpoint_path = cand
            break

if checkpoint_path is None:
    checkpoint_path = "weights/mapped_deblurgan.pth"

if os.path.exists(checkpoint_path):
    print(f"SUCCESS: Loading pre-trained intelligence from {checkpoint_path}...")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
        print("Model loaded with 100% strict matching.")
    except Exception as e:
        print(f"Notice: {e}. Falling back to strict=False...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
else:
    print(f"WARNING: {checkpoint_path} not found! Please place weights file in root or checkpoints/.")

model.eval()
print("Optimal Face Restoration Engine ready.")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main UI HTML file with UTF-8 encoding and cache-invalidation."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(
        content=html_content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.post("/reconstruct")
async def reconstruct_image(file: UploadFile = File(...), mode: str = Form("direct")):
    """
    Universal Blind Forensic CCTV Face Reconstruction:
    - Automatically classifies and mitigates interlaced scanlines, mosaic censorship, and noise.
    - Universally aligns face via 5-point canonical affine landmark registration.
    - Generates Top-K forensic restorations and seamless in-context scene paste-back.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return {"error": "Invalid image format."}
        
    # 1. Blind Signal Pre-Conditioning (Universal Detection)
    # A. Detect and dissolve analog CCTV comb scanlines (fixes interlaced surveillance)
    img_clean_bgr, is_interlaced, _ = detect_and_deinterlace_cctv(img_bgr)
    
    # B. Detect and dissolve mosaic / pixelation step-edges (fixes censored/blocked CCTV)
    img_clean_bgr, is_mosaic = detect_and_smooth_mosaic(img_clean_bgr)
    
    # C. Adaptive Edge-Preserving Denoising (impulse & sensor noise)
    img_clean_bgr = adaptive_cctv_denoise(img_clean_bgr)
    noise_sigma = estimate_noise_sigma(cv2.cvtColor(img_clean_bgr, cv2.COLOR_BGR2GRAY))
    
    # 2. Universal 3-Tier Canonical Face Registration
    # Normalizes pupils and facial symmetry to standard FFHQ canonical coordinates
    aligned_clean_bgr, affine_M, reg_method = align_canonical_face(img_clean_bgr, target_size=(256, 256))
    
    cctv_thumb_b64 = None
    if mode == "sub32":
        # Mode B: Sub-32x32 Super-Resolution Simulation (Thesis Benchmark)
        img_low_bgr = cv2.resize(aligned_clean_bgr, (32, 32), interpolation=cv2.INTER_AREA)
        thumb_display = cv2.resize(img_low_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
        _, thumb_buf = cv2.imencode('.png', thumb_display)
        cctv_thumb_b64 = f"data:image/png;base64,{base64.b64encode(thumb_buf).decode('utf-8')}"
    else:
        # Mode A: Direct Forensic Restoration (Canonical Native Scale)
        img_low_bgr = aligned_clean_bgr.copy()
        _, thumb_buf = cv2.imencode('.png', aligned_clean_bgr)
        cctv_thumb_b64 = f"data:image/png;base64,{base64.b64encode(thumb_buf).decode('utf-8')}"

    img_rgb = cv2.cvtColor(img_low_bgr, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] tensor
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # 3. Generate Top-K reconstructions
    k = 3
    dummy_target = torch.nn.functional.interpolate(input_tensor, size=(256, 256), mode='bilinear').to(device)
    top_k_reconstructions, scores = model.generate_top_k(input_tensor, dummy_target, k=k)
    
    # Reference low-resolution input scaled up for illumination calibration
    ref_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    lab_ref = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_ref_mean = float(lab_ref[:, :, 0].mean())
    l_ref_std = float(max(1e-5, lab_ref[:, :, 0].std()))

    # 4. Convert output tensors to base64 strings with guided detail enhancement & illumination preservation
    result_images = []
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    top1_rec_bgr = None
    
    for idx, rec_tensor in enumerate(top_k_reconstructions):
        rec_img_np = (rec_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
        
        # Adaptive Forensic Illumination Calibration
        lab_rec = cv2.cvtColor(rec_img_np, cv2.COLOR_RGB2LAB)
        l_rec_mean = float(lab_rec[:, :, 0].mean())
        l_rec_std = float(max(1e-5, lab_rec[:, :, 0].std()))
        
        if l_rec_mean < l_ref_mean:
            lab_rec[:, :, 0] = np.clip((lab_rec[:, :, 0] - l_rec_mean) * (l_ref_std / l_rec_std) + l_ref_mean, 0, 255).astype(np.uint8)
            
        # Apply CLAHE on L-channel to reveal sharp facial features
        lab_rec[:, :, 0] = clahe.apply(lab_rec[:, :, 0])
        rec_img_np = cv2.cvtColor(lab_rec, cv2.COLOR_LAB2RGB)

        # High-definition unsharp masking tailored per rank
        blurred = cv2.GaussianBlur(rec_img_np, (0, 0), 1.2)
        weight = 1.45 if idx == 1 else (1.35 if idx == 0 else 1.20)
        rec_img_np = cv2.addWeighted(rec_img_np, weight, blurred, -(weight - 1.0), 0)
        rec_img_np = np.clip(rec_img_np, 0, 255).astype(np.uint8)
        rec_bgr = cv2.cvtColor(rec_img_np, cv2.COLOR_RGB2BGR)
        
        if idx == 0:
            top1_rec_bgr = rec_bgr.copy()
        
        # Encode back to PNG buffer
        success, encoded_img = cv2.imencode('.png', rec_bgr)
        if success:
            base64_str = base64.b64encode(encoded_img).decode('utf-8')
            result_images.append({
                "rank": idx + 1,
                "score": round(scores[idx], 4),
                "image_data": f"data:image/png;base64,{base64_str}"
            })
            
    # 5. Seamless Contextual Paste-Back (Invert Affine Transform to restore face in original surveillance context)
    composite_scene_b64 = None
    if top1_rec_bgr is not None and affine_M is not None:
        composite_bgr = paste_back_canonical_face(img_bgr, top1_rec_bgr, affine_M)
        _, comp_buf = cv2.imencode('.png', composite_bgr)
        composite_scene_b64 = f"data:image/png;base64,{base64.b64encode(comp_buf).decode('utf-8')}"
        
    response_data = {
        "results": result_images,
        "mode": mode,
        "cctv_thumbnail": cctv_thumb_b64,
        "composite_scene": composite_scene_b64,
        "diagnostics": {
            "registration_method": reg_method,
            "interlacing_detected": bool(is_interlaced),
            "mosaic_detected": bool(is_mosaic),
            "noise_sigma": round(noise_sigma, 2)
        }
    }
        
    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
