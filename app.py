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
from degradation import adaptive_cctv_denoise

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
    Dual-Mode CCTV Face Reconstruction:
    - Mode 'direct': Direct Forensic Restoration (preserves full resolution, removes blur, maximizes facial sharpness).
    - Mode 'sub32': Sub-32x32 Super-Resolution Simulation (Thesis Benchmark: downsamples to 32x32 and reconstructs).
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return {"error": "Invalid image format."}
        
    # 1. Adaptive Edge-Preserving Pre-Denoising (Bilateral / Median)
    img_clean_bgr = adaptive_cctv_denoise(img_bgr)
    
    # 2. Forensic Target Extraction & Aspect-Ratio Preservation: Center-crop to 1:1 square
    h, w = img_clean_bgr.shape[:2]
    if h != w:
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        img_clean_bgr = img_clean_bgr[top:top+min_dim, left:left+min_dim]
    
    cctv_thumb_b64 = None
    if mode == "sub32":
        # Mode B: Sub-32x32 Super-Resolution Simulation (Thesis Benchmark)
        # Downsamples to 32x32 CCTV standard and generates pixelated preview
        img_low_bgr = cv2.resize(img_clean_bgr, (32, 32), interpolation=cv2.INTER_AREA)
        thumb_display = cv2.resize(img_low_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
        _, thumb_buf = cv2.imencode('.png', thumb_display)
        cctv_thumb_b64 = f"data:image/png;base64,{base64.b64encode(thumb_buf).decode('utf-8')}"
    else:
        # Mode A: Direct Forensic Restoration (Deblur & Maximum Detail)
        # Preserves full native resolution upscaled directly to 256x256 canonical space
        img_low_bgr = cv2.resize(img_clean_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)

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
    
    for idx, rec_tensor in enumerate(top_k_reconstructions):
        # tensor is (1, 3, 256, 256) in [0, 1] range
        rec_img_np = (rec_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
        
        # Adaptive Forensic Illumination Calibration:
        # Prevents atmospheric dehazing offset from underexposing normal CCTV inputs
        lab_rec = cv2.cvtColor(rec_img_np, cv2.COLOR_RGB2LAB)
        l_rec_mean = float(lab_rec[:, :, 0].mean())
        l_rec_std = float(max(1e-5, lab_rec[:, :, 0].std()))
        
        if l_rec_mean < l_ref_mean:
            lab_rec[:, :, 0] = np.clip((lab_rec[:, :, 0] - l_rec_mean) * (l_ref_std / l_rec_std) + l_ref_mean, 0, 255).astype(np.uint8)
            
        # Apply CLAHE on L-channel to reveal sharp facial features (eyes, nose, lips)
        lab_rec[:, :, 0] = clahe.apply(lab_rec[:, :, 0])
        rec_img_np = cv2.cvtColor(lab_rec, cv2.COLOR_LAB2RGB)

        # High-definition unsharp masking tailored per rank
        blurred = cv2.GaussianBlur(rec_img_np, (0, 0), 1.2)
        weight = 1.45 if idx == 1 else (1.35 if idx == 0 else 1.20)
        rec_img_np = cv2.addWeighted(rec_img_np, weight, blurred, -(weight - 1.0), 0)
        rec_img_np = np.clip(rec_img_np, 0, 255).astype(np.uint8)
        
        # Encode back to PNG buffer
        success, encoded_img = cv2.imencode('.png', cv2.cvtColor(rec_img_np, cv2.COLOR_RGB2BGR))
        if success:
            base64_str = base64.b64encode(encoded_img).decode('utf-8')
            result_images.append({
                "rank": idx + 1,
                "score": round(scores[idx], 4),
                "image_data": f"data:image/png;base64,{base64_str}"
            })
            
    response_data = {"results": result_images, "mode": mode}
    if cctv_thumb_b64:
        response_data["cctv_thumbnail"] = cctv_thumb_b64
        
    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
