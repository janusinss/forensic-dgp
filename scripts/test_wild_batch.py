import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import torch
import numpy as np
from models import DGPSynthesizer
from degradation import (
    adaptive_cctv_denoise,
    detect_and_deinterlace_cctv,
    detect_and_smooth_mosaic,
    align_canonical_face,
    paste_back_canonical_face
)

def run_universal_reconstruction(model, device, img_bgr, mode="direct"):
    # 1. Blind Signal Pre-Conditioning
    img_clean, is_interlaced, ratio = detect_and_deinterlace_cctv(img_bgr)
    img_clean, is_mosaic = detect_and_smooth_mosaic(img_clean)
    img_clean = adaptive_cctv_denoise(img_clean)
    
    # 2. Universal Canonical Face Registration
    aligned_clean_bgr, affine_M, reg_method = align_canonical_face(img_clean, target_size=(256, 256))
    
    if mode == "sub32":
        img_low_bgr = cv2.resize(aligned_clean_bgr, (32, 32), interpolation=cv2.INTER_AREA)
        input_preview = cv2.resize(img_low_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
    else:
        img_low_bgr = aligned_clean_bgr.copy()
        input_preview = aligned_clean_bgr.copy()
        
    img_rgb = cv2.cvtColor(img_low_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    dummy_target = torch.nn.functional.interpolate(input_tensor, size=(256, 256), mode='bilinear').to(device)
    with torch.no_grad():
        top_k_reconstructions, scores = model.generate_top_k(input_tensor, dummy_target, k=3)
        
    # Illumination and contrast post-processing
    ref_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    lab_ref = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_ref_mean = float(lab_ref[:, :, 0].mean())
    l_ref_std = float(max(1e-5, lab_ref[:, :, 0].std()))
    
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    results = []
    top1_rec_bgr = None
    
    for idx, rec_tensor in enumerate(top_k_reconstructions):
        rec_img_np = (rec_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
        lab_rec = cv2.cvtColor(rec_img_np, cv2.COLOR_RGB2LAB)
        l_rec_mean = float(lab_rec[:, :, 0].mean())
        l_rec_std = float(max(1e-5, lab_rec[:, :, 0].std()))
        
        if l_rec_mean < l_ref_mean:
            lab_rec[:, :, 0] = np.clip((lab_rec[:, :, 0] - l_rec_mean) * (l_ref_std / l_rec_std) + l_ref_mean, 0, 255).astype(np.uint8)
            
        lab_rec[:, :, 0] = clahe.apply(lab_rec[:, :, 0])
        rec_img_np = cv2.cvtColor(lab_rec, cv2.COLOR_LAB2RGB)
        
        blurred = cv2.GaussianBlur(rec_img_np, (0, 0), 1.2)
        weight = 1.45 if idx == 1 else (1.35 if idx == 0 else 1.20)
        rec_img_np = cv2.addWeighted(rec_img_np, weight, blurred, -(weight - 1.0), 0)
        rec_img_np = np.clip(rec_img_np, 0, 255).astype(np.uint8)
        rec_bgr = cv2.cvtColor(rec_img_np, cv2.COLOR_RGB2BGR)
        results.append((rec_bgr, scores[idx]))
        if idx == 0:
            top1_rec_bgr = rec_bgr.copy()
            
    # Seamless in-context composite paste-back
    composite_scene = paste_back_canonical_face(img_bgr, top1_rec_bgr, affine_M)
    
    diagnostics = {
        "interlaced": is_interlaced,
        "interlace_ratio": ratio,
        "mosaic": is_mosaic,
        "registration": reg_method
    }
    return input_preview, results, composite_scene, diagnostics

def main():
    os.makedirs("outputs/wild_tests", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = DGPSynthesizer().to(device)
    ckpt_path = "checkpoints/dgp_improved_epoch_20.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/dgp_improved_epoch_19.pth"
    print(f"Loading weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=True)
    model.eval()
    
    wild_files = [f"test_images/wild{i}.png" for i in range(1, 6)]
    
    rows = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    header = np.zeros((40, 256 * 5 + 40, 3), dtype=np.uint8)
    titles = ["RAW INPUT", "CANONICAL ALIGNED", "DIRECT RECONSTRUCT", "SUB-32 RECONSTRUCT", "IN-CONTEXT SCENE"]
    for c_idx, title in enumerate(titles):
        cv2.putText(header, title, (c_idx * 266 + 10, 26), font, 0.45, (0, 255, 200), 1, cv2.LINE_AA)
    rows.append(header)
    
    for idx, wpath in enumerate(wild_files, 1):
        if not os.path.exists(wpath):
            continue
            
        base_name = f"wild{idx}"
        raw_bgr = cv2.imread(wpath)
        print(f"\n--- Testing Universal Pipeline on {base_name} ({raw_bgr.shape}) ---")
        
        # Run Direct Mode
        in_direct, res_direct, comp_direct, diag = run_universal_reconstruction(model, device, raw_bgr, mode="direct")
        # Run Sub32 Mode
        in_sub32, res_sub32, _, _ = run_universal_reconstruction(model, device, raw_bgr, mode="sub32")
        
        rec_direct = res_direct[0][0]
        rec_sub32 = res_sub32[0][0]
        
        print(f"Diagnostics: Registration={diag['registration']} | Interlacing={diag['interlaced']} (ratio={diag['interlace_ratio']:.2f}) | Mosaic={diag['mosaic']}")
        
        # Save individual images
        cv2.imwrite(f"outputs/wild_tests/{base_name}_universal_aligned.png", in_direct)
        cv2.imwrite(f"outputs/wild_tests/{base_name}_universal_direct.png", rec_direct)
        cv2.imwrite(f"outputs/wild_tests/{base_name}_universal_sub32.png", rec_sub32)
        cv2.imwrite(f"outputs/wild_tests/{base_name}_universal_scene.png", comp_direct)
        
        # Assemble row for collage
        raw_square = cv2.resize(raw_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
        comp_square = cv2.resize(comp_direct, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        cv2.putText(raw_square, f"WILD {idx}", (10, 30), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        spacer = np.zeros((256, 10, 3), dtype=np.uint8)
        row = np.hstack([raw_square, spacer, in_direct, spacer, rec_direct, spacer, rec_sub32, spacer, comp_square])
        rows.append(row)
        
    collage = np.vstack(rows)
    cv2.imwrite("outputs/wild_tests/all_wild_comparison_universal.png", collage)
    print("\nSaved full universal comparison collage: outputs/wild_tests/all_wild_comparison_universal.png")

if __name__ == "__main__":
    main()
