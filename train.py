import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
except Exception:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

from dataloader import get_dataloader
from models import DGPSynthesizer, OptimalFaceRestorationLoss
from evaluation import Evaluator

def train(args):
    # Setup Device (Works on Cloud GPU or Local CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 1. Initialize DataLoader
    print("Initializing DataLoader with scale-curriculum degradation...")
    dataloader = get_dataloader(
        root_dir=args.data_dir, 
        batch_size=args.batch_size, 
        shuffle=True, 
        curriculum=not args.no_curriculum
    )
    
    # 2. Initialize Models
    print("Initializing DGPSynthesizer...")
    model = DGPSynthesizer().to(device)
    
    # Load Checkpoint / Resume Weights
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        try:
            state_dict = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            print("SUCCESS: 100% of pre-trained weights loaded with strict=True!")
        except Exception as e:
            print(f"Notice on strict load: {e}. Falling back to strict=False...")
            model.load_state_dict(torch.load(args.resume_from, map_location=device), strict=False)
    else:
        print(f"Starting without existing checkpoint at: {args.resume_from}")
        
    # Optimizer configuration: Differential Learning Rates
    # Backbone: fine-tunes gently (args.lr_backbone, default 5e-6)
    # Head & Fusion: learns active high-frequency synthesis (args.lr, default 5e-5)
    backbone_params = list(model.fpn.features.parameters())
    head_params = [p for p in model.parameters() if not any(p is bp for bp in backbone_params)]
    
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr}
    ], betas=(0.9, 0.999), weight_decay=1e-5)
    print(f"Differential Learning Rates -> Backbone: {args.lr_backbone:.1e}, Head/Fusion: {args.lr:.1e}")
    
    # Cosine Annealing Learning Rate Scheduler for smooth convergence across epochs
    total_epochs = max(1, args.epochs - args.start_epoch + 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
        
    # Composite Optimal Loss Function (6 Terms)
    criterion = OptimalFaceRestorationLoss(
        device=device,
        lambda_vgg=args.lambda_vgg,
        lambda_color=args.lambda_color,
        lambda_fan=args.lambda_fan,
        lambda_sobel=args.lambda_sobel,
        lambda_fft=args.lambda_fft
    )
    
    # Evaluator
    evaluator = Evaluator(device=device)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 3. Training Loop
    print(f"Starting Training from Epoch {args.start_epoch} to {args.epochs}...")
    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (low_res, hr_target, landmarks) in enumerate(progress_bar):
            low_res = low_res.to(device)
            hr_target = hr_target.to(device)
            landmarks = landmarks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(low_res)
            
            # Compute Composite Loss (Attention Charbonnier + Multi-Layer VGG + Color + FAN + Sobel + FFT)
            total_loss, loss_dict = criterion(reconstructed, hr_target, landmarks=landmarks)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "Comp": f"{loss_dict.get('Comp', 0.0):.3f}",
                "VGG": f"{loss_dict.get('VGG', 0.0):.3f}",
                "Sob": f"{loss_dict.get('Sobel', 0.0):.3f}",
                "FFT": f"{loss_dict.get('FFT', 0.0):.3f}",
                "FAN": f"{loss_dict.get('FAN', 0.0):.3f}"
            })
            
            # For Dry Run, break after 1 batch
            if args.dry_run:
                print("\n[Dry Run] Successfully completed 1 batch forward & backward pass with zero errors.")
                break
                
        avg_loss = epoch_loss / max(1, 1 if args.dry_run else len(dataloader))
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
        
        # Step LR Scheduler
        scheduler.step()
        curr_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Current LRs -> Backbone: {curr_lrs[0]:.2e}, Head: {curr_lrs[1]:.2e}")
        
        # 4. Evaluation Phase (End of Epoch)
        model.eval()
        with torch.no_grad():
            metrics = evaluator.compute_metrics(reconstructed, hr_target)
            print(f"Validation Metrics -> PSNR: {metrics['PSNR']} | SSIM: {metrics['SSIM']} | ArcFace: {metrics['ArcFace_Sim']}")
            
        # Save Checkpoint
        checkpoint_path = f"checkpoints/dgp_improved_epoch_{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}\n")
        
        if args.dry_run:
            break

if __name__ == "__main__":
    default_data_dir = "dataset/ffhq" if os.path.exists("dataset/ffhq") else "dataset/thumbnails128x128"
    parser = argparse.ArgumentParser(description="Train the Optimal Deep Generative Prior Face Restoration Model (Epochs 11-20)")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Path to FFHQ dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (Keep small for <8GB VRAM)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--start_epoch", type=int, default=11, help="Epoch to start counting from")
    parser.add_argument("--resume_from", type=str, default="checkpoints/dgp_improved_epoch_10.pth", help="Path to checkpoint .pth file to resume from")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for generative head")
    parser.add_argument("--lr_backbone", type=float, default=5e-6, help="Learning rate for feature backbone")
    parser.add_argument("--lambda_vgg", type=float, default=0.15, help="Weight for Multi-Layer VGG19 Perceptual Loss")
    parser.add_argument("--lambda_color", type=float, default=0.05, help="Weight for Color Consistency Loss")
    parser.add_argument("--lambda_fan", type=float, default=0.05, help="Weight for Morphological FAN Loss")
    parser.add_argument("--lambda_sobel", type=float, default=0.10, help="Weight for Sobel Directional Gradient Loss")
    parser.add_argument("--lambda_fft", type=float, default=0.05, help="Weight for Focal FFT Frequency Loss")
    parser.add_argument("--no_curriculum", action="store_true", help="Disable multi-scale curriculum degradation")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 batch to verify the pipeline")
    
    args = parser.parse_args()
    train(args)

