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
import json
import random
import numpy as np
from pathlib import Path

from dataloader import get_training_loaders
from models import DGPSynthesizer, OptimalFaceRestorationLoss
from evaluation import Evaluator, validate_model
from training_state import load_training_state, save_training_state

def train(args):
    if args.epochs < args.start_epoch or args.batch_size < 1:
        raise ValueError('Invalid epoch range or batch size')
    if args.resume_from and not Path(args.resume_from).is_file():
        raise FileNotFoundError(args.resume_from)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'metrics.jsonl').exists() and not args.resume_from.endswith('last_state.pth'):
        raise ValueError('Output directory already contains a run; choose a new --output_dir')
    # Setup Device (Works on Cloud GPU or Local CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 1. Initialize DataLoader
    print("Initializing DataLoader with scale-curriculum degradation...")
    dataloader, validation_loader = get_training_loaders(
        root_dir=args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, seed=args.seed,
        validation_fraction=args.validation_fraction,
        heavy_blur_probability=args.heavy_blur_probability,
        curriculum=not args.no_curriculum
    )
    
    # 2. Initialize Models
    print("Initializing DGPSynthesizer...")
    model = DGPSynthesizer().to(device)
    
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
    state = load_training_state(args.resume_from, model, optimizer, scheduler, device) if args.resume_from else None
    if state:
        if Path(args.resume_from).resolve().parent != output_dir.resolve():
            raise ValueError('Resume a full training state in its original --output_dir')
        for key in ('seed', 'data_dir', 'validation_fraction', 'heavy_blur_probability', 'epochs', 'batch_size', 'no_curriculum', 'lambda_vgg', 'lambda_color', 'lambda_fan', 'lambda_sobel', 'lambda_fft', 'dry_run', 'skip_arcface'):
            if state['config'].get(key) != vars(args).get(key):
                raise ValueError(f'Resume configuration differs for {key}; use a weights checkpoint for a new experiment')
        args.start_epoch = state['epoch'] + 1
        if args.start_epoch > args.epochs:
            raise ValueError('This training run is already complete')
        
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
    evaluator = Evaluator(device=device, use_arcface=not args.skip_arcface)
    (output_dir / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    split = {
        'train': dataloader.dataset.image_paths,
        'validation': validation_loader.dataset.image_paths}
    if state and json.loads((output_dir / 'split.json').read_text(encoding='utf-8')) != split:
        raise ValueError('Dataset membership changed since this run was saved')
    (output_dir / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    baseline = validate_model(model, validation_loader, evaluator, device,
                              str(output_dir / 'baseline.png'), 1 if args.dry_run else None)
    best_psnr = state['best_psnr'] if state else baseline['PSNR']
    if not state:
        torch.save(model.state_dict(), output_dir / 'best.pth')
    with open(output_dir / 'metrics.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'stage': 'baseline', **baseline}) + '\n')
    print(f'Baseline validation: {baseline}')
    
    # 3. Training Loop
    print(f"Starting Training from Epoch {args.start_epoch} to {args.epochs}...")
    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        dataloader.dataset.epoch = epoch
        # Validation and initialization must not change the next epoch's shuffle.
        torch.manual_seed(args.seed + epoch)
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
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
        metrics = validate_model(model, validation_loader, evaluator, device,
                                 str(output_dir / f'epoch_{epoch}.png'), 1 if args.dry_run else None)
        print(f'Validation: {metrics}')
            
        # Save Checkpoint
        checkpoint_path = output_dir / f'dgp_improved_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}\n")
        
        # Log to file
        if metrics['PSNR'] > best_psnr:
            best_psnr = metrics['PSNR']
            torch.save(model.state_dict(), output_dir / 'best.pth')
        save_training_state(output_dir / 'last_state.pth', model, optimizer, scheduler,
                            epoch, best_psnr, vars(args))
        with open(output_dir / 'metrics.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps({'epoch': epoch, 'loss': avg_loss, 'best_psnr': best_psnr, **metrics}) + '\n')
        
        if args.dry_run:
            break

if __name__ == "__main__":
    default_data_dir = "dataset/ffhq" if os.path.exists("dataset/ffhq") else "dataset/thumbnails128x128"
    parser = argparse.ArgumentParser(description="Fine-tune face restoration with repeatable validation")
    parser.add_argument("--data_dir", type=str, default=default_data_dir, help="Path to FFHQ dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 produces 4,375 batches per epoch)")
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
    parser.add_argument('--output_dir', default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--validation_fraction', type=float, default=0.05)
    parser.add_argument('--heavy_blur_probability', type=float, default=0.35)
    parser.add_argument('--skip_arcface', action='store_true', help='Disable identity evaluation; reports null, not zero')
    
    args = parser.parse_args()
    train(args)
