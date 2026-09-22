"""Phase 5 pilot: frozen ArcFace supervision, EMA and conservative selection."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import split_paths
from dataset import DegradedFacesDataset
from models import DGPSynthesizer, OptimalFaceRestorationLoss
from models.identity_loss import ArcFaceIdentityLoss
from phase5_utils import IndexedDataset, ModelEMA, evaluate_identity, qualifies
from training_state import load_training_state, save_training_state


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.lambda_identity < 0:
        raise ValueError('Invalid training configuration')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'metrics.jsonl').exists() and not args.resume_state:
        raise ValueError('Run already exists; use --resume_state or a new --output_dir')
    if args.resume_state and Path(args.resume_state).resolve() != (output/'last_state.pth').resolve():
        raise ValueError('Resume state must be this output directory\'s last_state.pth')
    checkpoint = Path(args.resume_state or args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    arcface_path = Path(args.arcface_model).expanduser()
    args.arcface_sha256 = hashlib.sha256(arcface_path.read_bytes()).hexdigest()
    datasets = [DegradedFacesDataset(args.data_dir, seed=args.seed, heavy_blur_probability=args.heavy_blur_probability,
                                    landmark_cache_dir=args.landmark_cache) for _ in range(2)]
    training, validation = datasets
    if args.split_file:
        split = json.loads(Path(args.split_file).read_text(encoding='utf-8'))
        train_paths, val_paths = split['train'], split['validation']
        expected = {str(Path(p).resolve()) for p in training.image_paths}
        train_set, val_set = ({str(Path(p).resolve()) for p in paths} for paths in (train_paths,val_paths))
        if (train_set & val_set or train_set | val_set != expected or
                len(train_set) != len(train_paths) or len(val_set) != len(val_paths)):
            raise ValueError('Split must contain every dataset image exactly once without overlap')
    else:
        train_paths, val_paths = split_paths(training.image_paths,.05,args.seed)
    if args.dry_run:
        # Explicitly isolated smoke test, never counted as a complete epoch.
        train_paths, val_paths = train_paths[:args.batch_size], val_paths[:args.batch_size]
    training.image_paths, validation.image_paths = train_paths, val_paths
    if not train_paths or not val_paths:
        raise ValueError('Training and validation must both contain images')
    split = {'train':train_paths,'validation':val_paths}
    if args.resume_state and json.loads((output/'split.json').read_text()) != split:
        raise ValueError('Dataset split changed since the saved run')
    (output/'split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    print(f'Device: {device}; train: {len(train_paths)}; validation: {len(val_paths)}',flush=True)
    print('ArcFace uses fixed reference alignment. Its scores differ from Phase 4 detection-based scores.',flush=True)
    options = dict(batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=device=='cuda')
    train_loader = DataLoader(training,shuffle=True,**options)
    val_loader = DataLoader(IndexedDataset(validation),shuffle=False,**options)
    model = DGPSynthesizer().to(device)
    backbone = list(model.fpn.features.parameters())
    backbone_ids = {id(p) for p in backbone}
    optimizer = torch.optim.Adam([
        {'params':backbone,'lr':args.lr_backbone},
        {'params':[p for p in model.parameters() if id(p) not in backbone_ids],'lr':args.lr}],weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-7)
    state = load_training_state(checkpoint,model,optimizer,scheduler,device)
    if state and not args.resume_state:
        raise ValueError('Use a weights file for --checkpoint; full states require --resume_state')
    ema = ModelEMA(model,args.ema_decay)
    if args.resume_state:
        if not state or 'extra' not in state or 'ema' not in state['extra']:
            raise ValueError('Not a Phase 5 training state')
        ignored = {'resume_state','checkpoint','num_workers','split_file','arcface_model','landmark_cache'}
        for key,value in vars(args).items():
            if key not in ignored and state['config'].get(key) != value:
                raise ValueError(f'Resume configuration mismatch: {key}')
        ema.model.load_state_dict(state['extra']['ema'],strict=True)
    print('Loading frozen differentiable ArcFace...',flush=True)
    identity_loss = ArcFaceIdentityLoss(arcface_path,device=device)
    criterion = OptimalFaceRestorationLoss(device=device)
    (output/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    if state:
        baseline,best = state['extra']['baseline'],state['extra']['best']
        start = state['epoch']+1
    else:
        baseline = evaluate_identity(model,val_loader,identity_loss,device,val_paths,str(output/'baseline.png'))
        if baseline['identity_pairs'] == 0:
            raise ValueError('No valid reference landmarks for identity evaluation')
        best = baseline
        start = 1
        torch.save(model.state_dict(),output/'best.pth')
        (output/'best_selection.json').write_text(json.dumps({'epoch':0,'reason':'Starting baseline','metrics':best},indent=2))
        with open(output/'metrics.jsonl','a',encoding='utf-8') as handle:
            handle.write(json.dumps({'stage':'baseline',**baseline})+'\n')
    print('Baseline:',baseline,flush=True)
    if start > args.epochs:
        print('Run already completed.',flush=True)
        return
    for epoch in range(start,args.epochs+1):
        torch.manual_seed(args.seed+epoch)
        training.epoch = epoch
        model.train()
        losses, samples, id_samples = 0.,0,0
        progress = tqdm(train_loader,desc=f'Phase 5 epoch {epoch}/{args.epochs}',dynamic_ncols=True)
        for low,target,landmarks in progress:
            low,target,landmarks = low.to(device),target.to(device),landmarks.to(device)
            optimizer.zero_grad(set_to_none=True)
            restored = model(low)
            base_loss, components = criterion(restored,target,landmarks=landmarks)
            id_loss, count = identity_loss(restored,target,landmarks) if args.lambda_identity else (restored.sum()*0,0)
            loss = base_loss + args.lambda_identity*id_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            ema.update(model)
            losses += loss.item()*len(low)
            samples += len(low)
            id_samples += count
            progress.set_postfix(loss=f'{loss.item():.4f}',identity=f'{id_loss.item():.4f}',id_pairs=f'{count}/{len(low)}')
        if args.lambda_identity and not id_samples:
            raise ValueError('No training sample contributed an identity gradient')
        scheduler.step()
        metrics = evaluate_identity(ema.model,val_loader,identity_loss,device,val_paths,str(output/f'epoch_{epoch}.png'))
        accepted = qualifies(metrics,baseline,best)
        torch.save(ema.model.state_dict(),output/f'epoch_{epoch}.pth')
        if accepted:
            best = metrics
            torch.save(ema.model.state_dict(),output/'best.pth')
            (output/'best_selection.json').write_text(json.dumps({'epoch':epoch,'reason':'Passed PSNR, SSIM and identity gates','metrics':best},indent=2))
        save_training_state(output/'last_state.pth',model,optimizer,scheduler,epoch,best['PSNR'],vars(args),
                            extra={'ema':ema.model.state_dict(),'baseline':baseline,'best':best})
        record = {'epoch':epoch,'loss':losses/samples,'training_identity_pairs':id_samples,'accepted':accepted,**metrics}
        with open(output/'metrics.jsonl','a',encoding='utf-8') as handle:
            handle.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
        if args.dry_run:
            print('Smoke test completed: one batch only, not a trained epoch.',flush=True)
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir',default='dataset/thumbnails128x128,dataset/asian_faces')
    parser.add_argument('--checkpoint',default='checkpoints/dgp_zamboanga_final.pth')
    parser.add_argument('--resume_state',default='')
    parser.add_argument('--output_dir',default='outputs/phase5_identity')
    parser.add_argument('--split_file',default='')
    parser.add_argument('--arcface_model',default='~/.insightface/models/buffalo_l/w600k_r50.onnx')
    parser.add_argument('--landmark_cache',default='outputs/landmark_cache')
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--num_workers',type=int,default=2)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--lr_backbone',type=float,default=2e-6)
    parser.add_argument('--lambda_identity',type=float,default=.1)
    parser.add_argument('--ema_decay',type=float,default=.999)
    parser.add_argument('--heavy_blur_probability',type=float,default=.35)
    parser.add_argument('--dry_run',action='store_true')
    train(parser.parse_args())
