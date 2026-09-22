import copy
import math
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ModelEMA:
    """Average unique parameters and buffers; MobileNet/FPN shares modules."""
    def __init__(self, model, decay=.999):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0, 1)')
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source_params = dict(model.named_parameters())
        for name, p in self.model.named_parameters():
            p.lerp_(source_params[name], 1-self.decay)
        source_buffers = dict(model.named_buffers())
        for name, buf in self.model.named_buffers():
            source = source_buffers[name]
            if buf.is_floating_point():
                buf.lerp_(source, 1-self.decay)
            else:
                buf.copy_(source)


def qualifies(candidate, baseline, best):
    """PSNR gain is accepted only without measured identity/SSIM regression.

    Fixed target alignment makes identity-pair membership independent of outputs.
    These strict experimental gates do not establish forensic identity fidelity.
    """
    for result in (candidate, baseline, best):
        if any(result.get(k) is None or not math.isfinite(result[k])
               for k in ('PSNR','SSIM','ArcFace_fixed')) or result['identity_pairs'] < 1:
            return False
    if not (candidate['PSNR'] > best['PSNR'] and candidate['SSIM'] >= baseline['SSIM']
            and candidate['ArcFace_fixed'] >= baseline['ArcFace_fixed']
            and candidate['identity_pairs'] == baseline['identity_pairs']):
        return False
    for group, reference in baseline.get('groups', {}).items():
        current = candidate.get('groups', {}).get(group)
        if current is None or reference['ArcFace_fixed'] is None or current['ArcFace_fixed'] is None:
            return False
        if (current['identity_pairs'] != reference['identity_pairs'] or
                current['ArcFace_fixed'] < reference['ArcFace_fixed'] or
                current['SSIM'] < reference['SSIM'] or current['PSNR'] < reference['PSNR']):
            return False
    return True


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (*self.dataset[index], index)


@torch.no_grad()
def evaluate_identity(model, loader, identity_loss, device, paths, preview_path=None):
    """Fixed reference alignment, same eligible identities every epoch.

    ArcFace_fixed differs from Phase 4's independent face-detection metric.
    Missing reference landmarks are excluded and coverage is reported by source.
    """
    from torchmetrics.functional.image import structural_similarity_index_measure
    model.eval()
    rows = []
    for batch_index, (low, target, landmarks, indices) in enumerate(tqdm(loader, desc='Fixed-pair validation', dynamic_ncols=True)):
        low, target, landmarks = low.to(device), target.to(device), landmarks.to(device)
        rec = model(low)
        if not torch.isfinite(rec).all():
            raise FloatingPointError('Non-finite restoration during validation')
        psnr = -10*torch.log10((rec-target).square().mean((1,2,3)).clamp_min(1e-12))
        ssim = structural_similarity_index_measure(rec,target,data_range=1.,reduction='none')
        similarities, valid = identity_loss.similarities(rec,target,landmarks)
        identity = dict(zip(valid.nonzero().flatten().tolist(), similarities.tolist()))
        for i, index in enumerate(indices.tolist()):
            source = 'asian_faces' if 'asian_faces' in paths[index].replace('\\','/').split('/') else 'ffhq'
            rows.append({'index':index,'source':source,'PSNR':psnr[i].item(),
                         'SSIM':ssim[i].item(),'ArcFace_fixed':identity.get(i)})
        if batch_index == 0 and preview_path:
            from torchvision.utils import save_image
            save_image(torch.stack((low[:4],rec[:4],target[:4]),1).flatten(0,1),preview_path,nrow=3)
    if not rows:
        raise ValueError('Validation is empty')
    def aggregate(items):
        valid_items = [r for r in items if r['ArcFace_fixed'] is not None]
        return {'PSNR':sum(r['PSNR'] for r in items)/len(items),
                'SSIM':sum(r['SSIM'] for r in items)/len(items),
                'ArcFace_fixed':sum(r['ArcFace_fixed'] for r in valid_items)/len(valid_items) if valid_items else None,
                'identity_pairs':len(valid_items), 'samples':len(items)}
    metrics = aggregate(rows)
    metrics['groups'] = {source:aggregate([r for r in rows if r['source']==source]) for source in sorted({r['source'] for r in rows})}
    if preview_path:
        import json
        from pathlib import Path
        Path(preview_path).with_suffix('.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return metrics
