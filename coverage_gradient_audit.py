"""Loss decomposition helpers for a zero-update diagnostic."""
import torch
from detector_training import segmentation_loss
from detector_replay import replay_consistency


def objective_terms(logits, target, synthetic, added, teacher):
    if synthetic.dtype != torch.bool or added.dtype != torch.bool or (synthetic & added).any():
        raise ValueError('Invalid disjoint sample tags')
    if not synthetic.any() or synthetic.all():
        raise ValueError('Both real and replay samples required')
    terms = {}
    covered = target.flatten(1).any(1)
    groups = {'real_existing_covered':~synthetic & ~added & covered,
              'real_added_covered':~synthetic & added & covered,
              'real_clear':~synthetic & ~covered,
              'replay_supervised':synthetic}
    for name,mask in groups.items():
        if mask.any():
            terms[name] = segmentation_loss(logits[mask],target[mask],.25,.1) * mask.sum()/len(logits)
    terms['teacher'] = replay_consistency(logits[synthetic],teacher,
        torch.ones(int(synthetic.sum()),dtype=torch.bool,device=logits.device))
    full = segmentation_loss(logits,target,.25,.1) + terms['teacher']
    return terms, full


def cosine(a,b):
    denominator = a.norm()*b.norm()
    return None if a.norm()<1e-8 or b.norm()<1e-8 else float(torch.dot(a,b)/denominator)
