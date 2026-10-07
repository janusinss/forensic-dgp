"""Same-call frozen identity reference. Original objective weights remain exact."""
import torch
from cctv_dgp_degraded_objective_v24 import assemble_terms,cohort_normalizers


def batchmatched_scores(b,pred,identity):
    """Evaluate baseline and prediction in one batch/grad context; freeze reference."""
    assert pred.shape==b['base'].shape and pred.shape[0]>0
    n=pred.shape[0]
    baseline=b['base']*b['mask']+b['x']*(1-b['mask'])
    prediction=pred*b['mask']+b['x']*(1-b['mask'])
    vectors=identity.embedding(torch.cat((baseline.detach(),prediction),0),
        torch.cat((b['mask'],b['mask']),0),torch.cat((b['grid'],b['grid']),0))
    assert vectors.shape[0]==2*n and b['truth'].shape==vectors[:n].shape
    reference=(vectors[:n].detach()*b['truth'].detach()).sum(1)
    current=(vectors[n:]*b['truth'].detach()).sum(1)
    return reference,current


def objective_terms(b,pred,identity,mean,feature_errors,ssim,normalizers):
    feature,interior=feature_errors(pred,b['target'],b['feature'],b['interior'])
    pixel=mean((pred-b['target']).square(),b['mask'])
    base_pixel=mean((b['base']-b['target']).square(),b['mask'])
    base_cosine,cosine=batchmatched_scores(b,pred,identity)
    score=ssim(pred,b['target'],b['valid7']);base_score=ssim(b['base'],b['target'],b['valid7'])
    anchor=mean((pred-b['base']).square(),b['mask'])
    return assemble_terms(feature,interior,pixel,base_pixel,score,base_score,cosine,base_cosine,anchor,
        b['degraded_weight'],b['clear_weight'],normalizers)
