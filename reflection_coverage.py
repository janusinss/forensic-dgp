"""Fixed reflection-coverage helpers; no model creation or training on import."""
import copy
import math

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from detector_training import segmentation_loss


def require(condition,message):
    if not condition:raise ValueError(message)


def supported_segmentation_loss(logits,mask,valid,background_weight=.25,hard_fraction=.1):
    require(logits.ndim==4 and logits.shape[1]==1 and logits.shape==mask.shape==valid.shape
            and torch.isfinite(logits).all() and torch.isfinite(mask).all() and torch.isfinite(valid).all()
            and ((mask==0)|(mask==1)).all() and ((valid==0)|(valid==1)).all(),
            'Expected finite matching N1HW logits and binary mask/support')
    require(math.isfinite(background_weight) and background_weight>=0 and 0<hard_fraction<=1,
            'Invalid support-loss weights')
    area=valid.sum((1,2,3))
    require((area>0).all() and not (mask*(1-valid)).any(), 'Empty source support or hole in padding')
    if bool((valid==1).all()):return segmentation_loss(logits,mask,background_weight,hard_fraction)
    probability=logits.sigmoid()
    bce=(F.binary_cross_entropy_with_logits(logits,mask,reduction='none')*valid).sum((1,2,3))/area
    dice=1-(2*(probability*mask*valid).sum((1,2,3))+1)/((probability*valid).sum((1,2,3))+(mask*valid).sum((1,2,3))+1)
    values=[]
    for logit,target,support in zip(logits,mask,valid):
        errors=F.softplus(logit)[(target==0)&(support==1)]
        if errors.numel():values.append(errors.topk(max(1,math.ceil(errors.numel()*hard_fraction))).values.mean())
    negative=torch.stack(values).mean() if values else logits.sum()*0
    return bce.mean()+dice.mean()+background_weight*negative


def camera_fixture(rgb,geometry,valid,seed,degraded):
    require(rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
            and geometry.shape==valid.shape==rgb.shape[:2]
            and np.isin(geometry,[0,1]).all() and np.isin(valid,[0,1]).all()
            and not np.any(geometry & (1-valid)), 'Invalid RGB/geometry/support fixture')
    rgb=rgb.copy();mask=geometry.copy();support=valid.copy();camera={'degraded':bool(degraded)}
    if degraded:
        rng=np.random.default_rng(seed);size=rgb.shape[0]
        sigma=float(rng.uniform(.5,1.2));low=int(rng.integers(max(8,size//2),size+1))
        noise_sigma=float(rng.uniform(1,8));quality=int(rng.integers(50,96))
        camera.update(blur_sigma=sigma,low_size=low,noise_sigma=noise_sigma,jpeg_quality=quality,
                      effect_dilation_radius=4,padding_influence_exclusion_radius=8)
        rgb=cv2.GaussianBlur(rgb,(7,7),sigma)
        rgb=cv2.resize(cv2.resize(rgb,(low,low),interpolation=cv2.INTER_AREA),
                       (size,size),interpolation=cv2.INTER_LINEAR)
        rgb=np.clip(rgb.astype(float)+rng.normal(0,noise_sigma,rgb.shape),0,255).astype(np.uint8)
        ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY,quality])
        require(ok,'Camera JPEG encoding failed')
        rgb=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        # Ignore the border influenced by neutral padding through filtering/JPEG.
        # Image edges without internal padding use OpenCV's reflected boundary.
        support=cv2.erode(support,np.ones((17,17),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=1)
        mask=cv2.dilate(mask,np.ones((9,9),np.uint8)) & support
    rgb[support==0]=96
    return {'input':rgb,'mask':mask.astype(np.uint8),'valid':support.astype(np.uint8),'camera':camera}


def supplemental_schedule(source_count,steps,seed):
    require(type(source_count) is int and source_count>0 and type(steps) is int and steps>0,
            'Positive supplemental source/step budget required')
    rng=np.random.default_rng(seed);streams={}
    for style in range(1,5):
        for condition in range(2):
            group=[10*i+2*style+condition for i in range(source_count)]
            stream=[]
            while len(stream)<math.ceil(steps/8):stream.extend(rng.permutation(group).tolist())
            streams[style,condition]=stream
    clear_stream=[]
    while len(clear_stream)<steps:
        clear_stream.extend(rng.permutation([10*i+c for i in range(source_count) for c in range(2)]).tolist())
    used={group:0 for group in streams};result=[]
    for step in range(steps):
        style=step%4+1;condition=(step//4)%2;group=(style,condition)
        result.append([streams[group][used[group]],clear_stream[step]]);used[group]+=1
    return result


def sanitize_schedule(batches,available_cases,quarantined_sources,real_count=73,seed=42):
    """Substitute training-only source conflicts without changing labels/caches.

    Real samples and safe cached samples keep their exact positions. Replacements
    have matching covered/clear and degradation strata. Both arms share this map.
    """
    available=set(available_cases);blocked=set(quarantined_sources)
    groups={}
    for case in sorted(available):
        if case//10 not in blocked:groups.setdefault((case%5==0,case%10>=5),[]).append(case)
    require(len(groups)==4 and all(groups.values()), 'Insufficient safe cases in all replay strata')
    rng=np.random.default_rng(seed);result=copy.deepcopy(batches);replacements=[]
    for epoch,steps in enumerate(result):
        for step,batch in enumerate(steps):
            require(len(batch)==8 and all(type(i) is int and i>=0 for i in batch), 'Expected fixed core batch size8')
            used={i-real_count for i in batch if i>=real_count and (i-real_count)//10 not in blocked}
            for slot,index in enumerate(batch):
                if index<real_count:continue
                case=index-real_count;require(case in available,'Core case absent from frozen cache')
                if case//10 not in blocked:continue
                eligible=[i for i in groups[case%5==0,case%10>=5] if i not in used]
                require(eligible,'No distinct source-safe replacement in replay stratum')
                replacement=int(rng.choice(eligible));batch[slot]=real_count+replacement;used.add(replacement)
                replacements.append({'epoch':epoch+1,'step':step+1,'slot':slot,'before':index,'after':batch[slot]})
    return {'batches':result,'replacements':replacements,'quarantined_sources':sorted(blocked)}
