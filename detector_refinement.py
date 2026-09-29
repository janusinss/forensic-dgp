"""Experimental image-only SAM prompt adapter; never reads reference masks."""
import cv2
import numpy as np


def detector_prompt(probabilities):
    p=np.asarray(probabilities)
    if p.ndim!=2 or not p.size or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
        raise ValueError('Expected finite HxW probabilities in [0,1]')
    raw=(p>=.5).astype(np.uint8)
    if not raw.any():return None
    count,labels,stats,_=cv2.connectedComponentsWithStats(raw,8)
    component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    region=(labels==component).astype(np.uint8)
    # Padding ensures distance is defined even for components reaching every edge.
    distance=cv2.distanceTransform(np.pad(region,1),cv2.DIST_L2,5)[1:-1,1:-1]
    y,x=np.unravel_index(np.argmax(distance),distance.shape)
    left,top,width,height,_=stats[component];dx=max(1,int(np.ceil(width*.1)));dy=max(1,int(np.ceil(height*.1)))
    box=np.array([max(0,left-dx),max(0,top-dy),min(p.shape[1]-1,left+width-1+dx),min(p.shape[0]-1,top+height-1+dy)],np.float32)
    return {'point_coords':np.array([[x,y]],np.float32),'point_labels':np.array([1],np.int32),'box':box}


def refine_mask(image,probabilities,predictor):
    prompt=detector_prompt(probabilities)
    if image.dtype!=np.uint8 or image.shape!=(*np.asarray(probabilities).shape,3):
        raise ValueError('Expected matching RGB uint8 image')
    raw=np.asarray(probabilities)>=.5
    if prompt is None:return raw,{'status':'empty_detector'}
    if raw.mean()>.85:return raw,{'status':'detector_coverage_unsupported'}
    predictor.set_image(image)
    masks,scores,_=predictor.predict(**prompt,multimask_output=True)
    masks=np.asarray(masks);scores=np.asarray(scores)
    if masks.ndim!=3 or masks.shape[1:]!=raw.shape or scores.shape!=(len(masks),) or not np.isfinite(scores).all() or not len(masks):
        raise ValueError('Invalid predictor output')
    if not np.isin(masks,[0,1]).all():raise ValueError('Expected binary predictor masks')
    candidate=int(np.argmax(scores));chosen=masks[candidate].astype(bool)
    meta={'candidate':candidate,'predicted_quality':float(scores[candidate]),'prompt':{k:v.tolist() for k,v in prompt.items()}}
    x,y=prompt['point_coords'][0].astype(int)
    if chosen.mean()>.85 or not chosen.any():return raw,{**meta,'status':'rejected_coverage'}
    if not chosen[y,x]:return raw,{**meta,'status':'rejected_seed'}
    return chosen,{**meta,'status':'refined'}
