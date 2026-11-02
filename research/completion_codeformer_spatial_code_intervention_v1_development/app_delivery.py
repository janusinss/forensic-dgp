"""Exact app float delivery only; no loss/autograd/optimizer in this probe."""

def _need(ok,message):
    if not ok:raise ValueError(message)

def delivered_float(raw512,source256,mask256,*,interpolate):
    """Clamp then bilinear then active-index compose, matching the app exactly."""
    _need(source256.ndim==4 and source256.shape[1]==3,'RGB NCHW source required')
    _need(mask256.shape==(source256.shape[0],1,*source256.shape[-2:]),'Single-channel source-sized mask required')
    result=source256.clone()
    active=mask256.sum((1,2,3))>0
    if not active.any():return result
    _need(raw512 is not None and raw512.ndim==4 and raw512.shape[:2]==(int(active.sum().item()),3),'Raw active RGB batch differs')
    _need(raw512.shape[-2:]==tuple(2*size for size in source256.shape[-2:]),'Exact2x raw-to-source geometry required')
    generated=((raw512+1)/2).clamp(0,1)
    generated=interpolate(generated,size=source256.shape[-2:],mode='bilinear',align_corners=False)
    result[active]=source256[active]*(1-mask256[active])+generated*mask256[active]
    return result

