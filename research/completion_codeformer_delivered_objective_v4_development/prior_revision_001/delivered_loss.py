"""Exact app-float delivery and masked256 RGB loss, with injected interpolation.

No scientific library is imported at module load: transfer-only CLI remains pure.
These operators accept plain tensors; they do not create models or optimizers.
Spatial geometry is a 2x downsample so small CPU graph witnesses use the exact
same operators. The guarded VM worker admits only512 output and256 source.
"""
OBJECTIVE_ID='app_float256_mask_RGB_L1_only_v1'

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

def delivered_hole_loss(raw512,source256,mask256,clean256,*,interpolate,microbatches=4):
    """Return delivered float, unscaled hole RGB L1 and once-divided micro loss.

    No separate boundary term,512 target, PNG rounding or gradient surrogate.
    The exact existing clean256 is supplied; no target conversion happens here.
    """
    _need(type(microbatches) is int and microbatches==4,'Frozen four-microbatch divisor required')
    _need(clean256.shape==source256.shape,'Exact source-sized clean256 target required')
    _need(mask256.sum().item()>0,'Empty support has no training loss; app bypasses it')
    app_float=delivered_float(raw512,source256,mask256,interpolate=interpolate)
    hole_RGB_L1=(abs(app_float-clean256)*mask256).sum()/(mask256.sum()*3)
    scaled_microbatch_loss=hole_RGB_L1/microbatches
    return app_float,hole_RGB_L1,scaled_microbatch_loss

def clipping_diagnostic(raw512,mask512):
    """Detached scalar observations only; never change clipping or gradients."""
    normalized=(raw512.detach()+1)/2
    _need(mask512.shape==(normalized.shape[0],1,*normalized.shape[-2:]),'Raw support geometry differs')
    channels=mask512.sum()*3
    _need(channels.item()>0,'Clipping diagnostic requires a nonempty hole')
    low=(normalized<0).to(normalized.dtype);high=(normalized>1).to(normalized.dtype)
    outside=low+high
    pixel_outside=(outside.sum(1,keepdim=True)>0).to(normalized.dtype)
    return {'raw512_hole_channel_count':int(channels.item()),
            'below0_channel_fraction_hole512':float(((low*mask512).sum()/channels).item()),
            'above1_channel_fraction_hole512':float(((high*mask512).sum()/channels).item()),
            'clipped_channel_fraction_hole512':float(((outside*mask512).sum()/channels).item()),
            'clipped_pixel_fraction_hole512':float(((pixel_outside*mask512).sum()/mask512.sum()).item()),
            'clamp_zero_gradient_for_strictly_outside_channels':True,
            'surrogate_gradient_used':False,'scope':'Detached out-of-range fractions, not identity or anatomy quality.'}
