"""Experimental input-only eye alignment; opt-in, never selected by dataset name."""
import cv2
import numpy as np
import torch
from completion import validate_mask


def load_eye_detector(path):
    from insightface.model_zoo import get_model
    detector=get_model(str(path),providers=['CPUExecutionProvider'])
    detector.prepare(ctx_id=-1,input_size=(256,256),det_thresh=.6)
    def eyes(rgb):
        h,w=rgb.shape[:2]
        image=cv2.resize((rgb*255).round().astype(np.uint8),(256,256))
        boxes,points=detector.detect(image[:,:,::-1].copy(),max_num=0)
        if len(boxes)!=1 or points is None or boxes[0,4]<.6:
            return None
        return points[0,:2]*np.array([w/256,h/256])
    return eyes


class SelectiveEyeAlignment(torch.nn.Module):
    """Provisional gate: only enlarged crops with two visible, near-level eyes.

    Thresholds are research hypotheses, not calibrated quality guarantees.
    Detection uses covered input only. Call events expose all fallback decisions.
    """
    def __init__(self, model, eye_detector):
        super().__init__()
        self.model=model
        self.eye_detector=eye_detector
        self.events=[]

    @torch.inference_mode()
    def forward(self,image,mask):
        validate_mask(image,mask)
        if image.shape[-1]!=image.shape[-2] or not torch.isfinite(image).all():
            raise ValueError('Expected finite square crops')
        if image.min()<0 or image.max()>1 or not ((mask==0)|(mask==1)).all():
            raise ValueError('Expected RGB in [0,1] and binary masks')
        outputs=[]
        for x,m in zip(image,mask):
            if not m.any():
                self.events.append({'decision':'fallback','reason':'empty_mask'})
                outputs.append(x);continue
            rgb=x.cpu().permute(1,2,0).numpy(); hole=m[0].cpu().numpy();n=hole.shape[0]
            eyes=self.eye_detector(rgb)
            reason='unreliable_eyes'
            eligible=False
            if eyes is not None:
                eyes=np.asarray(eyes,dtype=np.float64)
                if eyes.shape==(2,2) and np.isfinite(eyes).all():
                    eyes=eyes[np.argsort(eyes[:,0])];delta=eyes[1]-eyes[0]
                    distance=np.linalg.norm(delta)/n
                    radius=max(1,round(n*5/256));visible=True
                    for ex,ey in eyes:
                        px,py=int(round(ex)),int(round(ey))
                        if not (radius<=px<n-radius and radius<=py<n-radius):visible=False
                        elif hole[py-radius:py+radius+1,px-radius:px+radius+1].any():visible=False
                    eligible=visible and .32<distance<.5 and abs(np.degrees(np.arctan2(delta[1],delta[0])))<20
                    reason='geometry_or_visibility_gate'
            if not eligible:
                self.events.append({'decision':'fallback','reason':reason})
                outputs.append(self.model(x[None],m[None])[0]);continue
            target=np.array([[192.98138,239.94708],[318.90277,240.1936]])*n/512
            s=eyes[1]-eyes[0];t=target[1]-target[0]
            a=np.dot(t,s)/np.dot(s,s);b=(t[1]*s[0]-t[0]*s[1])/np.dot(s,s)
            linear=np.array([[a,-b],[b,a]])
            affine=np.column_stack([linear,target[0]-linear@eyes[0]])
            def warp(v,flags=cv2.INTER_LINEAR,border=cv2.BORDER_REFLECT_101):
                return cv2.warpAffine(v,affine,(n,n),flags=flags,borderMode=border)
            support=warp(1-hole)
            aligned=np.clip(warp(rgb*(1-hole[...,None]))/np.maximum(support[...,None],1e-8),0,1)
            am=warp(hole,cv2.INTER_NEAREST,cv2.BORDER_CONSTANT)
            def tensor(v):return torch.from_numpy(v.copy()).permute(2,0,1).to(x)[None]
            generated=self.model(tensor(aligned),tensor(am[...,None]))[0].cpu().permute(1,2,0).numpy()
            back=cv2.warpAffine(generated,cv2.invertAffineTransform(affine),(n,n),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            outputs.append(x*(1-m)+tensor(back)[0]*m)
            self.events.append({'decision':'aligned','eye_distance_fraction':float(distance)})
        return torch.stack(outputs)
