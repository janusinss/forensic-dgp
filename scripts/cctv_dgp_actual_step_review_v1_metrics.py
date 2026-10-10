"""Raw versus delivered metrics for inference proposals; no neural imports."""
import hashlib
import numpy as np
import cv2
from scipy.ndimage import convolve1d
from skimage.metrics import structural_similarity


def parameter_state_hash(values, layout):
    digest=hashlib.sha256(); rows={r['name']:r for r in layout}
    assert len(rows)==57 and values.dtype==np.float32 and values.shape==(17952,)
    for name in sorted(rows):
        r=rows[name];a=values[r['start']:r['end']].reshape(r['shape'])
        digest.update(name.encode()+b'\0torch.float32\0')
        digest.update(str(tuple(r['shape'])).encode()+b'\0'+a.tobytes())
    return digest.hexdigest()


def pixel_metrics(actual, target, mask):
    assert actual.dtype==np.float32 and actual.shape==(256,256,3)
    reference=target.astype(np.float32)/np.float32(255)
    error=actual-reference;mse=float(np.square(error[mask]).astype(np.float64).mean())
    _,ssmap=structural_similarity(reference,actual,data_range=1,channel_axis=-1,win_size=7,full=True)
    interior=cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    return {'MSE':mse,'PSNR':float(-10*np.log10(mse)) if mse else None,'perfect_match':mse==0,
            'SSIM':float(ssmap[interior].astype(np.float64).mean()),'MAE':float(np.abs(error[mask]).astype(np.float64).mean())}


def detail_float(actual,target,support):
    z=np.arange(-6,7,dtype=np.float64);kernel=np.exp(-.5*(z/2)**2);kernel/=kernel.sum()
    high=lambda a:a-convolve1d(convolve1d(a,kernel,axis=0,mode='reflect'),kernel,axis=1,mode='reflect')
    weights=np.array([.299,.587,.114]);a=(actual.astype(np.float64)*weights).sum(2)
    b=(target.astype(np.float64)/255*weights).sum(2)
    return float(np.square(high(a)-high(b))[support].mean())


def deliver(raw,camera,mask):
    assert raw.dtype==np.float32 and raw.shape==(256,256,3) and np.isfinite(raw).all()
    assert raw.min()>=0 and raw.max()<=1
    result=np.floor(raw*np.float32(255)).astype(np.uint8);result[~mask]=camera[~mask]
    return result


def mean_only(raw,zero_raw,camera,mask):
    shift=(raw-zero_raw)[mask].astype(np.float64).mean(0)
    value=np.clip(zero_raw.astype(np.float64)+shift,0,1).astype(np.float32)
    value[~mask]=camera[~mask].astype(np.float32)/np.float32(255)
    return value,deliver(value,camera,mask),shift


def review_groups(rows,stage):
    metrics=['MSE','SSIM','ArcFace_observed_fixed','landmark_high_frequency_MSE','constant_mean_shift_only_MSE']
    sources={r['source'] for r in rows}
    keys={'all','clear','degraded'}|{r['source']+'/'+s for r in rows for s in ['all','clear','degraded',r['profile']]}
    result={}
    for key in sorted(keys):
        selected=[r for r in rows if key in {'all','clear' if r['profile']=='clear' else 'degraded',r['source']+'/all',
                 r['source']+('/clear' if r['profile']=='clear' else '/degraded'),r['source']+'/'+r['profile']}]
        assert selected
        result[key]={'cases':len(selected),**{m:float(np.mean([r[stage][m] for r in selected])) for m in metrics}}
    assert len(result)==3+7*len(sources)
    return result


def finite_comparison(before,after):
    assert set(before)==set(after)
    failures=[]
    for key,b in before.items():
        a=after[key];assert b['cases']==a['cases']
        for metric in ['MSE','SSIM','ArcFace_observed_fixed']:
            bad=a[metric]>b[metric]+1e-12 if metric=='MSE' else a[metric]<b[metric]-1e-6
            if bad:failures.append({'group':key,'metric':metric,'baseline':b[metric],'proposal':a[metric]})
    b=before['degraded'];a=after['degraded']
    gain=1-a['landmark_high_frequency_MSE']/b['landmark_high_frequency_MSE']
    source_gains={k:1-after[k]['landmark_high_frequency_MSE']/v['landmark_high_frequency_MSE'] for k,v in before.items() if k.endswith('/degraded')}
    fraction=max(0,b['MSE']-a['constant_mean_shift_only_MSE'])/max(b['MSE']-a['MSE'],1e-12)
    return {'relative_feature_gain':gain,'source_feature_gains':source_gains,'preservation_failures':failures,
            'mean_only_fraction':fraction,'finite_preservation_pass':not failures,
            'full_TRAIN_capacity_not_tested':True,'diagnostic_does_not_promote_model':True}
