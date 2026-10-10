"""Preserve V1; align its PNG-detail arithmetic with the original PNG metric."""
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/cctv_dgp_multiscale_color_detail_v1'
OUT = ROOT/'outputs/cctv_dgp_multiscale_color_detail_v1_r2'
PACKET = ROOT/'outputs/cctv_dgp_multiscale_calibration_vm_v1'
RETURNED = ROOT/'outputs/cctv_dgp_multiscale_calibration_vm_v1_return/outputs'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def pixels(path,mode='RGB'):
    with Image.open(path) as image:
        assert image.size==(256,256)
        return np.asarray(image.convert(mode)).copy()


def main():
    start=time.monotonic()
    assert not OUT.exists()
    old=read(OLD/'results.json'); old_audit=read(OLD/'independent_audit.json')
    assert old_audit['complete'] and old_audit['results_sha256']==sha(OLD/'results.json')
    p=read(PACKET/'protocol.json'); baseline=read(RETURNED/'baseline/metrics.json')['groups']
    bindings={**read(OLD/'plan.json')['bindings'],**old['output_bindings'],
        (OLD/'results.json').relative_to(ROOT).as_posix():sha(OLD/'results.json'),
        (OLD/'independent_audit.json').relative_to(ROOT).as_posix():sha(OLD/'independent_audit.json'),
        (PACKET/'frozen_capacity_contract.py').relative_to(ROOT).as_posix():sha(PACKET/'frozen_capacity_contract.py'),
        Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__))}
    for name,digest in bindings.items():
        assert sha(ROOT/name)==digest,name
    OUT.mkdir()
    write(OUT/'plan.json',dict(complete=True,frozen_before_new_measurements=True,bindings=bindings,
        original_receipts_preserved=True,fix='PNG HF uses uint8-to-float64 conversion, matching frozen_capacity_contract.detail_metric',
        raw_MSE_SSIM_and_raw_HF_unchanged=True,scientific_thresholds_unchanged=True,
        no_new_images_or_models=True,local_neural_calls=0,local_optimizer_updates=0,worker_seconds=180))
    pos=np.arange(-6,7,dtype=np.float64); kernel=np.exp(-.5*(pos/2)**2); kernel/=kernel.sum()
    weights=np.array([.299,.587,.114]); controls=[]; worst=0.; old_difference=0.
    for summary in old['controls']:
        folder=OLD/summary['arm']/summary['variant']
        metrics=read(folder/'metrics.json')
        for case,row in zip(p['cases'],metrics['rows']):
            png=pixels(folder/'previews'/(case['id']+'.png')); target=pixels(PACKET/case['target'])
            mask=pixels(PACKET/case['observed'],'L')>0
            support=np.zeros((256,256),bool)
            for x,y in np.floor(case['landmarks5_canvas_xy']).astype(int):
                support[max(0,y-12):min(256,y+12),max(0,x-12):min(256,x+12)]=True
            support &= cv2.erode(mask.astype(np.uint8),np.ones((13,13),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            a=(png.astype(np.float64)/255*weights).sum(2); b=(target.astype(np.float64)/255*weights).sum(2)
            high=lambda value:value-convolve1d(convolve1d(value,kernel,axis=0,mode='reflect'),kernel,axis=1,mode='reflect')
            updated=float(np.square(high(a)-high(b))[support].mean())
            old_difference=max(old_difference,abs(updated-row['png']['landmark_high_frequency_MSE']))
            row['png']['landmark_high_frequency_MSE']=updated
            # Independent OpenCV replay with its own equivalent tap formula.
            xx=np.linspace(-6,6,13); taps=np.exp(-xx*xx/8); taps/=taps.sum()
            a2=a-cv2.sepFilter2D(a,cv2.CV_64F,taps,taps,borderType=cv2.BORDER_REFLECT)
            b2=b-cv2.sepFilter2D(b,cv2.CV_64F,taps,taps,borderType=cv2.BORDER_REFLECT)
            worst=max(worst,abs(float(np.square(a2-b2)[support].mean())-updated))
            assert time.monotonic()-start<180
        grouped=metrics['groups']['png']
        for key,group in grouped.items():
            selected=[r for r in metrics['rows'] if key in {'all',
                'clear' if r['profile']=='clear' else 'degraded',r['source']+'/all',
                r['source']+('/clear' if r['profile']=='clear' else '/degraded'),r['source']+'/'+r['profile']}]
            group['landmark_high_frequency_MSE']=float(np.mean([r['png']['landmark_high_frequency_MSE'] for r in selected]))
        comparison=metrics['comparisons']['png']
        comparison['relative_feature_gain']=1-grouped['degraded']['landmark_high_frequency_MSE']/baseline['png']['degraded']['landmark_high_frequency_MSE']
        comparison['pixel_only_requirements_pass']=comparison['relative_feature_gain']>=.01 and not comparison['preservation_failures']
        assert comparison['pixel_only_requirements_pass']==summary['comparisons']['png']['pixel_only_requirements_pass']
        assert metrics['comparisons']['raw']==summary['comparisons']['raw']
        assert comparison['preservation_failures']==summary['comparisons']['png']['preservation_failures']
        # All original pixel/SSIM values and raw values are exact.
        original=read(folder/'metrics.json')
        for old_row,new_row in zip(original['rows'],metrics['rows']):
            assert old_row['raw']==new_row['raw']
            for key,value in old_row['png'].items():
                if key!='landmark_high_frequency_MSE':
                    assert value==new_row['png'][key]
        file=OUT/(summary['arm']+'-'+summary['variant']+'.json')
        write(file,metrics)
        controls.append(dict(arm=summary['arm'],variant=summary['variant'],comparisons=metrics['comparisons'],
            metrics_file=file.relative_to(ROOT).as_posix(),metrics_sha256=sha(file)))
    assert worst<=1e-12
    for name,digest in bindings.items():
        assert sha(ROOT/name)==digest,name
    result=dict(complete=True,plan_sha256=sha(OUT/'plan.json'),controls=controls,
        corrected_PNG_detail_records=400,exact_raw_and_other_PNG_values_preserved=True,
        all_decisions_and_failure_locations_unchanged=True,maximum_independent_filter_difference=worst,
        maximum_original_PNG_detail_precision_difference=old_difference,
        independent_replay_scope='OpenCV filter versus SciPy; exact non-HF rows and decisions checked against retained V1',
        ArcFace_not_recomputed=True,visual_usefulness_not_assessed=True,native_pixels_decoded=0,final_pixels_decoded=0,
        local_neural_calls=0,local_gradient_queries=0,local_optimizer_updates=0,
        model_qualification=False,app_changed=False,goal_complete=False,seconds=time.monotonic()-start)
    write(OUT/'results.json',result)
    print(dict(complete=True,corrected_PNG_records=400,maximum_filter_difference=worst,
        original_precision_difference=old_difference,unchanged_decisions=True,seconds=time.monotonic()-start),flush=True)


if __name__=='__main__':
    main()
