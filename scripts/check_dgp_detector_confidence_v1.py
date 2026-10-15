"""Probability-boundary and forged-mask regressions; no neural inference."""
from pathlib import Path
import importlib.util
import json
import numpy as np
from scipy.ndimage import maximum_filter


def main():
    source=Path(__file__).with_name('audit_dgp_detector_confidence_v1.py')
    assert source.is_file(),'Independent detector-confidence auditor is not implemented'
    spec=importlib.util.spec_from_file_location('confidence_auditor',source);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    p=np.full((256,256),.25,np.float32);p[80:90,100:110]=.5
    operator=np.zeros((256,256),bool);operator[80:100,100:120]=True
    protected=np.zeros_like(operator);protected[0:40]=True
    raw=(p>=.5).astype(np.uint8);mask=maximum_filter(raw,size=7,mode='constant',cval=0).astype(np.uint8)
    first=audit.measure(p,raw,mask,operator,protected)
    assert first['raw_pixels']==100 and first['proposal_pixels']==256 and first['missed_operator_pixels']==231 and first['extra_pixels']==87
    assert first['scores']['operator']['selected_fraction']==.25 and first['scores']['protected']['maximum']==.25
    assert audit.measure(p,raw,mask,np.zeros_like(operator),np.zeros_like(operator))['scores']['operator']['pixels']==0
    rejected=[]
    for label,x in [('nonfinite',np.where(np.indices(p.shape)[0]==0,np.nan,p).astype(np.float32)),
                    ('negative',p-np.float32(1)),('greater_than_one',p+np.float32(1)),
                    ('float64',p.astype(np.float64)),('wrong_shape',p[...,None])]:
        try:audit.measure(x,raw,mask,operator,protected)
        except AssertionError:rejected.append(label)
        else:raise AssertionError('Invalid probability accepted: '+label)
    wrong=raw.copy();wrong[80,100]=0
    try:audit.measure(p,wrong,mask,operator,protected)
    except AssertionError:rejected.append('changed_half_threshold')
    else:raise AssertionError('Threshold equality was changed')
    wrong=mask.copy();wrong[0,0]=1
    try:audit.measure(p,raw,wrong,operator,protected)
    except AssertionError:rejected.append('visible_mask_expansion')
    else:raise AssertionError('Forged mask accepted')
    corrupt=dict(first,missed_operator_pixels=0)
    try:audit.compare_measurement(first,corrupt)
    except AssertionError:rejected.append('missed_region_erased')
    else:raise AssertionError('Erased missed region accepted')
    assert len(rejected)==8
    out=source.resolve().parents[1]/'outputs/dgp_detector_confidence_v1_checker.json'
    with out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(dict(complete=True,checker_sha256=audit.sha(__file__),auditor_sha256=audit.sha(source),
            invalid_cases_rejected=rejected,actual_half_threshold_equality_preserved=True,
            zero_area_controls_audited=True,synthetic_array_fixture_not_neural_evidence=True,
            neural_or_training_calls=0),f,indent=2);f.write('\n')
    print(dict(complete=True,rejected_corruptions=8,neural_or_training_calls=0))


if __name__=='__main__':main()
