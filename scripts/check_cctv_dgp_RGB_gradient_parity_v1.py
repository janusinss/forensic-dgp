"""Zero-reference report must serialize while keeping every original absolute bound."""
import importlib.util
import json
import numpy as np


def main():
    if importlib.util.find_spec('cctv_dgp_RGB_gradient_parity_v1'):
        from cctv_dgp_RGB_gradient_parity_v1 import report
    else:
        from cctv_dgp_cross_run_gradient_report_v1 import report
    z=np.zeros(12,np.float32);r=report(z,z)
    assert r['relative_vector_difference'] is None,'Relative difference at a zero reference must be explicitly undefined, not NaN'
    assert r['original_parity_check_pass'];json.dumps(r,allow_nan=False)
    bad=z.copy();bad[4]=np.float32(2e-8);r=report(bad,z)
    assert not r['original_parity_check_pass'] and r['mismatched_elements']==1 and not r['norm_check_pass']
    assert r['original_elementwise_atol']==1e-8 and r['original_norm_atol']==1e-10;json.dumps(r,allow_nan=False)
    from cctv_dgp_cross_run_gradient_report_v1 import report as original
    x=np.arange(1,13,dtype=np.float32)/10;assert report(x,x)==original(x,x)
    y=x.copy();y[0]+=1;assert report(y,x)==original(y,x)
    for value in [np.nan,np.inf]:
        a=z.copy();a[2]=value
        try:report(a,z)
        except AssertionError:pass
        else:raise AssertionError('Nonfinite gradient accepted')
    print(dict(complete=True,zero_reference_serializes=True,original_absolute_bounds_retained=True,nonzero_reports_exactly_original=True,neural_calls=0))


if __name__=='__main__':main()
