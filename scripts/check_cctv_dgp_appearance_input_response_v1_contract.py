"""Synthetic saved-RGB arithmetic regressions; no neural, gradient or training call."""
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert (ROOT/'scripts/cctv_dgp_appearance_input_response_v1_contract.py').is_file(), 'Input projection contract is not implemented'
    from cctv_dgp_appearance_input_response_v1_contract import input_projection
    from cctv_dgp_conditioning_response_v1_contract import response_receipt
    x=np.full((256,256,3),.5,np.float32)
    g=np.zeros_like(x);g[20:40,30:50]=np.float32(.125)
    plus=x.copy();minus=x.copy();plus[20:40,30:50]+=.001;minus[20:40,30:50]-=.001
    linear=lambda a:float((a.astype(np.float64)*g.astype(np.float64)).sum())
    a=input_projection(g,x,plus);b=input_projection(g,x,minus)
    expected=(plus.astype(np.float64)-x.astype(np.float64))*g.astype(np.float64)
    assert a==float(expected.sum()) and b<0<a
    receipt=response_receipt(linear(x),linear(plus),linear(minus),.001,a,b,0)
    assert receipt['resolved'] and receipt['central_comparison_within_diagnostic_bound']
    wrong=response_receipt(linear(x),linear(plus),linear(minus),.001,-a,-b,0)
    assert wrong['central_comparison_within_diagnostic_bound'] is False
    # Asymmetric RGB endpoints can encode upstream curvature; never force minus=-plus.
    unequal=x.copy();unequal[20:40,30:50]-=.0005
    assert abs(a+input_projection(g,x,unequal))>1e-3
    assert input_projection(np.zeros_like(g),x,plus)==0
    rejected=0
    invalid=[(g.astype(np.float64),x,plus),(g,x.astype(np.float64),plus),
             (g[:,1:],x,plus),(g,x,plus*3),(g,x,plus-2)]
    for v in [float('nan'),float('inf')]:
        bad=g.copy();bad[0,0,0]=v;invalid.append((bad,x,plus))
        bad=plus.copy();bad[0,0,0]=v;invalid.append((g,x,bad))
    for args in invalid:
        try:input_projection(*args)
        except (AssertionError,ValueError):rejected+=1
        else:raise AssertionError('Unsafe gradient/image arithmetic accepted')
    assert rejected==9
    print(dict(complete=True,linear_wrong_sign_asymmetric_zero_checks=4,unsafe_inputs_rejected=9,neural_or_gradient_or_training_calls=0))


if __name__=='__main__':main()
