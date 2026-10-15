"""Synthetic scalar response fixtures; no model, derivative or weight proposal."""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    a=argparse.ArgumentParser();a.add_argument('--receipt',type=Path,required=True);args=a.parse_args()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('cctv_dgp_conditioning_response_v1_contract.py')
    record=dict(complete=False,scope='Declared synthetic scalar fixtures only',new_neural_or_training_calls=0)
    try:
        assert source.is_file(),'Response contract not implemented'
        spec=importlib.util.spec_from_file_location('response_contract',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        f=module.response_receipt
        linear=f(1.,1.02,.98,.1,.02,-.02,0.)
        assert linear['resolved'] and linear['central_comparison_within_diagnostic_bound']
        assert abs(linear['observed_central_slope']-.2)<1e-12
        rounded=f(1.,1.018,.981,.1,.018,-.019,0.)
        assert rounded['central_comparison_within_diagnostic_bound']
        curved=f(1.,1.025,.985,.1,.02,-.02,0.)
        assert curved['central_comparison_within_diagnostic_bound'] and abs(curved['even_response']-.005)<1e-12
        reversed_=f(1.,.98,1.02,.1,.02,-.02,0.)
        assert reversed_['resolved'] and not reversed_['central_comparison_within_diagnostic_bound']
        noisy=f(1.,1.00000002,.99999998,.1,2e-8,-2e-8,1e-6)
        assert not noisy['resolved'] and noisy['central_comparison_within_diagnostic_bound'] is None
        stationary=f(1.,1.,1.,.1,0.,0.,0.)
        assert not stationary['resolved'] and stationary['central_comparison_within_diagnostic_bound'] is None
        rejected=[]
        for name,values in [('NaN',(1.,float('nan'),.98,.1,.02,-.02,0.)),
                            ('Inf',(1.,1.02,.98,.1,float('inf'),-.02,0.)),
                            ('zero_magnitude',(1.,1.02,.98,0.,.02,-.02,0.)),
                            ('negative_noise',(1.,1.02,.98,.1,.02,-.02,-1.))]:
            try:f(*values)
            except AssertionError:rejected.append(name)
            else:raise AssertionError(name)
        assert all(v['model_qualified'] is False and v['preservation_gate_relaxed'] is False for v in [linear,rounded,curved,reversed_,noisy,stationary])
        record.update(complete=True,fixtures=dict(linear=linear,actual_rounded_predictions=rounded,quadratic_even_response=curved,
                      wrong_direction=reversed_,noise_unresolved=noisy,stationary_unresolved=stationary),invalid_inputs_rejected=rejected)
    except Exception as e:
        record['failure']=str(e)
        with args.receipt.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
        raise
    with args.receipt.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,scalar_fixtures=6,invalid_inputs_rejected=rejected,new_neural_or_training_calls=0))


if __name__=='__main__':main()
