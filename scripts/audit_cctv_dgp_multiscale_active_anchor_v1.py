"""Rebuild fixed arithmetic directions and replay signed derivatives by blocks."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_multiscale_active_anchor_v1'
RETURNED = ROOT/'outputs/cctv_dgp_multiscale_calibration_vm_v1_return/outputs'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    start = time.monotonic()
    plan, result = read(OUT/'plan.json'), read(OUT/'results.json')
    assert result['complete'] and result['plan_sha256'] == sha(OUT/'plan.json')
    assert not (OUT/'independent_audit.json').exists()
    for name,digest in plan['bindings'].items():
        assert sha(ROOT/name) == digest,name
    matrix = np.load(RETURNED/'initial_gradients.npy',mmap_mode='r',allow_pickle=False)
    layout = read(RETURNED/'gradient_preflight.json')['layout']
    maximum_direction = maximum_prediction = 0.
    checked = 0
    for contrast in result['contrasts']:
        weights = plan['all3_fixed_contrasts'][contrast['name']]
        indices = [n//5 for n in plan['pools'][contrast['pool']]]
        direction = np.load(ROOT/contrast['direction'],allow_pickle=False)
        assert sha(ROOT/contrast['direction']) == contrast['direction_sha256']
        rebuilt = np.zeros(609219,np.float64)
        # Different accumulation order: terms first, then individual references.
        for term,weight in enumerate(weights):
            if weight:
                component = sum((np.asarray(matrix[i,term],np.float64) for i in indices),
                                start=np.zeros(609219,np.float64))/len(indices)
                rebuilt -= weight*component
        rebuilt /= np.linalg.norm(rebuilt)
        maximum_direction = max(maximum_direction,float(np.abs(direction-rebuilt).max()))
        for index,row in enumerate(contrast['rows']):
            for term,(key,value) in enumerate(row['predictions'].items()):
                blocks = []
                for section in layout:
                    begin,end = section['start'],section['end']
                    blocks.append(float(np.sum(matrix[index,term,begin:end].astype(np.float64)*direction[begin:end],dtype=np.float64)))
                prediction = math.fsum(blocks)
                maximum_prediction = max(maximum_prediction,abs(prediction-value))
                assert (prediction>0) == (value>0) and (prediction<0) == (value<0)
                checked += 1
        assert contrast['positive_clear_MSE_reference_slopes'] == sum(r['predictions']['MSE_clear']>0 for r in contrast['rows'])
        assert contrast['positive_HF_reference_slopes'] == sum(r['predictions']['HF_degraded']>0 for r in contrast['rows'])
        assert time.monotonic()-start < 180
    assert checked == 1680 and maximum_direction < 1e-12 and maximum_prediction < 1e-12
    for key in ['local_neural_calls','local_gradient_queries','local_optimizer_updates','mathematical_optimizer_calls']:
        assert result[key] == 0
    receipt = dict(complete=True, plan_sha256=sha(OUT/'plan.json'), results_sha256=sha(OUT/'results.json'),
        checker_sha256=sha(Path(__file__)), saved_directions_reconstructed=6, signed_products_replayed=1680,
        all_prediction_signs_unchanged=True, maximum_direction_error=maximum_direction,
        maximum_prediction_error=maximum_prediction, local_neural_calls=0,
        local_gradient_queries=0, local_optimizer_updates=0, mathematical_optimizer_calls=0,
        first_order_not_finite_quality=True, model_qualification=False, goal_complete=False,
        seconds=time.monotonic()-start)
    with (OUT/'independent_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True, signed_products=checked, maximum_error=maximum_prediction,
               seconds=time.monotonic()-start),flush=True)


if __name__ == '__main__':
    main()
