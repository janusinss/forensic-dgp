"""Independent V24 cohort receipt/filter checks; no training or head prediction."""
import math
from pathlib import Path

from import_cctv_dgp_degraded_detail_v24 import read, require

COHORT_SCALAR_TOLERANCE = 2e-9


def check_receipt(receipt, expected_rows):
    """Check policy, exact float32 cohort arithmetic and independent baseline rows."""
    import numpy as np
    policy = {'complete':True,'cases':50,'clear_controls':10,'degraded_cases':40,
              'normalizer_floor':1e-6,'clear_reward':False,'degraded_weight':1.25,
              'clear_baseline_anchor_weight':.05,'fixed_high_pass_calls':100,'optimizer_constructed':False}
    require(set(receipt) == set(policy) | {'feature_normalizer','interior_normalizer','rows'}, 'Cohort receipt schema differs')
    for key,value in policy.items():
        require(type(receipt[key]) is type(value) and receipt[key] == value, 'Cohort policy differs: '+key)
    rows = receipt['rows']
    require(isinstance(rows,list) and len(rows) == len(expected_rows) == 50, 'Cohort row count differs')
    maximum = 0.
    for row,expected in zip(rows,expected_rows):
        require(set(row) == {'id','clear','feature_MSE','interior_MSE'} and row['id'] == expected['id'] and
                type(row['clear']) is bool and row['clear'] is expected['clear'], 'Cohort ID/control order differs')
        for key in ['feature_MSE','interior_MSE']:
            value = row[key]
            require(type(value) in (int,float) and math.isfinite(value) and value > 0, 'Nonpositive/nonfinite cohort error')
            difference = abs(value-expected[key]); maximum = max(maximum,difference)
            require(difference <= COHORT_SCALAR_TOLERANCE, 'Independent cohort baseline differs: '+row['id']+'/'+key)
    require(len({r['id'] for r in rows}) == 50 and sum(r['clear'] for r in rows) == 10, 'Cohort identities/controls differ')
    degraded = [r for r in rows if not r['clear']]
    scalars = {}
    for field,key in [('feature_MSE','feature_normalizer'),('interior_MSE','interior_normalizer')]:
        value = receipt[key]
        require(type(value) in (int,float) and math.isfinite(value), 'Nonfinite cohort normalizer')
        exact = float(np.float32(max(sum(r[field] for r in degraded)/40,1e-6)))
        require(value == exact, 'Float32 cohort normalizer arithmetic differs: '+key)
        independent = float(np.float32(max(sum(r[field] for r in expected_rows if not r['clear'])/40,1e-6)))
        require(abs(value-independent) <= COHORT_SCALAR_TOLERANCE, 'Independent cohort normalizer differs')
        scalars[key] = value
    return {'receipt_present':True,'policy_and_all50_rows_verified':True,'exact_float32_cohort_arithmetic_verified':True,
            'maximum_independent_baseline_row_difference':maximum,'scalar_tolerance':COHORT_SCALAR_TOLERANCE,
            **scalars,'clear_controls':10,'degraded_cases':40,
            'limit':'Only fixed baseline scalar receipt compatibility. Original raw/PNG quality gates and replay tolerances remain exact and unchanged.'}


def audit_cohort(bundle, returned, protocol, head, clock):
    path = returned/'outputs/cohort_loss_setup.json'
    if not path.exists():
        require(not any((returned/'outputs'/name).exists() for name in ['results.json','failure.json','execution_receipt.json','one_batch_gradient_preflight.json']) and
                not list((returned/'outputs').glob('update*')), 'Training evidence lacks frozen cohort setup')
        return {'receipt_present':False,'fixed_CPU_high_pass_calls':0,'head_DGP_recognizer_forwards':0,
                'limit':'No cohort/training evidence; partial preflight/export only, no learning claim'}
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    before = {k:v.clone() for k,v in head.state_dict().items()}
    expected_rows = []
    with torch.inference_mode():
        for case in protocol['cases']:
            clock()
            raw = np.load(bundle/case['raw_dgp'],allow_pickle=False)
            with Image.open(bundle/case['target']) as im:target8 = np.asarray(im).copy()
            with Image.open(bundle/case['observed']) as im:observed = np.asarray(im).copy() > 0
            require(raw.shape == target8.shape == (256,256,3) and raw.dtype == np.float32 and target8.dtype == np.uint8 and
                    observed.shape == (256,256) and np.isfinite(raw).all(), 'Frozen cohort array schema differs')
            base = torch.from_numpy(raw.copy()).permute(2,0,1)[None]
            target = torch.from_numpy(target8.astype(np.float32)/np.float32(255)).permute(2,0,1)[None]
            interior = cv2.erode(observed.astype(np.uint8),np.ones((13,13),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            feature = np.zeros((256,256),bool)
            for point in case['landmarks5_canvas_xy']:
                x,y = np.floor(point).astype(int)
                feature[max(0,y-12):min(256,y+12),max(0,x-12):min(256,x+12)] = True
            feature &= interior
            require(feature.any() and interior.any(), 'Empty cohort support')
            weights = torch.tensor([.299,.587,.114],dtype=torch.float32)[None,:,None,None]
            delta = head.high((base*weights).sum(1,keepdim=True))-head.high((target*weights).sum(1,keepdim=True))
            square = delta.square()
            def error(mask):
                m = torch.from_numpy(mask.astype(np.float32))[None,None]
                return float((square*m).sum()/m.sum())
            expected_rows.append({'id':case['id'],'clear':case['profile']=='clear',
                                  'feature_MSE':error(feature),'interior_MSE':error(interior)})
    require(all(torch.equal(before[k],v) for k,v in head.state_dict().items()) and
            not any(v.requires_grad or v.grad is not None for v in head.parameters()), 'Cohort check mutated model/gradients')
    receipt = check_receipt(read(path),expected_rows)
    receipt.update({'fixed_CPU_high_pass_calls':100,'head_DGP_recognizer_forwards':0,'model_state_unchanged':True,
                    'backward_calls':0,'optimizer_updates':0})
    return receipt
