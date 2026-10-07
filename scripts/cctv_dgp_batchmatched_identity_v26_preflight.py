"""L4-only zero-baseline penalty/gradient proof before constructing an optimizer."""
import json
from pathlib import Path
import sys
import time


def prove(root,p,head,identity,items,require_vm):
    assert sys.platform=='linux' and root.is_relative_to((Path.home()/'forensic-dgp').resolve())
    require_vm(root,idle=True)
    import torch
    from torch.nn import functional as F
    from cctv_dgp_pilot import state_hash
    from cctv_dgp_batchmatched_identity_v26 import batchmatched_scores
    started=time.monotonic();before=state_hash(head);recognizer_before=state_hash(identity)
    parameters=list(head.parameters());assert len(items)==50 and len(parameters)==26
    assert all(v.requires_grad and v.grad is None for v in parameters)
    legacy_gradient=torch.zeros(53781,dtype=torch.float64,device='cuda');rows=[]
    legacy_value=0.;new_value=0.;gradient_calls=0
    for begin in range(0,50,5):
        torch.cuda.synchronize();assert time.monotonic()-started<180,'Identity proof cap180s'
        b={k:torch.cat([items[i][k] for i in range(begin,begin+5)]) for k in ['x','base','mask','grid','truth','base_cosine']}
        b['fpn']=tuple(torch.cat([items[i]['fpn'][j] for i in range(begin,begin+5)]) for j in range(5))
        pred=head(b['x'],b['base'],b['mask'],b['fpn'])
        assert torch.equal(pred,b['base']),'Exact baseline outputs required before identity proof'
        legacy_cosine=(identity.embedding(pred*b['mask']+b['x']*(1-b['mask']),b['mask'],b['grid'])*b['truth']).sum(1)
        legacy=5*F.relu(b['base_cosine']-legacy_cosine)
        reference,current=batchmatched_scores(b,pred,identity);matched=5*F.relu(reference-current)
        assert torch.equal(reference,current) and torch.count_nonzero(matched)==0,'Identical images must have exactly zero matched identity penalty'
        old_parts=torch.autograd.grad(legacy.mean()/10,parameters,retain_graph=True,create_graph=False,allow_unused=False);gradient_calls+=1
        new_parts=torch.autograd.grad(matched.mean()/10,parameters,retain_graph=False,create_graph=False,allow_unused=False);gradient_calls+=1
        assert all(torch.isfinite(v).all() for v in old_parts+new_parts),'Finite preflight derivatives'
        assert all(torch.count_nonzero(v)==0 for v in new_parts),'Identical images must have exactly zero identity gradients'
        vector=torch.cat([v.detach().reshape(-1).double() for v in old_parts]);legacy_gradient+=vector
        legacy_value+=float(legacy.mean().detach())/10;new_value+=float(matched.mean().detach())/10
        rows.append({'ids':[items[i]['case']['id'] for i in range(begin,begin+5)],
            'exact_cached_baseline_cases':5,'legacy_component_value':float(legacy.mean().detach())/10,
            'legacy_component_gradient_norm':float(torch.linalg.vector_norm(vector)),
            'batchmatched_component_value':0.,'batchmatched_component_gradient_norm':0.,
            'exact_reference_prediction_cosines':True,'all26_matched_gradient_tensors_exactly_zero':True})
    norm=float(torch.linalg.vector_norm(legacy_gradient));assert legacy_value>0 and norm>0,'Legacy baseline discrepancy must reproduce before this pilot'
    assert gradient_calls==20 and new_value==0 and state_hash(head)==before and state_hash(identity)==recognizer_before
    assert all(v.grad is None for v in parameters) and all(not v.requires_grad and v.grad is None for v in identity.parameters())
    assert all(not v.requires_grad and v.grad is None for item in items for v in item['fpn'])
    torch.cuda.synchronize()
    result={'complete':True,'cases':50,'batches':10,'gradient_calls':20,'head_forwards':10,'recognizer_forwards':20,
        'optimizer_constructed':False,'optimizer_updates':0,'legacy_component_value':legacy_value,
        'legacy_component_gradient_norm':norm,'batchmatched_component_value':0.,'batchmatched_component_gradient_norm':0.,
        'baseline_and_prediction_identical_cases':50,'head_state_before_after':before,'recognizer_state_before_after':recognizer_before,
        'head_grad_buffers_empty':True,'frozen_features_and_recognizer_have_no_gradients':True,'rows':rows,
        'same_original_penalty_weight':5,'identity_margin':0,'quality_gate_relaxed':False,'seconds':time.monotonic()-started}
    with (root/('batchmatched_identity_preflight_'+str(time.time_ns())+'.json')).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
