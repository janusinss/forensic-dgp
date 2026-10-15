"""Challenge exact prepared gate blocks with old measured gradients and scalars."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from import_cctv_dgp_joint_raw_png_forward_v1_return import ROOT, OUT, read, sha, write


def main():
    receipt=OUT/'cctv_dgp_live_conditioning_v1_r1_gate_fixtures.json';assert not receipt.exists()
    packet=OUT/'cctv_dgp_live_conditioning_v1_r1_vm';p=read(packet/'protocol.json')
    worker=packet/'scripts/cctv_dgp_live_conditioning_v1_r1_vm.py'
    auditor=ROOT/'scripts/audit_cctv_dgp_live_conditioning_v1_r1_return.py'
    transfer=read(OUT/'cctv_dgp_live_conditioning_v1_r1_transfer_audit.json')
    assert transfer['complete'] and transfer['protocol_sha256']==sha(packet/'protocol.json')
    trees=[ast.parse(path.read_text(encoding='utf-8')) for path in [worker,auditor]]
    w=[node for node in ast.walk(trees[0]) if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='source_name']
    a=[node for node in ast.walk(trees[1]) if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='source'
       and isinstance(node.iter,ast.Subscript) and isinstance(node.iter.value,ast.Name) and node.iter.value.id=='p']
    assert len(w)==len(a)==1
    snippets=[compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec')
              for node,path in [(w[0],worker),(a[0],auditor)]]
    sizes=[int(np.prod(p['selected_parameter_shapes'][name])) for name in p['learning_names']]
    edges=np.cumsum([0]+sizes)
    indices=[i for i,name in enumerate(p['learning_names']) if name.startswith(('live.style_conditioner.','live.spatial_conditioners.'))]
    assert len(indices)==12
    old_root=OUT/'cctv_dgp_bank_group_gradients_v1_r1_return/outputs/group_gradients_v1_r1'
    old=read(old_root/'run_results.json');groups={g['key']:g for g in old['groups']}
    measured={};bindings={}
    for source in p['conditioning_gradient_requirement_sources']:
        for group in p['conditioning_gradient_requirement_groups']:
            key=source+'/'+group;record=groups[key];path=old_root/record['file']
            assert sha(path)==record['sha256']
            vector=np.load(path,allow_pickle=False,mmap_mode='r')
            values={p['learning_names'][i]:float(np.linalg.norm(vector[edges[i]:edges[i+1]])) for i in indices}
            assert len(values)==12 and all(v==0 for v in values.values())
            measured[key]=values;bindings[key]=sha(path)
    def accepted(snippet,values):
        namespace={'np':np,'p':p,'live_norms':values if snippet is snippets[0] else {'groups':values}}
        try:exec(snippet,namespace)
        except AssertionError:return False
        return True
    results=[]
    for index,snippet in enumerate(snippets):
        assert not accepted(snippet,measured)
        positive={key:{name:1. for name in values} for key,values in measured.items()}
        assert accepted(snippet,positive)
        for value in [0.,-1.,float('nan'),float('inf')]:
            altered=deepcopy(positive)
            key='dataset/asian_faces/degraded/feature_MSE';name=next(iter(altered[key]))
            altered[key][name]=value
            assert not accepted(snippet,altered)
        results.append(dict(source='worker' if index==0 else 'future_auditor',
            actual_old_closed_conditioning_gradients_rejected=True,
            hypothetical_positive_scalars_accepted=True,zero_negative_NaN_infinite_scalars_rejected=True,
            hypothetical_fixtures_not_VM_returns_or_neural_quality_evidence=True))
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        protocol_sha256=sha(packet/'protocol.json'),worker_sha256=sha(worker),future_auditor_sha256=sha(auditor),
        actual_old_gradient_bindings=bindings,exact_gate_blocks_exercised=True,gate_checks=results,
        all12_conditioning_tensors_required_for_each_source_improvement_and_clear_preservation=True,
        no_local_neural_forward_autograd_or_optimizer=True,no_fabricated_return_arrays=True,
        new_VM_activation_still_unrun=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,gate_blocks=2,old_measured_zero_gradients_rejected=True,
        adversarial_scalar_sets_rejected=8,new_neural_calls=0))


if __name__=='__main__':main()
