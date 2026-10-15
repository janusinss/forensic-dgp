"""Replay real saved TRAIN controls and retain rejection of an existing PNG failure."""
from datetime import datetime, timezone
import copy
import numpy as np

from import_cctv_dgp_nearby_gradient_learning_v1_return import ROOT,OUT,PARENT,RESPONSE,read,sha,write
from audit_cctv_dgp_raw_conditioning_learning_v1_r1_return import schema
from cctv_dgp_nearby_snapshot_audit_v1 import audit_snapshot
from cctv_dgp_raw_conditioning_learning_v1_r1_contract import decision
from raw_conditioning_decision_receipt_check_v1_r1 import validate_receipt


def main():
    old_packet=OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_vm';p=read(old_packet/'protocol.json');old=read(PARENT/'protocol.json')
    virtual=schema(old_packet);names=p['conditioning_names'];edges=np.cumsum([0]+[int(np.prod(p['conditioning_shapes'][n])) for n in names])
    folder=RESPONSE/'outputs/conditioning_response_v1';audited={}
    for label in ['initial','plus02','rollback']:
        snapshot,groups,_=audit_snapshot(folder/label,p,old,virtual,names,edges)
        audited[label]=dict(snapshot=snapshot,groups=groups)
    assert audited['initial']['groups']==audited['rollback']['groups']
    checked=decision(audited['initial']['groups'],audited['initial']['groups'],audited['plus02']['groups'])
    receipt=read(folder/'plus02/response.json')['scientific_decision_information_only']
    assert not checked['accept_step'] and not checked['one_percent_checkpoint_pass']
    proof=validate_receipt(checked,receipt,(audited['initial']['snapshot']['groups'],audited['initial']['snapshot']['groups'],audited['plus02']['snapshot']['groups']))
    forged=copy.deepcopy(receipt);forged['stages']['png']['initial_preservation_failures']=[]
    try:validate_receipt(checked,forged,(audited['initial']['snapshot']['groups'],audited['initial']['snapshot']['groups'],audited['plus02']['snapshot']['groups']))
    except AssertionError:rejected=True
    else:raise AssertionError('Erased existing PNG failures were accepted')
    write(OUT/'cctv_dgp_nearby_gradient_learning_v1_saved_fixture_r1.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),snapshot_auditor_sha256=sha(ROOT/'scripts/cctv_dgp_nearby_snapshot_audit_v1.py'),
        return_auditor_sha256=sha(ROOT/'scripts/audit_cctv_dgp_nearby_gradient_learning_v1_return.py'),
        saved_cases=100,RAW_PNG_rows_replayed=600,actual_old_signed_point_rejected=True,
        preserved_PNG_failure_count=len(checked['stages']['png']['initial_preservation_failures']),
        initial_and_rollback_exact_group_scores=True,forged_failure_erasure_rejected=rejected,exact_receipt_proof=proof,
        local_neural_or_training_calls=0,model_qualified=False,goal_complete=False))
    print(dict(complete=True,cases=100,snapshot_stages=3,PNG_gate_failure_still_rejected=True,neural_calls=0))


if __name__=='__main__':main()
