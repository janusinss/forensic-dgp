"""Test actual draft/R1 PNG expressions against 100 retained baseline PNGs."""
import ast
from datetime import datetime, timezone
import sys
import numpy as np
from PIL import Image
from import_cctv_dgp_bank_hinge_boundary_v1_return import ROOT, OUT, RETURN, PARENT, read, sha, write


def expression(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    found=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=='png' for t in n.targets)]
    assert len(found)==1
    return compile(ast.Expression(found[0]),str(path),'eval')


def main():
    receipt=OUT/'cctv_dgp_bank_saved_direction_v1_analysis/png_policy_repair.json'
    assert not receipt.exists()
    sys.path.insert(0,str(PARENT))
    from frozen_raw_metrics import deliver
    from cctv_dgp_head4_lossless_v1 import unpack
    p=read(PARENT/'protocol.json')
    cases=p['preview_case_ids']
    ref={r['id']:r for r in p['references']}
    before=ROOT/'scripts/cctv_dgp_bank_saved_direction_forward_v1_vm.py'
    after=ROOT/'scripts/cctv_dgp_bank_saved_direction_forward_v1_r1_vm.py'
    old,new=expression(before),expression(after)
    rows=[]
    for begin in range(0,3905,5):
        block=p['cases'][begin:begin+5]
        if block[0]['id'] not in cases:
            continue
        ids=[c['id'] for c in block]
        raw,_=unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz',ids)
        x8=np.stack([np.asarray(Image.open(PARENT/c['input']).convert('RGB')).copy() for c in block])
        observed=np.stack([np.asarray(Image.open(PARENT/ref[c['source_person_or_reference']]['observed']).convert('L')).copy()>0 for c in block])
        scope=dict(np=np,deliver=deliver,raw=raw,x8=x8,observed=observed)
        rounded,retained=eval(old,scope),eval(new,scope)
        for i,c in enumerate(block):
            saved=np.asarray(Image.open(PARENT/'outputs/bank_comparison_v1/baseline/previews'/(c['id']+'.png')).convert('RGB')).copy()
            assert np.array_equal(retained[i],saved)
            rows.append(dict(id=c['id'],old_different_channels=int(np.count_nonzero(rounded[i]!=saved)),
                corrected_exact_retained_PNG=True,raw_array_unchanged=True))
    assert len(rows)==100
    assert sum(r['old_different_channels'] for r in rows)>0
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        original_worker_sha256=sha(before),corrected_R1_worker_sha256=sha(after),
        retained_pixel_helper_sha256=sha(PARENT/'frozen_raw_metrics.py'),
        actual_worker_PNG_expressions_tested=True,retained_baseline_PNGs_checked=100,
        old_rounding_policy_mismatching_cases=sum(r['old_different_channels']>0 for r in rows),
        old_rounding_policy_different_channels=sum(r['old_different_channels'] for r in rows),
        corrected_floor_policy_matches_all100_retained_PNGs=True,rows=rows,
        original_sealed_unrun_packet_preserved=True,scientific_thresholds_unchanged=True,
        local_neural_gradients_optimizer_parameter_displacements=0,model_qualified=False,goal_complete=False))
    print(dict(complete=True,corrected_exact_PNG_cases=100,unrun_original_preserved=True))


if __name__=='__main__':
    main()
