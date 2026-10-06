"""Independent saved loss-term/group arithmetic check; no models or fitting."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.monotonic()
    folder = ROOT/'outputs/cctv_dgp_detail_skip_v23_loss_audit_v1'
    result = json.loads((folder/'results.json').read_text())
    p = json.loads((ROOT/'outputs/cctv_dgp_detail_skip_vm_v23/protocol.json').read_text())
    for name, expected in result['source_bindings_sha256'].items():
        path = (ROOT/name).resolve(); assert path.is_relative_to(ROOT) and sha(path)==expected
    assert [r['id'] for r in result['rows']]==[c['id'] for c in p['cases']]
    maximum = 0.
    for row, case in zip(result['rows'], p['cases']):
        assert row['source']==case['source'] and row['profile']==case['profile']
        for state in row['states'].values():
            difference=abs(sum(state['terms'].values())-state['objective']);maximum=max(maximum,difference)
            assert difference<=1e-6 and abs(state['compiled_objective']-state['objective'])<=1e-6
    for name, group in result['groups'].items():
        rows=[r for r in result['rows'] if name=='all' or (r['profile']=='clear')==(name=='clear')]
        assert len(rows)==group['cases']
        for update in ['0','50']:
            objective=sum(r['states'][update]['objective'] for r in rows)/len(rows)
            assert abs(objective-group['states'][update]['objective'])<1e-12
            for term,value in group['states'][update]['terms'].items():
                assert abs(sum(r['states'][update]['terms'][term] for r in rows)/len(rows)-value)<1e-12
        reduction=group['states']['0']['objective']-group['states']['50']['objective']
        assert abs(reduction-group['objective_reduction'])<1e-12
        assert abs(reduction*len(rows)/50-group['contribution_to_equal50_case_objective_reduction'])<1e-12
    clear=result['groups']['clear']['contribution_to_equal50_case_objective_reduction']
    degraded=result['groups']['degraded']['contribution_to_equal50_case_objective_reduction']
    all_change=result['groups']['all']['objective_reduction']
    assert abs(clear+degraded-all_change)<1e-12 and clear>0 and degraded<0 and all_change>0
    assert result['counts']=={'recognizer_forwards':210,'saved_prediction_replays':100}
    assert result['backward_calls']==result['optimizer_updates']==result['head_forwards']==0
    receipt={'complete':True,'date':'2026-10-06','checker_sha256':sha(Path(__file__)),
        'results_sha256':sha(folder/'results.json'),'source_bindings_verified':len(result['source_bindings_sha256']),
        'case_states_verified':100,'group_states_verified':6,'maximum_term_sum_difference':maximum,
        'clear_contribution_fraction_of_net_objective_reduction':clear/all_change,
        'clear_contribution':clear,'degraded_contribution':degraded,'overall_reduction':all_change,
        'limit':'Verifies saved decomposition arithmetic/known-source reference, not GPU step gradients or independent recomputation of recognizer forwards',
        'neural_calls':0,'backward_calls':0,'optimizer_updates':0,'goal_complete':False,'seconds':time.monotonic()-started}
    with (folder/'independent_saved_loss_audit.json').open('x',encoding='utf-8',newline='\n') as out:
        out.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
