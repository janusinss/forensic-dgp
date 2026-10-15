"""Exercise actual future source/raw/PNG gate expressions on adversarial fixtures."""
import ast
import copy
from datetime import datetime, timezone
from import_cctv_dgp_bank_group_gradients_v1_r1_return import ROOT, OUT, read, sha, write


def main():
    base='cctv_dgp_bank_group_step_check_v1'
    receipt=OUT/(base+'_source_gate_audit.json')
    assert not receipt.exists()
    path=ROOT/'scripts'/f'audit_{base}_return.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignments={n.targets[0].id:n for n in ast.walk(main_node) if isinstance(n,ast.Assign)
                 and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    names=['sources','source_structure','source_pass']
    module=ast.Module(body=[assignments[name] for name in names],type_ignores=[])
    code=compile(ast.fix_missing_locations(module),str(path),'exec')
    p=read(OUT/(base+'_vm')/'protocol.json')
    assert p['additional_source_degraded_structure_gain_required']==.01
    parent=read(OUT/'cctv_dgp_bank_photometric_guard_v1_return/protocol.json')
    sources=sorted({c['source'] for c in parent['cases'] if c['id'] in p['case_ids']})
    assert len(sources)==2
    before={stage:{source+'/degraded':{'landmark_high_frequency_MSE':10.} for source in sources} for stage in ['raw','png']}
    good={stage:{source+'/degraded':{'landmark_high_frequency_MSE':9.85} for source in sources} for stage in ['raw','png']}
    rows=[]
    for label,stage,source,expected in [('both_sources_and_stages_pass',None,None,True),
        ('one_raw_source_misses', 'raw',sources[0],False),('one_PNG_source_misses','png',sources[1],False)]:
        after=copy.deepcopy(good)
        if stage is not None:after[stage][source+'/degraded']['landmark_high_frequency_MSE']=9.95
        context={'p':p,'rows':[{'source':s} for s in sources],'groups_before':before,'groups':after}
        exec(code,context)
        assert context['source_pass']==expected
        if stage is not None:
            # Equal source counts: aggregate gain still reaches1%, but a weak
            # source independently fails. An average cannot hide its regression.
            aggregate_gain=1-sum(after[stage][s+'/degraded']['landmark_high_frequency_MSE'] for s in sources)/20
            assert aggregate_gain>=.009999999999999
        rows.append(dict(fixture=label,synthetic_not_model_evidence=True,expected_pass=expected,
                         observed_pass=context['source_pass']))
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        future_auditor_sha256=sha(path),protocol_sha256=sha(OUT/(base+'_vm')/'protocol.json'),
        actual_future_AST_expressions_executed=True,both_sources_and_both_stages_independently_required=True,
        no_failed_source_hidden_by_passing_average=True,fixtures=rows,
        local_neural_gradient_optimizer_or_model_parameter_displacement_calls=0,model_qualified=False))
    print(dict(complete=True,source_gate_boundary_fixtures=3,local_neural_calls=0))


if __name__=='__main__':
    main()
