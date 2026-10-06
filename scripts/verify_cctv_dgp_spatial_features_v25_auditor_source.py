"""Independent AST readback: retain original full audit and finite quality stops."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_spatial_features_v25_return_audit_preparation'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(node):return ast.dump(node,include_attributes=False)


def main():
    start=time.monotonic()
    oldfile=ROOT/'scripts/audit_cctv_dgp_degraded_detail_v24.py'
    newfile=ROOT/'scripts/audit_cctv_dgp_spatial_features_v25.py'
    old=ast.parse(oldfile.read_text());new=ast.parse(newfile.read_text())
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):
                for a,b in [('Independent V25','Independent V24'),('Frozen local V25','Frozen local V24'),('pinned-local-V25','pinned-local-V24'),
                    ('import_cctv_dgp_spatial_features_v25','import_cctv_dgp_degraded_detail_v24'),
                    ('cctv_dgp_spatial_features_vm_v25','cctv_dgp_degraded_detail_vm_v24'),
                    ('cctv_dgp_spatial_features_v25_return','cctv_dgp_degraded_detail_v24_return'),
                    ('cctv_dgp_spatial_features_v25_independent_audit','cctv_dgp_degraded_detail_v24_independent_audit'),
                    ('dgp-spatial-feature-capacity-v25','dgp-degraded-detail-cohort-capacity-v24'),
                    ('scripts/cctv_dgp_spatial_features_v25_vm.py','scripts/cctv_dgp_degraded_detail_v24.py'),
                    ('scripts/run_v25.sh','scripts/run_v24.sh'),
                    ('audit_cctv_dgp_spatial_features_v25_execution','audit_cctv_dgp_degraded_detail_v24_execution')]:node.value=node.value.replace(a,b)
            return node
        def visit_ImportFrom(self,node):
            node.module=node.module.replace('import_cctv_dgp_spatial_features_v25','import_cctv_dgp_degraded_detail_v24').replace('audit_cctv_dgp_spatial_features_v25_execution','audit_cctv_dgp_degraded_detail_v24_execution')
            return node
    new=Normalize().visit(new)
    new.body[new.body.index(fn(new,'make_head'))]=copy.deepcopy(fn(old,'make_head'))
    verify=fn(new,'verify_bundle')
    for n in ast.walk(verify):
        if isinstance(n,ast.Constant) and n.value==235:n.value=221
    audit=fn(new,'audit_return')
    import_nodes=[n for n in audit.body if isinstance(n,ast.ImportFrom) and n.module=='audit_cctv_dgp_spatial_features_v25_features']
    assignments=[n for n in audit.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='audit_features']
    assert len(import_nodes)==len(assignments)==1
    for n in import_nodes+assignments:audit.body.remove(n)
    report=next(n.value for n in audit.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='report' for t in n.targets))
    for field in ['frozen_features_audit','feature_checker_sha256']:
        i=next(i for i,k in enumerate(report.keys) if isinstance(k,ast.Constant) and k.value==field);report.keys.pop(i);report.values.pop(i)
    i=next(i for i,k in enumerate(report.keys) if isinstance(k,ast.Constant) and k.value=='DGP_replay')
    assert report.values[i].value=='All50 original input-conditioned CPU forwards independently compare250 features within the prospectively frozen numerical bounds; original CUDA receipts and unchanged state also checked.'
    report.values[i].value='Saved4-case CUDA preflight checked; no original DGP forward is run by this local audit.'
    required=[n for n in ast.walk(audit) if isinstance(n,ast.Tuple) and len(n.elts)==7 and isinstance(n.elts[0],ast.Constant) and n.elts[0].value=='protocol.json']
    assert len(required)==1 and [x.value for x in required[0].elts[4:]]==['cctv_dgp_spatial_features_v25.py','cctv_dgp_spatial_features_v25_prepare.py','cctv_dgp_degraded_objective_v24.py']
    required[0].elts=required[0].elts[:4]
    counts=0;order=0;calls=0
    class Allow(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value==53781:node.value=4613
            return node
        def visit_Dict(self,node):
            nonlocal counts
            if any(isinstance(k,ast.Constant) and k.value=='DGP_CPU' for k in node.keys):
                i=next(i for i,k in enumerate(node.keys) if isinstance(k,ast.Constant) and k.value=='DGP_CPU');node.keys.pop(i);node.values.pop(i)
                j=next(i for i,k in enumerate(node.keys) if isinstance(k,ast.Constant) and k.value=='DGP');assert node.values[j].value==50;node.values[j].value=4;counts+=1
            return self.generic_visit(node)
        def visit_Compare(self,node):
            nonlocal order
            if node.comparators and isinstance(node.comparators[0],ast.ListComp) and "'fresh_DGP_parity_cases'" in dump(node.comparators[0]):
                node.comparators[0]=ast.parse("p['fresh_DGP_parity_cases']",mode='eval').body;order+=1
            return self.generic_visit(node)
        def visit_Call(self,node):
            nonlocal calls
            if isinstance(node.func,ast.Name) and node.func.id=='head' and len(node.args)==4:
                assert "frozen_features" in dump(node.args[-1]);node.args.pop();calls+=1
            return self.generic_visit(node)
    new=Allow().visit(new)
    assert (counts,order,calls)==(1,1,1)
    assert dump(old)==dump(new),'Unexpected change to original full audit/gates/replay/time/partial handling'
    prep=json.loads((OUT/'preparation_receipt.json').read_text())
    for n,d in prep['sources_sha256'].items():assert sha(ROOT/n)==d,n
    assert prep['tests_passed']==13
    record={'complete':True,'date':'2026-10-06','checker_sha256':sha(Path(__file__)),
        'original_full_auditor_AST_exact_after_declared_feature_transport_changes':True,
        'quality_replay_metric_state_time_selection_failure_logic_unchanged':True,
        'CUDA_parity_receipt_order_matches_frozen_VM_extraction_order':'Case order filtered by the four protocol parity IDs; exact same cases/2e-6 bound',
        'original_fixed_COHORT_scalar_tolerance':2e-9,'source_bindings_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in
            [oldfile,newfile,ROOT/'scripts/audit_cctv_dgp_spatial_features_v25_features.py',ROOT/'scripts/audit_cctv_dgp_spatial_features_v25_execution.py',OUT/'preparation_receipt.json']},
        'tests_bound':13,'neural_calls':0,'backward_calls':0,'optimizer_updates':0,'seconds':time.monotonic()-start}
    with (OUT/'independent_source_audit.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
