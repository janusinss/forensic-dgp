"""Read back tested return importer and audit its inherited safety logic."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_spatial_features_v25_preparation'


def sha(file):return hashlib.sha256(file.read_bytes()).hexdigest()


def main():
    start=time.monotonic()
    old_file=ROOT/'scripts/import_cctv_dgp_degraded_detail_v24.py'
    new_file=ROOT/'scripts/import_cctv_dgp_spatial_features_v25.py'
    old=ast.parse(old_file.read_text());new=ast.parse(new_file.read_text())
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):
                for a,b in [('user-returned V25','user-returned V24'),
                    ('ed43355b1fa0bd768d78e4e2b9ed3bd32cc046fe6221629605899a6a9b2e3175','76ab24695a40b411832e2d678c79b0e2a80f6320d4525970b2dd32db2379e114'),
                    ('cctv_dgp_spatial_features_v25_return','cctv_dgp_degraded_detail_v24_return'),
                    ('cctv-dgp-spatial-features-v25','cctv-dgp-degraded-detail-v24'),
                    ('scripts/cctv_dgp_spatial_features_v25_vm.py','scripts/cctv_dgp_degraded_detail_v24.py'),
                    ('scripts/run_v25.sh','scripts/run_v24.sh')]:node.value=node.value.replace(a,b)
            return node
    new=Normalize().visit(new)
    mandatory=[node for node in ast.walk(new) if isinstance(node,ast.Tuple) and len(node.elts)==7 and isinstance(node.elts[0],ast.Constant) and node.elts[0].value=='protocol.json']
    assert len(mandatory)==1
    assert [node.value for node in mandatory[0].elts[4:]]==['cctv_dgp_spatial_features_v25.py','cctv_dgp_spatial_features_v25_prepare.py','cctv_dgp_degraded_objective_v24.py']
    mandatory[0].elts=mandatory[0].elts[:4]
    for flag in ['--expected-sha','--expected-bytes']:
        def calls(tree):return [n for n in ast.walk(tree) if isinstance(n,ast.Call) and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value==flag]
        orig,changed=calls(old),calls(new);assert len(orig)==len(changed)==1
        assert {k.arg for k in changed[0].keywords}==({'help'} if flag=='--expected-sha' else {'type','help'})
        changed[0].keywords=copy.deepcopy(orig[0].keywords)
    assert ast.dump(old)==ast.dump(new), 'Unexpected change to inherited safe importer'
    receipt=json.loads((OUT/'transfer_regression_receipt.json').read_text())
    assert receipt['tests_passed']==10 and receipt['exit_code']==0 and receipt['neural_calls']==receipt['backward_calls']==receipt['optimizer_updates']==0
    for name,digest in receipt['source_bindings_sha256'].items():assert sha(ROOT/name)==digest,name
    # The largest planned full return has at most four 50-case snapshots:
    # raw/PNG/predicted embeddings; 50 targets at0; 250 feature arrays; extra receipts.
    bound=4*(50*3+2)+50+250+40
    assert bound==948 and bound<=1024
    record={'complete':True,'date':'2026-10-06','checker_sha256':sha(Path(__file__)),
        'source_bindings_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in [old_file,new_file,ROOT/'tests/test_cctv_dgp_spatial_features_v25_transfer.py',OUT/'transfer_regression_receipt.json']},
        'inherited_safe_importer_AST_exact_after_declared_changes':True,'return_required_frozen_sources':7,
        'transfer_regressions_verified':10,'planned_four_snapshot_return_member_bound':bound,'member_limit':1024,
        'actual_return_present':False,'independent_trained_quality_audit_pending':True,
        'neural_calls':0,'backward_calls':0,'optimizer_updates':0,'VM_actions':False,'seconds':time.monotonic()-start}
    with (OUT/'independent_transfer_preparation_audit.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
