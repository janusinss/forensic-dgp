"""Independent thin-packet/single-change/source/host-guard checks; no training."""
import ast
import copy
import hashlib
import json
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/cctv_dgp_spatial_features_vm_v25'
NEW=ROOT/'outputs/cctv_dgp_batchmatched_identity_vm_v26'
OUT=ROOT/'outputs/cctv_dgp_batchmatched_identity_v26_preparation'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def fn(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)


def source_contracts():
    old=ast.parse((OLD/'scripts/cctv_dgp_spatial_features_v25_vm.py').read_text())
    new=ast.parse((NEW/'scripts/cctv_dgp_batchmatched_identity_v26_vm.py').read_text())
    class Names(ast.NodeTransformer):
        def visit_Constant(self,n):
            if isinstance(n.value,str):
                for revised,original in [('dgp-spatial-batchmatched-identity-capacity-v26','dgp-spatial-feature-capacity-v25'),
                    ('scripts/cctv_dgp_batchmatched_identity_v26_vm.py','scripts/cctv_dgp_spatial_features_v25_vm.py'),
                    ('scripts/run_v26.sh','scripts/run_v25.sh'),('cctv_dgp_batchmatched_identity_v26_return','cctv_dgp_spatial_features_v25_return'),
                    ('cctv-dgp-batchmatched-identity-v26','cctv-dgp-spatial-features-v25'),('V26 update','V25 update')]:n.value=n.value.replace(revised,original)
            return n
        def visit_ImportFrom(self,n):
            if n.module=='cctv_dgp_batchmatched_identity_v26':
                assert [v.name for v in n.names]==['cohort_normalizers','objective_terms']
                n.module='cctv_dgp_degraded_objective_v24'
            return n
    normalized=Names().visit(copy.deepcopy(new))
    for name in ['sha','write','verify','require_vm','run','main']:assert dump(fn(old,name))==dump(fn(normalized,name)),name
    wrapper=ast.parse('''def prepare(root, p):
    require_vm(root, idle=True)
    from cctv_dgp_spatial_features_v25_prepare import prepare_features
    from cctv_dgp_batchmatched_identity_v26_preflight import prove
    head, identity, items, preflight = prepare_features(root, p, require_vm)
    preflight['batchmatched_identity_proof'] = prove(root, p, head, identity, items, require_vm)
    preflight['neural_forward_counts'] = dict(head.audit_forward_counts)
    return head, identity, items, preflight
''').body[0]
    assert dump(fn(new,'prepare'))==dump(wrapper),'Exact new proof wrapper'
    export=fn(normalized,'export')
    additions=ast.parse('''files+=sorted(root.glob('batchmatched_identity_preflight_*.json'))
files+=[root/name for name in ['cctv_dgp_batchmatched_identity_v26.py', 'cctv_dgp_batchmatched_identity_v26_preflight.py', 'scripts/install_v26.py', 'installation_receipt.json']]
''').body
    for node in additions:
        matches=[n for n in export.body if dump(n)==dump(node)];assert len(matches)==1
        export.body.remove(matches[0])
    assert dump(export)==dump(fn(old,'export')),'Only new proof/source/install exports'
    helper=ast.parse((NEW/'cctv_dgp_batchmatched_identity_v26.py').read_text())
    original=ast.parse((OLD/'cctv_dgp_degraded_objective_v24.py').read_text())
    objective=copy.deepcopy(fn(helper,'objective_terms'));baseline=fn(original,'objective_terms')
    assert len(objective.body)==len(baseline.body)
    assert dump(objective.body[3])==dump(ast.parse('base_cosine,cosine=batchmatched_scores(b,pred,identity)').body[0])
    objective.body[3]=copy.deepcopy(baseline.body[3])
    call=objective.body[-1].value
    assert isinstance(call,ast.Call) and isinstance(call.args[7],ast.Name) and call.args[7].id=='base_cosine'
    call.args[7]=copy.deepcopy(baseline.body[-1].value.args[7])
    assert dump(objective)==dump(baseline),'All six other objective terms and coefficients unchanged'
    score=fn(helper,'batchmatched_scores');text=ast.unparse(score)
    assert text.count('identity.embedding(')==1 and 'baseline.detach()' in text and 'vectors[:n].detach()' in text
    assert 'torch.cat((baseline.detach(), prediction), 0)' in text and "b['truth'].detach()" in text
    proof=ast.parse((NEW/'cctv_dgp_batchmatched_identity_v26_preflight.py').read_text())
    attrs={n.func.attr for n in ast.walk(proof) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not attrs.intersection({'backward','step','save'}) and 'optim' not in {n.attr for n in ast.walk(proof) if isinstance(n,ast.Attribute)}
    prooftext=(NEW/'cctv_dgp_batchmatched_identity_v26_preflight.py').read_text()
    for value in ['range(0,50,5)','gradient_calls==20','torch.count_nonzero(matched)==0','torch.count_nonzero(v)==0',
        'legacy_value>0 and norm>0','state_hash(head)==before','state_hash(identity)==recognizer_before',"assert sys.platform=='linux'",'time.monotonic()-started<180']:
        assert value in prooftext,value
    assert prooftext.index("assert sys.platform=='linux'")<prooftext.index('import torch')
    return True


def main():
    start=time.monotonic();prep=read(OUT/'preparation.json');p=read(NEW/'protocol.json');original=read(OLD/'protocol.json')
    pin=sha(NEW/'protocol.json');assert pin==prep['protocol_sha256']
    for key in ['cases','references','budgets','prospective_gates','fresh_DGP_parity_cases']:
        assert p[key]==original[key],key
    assert p['design']==original['design'] and p['closed_V25_protocol_sha256']==sha(OLD/'protocol.json')
    assert len(p['assets_sha256'])==240 and len(p['transfer_assets_sha256'])==5 and len(p['inherited_assets'])==235
    for name,digest in p['assets_sha256'].items():assert sha(NEW/name)==digest,name
    for name,row in p['inherited_assets'].items():assert sha(OLD/row['source'])==sha(NEW/name)==row['sha256'],name
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest,name
    assert p['identity_preflight_policy']['matched_identity_penalty_exactly_zero'] and p['identity_preflight_policy']['all26_matched_parameter_gradients_exactly_zero']
    assert p['identity_preflight_policy']['component_weight']==5 and p['identity_preflight_policy']['margin']==0
    source_contracts()
    archive=ROOT/'outputs/cctv-dgp-batchmatched-identity-v26-execution.tar.gz'
    assert sha(archive)==prep['archive_sha256'] and archive.stat().st_size==prep['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text().strip()==sha(archive)+'  '+archive.name
    expected={'protocol.json','protocol.sha256'}|set(p['transfer_assets_sha256']);seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar:
            path=PurePosixPath(m.name)
            assert m.isreg() and not m.issparse() and not path.is_absolute() and '..' not in path.parts and path.parts[0]==NEW.name
            assert '\\' not in m.name and ':' not in m.name and path.as_posix()==m.name
            name='/'.join(path.parts[1:]);assert name in expected and name not in seen and m.size<256*1024
            assert tar.extractfile(m).read()==(NEW/name).read_bytes();seen.add(name)
    assert seen==expected and len(seen)==7
    guard_results={}
    commands={'worker_verify':['scripts/cctv_dgp_batchmatched_identity_v26_vm.py','--verify-transfer'],
        'worker_training_rejection':['scripts/cctv_dgp_batchmatched_identity_v26_vm.py','--run'],
        'installer_verify':['scripts/install_v26.py','--verify-transfer'],
        'installer_rejection':['scripts/install_v26.py','--install']}
    for label,(script,mode) in commands.items():
        result=subprocess.run([sys.executable,'-B',str(NEW/script),'--root',str(NEW),'--protocol-sha',pin,mode],text=True,capture_output=True,timeout=20)
        reject='rejection' in label
        assert (result.returncode!=0 and 'Existing Linux VM only' in result.stderr) if reject else result.returncode==0,result.stderr
        guard_results[label]=True
    assert sys.platform=='win32' and not (NEW/'outputs').exists() and not (NEW/'installation_receipt.json').exists() and not (NEW/'installation_failure.json').exists()
    sources=['scripts/cctv_dgp_batchmatched_identity_v26.py','scripts/cctv_dgp_batchmatched_identity_v26_preflight.py',
        'scripts/install_cctv_dgp_batchmatched_identity_v26_vm.py','scripts/prepare_cctv_dgp_batchmatched_identity_v26.py',
        'scripts/cctv_dgp_batchmatched_identity_v26_vm.py','scripts/verify_cctv_dgp_batchmatched_identity_v26.py']
    for name in sources:ast.parse((ROOT/name).read_text(),feature_version=(3,10))
    result={'complete':True,'date':'2026-10-06','checker_sha256':sha(Path(__file__)),
        'protocol_sha256':pin,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'thin_regular_members':7,
        'inherited_original_assets_verified':235,'all_reconstructed_assets_verified':240,
        'same_original_data_head_objective_weights_quality_gates_and_finite_limits':True,
        'single_identity_reference_processing_difference_verified':True,'actual_Windows_guards':guard_results,
        'Python310_sources_parsed':len(sources),'actual_L4_zero_identity_proof_and_training_pending':True,
        'model_calls':0,'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,'VM_actions':False,
        'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-start}
    with (OUT/'independent_packet_source_audit.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
