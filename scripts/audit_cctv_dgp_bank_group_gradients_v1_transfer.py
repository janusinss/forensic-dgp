"""Independently check the sealed manual packet and prospective saved-data audit."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np
from import_cctv_dgp_bank_saved_direction_forward_v1_r2_return import ROOT, OUT, ANALYSIS, PARENT, read, sha, write


def module(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    result = importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def rejected(fn, *args):
    try:
        fn(*args)
    except (AssertionError,ValueError):
        return True
    raise AssertionError('Declared invalid preparation fixture accepted')


def main():
    prep_path = OUT/'cctv_dgp_bank_group_gradients_v1_preparation.json'
    prep = read(prep_path)
    packet = OUT/prep['name'];p = read(packet/'protocol.json')
    assert prep['complete'] and sha(packet/'protocol.json')==prep['protocol_sha256']
    archive = OUT/(prep['stem']+'-execution.tar.gz')
    assert sha(archive)==prep['archive_sha256'] and archive.stat().st_size==prep['archive_bytes']
    checksum = Path(str(archive)+'.sha256').read_bytes()
    assert checksum==f"{prep['archive_sha256']}  {archive.name}\n".encode('ascii') and b'\r' not in checksum
    bindings = {**p['assets_sha256'],'protocol.json':prep['protocol_sha256']}
    assert len(bindings)==6 and all(sha(packet/n)==h for n,h in bindings.items())
    with tarfile.open(archive,'r:gz') as tar:
        members = tar.getmembers()
        assert len(members)==6 and len({m.name.casefold() for m in members})==6
        for m in members:
            assert m.isfile() and not m.issym() and not m.islnk() and m.name.startswith(prep['name']+'/')
            name = m.name[len(prep['name'])+1:]
            assert name in bindings and '..' not in Path(name).parts and '\\' not in name
            with tar.extractfile(m) as stream:
                assert hashlib.sha256(stream.read()).hexdigest()==bindings[name]
    worker_path = packet/'scripts/cctv_dgp_bank_group_gradients_v1_vm.py'
    worker = module(worker_path,'prepared_group_gradient_worker')
    verified,old = worker.verify(packet.resolve(),PARENT.resolve(),prep['protocol_sha256'])
    assert verified==p and len(p['parent_required_sha256'])==5852
    contract = module(packet/'cctv_dgp_bank_group_gradients_v1_contract.py','prepared_group_gradient_contract')
    blocks,groups = contract.scopes(old['cases'],p['case_ids'])
    assert blocks==p['reference_blocks'] and groups==p['gradient_groups'] and len(groups)==32
    selected = [c for c in old['cases'] if c['id'] in p['case_ids']]
    refs = {r['id']:r for r in old['references']}
    assert len(contract.fixed_reference_matrices(selected,refs))==100
    old_worker = OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_vm/scripts/cctv_dgp_bank_saved_direction_forward_v1_r2_vm.py'
    a = next(n for n in ast.parse(old_worker.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='fixed_reference_matrices')
    b = next(n for n in ast.parse((packet/'cctv_dgp_bank_group_gradients_v1_contract.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='fixed_reference_matrices')
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    invalid_affines = []
    case = dict(source_person_or_reference='fixture')
    for label,value in [('landmarks5x2',[[1.,2.]]*5),('nonfinite',[[float('nan'),0,0],[0,1,0]]),('singular',[[0,0,0],[0,0,0]])]:
        assert rejected(contract.fixed_reference_matrices,[case],{'fixture':dict(matrix112=value)})
        invalid_affines.append(dict(fixture=label,rejected=True,declared_fixture_not_model_output=True))
    # Independent contribution accounting on small declared arithmetic fixtures.
    destinations = {g['key']:np.zeros(3,np.float64) for g in groups}
    expected = {g['key']:[] for g in groups}
    for block in blocks:
        for query in range(16):
            key = contract.query_key(block['source'],query)
            v = np.array([block['begin']/10,query,-query],np.float32)
            contract.accumulate(destinations[key],v,10);expected[key].append(v.astype(np.float64))
    assert sum(len(v) for v in expected.values())==320 and all(len(v)==10 for v in expected.values())
    assert all(np.allclose(destinations[k],np.stack(v).mean(0),rtol=0,atol=1e-12) for k,v in expected.items())
    terms = dict(MSE=np.arange(5.),SSIM_loss=np.arange(5.)+10,landmark_structure=np.arange(5.)+20)
    values = contract.batch_values(terms,np.arange(5.)+30)
    assert values==[x for i in range(5) for x in [float(i),float(i+10),float(i+30)]]+[22.5]
    assert contract.projection_seconds([1.,2.])==60.
    assert all(rejected(contract.projection_seconds,x) for x in [[0.,1.],[-1.,1.],[float('nan'),1.],[1.]])
    source = worker_path.read_text(encoding='utf-8');tree = ast.parse(source,feature_version=(3,10))
    forbidden = ['Adam','AdamW','SGD','step','backward','copy_','add_','sub_']
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree))
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='grad' for n in ast.walk(tree))==1
    run = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assert isinstance(run.body[0],ast.Expr) and isinstance(run.body[0].value,ast.Call) and run.body[0].value.func.id=='require_vm'
    assert source.index('require_vm(root,parent)') < source.index('import torch')
    assert 'retain_graph=query<15' in source and 'for query,value in enumerate(values)' in source
    assert 'candidate.load_state_dict(initial,strict=True)' in source and 'full_state.pth' not in source
    blocked = subprocess.run([sys.executable,'-B',str(worker_path),'--root',str(packet),'--parent-root',str(PARENT),
        '--protocol-sha',prep['protocol_sha256'],'--run'],capture_output=True,text=True,timeout=20)
    assert blocked.returncode!=0 and 'AssertionError' in blocked.stderr and 'require_vm' in blocked.stderr
    assert not (packet/'outputs').exists()
    importer = ROOT/'scripts/import_cctv_dgp_bank_group_gradients_v1_return.py'
    assert sha(importer)==prep['importer_sha256']
    checker = ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_return.py'
    audit = module(checker,'prospective_group_saved_auditor')
    actual_before = read(ANALYSIS/'finite_metrics.json')['baseline_rows']
    losses = [audit.baseline_value(g,actual_before) for g in groups]
    assert len(losses)==32 and all(np.isfinite(v) and v>0 for v in losses)
    d = dict(ordinary=np.array([2.,-1.,0.]),constrained=np.array([1.,0.,-1.]))
    v = np.array([.1,-.2,.3],np.float64)
    receipt = dict(gradient_norm=float(np.sqrt(np.square(v).sum())),saved_direction_dot={k:float((v*x).sum()) for k,x in d.items()})
    assert audit.check_vector(v,receipt,d)>0
    invalid_vector_checks = []
    fixtures = [('wrong_dtype',v.astype(np.float32),receipt),('nonfinite',np.array([float('nan'),0.,0.]),receipt),
                ('wrong_norm',v,dict(receipt,gradient_norm=0)),('wrong_dot',v,dict(receipt,saved_direction_dot=dict(ordinary=0,constrained=0)))]
    for label,vector,record in fixtures:
        assert rejected(audit.check_vector,vector,record,d)
        invalid_vector_checks.append(dict(fixture=label,rejected=True,declared_fixture_not_neural_gradient=True))
    assert all(sha(ROOT/n)==h for n,h in read(ANALYSIS/'local_protected_before.json')['files_sha256'].items())
    assert all(sha(packet/n)==h for n,h in bindings.items())
    write(OUT/'cctv_dgp_bank_group_gradients_v1_transfer_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),preparation_sha256=sha(prep_path),protocol_sha256=prep['protocol_sha256'],
        archive_sha256=prep['archive_sha256'],archive_bytes=prep['archive_bytes'],regular_archive_members_checked=6,
        packet_assets_checked=5,parent_hashes_checked=5852,all100_fixed_affines_checked=True,
        actual_affine_helper_AST_identical_to_audited_R2=True,invalid_affines=invalid_affines,
        all320_query_group_bindings_and32_aggregate_divisors_checked=True,
        aggregation_and_timing_fixtures_declared_not_model_results=True,all32_saved_baseline_loss_values_checked=True,
        invalid_vector_checks=invalid_vector_checks,future_auditor_sha256=sha(checker),
        Python310_source_parse_passed=True,Windows_actual_run_rejected_before_neural_import=True,
        Windows_rejection_stdout=blocked.stdout,Windows_rejection_stderr=blocked.stderr,
        checksum_exact_LF=True,optimizer_updates=0,local_neural_gradient_or_parameter_displacement=0,
        manual_VM_execution_unrun=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,groups=32,query_bindings=320,packet_members=6,parent_hashes=5852,actual_VM_unrun=True))


if __name__ == '__main__':
    main()
