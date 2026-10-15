"""Independent saved-direction KKT, transfer, source and Windows rejection audit."""
import ast
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np
import torch

from import_cctv_dgp_bank_hinge_boundary_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write


def main():
    receipt = OUT/'cctv_dgp_bank_saved_direction_forward_v1_transfer_audit.json'
    assert not receipt.exists()
    prep_path = OUT/'cctv_dgp_bank_saved_direction_forward_v1_preparation.json'
    prep = read(prep_path)
    assert prep['complete']
    packet = OUT/prep['name']
    p = read(packet/'protocol.json')
    assert sha(packet/'protocol.json') == prep['protocol_sha256']
    archive = OUT/(prep['stem']+'-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256']
    assert Path(str(archive)+'.sha256').read_bytes() == (prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    with tarfile.open(archive,'r:gz') as stream:
        members = stream.getmembers()
        expected = {prep['name']+'/'+name for name in p['assets_sha256']} | {prep['name']+'/protocol.json'}
        assert len(members) == 5 and {m.name for m in members} == expected
        for member in members:
            assert member.isfile() and not member.islnk() and not member.issym()
            assert stream.extractfile(member).read() == (OUT/member.name).read_bytes()
    for name,digest in p['evidence_sha256'].items():
        assert sha(ROOT/name) == digest and read(ROOT/name)['complete']
    worker = packet/'scripts/cctv_dgp_bank_saved_direction_forward_v1_vm.py'
    tree = ast.parse(worker.read_text(encoding='utf-8'),feature_version=(3,10))
    forbidden = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr in ['grad','backward','step','Adam','AdamW','SGD','enable_vm_learning']]
    assert not forbidden
    spec = importlib.util.spec_from_file_location('safe_forward_transfer',worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual,old = module.verify(packet,PARENT,prep['protocol_sha256'])
    assert actual == p
    assert p['paired_TRAIN_cases'] == len(p['case_ids']) == 100
    assert p['preserved_scientific_thresholds'] == old['scientific_thresholds']
    run_path = RETURN/'outputs/loss_alignment_v1/run_results.json'
    parent_audit = read(ANALYSIS/'independent_audit.json')
    assert sha(run_path) == parent_audit['run_results_sha256']
    run = read(run_path)
    selected = [r for r in run['rows'] if r['state']=='initial' and r['sampled_gradient_file']]
    assert len(selected) == 2
    torch.set_grad_enabled(False)
    initial = torch.load(PARENT/'initializers/initial_B.pth',map_location='cpu',weights_only=True)
    names = run['learning_names']
    values = np.concatenate([initial[name].numpy().ravel() for name in names])
    gradients = []
    for row in selected:
        file = RETURN/'outputs/loss_alignment_v1'/row['sampled_gradient_file']
        with np.load(file,allow_pickle=False) as source:
            gradients.append([np.concatenate([source[f'g{j:02d}_p{i:02d}'].ravel() for i in range(37)]) for j in range(9)])
    gradient = np.zeros_like(values)
    for reference in gradients:
        subtotal = np.zeros_like(values)
        for j in [0,1,2,3,5,6]:
            subtotal += reference[j]
        gradient += subtotal
    # The sealed direction sums references before division. Moving division into
    # the sum changes float32 subnormals; keep the physical arithmetic exact.
    gradient /= np.float32(2)
    factor = np.float32(min(1,1/(np.sqrt(np.square(gradient.astype(np.float64)).sum())+1e-6)))
    effective = gradient*factor+values*np.float32(1e-5)
    m = np.float32(.1)*effective
    v = np.float32(.001)*np.square(effective)
    ordinary = -(m/np.float32(.1))/(np.sqrt(v/np.float32(.001))+np.float32(1e-8))
    q = read(packet/'direction_review.json')
    with np.load(packet/'directions.npz',allow_pickle=False) as saved:
        assert len(saved.files) == 74
        flat = {label:np.concatenate([saved[f'{label}_p{i:02d}'].ravel() for i in range(37)])
                for label in ['ordinary','constrained']}
    assert np.array_equal(flat['ordinary'],ordinary)
    ix = [3,7,8,1,2]
    constraints = np.stack([reference[j] for reference in gradients for j in ix]).astype(np.float64)
    norm = np.sqrt(np.square(constraints).sum(1))
    unit = constraints/norm[:,None]
    gram = unit@unit.T
    assert np.allclose(gram,q['normalized_Gram'],rtol=1e-10,atol=1e-10)
    fractions = np.array([.01,.0001,.0001,.0001,.0001]*2)
    scalar = np.array([row['component_values'][j] for row in selected for j in ix])
    target = -fractions*scalar/(1e-4*norm)
    assert np.allclose(target,q['normalized_targets'],rtol=1e-10,atol=1e-10)
    dual = np.array(q['dual_multipliers'])
    assert np.all(dual >= 0)
    expected = ordinary.astype(np.float64)-unit.T@dual
    assert np.array_equal(expected.astype(np.float32),flat['constrained'])
    slack = unit@expected-target
    assert np.max(slack) < 1e-7 and np.max(abs(dual*slack)) < 1e-5
    assert np.linalg.eigvalsh(gram)[0] > 1e-9
    assert np.linalg.norm(expected)/np.linalg.norm(ordinary.astype(np.float64)) < 2
    # Re-opened bytes meet the recorded first-order targets to numerical precision.
    predicted = -1e-4*(constraints@flat['constrained'].astype(np.float64))/scalar
    assert np.all(predicted >= fractions-1e-9)
    rejected = subprocess.run([sys.executable,'-B',str(worker),'--root',str(packet),'--parent-root',str(PARENT),
        '--protocol-sha',prep['protocol_sha256'],'--run'],capture_output=True,text=True,timeout=15)
    assert rejected.returncode != 0 and 'platform.system()' in rejected.stderr and 'AssertionError' in rejected.stderr
    assert not (packet/'outputs').exists()
    write(OUT/'cctv_dgp_bank_saved_direction_forward_v1_windows_rejection.json',dict(complete=True,
        code=rejected.returncode,stderr=rejected.stderr,pre_neural_rejection=True,local_gradient_or_optimizer_calls=0))
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name) == digest for name,digest in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        preparation_sha256=sha(prep_path),protocol_sha256=prep['protocol_sha256'],archive_members_checked=5,
        assets_checked=4,parent_hashes_checked=5852,LF_checksum_checked=True,Python310_AST_checked=True,
        actual_safe_worker_transfer_function_checked=True,Windows_run_rejected_before_neural_import=True,
        independent_ordinary_direction_reconstruction_exact=True,constrained_direction_saved_bytes_exact=True,
        sum_before_division_rounding_preserved=True,
        arithmetic_order_note_sha256=sha(OUT/'cctv_dgp_bank_saved_direction_v1_analysis/independent_audit_arithmetic_order.json'),
        ten_constraint_Gram_targets_and_convex_KKT_independently_recomputed=True,
        numerical_first_order_checks_are_not_scientific_output_gates=True,
        first_order_minimum_sampled_detail_fraction=float(min(predicted[::5])),
        frozen_source_initialization_and_loss_scales_bound=True,
        full100_case_raw_PNG_and_finite_preservation_unproven=True,
        no_neural_gradient_optimizer_or_parameter_displacement_locally=True,
        no_agent_VM_connection_or_launch=True,protected_bindings_unchanged=True,
        actual_VM_forward_unrun=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,archive_members=5,parent_assets=5852,independent_constraint_checks=10,
        actual_VM_forward_unrun=True))


if __name__=='__main__':
    main()
