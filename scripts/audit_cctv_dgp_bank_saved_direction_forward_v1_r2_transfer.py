"""Check the actual repaired alignment helper, unchanged science and sealed transfer."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import subprocess
import sys
import tarfile
import cv2
import numpy as np

from import_cctv_dgp_bank_saved_direction_forward_v1_r1_return import (
    ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write,
)


def main():
    receipt=OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_transfer_audit.json'
    assert not receipt.exists()
    prep_path=OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_preparation.json'
    prep=read(prep_path)
    assert prep['complete'] and prep['actual_VM_forward_unrun']
    packet=OUT/prep['name']
    p=read(packet/'protocol.json')
    previous=OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_vm'
    before=read(previous/'protocol.json')
    assert sha(packet/'protocol.json')==prep['protocol_sha256']
    archive=OUT/(prep['stem']+'-execution.tar.gz')
    assert sha(archive)==prep['archive_sha256']
    assert (OUT/(archive.name+'.sha256')).read_bytes()==(prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers()
        expected={prep['name']+'/'+n for n in p['assets_sha256']}|{prep['name']+'/protocol.json'}
        assert len(members)==5 and {m.name for m in members}==expected
        for m in members:
            assert m.isfile() and not m.islnk() and not m.issym()
            assert stream.extractfile(m).read()==(OUT/m.name).read_bytes()
    allowed={'name','assets_sha256','evidence_sha256','supersedes_failed_R1_protocol_sha256',
        'supersedes_failed_R1_return_archive_sha256','alignment_policy',
        'all100_fixed_affines_shape_finite_determinant_checked_before_neural_import',
        'raw_output_saved_before_identity_measurement','recognizer_counter_incremented_after_each_actual_forward','repair_scope'}
    assert all(p[key]==value for key,value in before.items() if key not in allowed)
    assert set(p)-set(before)<=allowed
    for key in ['proposals','case_ids','learning_names','parent_required_sha256','preserved_scientific_thresholds',
                'minimum_free_bytes','disk_reserve_bytes','output_byte_cap','peak_VRAM_bytes',
                'worker_seconds','measurement_seconds','external_seconds','export_seconds','export_external_seconds',
                'model_forward_cap','recognizer_forward_cap','optimizer_updates','gradient_queries','full_epochs']:
        assert p[key]==before[key]
    assert sha(packet/'directions.npz')==sha(previous/'directions.npz')
    assert sha(packet/'direction_review.json')==sha(previous/'direction_review.json')
    for name,digest in p['evidence_sha256'].items():
        assert sha(ROOT/name)==digest and read(ROOT/name)['complete']
    worker=packet/'scripts/cctv_dgp_bank_saved_direction_forward_v1_r2_vm.py'
    tree=ast.parse(worker.read_text(encoding='utf-8'),feature_version=(3,10))
    old_worker=previous/'scripts/cctv_dgp_bank_saved_direction_forward_v1_r1_vm.py'
    old_tree=ast.parse(old_worker.read_text(encoding='utf-8'),feature_version=(3,10))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                   n.func.attr in ['grad','backward','step','Adam','AdamW','SGD','enable_vm_learning'] for n in ast.walk(tree))
    for name in ['candidate','identity.embedding','params[name].copy_']:
        current=[ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                 if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
        original=[ast.dump(n,include_attributes=False) for n in ast.walk(old_tree)
                  if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
        assert current==original and current, name
    grid_calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='grid112']
    assert len(grid_calls)==1 and ast.unparse(grid_calls[0].args[0])=='matrix'
    parents=[n for n in ast.walk(tree) if isinstance(n,ast.ListComp) and
             isinstance(n.elt,ast.Call) and ast.unparse(n.elt.func)=='grid112']
    assert len(parents)==1 and ast.unparse(parents[0].generators[0].iter)=='fixed_reference_matrices(cs, refs)'
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    source_lines=worker.read_text(encoding='utf-8').splitlines()
    raw_save=next(n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='np.save')
    embeddings=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='identity.embedding']
    assert len(embeddings)==2 and raw_save<min(embeddings)
    assert source_lines[embeddings[0]].strip()=="counts['recognizer'] += 1"
    assert source_lines[embeddings[1]].strip()=="counts['recognizer'] += 1"
    spec=importlib.util.spec_from_file_location('safe_forward_R2',worker)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual,old=module.verify(packet,PARENT,prep['protocol_sha256'])
    assert actual==p
    refs={r['id']:r for r in old['references']}
    cases=[c for c in old['cases'] if c['id'] in p['case_ids']]
    matrices=module.fixed_reference_matrices(cases,refs)
    assert len(matrices)==100
    pilot=PARENT/'cctv_dgp_pilot.py'
    pilot_tree=ast.parse(pilot.read_text(encoding='utf-8'))
    node=next(n for n in pilot_tree.body if isinstance(n,ast.FunctionDef) and n.name=='grid112')
    context={'np':np,'cv2':cv2}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(pilot),'exec'),context)
    grid112=context['grid112']
    yy,xx=np.meshgrid(np.arange(112),np.arange(112),indexing='ij')
    geometry=[]
    for c,matrix in zip(cases,matrices,strict=True):
        assert matrix==refs[c['source_person_or_reference']]['matrix112']
        grid=grid112(matrix)
        assert grid.shape==(112,112,2) and grid.dtype==np.float32 and np.isfinite(grid).all()
        a,b,tx=matrix[0];d,e,ty=matrix[1]
        determinant=a*e-b*d
        x=(e*(xx-tx)-b*(yy-ty))/determinant
        y=(-d*(xx-tx)+a*(yy-ty))/determinant
        direct=np.stack((2*(x+.5)/256-1,2*(y+.5)/256-1),axis=-1).astype(np.float32)
        error=float(abs(direct.astype(np.float64)-grid).max())
        assert error<5e-7  # Geometry arithmetic check, unrelated to scientific image gates.
        geometry.append(dict(id=c['id'],grid_sha256=hashlib.sha256(grid.tobytes()).hexdigest(),independent_equation_error=error))
    prior_geometry=read(ANALYSIS/'failure_audit.json')['fixed_geometry']
    assert [(r['id'],r['grid_sha256']) for r in geometry]==[(r['id'],r['fixed_grid_float32_sha256']) for r in prior_geometry]
    invalid=[]
    for label,matrix in [('landmarks_are_not_an_affine',cases[0]['landmarks5_canvas_xy']),
                         ('nonfinite_matrix',[[float('nan'),0,0],[0,1,0]]),
                         ('singular_matrix',[[1,2,3],[2,4,6]])]:
        bad=copy.deepcopy(refs)
        bad[cases[0]['source_person_or_reference']]['matrix112']=matrix
        try:
            module.fixed_reference_matrices(cases[:1],bad)
        except ValueError as error:
            invalid.append(dict(test=label,rejected=True,message=str(error)))
        else:
            raise AssertionError(label)
    rejected=subprocess.run([sys.executable,'-B',str(worker),'--root',str(packet),'--parent-root',str(PARENT),
        '--protocol-sha',prep['protocol_sha256'],'--run'],capture_output=True,text=True,timeout=20)
    assert rejected.returncode and 'require_vm' in rejected.stderr and 'AssertionError' in rejected.stderr
    assert 'run(root, parent, p, old' not in rejected.stderr and not (packet/'outputs').exists()
    write(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_windows_rejection.json',dict(complete=True,
        code=rejected.returncode,stderr=rejected.stderr,pre_neural_rejection=True,local_gradient_or_optimizer_calls=0))
    imported=read(ANALYSIS/'import.json')
    assert sha(OUT/'cctv-dgp-bank-saved-direction-forward-v1-r1-results.tar.gz')==imported['archive_sha256']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        preparation_sha256=sha(prep_path),protocol_sha256=prep['protocol_sha256'],
        archive_members_checked=5,assets_checked=4,parent_hashes_checked=5852,LF_checksum_checked=True,
        Python310_AST_checked=True,actual_safe_worker_transfer_function_checked=True,
        actual_fixed_matrix_helper_checked_on100_cases=True,all100_grid_hashes_match_retained_alignment=True,
        independent_geometry_equations_checked=geometry,invalid_matrix_boundary_checks=invalid,
        raw_saved_before_identity_measurement=True,recognizer_counts_follow_each_completed_forward=True,
        neural_forward_and_parameter_displacement_call_ASTs_unchanged=True,
        directions_exactly_unchanged=True,scientific_gates_proposals_inputs_and_finite_limits_unchanged=True,
        Windows_run_rejected_before_neural_import=True,R1_failure_retained=True,protected_bindings_unchanged=True,
        no_local_neural_gradient_optimizer_or_parameter_displacement=True,no_agent_VM_connection_or_launch=True,
        actual_VM_forward_unrun=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,actual_alignment_cases=100,invalid_matrix_boundary_checks=3,
        archive_members=5,actual_VM_forward_unrun=True))


if __name__=='__main__':
    main()
