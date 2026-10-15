"""Audit exact sealed bytes, NN call continuity, scope and prospective checkers."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import subprocess
import sys
import tarfile
import numpy as np
import cv2
from import_cctv_dgp_bank_group_gradients_v1_r1_return import ROOT, OUT, ANALYSIS, PARENT, read, sha, write

NAME='cctv_dgp_bank_group_step_check_v1_vm'
BASE='cctv_dgp_bank_group_step_check_v1'
STEM='cctv-dgp-bank-group-step-check-v1'


def main():
    receipt=OUT/(BASE+'_transfer_audit.json')
    assert not receipt.exists()
    prep_path=OUT/(BASE+'_preparation.json')
    prep=read(prep_path)
    assert prep['complete'] and prep['manual_VM_run_unrun']
    packet=OUT/NAME
    p=read(packet/'protocol.json')
    pin=sha(packet/'protocol.json')
    assert pin==prep['protocol_sha256']
    old_root=OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_return'
    old=read(old_root/'protocol.json')
    for key in ['case_ids','learning_names','initial_model_state','parent_required_sha256','preserved_scientific_thresholds',
                'model_forward_cap','recognizer_forward_cap','worker_seconds','measurement_seconds','external_seconds',
                'export_seconds','export_external_seconds','minimum_free_bytes','disk_reserve_bytes','output_byte_cap','peak_VRAM_bytes']:
        assert p[key]==old[key],key
    assert p['proposals']==[dict(label='balanced_'+s,direction='balanced',rate=float(s)) for s in ['-0.001','-0.0001','0.0001','0.001']]
    assert p['gradient_groups_used_in_direction']==32 and p['proposal_derivation_TRAIN_references']==20
    assert p['peak_RSS_bytes']==12*1024**3 and p['additional_source_degraded_structure_gain_required']==.01
    assert p['optimizer_updates']==p['gradient_queries']==p['backwards']==p['full_epochs']==0
    archive=OUT/(STEM+'-execution.tar.gz')
    assert sha(archive)==prep['archive_sha256']
    assert PathBytes(OUT/(archive.name+'.sha256'))==(prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        expected={NAME+'/'+n for n in p['assets_sha256']}|{NAME+'/protocol.json'}
        assert len(members)==5 and {m.name for m in members}==expected
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk()
            assert tar.extractfile(member).read()==(OUT/member.name).read_bytes()
    for name,h in p['assets_sha256'].items():assert sha(packet/name)==h
    for name,h in p['parent_required_sha256'].items():assert sha(PARENT/name)==h
    for name,h in p['evidence_sha256'].items():assert sha(ROOT/name)==h and read(ROOT/name)['complete']
    direction=read(packet/'direction_review.json')
    original=ANALYSIS/'common_direction'
    assert sha(packet/'directions.npz')==sha(original/'directions.npz')==direction['direction_file_sha256']
    assert sha(packet/'direction_review.json')==sha(original/'direction_review.json')
    assert read(original/'independent_audit.json')['all32_float32_slopes_and_KKT_constraints_checked']
    worker=packet/'scripts'/f'{BASE}_vm.py'
    source=worker.read_text(encoding='utf-8')
    tree=ast.parse(source,feature_version=(3,10))
    old_tree=ast.parse((old_root/'scripts/cctv_dgp_bank_saved_direction_forward_v1_r2_vm.py').read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
        n.func.attr in ['grad','backward','step','Adam','AdamW','SGD','enable_vm_learning'] for n in ast.walk(tree))
    for name in ['candidate','identity.embedding','params[name].copy_']:
        current=[ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
        previous=[ast.dump(n,include_attributes=False) for n in ast.walk(old_tree) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
        assert current==previous and current,name
    spec=importlib.util.spec_from_file_location('safe_group_forward',worker)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    actual,parent=module.verify(packet,PARENT,pin)
    assert actual==p
    refs={r['id']:r for r in parent['references']}
    cases=[c for c in parent['cases'] if c['id'] in p['case_ids']]
    matrices=module.fixed_reference_matrices(cases,refs)
    retained=read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_transfer_audit.json')['independent_geometry_equations_checked']
    pilot=ast.parse((PARENT/'cctv_dgp_pilot.py').read_text(encoding='utf-8'))
    grid_node=next(n for n in pilot.body if isinstance(n,ast.FunctionDef) and n.name=='grid112')
    context={'np':np,'cv2':cv2}
    exec(compile(ast.Module(body=[grid_node],type_ignores=[]),'retained_fixed_grid','exec'),context)
    for c,matrix,binding in zip(cases,matrices,retained,strict=True):
        assert c['id']==binding['id']
        assert hashlib.sha256(context['grid112'](matrix).tobytes()).hexdigest()==binding['grid_sha256']
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    run=functions['run']
    require_line=min(n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='require_vm')
    first_import=min(n.lineno for n in ast.walk(run) if isinstance(n,(ast.Import,ast.ImportFrom)))
    early_write=next(n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='write' and n.lineno<first_import)
    assert require_line<early_write<first_import
    assert "'peak_RSS_bytes'" in source and 'ru_maxrss' in source
    rejected=subprocess.run([sys.executable,'-B',str(worker),'--root',str(packet),'--parent-root',str(PARENT),
        '--protocol-sha',pin,'--run'],capture_output=True,text=True,timeout=20)
    assert rejected.returncode and 'require_vm' in rejected.stderr and 'AssertionError' in rejected.stderr
    assert 'run(root, parent, p, old' not in rejected.stderr and not (packet/'outputs').exists()
    write(OUT/(BASE+'_windows_rejection.json'),dict(complete=True,exit_code=rejected.returncode,
        stderr=rejected.stderr,pre_neural_rejection=True,local_gradient_or_optimizer_calls=0))
    importer=ROOT/'scripts'/f'import_{BASE}_return.py'
    auditor=ROOT/'scripts'/f'audit_{BASE}_return.py'
    importer_tree=ast.parse(importer.read_text(encoding='utf-8'))
    literals={n.targets[0].id:n.value.value for n in importer_tree.body if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    assert literals['NAME']==NAME and literals['STEM']==STEM and literals['PIN']==pin
    importer_source=importer.read_text(encoding='utf-8')
    assert BASE+'_return' in importer_source and BASE+'_analysis' in importer_source and BASE+'_transfer_audit.json' in importer_source
    auditor_source=auditor.read_text(encoding='utf-8')
    assert 'source_structure_gains' in auditor_source and "additional_source_degraded_structure_gain_required" in auditor_source
    assert "outputs/group_step_check_v1" in auditor_source and "timing_audit_repair_sha256" not in auditor_source
    helper=read(OUT/(BASE+'_return_checker_preparation.json'))
    assert helper['complete'] and helper['actual_saved_control_cases_checked']==100
    assert helper['importer_sha256']==sha(importer) and helper['auditor_sha256']==sha(auditor)
    assert helper['protocol_sha256']==pin
    assert p['negative_steps_diagnostic_only'] and p['export_storage_reserve_enforced']
    derivative=read(OUT/(BASE+'_signed_derivative_fixture_audit.json'))
    assert derivative['complete'] and derivative['groups_checked']==32
    assert derivative['checker_sha256']==sha(ROOT/'scripts/cctv_dgp_signed_step_audit.py')
    assert 'selected_parameter_change_statistics' in source and 'Export space reserve stop' in source
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        preparation_sha256=sha(prep_path),protocol_sha256=pin,archive_sha256=sha(archive),
        members_checked=5,assets_checked=4,parent_hashes_checked=5852,all100_fixed_alignment_grid_hashes_retained=True,
        LF_checksum_and_Python310_AST_checked=True,future_importer_exact_scope_PIN_and_names_checked=True,
        future_importer_sha256=sha(importer),future_auditor_sha256=sha(auditor),
        return_checker_arithmetic_audit_sha256=sha(OUT/(BASE+'_return_checker_preparation.json')),
        raw_PNG_metrics_and_retained_six_gate_boundaries_checked=True,
        neural_forward_and_parameter_displacement_call_ASTs_retained=True,
        all32_group_derived_direction_matches_independent_arithmetic_audit=True,
        structured_pre_model_storage_stop_and_RSS_limit_checked=True,Windows_run_rejected_before_neural_import=True,
        preserved_original_checkpoints_app_examples_and_failures=True,
        no_local_neural_gradient_optimizer_or_parameter_displacement=True,no_agent_VM_connection_or_launch=True,
        signed_derivative_fixture_audit_sha256=sha(OUT/(BASE+'_signed_derivative_fixture_audit.json')),
        actual_manual_VM_run_unrun=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,members=5,parent_hashes=5852,fixed_grids=100,actual_manual_VM_run_unrun=True))


def PathBytes(path):
    return path.read_bytes()


if __name__=='__main__':
    main()
