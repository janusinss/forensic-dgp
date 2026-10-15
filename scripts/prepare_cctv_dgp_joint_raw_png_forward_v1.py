"""Seal a distinct positive finite comparison from audited joint derivatives."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
from import_cctv_dgp_png_group_gradients_v1_r2_return import ROOT,OUT,RETURN,ANALYSIS,PARENT,read,sha,write

BASE='cctv_dgp_joint_raw_png_forward_v1'
NAME=BASE+'_vm'
STEM='cctv-dgp-joint-raw-png-forward-v1'
SOURCE_BASE='cctv_dgp_bank_group_step_check_v1'
SOURCE_STEM='cctv-dgp-bank-group-step-check-v1'
RATES=['0.0001','0.0005','0.005','0.02']


def once(text,old,new):
    assert text.count(old)==1, old[:100]
    return text.replace(old,new)


def scope(text):
    return text.replace(SOURCE_BASE,BASE).replace(SOURCE_STEM,STEM).replace('group_step_check_v1','joint_raw_png_forward_v1')


def new_text(path,text):
    ast.parse(text,feature_version=(3,10))
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)


def main():
    packet=OUT/NAME;archive=OUT/(STEM+'-execution.tar.gz');receipt=OUT/(BASE+'_preparation.json')
    assert not any(path.exists() for path in [packet,archive,receipt])
    direction_root=ANALYSIS/'joint_raw_png_direction_v1_r2'
    direction=read(direction_root/'direction_review.json');independent=read(direction_root/'independent_audit.json')
    assert direction['complete'] and independent['complete']
    assert sha(direction_root/'direction_review.json')==independent['direction_review_sha256']
    assert sha(direction_root/'directions.npz')==independent['direction_file_sha256']
    assert independent['all64_actual_saved_float32_derivatives_descend'] and independent['no_ridge_or_loss_group_removal']
    assert read(ANALYSIS/'independent_audit.json')['complete']
    source_root=OUT/'cctv_dgp_bank_group_step_check_v1_return'
    assert read(OUT/'cctv_dgp_bank_group_step_check_v1_analysis/independent_audit_r1.json')['complete']
    assert read(OUT/'cctv_dgp_bank_group_step_check_v1_analysis/visual_review.json')['all20_sheets_and800_cells_inspected']
    source=source_root/'scripts'/('cctv_dgp_bank_group_step_check_v1_vm.py')
    text=scope(source.read_text())
    old="""    assert p['signed_radii'] == [.001, .0001] and p['negative_steps_diagnostic_only']
    assert p['no_arbitrary_derivative_threshold_or_model_selection']
    assert p['export_storage_reserve_enforced']"""
    new="""    assert p['joint_RAW_gradient_groups']==p['joint_PNG_gradient_groups']==32
    assert p['all64_actual_saved_float32_derivatives_descend']
    assert p['export_storage_reserve_enforced'] and p['positive_steps_only']"""
    text=once(text,old,new)
    text=once(text,"for rate in ['-0.001', '-0.0001', '0.0001', '0.001']]", "for rate in ['0.0001', '0.0005', '0.005', '0.02']]")
    text=once(text,"assert p['gradient_groups_used_in_direction'] == 32", "assert p['gradient_groups_used_in_direction'] == 64")
    text=once(text,"    assert direction['rate_candidates'] == [.005, .01, .02, .04]", "    assert direction['rate_candidates'] == [.0025, .005, .01, .02]\n    assert p['proposed_rates_bounded_by_direction_certificate'] and max(v['rate'] for v in p['proposals']) <= .02")
    old_protocol=read(OUT/'cctv_dgp_bank_group_direction_forward_v1_return/protocol.json')
    p={k:v for k,v in old_protocol.items() if k not in ['assets_sha256','evidence_sha256']}
    p.update(name=NAME,format='joint-raw-delivered-positive-finite-comparison-v1',UTC=datetime.now(timezone.utc).isoformat(),
        proposals=[dict(label='balanced_'+r,direction='balanced',rate=float(r)) for r in RATES],
        gradient_groups_used_in_direction=64,joint_RAW_gradient_groups=32,joint_PNG_gradient_groups=32,
        all64_actual_saved_float32_derivatives_descend=True,positive_steps_only=True,
        proposed_rates_bounded_by_direction_certificate=True,export_storage_reserve_enforced=True,
        maximum_prospective_parameter_displacement_relative_L2=.02*direction['direction_norm']/direction['initial_parameter_norm'],
        smaller_steps_selected_from_retained_raw_and_PNG_curvature_failures=True,
        four_temporary_resets_not_training_epochs_or_committed_updates=True,
        unchanged1percent_global_and_each_source_required_in_both_raw_and_PNG=True,
        derivative_design_does_not_qualify_actual_preservation_or_more_epochs=True,
        PNG_backward_is_declared_approximation_not_true_quantization_derivative=True,
        scientific_recipe_change='New jointly constrained64-loss direction plus small and structure-scale positive finite steps; preserve all inputs/quality gates',
        actual_float32_parameter_change_statistics_saved=True,
        manual_tmux_required=True,independent_saved_output_and_all20_sheet_visual_review_required=True,
        app_promotion=False,automatic_follow_on=False,failed_run_resume_allowed=False)
    p['evidence_sha256']={path.relative_to(ROOT).as_posix():sha(path) for path in [
        ANALYSIS/'independent_audit.json',direction_root/'direction_review.json',direction_root/'independent_audit.json',
        OUT/'cctv_dgp_bank_group_step_check_v1_analysis/independent_audit_r1.json',
        OUT/'cctv_dgp_bank_group_step_check_v1_analysis/visual_review.json']}
    assert p['model_forward_cap']==100 and p['recognizer_forward_cap']==160
    assert p['optimizer_updates']==p['gradient_queries']==p['backwards']==p['full_epochs']==0
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    packet.mkdir();(packet/'scripts').mkdir()
    new_text(packet/'scripts'/(NAME+'.py'),text)
    for name in ['directions.npz','direction_review.json']:shutil.copyfile(direction_root/name,packet/name)
    shell=scope((source_root/'scripts/run_forward.sh').read_text())
    shell=once(shell,'diagnostic_exit=${statuses[0]}','diagnostic_exit=${statuses[0]}\ntest "${statuses[1]}" -eq 0 || diagnostic_exit=${statuses[1]}')
    shell=once(shell,'export_exit=${statuses[0]}','export_exit=${statuses[0]}\ntest "${statuses[1]}" -eq 0 || export_exit=${statuses[1]}')
    with (packet/'scripts/run_forward.sh').open('x',encoding='utf-8',newline='\n') as f:f.write(shell)
    p['assets_sha256']={f.relative_to(packet).as_posix():sha(f) for f in packet.rglob('*') if f.is_file()}
    assert len(p['assets_sha256'])==4
    write(packet/'protocol.json',p);pin=sha(packet/'protocol.json')
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for f in sorted(path for path in packet.rglob('*') if path.is_file()):tar.add(f,arcname=NAME+'/'+f.relative_to(packet).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+archive.name+'\n')
    old_base='cctv_dgp_bank_group_direction_forward_v1';old_stem='cctv-dgp-bank-group-direction-forward-v1'
    checks={}
    for prefix in ['import','audit']:
        original=ROOT/'scripts'/(prefix+'_'+old_base+'_return.py')
        s=original.read_text().replace(old_base,BASE).replace(old_stem,STEM).replace('group_direction_forward_v1','joint_raw_png_forward_v1')
        s=s.replace('92783a76effd2409d4a77433a6a4f52ea80e48628dffc444716e3e6df2d7bd9e',pin)
        if prefix=='audit':
            s=once(s,"'Group step .005','Group step .01','Group step .02','Group step .04'", "'Joint step .0001','Joint step .0005','Joint step .005','Joint step .02'")
        path=ROOT/'scripts'/(prefix+'_'+BASE+'_return.py');new_text(path,s);checks[path.relative_to(ROOT).as_posix()]=sha(path)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
        name=NAME,stem=STEM,protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        retained_source_worker_sha256=sha(source),direction_review_sha256=sha(direction_root/'direction_review.json'),
        independent_joint_direction_audit_sha256=sha(direction_root/'independent_audit.json'),
        future_checkers_sha256=checks,manual_VM_run_unrun=True,old_used_packets_unchanged=True,
        original_app_source_checkpoint_and_all_gates_preserved=True,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,no_VM_connection_or_launch=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,name=NAME,bytes=archive.stat().st_size,protocol_sha256=pin,archive_sha256=digest))


if __name__=='__main__':main()
