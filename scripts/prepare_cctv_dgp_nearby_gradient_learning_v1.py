"""Seal a finite human-manual learning packet using hash and file operations only."""
import ast
from datetime import datetime, timezone
import shutil
import subprocess
import sys
import tarfile

from import_cctv_dgp_nearby_gradient_learning_v1_return import ROOT,OUT,NAME,STEM,PARENT,RESPONSE,read,sha,write


def text_file(path,value,encoding='utf-8'):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding=encoding,newline='\n') as f:f.write(value)


def main():
    prior=OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_vm'
    source=read(prior/'protocol.json');old=read(PARENT/'protocol.json')
    source_files={**source['assets_sha256'],'protocol.json':sha(prior/'protocol.json')}
    assert source_files['protocol.json']=='6b42f9e828854cf7aaf6d7058964494b88f2b1c47fcb4cd4c93331463a278b61'
    assert all(sha(prior/n)==h for n,h in source_files.items())
    precision=OUT/'cctv_dgp_recognizer_precision_v1_analysis'
    proofs=[precision/n for n in ['independent_audit.json','precision_activation_review.json','publication/publication_audit.json']]
    proofs += [OUT/('cctv_dgp_nearby_gradient_learning_v1_'+n+'.json') for n in [
        'contract_green','certificate_green','source_paths_r1','saved_fixture_r1','worker_source','worker_review_r1','snapshot_source','archive']]
    assert all(read(f)['complete'] for f in proofs)
    fixture=read(OUT/'cctv_dgp_nearby_gradient_learning_v1_saved_fixture_r1.json')
    assert fixture['return_auditor_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_nearby_gradient_learning_v1_return.py')
    assert fixture['snapshot_auditor_sha256']==sha(ROOT/'scripts/cctv_dgp_nearby_snapshot_audit_v1.py')
    assert read(OUT/'cctv_dgp_nearby_gradient_learning_v1_source_paths_r1.json')['return_auditor_sha256']==fixture['return_auditor_sha256']
    guard=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/check_cctv_dgp_nearby_gradient_learning_v1_guard.py')],capture_output=True,text=True,timeout=30)
    assert guard.returncode==0,guard.stdout+guard.stderr
    parent_files=source['parent_required_sha256']
    response_files={n:h for n,h in read(RESPONSE/'export_manifest.json')['files'].items() if n!='export.log'}
    response_files['export_manifest.json']=sha(RESPONSE/'export_manifest.json')
    assert sha(RESPONSE/'protocol.json')=='e72d76e3f2330420ee93b43cf528d0b49da132f356a30c336a1ec578c8f82f44'
    for folder,bindings in [(PARENT,parent_files),(RESPONSE,response_files)]:
        assert all(sha(folder/n)==h for n,h in bindings.items())
    protected=read(precision/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    packet=OUT/NAME;assert not packet.exists();packet.mkdir()
    copies=[(ROOT/'scripts/cctv_dgp_nearby_gradient_learning_v1_vm.py','scripts/cctv_dgp_nearby_gradient_learning_v1_vm.py'),
        (ROOT/'scripts/cctv_dgp_nearby_gradient_learning_v1_contract.py','conditioning_contract.py'),
        (ROOT/'scripts/cctv_dgp_raw_conditioning_learning_v1_r1_contract.py','cctv_dgp_raw_conditioning_learning_v1_r1_contract.py')]
    copies += [(prior/n,n) for n in ['conditioning_losses.py','conditioning_raw_losses.py','conditioning_scopes.py','readouts.npz','activation_audit.json','subspace_audit.json']]
    origins={}
    for path,name in copies:
        if name.endswith('.py'):ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10))
        target=packet/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        origins[name]=dict(source_path=path.relative_to(ROOT).as_posix(),source_sha256=sha(path),exact_copy=True)
    model=prior/'conditioning_model.py';text=model.read_text(encoding='utf-8')
    assert text.count('cctv_dgp_raw_conditioning_learning_v1_r1_vm')==1
    text_file(packet/'conditioning_model.py',text.replace('cctv_dgp_raw_conditioning_learning_v1_r1_vm',NAME))
    origins['conditioning_model.py']=dict(source_path=model.relative_to(ROOT).as_posix(),source_sha256=sha(model),exact_copy=False,
        sole_change='Manual-only root namespace; unchanged forward, parameters and learning guards')
    evidence={}
    for f in proofs:
        name='evidence/'+f.relative_to(OUT).as_posix()
        target=packet/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target);evidence[name]=sha(f)
    text_file(packet/'evidence/design_review.md',(ROOT/'CCTV_DGP_NEARBY_GRADIENT_LEARNING_REVIEW_V1.md').read_text(encoding='utf-8'))
    write(packet/'evidence/manual_guard.json',dict(complete=True,worker_sha256=sha(copies[0][0]),checker_sha256=sha(ROOT/'scripts/check_cctv_dgp_nearby_gradient_learning_v1_guard.py'),
        exit_code=guard.returncode,output=guard.stdout,local_training_rejected_before_neural_loading=True,local_neural_or_training_calls=0))
    shell=f'''#!/usr/bin/env bash
set -u -o pipefail
cd "$HOME/forensic-dgp/{NAME}" || exit 1
test -n "${{TMUX:-}}" || {{ echo 'Human manual tmux required'; exit 1; }}
test "$#" -eq 1 || exit 1
test ! -e review.log || {{ echo 'Used training comparison; retain it'; exit 1; }}
PIN="$1"
PY="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
timeout --signal=TERM --kill-after=30s 1530s "$PY" -B -u scripts/cctv_dgp_nearby_gradient_learning_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$PIN" --run >review.log 2>&1
code=$?
printf '%s\\n' "$code" >review_exit_code.txt
cat review.log
printf 'Training comparison exit code: %s\\n' "$code"
timeout --signal=TERM --kill-after=30s 450s "$PY" -B -u scripts/cctv_dgp_nearby_gradient_learning_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$PIN" --export >export.log 2>&1
export_code=$?
printf '%s\\n' "$export_code" >export_exit_code.txt
cat export.log
printf 'Export exit code: %s\\n' "$export_code"
exit "$export_code"
'''
    text_file(packet/'scripts/run_learning.sh',shell)
    kept=['parent_protocol_sha256','case_ids','paired_TRAIN_cases','reference_blocks','raw_gradient_groups','conditioning_names','conditioning_shapes','conditioning_parameters',
        'all37_learning_names','initial_model_state','sample_gradient_begins','preserved_scientific_thresholds','maximum_trials_per_update',
        'maximum_run_relative_parameter_distance','maximum_rate','backtrack_factors','worker_seconds','measurement_seconds','gradient_round_seconds','solver_seconds',
        'external_seconds','export_seconds','export_external_seconds','kill_grace_seconds','minimum_free_bytes','disk_reserve_bytes','output_byte_cap','peak_VRAM_bytes','peak_RSS_bytes',
        'original_checkpoint_sha256']
    p={k:source[k] for k in kept}
    p.update(format='finite-nearby-gradient-learning-v1',name=NAME,UTC=datetime.now(timezone.utc).isoformat(),
        source_protocol_sha256=source_files['protocol.json'],source_required_sha256=source_files,
        parent_required_sha256=parent_files,response_protocol_sha256=sha(RESPONSE/'protocol.json'),response_required_sha256=response_files,
        excluded_volatile_VM_paths=dict(response=['export.log'],reason='Retain growing live export log; all other assets exactly bound'),
        expected_torch='2.9.1+cu129',sampling_sites=['initial','plus02','minus02'],sampling_state_roles='Failed signed states are derivative sampling only and never accepted candidates',
        direction_solves=1,gradient_rounds=3,maximum_accepted_updates=1,maximum_gradient_queries=960,model_forward_cap=200,recognizer_forward_cap=340,
        RAW_groups_per_round=32,PNG_groups_per_round=0,optimizer_constraints='96 nearby-state RAW constraints; actual original RAW/PNG gates',
        optimizer='Stateless certified nearby-group constrained descent; accepted change is actual training',momentum=None,scheduler=None,
        solver=dict(name='SLSQP minimum-norm primal in numerical gradient span; NNLS nonnegative dual certificate',
            inequality_rows=96,gradient_normalization='Group Euclidean norm; unchanged loss targets',feature_fraction=1.,other_loss_fraction=.1,
            Gram_rank_bound='96 * float64 epsilon * max(1, largest eigenvalue)',Gram_PSD_bound='smallest eigenvalue >= -10 * rank_bound',
            Gram_reconstruction_bound='max(1e-10, 10 * rank_bound)',SLSQP_ftol=1e-12,SLSQP_maxiter=2000,NNLS_maxiter=10000,
            active_slack_bound=1e-9,dual_slack_bound=1e-9,complementarity_bound=1e-9,primal_dual_distance_bound=1e-8,
            float32_original_fraction_constraint_bound=1e-6,no_ridge_or_loss_removal=True,no_convergence_or_finite_preservation_guarantee=True),
        original_repeat_elementwise_rtol=1e-4,original_repeat_elementwise_atol=1e-8,original_repeat_norm_rtol=1e-4,original_repeat_norm_atol=1e-10,
        current_decoder_readouts_original_prior_recognizer_buffers_frozen=True,external_pretrained_generator_frozen_and_declared=True,
        RAW_SSIM_backward_is_declared_float64_surrogate_unchanged=True,PNG_surrogate_used_for_optimization=False,
        exact_PNG_delivery_and_all17_scientific_acceptance_groups_retained=True,
        actual_RAW_PNG_acceptance_required_for_every_retained_training_update=True,
        positive_detail_progress_required_globally_and_each_source_RAW_PNG=True,
        early_endpoint_requires_1percent_global_and_both_sources_RAW_PNG=True,full_TRAIN_later_requirement_remains_10percent=True,
        maximum_derivative_sampling_displacements=2,sampling_failures_remain_binding=True,
        accepted_full_component_state_and_Torch_RNG_retained_even_if_endpoint_fails=True,always_restore_initial_model_and_RNG=True,
        native_pixels_used=0,final_identity_pixels=0,full_epochs=0,manual_tmux_required=True,failed_run_resume_allowed=False,
        automatic_follow_on=False,app_promotion=False,source_origins=origins,preparation_evidence_sha256=evidence,
        source_labels_not_ethnicity_or_native_claims=True,full_TRAIN_DEV_final_and_all20_visual_sheets_pending=True,
        independent_final_and_seven_covering_family_scope_unchanged=True,
        estimate_minutes=[8,20],estimate_with_export_minutes=[10,25],estimate_not_fresh_CUDA_evidence=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False)
    p['assets_sha256']={f.relative_to(packet).as_posix():sha(f) for f in packet.rglob('*') if f.is_file()}
    write(packet/'protocol.json',p);pin=sha(packet/'protocol.json')
    importer=ROOT/'scripts/import_cctv_dgp_nearby_gradient_learning_v1_return.py';template=importer.read_text(encoding='utf-8')
    assert template.count("PIN='__PIN__'")==1
    text_file(OUT/'cctv_dgp_nearby_gradient_learning_v1_importer_template.py',template)
    importer.write_text(template.replace("PIN='__PIN__'","PIN='"+pin+"'"),encoding='utf-8',newline='\n')
    archive=OUT/(STEM+'-execution.tar.gz')
    with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
        for f in sorted(packet.rglob('*')):
            if f.is_file():tar.add(f,arcname=NAME+'/'+f.relative_to(packet).as_posix(),recursive=False)
    digest=sha(archive);text_file(OUT/(archive.name+'.sha256'),digest+'  '+archive.name+'\n','ascii')
    write(OUT/'cctv_dgp_nearby_gradient_learning_v1_preparation.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        preparer_sha256=sha(__file__),protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        packet_bindings=len(p['assets_sha256'])+1,dependency_bindings=dict(parent=len(parent_files),response=len(response_files)),
        original_source_bindings=len(source_files),immutable_importer_sha256=sha(importer),
        preparation_evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in proofs},
        local_neural_or_training_calls=0,VM_launch=False,manual_UNRUN=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,protocol_sha256=pin,archive_sha256=digest,cases=100,maximum_accepted_updates=1,manual_UNRUN=True))


if __name__=='__main__':main()
