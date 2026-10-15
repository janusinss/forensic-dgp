"""Seal one finite manual training pilot after the returned activation audit."""
from datetime import datetime,timezone
from pathlib import Path
import shutil
import tarfile

from import_cctv_dgp_live_conditioning_v1_r1_return import ROOT,OUT,RETURN,ANALYSIS,PARENT,read,sha,write


def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
    assert sha(source)==sha(target)


def main():
    name='cctv_dgp_conditioning_learning_v1_r1_vm';stem='cctv-dgp-conditioning-learning-v1-r1'
    dest=OUT/name;assert not dest.exists()
    audit=read(ANALYSIS/'independent_audit.json');subspace=read(ANALYSIS/'learning_subspace_audit.json')
    assert audit['complete'] and subspace['complete']
    review=read(ANALYSIS/'learning_subspace_review.json')
    assert sha(ANALYSIS/'learning_subspace_review.json')==subspace['review_sha256']
    assert review['subspaces']['conditioning12']['supported']
    oldp=read(RETURN/'protocol.json');parentp=read(PARENT/'protocol.json')
    assert sha(RETURN/'protocol.json')==audit['protocol_sha256']
    assert all(sha(PARENT/n)==h for n,h in oldp['parent_required_sha256'].items())
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    dest.mkdir();(dest/'scripts').mkdir()
    for source,target in [
        (ROOT/'scripts/cctv_dgp_conditioning_learning_v1_r1_vm.py',dest/'scripts/cctv_dgp_conditioning_learning_v1_r1_vm.py'),
        (ROOT/'scripts/cctv_dgp_conditioning_learning_v1_r1_contract.py',dest/'conditioning_contract.py'),
        (RETURN/'cctv_dgp_live_conditioning_v1_r1_contract.py',dest/'conditioning_scopes.py'),
        (OUT/'cctv_dgp_png_group_gradients_v1_r2_vm/cctv_dgp_png_group_gradients_v1_r2_losses.py',dest/'conditioning_losses.py'),
        (RETURN/'readouts.npz',dest/'readouts.npz'),
        (ANALYSIS/'independent_audit.json',dest/'activation_audit.json'),
        (ANALYSIS/'learning_subspace_audit.json',dest/'subspace_audit.json'),
        (ANALYSIS/'learning_subspace_review.json',dest/'subspace_review.json')]:copy(source,target)
    model=(RETURN/'cctv_dgp_live_conditioning_v1_r1_model.py').read_text(encoding='utf-8')
    assert model.count("NAME = 'cctv_dgp_live_conditioning_v1_r1_vm'")==1
    model=model.replace("NAME = 'cctv_dgp_live_conditioning_v1_r1_vm'","NAME = 'cctv_dgp_conditioning_learning_v1_r1_vm'")
    with (dest/'conditioning_model.py').open('x',encoding='utf-8',newline='\n') as f:f.write(model)
    shell='''#!/usr/bin/env bash
set -u -o pipefail
cd "$HOME/forensic-dgp/cctv_dgp_conditioning_learning_v1_r1_vm" || exit 1
test -n "${TMUX:-}" || { echo 'Human manual tmux required'; exit 1; }
test ! -e review.log || { echo 'Used pilot; preserve it'; exit 1; }
PIN="$1"
PY="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
timeout --signal=TERM --kill-after=30s 1530s "$PY" -B -u scripts/cctv_dgp_conditioning_learning_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$PIN" --run >review.log 2>&1
code=$?
printf '%s\\n' "$code" >review_exit_code.txt
cat review.log
printf 'Trainer exit code: %s\\n' "$code"
timeout --signal=TERM --kill-after=30s 450s "$PY" -B -u scripts/cctv_dgp_conditioning_learning_v1_r1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$PIN" --export >export.log 2>&1
export_code=$?
printf '%s\\n' "$export_code" >export_exit_code.txt
cat export.log
printf 'Export exit code: %s\\n' "$export_code"
exit "$export_code"
'''
    with (dest/'scripts/run_learning.sh').open('x',encoding='utf-8',newline='\n') as f:f.write(shell)
    names=[n for n in oldp['learning_names'] if n.startswith(('live.style_conditioner.','live.spatial_conditioners.'))]
    assert len(names)==12
    p=dict(format='finite-conditioning-learning-v1-r1',name=name,UTC=datetime.now(timezone.utc).isoformat(),
        parent_protocol_sha256=oldp['parent_protocol_sha256'],parent_required_sha256=oldp['parent_required_sha256'],
        assets_sha256={f.relative_to(dest).as_posix():sha(f) for f in dest.rglob('*') if f.is_file()},
        case_ids=oldp['case_ids'],reference_blocks=oldp['reference_blocks'],raw_gradient_groups=oldp['gradient_groups'],
        paired_TRAIN_cases=100,conditioning_names=names,conditioning_parameters=2580864,
        conditioning_shapes={n:oldp['selected_parameter_shapes'][n] for n in names},
        all37_learning_names=oldp['learning_names'],initial_model_state=oldp['initial_model_state'],
        sample_gradient_begins=oldp['sample_gradient_begins'],
        activation_protocol_sha256=sha(RETURN/'protocol.json'),activation_return_archive_sha256=read(ANALYSIS/'import.json')['archive_sha256'],
        activation_audit_sha256=sha(ANALYSIS/'independent_audit.json'),subspace_audit_sha256=sha(ANALYSIS/'learning_subspace_audit.json'),
        preserved_scientific_thresholds=parentp['scientific_thresholds'],
        maximum_accepted_updates=2,maximum_trials_per_update=3,maximum_gradient_queries=1280,
        model_forward_cap=180,recognizer_forward_cap=360,full_epochs=0,
        maximum_run_relative_parameter_distance=.001,maximum_rate=.02,backtrack_factors=[1.,.5,.25],
        worker_seconds=1500,measurement_seconds=1350,gradient_round_seconds=600,solver_seconds=60,
        external_seconds=1530,export_seconds=420,export_external_seconds=450,kill_grace_seconds=30,
        minimum_free_bytes=9*1024**3,disk_reserve_bytes=1024**3,output_byte_cap=3584*1024**2,
        peak_VRAM_bytes=20*1024**3,peak_RSS_bytes=12*1024**3,
        current_decoder_and_all_readouts_and_original_prior_recognizer_buffers_frozen=True,
        same_observed_input_frozen_feature_reference=True,external_pretrained_generator_is_frozen_and_declared=True,
        RAW_groups_per_round=32,PNG_groups_per_round=32,
        canonical_exact_PNG_forward_with_explicit_approximate_ST_backward=True,
        raw_SSIM_gradient_float64_surrogate_declared=True,
        refresh_all64_gradients_after_every_accepted_update=True,
        actual_raw_and_PNG_scientific_preservation_gate_before_every_accepted_update=True,
        require_positive_detail_progress_globally_and_each_source_on_both_stages=True,
        final_finite_checkpoint_requires_1percent_globally_and_each_source_RAW_and_PNG=True,
        full_TRAIN_final_structure_requirement_remains_10percent=True,
        rejected_trials_and_partial_accepted_components_exported_with_true_update_counts=True,
        optimizer='Stateless constrained full-cohort gradient descent; accepted changes are actual training',
        momentum=None,scheduler=None,resume_allowed=False,
        original_checkpoint_sha256=oldp['frozen_original_checkpoint_sha256'],
        source_labels_not_ethnicity_or_native_claims=True,native_pixels_used=0,final_identity_pixels=0,
        manual_tmux_required=True,failed_run_resume_allowed=False,automatic_follow_on=False,app_promotion=False,
        full_TRAIN_and_useful_native_DEV_and_all_new_visual_sheets_audit_required=True,
        independent_final_and_seven_covering_family_scope_unchanged=True,
        evidence_is_activation_and_saved_RAW_subspace_arithmetic_not_trained_output_quality=True,
        new_training_recipe_unrun=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False)
    # Never allow an accidental duplicate key to hide a contract edit.
    assert p['raw_gradient_groups']==oldp['gradient_groups']
    write(dest/'protocol.json',p);pin=sha(dest/'protocol.json')
    archive=OUT/(stem+'-execution.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in sorted(f for f in dest.rglob('*') if f.is_file()):
            tar.add(file,arcname=name+'/'+file.relative_to(dest).as_posix(),recursive=False)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:stream.write(sha(archive)+'  '+archive.name+'\n')
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(OUT/'cctv_dgp_conditioning_learning_v1_r1_preparation.json',dict(complete=True,
        UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),protocol_sha256=pin,
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,packet_assets=len(p['assets_sha256']),
        parent_file_bindings=len(p['parent_required_sha256']),activation_audit_sha256=p['activation_audit_sha256'],
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,manual_pilot_unrun=True,
        prospective_transfer_and_return_audits_required=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,protocol_sha256=pin,archive_bytes=archive.stat().st_size,manual_unrun=True))


if __name__=='__main__':main()
