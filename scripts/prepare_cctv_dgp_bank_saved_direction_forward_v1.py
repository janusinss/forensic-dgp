"""Seal four finite temporary forward proposals for human tmux execution only."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import tarfile

from import_cctv_dgp_bank_hinge_boundary_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write

NAME = 'cctv_dgp_bank_saved_direction_forward_v1_vm'
STEM = 'cctv-dgp-bank-saved-direction-forward-v1'


def main():
    dest = OUT/NAME
    archive = OUT/(STEM+'-execution.tar.gz')
    receipt = OUT/'cctv_dgp_bank_saved_direction_forward_v1_preparation.json'
    assert not any(path.exists() for path in [dest, archive, receipt])
    direction_root = OUT/'cctv_dgp_bank_saved_direction_v1_analysis'
    evidence = [ANALYSIS/'import.json', ANALYSIS/'independent_audit.json', ANALYSIS/'loss_direction_review.json',
                direction_root/'direction_review.json']
    assert all(read(path)['complete'] for path in evidence)
    direction = read(evidence[-1])
    assert direction['no_local_neural_calls_gradients_optimizer_or_parameter_displacement']
    assert direction['maximum_primal_slack'] < 1e-7 and direction['direction_norm_ratio'] < 2
    assert direction['sample_scope_two_TRAIN_references_ten_cases']
    old = read(RETURN/'protocol.json')
    assert sha(RETURN/'protocol.json') == '1947d162217984694573bfcba8811f717630f66068474e6c9659d58c1a6b9099'
    for name,digest in old['parent_required_sha256'].items():
        assert sha(PARENT/name) == digest
    source_path = ROOT/'scripts/cctv_dgp_bank_saved_direction_forward_v1_vm.py'
    source = source_path.read_text(encoding='utf-8')
    tree = ast.parse(source, feature_version=(3,10))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in ['step', 'backward', 'grad', 'Adam', 'AdamW', 'SGD', 'enable_vm_learning'] for n in ast.walk(tree))
    dest.mkdir()
    (dest/'scripts').mkdir()
    (dest/'scripts'/source_path.name).write_text(source, encoding='utf-8', newline='\n')
    (dest/'directions.npz').write_bytes((direction_root/'directions.npz').read_bytes())
    (dest/'direction_review.json').write_bytes((direction_root/'direction_review.json').read_bytes())
    shell = (RETURN/'scripts/run_boundary.sh').read_text(encoding='utf-8')
    shell = shell.replace('cctv_dgp_bank_hinge_boundary_v1_vm.py',source_path.name)
    assert shell.count(' 930 python') == 1
    shell = shell.replace(' 930 python',' 1230 python')
    (dest/'scripts/run_forward.sh').write_text(shell, encoding='utf-8', newline='\n')
    raw_bound = 4*100*256*256*3*4+80*256
    png_bound = 400*256*1024
    embedding_bound = 80*2*15*512*4+80*2048
    forecast = raw_bound+png_bound+embedding_bound+(dest/'directions.npz').stat().st_size+4*1024**2
    assert forecast < 768*1024**2
    p = dict(format='manual-saved-post-Adam-direction-finite-forward-v1', name=NAME,
        parent_name=old['parent_name'], parent_protocol_sha256=old['parent_protocol_sha256'],
        parent_required_sha256=old['parent_required_sha256'], case_ids=old['case_ids'],
        learning_names=old['learning_names'], initial_model_state=old['model_states']['initial'],
        original_checkpoint_sha256=old['original_checkpoint_sha256'],
        preserved_scientific_thresholds=old['preserved_scientific_thresholds'],
        proposals=[dict(label=label+'_'+rate, direction=label,rate=float(rate))
            for label in ['ordinary','constrained'] for rate in ['0.0001','0.00001']],
        paired_TRAIN_cases=100, TRAIN_references=20, proposal_derivation_TRAIN_references=2,
        native_pixels_used=0, final_identity_pixels=0,
        optimizer_updates=0, gradient_queries=0, backwards=0, full_epochs=0,
        maximum_temporary_parameter_proposals=4, model_forward_cap=100, recognizer_forward_cap=160,
        worker_seconds=1200, measurement_seconds=900, external_seconds=1230,
        export_seconds=300, export_external_seconds=330, kill_grace_seconds=30,
        peak_VRAM_bytes=20*1024**3, minimum_free_bytes=4*1024**3, disk_reserve_bytes=1024**3,
        output_byte_cap=768*1024**2, forecast_maximum_payload_bytes=forecast,
        exact_initial_raw_controls=20, raw_case_replays=100, new_candidate_forward_cases=400,
        raw_and_PNG_outputs_and_embeddings_separate=True,
        actual_finite_parameter_displacements_on_manual_VM_only=True,
        no_training_worker_import=True, no_optimizer_construction=True, no_new_neural_derivatives=True,
        initial_state_restored_between_proposals_and_after_run=True, no_checkpoint_created=True,
        no_strength_or_step_selection_in_this_worker=True,
        derivative_projection_after_Adam_preconditioning=True,
        objective_omits_unconditional_absolute_anchors_but_retains_barrier_and_corrected_guard=True,
        ten_constraints_are_sampled_reference_loss_averages_not_all_profiles_or_all_DEV=True,
        future_full_accumulated_training_update_not_qualified=True,
        failed_run_resume_allowed=False, app_promotion=False, automatic_follow_on=False,
        manual_tmux_required=True, native_unpaired=True,
        all_seven_completion_families_pending=True, independent_final_review_pending=True, goal_complete=False,
        evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in evidence})
    p['assets_sha256']={f.relative_to(dest).as_posix():sha(f) for f in dest.rglob('*') if f.is_file()}
    write(dest/'protocol.json',p)
    pin=sha(dest/'protocol.json')
    with tarfile.open(archive,'x:gz',compresslevel=1) as stream:
        for file in sorted(dest.rglob('*')):
            if file.is_file():
                stream.add(file,arcname=NAME+'/'+file.relative_to(dest).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
        name=NAME,stem=STEM,protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        packet_assets=len(p['assets_sha256']),parent_required_assets=len(p['parent_required_sha256']),
        forecast_payload_bytes=forecast,actual_VM_forward_unrun=True,local_parameter_displacements=0,
        local_neural_gradient_or_optimizer_calls=0,no_agent_VM_launch=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,archive_bytes=archive.stat().st_size,protocol_sha256=pin,
               temporary_VM_proposals_planned=4,actual_VM_run_unrun=True))


if __name__=='__main__':
    main()
