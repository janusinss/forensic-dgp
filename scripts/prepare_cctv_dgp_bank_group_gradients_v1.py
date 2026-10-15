"""Seal a new measured-scope diagnostic; execute no neural or VM work."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
from import_cctv_dgp_bank_saved_direction_forward_v1_r2_return import ROOT, OUT, ANALYSIS, PARENT, read, sha, write
from cctv_dgp_bank_group_gradients_v1_contract import scopes

NAME = 'cctv_dgp_bank_group_gradients_v1_vm'
STEM = 'cctv-dgp-bank-group-gradients-v1'


def main():
    packet, archive = OUT/NAME, OUT/(STEM+'-execution.tar.gz')
    receipt = OUT/'cctv_dgp_bank_group_gradients_v1_preparation.json'
    assert not any(p.exists() for p in [packet,archive,receipt])
    evidence = [ANALYSIS/'independent_audit.json',ANALYSIS/'visual_review.json',ANALYSIS/'finite_direction_review.json']
    assert all(read(f)['complete'] for f in evidence)
    assert read(evidence[1])['all20_sheets_inspected'] and read(evidence[2])['bigger_proposals_cross1percent_mean_structure_but_source_and_preservation_fail']
    assert read(evidence[2])['detail_constraint_and_gate_both_use_four_degraded_profiles']
    old_packet = OUT/'cctv_dgp_bank_saved_direction_forward_v1_r2_vm'
    old = read(old_packet/'protocol.json')
    parent = read(PARENT/'protocol.json')
    assert sha(PARENT/'protocol.json') == old['parent_protocol_sha256']
    assert all(sha(PARENT/n)==h for n,h in old['parent_required_sha256'].items())
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    blocks, groups = scopes(parent['cases'],old['case_ids'])
    packet.mkdir();(packet/'scripts').mkdir()
    copied = ['cctv_dgp_bank_group_gradients_v1_vm.py','cctv_dgp_bank_group_gradients_v1_contract.py']
    for filename in copied:
        source = ROOT/'scripts'/filename
        text = source.read_text(encoding='utf-8')
        ast.parse(text,feature_version=(3,10))
        target = packet/('scripts' if filename.endswith('_vm.py') else '')/filename
        target.write_text(text,encoding='utf-8',newline='\n')
    for filename in ['directions.npz','direction_review.json']:
        assert sha(old_packet/filename) == old['assets_sha256'][filename]
        shutil.copyfile(old_packet/filename,packet/filename)
    shell = (old_packet/'scripts/run_forward.sh').read_text(encoding='utf-8')
    shell = shell.replace('cctv_dgp_bank_saved_direction_forward_v1_r2_vm.py','cctv_dgp_bank_group_gradients_v1_vm.py')
    shell = shell.replace('--kill-after=30 1230','--kill-after=30 930')
    (packet/'scripts/run_groups.sh').write_text(shell,encoding='utf-8',newline='\n')
    p = dict(format='frozen-source-profile-gradient-measurement-v1',name=NAME,UTC=datetime.now(timezone.utc).isoformat(),
        parent_protocol_sha256=old['parent_protocol_sha256'],parent_required_sha256=old['parent_required_sha256'],
        evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in evidence},
        compared_forward_R2_protocol_sha256=sha(old_packet/'protocol.json'),
        compared_forward_R2_archive_sha256=read(ANALYSIS/'import.json')['archive_sha256'],
        initial_model_state=old['initial_model_state'],learning_names=old['learning_names'],parameters=3364643,
        case_ids=old['case_ids'],reference_blocks=blocks,gradient_groups=groups,paired_TRAIN_cases=100,
        sample_gradient_begins=[180,1955],sample_queries=[0,2],
        frozen_original_checkpoint_sha256=parent['original_checkpoint_sha256'],
        source_labels_not_ethnicity_or_native_capture_claims=True,external_frozen_generator_not_own_trained=True,
        baseline_source_semantics='Same20 photographic TRAIN references; no native or reserved final images',
        gradient_scope='Two sources, five profiles, MSE/SSIM_loss/ArcFace_loss separately; source-degraded detail means',
        aggregate_gradient_dtype='float64',raw_neural_gradient_dtype='float32',mean_divisor_per_source=10,
        gradients_are_unscaled_scientific_metric_losses_not_absolute_anchor_gradients=True,
        saved_directions_used_for_dot_products_only_not_parameter_displacements=True,
        maximum_gradient_queries=320,model_forward_cap=40,recognizer_forward_cap=20,
        optimizer_updates=0,backwards=0,full_epochs=0,parameter_displacements=0,
        worker_seconds=900,measurement_seconds=600,export_seconds=300,external_seconds=930,export_external_seconds=330,kill_grace_seconds=30,
        minimum_free_bytes=4*1024**3,disk_reserve_bytes=1024**3,output_byte_cap=1280*1024**2,peak_VRAM_bytes=20*1024**3,peak_RSS_bytes=12*1024**3,
        raw_aggregate_gradient_sample_control_bytes=32*3364643*8+4*3364643*4+100*256*256*3*4,
        explicit_duration_projection={'first_completed_batches':2,'reference_batches':20,'safety_factor':1.5,'cap_seconds':600},
        gradient_vector_contributions_audited_from_logged_dots_but_not_independently_reexecuted_neurally=True,
        no_candidate_PNG_duplication_controls_identical_to_retained_baseline=True,
        preserved_scientific_thresholds=old['preserved_scientific_thresholds'],
        native_pixels_used=0,final_identity_pixels=0,failed_run_resume_allowed=False,app_promotion=False,automatic_follow_on=False,
        manual_tmux_required=True,independent_return_audit_required_before_new_direction_or_training=True,model_qualified=False,goal_complete=False)
    assert p['raw_aggregate_gradient_sample_control_bytes']+64*1024**2 < p['output_byte_cap']
    p['assets_sha256'] = {f.relative_to(packet).as_posix():sha(f) for f in packet.rglob('*') if f.is_file()}
    assert len(p['assets_sha256']) == 5
    write(packet/'protocol.json',p)
    pin = sha(packet/'protocol.json')
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in sorted(f for f in packet.rglob('*') if f.is_file()):
            tar.add(file,arcname=NAME+'/'+file.relative_to(packet).as_posix(),recursive=False)
    digest = sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    original_importer = ROOT/'scripts/import_cctv_dgp_bank_saved_direction_forward_v1_r2_return.py'
    new_importer = ROOT/'scripts/import_cctv_dgp_bank_group_gradients_v1_return.py'
    assert not new_importer.exists()
    text = original_importer.read_text(encoding='utf-8')
    text = text.replace('forward_v1_r2','group_gradients_v1').replace('forward-v1-r2','group-gradients-v1').replace(sha(old_packet/'protocol.json'),pin)
    ast.parse(text)
    new_importer.write_text(text,encoding='utf-8',newline='\n')
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
        name=NAME,stem=STEM,protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        packet_assets=5,parent_assets=len(p['parent_required_sha256']),gradient_groups=32,planned_gradient_queries=320,
        maximum_uncompressed_gradient_control_bytes=p['raw_aggregate_gradient_sample_control_bytes'],
        importer_sha256=sha(new_importer),importer_source_sha256=sha(original_importer),
        corrected_scientific_measurement_scope_not_unchanged_failed_recipe=True,
        scientific_thresholds_original_checkpoints_splits_and_failures_preserved=True,
        actual_manual_VM_diagnostic_unrun=True,local_neural_or_gradient_calls=0,local_optimizer_updates=0,
        agent_VM_connection_or_launch=False,model_qualified=False,goal_complete=False))
    print(dict(complete=True,protocol_sha256=pin,archive_bytes=archive.stat().st_size,planned_queries=320,manual_VM_diagnostic_unrun=True))


if __name__ == '__main__':
    main()
