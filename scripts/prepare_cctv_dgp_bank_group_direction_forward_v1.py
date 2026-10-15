"""Seal four forward-only manual L4 probes from all32 audited group derivatives."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import tarfile
from import_cctv_dgp_bank_group_gradients_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write

NAME='cctv_dgp_bank_group_direction_forward_v1_vm'
STEM='cctv-dgp-bank-group-direction-forward-v1'
OLD='cctv_dgp_bank_saved_direction_forward_v1_r2'
NEW='cctv_dgp_bank_group_direction_forward_v1'


def replace_once(source, old, new):
    assert source.count(old)==1, old
    return source.replace(old,new)


def scope(source):
    return source.replace(OLD,NEW).replace('cctv-dgp-bank-saved-direction-forward-v1-r2',STEM)


def write_source(path, value):
    ast.parse(value,feature_version=(3,10))
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(value)


def main():
    dest=OUT/NAME
    archive=OUT/(STEM+'-execution.tar.gz')
    receipt=OUT/(NEW+'_preparation.json')
    assert not any(f.exists() for f in [dest,archive,receipt])
    direction_root=ANALYSIS/'common_direction'
    direction=read(direction_root/'direction_review.json')
    audited=read(direction_root/'independent_audit.json')
    assert direction['complete'] and audited['complete']
    assert sha(direction_root/'direction_review.json')==audited['direction_review_sha256']
    assert sha(direction_root/'directions.npz')==audited['direction_file_sha256']
    assert direction['no_local_neural_calls_gradients_optimizer_or_parameter_displacement']
    previous=OUT/(OLD+'_return')
    old=read(previous/'protocol.json')
    assert sha(previous/'protocol.json')=='52e0bba3b4e6f203bdfef6a2793fe851280feb0574982a64afa5eee587696641'
    assert read(OUT/(OLD+'_analysis')/'independent_audit.json')['complete']
    assert read(OUT/(OLD+'_analysis')/'visual_review.json')['complete']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    for name,digest in old['parent_required_sha256'].items():
        assert sha(PARENT/name)==digest
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    original_worker=previous/'scripts'/f'{OLD}_vm.py'
    source=scope(original_worker.read_text(encoding='utf-8'))
    source=replace_once(source,"[{'label': label+'_'+rate, 'direction': label, 'rate': float(rate)}\n        for label in ['ordinary', 'constrained'] for rate in ['0.0001', '0.00001']]",
        "[{'label': 'balanced_'+rate, 'direction': 'balanced', 'rate': float(rate)}\n        for rate in ['0.005', '0.01', '0.02', '0.04']]")
    source=source.replace("['ordinary', 'constrained']","['balanced']")
    source=replace_once(source,"    assert p['worker_seconds'] == 1200 and p['external_seconds'] == 1230",
        "    assert p['worker_seconds'] == 1200 and p['external_seconds'] == 1230\n    assert p['peak_RSS_bytes'] == 12*1024**3\n    assert p['gradient_groups_used_in_direction'] == 32 and p['proposal_derivation_TRAIN_references'] == 20")
    source=replace_once(source,"    assert direction['no_local_neural_calls_gradients_optimizer_or_parameter_displacement']",
        "    assert direction['no_local_neural_calls_gradients_optimizer_or_parameter_displacement']\n    assert direction['rate_candidates'] == [.005, .01, .02, .04]\n    assert direction['maximum_prospective_displacement_relative_to_parameter_L2'] < .001")
    source=replace_once(source,"    assert shutil.disk_usage(root).free >= p['minimum_free_bytes'], 'Need4GiB free'",
        """    available = shutil.disk_usage(root).free
    if available < p['minimum_free_bytes']:
        early = root/'outputs/group_direction_forward_v1'
        early.mkdir(parents=True)
        write(early/'failure.json', dict(stage='pre-model storage preflight',
            available_free_bytes=available, required_free_bytes=p['minimum_free_bytes'],
            neural_calls=0, gradient_queries=0, optimizer_updates=0, parameter_displacements=0))
        raise AssertionError('Need4GiB free after install; structured storage stop retained')""")
    source=source.replace('outputs/saved_direction_forward_v1','outputs/group_direction_forward_v1')
    source=replace_once(source,'    import torch\n','    import torch\n    import resource\n')
    source=replace_once(source,"        assert torch.cuda.max_memory_allocated() <= p['peak_VRAM_bytes'], 'VRAM stop'",
        "        assert torch.cuda.max_memory_allocated() <= p['peak_VRAM_bytes'], 'VRAM stop'\n        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= p['peak_RSS_bytes'], 'RSS stop'")
    source=replace_once(source,'            candidate_initial_state_restored=True, original_and_recognizer_unchanged=True,',
        '            peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,\n            candidate_initial_state_restored=True, original_and_recognizer_unchanged=True,')
    source=replace_once(source,'                   neural_calls=0, gradient_queries=0, optimizer_updates=0, fixed_target_affines_checked=100)',
        "                   neural_calls=0, gradient_queries=0, optimizer_updates=0, fixed_target_affines_checked=100,\n                   available_free_bytes=shutil.disk_usage(root).free, required_free_bytes=p['minimum_free_bytes'])")
    source_path=ROOT/'scripts'/f'{NEW}_vm.py'
    write_source(source_path,source)
    tree=ast.parse(source,feature_version=(3,10))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                   n.func.attr in ['grad','backward','step','Adam','AdamW','SGD','enable_vm_learning'] for n in ast.walk(tree))
    dest.mkdir();(dest/'scripts').mkdir()
    (dest/'scripts'/source_path.name).write_bytes(source_path.read_bytes())
    for name in ['directions.npz','direction_review.json']:
        (dest/name).write_bytes((direction_root/name).read_bytes())
    shell=scope((previous/'scripts/run_forward.sh').read_text(encoding='utf-8'))
    (dest/'scripts/run_forward.sh').write_text(shell,encoding='utf-8',newline='\n')
    inherited=['parent_protocol_sha256','parent_required_sha256','case_ids','learning_names','initial_model_state',
        'preserved_scientific_thresholds','native_pixels_used','final_identity_pixels','optimizer_updates','gradient_queries',
        'backwards','full_epochs','maximum_temporary_parameter_proposals','model_forward_cap','recognizer_forward_cap',
        'worker_seconds','measurement_seconds','external_seconds','export_seconds','export_external_seconds','kill_grace_seconds',
        'peak_VRAM_bytes','minimum_free_bytes','disk_reserve_bytes','output_byte_cap','exact_initial_raw_controls',
        'raw_case_replays','new_candidate_forward_cases','PNG_export_policy','alignment_policy']
    p={key:old[key] for key in inherited}
    p.update(format='all-source-profile-common-direction-forward-v1',name=NAME,UTC=datetime.now(timezone.utc).isoformat(),
        proposals=[dict(label='balanced_'+rate,direction='balanced',rate=float(rate)) for rate in ['0.005','0.01','0.02','0.04']],
        paired_TRAIN_cases=100,TRAIN_references=20,proposal_derivation_TRAIN_references=20,
        gradient_groups_used_in_direction=32,peak_RSS_bytes=12*1024**3,
        prospective_maximum_parameter_displacement_relative_L2=direction['maximum_prospective_displacement_relative_to_parameter_L2'],
        finite_all17_group_raw_PNG_preservation_and_brightness_requirements_unchanged=True,
        additional_source_degraded_structure_gain_required=.01,
        direction_design='minimum L2 source/profile inequality solve from all32 saved derivatives; float32 dots independently audited',
        first_order_margin_for_all30_preservation_groups=.1,
        raw_and_PNG_outputs_and_embeddings_separate=True,actual_finite_parameter_displacements_on_manual_VM_only=True,
        no_training_worker_import=True,no_optimizer_construction=True,no_new_neural_derivatives=True,
        initial_state_restored_between_proposals_and_after_run=True,no_checkpoint_created=True,
        no_strength_or_step_selection_in_this_worker=True,failed_run_resume_allowed=False,
        app_promotion=False,automatic_follow_on=False,manual_tmux_required=True,native_unpaired=True,
        all_seven_completion_families_pending=True,independent_final_review_pending=True,goal_complete=False,
        all100_fixed_affines_shape_finite_determinant_checked_before_neural_import=True,
        raw_output_saved_before_identity_measurement=True,recognizer_counter_incremented_after_each_actual_forward=True,
        scientific_recipe_change='All20 reference gradients and distinct32 source/profile losses replace two-reference Adam projection',
        evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in [ANALYSIS/'import.json',
            ANALYSIS/'independent_audit.json',direction_root/'direction_review.json',direction_root/'independent_audit.json',
            OUT/(OLD+'_analysis')/'independent_audit.json',OUT/(OLD+'_analysis')/'visual_review.json']})
    p['assets_sha256']={f.relative_to(dest).as_posix():sha(f) for f in dest.rglob('*') if f.is_file()}
    write(dest/'protocol.json',p);pin=sha(dest/'protocol.json')
    with tarfile.open(archive,'x:gz',compresslevel=1) as stream:
        for file in sorted(dest.rglob('*')):
            if file.is_file():stream.add(file,arcname=NAME+'/'+file.relative_to(dest).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    old_pin='52e0bba3b4e6f203bdfef6a2793fe851280feb0574982a64afa5eee587696641'
    importer_path=ROOT/'scripts'/f'import_{NEW}_return.py'
    value=scope((ROOT/'scripts'/f'import_{OLD}_return.py').read_text(encoding='utf-8')).replace(old_pin,pin)
    write_source(importer_path,value)
    auditor_path=ROOT/'scripts'/f'audit_{NEW}_return.py'
    value=scope((ROOT/'scripts'/f'audit_{OLD}_return_timing_r1.py').read_text(encoding='utf-8'))
    value=value.replace('outputs/saved_direction_forward_v1','outputs/group_direction_forward_v1')
    value=replace_once(value,"        assert len(archive.files)==74","        assert len(archive.files)==37")
    value=value.replace("['ordinary','constrained']","['balanced']")
    value=replace_once(value,"        titles=['Input','Retained DGP','Ordinary 1e-4','Ordinary 1e-5','Constrained 1e-4','Constrained 1e-5','Clean TRAIN target','Observed mask']",
        "        titles=['Input','Retained DGP','Group step .005','Group step .01','Group step .02','Group step .04','Clean TRAIN target','Observed mask']")
    value=replace_once(value,"        timing_audit_repair_sha256=sha(ANALYSIS/'timing_audit_repair.json'),\n",'')
    value=replace_once(value,"        original_prepared_checker_sha256=sha(ROOT/'scripts/audit_cctv_dgp_bank_group_direction_forward_v1_return.py'),\n",'')
    value=replace_once(value,"    assert 0<run['seconds']<p['worker_seconds'] and 0<run['peak_VRAM']<=p['peak_VRAM_bytes']",
        "    assert 0<run['seconds']<p['worker_seconds'] and 0<run['peak_VRAM']<=p['peak_VRAM_bytes']\n    assert 0<run['peak_RSS']<=p['peak_RSS_bytes']==12*1024**3")
    value=replace_once(value,"        results[label]=dict(rows=rows,groups=groups,gates=gates,all_preview_requirements_pass=all(g['pass_all_preview_requirements'] for g in gates.values()),",
        """        sources=sorted({r['source'] for r in rows})
        source_structure={stage:{source:1-groups[stage][source+'/degraded']['landmark_high_frequency_MSE']/groups_before[stage][source+'/degraded']['landmark_high_frequency_MSE'] for source in sources} for stage in ['raw','png']}
        source_pass=all(gain>=p['additional_source_degraded_structure_gain_required'] for values in source_structure.values() for gain in values.values())
        results[label]=dict(rows=rows,groups=groups,gates=gates,source_structure_gains=source_structure,
            source_structure_requirement_pass=source_pass,all_preview_requirements_pass=source_pass and all(g['pass_all_preview_requirements'] for g in gates.values()),""")
    # Preserve infrastructure/partial failures without fabricating completed outputs.
    value=replace_once(value,"    assert manifest['complete_diagnostic'] and not manifest['failure_present']",
        """    if not manifest['complete_diagnostic']:
        assert all(sha(RETURN/n)==h for n,h in manifest['files'].items())
        assert all(sha(PARENT/n)==h for n,h in p['parent_required_sha256'].items())
        out=RETURN/'outputs/group_direction_forward_v1'
        assert int((RETURN/'review_exit_code.txt').read_text().strip())!=0
        failure=read(out/'failure.json') if (out/'failure.json').exists() else None
        assert manifest['failure_present']==bool(failure)
        if failure and failure.get('stage')=='pre-model storage preflight':
            assert failure['available_free_bytes']<failure['required_free_bytes']==p['minimum_free_bytes']
            assert failure['neural_calls']==failure['gradient_queries']==failure['optimizer_updates']==failure['parameter_displacements']==0
        protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
        assert all(sha(ROOT/n)==h for n,h in protected.items())
        write(ANALYSIS/'failure_audit.json',dict(complete=True,checker_sha256=sha(__file__),failure=failure,
            review_log_sha256=sha(RETURN/'review.log'),run_complete=False,partial_outputs_not_qualified=True,
            no_neural_success_or_execution_count_inferred_from_export_flag=True,
            no_local_neural_gradient_optimizer_or_parameter_displacement=True,model_qualified=False,goal_complete=False))
        print(dict(complete=True,retained_failure_audited=True,model_qualified=False));return
    assert not manifest['failure_present']""")
    write_source(auditor_path,value)
    helper_path=ROOT/'scripts'/f'prepare_{NEW}_return_checker.py'
    value=scope((ROOT/'scripts/prepare_cctv_dgp_bank_saved_direction_return_checker_v1_r2.py').read_text(encoding='utf-8'))
    write_source(helper_path,value)
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
        name=NAME,stem=STEM,protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        assets=len(p['assets_sha256']),parent_required_assets=len(p['parent_required_sha256']),
        common_direction_independent_audit_sha256=sha(direction_root/'independent_audit.json'),
        original_worker_sha256=sha(original_worker),new_worker_sha256=sha(source_path),
        future_checker_files_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in [importer_path,auditor_path,helper_path]},
        scientific_thresholds_and_identical_inputs_preserved=True,new_direction_from_all32_audited_groups=True,
        manual_VM_run_unrun=True,no_agent_VM_connection_or_launch=True,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,model_qualified=False,goal_complete=False))
    print(dict(complete=True,archive_bytes=archive.stat().st_size,protocol_sha256=pin,manual_VM_run_unrun=True))


if __name__=='__main__':
    main()
