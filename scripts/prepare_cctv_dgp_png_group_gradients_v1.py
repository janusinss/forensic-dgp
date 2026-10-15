"""Seal delivered-loss measurement for manual L4 use; no local neural calls."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile

from import_cctv_dgp_bank_group_step_check_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write

NAME = 'cctv_dgp_png_group_gradients_v1_vm'
STEM = 'cctv-dgp-png-group-gradients-v1'
BASE = 'cctv_dgp_png_group_gradients_v1'


def replace_once(text, old, new):
    assert text.count(old) == 1, 'Source transform not uniquely bound: ' + old[:90]
    return text.replace(old, new)


def build_worker():
    original = ROOT/'outputs/cctv_dgp_bank_group_gradients_v1_r1_return/scripts/cctv_dgp_bank_group_gradients_v1_r1_vm.py'
    text = original.read_text(encoding='utf-8').replace('group_gradients_v1_r1', 'png_group_gradients_v1').replace('group-gradients-v1-r1', 'png-group-gradients-v1')
    text = text.replace('cctv_dgp_bank_png_group_gradients', 'cctv_dgp_png_group_gradients')
    text = text.replace("['ordinary','constrained']", "['balanced']")
    text = replace_once(text,
        '    from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses',
        '    from cctv_dgp_png_group_gradients_v1_losses import delivered_surrogates, pixel_and_structure_losses')
    text = replace_once(text, "    direction = read(root/'direction_review.json')", "    assert p['metric_stage']=='PNG' and p['straight_through_backward_is_approximate']\n    direction = read(root/'direction_review.json')")
    text = replace_once(text,
        "metric_gradient_measurement_only=True))",
        "metric_gradient_measurement_only=True,metric_stage='PNG',\n            exact_delivered_pixels_forward=True,straight_through_backward_is_approximate=True))")
    text = replace_once(text,
        "                assert np.array_equal(raw,baseline), 'Same L4 initial raw control differs; retain stop'",
        """                assert np.array_equal(raw,baseline), 'Same L4 initial raw control differs; retain stop'
                png = np.floor(raw*np.float32(255)).astype(np.uint8)
                for index,case in enumerate(cs):
                    png[index][~masks[index]] = cameras[index][~masks[index]]
                    assert np.array_equal(png[index],pixels(parent/'outputs/bank_comparison_v1/baseline/previews'/(case['id']+'.png')))
                png_path = out/'controls'/f'b{begin//5:03d}_PNG.npy'
                with png_path.open('xb') as stream: np.save(stream,png,allow_pickle=False)""")
    text = replace_once(text,
        "            assert maximum_error <= 1e-6, 'Gradient/frozen forward disagreement; retain stop'\n            terms = pixel_and_structure_losses(pred,target,mask,feature)\n            vectors = identity.embedding(torch.cat([pred,baseline_tensor.detach(),target]),torch.cat([mask]*3),torch.cat([grids]*3))\n            assert np.allclose(vectors[5:10].detach().cpu().numpy(),base_emb[:,0],rtol=1e-6,atol=1e-6)",
        """            assert maximum_error == 0, 'Gradient/frozen raw forward differs; retain stop before surrogate'
            delivered_u8 = torch.from_numpy(png).permute(0,3,1,2).contiguous().cuda()
            pred_png, feature_png64 = delivered_surrogates(pred,delivered_u8)
            assert np.array_equal(pred_png.detach().permute(0,2,3,1).cpu().numpy(),png.astype(np.float32)/np.float32(255))
            assert np.array_equal(feature_png64.detach().permute(0,2,3,1).cpu().numpy(),png.astype(np.float64)/255.)
            terms = pixel_and_structure_losses(pred_png,target,mask,feature,feature_png64)
            baseline_png = tensor(png)
            vectors = identity.embedding(torch.cat([pred_png,baseline_png.detach(),target]),torch.cat([mask]*3),torch.cat([grids]*3))
            assert np.allclose(vectors[5:10].detach().cpu().numpy(),base_emb[:,1],rtol=1e-6,atol=1e-6)""")
    text = replace_once(text,
        "            values = batch_values(terms,arc)\n            query_rows = []",
        """            values = batch_values(terms,arc)
            baseline_metrics = read(parent/'outputs/bank_comparison_v1/baseline/metrics.json')
            baseline_rows = {row['id']:row for row in baseline_metrics['rows']}
            expected_values = [value for case in cs for value in
                [baseline_rows[case['id']]['png']['MSE'],1-baseline_rows[case['id']]['png']['SSIM'],
                 1-baseline_rows[case['id']]['png']['ArcFace_observed_fixed']]]
            expected_values += [float(np.mean([baseline_rows[case['id']]['png']['landmark_high_frequency_MSE'] for case in cs[1:]]))]
            assert len(values)==len(expected_values)==16
            for query,(value,expected) in enumerate(zip(values,expected_values)):
                tolerance = p['forward_loss_numeric_tolerances'][['MSE','SSIM_loss','ArcFace_loss'][query%3] if query<15 else 'feature_MSE']
                assert abs(float(value.detach())-expected)<=tolerance, 'Initial delivered loss definition differs; retain stop'
            embedding_path = out/'controls'/f'b{begin//5:03d}_PNG_embeddings.npy'
            with embedding_path.open('xb') as stream: np.save(stream,vectors.detach().cpu().numpy(),allow_pickle=False)
            query_rows = []""")
    text = replace_once(text,
        "                raw_control_exact=True,input_NCHW_contiguous=x.is_contiguous(),target_NCHW_contiguous=target.is_contiguous())",
        """                raw_control_exact=True,input_NCHW_contiguous=x.is_contiguous(),target_NCHW_contiguous=target.is_contiguous(),
                PNG_control_file=png_path.relative_to(out).as_posix(),PNG_control_sha256=sha(png_path),
                PNG_embeddings_file=embedding_path.relative_to(out).as_posix(),PNG_embeddings_sha256=sha(embedding_path),
                actual_PNG_RGB_sha256=hashlib.sha256(png.tobytes()).hexdigest(),
                exact_RGB32_and_feature64_PNG_forward=True,straight_through_backward_is_approximate=True,
                expected_initial_PNG_loss_values=expected_values,forward_loss_numeric_tolerances=p['forward_loss_numeric_tolerances'])""")
    text = replace_once(text,
        "            del pred,frozen,terms,vectors,arc,target_vector,values,x,target,mask,feature,grids,baseline_tensor,raw",
        "            del pred,pred_png,feature_png64,delivered_u8,baseline_png,frozen,terms,vectors,arc,target_vector,values,x,target,mask,feature,grids,baseline_tensor,raw,png")
    text = replace_once(text,
        "            groups_not_independent_final_identities=True,independent_saved_arithmetic_audit_pending=True,",
        "            groups_not_independent_final_identities=True,independent_saved_arithmetic_audit_pending=True,\n            metric_stage='PNG',straight_through_backward_is_approximate=True,all100_delivered_forward_pixels_exact=True,")
    text = replace_once(text,
        "    assert shutil.disk_usage(root).free >= p['disk_reserve_bytes']+sum(f.stat().st_size for f in files), 'Retain reserve before export'",
        "    payload_bytes=sum(f.stat().st_size for f in files)\n    assert shutil.disk_usage(root).free >= p['disk_reserve_bytes']+payload_bytes+1024**2, 'Retain reserve plus full payload before export'")
    text = replace_once(text,
        "            assert shutil.disk_usage(root).free >= p['disk_reserve_bytes'], 'Export disk reserve stop'",
        "            assert shutil.disk_usage(root).free >= p['disk_reserve_bytes']+file.stat().st_size, 'Export disk reserve plus next member stop'")
    text = replace_once(text,
        '    digest = sha(archive)\n    with Path(str(archive)',
        "    assert shutil.disk_usage(root).free>=p['disk_reserve_bytes'] and time.monotonic()-stamp<p['export_seconds']\n    digest = sha(archive)\n    with Path(str(archive)")
    assert 'group_gradients_v1_r1' not in text and 'cctv_dgp_bank_png_' not in text
    ast.parse(text, feature_version=(3, 10))
    return text, sha(original)


def build_contract():
    source = ROOT/'scripts/cctv_dgp_bank_group_gradients_v1_contract.py'
    text = source.read_text(encoding='utf-8')
    text = replace_once(text,
        "    assert len(groups) == 32 and len({g['key'] for g in groups}) == 32",
        "    groups=[dict(group,key='PNG/'+group['key'],stage='PNG') for group in groups]\n    assert len(groups) == 32 and len({g['key'] for g in groups}) == 32")
    text = replace_once(text,
        "    return source+('/'+PROFILES[query//3]+'/'+METRICS[query%3] if query < 15 else '/degraded/feature_MSE')",
        "    return 'PNG/'+source+('/'+PROFILES[query//3]+'/'+METRICS[query%3] if query < 15 else '/degraded/feature_MSE')")
    ast.parse(text, feature_version=(3, 10))
    return text, sha(source)


def main():
    packet, archive, receipt = OUT/NAME, OUT/(STEM+'-execution.tar.gz'), OUT/(BASE+'_preparation.json')
    assert not any(path.exists() for path in [packet, archive, receipt])
    evidence = ['independent_audit_r1.json', 'signed_derivative_review.json', 'actual_step_direction_review.json',
        'quantization_review.json', 'visual_review.json', 'review_binding_audit.json', 'next_design_review.json', 'publication_audit.json']
    assert all(read(ANALYSIS/name)['complete'] for name in evidence)
    forward_path=OUT/(BASE+'_forward_audit.json')
    forward=read(forward_path)
    assert forward['complete'] and forward['saved_forward_cases']==100
    assert forward['loss_helper_sha256']==sha(ROOT/'scripts/cctv_dgp_png_group_gradients_v1_losses.py')
    assert read(ANALYSIS/'next_design_review.json')['exact_png_floor_forward_with_explicit_surrogate_backward_is_only_a_hypothesis']
    assert read(ANALYSIS/'visual_review.json')['all20_sheets_and800_cells_inspected']
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    old = read(RETURN/'protocol.json')
    parent = read(PARENT/'protocol.json')
    assert sha(PARENT/'protocol.json')==old['parent_protocol_sha256']
    assert all(sha(PARENT/name)==digest for name,digest in old['parent_required_sha256'].items())
    worker, worker_source_sha = build_worker()
    contract, contract_source_sha = build_contract()
    packet.mkdir(); (packet/'scripts').mkdir()
    with (packet/'scripts/cctv_dgp_png_group_gradients_v1_vm.py').open('x',encoding='utf-8',newline='\n') as f: f.write(worker)
    with (packet/'cctv_dgp_png_group_gradients_v1_contract.py').open('x',encoding='utf-8',newline='\n') as f: f.write(contract)
    losses_path = ROOT/'scripts/cctv_dgp_png_group_gradients_v1_losses.py'
    ast.parse(losses_path.read_text(),feature_version=(3,10))
    shutil.copyfile(losses_path,packet/losses_path.name)
    for name in ['directions.npz','direction_review.json']:
        shutil.copyfile(RETURN/name,packet/name)
    shell = (ROOT/'outputs/cctv_dgp_bank_group_gradients_v1_r1_return/scripts/run_groups.sh').read_text()
    shell = shell.replace('cctv_dgp_bank_group_gradients_v1_r1_vm.py','cctv_dgp_png_group_gradients_v1_vm.py')
    shell = replace_once(shell,'diagnostic_exit=${statuses[0]}','diagnostic_exit=${statuses[0]}\ntest "${statuses[1]}" -eq 0 || diagnostic_exit=${statuses[1]}')
    shell = replace_once(shell,'export_exit=${statuses[0]}','export_exit=${statuses[0]}\ntest "${statuses[1]}" -eq 0 || export_exit=${statuses[1]}')
    with (packet/'scripts/run_png_groups.sh').open('x',encoding='utf-8',newline='\n') as f: f.write(shell)
    namespace = {}
    exec(compile(contract,'prepared pure contract','exec'),namespace)
    blocks, groups = namespace['scopes'](parent['cases'],old['case_ids'])
    p = dict(format='exact-delivered-forward-surrogate-gradient-measurement-v1', name=NAME, UTC=datetime.now(timezone.utc).isoformat(),
        parent_protocol_sha256=old['parent_protocol_sha256'],parent_required_sha256=old['parent_required_sha256'],
        initial_model_state=old['initial_model_state'],learning_names=old['learning_names'],parameters=3364643,
        case_ids=old['case_ids'],reference_blocks=blocks,gradient_groups=groups,paired_TRAIN_cases=100,
        sample_gradient_begins=[180,1955],sample_queries=[0,2],metric_stage='PNG',
        frozen_original_checkpoint_sha256=parent['original_checkpoint_sha256'],
        source_labels_not_ethnicity_or_native_capture_claims=True,external_frozen_generator_not_own_trained=True,
        baseline_source_semantics='Same20 photographic TRAIN references; no native or reserved final images',
        straight_through_backward_is_approximate=True,exact_RGB32_and_feature64_PNG_forward_required=True,
        surrogate_definition='raw+(dequantized_exact_PNG-raw).detach(); RGB32 and byte/255 feature64; scientific scikit-image SSIM forward with analytic float64 SSIM backward surrogate',
        SSIM_scientific_forward_from_actual_pixels_not_cached_loss_injection=True,
        no_gradient_of_discontinuous_PNG_claim=True,
        forward_loss_numeric_tolerances=dict(MSE=1e-13,SSIM_loss=1e-6,ArcFace_loss=1e-6,feature_MSE=1e-12),
        forward_loss_numeric_tolerances_are_reproduction_checks_not_quality_gate_changes=True,
        exact_zero_raw_gradient_vs_frozen_forward_error_required=True,
        aggregate_gradient_dtype='float64',raw_neural_gradient_dtype='float32',mean_divisor_per_source=10,
        saved_balanced_direction_used_for_dots_only=True,old_raw_gradient_groups_remain_separate=True,
        comparison_aim='Test whether exact delivered-image forward losses supply a different useful learning direction alongside retained raw constraints',
        maximum_gradient_queries=320,model_forward_cap=40,recognizer_forward_cap=20,
        optimizer_updates=0,backwards=0,full_epochs=0,parameter_displacements=0,
        worker_seconds=900,measurement_seconds=600,export_seconds=300,external_seconds=930,export_external_seconds=330,kill_grace_seconds=30,
        minimum_free_bytes=4*1024**3,disk_reserve_bytes=1024**3,output_byte_cap=1280*1024**2,peak_VRAM_bytes=20*1024**3,peak_RSS_bytes=12*1024**3,
        uncompressed_gradient_sample_control_bytes=32*3364643*8+4*3364643*4+100*256*256*3*5+20*15*512*4,
        explicit_duration_projection=dict(first_completed_batches=2,reference_batches=20,safety_factor=1.5,cap_seconds=600),
        finite_step_after_measurement_is_separate_manual_work=True,
        preserved_scientific_thresholds=old['preserved_scientific_thresholds'],
        native_pixels_used=0,final_identity_pixels=0,failed_run_resume_allowed=False,app_promotion=False,automatic_follow_on=False,
        manual_tmux_required=True,independent_return_audit_required_before_new_direction_or_training=True,
        evidence_sha256={'outputs/cctv_dgp_bank_group_step_check_v1_analysis/'+name:sha(ANALYSIS/name) for name in evidence},
        derivation_sha256=dict(worker_source=worker_source_sha,contract_source=contract_source_sha,loss_source=sha(losses_path)),
        same_original_checkpoint_splits_baseline_and_all_gate_failures_retained=True,
        meaningful_changed_learning_measurement_not_unchanged_failed_recipe=True,model_qualified=False,goal_complete=False)
    p['local_saved_value_forward_audit_sha256']=sha(forward_path)
    assert p['uncompressed_gradient_sample_control_bytes']+64*1024**2<p['output_byte_cap']
    p['assets_sha256']={path.relative_to(packet).as_posix():sha(path) for path in packet.rglob('*') if path.is_file()}
    assert len(p['assets_sha256'])==6
    write(packet/'protocol.json',p);pin=sha(packet/'protocol.json')
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in sorted(path for path in packet.rglob('*') if path.is_file()):
            tar.add(file,arcname=NAME+'/'+file.relative_to(packet).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f: f.write(digest+'  '+archive.name+'\n')
    importer_source=ROOT/'scripts/import_cctv_dgp_bank_group_step_check_v1_return.py'
    importer=importer_source.read_text().replace('cctv_dgp_bank_group_step_check_v1','cctv_dgp_png_group_gradients_v1').replace('cctv-dgp-bank-group-step-check-v1',STEM).replace(old_pin := sha(RETURN/'protocol.json'),pin)
    ast.parse(importer)
    new_importer=ROOT/'scripts/import_cctv_dgp_png_group_gradients_v1_return.py'
    with new_importer.open('x',encoding='utf-8',newline='\n') as f: f.write(importer)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
        name=NAME,stem=STEM,protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        packet_assets=6,parent_assets=len(p['parent_required_sha256']),gradient_groups=32,planned_gradient_queries=320,
        maximum_uncompressed_gradient_control_bytes=p['uncompressed_gradient_sample_control_bytes'],
        importer_sha256=sha(new_importer),importer_source_sha256=sha(importer_source),
        local_saved_value_forward_audit_sha256=sha(forward_path),
        actual_manual_VM_diagnostic_unrun=True,zero_local_neural_gradients_optimizer_or_parameter_displacement=True,
        no_VM_connection_or_launch=True,exact_delivered_forward_surrogate_backward_is_new_measurement=True,
        scientific_gates_original_checkpoint_splits_and_failures_preserved=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,name=NAME,protocol_sha256=pin,archive_sha256=digest,bytes=archive.stat().st_size))


if __name__=='__main__':main()
