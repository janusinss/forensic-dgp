"""Prepare a changed, untrained spatial-path diagnostic; never execute models."""
import ast
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import tarfile

import numpy as np
import torch

from import_cctv_dgp_joint_raw_png_forward_v1_return import ROOT, OUT, ANALYSIS, PARENT, read, sha, write

NAME='cctv_dgp_live_conditioning_v1_vm'
STEM='cctv-dgp-live-conditioning-v1'
OLD_NAME='cctv_dgp_bank_group_gradients_v1_r1_vm'


def replace_once(text, before, after):
    assert text.count(before)==1, before[:160]
    return text.replace(before,after)


def state_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous()
        h.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0')
        h.update(str(tuple(a.shape)).encode()+b'\0'+a.numpy().tobytes())
    return h.hexdigest()


def main():
    packet=OUT/NAME;archive=OUT/(STEM+'-execution.tar.gz')
    receipt=OUT/'cctv_dgp_live_conditioning_v1_preparation.json'
    assert not any(path.exists() for path in [packet,archive,receipt])
    review=read(ANALYSIS/'review_binding_audit.json')
    assert review['complete'] and review['all_four_proposals_rejected_for_adoption']
    assert review['exact_unchanged_conditioning_tensors']==12
    old_root=OUT/OLD_NAME
    old=read(old_root/'protocol.json')
    old_audit=read(OUT/'cctv_dgp_bank_group_gradients_v1_r1_analysis/independent_audit.json')
    assert old_audit['complete'] and old_audit['all100_raw_controls_match']
    assert sha(old_root/'protocol.json')==old_audit['protocol_sha256']
    for name,digest in old['assets_sha256'].items(): assert sha(old_root/name)==digest
    for name,digest in old['parent_required_sha256'].items(): assert sha(PARENT/name)==digest
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    packet.mkdir();(packet/'scripts').mkdir()
    torch.set_grad_enabled(False)
    initial=torch.load(PARENT/'initializers/initial_B.pth',map_location='cpu',weights_only=True)
    readouts={};stats={};rng=np.random.default_rng(20261013)
    for size in [16,32,64,128,256]:
        shape=tuple(initial[f'fusions.{size}.weight'].shape);dest,inputs=shape[:2]
        assert shape==(dest,inputs,1,1) and inputs in [64,128,256,512] and dest<inputs
        h=np.ones((1,1),np.int8)
        while len(h)<inputs: h=np.block([[h,h],[h,-h]])
        rows=rng.permutation(np.arange(1,inputs))[:dest]
        columns=rng.permutation(inputs)
        signs=rng.choice(np.array([-1,1],np.int8),size=inputs)
        weight=(h[rows][:,columns]*signs).astype(np.float32)*np.float32(.125/np.sqrt(inputs))
        assert np.allclose(weight.astype(np.float64)@weight.astype(np.float64).T,np.eye(dest)*.125**2,rtol=0,atol=2e-9)
        readouts[f'f{size}_weight']=weight[:,:,None,None].copy()
        readouts[f'f{size}_bias']=np.zeros(dest,np.float32)
        fixture=rng.normal(size=(inputs,9)).astype(np.float32)
        actual=weight@fixture;anchor=weight@fixture
        assert np.array_equal(actual-anchor,np.zeros_like(actual))
        delta=np.ones_like(fixture)*np.float32(.001);delta[::2]*=-1
        # Illustrative matrix arithmetic only; not a neural derivative/face test.
        response=float(np.linalg.norm(weight.astype(np.float64)@delta.astype(np.float64)))
        assert response>0
        stats[str(size)]=dict(shape=list(shape),row_L2=.125,orthonormal_rows=True,
            zero_parity_matrix_fixture=True,nonzero_matrix_response_fixture=response,
            fixture_not_neural_or_image_or_quality_evidence=True)
    np.savez_compressed(packet/'readouts.npz',**readouts)
    virtual={'live.'+name:value for name,value in initial.items()}
    for size in [16,32,64,128,256]:
        for term in ['weight','bias']:
            value=torch.from_numpy(readouts[f'f{size}_{term}'])
            virtual[f'live.fusions.{size}.{term}']=value
            virtual[f'anchor_fusions.{size}.{term}']=value
    for name,value in initial.items():
        if name.startswith('style_conditioner.'):
            virtual['anchor_style_conditioner.'+name.removeprefix('style_conditioner.')]=value
        elif name.startswith('spatial_conditioners.'):
            virtual['anchor_spatial_conditioners.'+name.removeprefix('spatial_conditioners.')]=value
    expected_state=state_sha(virtual)
    assert state_sha(initial)==old['initial_model_state']
    model_source=ROOT/'scripts/cctv_dgp_live_conditioning_v1_model.py'
    ast.parse(model_source.read_text(encoding='utf-8'),feature_version=(3,10))
    shutil.copyfile(model_source,packet/'cctv_dgp_live_conditioning_v1_model.py')
    source=(old_root/'scripts/cctv_dgp_bank_group_gradients_v1_r1_vm.py').read_text(encoding='utf-8')
    source=source.replace('cctv_dgp_bank_group_gradients_v1_r1','cctv_dgp_live_conditioning_v1')
    source=source.replace('group_gradients_v1_r1','live_conditioning_v1').replace('group-gradients-v1-r1','live-conditioning-v1')
    source=replace_once(source,"    assert direction['names'] == p['learning_names'] and direction['initial_file_sha256'] == sha(parent/'initializers/initial_B.pth')",
        "    assert direction['names'] == [n.removeprefix('live.') for n in p['learning_names']] and direction['initial_file_sha256'] == sha(parent/'initializers/initial_B.pth')\n"
        "    assert p['new_architecture_diagnostic_only'] and p['nonzero_readout_row_L2']==.125\n"
        "    assert p['conditioning_tensors_required']==12 and not p['training_launch_ready']")
    source=replace_once(source,'    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP',
        '    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP\n    from cctv_dgp_live_conditioning_v1_model import AnchoredBankDGP')
    source=replace_once(source,'        candidate.load_state_dict(initial,strict=True)\n        candidate.enable_vm_learning(parent)',
        "        candidate.load_state_dict(initial,strict=True)\n        candidate = AnchoredBankDGP(candidate,root/'readouts.npz',root).cuda()\n        candidate.enable_vm_learning(parent)")
    # Contribution scope stays identical; include the new norm fields in both
    # the query record and aggregate contribution record for strict binding.
    source=replace_once(source,"                contributions[key].append(contribution)\n                query_rows.append(dict(key=key,**contribution))",
        "                contribution['conditioning_tensor_gradient_norms']={names[i]:float(np.linalg.norm(arrays[i].astype(np.float64)))\n                    for i in range(37) if names[i].startswith(('live.style_conditioner.','live.spatial_conditioners.'))}\n                contributions[key].append(contribution)\n                query_rows.append(dict(key=key,**contribution))")
    source=replace_once(source,"        assert state_hash(candidate) == before and buffer_hash(candidate) == buffers",'''        edges=np.cumsum([0]+[v.numel() for v in params])
        closed=[i for i,n in enumerate(names) if n.startswith(('live.style_conditioner.','live.spatial_conditioners.'))]
        assert len(closed)==12
        live_norms={key:{names[i]:float(np.linalg.norm(aggregates[key][edges[i]:edges[i+1]]))
                    for i in closed} for key in keys}
        write(out/'conditioning_gradient_norms.json',dict(groups=live_norms,conditioning_tensors=12,
            diagnostic_only_not_preservation_or_structure_qualification=True))
        for source_name in ['dataset/asian_faces','dataset/thumbnails128x128']:
            for term in ['/degraded/feature_MSE','/clear/MSE']:
                assert all(np.isfinite(v) and v>0 for v in live_norms[source_name+term].values()), 'A conditioning path has no finite nonzero improvement/preservation gradient; retain stop'
        assert candidate.branch_calls==dict(live=40,anchor=40)
        assert state_hash(candidate) == before and buffer_hash(candidate) == buffers''')
    source=replace_once(source,'            parameters=p[\'parameters\'],learning_names=names,model_state=before,buffer_state=buffers,',
        "            parameters=p['parameters'],learning_names=names,model_state=before,buffer_state=buffers,\n"
        "            feature_branch_calls=dict(candidate.branch_calls),all12_conditioning_tensors_live_for_both_sources=True,\n"
        "            same_input_frozen_feature_reference=True,new_architecture_untrained=True,")
    source=replace_once(source,'    digest = sha(archive)',
        "    assert time.monotonic()-stamp < p['export_seconds'], 'Finite post-archive export stop'\n    digest = sha(archive)\n    assert time.monotonic()-stamp < p['export_seconds'], 'Finite post-hash export stop'")
    source=replace_once(source,"    assert direction['direction_file_sha256'] == sha(root/'directions.npz')",
        "    assert direction['direction_file_sha256'] == sha(root/'directions.npz')\n"
        "    assert p['illustrative_matrix_fixture_not_neural_training_or_quality_evidence']")
    ast.parse(source,feature_version=(3,10))
    (packet/'scripts/cctv_dgp_live_conditioning_v1_vm.py').write_text(source,encoding='utf-8',newline='\n')
    for name in ['directions.npz','direction_review.json']: shutil.copyfile(old_root/name,packet/name)
    shutil.copyfile(old_root/'cctv_dgp_bank_group_gradients_v1_r1_contract.py',packet/'cctv_dgp_live_conditioning_v1_contract.py')
    shell=(old_root/'scripts/run_groups.sh').read_text(encoding='utf-8').replace('cctv_dgp_bank_group_gradients_v1_r1','cctv_dgp_live_conditioning_v1')
    (packet/'scripts/run_conditioning.sh').write_text(shell,encoding='utf-8',newline='\n')
    p=dict(old)
    p.update(name=NAME,UTC=datetime.now(timezone.utc).isoformat(),
        worker_lineage_sha256=sha(old_root/'scripts/cctv_dgp_bank_group_gradients_v1_r1_vm.py'),
        preceding_return_review_sha256=sha(ANALYSIS/'review_binding_audit.json'),
        human_design_direction='apply best approach after three finite comparison failures',
        new_architecture_diagnostic_only=True,training_launch_ready=False,
        architecture='Nonzero readouts minus frozen same-input initial readouts; live style/spatial conditioning receives derivatives while initial output must match original DGP exactly',
        architecture_not_a_claim_of_permanent_fix_or_better_images=True,
        earlier_B50_already_changed_all37_tensors_and_remains_failed=True,
        nonzero_readout_initializer='Declared deterministic signed/permuted Hadamard channel rows; no pretrained spatial mapping',
        nonzero_readout_row_L2=.125,readout_matrix_fixtures=stats,
        illustrative_matrix_fixture_not_neural_training_or_quality_evidence=True,
        conditioning_tensors_required=12,conditioning_gradient_requirement_sources=['dataset/asian_faces','dataset/thumbnails128x128'],
        conditioning_gradient_requirement_groups=['degraded/feature_MSE','clear/MSE'],
        frozen_same_input_reference_no_clean_target_usage=True,
        explicit_geometry_warp_or_display_processing=False,
        initial_model_state=expected_state,parent_initial_B_state=old['initial_model_state'],
        learning_names=['live.'+name for name in old['learning_names']],
        selected_parameter_shapes={f'live.{name}':list(initial[name].shape) for name in old['learning_names']},
        architecture_initializer_is_not_optimizer_update_or_trained_checkpoint=True,
        feature_branch_passes_cap=80,strict_initial_raw_controls=100,
        previous_directions_are_dot_comparators_only_no_displacements=True,
        no_unchanged_recipe_repeat=True,no_local_model_construction_forwards_gradients_or_updates=True)
    p['assets_sha256']={f.relative_to(packet).as_posix():sha(f) for f in packet.rglob('*') if f.is_file()}
    assert len(p['assets_sha256'])==7
    write(packet/'protocol.json',p);pin=sha(packet/'protocol.json')
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in sorted(f for f in packet.rglob('*') if f.is_file()):
            tar.add(file,arcname=NAME+'/'+file.relative_to(packet).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream: stream.write(digest+'  '+archive.name+'\n')
    importer=(ROOT/'scripts/import_cctv_dgp_bank_group_gradients_v1_r1_return.py').read_text(encoding='utf-8')
    importer=importer.replace('cctv_dgp_bank_group_gradients_v1_r1','cctv_dgp_live_conditioning_v1')
    importer=importer.replace('group_gradients_v1_r1','live_conditioning_v1').replace('group-gradients-v1-r1','live-conditioning-v1')
    importer=importer.replace(old_audit['protocol_sha256'],pin)
    with (ROOT/'scripts/import_cctv_dgp_live_conditioning_v1_return.py').open('x',encoding='utf-8',newline='\n') as stream:stream.write(importer)
    audit=(ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_r1_return.py').read_text(encoding='utf-8')
    audit=audit.replace('cctv_dgp_bank_group_gradients_v1_r1','cctv_dgp_live_conditioning_v1').replace('group_gradients_v1_r1','live_conditioning_v1')
    audit=replace_once(audit,"            assert np.allclose(flat,expected,rtol=1e-4,atol=1e-8), 'Raw clear-MSE derivative differs from retained normalized derivative'",'''            keep=np.concatenate([np.full(int(np.prod(shape_records[i])),not p['learning_names'][i].startswith(('live.style_conditioner.','live.spatial_conditioners.')),bool) for i in range(37)])
            assert np.allclose(flat[keep],expected[keep],rtol=1e-4,atol=1e-8), 'Retained decoder/readout clear-MSE derivative differs'
            assert np.count_nonzero(flat[~keep])>0, 'New conditioning derivative missing'
            delta=float(np.abs(flat.astype(np.float64)[keep]-expected[keep]).max())''')
    audit=audit.replace('Raw clear-MSE derivative','Retained nonconditioning clear-MSE derivative')
    audit=replace_once(audit,'    assert all(sha(ROOT/n)==h for n,h in protected.items())\n    write(ANALYSIS/\'independent_audit.json\'', '''    live_norms=read(out/'conditioning_gradient_norms.json')
    assert live_norms['conditioning_tensors']==12 and run['feature_branch_calls']==dict(live=40,anchor=40)
    assert run['all12_conditioning_tensors_live_for_both_sources'] and run['same_input_frozen_feature_reference'] and run['new_architecture_untrained']
    edges=np.cumsum([0]+[int(np.prod(s)) for s in shape_records])
    for record in run['groups']:
        vector=np.load(out/record['file'],allow_pickle=False)
        for i,name in enumerate(p['learning_names']):
            if name.startswith(('live.style_conditioner.','live.spatial_conditioners.')):
                value=float(np.linalg.norm(vector[edges[i]:edges[i+1]]))
                assert np.isclose(value,live_norms['groups'][record['key']][name],rtol=1e-12,atol=1e-15)
    for source in p['conditioning_gradient_requirement_sources']:
        for group in p['conditioning_gradient_requirement_groups']:
            assert all(np.isfinite(v) and v>0 for v in live_norms['groups'][source+'/'+group].values())
    for sample in run['samples']:
        with np.load(out/sample['file'],allow_pickle=False) as a:
            for i,name in enumerate(p['learning_names']):
                if name.startswith(('live.style_conditioner.','live.spatial_conditioners.')):
                    assert np.isclose(float(np.linalg.norm(a[f'p{i:02d}'].astype(np.float64))),sample['conditioning_tensor_gradient_norms'][name],rtol=1e-12,atol=1e-15)
    for row in run['rows']:
        for q in row['queries']:
            values=q['conditioning_tensor_gradient_norms']
            assert len(values)==12 and all(np.isfinite(v) and v>=0 for v in values.values())
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(ANALYSIS/'independent_audit.json' ''')
    # The insertion above preserves the original call's comma; normalize the
    # deliberate trailing whitespace to keep the AST/readable call unchanged.
    audit=audit.replace("write(ANALYSIS/'independent_audit.json' ,dict", "write(ANALYSIS/'independent_audit.json',dict")
    audit=replace_once(audit,'        two_clear_MSE_sample_derivatives_compared_to_prior_measured_normalized_gradients=True,',
        '        two_clear_MSE_nonconditioning_sample_derivatives_compared_to_prior_measured_normalized_gradients=True,\n'
        '        all12_conditioning_aggregate_norms_for_both_sources_independently_checked=True,\n'
        '        frozen_same_input_reference_and_new_untrained_architecture=True,\n'
        '        gradient_activation_is_not_structure_PNG_preservation_or_training_qualification=True,')
    ast.parse(audit,feature_version=(3,10))
    with (ROOT/'scripts/audit_cctv_dgp_live_conditioning_v1_return.py').open('x',encoding='utf-8',newline='\n') as stream:stream.write(audit)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
        protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        changed_architecture=True,unrun=True,training_launch_ready=False,
        initial_architecture_state_reconstructed_from_saved_arrays_not_a_model=True,
        virtual_initial_state_sha256=expected_state,selected_parameters=3364643,readout_fixtures=stats,
        parent_hash_bindings_checked=len(p['parent_required_sha256']),
        no_local_neural_or_gradient_or_optimizer_calls=True,no_VM_connection_upload_or_launch=True,
        future_importer_sha256=sha(ROOT/'scripts/import_cctv_dgp_live_conditioning_v1_return.py'),
        future_auditor_sha256=sha(ROOT/'scripts/audit_cctv_dgp_live_conditioning_v1_return.py'),
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,protocol_sha256=pin,archive_sha256=digest,bytes=archive.stat().st_size,
        selected_parameters=3364643,optimizer_updates=0,unrun=True))


if __name__=='__main__':main()
