"""Read existing source, tensors and target bytes without constructing a model."""
import ast
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_frozen_reconstruction_path_review_v1'
RETURN=ROOT/'outputs/cctv_dgp_nearby_gradient_learning_v1_r1_return'
PARENT=ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
ANALYSIS=ROOT/'outputs/cctv_dgp_nearby_gradient_learning_v1_r1_analysis'


def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))


def sha(f):
    h=hashlib.sha256()
    with Path(f).open('rb') as s:
        for b in iter(lambda:s.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    assert not OUT.exists(), 'Preserve the completed review; no automatic repeat'
    p=read(RETURN/'protocol.json');parent=read(PARENT/'protocol.json')
    env=read(RETURN/'outputs/nearby_gradient_learning_v1/environment.json')
    assert sha(RETURN/'protocol.json')=='913b2d201f0567328840f179418ad4f4c7649bf41d5820c8d060bb7ba5e9ad24'
    assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256']
    assert parent['reference_auxiliary_native_paths_ancestral_metadata_only']
    assert env['conditioning_names']==p['conditioning_names'] and env['parameters']==p['conditioning_parameters']==2580864
    assert p['current_decoder_readouts_original_prior_recognizer_buffers_frozen'] and env['full_epochs']==0
    assert len(p['conditioning_names'])==12 and len(p['all37_learning_names'])==37
    assert all(n.startswith(('live.style_conditioner.','live.spatial_conditioners.')) for n in p['conditioning_names'])
    model=PARENT/'cctv_dgp_bank_comparison_v1_model.py'
    worker=RETURN/'scripts/cctv_dgp_nearby_gradient_learning_v1_r1_vm.py'
    anchor=RETURN/'conditioning_model.py'
    manifest=read(RETURN/'export_manifest.json')['files']
    assert sha(model)==p['parent_required_sha256'][model.relative_to(PARENT).as_posix()]
    for f in [worker,anchor]:assert sha(f)==p['assets_sha256'][f.relative_to(RETURN).as_posix()]==manifest[f.relative_to(RETURN).as_posix()]
    files=[RETURN/'protocol.json',PARENT/'protocol.json',model,worker,anchor,PARENT/'initializers/initial_B.pth',
        RETURN/'readouts.npz',RETURN/'outputs/nearby_gradient_learning_v1/environment.json',
        RETURN/'outputs/nearby_gradient_learning_v1/failure_restoration.json',
        RETURN/'outputs/nearby_gradient_learning_v1/initial_conditioning.npy',ANALYSIS/'failure_audit.json',
        ANALYSIS/'finite_response_review.json',ANALYSIS/'visual_review.json',ANALYSIS/'diagnostic_question.json',
        ROOT/'outputs/cctv_dgp_method_comparison_v2/selection.json',ROOT/'CCTV_DGP_BANK_PATH_ABLATION_V1_RESULTS.md',
        ROOT/'CCTV_DGP_POST_V42_ARCHITECTURE_REVIEW.md',ROOT/'CCTV_DGP_ACTIVE_ORIGINAL_DECODER_V28_RESULTS.md',
        ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth',Path(__file__),
        ROOT/'outputs/cctv_dgp_frozen_reconstruction_path_review_v1_attempt1.json']
    bindings={f.relative_to(ROOT).as_posix():sha(f) for f in files}
    assert sha(PARENT/'initializers/initial_B.pth')==p['parent_required_sha256']['initializers/initial_B.pth']
    assert sha(RETURN/'readouts.npz')==p['source_origins']['readouts.npz']['source_sha256']==manifest['readouts.npz']
    state=torch.load(PARENT/'initializers/initial_B.pth',map_location='cpu',weights_only=True)
    assert isinstance(state,dict) and all(isinstance(v,torch.Tensor) for v in state.values())
    tensors=[]
    for name in p['all37_learning_names']:
        value=state[name[len('live.'):]]
        tensors.append(dict(name=name,shape=list(value.shape),elements=value.numel(),
            selected_in_failed_run=name in p['conditioning_names'],section='conditioning' if name in p['conditioning_names']
            else 'readout' if name.startswith('live.fusions.') else 'reconstruction'))
    counts={section:dict(tensors=sum(r['section']==section for r in tensors),elements=sum(r['elements'] for r in tensors if r['section']==section))
            for section in ['conditioning','readout','reconstruction']}
    assert counts['reconstruction']==dict(tensors=15,elements=609219)
    assert counts['conditioning']==dict(tensors=12,elements=2580864) and counts['readout']['tensors']==10
    initial=np.load(RETURN/'outputs/nearby_gradient_learning_v1/initial_conditioning.npy',allow_pickle=False)
    expected=np.concatenate([state[n[len('live.'):]].numpy().reshape(-1) for n in p['conditioning_names']])
    np.testing.assert_array_equal(initial,expected)
    del expected,initial,state
    projections=[]
    with np.load(RETURN/'readouts.npz',allow_pickle=False) as z:
        assert set(z.files)=={f'f{size}_{term}' for size in [16,32,64,128,256] for term in ['weight','bias']}
        for size in [16,32,64,128,256]:
            w,b=z[f'f{size}_weight'],z[f'f{size}_bias']
            assert w.dtype==b.dtype==np.float32 and w.ndim==4 and w.shape[2:]==(1,1) and b.shape==(w.shape[0],)
            assert np.isfinite(w).all() and np.isfinite(b).all() and not b.any() and np.count_nonzero(w)>0
            rank=int(np.linalg.matrix_rank(w[:,:,0,0].astype(np.float64)))
            assert rank==w.shape[0]
            projections.append(dict(feature_scale=size,shape=list(w.shape),rank=rank,bias_zero=True,
                nonzero_coefficients=int(np.count_nonzero(w)),full_row_rank_not_learnability_proof=True))
    refs={r['id']:r for r in parent['references']};selected=[]
    for block in p['reference_blocks']:
        r=refs[block['reference']];assert r['role']=='train'
        f=PARENT/r['target'];digest=sha(f)
        assert digest==p['parent_required_sha256'][r['target']]
        bindings[f.relative_to(ROOT).as_posix()]=digest
        with Image.open(f) as im:
            assert im.size==(256,256);target=np.asarray(im.convert('RGB')).copy()
        rgb_hash=hashlib.sha256(target.tobytes()).hexdigest()
        assert rgb_hash==parent['canonical_target_RGB_sha256'][r['id']]
        source=ROOT/r['source_file'];native=None;control=None
        if source.is_file():
            assert sha(source)==r['source_sha256']
            bindings[source.relative_to(ROOT).as_posix()]=sha(source)
            with Image.open(source) as im:
                assert list(im.size)==r['native_size'];native=np.asarray(im.convert('RGB')).copy()
            if r['source']=='dataset/thumbnails128x128':
                assert native.shape==(128,128,3)
                reduced=np.asarray(Image.fromarray(native).resize((256,256),Image.Resampling.BILINEAR))
                reduced_hash=hashlib.sha256(reduced.tobytes()).hexdigest()
                assert reduced_hash==r['target_rgb_sha256'] and reduced_hash!=rgb_hash
                control=dict(method='Pillow bilinear 128-to-256 from bound original thumbnail',
                    RGB_sha256=reduced_hash,exactly_matches_ancestral_reduced_target_RGB=True,
                    target_equals_reduced_pixels=False,component_pixels_different=int(np.count_nonzero(target!=reduced)),
                    raw_RGB_MSE=float(np.square((target.astype(np.float64)-reduced.astype(np.float64))/255).mean()))
        selected.append(dict(id=r['id'],source=r['source'],native_size_metadata=r['native_size'],
            native_file_checked=source.relative_to(ROOT).as_posix() if native is not None else None,
            actual_target_size=[256,256],target_RGB_hash_verified_against_canonical_map=True,
            actual_target_RGB_sha256=rgb_hash,ancestral_reduced_target_RGB_sha256=r['target_rgb_sha256'],
            target_file=f.relative_to(ROOT).as_posix(),target_origin_kind=r.get('evaluation_source_kind','prepared_original_source'),
            higher_resolution_source_sha256=r.get('hq_source_sha256'),higher_resolution_source_file_not_reaudited_here=bool(r.get('hq_source_sha256')),
            reduced_target_comparison=control,auxiliary_native_and_reduced_paths_in_parent_are_metadata_not_required_files=True))
    assert len(selected)==20 and Counter(r['source'] for r in selected)=={'dataset/asian_faces':10,'dataset/thumbnails128x128':10}
    assert all(r['native_file_checked'] for r in selected)
    assert sum(r['reduced_target_comparison'] is not None for r in selected)==10
    model_text=model.read_text(encoding='utf-8');ast.parse(model_text,feature_version=(3,10))
    for s in ["extras['64']","extras['32']","extras['16']","extras['128']","extras['256']",'torch.tanh(self.net.final(y))','.clamp(-1, 1)']:
        assert s in model_text,s
    assert 'param.requires_grad_(name in names)' in worker.read_text(encoding='utf-8')
    assert 'self.live.fusions[key](features[key])-reference[key]' in anchor.read_text(encoding='utf-8')
    assert 'return self.live.current.reconstruct(image,maps,extras)' in anchor.read_text(encoding='utf-8')
    selection=read(ROOT/'outputs/cctv_dgp_method_comparison_v2/selection.json')['current_model']
    assert selection['checkpoint_sha256']==p['original_checkpoint_sha256']==sha(ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth')
    assert selection['selected_extra_epochs']==2 and selection['selected_extra_updates']==226 and not selection['lifetime_epochs_confirmed']
    failure=read(ANALYSIS/'failure_audit.json')
    assert failure['accepted_trials']==[] and failure['trials_replayed']==3 and not failure['completed_pilot']
    restore=read(RETURN/'outputs/nearby_gradient_learning_v1/failure_restoration.json')
    assert restore['complete'] and restore['fresh_state']==restore['expected_initial']==p['initial_model_state']
    question=read(ANALYSIS/'diagnostic_question.json');assert question['human_answer_pending'] and question['human_answer'] is None
    assert all(sha(ROOT/n)==h for n,h in bindings.items())
    OUT.mkdir()
    proof=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),source_sha256=bindings,
        checker_sha256=sha(__file__),read_only_serialized_tensor_inspection_not_neural_construction=True,
        selected_parameter_partitions=counts,all37_initial_learning_tensor_shapes=tensors,
        all12_initial_conditioning_values_match_returned_initial_state=True,frozen_nonzero_readout_projections=projections,
        initial_anchor_cancellation_previously_audited_not_rerun=True,frozen_observer_decoder_readouts_prior_and_fixed_recognizer=True,
        reconstructed_RGB_path='bank conditioning -> fixed multiscale readouts -> fixed repaired original decoder -> tanh + observed input -> clamp -> RAW -> original PNG delivery',
        direct_generator_RGB_or_512_synthesis_tail_not_used=True,actual_bank_features_through256_only=True,
        no_new_full_resolution_decoder_was_trained_in_latest_run=True,selected_TRAIN_references=selected,
        cohort_original_references=781,cohort_original_cases=3905,selected_TRAIN_reference_count=20,selected_TRAIN_case_count=100,
        full_TRAIN_or_final_representative_performance_not_established=True,ten_FFHQ_targets_differ_from_reduced128_targets=True,
        all_twenty_current_target_RGB_hashes_match_authoritative_canonical_map=True,
        all_ten_ancestral_FFHQ_RGB_hashes_match_reconstructed_thumbnail_controls=True,
        no_metadata_or_scientific_hash_rewritten=True,first_review_assumption_failure_preserved=True,
        all_targets_uniformly_low_resolution_claim_rejected=True,target_quality_still_requires_visual_review=True,
        primary_checkpoint_selected_extra_epochs=2,primary_checkpoint_selected_extra_updates=226,primary_lifetime_epochs_not_confirmed=True,
        latest_run_accepted_updates=0,latest_run_full_epochs=0,actual_trials_not_added_to_primary_model_training=True,
        decoder_capability_or_unique_failure_cause_not_proved=True,no_new_metric_or_threshold_selection=True,
        question_pending=True,no_fourth_restorer_recipe_selected_or_prepared=True,
        neural_forwards=0,gradient_queries=0,optimizer_updates=0,parameter_displacements=0,VM_calls=0,
        no_app_model_code_or_mask_change=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False)
    with (OUT/'review.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(proof,f,indent=2,allow_nan=False);f.write('\n')
    print(dict(complete=True,partitions=counts,selected_TRAIN_references=20,HQ_targets_different_from_reduced=10,
               source_bindings=len(bindings),new_neural_or_training_calls=0))


if __name__=='__main__':main()
