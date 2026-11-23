"""Freeze prospective predictor-only CE design from admitted retained evidence.

Metadata/files only. Does not import Torch, construct models or run training.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]

def need(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
 base=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return'
 paths={
 'actual_probe_transport':(base/'transport_audit.json','ee4980dcd22a04f4315d7369a16c6fb2ce8ad8704532f6e1b284d35a45d24fa4'),
 'actual_probe_state_codes':(base/'independent_state_code_audit.json','794fed73804eabfa081805cdd0d3db3dda89d5a1fbd6138df725b6b404ad922d'),
 'actual_probe_pixels':(base/'independent_saved_output_audit.json','4f5042edc19496bd3dee515dc052928c03e4f6d9e12b246efdcca4262a54d02f'),
 'actual_probe_root_visual':(base/'root_visual_review.json','3d2568cfd19614f7d8b98da71b027d471fc32c27753cee017ed512b394931a53'),
 'independent_design_research':(ROOT/'research/completion_codeformer_post_probe_predictor_review_v1/independent_design_research.md','b4bcb18388ccd33a8df58f688ffb86d74d47d757f9a471b5df64787fa46e81a7'),
 'partition':(ROOT/'research/completion_codeformer_post_probe_predictor_review_v1/independent_predictor_partition.json','286b42ab32449568061b5b62cd09368c40c5c5e9f7c77c2cfd4c2c1a67bc87fd'),
 'scoped_design_admission':(ROOT/'research/completion_codeformer_predictor_ce_design_independent_review_v1/independent_scoped_design_admission.json','3f6c67961efd28e4d405d628b6dcab285d839557f73df9b8b9944e74badc2f50'),
 'balanced128':(ROOT/'research/completion_codeformer_predictor_ce_design_independent_review_v1/balanced128_subset.json','8ec2d535cd6fb231af815d0ec833a1cd08f418458cbc4919c0ebd13f638f2a1d'),
 'parent114_Adam_mapping':(ROOT/'research/completion_codeformer_predictor_ce_design_independent_review_v1/predictor_parent_Adam_mapping.json','157bb777bda588ff89786d8d4a5a80bb4252ca1a66c412811dcc8f6039a92465'),
 'resource_arithmetic':(ROOT/'research/completion_codeformer_post_probe_resource_review_v1/serialization_resource_admission.json','00773a66b805c66b0d9daebc0fda7eb6108704ffd3f3d6e43003715403835e59'),
 }
 evidence={}
 for key,(path,digest) in paths.items():need(sha(path)==digest,'Evidence drift '+key);evidence[key]=pin(path)
 prospective=ROOT/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_DESIGN.md'
 evidence['readable_design']=pin(prospective)
 oldp=read(ROOT/'research/completion_codeformer_delivered_objective_v4/protocol.json')
 need(sha(ROOT/'research/completion_codeformer_delivered_objective_v4/protocol.json')=='8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d','Retained prior protocol drift')
 evidence['inherited_fixed_data_protocol']=pin(ROOT/'research/completion_codeformer_delivered_objective_v4/protocol.json')
 subset=read(paths['balanced128'][0]);cases=subset['cases_in_reference_role_order']
 need(len(cases)==128 and len({r['id'] for r in cases})==128,'Exact128 observations')
 refs={r['reference_id'] for r in cases};monitorrefs={r['reference_id'] for r in oldp['monitor_cases']}
 need(len(refs)==32 and len(monitorrefs)==8 and refs.isdisjoint(monitorrefs),'FIT/monitor reference separation')
 need(len({r['teacher_key'] for r in cases})==64,'Exact64FITlabel sets')
 gray=[r['id'] for r in oldp['challenge_cases'] if r['style']=='GRAY']
 need(len(gray)==8,'Exact8gray challenges')
 design={
 'format':'completion-codeformer-predictor-only-CE-finite-design-v1',
 'frozen_UTC':datetime.now(timezone.utc).isoformat(),
 'status':'FROZEN_PROSPECTIVE_DESIGN_RUNTIME_PREPARATION_NOT_TRANSFER_READY',
 'question':'Can ordinary covered-input code prediction improve delivered completion while exact R1 encoder/fusion and original prior remain fixed?',
 'evidence':evidence,
 'scope':'Paired adaptive TRAIN plus fixed unpaired public photographs. No independent FINAL/native CCTV clean truth, hidden identity or oracle deployment claim.',
 'state':{'original_checkpoint_sha256':'b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797','R1_parent_sha256':'0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd','R1_logical_sha256':'55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6','trainable114_scalars':19450880,'trainable114_names_from_partition':True,'trainable_allowlist':['position_emb','feat_emb.','ft_layers.','idx_pred_layer.'],'frozen383':{'encoder':164,'fusion':54,'original_prior':165},'all_modules_eval':True,'requires_grad_only114_during_FIT':True,'ordinary_inference_w':1,'adain':False,'oracle_override':False,'unchanged_app_input_and_delivery':True},
 'optimizer':{'objects':1,'algorithm':'Adam','active_tensors':114,'lr':1e-5,'betas':[.9,.99],'eps':1e-8,'weight_decay':0.,'clip_global_L2':1.,'name_mapped_parent_step':32,'maximum_final_step':64,'reset':False,'blind_positional278_to114_load':False,'actual_gradient_and_update_diagnostics':True,'every_intended_gradient_present_finite':True,'all_frozen_gradients_none':True,'zero_gradient_values_retained_honestly':True,'restore_exact_parent_RNG_after_initial_readouts':True},
 'objective':{'identifier':'predictor_only_CE_all256_same_inpainting_codebook_v1','CE_reduction':'mean_over_all256_positions','microbatches':4,'microbatch_divisor_once':4,'cached_same_original_inpainting_codes':True,'teacher_neural_calls':0,'latent_MSE':False,'pixel_loss':False,'fusion_loss':False,'GAN':False,'STE':False,'unavailable_clean_indices_at_evaluation':False},
 'data':{'full_retained_FIT_pool_cases':256,'selected_optimization_observations':128,'FIT_reference_ids':sorted(refs),'FIT_label_sets':64,'monitor_reference_ids':sorted(monitorrefs),'monitor_label_sets':16,'monitor_never_optimized':True,'all80_label_arrays_retained':True,'new_targets':0,'per_reference_observations':4,'per_reference_roles':{'anchor':2,'gray_augmentation':1,'RGB_augmentation':1},'total_anchors':64,'total_augmentations':64,'GRAY':32,'RGB':96,'each_augmentation_condition':16,'each_mask_geometry':32,'each_cycle_distinct_references':4,'subset_and_schedule_pin':evidence['balanced128'],'subset_selection_uses_quality_scores':False,'this_is_not_a_full256_epoch':True,'true_person_identity_or_pretraining_disjointness_established':False},
 'execution':{'maximum_optimizer_steps':32,'maximum_backwards':128,'microbatches_per_step':4,'paired_snapshots':[0,8,16,32],'paired_cases_each':48,'public_snapshots':[0,32],'public_cases_each':32,'saved_evaluation_outputs':256,'saved_evaluation_empty_bypasses':8,'nonempty_evaluation_full_calls':248,'fresh_parity_pairs':4,'fresh_parity_full_calls':8,'full_generation_calls':256,'code_only_calls':128,'all_model_invocations_planned':384,'all_model_invocations_hard_cap':420,'typed_max':{'parity':8,'fit_code':128,'monitor':128,'challenge':64,'public_development':56},'teacher_calls':0,'additional_empty_control_bypasses':160,'all_empty_bypasses':168,'historical_R1_matches_before_any_backward':48,'fresh_parity_before_any_backward':True,'pre_dispatch_guards':True,'snapshots_do_not_add_FIT_calls':True},
 'quality':{'acceptance_endpoint_only':32,'intermediate8_16_purpose':'safety and diagnostics only; never selectable winners','minimum_clear_PNG_hole_gain_fraction':.01,'minimum_challenge_PNG_hole_gain_fraction':.01,'all_existing_RAW_PNG_hole_boundary_macro_geometry_and_challenge_condition_nonworse':True,'nonworse_tolerance':1e-12,'catastrophic_stop_fraction':.10,'catastrophe_cohorts':['clear','challenge'],'outside_source_pixels_float_and_PNG_exact':True,'empty_exact':True,'GRAY_style_additional_gate':{'case_ids':gray,'RAW_measure':'mean_over_generated_support(max_RGB-min_RGB) on delivered float256','RAW_units':[0,1],'PNG_measure':'mean_over_generated_support(max_RGB-min_RGB) on downloaded uint8 PNG256','PNG_units':[0,255],'nonworse_groups':['macro','lower_face','eye_band','image_left_lens','cheek_object'],'cases_per_geometry':2,'RAW_tolerance':1e-12,'PNG_tolerance':1e-12,'required_only_final32':True,'all8_individual_values_and_visual_review':True,'no_recoloring':True,'limited_small_sample_regression_guard':True},'all_planned_outputs_and_sheets_reviewed':True,'user_and_independent_FINAL_acceptance_still_required':True,'lower_CE_or_higher_grid_agreement_not_qualification':True,'oracle_gain_not_achievable_promise':True,'failed_final_cannot_select_earlier_snapshot':True},
 'resources':{'worker_hard_seconds':1800,'cooperative_neural_seconds':1680,'terminal_save_reserve_seconds':120,'export_cooperative_seconds':900,'initial_free_bytes':8*1024**3,'installer_extra_free':'exact_uncompressed_install_payload','stop_free_bytes':2*1024**3,'CUDA_allocated_bytes':20*1024**3,'terminal_bytes':1024**3,'whole_output_bytes':2560*1024**2,'teacher_bytes':24*1024**2,'metadata_including_log_bytes':8*1024**2,'log_bytes':4*1024**2,'PNG_bytes_each':256*1024,'raw_tar_bytes':2816*1024**2,'gzip_bytes':2816*1024**2,'max_regular_return_members':1000,'max_return_path_UTF8_bytes':240,'generated_NPZ_max':388,'retained_teacher_NPZ':80,'NPZ_arrays_max':10,'NPY_header_bytes_max':512,'checkpoint_ZIP_members_max':1615,'checkpoint_pickle_bytes_max':1024**2,'exact_single_L4_RNG_schema_bound':True,'new_unsealed_resource_revision_not_historical_gate_change':True},
 'evidence_output':{'evaluation256_npz':['input256','mask256','result256','raw512_or_absent_empty','predicted_codes_or_empty'],'fresh_parity4_npz':'all10 pinned same-codebook parity schema arrays','FIT128_npz':['logits_FP32_1x256x512','target_codes_int64_1x256','case_index','cycle','microbatch'],'FIT_CE_replay_possible':True,'all_per_case_metrics_preserved':True,'no_false_predictions_for_empty_masks':True,'NPZ_count_includes_copied_teachers':468,'all_generation_public_cases_unpaired':True},
 'terminal':{'save_attempts_max':1,'actual_nonprior_model_tensors':332,'mixed_historical_code_Adam_slots':278,'updated_predictor_slots':114,'unchanged_encoder_slots':164,'unchanged_fusion_slots':54,'duplicate_active114_moment_payload':False,'actual_failed_prior165_if_mutated':True,'all497_actual_byte_hashes_and_finite_flags_before_gate':True,'partial114_gradients_retained':True,'RNG_sampler_progress_counts_retained':True,'successful_partial_alias_not_archived_twice':True,'hardkill_or_IO_can_prevent_recovery_explicit':True,'automatic_resume':False},
 'admission_pending':['independent_final_formal_design_review','runtime_ownership_gradient_mapping_and_loss_tests','runtime_byte_schema_and_terminal_failure_enforcement','source_data_protocol_and_actual_guarded_transfer_CLI','safe_installer_exporter_namespace_and_resource_fixtures','closed_execution_archive_and_manual_guide'],'ready_to_upload':False,'new_model_created':False,'model_qualified':False,'automatic_launch':False,'RESTORATION_IMPLEMENTATION_changed':False,'goal_complete':False}
 need(sum(design['execution']['typed_max'].values())==384,'Typed count mismatch')
 need(192+64==design['execution']['saved_evaluation_outputs'] and 248+8+128==384,'Evaluation count mismatch')
 target=ROOT/'research/completion_codeformer_predictor_ce_design_v1';need(not target.exists(),'Preserve frozen design');target.mkdir()
 out=target/'root_design.json'
 with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(design,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(pin(out)))

if __name__=='__main__':main()
