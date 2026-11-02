"""Pure metadata/arithmetic prospective serialization review; no models or training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
MiB=1024**2
GiB=1024**3

def need(value,label):
    if not value:raise ValueError(label)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def pin(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def compress_bound(n):return n+(n>>12)+(n>>14)+(n>>25)+13

def npz_bound(payload_bytes,*,arrays=10):
    # Split largest input across actual dtype/shape payloads; each NPY header <=512B.
    need(len(payload_bytes)<=arrays,'Too many arrays')
    lengths=list(payload_bytes)+[0]*(arrays-len(payload_bytes))
    per_array=[{'data_bytes':n,'NPY_header_bound':512,'deflate_bound':compress_bound(n+512),'ZIP_local_central_name_alignment_bound':512} for n in lengths]
    return {'arrays_max':arrays,'tensor_payload_bytes':sum(payload_bytes),'per_array':per_array,'ZIP_global_closure_bound':1024,'NPZ_bytes_bound':sum(r['deflate_bound']+512 for r in per_array)+1024}

def main():
    part_path=ROOT/'research/completion_codeformer_post_probe_predictor_review_v1/independent_predictor_partition.json'
    parent_path=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/independent_state_audit.json'
    prior_save=ROOT/'outputs/completion_codeformer_fusion_only_v3_return/independent_state_audit.json'
    teacher_path=ROOT/'outputs/completion_codeformer_coverage_v2_return/cctv_completion_codeformer_coverage_v2_results/outputs/teacher_labels/index.json'
    need(sha(part_path)=='286b42ab32449568061b5b62cd09368c40c5c5e9f7c77c2cfd4c2c1a67bc87fd','Frozen predictor partition differs')
    need(sha(parent_path)=='e920c3ddd9b8a85b50fd6f324bde8760c13c6df805f6be61d1e4393472d59b06','Parent audit differs')
    need(sha(prior_save)=='52d2c3040b050bd544231ba0ea1ed577444770f6b88655fc3ea081355909a365','Historical serialization audit differs')
    need(sha(teacher_path)=='0567a9ad54e6397390341a8dfc1ffa8ec50ceb9159da299605fee021ec678a69','Teacher index differs')
    part=read(part_path);parent=read(parent_path);historical=read(prior_save);teachers=read(teacher_path)
    groups={}
    for name,rows in part['full_groups'].items():
        need(all(r['dtype']=='torch.float32' and math.prod(r['shape'])==r['scalars'] for r in rows),'FP32 exact shape metadata required')
        groups[name]={'tensors':len(rows),'scalars':sum(r['scalars'] for r in rows),'FP32_bytes':sum(r['scalars']*4 for r in rows)}
    need(groups['predictor']=={'tensors':114,'scalars':19450880,'FP32_bytes':77803520},'Predictor partition changed')
    delta=groups['encoder']['FP32_bytes']+groups['predictor']['FP32_bytes']+groups['fusion']['FP32_bytes']
    need(delta==parent['trainable']['FP32_bytes']==233839360,'Full retained332delta metadata differs')
    code_bytes=groups['encoder']['FP32_bytes']+groups['predictor']['FP32_bytes']
    adam_moments=2*(code_bytes+groups['fusion']['FP32_bytes'])
    adam_steps=(278+54)*4
    rng_bytes=sum(r['bytes'] for r in parent['RNG']['tensors'])
    base_tensors=delta+adam_moments+adam_steps+rng_bytes
    gradients=groups['predictor']['FP32_bytes']
    failed_prior_bytes=groups['prior']['FP32_bytes']
    terminal_tensor=base_tensors+gradients+failed_prior_bytes
    terminal_storages=332+3*(278+54)+2+114+165
    need(terminal_storages==1609 and terminal_tensor==916100108,'Exact prospective terminal tensor arithmetic differs')
    terminal_ZIP_members=terminal_storages+6
    terminal_container=terminal_tensor+terminal_ZIP_members*1024+MiB+1024
    terminal_cap=1024*MiB
    need(terminal_container<terminal_cap,'1GiB terminal cap infeasible')
    terminal={'full_R1_delta332_FP32_bytes':delta,'full_named_code278_plus_fusion54_Adam_moments_bytes':adam_moments,'Adam_step_scalar_FP32_bytes':adam_steps,'CPU_CUDA_RNG_bytes':rng_bytes,'partial_predictor114_gradients_FP32_bytes_max':gradients,'failed_actual_prior165_FP32_bytes_max':failed_prior_bytes,'tensor_payload_bytes_max':terminal_tensor,'distinct_storage_records_max':terminal_storages,'ZIP_member_count_max':terminal_ZIP_members,'ZIP_per_member_header_name_descriptor_alignment_bound':1024,'primitive_pickle_metadata_cap':MiB,'global_ZIP_closure_bytes_bound':1024,'serialized_bytes_upper_bound':terminal_container,'terminal_category_cap':terminal_cap,'headroom_bytes':terminal_cap-terminal_container,'one_save_attempt_only':True,'one_combined_Adam_slot_representation':True,'full_parent_Adam_plus_duplicate_predictor_Adam_forbidden':True,'duplicate_prior165_or_full497_state_forbidden':True,'failure_prior_retention_policy':'Normal state keeps332actualdelta and immutable original165base reference. If frozen-prior invariance fails, retain one optional failed_actual_prior_state of165actualFP32tensors in the same bounded state; all497actual tensors then recoverable as evidence. Preserve actualfiniteflags and reject qualification; do not substitute clean originalprior when reporting a mutation.','success_commit_policy':'Rename the one owned.partial into terminal path; no second physical checkpoint. Failure retains the one partial. Never retry serialization into a new file after a failed save.','semantics':'114 predictor Adam slots map by exact parameter name and step32to64; unused164encoder/54fusion slots remain exact at step32 as inert historical dictionaries, no optimizer objects. Save full RNG/sampler/progress and actual partial gradients; no automatic resume.'}
    # Reserve a full-sized code slot and two scalar flags for empty cases, although actual empty codes/raw must be absent with explicit flags.
    full=npz_bound([3*256*256*4,256*256*4,3*256*256*4,3*512*512*4,256*8,1,1])
    empty=npz_bound([3*256*256*4,256*256*4,3*256*256*4,256*8,1,1])
    parity=npz_bound([3*256*256*4,256*256*4,*([3*512*512*4]*4),256*8,256*8,3*256*256*4,3*256*256*4])
    fit=npz_bound([256*512*4,256*8,8,8,8])
    kinds={'evaluation_nonempty':{'count':248,**full},'evaluation_empty_bypass':{'count':8,**empty},'fresh_parity':{'count':4,**parity},'FIT_preupdate_logits_labels':{'count':128,**fit}}
    generated_npz=sum(v['count']*v['NPZ_bytes_bound'] for v in kinds.values())
    teacher_actual=sum(r['bytes'] for r in teachers['labels']);need(len(teachers['labels'])==80 and teacher_actual<24*MiB,'Teacher metadata allowance infeasible')
    limits={'whole_output_bytes':5*GiB//2,'terminal_bytes':terminal_cap,'teacher_bytes':24*MiB,'metadata_log_protocol_supervisor_bytes_combined':8*MiB,'trainer_log_bytes_included_in_metadata':4*MiB,'PNG_bytes_each':256*1024,'PNG_count_max':256,'export_raw_tar_bytes':11*GiB//4,'export_gzip_bytes':11*GiB//4,'export_manifest_bytes':2*MiB,'return_regular_members_including_manifest_max':2048,'relative_path_UTF8_bytes_max':240,'NPZ_generated_count_max':388,'NPZ_copied_teacher_count_max':80,'NPZ_total_max':468,'NPZ_arrays_each_max':10,'NPY_header_bytes_each_max':512,'ZIP_entry_name_bytes_each_max':80,'NPZ_ZIP_extra_comment_bytes':'No unbounded custom fields; total per-entry ZIPoverhead <=512B, global <=1024B','checkpoint_primitive_pickle_bytes_max':MiB}
    output_parts={'all388_generated_NPZ_bound':generated_npz,'one_terminal_or_failed_partial_cap':terminal_cap,'all80_teacher_NPZ_category_cap':limits['teacher_bytes'],'all256_PNG_cap':256*limits['PNG_bytes_each'],'combined_metadata_log_protocol_supervisor_cap':limits['metadata_log_protocol_supervisor_bytes_combined']}
    output_bound=sum(output_parts.values());need(output_bound<limits['whole_output_bytes'],'Whole output cap infeasible')
    tar_bound=output_bound+limits['return_regular_members_including_manifest_max']*(511+2048)+limits['export_manifest_bytes']+10240
    gzip_bound=compress_bound(tar_bound)+1024
    need(tar_bound<limits['export_raw_tar_bytes'] and gzip_bound<limits['export_gzip_bytes'],'Export caps infeasible')
    disk_new_cap=limits['whole_output_bytes']+limits['export_gzip_bytes']
    counts={'steps':32,'FIT_microbatches_and_code_only_calls':128,'FIT_backwards':128,'paired_cases_per_snapshot':48,'paired_snapshots':[0,8,16,32],'paired_outputs':192,'public_outputs':64,'public_snapshots':[0,32],'saved_eval_outputs':256,'public_saved_empty_outputs':8,'evaluation_decoded_full_calls':248,'fresh_parity_pairs':4,'fresh_parity_decoded_full_calls':8,'decoded_full_calls_planned_and_hard_class_cap':256,'code_only_calls_planned_and_hard_class_cap':128,'total_network_calls_planned':384,'global_network_call_hard_cap':420,'external_empty_controls_40times4':160,'all_empty_bypasses':168,'teacher_neural_calls':0,'no_extra_forward_for_FIT_logits_or_saved_eval_codes':True,'pre_dispatch_class_and_total_call_guards_required':True,'bound_notes':'Global420cap is redundant when full256/code128 subcaps are enforced; no unexplained spare calls may be consumed. Finished run must equal384calls with exact256/128split.'}
    report={'format':'predictor-only-CE-prospective-call-save-resource-metadata-review-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'status':'CONDITIONALLY_FEASIBLE_SOURCE_METADATA_REVIEW_RUNTIME_UNREADY_NO_PACKET','scope':'Arithmetic and retained tensor/serialization metadata only, not runtime/source admission or permission to train. No checkpoint deserialization or neural calls.','evidence':{'predictor_partition':pin(part_path),'parent_state_audit':pin(parent_path),'historical_serialization_audit':pin(prior_save),'original_teacher_index':pin(teacher_path)},'groups':groups,'planned_counts':counts,'terminal_bound':terminal,'NPZ_bounds':kinds,'array_schema_notes':{'FIT_logits_shape':[1,256,512],'FIT_labels_shape':[1,256],'FIT_case_index_cycle_microbatch':'Three int64 scalars,24bytes perFITNPZ, included.','eval_predicted_codes_nonempty':'int64(1,256) for248nonempty outputs','eval_predicted_codes_empty':'int64(0,) for8empty bypasses; raw512 absent with explicit flag; bound conservatively reserves full256int64 code slots for empties.','decoded_eval_plus_parity_code_array_data_bytes':248*256*8+4*2*256*8,'decoded_eval_plus_parity_code_array_data_cap':512*1024,'final_gray_nonworse_guard':'Eightgraychallenge cases: channel-spreadmacro+fourgeometry raw[0,1]/PNG[0,255] metadata only; fits8MiBallmetadata cap.'},'actual_existing80_teacher_files_bytes':teacher_actual,'output_bound_parts':output_parts,'output_upper_bound_bytes':output_bound,'output_category_cap_bytes':limits['whole_output_bytes'],'output_headroom_bytes':limits['whole_output_bytes']-output_bound,'tar_upper_bound_bytes':tar_bound,'gzip_upper_bound_bytes':gzip_bound,'tar_headroom_bytes':limits['export_raw_tar_bytes']-tar_bound,'gzip_headroom_bytes':limits['export_gzip_bytes']-gzip_bound,'limits':limits,'disk':{'free_bytes_required_after_install_and_before_worker':8*GiB,'installer_free_bytes_before_extract':'8GiB + exact uncompressed execution payload, with archive already present','floor_free_bytes':2*GiB,'maximum_new_output_plus_gzip_caps_bytes':disk_new_cap,'maximum_new_output_plus_gzip_caps_GiB':disk_new_cap/GiB,'remaining_above2GiB_floor_from8GiB_start_bytes':8*GiB-2*GiB-disk_new_cap,'no_second_raw_tar_on_disk':True,'successful_archive_partial_alias':'Same inode hardlink only; do not duplicate disk payload. Return manifest must not archive the same checkpoint payload twice under both success and partial names.','failed_array_or_terminal_partial':'Counts toward the same per-category and maximum-file slots. One failed attempt only; retain it and stop rather than retrying a full replacement.'},'timing_recommendation':{'worker_hard_seconds':1800,'worker_cooperative_stop_seconds':1680,'terminal_save_reserve_seconds':120,'export_cooperative_seconds':900,'decision':'Prefer1800/1680 over900/840 for this first real predictor-backward topology and up to1GiB recoverable failure state, while retaining finite stops. Historical V3~0.7GB save phases took~6.4s locally onL4 evidence, but that does not guarantee latency under new IO/gradients.','save_entry_deadline':'Begin terminal copy/serialization no later than1680s, preserve remaining120s under supervisorhard1800; failures/earlystops retain actualstate+partialgradients.','no_epoch_extension_or_automatic_resume':True},'mandatory_runtime_admission_gaps':['Freeze exact balanced128case sampler and source/teacher-key exclusion assertions;16monitor labelsets never enter training.','Match runtime114namedparameters/383frozen state and exact per-nameAdamstep/moment mapping, preserving old164encoder and54fusion slots without additional optimizerobjects.','Enforce these array dtype/shape/member/header andprimitive-pickle caps before writes; flags for savedempty cases must not fabricate raw512, predicted codes or networkcalls.','Use one terminal save attempt and cumulative1GiB terminal budget; retain failed partial without retry/duplicateparent optimizer serialization.','Guard all total/class calls BEFORE each dispatch; snapshots/FIT logits must reuse their corresponding call outputs.','LiveO1outputledger must include outputs plus protocol/trainerlog/supervisor/metadata;8MiB includes4MiBlog, not8+4additional.','Export protocol/supervisor/log/outputs only, <=2048regularmembers and240bytepaths; count rawtar andgzipstream bytes, never create a second rawtar diskfile.','Independent source/fixtures/transfer and emitted schema audit remain required; this prospective arithmetic alone is not a release.'],'primary_source_support':[{'url':'https://raw.githubusercontent.com/madler/zlib/v1.3.1/compress.c','role':'compressBound formula under defaultwindow/memlevels; rawdeflateNPZ uses conservative inclusion ofzlibheader overhead.'},{'url':'https://raw.githubusercontent.com/python/cpython/v3.10.14/Lib/zipfile.py','role':'Bounded ZIP local/central/ZIP64 headers and rawdeflate backend; no custom comments/unbounded names allowed.'}],'writer_formula':{'NPZ':'sum(compressBound(data_bytes+512NPYheader)+512ZIPentryoverhead) +1024global; up to10arrays, includingzero-sizedheaderallowances','terminal':'916100108 tensorbytes including165failedactualprior +1615*1024ZIP/name/alignment +1048576primitivepickle +1024global','tar':'output_bound +2048*(511padding+2048PAX/header) +2MiBmanifest+10240tail','gzip':'compressBound(tar_bound)+1024gzipheader/tailallowance'},'local_actions':{'Torch_imports':0,'checkpoint_deserializations':0,'neural_calls':0,'gradients':0,'optimizer_objects':0,'optimizer_steps':0,'VM_calls':0,'packet_or_archive_created':False},'model_qualified':False,'goal_complete':False}
    output=OUT/'serialization_resource_admission.json'
    with output.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'receipt':pin(output),'output_bound':output_bound,'terminal_bound':terminal_container,'tar_bound':tar_bound,'gzip_bound':gzip_bound,'timing':[1800,1680],'status':report['status']}))
if __name__=='__main__':main()
