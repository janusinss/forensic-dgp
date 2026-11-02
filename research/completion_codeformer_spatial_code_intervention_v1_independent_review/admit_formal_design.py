"""Independent pre-source admission of frozen spatial-code diagnostic design.
Read-only file/array validation; no model, inference, optimizer, gradients or VM.
"""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import hashlib,json,os,zipfile,io
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DESIGN=ROOT/'research/completion_codeformer_spatial_code_intervention_v1_design/root_design.json'
DESIGN_SHA='1983f9fed9bede2078ffc6a1af0dd6ca08fe6ec532e5c3de9c4ce41d4db2c657'
OLD=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return/cctv_completion_codeformer_same_codebook_probe_v1_results'
STAGE=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_packet/draft_001/cctv_completion_codeformer_same_codebook_probe_v1_vm'
CHECKS=0

def need(ok,msg):
 global CHECKS
 CHECKS+=1
 if not bool(ok):raise ValueError(msg)

def ip(path):
 p=Path(path).resolve();return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p

def sha(path):
 h=hashlib.sha256()
 with ip(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def pin(path):return {'path':str(Path(path).resolve()),'bytes':ip(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(ip(path).read_text(encoding='utf-8-sig'))
def needpin(row):need(pin(row['path'])=={k:row[k] for k in ('path','bytes','sha256')},'Exact provenance evidence '+row['path'])
def arr(path,name):
 with zipfile.ZipFile(ip(path)) as z:
  info=z.getinfo(name+'.npy');need(0<info.file_size<4096 and not info.flag_bits&1 and info.compress_type in (0,8),'Bounded integer index member');v=np.load(io.BytesIO(z.read(info)),allow_pickle=False,max_header_size=512)
 need(v.dtype==np.int64 and v.shape==(1,256) and ((v>=0)&(v<512)).all(),'Same512book integer code schema');return v

def main():
 need(sha(DESIGN)==DESIGN_SHA,'Frozen formaldesign hash');d=read(DESIGN);need(d['format']=='completion-codeformer-spatial-code-intervention-v1-root-design-v1' and d['status']=='PROSPECTIVE_SOURCE_PREPARATION_NOT_READY_NO_ACTUAL_RUN','Explicit prospectivedesign only');need(not d['runtime_ready'] and not d['transfer_ready'] and d['actual_executions']==d['local_neural_calls']==d['local_training_calls']==0 and not d['new_model_created'] and not d['model_qualified'] and not d['full_goal_complete'],'No model/training/readiness claim')
 for row in d['evidence'].values():needpin(row)
 oldp=read(OLD/'protocol.json');need(sha(OLD/'protocol.json')=='51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec','Admitted source oracle protocol');need(oldp['monitor_cases']+oldp['challenge_cases'] and d['data']['case_ids_in_fixed_protocol_order']==[r['id'] for r in oldp['monitor_cases']+oldp['challenge_cases']],'Exact48same adaptiveTRAIN caseorder');need(d['data']['monitor_cases']==32 and d['data']['challenge_cases']==16 and d['data']['monitor_reference_ids']==sorted({r['reference_id'] for r in oldp['monitor_cases']}),'Exact32clear16challenge8monitor refs');need(d['data']['parity_cases_exact']==oldp['parity_cases'],'Exact oldfour paritycases');need(d['data']['bridge_case_ids']==['tr_ffhq_05179__lower_face','tr_ffhq_05302__eye_band','v2m_05179_g0_gd','v2m_06781_g2_gd'] and set(d['data']['bridge_case_ids'])<=set(d['data']['case_ids_in_fixed_protocol_order']),'Fixed4bridge IDs predeclared');need(d['data']['fixed_masks'] is True and not d['data']['new_data'] and not d['data']['new_teacher_labels'] and not d['data']['clean_RGB_pasted'] and d['data']['native_unpaired_cases_new']==0 and not d['data']['reserved_FINAL_used'],'No newdata/labels/cleanpaste/FINAL/masks')
 labels={r['teacher_key']:r for r in d['data']['original16labels_exact_only']};oldlabels={r['teacher_key']:r for r in oldp['oracle_labels']};need(len(labels)==d['data']['oracle_label_sets']==16 and set(labels)==set(oldlabels),'Only16retained samebook labels');codes={}
 for k,row in labels.items():
  needpin(row['source_pin']);need(all(row[field]==oldlabels[k][field] for field in ('path','bytes','sha256','clean512','clean512_sha256','teacher_original_logical_sha256')) and row['oracle_unavailable_clean_information'] is True,'Samebook label provenance/source exact');codes[k]=arr(row['source_pin']['path'],'nearest_codes')
 external=d['external_controls'];need(external['VM_root_relative']=='../cctv_completion_codeformer_same_codebook_probe_v1_vm/outputs' and external['source_protocol_sha256']==d['evidence']['retained_oracle_actual_protocol']['sha256'] and external['historical_not_fresh48_A_or_C_calls'] and external['exact_hashes_required_no_fallback_recompute'],'Exactexternal oldA/C controls no fallback');rows={r['relative_output_path']:r for r in external['files']};need(len(rows)==len(external['files'])==196,'Exactly196external control files');oldmanifest={r['path']:r for r in read(OLD/'export_manifest.json')['files']};expected=set();selected=[]
 for arm,folder in [('A','ACTUAL_A_R1_w1'),('C','ORACLE_C_unavailable_clean_information_w1')]:
  for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
   expected.add(folder+'/'+kind+'/results.json')
   for case in oldp[key]:
    tag=case['id']+('__ORACLE_unavailable_clean_information' if arm=='C' else '')
    for suffix in ('.npz','.png'):expected.add(folder+'/'+kind+'/'+tag+suffix)
    rel=folder+'/'+kind+'/'+tag+'.npz';row=rows[rel];pred=arr(row['path'],'predicted_codes');used=arr(row['path'],'used_codes');need(np.array_equal(used,pred if arm=='A' else codes[case['teacher_key']]),'Exact old selectedcode provenance '+rel)
    if arm=='A':selected.append((case['id'],pred))
    else:need(np.array_equal(pred,dict(selected)[case['id']]),'OldA/Cpredictions identical')
 need(set(rows)==expected,'Closed96NPZ96PNG4results namespace')
 for rel,row in rows.items():
  path=Path(row['path']);need(path.resolve()==(OLD/'outputs'/rel).resolve(),'External local path binding');needpin(row);prior=oldmanifest['outputs/'+rel];need(row['bytes']==prior['bytes'] and row['sha256']==prior['sha256'],'Controlexact prioradmitted manifest')
 model=d['models'];need(model['original_checkpoint']['sha256']=='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797' and model['exact_full_R1_parent']['sha256']=='0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd' and model['all497_frozen_R1_logical']=='55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6' and model['no_predictor_CE_terminal_initialization'],'Exact frozenR1original prior nofailednewterminal init')
 arms={r['id']:r for r in d['conditions']};need(set(arms)=={'A','H','V','C'} and [arms[k]['new_forwards'] for k in ('A','H','V','C')]==[0,48,48,4] and all(r['fusion_w']==1 for r in arms.values()) and not any(arms[k]['deployable'] for k in ('H','V','C')),'Exact ordinary/oracle intervention scope');grid=d['grid_rule'];need(grid['shape']==[16,16] and grid['row_major'] and grid['fixed_threshold']=='any' and grid['no_dilation_or_threshold_search'] and grid['not_a_receptive_field_map'] and grid['delivered256_mask_unchanged'],'Fixed geometrical proxy no search')
 r=d['resources'];need(r['forward_slots_planned']==108 and r['forward_slots_max']==120 and r['typed_forward_slots_exact']=={'parity':8,'fresh_C_bridge':4,'H':48,'V':48} and sum(r['typed_forward_slots_exact'].values())==108,'Exact108typed120hard');need(all(r[k]==0 for k in ('optimizer_objects','optimizer_updates','gradient_queries','backward_calls','teacher_neural_calls')) and r['state_tensors_frozen']==497 and r['fusion_w']==1 and r['codebook_size']==512,'Frozen zerooptimizer/backward/teacher');need(r['NPZ_count']==104==96+r['parity_NPZ_count']+r['fresh_C_bridge_NPZ_count'] and r['new_HV_PNG_count']==96 and r['normal_checkpoint_files']==0 and r['failure_checkpoint_files_at_most']==1,'Exact104newarrays96PNGs/failureonlystate');need(r['worker_hard_seconds']==900 and r['worker_cooperative_seconds']==840 and r['finish_reserve_seconds']==60 and r['export_hard_seconds']==900,'Finite worker/exportreserve');need(r['minimum_free_bytes']==6*1024**3 and r['stop_free_bytes']==2*1024**3 and r['CUDA_allocated_bytes_cap']==20*1024**3,'ExactdiskCUDAcaps');need(r['output_bytes']==1024**3 and r['raw_tar_bytes_cap']==r['gzip_bytes_cap']==1280*1024**2 and r['no_quality_based_early_stop'],'Fixed1GiBoutput1280MiBarchives nooraclequalityearlyselection')
 arithmetic=d['resource_arithmetic'];data=4*15208448+100*4985090;reserved=data+104*65536;need(arithmetic['parity_data_bytes_each']==15208448 and arithmetic['HV_or_bridge_data_bytes_each']==4985090 and arithmetic['all104_numeric_data_plus64KiB_each']==reserved and reserved<=r['output_category_caps']['arrays_bytes']==560*1024**2,'Correct104NPZnumeric+ZIPheadroom');need(4985090+65536<r['single_new_HV_or_bridge_NPZ_bytes_cap']==6*1024**2 and 15208448+65536<r['single_parity_NPZ_bytes_cap']==16*1024**2,'SingleNPZ6/16MiBcaps');need(sum(r['output_category_caps'].values())==976*1024**2==arithmetic['whole_category_reservation_bytes']<r['output_bytes'] and r['output_bytes']+r['gzip_bytes_cap']<r['minimum_free_bytes']-r['stop_free_bytes'],'Allcategoriesfit1GiB/diskcoexistence');need(r['relative_path_UTF8_bytes']==240 and r['NPZ_numeric_members_max']==10 and r['NPY_header_bytes_cap']==512 and r['normal_return_regular_member_cap']==400,'Finite namespace/header/memberbounds')
 need(d['failure_retention']['no_integrity_failure_veto_of_resource_bounded_failure_save'] and d['failure_retention']['do_not_export_original_or_full_parent_weights'],'Retain actual mutatedstatewithoutscienceveto nohugeparentcopy');need(d['diagnostic_analysis']['cohorts_separate'] and d['diagnostic_analysis']['existing_quality_requirements_unchanged'] and d['diagnostic_analysis']['no_diagnostic_arm_adopted'] and 'C-H-V+A' in d['diagnostic_analysis']['effects'],'Separate diagnostic effects noqualification')
 admission=read(d['evidence']['R1_source_admission']['path']);r1pins=admission['source_files']+[admission['candidate_manifest'],admission['checkpoint']]
 for row in r1pins:needpin(row)
 clarify={'fresh_C_bridge_no_redundant_input512':'Root decision27October2026: bridge NPZ uses same9 corefields asH/V, not input512; oldC neverretained input512. Exact own/app512conditioning is saved byfourfreshordinaryparity pairs. Therefore frozen6MiBbridge and560MiBarray caps remain valid; noadditional12MiB schema is authorized.','actual_call_instrumentation_to_verify':'Ownedforward bypassesnet.__call__. Runtime must independentlyinstrument actualquantizer.get_codebook_feat plusgeneratorfirstpre/lastpost perowned+ordinary invocation; all108/globaltypedcounts must agree. Name bridge versusfresh_C_bridge maytranslate consistently withoutchanging4calls.','retained_inputs_not_newreturnNPZ':'All16labels and196A/Ccontrol files are exact retained external/packetinputs; no copy into return.104NPZmeans96H/V+4parity+4freshC only.'}
 result={'format':'independent-spatial-code-intervention-v1-formal-design-admission','complete':True,'independent':True,'UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(__file__),'root_design':pin(DESIGN),'root_design_sha256':DESIGN_SHA,'scientific_scope_admitted_for_source_preparation':True,'checks_passed':CHECKS,'frozen_R1_no_failed_predictor_initialization':True,'same48cases16labels196controls_exact':True,'all96prior_control_code_provenance_verified':True,'fourfreshparity_then_fourfreshC_before96HV_required':True,'counts':{'newfullforwards':108,'typed':r['typed_forward_slots_exact'],'newNPZ':104,'newPNG':96,'historicalA_C_files':196,'newoptimizer_backwards_gradients_teacher':0},'resource_arithmetic':{'numeric_arrays_bytes':data,'with104times64KiBZIPreserve':reserved,'arrays_cap':r['output_category_caps']['arrays_bytes'],'whole_category_caps':sum(r['output_category_caps'].values()),'output_cap':r['output_bytes'],'bridge_input512_omitted_not_unauthorized_cap_raise':True},'source_binding_clarifications_and_checks_remaining':clarify,'all22currentR1pins_exact':True,'runtime_source_audited':False,'packet_or_transfer_ready':False,'actual_execution_authorized_by_this_receipt':False,'no_newmodel_or_qualityclaim':True,'model_qualified':False,'goal_complete':False,'local_actions':{'Torch_imported':False,'model_construction':0,'neural_calls':0,'gradient_calls':0,'optimizer_objects':0,'VM_calls':0,'root_design_modified':False}}
 target=Path(__file__).parent/'formal_design_admission.json'
 with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(target),'checks':CHECKS,'admitted_for_source_preparation':True,'runtime_ready':False},indent=2),flush=True)
if __name__=='__main__':main()
