"""Independent frozen-state, selected-code provenance and finite-count return audit.
Only saved numeric arrays, source hashes and safely admitted CPU tensor states.
No model construction, inference, gradient, optimizer or VM action.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,math,sys,time,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RET=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return';RUN=RET/'cctv_completion_codeformer_same_codebook_probe_v1_results';STAGE=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_packet/draft_001/cctv_completion_codeformer_same_codebook_probe_v1_vm';PROTOCOL='51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec';TRANSPORT='ee4980dcd22a04f4315d7369a16c6fb2ce8ad8704532f6e1b284d35a45d24fa4';CAP=1536*1024**2;CHECKS=0
ORIGINAL=Path(r'C:\xampp\htdocs\YEAR 4\Testing\checkpoints\codeformer_inpainting.pth');ORIGINAL_SHA='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797';PARENT=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results/outputs/terminal_training_state.pth';PARENT_SHA='0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd';R1='55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6'
def need(ok,msg):
 global CHECKS
 CHECKS+=1
 if not ok:raise ValueError(msg)
def io_path(p):
 p=Path(p).resolve();return p if str(p).startswith('\\\\?\\') else Path('\\\\?\\'+str(p))
def sha(p):
 h=hashlib.sha256()
 with io_path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(io_path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(Path(p).resolve()),'bytes':io_path(p).stat().st_size,'sha256':sha(p)}
def main():
 start=time.monotonic();need(sha(RET/'transport_audit.json')==TRANSPORT,'Exact admitted transport');transport=read(RET/'transport_audit.json');need(transport['complete'] is True and transport['all_payload_manifest_hashes_exact'] is True and transport['closed_namespace_and_safe_paths'] is True and transport['protocol_sha256']==PROTOCOL,'Transport status differs');p=read(RUN/'protocol.json');need(sha(RUN/'protocol.json')==sha(STAGE/'protocol.json')==PROTOCOL,'Controlling protocol differs');files={r['path']:r for r in p['files']};need(len(files)==254,'254 source manifest differs')
 for name,row in files.items():q=STAGE/name;need(io_path(q).is_file() and io_path(q).stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Source/asset drift '+name)
 manifest=read(RUN/'export_manifest.json');mf={r['path']:r for r in manifest['files']};need(len(mf)==413 and set(mf)==set(transport['files'])-{'export_manifest.json'},'Return manifest closed scope differs')
 def evidence(relative):
  q=RUN/relative;row=mf[relative];need(io_path(q).stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Returned evidence drift '+relative);return q
 supervisor=read(evidence('supervisor.json'));result=read(evidence('outputs/results.json'));before=read(evidence('outputs/state_before.json'));after=read(evidence('outputs/state_after.json'));compat=read(evidence('outputs/ACTUAL_R1_48case_compatibility.json'));parity=read(evidence('outputs/ACTUAL_fresh_app_parity/summary.json'));labelreceipt=read(evidence('outputs/oracle_label_receipt.json'))
 need(supervisor['complete'] is True and supervisor['worker_exit']==supervisor['actual_child_exit']==0 and supervisor['timeout'] is False and supervisor['interruption'] is None and supervisor['shutdown_mode'] is None and supervisor['external_cap_seconds']==900 and 0<result['seconds']<supervisor['seconds']<900,'Finite supervisor completion differs');need(supervisor['no_learned_checkpoint'] is True and supervisor['stdout_capture_bytes']==0 and io_path(RUN/'trainer.log').stat().st_size==0 and supervisor['stdout_cap_reached'] is False,'No-checkpoint/intentional-empty-log status differs')
 need(result['complete'] is True and result['protocol_sha256']==PROTOCOL and result['outputs_saved']==192 and result['oracle_outputs_explicitly_labeled']==96 and result['all497_state_before_after_identical'] is True and result['R1_logical_sha256']==R1,'Diagnostic final scope differs')
 counts=lambda n:{'forward_slots':n,'actual_whole_model_forward_calls':n,'backward_calls':0,'gradient_queries':0,'optimizer_objects':0,'optimizer_steps':0,'teacher_neural_calls':0}
 need(result['counts']==counts(200) and result['typed_forward_slots']=={'parity':8,'A':48,'B':48,'C':48,'D':48},'Actual200calls zerooptimizer/teacher/gradient counts');need(all(result[k] is True for k in ('oracle_not_deployable_or_identity_recovery','same_codebook_target_not_mathematical_upper_bound','diagnostic_only')) and all(result[k] is False for k in ('new_model_created','checkpoint_saved','model_qualified','automatic_app_promotion')) and result['optimizer_steps']==result['backward_calls']==0,'No learned/deployable/identity claim')
 need(before==after and len(before['tensors'])==497 and before['logical_sha256']==R1 and all(before[k] is True for k in ('all_tensors_finite','all_parameters_frozen','all_parameter_gradients_none','all_modules_eval')),'All497 frozen/eval/finite/no-gradient records differ');need(result['output_budget_bytes']<CAP and sum(r['bytes'] for r in mf.values())<CAP and not any(n.endswith(('.pth','.pt','.partial')) for n in mf),'Raw bounded/no checkpoint output')
 need(labelreceipt['sets']==16 and labelreceipt['teacher_calls']==0 and labelreceipt['existing_same_codebook_labels_only'] is True and labelreceipt['unused_prequant_latents_not_passed_to_model'] is True and labelreceipt['oracle_unavailable_clean_information'] is True and labelreceipt['source_pins']==p['oracle_labels'],'Same-codebook label receipt differs')
 labels={};labelpins=[]
 for row in p['oracle_labels']:
  q=STAGE/row['path'];need(sha(q)==row['sha256'] and io_path(q).stat().st_size==row['bytes'],'Retained label exact pin');need(sha(STAGE/row['clean512'])==row['clean512_sha256'] and row['teacher_original_logical_sha256']=='586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f','Original same-prior/clean-context provenance')
  with zipfile.ZipFile(io_path(q)) as z:need(z.testzip() is None and set(z.namelist())=={'prequant_latent.npy','nearest_codes.npy'},'Retained label ZIP/schema')
  with np.load(io_path(q),allow_pickle=False) as z:codes=z['nearest_codes'].copy();need(codes.dtype==np.int64 and codes.shape==(1,256) and ((codes>=0)&(codes<512)).all(),'Retained codes range/shape');labels[row['teacher_key']]=codes
  labelpins.append(pin(q))
 need(len(labels)==16,'Exactly16 retained labels')
 need(parity['complete'] is True and parity['all4_saved_before_gate'] is True and parity['tolerance']==1e-6 and parity['counts']==counts(8) and len(parity['cases'])==4,'All4 ordinary parity prefix differs')
 for row in parity['cases']:need(row['max_raw256_error']<=1e-6 and row['input512_exact'] is True and row['input_stride_exact'] is True and row['predicted_codes_exact'] is True and row['used_equals_predicted'] is True,'Ordinary owned/app parity failed')
 need(result['ordinary_app_parity']==parity['cases'] and result['actual_R1_historical_compatibility']==compat['cases'],'Final embedded parity/compatibility differs');need(compat['complete'] is True and compat['before_any_oracle'] is True and compat['counts']==counts(56) and compat['pass'] is True and len(compat['cases'])==48 and all(r['raw256_max_error']<=1e-6 and r['PNG_bytes_exact'] is True for r in compat['cases']),'All48 initial A admission before any oracle differs')
 arms=p['arms'];need([(r['id'],r['w'],r['oracle']) for r in arms]==[('A',1,False),('B',0,False),('C',1,True),('D',0,True)],'Fixed arm ordering differs');baseline={};selected_diagnostics=[];seenoutputs=[]
 for armindex,arm in enumerate(arms):
  for kind,casekey in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
   cases=p[casekey];summary=read(evidence('outputs/'+arm['filename_tag']+'/'+kind+'/results.json'));summary_n=8+armindex*48+(32 if kind=='monitor' else 48);need(summary['arm']==arm['id'] and summary['w']==arm['w'] and summary['oracle_unavailable_clean_information']==arm['oracle'] and summary['counts']==counts(summary_n) and [r['id'] for r in summary['records']]==[r['id'] for r in cases],'Cohort/typed counter prefix differs')
   for case,row in zip(cases,summary['records']):
    tag=case['id']+('__ORACLE_unavailable_clean_information' if arm['oracle'] else '');relative='outputs/'+arm['filename_tag']+'/'+kind+'/'+tag+'.npz';q=evidence(relative)
    with np.load(io_path(q),allow_pickle=False) as z:
     need(set(z.files)=={'input256','mask256','result256','raw512','predicted_codes','used_codes','arm_code','oracle_unavailable_clean_information'},'Exact arm array schema');pred=z['predicted_codes'].copy();used=z['used_codes'].copy();codeflag=z['arm_code'].copy();oracleflag=z['oracle_unavailable_clean_information'].copy()
    need(pred.dtype==used.dtype==np.int64 and pred.shape==used.shape==(1,256) and ((pred>=0)&(pred<512)).all() and ((used>=0)&(used<512)).all(),'Actual predicted/selected code shape/range')
    need(codeflag.dtype==oracleflag.dtype==np.uint8 and codeflag.shape==oracleflag.shape==() and int(codeflag)==ord(arm['id']) and bool(oracleflag)==arm['oracle'],'Saved explicit arm/oracle flags')
    if arm['id']=='A':baseline[case['id']]=pred
    need(np.array_equal(pred,baseline[case['id']]),'Observed predicted codes differ acrossarms')
    expected=labels[case['teacher_key']] if arm['oracle'] else pred;need(np.array_equal(used,expected),'Selected code provenance/override differs')
    agreement=float((pred==used).mean());need(abs(agreement-row['predicted_used_code_agreement'])<=1e-15 and row['teacher_key']==case['teacher_key'] and row['outside_exact'] is True and row['not_deployable']==(arm['id']!='A'),'Selected code receipt differs')
    need(row['oracle_unavailable_clean_information']==arm['oracle'] and ('ORACLE_unavailable_clean_information' in relative)==arm['oracle'] and sha(RUN/'outputs'/row['PNG'])==row['PNG_sha256'],'Persistent oracle naming and PNG pin')
    selected_diagnostics.append({'id':case['id'],'kind':kind,'arm':arm['id'],'teacher_key':case['teacher_key'],'predicted_sha256':hashlib.sha256(pred.tobytes()).hexdigest(),'used_sha256':hashlib.sha256(used.tobytes()).hexdigest(),'predicted_used_agreement':agreement,'selected_equals_declared_actual_or_oracle':True});seenoutputs.append(relative)
  complete=read(evidence('outputs/arm_'+arm['id']+'_complete.json'));need(complete['complete'] is True and complete['arm']==arm and complete['oracle_unavailable_clean_information']==arm['oracle'] and complete['counts']==counts(8+48*(armindex+1)),'Complete arm prefix differs')
 need(len(seenoutputs)==len(set(seenoutputs))==192 and sum(n.endswith('.npz') for n in mf)==196 and sum(n.endswith('.png') for n in mf)==192,'All192 armoutputs+4parity counts')
 # Reconstruct only tensor dictionaries after passive ZIP/CRC/restricted-global admission.
 helper=ROOT/'scripts/audit_completion_codeformer_delivered_objective_v4_state.py';spec=importlib.util.spec_from_file_location('independent_prior_passive_reader',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 passive={'original':module.passive(ORIGINAL,ORIGINAL_SHA),'R1_parent':module.passive(PARENT,PARENT_SHA)}
 import torch
 torch.set_num_threads(1);original=torch.load(ORIGINAL,map_location='cpu',weights_only=True)['params_ema'];parent=torch.load(PARENT,map_location='cpu',weights_only=True);need(len(original)==497 and len(parent['trainable_state'])==332 and parent['logical_state_sha256']==R1 and parent['base_checkpoint_sha256']==ORIGINAL_SHA,'Original/fullR1 source state differs');full={**original,**parent['trainable_state']};digest=hashlib.sha256();prior=0
 for name,value in sorted(full.items()):
  a=value.detach().cpu().contiguous();body=a.numpy().tobytes();need(a.dtype==torch.float32 and torch.isfinite(a).all().item(),'Source recovered tensor invalid');digest.update(name.encode());digest.update(str(tuple(a.shape)).encode());digest.update(str(a.dtype).encode());digest.update(body);expected={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(body).hexdigest(),'finite':True};need(before['tensors'][name]==after['tensors'][name]==expected,'Actual497 before/after record differs from recoveredR1 '+name)
  if name.startswith(('quantize.','generator.')):prior+=1;need(torch.equal(value,original[name]),'Original prior changed '+name)
 need(digest.hexdigest()==R1 and prior==165,'FullR1logical/original165prior differs')
 # Rehash admitted current app/model assets without importing them.
 admission=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json');R1pins=admission['source_files']+[admission['candidate_manifest'],admission['checkpoint']]
 for row in R1pins:q=Path(row['path']);need(io_path(q).stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Current R1 model/app drift '+str(q))
 need(sha(ORIGINAL)==ORIGINAL_SHA and sha(PARENT)==PARENT_SHA and sha(RET/'transport_audit.json')==TRANSPORT,'Source/admitted return changed during audit')
 report={'format':'independent-same-codebook-probe-frozen-state-and-code-provenance-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'transport_audit_sha256':TRANSPORT,'protocol_sha256':PROTOCOL,'supervisor':supervisor,'counts':result['counts'],'typed_forward_slots':result['typed_forward_slots'],'completed_arm_prefixes_checked':4,'all192_selected_code_sources_independently_verified':True,'predicted_codes_identical_across_all4arms':True,'actual_arms_used_codes_equal_predictions':True,'oracle_arms_used_codes_equal_exact_retained_samebook_labels':True,'oracle_label_source_pins':labelpins,'selected_code_diagnostics':selected_diagnostics,'state_invariance':{'before_and_after_all497_records_identical':True,'all497_records_equal_actual_reconstructed_R1_cpu_tensor_bytes':True,'full_logical_sha256':R1,'original_prior165_exact':True,'all_model_parameters_frozen_gradients_none_eval_flags_true':True,'no_new_checkpoint_expected_or_returned':True,'safe_original_R1_loading':{'passive_reviews':passive,'weights_only':True,'map_location':'cpu','model_instantiated':False}},'ordinary_parity_and_historical_preoracle_admission':{'fresh_pairs':4,'fresh_calls':8,'all48A_raw_maxerror':max(r['raw256_max_error'] for r in compat['cases']),'all48A_PNG_bytes_exact':True,'counts_before_control_oracle':counts(56)},'resources':{'hard_seconds':900,'actual_supervisor_seconds':supervisor['seconds'],'hard_forward_cap':220,'actual_forwards':200,'raw_output_payload_bytes':sum(r['bytes'] for r in mf.values()),'rawcap_bytes':CAP,'all196NPZ192PNG_retained':True,'firstWindowsMAXPATHattempt_preserved_not_admitted':True},'all254_staged_source_asset_pins_exact':True,'all22_current_R1_app_model_pins_exact':True,'checks_passed':CHECKS,'interpretation_limits':['Oracle C/D use unavailable clean hidden information; they are diagnostic, never a new deployable model or exact hidden identity.','Code provenance, pinned bridge and immutable tensor records are independently verified; no new neural forward was executed locally to reproduce decoding.','Returned before/after hashes cannot independently observe every transient internal VM state, but all497 retained records match actual fullR1 source tensor bytes.','Source-valid counts and successfulexport establish completed diagnosis, not useful image quality or predictability of oracle codes.','Paired adaptive TRAIN48cases remain separate from independent FINAL/native CCTV; no automatic optimizer pilot follows.'],'local_actions':{'CPU_weights_only_loads':2,'neural_models_or_forwards':0,'gradients':0,'optimizer_objects_or_steps':0,'VM_calls':0,'returned_code_executed':False},'diagnostic_only':True,'new_model_created':False,'model_qualified':False,'seconds':time.monotonic()-start}
 out=RET/'independent_state_code_audit.json'
 with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(out),'checks':CHECKS,'state497exact':True,'selected_codes_verified':192,'actualcalls':200,'optimizer_steps':0,'seconds':report['seconds']},indent=2),flush=True)
if __name__=='__main__':main()
