"""Independent actual return: named Adam, tensor state, saved CE and code provenance.
No returned Python execution, neural model, optimizer construction or VM action.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import argparse, hashlib, json, math, pickletools, time, zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001/cctv_completion_codeformer_predictor_ce_v1_vm'
PROTOCOL='15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b'
ORIGINAL=Path(r'C:\xampp\htdocs\YEAR 4\Testing\checkpoints\codeformer_inpainting.pth')
ORIGINAL_SHA='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
ORIGINAL_LOGICAL='586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f'
PARENT=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results/outputs/terminal_training_state.pth'
PARENT_SHA='0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd'
PARENT_LOGICAL='55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6'
SAFE_GLOBALS={'collections OrderedDict','torch._utils _rebuild_tensor_v2','torch FloatStorage','torch ByteStorage','torch LongStorage'}
CHECKS=0

def need(ok,msg):
 global CHECKS
 CHECKS+=1
 if not bool(ok):raise ValueError(msg)

def io_path(path):
 p=Path(path).resolve();return p if str(p).startswith('\\\\?\\') else Path('\\\\?\\'+str(p))

def sha(path):
 h=hashlib.sha256()
 with io_path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def pin(path):
 path=Path(path).resolve();need(io_path(path).is_file() and not path.is_symlink(),'Missing/linked evidence '+str(path))
 return {'path':str(path),'bytes':io_path(path).stat().st_size,'sha256':sha(path)}

def read(path):
 def reject(value):raise ValueError('Nonfinite JSON '+value)
 return json.loads(io_path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)

def js(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def predictor(n):return n=='position_emb' or n.startswith(('feat_emb.','ft_layers.','idx_pred_layer.'))

def numeric(path,keys):
 with zipfile.ZipFile(io_path(path)) as z:
  info=z.infolist();need(len(info)==len(keys) and {v.filename for v in info}=={k+'.npy' for k in keys},'Exact numeric ZIP namespace '+str(path));need(all(not v.is_dir() and not v.flag_bits&1 and v.file_size<8*1024**2 and v.compress_type in (0,8) for v in info),'Numeric ZIP resource/schema');need(z.testzip() is None,'Numeric ZIP CRC')
 with np.load(io_path(path),allow_pickle=False,max_header_size=512) as z:a={k:z[k].copy() for k in keys}
 for k,v in a.items():need(not v.dtype.hasobject and v.dtype in (np.float32,np.int64) and np.isfinite(v).all(),'Finite permitted numeric dtype '+k)
 return a

def passive(path,expected):
 record=pin(path);need(record['sha256']==expected,'Checkpoint hash differs '+str(path));need(record['bytes']<=1024**3,'Bounded tensor archive')
 with zipfile.ZipFile(io_path(path)) as z:
  members=z.infolist();names=[m.filename for m in members];need(0<len(names)<=1615 and len(names)==len(set(names)),'Unique bounded checkpoint members');roots=set();metadata=[]
  for m in members:
   n=PurePosixPath(m.filename);need(not m.is_dir() and not m.flag_bits&1 and m.compress_type==0 and not n.is_absolute() and '..' not in n.parts and '\\' not in m.filename and ':' not in m.filename and len(n.parts)>=2,'Safe Torch ZIP namespace');roots.add(n.parts[0]);rel='/'.join(n.parts[1:]);need(rel in ('data.pkl','byteorder','version','.data/serialization_id','.format_version','.storage_alignment') or rel.startswith('data/') and rel[5:].isdigit(),'Tensor-state ZIP only')
   if rel=='data.pkl':metadata.append(m)
  need(len(roots)==len(metadata)==1 and metadata[0].file_size<=1024**2,'One bounded primitive tensor pickle');need(z.testzip() is None,'Complete checkpoint CRC');raw=z.read(metadata[0]);seen=set();ops=0
  for op,arg,pos in pickletools.genops(raw):
   ops+=1;need(op.name!='STACK_GLOBAL','Dynamic pickle globals refused')
   if op.name=='GLOBAL':need(arg in SAFE_GLOBALS,'Unexpected static pickle global '+str(arg));seen.add(arg)
 return {'file':record,'complete_ZIP':True,'all_member_CRC':True,'members':len(members),'primitive_pickle_bytes':len(raw),'pickle_opcodes':ops,'static_globals':sorted(seen),'weights_only_CPU_load_permitted':True,'arbitrary_pickle_load':False,'safe_globals_added':False}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--return-root',type=Path,required=True);parser.add_argument('--transport-sha',required=True);a=parser.parse_args();start=time.monotonic();ret=a.return_root.resolve(strict=True);run=ret/'cctv_completion_codeformer_predictor_ce_v1_results'
 transport_path=ret/'transport_audit.json';need(sha(transport_path)==a.transport_sha,'Exact admitted transport binding');transport=read(transport_path);need(transport['complete'] is True,'Transport incomplete')
 p=read(run/'protocol.json');need(sha(run/'protocol.json')==sha(STAGE/'protocol.json')==PROTOCOL,'Protocol bytes differ');manifest=read(run/'export_manifest.json');mf={r['path']:r for r in manifest['files']};need(len(mf)==len(manifest['files'])==834,'Exact834payload manifest')
 def evidence(rel):
  need(rel in mf,'Missing returned evidence '+rel);path=run/rel;row=mf[rel];need(io_path(path).stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Returned evidence drift '+rel);return path
 for row in p['files']:q=STAGE/row['path'];need(io_path(q).stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Sealed source/asset drift '+row['path'])
 result=read(evidence('outputs/results.json'));supervisor=read(evidence('supervisor.json'));pre=read(evidence('outputs/pre_save_receipt.json'));initial=read(evidence('outputs/initial_state.json'));transition=read(evidence('outputs/RNG_transition.json'));before=read(evidence('outputs/R1_state_before.json'));after=read(evidence('outputs/actual_state_at_save.json'));compat=read(evidence('outputs/historical_R1_compatibility.json'));objective=read(evidence('outputs/objective_contract.json'))
 expected_counts={'forward_slots':384,'backward_calls':128,'predictor_optimizer_steps':32,'empty_bypasses':168};typed={'parity':8,'fit_code':128,'monitor':128,'challenge':64,'public_development':56}
 need(result['complete'] is True and result['protocol_sha256']==PROTOCOL and result['predictor_steps_completed_newphase']==32 and result['historical_R1_paired_cycles']==32 and result['stop_reason'] is None,'Exact completed finite32steps');need(result['counts_newphase']==expected_counts and result['typed_forward_slots']==typed and result['actual_model_forward_counts']=={'calls':384,'generator_calls':256,'code_only_calls':128} and result['FIT_logits_saved']==128 and result['teacher_neural_calls']==0,'Exact finite counters');need(result['model_qualified'] is False and result['automatic_app_promotion'] is False and result['numeric_screen']['prospective_numeric_screen_pass'] is False,'Retain quality rejection')
 need(supervisor['complete'] is True and supervisor['worker_exit']==supervisor['actual_child_exit']==2 and supervisor['timeout'] is False and supervisor['interruption'] is None and supervisor['shutdown_mode'] is None and supervisor['external_cap_seconds']==1800 and 0<result['seconds']<supervisor['seconds']<1800,'Finite non-timeout quality exit2');need(supervisor['terminal_state_present'] and not supervisor['terminal_partial_present'] and not supervisor['stdout_cap_reached'],'Bounded terminal/log status')
 need(objective['objective']=='predictor_only_CE_all256_same_inpainting_codebook_v1' and objective['teacher_neural_calls']==0 and objective['mean_CE_indices']==256 and objective['codebook_size']==512 and objective['microbatch_divisor_once']==4 and objective['latent_MSE_added'] is False and objective['image_loss_added'] is False and objective['fusion_w_during_evaluation']==1 and objective['hard_top1_used_in_ordinary_evaluation'] is True and objective['oracle_inference'] is False and objective['all_old_quality_gates_retained'] is True and objective['FIT_logits_saved_for_independent_CE_replay'] is True,'Fixed objective/path provenance')
 need(result['initial_R1_logical_state_sha256']==PARENT_LOGICAL and result['original_base_logical_state_sha256']==ORIGINAL_LOGICAL,'Ancestry logical binding')
 terminal=evidence('outputs/terminal_training_state.pth');t=result['terminal_training_state'];need(t['bytes']==io_path(terminal).stat().st_size and t['sha256']==sha(terminal) and t['cycle']==32 and t['retained_failure_gates']==[] and pre['cycle']==32 and pre['counts']==expected_counts and pre['retained_failure_gates']==[],'Committed actual terminal/no failuregates')
 need(not any(n.endswith('.partial') for n in mf) and 'outputs/failure.json' not in mf,'No returned failedpartial/unexpectedexception')
 print(json.dumps({'phase':'reading_exact_retained_labels_and_FIT_logits','seconds':time.monotonic()-start}),flush=True)
 labels={};labelpins=[];copyreceipt=read(evidence('outputs/cached_label_receipt.json'));need(copyreceipt['sets']==80 and copyreceipt['FIT_sets']==64 and copyreceipt['monitor_diagnostic_only_sets']==16 and copyreceipt['teacher_neural_calls']==0 and copyreceipt['new_labels_generated'] is False and copyreceipt['unused_prequant_latents_not_in_objective'] is True,'Retained label provenance')
 for row in p['cached_labels']:
  source=STAGE/row['path'];need(sha(source)==row['sha256'] and io_path(source).stat().st_size==row['bytes'],'Pinned original nearest-code labels');dest=evidence('outputs/teacher_labels/'+Path(row['path']).name);need(sha(dest)==row['sha256'],'Copied retained labels exact');arr=numeric(source,{'prequant_latent','nearest_codes'});v=arr['nearest_codes'];need(v.dtype==np.int64 and v.shape==(1,256) and ((v>=0)&(v<512)).all(),'Samebook512 labels valid');labels[row['teacher_key']]=v;labelpins.append({'teacher_key':row['teacher_key'],'sha256':row['sha256']})
 refs={r['id']:r for r in p['references']};fit={r['id']:r for r in p['fit_cases']};order={r['id']:i for i,r in enumerate(p['fit_cases'])};cycles=[];all_fit=[];ceerrors=[];lastupdate={};gradientzeros={};updatezeros={};clip=[];oldtime=0
 for cycle in range(1,33):
  row=read(evidence('outputs/training_cycle_%04d.json'%cycle));ids=p['schedule'][cycle-1];need(row['cycle']==cycle and row['microbatch_ids']==ids and len(ids)==len(set(ids))==4 and len({fit[k]['reference_id'] for k in ids})==4,'Fixed balanced4reference schedule');expected={'forward_slots':84+4*cycle+48*sum(cycle>s for s in (8,16,32)),'backward_calls':4*cycle,'predictor_optimizer_steps':cycle,'empty_bypasses':44+40*sum(cycle>s for s in (8,16,32))};need(row['counts']==expected,'Per-cycle actual counter prefix');need(row['objective']==objective['objective'] and oldtime<row['seconds']<result['seconds'],'Monotonic cycle seconds');oldtime=row['seconds'];need(len(row['predictor_losses'])==4,'Four saved loss rows');losses=[]
  for micro,(ident,l) in enumerate(zip(ids,row['predictor_losses']),1):
   case=fit[ident];need(refs[case['reference_id']]['role']=='fit' and l['id']==ident and l['reference_id']==case['reference_id'] and l['teacher_key']==case['teacher_key'] and l['style']==case['style'] and l['microbatch_divisor']==4,'Actual FIT labels/IDs only');q=evidence('outputs/'+l['FIT_logits']);need(sha(q)==l['FIT_logits_sha256'],'FIT array pin');v=numeric(q,{'logits','labels','case_index','cycle','microbatch'});logit=v['logits'];lab=v['labels'];need(logit.dtype==np.float32 and logit.shape==(1,256,512) and lab.dtype==np.int64 and lab.shape==(1,256) and np.array_equal(lab,labels[case['teacher_key']]),'Full128 logits/labels exact samebook');need(all(v[k].dtype==np.int64 and v[k].shape==() for k in ('case_index','cycle','microbatch')) and int(v['cycle'])==cycle and int(v['microbatch'])==micro and int(v['case_index'])==order[ident],'FIT metadata index/cycle/micro exact')
   flat=logit[0].astype(np.float64);maximum=flat.max(axis=1);ce=float((maximum+np.log(np.exp(flat-maximum[:,None]).sum(axis=1))-flat[np.arange(256),lab[0]]).mean());err=abs(ce-l['CE_mean_all256']);need(math.isfinite(l['CE_mean_all256']) and err<=2e-6,'FP32 CUDA mean CE differs from independent float64 replay');need(l['scaled_microbatch_loss']==l['CE_mean_all256']/4 and abs(l['scaled_microbatch_loss']-ce/4)<=5e-7,'Exactlyonce /4 scaling');agreement=float((flat.argmax(axis=1)==lab[0]).mean());need(agreement==l['clean_context_code_agreement_diagnostic'],'FIT whole-grid top1 diagnostic replay');ceerrors.append(err);all_fit.append(ident);losses.append({'id':ident,'reference_id':case['reference_id'],'teacher_key':case['teacher_key'],'CE_saved_FP32':l['CE_mean_all256'],'CE_replayed_float64':ce,'absolute_replay_error':err,'whole256_code_agreement':agreement,'loss_scaled_once_by4':True})
  groups=row['predictor_gradient_groups'];per=groups['per_parameter'];need(len(per)==114 and all(predictor(n) for n in per),'All114 per-parameter gradient norms present');need(len(row['predictor_actual_parameter_updates'])==114 and set(row['predictor_actual_parameter_updates'])==set(per),'All114 actual update deltas')
  for n,g in per.items():
   norm=g['L2'];need(g['finite'] is True and math.isfinite(norm) and norm>=0 and g['nonzero']==(norm>0) and g['zero_retained_honestly']==(norm==0),'Honest per-tensor gradient norm '+n);gradientzeros[n]=gradientzeros.get(n,0)+(norm==0)
   u=row['predictor_actual_parameter_updates'][n];need(u['finite'] is True and math.isfinite(u['delta_L2']) and math.isfinite(u['maximum_absolute_delta']) and u['delta_L2']>=0 and u['maximum_absolute_delta']>=0 and u['changed']==(u['before_sha256']!=u['after_sha256']),'Actual parameter delta finite/schema '+n)
   if n in lastupdate:need(lastupdate[n]==u['before_sha256'],'All114 parameter SHA update chains')
   lastupdate[n]=u['after_sha256'];updatezeros[n]=updatezeros.get(n,0)+(not u['changed'])
  grouping={'position':[n for n in per if n=='position_emb'],'feature':[n for n in per if n.startswith('feat_emb.')],'transformer':[n for n in per if n.startswith('ft_layers.')],'prediction':[n for n in per if n.startswith('idx_pred_layer.')]}
  need(set(groups)==set(grouping)|{'per_parameter'},'Four exact predictor gradient groups')
  for k,names0 in grouping.items():g=groups[k];norm=math.sqrt(sum(per[n]['L2']**2 for n in names0));need(g['tensors']==len(names0) and g['finite'] is True and g['nonzero']==(g['L2']>0) and math.isclose(g['L2'],norm,rel_tol=1e-10,abs_tol=1e-10),'Group norm equals actual reported114parts '+k)
  norm=math.sqrt(sum(per[n]['L2']**2 for n in per));need(math.isclose(row['predictor_preclip_L2'],norm,rel_tol=2e-5,abs_tol=1e-7) and row['gradient_clip_applied']==(row['predictor_preclip_L2']>1),'Preclip norm and clipping flag');clip.append(row['predictor_preclip_L2']);cycles.append({'cycle':cycle,'seconds':row['seconds'],'counts':expected,'FIT':losses,'preclip_L2':row['predictor_preclip_L2'],'gradient_clip_applied':row['gradient_clip_applied']})
 need(len(all_fit)==len(set(all_fit))==128 and set(all_fit)==set(fit),'All128 fixed distinct FIT observations exactlyonce');need(sum(n.startswith('outputs/FIT_logits/') and n.endswith('.npz') for n in mf)==128,'Exact128 FIT arrays saved')
 monitor_labels={r['teacher_key'] for r in p['monitor_cases']+p['challenge_cases']};usedlabels={fit[k]['teacher_key'] for k in all_fit};need(len(usedlabels)==64 and len(monitor_labels)==16 and not usedlabels&monitor_labels,'All64FIT labels,16monitor labels disjoint')
 # This is a whole256-grid target diagnostic, not masked accuracy or identity confidence.
 code_rows=[];code_summaries={}
 for cycle in (0,8,16,32):
  code_summaries[str(cycle)]={}
  for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
   summary=read(evidence('outputs/u%04d/%s/results.json'%(cycle,kind)));cases=p[key];need([r['id'] for r in summary['records']]==[r['id'] for r in cases],'Fixed all48 monitor snapshot caseorder');agreements=[]
   for case,r in zip(cases,summary['records']):
    q=evidence('outputs/u%04d/%s/%s.npz'%(cycle,kind,case['id']));v=numeric(q,{'input256','mask256','result256','raw512','predicted_codes'});codes=v['predicted_codes'];need(codes.dtype==np.int64 and codes.shape==(1,256) and ((codes>=0)&(codes<512)).all(),'Actual ordinary predicted512indices');agreement=float((codes==labels[case['teacher_key']]).mean());need(agreement==r['same_codebook_clean_context_agreement_diagnostic'],'Saved EVAL wholegrid agreement replay');agreements.append(agreement);code_rows.append({'cycle':cycle,'kind':kind,'id':case['id'],'mask_kind':case['mask_kind'],'condition':case.get('condition'),'style':case['style'],'teacher_key':case['teacher_key'],'whole256_code_agreement':agreement,'predicted_codes_sha256':hashlib.sha256(codes.tobytes()).hexdigest(),'samebook_clean_reference_codes_sha256':hashlib.sha256(labels[case['teacher_key']].tobytes()).hexdigest(),'actual_npz':str(q.resolve()),'teacher_npz':str((STAGE/next(x['path'] for x in p['cached_labels'] if x['teacher_key']==case['teacher_key'])).resolve())})
   code_summaries[str(cycle)][kind]={'cases':len(agreements),'mean_whole256_agreement':sum(agreements)/len(agreements),'minimum':min(agreements),'maximum':max(agreements)}
 print(json.dumps({'phase':'all128_CE_replayed_all192_eval_codes_bound','maximum_CE_abs_error':max(ceerrors),'code_summaries':code_summaries,'seconds':time.monotonic()-start}),flush=True)
 # Passive static-global and complete ZIP/CRC admission precedes every CPU load.
 passive_reports={'original':passive(ORIGINAL,ORIGINAL_SHA),'R1_parent':passive(PARENT,PARENT_SHA),'returned_terminal':passive(terminal,t['sha256'])}
 import torch
 torch.set_num_threads(1)
 original=torch.load(io_path(ORIGINAL),map_location='cpu',weights_only=True);parent=torch.load(io_path(PARENT),map_location='cpu',weights_only=True);state=torch.load(io_path(terminal),map_location='cpu',weights_only=True)
 def ts(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
 def logical(mapping):
  h=hashlib.sha256()
  for n,v in sorted(mapping.items()):c=v.detach().cpu().contiguous();h.update(n.encode());h.update(str(tuple(c.shape)).encode());h.update(str(c.dtype).encode());h.update(c.numpy().tobytes())
  return h.hexdigest()
 def tree(v):
  h=hashlib.sha256()
  def visit(x):
   if isinstance(x,torch.Tensor):c=x.detach().cpu().contiguous();need(torch.isfinite(c).all().item(),'Finite retained tensor tree');h.update(str(c.dtype).encode());h.update(str(tuple(c.shape)).encode());h.update(c.numpy().tobytes())
   elif isinstance(x,dict):
    for k in sorted(x,key=lambda k:(str(type(k)),str(k))):h.update(repr(k).encode());visit(x[k])
   elif isinstance(x,(tuple,list)):
    h.update(type(x).__name__.encode())
    for z in x:visit(z)
   else:h.update(repr(x).encode())
  visit(v);return h.hexdigest()
 def tensorbytes(v):
  if isinstance(v,torch.Tensor):return v.numel()*v.element_size()
  if isinstance(v,dict):return sum(tensorbytes(x) for x in v.values())
  if isinstance(v,(list,tuple)):return sum(tensorbytes(x) for x in v)
  return 0
 need(set(original)=={'params_ema'},'Original checkpoint envelope');base=original['params_ema'];need(len(base)==497 and logical(base)==ORIGINAL_LOGICAL,'Original497logical bytes');delta={n for n in base if not n.startswith(('quantize.','generator.'))};active={n for n in base if predictor(n)};frozen=set(base)-active;need(len(delta)==332 and len(active)==114 and len(frozen)==383,'Exact model partitions');need(set(parent['trainable_state'])==set(state['trainable_state'])==delta,'Full332 delta state retained');pf={**base,**parent['trainable_state']};full={**base,**state['trainable_state']};need(logical(pf)==PARENT_LOGICAL and logical(full)==state['logical_state_sha256']==result['final_logical_state_sha256']==t['logical_state_sha256'],'All original/R1/terminal logical hashes')
 need(parent['protocol_sha256']==p['parent']['protocol_sha256'] and parent['cycle']==32 and parent['partial_accumulated_gradients']=={} and parent['reason']=='completed_finite_pilot','Exact retained R1phase');need(state['format']=='completion-codeformer-predictor-ce-v1-trainable-resume-v1' and state['protocol_sha256']==PROTOCOL and state['base_checkpoint_sha256']==ORIGINAL_SHA and state['parent_training_state_sha256']==PARENT_SHA and state['parent_initial_logical_sha256']==PARENT_LOGICAL,'New terminal ancestry');need(state['counts']==expected_counts and state['contract']==p['contract'] and state['phase_name']=='predictor_ce_v1' and state['cycle']==state['phase_completed_predictor_steps']==state['completed_predictor_steps_newphase']==32 and state['completed_paired_cycles']==32 and state['reason']=='completed_finite_pilot','New exact32predictor phase');need(state['partial_accumulated_gradients']==state['failed_frozen_prior_state']=={} and state['retained_failure_gates']==[] and state['automatic_resume_permitted'] is False and state['model_qualified'] is False and state['state_is_qualified'] is False,'Terminal honest recovery/unqualified status');need(state['sampler_schedule']==p['schedule'] and state['next_cycle']==33 and state['next_clean_sampler_index']==128 and state['progress']=={'cycle':32,'route':'predictor','microbatch':4,'phase':'predictor_cycle_complete','microbatches_backward_complete':4},'Exact fullcycle128 sampler boundary')
 changes=[];firstupdates=read(run/'outputs/training_cycle_0001.json')['predictor_actual_parameter_updates']
 for n,v in sorted(full.items()):
  need(isinstance(v,torch.Tensor) and v.device.type=='cpu' and v.dtype==torch.float32 and v.shape==base[n].shape and torch.isfinite(v).all().item(),'All497finiteCPUFP32 shapes');same=torch.equal(v,pf[n]);need(same if n in frozen else True,'Frozen383R1 tensor drift '+n)
  expected={'shape':list(v.shape),'dtype':str(v.dtype),'sha256':ts(v),'finite':True,'frozen_in_newphase':n in frozen};need(after['tensors'][n]==expected,'Actual at-save497hash receipt differs '+n);need(before['tensors'][n]=={**expected,'sha256':ts(pf[n])},'ActualR1 before497 receipt differs '+n)
  if n in active:
   need(lastupdate[n]==ts(v) and firstupdates[n]['before_sha256']==ts(pf[n]),'Actual114firstlast SHA chain anchor');diff=v.double()-pf[n].double();changes.append({'name':n,'changed':not same,'changed_coordinates':int(torch.count_nonzero(diff).item()),'coordinates':v.numel(),'maximum_absolute_delta':float(diff.abs().max().item()),'L2_delta':float(torch.linalg.vector_norm(diff).item()),'parent_sha256':ts(pf[n]),'terminal_sha256':ts(v),'cycles_zero_gradient':gradientzeros[n],'cycles_unchanged_parameter':updatezeros[n]})
 need(before['state_tensors']==after['state_tensors']==497 and before['frozen_tensors']==after['frozen_tensors']==383 and before['finite'] is after['finite'] is True,'Complete actual497 receipts');need(all(before[k] is True and after[k] is True for k in ('all_modules_eval','frozen_requires_grad_false','frozen_parameter_gradients_none')),'Eval/frozen flags');need(before['logical_sha256']==PARENT_LOGICAL and after['logical_sha256']==logical(full),'Beforeafter logical receipts');frozenhash=tree({n:full[n] for n in frozen});need(frozenhash==tree({n:pf[n] for n in frozen})==state['frozen383_sha256']==state['actual_frozen383_sha256']==result['frozen383_sha256']==pre['frozen383_sha256'],'Exact383frozen digest');priorhash=logical({n:base[n] for n in base if n.startswith(('quantize.','generator.'))});need(priorhash==state['frozen_state_sha256']==state['actual_prior_sha256']==parent['frozen_state_sha256'],'Original165prior exact')
 names=state['optimizer_parameter_names'];need(names==parent['optimizer_parameter_names'] and len(names['code'])==278 and len(names['fusion'])==54 and state['active_optimizer_parameter_names']==[n for n in names['code'] if predictor(n)] and len(state['active_optimizer_parameter_names'])==114,'Named278historical114active54fusion mapping');need(state['active_optimizer_slots_stored_only_in_full_code_optimizer'] is True and state['full_code_optimizer_is_historical_mixed_step_container_not_active278_optimizer'] is True and state['expected_mixed_steps']=={'inactive_encoder164':32,'active_predictor114':64,'inactive_fusion54':32},'Historical mixedsteps explicitlydeclared')
 optimizers={};inactive_encoder={}
 for route in ('code','fusion'):
  opt=state[route+'_optimizer'];old=parent[route+'_optimizer'];ns=names[route];need(len(opt['param_groups'])==len(old['param_groups'])==1 and opt['param_groups']==old['param_groups'],'Retained original Adam group metadata');g=opt['param_groups'][0];need(g['params']==list(range(len(ns))) and set(opt['state'])==set(old['state'])==set(g['params']) and g['lr']==1e-5 and tuple(g['betas'])==(.9,.99) and g['eps']==1e-8 and g['weight_decay']==0,'Exact named state slots/Adamhyperparams');rows=[]
  for i,n in enumerate(ns):
   entry=opt['state'][i];oldentry=old['state'][i];isactive=route=='code' and predictor(n);need(set(entry)==set(oldentry)=={'step','exp_avg','exp_avg_sq'},'Adam slot schema');step=entry['step'];need(step.device.type=='cpu' and step.dtype==torch.float32 and step.shape==() and torch.isfinite(step).item() and step.item()==(64 if isactive else 32) and oldentry['step'].item()==32,'Exact finite pername32/64 steps');momentchanged=[]
   for k in ('exp_avg','exp_avg_sq'):
    v=entry[k];need(v.device.type=='cpu' and v.dtype==torch.float32 and v.shape==full[n].shape and torch.isfinite(v).all().item(),'Adam moment finite shape dtype');momentchanged.append(not torch.equal(v,oldentry[k]));need(isactive or torch.equal(v,oldentry[k]),'Frozen164/54moment bytes changed')
   need((entry['exp_avg_sq']>=0).all().item(),'Nonnegative second moment');rows.append({'name':n,'step':step.item(),'active_predictor':isactive,'changed_moment_tensors':sum(momentchanged),'slot_sha256':tree(entry)})
   if route=='code' and not isactive:inactive_encoder[n]=entry
  if route=='fusion':need(tree(opt)==tree(old),'All frozen54fusionAdam exact')
  optimizers[route]={'rows':rows,'original_group_exact':True,'actual_active_predictor_slots':sum(x['active_predictor'] for x in rows),'state_sha256':tree(opt)}
 need(tree(inactive_encoder)==state['inactive_encoder_Adam_sha256']==result['inactive_encoder_Adam_sha256'] and optimizers['fusion']['state_sha256']==state['inactive_fusion_Adam_sha256']==result['inactive_fusion_Adam_sha256'],'InactiveAdam receipt hashes')
 def rng(v):
  c=v['torch_CPU_RNG'];g=v['CUDA_RNG'];py=v['python_random_state'];nr=v['numpy_random_state'];need(c.dtype==torch.uint8 and c.shape==(5056,) and c.device.type=='cpu' and len(g)==1 and g[0].dtype==torch.uint8 and g[0].shape==(16,) and g[0].device.type=='cpu','Exact L4 CPU5056 CUDA16 byte RNG');need(isinstance(py,tuple) and len(py)==3 and py[0]==3 and isinstance(py[1],tuple) and len(py[1])==625 and all(isinstance(x,int) and 0<=x<2**32 for x in py[1][:-1]) and 0<=py[1][-1]<=624 and (py[2] is None or math.isfinite(py[2])),'PythonRNGprimitive schema');need(set(nr)=={'algorithm','keys','position','has_gauss','cached_gaussian'} and nr['algorithm']=='MT19937' and len(nr['keys'])==624 and all(isinstance(x,int) and 0<=x<2**32 for x in nr['keys']) and 0<=nr['position']<=624 and nr['has_gauss'] in (0,1) and math.isfinite(nr['cached_gaussian']),'NumPyRNGprimitive schema');return {'CPU':ts(c),'CUDA':[ts(x) for x in g],'Python':js(py),'NumPy':js(nr)}
 prng=rng(parent);srng=rng(state);need(transition['complete'] is True and transition['expected_hashes']==transition['restored_hashes']==prng and transition['parent_terminal_sha256']==PARENT_SHA and transition['parent_RNG_restored'] is True and state['parent_RNG_restored'] is True and state['parent_optimizer_restored'] is True,'Actual parent RNG/Adam boundary metadata');need(state['parent_metadata']['RNG_boundary_hashes']=={'expected':prng,'restored':prng} and state['parent_metadata']['counts']==parent['counts'] and state['parent_metadata']['sampler_schedule']==parent['sampler_schedule'],'Parent retained boundary metadata exact')
 actualtensorbytes=sum(tensorbytes(state[k]) for k in ('trainable_state','failed_frozen_prior_state','code_optimizer','fusion_optimizer','partial_accumulated_gradients'));need(actualtensorbytes==pre['planned_tensor_payload_bytes']<p['contract']['terminal_bytes'],'Whole actual retained recovery payload size')
 need(compat['complete'] is True and compat['all48_before_update'] is True and compat['pass'] is True and compat['updates_before']==0 and len(compat['cases'])==48 and all(r['max_raw256_error']<=1e-6 and r['PNG_bytes_exact'] is True for r in compat['cases']),'All48initialR1 compatibility receipts');need(len(initial['fresh_app_parity'])==4 and initial['counts']=={'forward_slots':8,'backward_calls':0,'predictor_optimizer_steps':0,'empty_bypasses':0} and all(r['max_raw256_error']<=1e-6 and r['model_raw512_max_error']<=1e-6 and r['input512_exact'] and r['input_stride_exact'] and r['selected_codes_exact'] and r['optimizer_steps_before_parity']==0 for r in initial['fresh_app_parity']),'Fourfresh app parity receipts')
 phases=[]
 for rel in sorted(mf):
  if rel.startswith('outputs/phase_') and rel.endswith('_end.json'):
   b=read(evidence(rel));a0=read(evidence(rel[:-9]+'_start.json'));need(a0['phase']==b['phase'] and a0['start_seconds']==b['start_seconds'] and a0['complete'] is False and b['complete'] is True and 0<=b['start_seconds']<=b['end_seconds']<result['seconds'] and math.isclose(b['end_seconds']-b['start_seconds'],b['seconds'],abs_tol=1e-8),'Complete phase timing pair');phases.append(b)
 need(all(phases[i]['end_seconds']<=phases[i+1]['start_seconds'] for i in range(len(phases)-1)),'Sequential measured phases');phaseby={r['phase']:r for r in phases};save=phaseby['terminal_serialization'];commit=phaseby['terminal_hash_and_atomic_commit'];need(save['serialized_bytes']==t['bytes'] and save['write_callbacks']>0 and commit['sha256']==t['sha256'] and commit['bytes']==t['bytes'] and commit['ledger_total']<p['contract']['output_bytes'],'Actual bounded serialization andcommit');need(sha(ORIGINAL)==ORIGINAL_SHA and sha(PARENT)==PARENT_SHA and sha(terminal)==t['sha256'] and sha(transport_path)==a.transport_sha,'No protected source/return drift duringaudit')
 admission=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json');pins=admission['source_files']+[admission['candidate_manifest'],admission['checkpoint']]
 for row in pins:need(io_path(row['path']).stat().st_size==row['bytes'] and sha(row['path'])==row['sha256'],'R1 app/model protected pin exact')
 report={'format':'independent-predictor-CE-v1-state-Adam-RNG-logits-code-audit','complete':True,'independent':True,'UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(__file__),'protocol_sha256':PROTOCOL,'transport_audit_sha256':a.transport_sha,'checks_passed':CHECKS,'status':'VALID_COMPLETE_STATE_QUALITY_SCREEN_FAILED_NO_ADOPTION','supervisor':supervisor,'newphase_counts':expected_counts,'typed_forward_slots':typed,'actual_hook_counts':result['actual_model_forward_counts'],'model_state':{'full_tensors':497,'frozen_bitexact_R1':383,'active_predictor_tensors':114,'actual_changed_predictor_tensors':sum(r['changed'] for r in changes),'original_prior165_exact':True,'terminal_logical_sha256':logical(full),'frozen383_sha256':frozenhash,'actual_predictor_changes':changes},'optimizers':optimizers,'RNG':{'parent':prng,'terminal':srng,'equal_to_parent_by_family':{k:prng[k]==srng[k] for k in prng},'restoration_boundary_matches_actual_parent':True,'all_families_retained_schema_valid':True,'transient_VM_observation_limitation':True},'sampler':{'all128_distinct_balanced_selected_cases_once':True,'all64FIT_no16monitor_label_overlap':True,'next_cycle':33,'next_clean_sampler_index':128,'progress':state['progress'],'partial_accumulated_gradients_empty':True},'FIT_CE_replay':{'arrays':128,'mean_positions':256,'codebook':512,'float64_replay_of_saved_FP32_logits':True,'maximum_absolute_mean_CE_error':max(ceerrors),'FP32_vs_float64_reduction_absolute_tolerance':2e-6,'maximum_abs_scaled_loss_error':max(ceerrors)/4,'exact_divide_by4_once':True,'all_cycles':cycles,'gradient_clip_steps':sum(x>1 for x in clip),'min_preclip_L2':min(clip),'max_preclip_L2':max(clip),'important_limit':'Distinct one-pass FIT observations; first/last CE is not a matched learning curve. Full114 gradients were not saved, so Adam dynamics and autograd are not numerically re-executed locally.'},'fixed_monitor_whole256_code_agreement':{'scope':'All256positions, NOT masked-region accuracy, face identification or hidden identity confidence. Exactsame48fixedcases at4snapshots. Clean labels unavailable atdeployment.','cohort_summaries':code_summaries,'per_case':code_rows},'checkpoint_passive_admission':passive_reports,'terminal_save':{'actual_tensor_payload_bytes':actualtensorbytes,'serialized_bytes':t['bytes'],'serialization_seconds':save['seconds'],'hash_commit_seconds':commit['seconds'],'phases':phases,'committed_recoverable_state_valid':True,'earlier_images_are_not_recoverable_selected_states':True},'all22_current_R1_app_model_pins_exact':True,'numeric_screen_reported':result['numeric_screen'],'model_qualified':False,'goal_complete':False,'local_actions':{'CPU_weights_only_loads':3,'safe_globals_added':False,'arbitrary_pickle_load':False,'neural_models_or_forwards':0,'gradient_calls':0,'optimizer_objects_or_steps':0,'VM_calls':0,'returned_Python_executed':False,'candidate_packaging':0},'seconds':time.monotonic()-start}
 dest=ROOT/'research/completion_codeformer_predictor_ce_v1_state_review/independent_state_Adam_logits_audit.json';dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'complete':True,'report':pin(dest),'checks':CHECKS,'valid_recovery_state':True,'changed_predictor_tensors':report['model_state']['actual_changed_predictor_tensors'],'CE_replay128':True,'quality_screen_pass':False,'model_qualified':False,'seconds':report['seconds']},indent=2),flush=True)
if __name__=='__main__':main()
