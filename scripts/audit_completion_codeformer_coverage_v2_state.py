"""Independent V2 timeout/recovery audit; no model, gradients or VM operations.

Incomplete .partial is passively examined only, never rebuilt or deserialized.
The already audited complete R1 parent is passively admitted before CPU safe load.
"""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import hashlib,json,math,pickletools,time,zipfile
ROOT=Path(__file__).resolve().parents[1]
RET=ROOT/'outputs/completion_codeformer_coverage_v2_return'
RUN=RET/'cctv_completion_codeformer_coverage_v2_results'
EXPECTED='c838f4472cf87c138b2d518ef4f543bfcec3e8149ea7a5411b863397bf66859d'
PARENT=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results/outputs/terminal_training_state.pth'
PARENT_SHA='0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd'
PARENT_PROTOCOL='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
PARENT_LOGICAL='55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6'
ORIGINAL_SHA='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
SAFE_GLOBALS={'collections OrderedDict','torch._utils _rebuild_tensor_v2','torch FloatStorage','torch ByteStorage','torch LongStorage'}
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def pin(path):
 path=Path(path);need(path.is_file() and not path.is_symlink(),'Missing/linked evidence');return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def read(path):
 def reject(x):raise ValueError('Nonfinite JSON '+x)
 return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)
def passive_complete(path):
 report={'checkpoint':pin(path),'is_ZIP':zipfile.is_zipfile(path),'arbitrary_pickle_load':False,'safe_globals_added':False}
 need(report['is_ZIP'],'Complete parent is not ZIP')
 with zipfile.ZipFile(path) as z:
  members=z.infolist();names=[x.filename for x in members]
  need(len(names)==len(set(names)) and all(not x.is_dir() and not x.flag_bits&1 for x in members),'Checkpoint ZIP namespace')
  need(all(not PurePosixPath(x).is_absolute() and '..' not in PurePosixPath(x).parts and '\\' not in x and ':' not in x for x in names),'Checkpoint ZIP path invalid')
  need(z.testzip() is None,'Checkpoint CRC failed');metadata=[n for n in names if n.endswith('/data.pkl')];need(len(metadata)==1,'Expected one tensor pickle')
  raw=z.read(metadata[0]);need(len(raw)<=2*1024**2,'Unbounded pickle metadata');found=[];count=0
  for op,arg,pos in pickletools.genops(raw):
   count+=1;need(op.name!='STACK_GLOBAL','Unexpected dynamic pickle global')
   if op.name=='GLOBAL':need(arg in SAFE_GLOBALS,'Unexpected tensor pickle global '+str(arg));found.append(arg)
  report.update(ZIP_entries=len(members),uncompressed_bytes=sum(m.file_size for m in members),CRC_all_members_pass=True,pickle_bytes=len(raw),pickle_opcodes=count,pickle_globals=sorted(set(found)),allowlisted_globals_only=True)
 return report
def json_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 started=time.monotonic();transport=read(RET/'transport_audit.json');need(transport['complete'] is True,'Transport admission absent')
 protocol=read(RUN/'protocol.json');need(sha(RUN/'protocol.json')==EXPECTED,'Wrong V2 protocol')
 sealed=ROOT/'outputs/completion_codeformer_coverage_v2_packet/draft_003/cctv_completion_codeformer_coverage_v2_vm'
 need((RUN/'protocol.json').read_bytes()==(sealed/'protocol.json').read_bytes(),'Sealed protocol differs')
 manifest=read(RUN/'export_manifest.json');files={row['path']:row for row in manifest['files']};need(len(files)==623 and len(files)==len(manifest['files']),'Return namespace count differs')
 relevant=['supervisor.json','trainer.log','protocol.json','outputs/initial_state.json','outputs/RNG_transition.json','outputs/terminal_training_state.pth.partial','outputs/u0064/monitor/results.json','outputs/u0064/challenge/results.json','outputs/u0064/public_development/results.json']+[f'outputs/training_cycle_{i:04d}.json' for i in range(1,65)]+[f'outputs/screen_u{i:04d}.json' for i in (16,32,64)]
 pins=[]
 for relative in relevant:
  need(relative in files,'Missing progress artifact '+relative);actual=pin(RUN/relative);row=files[relative]
  need(actual['bytes']==row['bytes'] and actual['sha256']==row['sha256'],'Progress manifest binding differs '+relative);pins.append(actual)
 supervisor=read(RUN/'supervisor.json');initial=read(RUN/'outputs/initial_state.json');transition=read(RUN/'outputs/RNG_transition.json')
 need(supervisor['complete'] is True and supervisor['worker_exit']==124 and supervisor['actual_child_exit']==-9 and supervisor['timeout'] is True and supervisor['shutdown_mode']=='hard_cap_SIGKILL','Not observed supervisor hard timeout')
 need(supervisor['external_cap_seconds']==1800 and 1800<=supervisor['seconds']<1805 and supervisor['interruption'] is None,'Supervisor deadline receipt differs')
 need(supervisor['terminal_state_present'] is False and supervisor['terminal_partial_present'] is True and supervisor['stdout_cap_reached'] is False,'Terminal/log status differs')
 need(not(RUN/'outputs/results.json').exists() and not(RUN/'outputs/failure.json').exists() and not(RUN/'outputs/terminal_training_state.pth').exists(),'Unexpected committed/failure terminal file')
 partial=RUN/'outputs/terminal_training_state.pth.partial';partial_report={'file':pin(partial),'is_ZIP':zipfile.is_zipfile(partial),'safe_load_attempted':False,'reconstruction_attempted':False,'promotion_permitted':False,'status':'INCOMPLETE_SERIALIZATION_NOT_A_CHECKPOINT'}
 need(partial_report['is_ZIP'] is False,'Partial unexpectedly complete; requires separate full audit branch')
 try:
  with zipfile.ZipFile(partial) as z:z.infolist()
 except zipfile.BadZipFile as exc:partial_report['ZIP_open_error']=repr(exc)
 else:raise ValueError('Partial opening unexpectedly succeeded')
 with partial.open('rb') as stream:
  partial_report['prefix_hex']=stream.read(16).hex();stream.seek(max(0,partial.stat().st_size-65536));tail=stream.read()
 partial_report['central_directory_end_signature_in_last64KiB']=b'PK\x05\x06' in tail
 partial_report['ZIP64_end_signature_in_last64KiB']=b'PK\x06\x06' in tail
 need(partial.stat().st_size<701952165,'Partial not smaller than expected trained state')
 cycles=[];case_ids=[];clip={'code':[],'fusion':[]}
 groups={'code_gradient_groups':{'encoder':164,'position':1,'feature':2,'transformer':108,'prediction':3},'fusion_gradient_groups':{'32':18,'64':18,'128':18}}
 for i in range(1,65):
  row=read(RUN/f'outputs/training_cycle_{i:04d}.json');need(row['cycle']==i and row['microbatch_ids']==protocol['schedule'][i-1],'Cycle schedule differs')
  expected_counts={'forward_slots':244+8*i+32*sum(i>s for s in (16,32,64)),'backward_calls':8*i,'code_optimizer_steps':i,'fusion_optimizer_steps':i,'empty_bypasses':44+40*sum(i>s for s in (16,32,64))}
  need(row['counts']==expected_counts,'Cycle cumulative phase counts differ')
  ids=row['microbatch_ids'];need(len(ids)==len(set(ids))==4,'Duplicated cycle IDs');case_ids.extend(ids)
  for key,losskeys in [('code_losses',('CE','latent_MSE')),('fusion_losses',('mask_RGB512_L1','inner_boundary_RGB512_L1'))]:
   need([x['id'] for x in row[key]]==ids,'Loss IDs differ')
   need(all(all(math.isfinite(x[k]) and x[k]>=0 for k in losskeys) for x in row[key]),'Loss receipt invalid')
  for key,expected_groups in groups.items():
   need(set(row[key])==set(expected_groups),'Intended gradient groups changed')
   for name,n in expected_groups.items():
    r=row[key][name];need(r['tensors']==n and r['finite'] is True and math.isfinite(r['L2']) and r['L2']>=0 and r['nonzero']==(r['L2']>0),'Gradient receipt invalid')
   route='code' if key.startswith('code') else 'fusion';summed=math.sqrt(sum(r['L2']**2 for r in row[key].values()));norm=row[route+'_preclip_L2'];need(math.isfinite(norm) and math.isclose(summed,norm,rel_tol=2e-5,abs_tol=1e-7),'Gradient aggregate differs');clip[route].append(norm)
  need(0<row['seconds']<1800 and (not cycles or cycles[-1]['seconds']<row['seconds']),'Cycle timing invalid')
  cycles.append({'cycle':i,'seconds':row['seconds'],'counts':row['counts'],'microbatch_ids':ids,'code_preclip_L2':row['code_preclip_L2'],'fusion_preclip_L2':row['fusion_preclip_L2']})
 need(len(case_ids)==len(set(case_ids))==256 and set(case_ids)=={r['id'] for r in protocol['fit_cases']},'256case coverage differs')
 terminal_summary=read(RUN/'outputs/u0064/public_development/results.json');counts=terminal_summary['counts'];need(counts=={'forward_slots':896,'backward_calls':512,'code_optimizer_steps':64,'fusion_optimizer_steps':64,'empty_bypasses':168},'Final evaluated phase counts differ')
 screens={str(i):read(RUN/f'outputs/screen_u{i:04d}.json') for i in (16,32,64)}
 need(all(row['catastrophic_regression_stop'] is False for row in screens.values()),'Observed quality catastrophe incompatible with64cycles')
 log=(RUN/'trainer.log').read_text(encoding='utf-8');printed=[json.loads(line) for line in log.splitlines() if line.startswith('{')]
 need(len(printed)==64 and printed[-1]['cycle']==64 and printed[-1]['counts']=={'forward_slots':852,'backward_calls':512,'code_optimizer_steps':64,'fusion_optimizer_steps':64,'empty_bypasses':164},'Trainer last printed cycle differs')
 need(printed[-1]['seconds']<1000,'Last log did not complete atobserved923s')
 need(initial['initial_R1_logical_state_sha256']==PARENT_LOGICAL and initial['parent_training_state_sha256']==PARENT_SHA and initial['checkpoint_sha256']==ORIGINAL_SHA,'Initial R1 provenance differs')
 need(initial['counts']=={'forward_slots':168,'backward_calls':0,'code_optimizer_steps':0,'fusion_optimizer_steps':0,'empty_bypasses':0},'Initial before-learning counts differ')
 need(transition['complete'] is True and transition['parent_terminal_sha256']==PARENT_SHA and transition['parent_RNG_restored'] is True and transition['expected_hashes']==transition['restored_hashes'],'RNG receipt differs')
 # Passive review and safe CPU load only of complete, previously retained R1 parent.
 need(pin(PARENT)['sha256']==PARENT_SHA and PARENT.stat().st_size==701952165,'Retained complete R1 parent changed')
 parent_passive=passive_complete(PARENT)
 import torch
 torch.set_num_threads(1)
 parent=torch.load(PARENT,map_location='cpu',weights_only=True)
 need(parent['protocol_sha256']==PARENT_PROTOCOL and parent['logical_state_sha256']==PARENT_LOGICAL and parent['base_checkpoint_sha256']==ORIGINAL_SHA,'Complete parent provenance differs')
 need(parent['cycle']==parent['completed_paired_cycles']==32 and parent['counts']=={'forward_slots':528,'backward_calls':256,'code_optimizer_steps':32,'fusion_optimizer_steps':32,'empty_bypasses':168} and parent['partial_accumulated_gradients']=={},'Parent terminal progress differs')
 need(len(parent['trainable_state'])==332 and sum(x.numel() for x in parent['trainable_state'].values())==58459840,'Parent trainable partition differs')
 need(all(x.dtype==torch.float32 and x.device.type=='cpu' and torch.isfinite(x).all().item() for x in parent['trainable_state'].values()),'Parent trainable tensor invalid')
 def tensor_sha(x):return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()
 actual_parent_rng={'CPU':tensor_sha(parent['torch_CPU_RNG']),'CUDA':[tensor_sha(x) for x in parent['CUDA_RNG']],'Python':json_digest(parent['python_random_state']),'NumPy':json_digest(parent['numpy_random_state'])}
 need(actual_parent_rng==transition['expected_hashes']==transition['restored_hashes'],'Returned RNG boundary hashes do not match exact saved parent')
 optimizers={}
 for route,n in [('code',278),('fusion',54)]:
  optimizer=parent[route+'_optimizer'];names=parent['optimizer_parameter_names'][route];need(len(names)==len(set(names))==n and len(optimizer['param_groups'])==1 and len(optimizer['state'])==n,'Parent namedAdam mappings differ')
  group=optimizer['param_groups'][0];need(len(group['params'])==n and group['params']==list(range(n)) and group['lr']==1e-5 and tuple(group['betas'])==(.9,.99) and group['eps']==1e-8 and group['weight_decay']==0,'Parent Adam options differ')
  numel=0
  for index,name in enumerate(names):
   entry=optimizer['state'][group['params'][index]];ref=parent['trainable_state'][name]
   need(set(entry)=={'step','exp_avg','exp_avg_sq'} and float(entry['step'].item())==32,'Parent Adam step/schema differs')
   for key in ('exp_avg','exp_avg_sq'):need(entry[key].shape==ref.shape and entry[key].dtype==ref.dtype==torch.float32 and torch.isfinite(entry[key]).all().item(),'Parent Adam moments invalid')
   need((entry['exp_avg_sq']>=0).all().item(),'Negative Adam second moment');numel+=ref.numel()
  expected={'parameter_tensors':n,'step_before':32,'named_order_exact':True,'moments_exact':True,'lr':1e-5}
  need(initial['parent_optimizer_admission'][route]==expected,'VM parent optimizer admission differs')
  optimizers[route]={'tensors':n,'coordinates':numel,'all_parent_steps':32,'parent_mapping_moments_finite':True,'VM_named_order_and_exact_moment_restore_receipt':initial['parent_optimizer_admission'][route],'terminal_V2_steps96_not_saved_or_verified':True}
 need(sha(PARENT)==PARENT_SHA,'R1 parent changed duringaudit')
 full_output_seen=all((RUN/f'outputs/u0064/{group}/results.json').is_file() for group in ('monitor','challenge','public_development'))
 result={'format':'completion-coverage-v2-independent-timeout-state-audit-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'transport_audit_sha256':sha(RET/'transport_audit.json'),'protocol_sha256':EXPECTED,'progress_artifacts':pins,'supervisor':supervisor,'partial_state':partial_report,'committed_terminal_available':False,'valid_new_recovery_checkpoint':False,'new_state_load_attempted':False,'new_state_reconstruction_attempted':False,'new_candidate_packaged':False,'training_phase_receipts':{'all64cycles':cycles,'all256cases_once_perroute':True,'newphase_optimizer_updates_reported':128,'newphase_backward_calls_reported':512,'cumulative_optimizer_updates_reported':192,'cumulative_backward_calls_reported':768,'phasefinal_counts':counts,'phasefinal_typed_slots_reported':896,'cumulative_typed_slots_reported':1424,'finalAdam_step96_inferred_fromreceipt_not_observed_invalidpartial':True,'all_planned_eval_group_summaries_present':full_output_seen,'last_printed_cycle_seconds':printed[-1]['seconds'],'last_printed_counts':printed[-1]['counts'],'logical_terminal_candidate_unverified':True},'gradient_clipping_receipts':{route:{'steps':len(values),'preclip_min':min(values),'preclip_max':max(values),'clipped_steps':sum(x>1 for x in values)} for route,values in clip.items()},'initial_state_receipt':initial,'parent_passive_review':parent_passive,'parent_safe_loading':{'torch':str(torch.__version__),'weights_only':True,'map_location':'cpu','model_class_imported':False,'model_instantiated':False,'parent_only':True},'parent_optimizers':optimizers,'RNG_boundary':{'actual_parent_hashes':actual_parent_rng,'VM_expected_equal_saved_parent':True,'VM_restored_hashes_equal_saved_parent':True,'limitation':'Returned receipts plus exact savedparent hashes establish reported restoration, not independent observation of transient VMmemory.'},'termination_findings':{'training_loop_completed_beforetimeout':True,'allplannedforwardslots_reported_complete':True,'supervisor_hardkill_duringterminalserialization':True,'serialization_incompleteZIP':True,'no_python_traceback_or_failure_receipt_returned':True,'source_call_order':'finaleval->allasset/countguards->save_state(torch.save to exclusive.partial)->atomicrename->results.json. Partialcreated+allfinalsummaries butnoZIPcentraldirectory/commit confirms interruptedsave path.','underlying_performance_cause':'Not established by return alone. Per-write OutputBudget repeatedly resolves/categorizes all trackedpaths; this is a concrete source-level complexity concern, not yet a measured VM bottleneck. No evidence authorizes raising time cap/repeatingrecipe unchanged.'},'quality_status':'Savedimages/metrics may be audited, but no usable final learned state exists and no new app model may be promoted from this return.','retention':'Keep completeR1parent/candidate, actualV2archive/alloutputs/incompletepartial and gate failures; do not rename.partial to.pth or silentlyrebuild fromfragment.','local_actions':{'CPUweights_only_parent_loads':1,'V2partial_safe_loads':0,'model_loads':0,'model_instantiations':0,'neural_forwards':0,'gradient_calls':0,'optimizer_operations':0,'VM_calls':0,'returned_code_executed':False},'model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-started}
 target=RET/'independent_state_audit.json'
 with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps({'complete':True,'audit':str(target),'sha256':sha(target),'valid_new_recovery_checkpoint':False,'partial_ZIP':False,'actual_training_cycles_reported':64,'newupdates_reported':128,'RNG_parent_hashes_exact':True,'parentAdamsteps':32,'seconds':result['seconds']},indent=2))
if __name__=='__main__':main()