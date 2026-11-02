"""Independent V4 recovery-state audit. Passive ZIP admission then weights_only CPU reads.
No returned Python is executed; no neural model, gradient, optimizer or VM operation.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib, json, math, pickletools, time, zipfile
ROOT=Path(__file__).resolve().parents[1]
RET=ROOT/'outputs/completion_codeformer_delivered_objective_v4_return'
RUN=RET/'cctv_completion_codeformer_delivered_objective_v4_results'
PROTOCOL='8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d'
TRANSPORT='187be2e99a5439528288ccbdf3d196752b14dd3e8262429046ec9b1958088570'
TERMINAL_SHA='24e92d8c937c93d942e23b96ebcc6be66390ce0c8e77391f894404e3689e8850'
TERMINAL_BYTES=701936805
TERMINAL_LOGICAL='d26bae8456cb94633ecc3a24d47540a42b1b158b687352903cf77cd9b424254a'
ORIGINAL=Path(r'C:\xampp\htdocs\YEAR 4\Testing\checkpoints\codeformer_inpainting.pth')
ORIGINAL_SHA='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
ORIGINAL_LOGICAL='586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f'
PARENT=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results/outputs/terminal_training_state.pth'
PARENT_SHA='0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd'
PARENT_PROTOCOL='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
PARENT_LOGICAL='55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6'
SAFE_GLOBALS={'collections OrderedDict','torch._utils _rebuild_tensor_v2','torch FloatStorage','torch ByteStorage','torch LongStorage'}
CHECKS=0

def need(ok,message):
 global CHECKS
 CHECKS+=1
 if not ok: raise ValueError(message)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()

def pin(path):
 path=Path(path);need(path.is_file() and not path.is_symlink(),'Missing or linked file '+str(path))
 return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}

def read(path):
 def reject(v):raise ValueError('Nonfinite JSON '+v)
 return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)

def json_digest(v):
 return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def passive(path,expected):
 report={'file':pin(path),'safe_globals_added':False,'arbitrary_pickle_load':False}
 need(report['file']['sha256']==expected,'Checkpoint hash differs '+str(path))
 need(zipfile.is_zipfile(path),'Checkpoint not complete ZIP '+str(path))
 with zipfile.ZipFile(path) as z:
  members=z.infolist();names=[m.filename for m in members]
  need(len(names)==len(set(names)) and all(not m.is_dir() and not m.flag_bits&1 for m in members),'ZIP members unsafe')
  need(all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and '\\' not in n and ':' not in n for n in names),'ZIP path unsafe')
  need(all('/code/' not in n and not n.endswith(('.py','.pyc','.so','.dll')) for n in names),'Executable ZIP payload')
  need(z.testzip() is None,'ZIP CRC mismatch')
  metadata=[n for n in names if n.endswith('/data.pkl')];need(len(metadata)==1,'Tensor metadata count differs')
  raw=z.read(metadata[0]);need(len(raw)<2*1024**2,'Unbounded tensor metadata')
  globals_seen=[];opcodes=0
  for op,arg,pos in pickletools.genops(raw):
   opcodes+=1;need(op.name!='STACK_GLOBAL','Dynamic pickle global refused')
   if op.name=='GLOBAL':need(arg in SAFE_GLOBALS,'Unexpected pickle global '+str(arg));globals_seen.append(arg)
  report.update(complete_ZIP=True,CRC_all_members_pass=True,ZIP_entries=len(members),uncompressed_bytes=sum(m.file_size for m in members),pickle_bytes=len(raw),pickle_opcodes=opcodes,pickle_globals=sorted(set(globals_seen)),allowlisted_globals_only=True)
 return report

def main():
 started=time.monotonic()
 transport=read(RET/'transport_audit.json');need(sha(RET/'transport_audit.json')==TRANSPORT,'Transport receipt changed')
 need(transport['complete'] is True and transport['all_payload_manifest_hashes_exact'] is True and transport['closed_namespace_and_safe_paths'] is True,'Transport not admitted')
 need(transport['manifest_payload_files']==583 and transport['regular_files_including_manifest']==584 and transport['worker_exit']==2 and transport['terminal_state_committed'] is True and transport['partial_files_retained']==[],'Transport state/counts differ')
 protocol=read(RUN/'protocol.json');need(sha(RUN/'protocol.json')==PROTOCOL,'V4 protocol differs')
 sealed=ROOT/'outputs/completion_codeformer_delivered_objective_v4_packet/draft_001/cctv_completion_codeformer_delivered_objective_v4_vm/protocol.json'
 need((RUN/'protocol.json').read_bytes()==sealed.read_bytes(),'Sealed controlling protocol differs')
 manifest=read(RUN/'export_manifest.json');rows=manifest['files'];files={r['path']:r for r in rows}
 need(len(files)==len(rows)==583 and manifest['protocol_sha256']==PROTOCOL and manifest['worker_exit']==2 and manifest['worker_timeout'] is False and manifest['terminal_state_committed'] is True and manifest['partial_files_retained']==[],'Manifest terminal binding differs')
 relevant=['protocol.json','supervisor.json','trainer.log','outputs/results.json','outputs/pre_save_receipt.json','outputs/initial_state.json','outputs/RNG_transition.json','outputs/historical_R1_compatibility.json','outputs/objective_contract.json','outputs/terminal_training_state.pth']+[f'outputs/training_cycle_{i:04d}.json' for i in range(1,65)]+[f'outputs/screen_u{i:04d}.json' for i in (16,32,64)]+['outputs/'+p.name for p in (RUN/'outputs').glob('phase_*.json')]
 pins=[]
 for rel in relevant:
  need(rel in files,'Missing manifest evidence '+rel)
  actual=pin(RUN/rel);entry=files[rel]
  need(actual['bytes']==entry['bytes'] and actual['sha256']==entry['sha256'],'Evidence hash differs '+rel);pins.append(actual)
 need(not list(RUN.rglob('*.partial')),'Unexpected partial state returned')
 result=read(RUN/'outputs/results.json');supervisor=read(RUN/'supervisor.json');pre=read(RUN/'outputs/pre_save_receipt.json');initial=read(RUN/'outputs/initial_state.json');transition=read(RUN/'outputs/RNG_transition.json');compat=read(RUN/'outputs/historical_R1_compatibility.json')
 objective=read(RUN/'outputs/objective_contract.json')
 need(objective=={'objective':'app_float256_mask_RGB_L1_only_v1','loss_source':'exact pinned clean256, existing clean[teacher_key][0]','delivery':'clamp raw512 RGB then bilinear align_corners=False to256, active-index compose source','boundary_loss_added':False,'all_boundary_acceptance_gates_retained':True,'microbatch_divisor_once':4,'PNG_rounding_or_STE_in_training':False,'clipping_metadata_is_detached_diagnostic_only':True,'R1_incumbent_and_failed_V2_V3_preserved':True},'Actual objective contract differs')
 need(result['objective']=='app_float256_mask_RGB_L1_only_v1' and result['boundary_loss_added'] is False and result['PNG_rounding_or_STE_in_training'] is False,'Final objective receipts differ')
 contract=protocol['contract']
 need(contract['objective_resolution']==256 and contract['clean_target_index']==0 and contract['microbatch_divisor_once']==4 and contract['additive_boundary_loss'] is False and contract['training_PNG_rounding'] is False and contract['saturation_gradient_surrogate'] is False,'Sealed V4 objective differs')
 need(contract['selection_min_macro_PNG_hole_gain']==.01 and contract['nonworse_tolerance']==1e-12 and contract['catastrophic_regression_fraction']==.1 and contract['worker_seconds']==1800 and contract['cooperative_stop_seconds']==1680 and contract['terminal_bytes']==1073741824 and contract['output_bytes']==2304*1024**2,'Current quality/resource gates changed')
 need(result['complete'] is True and result['protocol_sha256']==PROTOCOL and result['cycles_completed_newphase']==64 and result['cycles_completed_cumulative']==96 and result['stop_reason'] is None,'Finite completion receipt differs')
 need(supervisor['complete'] is True and supervisor['worker_exit']==supervisor['actual_child_exit']==2 and supervisor['timeout'] is False and supervisor['interruption'] is None and supervisor['shutdown_mode'] is None,'Worker did not stop normally on quality')
 need(supervisor['external_cap_seconds']==1800 and 0<result['seconds']<supervisor['seconds']<1800 and supervisor['terminal_state_present'] is True and supervisor['terminal_partial_present'] is False and supervisor['stdout_cap_reached'] is False,'Finite supervisor limits/status differ')
 expected_counts={'forward_slots':480,'backward_calls':256,'code_optimizer_steps':0,'fusion_optimizer_steps':64,'empty_bypasses':168}
 cumulative={'backward_calls':512,'code_optimizer_steps':32,'fusion_optimizer_steps':96,'optimizer_steps_total':128,'forward_slots':1008,'actual_whole_model_forward_calls':928}
 need(result['counts_newphase']==expected_counts and result['counts_cumulative']==cumulative,'Final actual counts differ')
 need(result['typed_forward_slots']=={'parity':8,'fit_RGB':256,'monitor':128,'challenge':32,'public_development':56} and result['actual_whole_model_forward_calls']==480 and result['teacher_neural_calls']==0,'Typed/whole/teacher counts differ')
 need(result['model_qualified'] is False and result['automatic_app_promotion'] is False and result['numeric_screen']['prospective_numeric_screen_pass'] is False,'Quality failure must remain unqualified')
 need(result['initial_R1_logical_state_sha256']==PARENT_LOGICAL and result['original_base_logical_state_sha256']==ORIGINAL_LOGICAL and result['final_logical_state_sha256']==TERMINAL_LOGICAL,'Reported logical ancestry differs')
 terminal=RUN/'outputs/terminal_training_state.pth';t=result['terminal_training_state']
 need(t['path']==terminal.name and t['bytes']==TERMINAL_BYTES and t['sha256']==TERMINAL_SHA and t['logical_state_sha256']==TERMINAL_LOGICAL and t['cycle']==64 and t['retained_failure_gates']==[],'Terminal receipt differs')
 need(terminal.stat().st_size==TERMINAL_BYTES and pre['cycle']==64 and pre['counts']==expected_counts and pre['logical_state_sha256']==TERMINAL_LOGICAL and pre['serialization_started'] is False and pre['retained_failure_gates']==[],'Pre-save receipt differs')
 need(pre['planned_tensor_payload_bytes']<1073741824 and pre['free_disk_bytes']>=2147483648 and 0<pre['remaining_hard_seconds']<1800 and pre['ledger_totals']['terminal']==0,'Pre-save bounded resources differ')
 # Per-cycle counts are recorded BEFORE same-cycle monitoring/empty controls.
 cycles=[];case_ids=[];clip=[]
 for i in range(1,65):
  row=read(RUN/f'outputs/training_cycle_{i:04d}.json');ids=row['microbatch_ids']
  need(row['cycle']==i and ids==protocol['schedule'][i-1] and len(ids)==len(set(ids))==4,'Cycle schedule differs')
  case_ids.extend(ids)
  expected={'forward_slots':84+4*i+32*sum(i>s for s in (16,32,64)),'backward_calls':4*i,'code_optimizer_steps':0,'fusion_optimizer_steps':i,'empty_bypasses':44+40*sum(i>s for s in (16,32,64))}
  need(row['counts']==expected,'Cycle operation counts differ')
  need(row['code_route_frozen'] is True and row['code_losses']==[] and row['code_gradient_groups']=={} and row['code_preclip_L2'] is None,'Frozen code route receipt differs')
  need([x['id'] for x in row['fusion_losses']]==ids and row['objective']=='app_float256_mask_RGB_L1_only_v1','Fusion objective/IDs differ')
  byid={case['id']:case for case in protocol['fit_cases']}
  for lossrow in row['fusion_losses']:
   case=byid[lossrow['id']]
   need(lossrow['style']==case['style'] and lossrow['clean256']==case['clean256'] and lossrow['teacher_key']==case['teacher_key'] and lossrow['objective']=='app_float256_mask_RGB_L1_only_v1','Exact case clean256 target differs')
   need(math.isfinite(lossrow['mask_RGB256_L1']) and lossrow['mask_RGB256_L1']>=0 and lossrow['scaled_microbatch_loss']==lossrow['mask_RGB256_L1']/4 and lossrow['microbatch_divisor']==4 and lossrow['separate_boundary_loss_added'] is False and lossrow['PNG_rounding_or_STE_in_training'] is False,'Actual delivered256 loss scaling/no seam/PNG/STE differs')
   diagnostic=lossrow['clipping'];need(diagnostic['raw512_hole_channel_count']==case['mask_pixels']*4*3 and diagnostic['clamp_zero_gradient_for_strictly_outside_channels'] is True and diagnostic['surrogate_gradient_used'] is False,'Clipping support/surrogate receipt differs')
   for field in ('below0_channel_fraction_hole512','above1_channel_fraction_hole512','clipped_channel_fraction_hole512','clipped_pixel_fraction_hole512'):need(math.isfinite(diagnostic[field]) and 0<=diagnostic[field]<=1,'Invalid clipping fraction')
   need(math.isclose(diagnostic['below0_channel_fraction_hole512']+diagnostic['above1_channel_fraction_hole512'],diagnostic['clipped_channel_fraction_hole512'],rel_tol=1e-5,abs_tol=1e-7),'Clipping fraction sum differs')
  need(row['fusion_gradient_clipping_applied']==(row['fusion_preclip_L2']>1),'Gradient clipping applied receipt differs')
  need(set(row['fusion_gradient_groups'])=={'32','64','128'},'Fusion gradient groups differ')
  for g in row['fusion_gradient_groups'].values():need(g['tensors']==18 and g['finite'] is True and math.isfinite(g['L2']) and g['L2']>=0 and g['nonzero']==(g['L2']>0),'Fusion gradient receipt invalid')
  aggregate=math.sqrt(sum(g['L2']**2 for g in row['fusion_gradient_groups'].values()));norm=row['fusion_preclip_L2']
  need(math.isfinite(norm) and math.isclose(aggregate,norm,rel_tol=2e-5,abs_tol=1e-7),'Fusion clipping norm differs');clip.append(norm)
  need(0<row['seconds']<result['seconds'] and (not cycles or cycles[-1]['seconds']<row['seconds']),'Cycle timings invalid')
  cycles.append({'cycle':i,'seconds':row['seconds'],'counts':row['counts'],'microbatch_ids':ids,'fusion_preclip_L2':norm})
 need(len(case_ids)==len(set(case_ids))==256 and set(case_ids)=={r['id'] for r in protocol['fit_cases']},'FIT cases not covered exactly once')
 screens={str(i):read(RUN/f'outputs/screen_u{i:04d}.json') for i in (16,32,64)}
 need(all(not r['catastrophic_regression_stop'] for r in screens.values()),'Early stop contradictory to all64 cycles')
 printed=[json.loads(line) for line in (RUN/'trainer.log').read_text().splitlines() if line.startswith('{')]
 need(len(printed)==64 and [r['cycle'] for r in printed]==list(range(1,65)) and printed[-1]['counts']=={'forward_slots':436,'backward_calls':256,'code_optimizer_steps':0,'fusion_optimizer_steps':64,'empty_bypasses':164},'Trainer completion log differs')
 need(initial['checkpoint_sha256']==ORIGINAL_SHA and initial['parent_training_state_sha256']==PARENT_SHA and initial['initial_R1_logical_state_sha256']==PARENT_LOGICAL and initial['original_base_logical_state_sha256']==ORIGINAL_LOGICAL,'Initial ancestry differs')
 need(initial['counts']=={'forward_slots':8,'backward_calls':0,'code_optimizer_steps':0,'fusion_optimizer_steps':0,'empty_bypasses':0} and len(initial['fresh_app_parity'])==4 and initial['teacher_neural_calls']==0,'Initial pre-learning counts differ')
 for i,r in enumerate(initial['fresh_app_parity'],1):
  need(r['max_raw256_error']<=1e-6 and r['model_raw512_max_error']<=1e-6 and r['input512_exact'] is True and r['input_stride_exact'] is True and r['selected_codes_exact'] is True and r['optimizer_steps_before_parity']==0,'Fresh R1 parity failed')
  need(r['counts']=={'forward_slots':2*i,'backward_calls':0,'code_optimizer_steps':0,'fusion_optimizer_steps':0,'empty_bypasses':0},'Parity counting differs')
 need(compat['complete'] is True and compat['all48_before_update'] is True and compat['raw_tolerance']==1e-6 and compat['updates_before']==0 and compat['pass'] is True and len(compat['cases'])==48,'Historical compatibility pre-update admission differs')
 expected_compat={(kind,r['id']) for kind in ('monitor','challenge') for r in protocol['historical_control']['initial_rows'][kind]}
 need({(r['kind'],r['id']) for r in compat['cases']}==expected_compat and all(r['max_raw256_error']<=1e-6 and r['PNG_bytes_exact'] is True for r in compat['cases']),'Historical48 recorded outputs differ')
 # All three archives are passively admitted before Torch import or deserialization.
 passive_reports={'original':passive(ORIGINAL,ORIGINAL_SHA),'R1_parent':passive(PARENT,PARENT_SHA),'V4_terminal':passive(terminal,TERMINAL_SHA)}
 print(json.dumps({'phase':'passive_ZIP_CRC_pickle_allowlist_passed','checkpoints':3,'seconds':time.monotonic()-started}),flush=True)
 import torch
 torch.set_num_threads(1)
 original=torch.load(ORIGINAL,map_location='cpu',weights_only=True)
 parent=torch.load(PARENT,map_location='cpu',weights_only=True)
 state=torch.load(terminal,map_location='cpu',weights_only=True)
 def tensor_sha(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
 def logical(mapping):
  h=hashlib.sha256()
  for name,v in sorted(mapping.items()):
   a=v.detach().cpu().contiguous();h.update(name.encode());h.update(str(tuple(a.shape)).encode());h.update(str(a.dtype).encode());h.update(a.numpy().tobytes())
  return h.hexdigest()
 def tree_digest(v):
  h=hashlib.sha256()
  def visit(x):
   if isinstance(x,torch.Tensor):
    a=x.detach().cpu().contiguous();need(torch.isfinite(a).all().item(),'Nonfinite recovery tensor');h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
   elif isinstance(x,dict):
    for key in sorted(x,key=lambda k:(str(type(k)),str(k))):h.update(repr(key).encode());visit(x[key])
   elif isinstance(x,(tuple,list)):
    h.update(type(x).__name__.encode())
    for item in x:visit(item)
   else:h.update(repr(x).encode())
  visit(v);return h.hexdigest()
 def tensor_bytes(v):
  if isinstance(v,torch.Tensor):return v.numel()*v.element_size()
  if isinstance(v,dict):return sum(tensor_bytes(x) for x in v.values())
  if isinstance(v,(list,tuple)):return sum(tensor_bytes(x) for x in v)
  return 0
 need(set(original)=={'params_ema'} and len(original['params_ema'])==497,'Original full state schema differs')
 base=original['params_ema'];delta_names={n for n in base if not n.startswith(('quantize.','generator.'))};fusion_names={n for n in base if n.startswith('fuse_convs_dict.')};frozen_names=set(base)-fusion_names;prior_names=set(base)-delta_names
 need(len(delta_names)==332 and len(fusion_names)==54 and len(frozen_names)==443 and len(prior_names)==165,'Reconstruction partitions differ')
 for label,d in [('base',base),('parent',parent['trainable_state']),('new',state['trainable_state'])]:
  if label!='base':need(set(d)==delta_names,'Delta names differ '+label)
  for n,v in d.items():need(isinstance(v,torch.Tensor) and v.device.type=='cpu' and v.dtype==torch.float32 and v.shape==base[n].shape and torch.isfinite(v).all().item(),'Invalid CPU tensor '+label+'/'+n)
 need(logical(base)==ORIGINAL_LOGICAL,'Original model logical bytes differ')
 parent_full={**base,**parent['trainable_state']};full={**base,**state['trainable_state']}
 need(logical(parent_full)==PARENT_LOGICAL and parent['logical_state_sha256']==PARENT_LOGICAL,'Full R1 reconstruction differs')
 need(logical(full)==TERMINAL_LOGICAL==state['logical_state_sha256'],'Full V4 reconstruction differs')
 need(parent['protocol_sha256']==PARENT_PROTOCOL and parent['base_checkpoint_sha256']==ORIGINAL_SHA and parent['cycle']==parent['completed_paired_cycles']==32 and parent['reason']=='completed_finite_pilot' and parent['partial_accumulated_gradients']=={},'Parent original provenance/state incomplete')
 need(state['format']=='completion-codeformer-delivered-objective-v4-trainable-resume-v1' and state['protocol_sha256']==PROTOCOL and state['base_checkpoint_sha256']==ORIGINAL_SHA and state['parent_training_state_sha256']==PARENT_SHA and state['parent_protocol_sha256']==PARENT_PROTOCOL and state['parent_initial_logical_sha256']==PARENT_LOGICAL,'New recovery provenance differs')
 need(state['parent_optimizer_restored'] is True and state['parent_RNG_restored'] is True and state['counts']==expected_counts and state['cumulative_counts']==cumulative and state['contract']==protocol['contract'],'New recovery contract/counts differ')
 need(state['cycle']==state['phase_completed_paired_cycles']==64 and state['completed_paired_cycles']==state['cumulative_completed_paired_cycles']==96 and state['phase_name']=='delivered_objective_v4' and state['reason']=='completed_finite_pilot','New recovery phase differs')
 need(state['sampler_schedule']==protocol['schedule'] and state['next_cycle']==65 and state['next_clean_sampler_index']==256,'Sampler boundary differs')
 need(state['progress']=={'phase':'fusion_cycle_complete','cycle':64,'route':'fusion','microbatch':4,'microbatches_backward_complete':4},'Full-cycle progress boundary differs')
 need(state['partial_accumulated_gradients']=={} and state['failed_frozen_prior_state']=={} and state['retained_failure_gates']==[],'Unexpected partial/frozen failure state')
 need(state['resume_progress_requires_new_manual_audit'] is True and state['automatic_resume_permitted'] is False and state['model_qualified'] is False and state['state_is_qualified'] is False,'Recovery status must not imply qualification/resume')
 changes=[]
 for n in sorted(base):
  a=parent_full[n];b=full[n];same=torch.equal(a,b)
  if n in frozen_names:need(same,'Frozen R1 tensor changed '+n)
  else:
   need(not same,'Expected active fusion tensor unchanged '+n)
   diff=b.double()-a.double();changes.append({'name':n,'shape':list(b.shape),'coordinates':b.numel(),'changed_coordinates':int(torch.count_nonzero(diff).item()),'maximum_absolute_delta':float(diff.abs().max().item()),'L2_delta':float(torch.linalg.vector_norm(diff).item()),'parent_sha256':tensor_sha(a),'terminal_sha256':tensor_sha(b)})
 frozen_hash=tree_digest({n:full[n] for n in frozen_names});prior_hash=logical({n:base[n] for n in prior_names})
 need(frozen_hash==tree_digest({n:parent_full[n] for n in frozen_names})==state['frozen443_sha256']==state['actual_frozen443_sha256']==result['frozen443_sha256']==pre['frozen443_sha256'],'Frozen443 digest differs')
 need(prior_hash==parent['frozen_state_sha256']==state['frozen_state_sha256']==state['actual_prior_sha256']==initial['frozen_generator_codebook_sha256'],'Original165 prior digest differs')
 optimizers={}
 for route,n,expected_step in [('code',278,32),('fusion',54,96)]:
  names=state['optimizer_parameter_names'][route];oldnames=parent['optimizer_parameter_names'][route];opt=state[route+'_optimizer'];old=parent[route+'_optimizer']
  need(names==oldnames and len(names)==len(set(names))==n and set(names)==(delta_names-fusion_names if route=='code' else fusion_names),'Named Adam order differs '+route)
  need(len(opt['param_groups'])==len(old['param_groups'])==1 and opt['param_groups']==old['param_groups'],'Adam group configuration changed '+route)
  g=opt['param_groups'][0];need(g['params']==list(range(n)) and g['lr']==1e-5 and tuple(g['betas'])==(.9,.99) and g['eps']==1e-8 and g['weight_decay']==0,'Adam options differ '+route)
  need(set(opt['state'])==set(range(n)) and set(old['state'])==set(range(n)),'Adam state map differs '+route)
  moment_changes=0
  for index,name in enumerate(names):
   entry=opt['state'][index];before=old['state'][index];ref=full[name]
   need(set(entry)==set(before)=={'step','exp_avg','exp_avg_sq'},'Adam entry schema differs')
   need(entry['step'].device.type=='cpu' and entry['step'].numel()==1 and math.isfinite(float(entry['step'].item())) and float(entry['step'].item())==expected_step and float(before['step'].item())==32,'Adam step differs '+route)
   for k in ('exp_avg','exp_avg_sq'):
    v=entry[k];need(v.shape==ref.shape and v.dtype==torch.float32 and v.device.type=='cpu' and torch.isfinite(v).all().item(),'Adam moment invalid '+route)
    if not torch.equal(v,before[k]):moment_changes+=1
   need((entry['exp_avg_sq']>=0).all().item(),'Adam negative second moment')
  if route=='code':need(tree_digest(opt)==tree_digest(old),'Frozen code Adam changed')
  else:need(moment_changes==108,'All54 fusion moments expected to change')
  optimizers[route]={'parameter_tensors':n,'named_order_exact':True,'group_configuration_equal_R1':True,'step_before':32,'step_after':expected_step,'all_state_moments_finite':True,'changed_moment_tensors':moment_changes,'state_sha256':tree_digest(opt)}
 need(optimizers['code']['state_sha256']==state['code_optimizer_unchanged_sha256']==state['actual_code_Adam_sha256']==result['code_Adam_unchanged_sha256']==pre['code_Adam_sha256'],'Frozen code Adam digest receipts differ')
 def rng_report(v):
  cpu=v['torch_CPU_RNG'];cuda=v['CUDA_RNG'];py=v['python_random_state'];np=v['numpy_random_state']
  need(isinstance(cpu,torch.Tensor) and cpu.dtype==torch.uint8 and cpu.ndim==1 and cpu.numel()>0 and cpu.device.type=='cpu','CPU RNG schema invalid')
  need(isinstance(cuda,list) and len(cuda)==1 and all(isinstance(t,torch.Tensor) and t.dtype==torch.uint8 and t.ndim==1 and t.numel()>0 and t.device.type=='cpu' for t in cuda),'CUDA RNG schema invalid')
  need(isinstance(py,tuple) and len(py)==3 and py[0]==3 and isinstance(py[1],tuple) and len(py[1])==625 and all(isinstance(x,int) and 0<=x<2**32 for x in py[1][:-1]) and 0<=py[1][-1]<=624 and (py[2] is None or math.isfinite(py[2])),'Python RNG schema invalid')
  need(set(np)=={'algorithm','keys','position','has_gauss','cached_gaussian'} and np['algorithm']=='MT19937' and isinstance(np['keys'],list) and len(np['keys'])==624 and all(isinstance(x,int) and 0<=x<2**32 for x in np['keys']) and 0<=np['position']<=624 and np['has_gauss'] in (0,1) and math.isfinite(np['cached_gaussian']),'NumPy RNG schema invalid')
  return {'CPU':tensor_sha(cpu),'CUDA':[tensor_sha(t) for t in cuda],'Python':json_digest(py),'NumPy':json_digest(np)}
 parent_rng=rng_report(parent);terminal_rng=rng_report(state)
 need(transition['complete'] is True and transition['parent_terminal_sha256']==PARENT_SHA and transition['parent_RNG_restored'] is True and transition['expected_hashes']==transition['restored_hashes']==parent_rng and transition['new_sampler_prematerialized'] is True and transition['global_rng_not_consumed_for_sampler'] is True,'RNG restoration receipt does not match actual parent bytes')
 metadata=state['parent_metadata'];need(metadata['terminal_sha256']==PARENT_SHA and metadata['protocol_sha256']==PARENT_PROTOCOL and metadata['logical_state_sha256']==PARENT_LOGICAL and metadata['counts']==parent['counts'] and metadata['sampler_schedule']==parent['sampler_schedule'] and metadata['completed_cycles']==32 and metadata['RNG_boundary_hashes']=={'expected':parent_rng,'restored':parent_rng},'Embedded parent boundary metadata differs')
 need(sum(tensor_bytes(state[k]) for k in ('trainable_state','failed_frozen_prior_state','code_optimizer','fusion_optimizer','partial_accumulated_gradients'))==pre['planned_tensor_payload_bytes'],'Pre-save tensor byte estimate differs from actual tensors')
 phase_receipts=[]
 for i,p in enumerate(t['phase_timings'],1):
  a=read(RUN/('outputs/phase_%03d_%s_start.json'%(i,p['phase'])));b=read(RUN/('outputs/phase_%03d_%s_end.json'%(i,p['phase'])))
  need(b==p and a['phase']==p['phase'] and a['start_seconds']==p['start_seconds'] and a['complete'] is False and p['complete'] is True and 0<=p['start_seconds']<=p['end_seconds']<result['seconds'] and math.isclose(p['end_seconds']-p['start_seconds'],p['seconds'],rel_tol=1e-10,abs_tol=1e-9),'Phase timing receipts differ')
  need(not phase_receipts or phase_receipts[-1]['end_seconds']<=p['start_seconds'],'Phases overlap/out of order');phase_receipts.append(p)
 need(len(phase_receipts)==18,'Expected18 measured phases')
 final_phases={p['phase']:p for p in phase_receipts[-3:]}
 assembly=final_phases['terminal_CPU_copy_and_assembly'];serial=final_phases['terminal_serialization'];commit=final_phases['terminal_hash_and_atomic_commit']
 need(assembly['tensor_payload_bytes']==pre['planned_tensor_payload_bytes'] and serial['serialized_bytes']==TERMINAL_BYTES and serial['write_callbacks']>0 and serial['start_seconds']<=serial['serializer_finished_seconds']<=serial['end_seconds'] and 0<=serial['flush_fsync_seconds']<=serial['seconds'] and commit['bytes']==TERMINAL_BYTES and commit['sha256']==TERMINAL_SHA and commit['ledger_total']<protocol['contract']['output_bytes'],'Measured terminal save/commit bounds differ')
 need(math.isclose(pre['remaining_hard_seconds'],1800-assembly['start_seconds'],rel_tol=0,abs_tol=0.1),'Pre-save remaining time differs')
 for label,path,expected in [('base',ORIGINAL,ORIGINAL_SHA),('parent',PARENT,PARENT_SHA),('terminal',terminal,TERMINAL_SHA)]:need(sha(path)==expected,'Checkpoint changed during audit '+label)
 report={'format':'completion-codeformer-delivered-objective-v4-independent-state-audit-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'transport_audit_sha256':TRANSPORT,'protocol_sha256':PROTOCOL,'checks_passed':CHECKS,'evidence':pins,'supervisor':supervisor,'state_status':{'committed_terminal_available':True,'valid_new_recovery_checkpoint':True,'reconstruction_complete':True,'qualified_model':False,'automatic_resume_permitted':False,'new_candidate_packaged':False,'quality_failure_is_separate_from_state_validity':True},'checkpoint_passive_admission':passive_reports,'safe_loading':{'torch':str(torch.__version__),'weights_only':True,'map_location':'cpu','all_three_passive_checks_preceded_first_load':True,'safe_globals_added':False,'returned_code_executed':False,'model_instantiated':False},'model_state':{'full_state_tensors':497,'delta_tensors':332,'frozen_bitexact_to_R1':443,'fusion_tensors_changed':54,'original_prior_tensors_bitexact':165,'base_logical_sha256':ORIGINAL_LOGICAL,'parent_logical_sha256':PARENT_LOGICAL,'terminal_logical_sha256':TERMINAL_LOGICAL,'frozen443_sha256':frozen_hash,'original_prior_sha256':prior_hash,'fusion_changes':changes},'optimizers':optimizers,'RNG':{'actual_parent_hashes':parent_rng,'boundary_expected_and_restored_equal_actual_parent':True,'terminal_hashes':terminal_rng,'terminal_equal_parent_by_family':{k:terminal_rng[k]==parent_rng[k] for k in parent_rng},'all_four_families_retained_and_schema_valid':True,'limitation':'Saved parent and returned boundary receipts establish reported exact restoration, not independent observation of transient VM memory.'},'sampler':{'schedule_exact':True,'each_of_256_FIT_cases_once':True,'next_cycle':65,'next_clean_sampler_index':256,'partial_accumulated_gradients_empty':True,'progress':state['progress']},'training':{'all64cycle_receipts':cycles,'newphase_counts':expected_counts,'cumulative_counts':cumulative,'typed_slots':result['typed_forward_slots'],'whole_model_forwards_newphase':480,'teacher_neural_calls':0,'gradient_clipping':{'steps':64,'preclip_min':min(clip),'preclip_max':max(clip),'clipped_steps':sum(x>1 for x in clip)},'all48_historical_R1_beforeupdate_compatibility_receipts_pass':True,'all4_fresh_app_parity_receipts_pass':True,'parity_limitation':'Raw/PNG saved-output parity is independently audited separately; state audit validates source-bound receipts without neural replay.'},'terminal_save':{'pre_save_receipt':pre,'phases':final_phases,'serialized_bytes':TERMINAL_BYTES,'serialization_seconds':serial['seconds'],'serializer_seconds_before_flush':serial['serializer_finished_seconds']-serial['start_seconds'],'fsync_seconds':serial['flush_fsync_seconds'],'hash_commit_seconds':commit['seconds'],'partial_files_present':False,'actual_terminal_pending_tensor_bytes':pre['planned_tensor_payload_bytes'],'repair_observed':'Complete CRC-valid ZIP and exclusive commit returned inside existing1800s bound; state validity is separate from failed model quality.'},'objective_contract':objective,'numeric_screen_reported':result['numeric_screen'],'status':'RECOVERY_STATE_VALID_QUALITY_SCREEN_FAILED_NO_ADOPTION','local_actions':{'CPU_weights_only_loads':3,'neural_models_instantiated':0,'neural_forwards':0,'gradient_calls':0,'optimizer_objects_or_steps':0,'VM_calls':0,'candidate_packaging':0},'model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-started}
 target=RET/'independent_state_audit.json'
 with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'complete':True,'audit':str(target),'sha256':sha(target),'checks':CHECKS,'valid_new_recovery_checkpoint':True,'frozen443_exact':True,'fusion54_changed':True,'code_Adam_step':32,'fusion_Adam_step':96,'actual_terminal_serialization_seconds':serial['seconds'],'quality_screen_pass':False,'model_qualified':False,'seconds':report['seconds']},indent=2),flush=True)
if __name__=='__main__':main()
