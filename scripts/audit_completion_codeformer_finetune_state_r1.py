"""Independent CPU-only R1 saved training-state audit, without model construction.
Reviews passive ZIP/pickle globals before weights_only deserialization. Does not
run a forward, backward, optimizer, CUDA query, VM or returned executable code.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib,json,math,random,time
from pathlib import Path
import numpy as np
from audit_completion_codeformer_finetune_return_v1 import passive,pin,read,bound,logical,need,PREFIX
BASE=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION')
DATA=Path(r'C:\xampp\htdocs\YEAR 4\Testing')
PARENT=BASE/'outputs/completion_codeformer_finetune_v1_r1_return'
ROOT=PARENT/'cctv_completion_codeformer_finetune_v1_r1_results'
EXPECTED='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
BASE_SHA='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
ORIGINAL_LOGICAL='586a8854ed7b5e3f0e202c258addb7a87de63cd3068d153741b065379ff6668f'
EXPECTED_COUNTS={'forward_slots':528,'backward_calls':256,'code_optimizer_steps':32,'fusion_optimizer_steps':32,'empty_bypasses':168}
EXPECTED_TYPED={'teacher_encoder':40,'teacher_quantizer':40,'parity':8,'fit_code':128,'fit_RGB':128,'monitor':128,'public_development':56}

def main():
    started=time.monotonic()
    def guard():need(time.monotonic()-started<300,'State audit300second bound')
    ppin=pin(ROOT/'protocol.json');need(ppin['sha256']==EXPECTED,'Wrong returned R1 protocol')
    protocol=read(ROOT/'protocol.json');sealed=BASE/'outputs/completion_codeformer_finetune_v1_r1_packet/protocol.json'
    need((ROOT/'protocol.json').read_bytes()==sealed.read_bytes(),'Returned/sealed protocol bytes differ')
    manifest=read(ROOT/'export_manifest.json');need(manifest['protocol_sha256']==EXPECTED and manifest['worker_exit']==0,'Wrong returned manifest/status')
    files={r['path']:r for r in manifest['files']};need(len(files)==len(manifest['files']),'Duplicate manifest paths')
    relevant=['outputs/results.json','supervisor.json','outputs/initial_state.json','outputs/terminal_training_state.pth']+[f'outputs/training_cycle_{i:04d}.json' for i in range(1,33)]
    relevant_pins=[]
    for name in relevant:guard();need(name in files,'Missing state audit artifact '+name);relevant_pins.append(bound(ROOT/name,files[name]))
    result=read(ROOT/'outputs/results.json');supervisor=read(ROOT/'supervisor.json');initial=read(ROOT/'outputs/initial_state.json')
    need(result['complete'] is True and result['protocol_sha256']==EXPECTED and result['cycles_completed']==32 and result['stop_reason'] is None,'Not full terminal R1')
    need(result['counts']==EXPECTED_COUNTS and result['typed_forward_slots']==EXPECTED_TYPED and result['actual_whole_model_forward_calls']==448,'Reported finite counts differ')
    need(supervisor['complete'] is True and supervisor['worker_exit']==supervisor['actual_child_exit']==0 and not supervisor['timeout'] and supervisor['interruption'] is None,'Supervisor not clean successful exit')
    need(supervisor['terminal_state_present'] is True and supervisor['external_cap_seconds']==1800 and 0<result['seconds']<supervisor['seconds']<1800,'Run timing/cap/terminal differs')
    need(not (ROOT/'outputs/failure.json').exists(),'Unexpected failure in successful return')
    need(result['model_qualified'] is False and result['visual_review_pending'] is True and result['user_acceptance_pending'] is True and result['automatic_app_promotion'] is False,'Overbroad result flags')
    need(result['original_checkpoint_unchanged'] is True and result['frozen_state_unchanged'] is True,'Original/frozen receipt not retained')
    snapshots=result['metric_snapshots'];need([r['cycle'] for r in snapshots]==[8,16,32] and all(r['state_saved'] is False and r['recoverable_checkpoint'] is False for r in snapshots),'Intermediate checkpoints rescue differs')
    expected_names=read(BASE/'outputs/completion_codeformer_gradient_v1_return/cctv_completion_codeformer_gradient_v1_results/outputs/results.json')['cases'][0]
    names_by_route={route:[PREFIX[group]+tensor['name'] for group,record in expected_names[key].items() for tensor in record['gradient_tensors']] for route,key in [('code','code_gradients'),('fusion','fusion_gradients')]}
    cycles=[];group_norms={};all_caseids=[]
    for i in range(1,33):
        guard();row=read(ROOT/f'outputs/training_cycle_{i:04d}.json')
        need(row['cycle']==i and row['microbatch_ids']==protocol['schedule'][i-1],'Cycle/sampler sequence differs')
        need(len(set(row['microbatch_ids']))==4,'Cycle repeated case')
        all_caseids.extend(row['microbatch_ids'])
        need(row['counts']=={'forward_slots':148+8*i+32*sum(i>s for s in (8,16,32)), 'backward_calls':8*i,'code_optimizer_steps':i,'fusion_optimizer_steps':i,'empty_bypasses':44+40*sum(i>s for s in (8,16,32))},'Cycle cumulative counts differ')
        # Cycle receipts precede snapshot work at that same cycle. Baseline slots:
        # teacher80 +parity8 +monitor32 +publicDEV28 =148; empty controls use0.
        expected_groups={'code_gradient_groups':{'encoder':164,'position':1,'feature':2,'transformer':108,'prediction':3},'fusion_gradient_groups':{'32':18,'64':18,'128':18}}
        gradient_summary={}
        for key,groups in expected_groups.items():
            need(set(row[key])==set(groups),'Gradient group namespace differs')
            for name,count in groups.items():
                entry=row[key][name];norm=entry['L2']
                need(entry['tensors']==count and entry['finite'] is True and entry['nonzero'] is True and math.isfinite(norm) and norm>0,'Nonfinite/missing/zero gradient group receipt')
                group_norms.setdefault(key+':'+name,[]).append(norm)
            aggregate=math.sqrt(sum(entry['L2']**2 for entry in row[key].values()))
            clipping_key='code_preclip_L2' if key=='code_gradient_groups' else 'fusion_preclip_L2'
            need(math.isfinite(row[clipping_key]) and math.isclose(aggregate,row[clipping_key],rel_tol=2e-5,abs_tol=1e-7),'Preclip/group norm receipts inconsistent')
            gradient_summary[key]={'preclip_L2':row[clipping_key],'summed_group_L2':aggregate}
        for key,losskeys in [('code_losses',{'CE','latent_MSE'}),('fusion_losses',{'mask_RGB512_L1','inner_boundary_RGB512_L1'})]:
            need([entry['id'] for entry in row[key]]==row['microbatch_ids'],'Loss case ordering differs')
            for entry in row[key]:
                need(set(entry)==losskeys|{'id'} and all(math.isfinite(entry[k]) and entry[k]>=0 for k in losskeys),'Invalid loss receipt')
        need(0<row['seconds']<result['seconds'] and (not cycles or cycles[-1]['seconds']<row['seconds']),'Cycle timing sequence differs')
        cycles.append({'cycle':i,'microbatch_ids':row['microbatch_ids'],'counts':row['counts'],'seconds':row['seconds'],'gradient_summary':gradient_summary,'code_losses':row['code_losses'],'fusion_losses':row['fusion_losses']})
    need(len(all_caseids)==len(set(all_caseids))==128 and set(all_caseids)=={r['id'] for r in protocol['fit_cases']},'FIT coverage not exactly once')
    original=DATA/'checkpoints/codeformer_inpainting.pth';base_pin=pin(original);need(base_pin['sha256']==BASE_SHA,'Original checkpoint changed')
    terminal=ROOT/'outputs/terminal_training_state.pth';terminal_pin=pin(terminal)
    saved_receipt=result['terminal_training_state'];need(terminal_pin['bytes']==saved_receipt['bytes'] and terminal_pin['sha256']==saved_receipt['sha256'],'Terminal file pin differs')
    passive_reviews={'original':passive(original),'terminal':passive(terminal)}
    guard()
    import torch
    torch.set_num_threads(1)
    original_object=torch.load(original,map_location='cpu',weights_only=True)
    saved=torch.load(terminal,map_location='cpu',weights_only=True)
    need(isinstance(original_object,dict) and isinstance(original_object.get('params_ema'),dict) and isinstance(saved,dict),'State object schema differs')
    state=original_object['params_ema'];trainable=saved['trainable_state'];frozen_names={n for n in state if n.startswith(('quantize.','generator.'))};trainable_names=set(state)-frozen_names
    need(len(state)==497 and len(frozen_names)==165 and len(trainable_names)==len(trainable)==332 and set(trainable)==trainable_names,'State partition differs')
    need(saved['format']=='completion-codeformer-trainable-delta-resume-v1' and saved['base_checkpoint_sha256']==BASE_SHA and saved['protocol_sha256']==EXPECTED and saved['contract']==protocol['contract'],'Saved base/protocol/contract differs')
    need(saved['cycle']==saved['completed_paired_cycles']==32 and saved['next_cycle']==33 and saved['next_clean_sampler_index']==128,'Saved sampler/cycle differs')
    need(saved['counts']==EXPECTED_COUNTS and saved['sampler_schedule']==protocol['schedule'],'Saved counts/schedule differs')
    need(saved['progress']=={'cycle':32,'route':'fusion','microbatch':4,'microbatches_backward_complete':4,'phase':'paired_cycle_complete'},'Terminal progress differs')
    need(saved['reason']=='completed_finite_pilot' and saved['automatic_resume_permitted'] is False and saved['resume_progress_requires_new_manual_audit'] is True and saved['model_qualified'] is False and saved['partial_accumulated_gradients']=={},'Terminal resume/gradients/reason flags differ')
    changes=[];coordinates=bytecount=0
    for name,value in trainable.items():
        guard();reference=state[name]
        need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32 and value.shape==reference.shape,'Invalid trained tensor '+name)
        a=value.detach().contiguous().numpy();b=reference.detach().contiguous().numpy();need(np.isfinite(a).all(),'Nonfinite trained state '+name)
        delta=a.astype(np.float64)-b.astype(np.float64);changed=int(np.count_nonzero(delta))
        changes.append({'name':name,'shape':list(a.shape),'coordinates':a.size,'changed_coordinates':changed,'maximum_absolute_change':float(np.max(np.abs(delta))),'change_L2':float(np.sqrt(np.square(delta).sum()))})
        coordinates+=a.size;bytecount+=a.nbytes
    need(coordinates==58459840 and bytecount==233839360,'Trainable size differs')
    need(sum(r['changed_coordinates'] for r in changes)>0,'No saved learned change')
    base_logical=logical(state);need(base_logical==ORIGINAL_LOGICAL==initial['original_logical_state_sha256']==result['initial_state_sha256'],'Initial base logical hash differs')
    reconstructed=dict(state);reconstructed.update(trainable);reconstructed_logical=logical(reconstructed);frozen_logical=logical({n:state[n] for n in frozen_names})
    need(reconstructed_logical==saved['logical_state_sha256']==saved_receipt['logical_state_sha256']==result['final_logical_state_sha256']==snapshots[-1]['logical_state_sha256'],'Reconstructed terminal logical hash differs')
    need(frozen_logical==saved['frozen_state_sha256']==initial['frozen_generator_codebook_sha256'],'Frozen base hash differs')
    optimizers={}
    for route,count in [('code',278),('fusion',54)]:
        names=saved['optimizer_parameter_names'][route];need(names==names_by_route[route] and len(names)==len(set(names))==count,'Ordered optimizer tensor names differ '+route)
        optimizer=saved[route+'_optimizer'];need(len(optimizer['param_groups'])==1 and set(optimizer['state'])==set(range(count)),'Adam state entries missing '+route)
        group=optimizer['param_groups'][0];need(group['params']==list(range(count)) and group['lr']==1e-5 and tuple(group['betas'])==(.9,.99) and group['eps']==1e-8 and group['weight_decay']==0,'Adam mapping/hyperparameters differ '+route)
        moment_rows=[]
        for index,name in enumerate(names):
            guard();record=optimizer['state'][index];need(set(record)=={'step','exp_avg','exp_avg_sq'},'Unexpected Adam state keys')
            step=record['step'];need(isinstance(step,torch.Tensor) and step.numel()==1 and float(step.item())==32.,'Adam step not32 '+name)
            row={'name':name,'step':32}
            for key in ('exp_avg','exp_avg_sq'):
                value=record[key];need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32 and value.shape==trainable[name].shape,'Adam moment shape/dtype differs '+name)
                array=value.detach().contiguous().numpy();need(np.isfinite(array).all(),'Nonfinite Adam moment')
                if key=='exp_avg_sq':need((array>=0).all(),'Negative Adam squared moment')
                row[key]={'shape':list(array.shape),'nonzero_coordinates':int(np.count_nonzero(array)),'maximum_absolute':float(np.max(np.abs(array))),'sha256':hashlib.sha256(array.tobytes()).hexdigest()}
            moment_rows.append(row)
        optimizers[route]={'parameter_tensors':count,'ordered_names_match_independent_gradient_diagnostic':True,'names':names,'Adam_state_entries':len(optimizer['state']),'moment_tensors':2*count,'all_step_counters':32,'hyperparameters':{k:v for k,v in group.items() if k!='params'},'moments':moment_rows}
    need(saved['python_random_state']==random.Random(240).getstate(),'Python RNG mismatch')
    np_expected=np.random.RandomState(240).get_state();np_saved=saved['numpy_random_state'];need(np_saved['algorithm']==np_expected[0] and np.array_equal(np.asarray(np_saved['keys'],dtype=np.uint32),np_expected[1]) and np_saved['position']==np_expected[2] and np_saved['has_gauss']==np_expected[3] and np_saved['cached_gaussian']==np_expected[4],'Numpy seeded RNG mismatch')
    rng=[]
    need(len(saved['CUDA_RNG'])==1,'CUDA RNG count differs')
    for kind,value in [('CPU',saved['torch_CPU_RNG']),('CUDA0',saved['CUDA_RNG'][0])]:
        need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.uint8 and value.ndim==1 and value.numel()>0,'RNG tensor schema differs')
        array=value.numpy();rng.append({'kind':kind,'bytes':array.nbytes,'sha256':hashlib.sha256(array.tobytes()).hexdigest(),'replay_performed':False})
    for item in [base_pin,terminal_pin,*relevant_pins]:guard();bound(item['path'],item)
    audit={'format':'completion-codeformer-finetune-r1-independent-saved-state-audit-v1','complete':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'protocol':ppin,'base_checkpoint':base_pin,'terminal_checkpoint':terminal_pin,'passive_reviews':passive_reviews,'counts':EXPECTED_COUNTS,'typed_slots':EXPECTED_TYPED,'supervisor':supervisor,'cycle_receipts':{'count':32,'schedule_exact':True,'all128fit_cases_covered_once':True,'receipts_not_gradient_replay':True,'all_group_receipts_finite_positive':True,'group_norm_ranges':{k:{'minimum':min(v),'maximum':max(v),'cycles':len(v)} for k,v in group_norms.items()},'preclip_norm_check_tolerance':{'relative':2e-5,'absolute':1e-7},'cycles':cycles},'trainable':{'tensors':332,'coordinates':coordinates,'FP32_bytes':bytecount,'changed_tensors':sum(r['changed_coordinates']>0 for r in changes),'changed_coordinates':sum(r['changed_coordinates'] for r in changes),'changes':changes},'frozen_reconstruction':{'original_tensors':165,'frozen_hash':frozen_logical,'full_tensors':497,'original_logical_sha256':base_logical,'trained_logical_sha256':reconstructed_logical,'all_reported_terminal_hashes_match':True,'limitation':'Frozen165 supplied by retained original checkpoint; receipt consistency is not observation of all past VM memory states.'},'optimizers':optimizers,'RNG':{'Python_and_NumPy_seed240_exact':True,'tensors':rng},'terminal_state_recoverability':'Complete332trainable parameters +bothnamedAdam states +RNG/schedule/counts/progress and retained frozenbase; automaticresume forbidden','local_actions':{'weights_only_CPU_loads':2,'custom_safe_globals_added':False,'model_class_imports':0,'model_instantiations':0,'neural_forwards':0,'gradient_queries':0,'backward_calls':0,'optimizer_updates':0,'VM_calls':0,'returned_code_executed':False},'data_scope':'Adaptive originalTRAIN fit andmonitor, not independent final or identity qualification','model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-started}
    target=PARENT/'independent_state_audit.json'
    with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(audit,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'complete':True,'audit':pin(target),'trained_logical_sha256':reconstructed_logical,'changed_tensors':audit['trainable']['changed_tensors'],'changed_coordinates':audit['trainable']['changed_coordinates'],'optimizer_steps':64,'seconds':audit['seconds']},indent=2))
if __name__=='__main__':main()