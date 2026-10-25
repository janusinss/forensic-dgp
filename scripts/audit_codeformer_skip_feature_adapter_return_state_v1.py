"""Independent returned skip-adapter states/operations; no neural execution.

Only exact imported-and-pinned checkpoint bytes are read with weights_only=True
on CPU. No returned Python, optimizer, model architecture or training is run.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import stat
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RETURN=ROOT/'outputs/codeformer_skip_feature_adapter_v1_return'
REVIEW=ROOT/'outputs/codeformer_skip_feature_adapter_return_review_v1'
OUT=REVIEW/'state_v1'
PREFIX='outputs/skip_feature_adapter_v1/'
PROTOCOL_SHA='0d358ab29e0344c7243b6312cb1d04ca6cc0e0a1de270bca7e6b4040dd566c90'
ARCHIVE_SHA='f445f5ffe3a97215046cb8e265eece2dd2d36961ac0b607948a52a29a5809c76'
IMPORTER_SHA='790dfeb3f3e964ae5db99250b6c3c84ce4bc7ea3fc5e7cfc74bdf1032e15dcb8'
R2_FILE='weights/codeformer_r2.pth'
R2_SHA='d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
R2_STATE='a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
CAPS={'original_baseline_forwards':478,'parent_baseline_forwards':478,'R2_baseline_forwards':478,
    'R2_clean_feature_teacher_forwards':55,'original_clear_teacher_forwards':55,'R2_dark_teacher_forwards':55,
    'candidate_snapshot_forwards':1434,'candidate_training_forwards':440,'candidate_TRAIN_guard_forwards':96,
    'candidate_witness_forwards':3,'R2_auxiliary_forwards':18,'candidate_auxiliary_forwards':54,
    'backward_calls':440,'optimizer_step_attempts':110,'gradient_queries':4}
SITES={'32':256,'64':256,'128':128,'256':128}
START=time.monotonic()


def need(value,message):
    if not value:raise ValueError(message)


def check():need(time.monotonic()-START<300,'Finite300s CPU state audit cap')


def safe(path):
    path=Path(path).absolute()
    need(path.resolve(strict=True).is_relative_to(ROOT),'Workspace evidence required')
    for part in (path,*path.parents):
        row=part.lstat()
        need(not stat.S_ISLNK(row.st_mode) and not getattr(row,'st_file_attributes',0)&1024,'Linked/reparse evidence refused')
    row=path.stat();need(stat.S_ISREG(row.st_mode) and row.st_nlink==1,'Unique regular evidence required')
    return path


def pin(path):
    check();path=safe(path);before=path.stat();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):check();digest.update(block)
    after=path.stat()
    need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Evidence changed while hashing')
    return {'bytes':before.st_size,'sha256':digest.hexdigest()}


def relative(path):return Path(path).absolute().relative_to(ROOT).as_posix()


def parse_json(value):
    def unique(pairs):
        value={}
        for name,item in pairs:need(name not in value,'Duplicate JSON key');value[name]=item
        return value
    return json.loads(value,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def read(path,cap=4*1024**2):
    path=safe(path);need(path.stat().st_size<=cap,'Bounded JSON required')
    return parse_json(path.read_bytes())


def output_path(name):
    need(isinstance(name,str) and '\\' not in name and ':' not in name and not name.startswith('/') and
        all(p not in ('','.','..') for p in name.split('/')),'Portable canonical returned path required')
    return safe(RETURN.joinpath(*PurePosixPath(name).parts))


def shapes():
    result={}
    for site,c in SITES.items():
        for name,shape in {'down.weight':(32,c,1,1),'down.bias':(32,),
            'depthwise.weight':(32,1,3,3),'depthwise.bias':(32,),
            'up.weight':(c,32,1,1),'up.bias':(c,)}.items():result['adapters.'+site+'.'+name]=shape
    need(len(result)==24 and sum(math.prod(v) for v in result.values())==51328,'Exact adapter schema differs')
    return result


def digest_state(state,torch):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        need(isinstance(tensor,torch.Tensor),'Tensor-only state required')
        t=tensor.detach().cpu().contiguous()
        header=json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode()
        h.update(len(header).to_bytes(8,'big'));h.update(header);h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def exact_count_map(value):
    need(type(value) is dict and set(value)==set(CAPS) and all(type(v) is int and 0<=v<=CAPS[k] for k,v in value.items()),
        'Exact finite counter map required')


def close_operations(rows):
    """Return durable prefix totals; no missing return is silently completed."""
    reserved=dict.fromkeys(CAPS,0);returned=dict.fromkeys(CAPS,0);pending={};sequence=0
    prefixes=[(dict(reserved),dict(returned),dict(pending))]
    reservations={};returns={}
    for row in rows:
        need(type(row) is dict,'Operation JSON object required')
        if row.get('event')=='reserved':
            sequence+=1;kind=row.get('kind');need(kind in CAPS and row.get('sequence')==sequence,'Ordered reservation identity differs')
            reserved[kind]+=1;need(reserved[kind]<=CAPS[kind] and row.get('count')==reserved[kind],'Operation cap/count differs')
            need(type(row.get('context')) is dict,'Reservation context required')
            pending[sequence]=kind;reservations[sequence]=row
        elif row.get('event')=='returned':
            number=row.get('reservation');need(type(number) is int and number in pending and row.get('kind')==pending[number],
                'Unreserved/duplicate/wrong-kind returned operation')
            returned[pending.pop(number)]+=1;returns[number]=row
        else:raise ValueError('Unknown durable operation event')
        prefixes.append((dict(reserved),dict(returned),dict(pending)))
    return {'reserved':reserved,'returned':returned,'pending':pending,'reservations':reservations,'returns':returns,'prefixes':prefixes}


def learning_rate(update):
    need(type(update) is int and 1<=update<=110,'Finite learning-rate update required')
    return 1e-4*min(1.,update/10)


def finite_scalar(value):return type(value) in (float,int) and math.isfinite(value)


def validate_rng(saved,torch):
    for key in ('CPU_RNG','CUDA_RNG'):
        value=saved[key]
        need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.uint8 and
            value.ndim==1 and 1<=value.numel()<=65536,'Bounded CPU-stored RNG byte vector required')
    p=saved['python_RNG'];need(isinstance(p,tuple) and len(p)==3,'Python RNG tuple required')
    local=random.Random();local.setstate(p)
    value=saved['NumPy_RNG'];need(set(value)=={'algorithm','state','position','has_gauss','cached_gaussian'} and
        value['algorithm']=='MT19937' and isinstance(value['state'],torch.Tensor) and
        value['state'].shape==(624,) and value['state'].dtype==torch.int64 and value['state'].device.type=='cpu' and
        0<=value['state'].min().item()<=value['state'].max().item()<2**32 and
        type(value['position']) is int and 0<=value['position']<=624 and value['has_gauss'] in (0,1) and
        finite_scalar(value['cached_gaussian']),'Complete bounded NumPy RNG representation required')
    return {'CPU_bytes':saved['CPU_RNG'].numel(),'CUDA_bytes':saved['CUDA_RNG'].numel(),
        'Python_state_validated_on_isolated_Random':True,'NumPy_MT19937_624words_valid':True,
        'CPU_SHA256':hashlib.sha256(saved['CPU_RNG'].numpy().tobytes()).hexdigest(),
        'CUDA_SHA256':hashlib.sha256(saved['CUDA_RNG'].numpy().tobytes()).hexdigest()}


def validate_adam(optimizer,adapter_shapes,updates,torch):
    need(type(optimizer) is dict and set(optimizer)=={'state','param_groups'} and
        type(optimizer['state']) is dict and len(optimizer['param_groups'])==1,'One actual Adam group required')
    group=optimizer['param_groups'][0];names=sorted(adapter_shapes)
    need(group['name']=='skip_adapters' and group['params']==list(range(24)) and
        group['betas']==(.9,.99) and group['eps']==1e-8 and group['weight_decay']==0 and
        group['base_lr']==1e-4 and group['lr']==learning_rate(max(1,updates)), 'Exact finite Adam options/order/LR differ')
    need(set(optimizer['state'])==(set(range(24)) if updates else set()),'Adam state presence differs from committed updates')
    reports={}
    for index,name in enumerate(names):
        if not updates:break
        row=optimizer['state'][index];need(set(row)=={'step','exp_avg','exp_avg_sq'},'Exact ordinary Adam state fields required')
        step=row['step'];need(isinstance(step,torch.Tensor) and step.numel()==1 and torch.isfinite(step).all().item() and
            step.item()==updates,'Each actual Adam step must equal committed updates')
        for key in ('exp_avg','exp_avg_sq'):
            value=row[key];need(isinstance(value,torch.Tensor) and tuple(value.shape)==adapter_shapes[name] and
                value.dtype==torch.float32 and value.device.type=='cpu' and torch.isfinite(value).all().item(),
                'Finite CPU Adam moment geometry differs')
        need(row['exp_avg_sq'].min().item()>=0,'Adam second moment must remain nonnegative')
        reports[name]={'step':int(step.item()),'first_moment_l2':float(row['exp_avg'].double().square().sum().sqrt().item()),
            'second_moment_max':float(row['exp_avg_sq'].max().item())}
    return {'parameter_groups':1,'parameters':24,'state_entries':len(optimizer['state']),
        'updates':updates,'LR':group['lr'],'all_moments':reports,'optimizer_not_instantiated_or_stepped':True}


def main():
    need(not OUT.exists(),'Preserve previous/partial state audit; no overwrite')
    imported_path=REVIEW/'return_import_v1.json';import_pin=pin(imported_path);imported=read(imported_path)
    need(imported.get('complete') is True and imported.get('transport_pass') is True and
        imported.get('protocol_sha256')==PROTOCOL_SHA and imported.get('archive',{}).get('sha256')==ARCHIVE_SHA and
        imported.get('returned_code_executed') is False and imported.get('checkpoint_or_array_deserialized') is False,
        'Actual exact successful inert return-import receipt required before state reads')
    need(imported['source']['sha256']==IMPORTER_SHA and pin(ROOT/'scripts/import_codeformer_skip_feature_adapter_return_v1.py')==
        {k:imported['source'][k] for k in ('bytes','sha256')},'Exact independently reviewed importer source required')
    files=imported['files'];used={}
    def bound(name,json_value=True,cap=4*1024**2):
        need(name in files,'Required imported member missing: '+name)
        path=output_path(name);actual=pin(path);need(actual==files[name],'Imported member changed: '+name)
        used[name]=actual
        return read(path,cap) if json_value else path
    protocol=bound('protocol.json');need(pin(output_path('protocol.json'))['sha256']==PROTOCOL_SHA and protocol['call_caps']==CAPS,
        'Frozen protocol/call caps changed')
    schedule=bound(protocol['data']['schedule_path']);evaluation=bound(protocol['evaluation_path']);auxiliary=bound(protocol['auxiliary_path'])
    need(len(schedule['updates'])==110 and len(schedule['anchor_source_order'])==55 and len(evaluation['cases'])==478 and
        len(auxiliary['cases'])==18,'Exact finite source/schedule/evaluation census required')
    runtime=bound(PREFIX+'runtime.json');initial=bound(PREFIX+'model_initialization.json');witness=bound(PREFIX+'witness/receipt.json')
    need(runtime['GPU']=='NVIDIA L4' and runtime['FP32'] is True and runtime['TF32'] is False and
        runtime['seed']==20261019 and runtime['deterministic_algorithms'] is True and runtime['manual_only'] is True,
        'Actual fixed seed/manual L4 FP32 runtime differs')
    need(initial['census']['trainable_tensors']==24 and initial['census']['frozen_tensors']==515 and
        initial['census']['trainable_scalar_parameters']==51328 and initial['candidate']['original_all515_frozen'] is True,
        'Actual new-adapter initialization census differs')
    need(witness['complete'] is True and witness['candidate_forwards']==3 and witness['gradient_queries']==4 and
        witness['optimizer_updates']==0 and witness['restoration']['pass'] is True and
        witness['image_only_query_separate_from_combined_feature_image_query'] is True,
        'Actual three-forward/four-query witness prerequisite required')
    filter_witness=witness['roles']['filter']
    for field in ('total_gradient','image_only_gradient'):
        summary=filter_witness[field]
        need(summary['queried_tensors']==24 and set(summary['sites'])==set(SITES) and
            all(row['tensors']==6 and row['none']==0 and finite_scalar(row['up_l2']) and row['up_l2']>0
                for row in summary['sites'].values()),'Actual isolated/combined image gradient is disconnected')
    need(witness['roles']['dark']['scalar']==0 and all(row['l2']==0 for row in witness['roles']['dark']['total_gradient']['sites'].values()),
        'Initial R2 dark anchor was not exactly zero')
    journal_path=bound(PREFIX+'operations.jsonl',False);need(journal_path.stat().st_size<=16*1024**2,'Bounded operation JSONL required')
    operation_rows=[]
    for line in journal_path.read_bytes().splitlines():
        need(len(line)<=1024**2,'Bounded operation record required');operation_rows.append(parse_json(line))
    operations=close_operations(operation_rows)
    terminal=imported['independently_recomputed_status'];terminal_name=terminal['worker_terminal_evidence']
    need(terminal_name in files,'Explicit actual terminal receipt required for scientific state audit')
    failure=bound(terminal_name);completed=failure['completed_training_updates']
    need(failure['counts']==operations['reserved'] and failure['actual_return_counts']==operations['returned'] and
        {int(k):v for k,v in failure.get('pending_reservations',{}).items()}==operations['pending'],
        'Terminal totals contradict independently closed operation journal')
    need(completed==terminal['completed_training_updates'] and failure['partial_microbatches']==terminal['partial_microbatches'] and
        failure['optimizer_inflight']==terminal['optimizer_inflight'],'Terminal physical imported status differs')
    # Check every committed update against exact schedule, LR, finite losses and durable calls.
    train_reports=[]
    reserved=list(operations['reservations'].values())
    for update in range(1,completed+1):
        planned=schedule['updates'][update-1];need(planned['update']==update,'Training schedule order differs')
        slots=[]
        for slot in planned['slots']:
            row=bound(PREFIX+f'training/u{update:04d}_s{slot["slot"]}.json')
            need(row['update']==update and row['slot']==slot['slot'] and row['id']==slot['id'] and row['role']==slot['role'] and
                row['reference_id']==slot['reference_id'] and row['LR']==learning_rate(update) and
                finite_scalar(row['scalar']) and finite_scalar(row['divided_scalar']) and row['scalar']/4==row['divided_scalar'] and
                all(finite_scalar(v) for v in row['terms'].values()),'Actual training row differs from frozen schedule/loss scope')
            if row['role']=='filter':
                need(set(row['terms'])=={'normalized_feature_l1','spatial_rgb_l1','adjacent_residual_l1'} and
                    set(row['feature_scales'])==set(SITES) and all(finite_scalar(s) and s>=1e-6 for s in row['feature_scales'].values()),
                    'Frozen four-site blur loss fields differ')
                expected=.1*row['terms']['normalized_feature_l1']+row['terms']['spatial_rgb_l1']+.25*row['terms']['adjacent_residual_l1']
            else:
                need(set(row['terms'])=={'teacher_rgb_l1'} and row['feature_scales'] is None,'Behavior role loss fields differ')
                expected=2*row['terms']['teacher_rgb_l1']
            need(abs(row['scalar']-expected)<=2e-7*max(1.,abs(expected)),
                'Actual loss readouts contradict FP32 scalar composition')
            for kind in ('candidate_training_forwards','backward_calls'):
                calls=[x for x in reserved if x['kind']==kind and x['context'].get('update')==update and
                    x['context'].get('slot')==slot['slot']]
                need(len(calls)==1 and calls[0]['sequence'] in operations['returns'],'Missing/duplicate training forward/backward reservation')
            slots.append(row['scalar'])
        commit=bound(PREFIX+f'training/u{update:04d}_commit.json')
        need(commit['complete'] is True and commit['update']==update and commit['LR']==learning_rate(update) and
            commit['gradient_clip_max_norm']==1 and finite_scalar(commit['preclip_global_norm']) and
            commit['slot_scalars']==slots and commit['sampler']=={'completed_updates':update,'completed_microbatches':4*update,'seed':20261019} and
            commit['all515_base_values_unchanged'] is True,'Actual four-slot optimizer commit metadata differs')
        calls=[x for x in reserved if x['kind']=='optimizer_step_attempts' and x['context'].get('update')==update]
        need(len(calls)==1 and calls[0]['sequence'] in operations['returns'],'Actual committed update lacks returned Adam call')
        train_reports.append({'update':update,'LR':commit['LR'],'four_scalars':slots,'preclip_gradient_norm':commit['preclip_global_norm']})
    need(operations['returned']['candidate_training_forwards']>=4*completed and operations['returned']['backward_calls']>=4*completed and
        operations['returned']['optimizer_step_attempts']>=completed,'Physical committed training count differs')
    # At this point every checkpoint input has an exact imported pin; no executable return imports occur.
    import torch
    torch.set_num_threads(4)
    r2_path=bound(R2_FILE,False);need(files[R2_FILE]['sha256']==R2_SHA,'Exact retained R2 weights required')
    original=torch.load(r2_path,map_location='cpu',weights_only=True,mmap=True)
    need(set(original)=={'params_ema'} and len(original['params_ema'])==515,'Exact frozen R2 loader container required')
    base=original['params_ema'];need(digest_state(base,torch)==R2_STATE,'Actual R2 canonical state differs')
    adapter_shapes=shapes();expected_names={'base.'+name for name in base}|set(adapter_shapes)
    checkpoint_names=terminal['checkpoint_members'];need(checkpoint_names,'At least one actual checkpoint required')
    reports=[];adapter_initial=None;checkpoint_pins=[]
    for name in checkpoint_names:
        need((name.startswith(PREFIX) and name.endswith('/checkpoint.pt')) or name==PREFIX+'stopped_checkpoint.pt',
            'Incomplete temporary is retained but cannot be treated as complete parsed state')
        path=bound(name,False);need(path.stat().st_size<=384*1024**2,'Finite physical checkpoint required')
        saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        need(type(saved) is dict and saved['format']=='codeformer-skip-feature-adapter-training-state-v1' and
            saved['protocol_sha256']==PROTOCOL_SHA and saved['call_caps']==CAPS and saved['automatic_resume_safe'] is False and
            saved['model_qualified'] is False,'Actual new checkpoint schema/protocol differs')
        model=saved['model'];adapters=saved['adapter_state']
        need(type(model) is dict and set(model)==expected_names and len(model)==539 and
            type(adapters) is dict and set(adapters)==set(adapter_shapes),'Exact frozen515+new24 state schema required')
        need(all(isinstance(v,torch.Tensor) and v.device.type=='cpu' and v.dtype==torch.float32 and torch.isfinite(v).all().item()
            for v in model.values()),'Every candidate state tensor must be finite CPU FP32')
        need(all(torch.equal(model['base.'+key],value) for key,value in base.items()),'An original R2 tensor changed')
        need(saved['frozen_base_tensor_sha256']==R2_STATE,'Recorded base state differs')
        for key,shape in adapter_shapes.items():
            need(tuple(model[key].shape)==shape and isinstance(adapters[key],torch.Tensor) and
                torch.equal(adapters[key],model[key]),'Adapter duplicated state or shape differs')
        progress=saved['progress'];updates=progress['completed_training_updates']
        need(type(updates) is int and 0<=updates<=completed and saved['sampler']==
            {'completed_updates':updates,'completed_microbatches':4*updates,'seed':20261019},'Checkpoint sampler/committed progress differs')
        exact_count_map(saved['counts']);exact_count_map(saved['actual_return_counts'])
        pending={int(k):v for k,v in saved['pending_reservations'].items()}
        need(any(saved['counts']==a and saved['actual_return_counts']==b and pending==c for a,b,c in operations['prefixes']),
            'Checkpoint counters are not an actual durable operation prefix')
        adam=validate_adam(saved['optimizer'],adapter_shapes,updates,torch);rng=validate_rng(saved,torch)
        if updates==0:
            need(all(torch.count_nonzero(v).item()==0 for key,v in adapters.items() if '.up.' in key),
                'Identity initializer final adapter projections must be exactly zero')
            adapter_initial={key:value.clone() for key,value in adapters.items()}
        changes={}
        if adapter_initial is not None:
            for site in SITES:
                chosen=[key for key in adapter_shapes if key.startswith('adapters.'+site+'.')]
                norm=sum((adapters[key].double()-adapter_initial[key].double()).square().sum().item() for key in chosen)
                changes[site]={'changed_tensors':sum(not torch.equal(adapters[key],adapter_initial[key]) for key in chosen),
                    'change_l2':math.sqrt(norm)}
            if updates:need(all(v['changed_tensors']>0 and v['change_l2']>0 for v in changes.values()),'No learned adapter change at a site')
        checkpoint_pins.append({'path':name,**files[name]})
        reports.append({'path':name,'bytes':files[name]['bytes'],'sha256':files[name]['sha256'],
            'updates':updates,'partial_microbatches':progress['partial_microbatches'],'optimizer_inflight':progress['optimizer_inflight'],
            'all515_frozen_base_tensors_exact':True,'base_canonical_sha256':R2_STATE,
            'all24_adapter_duplicates_exact':True,'model539_canonical_sha256':digest_state(model,torch),
            'adapter24_canonical_sha256':digest_state(adapters,torch),'adapter_changes_from_initial':changes,
            'Adam':adam,'RNG':rng,'checkpoint_counts':saved['counts'],'checkpoint_returns':saved['actual_return_counts'],
            'checkpoint_is_pre_snapshot_forward_durable_prefix':True,
            'checkpoint_progress_prior_state_pointer_not_misreported_as_current':True})
        del saved,model,adapters
    gates={}
    for name in sorted(files):
        if re.fullmatch(re.escape(PREFIX)+r'(snapshot|guard)_\d{4}/gates.json',name):gates[name]=bound(name)
    supervised=bound('outputs/supervision/run_receipt.json');finished_name='outputs/supervision/finished.json'
    finished_path=output_path(finished_name);finished_pin=pin(finished_path)
    need(finished_pin==imported['external_artifacts_reconstructed_from_bound_companion'][finished_name],
        'Externally reconstructed final supervisor evidence changed')
    finished=read(finished_path)
    run=[x for x in supervised['stages'] if x['stage']=='--run'];need(len(run)==1,'Unique manual actual run required')
    need(run[0]==finished['stages'][1] and run[0]['exit_code']==terminal['worker_run_exit_code'] and
        run[0]['outer_watchdog_expired'] is False and run[0]['log_capped'] is False,'Actual process completion/timing differs')
    need(pin(imported_path)==import_pin and pin(finished_path)==finished_pin and
        all(pin(output_path(name))==value for name,value in used.items()),
        'Imported evidence changed during state audit')
    receipt={'format':'codeformer-skip-feature-adapter-independent-state-operation-audit-v1','complete':True,'audit_pass':True,
        'UTC':datetime.now(timezone.utc).isoformat(),'source_auditor':{'path':relative(__file__),**pin(__file__)},
        'actual_import':{'path':relative(imported_path),**import_pin},'protocol_sha256':PROTOCOL_SHA,'archive_sha256':ARCHIVE_SHA,
        'used_member_pins':used,'actual_runtime':runtime,'actual_operation_reservations':operations['reserved'],
        'actual_operation_returns':operations['returned'],'pending_operations':operations['pending'],
        'durable_journal_records':len(operation_rows),'actual_completed_updates':completed,'actual_microbatches':4*completed,
        'actual_training_metadata':train_reports,'checkpoints':reports,'checkpoint_pins':checkpoint_pins,
        'actual_witness':witness,'actual_saved_gates':gates,'terminal_failure_retained':failure,
        'actual_supervisor_run':run[0],'supervisor_final_reconstructed_pin':{'path':finished_name,**finished_pin},
        'training_process_success':terminal['complete_training_process'],
        'model_qualified':False,'app_default_changed':False,'independent_pixel_and_visual_review_pending':True,
        'quality_failure_not_overridden_by_state_integrity':True,'no_new_model_exported':True,
        'execution_boundary':{'weights_only_CPU_reads':1+len(checkpoint_names),'model_constructions':0,'neural_forwards':0,
            'backward_calls':0,'gradient_queries':0,'optimizer_constructions':0,'optimizer_steps':0,'VM_calls':0,
            'returned_code_imports':0,'browser_calls':0,'root_document_edits':0},'seconds':time.monotonic()-START}
    OUT.mkdir(parents=True)
    target=OUT/'state_audit_v1.json'
    with target.open('x',encoding='utf8',newline='\n') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'complete':True,'audit_pass':True,'receipt':{'path':relative(target),**pin(target)},
        'completed_updates':completed,'checkpoint_states':len(reports),'actual_local_neural_calls':0,'model_qualified':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',action='store_true',required=True)
    parser.parse_args();main()
