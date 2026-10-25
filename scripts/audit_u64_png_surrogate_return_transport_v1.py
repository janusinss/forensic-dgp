"""Independent stdlib-only audit of the returned zero-update PNG measurement.

No extraction, arrays, worker/model import, neural execution or VM access.
The complete streaming pass authenticates every payload before JSON replay.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
PARENT = 'cctv_codeformer_u64_cleananchors_continuation_v1_vm'
PREFIX = 'cctv-codeformer-u64-png-surrogate-measurement-v1'
OPREFIX = 'outputs/u64_png_surrogate_measurement_v1/'
PROTOCOL_SHA = '0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c'
ARCHIVE_SHA = '0fe590ba344a6d0fe11a318f54dd66d0f1814ad751b7142927d471c29e7af3b2'
ARCHIVE_BYTES = 979202030
OUT = ROOT/'outputs/codeformer_u64_png_surrogate_measurement_v1_return_transport'
CAPS = dict(zip(('full_forward_attempts','full_forward_returns','gradient_query_attempts',
                'gradient_query_returns','parity_forwards','repeat_forwards','diagnostic_forwards'),
                (72,72,256,256,4,4,64)))
PHASE_CATEGORY = {'historical_u64_parity':'parity_forwards',
                  'deterministic_same_u64_repeat':'repeat_forwards',
                  'paired_TRAIN_gradient_diagnostic':'diagnostic_forwards'}
QUERIES = ['raw_mae','raw_adjacent','png_surrogate_mae','png_surrogate_adjacent']
OPS = {'forward':('full_forward_attempts','full_forward_returns'),
       'gradient_query':('gradient_query_attempts','gradient_query_returns')}
POST_EXPORT = {'export.json','export.log','export_failure.json','supervisor.export.json',
               'supervisor.export.resources.jsonl','supervisor.export.started.json',
               'supervisor.json','supervisor_failure.json'}
SPEC = importlib.util.spec_from_file_location('pure_return_framing',ROOT/'scripts/audit_u64_role_gradients_return_transport_v1.py')
F = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(F)
need, strict_json, confined, pin = F.need, F.strict_json, F.confined, F.pin


def write_new(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def capture():
    OUT.mkdir(exist_ok=True)
    archive = ROOT/'outputs'/(PREFIX+'-results.tar.gz')
    export_path = ROOT/'outputs'/(PREFIX+'-export.json')
    sidecar = archive.with_name(archive.name+'.sha256')
    export_pin,sidecar_pin = pin(export_path),pin(sidecar)
    export = strict_json(export_path.read_bytes())
    need(export['archive']['bytes']==ARCHIVE_BYTES and export['archive']['sha256']==ARCHIVE_SHA and
         export['protocol_sha256']==PROTOCOL_SHA and archive.stat().st_size==ARCHIVE_BYTES,
         'Exact independent archive/protocol identity required')
    need(sidecar.read_bytes()==(ARCHIVE_SHA+'  '+archive.name+'\n').encode('ascii'),'Exact LF checksum required')
    expected = export['files']; need(len(expected)==673,'Exact673 returned members required')
    for name,value in expected.items():
        confined(name); need(value['path']==name and set(value)=={'path','bytes','sha256'},'Exact export pin required')
    before = archive.stat()
    with archive.open('rb') as stream: parsed = F.stream_archive(stream,expected,STAGE)
    after = archive.stat()
    identity = lambda s:(s.st_ino,s.st_size,s.st_mtime_ns)
    need(identity(before)==identity(after),'Archive changed during streaming')
    framing=parsed['framing']
    need(framing['compressed_bytes']==ARCHIVE_BYTES and framing['compressed_sha256']==ARCHIVE_SHA and
         framing['payload_bytes']==1093522436,'Actual compressed and payload identities differ')
    need(pin(export_path)==export_pin and pin(sidecar)==sidecar_pin,'Export/sidecar changed during streaming')
    value={'complete':True,'framing':framing,'files':parsed['files'],
           'authenticated_small_text':{k:v.decode('utf-8') for k,v in parsed['metadata'].items()},
           'archive':{'path':archive.relative_to(ROOT).as_posix(),'bytes':ARCHIVE_BYTES,'sha256':ARCHIVE_SHA},
           'archive_stat':list(identity(after)),'export':export_pin,'sidecar':sidecar_pin,
           'framing_parser':pin(Path(F.__file__)),'actual_array_decodes_extractions_worker_model_VM_calls':0}
    target=OUT/'stream_capture_v1.json';write_new(target,value)
    print(json.dumps({'capture':pin(target),'framing':framing},indent=2))


def expanded_plan(plan):
    need(len(plan)==72 and [r['slot'] for r in plan]==list(range(1,73)),'Exact72-row plan required')
    sequence=[]
    for index,row in enumerate(plan):
        expected_phase='historical_u64_parity' if index<4 else 'deterministic_same_u64_repeat' if index<8 else 'paired_TRAIN_gradient_diagnostic'
        need(row['phase']==expected_phase and row['gradient_queries']==([] if index<8 else QUERIES),
             'Exact parity/repeat/query ordering required')
        sequence.append(('forward',row,None))
        sequence.extend(('gradient_query',row,q) for q in row['gradient_queries'])
    need(len(sequence)==328,'Exact328-operation sequence required');return sequence


def replay(raw,plan):
    need(raw.endswith(b'\n') and len(raw)<32*1024**2,'Complete bounded event journal required')
    sequence=expanded_plan(plan);counts=dict.fromkeys(CAPS,0)
    position=token=0;pending=None;previous=-1.;kinds=Counter();rows=[];phases=[]
    for line in raw.splitlines():
        row=strict_json(line);rows.append(row);kind=row['kind'];kinds[kind]+=1
        elapsed=row.get('elapsed_seconds')
        need(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=previous,'Monotonic finite event time required')
        previous=elapsed
        reserved=kind in ('forward_reserved','gradient_query_reserved')
        returned=kind in ('forward_returned','gradient_query_returned')
        need(not kind.endswith(('_reserved','_returned')) or reserved or returned or kind=='phase_returned',
             'Unknown operation reservation/return refused')
        if reserved:
            need(pending is None and position<len(sequence),'Nonoverlapping bounded reservation required')
            operation,call,query=sequence[position];token+=1
            need(kind==operation+'_reserved' and type(row['token']) is int and row['token']==token,
                 'Exact operation/token order required')
            pending={'token':token,'id':call['id'],'plan_slot':call['slot'],
                     'phase':'call_'+str(call['slot']).zfill(4)}
            if query is not None:pending['query']=query
            else:pending['category']=PHASE_CATEGORY[call['phase']];counts[pending['category']]+=1
            counts[OPS[operation][0]]+=1
        elif returned:
            need(pending is not None,'Operation returned without reservation')
            operation,call,query=sequence[position]
            need(kind==operation+'_returned' and row.get('synchronized_return') is True,
                 'Exact synchronized return required')
            counts[OPS[operation][1]]+=1
        if reserved or returned:
            need(all(row.get(k)==v for k,v in pending.items()),'Operation identity/phase/slot/query differs')
            need(row['counters']==counts,'Full independent counter replay differs')
            need(all(type(v) is int and 0<=v<=CAPS[k] for k,v in counts.items()),'Operation cap exceeded')
            if returned:pending=None;position+=1
        elif 'counters' in row:need(row['counters']==counts,'Nonoperation counters differ')
        if kind=='phase_returned':phases.append(row)
    unknown=[] if pending is None else [pending]
    return {'counters':counts,'journal_events_replayed':len(rows),'event_kinds':dict(kinds),
            'completed_operation_pairs':position,'maximum_operation_pairs':len(sequence),'last_token':token,
            'unknown_reservations':unknown,'operation_outcome_unknown':bool(unknown),
            'full_exact_plan_completed':not unknown and position==len(sequence),'valid_sealed_plan_prefix':True,
            'last_elapsed_seconds':previous,'phase_returns':phases},rows


def resource_audit(raw,protocol,worker,events):
    limits=protocol['limits'];resources=[strict_json(x) for x in raw[OPREFIX+'resources.jsonl'].splitlines()]
    watchdog=[strict_json(x) for x in raw['supervisor.worker.resources.jsonl'].splitlines()]
    need(resources and watchdog and raw[OPREFIX+'resources.jsonl'].endswith(b'\n') and
         raw['supervisor.worker.resources.jsonl'].endswith(b'\n'),'Complete nonempty resource journals required')
    labels=['before','after']*4
    for call in protocol['plan']:
        labels+=['before','before_forward']
        for query in call['gradient_queries']:labels+=['before_gradient_query','after_gradient_query']
        labels+=['after_forward_and_queries','after']
    labels+=['before','after']*2
    need([r['label'] for r in resources]==labels,'Exact resource boundary sample ordering differs')
    previous=-1.
    for row in resources:
        need(previous<=row['elapsed_seconds']<limits['worker_seconds'],'Worker time cap differs');previous=row['elapsed_seconds']
        for key in ('free_disk_bytes','current_RSS_bytes','peak_RSS_bytes','CUDA_reserved_bytes','CUDA_peak_reserved_bytes','output_bytes'):
            need(type(row[key]) is int and row[key]>=0,'Nonnegative integer resource measurement required')
        need(row['free_disk_bytes']>=limits['continuing_free_disk_bytes'] and
             max(row['current_RSS_bytes'],row['peak_RSS_bytes'])<=limits['maximum_RSS_bytes'] and
             max(row['CUDA_reserved_bytes'],row['CUDA_peak_reserved_bytes'])<=limits['maximum_CUDA_reserved_bytes'] and
             row['output_bytes']<=limits['maximum_output_bytes'],'Worker resource bound exceeded')
    need(resources[0]['free_disk_bytes']>=limits['initial_free_disk_bytes'],'Initial disk admission measurement differs')
    previous=-1.
    for row in watchdog:
        need(row['child_pid']==worker['child_pid'] and row['phase']=='worker' and
             previous<=row['elapsed_seconds']<limits['external_seconds'],'Watchdog identity/time differs');previous=row['elapsed_seconds']
        need(row['free_disk_bytes']>=limits['continuing_free_disk_bytes'] and
             (row['current_RSS_bytes'] is None or 0<=row['current_RSS_bytes']<=limits['maximum_RSS_bytes']) and
             0<=row['output_bytes']<=limits['maximum_output_bytes'] and
             0<=row['archive_bytes']<=limits['maximum_archive_bytes'],'Watchdog resource bound exceeded')
    need(worker['resource_samples']==len(watchdog),'Supervisor resource count differs')
    for key in ('output_bytes','current_RSS_bytes','archive_bytes'):
        need(worker['resource_maxima'][key]==max((r[key] for r in watchdog if r[key] is not None),default=0),
             'Supervisor resource maximum differs')
    need(worker['minimum_free_disk_bytes']==min(r['free_disk_bytes'] for r in watchdog),'Supervisor free disk minimum differs')
    phases=[r for r in events if r['kind']=='phase_returned']
    expected=['idle_GPU_before_framework','dependencies_and_FP32_policy','exact_trained_u64_initialization_no_forward',
              'initial_full575_value_witness']+['call_'+str(r['slot']).zfill(4) for r in protocol['plan']]+[
              'saved_gradient_geometry_no_update','closing_full575_value_witness_and_integrity']
    need([r['phase'] for r in phases]==expected and all(0<=r['phase_seconds']<limits['per_phase_seconds'] for r in phases),
         'Exact completed phase sequence/time cap differs')
    return {'worker_resource_samples':len(resources),'external_watchdog_samples':len(watchdog),
            'exact_resource_boundary_labels_verified':True,'phase_returns':len(phases),
            'maximum_phase_seconds':max(r['phase_seconds'] for r in phases),
            'worker_resource_maxima':{k:max(r[k] for r in resources) for k in
                ('current_RSS_bytes','peak_RSS_bytes','CUDA_reserved_bytes','CUDA_peak_reserved_bytes','output_bytes')},
            'minimum_worker_sampled_free_disk_bytes':min(r['free_disk_bytes'] for r in resources),
            'watchdog_maxima':worker['resource_maxima'],'watchdog_minimum_free_disk_bytes':worker['minimum_free_disk_bytes'],
            'scope':'Authenticated returned sampled measurements; no current process or VM query'}


def runtime_audit(docs,raw,protocol,replayed,events,files,export):
    limits=protocol['limits'];status=docs[OPREFIX+'status.json'];result=docs[OPREFIX+'result.json']
    need(replayed['counters']==CAPS and replayed['full_exact_plan_completed'] and len(events)==992,'Full actual72/256/328 replay required')
    for value in (status,result,export):
        need(value['complete'] is True and value['counters']==CAPS and value['protocol_sha256']==PROTOCOL_SHA and
             value['unknown_reservations']==[] and value['operation_outcome_unknown'] is False and value['model_qualified'] is False,
             'Status/result/export full operation outcome differs')
    need(status['active_slot']==72 and status['active_case']==protocol['plan'][-1]['id'] and status['active_query'] is None and
         status['elapsed_seconds']<limits['worker_seconds'],'Final status identity/cap differs')
    need(result['optimizer_constructions']==result['optimizer_steps']==result['parameter_applications']==0,
         'Zero optimizer/application declarations differ')
    need(result['true_PNG_derivative_claimed'] is False and result['PNG_surrogate_is_biased'] is True and
         result['PNG_gradient_claimed'] is False,'Surrogate epistemic boundary differs')
    need(OPREFIX+'failure.json' not in files,'Returned worker failure present')
    start=docs['supervisor.start.json'];spawned=docs['supervisor.worker.started.json'];worker=docs['supervisor.worker.json']
    need(start['protocol_sha256']==PROTOCOL_SHA and start['worker_spawn_reserved'] is True and
         start['read_only_parent'].endswith('/'+PARENT),'Supervisor start identity differs')
    need(spawned['child_pid']==spawned['process_group']==worker['child_pid'] and type(worker['child_pid']) is int and worker['child_pid']>0 and
         spawned['phase']==worker['phase']=='worker' and spawned['external_seconds']==limits['external_seconds'],
         'Known worker PID/process group/cap differs')
    need(0<=worker['elapsed_seconds']<limits['external_seconds'] and worker['external_seconds']==limits['external_seconds'],
         'Supervisor measured duration/cap differs')
    cleanup=worker['cleanup']
    need(worker['complete'] is True and worker['exit_code']==0 and worker['timeout'] is False and worker['error'] is None and
         worker['spawn_returned'] is True and cleanup['child_reaped'] is True and cleanup['process_group_absent'] is True and
         cleanup['actions']==[] and 0<=cleanup['seconds']<=limits['termination_grace_seconds'],'Worker success/known cleanup differs')
    need(files['run.log']['bytes']==0,'Unexpected worker log output')
    allowed={'static_VM_supervisor_guards_passed_before_Torch','phase_returned','run_result_saved',
             'forward_reserved','forward_returned','gradient_query_reserved','gradient_query_returned','gradient_warning_scope'}
    need(all(r['kind'] in allowed for r in events) and events[0]['kind']=='static_VM_supervisor_guards_passed_before_Torch' and
         events[0]['Torch_imported'] is False and events[-2]['kind']=='run_result_saved','Exact event namespace/boundary differs')
    signatures=[('static_VM_supervisor_guards_passed_before_Torch','static_guards')]
    signatures.extend(('phase_returned',phase) for phase in ('idle_GPU_before_framework','dependencies_and_FP32_policy',
        'exact_trained_u64_initialization_no_forward','initial_full575_value_witness'))
    for call in protocol['plan']:
        phase='call_'+str(call['slot']).zfill(4)
        signatures.extend((kind,phase) for kind in ('forward_reserved','forward_returned'))
        for query in call['gradient_queries']:
            signatures.extend((kind,phase) for kind in ('gradient_query_reserved','gradient_query_returned','gradient_warning_scope'))
        signatures.append(('phase_returned',phase))
    signatures.extend([('phase_returned','saved_gradient_geometry_no_update'),
        ('run_result_saved','closing_full575_value_witness_and_integrity'),
        ('phase_returned','closing_full575_value_witness_and_integrity')])
    need([(r['kind'],r['phase']) for r in events]==signatures,'Full992event interleaving/phase sequence differs')
    warnings=[r for r in events if r['kind']=='gradient_warning_scope']
    need(len(warnings)==256 and all(r['unexpected']==[] and r['strict_flags_restored'] is True and
         r['bitwise_CUDA_gradient_repeatability_claimed'] is False for r in warnings),'Warning scope differs')
    for row in warnings:
        for warning in row['warnings']:
            need(re.match(r'^upsample_bilinear2d_backward_out_cuda\b',warning['message']) is not None,'Unexpected gradient warning')
    resources=resource_audit(raw,protocol,worker,events)
    own_bytes=sum(v['bytes'] for k,v in files.items() if k.startswith(OPREFIX))
    need(own_bytes<=limits['maximum_output_bytes'],'Actual own output byte cap exceeded')
    return {'complete':True,'terminal_classification':'MEASUREMENT_COMPLETED_ZERO_UPDATES_NOT_MODEL_QUALIFICATION',
            'worker':worker,'status':status,'resources':resources,'warning_scopes':len(warnings),
            'actual_archived_own_output_bytes':own_bytes,'empty_run_log_valid':True,
            'post_export_receipts_absent_by_design':sorted(POST_EXPORT),'final_export_exit_code_available':False,
            'no_final_export_exit_code_inferred_from_complete_archive':True,
            'zero_parameter_applications_recorded':True,'scientific_state_gradient_and_pixel_claims_require_separate_audit':True}


def metadata_bindings_audit(docs,protocol,files):
    """Authenticate record joins and exact expected names; do not validate arrays."""
    expected=set(protocol['files'])|{'protocol.json','supervisor.start.json','supervisor.worker.started.json',
        'supervisor.worker.json','supervisor.worker.resources.jsonl','run.log'}
    basic=('started.json','operation_plan.json','events.jsonl','resources.jsonl','head_schema.json',
        'RNG_sampler_initial.json','RNG_sampler_closing.json','result.json','status.json','saved_gradient_geometry.json',
        'full_states/initial.json','full_states/initial.npz','full_states/closing.json','full_states/closing.npz')
    expected.update(OPREFIX+n for n in basic)
    result=docs[OPREFIX+'result.json'];diagnostic_pins=[]
    selection={r['id']:r for r in protocol['selection']['monitors']+protocol['selection']['optimization']}
    def bound(value):
        need(type(value) is dict and set(value)=={'path','bytes','sha256'},'Exact saved binding required')
        need(files[confined(value['path'])]==value,'Saved JSON binding differs from actual archived bytes')
    for call in protocol['plan']:
        stem=OPREFIX+'calls/'+str(call['slot']).zfill(4)+'/'
        names=['raw.npy','result.png','record.json']
        if call['gradient_queries']:
            names+=['png_surrogate_forward.npy','gradients.json']+['gradients_'+q+'.npz' for q in QUERIES]
        else:names+=['unclipped.npy','native.npy','logits.npy','LQ.npy']
        expected.update(stem+n for n in names)
        record=docs[stem+'record.json'];source=selection[call['id']]
        need(record['complete'] is True and record['call']==call and record['id']==call['id'] and
             record['reference_id']==call['reference_id'] and record['role']==call['role'] and record['split']=='TRAIN' and
             record['input']==source['input'] and record['target']==source['target'],'Call input/target/plan join differs')
        for field,value in record['outputs'].items():bound(value)
        if not call['gradient_queries']:
            for value in record['stock'].values():bound(value['file'])
        else:
            scalar=docs[stem+'gradients.json'];need(scalar['call']==call and scalar['complete'] is True and
                set(scalar['gradients'])==set(QUERIES),'Exact four-query metadata join differs')
            bound(scalar['output_record']);bound(scalar['PNG_forward_reference']);bound(scalar['PNG_RGB8_file'])
            need(scalar['output_record']==files[stem+'record.json'] and scalar['PNG_RGB8_file']==files[stem+'result.png'] and
                 scalar['PNG_forward_reference']==files[stem+'png_surrogate_forward.npy'],'Exact PNG/record binding differs')
            for query in QUERIES:
                bound(scalar['gradients'][query]['file'])
                need(scalar['gradients'][query]['file']==files[stem+'gradients_'+query+'.npz'],'Query witness path differs')
            diagnostic_pins.append(files[stem+'gradients.json'])
    need(set(files)==expected and len(expected)==673,'Independently reconstructed full673file namespace differs')
    need(result['diagnostic_records']==diagnostic_pins,'Result64 diagnostic record bindings differ')
    bound(result['closing_state']);bound(result['saved_gradient_geometry'])
    for label in ('initial','closing'):
        value=docs[OPREFIX+'full_states/'+label+'.json'];bound(value['state'])
        need(value['state']==files[OPREFIX+'full_states/'+label+'.npz'],'State description/file binding differs')
    initial=docs[OPREFIX+'RNG_sampler_initial.json'];closing=docs[OPREFIX+'RNG_sampler_closing.json']
    need(initial['RNG']==closing['RNG'] and initial['optimizer'] is closing['optimizer'] is None and
         initial['sampler']=={'automatic_resume_permitted':False,'fixed_photo_major_plan':True,'next_slot':1} and
         closing['sampler']=={'automatic_resume_permitted':False,'fixed_photo_major_plan':True,'next_slot':None},
         'Saved RNG/sampler/optimizer boundary differs')
    return {'complete':True,'independently_expected_file_names':673,'call_record_plan_input_target_joins':72,
            'gradient_record_joins':64,'gradient_witness_file_pins':256,'state_record_file_pin_joins':2,
            'PNG_forward_reference_file_pin_joins':64,'saved_run_RNG_records_equal':True,
            'fixed_sampler_and_no_optimizer_records_verified':True,'arrays_or_images_decoded':False,
            'scope':'Metadata-to-actual-payload joins only; independent value/gradient/image audits remain separate'}


def audit():
    consumed={}
    def checked(path):
        value=pin(path);consumed[value['path']]=value;return value
    def doc(path):checked(path);return strict_json(path.read_bytes())
    capture_path=OUT/'stream_capture_v1.json';captured=doc(capture_path)
    archive=ROOT/captured['archive']['path'];info=archive.stat()
    need([info.st_ino,info.st_size,info.st_mtime_ns]==captured['archive_stat'],'Archive changed after full streaming pass')
    export_path=ROOT/'outputs'/(PREFIX+'-export.json');export=doc(export_path)
    need(checked(export_path)==captured['export'] and checked(ROOT/captured['sidecar']['path'])==captured['sidecar'] and
         checked(Path(F.__file__))==captured['framing_parser'],'Stream consumed evidence changed')
    framing=captured['framing'];files=captured['files'];raw={k:v.encode('utf-8') for k,v in captured['authenticated_small_text'].items()}
    need(files==export['files'] and framing['compressed_sha256']==ARCHIVE_SHA and framing['regular_members']==673,
         'Captured/export file identities differ')
    protocol_path=ROOT/'outputs'/STAGE/'protocol.json';protocolraw=protocol_path.read_bytes();protocol=doc(protocol_path)
    need(len(protocolraw)==188285 and hashlib.sha256(protocolraw).hexdigest()==PROTOCOL_SHA and raw['protocol.json']==protocolraw,
         'Exact returned/sealed protocol required')
    need(len(protocol['files'])==21,'Exact21static source/evidence files required')
    for name,value in protocol['files'].items():
        need(files[name]=={'path':name,**value},'Returned sealed static payload differs: '+name)
        actual=checked(ROOT/'outputs'/STAGE/name)
        need({k:actual[k] for k in ('bytes','sha256')}==value,'Original sealed stage changed: '+name)
    admissions={}
    for name,flag in [('evidence/source_review.json','source_review_pass'),('evidence/source_tests.json','all_tests_pass')]:
        proof=strict_json(raw[name]);need(proof['complete'] is True and proof[flag] is True,'Unapproved returned sources')
        if 'review' in name:need(proof['unresolved_blockers']==[],'Source review blockers')
        else:need(proof['neural_gradient_optimizer_training_VM_calls']==0,'Source tests not pure')
        approved={r['path']:{'bytes':r['bytes'],'sha256':r['sha256']} for r in proof['sources']}
        need(len(approved)==len(proof['sources'])==12,'Exact12unique approved sources required')
        for source,value in protocol['files'].items():
            if source.startswith(('research/','scripts/')):
                key='research/codeformer_u64_png_surrogate_measurement_v1/run_png_surrogate.sh' if source=='scripts/run_png_surrogate.sh' else source
                need(approved[key]==value,'Returned source approval differs: '+source)
        admissions[name]={'sources_verified':12,'proof_pin':files[name]}
    parent_protocol=strict_json(raw['evidence/parent_protocol.json']);readback=strict_json(raw['evidence/local_parent_readback.json'])
    actual_parent_protocol=checked(ROOT/'outputs'/PARENT/'protocol.json')
    need({k:actual_parent_protocol[k] for k in ('bytes','sha256')}=={k:protocol['parent_protocol'][k] for k in ('bytes','sha256')} and
         (ROOT/actual_parent_protocol['path']).read_bytes()==raw['evidence/parent_protocol.json'],'Original parent protocol differs')
    need(export['parent_consumed_bindings']==parent_protocol['files'] and len(parent_protocol['files'])==469 and
         export['parent_consumed_static_bindings']==469 and export['parent_protocol']==protocol['parent_protocol'] and
         export['parent_payload_included'] is False,'Exact469 consumed parent bindings differ')
    bindings={r['parent_relative']:r['actual_local'] for r in readback['files']}
    need(len(bindings)==len(readback['files'])==469 and set(bindings)==set(parent_protocol['files']),'Parent provenance namespace differs')
    parentrows=[];aliases=[]
    for name,value in parent_protocol['files'].items():
        confined(name);confined(bindings[name]['path']);source=ROOT/'outputs'/PARENT/name
        if not source.is_file():
            need(name in ('weights/codeformer_r2.pth','weights/codeformer_u96_adapter.pth'),'Unexpected parent alias')
            source=ROOT/bindings[name]['path'];aliases.append({'parent_relative':name,'actual_local':bindings[name]})
        actual=checked(source)
        need(actual==bindings[name] and {k:actual[k] for k in ('bytes','sha256')}==value,'Actual parent binding differs: '+name)
        parentrows.append({'parent_relative':name,'actual_local':actual})
    need(len(aliases)==2 and readback['VM_requires_all469_at_actual_parent_stage'] is True and readback['VM_inventory_performed'] is False,
         'Two local model aliases only; realVM469 scope differs')
    docs={k:strict_json(v) for k,v in raw.items() if k.endswith('.json')}
    need(docs[OPREFIX+'operation_plan.json']['calls']==protocol['plan'],'Returned plan differs from seal')
    replayed,events=replay(raw[OPREFIX+'events.jsonl'],protocol['plan'])
    for key in ('counters','journal_events_replayed','completed_operation_pairs','maximum_operation_pairs','last_token',
                'unknown_reservations','operation_outcome_unknown','full_exact_plan_completed','valid_sealed_plan_prefix'):
        need(export[key]==replayed[key],'Independent full journal differs from export: '+key)
    need(set(export['post_export_receipts_not_in_archive'])==POST_EXPORT and not POST_EXPORT.intersection(files),
         'Post-export receipt namespace differs')
    runtime=runtime_audit(docs,raw,protocol,replayed,events,files,export)
    metadata_bindings=metadata_bindings_audit(docs,protocol,files)
    worker_path=ROOT/'outputs'/STAGE/'research/codeformer_u64_png_surrogate_measurement_v1/worker.py'
    source_ast=F.audit_worker_ast(worker_path)
    metadata_path=OUT/'authenticated_small_metadata_v1.json'
    write_new(metadata_path,{'archive_sha256':ARCHIVE_SHA,'documents':docs,'events':events})
    checked(metadata_path);checked(Path(__file__))
    for name,value in consumed.items():need(pin(ROOT/name)==value,'Consumed file changed after analysis')
    consumed[captured['archive']['path']]=captured['archive']
    result={'complete':True,'transport_audit_pass':True,'status':runtime['terminal_classification'],
            'created_UTC':datetime.now(timezone.utc).isoformat(),'archive':captured['archive'],'export':captured['export'],
            'sidecar':captured['sidecar'],'protocol':consumed[protocol_path.relative_to(ROOT).as_posix()],
            'framing':framing,'all673_actual_member_pins_match_export':True,'all_member_pins':files,
            'sealed_static_sources_verified':21,'source_admission':admissions,
            'parent_static_bindings_rehashed':469,'local_model_aliases':aliases,'parent_bindings':parentrows,
            'journal':replayed,'runtime':runtime,'worker_source_AST':source_ast,
            'metadata_bindings':metadata_bindings,
            'unresolved_transport_blockers':[],'authenticated_metadata':pin(metadata_path),'source':pin(Path(__file__)),
            'consumed_files':list(consumed.values()),'all_consumed_files_unchanged_after':True,
            'archive_fully_hashed_once_and_stat_unchanged_after':True,
            'scope':'Complete returned archive integrity, exact sealed source/parent pins, full ordered operation replay and authenticated runtime/cleanup/resource evidence. State, gradient, geometry and image validity are separate audits.',
            'actual_local_array_decode_worker_model_Torch_neural_gradient_parameter_VM_browser_calls':0,
            'model_qualified':False,'transport_integrity_does_not_establish_quality':True}
    target=OUT/'receipt_v1.json';write_new(target,result)
    print(json.dumps({'receipt':pin(target),'status':result['status'],'framing':framing,
                     'journal':{k:v for k,v in replayed.items() if k not in ('phase_returns','event_kinds')},
                     'runtime':runtime,'unresolved_transport_blockers':[]},indent=2))


class PureTests(unittest.TestCase):
    def plan(self):
        return [{'slot':i,'id':'case'+str(i),'phase':'historical_u64_parity' if i<=4 else
            'deterministic_same_u64_repeat' if i<=8 else 'paired_TRAIN_gradient_diagnostic',
            'gradient_queries':[] if i<=8 else list(QUERIES)} for i in range(1,73)]

    def journal(self):
        rows=[];counts=dict.fromkeys(CAPS,0)
        for token,(operation,call,query) in enumerate(expanded_plan(self.plan()),1):
            base={'id':call['id'],'plan_slot':call['slot'],'phase':'call_'+str(call['slot']).zfill(4),'token':token}
            if query is None:base['category']=PHASE_CATEGORY[call['phase']];counts[base['category']]+=1
            else:base['query']=query
            for kind in ('reserved','returned'):
                counts[OPS[operation][kind=='returned']]+=1
                rows.append({**base,'kind':operation+'_'+kind,'elapsed_seconds':len(rows)/100.,'counters':dict(counts),
                             **({'synchronized_return':True} if kind=='returned' else {})})
        return rows

    def run_rows(self,rows):return replay(('\n'.join(json.dumps(r) for r in rows)+'\n').encode(),self.plan())[0]

    def test_full_order(self):
        x=self.run_rows(self.journal());self.assertEqual(x['counters'],CAPS);self.assertEqual(x['completed_operation_pairs'],328)
    def test_partial_pending(self):
        x=self.run_rows(self.journal()[:1]);self.assertTrue(x['operation_outcome_unknown']);self.assertEqual(x['completed_operation_pairs'],0)
    def test_partial_known(self):
        x=self.run_rows(self.journal()[:2]);self.assertFalse(x['operation_outcome_unknown']);self.assertFalse(x['full_exact_plan_completed'])
    def test_query_order(self):
        rows=self.journal();rows[18]['query']='wrong'
        with self.assertRaises(ValueError):self.run_rows(rows)
    def test_missing_return(self):
        rows=self.journal();del rows[1]
        with self.assertRaises(ValueError):self.run_rows(rows)
    def test_unknown_tokenless_reservation(self):
        with self.assertRaises(ValueError):self.run_rows([{'kind':'optimizer_reserved','elapsed_seconds':0}])
    def test_unknown_return(self):
        with self.assertRaises(ValueError):self.run_rows([{'kind':'parameter_application_returned','elapsed_seconds':0,'token':1}])
    def test_reversed_time(self):
        rows=self.journal();rows[2]['elapsed_seconds']=-1
        with self.assertRaises(ValueError):self.run_rows(rows)
    def test_counter_lie(self):
        rows=self.journal();rows[1]['counters']['gradient_query_returns']=1
        with self.assertRaises(ValueError):self.run_rows(rows)
    def test_wrong_phase(self):
        rows=self.journal();rows[1]['phase']='call_0002'
        with self.assertRaises(ValueError):self.run_rows(rows)
    def test_unsynchronized_return(self):
        rows=self.journal();rows[1]['synchronized_return']=False
        with self.assertRaises(ValueError):self.run_rows(rows)
    def test_sealed_plan_four_query_order(self):
        plan=self.plan();plan[8]['gradient_queries']=list(reversed(QUERIES))
        with self.assertRaises(ValueError):expanded_plan(plan)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(PureTests),
                                 unittest.defaultTestLoader.loadTestsFromTestCase(F.PureTests)])
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    elif args.capture:capture()
    else:audit()
