"""Independent stdlib-only pilot return integrity and finite-prefix audit.

Streams every gzip/tar byte and hashes all payloads without extraction or array
decoding. Replays the authenticated journal independently of exporter/runtime.
No scientific worker, model, neural framework, gradient, parameter or VM call.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
STAGE='cctv_codeformer_u64_gate_aligned_pilot_v1_vm'
PARENT='cctv_codeformer_u64_cleananchors_continuation_v1_vm'
PREFIX='cctv-codeformer-u64-gate-aligned-pilot-v1'
OPREFIX='outputs/u64_gate_aligned_pilot_v1/'
PROTOCOL_SHA='35a5cfd877a97355622c6c69d9792fd9238ccf312335dd679081a8636ba23b48'
ARCHIVE_SHA='005167e9be87c37efab55382652b9e5776466a61ef4f201f485fdafd527281d0'
OUT=ROOT/'outputs/codeformer_u64_gate_aligned_pilot_v1_return_transport'
CAPS={'full_forward_attempts':540,'full_forward_returns':540,
      'gradient_query_attempts':512,'gradient_query_returns':512,
      'parameter_application_attempts':4,'parameter_applications':4,
      'initial_quality_forwards':56,'initial_repeat_forwards':4,
      'optimization_forwards':256,'quality_forwards':224}
CATEGORIES=('initial_quality_forwards','initial_repeat_forwards','optimization_forwards','quality_forwards')
OPS={'forward':('full_forward_attempts','full_forward_returns'),
     'gradient_query':('gradient_query_attempts','gradient_query_returns'),
     'parameter_application':('parameter_application_attempts','parameter_applications')}
POST_EXPORT={'export.json','export.log','export_failure.json','supervisor.export.json',
             'supervisor.export.resources.jsonl','supervisor.export.started.json','supervisor.json','supervisor_failure.json'}

# Previously pure-tested framing parser; imported as inert stdlib source only.
SPEC=importlib.util.spec_from_file_location('independent_framing_parser',ROOT/'scripts/audit_u64_role_gradients_return_transport_v1.py')
F=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(F)
need, strict_json, confined, pin=F.need,F.strict_json,F.confined,F.pin


def expanded_plan(plan):
    need(len(plan)==544 and [r['slot']for r in plan]==list(range(1,545)),'Exact sealed544-row plan required')
    result=[]
    for row in plan:
        kind=row['kind'];need(kind in ('forward','parameter_application'),'Known sealed operation required')
        result.append((kind,row,None))
        if kind=='forward':result.extend(('gradient_query',row,q)for q in row['gradient_queries'])
        else:need(row['gradient_queries']==[],'Applications may have no implicit queries')
    need(len(result)==1056,'Exact sealed1056-operation expansion required')
    return result


def replay(raw,plan):
    need(raw.endswith(b'\n')and len(raw)<=32*1024**2,'Complete bounded journal required')
    sequence=expanded_plan(plan);counts=dict.fromkeys(CAPS,0)
    pending=None;position=token=0;previous=-1.;kinds=Counter();phases=[];rows=[]
    for line in raw.splitlines():
        row=strict_json(line);rows.append(row);kind=row['kind'];kinds[kind]+=1
        elapsed=row.get('elapsed_seconds')
        need(type(elapsed)in(int,float)and math.isfinite(elapsed)and elapsed>=previous,'Monotonic finite event time required')
        previous=elapsed
        reserved=kind.endswith('_reserved')and kind[:-9]in OPS
        returned=kind.endswith('_returned')and kind[:-9]in OPS
        need(not('token'in row and(kind.endswith('_reserved')or kind.endswith('_returned')))or reserved or returned,
             'Unknown operation event refused')
        if reserved:
            need(pending is None and position<len(sequence),'Known nonoverlapping operation required')
            operation,call,query=sequence[position];token+=1
            need(kind==operation+'_reserved'and row['token']==token,'Exact operation/token ordering required')
            pending={'token':token,'id':call['id'],'plan_slot':call['slot'],'new_update':call['new_update']}
            if query is not None:pending['query']=query
            if operation=='forward':pending['category']=call['category'];counts[call['category']]+=1
            counts[OPS[operation][0]]+=1
        elif returned:
            need(pending is not None,'Returned operation without reservation')
            operation,call,query=sequence[position]
            need(kind==operation+'_returned'and row.get('synchronized_return')is True,'Exact synchronized returned operation required')
            counts[OPS[operation][1]]+=1
        if reserved or returned:
            need(all(row.get(k)==v for k,v in pending.items()),'Returned/reserved identity/slot/update/query differs')
            need(row.get('counters')==counts,'Full operation counter replay differs')
            need(all(type(v)is int and 0<=v<=CAPS[k]for k,v in counts.items()),'Finite operation cap exceeded')
            if returned:pending=None;position+=1
        elif 'counters'in row:need(row['counters']==counts,'Nonoperation event counters differ')
        if kind=='phase_returned':phases.append(row)
    return {'counters':counts,'journal_events_replayed':len(rows),'event_kinds':dict(kinds),
            'completed_operation_pairs':position,'maximum_operation_pairs':len(sequence),'last_token':token,
            'unknown_reservations':[]if pending is None else[pending],
            'operation_outcome_unknown':pending is not None,'full_exact_plan_completed':pending is None and position==len(sequence),
            'valid_sealed_plan_prefix':True,'last_elapsed_seconds':previous,'phase_returns':phases},rows


def runtime_evidence(docs,raw,protocol,replayed,events,files):
    limits=protocol['limits'];status=docs[OPREFIX+'status.json'];failure=docs.get(OPREFIX+'failure.json')
    worker=docs['supervisor.worker.json'];start=docs['supervisor.start.json'];spawned=docs['supervisor.worker.started.json']
    need(status['protocol_sha256']==PROTOCOL_SHA and status['counters']==replayed['counters']and
         status['operation_outcome_unknown']==replayed['operation_outcome_unknown']and
         status['unknown_reservations']==replayed['unknown_reservations'],'Status/journal actual prefix differs')
    need(failure is not None and failure['counters']==replayed['counters']and
         failure['unknown_reservations']==replayed['unknown_reservations']and
         failure['protocol_sha256']==PROTOCOL_SHA,'Failure/journal actual prefix differs')
    need(start['protocol_sha256']==PROTOCOL_SHA and start['worker_spawn_reserved']is True and
         start['read_only_parent'].endswith('/'+PARENT),'Supervisor root/protocol identity differs')
    need(worker['child_pid']==spawned['child_pid']==spawned['process_group']and type(worker['child_pid'])is int and
         worker['child_pid']>0 and spawned['external_seconds']==limits['external_seconds'],'Supervisor actual child/cap differs')
    cleanup=worker['cleanup'];need(worker['spawn_returned']is True and cleanup['child_reaped']is True and
         cleanup['process_group_absent']is True and 0<=cleanup['seconds']<=limits['termination_grace_seconds'],
         'Known bounded worker cleanup required')
    resources=[strict_json(line)for line in raw[OPREFIX+'resources.jsonl'].splitlines()]
    watchdog=[strict_json(line)for line in raw['supervisor.worker.resources.jsonl'].splitlines()]
    for name in(OPREFIX+'resources.jsonl','supervisor.worker.resources.jsonl'):
        need(raw[name].endswith(b'\n'),'Complete resource journal required')
    previous=-1.
    for row in resources:
        need(row['elapsed_seconds']>=previous and row['elapsed_seconds']<limits['worker_seconds'],'Worker time cap differs');previous=row['elapsed_seconds']
        need(row['free_disk_bytes']>=limits['continuing_free_disk_bytes']and
             max(row['current_RSS_bytes'],row['peak_RSS_bytes'])<=limits['maximum_RSS_bytes']and
             max(row['CUDA_reserved_bytes'],row['CUDA_peak_reserved_bytes'])<=limits['maximum_CUDA_reserved_bytes']and
             row['output_bytes']<=limits['maximum_output_bytes'],'Worker resource cap exceeded')
    need(resources[0]['free_disk_bytes']>=limits['initial_free_disk_bytes'],'Initial sampled disk admission differs')
    previous=-1.
    for row in watchdog:
        need(row['child_pid']==worker['child_pid']and row['phase']=='worker'and row['elapsed_seconds']>=previous and
             row['elapsed_seconds']<limits['external_seconds'],'Watchdog identity/time differs');previous=row['elapsed_seconds']
        need(row['free_disk_bytes']>=limits['continuing_free_disk_bytes']and
             (row['current_RSS_bytes']is None or row['current_RSS_bytes']<=limits['maximum_RSS_bytes'])and
             row['output_bytes']<=limits['maximum_output_bytes']and row['archive_bytes']<=limits['maximum_archive_bytes'],
             'External watchdog resource cap exceeded')
    need(worker['resource_samples']==len(watchdog),'Supervisor sample count differs')
    for key in ('output_bytes','current_RSS_bytes','archive_bytes'):
        measured=max((r[key]for r in watchdog if r[key]is not None),default=0)
        need(worker['resource_maxima'][key]==measured,'Supervisor maxima differ from actual samples')
    need(worker['minimum_free_disk_bytes']==min(r['free_disk_bytes']for r in watchdog),'Supervisor disk minimum differs')
    need(all(0<=r['phase_seconds']<limits['per_phase_seconds']for r in replayed['phase_returns']),'Phase cap exceeded')
    own_bytes=sum(v['bytes']for k,v in files.items()if k.startswith(OPREFIX))
    need(own_bytes<=limits['maximum_output_bytes'],'Actual archived own output exceeds cap')
    qualified_stop=(worker['exit_code']==2 and failure.get('quality_stop')is True and
                    worker['timeout']is False and worker['error']is None and not replayed['operation_outcome_unknown'])
    return {'complete':True,'terminal_classification':'QUALITY_STOP_RETAINED_FAILED_PILOT_NOT_QUALIFIED'if qualified_stop else
            'OTHER_RETURNED_FAILURE_OR_PARTIAL_REQUIRES_REVIEW','quality_stop_receipt_verified':qualified_stop,
            'worker':worker,'failure':failure,'status':status,'worker_resource_samples':len(resources),
            'supervisor_watchdog_samples':len(watchdog),'worker_resource_maxima':{k:max(r[k]for r in resources)for k in
                ('current_RSS_bytes','peak_RSS_bytes','CUDA_reserved_bytes','CUDA_peak_reserved_bytes','output_bytes')},
            'minimum_sampled_worker_free_disk_bytes':min(r['free_disk_bytes']for r in resources),
            'maximum_phase_seconds':max(r['phase_seconds']for r in replayed['phase_returns']),
            'actual_archived_own_output_bytes':own_bytes,'post_export_receipts_absent_by_design':sorted(POST_EXPORT),
            'final_supervisor_export_exit_codes_available':False,'failure_quality_details_not_independently_recomputed_here':True,
            'resource_scope':'Authenticated saved measurements/cleanup declarations; no current VM/process query',
            'model_qualified':False}


def audit():
    OUT.mkdir(exist_ok=True);consumed={}
    def checked(path):
        value=pin(path);consumed[value['path']]=value;return value
    def doc(path):checked(path);return strict_json(path.read_bytes())
    export_path=ROOT/'outputs'/(PREFIX+'-export.json');export=doc(export_path)
    archive_path=ROOT/'outputs'/(PREFIX+'-results.tar.gz');sidecar=ROOT/'outputs'/(archive_path.name+'.sha256')
    protocol_path=ROOT/'outputs'/STAGE/'protocol.json';protocolraw=protocol_path.read_bytes();protocol=doc(protocol_path)
    need(len(protocolraw)==616319 and hashlib.sha256(protocolraw).hexdigest()==PROTOCOL_SHA,'Exact sealed pilot protocol required')
    need(export['protocol_sha256']==PROTOCOL_SHA and export['archive']['sha256']==ARCHIVE_SHA and
         export['archive']['bytes']==1166765803 and archive_path.stat().st_size==1166765803,'Actual return archive identity differs')
    need(sidecar.read_bytes()==(ARCHIVE_SHA+'  '+archive_path.name+'\n').encode()and sidecar.stat().st_size==123,
         'Exact LF checksum required');checked(sidecar)
    expected=export['files'];need(len(expected)==1643,'Exact1643member namespace required')
    for name,binding in expected.items():
        confined(name);need(binding['path']==name and set(binding)=={'path','bytes','sha256'},'Exact unique file binding required')
    before=archive_path.stat()
    with archive_path.open('rb')as f:parsed=F.stream_archive(f,expected,STAGE)
    after=archive_path.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),
                                 'Archive changed while streaming')
    framing=parsed['framing'];need(framing['compressed_bytes']==1166765803 and framing['compressed_sha256']==ARCHIVE_SHA and
                                 framing['payload_bytes']==1284390829,'Independent compressed/payload pin differs')
    consumed[archive_path.relative_to(ROOT).as_posix()]={'path':archive_path.relative_to(ROOT).as_posix(),
                                                      'bytes':1166765803,'sha256':ARCHIVE_SHA}
    files,raw=parsed['files'],parsed['metadata'];need(raw['protocol.json']==protocolraw,'Returned protocol bytes differ from seal')
    need(len(protocol['files'])==26,'Exact26sealed static files required')
    for name,value in protocol['files'].items():
        need(files[name]=={'path':name,**value},'Returned sealed source differs: '+name)
        actual=checked(ROOT/'outputs'/STAGE/name);need({k:actual[k]for k in ('bytes','sha256')}==value,'Local sealed stage changed')
    admissions={}
    for name,flag in [('evidence/source_review.json','source_review_pass'),('evidence/source_tests.json','all_tests_pass')]:
        value=strict_json(raw[name]);need(value['complete']is True and value[flag]is True,'Unapproved returned source')
        if 'review'in name:need(value['unresolved_blockers']==[],'Source review blockers')
        else:need(value['neural_gradient_optimizer_training_VM_calls']==0,'Source tests not pure')
        approved={r['path']:{'bytes':r['bytes'],'sha256':r['sha256']}for r in value['sources']}
        need(len(approved)==len(value['sources'])==15,'Exact15approved sources required')
        for source,pinrow in protocol['files'].items():
            if source.startswith(('research/','scripts/')):
                key='research/codeformer_u64_gate_aligned_pilot_v1/run_gate_aligned.sh'if source=='scripts/run_gate_aligned.sh'else source
                need(approved[key]==pinrow,'Returned source admission differs: '+source)
        admissions[name]={'sources_verified':len(approved),'proof_pin':files[name]}
    parent_protocol=strict_json(raw['evidence/parent_protocol.json']);readback=strict_json(raw['evidence/local_parent_readback.json'])
    need(export['parent_consumed_bindings']==parent_protocol['files']and len(parent_protocol['files'])==469,
         'Consumed469parent bindings differ from sealed copy')
    parentproof={r['parent_relative']:r['actual_local']for r in readback['files']};need(set(parentproof)==set(parent_protocol['files']),
                     'Parent local readback namespace differs');aliases=[];parentrows=[]
    for name,value in parent_protocol['files'].items():
        confined(name);source=ROOT/'outputs'/PARENT/name
        if not source.is_file():
            need(name in {'weights/codeformer_r2.pth','weights/codeformer_u96_adapter.pth'},'Unexpected parent alias')
            source=ROOT/confined(parentproof[name]['path']);aliases.append(name)
        actual=checked(source);need({k:actual[k]for k in ('bytes','sha256')}==value and actual==parentproof[name],
                                   'Consumed parent byte mismatch: '+name)
        parentrows.append({'parent_relative':name,'actual_local':actual})
    need(len(aliases)==2 and export['parent_payload_included']is False and export['parent_protocol']==protocol['parent_protocol'],
         'Exact parent alias/payload scope differs')
    docs={name:strict_json(value)for name,value in raw.items()if name.endswith('.json')}
    need(docs[OPREFIX+'operation_plan.json']['calls']==protocol['plan'],'Returned operation plan differs from sealed plan')
    replayed,events=replay(raw[OPREFIX+'events.jsonl'],protocol['plan'])
    for key in ('counters','journal_events_replayed','completed_operation_pairs','maximum_operation_pairs','last_token',
                'unknown_reservations','operation_outcome_unknown','full_exact_plan_completed','valid_sealed_plan_prefix'):
        need(export[key]==replayed[key],'Export/full independent journal replay differs: '+key)
    need(set(export['post_export_receipts_not_in_archive'])==POST_EXPORT and not POST_EXPORT.intersection(files),
         'Unexpected postexport receipt scope')
    runtime=runtime_evidence(docs,raw,protocol,replayed,events,files)
    metadata_path=OUT/'authenticated_small_metadata_v1.json'
    with metadata_path.open('x',encoding='utf-8',newline='\n')as f:
        json.dump({'archive_sha256':ARCHIVE_SHA,'documents':docs,'events':events},f,indent=2,allow_nan=False);f.write('\n')
    checked(metadata_path);checked(Path(__file__));checked(Path(F.__file__))
    # Rehash all small evidence and469parent payloads; the compressed archive was
    # already independently hashed in the complete framing pass above.
    for name,binding in consumed.items():
        if name!=archive_path.relative_to(ROOT).as_posix():need(pin(ROOT/name)==binding,'Consumed file changed after audit')
    receipt={'complete':True,'transport_audit_pass':True,'status':runtime['terminal_classification'],
             'created_UTC':datetime.now(timezone.utc).isoformat(),'archive':consumed[archive_path.relative_to(ROOT).as_posix()],
             'export':consumed[export_path.relative_to(ROOT).as_posix()],'sidecar':consumed[sidecar.relative_to(ROOT).as_posix()],
             'protocol':consumed[protocol_path.relative_to(ROOT).as_posix()],'framing':framing,
             'all1643_actual_member_pins_match_export':True,'all_member_pins':files,'sealed_static_sources_verified':26,
             'source_admission':admissions,'parent_static_bindings_rehashed':469,'local_model_aliases':aliases,
             'parent_bindings':parentrows,'journal':replayed,'runtime':runtime,'unresolved_transport_blockers':[],
             'authenticated_metadata':pin(metadata_path),'source':pin(Path(__file__)),
             'consumed_files':list(consumed.values()),'all_consumed_files_unchanged_after':True,
             'scope':'Archive integrity, exact source/parent binding, independent finite plan-prefix replay and authenticated runtime/failure/resource receipts only. Quality metrics, gradients, states and images require separate audits.',
             'actual_local_worker_model_Torch_neural_gradient_parameter_VM_calls':0,'model_qualified':False,
             'transport_integrity_does_not_qualify_model_or_overrule_quality_stop':True}
    target=OUT/'audit_v1.json'
    with target.open('x',encoding='utf-8',newline='\n')as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'receipt':pin(target),'status':runtime['terminal_classification'],'framing':framing,
                     'journal':{k:v for k,v in replayed.items()if k not in ('phase_returns','event_kinds')},
                     'worker':runtime['worker'],'failure':runtime['failure'],
                     'resources':{k:runtime[k]for k in ('worker_resource_samples','supervisor_watchdog_samples',
                         'worker_resource_maxima','minimum_sampled_worker_free_disk_bytes','maximum_phase_seconds')},
                     'post_export_receipts_absent':sorted(POST_EXPORT)},indent=2))


if __name__=='__main__':audit()
